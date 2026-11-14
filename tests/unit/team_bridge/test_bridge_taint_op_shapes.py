"""Every bridge op, run for real, lands on the right side of the taint predicate.

``harness/bridge_egress.py: bridge_result_is_inbound`` decides whether a ``bridge``
result latches the session from the SHAPE of the tool's ``details``. The unit table in
``tests/unit/harness/test_bridge_egress.py`` pins the predicate against hand-written
shapes; this file pins the other half, that the tool still BUILDS those shapes. Each op
runs through ``build_bridge_tool(...).execute`` against a real ``TeamBridgeClient`` whose
transport is a mock hub, the ``ToolResult`` becomes a ``Message`` the way the loop does it
(``Message.tool_result``), and the live message and its journal row are classified.
A tool change that makes a receipt lose ``receipt_stage``, or a read op start returning a
``members`` list, fails here instead of silently opening or closing the latch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.aida import state as aida_state
from local_operator.harness import bridge_egress as be
from local_operator.harness.types import Message, ToolContext
from local_operator.team_bridge import authority as authority_mod
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import (
    BRIDGE_OPS,
    BridgeRuntime,
    body_digest,
    build_bridge_tool,
)
from tests.unit.harness.test_bridge_egress import OP_LATCHES
from tests.unit.team_bridge.test_client import FakeStore
from tests.unit.team_bridge.test_owner_audit_isolation import (
    CID,
    OP_ARGS,
    ROOT,
    TENANT,
    _hub_answer,
)

TAKE_UP_BODY = "a screened teammate body"


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the fences so each op reaches its builder (they have their own tests)."""

    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(authority_mod, "authority_fence", lambda runtime: None)
    monkeypatch.setattr(authority_mod, "outbound_authority_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=1,
        account_id="acc_1",
        tenant_id=TENANT,
        endpoint_id="bep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _hub(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/bmsg_1"):
        # ``take_up`` fetches one pair-lane message by id.
        return httpx.Response(
            200, json={"msg": "ok", "result": {"message": {"body": {"text": TAKE_UP_BODY}}}}
        )
    if request.method == "POST" and "/conversations" not in request.url.path:
        # The pair lane (``send``/``reply``) confirms with a 201 and the new message's id.
        return httpx.Response(201, json={"msg": "ok", "result": {"message": {"id": "bmsg_9"}}})
    return _hub_answer(request)


def _runtime(tmp_path: Path, handler: Any = _hub) -> BridgeRuntime:
    store = Store(tmp_path)
    store.commit_enrollment(_owner(), OwnerReceipt(consent_version="cv-1", consent_ref="cr-1"))
    client = TeamBridgeClient(
        tmp_path,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )
    return BridgeRuntime(
        client=client,
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="sess_1",
    )


async def _classify(runtime: BridgeRuntime, args: dict[str, Any]) -> tuple[Message, bool, bool]:
    """Run one call through the tool; return the live message and its two verdicts."""

    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    result = await tool.execute("call_1", args, None, None, ToolContext(bridge_runtime=runtime))
    message = Message.tool_result(result)
    payload = message.model_dump(mode="json", exclude_none=True)
    payload["kind"] = "message"
    live = be.message_is_taint_source(message)
    durable = be.entry_is_taint_source("message", payload)
    return message, live, durable


def _args(op: str) -> dict[str, Any]:
    args = {"op": op, **OP_ARGS[op]}
    if op == "take_up":
        args["message_id"] = "bmsg_1"
        args["body_sha256"] = body_digest(TAKE_UP_BODY)
    return args


def test_the_driver_table_covers_every_op() -> None:
    assert set(OP_ARGS) == set(OP_LATCHES) == set(BRIDGE_OPS)


@pytest.mark.asyncio
@pytest.mark.parametrize("op", sorted(OP_LATCHES))
async def test_a_real_call_of_each_op_is_classified_as_expected(tmp_path: Path, op: str) -> None:
    # The shared reads need the durable chief of staff (a task session is refused).
    aida_state.write_state(tmp_path, {"session_id": "sess_1"})
    runtime = _runtime(tmp_path)
    if op == "ack":
        # ``done`` needs a live claim by this session; claim it with the real op first.
        await _classify(runtime, {"op": "ack", "message_id": "bmsg_1", "state": "working"})
    message, live, durable = await _classify(runtime, _args(op))
    details = (message.provider_payload or {}).get("details") or {}
    # Not vacuous: the op ran to its success shape, it was not refused before it started.
    assert details.get("ok") is True, (op, details)
    assert live is durable is OP_LATCHES[op], (op, details)


@pytest.mark.asyncio
async def test_a_dm_goes_through_the_same_receipt_as_a_channel_post(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for op in ("dm", "channel_post"):
        message, live, _ = await _classify(runtime, _args(op))
        details = (message.provider_payload or {})["details"]
        assert details["receipt_stage"] == "server_accepted", op
        assert live is False


@pytest.mark.asyncio
async def test_a_second_identical_send_is_the_skipped_result_and_does_not_latch(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    first = await _classify(runtime, _args("channel_post"))
    second = await _classify(runtime, _args("channel_post"))
    details = (second[0].provider_payload or {})["details"]
    assert first[1] is False and second[1] is False
    assert details.get("skipped") is True, details


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"op": "decide", "message_id": "bmsg_1"},
        {"op": "channel_history"},
        {"op": "channel_post", "channel_id": CID, "body": ""},
        {"op": "channel_thread", "channel_id": CID},
        {"op": "react", "channel_id": CID, "message_id": ROOT},
    ],
)
async def test_a_refusal_does_not_latch(tmp_path: Path, args: dict[str, Any]) -> None:
    runtime = _runtime(tmp_path)
    message, live, durable = await _classify(runtime, args)
    details = (message.provider_payload or {})["details"]
    assert details["ok"] is False and message.is_error is True, details
    assert live is durable is False


@pytest.mark.asyncio
async def test_a_hub_failure_notice_does_not_latch(tmp_path: Path) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"msg": "boom"})

    runtime = _runtime(tmp_path, failing)
    message, live, durable = await _classify(runtime, _args("channels"))
    assert (message.provider_payload or {})["details"]["ok"] is False
    assert live is durable is False
