"""H5d — the bridge tool: hash-verified take-up, local claim/release, sends
through the outbox, and NO decide authority."""

from __future__ import annotations

import json
from typing import cast

import httpx
import pytest

from local_operator.harness.types import ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import (
    BridgeRuntime,
    body_digest,
    build_bridge_tool,
    execute_bridge,
)
from tests.unit.team_bridge.test_client import FakeStore, _ok

TENANT = "ten_1"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _gate_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open BOTH entry gates for the op-level subjects below.

    The tool now calls the LOCAL authority fence FIRST and the admission gate
    second (S-1 remediation), and these tests exercise hash verification,
    claims and sends — not the gates themselves (the fence and the admission
    states, including reads-vs-mutations, are covered by
    ``test_h5_remediation_fence.py`` / ``_coalesce`` / ``_doors``). Both are
    patched at the module attributes the tool imports at CALL time, so the
    pending default cannot shadow the subject here.
    """

    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
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


def _client(handler) -> TeamBridgeClient:
    return TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )


def test_the_builder_never_advertises_without_a_runtime() -> None:
    assert build_bridge_tool(ToolContext()) is None
    # The builder reads no client; ``BridgeRuntime`` is typed against the real
    # one, so the stand-in is cast rather than the annotation weakened (see
    # test_h5_remediation_doors).
    runtime = BridgeRuntime(
        client=cast(TeamBridgeClient, object()),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None and tool.name == "bridge"
    op_schema = tool.parameters["properties"]["op"]
    assert "decide" not in json.dumps(op_schema)


@pytest.mark.asyncio
async def test_take_up_verifies_the_body_digest_before_handing_it_over(tmp_path) -> None:
    text = "the full screened body, longer than any excerpt"
    runtime = BridgeRuntime(
        client=_client(lambda request: _ok({"message": {"body": {"text": text}}})),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )
    good = await execute_bridge(
        runtime, {"op": "take_up", "message_id": "bmsg_1", "body_sha256": body_digest(text)}
    )
    assert good["ok"] is True and good["text"] == text

    bad = await execute_bridge(
        runtime, {"op": "take_up", "message_id": "bmsg_1", "body_sha256": "0" * 64}
    )
    assert bad["ok"] is False and bad["code"] == "body_hash_mismatch"


@pytest.mark.asyncio
async def test_the_tool_refuses_the_decide_word_outright(tmp_path) -> None:
    runtime = BridgeRuntime(
        client=_client(lambda request: _ok({})),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )
    outcome = await execute_bridge(runtime, {"op": "decide", "message_id": "bmsg_1"})
    assert outcome["ok"] is False and outcome["code"] == "not_permitted"
    assert "human" in outcome["error"]


@pytest.mark.asyncio
async def test_ack_working_claims_and_done_releases_with_the_same_token(tmp_path) -> None:
    owner = _owner()
    store = Store(tmp_path)
    store.commit_enrollment(owner, OwnerReceipt(consent_version="cv-1", consent_ref="cr-1"))
    runtime = BridgeRuntime(
        client=_client(lambda request: _ok({})),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=owner,
        session_id="s1",
    )
    working = await execute_bridge(
        runtime, {"op": "ack", "message_id": "bmsg_1", "state": "working"}
    )
    assert working["ok"] is True and "bmsg_1" in runtime.claims

    # A second session (different runtime) cannot grab the same message.
    other = BridgeRuntime(
        client=runtime.client, config_dir=tmp_path, tenant_id=TENANT, owner=owner, session_id="s2"
    )
    contended = await execute_bridge(
        other, {"op": "ack", "message_id": "bmsg_1", "state": "working"}
    )
    assert contended["ok"] is False and contended["code"] == "claimed"

    done = await execute_bridge(runtime, {"op": "ack", "message_id": "bmsg_1", "state": "done"})
    assert done["ok"] is True and "bmsg_1" not in runtime.claims
    # 'done' without a live claim by THIS session is refused, not stolen.
    again = await execute_bridge(runtime, {"op": "ack", "message_id": "bmsg_1", "state": "done"})
    assert again["ok"] is False and again["code"] == "not_holder"


@pytest.mark.parametrize(
    "target",
    [
        {"account_id": "acc_2", "kind": "agent"},
        {"endpoint_id": "bep_9", "kind": "agent"},
        {"account_id": "acc_2", "endpoint_id": "bep_9"},
        {},
    ],
)
def test_tool_recipient_schema_is_closed(target):
    from pydantic import ValidationError

    from local_operator.team_bridge.tool import BridgeToolParams

    with pytest.raises(ValidationError):
        BridgeToolParams.model_validate({"op": "send", "to": target, "text": "synthetic"})


@pytest.mark.parametrize(
    ("target", "valid"),
    [
        ({"endpoint_id": "bep_9"}, True),
        ({"account_id": "acc_2"}, True),
        ({"account_id": "acc_2", "kind": "agent"}, False),
        ({"endpoint_id": "bep_9", "kind": "agent"}, False),
        ({"account_id": "acc_2", "endpoint_id": "bep_9"}, False),
        ({"account_id": ""}, False),
        ({"account_id": 42}, False),
        ({}, False),
    ],
)
def test_compact_recipient_wire_schema_preserves_closed_one_of(target, valid):
    """Schema compression must not just rely on the executor catching bad input."""
    from jsonschema import Draft202012Validator

    from local_operator.providers.clients import (
        _tools_to_openai,
        _tools_to_openai_responses,
    )
    from local_operator.team_bridge.tool import BridgeToolParams

    runtime = BridgeRuntime(
        client=cast(TeamBridgeClient, object()),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-schema",
    )
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    schema = tool.parameters
    assert "propertyNames" not in json.dumps(schema)
    for published in (
        schema,
        _tools_to_openai([tool])[0]["function"]["parameters"],
        _tools_to_openai_responses([tool])[0]["parameters"],
    ):
        assert published == schema
        Draft202012Validator.check_schema(published)
        errors = list(
            Draft202012Validator(published).iter_errors(
                {"op": "send", "to": target, "text": "synthetic"}
            )
        )
        assert (not errors) is valid
    if valid:
        assert (
            BridgeToolParams.model_validate({"op": "send", "to": target}).model_dump()["to"]
            == target
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["send", "reply"])
@pytest.mark.parametrize(
    "target",
    [
        {"account_id": "acc_2", "kind": "agent"},
        {"endpoint_id": "bep_9", "kind": "agent"},
        {"account_id": "acc_2", "typo": "ignored-before"},
        {"account_id": "acc_2", "endpoint_id": ""},
        {"account_id": "acc_2", "endpoint_id": "bep_9"},
        {"account_id": "   "},
        {},
    ],
)
async def test_send_recipient_is_closed_before_outbox(tmp_path, monkeypatch, op, target) -> None:
    """A refused target cannot leave a retryable intent or dispatch a request."""
    from local_operator.team_bridge.outbox import Outbox

    records = []
    admissions = []
    original = Outbox.record_send

    def record(self, *args, **kwargs):
        records.append(kwargs)
        return original(self, *args, **kwargs)

    def admission(runtime, *, is_read=False):
        admissions.append(runtime)
        return None

    monkeypatch.setattr(Outbox, "record_send", record)
    monkeypatch.setattr(host_mod, "admission_refusal", admission)
    requests = []
    runtime = BridgeRuntime(
        client=_client(lambda request: (requests.append(request), _ok({}))[1]),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )
    result = await execute_bridge(
        runtime, {"op": op, "to": target, "text": "synthetic", "thread_id": "bth_1"}
    )
    assert result["ok"] is False and result["code"] == "invalid_request"
    assert records == []
    assert admissions == []
    assert requests == []


@pytest.mark.asyncio
async def test_a_reply_pairs_thread_and_to_and_the_send_confirms_in_the_outbox(tmp_path) -> None:
    requests: list[httpx.Request] = []
    runtime = BridgeRuntime(
        client=_client(
            lambda request: (
                requests.append(request),
                _ok({"message": {"id": "bmsg_9"}}, status=201),
            )[1]
        ),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="s1",
    )
    missing_thread = await execute_bridge(
        runtime, {"op": "reply", "to": {"account_id": "acc_2"}, "text": "hi"}
    )
    assert missing_thread["ok"] is False and requests == []

    sent = await execute_bridge(
        runtime,
        {
            "op": "reply",
            "to": {"account_id": "acc_2"},
            "thread_id": "bth_1",
            "text": "hi",
            "type": "note",
        },
    )
    assert sent["ok"] is True and sent["message_id"] == "bmsg_9"
    body = json.loads(requests[0].content)
    assert body["thread_id"] == "bth_1" and body["to"] == {"account_id": "acc_2"}
    assert body["client_ref"]

    from local_operator.team_bridge.outbox import Outbox

    assert Outbox(tmp_path).pending(runtime.key()) == []
