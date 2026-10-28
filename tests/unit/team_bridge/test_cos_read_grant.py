"""The chief-of-staff read grant is a HUMAN act: no model path may change it.

WHY THIS FILE EXISTS (review round 1 MAJOR 4). The bridge tool must never be able
to widen its own reach by sharing one of its owner's conversations with itself.
That is enforced in the client by a dispatch-time path gate
(``_refuse_unscoped_owner_audit``'s PUT/DELETE arm plus the
``_COS_READ_GRANT_SCOPE`` contextvar) rather than by convention, because a
convention is exactly what a later refactor drops. Two properties, both pinned
here so the gate cannot be lost silently:

1. A ``_send``/``_dispatch`` on ``PUT|DELETE …/cos-read`` from anywhere outside
   the two human-only methods raises ``cos_read_grant_not_in_scope`` before a
   byte moves — including the exact URL shapes a model could hand-build
   (percent-encoded, doubled slashes, upper-case, a dot segment).
2. The real methods still work: ``cos_read_grant`` / ``cos_read_revoke`` reach
   the wire.

A third property is static: the grant method names appear only in the modules
allowed to name them — the client that defines them and the desktop route family
that serves the renderer. The bridge tool, read_context and every other
agent-context path must not appear in that set, and the tool's op table must not
contain an op that could reach them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

import local_operator
from local_operator.harness.types import ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from local_operator.team_bridge.tool import BRIDGE_OPS, build_bridge_tool
from tests.unit.team_bridge.test_channel_tool import RecordingClient, _runtime
from tests.unit.team_bridge.test_client import FakeStore, _ok

PACKAGE = Path(local_operator.__file__).parent

#: The only modules allowed to name the grant surface: the client defines it and
#: the desktop routes serve the renderer. The bridge tool is NOT here.
GRANT_CALLERS = {
    "team_bridge/client.py",
    "server/routes/desktop_team_bridge.py",
}
GRANT_METHODS = ("cos_read_grant", "cos_read_revoke")


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the destination allow-list: this file tests the GRANT path gate.

    Without it the transport refuses `https://hub.example` first, with
    `bridge_destination_refused`, and every assertion below would be measuring the
    wrong refusal (found by running the file).
    """

    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


class _Recorder:
    """A transport that records what actually reached the wire."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return _ok({"conversation_id": "bcv_abcd1234", "cos_read": {"endpoint_id": "bep_1"}})


def _client(tmp_path: Path, recorder: _Recorder) -> TeamBridgeClient:
    return TeamBridgeClient(
        tmp_path,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(recorder.handler),
    )


@pytest.mark.parametrize(
    "path",
    [
        "/messaging/conversations/bcv_abcd1234/cos-read",
        # A model can spell a URL; the gate must read every one of these as the
        # same route the hub does.
        "//messaging//conversations/bcv_abcd1234/cos-read",
        "/messaging/conversations/bcv_abcd1234/COS-READ",
        "/messaging/conversations/%62cv_abcd1234/cos-read",
        "/messaging/conversations/bcv_abcd1234/cos-read/",
    ],
)
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["PUT", "DELETE"])
async def test_a_hand_built_grant_url_is_refused_before_any_request(
    tmp_path: Path, method: str, path: str
) -> None:
    recorder = _Recorder()
    client = _client(tmp_path, recorder)

    with pytest.raises(BridgeRefusal) as refused:
        await client._send(method, path, body={"tenant_id": "ten_1"})

    assert refusal_code(refused.value) == "cos_read_grant_not_in_scope"
    assert recorder.requests == [], "the refusal must come before any dispatch"


def refusal_code(refusal: BaseException) -> str:
    return str(getattr(refusal, "code", ""))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["PUT", "DELETE"])
async def test_a_dot_segment_cannot_reach_the_grant_route(tmp_path: Path, method: str) -> None:
    recorder = _Recorder()
    client = _client(tmp_path, recorder)

    with pytest.raises(BridgeRefusal) as refused:
        await client._send(
            method, "/messaging/conversations/../conversations/bcv_abcd1234/cos-read"
        )

    assert refusal_code(refused.value) in ("invalid_request", "cos_read_grant_not_in_scope")
    assert recorder.requests == []


@pytest.mark.asyncio
async def test_the_human_methods_still_reach_the_wire(tmp_path: Path) -> None:
    recorder = _Recorder()
    client = _client(tmp_path, recorder)

    await client.cos_read_grant("bcv_abcd1234", tenant_id="ten_1", expected_account_id="acc_1")
    assert [r.method for r in recorder.requests] == ["PUT"]
    assert recorder.requests[0].url.path.endswith("/conversations/bcv_abcd1234/cos-read")
    # The PUT carries the tenant in its BODY (a strict-decoded write), the DELETE in
    # the query string (a DELETE body is not reliably carried).
    import json as _json

    assert _json.loads(recorder.requests[0].content)["tenant_id"] == "ten_1"

    await client.cos_read_revoke("bcv_abcd1234", tenant_id="ten_1", expected_account_id="acc_1")
    assert [r.method for r in recorder.requests] == ["PUT", "DELETE"]
    assert "tenant_id=ten_1" in str(recorder.requests[1].url)


@pytest.mark.asyncio
async def test_reads_are_not_gated_but_carry_the_captured_owner(tmp_path: Path) -> None:
    """The agent-side reads are ordinary GETs; they must still name the owner."""

    recorder = _Recorder()
    client = _client(tmp_path, recorder)

    await client.cos_read_conversations(
        actor_endpoint_id="bep_1", tenant_id="ten_1", expected_account_id="acc_1"
    )
    await client.cos_read_messages(
        "bcv_abcd1234",
        actor_endpoint_id="bep_1",
        tenant_id="ten_1",
        expected_account_id="acc_1",
    )
    assert [r.url.path for r in recorder.requests] == [
        "/v1/messaging/cos-read/conversations",
        "/v1/messaging/conversations/bcv_abcd1234/cos-read/messages",
    ]


def test_no_bridge_tool_op_can_reach_a_grant_method() -> None:
    """Static half: the agent surface never names the human-only methods."""

    # The op table is the only thing a model can invoke, and its names are not
    # the grant methods, so no op dispatches there by construction.
    assert not set(BRIDGE_OPS) & set(GRANT_METHODS)

    offenders: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE).as_posix()
        if relative in GRANT_CALLERS:
            continue
        source = path.read_text(encoding="utf-8")
        if any(method in source for method in GRANT_METHODS):
            offenders.append(relative)
    assert offenders == [], f"agent-context modules named a human-only grant method: {offenders}"


def test_the_desktop_grant_routes_are_the_documented_path() -> None:
    """The tool description must point at tool://bridge, never at a grant API."""

    tool = build_bridge_tool(ToolContext(bridge_runtime=_runtime(RecordingClient())))
    assert tool is not None
    description: Any = tool.description
    assert "read tool://bridge" in description
    assert "cos_read" not in description
