"""Run the bridge tool, persisted authority fence, and real client over loopback.

The synthetic hub only serves GETs. Record every request to prove the new read
never probes an unclassified id or runs a collection preflight of its own.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlsplit

import pytest

from local_operator.aida.state import write_state
from local_operator.harness.types import ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import (
    BridgeRuntime,
    build_bridge_tool,
    execute_bridge,
)
from tests.unit.team_bridge.test_client import FakeRow, FakeStore

ROOM = "bcv_testroom1"
GROUP = "bcv_testgroup"
BUILTIN = "bcv_testbuiltin"
DM = "bcv_testdm01"
FOREIGN = "bcv_testforeign"
ABSENT = "bcv_testabsent"
TENANT = "tnt_roster"
OWNER = "acc_roster_owner"
COS = "sess_roster_cos"
LIST_PATH = "/v1/messaging/conversations"
logger = logging.getLogger(__name__)


def room(room_id: str = ROOM, *, kind: str = "channel", tenant: str = TENANT) -> dict[str, Any]:
    return {
        "id": room_id,
        "tenant_id": tenant,
        "kind": kind,
        "status": "open",
        "member": True,
        "discoverable": False,
    }


def member(account: str, *, state: str = "active", **extra: Any) -> dict[str, Any]:
    return {"account_id": account, "kind": "human", "role": "member", "state": state, **extra}


@dataclass
class WireHub:
    base: str = ""
    calls: list[tuple[str, str, str, int]] = field(default_factory=list)
    page: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    detail_status: int = 200
    before_detail: Callable[[], None] | None = None


@pytest.fixture
def hub() -> Iterator[WireHub]:
    hub = WireHub(
        page={
            "conversations": [
                room(),
                room(GROUP, kind="group"),
                {**room(BUILTIN), "builtin": "general"},
                room(DM, kind="dm"),
                room(FOREIGN, tenant="tnt_other"),
            ]
        },
        details={
            **room(),
            "name": "synthetic name to omit",
            "topic": "synthetic topic to omit",
            "notify": "mute",
            "last_read_seq": 100,
            "members": [
                member(OWNER, role="owner", display_label="private-label@example.test"),
                member(OWNER, kind="agent", state="invited", endpoint_id="bep_roster1"),
                member("acc_other", kind="agent", endpoint_id="bep_roster2", notify="all"),
                member("acc_left", state="left"),
                member("acc_removed", state="removed"),
            ],
        },
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            req = urlsplit(self.path)
            if self.headers.get("Authorization") != "Bearer fixture-only":
                status, body = 401, {"code": "unauthorized"}
            elif req.query != f"tenant_id={TENANT}" and not (
                req.path == LIST_PATH and req.query == f"tenant_id={TENANT}&limit=50"
            ):
                status, body = 403, {"code": "invalid_tenant"}
            elif req.path == LIST_PATH:
                status, body = 200, {"msg": "ok", "result": hub.page}
            else:
                if hub.before_detail is not None:
                    hub.before_detail()
                status = hub.detail_status
                body = (
                    {"msg": "ok", "result": hub.details}
                    if status == 200
                    else {"code": "conversation_not_found", "error": "fixture-private-name"}
                )
            hub.calls.append((self.command, req.path, req.query, status))
            payload = json.dumps(body).encode("utf-8")
            logger.debug("fixture wire GET %s => %s %s", self.path, status, payload.decode())
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # The structured calls above deliberately omit auth headers.

    with HTTPServer(("127.0.0.1", 0), Handler) as http:
        hub.base = f"http://127.0.0.1:{http.server_port}/v1"
        thread = Thread(target=http.serve_forever)
        thread.start()
        try:
            yield hub
        finally:
            http.shutdown()
            thread.join()


@pytest.fixture
def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: WireHub
) -> tuple[BridgeRuntime, FakeStore]:
    # Override destination trust only for this exact synthetic TCP fixture; the
    # production client, local account/tenant fence and bearer pin still run.
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda url: url == hub.base)
    (tmp_path / "config.yml").write_text("values:\n  team_bridge:\n    enabled: true\n")
    write_state(tmp_path, {"session_id": COS})
    owner = OwnerSnapshot(
        credential_id=1,
        account_id=OWNER,
        tenant_id=TENANT,
        endpoint_id="bep_roster1",
        canonical_base_url=hub.base,
        enrollment_generation=1,
    )
    Store(tmp_path).commit_enrollment(
        owner, OwnerReceipt("bridge_consent_v1", "fixture"), cos_session_id=COS
    )
    auth = FakeStore(rows=[FakeRow(1, data={"access": "fixture-only", "account_id": OWNER})])
    client = TeamBridgeClient(tmp_path, base_url=hub.base, credential_id=1, store=auth)
    return BridgeRuntime(client, tmp_path, TENANT, owner, COS, admission="admitted"), auth


async def roster(runtime: BridgeRuntime, channel_id: str = ROOM) -> dict[str, Any]:
    return await execute_bridge(runtime, {"op": "channel_members", "channel_id": channel_id})


async def list_rooms(runtime: BridgeRuntime) -> None:
    outcome = await execute_bridge(runtime, {"op": "channels"})
    assert outcome["ok"], outcome


@pytest.mark.asyncio
async def test_roster_through_actual_tool_uses_exact_hub_state_and_minimal_fields(runtime, hub):
    rt, _ = runtime
    ctx = ToolContext(bridge_runtime=rt, session_id=COS)
    tool = build_bridge_tool(ctx)
    assert tool is not None
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"op": "channel_members"}) == "read"
    assert "channel_members" in tool.parameters["properties"]["op"]["enum"]
    child = build_bridge_tool(
        ToolContext(bridge_runtime=rt, session_id=COS, ancestor_session_ids=("sess_parent",))
    )
    assert child is not None
    expected = json.loads(json.dumps(tool.parameters))
    expected["properties"]["op"]["enum"].remove("channel_members")
    assert child.parameters == expected

    await list_rooms(rt)
    result = await tool.execute(
        "fixture-roster", {"op": "channel_members", "channel_id": ROOM}, None, None, ctx
    )
    roster_body = {
        "conversation_id": ROOM,
        "members": [
            member(OWNER, role="owner"),
            member(OWNER, kind="agent", state="pending", endpoint_id="bep_roster1"),
            member("acc_other", kind="agent", endpoint_id="bep_roster2"),
        ],
    }
    assert not result.is_error
    assert result.details == {
        "ok": True,
        "state": "ok",
        "conversation_id": ROOM,
        "content": json.dumps(roster_body),
    }
    assert result.content[0].model_dump()["text"] == json.dumps(roster_body)
    logger.debug("fixture model projection %s", result.content[0].model_dump()["text"])
    assert hub.calls == [
        ("GET", LIST_PATH, f"tenant_id={TENANT}&limit=50", 200),
        ("GET", f"{LIST_PATH}/{ROOM}", f"tenant_id={TENANT}", 200),
    ]

    # Same action works for groups; the body has no cached channel-member rows.
    hub.details.update(room(GROUP, kind="group"))
    hub.details["members"] = [member(OWNER)]
    group = await roster(rt, GROUP)
    assert json.loads(group["content"]) == {"conversation_id": GROUP, "members": [member(OWNER)]}
    assert hub.calls[-1] == ("GET", f"{LIST_PATH}/{GROUP}", f"tenant_id={TENANT}", 200)


@pytest.mark.asyncio
async def test_unclassified_rooms_and_wrong_session_are_offline(runtime, hub):
    rt, auth = runtime
    assert (await roster(rt))["code"] == "conversation_not_found"  # no implicit collection read
    assert auth.resolve_requests == 0
    await list_rooms(rt)
    auth_count = auth.resolve_requests
    request_count = len(hub.calls)
    task = replace(rt, session_id="sess_task")
    assert (await roster(task, "not-even-an-id"))["code"] == "not_designated"
    assert (await roster(replace(rt, tenant_id="tnt_other")))["code"] == "owner_changed"
    write_state(rt.config_dir, {"session_id": "sess_other_cos"})
    assert (await roster(rt))["code"] == "not_chief_of_staff"
    write_state(rt.config_dir, {"session_id": COS})
    hidden = await roster(rt, ABSENT)
    assert hidden["code"] == "conversation_not_found"
    for room_id in (BUILTIN, DM, FOREIGN):
        assert await roster(rt, room_id) == hidden
    for room_id in ("testroom1", f"{ROOM} ", ROOM.upper()):
        assert (await roster(rt, room_id))["code"] == "invalid_request"
    assert len(hub.calls) == request_count
    assert auth.resolve_requests == auth_count


@pytest.mark.asyncio
async def test_private_refusal_and_bad_binding_are_never_passed_through(runtime, hub):
    rt, auth = runtime
    await list_rooms(rt)
    opaque = await roster(rt, ABSENT)
    hub.detail_status = 404  # a member can lose access after the list snapshot
    assert await roster(rt) == opaque
    assert hub.calls[-1] == ("GET", f"{LIST_PATH}/{ROOM}", f"tenant_id={TENANT}", 404)

    # Even a 200 must bind to the requested room, tenant and active human caller.
    hub.detail_status = 200
    good = hub.details
    for updates in (
        {"id": GROUP},
        {"tenant_id": "tnt_other"},
        {"member": False},
        {"builtin": "general"},
        {"kind": "dm"},
        {"members": [member(OWNER, state="invited")]},
        {"members": [member(OWNER), member("acc_unknown", state="pending")]},
        {"members": [member(OWNER), {"account_id": "acc_unknown", "kind": "human"}]},
    ):
        hub.details = {**good, **updates}
        assert await roster(rt) == opaque
        assert hub.calls[-1] == ("GET", f"{LIST_PATH}/{ROOM}", f"tenant_id={TENANT}", 200)

    auth.rows[0].data["account_id"] = "acc_wrong"
    request_count = len(hub.calls)
    assert (await roster(rt))["code"] == "account_mismatch"
    assert len(hub.calls) == request_count
    auth.rows[0].data = {"access": "rejected-fixture-token", "account_id": OWNER}
    assert (await roster(rt))["code"] == "bridge_signin_required"
    assert hub.calls[request_count:] == [
        ("GET", f"{LIST_PATH}/{ROOM}", f"tenant_id={TENANT}", 401),
        ("GET", f"{LIST_PATH}/{ROOM}", f"tenant_id={TENANT}", 401),
    ]


@pytest.mark.asyncio
async def test_designation_is_rechecked_at_dispatch_and_after_response(runtime, hub, monkeypatch):
    rt, auth = runtime
    await list_rooms(rt)
    fresh = auth.ensure_oauth_fresh_or_raise

    async def designate_elsewhere(credential_id: int, *, force: bool = False):
        result = await fresh(credential_id, force=force)
        write_state(rt.config_dir, {"session_id": "sess_other_cos"})
        return result

    with monkeypatch.context() as mp:
        mp.setattr(auth, "ensure_oauth_fresh_or_raise", designate_elsewhere)
        assert (await roster(rt))["code"] == "not_chief_of_staff"
    assert len(hub.calls) == 1  # just the explicit list; no roster GET left the client

    write_state(rt.config_dir, {"session_id": COS})
    hub.before_detail = lambda: write_state(rt.config_dir, {"session_id": "sess_other_cos"})
    refusal = await roster(rt)
    assert refusal["code"] == "not_chief_of_staff"
    assert "content" not in refusal
    assert hub.calls[-1] == ("GET", f"{LIST_PATH}/{ROOM}", f"tenant_id={TENANT}", 200)


@pytest.mark.asyncio
async def test_room_scope_is_only_the_latest_explicit_page(runtime, hub):
    rt, _ = runtime
    await list_rooms(rt)
    hub.page["conversations"] = []
    await list_rooms(rt)
    assert (await roster(rt))["code"] == "conversation_not_found"
    assert len(hub.calls) == 2
    assert all(call[1] == LIST_PATH for call in hub.calls)
