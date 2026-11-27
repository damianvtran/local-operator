"""Exercise the tool and persisted fence against synthetic hub and OAuth state.

The client uses MockTransport, with no socket or live credentials. Every GET
is recorded to pin the contract without touching real rooms or transports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.aida.state import write_state
from local_operator.harness.types import ToolContext
from local_operator.providers.auth_store import AuthStore
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.host import admit_bridge_runtime
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import (
    BridgeRuntime,
    build_bridge_tool,
    execute_bridge,
)
from local_operator.tools.tool_docs import render_tool_doc
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
STATUS_PATH = "/v1/messaging/status"
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
    base: str = "https://roster.fixture.test/v1"
    calls: list[tuple[str, str, str, int]] = field(default_factory=list)
    page: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    list_status: int = 200
    detail_status: int = 200
    after_list: Callable[[], None] | None = None
    after_detail: Callable[[], None] | None = None

    def answer(self, req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"  # neither discovery nor roster may mutate a room
        path, query = req.url.path, req.url.query.decode("ascii")
        is_list = path == LIST_PATH
        if req.headers.get("Authorization") != "Bearer fixture-only":
            status, body = 401, {"code": "unauthorized"}
        elif query != (f"tenant_id={TENANT}&limit=50" if is_list else f"tenant_id={TENANT}"):
            status, body = 403, {"code": "invalid_tenant"}
        else:
            status = self.list_status if is_list else self.detail_status
            body = (
                {"msg": "ok", "result": self.page if is_list else self.details}
                if status == 200
                else {"code": "conversation_not_found", "error": "fixture-private-name"}
            )
        self.calls.append((req.method, path, query, status))
        hook = self.after_list if is_list else self.after_detail
        if hook is not None:
            hook()
        logger.debug("fixture request GET %s?%s => %s %s", path, query, status, json.dumps(body))
        return httpx.Response(status, json=body)


@pytest.fixture
def hub() -> WireHub:
    return WireHub(
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


@pytest.fixture
def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: WireHub
) -> tuple[BridgeRuntime, FakeStore]:
    # Override the normal destination allowlist only for the synthetic fixture;
    # the real client still checks exact enrollment, authority and bearer pin.
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
    client = TeamBridgeClient(
        tmp_path,
        base_url=hub.base,
        credential_id=1,
        store=auth,
        transport=httpx.MockTransport(hub.answer),
    )
    return BridgeRuntime(client, tmp_path, TENANT, owner, COS, admission="admitted"), auth


async def roster(runtime: BridgeRuntime, channel_id: str = ROOM) -> dict[str, Any]:
    return await execute_bridge(runtime, {"op": "channel_members", "channel_id": channel_id})


async def list_rooms(runtime: BridgeRuntime) -> None:
    outcome = await execute_bridge(runtime, {"op": "channels"})
    assert outcome["ok"], outcome


def status_result(runtime: BridgeRuntime) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "msg": "ok",
            "result": {
                "api_version": "1",
                "tenants": [
                    {
                        "tenant_id": runtime.tenant_id,
                        "eligible": True,
                        "endpoint": {"id": runtime.owner.endpoint_id, "state": "active"},
                    }
                ],
            },
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("challenge", [False, True])
async def test_designated_channels_recovers_after_offline_boot(runtime, hub, challenge):
    rt, auth = runtime
    rt.admission = "pending"
    retry_started, reply_ready = asyncio.Event(), asyncio.Event()
    status_calls = 0

    async def answer(req: httpx.Request) -> httpx.Response:
        nonlocal status_calls
        if req.url.path != STATUS_PATH:
            return hub.answer(req)
        assert req.method == "GET" and req.headers["Authorization"] == "Bearer fixture-only"
        status_calls += 1
        if status_calls == 1:
            raise httpx.ConnectError("fixture offline at boot", request=req)
        retry_started.set()
        await reply_ready.wait()
        if challenge and status_calls == 2:
            return httpx.Response(401)
        return status_result(rt)

    rt.client = TeamBridgeClient(
        rt.config_dir,
        base_url=hub.base,
        credential_id=1,
        store=auth,
        transport=httpx.MockTransport(answer),
    )
    assert await admit_bridge_runtime(rt) == "pending"
    assert rt.admission_note == "unreachable" and status_calls == 1
    assert (await roster(rt))["code"] == "conversation_not_found"
    assert status_calls == 1 and hub.calls == []  # detail never starts a retry or list

    cos_context = ToolContext(bridge_runtime=rt, session_id=COS)
    child_context = ToolContext(
        bridge_runtime=rt, session_id=COS, ancestor_session_ids=("sess_parent",)
    )
    cos_tool, child_tool = build_bridge_tool(cos_context), build_bridge_tool(child_context)
    assert cos_tool is not None and child_tool is not None
    boot_task = rt._admission_task
    for inherited in (child_tool, cos_tool):
        refused = await inherited.execute(
            "child-list", {"op": "channels"}, None, None, child_context
        )
        assert refused.details is not None and refused.details["code"] == "bridge_not_ready"
        assert rt._admission_task is boot_task and status_calls == 1

    refused = await cos_tool.execute("cos-list", {"op": "channels"}, None, None, cos_context)
    assert refused.details is not None and refused.details["code"] == "bridge_not_ready"
    attempt = rt._admission_task
    assert attempt is not None and not attempt.done()  # the op edge re-arms the boot slot
    try:
        await asyncio.wait_for(retry_started.wait(), timeout=5)
        for _ in range(3):
            assert (await execute_bridge(rt, {"op": "channels"}))["code"] == "bridge_not_ready"
            assert rt._admission_task is attempt
        assert status_calls == 2 and hub.calls == [] and rt.roster_discovery is None
    finally:
        reply_ready.set()
        await attempt

    assert rt.admission == "admitted"
    assert status_calls == (3 if challenge else 2)
    assert auth.refresh_calls == [(1, False), (1, False)] + ([(1, True)] if challenge else [])
    assert hub.calls == [] and rt.roster_discovery is None  # status is not a room read
    await list_rooms(rt)
    assert rt.roster_discovery is not None and ROOM in rt.roster_discovery.room_kinds
    assert [call[1] for call in hub.calls] == [LIST_PATH]
    assert (await roster(rt))["ok"]
    assert [call[1] for call in hub.calls] == [LIST_PATH, f"{LIST_PATH}/{ROOM}"]


@pytest.mark.asyncio
async def test_pending_roster_never_schedules_even_with_a_classified_room(runtime, hub):
    rt, auth = runtime
    await list_rooms(rt)
    rt.admission = "pending"

    assert (await roster(rt))["code"] == "bridge_not_ready"
    assert rt._admission_task is None
    assert [call[1] for call in hub.calls] == [LIST_PATH]
    assert auth.refresh_calls == [(1, False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["consent", "different_base", "different_pin", "account"])
async def test_pending_channels_with_invalid_binding_does_not_schedule(
    runtime, hub, monkeypatch, drift
):
    rt, auth = runtime
    rt.admission = "pending"
    code = "bridge_not_ready"
    if drift == "consent":
        with Store(rt.config_dir)._connect() as conn:
            conn.execute("UPDATE enrollments SET consent_version=NULL, consent_ref=NULL")
        code = "consent_revoked"
    elif drift == "account":
        auth.rows[0].data["account_id"] = "acc_wrong"
    else:
        # An otherwise allowed origin and a same-account sibling pin are not
        # the enrollment's actual destination and credential.
        monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda _: True)
        base = "https://alternate.fixture.test/v1" if drift == "different_base" else hub.base
        auth.rows.append(FakeRow(2, data=dict(auth.rows[0].data)))
        rt.client = TeamBridgeClient(
            rt.config_dir,
            base_url=base,
            credential_id=2 if drift == "different_pin" else 1,
            store=auth,
            transport=httpx.MockTransport(hub.answer),
        )
        if drift == "different_base":
            code = "bridge_destination_refused"
    assert (await execute_bridge(rt, {"op": "channels"}))["code"] == code
    assert rt._admission_task is None
    assert rt.admission == "pending" and rt.roster_discovery is None
    assert hub.calls == [] and auth.refresh_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status,drift", [(401, "consent"), (200, "designation"), (200, "account")])
async def test_pending_channels_discards_status_after_binding_changes(runtime, hub, status, drift):
    rt, auth = runtime
    rt.admission = "pending"
    status_calls = 0

    def answer(req: httpx.Request) -> httpx.Response:
        nonlocal status_calls
        assert req.method == "GET" and req.url.path == STATUS_PATH
        status_calls += 1
        if drift == "consent":
            with Store(rt.config_dir)._connect() as conn:
                conn.execute("UPDATE enrollments SET consent_version=NULL, consent_ref=NULL")
        elif drift == "designation":
            Store(rt.config_dir).update_designated_session(rt.owner, "sess_other_cos")
        else:
            auth.rows[0].data["account_id"] = "acc_wrong"
        return httpx.Response(401) if status == 401 else status_result(rt)

    rt.client = TeamBridgeClient(
        rt.config_dir,
        base_url=hub.base,
        credential_id=1,
        store=auth,
        transport=httpx.MockTransport(answer),
    )
    assert (await execute_bridge(rt, {"op": "channels"}))["code"] == "bridge_not_ready"
    attempt = rt._admission_task
    assert attempt is not None
    await attempt
    assert rt.admission == "pending" and rt.roster_discovery is None
    assert status_calls == 1 and auth.refresh_calls == [(1, False)]  # no forced refresh/retry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    (
        "other_session",
        "stale_designation",
        "inherited",
        "missing",
        "null",
        "empty_list",
        "empty_str",
    ),
)
async def test_roster_tool_requires_positive_root_authority(runtime, hub, case):
    rt, _ = runtime
    root = ToolContext(bridge_runtime=rt, session_id=COS)
    privileged = build_bridge_tool(root)
    assert privileged is not None
    assert "- channel_members:" in render_tool_doc(privileged)
    await list_rooms(rt)
    prior_calls = hub.calls.copy()

    if case == "other_session":
        denied = root.model_copy(update={"session_id": "sess_other"})
    elif case == "stale_designation":
        write_state(rt.config_dir, {"session_id": "sess_other"})
        denied = root
    else:
        # Simulate corrupt host state without Pydantic normalizing supplied values.
        ancestry = {"inherited": ("sess_parent",), "null": None, "empty_list": [], "empty_str": ""}
        denied = root.model_copy(update={"ancestor_session_ids": ancestry.get(case)})
        if case == "missing":
            del denied.ancestor_session_ids
            assert not hasattr(denied, "ancestor_session_ids")

    ordinary = build_bridge_tool(denied)
    assert ordinary is not None
    ops = ordinary.parameters["properties"]["op"]["enum"]
    assert "channels" in ops and "channel_members" not in ops
    help_text = render_tool_doc(ordinary)
    assert "- channels:" in help_text and "channel_members" not in help_text

    # Even a forged call to a previously privileged tool cannot reach the hub.
    args = {"op": "channel_members", "channel_id": ROOM}
    for tool, context in ((ordinary, denied), (privileged, denied), (ordinary, root)):
        assert tool.execute is not None
        result = await tool.execute("forged-roster", args, None, None, context)
        assert result.is_error and result.details is not None
        assert result.details["code"] == "not_chief_of_staff"
    assert hub.calls == prior_calls


@pytest.mark.asyncio
async def test_roster_through_actual_tool_uses_exact_hub_state_and_minimal_fields(runtime, hub):
    rt, _ = runtime
    ctx = ToolContext(bridge_runtime=rt, session_id=COS)
    tool = build_bridge_tool(ctx)
    assert tool is not None
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"op": "channel_members"}) == "read"
    assert "channel_members" in tool.parameters["properties"]["op"]["enum"]
    child_ctx = ToolContext(
        bridge_runtime=rt, session_id=COS, ancestor_session_ids=("sess_parent",)
    )
    child = build_bridge_tool(child_ctx)
    assert child is not None
    expected = json.loads(json.dumps(tool.parameters))
    expected["properties"]["op"]["enum"].remove("channel_members")
    assert child.parameters == expected

    await list_rooms(rt)
    # Schema visibility is not authority: a forwarded call with inherited
    # runtime and CoS ID still must not consume the parent's certified page.
    assert child.execute is not None
    for receiver, call_ctx in ((child, child_ctx), (tool, child_ctx)):
        assert receiver.execute is not None
        refused = await receiver.execute(
            "child-roster", {"op": "channel_members", "channel_id": ROOM}, None, None, call_ctx
        )
        assert refused.is_error and refused.details is not None
        assert refused.details["code"] == "not_chief_of_staff"
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

    async def designate_elsewhere(credential_id: int, *, force: bool = False, row_guard=None):
        result = await fresh(credential_id, force=force, row_guard=row_guard)
        write_state(rt.config_dir, {"session_id": "sess_other_cos"})
        return result

    with monkeypatch.context() as mp:
        mp.setattr(auth, "ensure_oauth_fresh_or_raise", designate_elsewhere)
        assert (await roster(rt))["code"] == "not_chief_of_staff"
    assert len(hub.calls) == 1  # just the explicit list; no roster GET left the client

    write_state(rt.config_dir, {"session_id": COS})
    await list_rooms(rt)  # the observed authority loss invalidated the prior certification
    hub.after_detail = lambda: write_state(rt.config_dir, {"session_id": "sess_other_cos"})
    refusal = await roster(rt)
    assert refusal["code"] == "not_chief_of_staff"
    assert "content" not in refusal
    assert hub.calls[-1] == ("GET", f"{LIST_PATH}/{ROOM}", f"tenant_id={TENANT}", 200)


@pytest.mark.asyncio
async def test_discovery_drops_the_page_on_designation_or_generation_drift(runtime, hub):
    rt, auth = runtime
    native = Store(rt.config_dir)
    await list_rooms(rt)
    hub.after_list = lambda: native.update_designated_session(rt.owner, "sess_other_cos")
    outcome = await execute_bridge(rt, {"op": "channels"})
    assert outcome["code"] == "not_designated" and "content" not in outcome
    assert rt.roster_discovery is None
    native.update_designated_session(rt.owner, COS)
    before = (len(hub.calls), len(auth.refresh_calls))
    assert (await roster(rt))["code"] == "conversation_not_found"
    assert before == (len(hub.calls), len(auth.refresh_calls))

    rebound = replace(rt.owner, enrollment_generation=2)
    hub.after_list = lambda: native.commit_enrollment(
        rebound, OwnerReceipt("bridge_consent_v1", "other-incarnation"), cos_session_id=COS
    )
    outcome = await execute_bridge(rt, {"op": "channels"})
    assert outcome["code"] == "owner_changed" and "content" not in outcome
    assert rt.roster_discovery is None


@pytest.mark.asyncio
async def test_discovery_rechecks_authority_before_dispatch(runtime, hub, monkeypatch):
    rt, auth = runtime
    fresh = auth.ensure_oauth_fresh_or_raise

    async def move_designation(credential_id: int, *, force: bool = False, row_guard=None):
        result = await fresh(credential_id, force=force, row_guard=row_guard)
        Store(rt.config_dir).update_designated_session(rt.owner, "sess_other_cos")
        return result

    monkeypatch.setattr(auth, "ensure_oauth_fresh_or_raise", move_designation)
    outcome = await execute_bridge(rt, {"op": "channels"})
    assert outcome["code"] == "not_designated"
    assert hub.calls == [] and rt.roster_discovery is None


@pytest.mark.asyncio
async def test_allowed_but_unenrolled_destination_never_resolves_credentials(
    runtime, hub, monkeypatch
):
    rt, auth = runtime
    await list_rooms(rt)
    before = (auth.resolve_requests, len(auth.refresh_calls), len(hub.calls))
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda url: True)
    # Same origin alone is insufficient: the canonical API base includes its path.
    for base in ("https://other.fixture.test/v1", "https://roster.fixture.test/v2"):
        swapped = replace(
            rt,
            client=TeamBridgeClient(
                rt.config_dir,
                base_url=base,
                credential_id=1,
                store=auth,
                transport=httpx.MockTransport(hub.answer),
            ),
        )
        assert (await roster(swapped))["code"] == "bridge_destination_refused"
        assert (await execute_bridge(swapped, {"op": "channels"}))[
            "code"
        ] == "bridge_destination_refused"
        assert swapped.roster_discovery is None
    assert before == (auth.resolve_requests, len(auth.refresh_calls), len(hub.calls))


@pytest.mark.asyncio
async def test_discovery_is_bound_to_exact_client_and_credential_row(runtime, hub):
    rt, auth = runtime
    await list_rooms(rt)
    auth.rows.append(FakeRow(2, data={"access": "fixture-only", "account_id": OWNER}))
    before = (auth.resolve_requests, len(auth.refresh_calls), len(hub.calls))
    for pin, code in ((1, "conversation_not_found"), (2, "bridge_signin_required")):
        swapped = replace(
            rt,
            client=TeamBridgeClient(
                rt.config_dir,
                base_url=hub.base,
                credential_id=pin,
                store=auth,
                transport=httpx.MockTransport(hub.answer),
            ),
        )
        assert (await roster(swapped))["code"] == "conversation_not_found"
        if pin == 2:
            assert (await execute_bridge(swapped, {"op": "channels"}))["code"] == code
    assert before == (auth.resolve_requests, len(auth.refresh_calls), len(hub.calls))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["channels", "channel_members"])
@pytest.mark.parametrize("drift", ["account", "designation", "consent"])
async def test_401_revalidates_authority_and_exact_row_before_refresh(
    runtime, hub, operation, drift
):
    rt, auth = runtime
    if operation == "channel_members":
        await list_rooms(rt)
    before, fresh_before = len(hub.calls), len(auth.refresh_calls)

    def change_binding() -> None:
        if drift == "account":
            auth.rows[0].data["account_id"] = "acc_wrong"
        elif drift == "designation":
            Store(rt.config_dir).update_designated_session(rt.owner, "sess_other_cos")
        else:
            with Store(rt.config_dir)._connect() as conn:
                conn.execute("UPDATE enrollments SET consent_version=NULL, consent_ref=NULL")

    hub.list_status = hub.detail_status = 401
    hub.after_list = hub.after_detail = change_binding
    outcome = await execute_bridge(rt, {"op": operation, "channel_id": ROOM})
    assert (
        outcome["code"]
        == {
            "account": "account_mismatch",
            "designation": "not_designated",
            "consent": "consent_revoked",
        }[drift]
    )
    assert "content" not in outcome
    assert [request[-1] for request in hub.calls[before:]] == [401]
    assert auth.refresh_calls[fresh_before:] == [(1, False)]  # no forced refresh


@pytest.mark.asyncio
async def test_wrong_expired_account_is_refused_before_store_freshness(runtime, hub, monkeypatch):
    rt, auth = runtime
    await list_rooms(rt)
    before = len(hub.calls)
    auth.rows[0].data.update(account_id="acc_wrong", expires=0, refresh="fixture-grant")

    async def unexpected_freshness(*args, **kwargs):
        pytest.fail("another account's exact row must not enter freshness")

    monkeypatch.setattr(auth, "ensure_oauth_fresh_or_raise", unexpected_freshness)
    assert (await roster(rt))["code"] == "account_mismatch"
    assert (await execute_bridge(rt, {"op": "channels"}))["code"] == "account_mismatch"
    assert len(hub.calls) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["account", "designation", None])
async def test_roster_refresh_rechecks_the_real_store_snapshot_under_lock(
    runtime, hub, monkeypatch, drift
):
    rt, _ = runtime
    auth = AuthStore(db_path=Path(rt.config_dir) / "fixture-auth.db")
    row = auth.upsert_credential(
        "radient",
        {"access": "fixture-only", "refresh": "synthetic", "expires": 0, "account_id": OWNER},
    )
    assert row.id == rt.owner.credential_id
    rt.client = TeamBridgeClient(
        rt.config_dir,
        base_url=hub.base,
        credential_id=row.id,
        store=auth,
        transport=httpx.MockTransport(hub.answer),
    )
    requested_refresh = asyncio.Event()
    original = auth._refresh_lock_for
    calls: list[str] = []

    def observed_lock(credential_id: int):
        requested_refresh.set()
        return original(credential_id)

    async def synthetic_refresh(creds):
        calls.append(creds["account_id"])
        return {"access": "fixture-only", "expires": int(time.time() * 1000) + 3600_000}

    monkeypatch.setattr(auth, "_refresh_lock_for", observed_lock)
    monkeypatch.setattr(auth, "_refresh_fn", lambda provider: synthetic_refresh)
    try:
        async with asyncio.TaskGroup() as tasks:
            async with original(row.id):
                listing = tasks.create_task(execute_bridge(rt, {"op": "channels"}))
                await asyncio.wait_for(requested_refresh.wait(), timeout=5)
                if drift == "account":
                    assert auth._update_payload(
                        row.id, lambda data: data.update(account_id="acc_wrong") is None
                    )
                elif drift == "designation":
                    Store(rt.config_dir).update_designated_session(rt.owner, "sess_other_cos")
        outcome = listing.result()
        if drift is None:
            assert calls == [OWNER]
            assert outcome["ok"] and rt.roster_discovery is not None
            assert [call[-1] for call in hub.calls] == [200]
        else:
            assert outcome["code"] == (
                "account_mismatch" if drift == "account" else "not_designated"
            )
            assert calls == [] and hub.calls == [] and rt.roster_discovery is None
    finally:
        auth.close()


@pytest.mark.asyncio
async def test_room_scope_is_only_the_latest_explicit_page(runtime, hub):
    rt, _ = runtime
    await list_rooms(rt)
    hub.page["conversations"] = []
    await list_rooms(rt)
    assert (await roster(rt))["code"] == "conversation_not_found"
    assert len(hub.calls) == 2
    assert all(call[1] == LIST_PATH for call in hub.calls)
