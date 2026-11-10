"""Real signed continuations across busy sockets and unavailable team snapshots.

The socket ACK is only a hint receipt. Durable publication and one-shot inverse
outcomes must remain correct across request/session lifetime and storage faults.
"""

import asyncio
import contextlib
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException

from local_operator import profile_approval as P
from local_operator.agents import AgentRegistry
from local_operator.harness.approval import mint_operator_cap, remember_operator_cap
from local_operator.harness.types import (
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
    ToolContext,
    ToolExecutionEndEvent,
    ToolResult,
)
from local_operator.mobile.attach_client import AttachClient
from local_operator.mobile.peer_client import send_peer_message
from local_operator.network import approvals as A
from local_operator.network.types import MeshRefusal
from local_operator.profile_actions import ProfileActions
from local_operator.profile_scope import scope_agent_registry, scope_team_registry
from local_operator.server.routes import desktop_sessions as routes
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import FrontendStateStore
from local_operator.session.runtime.process import _drain_inbox_into
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import TeamEditFields, TeamRegistry
from local_operator.tools.agent_tool import execute_agent
from tests.unit import test_profile_approval as fixtures
from tests.unit.session.runtime import test_approval_authority_seam as seam
from tests.unit.session.test_account_profile_consent import _never_stream

root = fixtures.root
synthetic_presence = fixtures.synthetic_presence


def new_session(root: Path, directory: Path) -> Session:
    return Session(
        model=ModelSpec(provider="test", model_id="fail-fast", context_window=100000),
        stream_fn=_never_stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        agent_registry=scope_agent_registry(AgentRegistry(root, migrate=False), directory),
        team_registry=scope_team_registry(TeamRegistry(root), directory),
        owner_prompt_consent=True,
    )


async def file_legacy_signed_receipt(session: Session, directory: Path, call_id: str, args):
    """Mint a GENUINE legacy ``queued`` signed-continuation receipt.

    WHY: the ``agent`` tool no longer files signed mesh requests (an unasked
    account write is now a proposal in Agents → <name>), but signed requests
    filed before that change -- and their spooled continuations -- must still
    complete exactly once. So the receipt is built the way the pre-proposal
    host did (``Session._authorize_durable_write`` on main before PR #3704):
    the real tool computes the bounded ``intent.proposal`` through the consent
    seam, then ``profile_approval.build`` + ``file_request`` file it, and the
    tool result carries the legacy ``queued`` details. Nothing is labelled by
    hand except the receipt shape the old ``_queued_profile_receipt`` wrote.
    """
    captured = []

    async def capture(intent):
        captured.append(intent)
        return None  # Refuse: this call only derives the authored proposal.

    await execute_agent(
        call_id,
        args,
        None,
        None,
        ToolContext(
            cwd=".",
            session_dir=str(directory),
            agent_registry=session.agent_registry,
            authorize_durable_write=capture,
        ),
    )
    [intent] = captured
    assert intent.proposal is not None, "a bounded update must carry a proposal"
    root = Path(session.agent_registry.config_dir)
    data = P.build(root, session_id=session._session_id, tool_call_id=call_id, **intent.proposal)
    record = P.file_request(root, data)
    return ToolResult(
        tool_call_id=call_id,
        tool_name="agent",
        content=[TextContent(text=f"Queued agent {data.name!r} for durable approval.")],
        details={
            "status": "queued",
            "fields": list(intent.changes),
            "diff": intent.summary.splitlines(),
            "approval_id": record["approval_id"],
            "request_digest": record["request_digest"],
            "expires_at": record["expires_at"],
            "receipt_id": data.receipt_id,
            "profile_kind": data.profile_kind,
            "profile_name": data.name,
            "scope": "global",
        },
    )


def applied_rows(directory: Path, call_id: str):
    return [
        row
        for row in Transcript(directory).build_llm_history()
        if isinstance(row, Message)
        and row.tool_call_id == call_id
        and ((row.provider_payload or {}).get("details") or {}).get("status") == "applied"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("lifetime", ["normal", "duplicate", "cancel", "dispose"])
async def test_busy_actual_socket_hint_has_one_owned_continuation(root, monkeypatch, lifetime):
    fixtures.proposal(root)
    directory = root / "sessions" / "synthetic-session"
    session = new_session(root, directory)
    args = {"op": "update", "name": "reviewer", "instructions": "Signed while busy."}
    # The CURRENT host files this unasked update as an Agents proposal, never a
    # signed mesh request -- pinned here so the legacy rig below cannot drift
    # into describing the live path.
    current = await execute_agent(
        "busy-current-call",
        args,
        None,
        None,
        ToolContext(
            cwd=".",
            session_dir=str(directory),
            agent_registry=session.agent_registry,
            authorize_durable_write=session._authorize_durable_write,
        ),
    )
    assert current.details is not None and current.details["status"] == "proposed"
    assert "approval_id" not in current.details and "receipt_id" not in current.details
    queued = await file_legacy_signed_receipt(session, directory, "busy-real-call", args)
    details = queued.details
    assert details is not None and details["status"] == "queued"
    pair = [
        Message.assistant(
            tool_calls=[ToolCall(id=queued.tool_call_id, name="agent", arguments=args)]
        ),
        Message.tool_result(queued),
    ]
    await session.transcript.append_messages(pair)
    session._context.messages.extend(pair)
    record = A._load_raw(details["approval_id"], root)
    fixtures.approve(root, record)
    events = []
    session.subscribe(events.append)
    monkeypatch.setattr(seam, "_AttachableSession", lambda: session)
    live = await seam._serve(root, operator_cap=None)
    locked = False
    try:
        await session._turn_lock.acquire()
        locked = True
        assert P.apply(record["approval_id"], P.expected(record), root)["state"] == "complete"
        # A daemon dial replaces its predecessor by protocol design. Let the
        # automatic notifier ACK while still busy before manually duplicating
        # that dial; racing the two would test reconnect eviction, not receipt
        # ownership. Neither publication nor a forced inbox drain happens here.
        publications = tuple(P._live_publications)
        if publications:
            await asyncio.wait_for(asyncio.gather(*publications), 3)

        async def notify():
            return await send_peer_message(
                live.record,
                text="receipt ready",
                mode="mailbox",
                wake=False,
                sender={"source": "profile_save"},
                message_id=details["receipt_id"],
                custom_type="profile_save_result",
                deadline_s=0.5,
            )

        # ACK must not depend on the busy turn boundary or notifier lifetime.
        await asyncio.wait_for(notify(), 1)
        if lifetime == "duplicate":
            await asyncio.wait_for(notify(), 1)
        if lifetime == "cancel":
            from local_operator.session.runtime.serving import ServingSessionHandle

            acknowledged_in_owner = asyncio.Event()
            release_reply = asyncio.Event()
            assert isinstance(live.runtime._handle, ServingSessionHandle)
            original_hint = live.runtime._handle.receive_profile_save_result

            async def lose_transport_reply(receipt_id):
                await original_hint(receipt_id)
                acknowledged_in_owner.set()
                await release_reply.wait()  # Only the caller loses its reply.

            monkeypatch.setattr(
                live.runtime._handle, "receive_profile_save_result", lose_transport_reply
            )
            request = asyncio.create_task(notify())
            await asyncio.wait_for(acknowledged_in_owner.wait(), 3)
            request.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request
            # Finish the carrier-side rig too; an intentionally parked request
            # must not escape this case and interfere with the next live owner.
            release_reply.set()
            monkeypatch.setattr(live.runtime._handle, "receive_profile_save_result", original_hint)
            await asyncio.sleep(0)
        assert len(session._profile_result_waiters) == 1
        assert not [e for e in events if isinstance(e, ToolExecutionEndEvent)]
        assert not applied_rows(directory, queued.tool_call_id)
        if lifetime == "dispose":
            # Disposal cancels the owned waiter, never its durable spool.
            await asyncio.wait_for(session.dispose(), 3)
        session._turn_lock.release()
        locked = False
        if lifetime == "dispose":
            resumed = new_session(root, directory)
            resumed.subscribe(events.append)
            try:
                assert (
                    await _drain_inbox_into(
                        SimpleNamespace(
                            _session=resumed, receive_peer_message=resumed.receive_peer_message
                        )
                    )
                    == 1
                )
            finally:
                await resumed.dispose()
        else:
            # Wait for BOTH halves of delivery: ``receive_profile_save_result``
            # commits the applied row to the transcript and only then awaits
            # its ``ToolExecutionEndEvent`` emit, so polling the row alone
            # races the event count asserted below (observed locally as a
            # 1-in-4 ``0 == 1`` on the end-event count).
            async with asyncio.timeout(3):
                while not applied_rows(directory, queued.tool_call_id) or not [
                    e for e in events if isinstance(e, ToolExecutionEndEvent)
                ]:
                    await asyncio.sleep(0.01)
        assert len([e for e in events if isinstance(e, ToolExecutionEndEvent)]) == 1
        assert len(applied_rows(directory, queued.tool_call_id)) == 1
        assert not session._run_owner_texts and not session._run_typed_user
    finally:
        if locked:
            session._turn_lock.release()
        await live.close(root)


@contextlib.asynccontextmanager
async def http_owner(root, session, monkeypatch):
    monkeypatch.setattr(seam, "_AttachableSession", lambda: session)
    cap = mint_operator_cap()
    live = await seam._serve(root, operator_cap=cap)
    remember_operator_cap(live.record.pid, cap)
    wire = AttachClient(lambda _: None, lambda _: None, surface="desktop")
    await wire.connect(live.record, live.record.session_id)
    remote = AttachedSession(
        config_dir=root,
        session_id=live.record.session_id,
        takeover_factory=lambda: None,
        surface="desktop",
    )
    remote._client = wire
    remote._frontend_store = FrontendStateStore(session.frontend_state)

    @contextlib.asynccontextmanager
    async def bind(sid, **kwargs):
        if sid != live.record.session_id:
            raise HTTPException(404, "unknown fixture session")
        yield SimpleNamespace(remote=remote)

    app = FastAPI()
    app.state.desktop_sessions = SimpleNamespace(session=bind)
    app.state.config_manager = SimpleNamespace(config_dir=root)
    app.include_router(routes.router)
    token = "profile-continuation-fixture"
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
            headers={"Authorization": "Bearer profile-continuation-fixture"},
            timeout=5,
        ) as http:
            yield http, "/v1/desktop/sessions/" + live.record.session_id
    finally:
        server.should_exit = True
        await task
        sock.close()
        wire.close()
        await live.close(root)


@pytest.mark.asyncio
async def test_signed_team_inverse_unknown_is_closed_http_pending_and_never_rearmed(
    root, monkeypatch
):
    fixtures.proposal(root)
    registry = TeamRegistry(root)
    team = registry.create_team(
        TeamEditFields(name="lane", manager="reviewer", instructions="Before.")
    )
    before = P.snapshot(root, "team", team.id)
    assert before is not None
    proposal = P.build(
        root,
        profile_kind="team",
        operation="update",
        target_id=team.id,
        name=team.name,
        session_id="team-fault",
        tool_call_id="team-call",
        before=before,
        after=dict(before, instructions="Approved."),
    )
    record = P.file_request(root, proposal)
    fixtures.approve(root, record)
    action = P.apply(record["approval_id"], P.expected(record), root)["profile_action"]
    registry = TeamRegistry(root)
    registry.get_team(team.id)
    query = {
        key: action[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
    }
    query["tool_call_id"] = "team-call"
    directory = root / "sessions" / "team-fault"
    session = new_session(root, directory)
    replace = os.replace
    attempts = []

    def fail_publish(source, target):
        if Path(target).name == team.id and Path(source).name.startswith("."):
            attempts.append((Path(source).name, Path(target).name))
            raise OSError("primary signed TEAM publish/recovery fault")
        return replace(source, target)

    async with http_owner(root, session, monkeypatch) as (http, url):
        ready = await http.post(url + "/profile-action-status", json=query)
        assert ready.status_code == 200 and ready.json()["result"]["state"] == "available"
        with monkeypatch.context() as fault:
            fault.setattr(os, "replace", fail_publish)
            answer = await http.post(
                url + "/answers",
                json={
                    "epoch": session.frontend_state.epoch,
                    "request_id": action["action_id"],
                    "approved": True,
                },
            )
            assert answer.status_code == 200, answer.text
            state = await http.post(url + "/profile-action-status", json=query)
            assert state.status_code == 200 and state.json()["result"]["state"] == "pending"
            fresh = ProfileActions("team-fault", root)
            assert fresh.status(**query)["state"] == "pending"
            count = len(attempts)
            duplicate = await http.post(
                url + "/answers",
                json={
                    "epoch": session.frontend_state.epoch,
                    "request_id": action["action_id"],
                    "approved": True,
                },
            )
            assert duplicate.status_code == 200 and len(attempts) == count
            journal = root / "profile-transactions" / ("revert_" + action["action_id"] + ".json")
            assert json.loads(journal.read_text())["state"] == "applying"
        assert TeamRegistry(root).get_team(team.id).instructions == "Approved."
        assert registry.get_team(team.id).instructions == "Approved."
        recovered = await http.post(url + "/profile-action-status", json=query)
        assert recovered.status_code == 200 and recovered.json()["result"]["state"] == "pending"
        await http.post(
            url + "/answers",
            json={
                "epoch": session.frontend_state.epoch,
                "request_id": action["action_id"],
                "approved": True,
            },
        )
        assert TeamRegistry(root).get_team(team.id).instructions == "Approved."


def test_signed_team_save_preserves_primary_unknown_and_normal_recovery(root, monkeypatch):
    fixtures.proposal(root)
    registry = TeamRegistry(root)
    team = registry.create_team(
        TeamEditFields(name="save-fault", manager="reviewer", instructions="Before.")
    )
    before = P.snapshot(root, "team", team.id)
    assert before is not None
    proposal = P.build(
        root,
        profile_kind="team",
        operation="update",
        target_id=team.id,
        name=team.name,
        session_id="save-fault",
        tool_call_id="save-fault-call",
        before=before,
        after=dict(before, instructions="After."),
    )
    record = P.file_request(root, proposal)
    fixtures.approve(root, record)
    replace = os.replace

    def fail_publish(source, target):
        if Path(target).name == team.id and Path(source).name.startswith("."):
            raise OSError("primary signed TEAM save fault")
        return replace(source, target)

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", fail_publish)
        with pytest.raises(
            MeshRefusal, match="Failed to save team metadata: primary signed TEAM save fault"
        ) as raised:
            P.apply(record["approval_id"], P.expected(record), root)
        primary = raised.value.__cause__
        assert primary is not None
        assert "Failed to save team metadata" in str(primary)
        assert any("unconfirmed" in note for note in primary.__notes__)
        journal = json.loads(
            (root / "profile-transactions" / (proposal.receipt_id + ".json")).read_text()
        )
        assert journal["state"] == "prepared"
        paths = sorted(path.name for path in registry.teams_dir.iterdir())
        from local_operator.teams import read_team_snapshot
        from local_operator.types import TeamRegistryRecoveryError

        with pytest.raises(TeamRegistryRecoveryError):
            read_team_snapshot(registry.teams_dir, team.id)
        assert sorted(path.name for path in registry.teams_dir.iterdir()) == paths
    assert TeamRegistry(root).get_team(team.id).instructions == "Before."
    assert registry.get_team(team.id).instructions == "Before."


def test_prestart_team_unavailable_does_not_consume_and_terminal_success_is_cached(root):
    fixtures.proposal(root)
    registry = TeamRegistry(root)
    team = registry.create_team(
        TeamEditFields(name="prestart", manager="reviewer", instructions="Before.")
    )
    before = P.snapshot(root, "team", team.id)
    assert before is not None
    proposal = P.build(
        root,
        profile_kind="team",
        operation="update",
        target_id=team.id,
        name=team.name,
        session_id="prestart",
        tool_call_id="prestart-call",
        before=before,
        after=dict(before, instructions="After."),
    )
    record = P.file_request(root, proposal)
    fixtures.approve(root, record)
    action = P.apply(record["approval_id"], P.expected(record), root)["profile_action"]
    query = {
        key: action[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
    }
    query["tool_call_id"] = "prestart-call"
    owner = ProfileActions("prestart", root)
    assert owner.status(**query)["state"] == "available"
    journal = root / "profile-transactions" / ("revert_" + action["action_id"] + ".json")
    prepared = journal.read_bytes()
    target = root / "teams" / team.id
    backup = root / "teams" / ("." + team.id + ".backup.synthetic")
    target.rename(backup)
    assert owner.status(**query)["state"] == "unavailable"
    assert (
        owner.activate(action["action_id"], approved=True, remember=False)["state"] == "unavailable"
    )
    assert journal.read_bytes() == prepared
    backup.rename(target)
    assert owner.status(**query)["state"] == "available"
    assert (
        owner.activate(action["action_id"], approved=True, remember=False)["state"] == "succeeded"
    )
    terminal = journal.read_bytes()
    target.rename(backup)
    try:
        assert owner.status(**query)["state"] == "succeeded"
        assert ProfileActions("prestart", root).status(**query)["state"] == "succeeded"
        assert (
            owner.activate(action["action_id"], approved=True, remember=False)["state"]
            == "succeeded"
        )
        assert journal.read_bytes() == terminal
    finally:
        backup.rename(target)
