"""L3 runtime transport evidence, NOT real-hub/system E2E.

UI DMs use conversation ids, not the legacy bmsg pair lane. The hub HTTP
transport and inference are synthetic; connector routing, SQLite screening,
card composition, runtime discovery, authenticated TCP, serving handle and
recipient turns are real. The direct screening entry has an injected policy;
this does not prove authenticated scheduler status acquisition or model compliance.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE
from local_operator.harness.types import (
    AgentTool,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolResult,
)
from local_operator.mobile.peer_client import send_peer_message
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.team_bridge import pull, screening
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.connector import conversation, loop
from local_operator.team_bridge.store import state_db_path
from local_operator.team_bridge.tool import BridgeRuntime

# Importing the fixture applies the same explicit recorder lifecycle to both rigs.
from tests.unit.team_bridge.test_body_turn_e2e import telemetry_off  # noqa: F401
from tests.unit.team_bridge.test_body_turn_e2e import MODEL, CapturingStream, _tool_call
from tests.unit.team_bridge.test_client import FakeRow, FakeStore
from tests.unit.team_bridge.test_recipient_states import (
    BODY,
    CONVERSATION_ID,
    MESSAGE_ID,
    _enroll,
    _owner,
)
from tests.unit.team_bridge.test_screening import Policy, Runner

SESSION_ID = "sess_transport"


async def settle(session):
    # receive_peer_message schedules before it sets is_streaming: waiting only
    # on that flag would dispose the recipient before its turn had even begun.
    async with asyncio.timeout(10):
        while session._background_tasks or session._is_streaming:
            await asyncio.sleep(0.01)


def frames(root):
    path = root / "sessions" / SESSION_ID / "transcript.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def bridge_frames(root):
    return [
        row
        for row in frames(root)
        if row.get("payload", {}).get("custom_type") == TEAM_BRIDGE_MESSAGE_TYPE
    ]


@asynccontextmanager
async def runtime_rig(tmp_path, monkeypatch, *, sender_kind="human", verdict="deliver", tools=()):
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.chdir(tmp_path)
    owner = _owner("receiver", 1)
    store = _enroll(root, owner, SESSION_ID)
    wire_calls = []
    admitted = []
    messages = {MESSAGE_ID: BODY}
    unavailable = set()
    body_fetches = []
    body_gates = {}
    event_filter = set()

    async def hub(request):
        # Never log headers: even this fake bearer must not establish a habit
        # that leaks the real fixture's short-lived token when it replaces this.
        wire_calls.append((request.method, request.url.path))
        path = request.url.path
        if path.endswith("/events"):
            result = {
                "cursor": str(len(messages)),
                "events": [
                    {
                        "type": "message_available",
                        "at": "2026-10-25T00:00:00Z",
                        "ids": {
                            "conversation_id": CONVERSATION_ID,
                            "message_id": message_id,
                            "conversation_seq": seq,
                            "mention": True,
                        },
                    }
                    for seq, message_id in enumerate(messages, start=1)
                    if not event_filter or message_id in event_filter
                ],
            }
        elif path.endswith("/body"):
            message_id = path.split("/")[-2]
            body_fetches.append(message_id)
            if message_id in body_gates:
                entered, resume = body_gates[message_id]
                entered.set()
                await resume.wait()
            if message_id in unavailable:
                return httpx.Response(404, json={"code": "not_found", "error": "unavailable"})
            body = messages[message_id]
            result = {
                "message_id": message_id,
                "body": body,
                "body_hash": hashlib.sha256(body.encode()).hexdigest(),
                "conversation_name": "DM fixture",
                "sender_label": "Other identity",
                "sender_kind": sender_kind,
            }
        elif path.endswith("/admit"):
            admitted.append(json.loads(request.content))
            result = {"ok": True}
        else:
            raise AssertionError(f"unexpected mock hub route: {request.method} {path}")
        return httpx.Response(200, json={"msg": "ok", "result": result})

    credentials = FakeStore(
        [FakeRow(1, {"access": "synthetic-only", "account_id": owner.account_id})]
    )
    client = TeamBridgeClient(
        root,
        base_url=owner.canonical_base_url,
        credential_id=1,
        store=credentials,
        transport=httpx.MockTransport(hub),
    )
    monkeypatch.setattr(pull, "_client_for", lambda config_dir, client=None: rig.client)

    async def wire_for(*args, **kwargs):
        return client

    runner = Runner(reply=f"VERDICT: {verdict}\nREASON: prompt_injection fixture.")

    async def screen(owner, item, *, config_dir, **kwargs):
        # Unlike test_recipient_states this produces the receipt via the REAL
        # parser, account lease, budget and SQLite screening transaction.
        return await screening.screen_message_async(
            owner,
            item,
            config_dir=config_dir,
            policy=Policy(),
            runner=runner,
        )

    monkeypatch.setattr(conversation, "_wire_for", wire_for)
    monkeypatch.setattr(conversation, "screen_item", screen)
    stream = CapturingStream(
        [[StreamTextDelta(delta="settled"), StreamEndEvent(stop_reason="stop")]]
    )
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=list(tools),
        transcript=Transcript(root / "sessions" / SESSION_ID),
        system_blocks_provider=lambda: ["fixture", "env"],
        bridge_runtime=BridgeRuntime(client, root, owner.tenant_id, owner, SESSION_ID),
    )
    await session.prompt("owner starts this existing session")
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(tmp_path), install_gates=False
    )
    # Hosts publish ownership separately from the discovery registry. Without
    # this marker engage_runtime correctly tries to create the missing owner.
    marker = root / "sessions" / SESSION_ID / ".session.pid"
    marker.write_text(str(os.getpid()))
    from local_operator.session.runtime import launch

    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("L3 fixture must not launch another runtime")

    monkeypatch.setattr(launch, "_spawn_runtime", forbidden_spawn)
    server = RuntimeServer(handle, kind="tui")
    rig = SimpleNamespace(
        root=root,
        owner=owner,
        store=store,
        client=client,
        session=session,
        stream=stream,
        server=server,
        admitted=admitted,
        runner=runner,
        wire_calls=wire_calls,
        messages=messages,
        unavailable=unavailable,
        body_fetches=body_fetches,
        body_gates=body_gates,
        event_filter=event_filter,
    )
    server.start()
    server.set_record_started(True)
    try:
        async with asyncio.timeout(5):
            while True:
                found = [
                    r
                    for r, state in registry.scan()
                    if state == "live" and r.session_id == SESSION_ID
                ]
                if found:
                    rig.record = found[0]
                    break
                await asyncio.sleep(0.02)
        from local_operator.mobile.attach_client import find_runtime_record

        discovered, owner_pid = find_runtime_record(root, SESSION_ID)
        assert discovered is not None and owner_pid == os.getpid(), "fixture not discoverable"
        yield rig
    finally:
        # Match process._clean_exit: retire the serving handle/session first,
        # then await the thread-hosted server off-loop. Closing the socket
        # synchronously before disposal can block host callbacks needed by
        # shutdown and permit disposal publications after the unpublish.
        try:
            await handle.dispose()
            assert all(task.done() for task in session._background_tasks)
            waiter = session._pending_body_followup_task
            assert waiter is None or waiter.done()
        finally:
            await server.aclose_remote()
            marker.unlink(missing_ok=True)
        # A closed listener and no discovery row prove teardown, not merely a
        # finally block that attempted it. No process is spawned by this fixture.
        assert not [r for r, _ in registry.scan() if r.session_id == SESSION_ID]
        if hasattr(rig, "record"):
            with pytest.raises((ConnectionRefusedError, OSError)):
                reader, writer = await asyncio.open_connection("127.0.0.1", rig.record.control_port)
                writer.close()
                await writer.wait_closed()


async def deliver(rig):
    return await loop.run_iteration(
        rig.owner,
        config_dir=rig.root,
        auth_store=None,
        access=await rig.client.current_access(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sender_kind", ["human", "agent"])
async def test_dm_real_tcp_idle_provenance_and_replay(tmp_path, monkeypatch, sender_kind):
    async with runtime_rig(tmp_path, monkeypatch, sender_kind=sender_kind) as rig:
        before = len(rig.stream.requests)
        directories = sorted(p.name for p in (rig.root / "sessions").iterdir())
        result = await deliver(rig)
        await settle(rig.session)
        assert result["handled"][0]["action"] == "delivered", result
        rows = bridge_frames(rig.root)
        assert len(rows) == 1
        text = rows[0]["payload"]["details"]["text"]
        assert f"Other identity ({sender_kind}, same tenant)" in text
        assert "mention: yes" in text and "untrusted" in text
        assert len(rig.stream.requests) == before + 1
        assert BODY in "\n".join(rig.stream.texts(-1))
        assert rig.store.receipt_for_message(rig.owner, MESSAGE_ID)["verdict"] == "allow"
        assert len(rig.runner.calls) == 1
        assert rig.admitted[0]["verdict"] == "allow"
        # This is a TCP-frame replay, separate from the hub event below: the
        # sender-minted peer id must survive every hop into the transcript.
        replay = await send_peer_message(
            rig.record,
            text=rows[0]["payload"]["details"]["text"],
            mode="mailbox",
            wake=True,
            sender={},
            message_id=rows[0]["id"],
            custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        )
        assert replay.delivery["duplicate"] is True
        await settle(rig.session)
        assert len(bridge_frames(rig.root)) == 1
        assert len(rig.stream.requests) == before + 1
        # Replaying the upstream event must not create a second wake/card.
        again = await deliver(rig)
        await settle(rig.session)
        assert again["handled"][0]["action"] == "duplicate", again
        assert len(bridge_frames(rig.root)) == 1
        assert len(rig.stream.requests) == before + 1
        assert sorted(p.name for p in (rig.root / "sessions").iterdir()) == directories


@pytest.mark.asyncio
async def test_dm_moderation_deny_has_no_recipient_body_or_wake(tmp_path, monkeypatch):
    async with runtime_rig(tmp_path, monkeypatch, verdict="bounce") as rig:
        before = len(rig.stream.requests)
        result = await deliver(rig)
        await settle(rig.session)
        assert result["handled"][0]["action"] == "rejected", result["handled"]
        assert rig.admitted[0]["verdict"] == "reject"
        assert rig.store.receipt_for_message(rig.owner, MESSAGE_ID)["verdict"] == "reject"
        assert not bridge_frames(rig.root)
        assert len(rig.stream.requests) == before
        assert BODY not in json.dumps(frames(rig.root))
        assert rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID) is None


@pytest.mark.asyncio
async def test_tcp_wrong_key_refuses_without_recipient_work(tmp_path, monkeypatch):
    async with runtime_rig(tmp_path, monkeypatch) as rig:
        before = len(rig.stream.requests)
        bad = dataclasses.replace(rig.record, control_key="wrong-test-instance-key")
        with pytest.raises((RuntimeError, ConnectionError, OSError)):
            await send_peer_message(
                bad,
                text="wrong-key-canary",
                mode="mailbox",
                wake=True,
                sender={},
                deadline_s=1,
            )
        await settle(rig.session)
        assert len(rig.stream.requests) == before
        assert "wrong-key-canary" not in json.dumps(frames(rig.root))


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_type", [None, TEAM_BRIDGE_MESSAGE_TYPE])
async def test_idle_peer_without_native_body_keeps_ordinary_wake(
    tmp_path, monkeypatch, custom_type
):
    async with runtime_rig(tmp_path, monkeypatch) as rig:
        before = len(rig.stream.requests)
        await send_peer_message(
            rig.record,
            text="ordinary-idle-message-8843",
            mode="mailbox",
            wake=True,
            sender={},
            custom_type=custom_type,
        )
        await settle(rig.session)
        assert len(rig.stream.requests) == before + 1
        assert "ordinary-idle-message-8843" in "\n".join(rig.stream.texts(-1))
        assert rig.session._pending_body_followup_task is None
        assert rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID) is None


@pytest.mark.asyncio
async def test_later_idle_arrival_skips_retained_failed_body(tmp_path, monkeypatch):
    """A completed waiter cannot leave every later idle arrival stuck behind A."""
    tool_entered, finish_tool = asyncio.Event(), asyncio.Event()
    completed = []

    async def execute(tool_call_id, args, signal, on_update, context):
        tool_entered.set()
        await finish_tool.wait()
        completed.append("original-work-finished")
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="held_work",
            content=[TextContent(text="original work finished")],
        )

    tool = AgentTool(name="held_work", description="bounded test work", execute=execute)
    async with runtime_rig(tmp_path, monkeypatch, tools=[tool]) as rig:
        directories = sorted(p.name for p in (rig.root / "sessions").iterdir())
        rig.stream.scripted = [
            [],
            _tool_call("held-1", "held_work"),
            [StreamTextDelta(delta="settled"), StreamEndEvent(stop_reason="stop")],
        ]
        working = asyncio.create_task(rig.session.prompt("finish existing work"))
        try:
            await asyncio.wait_for(tool_entered.wait(), 5)
            assert (await deliver(rig))["handled"][0]["action"] == "delivered"
            first_waiter = rig.session._pending_body_followup_task
            assert first_waiter is not None
            rig.unavailable.add(MESSAGE_ID)
            finish_tool.set()
            await asyncio.wait_for(working, 10)
            await settle(rig.session)
            assert first_waiter.done(), "B must arrive AFTER the prior waiter retired"
            assert not rig.session._is_streaming and not rig.session._turn_lock.locked()
            assert completed == ["original-work-finished"]
            first = rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID)
            assert first["state"] == "pending" and first["attempts"] == 1
            assert first["last_error"] == "binding_drift"
            assert len(rig.stream.requests) == 3

            second_id, second_body = "bcmsg_abcdef02", "LATER-IDLE-BODY-7732"
            rig.messages[second_id] = second_body
            rig.event_filter.add(second_id)
            assert (await deliver(rig))["handled"][0]["action"] == "delivered"
            await settle(rig.session)
            second = rig.store.body_reservation(rig.owner, CONVERSATION_ID, second_id)
            assert second["state"] == "completed", second
            first = rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID)
            # A legitimate new arrival may retry A once in its new finite pass;
            # it must retain A/reason, reach B and terminate rather than spin.
            assert first["state"] == "pending" and first["attempts"] == 2
            assert first["last_error"] == "binding_drift"
            assert rig.body_fetches.count(MESSAGE_ID) == 3
            assert rig.body_fetches.count(second_id) == 2
            assert len(rig.stream.requests) == 4
            assert second_body in "\n".join(rig.stream.texts(3))
            assert BODY not in "\n".join(rig.stream.texts(3))
            assert len(bridge_frames(rig.root)) == 2
            assert second_body not in json.dumps(frames(rig.root))
            assert sorted(p.name for p in (rig.root / "sessions").iterdir()) == directories
        finally:
            finish_tool.set()
            if not working.done():
                await asyncio.wait_for(working, 10)


@pytest.mark.asyncio
async def test_self_contained_card_keeps_its_wake_behind_retained_failed_body(
    tmp_path, monkeypatch
):
    """PR #2951 round 1 MAJOR-1: a woken card without a body of its own must open a turn.

    A retained failed reservation A routes the idle card through the body-only
    pass; when that pass opens no turn (A fails again, nothing else eligible),
    the card is still owed the ordinary wake it would have had on main.
    """
    tool_entered, finish_tool = asyncio.Event(), asyncio.Event()

    async def execute(tool_call_id, args, signal, on_update, context):
        tool_entered.set()
        await finish_tool.wait()
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="held_work",
            content=[TextContent(text="original work finished")],
        )

    tool = AgentTool(name="held_work", description="bounded test work", execute=execute)
    async with runtime_rig(tmp_path, monkeypatch, tools=[tool]) as rig:
        directories = sorted(p.name for p in (rig.root / "sessions").iterdir())
        rig.stream.scripted = [
            [],
            _tool_call("held-1", "held_work"),
            [StreamTextDelta(delta="settled"), StreamEndEvent(stop_reason="stop")],
        ]
        working = asyncio.create_task(rig.session.prompt("finish existing work"))
        try:
            await asyncio.wait_for(tool_entered.wait(), 5)
            assert (await deliver(rig))["handled"][0]["action"] == "delivered"
            first_waiter = rig.session._pending_body_followup_task
            rig.unavailable.add(MESSAGE_ID)
            finish_tool.set()
            await asyncio.wait_for(working, 10)
            await settle(rig.session)
            assert first_waiter.done()
            first = rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID)
            assert first["state"] == "pending" and first["attempts"] == 1
            assert len(rig.stream.requests) == 3

            # A self-contained bridge card (pair-lane shape: screened excerpt,
            # no reservation of its own) arriving at idle.
            result = await send_peer_message(
                rig.record,
                text="SELF-CONTAINED-CARD-6141",
                mode="mailbox",
                wake=True,
                sender={},
                custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
            )
            await settle(rig.session)
            assert len(rig.stream.requests) == 4
            assert "SELF-CONTAINED-CARD-6141" in "\n".join(rig.stream.texts(-1))
            assert BODY not in "\n".join(rig.stream.texts(-1))
            # NIT-1: the sender is told the truth — this arrival woke the session.
            assert str(result) == "delivered and woke the session", str(result)
            first = rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID)
            # A is retained with its reason, never consumed or quarantined;
            # the bounded retries are the follow-up pass plus the ordinary wake.
            assert first["state"] == "pending" and first["last_error"] == "binding_drift"
            assert first["attempts"] <= 3
            assert not rig.session._turn_lock.locked()
            assert rig.session._pending_body_followup_task.done()
            assert sorted(p.name for p in (rig.root / "sessions").iterdir()) == directories
        finally:
            finish_tool.set()
            if not working.done():
                await asyncio.wait_for(working, 10)


@pytest.mark.asyncio
async def test_arrival_during_failed_hydration_rearms_same_waiter(tmp_path, monkeypatch):
    """B arrives after A's finite-pass horizon, before is_streaming turns true."""
    tool_entered, finish_tool = asyncio.Event(), asyncio.Event()
    hydration_entered, finish_hydration = asyncio.Event(), asyncio.Event()
    completed = []

    async def execute(tool_call_id, args, signal, on_update, context):
        tool_entered.set()
        await finish_tool.wait()
        completed.append("original-work-finished")
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="held_work",
            content=[TextContent(text="original work finished")],
        )

    tool = AgentTool(name="held_work", description="bounded test work", execute=execute)
    async with runtime_rig(tmp_path, monkeypatch, tools=[tool]) as rig:
        directories = sorted(p.name for p in (rig.root / "sessions").iterdir())
        rig.stream.scripted = [
            [],
            _tool_call("held-1", "held_work"),
            [StreamTextDelta(delta="settled"), StreamEndEvent(stop_reason="stop")],
        ]
        working = asyncio.create_task(rig.session.prompt("finish existing work"))
        try:
            await asyncio.wait_for(tool_entered.wait(), 5)
            assert (await deliver(rig))["handled"][0]["action"] == "delivered"
            waiter = rig.session._pending_body_followup_task
            assert waiter is not None
            rig.body_gates[MESSAGE_ID] = (hydration_entered, finish_hydration)
            rig.unavailable.add(MESSAGE_ID)
            finish_tool.set()
            await asyncio.wait_for(working, 10)
            await asyncio.wait_for(hydration_entered.wait(), 5)
            after_horizon = time.time()
            assert not rig.session._is_streaming and rig.session._turn_lock.locked()
            assert completed == ["original-work-finished"]

            second_id, second_body = "bcmsg_abcdef02", "AFTER-HORIZON-BODY-5529"
            rig.messages[second_id] = second_body
            rig.event_filter.add(second_id)
            assert (await deliver(rig))["handled"][0]["action"] == "delivered"
            second = rig.store.body_reservation(rig.owner, CONVERSATION_ID, second_id)
            assert second["created_at"] > after_horizon
            assert rig.session._pending_body_followup_task is waiter
            finish_hydration.set()
            await settle(rig.session)

            # No manual prompt: the arrival must rearm the existing finite pass
            # even during its non-streaming hydration window, without retrying A.
            second = rig.store.body_reservation(rig.owner, CONVERSATION_ID, second_id)
            assert second["state"] == "completed", second
            first = rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID)
            assert first["state"] == "pending" and first["attempts"] == 1
            assert first["last_error"] == "binding_drift"
            assert rig.body_fetches.count(MESSAGE_ID) == 2
            assert rig.body_fetches.count(second_id) == 2
            assert len(rig.stream.requests) == 4
            assert second_body in "\n".join(rig.stream.texts(3))
            assert BODY not in "\n".join(rig.stream.texts(3))
            assert len(bridge_frames(rig.root)) == 2
            assert second_body not in json.dumps(frames(rig.root))
            assert sorted(p.name for p in (rig.root / "sessions").iterdir()) == directories
        finally:
            finish_tool.set()
            finish_hydration.set()
            if not working.done():
                await asyncio.wait_for(working, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "single",
        "multi",
        "revoke",
        "wrong_owner",
        "reject",
        "hold",
        "owner_after_hydration",
        "first_unavailable",
    ],
)
async def test_busy_dm_pickup_preserves_running_tool(tmp_path, monkeypatch, case):
    entered, release = asyncio.Event(), asyncio.Event()
    hydrated, resume_pipeline = asyncio.Event(), asyncio.Event()
    completed = []

    async def execute(tool_call_id, args, signal, on_update, context):
        entered.set()
        await release.wait()
        completed.append("original-work-finished")
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="held_work",
            content=[TextContent(text="original work finished")],
        )

    tool = AgentTool(name="held_work", description="bounded test work", execute=execute)
    verdict = {"reject": "bounce", "hold": "hold"}.get(case, "deliver")
    async with runtime_rig(tmp_path, monkeypatch, tools=[tool], verdict=verdict) as rig:
        if case in ("multi", "first_unavailable"):
            rig.messages["bcmsg_abcdef02"] = "SECOND-ADMITTED-BODY-4418"
        if case == "owner_after_hydration":
            append = rig.session.transcript.append_custom

            async def paused_append(kind, details, *args, **kwargs):
                frame = rig.session._pending_body_turn
                if kind == "attention_started" and frame is not None and frame.has_claim:
                    hydrated.set()
                    await resume_pipeline.wait()
                return await append(kind, details, *args, **kwargs)

            monkeypatch.setattr(rig.session.transcript, "append_custom", paused_append)
        directories = sorted(p.name for p in (rig.root / "sessions").iterdir())
        # Absolute provider indices include the owner's engagement request.
        rig.stream.scripted = [
            [],
            _tool_call("held-1", "held_work"),
            [StreamTextDelta(delta="original work complete"), StreamEndEvent(stop_reason="stop")],
        ]
        working = asyncio.create_task(rig.session.prompt("finish existing work"))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            result = await deliver(rig)
            expected = {"reject": "rejected", "hold": "hold"}.get(case, "delivered")
            assert all(item["action"] == expected for item in result["handled"]), result
            assert not working.done() and not completed
            waiter = rig.session._pending_body_followup_task
            if case in ("single", "multi"):
                assert waiter is not None
                replay = await deliver(rig)
                assert all(item["action"] == "duplicate" for item in replay["handled"])
                assert rig.session._pending_body_followup_task is waiter, "one coalesced waiter"
            elif case == "revoke":
                # Model the committed withdrawal while the waiter is blocked
                # behind original work; the lock guard must read it afresh.
                with sqlite3.connect(state_db_path(rig.root)) as conn:
                    conn.execute("UPDATE enrollments SET consent_version=NULL, consent_ref=NULL")
            elif case == "wrong_owner":
                rig.session.bridge_runtime.owner = dataclasses.replace(
                    rig.owner, account_id="acc_unrelated"
                )
            elif case == "first_unavailable":
                # Both messages were admitted; only subsequent A hydration
                # fails. Selection must leave A's reason and still reach B.
                rig.unavailable.add(MESSAGE_ID)
            elif case == "owner_after_hydration":
                assert waiter is not None
            else:
                assert waiter is None, "screening did not admit any body work"
            release.set()
            await asyncio.wait_for(working, 10)
            if case == "owner_after_hydration":
                await asyncio.wait_for(hydrated.wait(), 5)
                row = rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID)
                assert row["state"] == "claimed"
                assert rig.store.active_owner() == rig.owner, "old enrollment remains live"
                rig.session.bridge_runtime.owner = dataclasses.replace(
                    rig.owner, account_id="acc_unrelated"
                )
                resume_pipeline.set()
            await settle(rig.session)
            assert completed == ["original-work-finished"]
            # RED log and exact source retained in scratch. The ORIGINAL turn
            # remains immutable; no manual prompt manufactures the next turn.
            assert BODY not in "\n".join(rig.stream.texts(2))
            if case in ("single", "multi"):
                assert len(rig.stream.requests) == 3 + len(rig.messages)
                for index, (message_id, body) in enumerate(rig.messages.items(), start=3):
                    request = "\n".join(rig.stream.texts(index))
                    assert request.count(body) == 1
                    assert all(
                        other not in request for other in rig.messages.values() if other != body
                    )
                    row = rig.store.body_reservation(rig.owner, CONVERSATION_ID, message_id)
                    assert row["state"] == "completed"
                assert len(bridge_frames(rig.root)) == len(rig.messages)
            elif case == "first_unavailable":
                assert len(rig.stream.requests) == 4
                assert "SECOND-ADMITTED-BODY-4418" in "\n".join(rig.stream.texts(3))
                first = rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID)
                second = rig.store.body_reservation(rig.owner, CONVERSATION_ID, "bcmsg_abcdef02")
                assert first["state"] == "pending" and first["attempts"] == 1
                assert first["last_error"] == "binding_drift"
                assert second["state"] == "completed" and second["attempts"] == 1
                assert rig.body_fetches.count(MESSAGE_ID) == 2, "admission + one failed hydration"
                assert rig.body_fetches.count("bcmsg_abcdef02") == 2
            else:
                assert len(rig.stream.requests) == 3, "no authority means no automatic turn"
                assert BODY not in "\n".join(rig.stream.texts(-1))
                row = rig.store.body_reservation(rig.owner, CONVERSATION_ID, MESSAGE_ID)
                if case in ("reject", "hold"):
                    assert row is None and bridge_frames(rig.root) == []
                elif case == "owner_after_hydration":
                    assert row["state"] == "pending" and row["claim_token"] is None
                    assert row["attempts"] == 1 and row["last_error"] == "admission_drift"
                    assert rig.store.active_owner() == rig.owner
                else:
                    assert row["state"] == "pending" and row["attempts"] == 0
            assert sorted(p.name for p in (rig.root / "sessions").iterdir()) == directories
            assert BODY not in json.dumps(frames(rig.root)), "body stays out of durable transcript"
        finally:
            release.set()
            resume_pipeline.set()
            if not working.done():
                await asyncio.wait_for(working, 10)
