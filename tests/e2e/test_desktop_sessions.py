"""Real HTTP + canonical Session/ServingSessionHandle/RuntimeServer/AttachClient.

Only the provider stream is scripted. No session, socket, admission, transcript,
gate or bridge is mocked: this catches the seams a green adapter suite cannot.
"""

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import secrets
import socket
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from local_operator.mobile.attach_client import AttachClient
from local_operator.server.app import app
from local_operator.session.attention import AttentionStore
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import (
    E2E_ORACLE_MODEL,
    ScriptedStream,
    build_session,
    text_turn,
    tool_call_turn,
)
from tests.notification_opt_in import notification_path_opt_in

pytestmark = pytest.mark.e2e


@pytest.fixture
def notifications_on(headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Let these cells HEAR the notification path, deliberately.

    ``tests/e2e/conftest.py`` arms ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` for every
    e2e test (a composed frame must not reach the developer's machine while
    fifty sessions are open), and the runtime and the desktop bridge now skip an
    announced completion while that switch is on — which is the subject of every
    cell that asks for this fixture: one asserts exactly one ``notification``
    frame after N turn ends, one asserts the runtime speaks once the desktop
    lease is withdrawn.

    Ordering is why this fixture TAKES ``headless_tui_env``: pytest builds an
    autouse fixture before a same-scope one that does not depend on it, so a
    bare fixture would clear the switch and be re-armed by the autouse one a
    moment later. Set and restored BY HAND rather than through ``monkeypatch``,
    which is function-scoped and shared with the test: a test that calls
    ``monkeypatch.undo()`` would otherwise re-arm the gate mid-cell.

    THE BODY IS THE SHARED OPT-IN (``tests/notification_opt_in``), which clears
    that switch AND waives the test-hosting rule — the sessions these cells
    drive carry a real selection row, so the per-session rule would suppress the
    very frame they assert without it.

    Opening the gate is safe ONLY because the fixture also doubles
    ``tui.notify.detached_notify`` — the detached banner is a real ``osascript``
    call on this host, and a cell that opens the gate without the double is a
    cell that can put a banner on the developer's screen. Every cell that asks
    for this fixture asserts on an in-process frame (or on its own recorder,
    which it installs over this one) instead.

    The double answers FALSE ("no banner was raised"), which is the production
    answer for a silenced process and the one these cells depend on: the
    runtime's rung-4 arm releases the delivery claim it took when the raise
    does not land, and ``test_a_real_turn_emits_exactly_one_notification...``
    ends by asserting the RENDERER can still claim the same completion. A
    double answering True leaves the watermark spent and that claim fails —
    which is what this fixture did before the answer mattered.
    """
    from local_operator.tui import notify as notify_module

    monkeypatch.setattr(notify_module, "detached_notify", lambda *args, **kwargs: False)
    with notification_path_opt_in():
        yield


async def next_frame(lines, predicate):
    async def read():
        async for line in lines:
            if line.startswith("data: "):
                frame = json.loads(line[6:])
                if predicate(frame):
                    return frame
        raise AssertionError("stream ended before the expected frame")

    return await asyncio.wait_for(read(), 30)


@pytest.mark.asyncio
async def test_canonical_desktop_over_http(headless_tui_env: Path, workspace: Path, monkeypatch):
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    runtime = handle = terminal = None
    try:
        for _ in range(10000):
            if server.started:
                break
            if serving.done():
                await serving
            await asyncio.sleep(0)
        assert server.started
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            path = "/v1/desktop/sessions"
            assert (await client.get(path)).status_code == 401
            client.headers["Authorization"] = "Bearer incorrect"
            assert (await client.get(path)).status_code == 401
            client.headers["Authorization"] = f"Bearer {token}"
            for origin in ("null", "https://evil.example"):
                assert (await client.get(path, headers={"Origin": origin})).status_code == 403
            invalid = await client.post(path, json={"request_id": "short", "cwd": str(workspace)})
            assert invalid.status_code == 422
            created = await client.post(
                path,
                json={"request_id": "11111111-1111-4111-8111-111111111111", "cwd": str(workspace)},
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            again = await client.post(
                path,
                json={"request_id": "11111111-1111-4111-8111-111111111111", "cwd": str(workspace)},
            )
            assert again.json()["result"]["session_id"] == sid
            target = path + "/" + sid
            cold = await client.get(target)
            assert cold.status_code == 200, cold.text
            assert cold.json()["result"]["payload"]["cold"]
            assert not (root / "sessions" / sid / ".session.pid").exists()
            print(
                "HTTP auth: missing/wrong token401, evil/null Origin403, invalid422; "
                "create200 stable retry; cold GET creates no owner"
            )

            stream = ScriptedStream(
                [
                    text_turn("The goal request arrived once."),
                    text_turn("The canonical runtime answered."),
                    text_turn("The team request arrived once."),
                    text_turn("The image arrived without invented text."),
                    text_turn("The stored image arrived without invented text."),
                ]
            )
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
            # Named so the gate assertion further down is a real one: an
            # unnamed session would publish an empty `session_name` and the
            # D3 check would pass vacuously.
            session.set_conversation_name("HTTP contract run", user_set=True)
            from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

            session.team_registry = TeamRegistry(root)
            session.team_registry.create_team(
                TeamEditFields(
                    name="http-team",
                    description="HTTP test team",
                    manager="manager",
                    members=[TeamMember(role="coder")],
                )
            )
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            # The assembled test owns this in-process Session; production's
            # process launcher writes the same claim marker before publishing.
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
            from local_operator.mobile.attach_client import find_runtime_record

            record, _ = find_runtime_record(root, sid)
            assert record is not None
            terminal = AttachClient(lambda _: None, lambda _: None)
            await terminal.connect(record, sid)

            async with client.stream("GET", target + "/events") as response:
                assert response.status_code == 200
                lines = response.aiter_lines()
                opened = await next_frame(lines, lambda f: f["type"] == "open")
                subscription = opened["payload"]["subscription_id"]
                snapshot = await next_frame(lines, lambda f: f["type"] == "snapshot")
                assert snapshot["payload"]["frontend"]["snapshot"]["session_id"] == sid
                watched = await client.post(
                    target + "/watch",
                    json={
                        "subscription_id": subscription,
                        "visible": True,
                        "can_notify": True,
                    },
                )
                assert watched.status_code == 200, watched.text
                assert "desktop" in runtime.watching_surfaces()
                assert "desktop" in runtime.notification_surfaces()
                result = await client.post(
                    target + "/commands",
                    json={
                        "request_id": "22222222-2222-4222-8222-222222222222",
                        "command": "goal",
                        "args": "Preserve one identity",
                    },
                )
                assert result.status_code == 200, result.text
                assert session.goal == "Preserve one identity"
                assert "Preserve one identity" in json.dumps(
                    await terminal.slash_result("goal", "")
                )
                # BESIDE the stored goal, the argument is an ordinary user turn.
                # The TUI does both on one Enter, and this route is the host of the
                # viewer that declared the receipt as its own (see
                # ``desktop_viewer_must_submit``): setting the goal and dropping the
                # request is the silent drop that declaration exists to prevent.
                assert result.json()["result"]["result"]["admission"]["status"] == "admitted"
                await next_frame(
                    lines,
                    lambda f: f["type"] == "event" and f["payload"].get("type") == "agent_end",
                )
                goal_history = await client.get(target + "/history")
                assert goal_history.status_code == 200, goal_history.text
                goal_entries = goal_history.json()["result"]["entries"]
                assert (
                    sum(row["id"] == "22222222-2222-4222-8222-222222222222" for row in goal_entries)
                    == 1
                )
                assert "Preserve one identity" in json.dumps(goal_entries)
                assert len(stream.requests) == 1, "the goal command must consume exactly one turn"
                duplicate = await client.post(
                    target + "/commands",
                    json={
                        "request_id": "22222222-2222-4222-8222-222222222222",
                        "command": "goal",
                        "args": "Preserve one identity",
                    },
                )
                assert duplicate.json()["result"]["replayed"]
                assert len(stream.requests) == 1, "a replayed goal command must not submit a turn"
                print(
                    "HTTP /goal200 stores the goal AND admits the same argument as an "
                    "ordinary user turn: one durable row under the goal, one scripted "
                    "provider call; retry replayed without a second turn"
                )
                changed = await client.post(
                    target + "/commands",
                    json={
                        "request_id": "22222222-2222-4222-8222-222222222222",
                        "command": "goal",
                        "args": "different",
                    },
                )
                assert changed.status_code == 409
                print(
                    "Same-session HTTP /goal200; terminal sees exact goal; retry "
                    "replayed; changed request409"
                )

                # Explicit naming prevents the owner's separate title-model
                # errand from consuming this one-turn provider script.
                named = await client.post(
                    target + "/commands",
                    json={
                        "request_id": "44444444-4444-4444-8444-444444444444",
                        "command": "rename",
                        "args": "HTTP canonical evidence",
                    },
                )
                assert named.status_code == 200, named.text
                bad_slash = await client.post(
                    target + "/messages",
                    json={
                        "request_id": "33333333-3333-4333-8333-333333333333",
                        "text": "/settings",
                    },
                )
                assert bad_slash.status_code == 422
                message = {
                    "request_id": "33333333-3333-4333-8333-333333333333",
                    "text": "A canonical turn",
                }
                admitted = await client.post(target + "/messages", json=message)
                assert admitted.status_code == 200, admitted.text
                assert admitted.json()["result"]["status"] == "admitted"
                await next_frame(
                    lines,
                    lambda f: f["type"] == "event" and f["payload"].get("type") == "agent_end",
                )
                retried = await client.post(target + "/messages", json=message)
                assert retried.json()["result"]["replayed"]
                history = await client.get(target + "/history")
                assert history.status_code == 200, history.text
                rows = history.json()["result"]["entries"]
                assert sum("A canonical turn" in json.dumps(row) for row in rows) == 1
                assert "The canonical runtime answered." in json.dumps(rows)
                assert len(stream.requests) == 2
                print(
                    "Prompt admission200; canonical agent_end received; one durable "
                    "user row and one real scripted provider call; retry did not "
                    "repeat"
                )
                stale = await client.post(
                    target + "/answers",
                    json={
                        "epoch": "old-owner",
                        "request_id": "obsolete",
                        "approved": True,
                    },
                )
                assert stale.status_code == 409
                stale = await client.post(
                    target + "/answers",
                    json={
                        "epoch": session.frontend_state.epoch,
                        "request_id": "obsolete",
                        "approved": True,
                    },
                )
                assert stale.status_code == 409
                assert (await client.get(path + "/aaaaaaaaaaaa")).status_code == 404

                team_body = {
                    "request_id": "66666666-6666-4666-8666-666666666666",
                    "command": "team",
                    "args": "http-team Check the attached team",
                }
                team_result = await client.post(target + "/commands", json=team_body)
                assert team_result.status_code == 200, team_result.text
                assert team_result.json()["result"]["result"]["admission"]["status"] == "admitted"
                await next_frame(
                    lines,
                    lambda f: f["type"] == "event" and f["payload"].get("type") == "agent_end",
                )
                team_retry = await client.post(target + "/commands", json=team_body)
                assert team_retry.json()["result"]["replayed"]
                assert session.frontend_state.active_team == "http-team"
                team_history = await client.get(target + "/history")
                assert (
                    sum(
                        row["id"] == team_body["request_id"]
                        for row in team_history.json()["result"]["entries"]
                    )
                    == 1
                )
                assert len(stream.requests) == 3
                print(
                    "Owner /team attaches real registry team and admits consumed "
                    "request once; retry replayed without second turn"
                )

                image_body = {
                    "request_id": "77777777-7777-4777-8777-777777777777",
                    "text": "",
                    "images": [
                        {
                            "mime_type": "image/png",
                            "data_b64": (
                                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
                                "QVR42mP8/x8AAwMCAO+j3ioAAAAASUVORK5CYII="
                            ),
                        }
                    ],
                }
                image_result = await client.post(target + "/messages", json=image_body)
                assert image_result.status_code == 200, image_result.text
                await next_frame(
                    lines,
                    lambda f: f["type"] == "event" and f["payload"].get("type") == "agent_end",
                )
                image_retry = await client.post(target + "/messages", json=image_body)
                assert image_retry.json()["result"]["replayed"]
                image_history = await client.get(target + "/history")
                assert (
                    sum(
                        row["id"] == image_body["request_id"]
                        for row in image_history.json()["result"]["entries"]
                    )
                    == 1
                )
                assert len(stream.requests) == 4
                print(
                    "Image-only prompt admitted200 without synthetic text; "
                    "one durable user row; retry did not duplicate image"
                )

                # The image above is 70 bytes of base64 -- BELOW the 1024-byte
                # externalization floor -- so it never reaches the attachment
                # store, and until now this file's single image body exercised
                # admission with the store path untouched. That is the gap an
                # image send walked through on 2026-09-17: the store write is the
                # largest write in the flow, so it is the FIRST to fail on a
                # nearly-full volume, while a few-KB text write still lands. That
                # is the whole reason an image was refused and the same message
                # without one was not.
                #
                # This second message carries a screenshot-sized image and pins
                # the path end to end: admission, the bytes on disk, the
                # REFERENCE the transcript keeps instead of the bytes, and the
                # read route that serves them back.
                from PIL import Image as PILImage

                buffer = io.BytesIO()
                PILImage.frombytes("RGB", (300, 300), os.urandom(300 * 300 * 3)).save(
                    buffer, format="PNG"
                )
                raw = buffer.getvalue()
                large_b64 = base64.b64encode(raw).decode()
                assert len(large_b64) > 1024, "below the externalization floor"
                large_body = {
                    "request_id": "88888888-8888-4888-8888-888888888888",
                    "text": "",
                    "images": [{"mime_type": "image/png", "data_b64": large_b64}],
                }
                large_result = await client.post(target + "/messages", json=large_body)
                assert large_result.status_code == 200, large_result.text
                await next_frame(
                    lines,
                    lambda f: f["type"] == "event" and f["payload"].get("type") == "agent_end",
                )

                digest = hashlib.sha256(raw).hexdigest()[:32]
                store_dir = root / "attachments"
                stored = store_dir / f"{digest}.bin"
                # A store that is absent is the failure mode this guards (the
                # write never happened), so the diagnostic has to survive it.
                present = (
                    sorted(entry.name for entry in store_dir.iterdir())
                    if store_dir.exists()
                    else "no attachments directory at all"
                )
                assert stored.exists(), f"image never reached the store: {present}"
                assert stored.read_bytes() == raw

                rows = [
                    json.loads(line)
                    for line in (root / "sessions" / sid / "transcript.jsonl")
                    .read_text()
                    .splitlines()
                ]
                (row,) = [entry for entry in rows if entry.get("id") == large_body["request_id"]]
                (block,) = [entry for entry in row["payload"]["content"] if entry.get("mime_type")]
                # The reference, not the bytes: the row carries a digest and the
                # store owns the payload, which is what keeps a screenshot out of
                # the JSONL the model is replayed from.
                assert block["attachment"] == digest
                assert "data" not in block
                assert len(json.dumps(row)) < len(large_b64)

                served = await client.get(target + "/attachments/" + digest)
                assert served.status_code == 200, served.text
                assert served.content == raw
                assert served.headers["content-type"].startswith("image/")
                assert len(stream.requests) == 5
                print(
                    f"Image {len(raw)}B persisted to the attachment store and "
                    "referenced from the transcript; served back over HTTP"
                )

                # Exercise the actual installed owner gate closures. Invalid
                # answers must leave the same future pending; another window's
                # successful answer makes the original popup stale.
                from local_operator.harness.types import AskQuestion

                question = AskQuestion(
                    id="HTTP_TEST_SECRET", question="Test masked input", secret=True
                )
                asking = asyncio.create_task(handle._ask_gate([question]))
                gate_frame = await next_frame(
                    lines,
                    lambda f: f["type"] == "frontend.update"
                    and bool(f["payload"]["changes"].get("pending_gate")),
                )
                gate = gate_frame["payload"]["changes"]["pending_gate"]
                # D3: a parked gate carries the conversation's name, so a
                # desktop banner for it can be triaged. The card travels on
                # THIS existing path rather than on a second notification
                # channel, which is why the assertion belongs on the gate frame
                # a real `_ask_gate` produced rather than on a new frame type.
                assert gate["session_name"] == session.conversation_name
                assert gate["session_name"], "a real parked gate published an anonymous card"
                answer_body = {
                    "epoch": session.frontend_state.epoch,
                    "request_id": gate["request_id"],
                    "question_index": gate["question_index"],
                    "value": "synthetic-answer",
                }
                invalid_answer = await client.post(
                    target + "/answers", json={**answer_body, "question_index": 9}
                )
                assert invalid_answer.status_code == 409 and not asking.done()
                invalid_answer = await client.post(
                    target + "/answers", json={**answer_body, "approved": True}
                )
                assert invalid_answer.status_code == 422 and not asking.done()
                answered = await client.post(target + "/answers", json=answer_body)
                assert answered.status_code == 200, answered.text
                assert "synthetic-answer" not in answered.text
                assert await asyncio.wait_for(asking, 30) == {
                    "HTTP_TEST_SECRET": ["synthetic-answer"]
                }
                assert (await client.post(target + "/answers", json=answer_body)).status_code == 409
                approving = asyncio.create_task(
                    handle._approval_gate("test-operation", "No side effect in this gate probe")
                )
                gate_frame = await next_frame(
                    lines,
                    lambda f: f["type"] == "frontend.update"
                    and (f["payload"]["changes"].get("pending_gate") or {}).get("kind")
                    == "approval",
                )
                approval = gate_frame["payload"]["changes"]["pending_gate"]
                denied = await client.post(
                    target + "/answers",
                    json={
                        "epoch": session.frontend_state.epoch,
                        "request_id": approval["request_id"],
                        "approved": False,
                    },
                )
                assert denied.status_code == 200 and await asyncio.wait_for(approving, 30) is False
                print(
                    "Real owner ask200/approval denial200; wrong-index409 and invalid-"
                    "shape422 leave gate pending; stale answer409; secret absent from "
                    "response"
                )

                async with client.stream(
                    "GET",
                    target + "/events",
                    params={"epoch": opened["epoch"], "after_seq": opened["seq"]},
                ) as replay_response:
                    replay_lines = replay_response.aiter_lines()
                    replay_open = await next_frame(replay_lines, lambda f: f["type"] == "open")
                    assert not replay_open["payload"]["gap"]
                    replayed_events = []
                    while True:
                        frame = await next_frame(replay_lines, lambda _: True)
                        if frame["type"] == "snapshot":
                            assert "The canonical runtime answered." in json.dumps(
                                frame["payload"]["history"]
                            )
                            break
                        replayed_events.append(frame)
                    assert any(
                        f["type"] == "event" and f["payload"].get("type") == "agent_end"
                        for f in replayed_events
                    )
                    assert [f["seq"] for f in replayed_events] == sorted(
                        {f["seq"] for f in replayed_events}
                    )
                print(
                    "Concurrent reconnect replays ordered semantic agent_end BEFORE "
                    "newer authoritative snapshot; no receipt skipped by snapshot "
                    "watermark"
                )

                other = await client.post(
                    path,
                    json={
                        "request_id": "55555555-5555-4555-8555-555555555555",
                        "cwd": str(workspace),
                    },
                )
                other_path = path + "/" + other.json()["result"]["session_id"]
                isolated = await client.get(other_path)
                assert isolated.json()["result"]["payload"]["frontend"]["snapshot"]["goal"] == ""
                cross_watch = await client.post(
                    other_path + "/watch",
                    json={"subscription_id": subscription, "visible": True, "can_notify": True},
                )
                assert cross_watch.status_code == 404
                assert "desktop" in runtime.watching_surfaces()
                print(
                    "Second canonical session has isolated state; first stream ID "
                    "rejected404 by second session"
                )

            # Closing only the HTTP reader leaves the canonical owner and its
            # independent terminal client intact. Await bridge release directly
            # as an invariant rather than assuming a wall-clock sleep is enough.
            for _ in range(10000):
                if (
                    not app.state.desktop_sessions.bridges[sid].subscribers
                    and "desktop" not in runtime.watching_surfaces()
                ):
                    break
                await asyncio.sleep(0)
            assert not app.state.desktop_sessions.bridges[sid].subscribers
            assert "desktop" not in runtime.watching_surfaces()
            assert "desktop" not in runtime.notification_surfaces()
            assert terminal.connected
            reopened = await client.get(target)
            assert reopened.status_code == 200, reopened.text
            assert (
                reopened.json()["result"]["payload"]["frontend"]["snapshot"]["goal"]
                == "Preserve one identity"
            )
            stale_watch = await client.post(
                target + "/watch",
                json={
                    "subscription_id": subscription,
                    "visible": True,
                    "can_notify": True,
                },
            )
            assert stale_watch.status_code == 404
            print(
                "Close/reopen200 retains session/goal; stale answers409 and "
                "disconnected watch404; desktop lease removed, terminal owner remains"
            )
    finally:
        if terminal is not None:
            await terminal.detach()
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()


@pytest.mark.asyncio
async def test_a_bulk_read_receipt_clears_only_the_completions_it_was_given(
    headless_tui_env: Path, workspace: Path, monkeypatch
):
    """The whole matrix over real loopback HTTP, against a real receipt store.

    The feature is a bulk write, so the two things that can only be tested here
    are (a) that the WIRE admits exactly the items it was handed and answers a
    verdict for each, and (b) that the write is narrow: a completion published
    after the client's render stays unread, an id this machine has no session for
    earns `unknown` rather than a refusal of the call, and nothing about
    DELIVERY moves -- notifying is not reading, so an already-bannered
    conversation keeps its banner while its mark clears.

    Deliberately no bridge, runtime or transcript is involved: a read receipt is
    cold by construction, and the desktop pool has no owner here at all.
    """
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server, serving = await serve_app(listener, token)
    store = AttentionStore(root / "attention.db")
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            path = "/v1/desktop/sessions"

            async def new_session() -> str:
                created = await client.post(
                    path,
                    json={"request_id": str(uuid.uuid4()), "cwd": str(workspace)},
                )
                assert created.status_code == 200, created.text
                return created.json()["result"]["session_id"]

            def publish(session_id: str, anchor: str) -> str:
                completion = str(uuid.uuid4())
                store.publish(f"session/{session_id}", completion, anchor, "complete")
                return completion

            settled, racing, delivered = (
                await new_session(),
                await new_session(),
                await new_session(),
            )
            observed = {
                settled: publish(settled, "anchor-settled"),
                # Two completions for this conversation: the token the client was
                # shown is no longer current by the time it clicks.
                racing: publish(racing, "anchor-racing-1"),
                delivered: publish(delivered, "anchor-delivered"),
            }
            publish(racing, "anchor-racing-2")
            assert store.claim_delivery(f"session/{delivered}", observed[delivered], "desktop")

            revision_before = store.revision()
            with contextlib.closing(sqlite3.connect(root / "attention.db")) as conn:
                supersedes_before = conn.execute("SELECT COUNT(*) FROM supersede_log").fetchone()[0]
                deliveries_before = conn.execute("SELECT * FROM deliveries ORDER BY 1").fetchall()
                completions_before = conn.execute("SELECT * FROM completions ORDER BY 1").fetchall()

            absent = "0123456789ab"
            response = await client.post(
                "/v1/desktop/attention/seen",
                json={
                    "items": [
                        {"session_id": settled, "completion_token": observed[settled]},
                        {"session_id": racing, "completion_token": observed[racing]},
                        {"session_id": delivered, "completion_token": observed[delivered]},
                        {"session_id": absent, "completion_token": str(uuid.uuid4())},
                    ]
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["message"] == "Completion receipts marked read."
            assert body["result"]["superseded"] == [racing]
            assert body["result"]["unknown"] == [absent]
            assert sorted(entry["conversation_id"] for entry in body["result"]["read"]) == sorted(
                [f"session/{settled}", f"session/{delivered}"]
            )
            for entry in body["result"]["read"]:
                assert entry["unseen"] is False
                # R8: the store's own state, so the wire omits `supported`
                # entirely and the renderer's merge inherits what it knew.
                assert "supported" not in entry, entry

            # The store, read back through the same object every surface uses.
            assert store.state(f"session/{settled}")["unseen"] is False
            assert store.state(f"session/{delivered}")["unseen"] is False
            superseded_state = store.state(f"session/{racing}")
            assert superseded_state["unseen"] is True, "a superseded item was cleared"
            assert superseded_state["anchor_id"] == "anchor-racing-2"
            assert store.state(f"session/{absent}")["unseen"] is False

            # R2: a read is not a notification. Delivery, the supersede log and
            # the completion rows are untouched, and MAX(sequence) has not moved,
            # so nothing read can be resurrected as unread.
            with contextlib.closing(sqlite3.connect(root / "attention.db")) as conn:
                assert conn.execute("SELECT COUNT(*) FROM supersede_log").fetchone()[0] == (
                    supersedes_before
                )
                assert conn.execute("SELECT * FROM deliveries ORDER BY 1").fetchall() == (
                    deliveries_before
                )
                assert conn.execute("SELECT * FROM completions ORDER BY 1").fetchall() == (
                    completions_before
                )
            assert store.revision()[0] == revision_before[0]
            assert store.revision()[2] == revision_before[2]
            assert store.revision()[1] > revision_before[1], "the ack did not move the watermark"

            # A batch that clears nothing is still 200 -- the three buckets are
            # the answer -- and the malformed bodies are refused before the store
            # is ever consulted.
            nothing = await client.post(
                "/v1/desktop/attention/seen",
                json={"items": [{"session_id": absent, "completion_token": str(uuid.uuid4())}]},
            )
            assert nothing.status_code == 200, nothing.text
            assert nothing.json()["result"] == {"read": [], "superseded": [], "unknown": [absent]}
            for payload in (
                {"items": []},
                {"items": [{"session_id": "short", "completion_token": str(uuid.uuid4())}]},
                {"items": [{"session_id": absent, "completion_token": "not-a-token"}]},
                {
                    "items": [
                        {"session_id": absent, "completion_token": str(uuid.uuid4())}
                        for _ in range(501)
                    ]
                },
            ):
                refused = await client.post("/v1/desktop/attention/seen", json=payload)
                assert refused.status_code == 422, (payload, refused.text)

            for headers in ({"Authorization": ""}, {"Authorization": "Bearer nope"}):
                denied = await client.post(
                    "/v1/desktop/attention/seen",
                    json={
                        "items": [{"session_id": settled, "completion_token": str(uuid.uuid4())}]
                    },
                    headers=headers,
                )
                assert denied.status_code == 401, headers
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(serving, timeout=10)


async def collect_frames(lines, until, *, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Every frame up to and including the one `until` accepts.

    The notification assertions below are about the CONTENTS OF A SEQUENCE —
    "exactly one notification after N turn_ends", "zero notifications while a
    child runs" — and `next_frame` cannot express either, because it discards
    everything it skipped past. A frame log is the only shape in which "one"
    and "none" are checkable at all.
    """

    async def read() -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        async for line in lines:
            if not line.startswith("data: "):
                continue
            frame = json.loads(line[6:])
            frames.append(frame)
            if until(frame):
                return frames
        raise AssertionError("stream ended before the expected frame")

    return await asyncio.wait_for(read(), timeout)


async def serve_app(listener, token: str):
    """Start uvicorn on `listener` and wait until it is genuinely accepting."""
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    for _ in range(10000):
        if server.started:
            break
        if serving.done():
            await serving
        await asyncio.sleep(0)
    assert server.started
    return server, serving


@pytest.mark.asyncio
async def test_a_real_turn_emits_exactly_one_notification_after_many_turn_ends(
    headless_tui_env: Path, notifications_on: None, workspace: Path, monkeypatch
):
    """The reported defect, proven fixed at the real transport.

    THE DEFECT: the app toasted on every `turn_end` — which is ONE MODEL CALL,
    not a finished turn — so an agentic turn that called a tool and then
    answered produced a banner per step, each asserting "The agent finished its
    turn." This drives exactly that shape (a `write` tool call, then a text
    answer, so the loop emits two `turn_end`s) over real loopback HTTP with the
    production Session/RuntimeServer/AttachClient and only the provider stream
    scripted.

    What is asserted is the ORDERED FRAME LOG, because the property is a count:
    N `turn_end` frames, one `agent_end`, and exactly ONE `notification` — with
    the model's own last line as its body, which is the content half of the
    feature. A `next_frame`-style assertion could not fail on a second banner.
    """
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server, serving = await serve_app(listener, token)
    runtime = handle = None
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            path = "/v1/desktop/sessions"
            created = await client.post(
                path,
                json={"request_id": "aaaaaaaa-1111-4111-8111-111111111111", "cwd": str(workspace)},
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            target = path + "/" + sid

            from local_operator.tools.builtin import build_write_tool

            written = workspace / "notification-evidence.txt"
            stream = ScriptedStream(
                [
                    tool_call_turn(
                        text="Writing the report.",
                        tool_name="write",
                        tool_call_id="evidence-write",
                        arguments={"path": str(written), "content": "done"},
                    ),
                    text_turn("Wrote the report to notification-evidence.txt."),
                ]
            )
            session = build_session(
                root / "sessions" / sid,
                stream,
                tools=[build_write_tool()],
                cwd=workspace,
                # The bridge SKIPS a session recorded on the test hosting (a test
                # session is not news), and this cell's subject is the bridge's
                # frame contract — one notification after many turn ends — so the
                # session it drives has to be one the bridge does not skip.
                model=E2E_ORACLE_MODEL,
            )
            # NAMED UP FRONT, and not for cosmetics: an unnamed session fires
            # the runtime's one-shot auto-naming errand, which is a real
            # provider call and therefore consumes a `ScriptedStream` entry —
            # shifting every scripted turn by one and collapsing the multi-step
            # turn this test exists to produce. Naming it also makes the
            # banner's title assertion below a real one.
            session.set_conversation_name("Report run", user_set=True)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))

            async with client.stream("GET", target + "/events") as response:
                assert response.status_code == 200
                lines = response.aiter_lines()
                await next_frame(lines, lambda f: f["type"] == "snapshot")

                admitted = await client.post(
                    target + "/messages",
                    json={
                        "request_id": "bbbbbbbb-2222-4222-8222-222222222222",
                        "text": "Write the report",
                    },
                )
                assert admitted.status_code == 200, admitted.text

                # The notification rides the 1 s attention poll, which fires
                # AFTER `agent_end` — so the log has to run until the banner
                # arrives, not until the turn ends.
                frames = await collect_frames(
                    lines, lambda f: f["type"] == "notification", timeout=45
                )

            kinds = [f["payload"].get("type") for f in frames if f["type"] == "event"]
            notifications = [f for f in frames if f["type"] == "notification"]
            turn_ends = kinds.count("turn_end")

            # The shape the defect needs in order to be visible: MORE THAN ONE
            # model call in one logical turn. Without this the test would pass
            # against the buggy code too.
            assert turn_ends >= 2, f"expected a multi-step turn, saw {kinds}"
            assert kinds.count("agent_end") == 1
            assert len(notifications) == 1, (
                f"{turn_ends} turn_end frames and {kinds.count('agent_end')} agent_end "
                f"produced {len(notifications)} notifications"
            )

            payload = notifications[0]["payload"]
            assert payload["kind"] == "complete"
            assert payload["status"] == "Complete"
            assert payload["title"] == "Report run"
            assert payload["title_is_session_name"] is True
            assert payload["body"] == "Wrote the report to notification-evidence.txt."
            assert payload["body_is_snippet"] is True
            assert payload["completion_token"]
            # Prefix is the frame's own kind (round 1, n1) — pinned relative to
            # the kind asserted just above rather than as a separate literal.
            assert payload["dedupe_key"] == (
                f"{payload['kind']}:{sid}:{payload['completion_token']}"
            )
            # The banner follows the receipt state that explains it.
            assert [f["type"] for f in frames].index("attention") < [
                f["type"] for f in frames
            ].index("notification")
            assert written.read_text() == "done", "the turn really ran its tool"

            # The claim is a separate step the renderer takes, and it is
            # single-winner: this is the real HTTP route, not the store.
            claim = await client.post(
                target + "/notified", json={"completion_token": payload["completion_token"]}
            )
            assert claim.status_code == 200, claim.text
            assert claim.json()["result"]["claimed"] is True
            again = await client.post(
                target + "/notified", json={"completion_token": payload["completion_token"]}
            )
            assert again.json()["result"]["claimed"] is False
            # And it did NOT mark the conversation read: notifying is not reading.
            state = await client.get(target)
            attention = state.json()["result"]["payload"]["frontend"]["snapshot"]["attention"]
            assert attention["unseen"] is True

            print(
                f"Real multi-step turn over HTTP: {turn_ends} turn_end + 1 agent_end frames "
                f"produced exactly 1 notification, body={payload['body']!r}; "
                "second /notified claimed=false; session still unseen"
            )
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()


@pytest.mark.asyncio
async def test_a_delegating_turn_stays_silent_until_its_child_settles(
    headless_tui_env: Path, notifications_on: None, workspace: Path, monkeypatch
):
    """The other half of the defect: `agent_end` while children still work.

    The harness's `task` tool returns as soon as a child is REGISTERED, so a
    delegating parent reaches `agent_end` with the work still running. The app
    toasted there, telling the user their task was done while it was not.

    This drives a real registered `task` job over the real bridge and asserts
    ZERO notification frames while the child runs — then releases the child and
    asserts the banner appears only for the re-entry turn its result opens.
    Nothing about this is reconstructed in the bridge: the session's own
    `_publish_attention_outcome` refuses to publish while a `task` job is
    running, so there is no completion row for the bridge to observe.
    """
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server, serving = await serve_app(listener, token)
    runtime = handle = None
    child_release = asyncio.Event()
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            path = "/v1/desktop/sessions"
            created = await client.post(
                path,
                json={"request_id": "cccccccc-3333-4333-8333-333333333333", "cwd": str(workspace)},
            )
            sid = created.json()["result"]["session_id"]
            target = path + "/" + sid

            stream = ScriptedStream(
                [
                    text_turn("Delegated the audit to a subagent."),
                    text_turn("The subagent finished; the audit is clean."),
                ]
            )
            # Same reason as the cell above: the bridge skips a test-hosted
            # session, and this cell measures WHEN the bridge speaks.
            session = build_session(
                root / "sessions" / sid, stream, cwd=workspace, model=E2E_ORACLE_MODEL
            )
            # See the note in the test above: an unnamed session spends a
            # scripted turn on the auto-naming errand, which here would leave
            # the delegating turn with no script and end it as an `error`
            # rather than the `complete` this asserts.
            session.set_conversation_name("Config audit", user_set=True)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))

            async with client.stream("GET", target + "/events") as response:
                lines = response.aiter_lines()
                await next_frame(lines, lambda f: f["type"] == "snapshot")

                # A REAL `task` job on the real manager, registered before the
                # turn ends — exactly what the `task` tool leaves behind when
                # it returns at registration.
                async def child(job_id, signal, progress):
                    await child_release.wait()
                    return "audit clean"

                job_id = session.jobs.register("task", "audit subagent", child)

                def job_status() -> str:
                    """The child's live status. `jobs.get` is Optional by contract."""
                    row = session.jobs.get(job_id)
                    assert row is not None, "the registered task job vanished from the manager"
                    return row.status

                assert job_status() == "running"

                admitted = await client.post(
                    target + "/messages",
                    json={
                        "request_id": "dddddddd-4444-4444-8444-444444444444",
                        "text": "Audit the config",
                    },
                )
                assert admitted.status_code == 200, admitted.text

                # The parent's turn ends here, with the child still working.
                during = await collect_frames(
                    lines,
                    lambda f: f["type"] == "event" and f["payload"].get("type") == "agent_end",
                )
                # Give the 1 s attention poll several chances to fire. A banner
                # for this turn would be the defect, so the wait has to outlast
                # the mechanism that would deliver it.
                await asyncio.sleep(3.0)
                assert job_status() == "running", "the child must still be live"
                assert [f for f in during if f["type"] == "notification"] == []

                # Nothing was even published: the refusal is one layer below
                # every frontend, in the process that owns the job manager.
                from local_operator.session.attention import AttentionStore

                store = AttentionStore(root / "attention.db")
                assert store.state(f"session/{sid}")["completion_token"] is None

                # Release the child. Its result re-enters as a fresh turn, and
                # THAT turn's completion is the notifiable one.
                child_release.set()
                after = await collect_frames(
                    lines, lambda f: f["type"] == "notification", timeout=60
                )

            notifications = [f for f in after if f["type"] == "notification"]
            assert len(notifications) == 1
            payload = notifications[0]["payload"]
            assert payload["kind"] == "complete"
            assert payload["body"] == "The subagent finished; the audit is clean."
            assert job_status() == "completed"

            print(
                "Delegating turn: agent_end with a live task child produced 0 notifications "
                "and no completion row; after the child settled, the re-entry turn produced "
                f"exactly 1, body={payload['body']!r}"
            )
    finally:
        child_release.set()
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()


class RepeatingTextStream:
    """Serves the same text turn to every provider call, and records the calls.

    ``ScriptedStream`` indexes its script by call number and raises past the
    end, which is right where the test owns the parent's whole turn sequence.
    The child launched below is built by PRODUCTION code and the test does not
    own its call count (a naming errand or a continuation is one more call), so
    this double answers any number of calls instead of pinning a sequence it
    cannot know.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[Any] = []

    def __call__(self, request, signal=None):  # noqa: ANN001
        self.requests.append(request)

        async def gen():
            for event in text_turn(self.text):
                yield event

        return gen()


@pytest.mark.asyncio
async def test_a_real_child_transcript_is_readable_through_the_parent_route(
    headless_tui_env: Path, workspace: Path, monkeypatch
):
    """The sidebar's child reader against the ASSEMBLED application (§ 9.1).

    Everything here is production: uvicorn, the bearer/origin gate, the route,
    the adapter, a real parent `Session`, and a real child launched by
    `Session._launch_subagent` — the production path that writes the child's
    transcript, stamps its `origin.json` as a subagent, and persists the
    parent's roster SIDECAR. The only double is the provider stream, as
    everywhere in this stage.

    What the route must do on those real artifacts: return the child's rows
    verbatim in the parent's envelope with `state: "ready"`, page backwards on
    the child's own file, answer 404 `child_not_found` for a pair the parent
    never named, and answer the derived `gone` as an ANSWER once the child's
    directory is removed — a reader cannot tell those two absences apart from
    the roster row, which is why the backend derives it at all.
    """
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server, serving = await serve_app(listener, token)

    from local_operator.resume import ORIGIN_SUBAGENT, session_origin
    from local_operator.session.restored_rows import record_field, roster_records
    from local_operator.session.session import (
        SUBAGENT_ROSTER_SIDECAR,
        _read_roster_sidecar,
    )
    from local_operator.session.transcript import read_transcript_page

    parent_id = "abcdefabcdef"
    parent_dir = root / "sessions" / parent_id
    parent = build_session(parent_dir, RepeatingTextStream("child did the work"), cwd=workspace)
    job_id: str | None = None

    def named_child() -> Path | None:
        """The first child the parent's OWN persisted roster names, or None.

        Reading the sidecar rather than a private accessor keeps the test on
        the same artifact the route's containment proof consumes.
        """
        payload = _read_roster_sidecar(parent_dir / SUBAGENT_ROSTER_SIDECAR)
        for record in roster_records(payload):
            raw = record_field(record, "session_dir")
            if raw:
                return Path(str(raw))
        return None

    async def settled() -> Path:
        """The child's directory once the child has STOPPED writing.

        A running child appends as it goes, so a test that reads its file twice
        and demands equality would assert something the design does not promise
        — the route returns what was on disk at ITS read instant, and a
        half-written line degrades to "not there yet" (that is the whole reason
        the reader is a fresh `Transcript` per call). Quiescence is what makes
        the comparison meaningful: the launched job reaches `completed` and the
        file's size and mtime stop moving.
        """
        async with asyncio.timeout(90):
            while True:
                child = named_child()
                job = parent.jobs.get(job_id) if job_id else None
                if child is not None and job is not None and job.status == "completed":
                    transcript = child / "transcript.jsonl"
                    if transcript.is_file():
                        before = transcript.stat()
                        await asyncio.sleep(0.3)
                        after = transcript.stat()
                        if (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime):
                            return child
                await asyncio.sleep(0.05)

    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            capabilities = (await client.get("/v1/capabilities")).json()["result"]
            assert capabilities["features"]["subagent_transcript"] == 1

            await parent.async_init()
            job_id = parent._launch_subagent(label="audit", prompt="audit the config")
            child_dir = await settled()
            child_id = child_dir.name
            assert session_origin(child_dir) == ORIGIN_SUBAGENT
            assert len(child_id) == 12 and child_id == child_id.lower()

            url = f"/v1/desktop/sessions/{parent_id}/children/{child_id}/transcript"
            response = await client.get(url)
            assert response.status_code == 200, response.text
            result = response.json()["result"]
            expected = [
                json.loads(row.to_json())
                for row in read_transcript_page(child_dir, limit=500).entries
            ]
            assert expected, "the real child wrote no transcript rows"
            assert result == {
                "entries": expected,
                "has_more": False,
                "cursor_missing": False,
                "state": "ready",
            }
            assert response.headers["cache-control"] == "no-store"

            # Paging runs on the CHILD's file, and `has_more` is the child's.
            if len(expected) >= 2:
                tail = (await client.get(url + "?limit=1")).json()["result"]
                assert [row["id"] for row in tail["entries"]] == [expected[-1]["id"]]
                assert tail["has_more"] is True
                older = (await client.get(url + f"?before_id={expected[-1]['id']}&limit=1")).json()
                assert [row["id"] for row in older["result"]["entries"]] == [expected[-2]["id"]]

            # A conversation this parent never launched is not its child, and
            # the refusal is the contract's coded 404 rather than a bare one.
            refused = await client.get(
                f"/v1/desktop/sessions/{parent_id}/children/{'fedcba987654'}/transcript"
            )
            assert refused.status_code == 404
            assert refused.json()["detail"]["code"] == "child_not_found"

            # `gone` is an ANSWER: the roster still names the child, so the
            # read is legal and only the filesystem says the rows are final.
            child_dir.rename(root / "retired-child")
            gone = (await client.get(url)).json()["result"]
            assert gone == {
                "entries": [],
                "has_more": False,
                "cursor_missing": False,
                "state": "gone",
            }
            print(
                f"child route over real HTTP: {len(expected)} verbatim rows ready, "
                "paging cursor on the child's file, unnamed pair 404 child_not_found, "
                "removed directory 200 gone"
            )
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        await parent.dispose()


@pytest.mark.asyncio
async def test_the_desktop_presence_decides_whether_the_runtime_speaks(
    headless_tui_env: Path, notifications_on: None, workspace: Path, monkeypatch
):
    """RUNG 2, over real loopback HTTP and the production runtime handle.

    The rule the whole ladder exists for, in the one shape that cannot be faked
    by a unit double: a background completion must be announced by the DESKTOP
    while a notify-capable app is connected, and by the RUNTIME the moment that
    lease is withdrawn. A predicate that answered from a mode set at boot passes
    a unit test and fails this one.

    The lease is real (`POST /v1/desktop/presence` against the live feed
    subscription, materialised at ``run/desktop/delivery.json``) and it is
    revoked by closing the SSE socket, which is the signal the design makes
    load-bearing: withdrawing only on a missed heartbeat would leave 45 s in
    which every runtime on the machine stays silent for a banner nobody can
    raise.
    """
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))

    banners: list[dict[str, Any]] = []
    from local_operator.session.runtime.presence import reset_cache
    from local_operator.tui import notify as notify_module

    monkeypatch.setattr(
        notify_module,
        "detached_notify",
        lambda title, body, **kwargs: banners.append({"title": title, "body": body, **kwargs})
        or True,
    )
    session = handle = runtime = None
    try:
        for _ in range(10000):
            if server.started:
                break
            if serving.done():
                await serving
            await asyncio.sleep(0)
        assert server.started
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            created = await client.post(
                "/v1/desktop/sessions",
                json={
                    "request_id": "55555555-5555-4555-8555-555555555555",
                    "cwd": str(workspace),
                },
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            stream = ScriptedStream([text_turn("background work finished")])
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()

            store = AttentionStore(root / "attention.db")
            identity = f"session/{sid}"

            # Nothing connected yet: the runtime is the last rung and must speak.
            store.publish(identity, str(uuid.uuid4()), "a1", "complete")
            await asyncio.to_thread(handle._announce_completion)
            assert len(banners) == 1, banners
            assert banners[0]["session_id"] == sid
            # The claim is spent, so the next attempt for THIS completion is a
            # no-op until another surface hands it back.
            await asyncio.to_thread(handle._announce_completion)
            assert len(banners) == 1

            async with client.stream("GET", "/v1/desktop/events") as feed:
                assert feed.status_code == 200
                feed_lines = feed.aiter_lines()
                opened = await next_frame(feed_lines, lambda f: f["type"] == "open")
                beat = await client.post(
                    "/v1/desktop/presence",
                    json={
                        "subscription_id": opened["payload"]["subscription_id"],
                        "can_notify": True,
                        "can_notify_kinds": ["complete", "error"],
                        "window": {
                            "exists": True,
                            "focused": True,
                            "visible": True,
                            "minimized": False,
                        },
                    },
                )
                assert beat.status_code == 200, beat.text
                assert beat.json()["result"]["lease_seconds"] == 45
                reset_cache()

                store.publish(identity, str(uuid.uuid4()), "a2", "complete")
                await asyncio.to_thread(handle._announce_completion)
                assert len(banners) == 1, "a notify-capable desktop must silence the runtime"

            # The socket is closed, so the lease is revoked with it.
            from local_operator.session.runtime.presence import desktop_delivery_present

            for _ in range(100):
                await asyncio.sleep(0.05)
                reset_cache()
                if not desktop_delivery_present(root, "complete"):
                    break
            assert not desktop_delivery_present(root, "complete")
            await asyncio.to_thread(handle._announce_completion)
            assert len(banners) == 2, "the runtime stayed silent for a withdrawn lease"
    finally:
        if session is not None:
            await session.dispose()
        server.should_exit = True
        await serving
        engine = getattr(app.state, "desktop_feed", None)
        if engine is not None:
            await engine.close()
        app.state.desktop_feed = None
        app.state.desktop_sessions = None
        reset_cache()
