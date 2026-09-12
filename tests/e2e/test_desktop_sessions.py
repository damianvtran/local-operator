"""Real HTTP + canonical Session/ServingSessionHandle/RuntimeServer/AttachClient.

Only the provider stream is scripted. No session, socket, admission, transcript,
gate or bridge is mocked: this catches the seams a green adapter suite cannot.
"""

import asyncio
import json
import os
import secrets
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from local_operator.mobile.attach_client import AttachClient
from local_operator.server.app import app
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn, tool_call_turn

pytestmark = pytest.mark.e2e


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
                    text_turn("The canonical runtime answered."),
                    text_turn("The team request arrived once."),
                    text_turn("The image arrived without invented text."),
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
                duplicate = await client.post(
                    target + "/commands",
                    json={
                        "request_id": "22222222-2222-4222-8222-222222222222",
                        "command": "goal",
                        "args": "Preserve one identity",
                    },
                )
                assert duplicate.json()["result"]["replayed"]
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
                assert len(stream.requests) == 1
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
                assert len(stream.requests) == 2
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
                assert len(stream.requests) == 3
                print(
                    "Image-only prompt admitted200 without synthetic text; "
                    "one durable user row; retry did not duplicate image"
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
    headless_tui_env: Path, workspace: Path, monkeypatch
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
                root / "sessions" / sid, stream, tools=[build_write_tool()], cwd=workspace
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
            assert payload["dedupe_key"] == f"complete:{sid}:{payload['completion_token']}"
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
    headless_tui_env: Path, workspace: Path, monkeypatch
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
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
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
