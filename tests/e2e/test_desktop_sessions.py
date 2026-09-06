"""Real HTTP + canonical Session/OwnedSessionHandle/RuntimeServer/AttachClient.

Only the provider stream is scripted. No session, socket, admission, transcript,
gate or bridge is mocked: this catches the seams a green adapter suite cannot.
"""

import asyncio
import json
import os
import secrets
import socket
from pathlib import Path

import httpx
import pytest
import uvicorn

from local_operator.mobile.attach_client import AttachClient
from local_operator.server.app import app
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from tests.e2e.harness import ScriptedStream, build_session, text_turn

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
            handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            # The assembled test owns this in-process Session; production's
            # process launcher writes the same claim marker before publishing.
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
            from local_operator.mobile.attach_client import find_owner_record

            record, _ = find_owner_record(root, sid)
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
