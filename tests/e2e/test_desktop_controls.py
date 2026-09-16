"""Assembled HTTP/owner/MCP controls; only external model replies are scripted."""

import asyncio
import json
import os
import secrets
import socket
import sys
import uuid
from pathlib import Path

import httpx
import pytest
import uvicorn

from local_operator.mcp.manager import McpManager
from local_operator.server.app import app
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.slash_commands import SLASH_COMMANDS
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e


class ControlledStream(ScriptedStream):
    def __init__(self, turns):
        super().__init__(turns)
        self.block = False
        self.started = asyncio.Event()

    def __call__(self, request, signal=None):
        if not self.block:
            return super().__call__(request, signal)
        self.requests.append(request)

        async def blocked():
            self.started.set()
            assert signal is not None
            await signal.wait()
            for event in text_turn("Interrupted"):
                yield event

        return blocked()


def request_id():
    return str(uuid.uuid4())


async def until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_desktop_control_surface(headless_tui_env: Path, workspace: Path, monkeypatch):
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
    runtime = handle = manager = None
    spawned_child = None
    try:
        await until(lambda: server.started)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            for route in (
                "/v1/desktop/commands",
                "/v1/desktop/models",
                "/v1/desktop/usage",
                "/v1/desktop/analytics",
            ):
                assert (await client.get(route)).status_code == 401
                assert (
                    await client.get(route, headers={"Authorization": "Bearer wrong"})
                ).status_code == 401
                assert (
                    await client.get(
                        route,
                        headers={
                            "Authorization": "Bearer " + token,
                            "Origin": "https://evil.example",
                        },
                    )
                ).status_code == 403
            client.headers["Authorization"] = "Bearer " + token
            catalog = (await client.get("/v1/desktop/commands")).json()["result"]["commands"]
            # The catalogue is the registry MINUS the entries deliberately not
            # offered on the desktop (no `desktop_destination`): `/mobile`,
            # whose provisioning has no desktop proxy. `/info` used to be the
            # other one — its every field describes the PROCESS AND HOST it
            # runs in (install prefix, resolved import path, pid, control port,
            # this machine's session registry) — and it is now OFFERED instead,
            # because `GET /v1/desktop/info` serves that same host read rather
            # than a proxied guess at another machine, and the caveat it used to
            # carry is the panel's own host label. Derived from
            # the registry rather than pinned as a literal, because the equality
            # against EVERY registry name could only ever hold if the withheld
            # commands were offered, which is the bug the field exists to
            # prevent; the withheld set is asserted separately so silently
            # dropping a command from the desktop still fails here.
            offered = {spec.name for spec in SLASH_COMMANDS if spec.desktop_destination}
            withheld = {spec.name for spec in SLASH_COMMANDS if not spec.desktop_destination}
            assert {row["name"] for row in catalog} == offered
            assert withheld == {"mobile"}
            assert len(catalog) == len(SLASH_COMMANDS) - len(withheld)
            # The literal is DELIBERATE, unlike its three neighbours. The
            # catalogue's `aliases` are copied straight off `spec.aliases`
            # (`desktop_commands.py:39`), so deriving this bound from
            # SLASH_COMMANDS would compare the registry with itself and pass for
            # any alias added or dropped — the one thing this line exists to
            # notice. Update the number when you intend to change the offered
            # alias surface; a diff here is the review prompt.
            assert sum(len(row["aliases"]) for row in catalog) == 9
            created = await client.post(
                "/v1/desktop/sessions", json={"request_id": request_id(), "cwd": str(workspace)}
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            target = "/v1/desktop/sessions/" + sid
            stream = ControlledStream(
                [
                    text_turn("Seed answer"),
                    # The goal command's own admitted turn, in call order: the
                    # standing objective is stored AND its argument is submitted,
                    # so it consumes a provider turn before the loop starts.
                    text_turn("Goal turn answered"),
                    text_turn("First loop step"),
                    text_turn("Second loop step"),
                    text_turn("Private aside answer"),
                    text_turn("Goal work finished"),
                    text_turn("VERDICT: ACHIEVED\nFixture verified"),
                ]
            )
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
            manager = McpManager(str(workspace))
            session.mcp_manager = manager
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))

            async def command(name, args="", rid=None):
                result = await client.post(
                    target + "/commands",
                    json={"request_id": rid or request_id(), "command": name, "args": args},
                )
                assert result.status_code == 200, result.text
                return result.json()["result"]["result"]

            # Every OFFERED canonical name and alias is admitted. Native
            # receipts must carry a destination; owner receipts are not counted
            # as UI proof. A withheld command is asserted the other way round:
            # the route refuses it, which is what keeps "not offered" from
            # degrading into "offered and broken".
            for spec in SLASH_COMMANDS:
                for name in (spec.name, *spec.aliases):
                    if not spec.desktop_destination:
                        refused = await client.post(
                            target + "/commands",
                            json={"request_id": request_id(), "command": name, "args": ""},
                        )
                        assert refused.status_code == 422, refused.text
                        continue
                    result = await command(name)
                    if result["kind"] == "native_action":
                        assert result["destination"] == spec.desktop_destination
                        assert result["session_id"] == sid
            assert not stream.requests
            print(
                (
                    f"Command census: all {len(catalog)} offered canonical +8 "
                    "aliases HTTP200; actionable native destinations, no model "
                    "prompts"
                )
            )

            for route in (
                "/v1/desktop/models",
                "/v1/desktop/usage",
                "/v1/desktop/analytics",
                target + "/failovers",
                "/v1/desktop/skills?session_id=" + sid,
            ):
                response = await client.get(route)
                assert response.status_code == 200, response.text
            for name in ("model", "effort", "approvals", "goal", "team", "agent"):
                response = await client.get(target + "/command-entities", params={"command": name})
                assert response.status_code == 200, response.text
            original_model = session.model
            session.set_model(
                original_model.model_copy(
                    update={"reasoning_efforts": ("low", "high"), "reasoning_effort": "high"}
                )
            )
            choices = await client.get(target + "/command-entities?command=effort")
            assert choices.json()["result"]["entities"] == [{"value": "low"}, {"value": "high"}]
            assert choices.json()["result"]["current"] == "high"
            session.set_model(original_model)
            assert (await client.get("/v1/desktop/usage?provider=nonexistent")).status_code == 422
            assert (
                await client.get("/v1/desktop/analytics?since_ms=2&until_ms=1")
            ).status_code == 422

            # Secret bytes are generated here and never printed on assertion
            # failure. Search durable transcript and receipts, not just responses.
            secret = secrets.token_hex(24)
            saved = await client.post(
                target + "/credentials",
                json={"action": "store", "key": "DESKTOP_TEST_SECRET", "value": secret},
            )
            assert saved.status_code == 200
            assert secret not in saved.text
            listed = await client.post(target + "/credentials", json={"action": "list"})
            assert listed.status_code == 200, listed.text
            assert "DESKTOP_TEST_SECRET" in listed.text and secret not in listed.text
            rejected = await client.post(
                target + "/commands",
                json={"request_id": request_id(), "command": "cred", "args": secret},
            )
            assert rejected.status_code == 422 and secret not in rejected.text
            assert (
                await client.post(
                    target + "/credentials", json={"action": "forget", "key": "DESKTOP_TEST_SECRET"}
                )
            ).status_code == 422
            forgot = await client.post(
                target + "/credentials",
                json={"action": "forget", "key": "DESKTOP_TEST_SECRET", "confirmed": True},
            )
            assert forgot.json()["result"]["data"]["removed"]
            for p in [root / "sessions" / sid / "transcript.jsonl", root / "desktop-receipts.db"]:
                if p.exists():
                    assert secret.encode() not in p.read_bytes()
            print(
                (
                    "Credential store/list/confirmed forget HTTP200; missing "
                    "confirmation422; secret absent from replies/transcript/receipt DB"
                )
            )

            chart = await command("team", "chart missing")
            assert chart["kind"] == "native_action" and chart["data"]["mode"] == "chart"
            defaults = await command("approvals", "default auto")
            assert defaults["data"]["scope"] == "default"
            assert defaults["data"]["submit"]["path"] == "/v1/settings/tool_approval_mode"
            await command("rename", "Control fixture")
            await session.prompt("Seed history")
            before = (await client.get(target + "/history")).json()["result"]
            await command("clear")
            assert (await client.get(target + "/history")).json()["result"] == before
            # ``/goal <text>`` is the one command that BOTH stores a standing
            # objective and submits its argument as an ordinary user turn, and
            # the desktop host is the one that completes the receipt (see
            # ``desktop_viewer_must_submit`` in the route and
            # ``test_desktop_goal_admission`` for the rule and its guards). Its
            # turn is counted with every other provider call below.
            #
            # THE THREE CENSUS NUMBERS (here, and after the achieved loop and
            # the cancelled live loop) are positional hand-counts, and each is
            # this command's ONE turn more than it would be without it. The
            # scripted answers above shift with it, so the next edit that adds
            # or removes a turn must move all three — they are a census of
            # provider calls, deliberately absolute so that an unnoticed extra
            # turn fails here rather than passing as a proportional difference.
            goal_stored = await command("goal", "Complete two steps")
            assert goal_stored["admission"]["status"] == "admitted"
            loop_id = request_id()
            started = await command("loop", "2", loop_id)
            assert started["data"]["status"] == "running"
            await until(lambda: handle._goal_loop.state["status"] == "completed")
            await command("loop", "2", loop_id)
            assert len(stream.requests) == 4
            snapshot = (await client.get(target)).json()["result"]["payload"]["frontend"][
                "snapshot"
            ]
            assert snapshot["loop"]["completed"] == 2
            print(
                (
                    "Canonical count loop ran two actual model turns, persisted state, "
                    "replay did not restart; /clear preserved history"
                )
            )

            prior = (await client.get(target + "/history")).json()["result"]
            aside = await client.post(
                target + "/asides", json={"request_id": request_id(), "text": "Private question"}
            )
            assert aside.status_code == 200, aside.text
            aside_id = aside.json()["result"]["data"]["aside_id"]
            assert (await client.get(target + "/history")).json()["result"] == prior
            adopt_id = request_id()
            adopted = await client.post(
                target + f"/asides/{aside_id}/adopt",
                json={"request_id": adopt_id, "confirmed": True},
            )
            assert adopted.status_code == 200, adopted.text
            assert "Private aside answer" in (await client.get(target + "/history")).text
            assert (
                await client.post(
                    target + f"/asides/{aside_id}/adopt",
                    json={"request_id": adopt_id, "confirmed": True},
                )
            ).json()["result"]["replayed"]
            print(
                (
                    "Aside HTTP200 off-record until confirmed adoption; canonical history"
                    " then contains exact exchange once"
                )
            )

            forked = await client.post(
                target + "/fork", json={"request_id": request_id(), "boundary": "next_safe"}
            )
            assert forked.status_code == 200, forked.text
            child = forked.json()["result"]["data"]["session_id"]
            assert child != sid
            assert (
                "Private aside answer"
                in (await client.get("/v1/desktop/sessions/" + child + "/history")).text
            )
            print(
                (
                    "Canonical safe-boundary fork HTTP200, distinct identity and "
                    "inherited history; parent remains intact"
                )
            )

            fork_request = {
                "request_id": request_id(),
                "message": "Fork request once",
                "boundary": "next_safe",
            }
            fork_with_prompt = await client.post(target + "/fork", json=fork_request)
            assert fork_with_prompt.status_code == 200, fork_with_prompt.text
            spawned_child = fork_with_prompt.json()["result"]["data"]["session_id"]
            assert fork_with_prompt.json()["result"]["data"]["admission"]["status"] == "admitted"
            replayed_fork = await client.post(target + "/fork", json=fork_request)
            assert replayed_fork.json()["result"]["data"]["session_id"] == spawned_child
            assert replayed_fork.json()["result"]["replayed"]
            child_history = await client.get("/v1/desktop/sessions/" + spawned_child + "/history")
            assert (
                sum(
                    row["id"] == fork_request["request_id"]
                    for row in child_history.json()["result"]["entries"]
                )
                == 1
            )
            print(
                "Fork optional request: real detached child admitted the retained UUID once; "
                "retry returned the same child and one durable user row"
            )

            await command("loop", "Verify the fixture goal")
            await until(lambda: handle._goal_loop.state["status"] == "achieved")
            assert session.goal == "Complete two steps"
            assert len(stream.requests) == 7
            stream.block = True
            await command("loop", "3")
            await asyncio.wait_for(stream.started.wait(), 15)
            cancelled = await command("loop", "cancel")
            assert cancelled["data"]["status"] == "cancelled"
            await until(lambda: not session.is_streaming and not handle._prompt_queue)
            assert len(stream.requests) == 8
            print(
                (
                    "Goal loop judged ACHIEVED off-record without replacing standing "
                    "goal; live count loop cancellation aborted its own turn and "
                    "submitted no next iteration"
                )
            )

            mcp = target + "/mcp"
            bad = await client.post(
                mcp,
                json={"action": "add", "name": "fixture", "url": "https://user:secret@example.org"},
            )
            assert bad.status_code == 422 and "user:secret" not in bad.text
            added = await client.post(
                mcp,
                json={
                    "action": "add",
                    "name": "fixture",
                    "command": sys.executable,
                    "args": [str(Path(__file__).with_name("desktop_mcp_fixture.py"))],
                },
            )
            assert added.status_code == 200, added.text
            await manager.wait_for_connection("fixture")
            assert manager.get_connection_status("fixture") == "connected"
            assert len(manager.get_server_tools("fixture")) == 1
            assert "fixture" in json.loads((root / "mcp.json").read_text())["mcpServers"]
            unsupported = await client.post(mcp, json={"action": "login", "name": "fixture"})
            assert unsupported.status_code == 409
            assert (
                await client.post(mcp, json={"action": "remove", "name": "fixture"})
            ).status_code == 422
            removed = await client.post(
                mcp, json={"action": "remove", "name": "fixture", "confirmed": True}
            )
            assert removed.status_code == 200, removed.text
            assert "fixture" not in manager.get_all_server_names()
            assert "fixture" not in json.loads((root / "mcp.json").read_text())["mcpServers"]
            print(
                (
                    "Real stdio MCP add/connect/tool discovery/remove HTTP200 and file "
                    "side effects; inline secret422, stdio OAuth409, unconfirmed "
                    "remove422"
                )
            )
            stop_payload = {"request_id": request_id(), "targets": [sid, child], "confirmed": False}
            assert (await client.post("/v1/desktop/stop", json=stop_payload)).status_code == 422
            stop_payload["confirmed"] = True
            stopped = await client.post("/v1/desktop/stop", json=stop_payload)
            assert stopped.status_code == 200, stopped.text
            statuses = [row["status"] for row in stopped.json()["result"]["data"]["sessions"]]
            assert statuses == ["stop_requested", "already_stopped"]
            await until(lambda: handle._disposing)
            assert (await client.post("/v1/desktop/stop", json=stop_payload)).json()["result"][
                "replayed"
            ]
            print(
                (
                    "Confirmed selected stop200 acknowledged real owner stop; cold fork "
                    "was not spawned, unconfirmed422, replay did not repeat"
                )
            )
    finally:
        if spawned_child is not None:
            from local_operator.mobile.attach_client import (
                AttachClient,
                find_runtime_record,
            )

            record, _ = await asyncio.to_thread(find_runtime_record, root, spawned_child)
            if record is not None:
                child_client = AttachClient(lambda _: None, lambda _: None)
                await child_client.connect(record, spawned_child)
                try:
                    await child_client.request_stop()
                finally:
                    await child_client.detach()
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
        if manager is not None:
            await manager.disconnect_all()


@pytest.mark.asyncio
async def test_desktop_interrupt_stops_the_turn_and_keeps_the_session(
    headless_tui_env: Path, workspace: Path, monkeypatch
):
    """The Stop button's rung, against the assembled app.

    REPORTED DEFECT (2026-09-15): "When I pressed stop/interrupt, it didn't seem
    to work in local-operator-ui." The composer posted
    ``sessions.command`` with ``command: "stop"``, which is not an
    ``OWNER_COMMAND``, so this API answered a ``native_action`` PRESENTATION for
    ``POST /v1/desktop/stop`` and stopped no turn at all — the transport was
    fine and the button called the wrong op. That half is reproduced below and
    must stay reproduced: it is the reason the fix is a new route rather than a
    change to the command catalogue.

    THE FIX IS THE OTHER RUNG. ``/interrupt`` stops the turn that is running and
    leaves the session, its runtime and its process alive — the phone relay's
    ``abort``, reachable from HTTP. Pointing the button at ``/stop`` instead
    would have ended the user's session under a control promising it would not,
    so the kill switch keeps its meaning and this test proves the two differ by
    driving a SECOND real turn after the interrupt.

    Every failure shape the route declares is exercised here rather than only
    the happy path: a cold session answers ``idle`` without spawning a runtime,
    a bad body shape is a 422, an unknown or malformed session id is a 404, and
    the unauthenticated/browser-originated answers are asserted before the rest.
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
    runtime = handle = None
    try:
        await until(lambda: server.started)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            # The door first, on the real route: no bearer, a wrong bearer, and
            # a browser origin that is not allowed.
            no_auth = await client.post("/v1/desktop/sessions/x/interrupt", json={})
            assert no_auth.status_code == 401, no_auth.text
            wrong = await client.post(
                "/v1/desktop/sessions/x/interrupt",
                json={},
                headers={"Authorization": "Bearer wrong"},
            )
            assert wrong.status_code == 401, wrong.text
            browser = await client.post(
                "/v1/desktop/sessions/x/interrupt",
                json={},
                headers={
                    "Authorization": "Bearer " + token,
                    "Origin": "https://evil.example",
                },
            )
            assert browser.status_code == 403, browser.text
            client.headers["Authorization"] = "Bearer " + token

            created = await client.post(
                "/v1/desktop/sessions", json={"request_id": request_id(), "cwd": str(workspace)}
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            target = "/v1/desktop/sessions/" + sid
            stream = ControlledStream(
                # Repeated because the headless naming worker makes a model call of
                # its own through this same stream: pinning the second turn to an
                # exact index made the script's own bookkeeping, not the fix,
                # decide whether this test passed.
                [text_turn("Seed answer")]
                + [text_turn("Second answer")] * 4
            )
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))

            # A real turn, parked mid-stream by a scripted model that blocks until
            # the abort signal fires.
            stream.block = True
            admitted = await client.post(
                target + "/messages",
                json={"request_id": request_id(), "text": "start a long turn"},
            )
            assert admitted.status_code == 200, admitted.text
            await asyncio.wait_for(stream.started.wait(), 15)
            assert session.is_streaming

            # THE BUG, reproduced exactly as the composer posted it.
            legacy = await client.post(
                target + "/commands",
                json={"request_id": request_id(), "command": "stop"},
            )
            assert legacy.status_code == 200, legacy.text
            legacy_result = legacy.json()["result"]["result"]
            assert legacy_result["kind"] == "native_action", legacy_result
            assert legacy_result["destination"] == "sessions.stop", legacy_result
            assert session.is_streaming, "the old op stopped a turn; this is not the defect"
            print(
                "REPRO (the reported bug): POST /commands {command: 'stop'} answered "
                "HTTP200 native_action -> destinations.sessions.stop and the turn was "
                "STILL streaming; nothing was interrupted"
            )

            # THE FIX: the same press, aimed at the rung that stops a turn.
            press = request_id()
            answer = await client.post(target + "/interrupt", json={"request_id": press})
            assert answer.status_code == 200, answer.text
            result = answer.json()["result"]
            assert result["status"] == "interrupted", result
            # The owner's own sentence, not one this layer composed.
            assert result["receipt"].startswith("stopping this turn"), result
            assert (result["children_running"], result["background_jobs"]) == (0, 0)
            assert result["replayed"] is False
            await until(lambda: not session.is_streaming and not handle._prompt_queue)
            snapshot = (await client.get(target)).json()["result"]["payload"]["frontend"][
                "snapshot"
            ]
            assert snapshot["streaming"] is False
            # NOT the kill switch: the runtime is alive, undisposed and serving.
            assert not handle._disposing
            print(
                f"FIX: POST {target}/interrupt HTTP200 status=interrupted "
                f"receipt={result['receipt']!r}; session still streaming=False and the "
                "runtime was not disposed"
            )

            # The journal replays rather than firing a second interrupt.
            again = await client.post(target + "/interrupt", json={"request_id": press})
            assert again.status_code == 200, again.text
            assert again.json()["result"]["replayed"] is True
            print("IDEMPOTENT: the same request_id replayed the stored receipt")

            # ...and the session is genuinely still usable: a SECOND real turn.
            calls_before = len(stream.requests)
            stream.block = False
            second = await client.post(
                target + "/messages",
                json={"request_id": request_id(), "text": "and now a second turn"},
            )
            assert second.status_code == 200, second.text
            await until(lambda: not session.is_streaming)
            history = (await client.get(target + "/history")).json()["result"]
            assert "Second answer" in json.dumps(history)
            assert len(stream.requests) > calls_before, "the follow-up turn never reached the model"
            print(
                f"AFTER: a second real turn completed on the SAME session "
                f"({len(stream.requests) - calls_before} more model call(s)), so the "
                "interrupt stopped the work and not the session"
            )

            # A WARM session with nothing to stop: the same `idle` answer as a
            # cold one, because `interrupted` is a claim that work was stopped.
            quiet = await client.post(target + "/interrupt", json={"request_id": request_id()})
            assert quiet.status_code == 200, quiet.text
            quiet_result = quiet.json()["result"]
            assert quiet_result["status"] == "idle", quiet_result
            assert quiet_result["receipt"] == "", "an idle answer must not invent a receipt"
            print(
                "IDLE (warm): a settled session answered HTTP200 status=idle with an empty "
                "receipt and nothing stopped"
            )

            # A COLD session: idle, and nothing spawned to answer it.
            cold = await client.post(
                "/v1/desktop/sessions", json={"request_id": request_id(), "cwd": str(workspace)}
            )
            assert cold.status_code == 200, cold.text
            cold_id = cold.json()["result"]["session_id"]
            from local_operator.mobile.attach_client import find_runtime_record

            assert (await asyncio.to_thread(find_runtime_record, root, cold_id))[
                0
            ] is None, "the cold fixture session already had a runtime"
            cold_answer = await client.post(
                f"/v1/desktop/sessions/{cold_id}/interrupt", json={"request_id": request_id()}
            )
            assert cold_answer.status_code == 200, cold_answer.text
            cold_result = cold_answer.json()["result"]
            assert cold_result["status"] == "idle", cold_result
            assert cold_result["receipt"] == "", "an idle answer must not invent a receipt"
            assert (await asyncio.to_thread(find_runtime_record, root, cold_id))[
                0
            ] is None, "the interrupt spawned a runtime for a cold session"
            print(
                "IDLE: a cold session answered HTTP200 status=idle with an empty receipt "
                "and no runtime record was created"
            )

            # The body's own shape, and the two ids that name nothing.
            bad_uuid = await client.post(target + "/interrupt", json={"request_id": "not-a-uuid"})
            assert bad_uuid.status_code == 422, bad_uuid.text
            extra = await client.post(
                target + "/interrupt",
                json={"request_id": request_id(), "confirmed": True},
            )
            assert extra.status_code == 422, extra.text
            unknown = await client.post(
                "/v1/desktop/sessions/000000000000/interrupt", json={"request_id": request_id()}
            )
            assert unknown.status_code == 404, unknown.text
            malformed = await client.post(
                "/v1/desktop/sessions/not-a-session/interrupt", json={"request_id": request_id()}
            )
            assert malformed.status_code == 404, malformed.text
            print(
                "SHAPES: non-uuid request_id 422, an extra field 422, an unknown session "
                "404 and a malformed id 404"
            )
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
