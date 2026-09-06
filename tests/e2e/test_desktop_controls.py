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
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
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
            # offered on the desktop (no `desktop_destination` — today just
            # `/mobile`, whose provisioning has no desktop proxy). Derived from
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
            assert sum(len(row["aliases"]) for row in catalog) == 8
            created = await client.post(
                "/v1/desktop/sessions", json={"request_id": request_id(), "cwd": str(workspace)}
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            target = "/v1/desktop/sessions/" + sid
            stream = ControlledStream(
                [
                    text_turn("Seed answer"),
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
            handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
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
            await command("goal", "Complete two steps")
            loop_id = request_id()
            started = await command("loop", "2", loop_id)
            assert started["data"]["status"] == "running"
            await until(lambda: handle._goal_loop.state["status"] == "completed")
            await command("loop", "2", loop_id)
            assert len(stream.requests) == 3
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
            assert len(stream.requests) == 6
            stream.block = True
            await command("loop", "3")
            await asyncio.wait_for(stream.started.wait(), 15)
            cancelled = await command("loop", "cancel")
            assert cancelled["data"]["status"] == "cancelled"
            await until(lambda: not session.is_streaming and not handle._prompt_queue)
            assert len(stream.requests) == 7
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
                find_owner_record,
            )

            record, _ = await asyncio.to_thread(find_owner_record, root, spawned_child)
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
