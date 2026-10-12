"""Real HTTP, Session, producer, runtime and replay; no desktop IPC substitute.

The provider is scripted. A named fixture_wait tool deliberately holds one real
execution event on an asyncio gate so the lazy-aside seam can be inspected before
and after materialization. Read is the production tool and touches synthetic data
only. This proves backend/delivery/projection, not a full desktop/browser matrix.
"""

import asyncio
import os
import secrets
import socket
import uuid
from pathlib import Path

import httpx
import pytest
import uvicorn

from local_operator.harness.comms import SubagentComms
from local_operator.harness.types import AgentTool, TextContent, ToolResult
from local_operator.mobile.projection import fold_messages_to_entries
from local_operator.server.app import app
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tools.builtin import build_read_tool
from tests.e2e.harness import ScriptedStream, build_session, text_turn, tool_call_turn

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_materialized_hub_preserves_real_source_identity_and_structured_replay(
    headless_tui_env: Path, workspace: Path, monkeypatch
) -> None:
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    data = workspace / "synthetic.txt"
    data.write_text("Only this isolated fixture file is read.\n")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    runtime = handle = session = None
    released = asyncio.Event()
    waiting = asyncio.Event()

    async def hold(call_id, args, abort, update, context):
        waiting.set()
        await released.wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name="fixture_wait",
            content=[TextContent(text="gate released")],
        )

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
            assert (await client.get("/v1/desktop/sessions")).status_code == 401
            client.headers["Authorization"] = f"Bearer {token}"
            created = await client.post(
                "/v1/desktop/sessions",
                json={"request_id": str(uuid.uuid4()), "cwd": str(workspace)},
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            target = f"/v1/desktop/sessions/{sid}"
            stream = ScriptedStream(
                [
                    tool_call_turn(
                        text="",
                        tool_name="read",
                        tool_call_id="hub-read-1",
                        arguments={"path": str(data)},
                    ),
                    tool_call_turn(
                        text="", tool_name="fixture_wait", tool_call_id="hub-wait-2", arguments={}
                    ),
                    tool_call_turn(
                        text="",
                        tool_name="read",
                        tool_call_id="hub-read-3",
                        arguments={"path": str(data)},
                    ),
                    tool_call_turn(
                        text="",
                        tool_name="read",
                        tool_call_id="hub-read-4",
                        arguments={"path": str(data)},
                    ),
                    text_turn("Isolated hub delivery complete."),
                ]
            )
            gate_tool = AgentTool(
                name="fixture_wait",
                label="Private fixture gate",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                execute=hold,
            )
            session = build_session(
                root / "sessions" / sid, stream, tools=[build_read_tool(), gate_tool], cwd=workspace
            )
            session.set_conversation_name("Synthetic hub E2E", user_set=True)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
            admitted = await client.post(
                target + "/messages",
                json={"request_id": str(uuid.uuid4()), "text": "Read this fixture and wait"},
            )
            assert admitted.status_code == 200, admitted.text
            await asyncio.wait_for(waiting.wait(), 10)
            body = "  First paragraph\n\nFull second paragraph  "
            receipt = SubagentComms(session)._to_child_message(body, expects_reply=True)
            # A withdrawn thunk is no delivery. Neither it nor the valid thunk
            # can paint a receipt before the source is durably materialized.
            session.queue_aside(lambda: None)
            session.queue_aside(lambda: receipt)
            before = await client.get(target + "/history")
            assert before.status_code == 200
            assert receipt.id not in {entry["id"] for entry in before.json()["result"]["entries"]}
            released.set()
            for _ in range(1000):
                after = await client.get(target + "/history")
                entries = after.json()["result"]["entries"]
                finals = [
                    entry
                    for entry in entries
                    if entry.get("payload", {}).get("stop_reason") == "stop"
                ]
                if finals:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("real runtime did not persist its final")
            rows = [entry for entry in entries if entry["id"] == receipt.id]
            assert len(rows) == 1
            assert rows[0]["payload"]["details"]["body"] == body
            results = [
                entry["payload"]["tool_call_id"]
                for entry in entries
                if entry.get("payload", {}).get("role") == "tool"
            ]
            assert results == ["hub-read-1", "hub-wait-2", "hub-read-3", "hub-read-4"]
            replay = fold_messages_to_entries(session.history())
            projected = [row for row in replay if row.id == receipt.id]
            assert len(projected) == 1
            assert projected[0].kind == "parent_message"
            assert projected[0].text == body
            assert projected[0].details["body"] == body
            assert projected[0].details["expects_reply"] is True
            assert projected[0].details["direction"] == "to_child"
            assert "parent_id" not in projected[0].details
            print(
                "HTTP401/200; lazy queued receipt absent; one real producer hub persisted; "
                "four actual tool results; full replay body+intent/sourceID preserved"
            )
    finally:
        released.set()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
        server.should_exit = True
        await serving
        listener.close()
