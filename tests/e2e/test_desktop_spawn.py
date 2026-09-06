"""Real detached process launch through the authenticated canonical HTTP API.

No monkeypatch of engage_runtime, session construction, AttachClient or the
owner. The built-in test provider avoids external credentials; the runtime is
started under isolated HOME/config/TMPDIR and stopped through its own protocol.
"""

import asyncio
import json
import secrets
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn

from local_operator.mobile.attach_client import AttachClient, find_owner_record
from local_operator.server.app import app
from tests.e2e.test_desktop_sessions import next_frame

pytestmark = pytest.mark.e2e


@asynccontextmanager
async def serve_http():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(10000):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0)
        assert server.started
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 30)
        listener.close()


@pytest.mark.asyncio
async def test_spawn_and_reopen_after_http_restart(
    headless_tui_env: Path, workspace: Path, monkeypatch
):
    root = headless_tui_env
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", secrets.token_hex(32))
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    import os

    headers = {"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]}
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    sid = None
    try:
        async with (
            serve_http() as url,
            httpx.AsyncClient(base_url=url, headers=headers, timeout=60) as client,
        ):
            result = await client.post(
                "/v1/desktop/sessions",
                json={
                    "request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "cwd": str(workspace),
                },
            )
            assert result.status_code == 200, result.text
            sid = result.json()["result"]["session_id"]
            path = "/v1/desktop/sessions/" + sid
            async with client.stream("GET", path + "/events") as stream:
                lines = stream.aiter_lines()
                opened = await next_frame(lines, lambda f: f["type"] == "open")
                initial = await next_frame(lines, lambda f: f["type"] == "snapshot")
                assert initial["payload"]["cold"]
                watch = await client.post(
                    path + "/watch",
                    json={
                        "subscription_id": opened["payload"]["subscription_id"],
                        "visible": True,
                        "can_notify": False,
                    },
                )
                assert watch.status_code == 200
                # This owner control is the FIRST operation allowed to spawn.
                renamed = await client.post(
                    path + "/commands",
                    json={
                        "request_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                        "command": "rename",
                        "args": "Actual spawned HTTP runtime",
                    },
                )
                assert renamed.status_code == 200, renamed.text
                record, _ = await asyncio.to_thread(find_owner_record, root, sid)
                assert record is not None and record.pid != os.getpid()
                owner_pid = record.pid
                body = {
                    "request_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                    "text": "Reply briefly for the HTTP integration test",
                }
                admitted = await client.post(path + "/messages", json=body)
                assert admitted.status_code == 200, admitted.text
                end = await next_frame(
                    lines,
                    lambda f: f["type"] == "event" and f["payload"].get("type") == "agent_end",
                )
                old_epoch, old_seq = end["epoch"], end["seq"]
                transcript = await client.get(path + "/history")
                assert body["text"] in transcript.text
                rows_before = transcript.json()["result"]["entries"]
                print(
                    "Real HTTP control spawned detached owner PID",
                    owner_pid,
                    "and canonical prompt reached agent_end and durable history",
                )

        # Recreate the assembled HTTP lifespan with NO bridge or receipt object
        # reused. Durable request receipts and canonical identity remain valid.
        async with (
            serve_http() as url,
            httpx.AsyncClient(base_url=url, headers=headers, timeout=60) as client,
        ):
            reopened = await client.get(path)
            assert reopened.status_code == 200, reopened.text
            state = reopened.json()["result"]["payload"]["frontend"]["snapshot"]
            assert (
                state["session_id"] == sid
                and state["conversation_title"] == "Actual spawned HTTP runtime"
            )
            retried = await client.post(path + "/messages", json=body)
            assert retried.status_code == 200 and retried.json()["result"]["replayed"]
            transcript = await client.get(path + "/history")
            assert (
                sum(
                    row["id"] == body["request_id"]
                    for row in transcript.json()["result"]["entries"]
                )
                == 1
            )
            assert sum(row["id"] == body["request_id"] for row in rows_before) == 1
            async with client.stream(
                "GET", path + "/events", params={"epoch": old_epoch, "after_seq": old_seq}
            ) as stream:
                lines = stream.aiter_lines()
                reset = await next_frame(lines, lambda f: f["type"] == "open")
                assert reset["payload"]["gap"] and reset["epoch"] != old_epoch
                authoritative = await next_frame(lines, lambda f: f["type"] == "snapshot")
                assert body["text"] in json.dumps(authoritative["payload"]["history"])
            print(
                "HTTP lifespan restart: same canonical ID/title200, durable retry "
                "receipt200 and exactly one user row; old stream epoch forces gap + "
                "authoritative history"
            )
    finally:
        if sid is not None:
            record, _ = await asyncio.to_thread(find_owner_record, root, sid)
            if record is not None:
                client = AttachClient(lambda _: None, lambda _: None)
                await client.connect(record, sid)
                try:
                    await client.request_stop()
                finally:
                    await client.detach()
                print("Test-owned detached owner stopped through canonical control protocol")
