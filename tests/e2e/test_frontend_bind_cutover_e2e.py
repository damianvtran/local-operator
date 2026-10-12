"""Native cutover over HTTP/sockets; child provider alone is scripted.

The fixture child has an independent real PID/record. This deliberately does
not validate the production launcher/provider bootstrap. Two owners in one PID
would overwrite the native per-PID discovery record and manufacture a 503.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import secrets
import signal
import socket
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from local_operator.harness.types import Message, TextContent
from local_operator.server.app import app
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn

pytestmark = pytest.mark.e2e


async def _child_owner(sid: str) -> None:
    root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    session = build_session(root / "sessions" / sid, ScriptedStream([text_turn("Child answer")]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    runtime = RuntimeServer(handle, kind="daemon")
    await runtime.start_in_process()
    path = runtime.record_path
    (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
    print(json.dumps({"pid": os.getpid(), "sid": sid, "path": str(path)}), flush=True)
    stopped = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
    try:
        await stopped.wait()
    finally:
        await runtime.aclose()
        await handle.dispose()
        assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("handoff", [False, True])
async def test_native_fork_cutover_keeps_prompt_receipt_and_replay(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch, handoff: bool
) -> None:
    from local_operator.mobile.attach_client import find_runtime_record

    root = headless_tui_env
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", secrets.token_hex(32))
    directory = root / "sessions" / uuid.uuid4().hex[:12]
    await seed_transcript(
        directory,
        [
            Message(role="user", content=[TextContent(text="Synthetic seed")]),
            Message(role="assistant", content=[TextContent(text="Synthetic seed reply")]),
        ],
    )
    session = build_session(directory, ScriptedStream([]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    runtime = RuntimeServer(handle, kind="daemon")
    await runtime.start_in_process()
    record_path = runtime.record_path
    sid = session.session_id
    (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
    registered, release, pending = asyncio.Event(), asyncio.Event(), asyncio.Event()
    armed = taken = False
    child: asyncio.subprocess.Process | None = None
    child_path: Path | None = None
    spawned: list[str] = []
    native_sub = ServingSessionHandle.subscribe_frontend
    native_grace = RuntimeServer._bind_with_grace
    native_relay = RuntimeServer._relay_frontend_to_on_loop
    native_slash = ServingSessionHandle.run_slash_authoritative

    async def stop_child() -> None:
        nonlocal child
        if child is None:
            return
        if child.returncode is None:
            child.terminate()
        try:
            await asyncio.wait_for(child.wait(), 10)
        except TimeoutError:
            child.kill()
            await child.wait()
            raise
        assert child.returncode == 0
        assert child_path is not None and not child_path.exists()
        child = None

    @functools.wraps(native_sub)
    async def gated_sub(owner: Any, callback: Any, **kwargs: Any) -> Any:
        sub = await native_sub(owner, callback, **kwargs)
        if armed and taken and owner is handle and not registered.is_set():
            registered.set()
            await release.wait()
        return sub

    @functools.wraps(native_grace)
    async def choose(server: Any, task: Any) -> Any:
        nonlocal taken
        if server is not runtime or not armed or taken:
            return await native_grace(server, task)
        taken = True
        await registered.wait()
        session.set_conversation_name("Native cutover edge")
        await pending.wait()
        if handoff:
            return None
        release.set()
        return await native_grace(server, task)

    @functools.wraps(native_relay)
    def relay(server: Any, conn: Any, data: Any, *args: Any) -> None:
        native_relay(server, conn, data, *args)
        if server is runtime and conn.frontend_pending:
            pending.set()

    @functools.wraps(native_slash)
    async def fork(owner: Any, command: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal child, child_path
        result = await native_slash(owner, command, *args, **kwargs)
        if command != "fork":
            return result
        await stop_child()
        csid = result["data"]["session_id"]
        spawned.append(csid)
        # Explicit allowlist: no ambient CMUX/LOP/XPC or provider credentials.
        env = {
            k: os.environ[k]
            for k in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR", "TMPDIR", "PATH", "TERM")
            if k in os.environ
        }
        env.update(
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
            PYTHONDONTWRITEBYTECODE="1",
            LOCAL_OPERATOR_NO_NOTIFICATIONS="1",
            LOCAL_OPERATOR_NO_TERMINAL_TITLE="1",
        )
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            __file__,
            "--child",
            csid,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert child.stdout is not None
        assert child.stdout is not None
        ready = json.loads(await asyncio.wait_for(child.stdout.readline(), 15))
        child_path = Path(ready["path"])
        assert ready["pid"] != os.getpid() and child_path != record_path
        for wanted in (sid, csid):
            found, _ = await asyncio.to_thread(find_runtime_record, root, wanted)
            assert found is not None and found.session_id == wanted
        return result

    monkeypatch.setattr(ServingSessionHandle, "subscribe_frontend", gated_sub)
    monkeypatch.setattr(RuntimeServer, "_bind_with_grace", choose)
    monkeypatch.setattr(RuntimeServer, "_relay_frontend_to_on_loop", relay)
    monkeypatch.setattr(ServingSessionHandle, "run_slash_authoritative", fork)
    # Prevent external tokenizer/catalogue fetches without altering model replies
    # or loopback/unix IPC. Scope restoration belongs to monkeypatch.
    dns = socket.getaddrinfo

    def local_dns(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host not in ("127.0.0.1", "::1", "localhost", None):
            raise OSError("external DNS is unavailable in this isolated fixture")
        return dns(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", local_dns)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    ready_event = asyncio.Event()
    startup = server.startup

    async def started(*args: Any, **kwargs: Any) -> None:
        await startup(*args, **kwargs)
        ready_event.set()

    monkeypatch.setattr(server, "startup", started)
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await asyncio.wait_for(ready_event.wait(), 15)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}",
            timeout=20,
            headers={"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]},
        ) as client:
            target = f"/v1/desktop/sessions/{sid}/fork"
            first = await client.post(
                target, json={"request_id": str(uuid.uuid4()), "boundary": "next_safe"}
            )
            assert first.status_code == 200, first.text
            first_child = first.json()["result"]["data"]["session_id"]
            assert (
                await client.get(f"/v1/desktop/sessions/{first_child}/history")
            ).status_code == 200
            await stop_child()
            armed = True
            request_id = str(uuid.uuid4())
            body = {"request_id": request_id, "boundary": "next_safe", "message": "Cutover prompt"}
            response = await client.post(target, json=body)
            assert response.status_code == 200, response.text
            data = response.json()["result"]["data"]
            assert data["admission"]["status"] == "admitted"
            replay = await client.post(target, json=body)
            assert replay.status_code == 200 and replay.json()["result"]["replayed"]
            assert replay.json()["result"]["data"]["session_id"] == data["session_id"]
            assert len(spawned) == 2
            rows = [
                json.loads(line)
                for line in (root / "sessions" / data["session_id"] / "transcript.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            matching = [
                row
                for row in rows
                if row.get("type") == "message"
                and row.get("payload", {}).get("producer_command_id") == request_id
            ]
            assert len(matching) == 1 and matching[0]["id"] == request_id
            assert not runtime._closed.is_set() and not handle._disposing
            print(f"Native cutover handoff={handoff}: HTTP200/replay, exact UUID row {request_id}")
    finally:
        release.set()
        await stop_child()
        server.should_exit = True
        await asyncio.wait_for(serving, 15)
        listener.close()
        await runtime.aclose()
        await handle.dispose()
        assert not record_path.exists()


if __name__ == "__main__":
    # The isolated child must never invoke keychain/native provisioning or reach
    # an external tokenizer/provider while running its real serving machinery.
    def _audit(event: str, args: Any) -> None:
        if event == "subprocess.Popen":
            raise RuntimeError("child subprocess/keychain access is forbidden")
        if event == "socket.getaddrinfo" and args[0] not in ("127.0.0.1", "::1", "localhost", None):
            raise OSError("child external DNS is forbidden")

    sys.addaudithook(_audit)
    asyncio.run(_child_owner(sys.argv[2]))
