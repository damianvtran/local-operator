"""Pilot refusal honesty over two real relays, not a substituted local viewer.

Faults affect only this fixture's runtime writes. The half-dead case retains its
socket after the welcome but drops all later frames; the busy control sends an
additional legitimate projection before withholding the expensive frontend sync.
The CLI must distinguish those facts, not infer liveness from a socket with no EOF.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime.server import RuntimeServer
from tests.unit.network.test_pilot_busy_runtime import _await_record
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixture registration
    _pair_settled,
    devices,
)
from tests.unit.network.test_session_plane import SESSION, _Handle, _listen, _seed


class _HeldHandle(_Handle):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.bind_tasks: set[asyncio.Task[Any]] = set()

    def subscribe_frontend(self, on_update: Any, *, display_window: bool = False) -> Any:
        async def pending() -> Any:
            task = asyncio.current_task()
            assert task is not None
            self.bind_tasks.add(task)
            try:
                await asyncio.Event().wait()
            finally:
                self.bind_tasks.discard(task)

        return pending()

    async def cancel_binds(self) -> None:
        tasks = list(self.bind_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class _RefusingHandle(_HeldHandle):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.welcome_forwarded = threading.Semaphore(0)

    def subscribe_frontend(self, on_update: Any, *, display_window: bool = False) -> Any:
        async def refuse_after_welcome() -> Any:
            task = asyncio.current_task()
            assert task is not None
            self.bind_tasks.add(task)
            try:
                await asyncio.to_thread(self.welcome_forwarded.acquire)
                raise RuntimeError("synthetic owner cannot prepare the canonical view")
            finally:
                self.bind_tasks.discard(task)

        return refuse_after_welcome()

    async def cancel_binds(self) -> None:
        for _ in self.bind_tasks:
            self.welcome_forwarded.release()
        await super().cancel_binds()


class _FaultRuntime(RuntimeServer):
    def __init__(self, handle: _Handle, mode: str) -> None:
        super().__init__(handle, kind="tui")
        self.mode = mode
        self.welcomes = 0
        self.after_welcome = 0

    async def _send_to(self, conn: Any, frame: dict[str, Any]) -> None:
        if conn.sending_welcome:
            await super()._send_to(conn, frame)
            self.welcomes += 1
            if self.mode == "busy":
                # A second, valid runtime projection is positive evidence on
                # THIS pipe. It does not supply or fake the awaited FrontendSync.
                await super()._send_to(conn, frame)
                self.after_welcome += 1
            return
        if self.mode == "half-dead":
            return
        self.after_welcome += 1
        await super()._send_to(conn, frame)


def _cli(root: Path, home: Path, peer: str) -> tuple[int, dict[str, Any], str]:
    tree = Path(__file__).resolve().parents[3]
    # Scale only existing clocks: the same production CLI, real remote facade,
    # framing and classification run unchanged. No 120-second rig is needed.
    program = """import ipaddress,pathlib,runpy,socket,sys
from local_operator.network import cli,projection,addresses
from local_operator.session import attached
root=pathlib.Path(sys.argv[1]);sys.argv=sys.argv[2:]
loaded=(pathlib.Path(m.__file__).resolve() for m in (cli,projection,attached))
assert all(p.is_relative_to(root) for p in loaded)
def guard(event,args):
 if event=='socket.connect' and isinstance(args[1],tuple):
  if not ipaddress.ip_address(args[1][0]).is_loopback:raise RuntimeError('non-loopback refused')
sys.addaudithook(guard)
with socket.socket() as s:
 try:s.connect(('203.0.113.9',1))
 except RuntimeError:pass
 else:raise AssertionError('socket negative control failed')
addresses._default_route_address=lambda:None
cli.PILOT_BIND_TIMEOUT_S=6.0
cli._PILOT_BIND_SLACK_S=1.0
projection.ACK_TIMEOUT_S=0.5
attached.FRONTEND_SYNC_FOREGROUND_S=0.5
runpy.run_module('local_operator.cli',run_name='__main__')
"""
    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(tree),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TERM": "xterm-256color",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            program,
            str(tree),
            "lop",
            "network",
            "sessions",
            "--json",
            "--peer",
            peer,
            "--send",
            SESSION,
            "not delivered by this fixture",
        ],
        env=env,
        cwd=home,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
    assert process.returncode is not None
    # A child that exited is not sufficient if it leaked a descendant group.
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
    rows = [line for line in out.splitlines() if line.startswith("{")]
    assert rows, (process.returncode, out, err)
    return process.returncode, json.loads(rows[-1]), err


@pytest.mark.parametrize("mode", ["late-welcome", "busy", "fast-refusal", "half-dead"])
def test_the_real_pilot_distinguishes_runtime_evidence(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    pair = request.getfixturevalue("devices")
    owner, borrower, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="drive")
    _listen(borrower)
    if mode == "late-welcome":
        route = borrower.route_stream_push

        def hold_forwarded_welcome(link: Any, frame: dict[str, Any]) -> bool:
            payload = frame.get("frame")
            if isinstance(payload, dict) and payload.get("op") in ("welcome", "projection"):
                # Admission already consumed the runtime's real welcome. Drop
                # only the subsequent forwarding hop; otherwise this fixture
                # measures the owner's pre-admission dial, not this defect.
                return borrower._stream_for(link, str(frame.get("stream") or "")) is not None
            return route(link, frame)

        monkeypatch.setattr(borrower, "route_stream_push", hold_forwarded_welcome)
    _seed(owner.root, SESSION)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(owner.root))
    (owner.root / "sessions" / SESSION / ".session.pid").write_text(str(os.getpid()))
    handle = _RefusingHandle(SESSION) if mode == "fast-refusal" else _HeldHandle(SESSION)
    runtime = _FaultRuntime(handle, mode)
    if isinstance(handle, _RefusingHandle):
        from local_operator.network.relay import _Stream

        write = _Stream.write_to_viewer
        forwarded: set[str] = set()
        fence_lock = threading.Lock()

        def fence_welcome(stream: Any, frame: dict[str, Any]) -> bool:
            # The fault is an immediate refusal AFTER the welcome reaches the
            # viewer socket, not the separate admission/close race. No sleeps:
            # flush calls this same writer after marking buffered frames ready.
            ready = stream.ready and not stream.closed
            result = write(stream, frame)
            if (
                ready
                and result
                and not stream.closed
                and frame.get("op") in ("welcome", "projection")
            ):
                with fence_lock:
                    if stream.stream_id not in forwarded:
                        forwarded.add(stream.stream_id)
                        handle.welcome_forwarded.release()
            return result

        monkeypatch.setattr(_Stream, "write_to_viewer", fence_welcome)
    home = tmp_path / "shell-home"
    home.mkdir()
    (borrower.root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    runtime.start()
    try:
        _await_record(owner.root)
        rc, payload, _err = _cli(borrower.root, home, owner.identity.device_id)
        receipt = {
            "case": mode,
            "rc": rc,
            "reply": payload,
            "welcomes": runtime.welcomes,
            "post_welcome_frames": runtime.after_welcome,
            "owner_prompt_calls": len(handle.prompts()),
        }
        (tmp_path / "pilot-receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt))
        assert rc == 1, payload
        assert not handle.prompts(), "a failed bind must never deliver the user's turn"
        if mode == "half-dead":
            assert runtime.welcomes > 0 and runtime.after_welcome == 0
            assert payload["code"] == "peer_unreachable", payload
            assert "doctor" in payload["message"] and "too busy" not in payload["message"]
        else:
            if mode == "late-welcome":
                assert runtime.welcomes > 0, "the owner must have admitted a real runtime pipe"
            assert payload["code"] == "session_unreachable", payload
            assert "doctor" not in payload["message"], payload
            if mode == "fast-refusal":
                assert "synthetic owner cannot prepare the canonical view" in payload["message"]
                assert "too busy" not in payload["message"], payload
                assert "within 1s" not in payload["message"], payload
            else:
                assert "too busy" in payload["message"], payload
                assert "was not delivered" in payload["message"], payload
                if mode == "busy":
                    assert runtime.after_welcome > 0
    finally:
        if isinstance(handle, _HeldHandle) and runtime._loop is not None:
            asyncio.run_coroutine_threadsafe(handle.cancel_binds(), runtime._loop).result(timeout=5)
        runtime.close()
