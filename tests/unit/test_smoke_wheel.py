"""Regressions for ``scripts/smoke_wheel.py``'s teardown and health check.

The smoke is executed by path in CI rather than imported as a package module, so
these tests load it with ``runpy.run_path`` and exercise the two process-level
guarantees a payload assertion cannot see:

* teardown kills the WHOLE process group, not just the leader (review R1/Q1);
* a server that dies immediately after a valid ``/health`` payload fails
  (review R2).

Both defects were found in review of PR #2615. The probes use only
``sys.executable`` and the standard library and are skipped on Windows because
``os.killpg`` is POSIX.
"""

from __future__ import annotations

import os
import runpy
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_wheel.py"

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="process-group teardown is POSIX-only (os.killpg)"
)


#: A worker that ignores SIGTERM. It writes ``ready`` only AFTER the handler is
#: installed, so a leader that waits for that file cannot race the disposition
#: change and hand the test a worker a plain SIGTERM would have killed.
_SIGTERM_IMMUNE_CHILD = textwrap.dedent("""
    import pathlib, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text("ready")
    time.sleep(60)
    """)

#: A synthetic ``lop serve``: answers /health with a valid payload, then either
#: exits 17 (the one-shot shape R2 is about) or keeps serving until killed.
#: ``_serve_check`` invokes it as ``<script> serve --port <n>``, so the pid file
#: and mode are baked into the generated text rather than passed as argv.
_FAKE_SERVER_TEMPLATE = """
import http.server, json, os, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
payload = json.dumps(
    {"status": 200, "message": "ok", "result": {"version": "9.9.9", "install_kind": "pip"}}
).encode()
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *args):
        pass
class Server(http.server.HTTPServer):
    allow_reuse_address = True
httpd = Server(("127.0.0.1", port), Handler)
with open("__PID_FILE__", "w") as fh:
    fh.write(str(os.getpid()))
if "__MODE__" == "one-shot":
    httpd.handle_request()
    sys.exit(17)
while True:
    httpd.handle_request()
"""


@pytest.fixture(scope="module")
def smoke() -> dict[str, Any]:
    """The script's globals, without running its CLI entry point."""
    return runpy.run_path(str(SMOKE_SCRIPT))


def _spawn_leader(tmp_path: Path, *, leader_exits: bool) -> tuple[subprocess.Popen[str], int]:
    """Start a session leader plus a SIGTERM-immune worker in its group.

    The worker signals readiness before the leader prints its pid, so each test
    starts from a worker that really would survive a bare SIGTERM. When
    ``leader_exits`` is true the leader returns immediately, leaving the worker
    behind in the original process group — the already-exited-leader shape.
    """
    ready = tmp_path / "worker-ready"
    leader = textwrap.dedent(f"""
        import pathlib, subprocess, sys, time
        child = subprocess.Popen(
            [sys.executable, "-c", {_SIGTERM_IMMUNE_CHILD!r}, {str(ready)!r}]
        )
        deadline = time.monotonic() + 10
        while not pathlib.Path({str(ready)!r}).exists():
            if time.monotonic() >= deadline:
                raise SystemExit("worker never signalled ready")
            time.sleep(0.01)
        print(child.pid, flush=True)
        {"" if leader_exits else "time.sleep(60)"}
        """)
    proc = subprocess.Popen(
        [sys.executable, "-c", leader],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line, "the leader never reported the worker pid"
    return proc, int(line)


def _kill_group(pgid: int) -> None:
    """The probe's own cleanup: exact pgid, never a pattern match."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _assert_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _fake_server(tmp_path: Path, mode: str) -> tuple[Path, Path]:
    pid_file = tmp_path / f"fake-server-{mode}.pid"
    script = tmp_path / f"fake-lop-{mode}"
    body = _FAKE_SERVER_TEMPLATE.replace("__PID_FILE__", str(pid_file)).replace("__MODE__", mode)
    script.write_text(f"#!{sys.executable}\n{body}")
    script.chmod(0o755)
    return script, pid_file


def _run_serve(smoke: dict[str, Any], fake: Path, tmp_path: Path) -> None:
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / "config"),
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
    }
    (tmp_path / "config").mkdir(exist_ok=True)
    smoke["_serve_check"](
        fake,
        "9.9.9",
        env,
        tmp_path,
        timeout=15.0,
        log_path=tmp_path / "serve.log",
    )


def test_reap_kills_a_worker_that_ignores_sigterm(smoke, tmp_path: Path) -> None:
    """R1/Q1: the SIGKILL must not depend on the leader timing out."""
    proc, worker_pid = _spawn_leader(tmp_path, leader_exits=False)
    try:
        smoke["_reap"](proc)
        _assert_gone(worker_pid)
    finally:
        _kill_group(proc.pid)
        if proc.stdout is not None:
            proc.stdout.close()


def test_reap_kills_a_worker_when_the_leader_already_exited(smoke, tmp_path: Path) -> None:
    """R1/Q1: an already-exited leader must not short-circuit teardown."""
    proc, worker_pid = _spawn_leader(tmp_path, leader_exits=True)
    try:
        assert proc.wait(timeout=10) == 0
        smoke["_reap"](proc)
        _assert_gone(worker_pid)
    finally:
        _kill_group(proc.pid)
        if proc.stdout is not None:
            proc.stdout.close()


def test_serve_check_rejects_a_server_that_exits_after_answering(smoke, tmp_path: Path) -> None:
    """R2: a valid payload followed by exit 17 is a failure, not a pass."""
    fake, _ = _fake_server(tmp_path, "one-shot")
    with pytest.raises(smoke["SmokeError"], match="exited with 17"):
        _run_serve(smoke, fake, tmp_path)


def test_serve_check_accepts_a_live_server_and_reaps_it(smoke, tmp_path: Path) -> None:
    """The happy path still passes, and teardown leaves no process behind."""
    fake, pid_file = _fake_server(tmp_path, "persistent")
    _run_serve(smoke, fake, tmp_path)
    _assert_gone(int(pid_file.read_text()))
