"""Install, supervise and introspect the mobile daemon (macOS launchd).

One LaunchAgent re-runs THIS interpreter's package as ``python -m
local_operator.mobile.service`` — re-entering the installed code rather than
a hardcoded binary path means an upgrade (``lop-update``) changes what the
agent runs with no reinstall, and ``restart`` picks it up. That is the omp
mobile lesson applied to a Python entry point.

The LaunchAgent label is fixed (``com.local-operator.mobile``): the label owns
the port, so a second daemon cannot split-brain the control plane — it fails
to bind and exits loudly.

Linux/Windows: the daemon itself is portable and foreground-runnable; only
the supervisor is macOS-specific, and the CLI says so rather than writing a
broken unit file.
"""

from __future__ import annotations

import json
import plistlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from local_operator.mobile.auth import generate_password, load_password, store_password
from local_operator.mobile.daemon import DEFAULT_PORT
from local_operator.paths import log_dir

LABEL = "com.local-operator.mobile"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def log_path() -> Path:
    return log_dir() / "mobile.log"


def render_plist(port: int = DEFAULT_PORT) -> dict[str, object]:
    """The whole supervised-unit plan in one pure function — every consumer
    (install, status, tests) reads the same rendering."""
    return {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "local_operator.mobile.service",
            "--port",
            str(port),
        ],
        "RunAtLoad": True,
        # Restart on crash, throttled by launchd's own 10s floor; an
        # exit-code-2 (no password) stays down because KeepAlive keys on
        # successful exit only being false — crashes restart, refusals don't
        # flap.
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path()),
        "StandardErrorPath": str(log_path()),
        # SSE-holding daemons must not be App Nap'd into suspending timers.
        "ProcessType": "Interactive",
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


def _domain() -> str:
    import os

    return f"gui/{os.getuid()}"


def is_supported() -> bool:
    return sys.platform == "darwin" and shutil.which("launchctl") is not None


def health(port: int = DEFAULT_PORT, timeout: float = 3.0) -> dict[str, object] | None:
    """Probe the daemon's unauthenticated liveness endpoint."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=timeout
        ) as response:
            return json.loads(response.read().decode())  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001 — health is a probe, absence is the answer
        return None


def gate_closed(port: int = DEFAULT_PORT, timeout: float = 3.0) -> bool:
    """Assert the AUTH gate, not mere liveness: a daemon that serves the API
    without a cookie is a boundary failure, however healthy its process."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/sessions")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 401
    except urllib.error.HTTPError as exc:
        return exc.code == 401
    except Exception:  # noqa: BLE001
        return False


def install(port: int = DEFAULT_PORT, *, dry_run: bool = False) -> dict[str, object]:
    """Idempotent one-shot: password (kept if present), plist, load, verify."""
    steps: list[str] = []
    if not is_supported():
        return {
            "ok": False,
            "steps": [],
            "error": (
                "install needs macOS launchd; " "run `lop mobile serve` in the foreground elsewhere"
            ),
        }

    password = load_password()
    if password is None:
        password = generate_password()
        if not dry_run:
            store_password(password)
        steps.append("generated a new portal password (Keychain: lop-mobile)")
    else:
        steps.append("kept the existing portal password")

    plist_path().parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        plist_path().write_bytes(plistlib.dumps(render_plist(port)))
    steps.append(f"wrote {plist_path()}")

    if not dry_run:
        _launchctl("bootout", _domain(), str(plist_path()))  # stale copy: fine if absent
        result = _launchctl("bootstrap", _domain(), str(plist_path()))
        if result.returncode != 0:
            return {"ok": False, "steps": steps, "error": result.stderr.strip()[:300]}
        steps.append("loaded the LaunchAgent")

        deadline = time.time() + 15
        while time.time() < deadline:
            if health(port) and gate_closed(port):
                steps.append("health check passed and the auth gate is closed")
                # Never return the password: a `--json` dump or an agent
                # capturing stdout would put it in the transcript.
                return {"ok": True, "steps": steps}
            time.sleep(0.5)
        return {
            "ok": False,
            "steps": steps,
            "error": f"daemon did not come up healthy; see {log_path()}",
        }
    steps.append("dry run: skipped load and verification")
    return {"ok": True, "steps": steps}


def uninstall(*, purge: bool = False, dry_run: bool = False) -> dict[str, object]:
    steps: list[str] = []
    if not dry_run:
        _launchctl("bootout", _domain(), str(plist_path()))
        plist_path().unlink(missing_ok=True)
    steps.append("removed the LaunchAgent")
    if purge:
        if not dry_run:
            from local_operator.mobile.auth import delete_password

            delete_password()
        steps.append("deleted the Keychain password")
    return {"ok": True, "steps": steps}


def service_action(action: str) -> dict[str, object]:
    """start|stop|restart via launchctl kickstart/kill."""
    if action == "start":
        result = _launchctl("kickstart", f"{_domain()}/{LABEL}")
    elif action == "stop":
        result = _launchctl("kill", "SIGTERM", f"{_domain()}/{LABEL}")
    else:  # restart
        result = _launchctl("kickstart", "-k", f"{_domain()}/{LABEL}")
    ok = result.returncode == 0
    return {"ok": ok, "error": "" if ok else result.stderr.strip()[:300]}


def status(port: int = DEFAULT_PORT) -> dict[str, object]:
    """What a human (or `lop mobile status`) needs: install state, live
    health, gate state, registered sessions, log path."""
    from local_operator.mobile import registry

    probe = health(port)
    records = registry.scan()
    return {
        "installed": plist_path().exists(),
        "password_set": load_password() is not None,
        "healthy": probe is not None,
        "gate_closed": gate_closed(port),
        "health": probe,
        "port": port,
        "log": str(log_path()),
        "sessions": [
            {
                "pid": record.pid,
                "kind": record.kind,
                "session_id": record.session_id,
                "conversation_name": record.conversation_name,
                "model_label": record.model_label,
                "state": state,
            }
            for record, state in records
        ],
    }
