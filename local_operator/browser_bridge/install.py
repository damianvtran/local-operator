"""Install and supervise the browser bridge on macOS or Linux.

The unit re-enters this interpreter's package rather than pinning a checkout;
updating the installed Local Operator package therefore updates the daemon on
its next restart without rewriting supervisor configuration.
"""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.daemon import (
    DEFAULT_PORT,
    pairing_status,
    reset_pairing,
)
from local_operator.paths import log_dir

LABEL = "com.local-operator.browser"
SYSTEMD_UNIT = "local-operator-browser.service"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def systemd_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def log_path() -> Path:
    return log_dir() / "browser-bridge.log"


def render_plist(port: int = DEFAULT_PORT) -> dict[str, object]:
    return {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "local_operator.browser_bridge.daemon",
            "--port",
            str(port),
        ],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path()),
        "StandardErrorPath": str(log_path()),
        "ProcessType": "Interactive",
    }


def render_systemd(port: int = DEFAULT_PORT) -> str:
    command = f"{sys.executable} -m local_operator.browser_bridge.daemon --port {port}"
    return f"""[Unit]
Description=Local Operator browser bridge

[Service]
ExecStart={command}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


def health(port: int = DEFAULT_PORT, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            value = json.loads(response.read().decode())
            return value if isinstance(value, dict) else None
    except Exception:  # noqa: BLE001 - a probe answers absent rather than raising
        return None


def stale_heartbeat_age(root: Path | None = None) -> float | None:
    """Heartbeat age when the discovery file is stale but the daemon is alive.

    ``None`` whenever there is nothing worth reporting (no daemon, or a fresh
    heartbeat). This is the diagnostic that makes the incident's central
    contradiction visible: ``status`` reads the live socket, every session
    reads this file, and when the file stops being refreshed the two disagree
    silently — status says "connected" while every session falls back to cmux.
    """
    try:
        status_value, current = state_store.liveness(root)
    except Exception:  # noqa: BLE001 - a diagnostic never raises
        return None
    if status_value is not state_store.Liveness.STALE or current is None:
        return None
    return state_store.heartbeat_age(current)


def repair(port: int | None = None, root: Path | None = None) -> dict[str, Any]:
    """Reconcile the daemon's advertised state against reality.

    The user's remedy in the incident was "there isn't one": the bridge
    advertised a tab that no longer existed and a heartbeat that had stopped,
    and the only lever anyone had was killing a healthy daemon. This asks the
    daemon to prune what reality does not back and to republish a fresh
    heartbeat, then reports what it cleaned.

    Safe while sessions are live: the daemon's ``/repair`` closes no tab,
    cancels no in-flight command, and touches no pairing.
    """
    current = state_store.read(root)
    resolved_port = port or (current.port if current else DEFAULT_PORT)
    steps: list[str] = []
    if current is None:
        return {
            "ok": False,
            "steps": steps,
            "error": "no bridge daemon state found; 'lop browser install' starts it.",
        }
    age = state_store.heartbeat_age(current)
    steps.append(f"daemon pid {current.pid} on port {resolved_port}, heartbeat {age:.0f}s old")
    if not state_store.pid_alive(current.pid):
        return {
            "ok": False,
            "steps": steps,
            "error": (
                f"daemon pid {current.pid} is not running; run 'lop browser restart' "
                "to start a fresh one."
            ),
        }
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{resolved_port}/repair", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload: Any = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            # The daemon answers, but predates /repair. Restarting is the right
            # advice and the reason matters: this is exactly the long-running
            # daemon whose heartbeat died, so it is the one most likely to be
            # running older code than the CLI asking it to repair itself.
            return {
                "ok": False,
                "steps": steps,
                "error": (
                    f"the daemon on port {resolved_port} is running a build with no "
                    "/repair endpoint (it predates this fix). Run 'lop browser restart' "
                    "to load the current build."
                ),
            }
        return {
            "ok": False,
            "steps": steps,
            "error": f"daemon returned HTTP {error.code} on port {resolved_port}.",
        }
    except Exception as error:  # noqa: BLE001 - report, do not raise, at a CLI edge
        return {
            "ok": False,
            "steps": steps,
            "error": (
                f"daemon is not answering on port {resolved_port} ({error}); "
                "run 'lop browser restart'."
            ),
        }
    cleared = payload.get("cleared_tabs") or []
    if cleared:
        for url in cleared:
            steps.append(f"cleared phantom driven tab: {url}")
    else:
        steps.append("no phantom driven tabs to clear")
    steps.append(f"driven tabs now: {payload.get('driven_tabs', 0)}")
    if payload.get("heartbeat_republished"):
        fresh = state_store.read(root)
        refreshed = state_store.heartbeat_age(fresh) if fresh else None
        steps.append(
            "heartbeat republished"
            + (f" (now {refreshed:.0f}s old)" if refreshed is not None else "")
        )
    else:
        steps.append("heartbeat could NOT be republished — check disk space and permissions")
    return {"ok": True, "steps": steps, "error": ""}


def install(port: int = DEFAULT_PORT, *, dry_run: bool = False) -> dict[str, object]:
    steps: list[str] = []
    log_path().parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin" and shutil.which("launchctl"):
        plist_path().parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            plist_path().write_bytes(plistlib.dumps(render_plist(port)))
        steps.append(f"wrote {plist_path()}")
        if not dry_run:
            _launchctl("bootout", _domain(), str(plist_path()))
            loaded = _launchctl("bootstrap", _domain(), str(plist_path()))
            if loaded.returncode:
                return {"ok": False, "steps": steps, "error": loaded.stderr.strip()[:300]}
            steps.append("loaded the LaunchAgent")
    elif sys.platform.startswith("linux") and shutil.which("systemctl"):
        systemd_path().parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            systemd_path().write_text(render_systemd(port), encoding="utf-8")
        steps.append(f"wrote {systemd_path()}")
        if not dry_run:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            loaded = subprocess.run(
                ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT],
                capture_output=True,
                text=True,
            )
            if loaded.returncode:
                return {"ok": False, "steps": steps, "error": loaded.stderr.strip()[:300]}
            steps.append("enabled the systemd user service")
    else:
        return {
            "ok": False,
            "steps": steps,
            "error": "no supported user supervisor; run `lop browser serve` in the foreground",
        }
    if dry_run:
        return {"ok": True, "steps": [*steps, "dry run: skipped load and verification"]}
    deadline = time.time() + 20
    while time.time() < deadline:
        if health(port) is not None:
            return {"ok": True, "steps": [*steps, "health check passed"]}
        time.sleep(0.5)
    return {
        "ok": False,
        "steps": steps,
        "error": f"daemon did not become healthy; see {log_path()}",
    }


def uninstall(*, purge: bool = False, dry_run: bool = False) -> dict[str, object]:
    steps: list[str] = []
    if sys.platform == "darwin":
        if not dry_run:
            _launchctl("bootout", _domain(), str(plist_path()))
            plist_path().unlink(missing_ok=True)
        steps.append("removed the LaunchAgent")
    elif sys.platform.startswith("linux"):
        if not dry_run:
            subprocess.run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT], check=False)
            systemd_path().unlink(missing_ok=True)
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        steps.append("removed the systemd user service")
    if purge:
        if not dry_run:
            reset_pairing()
            state_store.remove()
        steps.append("deleted pairing and bridge state")
    return {"ok": True, "steps": steps}


def service_action(action: str) -> dict[str, object]:
    if sys.platform == "darwin":
        if action in ("start", "restart") and plist_path().exists():
            if _launchctl("print", f"{_domain()}/{LABEL}").returncode:
                loaded = _launchctl("bootstrap", _domain(), str(plist_path()))
                if loaded.returncode:
                    return {"ok": False, "error": loaded.stderr.strip()[:300]}
        args = {
            "start": ("kickstart", f"{_domain()}/{LABEL}"),
            "stop": ("kill", "SIGTERM", f"{_domain()}/{LABEL}"),
            "restart": ("kickstart", "-k", f"{_domain()}/{LABEL}"),
        }[action]
        result = _launchctl(*args)
    else:
        result = subprocess.run(
            ["systemctl", "--user", action, SYSTEMD_UNIT], capture_output=True, text=True
        )
    return {"ok": result.returncode == 0, "error": result.stderr.strip()[:300]}


def status(port: int | None = None) -> dict[str, object]:
    current = state_store.read()
    resolved_port = port or (current.port if current else DEFAULT_PORT)
    probe = health(resolved_port)
    pairing = pairing_status()
    return {
        "installed": plist_path().exists() if sys.platform == "darwin" else systemd_path().exists(),
        "healthy": probe is not None,
        "health": probe,
        "port": resolved_port,
        "state": current.model_dump(mode="json", exclude={"session_key"}) if current else None,
        "paired": pairing["paired"],
        "extension_id": pairing["extension_id"],
        "pending_code": pairing["pending_code"],
        "pending_expires_at": pairing["pending_expires_at"],
        "log": str(log_path()),
    }
