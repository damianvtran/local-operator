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
