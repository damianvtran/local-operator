"""User-level service lifecycle; no root installer or global cloudflared login."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from local_operator.paths import config_dir
from local_operator.tunnels import config

LABEL = "com.local-operator.tunnel"


def service_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    if sys.platform.startswith("linux"):
        return Path.home() / ".config" / "systemd" / "user" / "lop-tunnel.service"
    raise ValueError("Use lop tunnel serve in the foreground on this platform.")


def _run(args: list[str], *, checked: bool = True) -> None:
    result = subprocess.run(args, capture_output=True, timeout=20)
    if checked and result.returncode:
        raise ValueError("Tunnel service action failed; check your user service manager.")


def install() -> None:
    path = service_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        log = config.directory() / "service.log"
        config.private_write(log, "")
        value = {
            "Label": LABEL,
            "ProgramArguments": [sys.executable, "-m", "local_operator.tunnels.service"],
            "EnvironmentVariables": {"LOCAL_OPERATOR_CONFIG_DIR": str(config_dir())},
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 10,
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
        }
        path.write_bytes(plistlib.dumps(value))
        path.chmod(0o600)
        _run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], checked=False)
        _run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)])
    else:
        # Systemd quoting is its own grammar, not shell escaping. Percent is
        # doubled because unit specifiers expand even inside quoted strings.
        def quoted(value: str) -> str:
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'

        text = (
            "[Unit]\nDescription=Radient personal tunnel\nAfter=network-online.target\n"
            "[Service]\nType=simple\n"
            f"ExecStart={quoted(sys.executable)} -m local_operator.tunnels.service\n"
            f"Environment={quoted('LOCAL_OPERATOR_CONFIG_DIR=' + str(config_dir()))}\n"
            "Restart=on-failure\nRestartSec=10\nUMask=0077\n"
            "[Install]\nWantedBy=default.target\n"
        )
        path.write_text(text)
        path.chmod(0o600)
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", "--now", path.name])


def action(name: str) -> None:
    path = service_path()
    if not path.exists():
        raise ValueError(
            "Tunnel service not installed. Run lop tunnel install or lop tunnel serve."
        )
    if sys.platform == "darwin":
        domain = f"gui/{os.getuid()}"
        if name == "stop":
            _run(["launchctl", "bootout", domain + "/" + LABEL], checked=False)
        else:
            _run(["launchctl", "bootstrap", domain, str(path)], checked=False)
            _run(["launchctl", "kickstart", "-k", domain + "/" + LABEL])
    else:
        _run(["systemctl", "--user", name, path.name])


def uninstall() -> None:
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        return  # This platform only supports the foreground connector.
    path = service_path()
    if not path.exists():
        return
    action("stop")
    if sys.platform.startswith("linux"):
        _run(["systemctl", "--user", "disable", path.name])
    path.unlink()
