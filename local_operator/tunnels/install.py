"""User-level service lifecycle; no root installer or global cloudflared login."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from local_operator import launchd, procname
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


def render_plist(config_base: Path | None = None) -> dict[str, object]:
    """The whole supervised-unit plan, for the store ``config_base`` names.

    ``config_base`` defaults to this process's config dir, which is what
    ``install`` wants. The LaunchAgent repair passes the store recorded in the
    plist it is replacing instead — see
    :func:`local_operator.tunnels.config.directory`.

    ``launchd_job`` rather than ``launchd_program``: ``Program`` is the branded
    image and ``ProgramArguments[0]`` is this daemon's role label, which is what
    stops the tunnel reading as the same bare ``Local Operator`` row as the
    other three daemons (and, before branding, as ``python3``). macOS names a
    background item by the basename of ``ProgramArguments[0]``, so the old
    ``sys.executable`` here is what made installing notify that 'python3 is
    running in the background'. The trade-off that shape accepts is recorded in
    ``procname.launchd_job``; with no link to plant this is the same plist as
    before. On a machine with the generation layout ``Program`` is the stable
    shim instead (see ``procname.supervised_image``) and the role label is
    unchanged.
    """
    base = config_base if config_base is not None else config_dir()
    return {
        "Label": LABEL,
        **procname.launchd_job(
            "local_operator.tunnels.service",
            label=procname.branded_argv0(procname.LABEL_TUNNEL),
        ),
        "EnvironmentVariables": {"LOCAL_OPERATOR_CONFIG_DIR": str(base)},
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "StandardOutPath": str(config.directory(base) / "service.log"),
        "StandardErrorPath": str(config.directory(base) / "service.log"),
    }


def refresh_plist_if_stale() -> launchd.PlistRefresh:
    """Rewrite this daemon's LaunchAgent when an older build wrote it.

    The tunnel had no repair path at all: ``lop-update`` bounced the mobile
    daemon and nothing else, so a plist written before branding kept running a
    bare ``python3.14 -m local_operator.tunnels.service`` forever — measured on
    the operator's machine as ``com.local-operator.tunnel.plist`` and pid 92821.

    Two guards, both inherited from ``wakes.install`` rather than reinvented:
    the plist must be the one the real passwd home produces (``launchctl``
    always addresses the real user's session, whatever ``HOME`` says), and the
    store it records must live under the real home (otherwise this would point
    the operator's real LaunchAgent at a sandbox store that disappears).

    Never raises; a repair that cannot run leaves the upgrade that called it
    exactly as it was.
    """
    name = "tunnel"
    try:
        if sys.platform != "darwin":
            # The systemd user unit has no plist to repair; it re-reads its unit
            # file on every start, so it has never had this failure mode.
            return launchd.PlistRefresh(name=name, kind="unsupported")
        path = service_path()
        if not launchd.is_own_plist(path, LABEL):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        current = launchd.load(path)
        base = launchd.config_dir_from_plist(current) or config_dir()
        if not launchd.config_lives_in_real_home(base):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        outcome = launchd.rewrite_if_stale(name=name, path=path, rendered=render_plist(base))
        if outcome.kind != "repaired":
            return outcome
        domain = f"gui/{os.getuid()}"
        # bootout + bootstrap, NOT kickstart -k: a kickstart restarts the job
        # from launchd's in-memory definition, so it would keep running the old
        # argv after this rewrite. Measured; see :mod:`local_operator.launchd`.
        _run(["launchctl", "bootout", f"{domain}/{LABEL}"], checked=False)
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["launchctl", "bootstrap", domain, str(path)], capture_output=True, timeout=20
        )
        if result.returncode:
            # Names the recovery, because the job is DOWN at this point: see
            # `launchd.reload_failure`.
            detail = result.stderr.decode(errors="replace").strip()[:200]
            return launchd.reload_failure(
                name, path, "lop tunnel install", detail or str(result.returncode)
            )
        return outcome
    except Exception as exc:  # noqa: BLE001 — a repair must never fail an upgrade
        return launchd.PlistRefresh(name=name, kind="failed", detail=str(exc))


def install() -> None:
    path = service_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        log = config.directory() / "service.log"
        config.private_write(log, "")
        value = render_plist()
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
            # The stable shim when this machine has the generation layout, else
            # this process's interpreter: a unit restarts, and a path inside a
            # tree a flip or a prune replaced is a daemon that dies at load.
            f"ExecStart={quoted(str(procname.supervised_image() or sys.executable))} "
            "-m local_operator.tunnels.service\n"
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
