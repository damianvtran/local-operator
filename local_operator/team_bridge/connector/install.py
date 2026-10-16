"""Service-unit rendering for the connector (H4a) — RENDER ONLY.

Mechanics are the wake supervisor's family (`wakes/install.py`), under a NEW
dedicated label so the two daemons never share a plist/unit identity:

* macOS: a LaunchAgent plist with the supervisor's lifecycle contract
  (``RunAtLoad``; the connector is ALWAYS-ON, so ``KeepAlive: true`` — unlike
  the self-retiring wake supervisor, an empty poll window is its normal state
  and not an exit condition).
* Linux: a ``systemd --user`` unit with ``Restart=always``.
* Windows: a Task Scheduler XML whose trigger runs at logon and repeats.

WHAT THIS MODULE MUST NEVER DO: install, bootstrap, `launchctl load`, touch
``~/Library/LaunchAgents``, ``systemctl`` or ``schtasks``. :func:`write_plist`
writes to a caller-provided directory ONLY — production wiring and the
operator's machine are the CLI's separate act, deliberately not in this pass.
Tests render every artefact to an isolated temp dir and assert delivery
targets, exactly like the wake supervisor's own tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The connector's own identity — a NEW label per the manager decision; the
#: wake supervisor's ``com.local-operator.wakes`` is never reused or altered.
LABEL = "com.local-operator.bridge-connector"
TASK_NAME = "Local Operator bridge connector"
SYSTEMD_UNIT = "local-operator-bridge-connector.service"


def _argv(config_dir: Path) -> list[str]:
    """What the supervisor runs: this interpreter, this package, this dir."""

    return [
        sys.executable,
        "-m",
        "local_operator.team_bridge.connector",
        "--config-dir",
        str(config_dir),
    ]


def plist_path() -> Path:
    """The real install target (rendered/tests never write there)."""

    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def render_plist(config_dir: Path) -> dict[str, object]:
    """The macOS job plan — pure, so tests assert exactly what launchd reads."""

    return {
        "Label": LABEL,
        "ProgramArguments": _argv(config_dir),
        "RunAtLoad": True,
        # ALWAYS-ON: restarted regardless of exit status. The wake
        # supervisor's `SuccessfulExit: False` is its self-retirement
        # contract; the connector has no empty-index exit, so it keeps the
        # plain dictionary form.
        "KeepAlive": True,
        "EnvironmentVariables": {"LOCAL_OPERATOR_CONNECTOR": "1"},
    }


def render_systemd(config_dir: Path) -> str:
    """The Linux user unit — the same always-on contract, in systemd's words."""

    argv = " ".join(_argv(config_dir))
    return (
        "[Unit]\n"
        f"Description={TASK_NAME}\n"
        "\n"
        "[Service]\n"
        f"ExecStart={argv}\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def render_task_xml(config_dir: Path) -> str:
    """The Windows task: run at logon, restarted by Task Scheduler."""

    argv = " ".join(_argv(config_dir))
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        f"  <Description>{TASK_NAME}</Description>\n"
        "  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>\n"
        "  <Settings><RestartOnFailure><Interval>PT1M</Interval><Count>999</Count>"
        "</RestartOnFailure></Settings>\n"
        "  <Actions><Exec><Command>{}</Command></Exec></Actions>\n"
        "</Task>\n"
    ).format(argv)


def write_plist(config_dir: Path, target_dir: Path) -> Path:
    """Render to ``target_dir`` ONLY — never the operator's LaunchAgents."""

    import plistlib

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{LABEL}.plist"
    path.write_bytes(plistlib.dumps(render_plist(config_dir)))
    return path
