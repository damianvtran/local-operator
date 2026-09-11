"""``lop wake status`` / ``list`` — is the thing that fires wakes alive?

The command reported ``installed`` from ``plist_path().exists()``, which is an
even weaker test than the install hook's own: a supervisor that had exited
still has its plist on disk, so the one surface an operator would check to
answer "why did my wake not fire" confidently answered "installed" while
nothing was running. It also reported no per-schedule state at all — not when
a wake last fired, not whether it is overdue now, not whether it has gone past
the staleness bound the supervisor silently stops firing at.

These tests never touch launchd: ``supervisor_state`` is patched, which is what
keeps them inside the safety contract in ``test_install.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

from local_operator.wakes.install import SupervisorState
from local_operator.wakes.store import write_entry

NOW_MS = int(time.time() * 1000)


def _args(**kwargs: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "wake_command": "status",
        "json": False,
        "install": False,
        "uninstall": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def _installer_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend this platform has an installer.

    The status RENDERING is platform-independent, but the command asks
    `is_supported()` before probing at all — so on Linux (where there is no
    installer and the honest answer is "wakes fire only while a session is
    open") these fixtures would never be consulted and every assertion below
    would be about the unsupported branch instead. Patching the capability
    keeps the rendering under test on every runner rather than only on macOS.
    """
    monkeypatch.setattr("local_operator.wakes.install.is_supported", lambda: True)


@pytest.fixture
def stopped_supervisor(monkeypatch: pytest.MonkeyPatch, _installer_available: None) -> None:
    """Loaded, exited — the state that produced the permanent misses."""
    monkeypatch.setattr(
        "local_operator.wakes.install.supervisor_state",
        lambda _config: SupervisorState(loaded=True, running=False, detail="not running"),
    )


@pytest.fixture
def running_supervisor(monkeypatch: pytest.MonkeyPatch, _installer_available: None) -> None:
    monkeypatch.setattr(
        "local_operator.wakes.install.supervisor_state",
        lambda _config: SupervisorState(loaded=True, running=True, pid=4242, detail="running"),
    )
    monkeypatch.setattr("local_operator.cli._process_uptime_s", lambda _pid: 3600.0)


def test_a_stopped_supervisor_is_not_reported_as_installed(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE BLIND SPOT, on the surface an operator actually reads.

    ``plist_path().exists()`` is true of a supervisor that exited hours ago.
    Reporting that as "installed" beside an overdue wake is precisely the
    reassurance that kept the misses invisible.
    """
    from local_operator.cli import wake_command

    write_entry(
        tmp_path,
        "statussess01",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch prod", "next_due_at": NOW_MS - 60_000}],
    )

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    assert "loaded but NOT running" in out, out
    assert "lop wake install" in out, "the actionable repair must be named"


def test_a_running_supervisor_reports_its_pid_and_uptime(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """The positive half: "running" must be distinguishable from "present"."""
    from local_operator.cli import wake_command

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    assert "running (pid 4242)" in out, out
    assert "up 1h" in out, out


def test_status_reports_overdue_and_stale_counts(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wake past ``STALE_AFTER_S`` is skipped by the supervisor FOREVER.

    That is deliberate — the session's own catch-up owns it — but it must be
    visible, because it is indistinguishable from a working wake otherwise.
    """
    from local_operator.cli import wake_command

    write_entry(
        tmp_path,
        "statussess02",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "late watch", "next_due_at": NOW_MS - 600_000}],
    )
    write_entry(
        tmp_path,
        "statussess03",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "forgotten", "next_due_at": NOW_MS - 9 * 24 * 3600_000}],
    )

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    assert "overdue:     2" in out, out
    assert "stale:       1 past 7d" in out, out


def test_the_json_form_carries_the_machine_readable_state(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """A monitoring caller must not have to scrape the human rendering."""
    from local_operator.cli import wake_command

    write_entry(
        tmp_path,
        "statussess04",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch prod", "next_due_at": NOW_MS - 120_000}],
    )

    assert wake_command(_args(json=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["supervisor"]["running"] is False
    assert payload["supervisor"]["loaded"] is True
    # The legacy key keeps its name but now means RUNNING, which is the only
    # reading of "installed" that answers "will my wake fire".
    assert payload["installed"] is False
    assert payload["overdue"] == 1
    assert payload["max_overdue_s"] >= 120


def test_list_marks_overdue_and_stale_schedules(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import wake_command

    write_entry(
        tmp_path,
        "statussess05",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "late watch", "next_due_at": NOW_MS - 600_000}],
    )
    write_entry(
        tmp_path,
        "statussess06",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "forgotten", "next_due_at": NOW_MS - 9 * 24 * 3600_000}],
    )

    assert wake_command(_args(wake_command="list", json=True)) == 0
    rows = {row["session_id"]: row for row in json.loads(capsys.readouterr().out)}

    assert rows["statussess05"]["overdue"] is True
    assert rows["statussess05"]["stale"] is False
    assert rows["statussess06"]["stale"] is True

    assert wake_command(_args(wake_command="list", json=False)) == 0
    out = capsys.readouterr().out
    assert "(OVERDUE)" in out
    # Stale is the STRONGER statement (the supervisor has stopped trying), so
    # it replaces the overdue mark rather than doubling up on one line.
    assert "(STALE" in out
    assert out.count("(OVERDUE)") == 1


def test_the_install_subcommand_reaches_the_repair_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _installer_available: None,
) -> None:
    """THE ROLLOUT PATH.

    Repair-on-demand otherwise runs only on a wake PERSIST, so a machine whose
    supervisor is stale cannot be fixed without some session happening to
    schedule a wake — exactly the wrong dependency after an upgrade, when the
    running supervisor is still executing the old code.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes.install import InstallOutcome

    called: list[Path] = []

    def fake_install(config_dir: Path) -> InstallOutcome:
        called.append(config_dir)
        return InstallOutcome(installed=True, reason="restarted a stopped supervisor")

    monkeypatch.setattr("local_operator.wakes.install.ensure_supervisor_installed", fake_install)
    monkeypatch.setattr(
        "local_operator.wakes.install.supervisor_state",
        lambda _config: SupervisorState(loaded=True, running=True, pid=7, detail="running"),
    )

    assert wake_command(_args(wake_command="install")) == 0

    assert called, "the install subcommand did not reach the repair hook"
    assert "restarted a stopped supervisor" in capsys.readouterr().out


def test_an_unsupervisable_store_never_reports_another_stores_supervisor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE ISOLATED-RUN CASE, end to end through the command.

    No patching of the probe: this drives the real `supervisor_state` against
    a config dir outside the real home — the shape every isolated run has —
    and the command must report a state of its own rather than the operator's
    live LaunchAgent. Before this, the same invocation printed
    `running (pid 47545)`, which is a different store's process.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes.install import is_supported

    if not is_supported():
        pytest.skip("launchd scoping is only meaningful on darwin with launchctl")

    write_entry(
        tmp_path,
        "statussess07",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch prod", "next_due_at": NOW_MS - 60_000}],
    )

    assert wake_command(_args()) == 0
    out = capsys.readouterr().out

    assert "cannot be verified for this store" in out, out
    assert "running" not in out.split("scheduled:")[0].replace("not running", ""), out
    assert "pid" not in out, "another store's pid was printed"

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["supervisor"]["verifiable"] is False
    assert payload["supervisor"]["running"] is False
    assert payload["supervisor"]["pid"] is None
    assert payload["installed"] is False
