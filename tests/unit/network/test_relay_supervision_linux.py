"""The relay's Linux supervision arm — systemd ``--user`` (remote onboarding slice (b)).

WHY THIS FILE EXISTS SEPARATELY. ``test_purge`` pins the LAUNCHD half of the
supervision surface plus the no-supervisor refusal; these cells pin the half
that did not exist until slice (b): the unit render, the write/enable path (dry
run and file half only — a test never reaches a user manager), and the
missing-unit start that installs. ``sys.platform`` and ``shutil.which`` are
patched to the Linux answer rather than stubbing ``is_supported`` itself, so
the real expression runs (the same discipline ``test_purge`` documents).

NO CELL HERE MAY REACH ``systemctl``: every one that could is either a dry run
or has ``systemctl_user`` replaced by something that raises when called.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import supervisors
from local_operator.network import relay

# tmp_path fixtures -------------------------------------------------------


@pytest.fixture()
def linux_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A HOME the unit path resolves under, claiming linux + systemctl."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(relay.sys, "platform", "linux")
    monkeypatch.setattr(
        relay.shutil,
        "which",
        lambda name: "/usr/bin/systemctl" if name == "systemctl" else f"/usr/bin/{name}",
    )
    return home


def _unit(home: Path) -> Path:
    return home / ".config" / "systemd" / "user" / relay.SYSTEMD_UNIT


def _no_systemctl(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: str, **kwargs: Any) -> Any:
        raise AssertionError(f"systemctl was invoked: {args}")

    monkeypatch.setattr(supervisors, "systemctl_user", _boom)


# tmp_path render ---------------------------------------------------------


def test_render_systemd_has_the_contract_every_unit_shares(tmp_path: Path) -> None:
    text = relay.render_systemd(4097)
    assert text.startswith("[Unit]\n")
    assert "Description=Local Operator network relay" in text
    assert "-m local_operator.network.relay --port 4097" in text
    # The config-dir pin is present and QUOTED (an unquoted assignment truncates
    # at the first space — the mobile daemon's own measured lesson).
    assert 'Environment="LOCAL_OPERATOR_CONFIG_DIR=' in text
    assert "Restart=on-failure" in text
    assert "WantedBy=default.target" in text


def test_install_dry_run_names_the_unit_and_touches_nothing(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run REPORTS what it would write; it writes nothing and reaches no manager.

    The same contract ``_install_launchd``'s dry run has: the receipt names the
    unit path so an operator can see where it would land, and the file itself is
    the non-dry-run half — which the redirected-home cell below exercises (the
    file half is testable; the enable half is what refuses there).
    """
    _no_systemctl(monkeypatch)

    result = relay.install(4097, dry_run=True)

    assert result["ok"] is True, result
    assert not _unit(linux_host).exists(), "a dry run must not write the unit"
    assert any(str(_unit(linux_host)) in step for step in result["steps"])
    assert any("dry run" in step for step in result["steps"])


def test_a_redirected_home_never_reaches_the_user_manager(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The addressability guard, asserted on the REAL expression.

    The test's own HOME is redirected and therefore not the passwd home, so
    ``_install_systemd`` must refuse AFTER writing the file (the file half is
    testable) and BEFORE any enable — with ``systemctl_user`` armed to fail.
    """
    _no_systemctl(monkeypatch)

    result = relay.install(4097)

    assert result["ok"] is False
    assert result["reason"] == "isolated_home"
    assert _unit(linux_host).exists(), "the file half is still written"


# -- the honesty wrapper (drill finding, 2026-10-04) -------------------------
#
# The drill's exact shape, encoded: `lop network restart` returned ok while the
# listening PID did not change, because a hand-started relay held the port and
# the readiness check accepted the OLD process's answer as the new one's. These
# cells pin the contract the fix states: a hand-started relay of ours is
# ADOPTED (stopped so the supervised one can bind); the supervisor's own pid
# must be what answers afterwards, and a restart must have replaced it; a
# holder that is not ours is refused with pid and cmdline named. The probes are
# stubbed so no cell here can reach systemctl, launchctl, or a real signal.


class _Script:
    """One scripted value per call, the last repeating (a real probe's cadence)."""

    def __init__(self, *values: Any) -> None:
        self._values = list(values)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0] if self._values else None


class _RelayRecordScript:
    """``store.scan_own_relay`` as a script: records whose pid changes per call."""

    def __init__(self, *pids: int | None) -> None:
        self._pids = list(pids)

    def __call__(self, root: Any = None) -> tuple[Any, str]:
        pid = self._pids.pop(0) if len(self._pids) > 1 else (self._pids[0] if self._pids else None)
        if pid is None:
            return (None, "")
        return (SimpleNamespace(pid=pid), "live")


def _pin_supervision(
    monkeypatch: pytest.MonkeyPatch,
    *,
    managed: Any = None,
    holder_pids: tuple[int, ...] = (),
    record: Any = None,
    answering: bool = True,
    cmdline: Any = None,
    stop_outcome: str = "stopped",
    stop_stub: bool = True,
    stop_calls: list[dict[str, Any]] | None = None,
) -> None:
    """Pin every probe the honesty wrapper reads, so no cell reaches a manager.

    ``stop_stub=False`` leaves the REAL ``_stop_manual_holder`` in place — the
    cells about identity (pid reuse, a changing command line) must exercise it,
    not a stub, or they would pin their own assumption.
    """
    monkeypatch.setattr(relay, "SERVICE_VERIFY_WINDOW_S", 0.0)
    monkeypatch.setattr(relay, "_port_holder_pids", lambda port: list(holder_pids))
    monkeypatch.setattr(
        relay,
        "_pid_cmdline",
        cmdline or (lambda pid: f"/usr/bin/lop network serve --pid {pid}"),
    )
    monkeypatch.setattr(
        relay, "_managed_service_pid", managed if callable(managed) else _Script(managed)
    )
    monkeypatch.setattr(relay, "health", lambda *a, **k: ({"ok": True} if answering else None))
    monkeypatch.setattr(
        relay.store,
        "scan_own_relay",
        record if callable(record) else _RelayRecordScript(record),
    )
    if not stop_stub:
        return

    def _fake_stop(holder: dict[str, Any], timeout: float = 8.0) -> str:
        if stop_calls is not None:
            stop_calls.append(dict(holder))
        return stop_outcome

    monkeypatch.setattr(relay, "_stop_manual_holder", _fake_stop)


def test_service_action_start_installs_a_missing_unit(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``join`` never installed a unit, so start installs one — via ``_install_systemd``.

    The install is faked, so the VERIFICATION is what this cell really drives:
    the supervisor must report its own running process (the wrapper refuses to
    forward the arm's ``ok`` otherwise).
    """
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    calls: list[int] = []

    def _fake_install(port: int, *, dry_run: bool = False) -> dict[str, Any]:
        calls.append(port)
        return {"ok": True, "steps": ["installed"]}

    monkeypatch.setattr(relay, "_install_systemd", _fake_install)
    _no_systemctl(monkeypatch)
    _pin_supervision(
        monkeypatch, managed=_Script(None, 4242), record=_RelayRecordScript(None, 4242)
    )

    result = relay.service_action("start")

    assert result["ok"] is True, result
    assert calls, "_install_systemd must be what starts a relay that has no unit"


def test_service_action_drives_systemctl_when_a_unit_is_there(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")
    seen: list[tuple[str, ...]] = []

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def _record(*args: str, **kwargs: Any) -> Any:
        seen.append(tuple(args))
        return Done()

    monkeypatch.setattr(supervisors, "systemctl_user", _record)
    # The manager's process is REPLACED (1000 → 2000) and its record follows.
    _pin_supervision(
        monkeypatch,
        managed=_Script(1000, 2000),
        record=_RelayRecordScript(1000, 2000),
    )

    assert relay.service_action("restart")["ok"] is True
    assert seen == [("restart", relay.SYSTEMD_UNIT)]


def test_a_restart_that_replaces_nothing_is_not_ok(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE DRILL'S EXACT SHAPE: systemctl returns 0, the same pid keeps
    serving, and the verb used to answer ``ok: true``. It must not."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(supervisors, "systemctl_user", lambda *a, **k: Done())
    # The SAME pid before and after: the process it manages did not restart.
    _pin_supervision(monkeypatch, managed=333612, record=333612, holder_pids=(333612,))

    result = relay.service_action("restart")

    assert result["ok"] is False, result
    assert result["reason"] == "not_restarted"
    assert "same process" in result["error"]


def test_a_hand_started_relay_is_adopted_before_the_service_binds(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node's shape: a hand-started ``lop network serve`` holds :4097, no
    unit exists. Start/restart STOPS it (the sanctioned state change — the flow
    owns it, not a hand-kill) and then installs; the result names the replacement."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    stops: list[dict[str, Any]] = []

    def _fake_install(port: int, *, dry_run: bool = False) -> dict[str, Any]:
        return {"ok": True, "steps": ["installed"]}

    monkeypatch.setattr(relay, "_install_systemd", _fake_install)
    _no_systemctl(monkeypatch)
    _pin_supervision(
        monkeypatch,
        managed=_Script(None, 9001),
        holder_pids=(333612,),
        record=_RelayRecordScript(333612, 9001),
        stop_calls=stops,
    )

    result = relay.service_action("restart")

    assert result["ok"] is True, result
    assert [h["pid"] for h in stops] == [333612], stops
    assert any("hand-started relay (pid 333612)" in step for step in result.get("steps", []))


def test_a_hand_started_relay_that_will_not_stop_refuses_with_its_pid(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    _no_systemctl(monkeypatch)
    _pin_supervision(
        monkeypatch,
        managed=None,
        holder_pids=(333612,),
        record=333612,
        stop_outcome="stubborn",
    )

    result = relay.service_action("restart")

    assert result["ok"] is False, result
    assert result["reason"] == "port_held"
    assert "pid 333612" in result["error"]
    assert "lop network serve" in result["error"]
    # Round-1 D2: the kind is named ONCE — the outer clause no longer stutters
    # into its own gloss.
    assert result["error"].count("started by hand") == 1, result["error"]
    assert "did not stop" in result["error"]


def test_a_foreign_listener_blocks_the_action_with_pid_and_cmdline_named(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder that is NOT ours: the managed relay cannot come up, and the
    refusal must name the process (pid/port/cmdline) instead of reporting a
    service problem. Never killed — not this install's to stop."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    stops: list[dict[str, Any]] = []
    monkeypatch.setattr(supervisors, "systemctl_user", lambda *a, **k: Done())
    _pin_supervision(
        monkeypatch,
        managed=_Script(700, None),
        holder_pids=(60123,),
        record=_RelayRecordScript(700, 700),
        cmdline=lambda pid: "caddy reverse-proxy --from :4097" if pid == 60123 else "lop relay",
        stop_calls=stops,
    )

    result = relay.service_action("restart")

    assert result["ok"] is False, result
    assert result["reason"] == "not_restarted"
    assert "pid 60123" in result["error"]
    assert "caddy reverse-proxy" in result["error"]
    assert stops == [], "a foreign process must never be signalled"


def test_stop_reports_a_surviving_hand_started_relay_rather_than_ok(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service stops, but a hand-started relay of ours keeps answering: the
    verb says so (nothing of ours was left serving by accident, and the unit
    alone stopping is not the fact a reader asked for)."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(supervisors, "systemctl_user", lambda *a, **k: Done())
    _pin_supervision(
        monkeypatch, managed=1000, record=333612, holder_pids=(333612,), answering=True
    )

    result = relay.service_action("stop")

    assert result["ok"] is False, result
    assert result["reason"] == "not_stopped"
    assert "pid 333612" in result["error"]


def test_stop_is_ok_once_nothing_of_ours_answers(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(supervisors, "systemctl_user", lambda *a, **k: Done())
    _pin_supervision(monkeypatch, managed=_Script(1000, None), record=None, answering=False)

    assert relay.service_action("stop")["ok"] is True


def test_the_adoption_signal_is_scoped_to_the_named_pid(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_stop_manual_holder`` with its real teeth, against REAL processes: a
    spawned argv carrying the relay markers is signalled and observed to exit;
    a pid that is already gone is ``gone`` (nothing to signal); and the guards
    refuse to signal this process or pid 1. Nothing here ever touches a process
    this session did not create."""
    import os
    import subprocess

    assert relay._stop_manual_holder({"pid": os.getpid()}) == "unverified", "never signal self"
    assert relay._stop_manual_holder({"pid": 1}) == "unverified", "never signal pid 1"
    # A real process whose ARGV carries the product's markers — the stand-in
    # for a hand-started `lop network serve`. `exec -a` puts the marker text
    # into argv[0]; nothing is executed but `sleep`. The wait is for the exec
    # itself: until bash hands over, `ps` still shows bash's own command line,
    # which is exactly the "command line does not verify yet" state the check
    # is entitled to refuse.
    import time

    child = subprocess.Popen(["bash", "-c", 'exec -a "lop network serve" sleep 60'])
    try:
        fetched = ""
        for _ in range(60):
            fetched = relay._pid_cmdline(child.pid)
            if relay._is_own_relay_command(fetched):
                break
            time.sleep(0.05)
        assert "network serve" in fetched, fetched
        assert relay._is_own_relay_command(fetched), fetched
        outcome = relay._stop_manual_holder({"pid": child.pid, "cmdline": fetched}, timeout=8.0)
        assert outcome == "stopped", outcome
        assert child.poll() is not None, "the holder must actually have stopped"
    finally:
        child.kill()
        child.wait()
    assert (
        relay._stop_manual_holder({"pid": child.pid, "cmdline": "lop network serve"}) == "gone"
    ), "already-gone is nothing to signal"


def test_a_reused_record_pid_is_never_signalled(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-1 MAJOR-1's repro, pinned as the guard it must now be: the relay
    record outlives an unclean death, and once the OS reuses its number for
    anything else, adoption must not reach it. The stored pid is a REAL live
    process whose command line carries no product markers and which listens on
    nothing — not a holder at all: no signal, no refusal, and the action
    proceeds (the port is free). The real stop path is left in place so a
    regression that signalled would be observable."""
    import subprocess
    import sys

    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)

    def _fake_install(port: int, *, dry_run: bool = False) -> dict[str, Any]:
        return {"ok": True, "steps": ["installed"]}

    monkeypatch.setattr(relay, "_install_systemd", _fake_install)
    _no_systemctl(monkeypatch)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _pin_supervision(
            monkeypatch,
            managed=_Script(None, 9001),
            holder_pids=(),  # no lsof evidence: the number came from the record
            record=_RelayRecordScript(child.pid, 9001),
            cmdline=lambda pid: "sleep 300" if pid == child.pid else "x",
            stop_stub=False,
        )

        result = relay.service_action("restart")

        assert result["ok"] is True, result
        assert child.poll() is None, "the unrelated process must still be alive"
    finally:
        child.kill()
        child.wait()


def test_a_record_claimed_listener_without_markers_is_named_never_signalled(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder that LISTS and whose command line does not verify: never
    pre-signalled — the adoption loop only touches marker-verified holders —
    and when the arm fails, the refusal names it as a process this install does
    not manage. The record naming it does not make it ours."""
    import subprocess
    import sys

    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")

    class Fail:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(supervisors, "systemctl_user", lambda *a, **k: Fail())
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _pin_supervision(
            monkeypatch,
            managed=_Script(700, 700),
            holder_pids=(child.pid,),
            record=_RelayRecordScript(child.pid, child.pid),
            cmdline=lambda pid: "postgres -D :4097" if pid == child.pid else "x",
            stop_stub=False,
        )

        result = relay.service_action("restart")

        assert result["ok"] is False, result
        assert str(child.pid) in result["error"]
        assert "does not manage" in result["error"]
        assert child.poll() is None, "an unverifiable listener must not be signalled"
    finally:
        child.kill()
        child.wait()


def test_a_command_line_that_changes_before_the_signal_is_refused(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-1 MAJOR-1's second half: a pid can be reused BETWEEN the probe and
    the signal. The already-fetched command line matches; the fresh re-read does
    not — the adoption refuses with the holder named and nothing is signalled.
    The stored pid is a REAL live process, so a signal would be observable."""
    import subprocess
    import sys

    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    _no_systemctl(monkeypatch)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _pin_supervision(
            monkeypatch,
            managed=None,
            holder_pids=(child.pid,),
            record=None,
            cmdline=_Script("/usr/bin/lop network serve", "sleep 300"),
            stop_stub=False,
        )

        result = relay.service_action("restart")

        assert result["ok"] is False, result
        assert result["reason"] == "port_held"
        assert "could not be verified" in result["error"]
        assert str(child.pid) in result["error"]
        assert child.poll() is None, "nothing may be signalled without the fresh match"
    finally:
        child.kill()
        child.wait()


def test_a_failed_arm_says_the_hand_started_relay_was_already_stopped(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-1 MINOR-2: adoption succeeded (the operator's relay is STOPPED and
    nothing of theirs is serving), then the arm failed — the response must say
    both facts, not only the arm's error."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")

    class Fail:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(supervisors, "systemctl_user", lambda *a, **k: Fail())
    stops: list[dict[str, Any]] = []
    _pin_supervision(
        monkeypatch,
        managed=None,
        holder_pids=(333612,),
        record=_RelayRecordScript(333612, None),
        stop_calls=stops,
        stop_outcome="stopped",
    )

    result = relay.service_action("restart")

    assert result["ok"] is False, result
    assert [h["pid"] for h in stops] == [333612], stops
    assert "was stopped first" in result["error"], result["error"]
    assert "pid 333612" in result["error"]
    assert result.get("stopped") == [333612], result
