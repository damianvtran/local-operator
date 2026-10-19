"""The relay's Linux supervision arm — systemd ``--user`` (remote onboarding slice (b)).

WHY THIS FILE EXISTS SEPARATELY. ``test_purge`` pins the LAUNCHD half of the
supervision surface plus the no-supervisor refusal; these cells pin the half
that did not exist until slice (b): the unit render, the write/enable path (dry
run and file half only — a test never reaches a user manager), and the
missing-unit start that installs. ``sys.platform`` and ``shutil.which`` are
patched to the Linux answer rather than stubbing ``is_supported`` itself, so
the real expression runs (the same discipline ``test_purge`` documents).

NO CELL HERE MAY REACH ``systemctl``: every one that could is either a dry run,
has ``systemctl_user`` replaced by something that raises when called, or — the
``refresh_unit_if_stale`` cells, whose subject IS the manager conversation — by
a scripted recorder whose calls the cell counts. No cell reaches a real
manager.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import procname, supervisors
from local_operator import update as update_mod
from local_operator.network import relay

# tmp_path fixtures -------------------------------------------------------


@pytest.fixture()
def linux_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A HOME the unit path resolves under, claiming linux + systemctl."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(supervisors, "enable_linger", lambda: True)
    monkeypatch.setattr(supervisors, "linger_enabled", lambda: True)
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
    # THE SESSION-SAFETY FIX (2026-10-11, stale-relay memo): the relay's cgroup
    # holds the device's session runtimes, so its unit must kill only the main
    # process — the default killed every session on every restart. The decision
    # pin beside it: NO ExecStop/ExecStopPost reaper, because those fire on
    # RESTART too and would recreate exactly the defect this line removes.
    assert "KillMode=process" in text
    assert "ExecStop" not in text


def test_no_other_daemons_unit_grows_the_relays_kill_mode(tmp_path: Path) -> None:
    """``kill_mode`` is opt-in: the shared renderer and every sibling unit are
    byte-unchanged, which is the other half of the KillMode pin.

    The relay is deliberately the ONE unit that names a kill mode (see
    ``supervisors.render_systemd_unit`` for why); a sibling that grew the line
    would change what its own daemon's stop does, which no one asked for.
    """
    from local_operator.browser_bridge import install as browser_install
    from local_operator.mobile import install as mobile_install
    from local_operator.tunnels import install as tunnel_install
    from local_operator.wakes import install as wakes_install

    base = supervisors.render_systemd_unit(description="d", exec_start="/bin/true")
    assert "KillMode" not in base
    rendered = {
        # The appended units' own renders, each called the way its installer
        # calls it (browser's version probe is pinned, not detected).
        "browser bridge": browser_install.render_systemd(4099, version=255),
        "tunnel": tunnel_install.render_systemd(),
        "wakes supervisor": wakes_install.render_systemd(tmp_path / "cfg"),
        "mobile": mobile_install.render_systemd(4098),
    }
    for name, text in rendered.items():
        assert "KillMode" not in text, name
        assert "ExecStop" not in text, name


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
    stop_outcome: str | list[str] = "stopped",
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
        if isinstance(stop_outcome, list):
            return stop_outcome.pop(0) if stop_outcome else "stubborn"
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
    assert seen == [
        ("restart", relay.SYSTEMD_UNIT),
        ("is-enabled", relay.SYSTEMD_UNIT),
        ("is-active", relay.SYSTEMD_UNIT),
    ]


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
    # D-N1: the imperative names its own holder by pid, not an ambiguous
    # "that process" that a context sentence could reassign.
    assert "Stop that relay (pid 333612) and retry" in result["error"], result["error"]


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


def test_a_refusal_after_an_earlier_adoption_reports_the_stop(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 MINOR-1: holder A is adopted (stopped), holder B then fails —
    the refusal must say A was stopped and carry ``stopped``; "Nothing was
    changed" is only true when nothing was."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    _no_systemctl(monkeypatch)
    _pin_supervision(
        monkeypatch,
        managed=None,
        holder_pids=(111, 222),
        record=None,
        stop_outcome=["stopped", "stubborn"],
    )

    result = relay.service_action("restart")

    assert result["ok"] is False, result
    assert result["reason"] == "port_held"
    assert "Nothing was changed" not in result["error"], result["error"]
    assert "was stopped first by this command" in result["error"]
    assert "did not stop" in result["error"]
    assert "pid 111" in result["error"]
    assert "pid 222" in result["error"]
    assert result.get("stopped") == [111], result


def test_a_gone_holder_is_not_folded_into_was_stopped_first(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-R3-1: an earlier holder that was ALREADY GONE (nothing signalled) must
    not read as "was stopped first by this command" in a later refusal's
    context — NIT-2's distinction, on the refusal's own sentence. D-N1 rides the
    same cell: the stubborn imperative names ITS holder by pid, so "that relay"
    cannot attach to the gone pid the context just named."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    _no_systemctl(monkeypatch)
    _pin_supervision(
        monkeypatch,
        managed=None,
        holder_pids=(111, 222),
        record=None,
        stop_outcome=["gone", "stubborn"],
    )

    result = relay.service_action("restart")

    assert result["ok"] is False, result
    assert result["reason"] == "port_held"
    error = result["error"]
    assert "The hand-started relay (pid 111) was already gone." in error, error
    assert "was stopped first by this command" not in error, error
    assert "Nothing was changed" not in error, error
    assert "Stop that relay (pid 222) and retry" in error, error


def test_a_gone_holder_is_labelled_gone_not_stopped(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 NIT-2: a holder that had already exited when the loop reached it
    was not signalled — the report says "already gone", not "stopped"."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)

    def _fake_install(port: int, *, dry_run: bool = False) -> dict[str, Any]:
        return {"ok": True, "steps": ["installed"]}

    monkeypatch.setattr(relay, "_install_systemd", _fake_install)
    _no_systemctl(monkeypatch)
    _pin_supervision(
        monkeypatch,
        managed=_Script(None, 9001),
        holder_pids=(333612,),
        record=_RelayRecordScript(333612, 9001),
        stop_outcome="gone",
    )

    result = relay.service_action("restart")

    assert result["ok"] is True, result
    assert any("was already gone" in step for step in result.get("steps", [])), result
    assert not any(
        "stopped the hand-started relay" in step for step in result.get("steps", [])
    ), result


def test_status_installed_and_served_by_follow_the_facts(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 MINOR-2 + D6: ``status()`` reports ``installed`` from THIS
    platform's unit file (a revert to launchd-only would read false under the
    linux fixture), and ``relay_served_by`` from the SERVING PROCESS — the drill
    node's exact state is a unit file that EXISTS while a hand-started
    ``lop network serve`` holds :4097, and the file must not be what names the
    kind. A module-form relay whose supervisor pid cannot be read stays None."""
    from types import SimpleNamespace

    monkeypatch.setattr(relay, "is_supported", lambda: True)
    record = SimpleNamespace(pid=9001, to_json=lambda: {"pid": 9001})

    def _pin_status_record() -> None:
        monkeypatch.setattr(relay.store, "scan_own_relay", lambda root=None: (record, "live"))

    # The unit file half: written → installed true; removed → false.
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")
    _pin_supervision(monkeypatch, managed=None, cmdline=lambda pid: "/srv/lop network serve")
    _pin_status_record()
    first = relay.status()
    assert first["installed"] is True, first
    # THE NODE'S STATE: the file exists AND the kind is the process's — the
    # foreground serve shape no unit of this product runs.
    assert first["relay_served_by"] == "manual", first

    unit.unlink()
    assert relay.status()["installed"] is False

    # The verified service case: the supervisor reports THIS pid as its own.
    _pin_supervision(
        monkeypatch,
        managed=9001,
        cmdline=lambda pid: "/usr/bin/python -m local_operator.network.relay --port 4097",
    )
    _pin_status_record()
    assert relay.status()["relay_served_by"] == "service"

    # A module-form relay whose supervisor pid cannot be read: no claim.
    _pin_supervision(
        monkeypatch,
        managed=None,
        cmdline=lambda pid: "/usr/bin/python -m local_operator.network.relay --port 4097",
    )
    _pin_status_record()
    assert relay.status()["relay_served_by"] is None

    # The darwin branch of the same fact, read from ITS unit path.
    monkeypatch.setattr(relay.sys, "platform", "darwin")
    plist = relay.plist_path()
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text("(plist)\n", encoding="utf-8")
    assert relay.status()["installed"] is True
    plist.unlink()
    assert relay.status()["installed"] is False


# refresh_unit_if_stale: the daemon-refresh seam's Linux arm (2026-10-11) ------
#
# The stale-relay memo's T2 matrix, cell for cell. The discipline is the
# sibling file's (tests/unit/test_daemon_plist_refresh.py): the guard cells arm
# ``systemctl_user`` to RAISE and touch nothing; the manager cells script it and
# COUNT — the recorder is what "exactly one restart" reads. Nothing here is
# allowed to reach a real manager.


_GEN = "20260921T125352Z-0.61.12"


def _fake_shim(tmp_path: Path) -> Path:
    return tmp_path / "lop" / "bin" / "python3"


def _shim_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both sides of the image question, pointed at one fake shim.

    ``render_systemd`` names it and ``_names_the_shim`` compares against it, so
    a content compare in a cell is a statement about the unit and not about the
    test venv's own interpreter.
    """
    shim = _fake_shim(tmp_path)
    monkeypatch.setattr(procname, "supervised_image", lambda: shim)
    monkeypatch.setattr(update_mod, "daemon_image_path", lambda: shim)


def _manager_recorder(
    *, returncode: int = 0, stderr: str = ""
) -> tuple[list[tuple[str, ...]], Any]:
    """A scripted ``systemctl_user``: records calls, answers with the given rc."""
    calls: list[tuple[str, ...]] = []

    def _record(*args: str, **kwargs: Any) -> Any:
        calls.append(tuple(args))
        return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)

    return calls, _record


def _migratable_unit() -> str:
    """A pre-KillMode relay unit — the shape the migration exists to move."""
    return relay.render_systemd().replace("KillMode=process\n", "")


def test_refresh_unit_missing_declines_before_any_manager_call(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No unit, no question: decline, and a manager is never touched."""
    _no_systemctl(monkeypatch)

    outcome = relay.refresh_unit_if_stale()

    assert outcome.kind == "not-installed", outcome
    assert outcome.summary() == "" and outcome.warning() == ""


def test_refresh_unit_it_cannot_address_leaves_the_file_alone(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redirected HOME declines BEFORE anything is read, rewritten or run."""
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(_migratable_unit(), encoding="utf-8")
    before = unit.read_text(encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: False)
    _no_systemctl(monkeypatch)

    outcome = relay.refresh_unit_if_stale()

    assert outcome.kind == "not-addressable", outcome
    assert unit.read_text(encoding="utf-8") == before


def test_refresh_unit_with_a_foreign_image_is_not_addressable(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit that would restart through another tree is not this run's to move.

    The image guard uses the REAL ``_names_the_shim`` — the unit's ``ExecStart``
    names a foreign interpreter, so the shim comparison is what declines and the
    file is never rewritten toward this run's tree.
    """
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(
        "[Unit]\n"
        "Description=Local Operator network relay\n"
        "\n"
        "[Service]\n"
        'ExecStart="/opt/other-tree/bin/python3" -m local_operator.network.relay --port 4097\n'
        "Restart=on-failure\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n",
        encoding="utf-8",
    )
    before = unit.read_text(encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: True)
    _no_systemctl(monkeypatch)

    outcome = relay.refresh_unit_if_stale()

    assert outcome.kind == "not-addressable", outcome
    assert unit.read_text(encoding="utf-8") == before


def test_refresh_unit_is_unsupported_off_linux(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The darwin-here case: no Linux arm runs on a non-Linux host."""
    monkeypatch.setattr(relay.sys, "platform", "darwin")
    _no_systemctl(monkeypatch)

    outcome = relay.refresh_unit_if_stale()

    assert outcome.kind == "unsupported", outcome


def test_refresh_unit_migrates_a_stale_unit_and_reloads_without_a_restart(
    linux_host: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migration half: rewrite + daemon-reload, and NOTHING restarts.

    This is how an installed unit grows ``KillMode=process`` (the
    session-safety fix), and it is also the first half's safety ORDER: the
    rewrite lands BEFORE any restart the second half might perform. A rewrite
    alone restarts nothing (``daemon-reload`` covers the next start), so the
    summary stays silent — the launchd ``repaired`` sentence would claim a
    restart that did not happen here.
    """
    _shim_rig(tmp_path, monkeypatch)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    legacy = _migratable_unit()
    assert "KillMode=process\n" not in legacy
    unit.write_text(legacy, encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: True)
    calls, record = _manager_recorder()
    monkeypatch.setattr(supervisors, "systemctl_user", record)
    _pin_supervision(monkeypatch, managed=_Script(111), record=_RelayRecordScript(111))
    monkeypatch.setattr(update_mod, "stale_generation_of_process", lambda pid: None)

    outcome = relay.refresh_unit_if_stale()

    assert calls == [("daemon-reload",)], calls
    written = unit.read_text(encoding="utf-8")
    assert written == relay.render_systemd(), "the unit is now this build's render"
    assert "KillMode=process\n" in written, "the migration is the point"
    assert outcome.kind == "current", outcome
    assert outcome.summary() == "" and outcome.warning() == ""


def test_refresh_unit_restarts_exactly_once_on_a_proven_move(
    linux_host: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fleet's first-fix shape: migrate, reload, THEN restart — one restart.

    The proven move is the only thing that restarts (the second question's
    contract), the reload must precede it (the session-safe kill semantics have
    to be in force before the cgroup is touched), and the line the operator
    reads is the macOS family's, byte-identical, one row at 80.
    """
    _shim_rig(tmp_path, monkeypatch)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(_migratable_unit(), encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: True)
    calls, record = _manager_recorder()
    monkeypatch.setattr(supervisors, "systemctl_user", record)
    _pin_supervision(
        monkeypatch,
        managed=_Script(111, 111, 222),
        record=_RelayRecordScript(111, 222),
    )
    generation = tmp_path / "lop" / "generations" / _GEN
    monkeypatch.setattr(
        update_mod,
        "stale_generation_of_process",
        lambda pid: generation if pid == 111 else None,
    )

    outcome = relay.refresh_unit_if_stale()

    # The reload PRECEDES the restart: the order is the safety property.
    assert calls == [("daemon-reload",), ("restart", relay.SYSTEMD_UNIT)], calls
    assert outcome.kind == "restarted", outcome
    assert outcome.detail == _GEN, outcome.detail
    line = outcome.summary()
    assert line == (
        "network relay daemon: restarted onto the new build — expect a brief mesh blip"
    ), line
    assert len(line) <= 80, line
    assert outcome.warning() == ""


def test_refresh_unit_does_not_bounce_a_current_build(
    linux_host: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """False/None declines: a current build is read and left alone.

    ``systemctl_user`` is armed to RAISE here — the cell proves the reading
    stops at the probes and no manager call happens on the no-move path.
    """
    _shim_rig(tmp_path, monkeypatch)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(relay.render_systemd(), encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: True)
    _no_systemctl(monkeypatch)
    _pin_supervision(monkeypatch, managed=_Script(111), record=_RelayRecordScript(111))
    monkeypatch.setattr(update_mod, "stale_generation_of_process", lambda pid: None)

    outcome = relay.refresh_unit_if_stale()

    assert outcome.kind == "current", outcome
    assert outcome.summary() == "" and outcome.warning() == ""


def test_refresh_unit_a_same_pid_restart_is_a_refusal_with_the_remedy(
    linux_host: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drill's exact shape: the restart 'succeeded' and replaced nothing.

    One restart is ATTEMPTED and exactly one manager restart call is made; the
    honesty wrapper refuses it (same pid serving), and the outcome is the
    failure family's sentence with the installer named — never a claim.
    """
    _shim_rig(tmp_path, monkeypatch)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(relay.render_systemd(), encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: True)
    calls, record = _manager_recorder()
    monkeypatch.setattr(supervisors, "systemctl_user", record)
    _pin_supervision(
        monkeypatch,
        managed=_Script(333612),
        record=_RelayRecordScript(333612),
        holder_pids=(333612,),
    )
    monkeypatch.setattr(update_mod, "stale_generation_of_process", lambda pid: tmp_path / _GEN)

    outcome = relay.refresh_unit_if_stale()

    assert calls == [("daemon-reload",), ("restart", relay.SYSTEMD_UNIT)], calls
    assert outcome.kind == "failed", outcome
    assert outcome.detail == (
        f"it was still on an older install ({_GEN}) and the restart it accepted "
        "did not take — run `lop network install` to move it onto this build"
    ), outcome.detail
    assert outcome.summary() == ""
    assert outcome.warning() == (
        "warning: network relay daemon was not refreshed: " + outcome.detail
    )


def test_refresh_unit_reload_failure_stops_before_any_restart(
    linux_host: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewritten-but-unread unit is a FAILURE and no restart follows.

    The restart is refused at this point on purpose: the manager has not
    re-read the definition, so the kill semantics that make a restart safe are
    unproven — restarting here is the cgroup kill this slice removes.
    """
    _shim_rig(tmp_path, monkeypatch)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(_migratable_unit(), encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: True)
    calls, record = _manager_recorder(returncode=1, stderr="Failed to connect to bus: boom")
    monkeypatch.setattr(supervisors, "systemctl_user", record)
    _pin_supervision(
        monkeypatch,
        managed=_Script(111, 111, 222),
        record=_RelayRecordScript(111, 222),
    )
    monkeypatch.setattr(
        update_mod,
        "stale_generation_of_process",
        lambda pid: tmp_path / _GEN,
    )

    outcome = relay.refresh_unit_if_stale()

    assert calls == [("daemon-reload",)], calls
    assert outcome.kind == "failed", outcome
    assert "could not re-read it" in outcome.detail, outcome.detail
    assert "lop network install" in outcome.detail, outcome.detail
    # The file WAS rewritten (the failure is post-write) and now carries the
    # migration — the running relay is untouched, which the no-restart asserts.
    assert "KillMode=process\n" in unit.read_text(encoding="utf-8")


@pytest.mark.parametrize("retry_succeeds", [False, True])
def test_refresh_unit_retries_the_loaded_policy_after_a_failed_reload(
    linux_host: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_succeeds: bool,
) -> None:
    """Disk-current after a failed reload is not manager-current (R1/Q1).

    Model loaded policy separately from disk: only a successful reload changes
    KillMode. The real restart wrapper must verify a new serving PID as well.
    No real systemctl is reachable from this cell.
    """
    _shim_rig(tmp_path, monkeypatch)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(_migratable_unit(), encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: True)
    calls: list[tuple[str, ...]] = []
    loaded_mode = "control-group"
    restart_modes: list[str] = []
    reloads = 0

    def _manager(*args: str, **kwargs: Any) -> Any:
        nonlocal loaded_mode, reloads
        calls.append(tuple(args))
        if args == ("daemon-reload",):
            reloads += 1
            if reloads == 1 or not retry_succeeds:
                return SimpleNamespace(returncode=1, stdout="", stderr="reload failed")
            assert "KillMode=process\n" in unit.read_text(encoding="utf-8")
            loaded_mode = "process"
        elif args == ("restart", relay.SYSTEMD_UNIT):
            restart_modes.append(loaded_mode)
        else:
            raise AssertionError(f"unexpected manager call: {args}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(supervisors, "systemctl_user", _manager)
    # First call must fail before the PID probe; the second call owns all three
    # readings consumed by the refresh and the real restart verification.
    _pin_supervision(
        monkeypatch,
        managed=_Script(111, 111, 222),
        record=_RelayRecordScript(111, 222),
    )
    monkeypatch.setattr(update_mod, "stale_generation_of_process", lambda pid: tmp_path / _GEN)

    first = relay.refresh_unit_if_stale()
    assert first.kind == "failed", first
    assert calls == [("daemon-reload",)], calls
    assert "KillMode=process\n" in unit.read_text(encoding="utf-8")
    assert loaded_mode == "control-group"

    second = relay.refresh_unit_if_stale()
    assert calls[:2] == [("daemon-reload",), ("daemon-reload",)], calls
    if retry_succeeds:
        assert calls == [
            ("daemon-reload",),
            ("daemon-reload",),
            ("restart", relay.SYSTEMD_UNIT),
        ], calls
        assert restart_modes == ["process"], restart_modes
        assert second.kind == "restarted", second
        assert second.summary() == (
            "network relay daemon: restarted onto the new build — expect a brief mesh blip"
        )
    else:
        assert calls == [("daemon-reload",), ("daemon-reload",)], calls
        assert restart_modes == []
        assert loaded_mode == "control-group"
        assert second.kind == "failed", second
        assert "lop network install" in second.detail


def test_refresh_unit_a_raising_probe_is_a_failure_never_an_escape(
    linux_host: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected raise is reported — the upgrade already succeeded."""
    _shim_rig(tmp_path, monkeypatch)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(relay.render_systemd(), encoding="utf-8")
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda name: True)
    _no_systemctl(monkeypatch)
    _pin_supervision(monkeypatch, managed=_Script(111), record=_RelayRecordScript(111))

    def _boom(pid: int) -> Any:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(update_mod, "stale_generation_of_process", _boom)

    outcome = relay.refresh_unit_if_stale()

    assert outcome.kind == "failed", outcome
    assert "probe exploded" in outcome.detail, outcome.detail
    assert "lop network install" in outcome.detail, outcome.detail


@pytest.mark.parametrize(
    ("installed", "linger", "enabled", "persistent"),
    [
        (True, True, True, True),
        (True, False, True, False),
        (False, True, True, False),
        (True, True, False, False),
    ],
)
def test_supervision_state_requires_unit_enabled_and_linger(
    linux_host, monkeypatch, installed, linger, enabled, persistent
) -> None:
    """An active unit alone does not prove restart-on-boot durability."""
    if installed:
        unit = _unit(linux_host)
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text("(unit)\n")
    calls = []

    def systemctl(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0 if args[0] != "is-enabled" or enabled else 1)

    monkeypatch.setattr(supervisors, "systemctl_user", systemctl)
    monkeypatch.setattr(supervisors, "linger_enabled", lambda: linger)
    state = relay.supervision_state()
    assert state == {
        "platform": "systemd",
        "unit": relay.SYSTEMD_UNIT,
        "installed": installed,
        "enabled": enabled,
        "active": True,
        "linger": linger,
        "persistent": persistent,
    }
    assert calls == [("is-enabled", relay.SYSTEMD_UNIT), ("is-active", relay.SYSTEMD_UNIT)]


@pytest.mark.parametrize("linger", [True, False])
def test_existing_unit_start_repairs_linger_without_failing_the_start(
    linux_host, monkeypatch, linger
) -> None:
    """Existing installations must repair linger, not only missing units."""
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n")
    monkeypatch.setattr(relay, "_systemd_guard", lambda action: None)
    calls = []
    monkeypatch.setattr(supervisors, "enable_linger", lambda: calls.append("linger") or linger)
    monkeypatch.setattr(supervisors, "linger_enabled", lambda: linger)
    monkeypatch.setattr(
        supervisors, "systemctl_user", lambda *a, **k: SimpleNamespace(returncode=0, stderr="")
    )
    result = relay._service_action_systemd("start")
    assert calls == ["linger"]
    assert result["ok"] is True
    assert result["linger"] is linger and result["persistent"] is linger


@pytest.mark.parametrize(
    "stdout, expected", [("Linger=yes\n", True), ("Linger=no\n", False), ("", False)]
)
def test_linger_state_reads_loginctl_without_mutating_it(monkeypatch, stdout, expected) -> None:
    """Diagnostics query loginctl; only a service action may enable linger."""
    calls = []
    monkeypatch.setattr(supervisors.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setenv("USER", "synthetic-user")

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(supervisors.subprocess, "run", run)
    assert supervisors.linger_enabled() is expected
    assert calls == [["/usr/bin/loginctl", "show-user", "synthetic-user", "-p", "Linger"]]


def test_supervision_state_without_a_supervisor_is_not_persistent(monkeypatch) -> None:
    """A host without a manager cannot claim durability."""
    monkeypatch.setattr(supervisors, "supervisor", lambda: None)
    state = relay.supervision_state()
    assert state["platform"] == "none" and not state["persistent"] and not state["installed"]


def test_supervision_state_survives_an_unavailable_user_bus(linux_host, monkeypatch) -> None:
    """Unavailable systemd is diagnostic evidence, never an exception to callers."""

    def unavailable(*args, **kwargs):
        raise OSError("synthetic bus unavailable")

    monkeypatch.setattr(supervisors, "systemctl_user", unavailable)
    state = relay.supervision_state()
    assert not state["enabled"] and not state["active"] and not state["persistent"]


@pytest.mark.parametrize("loaded", [False, True])
def test_launchd_persistence_follows_the_installed_run_at_load_plist(
    monkeypatch, tmp_path, loaded
) -> None:
    """Launchd boot persistence is distinct from whether the job is loaded now."""
    plist = tmp_path / "relay.plist"
    plist.write_text("synthetic plist")
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
    monkeypatch.setattr(relay, "plist_path", lambda: plist)
    monkeypatch.setattr(relay, "_domain", lambda: "gui/synthetic")
    monkeypatch.setattr(
        relay, "_launchctl", lambda *a: SimpleNamespace(returncode=0 if loaded else 1)
    )
    state = relay.supervision_state()
    assert state["platform"] == "launchd" and state["installed"]
    assert state["persistent"] and state["active"] is loaded
    assert state["linger"] is None
