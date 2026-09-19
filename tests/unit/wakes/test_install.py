"""Install-on-demand for the wake supervisor.

**These tests must never install a real LaunchAgent.** ``launchctl`` has no
sandbox: it addresses the calling user's live session whatever ``Path.home()``
says, so the obvious installer test — patch home, call the installer — wrote a
plist under a pytest tmpdir and then bootstrapped a REAL supervised unit into
the developer's session pointed at that tmpdir. It was observed here:
``launchctl print gui/501/com.local-operator.wakes`` reported a live unit whose
plist lived under ``/private/var/folders/…/pytest-of-damian/``.

``_launchd_is_addressable`` is the guard that makes this file safe: the plist
is written wherever ``plist_path()`` points, but launchd is only addressed when
that path is inside the real passwd home. Every test below runs under a
redirected home, so the file half is exercised in full and the process half
refuses. A test that needs to assert on ``launchctl`` behaviour must fake the
subprocess, never call it.
"""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path
from typing import cast

import pytest

from local_operator.wakes.install import (
    LABEL,
    SELF_HEAL_INTERVAL_S,
    UNSUPPORTED_REASON,
    ensure_supervisor_installed,
    plist_path,
    render_plist,
)


def _supervisor_kind() -> str | None:
    """Which supervisor THIS machine would be addressed on.

    The launchd-shaped tests below used to gate on ``is_supported()``, which
    was true on darwin only — so on a Linux runner they fell into the
    "unsupported" branch and asserted nothing. Now that ``is_supported()`` is
    true on all three platforms, the honest gate is the supervisor's IDENTITY:
    these tests fake ``launchctl``, so they are only meaningful where launchctl
    is what the installer would call.
    """
    from local_operator import supervisors

    return supervisors.supervisor()


@pytest.fixture
def redirected_home(tmp_path: Path, monkeypatch) -> Path:  # noqa: ANN001
    """Point ``Path.home()`` at a tmpdir. The launchd guard does the rest."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_the_installer_never_touches_the_real_launch_agents_directory(
    redirected_home: Path,
) -> None:
    """The guard itself, asserted rather than assumed.

    This is the test that would have caught the stray unit: under a redirected
    home the installer must decline to reach launchd, whatever else it does.
    """
    from local_operator.wakes.install import _launchd_is_addressable

    assert _launchd_is_addressable() is False


def test_install_writes_a_plist_but_declines_to_load_it(
    redirected_home: Path, tmp_path: Path
) -> None:
    """The file half runs; the half that reaches a live supervisor refuses.

    On macOS that means a plist and a launchd call that declines because the
    path is not the one the real passwd home owns. On Linux the same two halves
    are a systemd unit and a ``systemctl`` call that declines for the same
    reason, and the unit file it wrote is the real thing a user would get — so
    this asserts the CONTENT per supervisor rather than assuming a plist.
    """
    outcome = ensure_supervisor_installed(tmp_path / "config")
    kind = _supervisor_kind()

    if kind == "launchctl":
        # The file half runs in full: this is what launchd WOULD be handed.
        assert plist_path().exists()
        written = plistlib.loads(plist_path().read_bytes())
        assert written == render_plist(tmp_path / "config")
    elif kind == "systemctl":
        # The unit, not a plist: same guard, same refusal, different file.
        from local_operator.wakes.install import SYSTEMD_TIMER, SYSTEMD_UNIT

        unit = plist_path()
        assert unit.name == SYSTEMD_UNIT
        text = unit.read_text(encoding="utf-8")
        assert "ExecStart=" in text and "local_operator.wakes.supervisor" in text
        # The store travels in the unit, or a second profile would be supervised
        # by a unit watching the default one.
        assert f"LOCAL_OPERATOR_CONFIG_DIR={tmp_path / 'config'}" in text
        # Restart=on-failure and NOT Restart=always: the supervisor exits 0 when
        # the index empties and that exit has to stick.
        assert "Restart=on-failure" in text
        timer = unit.with_name(SYSTEMD_TIMER)
        assert timer.exists()
        assert f"OnUnitInactiveSec={SELF_HEAL_INTERVAL_S}s" in timer.read_text(encoding="utf-8")
    else:
        assert outcome.installed is False
        assert outcome.reason == UNSUPPORTED_REASON
        return

    # The process half refuses under a redirected home on every platform.
    assert outcome.installed is False
    assert "not addressable" in outcome.reason or "not the one the real home owns" in outcome.reason


@pytest.mark.skipif(sys.platform != "darwin", reason="LaunchAgent plist is macOS-only")
def test_the_unsupported_reason_names_a_supervisor_and_a_remedy() -> None:
    """D6: the text said "no supervisor installer for this platform".

    That was the pre-PR fact. After this branch all three platforms HAVE an
    installer, so what is missing on such a host is a SUPERVISOR — and the three
    sibling daemons already say it that way
    (``supervisors.no_supervisor_error``), so a user reading four refusals must
    not get two different stories. Asserted on the copy itself: a test that only
    compares the outcome to the constant passes whatever the constant says.
    """
    assert "no supervisor installer" not in UNSUPPORTED_REASON
    for named in ("launchctl", "systemctl --user", "schtasks"):
        assert named in UNSUPPORTED_REASON, UNSUPPORTED_REASON


def test_the_plist_lets_a_finished_supervisor_stay_down(tmp_path: Path) -> None:
    """``KeepAlive: {SuccessfulExit: False}`` is load-bearing, not decoration.

    The supervisor exits 0 when the wake index empties. This key is what makes
    that exit stick — a plain ``KeepAlive: true`` would restart it forever
    against an empty index, which is exactly the always-on cost the design set
    out to avoid.
    """
    plan = render_plist(tmp_path)
    assert plan["KeepAlive"] == {"SuccessfulExit": False}
    assert plan["Label"] == LABEL
    argv = cast(list[str], plan["ProgramArguments"])
    assert argv[1:] == ["-m", "local_operator.wakes.supervisor"]
    # The config dir travels in the environment: a second profile (or a test)
    # must be able to supervise its own store rather than the default one.
    env = cast(dict[str, str], plan["EnvironmentVariables"])
    assert env["LOCAL_OPERATOR_CONFIG_DIR"] == str(tmp_path)


@pytest.mark.skipif(sys.platform != "darwin", reason="LaunchAgent plist is macOS-only")
def test_a_stale_plist_is_rewritten_rather_than_trusted(
    redirected_home: Path, tmp_path: Path
) -> None:
    """Idempotent by CONTENT, not by existence.

    A plist from an older release names a different interpreter or a different
    config dir. Treating "a file is there" as "installed" would leave that
    stale unit supervising the wrong store forever.
    """
    plist_path().parent.mkdir(parents=True, exist_ok=True)
    plist_path().write_bytes(plistlib.dumps({"Label": LABEL, "ProgramArguments": ["/bin/false"]}))

    ensure_supervisor_installed(tmp_path / "config")

    rewritten = plistlib.loads(plist_path().read_bytes())
    assert rewritten == render_plist(tmp_path / "config")


def test_install_never_raises_even_on_an_unwritable_target(
    redirected_home: Path, tmp_path: Path, monkeypatch
) -> None:
    """The caller is the wake persist path, which has already succeeded.

    An installer failure must be reported through the outcome and never
    propagate: the schedule is durable, and a wake the user asked for must not
    turn into an exception because a plist could not be written.
    """

    def _boom(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "write_bytes", _boom)

    outcome = ensure_supervisor_installed(tmp_path / "config")
    assert outcome.installed is False
    assert outcome.reason


def test_the_guard_holds_when_the_redirected_home_is_inside_the_real_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode the other guard test structurally cannot reach.

    Round 1 (R4): the guard asked whether the plist path was *under* the real
    passwd home, which is a LOCATION test. Set `TMPDIR` inside `$HOME` — not
    exotic; it is how you avoid `/var/folders` cleanup races — and pytest's
    `tmp_path`, and so a patched `Path.home()`, lands inside the real home and
    satisfies containment. The guard would then wave a real launchd bootstrap
    through against a directory about to be deleted, which is the exact
    incident it exists to prevent.

    Every existing test uses `tmp_path`, whose location follows `TMPDIR`, so
    none of them can express this. This one builds the pathological home
    explicitly under the REAL passwd home and asserts the guard still refuses.
    """
    import pwd

    from local_operator.wakes.install import _launchd_is_addressable

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    # A path shaped like the dangerous case. Never created: the guard is a
    # pure path comparison, and this test must not write into the real home.
    pathological = real_home / "tmp-pytest-sandbox" / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: pathological))

    assert _launchd_is_addressable() is False, (
        "a redirected home INSIDE the real home was accepted as addressable; "
        "the guard has regressed from an identity test to a location test"
    )


def test_the_guard_accepts_only_the_genuine_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive half: the guard is not vacuously False.

    Without this, replacing the body with `return False` would pass every
    other test in this file while silently disabling install-on-demand for
    every real user.
    """
    import pwd

    from local_operator.wakes.install import _launchd_is_addressable

    if _supervisor_kind() != "launchctl":
        pytest.skip("launchd guard is only meaningful on darwin with launchctl")

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: real_home))

    assert _launchd_is_addressable() is True


def test_the_write_guard_refuses_config_dirs_outside_the_real_home(
    tmp_path: Path,
) -> None:
    """The round-2 escape: real HOME, redirected config dir.

    The round-1 guard covered the `launchctl` call but not the WRITE, and the
    write is the half that escapes that combination: `plist_path()` is the
    real `~/Library/LaunchAgents` whenever HOME is real, so a sandbox run
    planted a supervised unit in the operator's live launchd domain, pointed
    at a store that dies with the sandbox.

    Asserted on the decision predicate rather than by running the installer
    against the real home: a test that exercises the escape end to end would
    REPRODUCE the incident on the very machine it guards if the guard ever
    regressed. The predicate is the whole decision — the branch that consumes
    it is two lines.
    """
    import pwd

    from local_operator.wakes.install import _config_lives_in_real_home

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert _config_lives_in_real_home(tmp_path / "sandbox-cfg") is False
    assert _config_lives_in_real_home(real_home / ".local-operator") is True


# --- The running probe, and the permanent miss it closes ----------------------
#
# These never call `launchctl`: `_launchctl` is patched, which is also what
# keeps them inside this file's safety contract (see the module docstring).
# `_parse_supervisor_state` is pure, so the recorded launchd output below is
# verbatim from this machine rather than invented.

#: Real `launchctl print` output for a RUNNING job, trimmed to the lines the
#: parse reads. Kept verbatim so a launchd format change fails this test
#: rather than silently changing what "running" means.
_PRINT_RUNNING = """\
gui/501/com.local-operator.wakes = {
\tactive count = 1
\tpath = /Users/x/Library/LaunchAgents/com.local-operator.wakes.plist
\tstate = running

\tdomain = gui/501 [100023]
\truns = 3
\tpid = 47545
}
"""

#: Real output for a job that has EXITED. This is the whole defect: launchd
#: still prints it, still exits 0, and reports no pid.
_PRINT_EXITED = """\
gui/501/com.local-operator.waketest = {
\tactive count = 0
\tpath = /Users/x/Library/LaunchAgents/com.local-operator.waketest.plist
\tstate = not running

\truns = 1
\tlast exit code = 0
}
"""


def _reachable(mod, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """Let the probe reach its (faked) launchctl.

    `supervisor_state` applies both installer guards before probing, so that
    an isolated store can never be told about the REAL user's supervisor.
    Tests about the PARSE have to step past them; the guards themselves are
    asserted directly, above and below.
    """
    monkeypatch.setattr(mod, "_launchd_is_addressable", lambda: True)
    monkeypatch.setattr(mod, "_config_lives_in_real_home", lambda _config: True)


class _FakeLaunchctl:
    """Records every ``launchctl`` argv and replays canned results."""

    def __init__(self, results: dict[str, tuple[int, str]]) -> None:
        self._results = results
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str):  # noqa: ANN204
        import subprocess

        self.calls.append(args)
        code, stdout = self._results.get(args[0], (0, ""))
        return subprocess.CompletedProcess(list(args), code, stdout, "")


def test_a_loaded_but_exited_job_is_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE REGRESSION TEST for the permanent miss.

    `launchctl print` returns 0 for a job that has exited, so the old
    `_is_loaded()` check ("print returned 0") reported an armed wake as
    supervised while nothing was running to fire it. The probe must read the
    OUTPUT, not the exit code.
    """
    from local_operator.wakes import install as mod

    if _supervisor_kind() != "launchctl":
        pytest.skip("launchd probe is only meaningful on darwin with launchctl")

    fake = _FakeLaunchctl({"print": (0, _PRINT_EXITED)})
    monkeypatch.setattr(mod, "_launchctl", fake)
    _reachable(mod, monkeypatch)

    state = mod.supervisor_state(Path("/anywhere"))

    assert state.loaded is True, "launchd does still know the label"
    assert state.running is False, (
        "a loaded-but-exited job was reported as running; the probe has regressed to "
        "trusting launchctl's exit code, which is how an armed wake was left with no "
        "live supervisor"
    )
    assert state.pid is None


def test_a_running_job_is_reported_with_its_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive half: the probe is not vacuously False.

    Without this, `return SupervisorState(loaded=True, running=False, ...)`
    would pass the regression test above while making every persist kickstart
    a perfectly healthy supervisor.
    """
    from local_operator.wakes import install as mod

    if _supervisor_kind() != "launchctl":
        pytest.skip("launchd probe is only meaningful on darwin with launchctl")

    monkeypatch.setattr(mod, "_launchctl", _FakeLaunchctl({"print": (0, _PRINT_RUNNING)}))
    _reachable(mod, monkeypatch)

    state = mod.supervisor_state(Path("/anywhere"))

    assert (state.loaded, state.running, state.pid) == (True, True, 47545)


def test_an_absent_job_is_neither_loaded_nor_running(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.wakes import install as mod

    if _supervisor_kind() != "launchctl":
        pytest.skip("launchd probe is only meaningful on darwin with launchctl")

    monkeypatch.setattr(mod, "_launchctl", _FakeLaunchctl({"print": (1, "")}))
    _reachable(mod, monkeypatch)

    state = mod.supervisor_state(Path("/anywhere"))
    assert (state.loaded, state.running) == (False, False)


def test_unparseable_output_fails_safe_as_running() -> None:
    """Fail SAFE, because the two errors do not cost the same.

    Reading an unknown state as stopped would kickstart a healthy supervisor
    on every persist, forever. Reading it as running only forgoes a repair.
    """
    from local_operator.wakes.install import _parse_supervisor_state

    running, pid, _detail = _parse_supervisor_state("gui/501/x = {\n\tsomething = else\n}\n")

    assert running is True
    assert pid is None


def test_a_stopped_supervisor_is_kickstarted_rather_than_called_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair-on-demand: the hook must FIX a loaded-but-dead supervisor.

    Reaching this state used to end in "already installed" with nothing
    running. The plist here is already correct, so the only thing that can
    make the wake fire again is the restart.
    """
    from local_operator.wakes import install as mod

    if _supervisor_kind() != "launchctl":
        pytest.skip("launchd repair is only meaningful on darwin with launchctl")

    config = tmp_path / "config"
    # Stand in for the real home so both guards pass; `_launchctl` is faked, so
    # nothing reaches a real launchd domain.
    monkeypatch.setattr(mod, "_launchd_is_addressable", lambda: True)
    monkeypatch.setattr(mod, "_config_lives_in_real_home", lambda _config: True)
    plist = tmp_path / "agents" / f"{LABEL}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(plistlib.dumps(render_plist(config)))
    monkeypatch.setattr(mod, "plist_path", lambda: plist)
    fake = _FakeLaunchctl({"print": (0, _PRINT_EXITED), "kickstart": (0, "")})
    monkeypatch.setattr(mod, "_launchctl", fake)

    outcome = mod.ensure_supervisor_installed(config)

    assert outcome.installed is True
    assert "restarted" in outcome.reason
    assert any(
        call[0] == "kickstart" for call in fake.calls
    ), f"the stopped supervisor was not restarted; launchctl calls were {fake.calls}"


def test_a_running_supervisor_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotence still holds: the hook runs on EVERY persist.

    Kickstarting a healthy supervisor on every wake write would restart it
    constantly and interrupt the very sweeps it is meant to run.
    """
    from local_operator.wakes import install as mod

    if _supervisor_kind() != "launchctl":
        pytest.skip("launchd repair is only meaningful on darwin with launchctl")

    config = tmp_path / "config"
    monkeypatch.setattr(mod, "_launchd_is_addressable", lambda: True)
    monkeypatch.setattr(mod, "_config_lives_in_real_home", lambda _config: True)
    plist = tmp_path / "agents" / f"{LABEL}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(plistlib.dumps(render_plist(config)))
    monkeypatch.setattr(mod, "plist_path", lambda: plist)
    fake = _FakeLaunchctl({"print": (0, _PRINT_RUNNING)})
    monkeypatch.setattr(mod, "_launchctl", fake)

    outcome = mod.ensure_supervisor_installed(config)

    assert outcome.reason == "already installed"
    assert [call[0] for call in fake.calls] == [
        "print"
    ], f"a healthy supervisor was disturbed; launchctl calls were {fake.calls}"


@pytest.mark.skipif(sys.platform != "darwin", reason="LaunchAgent plist is macOS-only")
def test_the_plist_carries_the_bounded_self_heal(tmp_path: Path) -> None:
    """``StartInterval`` SOFTENS the self-retirement invariant, deliberately.

    "A machine with no wakes left runs no supervisor at all" is now "runs it
    briefly every 15 minutes, where it reads an empty index and exits again".
    That is bought knowingly: every other repair path needs something to call
    it, and the failure being fixed is exactly the one where nothing is left
    running to make that call.

    Measured on a scratch label (20 s job, 5 s interval): launchd started runs
    at +0/+25/+51 s — it does NOT start a second instance while one runs, and
    it DOES re-run a job that exited 0.
    """
    from local_operator.wakes.install import SELF_HEAL_INTERVAL_S

    plan = render_plist(tmp_path)

    assert plan["StartInterval"] == SELF_HEAL_INTERVAL_S
    # Must stay under the shortest realistic wake cadence, or a resurrected
    # supervisor misses the occurrence it was resurrected for.
    assert SELF_HEAL_INTERVAL_S <= 900
    # The retirement key stays: the self-heal bounds the outage, it does not
    # replace "a finished supervisor stays down".
    assert plan["KeepAlive"] == {"SuccessfulExit": False}


def test_a_store_launchd_cannot_supervise_is_reported_as_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ISOLATED-RUN LIE, closed at the probe.

    `launchctl` has no sandbox: it answers about the calling user's live
    session whatever store was asked about. During validation an isolated run
    (redirected HOME, config dir in a tmpdir) reported
    `running (pid 47545)` — the operator's REAL LaunchAgent, supervising the
    operator's REAL store. That is the same class of wrong answer this module
    exists to remove, so the probe applies the installer's own two guards and
    reports a distinct third state rather than someone else's process.
    """
    from local_operator.wakes import install as mod

    if _supervisor_kind() != "launchctl":
        pytest.skip("launchd scoping is only meaningful on darwin with launchctl")

    fake = _FakeLaunchctl({"print": (0, _PRINT_RUNNING)})
    monkeypatch.setattr(mod, "_launchctl", fake)

    # A config dir outside the real home: the sandbox shape exactly.
    state = mod.supervisor_state(tmp_path / "sandbox-cfg")

    assert state.verifiable is False
    assert state.running is False, "an isolated store was told about another store's supervisor"
    assert state.pid is None
    assert fake.calls == [], "launchd was addressed on behalf of a store it cannot supervise"


def test_a_repeat_install_in_a_foreign_store_does_not_claim_to_be_installed(
    redirected_home: Path, tmp_path: Path
) -> None:
    """Round 1 (Q1): three surfaces must not disagree about one store.

    Under a redirected home the FIRST `wake create` says "plist written;
    launchd not addressable from here" and `wake status` says "cannot be
    verified for this store" — but the second create said "already installed",
    which reads as "a supervisor is in place" for a store nothing supervises.
    The reason string is the only feedback `wake create` gives, and this PR's
    whole thesis is that "installed" stops meaning "a file exists".
    """
    config = tmp_path / "config"

    first = ensure_supervisor_installed(config)
    second = ensure_supervisor_installed(config)  # same store, plist now matches

    if _supervisor_kind() != "launchctl":
        assert second.reason == UNSUPPORTED_REASON
        return

    assert second.installed is False, "a store launchd cannot reach reported as installed"
    assert "not addressable" in second.reason, second.reason
    # Both calls answer in the same vocabulary; only the tense differs.
    assert "not addressable" in first.reason, first.reason


# ---------------------------------------------------------------------------
# The Linux and Windows arms (A9/A1/A8)
#
# The defect these exist for is not a crash but a SILENCE: on Linux and Windows
# `ensure_supervisor_installed` returned
# ``installed=False, "no supervisor installer for this platform"``, the wake
# persisted, and it then fired only when a human next opened that session — the
# exact case `lop wake` exists for. macOS was unaffected, so nothing caught it.
#
# The fakes below stand in for `systemctl --user` and `schtasks`. Whether a REAL
# systemd accepts these units was settled separately, on systemd 255 in an
# Ubuntu 24.04 container: the unit is written, `systemctl --user enable --now`
# starts it, and the timer re-runs it (see the PR's evidence section).
# ---------------------------------------------------------------------------


class _FakeSystemctl:
    """Records every ``systemctl --user`` argv and replays canned results."""

    def __init__(self, results: dict[str, tuple[int, str]] | None = None) -> None:
        self._results = results or {}
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str, **kwargs: object):  # noqa: ANN204
        import subprocess

        self.calls.append(args)
        code, stdout = self._results.get(args[0], (0, ""))
        return subprocess.CompletedProcess(list(args), code, stdout, "")

    @property
    def verbs(self) -> list[str]:
        return [call[0] for call in self.calls]


def _systemd(
    monkeypatch: pytest.MonkeyPatch,
    *,
    unit_addressable: bool = True,
    store_in_real_home: bool = True,
) -> _FakeSystemctl:
    """Point the installer at Linux/systemd with fakes for every process reach.

    ``unit_addressable`` (does a ``systemctl --user`` call here address the REAL
    user manager?) and ``store_in_real_home`` (does the supervised store outlive
    this process?) are the two INDEPENDENT inputs the write guard decides on,
    and they are separate parameters because the defect they guard against lives
    in the mixed corner: a real HOME with a store outside it. This fixture used
    to set both from ONE ``addressable`` argument, so only ``True/True`` and
    ``False/False`` could be expressed and the mixed case — the one that
    overwrote an operator's live systemd unit — was untestable by construction.
    """
    from local_operator import supervisors
    from local_operator.wakes import install as mod

    fake = _FakeSystemctl()
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(supervisors, "systemctl_user", fake)
    monkeypatch.setattr(supervisors, "enable_linger", lambda: True)
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda _unit: unit_addressable)
    monkeypatch.setattr(mod, "_config_lives_in_real_home", lambda _config: store_in_real_home)
    monkeypatch.setattr(supervisors, "systemd_version", lambda: 255)
    return fake


def test_the_linux_arm_installs_a_unit_a_timer_and_enables_the_timer(
    redirected_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The deliverable: a wake armed on Linux is supervised when no TUI is open."""
    from local_operator.wakes import install as mod

    fake = _systemd(monkeypatch)
    store = tmp_path / "config"

    outcome = mod.ensure_supervisor_installed(store)

    assert outcome.installed is True, outcome.reason
    unit = mod.plist_path()
    assert unit.name == mod.SYSTEMD_UNIT
    text = unit.read_text(encoding="utf-8")
    assert "ExecStart=" in text and "local_operator.wakes.supervisor" in text
    assert f"LOCAL_OPERATOR_CONFIG_DIR={store}" in text
    assert "Restart=on-failure" in text
    # The log the other two platforms' supervisors create, which systemd only
    # creates when it is new enough to understand `append:`.
    assert "StandardOutput=append:" in text
    timer = unit.with_name(mod.SYSTEMD_TIMER)
    assert timer.exists()
    assert f"OnUnitInactiveSec={mod.SELF_HEAL_INTERVAL_S}s" in timer.read_text(encoding="utf-8")
    # The TIMER is enabled so the supervisor re-runs itself, and the SERVICE is
    # enabled for start-at-login — enabling a timer does not enable the unit it
    # activates, so both are needed for the launchd pair being reproduced
    # (`RunAtLoad` + `StartInterval`).
    assert ("enable", "--now", mod.SYSTEMD_TIMER) in fake.calls
    assert ("enable", mod.SYSTEMD_UNIT) in fake.calls
    # And started NOW: the wake that triggered this install is already due.
    assert ("start", mod.SYSTEMD_UNIT) in fake.calls
    assert ("daemon-reload",) in fake.calls


def test_the_linux_arm_refuses_to_load_a_unit_from_a_redirected_home(
    redirected_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file half runs; the half that reaches a LIVE user manager refuses.

    ``systemctl --user`` addresses the calling user's instance whatever ``$HOME``
    says, so without this an isolated run would enable a real unit pointed at a
    store that is deleted when the test ends.
    """
    from local_operator.wakes import install as mod

    fake = _systemd(monkeypatch, unit_addressable=False, store_in_real_home=False)

    outcome = mod.ensure_supervisor_installed(tmp_path / "config")

    assert outcome.installed is False
    assert "not addressable" in outcome.reason
    assert mod.plist_path().exists(), "the unit file half must still be testable"
    assert fake.calls == [], "the user manager was addressed from a redirected home"


def test_the_linux_arm_refuses_a_real_unit_path_with_a_store_outside_the_home(
    redirected_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mix the old fixture could not express: real manager, sandbox store.

    ``_systemd``'s two flags are independent on purpose, and this is the corner
    they were collapsed over: ``unit_addressable`` True (the unit path is the
    real ``~/.config/systemd/user``) with ``store_in_real_home`` False. The
    write guard used to fold the store test into ``addressable``, which made its
    own refusal — ``addressable and not _config_lives_in_real_home`` —
    unsatisfiable, so this exact input overwrote an operator's live unit with
    one pointed at a store that dies with the sandbox (reviewer A A1).

    Asserted on the outcome and on the file NOT being written, which is the
    damage; the predicate alone would pass either way.
    """
    from local_operator.wakes import install as mod

    fake = _systemd(monkeypatch, unit_addressable=True, store_in_real_home=False)

    outcome = mod.ensure_supervisor_installed(tmp_path / "sandbox-store")

    assert outcome.installed is False
    assert "outside the real home" in outcome.reason
    unit = mod.plist_path()
    assert not unit.exists(), "a sandbox store was written into the real systemd user dir"
    assert not unit.with_name(mod.SYSTEMD_TIMER).exists()
    assert fake.calls == [], "the real user manager was addressed for a sandbox store"


def test_a_repeat_install_is_idempotent_by_content(
    redirected_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second write of the same unit must not bounce a healthy supervisor."""
    from local_operator.wakes import install as mod

    store = tmp_path / "config"
    fake = _systemd(monkeypatch)
    monkeypatch.setattr(
        mod,
        "_systemd_supervisor_state",
        lambda _config: mod.SupervisorState(loaded=True, running=True, pid=4321, detail="active"),
    )

    mod.ensure_supervisor_installed(store)
    fake.calls.clear()
    second = mod.ensure_supervisor_installed(store)

    assert second.installed is True
    assert second.reason == "already installed"
    assert fake.calls == [], "an unchanged, running unit was restarted"


def test_a_loaded_but_dead_supervisor_is_started_again(
    redirected_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The self-retirement shape: unit correct, systemd knows it, nothing runs.

    This is the Linux half of the permanent-miss bug — the supervisor exits 0 on
    an empty index, so "already installed" on a dead unit is exactly the state
    that let armed wakes sit unfired.
    """
    from local_operator.wakes import install as mod

    store = tmp_path / "config"
    fake = _systemd(monkeypatch)
    monkeypatch.setattr(
        mod,
        "_systemd_supervisor_state",
        lambda _config: mod.SupervisorState(loaded=True, running=False, detail="inactive/dead"),
    )

    mod.ensure_supervisor_installed(store)
    fake.calls.clear()
    outcome = mod.ensure_supervisor_installed(store)

    assert outcome.installed is True
    assert outcome.reason == "restarted a stopped supervisor"
    assert ("start", mod.SYSTEMD_UNIT) in fake.calls


def test_systemctl_refusing_to_enable_is_reported_with_its_own_words(
    redirected_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Not a traceback, and not a claimed success either."""
    from local_operator.wakes import install as mod

    fake = _systemd(monkeypatch)

    def failing(*args: str, **kwargs: object):  # noqa: ANN202
        if args[0] == "enable":
            import subprocess

            return subprocess.CompletedProcess(
                list(args), 1, "", "Failed to connect to bus: No medium found"
            )
        return fake(*args, **kwargs)  # type: ignore[arg-type]

    from local_operator import supervisors

    monkeypatch.setattr(supervisors, "systemctl_user", failing)

    outcome = mod.ensure_supervisor_installed(tmp_path / "config")

    assert outcome.installed is False
    assert "loginctl enable-linger" in outcome.reason, "the remedy must be named"


def test_uninstall_disables_the_timer_before_the_service(
    redirected_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling the service alone leaves a timer that starts it again."""
    from local_operator.wakes import install as mod

    fake = _systemd(monkeypatch)

    mod.uninstall()

    assert fake.calls.index(("disable", "--now", mod.SYSTEMD_TIMER)) < fake.calls.index(
        ("disable", "--now", mod.SYSTEMD_UNIT)
    )


def test_the_windows_uninstall_ends_the_task_before_deleting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A2: the delete deregisters without stopping the supervisor.

    ``lop wake uninstall`` on Windows used to leave a supervisor still firing
    wakes from a store the caller believed it had removed — the other two
    platforms stop the job as part of the uninstall (``bootout``,
    ``disable --now``) and Windows needs its own ``/End``.
    """
    import subprocess

    from local_operator import supervisors
    from local_operator.wakes import install as mod

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(mod, "ambient_config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: calls.append(args)
        or subprocess.CompletedProcess(list(args), 0, "", ""),
    )
    monkeypatch.setattr(
        supervisors,
        "delete_task",
        lambda name: calls.append(("/Delete", "/TN", name)) or (True, "deleted"),
    )
    record = mod.task_record_path(tmp_path)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("<Task/>", encoding="utf-8")

    outcome = mod.uninstall()

    assert outcome.installed is False and outcome.reason == "uninstalled", outcome
    assert [call[0] for call in calls] == ["/End", "/Delete"], calls
    assert calls[0] == tuple(supervisors.task_end_args(mod.TASK_NAME))
    assert not record.exists()


def test_a_refused_windows_uninstall_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refusal must not read as removed: the supervisor is still registered."""
    import subprocess

    from local_operator import supervisors
    from local_operator.wakes import install as mod

    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(mod, "ambient_config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: subprocess.CompletedProcess(list(args), 0, "", ""),
    )
    monkeypatch.setattr(
        supervisors, "delete_task", lambda _name: (False, "ERROR: Access is denied.")
    )

    outcome = mod.uninstall()

    assert outcome.installed is False
    assert "Access is denied." in outcome.reason


def test_the_linux_unit_quotes_paths_systemd_would_otherwise_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A5, measured on systemd 255: unquoted values are silently truncated.

    ``Environment=LOCAL_OPERATOR_CONFIG_DIR=/home/a b/store`` made systemd log
    ``Invalid environment assignment, ignoring: b/store`` and keep the variable
    at its first word — so the supervisor supervised a DIFFERENT store than the
    one the wake was armed in, with nothing on screen saying so. A space in the
    interpreter path was worse: ``Command /home/a is not executable``, a unit
    that can never start.
    """
    from local_operator.wakes import install as mod

    monkeypatch.setattr(mod.sys, "executable", "/home/a b/python3")
    monkeypatch.setattr(mod.procname, "supervised_image", lambda: None)
    store = tmp_path / "a b" / "100%" / "store"

    text = mod.render_systemd(store)

    # ``%`` is doubled because systemd expands unit specifiers even inside a
    # quoted string; ``%%`` is the escape, and a single ``%`` here would be the
    # silent-mangling half of the same defect.
    assert f'Environment="LOCAL_OPERATOR_CONFIG_DIR={store}"'.replace("100%", "100%%") in text
    assert 'ExecStart="/home/a b/python3" -m local_operator.wakes.supervisor' in text


def test_the_systemd_probe_reads_state_rather_than_an_exit_code() -> None:
    """Pure parse, against recorded ``systemctl show`` output."""
    from local_operator.wakes.install import _parse_systemd_state

    exited = _parse_systemd_state(
        "LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0"
    )
    assert exited.loaded is True and exited.running is False and exited.pid is None

    live = _parse_systemd_state(
        "LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=8123"
    )
    assert live.running is True and live.pid == 8123

    absent = _parse_systemd_state(
        "LoadState=not-found\nActiveState=inactive\nSubState=dead\nMainPID=0"
    )
    assert absent.loaded is False and absent.running is False
    assert "not loaded" in absent.detail


def test_an_auto_restarting_unit_is_not_a_running_supervisor() -> None:
    """A crash loop must not read as "running" on the status surface.

    MEASURED on real systemd 255 (Ubuntu 24.04 container): a unit whose process
    keeps exiting reports ``activating/auto-restart``. systemd is retrying, so
    nothing is serving the wakes — and a `start` on it is a no-op, which is why
    classifying it as stopped is safe as well as honest.
    """
    from local_operator.wakes.install import _parse_systemd_state

    crashing = _parse_systemd_state(
        "LoadState=loaded\nActiveState=activating\nSubState=auto-restart\nMainPID=162"
    )

    assert crashing.loaded is True
    assert crashing.running is False
    assert "auto-restart" in crashing.detail


def test_the_systemd_probe_fails_safe_on_vocabulary_it_does_not_know() -> None:
    """Unknown vocabulary is NOT read as stopped.

    Reading an unknown word as stopped would restart a healthy supervisor on
    every wake write; reading it as running only skips a repair the timer would
    have made anyway, so the asymmetry is deliberate.
    """
    from local_operator.wakes.install import _parse_systemd_state

    unknown = _parse_systemd_state(
        "LoadState=loaded\nActiveState=reloading\nSubState=reload\nMainPID=9"
    )
    assert unknown.running is True


def test_an_unreachable_user_manager_is_not_verifiable_rather_than_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ "Cannot ask" and "nothing installed" are different answers to the user."""
    from local_operator import supervisors
    from local_operator.wakes import install as mod

    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda _unit: True)
    monkeypatch.setattr(mod, "_config_lives_in_real_home", lambda _config: True)

    def no_bus(*args: str, **kwargs: object):  # noqa: ANN202
        import subprocess

        return subprocess.CompletedProcess(
            list(args), 1, "", "Failed to connect to bus: No medium found"
        )

    monkeypatch.setattr(supervisors, "systemctl_user", no_bus)
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")

    state = mod.supervisor_state(tmp_path / "config")

    assert state.verifiable is False
    assert "loginctl enable-linger" in state.detail
    # And the isolated-store answer is unchanged: not verifiable, no pid.
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda _unit: False)
    assert mod.supervisor_state(tmp_path / "config").verifiable is False


def test_the_windows_arm_registers_and_starts_a_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A8: nothing in this repo attempted Windows task scheduling before this."""
    import subprocess

    from local_operator import supervisors
    from local_operator.wakes import install as mod

    created: list[tuple[str, str]] = []
    runs: list[tuple[str, ...]] = []

    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(supervisors, "task_scheduler_is_addressable", lambda _c: True)
    monkeypatch.setattr(supervisors, "task_state", lambda _name: (False, False, "not registered"))
    monkeypatch.setattr(mod, "_config_lives_in_real_home", lambda _c: True)

    def fake_create(name: str, xml: str) -> tuple[bool, str]:
        created.append((name, xml))
        return True, "registered"

    monkeypatch.setattr(supervisors, "create_task", fake_create)
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kwargs: runs.append(args)
        or subprocess.CompletedProcess(list(args), 0, "", ""),
    )

    store = tmp_path / "config"
    outcome = mod.ensure_supervisor_installed(store)

    assert outcome.installed is True, outcome.reason
    name, xml = created[0]
    assert name == mod.TASK_NAME
    assert "local_operator.wakes.supervisor" in xml
    assert "PT15M" in xml, "the self-heal interval is what makes an unattended wake fire"
    assert str(store) in xml, "the supervised store is part of the contract"
    assert ("/Run", "/TN", mod.TASK_NAME) in runs
    assert mod.task_record_path(store).exists()


def test_the_windows_arm_reports_schtasks_refusal_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No Windows host here, so a wrong guess must fail LOUDLY with its own words."""
    from local_operator import supervisors
    from local_operator.wakes import install as mod

    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(supervisors, "task_scheduler_is_addressable", lambda _c: True)
    monkeypatch.setattr(
        supervisors, "create_task", lambda _n, _x: (False, "ERROR: Access is denied.")
    )

    outcome = mod.ensure_supervisor_installed(tmp_path / "config")

    assert outcome.installed is False
    assert "Access is denied." in outcome.reason


def test_the_windows_arm_refuses_a_store_outside_the_real_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator import supervisors
    from local_operator.wakes import install as mod

    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(supervisors, "task_scheduler_is_addressable", lambda _c: False)
    called: list[str] = []
    monkeypatch.setattr(supervisors, "create_task", lambda n, x: called.append(n) or (True, ""))

    outcome = mod.ensure_supervisor_installed(tmp_path / "config")

    assert outcome.installed is False
    assert "outside the real profile" in outcome.reason
    assert called == [], "a task was registered for a sandbox store"
