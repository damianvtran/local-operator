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
    UNSUPPORTED_REASON,
    ensure_supervisor_installed,
    is_supported,
    plist_path,
    render_plist,
)


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
    outcome = ensure_supervisor_installed(tmp_path / "config")

    if not is_supported():
        assert outcome.installed is False
        assert outcome.reason == UNSUPPORTED_REASON
        return

    # The file half runs in full: this is what launchd WOULD be handed.
    assert plist_path().exists()
    written = plistlib.loads(plist_path().read_bytes())
    assert written == render_plist(tmp_path / "config")
    # The process half refuses under a redirected home.
    assert outcome.installed is False
    assert "not addressable" in outcome.reason


@pytest.mark.skipif(sys.platform != "darwin", reason="LaunchAgent plist is macOS-only")
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

    from local_operator.wakes.install import _launchd_is_addressable, is_supported

    if not is_supported():
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

    if not is_supported():
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

    if not is_supported():
        pytest.skip("launchd probe is only meaningful on darwin with launchctl")

    monkeypatch.setattr(mod, "_launchctl", _FakeLaunchctl({"print": (0, _PRINT_RUNNING)}))
    _reachable(mod, monkeypatch)

    state = mod.supervisor_state(Path("/anywhere"))

    assert (state.loaded, state.running, state.pid) == (True, True, 47545)


def test_an_absent_job_is_neither_loaded_nor_running(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.wakes import install as mod

    if not is_supported():
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

    if not is_supported():
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

    if not is_supported():
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

    if not is_supported():
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

    if not is_supported():
        assert second.reason == UNSUPPORTED_REASON
        return

    assert second.installed is False, "a store launchd cannot reach reported as installed"
    assert "not addressable" in second.reason, second.reason
    # Both calls answer in the same vocabulary; only the tense differs.
    assert "not addressable" in first.reason, first.reason
