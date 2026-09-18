"""The platform process primitives, and every caller that used to own a copy.

WHY THESE TESTS EXIST (cross-platform work, 2026-09-18). ``os.kill(pid, 0)`` is
a liveness probe on POSIX and a KILL on Windows — CPython's ``os_kill_impl``
takes the ``GenerateConsoleCtrlEvent`` path only for
``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` and otherwise calls
``TerminateProcess(handle, sig)``. Three modules had written that probe at the
call site, so on Windows the probe TERMINATED the process it was asked about and
then reported it alive: silent to the caller, destructive to the target, on a
path (``registry.pid_alive``) that runs on every ``lop`` invocation.

The discriminating assertions here are the ones that fail on the pre-fix code:
``os.kill`` must not be reached at all when the platform is win32, and the
Windows answer must come from ``OpenProcess`` (the primitive
``session_lease`` already used) rather than from a signal.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator import procstate
from local_operator.browser_bridge import state as bridge_state
from local_operator.mobile import resources as mobile_resources
from local_operator.session.runtime import registry


def _forbid_os_kill(monkeypatch: pytest.MonkeyPatch, killed: list[tuple[int, int]]) -> None:
    """Make any ``os.kill`` call both recorded and fatal.

    Recorded as well as fatal because the failure mode being pinned is not "an
    exception escaped" but "the probe ran at all": an assertion on the empty
    list is what proves the branch was taken.
    """

    def _kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        raise AssertionError("os.kill must not run on win32")

    monkeypatch.setattr(os, "kill", _kill)


@pytest.fixture
def as_windows(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Pretend this process is on win32, and refuse to let os.kill be called.

    ``procstate._PLATFORM`` rather than ``os.name``: it is the ONE name every
    platform branch in the package reads, and patching ``os.name`` process-wide
    makes ``pathlib`` refuse to build a path on the host running the test.
    """
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(procstate, "_PLATFORM", "win32")
    _forbid_os_kill(monkeypatch, killed)
    return killed


# ---------------------------------------------------------------------------
# pid_liveness / pid_alive
# ---------------------------------------------------------------------------


def test_pid_alive_asks_the_kernel_on_windows_and_never_signals(
    as_windows: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(procstate, "_windows_liveness", lambda pid: True)
    assert procstate.pid_alive(4242) is True
    assert procstate.pid_liveness(4242) is True
    # THE WHOLE POINT: a live pid is reported alive WITHOUT being terminated.
    assert as_windows == []


def test_a_gone_windows_pid_is_dead_not_alive(
    as_windows: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(procstate, "_windows_liveness", lambda pid: False)
    assert procstate.pid_alive(4242) is False
    assert as_windows == []


def test_an_unprovable_windows_answer_fails_closed(
    as_windows: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Access denial is doubt, and doubt may never read as "gone".

    Calling a live owner dead is what lets a second writer take a transcript a
    working runtime is still appending to; the reverse only delays a recovery.
    """
    monkeypatch.setattr(procstate, "_windows_liveness", lambda pid: None)
    assert procstate.pid_liveness(4242) is None
    assert procstate.pid_alive(4242) is True
    assert as_windows == []


def test_pid_liveness_posix_answers_about_a_real_process() -> None:
    assert procstate.pid_liveness(os.getpid()) is True
    assert procstate.pid_alive(os.getpid()) is True
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert procstate.pid_liveness(child.pid) is True
    finally:
        child.kill()
        child.wait(timeout=10)


def test_a_pid_with_no_process_reads_as_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    """ESRCH is the one POSIX answer that means "gone".

    Injected rather than raced: reaped pids are reusable, and on a host running
    a fleet of sessions a just-freed pid can belong to somebody else by the time
    the probe runs, which would make the test flaky rather than wrong.
    """

    def _kill(pid: int, sig: int) -> None:
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "kill", _kill)
    assert procstate.pid_liveness(1234) is False
    assert procstate.pid_alive(1234) is False


def test_another_account_s_process_reads_as_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """EPERM is "exists but not ours", which for every caller is "alive"."""

    def _kill(pid: int, sig: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "kill", _kill)
    assert procstate.pid_liveness(1) is True
    assert procstate.pid_alive(1) is True


def test_an_unexpected_posix_probe_error_is_uncertain_not_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _kill(pid: int, sig: int) -> None:
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(os, "kill", _kill)
    assert procstate.pid_liveness(1234) is None
    assert procstate.pid_alive(1234) is True


# ---------------------------------------------------------------------------
# Every caller shares it — none of them may grow its own os.kill again
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(lambda pid: registry.pid_alive(pid), id="registry.pid_alive"),
        pytest.param(bridge_state.pid_alive, id="browser_bridge.state.pid_alive"),
        pytest.param(mobile_resources._pid_exists, id="mobile.resources._pid_exists"),
    ],
)
def test_every_shared_liveness_probe_avoids_os_kill_on_windows(
    probe: Callable[[int], bool], as_windows: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(procstate, "_windows_liveness", lambda pid: True)
    assert probe(4242) is True
    assert as_windows == []


def test_the_fourth_caller_is_the_lease_and_it_keeps_its_tri_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``session_lease`` distinguishes "uncertain" from "dead", and must keep doing so.

    Its callers move a sole-writer claim only from a holder that is PROVEN dead,
    so collapsing doubt into either answer would either strand a transcript or
    take one that is still being written.
    """
    from local_operator import session_lease

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(procstate, "_PLATFORM", "win32")
    _forbid_os_kill(monkeypatch, killed)

    monkeypatch.setattr(procstate, "_windows_liveness", lambda pid: True)
    assert session_lease._pid_state(4242) == "live"
    monkeypatch.setattr(procstate, "_windows_liveness", lambda pid: False)
    assert session_lease._pid_state(4242) == "dead"
    monkeypatch.setattr(procstate, "_windows_liveness", lambda pid: None)
    assert session_lease._pid_state(4242) == "uncertain"
    assert killed == []


@pytest.mark.skipif(
    os.name != "posix", reason="fork is POSIX-only; the Windows lease answer is pinned above"
)
def test_the_lease_still_reports_a_zombie_holder_as_dead() -> None:
    """The POSIX half of the same probe, unchanged: a corpse is not a writer.

    ``signal 0`` succeeds against an exited-but-unreaped child, so signal 0
    alone would call a killed owner live FOREVER — and the lease moves a claim
    only from a holder it can PROVE dead. ``fork`` rather than ``Popen`` because
    the point is to leave the child unreaped: ``Popen.poll`` reaps.
    """
    from local_operator import session_lease

    with warnings.catch_warnings():
        # Python 3.14 warns about `fork` in a threaded process, and pytest runs
        # with xdist workers. The child does nothing but `_exit`, so there is no
        # inherited lock or half-written buffer to corrupt: the warning is about
        # the general shape, not this one.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:  # pragma: no cover - the child exits immediately
        os._exit(0)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if session_lease._pid_state(pid) == "dead":
                break
            time.sleep(0.02)
        # Existence says alive (it is still on the process table); the zombie
        # question is what says the writer is gone.
        assert session_lease._pid_state(pid, check_zombie=False) == "live"
        assert session_lease._pid_state(pid) == "dead"
    finally:
        os.waitpid(pid, 0)


# ---------------------------------------------------------------------------
# Detachment
# ---------------------------------------------------------------------------


def test_detached_popen_kwargs_is_setsid_on_posix() -> None:
    """POSIX keeps the exact kwarg it always passed — no macOS behaviour change."""
    assert procstate.detached_popen_kwargs() == {"start_new_session": True}


def test_detached_popen_kwargs_really_detaches_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kwarg that says what Windows DOES, not one it silently ignores.

    ``start_new_session`` is documented "(POSIX only)" and the Windows
    ``_execute_child`` parameter is literally named ``unused_start_new_session``
    — no error, no effect, so a "detached" child kept the parent's console and
    both Ctrl-C and a console close reached it.
    """
    monkeypatch.setattr(procstate, "_PLATFORM", "win32")
    kwargs = procstate.detached_popen_kwargs()
    assert "start_new_session" not in kwargs
    flags = int(kwargs["creationflags"])  # type: ignore[arg-type]
    assert flags & 0x00000008, "DETACHED_PROCESS: no console inherited at all"
    assert flags & 0x00000200, "CREATE_NEW_PROCESS_GROUP: its own group"


def test_detached_popen_kwargs_are_accepted_by_popen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The kwargs are real Popen arguments on this platform, not just a dict."""
    kwargs = dict(procstate.detached_popen_kwargs())
    child = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,  # type: ignore[arg-type]
    )
    assert child.wait(timeout=30) == 0


# ---------------------------------------------------------------------------
# terminate_process_tree
# ---------------------------------------------------------------------------


def test_terminate_process_tree_kills_the_group_and_descendants(tmp_path: Path) -> None:
    """A shell command's children die with it — the group IS the point.

    Written as a real spawn rather than a mock: on POSIX the whole value of
    routing through this helper is that it still signals the GROUP
    (``killpg``), so a descendant of the shell must be gone too.
    """
    marker = tmp_path / "child.pid"
    child = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        ["/bin/sh", "-c", f"(sleep 30 & echo $! > {marker}); sleep 30"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 20.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "the command never started its descendant"
        assert procstate.terminate_process_tree(child.pid, force=True) is True
        assert child.wait(timeout=10) == -signal.SIGKILL
        descendant = int(marker.read_text().strip())
        with pytest.raises(ProcessLookupError):
            os.kill(descendant, 0)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_terminate_process_tree_on_a_dead_pid_answers_false() -> None:
    """False means "nothing was there to stop", which is not a failure."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert procstate.terminate_process_tree(child.pid, force=True) is False
    assert procstate.terminate_process_tree(0, force=True) is False


def test_terminate_process_tree_signals_by_pid_when_there_is_no_group() -> None:
    """A child spawned WITHOUT a new session must not cost the CALLER its group.

    This is the dangerous shape: ``os.getpgid(child)`` answers with the group
    the caller itself belongs to, so an unconditional ``killpg`` here would
    signal the whole test runner. The helper must notice the pid is not a group
    leader and signal the pid alone.
    """
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert os.getpgid(child.pid) == os.getpgid(os.getpid())  # not a leader
        assert procstate.terminate_process_tree(child.pid, force=False) is True
        assert child.wait(timeout=10) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


# ---------------------------------------------------------------------------
# Loop signal handlers
# ---------------------------------------------------------------------------


def test_supports_loop_signals_is_true_where_it_is_probed() -> None:
    """POSIX keeps the loop's own handler; the fallback is for the platform that has none."""
    assert procstate.supports_loop_signals() is True


def test_supports_loop_signals_is_false_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(procstate, "_PLATFORM", "win32")
    assert procstate.supports_loop_signals() is False


def test_hard_kill_signal_is_sigkill_here_and_none_where_it_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert procstate.hard_kill_signal() == signal.SIGKILL
    # "Availability: Unix" for signal.SIGKILL: the constant is simply absent
    # there, so the ladder must ask rather than name it.
    monkeypatch.delattr(signal, "SIGKILL")
    assert procstate.hard_kill_signal() is None


class _RefusingLoop:
    """The Windows Proactor loop's shape: the base stub plus real scheduling."""

    def __init__(self) -> None:
        self.scheduled: list[Callable[[], None]] = []

    def add_signal_handler(self, sig: int, callback: Callable[[], None]) -> None:
        raise NotImplementedError

    def call_soon_threadsafe(self, callback: Callable[[], None]) -> None:
        self.scheduled.append(callback)


class _AcceptingLoop(_RefusingLoop):
    def __init__(self) -> None:
        super().__init__()
        self.installed: dict[int, Callable[[], None]] = {}

    def add_signal_handler(self, sig: int, callback: Callable[[], None]) -> None:
        self.installed[sig] = callback


def test_install_loop_signal_handlers_uses_the_loop_where_it_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signal, "signal", lambda *_: pytest.fail("no fallback needed here"))
    loop = _AcceptingLoop()
    handlers: dict[int, Callable[[], None]] = {signal.SIGTERM: lambda: None}
    assert procstate.install_loop_signal_handlers(loop, handlers) is True
    assert signal.SIGTERM in loop.installed


def test_install_loop_signal_handlers_degrades_to_signal_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relay/runtime boot path used to die here instead of degrading.

    ``add_signal_handler`` raises ``NotImplementedError`` on the Windows
    Proactor loop, and nothing in this package switches Windows to a selector
    policy — so an unguarded call meant a traceback at startup.
    """
    installed: dict[int, Any] = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.setdefault(sig, handler))
    loop = _RefusingLoop()
    fired: list[str] = []

    handlers: dict[int, Callable[[], None]] = {signal.SIGTERM: lambda: fired.append("stop")}
    assert procstate.install_loop_signal_handlers(loop, handlers) is True
    assert set(installed) == {signal.SIGTERM}
    # The handler runs on the main thread OUTSIDE the loop, so it must hand the
    # callback over rather than touching loop state from a signal context.
    installed[signal.SIGTERM](signal.SIGTERM, None)
    assert fired == []
    assert len(loop.scheduled) == 1
    loop.scheduled[0]()
    assert fired == ["stop"]


def test_install_loop_signal_handlers_survives_an_unhandleable_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signal this platform will not take is not a reason to fail to boot."""

    def _refuse(sig: int, handler: Any) -> None:
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "signal", _refuse)
    assert (
        procstate.install_loop_signal_handlers(_RefusingLoop(), {signal.SIGTERM: lambda: None})
        is False
    )


def test_omitting_a_signal_the_platform_lacks_is_the_callers_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SIGUSR1`` has no Windows equivalent, and a missing constant is a NameError.

    The helper takes a mapping precisely so an optional signal is filtered at
    the call site — this pins that an absent constant is not silently installed.
    """
    monkeypatch.delattr(signal, "SIGUSR1")
    assert getattr(signal, "SIGUSR1", None) is None
    handlers: dict[int, Callable[[], None]] = {signal.SIGTERM: lambda: None}
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    assert procstate.install_loop_signal_handlers(_RefusingLoop(), handlers) is True
