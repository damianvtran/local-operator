"""A runtime's lifetime is its own; no interface event may end a session's work.

The operator's report, verbatim: "I noticed an issue in the TUI where, when I
closed ghostty a bunch of sessions got cancelled at once ... it's unexpected
since all session runtimes are supposed to be detached from interfaces and they
should be able to run in the background when the parent process is quit".

The architecture already intends exactly that. ``process.py`` is spawned by
``launch._spawn_runtime`` with ``start_new_session=True`` (pinned here and in
``test_launch_arbitration``), the TUI is a viewer whose takeover factory raises
by construction (``cli.py``: a terminal must never win the transcript lease),
and nothing in the tree stops a runtime because a viewer went away. The gap this
file pins is the one signal a terminal teardown produces and the runtime did not
handle: ``SIGHUP``.

``SIGHUP`` carries no handler, so its default disposition applies and the
interpreter dies outright — no caused turn outcome, no lease release, no record
unpublish, not even a ``session runtime: exiting`` line. The session is then
left reading as an anonymous "cause could not be determined" cut-off, which is
indistinguishable from a crash. A runtime is spawned detached and writes its log
to a file, so losing a controlling terminal is not a reason to leave work
half-done; the runtime now ignores a HUP. There is no acceptable HUP death.

WHY THE CELLS ARE SHAPED THIS WAY.

* The child is REAL: ``launch._spawn_runtime`` is the production spawn, the
  runtime is the production module, and the assertions read the child's own log.
  A double here would pin the source of the handler, not the behaviour of the
  process.
* ``test_a_detached_runtime_survives_a_terminal_hangup`` fails on the tree
  before the handler existed (the child dies and never logs the ignore) and
  passes after it. Both directions were run; see the PR. Its assertions are
  EVENTS — the child's own ``ignoring SIGHUP`` line, then aliveness, then the
  clean SIGTERM exit line — never a sleep standing in for one.
* SIGNAL READINESS IS GATED, not assumed. A signal sent at a runtime that has
  not yet reached ``amain``'s handler block measures the spawn, not the handler.
  The sibling precedent is ``tests/unit/test_exec_mode.py``'s SIGTERM worker
  test, whose comment records this exact failure ("the child took SIGTERM before
  installing the handler, died with rc=-15"). So this test waits for the child's
  own ``SIGUSR1`` task dump (``LOP_RUNTIME_DEBUG_STACKS=1``, on by default
  since 2026-09-20 and set explicitly here, armed later in ``amain`` than the
  SIGHUP handler) before it sends
  anything.
  ``SIGUSR1``'s default disposition is fatal too, so the PROBE is made harmless
  at the source rather than in the product: this process sets ``SIGUSR1`` to
  ``SIG_IGN`` before spawning, and CPython's ``subprocess`` restores only
  ``SIGPIPE``/``SIGXFZ``/``SIGXFSZ`` to ``SIG_DFL`` in the child, so the child
  inherits the ignore and its own ``add_signal_handler`` replaces it once armed.
  That inheritance touches ``SIGUSR1`` only — never ``SIGHUP`` — so it cannot
  make the SIGHUP assertion pass on a tree that lacks the handler.

Isolation: a scratch ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR``, with every
inherited ``LOP_*``/``CMUX_*``/``HERDR_*`` variable stripped before the spawn,
because this suite is routinely run from inside an operator session whose own
values would otherwise be inherited by a real runtime (#648).
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import launch as launch_module
from local_operator.session.runtime import registry

#: The child boots a real session, so it costs seconds rather than milliseconds:
#: the marker the suite uses for a test that spawns real subprocesses.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not hasattr(signal, "SIGHUP"), reason="SIGHUP is POSIX-only, as is pty/setsid"
    ),
]

#: The session is a real one on the mock provider, so no API key and no network
#: are involved. Mirrors ``tests/e2e/test_cut_off_turns_e2e.py::_seed``.
_SESSION_ID = "sighupdetach01"

#: Long enough that a HUP storm cannot fill the log, short enough to fail loudly
#: rather than hang the worker (there is no pytest-timeout in this suite).
_WAIT_S = 30.0


def _seed(config_dir: Path) -> Path:
    """A resumable session on the mock provider, with one durable row.

    The runtime is spawned the way a viewer engages real work, so the directory
    it adopts must look like one a viewer would resume — not an empty id.
    """
    directory = config_dir / "sessions" / _SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    return directory


def _isolate(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    """A child environment that can only touch ``config_dir``.

    ``launch._spawn_runtime`` passes ``dict(os.environ)`` to the child, so the
    scrubbing has to happen on THIS process's environment. The two families are
    the ones that name a real session or window: an inherited
    ``CMUX_WORKSPACE_ID`` has already renamed the operator's live cmux
    workspaces once (#648), and an inherited ``LOP_MOBILE_CHILD_*`` makes a
    spawn adopt a session that is not the test's.
    """
    for key in tuple(os.environ):
        if key.startswith(("CMUX_", "LOP_", "HERDR_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    # The residency drain, not a convenience: the default grace is 3 s, so an
    # unviewed runtime would exit on its own and every "is it still alive"
    # assertion below would be measuring the reaper instead of the signal.
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "600")
    # The arming gate: this installs the SIGUSR1 task dump, which `amain` arms
    # after the SIGHUP handler (see the module docstring), so observing it is
    # proof the disposition under test is installed too. Set EXPLICITLY even
    # though the dump is now on by default (it stopped being opt-in on
    # 2026-09-20, see `process.amain`): this test's readiness gate must not
    # depend on a default that another change could flip back.
    monkeypatch.setenv("LOP_RUNTIME_DEBUG_STACKS", "1")
    # Defence in depth for the WORKSPACE families the loop above already
    # removed: nothing in this module may address a real pane.
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")


def _log_text(config_dir: Path) -> str:
    """The child's own log, or "" before it exists.

    ``main()`` points logging at ``log_dir()/runtime.log`` — the runtimes' own
    file, deliberately not the daemon's launchd-owned ``mobile.log`` (see
    ``paths.runtime_log_path``) — and ``log_dir()`` honours
    ``LOCAL_OPERATOR_CONFIG_DIR`` first, which is what makes the child's exit
    reason readable without touching the operator's real log.
    """
    try:
        return (config_dir / "logs" / "runtime.log").read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _capture_text(child: Any) -> str:
    """Whatever the child wrote to the stdio capture, for a failure message."""
    path = getattr(child, "lop_capture_path", None)
    if path is None:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        return ""


def _await_log(config_dir: Path, needle: str, child: Any, *, timeout: float = _WAIT_S) -> str:
    """The log text, once it contains ``needle`` — or the last text read.

    Bounded by the event (the child's own line) rather than by a wall-clock
    guess about how long the work "should" take; the deadline exists only so a
    genuine hang fails instead of blocking the worker forever. Gives up early
    when the child has exited, because then the line can never arrive.
    """
    deadline = time.monotonic() + timeout
    while True:
        text = _log_text(config_dir)
        if needle in text:
            return text
        if child.poll() is not None or time.monotonic() >= deadline:
            return text
        time.sleep(0.1)


def _wait_for_record(config_dir: Path, *, timeout: float = _WAIT_S) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == _SESSION_ID:
                return record
        time.sleep(0.05)
    raise AssertionError(f"no record for {_SESSION_ID} within {timeout}s")


def _reap(child: Any, config_dir: Path) -> None:
    """Take the child AND its group down; nothing here survives the test.

    The child is its own session and group leader (that is the contract under
    test), so a group kill is the only way to be sure a turn's own children go
    too. Never ``waitpid`` on a forked/pty child under xdist — see the precedent
    in ``tests/unit/providers/test_login_cancel_cli.py`` — but a ``Popen`` we
    spawned is ours to reap, and one we cannot reap is already gone.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        child.kill()
    try:
        child.wait(timeout=10)
    except Exception:  # noqa: BLE001 — teardown must not replace a real failure
        pass
    capture = getattr(child, "lop_capture_path", None)
    if capture is not None:
        Path(capture).unlink(missing_ok=True)
    # A record the child left behind would make the NEXT scan in this worker
    # (or a sibling test) see a dead owner as a live one.
    registry.scan(config_dir)


def test_a_detached_runtime_survives_a_terminal_hangup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant, on a real runtime: HUP is ignored, SIGTERM still works.

    Three properties, in the order they must be established:

    (a) the spawn put the child in its OWN session and group — the detachment
        contract a terminal teardown has to be unable to reach;
    (b) a real ``SIGHUP`` does not end it, and the ignore is dated by the child's
        own log line (so the assertion is "the handler ran", not "we looked a
        moment later and it happened to be alive");
    (c) ``SIGTERM`` afterwards still ends it cleanly and loudly — proof that
        ignoring a HUP did not disable the kill switch.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _seed(config_dir)
    _isolate(monkeypatch, config_dir)

    # The probe's own disposition, set on THIS process so the child inherits it.
    # CPython restores only SIGPIPE/SIGXFZ/SIGXFSZ in a child before exec, so a
    # SIG_IGN set here reaches the child as SIG_IGN, where the runtime's own
    # add_signal_handler replaces it the moment it is armed. Without this, a
    # probe that landed one moment early would kill the child it is asking.
    previous_usr1 = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    child = None
    try:
        child = launch_module._spawn_runtime(
            _SESSION_ID,
            str(config_dir),
            # The viewer's engage path for real work: the runtime owns a real
            # session directory rather than a speculatively warmed lease.
            defer_materialise=False,
        )
        pid = child.pid

        # (a) THE DETACHMENT CONTRACT. `start_new_session=True` in launch.py is
        # what makes a terminal teardown structurally unable to reach this
        # process: it has no controlling terminal and is not in the interface's
        # process group. Asserted on the real child, not on the kwarg alone
        # (test_launch_arbitration pins the kwarg; this pins its effect).
        assert os.getsid(pid) == pid, (
            f"runtime {pid} is not its own session leader: sid={os.getsid(pid)}, "
            "so an interface's teardown could signal it"
        )
        assert (
            os.getpgid(pid) == pid
        ), f"runtime {pid} is not its own group leader: pgid={os.getpgid(pid)}"

        # READINESS, by the child's own hand. The SIGUSR1 dump is armed later in
        # ``amain`` than the SIGHUP handler, so seeing it is proof that the
        # disposition under test is installed. Sent in a bounded loop because a
        # probe that arrives before arming is discarded by the inherited SIG_IGN
        # rather than queued.
        deadline = time.monotonic() + _WAIT_S
        while "state: streaming=" not in _log_text(config_dir):
            assert child.poll() is None, (
                f"the runtime exited (rc={child.returncode}) before it armed its "
                f"signal handlers:\n{_capture_text(child)}\n{_log_text(config_dir)[-2000:]}"
            )
            assert time.monotonic() < deadline, (
                "the runtime never armed (no SIGUSR1 dump in its log); the debug hook "
                f"is gone or the block never ran:\n{_log_text(config_dir)[-2000:]}"
            )
            os.kill(pid, signal.SIGUSR1)
            time.sleep(0.2)

        record = _wait_for_record(config_dir)
        assert int(record.pid) == pid, (record.pid, pid)

        # (b) THE HANGUP, for real. On the tree before the handler existed this
        # kills the child outright and the log line below never appears.
        os.kill(pid, signal.SIGHUP)
        text = _await_log(config_dir, "ignoring SIGHUP", child)
        assert "ignoring SIGHUP" in text, (
            "the runtime did not survive a SIGHUP (or did not log ignoring it): "
            f"alive={child.poll() is None} rc={child.returncode}\n"
            f"{_capture_text(child)}\n{text[-2000:]}"
        )
        # ...and it stays alive over a window rather than merely at one instant:
        # the latch is a handler, not a race won against the default
        # disposition. Bounded by the deadline, and the assertion is liveness.
        window_end = time.monotonic() + 3.0
        while time.monotonic() < window_end:
            assert child.poll() is None, (
                f"the runtime died {window_end - time.monotonic():.2f}s after a SIGHUP "
                f"(rc={child.returncode})"
            )
            time.sleep(0.1)
        # A HUP storm must not be able to spam the log either: the ignore is a
        # latch, so N HUPs produce exactly one line.
        for _ in range(5):
            os.kill(pid, signal.SIGHUP)
        time.sleep(0.5)
        assert _log_text(config_dir).count("ignoring SIGHUP") == 1, (
            "the SIGHUP ignore is not latched: "
            f"{_log_text(config_dir).count('ignoring SIGHUP')} lines"
        )

        # (c) THE KILL SWITCH STILL WORKS, and it still says so. This is the
        # other half of the guarantee: a runtime that ignored everything would
        # pass (b) and be unfixable, so the clean logged exit is asserted, not
        # merely the death.
        os.kill(pid, signal.SIGTERM)
        try:
            child.wait(timeout=_WAIT_S)
        except Exception:  # noqa: BLE001 — re-raised below as the assertion
            pass
        assert child.returncode == 0, (
            f"SIGTERM did not produce the clean exit (rc={child.returncode})\n"
            f"{_capture_text(child)}\n{_log_text(config_dir)[-2000:]}"
        )
        assert "session runtime: exiting (SIGTERM" in _log_text(config_dir), (
            "the SIGTERM exit left no line naming its trigger: the session would read "
            f"as an anonymous cut-off\n{_log_text(config_dir)[-2000:]}"
        )
    finally:
        if child is not None:
            _reap(child, config_dir)
        signal.signal(signal.SIGUSR1, previous_usr1)


# ---------------------------------------------------------------------------
# Q-XH-5: a spawned runtime that exits is REAPED, not left as a corpse
# ---------------------------------------------------------------------------


def _await_collected(child: Any, *, timeout: float = _WAIT_S) -> bool:
    """Whether the child's status was collected, observed WITHOUT collecting it.

    ``Popen.poll()``/``wait()`` reap the child themselves, so either would repair
    the condition under test and this test would pass on a tree with no reaper at
    all — the dead-instrument failure this repo names by name. The two reads below
    are both non-collecting: ``returncode`` is only filled in by a *waiter*, and
    the zombie probe forks ``ps`` and reads the state column.
    """
    from local_operator import procstate

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = procstate.zombie_states([child.pid]).get(child.pid)
        if child.returncode is not None:
            assert (
                state is not True
            ), f"pid {child.pid} has a collected status AND reads as a zombie"
            return True
        time.sleep(0.05)
    return False


def test_the_zombie_probe_reads_a_child_nobody_reaped() -> None:
    """The instrument's own check, first: a corpse reads as one.

    Without this, the cell below could pass because the probe answers "not a
    zombie" about every pid, which is the reading a broken instrument gives.
    """
    import subprocess
    import sys

    from local_operator import procstate

    corpse = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 — fixed argv
    try:
        deadline = time.monotonic() + _WAIT_S
        while time.monotonic() < deadline:
            if procstate.zombie_states([corpse.pid]).get(corpse.pid) is True:
                break
            time.sleep(0.05)
        else:  # pragma: no cover — a host where this never reads true is not testable
            raise AssertionError(
                "an exited, unreaped child never read as a zombie: the probe this "
                "test rests on is not reading anything"
            )
    finally:
        corpse.wait(timeout=_WAIT_S)


def test_a_spawned_child_that_exits_is_reaped_by_its_spawner() -> None:
    """Q-XH-5: the exit is harvested, so no ``<defunct>`` accumulates.

    Measured on the tree before the reaper: a relay that had spawned runtimes across
    two EC2 peers held them as ``Zs`` children for 18-47 minutes — one per remote
    engage, alive as corpses until the RELAY exited, which for a supervised relay is
    weeks. The property is asserted on a child that exits immediately, because a real
    runtime's own exit is minutes away and a cell that waited for it would be
    measuring the wait; the wiring is pinned on the production spawn by
    ``test_a_spawned_runtime_is_reaped_by_the_process_that_spawned_it``.
    """
    import subprocess
    import sys

    from local_operator import procstate

    # (1) THE MECHANISM, on a child that exits immediately: no wait is issued by
    # this test anywhere, so a collected status can only have come from the
    # reaper that the spawn path installed.
    quick = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 — fixed argv
    launch_module._harvest_on_exit(quick)  # noqa: SLF001 — the seam under test
    assert _await_collected(
        quick
    ), f"pid {quick.pid} is still uncollected after its exit: nothing reaped it"
    assert procstate.zombie_states([quick.pid]).get(quick.pid) is not True


def test_a_spawned_runtime_is_reaped_by_the_process_that_spawned_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same property through ``launch._spawn_runtime``, on a REAL runtime.

    The child is the production runtime on the mock provider, so this is the process
    a relay leaves behind when a remote session is stopped or moves, and the exit is
    an immediate ``SIGKILL`` — the worst case, where no clean exit path could have
    reaped anything on its own. Nothing here waits on the child: the status has to
    arrive on its own, and the pid has to stop reading as a corpse.
    """
    from local_operator import procstate

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _seed(config_dir)
    _isolate(monkeypatch, config_dir)
    child = None
    try:
        child = launch_module._spawn_runtime(_SESSION_ID, str(config_dir), defer_materialise=True)
        pid = child.pid
        # It is really up before the kill, so the cell is about a spawned process
        # rather than about a spawn that failed.
        deadline = time.monotonic() + _WAIT_S
        while child.poll() is None and time.monotonic() < deadline:
            if procstate.pid_liveness(pid) == "live":
                break
            time.sleep(0.05)
        os.kill(pid, signal.SIGKILL)
        assert _await_collected(child), (
            f"the killed runtime {pid} was never reaped by its spawner: it is a "
            f"corpse in ``ps`` (state={procstate.zombie_states([pid]).get(pid)})"
        )
        assert procstate.zombie_states([pid]).get(pid) is not True
    finally:
        if child is not None:
            _reap(child, config_dir)
