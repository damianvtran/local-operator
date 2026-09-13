"""The e2e death witnesses, pinned without booting the application.

WHY THIS EXISTS. ``tests/e2e/test_terminal_close_survives_e2e.py`` establishes
that the interface died before it asserts that the runtime survived — without
that precondition, "the runtime outlived the terminal" is consistent with
"nothing died at all", which is the vacuous pass the arm exists to prevent.
The macOS leg of CI then failed that precondition in 7 of 8 recent failing runs
across `main` and two open branches, and the failure ("the interface never
died") could not distinguish an interface that survived the ``SIGKILL`` from a
harness that could not observe the death. The arm grew a second witness and a
re-armed kill; THIS file is what holds those two facts still, because the
alternative is holding them with an 8-minute, two-platform, app-booting e2e
run whose failure mode is the thing under test.

WHY IT IS A UNIT TEST AND NOT MORE E2E COVERAGE. Everything pinned here is a
property of the pty and of the process the kernel kills — not of lop — so it
needs a pty child, not a conversation. Precedent: the shard stall watchdog's
helper is pinned the same way in ``tests/unit/test_shard_stall_watchdog.py``,
and the e2e module already imports helpers from a sibling e2e test module, so
importing this one is the established shape rather than a new coupling.

WHAT IS DELIBERATELY NOT PINNED: that a real TUI is reaped within ``DEATH_S``.
That is a claim about the platform under load, and a test that asserted it would
be the same bet on machine load this repo's timing section forbids. The
measured figures live in ``DEATH_S``'s comment instead.
"""

from __future__ import annotations

import contextlib
import os
import pty
import signal
import sys
import time
from collections.abc import Iterator

import pytest

from tests.e2e.test_terminal_close_survives_e2e import (
    _await_interface_death,
    _kill_interface,
    _Pty,
    _reaped,
)

#: A child that outlives every timeout here and never touches the terminal.
_SLEEPER = "import time; time.sleep(120)"

#: A child that exits on its own, to close the slave side from the inside.
_QUITTER = "pass"


def _reap_within(pid: int, *, timeout: float) -> bool:
    """Whether ``pid`` is reaped within ``timeout``; the unit test's own wall.

    Kept here rather than imported: the arm's `_await_interface_death` is the
    thing under test, so teardown must not depend on it to clean up.
    ``ChildProcessError`` counts as reaped — see
    ``test_a_child_reaped_by_someone_else_counts_as_a_death`` for why that is
    the same fact and not an error.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if os.waitpid(pid, os.WNOHANG)[0]:
                return True
        except ChildProcessError:
            return True
        time.sleep(0.02)
    return False


@contextlib.contextmanager
def _pty_child(script: str) -> Iterator[tuple[int, _Pty]]:
    """A real pty whose child runs ``script``, torn down whatever happens.

    A leaked child is not a cosmetic problem in this repo: the e2e suite
    contends real lock files and drives whole app lifecycles, and a stray
    process group is what the next run's death assertion would find instead of
    its own. So teardown SIGKILLs the group and *waits* for the reap, and says
    so (``assert``) if it could not — a shutdown that silently leaves a process
    behind is how a test suite starts lying about its own isolation.
    """
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os.execve(sys.executable, [sys.executable, "-c", script], os.environ)
        os._exit(127)
    terminal = _Pty(fd)
    try:
        yield pid, terminal
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
        reaped = _reap_within(pid, timeout=10.0)
        terminal.close()
        assert reaped, "the pty child outlived teardown; the next run inherits it"


def _drain_until_eof(terminal: _Pty, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not terminal.at_eof:
        terminal.drain(0.1)


def test_eof_on_the_master_is_the_witness_for_the_slave_side_closing() -> None:
    """A child that exits closes the slave, and the master reads EOF.

    This is the second witness's whole mechanism: no ``waitpid`` is involved, so
    a parent that never gets to reap the child can still observe the death.
    """
    with _pty_child(_QUITTER) as (pid, terminal):
        _drain_until_eof(terminal, timeout=10.0)
        assert terminal.at_eof is True


def test_closing_our_own_master_is_not_an_eof_witness() -> None:
    """``close()`` must not fake the witness for the arm that calls it.

    The ``pty-close`` arm closes the master itself — that IS the death shape it
    tests. If ``close()`` set the EOF flag, that arm's precondition would be
    satisfied by the test's own teardown rather than by the interface dying,
    which is exactly the vacuous pass the witnesses exist to prevent. The child
    is still running when the master goes away, so the honest answer is "no
    evidence available", not "gone".
    """
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os.execve(sys.executable, [sys.executable, "-c", _SLEEPER], os.environ)
        os._exit(127)
    terminal = _Pty(fd)
    try:
        assert _reaped(pid) is False
        terminal.close()
        assert terminal.is_open is False
        assert terminal.at_eof is False, "the master closed by the test is not a death"
    finally:
        _kill_interface(pid)
        assert _reap_within(pid, timeout=10.0)


def test_a_killed_interface_is_detected_by_the_loop() -> None:
    """The arm's real path: group-kill the child, then watch it die."""
    with _pty_child(_SLEEPER) as (pid, terminal):
        assert _reaped(pid) is False
        _kill_interface(pid)
        assert _await_interface_death(terminal, pid, timeout=10.0) is True


def test_a_living_interface_is_not_reported_dead() -> None:
    """The precondition must not pass for an interface that is still alive.

    The loop drains and polls; neither witness may fire while the child runs.
    Bounded at 0.6 s because this asserts a *negative* — the point is that no
    witness fires, not how long the loop can spin.
    """
    with _pty_child(_SLEEPER) as (pid, terminal):
        assert _await_interface_death(terminal, pid, timeout=0.6) is False
        assert _reaped(pid) is False
        assert terminal.at_eof is False


def test_a_child_reaped_by_someone_else_counts_as_a_death() -> None:
    """A runner that reaps the child reports the same fact from the other side.

    ``waitpid`` raises ``ChildProcessError`` once the exit status has been
    consumed by anything else, and that is a death, not an error to propagate;
    a pid that was never ours answers identically, which is why the helper
    treats "no such child" as gone rather than as a failure to observe.
    """
    with _pty_child(_QUITTER) as (pid, terminal):
        assert _reap_within(pid, timeout=10.0) is True
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        assert _reaped(pid) is True
