"""The wall-clock bound for tests whose failure mode is a hang.

Why this is not ``asyncio.wait_for``
------------------------------------

Every other bound in this suite is a Python-level one, and none of them can
work here. The regression this package exists to catch (#401) deadlocks the
process at the *kernel* level: a worker thread parked in ``fcntl.flock()`` on a
descriptor that the event-loop thread then calls ``os.close()`` on, which on
macOS/BSD blocks until that ``flock()`` returns. Both threads are inside a
syscall holding no bytecode, so:

* ``asyncio.wait_for`` never fires, because the loop is the thing that stopped;
* a ``threading.Thread`` watchdog never fires either, because it needs the GIL
  released by a thread that will never return to release it deliberately, and
  even when it does wake, the interpreter-level ``faulthandler.dump_traceback``
  raises ``RuntimeError: file.fileno() is not a valid file descriptor`` under
  Textual (which has taken over the terminal), and ``os._exit`` never runs;
* ``pytest-timeout``'s default ``signal`` method never fires, because a Python
  signal handler only runs between bytecodes.

Measured, not assumed: driving the pre-fix code through ``/resume`` produced a
process that ignored a 20 s thread watchdog and had to be ``kill -9``'d.

``faulthandler.dump_traceback_later`` is the one instrument that survives it.
It arms a timer in a dedicated **C** thread that writes the stacks of every
thread with ``write(2)`` directly to a file descriptor without the GIL, Python
frames or interpreter state. E2E bounds are dump-only: pytest cannot safely be
ended from the native callback, so CI must inspect the retained dump and its
outer job timeout remains the eventual failure bound.

The GRANULARITY is deliberate: the timer is armed around the specific step
under test rather than around the whole test, so the dump names the operation
that hung instead of "the test was slow".

ONE PROCESS-GLOBAL TIMER, SHARED WITH THE SHARD WATCHDOG
``faulthandler``'s timer is process-wide, so a ``bounded`` block displaces any
other armed timer and its ``finally`` leaves nothing armed. ``tests.shard_stall_watchdog``
arms the same timer around one xdist test item, so a test body that entered
``bounded`` would take that worker's stacks away for the rest of the test (the
controller would still name the test from xdist reports; only the stacks are
lost). No CI configuration runs the two in one process -- the shard job sets
``LOCAL_OPERATOR_SHARD_STALL_SECONDS`` with ``e2e`` deselected, and this stage
runs ``-n0`` without it, so there are no xdist workers here -- and that
invariant is pinned by
``tests/unit/test_shard_stall_watchdog.py::test_no_ci_job_runs_both_watchdogs_in_one_process``.
Nothing interlocks the timers, because nothing runs them together; if a future
job did, the interlocks would have to be added deliberately.
The product has a third timer (``session/runtime/stall_watchdog``, the runtime's
own bound). It cannot reach this stage either, and not because of job
configuration: it arms only inside the runtime child's ``__main__`` branch, so
it lives in a spawned runtime process, never in the pytest process this file's
``bounded`` blocks run in.

The watchdog is diagnostic-only. ``exit=True`` would terminate the whole pytest
worker from a C callback, losing pytest's normal reporting and any unrelated
work in that process. A fired dump survives for CI to report, while the workflow's
outer timeout remains the failure boundary for a test that never returns.
"""

from __future__ import annotations

import contextlib
import faulthandler
import itertools
import os
import sys
from collections.abc import Iterator
from pathlib import Path

#: Where the C-level dump is written. A real file rather than ``sys.stderr``:
#: under a Textual pilot the app owns the terminal and pytest has replaced the
#: stderr OBJECT, so ``faulthandler``'s fileno() lookup fails outright on it —
#: the dump has to land somewhere with a genuine descriptor behind it. Kept in
#: the OS temp dir rather than a tmp_path fixture so the path remains available to
#: the post-run CI report even when a hung test never returns.
DUMP_DIR = Path(os.environ.get("TMPDIR", "/tmp"))

#: Filename prefix for one bounded block's dump. Each block gets its OWN file
#: rather than a single shared path. A shared path is truncated by whichever
#: block runs next, so a real hang's stacks could be destroyed by a later
#: block's header; preserving per-block files keeps each fired diagnostic attributable.
DUMP_PREFIX = "lo-tui-e2e-hang"

#: The marker ``faulthandler`` itself writes when the timer actually fires
#: (``Timeout (0:01:00)!``). Presence of this string is what distinguishes a
#: genuine dump from a file whose header was written at arm time, so a reader
#: is never handed a timeout claim for a test that failed on an assertion.
FIRED_MARKER = "Timeout ("


@contextlib.contextmanager
def bounded(seconds: float, what: str) -> Iterator[None]:
    """Capture every thread's stack if ``what`` outlives its diagnostic bound.

    Wrap the smallest step that can hang, not the whole test: the dump names the
    parked operation, while a bound around ten steps only says one of them stopped.

    The dump file's EXISTENCE is the signal. ``faulthandler`` writes through a
    raw file descriptor from a C thread, so the handle and header exist before
    the timer is armed. Ordinary exit paths remove the file; when the timer fires,
    this context manager cannot run while the blocked code holds the GIL, so the
    diagnostic remains for CI to report. The C callback does not exit the worker.
    """
    # Unique per block and per process: pytest may run several bounded blocks,
    # and under a fired watchdog the surviving file must be attributable to the
    # block that actually hung rather than to whichever ran last.
    path = DUMP_DIR / f"{DUMP_PREFIX}-{os.getpid()}-{next(_COUNTER)}.log"
    handle = path.open("w", encoding="utf-8")
    try:
        handle.write(f"[e2e watchdog] {what!r} exceeded {seconds:g}s; every thread follows.\n")
        handle.flush()
        # cancel_dump_traceback_later() in the finally makes sequential bounds
        # re-entrant-safe. A diagnostic must not terminate pytest from the C thread:
        # normal assertion reporting and the CI dump reporter stay in control.
        faulthandler.dump_traceback_later(seconds, file=handle, exit=False)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
        # A cooperative operation may return after the native timer wrote its
        # diagnostic. Keep the test red in that case; the C callback itself must
        # not terminate pytest to report the bound.
        if _is_real_dump(path):
            raise TimeoutError(f"{what!r} exceeded {seconds:g}s; thread dump retained at {path}")
    finally:
        handle.close()
        # The timer's C thread may have written a real dump before the step returned.
        # Keep that diagnostic for CI; only a header without a fire is disposable.
        if not _is_real_dump(path):
            with contextlib.suppress(OSError):
                path.unlink()


#: Distinguishes concurrent bounded blocks within one process.
_COUNTER = itertools.count()


def report_previous_hang() -> str:
    """The stacks of a bound that actually fired, or an explicit statement that
    none did.

    Reads back what the C timer wrote while the tested operation was blocked. Only
    files carrying :data:`FIRED_MARKER` are reported, so a header-only file cannot
    turn an ordinary assertion failure into a timeout claim.
    """
    dumps = sorted(
        (path for path in DUMP_DIR.glob(f"{DUMP_PREFIX}-*.log") if _is_real_dump(path)),
        key=lambda path: path.stat().st_mtime,
    )
    if not dumps:
        return (
            "(no watchdog dump: nothing exceeded its bound in this run, so any "
            "failure above is an ordinary assertion failure, not a hang)\n"
        )
    # Newest last so the most recent hang is what a reader sees at the bottom;
    # every surviving dump is printed because a matrix leg can hang more than
    # once and discarding the earlier one hides evidence.
    return "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in dumps)


def _is_real_dump(path: Path) -> bool:
    """True only for a file ``faulthandler`` actually wrote stacks into."""
    try:
        return FIRED_MARKER in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def print_previous_hang_and_exit() -> None:  # pragma: no cover - operator aid
    """``python -m tests.e2e.watchdog`` — show the last hang's stacks."""
    sys.stdout.write(report_previous_hang())


if __name__ == "__main__":  # pragma: no cover
    print_previous_hang_and_exit()
