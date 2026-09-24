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
frames or interpreter state. It is armed with ``exit=False``, so it is a
DIAGNOSTIC and never the fast-fail: a step that can return after its dump does
so, and ``bounded`` then raises a normal ``TimeoutError`` for pytest to report.

THE FAST-FAIL IS A SECOND, KERNEL-LEVEL ARM (``SIGALRM``, see
:data:`BOUND_GRACE_S`), because the case that needs it is precisely the case
nothing in-process can reach: a process wedged with the GIL held takes no
Python signal handler and no thread, so the only instrument that can still fail
it is one the KERNEL acts on. It fires ``BOUND_GRACE_S`` after the dump was due,
so the order is dump first, terminate second, and a wedged run fails at its own
bound instead of holding a CI slot until the job ceiling. The process surviving
is not the failure; a bound that expires without a dump would be, and the
stage's own reporting step says which of the two it is looking at.

The GRANULARITY is deliberate: the timer is armed around the specific step
under test rather than around the whole test, so the dump names the operation
that hung instead of "the test was slow".

ONE PROCESS-GLOBAL TIMER, SHARED WITH THE SHARD WATCHDOG
``faulthandler``'s timer is process-wide, so a ``bounded`` block displaces any
other armed timer and its ``finally`` leaves nothing armed. The ``SIGALRM`` arm
is one interval timer per process too, and a block displaces whatever was
PENDING on it for its duration -- but unlike the dump timer it PUTS THAT BACK on
the way out (R4-3: the handler together with the timer's remaining delay),
because an alarm this block silently cancelled is the one failure a watchdog
must not hand out. So two nested ``bounded`` blocks leave only the INNER one's
backstop armed while it runs, and the outer one's resumes when the inner exits
(late by the inner block's duration); neither instrument is a substitute for the
outer job ceiling, which is why the stage ships one. ``tests.shard_stall_watchdog``
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

The watchdog is diagnostic-only where the PRODUCT's bound is concerned -- and
that is the property this file must never break: nothing here ends a runtime, and
no arm here (the dump timer or the ``SIGALRM`` backstop) is ever taken by
the runtime child. ``exit=True`` would terminate the whole pytest
worker from a C callback, losing pytest's normal reporting and any unrelated
work in that process, so the C callback stays diagnostic and the fast-fail lives
one rung out, on the pytest process the test's own bound owns. A fired dump
survives for CI to report, and the workflow's outer timeout remains the backstop
for anything that escapes the ``SIGALRM`` arm (an unkillable syscall, a stop the
kernel cannot deliver).
"""

from __future__ import annotations

import contextlib
import faulthandler
import itertools
import os
import signal
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


#: How long after a block's own bound the KERNEL ends the process if the step is still
#: wedged. THIS IS THE FAST-FAIL, and it is deliberately not the dump timer: the bound
#: that catches a GIL-held hang cannot be a Python one (see the module docstring), so the
#: instrument that acts on it is ``SIGALRM`` at its default disposition, which the kernel
#: delivers without needing the GIL, a bytecode, or a thread of ours.
#:
#: THE ORDER IS THE WHOLE DESIGN, and it is what makes this honest under a dump-only
#: policy: the C timer writes every thread's stacks AT ``seconds`` and this ends the
#: process ``BOUND_GRACE_S`` later, so the artifact exists before anything dies and a
#: wedged step fails at its own bound rather than holding a CI runner until
#: ``timeout-minutes``. The grace is not a second bound on the hang -- it is the
#: settling allowance for the DUMP WRITE, which runs in one pass from a C thread and is
#: measured in milliseconds for the handful of threads these stages park (against a
#: bound of 60-300 s on every call site, and a 20-minute job ceiling). It is therefore
#: two orders of magnitude above the write it waits for and still seconds, not minutes,
#: after the failure it reports.
#:
#: IT IS NOT A WALL-CLOCK DETECTOR: nothing here decides that a test has hung. The C
#: timer's expiry is that decision, and it is what leaves the evidence; this only turns
#: "dumped and still wedged" into a process end, which is the part no in-process
#: instrument can do. A step that returns after its dump is never reached by it -- the
#: alarm is cancelled in ``bounded``'s ``finally``, and the ``TimeoutError`` below is
#: then what reports the failure, through pytest's own machinery.
#:
#: WHAT THE GRACE DECIDES, stated because it is the visible trade: a step that
#: overshoots its bound by LESS than the grace returns and fails as one test, with a
#: ``TimeoutError`` pytest reports normally; one still running when the grace expires is
#: ended by the kernel, so its shard ends with a signal and no XML. Before this PR the
#: same overshoot ended the process AT the bound (``exit=True``), so the grace only
#: widens the window in which a slow-but-returning step can still fail cleanly; it does
#: not narrow one.
BOUND_GRACE_S = 5.0


@contextlib.contextmanager
def bounded(seconds: float, what: str) -> Iterator[None]:
    """Capture every thread's stack if ``what`` outlives its diagnostic bound.

    Wrap the smallest step that can hang, not the whole test: the dump names the
    parked operation, while a bound around ten steps only says one of them stopped.

    The dump file's EXISTENCE is the signal. ``faulthandler`` writes through a
    raw file descriptor from a C thread, so the handle and header exist before
    the timer is armed. Ordinary exit paths remove the file; when the timer fires,
    this context manager cannot run while the blocked code holds the GIL, so the
    diagnostic remains for CI to report. The C callback does not exit the worker;
    the ``SIGALRM`` arm ends the process one grace later if the step is still
    wedged, with the dump already written.
    """
    # Unique per block and per process: pytest may run several bounded blocks,
    # and under a fired watchdog the surviving file must be attributable to the
    # block that actually hung rather than to whichever ran last.
    path = DUMP_DIR / f"{DUMP_PREFIX}-{os.getpid()}-{next(_COUNTER)}.log"
    handle = path.open("w", encoding="utf-8")
    # ``SIG_DFL`` RATHER THAN A PYTHON HANDLER, and this is the point rather than a
    # shortcut: a Python-level handler only runs between bytecodes, which is exactly what
    # a wedged process never produces -- the same reason this file does not use
    # ``pytest-timeout``'s signal method.
    #
    # WHAT A BLOCK TAKES, AND WHY IT TAKES BOTH HALVES (agent review round 4, R4-3).
    # ``ITIMER_REAL`` is process-wide and ``setitimer`` REPLACES whatever was pending on
    # it, so a block owns the alarm for its duration -- and restoring the handler alone
    # would silently CANCEL the third party's timer that the block displaced (the
    # ``finally`` zeroes it and never puts it back): an alarm that simply never arrives,
    # which is the one failure a watchdog must not have. Nothing in ``tests/e2e`` arms
    # ``ITIMER_REAL`` today (this is the tree's only site), so the restoration is latent
    # rather than load-bearing -- which is exactly when it is cheap to get right. Stated
    # rather than implied, the other half of owning it: an alarm a third party had
    # PENDING at a shorter deadline than the block runs lands while ``SIG_DFL`` is
    # installed and takes its default action. Restoring the disposition cannot undo that,
    # so the honest arrangement is that no other caller shares this timer.
    previous_alarm = signal.signal(signal.SIGALRM, signal.SIG_DFL)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        handle.write(f"[e2e watchdog] {what!r} exceeded {seconds:g}s; every thread follows.\n")
        handle.flush()
        # cancel_dump_traceback_later() in the finally makes sequential bounds
        # re-entrant-safe. A diagnostic must not terminate pytest from the C thread:
        # normal assertion reporting and the CI dump reporter stay in control.
        faulthandler.dump_traceback_later(seconds, file=handle, exit=False)
        # THE FAST-FAIL, one grace after the dump is due (see BOUND_GRACE_S). Set AFTER
        # the dump timer so the two expiries cannot invert on a loaded host.
        signal.setitimer(signal.ITIMER_REAL, seconds + BOUND_GRACE_S)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
            signal.setitimer(signal.ITIMER_REAL, 0)
        # A cooperative operation may return after the native timer wrote its
        # diagnostic. Keep the test red in that case; the C callback itself must
        # not terminate pytest to report the bound.
        if _is_real_dump(path):
            raise TimeoutError(f"{what!r} exceeded {seconds:g}s; thread dump retained at {path}")
    finally:
        # Reached on the ordinary paths only: a step that is still wedged when the grace
        # expires is ended by the kernel and never gets here, which is the fast-fail.
        #
        # PUT BACK WHAT WAS THERE, IN THE STATE IT WAS IN (R4-3): the timer as well as
        # the handler. A pending third-party timer is restored at its SAVED delay rather
        # than at a delay reduced by how long this block ran, which can only make such an
        # alarm LATE and never silent -- the direction a watchdog may be wrong in.
        # ``(0.0, 0.0)``, the ordinary case here, is a plain cancel.
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_alarm)
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
