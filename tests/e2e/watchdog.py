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

``faulthandler.dump_traceback_later`` is the one bound that survives it. It
arms a timer in a dedicated **C** thread that writes the stacks of every thread
with ``write(2)`` directly to a file descriptor and then calls ``_exit(1)`` —
no GIL, no Python frames, no interpreter state required. That makes a
deadlocked run fail in seconds with the exact stacks of the parked threads,
which is precisely the diagnostic a reader needs.

The GRANULARITY is deliberate: the timer is armed around the specific step
under test rather than around the whole test, so the dump names the operation
that hung instead of "the test was slow".

``exit=True`` kills the whole process, so a fired watchdog takes down the
pytest worker with it. That is the intended behaviour and not a rough edge: a
deadlocked interpreter cannot report a test failure through any gentler
channel, and a hard exit carrying the stacks is strictly better than a job that
sits until the CI wall clock reaps it with no diagnostic at all.
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
#: the OS temp dir rather than a tmp_path fixture so the path is stable and
#: printable in the failure message a surviving process can still emit.
DUMP_DIR = Path(os.environ.get("TMPDIR", "/tmp"))

#: Filename prefix for one bounded block's dump. Each block gets its OWN file
#: rather than a single shared path. A shared path is truncated by whichever
#: block runs next, so a real hang's stacks could be destroyed by a later
#: block's header — and the dump only survived at all because the hanging test
#: happened to be the last one in the file.
DUMP_PREFIX = "lo-tui-e2e-hang"

#: The marker ``faulthandler`` itself writes when the timer actually fires
#: (``Timeout (0:01:00)!``). Presence of this string is what distinguishes a
#: genuine dump from a file whose header was written at arm time, so a reader
#: is never handed a timeout claim for a test that failed on an assertion.
FIRED_MARKER = "Timeout ("


@contextlib.contextmanager
def bounded(seconds: float, what: str) -> Iterator[None]:
    """Fail loudly, with every thread's stack, if ``what`` takes too long.

    Wrap the smallest step that can hang, not the whole test: the dump's value
    is that it names the parked operation, and a bound around ten steps only
    tells you one of them stopped.

    The dump file's EXISTENCE is the signal, and keeping that honest drives the
    shape here. ``faulthandler`` writes with a raw file descriptor from a C
    thread, so the handle and its header must exist BEFORE the timer is armed —
    the header cannot be deferred to the moment it fires. What can be deferred
    is the file's survival: every ordinary exit path (success, assertion
    failure, error) removes it, and only a fired watchdog leaves it behind,
    because that path exits the process immediately and never reaches the
    cleanup. So a file that is still there means the bound really tripped,
    which is exactly what the CI reporting step claims when it prints one.
    """
    # Unique per block and per process: pytest may run several bounded blocks,
    # and under a fired watchdog the surviving file must be attributable to the
    # block that actually hung rather than to whichever ran last.
    path = DUMP_DIR / f"{DUMP_PREFIX}-{os.getpid()}-{next(_COUNTER)}.log"
    handle = path.open("w", encoding="utf-8")
    try:
        handle.write(f"[e2e watchdog] {what!r} exceeded {seconds:g}s; every thread follows.\n")
        handle.flush()
        # cancel_dump_traceback_later() in the finally is what makes this
        # re-entrant-safe across sequential steps in one test: each `bounded`
        # block owns the single process-wide timer for its own duration.
        faulthandler.dump_traceback_later(seconds, file=handle, exit=True)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
    finally:
        handle.close()
        # Reached on success AND on an ordinary failure, never on a fired
        # watchdog (which _exit()s from the C thread). This is what stops a
        # green run or a plain assertion failure from leaving behind a file
        # that reads as a hang report.
        with contextlib.suppress(OSError):
            path.unlink()


#: Distinguishes concurrent bounded blocks within one process.
_COUNTER = itertools.count()


def report_previous_hang() -> str:
    """The stacks of a bound that actually fired, or an explicit statement that
    none did.

    Reads back what the dying process wrote: a fired watchdog ``_exit``s
    immediately, so nothing in that process can attach the dump to a test
    report. Only files carrying :data:`FIRED_MARKER` are reported, so this can
    never present a header-only file as a timeout — the failure mode that made
    an ordinary 3-second assertion failure look like a 45-second hang.
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
