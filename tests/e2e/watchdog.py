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
DUMP_PATH = Path(os.environ.get("TMPDIR", "/tmp")) / "lo-tui-e2e-hang.log"


@contextlib.contextmanager
def bounded(seconds: float, what: str) -> Iterator[None]:
    """Fail loudly, with every thread's stack, if ``what`` takes too long.

    Wrap the smallest step that can hang, not the whole test: the dump's value
    is that it names the parked operation, and a bound around ten steps only
    tells you one of them stopped.
    """
    handle = DUMP_PATH.open("w", encoding="utf-8")
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


def report_previous_hang() -> str:
    """The last watchdog dump, for a runner reconstructing why a worker died.

    The dump is written by a process that then exits immediately, so nothing in
    that process can attach it to a test report. This reads it back so the
    surviving pytest session (or a human reading CI logs) can print the stacks.
    """
    if not DUMP_PATH.is_file():
        return "(no watchdog dump was written)"
    return DUMP_PATH.read_text(encoding="utf-8", errors="replace")


def print_previous_hang_and_exit() -> None:  # pragma: no cover - operator aid
    """``python -m tests.e2e.watchdog`` — show the last hang's stacks."""
    sys.stdout.write(report_previous_hang())


if __name__ == "__main__":  # pragma: no cover
    print_previous_hang_and_exit()
