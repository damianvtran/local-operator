"""A real exited-but-unreaped process, for tests of liveness probes.

A zombie is the one process state that ``kill(pid, 0)`` cannot tell apart from a
running process, and it is what a killed runtime leaves behind when its parent —
usually a long-lived TUI — never reaps it. Testing anything that depends on
seeing through that state needs a genuine zombie, and producing one has two
requirements that are easy to lose:

1. **It must be a child of the test process.** Only the parent can reap it. A
   helper that spawns a child and then exits hands it to the init process, which
   reaps it immediately, and the window a test needs closes with it.

2. **Its ``Popen`` object must stay referenced for as long as the test needs
   it.** ``subprocess`` reaps the instances it has finalised, on the next
   ``Popen`` it creates (``_cleanup``); an instance the test still holds is not
   in that list, so the zombie survives the probe's own ``ps`` fork. Dropping
   the last reference and letting ``__del__`` run would let the very next probe
   reap it — the test would then assert against a pid that is fully dead, which
   is the state every probe already agreed on, and it would pass without ever
   exercising the zombie case it was written for.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: How long to wait for the kernel to move the killed child to ``Z``. Normally
#: immediate; generous so a loaded CI runner reports a real failure rather than
#: a scheduling race.
_ZOMBIE_TIMEOUT_S = 10.0


def process_state(pid: int) -> str:
    """The process table's state letter for ``pid``, or ``""`` once it is gone.

    Linux answers from ``/proc``; macOS has no ``/proc`` and needs the ``ps``
    fork, exactly as :func:`local_operator.procstate.is_zombie` documents.
    """
    proc_status = Path(f"/proc/{pid}/stat")
    if proc_status.exists():
        # `comm` can contain spaces and parentheses; state is the field after
        # the LAST ')'.
        return proc_status.read_text().rpartition(")")[2].strip()[:1]
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["/bin/ps", "-o", "state=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()[:1]


@contextmanager
def unreaped_child() -> Iterator[int]:
    """Yield the pid of a real zombie, then reap it on the way out.

    The child is killed and deliberately left unreaped for the whole block, so
    ``os.kill(pid, 0)`` succeeds against it for as long as the caller holds it —
    which is precisely the state a liveness probe has to see through.
    """
    proc = subprocess.Popen(["sleep", "60"])  # noqa: S603,S607 — fixed argv
    try:
        proc.kill()
        deadline = time.monotonic() + _ZOMBIE_TIMEOUT_S
        state = process_state(proc.pid)
        while not state.startswith("Z") and time.monotonic() < deadline:
            time.sleep(0.02)
            state = process_state(proc.pid)
        if not state.startswith("Z"):
            raise AssertionError(f"child {proc.pid} never became a zombie (state {state!r})")
        yield proc.pid
    finally:
        proc.wait(timeout=10)
