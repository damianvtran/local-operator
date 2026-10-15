"""Take the protocol pipes away from everything else in the process (D4).

``lop acp`` speaks JSON-RPC on fd 0 and fd 1, and two things in an ordinary lop
process would corrupt that stream:

* tool children inherit fd 0 — the ``bash`` tool sets only stdout/stderr on its
  child — so a command that reads stdin (``cat``, a prompt) would swallow
  protocol frames meant for this agent;
* the library still has stray ``print()`` calls that write to fd 1, which is
  the client's input stream.

So, before anything else runs, the two descriptors are duplicated into private
ones (``os.dup`` returns non-inheritable descriptors, PEP 446), fd 0 is pointed
at ``/dev/null`` and fd 1 at stderr. Children then inherit a closed stdin and a
stdout that is the log, and only the agent's connection holds the real pipes.

Standard library only: this runs before the optional ``acp`` package is
imported, and it must work (and be testable) without it.
"""

from __future__ import annotations

import os
import sys


def claim_stdio() -> tuple[int, int]:
    """Make fd 0/1 private to the caller; return ``(reader_fd, writer_fd)``.

    The returned descriptors are the ORIGINAL stdin and stdout, not inheritable
    by any child. After the call, fd 0 reads ``/dev/null`` and fd 1 writes to
    fd 2, for this process and for every child it starts.
    """
    # Anything already buffered for fd 1 belongs to nobody on the protocol
    # stream; flushing first keeps it from landing there after the swap.
    sys.stdout.flush()
    reader = os.dup(0)
    writer = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(devnull, 0)
    finally:
        os.close(devnull)
    os.dup2(2, 1)
    return reader, writer
