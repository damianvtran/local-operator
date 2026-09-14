"""A deliberately slow stdio MCP server: answers nothing, ever.

Declared by the ``mcp`` variant of ``scripts/bench_cold_engage.py`` to put "this
machine has an MCP server that never completes a handshake" in front of a cold
engage without touching the network. That is the shape the operator's own
``runtime.log`` shows for ``gitlab`` and ``google-workspace``.

It reads stdin until EOF or its own deadline, then exits, so it cannot outlive a
benchmark run and cannot be left behind holding a pipe.
"""

from __future__ import annotations

import sys
import time

DEADLINE_S = 25.0

started = time.time()
while time.time() - started < DEADLINE_S:
    line = sys.stdin.readline()
    if not line:
        break
    # Deliberately swallow the request: an unanswered `initialize` is the cost
    # a hanging server puts on the runtime that declared it.
    time.sleep(1.0)
