"""The agent-facing half of the app's console surfaces (design §17.1 row C).

Two modules, each a sibling of the browser's own:

* :mod:`.state` — the FILE read that decides whether the ``console`` tool exists
  at all. The record is the app's, and the console rides it rather than owning a
  second discovery namespace (design §10.1).
* :mod:`.backend` — the authenticated loopback client. The transport is NOT
  re-implemented here: it is
  :class:`~local_operator.browser_bridge.backend.HostClient` pointed at this
  package's record, with the console's own failure copy and timeouts.

The console is one namespace on the app's EXISTING host — one key, one
heartbeat, one set of safety rules (bind ``127.0.0.1``, require
``X-Bridge-Key`` on every request, send no CORS headers, expose no CDP). It is
not a second endpoint, because a second listener would mean a second key, a
second heartbeat and a second set of those rules to keep true.
"""

from __future__ import annotations

from local_operator.ui_console.backend import (
    CONSOLE_COPY,
    CONSOLE_METHODS,
    CONSOLE_TIMEOUTS,
    ConsoleHostClient,
    console_error_text,
    console_timeout,
    ui_console_advertisable,
    ui_console_available,
    ui_console_liveness,
)
from local_operator.ui_console.state import ConsoleHostState

__all__ = [
    "CONSOLE_COPY",
    "CONSOLE_METHODS",
    "CONSOLE_TIMEOUTS",
    "ConsoleHostClient",
    "ConsoleHostState",
    "console_error_text",
    "console_timeout",
    "ui_console_advertisable",
    "ui_console_available",
    "ui_console_liveness",
]
