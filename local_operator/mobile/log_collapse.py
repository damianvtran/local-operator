"""Collapse a repeating log record so an out-of-files daemon cannot fill the disk.

WHY THIS EXISTS. When the phone daemon reaches its open-file limit, asyncio's
accept loop fails on every readiness wakeup and its default exception handler
logs ``socket.accept() out of system resource`` WITH a full traceback each
time. Measured on a rig at a forced 64-descriptor limit: ~7 MB/s into
``mobile.log`` (1.78 GB in about four minutes). That is the same failure path
as the "Asleep / Not answering" report — an fd-starved daemon — and filling the
disk on top of it turns one broken service into a broken machine.

THE INVARIANT this filter keeps: a given message (same logger, level and first
line) is written AT MOST ONCE PER ``interval`` seconds, and no occurrence is
lost from the record — the suppressed count is carried onto the next copy that
is written ("… [N similar suppressed in the last Ts]"). A reader therefore still
sees that it kept happening and how often, which is what `lop mobile doctor`'s
log scan counts on; only the identical tracebacks are dropped.

Scope is deliberately narrow: it is installed on the ``asyncio`` logger only
(`install`), whose records are loop-level faults with no per-request content,
so collapsing two genuinely different sessions' messages into one is not a risk.
A count still pending when the process exits is not written — the next start's
first occurrence begins a new count — which is the accepted cost of not adding
a timer thread to a logging path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

#: Long enough that a sustained fault costs one traceback a minute (bounded
#: bytes), short enough that a reader tailing the log sees it is still ongoing.
DEFAULT_INTERVAL_S = 60.0


class RepeatCollapse(logging.Filter):
    """Admit a repeating record once per interval; carry the suppressed count."""

    def __init__(
        self, interval: float = DEFAULT_INTERVAL_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        super().__init__()
        self.interval = interval
        self.clock = clock
        # key -> (time the last copy was written, occurrences suppressed since)
        self._seen: dict[tuple[str, int, str], tuple[float, int]] = {}
        # Loggers are called from any thread; the read-modify-write must be atomic.
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            first_line = record.getMessage().split("\n", 1)[0]
        except Exception:  # noqa: BLE001 — a malformed record is never ours to drop
            return True
        key = (record.name, record.levelno, first_line)
        now = self.clock()
        with self._lock:
            last = self._seen.get(key)
            if last is not None and now - last[0] < self.interval:
                self._seen[key] = (last[0], last[1] + 1)
                return False
            suppressed = last[1] if last is not None else 0
            self._seen[key] = (now, 0)
        if suppressed:
            # Appended to the rendered message (args folded in first) so a
            # %-format record cannot be broken by the suffix.
            record.msg = (
                f"{record.getMessage()} [{suppressed} similar suppressed in the last "
                f"{int(now - last[0]) if last else 0}s]"
            )
            record.args = None
        return True


def install(logger_name: str = "asyncio", interval: float = DEFAULT_INTERVAL_S) -> RepeatCollapse:
    """Attach one collapse filter to ``logger_name`` (idempotent per process)."""
    logger = logging.getLogger(logger_name)
    for existing in logger.filters:
        if isinstance(existing, RepeatCollapse):
            return existing
    collapse = RepeatCollapse(interval)
    logger.addFilter(collapse)
    return collapse
