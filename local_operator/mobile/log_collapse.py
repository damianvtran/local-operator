"""Collapse a repeating log record so an out-of-files daemon cannot fill the disk.

WHY THIS EXISTS. When the phone daemon reaches its open-file limit, asyncio's
accept loop fails on every readiness wakeup and its default exception handler
logs ``socket.accept() out of system resource`` WITH a full traceback each
time. Measured on a rig at a forced 64-descriptor limit: ~7 MB/s into
``mobile.log`` (1.78 GB in about four minutes). That is the same failure path
as the "Asleep / Not answering" report — an fd-starved daemon — and filling the
disk on top of it turns one broken service into a broken machine.

THE INVARIANT this filter keeps: a given message (same logger, level, first
line AND attached exception — its type and text) is written AT MOST ONCE PER
``interval`` seconds, and no occurrence is lost from the record — the
suppressed count is carried onto the next copy that is written ("… [N similar
suppressed in the last Ts]"). A reader therefore still sees that it kept
happening and how often, which is what `lop mobile doctor`'s log scan counts
on; only the identical tracebacks are dropped.

WHY THE EXCEPTION IS PART OF THE KEY (agent review round 1, R1-2). asyncio's
default exception handler writes a CONSTANT first line ("Task exception was
never retrieved", "socket.accept() out of system resource") and puts the cause
in ``exc_info``. Keyed on the first line alone, a second task's different
exception, or an ENFILE right after an EMFILE, was swallowed into the first
one's count and its traceback never written. With the exception in the key the
measured flood (the same ``OSError(24)`` thousands of times a second) still
collapses, and every distinct fault still gets its first copy.

Scope is deliberately narrow: it is installed on the ``asyncio`` logger only
(`install`), whose records are loop-level faults with no per-request content.
Since the accept backoff (#3954) removes the flood at its source, this filter is
defence in depth: it only has to keep a future flood from filling the disk.

BOUNDED MEMORY (R1-5): the key table is pruned once it passes
:data:`MAX_KEYS`, dropping keys whose last copy is older than ``interval`` and
that have nothing suppressed — forgetting those loses no count, and the next
occurrence is simply written as a first copy, which it would be anyway.
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
#: Prune the key table past this many keys. A loop-level fault set is small (a
#: handful of distinct messages per process); this is far above it and only
#: stops a stream of one-off messages ("Exception in callback <Handle …>" with a
#: varying repr) from growing the table for the daemon's lifetime.
MAX_KEYS = 256

_Key = tuple[str, int, str, str, str]


def _key(record: logging.LogRecord, first_line: str) -> _Key:
    """The identity of a repeat: logger, level, first line and its exception."""
    exc = record.exc_info[1] if record.exc_info else None
    if exc is None:
        return (record.name, record.levelno, first_line, "", "")
    try:
        detail = str(exc)
    except Exception:  # noqa: BLE001 — a broken __str__ must not drop the record
        detail = ""
    return (record.name, record.levelno, first_line, type(exc).__name__, detail)


class RepeatCollapse(logging.Filter):
    """Admit a repeating record once per interval; carry the suppressed count."""

    def __init__(
        self, interval: float = DEFAULT_INTERVAL_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        super().__init__()
        self.interval = interval
        self.clock = clock
        # key -> (time the last copy was written, occurrences suppressed since)
        self._seen: dict[_Key, tuple[float, int]] = {}
        # Loggers are called from any thread; the read-modify-write must be atomic.
        self._lock = threading.Lock()
        # When the table was last pruned. A prune scans every key, so it runs at
        # most once per interval: a key only becomes prunable after `interval`
        # anyway, and a burst of FRESH distinct keys (none prunable yet) must not
        # turn every record into a full scan.
        self._pruned_at = float("-inf")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            first_line = record.getMessage().split("\n", 1)[0]
        except Exception:  # noqa: BLE001 — a malformed record is never ours to drop
            return True
        key = _key(record, first_line)
        now = self.clock()
        with self._lock:
            if len(self._seen) > MAX_KEYS and now - self._pruned_at >= self.interval:
                self._prune(now)
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

    def _prune(self, now: float) -> None:
        """Drop stale keys holding no count (caller holds the lock)."""
        self._pruned_at = now
        for key, (written, suppressed) in list(self._seen.items()):
            if suppressed == 0 and now - written >= self.interval:
                del self._seen[key]


def install(logger_name: str = "asyncio", interval: float = DEFAULT_INTERVAL_S) -> RepeatCollapse:
    """Attach one collapse filter to ``logger_name`` (idempotent per process)."""
    logger = logging.getLogger(logger_name)
    for existing in logger.filters:
        if isinstance(existing, RepeatCollapse):
            return existing
    collapse = RepeatCollapse(interval)
    logger.addFilter(collapse)
    return collapse
