"""Serve-only isolation from a desktop stderr pipe that may stop draining.

Do not use QueueHandler.prepare: it formats exceptions on the producer. Only
this daemon consumer formats/writes; it is deliberately NOT a logging Handler,
so logging.shutdown cannot acquire its sink lock or flush a wedged stream.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque
from typing import TextIO

from local_operator.logger import DEFAULT_LOG_FORMAT, _get_log_level

# A 1,000-record ledger-warning sample (Python 3.12) measured median 1,052 / max
# 1,108 shallow bytes per record. 1,024 pending is about 1.1 MiB for that sample.
# The invariant is RECORD COUNT, not a byte guarantee: arbitrary logging args and
# exception tracebacks can retain larger graphs. Half the slots are reserved for
# WARNING+, so an INFO flood cannot displace all diagnostic records.
SERVE_LOG_CAPACITY = 1024


class ServeLogHandler(logging.Handler):
    """Bounded best-effort queue; no producer ever takes the sink's lock.

    Overflow drops oldest low-priority records first; when only warnings/errors
    remain, their oldest record is dropped. Counters remain inspectable even if
    stderr never recovers. Summaries use the consumer, never a fallback write on
    the event loop. The queue mutex protects only deque/counter operations.
    """

    def __init__(self, stream: TextIO, capacity: int = SERVE_LOG_CAPACITY) -> None:
        if capacity < 2:
            raise ValueError("serve log capacity must be at least two")
        # Keep Handler's standard producer lock (required by Python 3.14).
        # It protects only bounded enqueue work: the consumer never acquires it,
        # and inherited flush() is a no-op, so shutdown cannot wait on sink I/O.
        super().__init__()
        self.capacity = capacity
        self._normal: deque[logging.LogRecord] = deque()
        self._important: deque[logging.LogRecord] = deque()
        self._condition = threading.Condition()
        self._stopped = False
        self.dropped = 0
        self.sink_errors = 0
        self._stream = stream
        self._formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        # Do not hold sys.stderr's TextIOWrapper lock across a blocked pipe
        # write: interpreter shutdown flushes that wrapper even for daemons.
        # A private fd also keeps close() from racing the caller's descriptor.
        try:
            self._fd: int | None = os.dup(stream.fileno())
        except (AttributeError, OSError, ValueError):
            self._fd = None
        self._worker = threading.Thread(target=self._consume, name="serve-log-sink", daemon=True)
        self._worker.start()

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def pending(self) -> int:
        with self._condition:
            return len(self._normal) + len(self._important)

    def emit(self, record: logging.LogRecord) -> None:
        with self._condition:
            if self._stopped:
                self.dropped += 1
                return
            important = record.levelno >= logging.WARNING
            if not important and len(self._normal) >= self.capacity // 2:
                self._normal.popleft()
                self.dropped += 1
            elif len(self._normal) + len(self._important) >= self.capacity:
                (self._normal if self._normal else self._important).popleft()
                self.dropped += 1
            (self._important if important else self._normal).append(record)
            self._condition.notify()

    def _write(self, text: str) -> None:
        if self._fd is None:
            # EMFILE and streams without fileno must never fall back to the
            # caller's wrapper: its lock is flushed by interpreter shutdown.
            # The consumer accounts for the lost record without logging again.
            raise OSError("private serve logging descriptor unavailable")
        data = text.encode(
            getattr(self._stream, "encoding", None) or "utf-8", errors="backslashreplace"
        )
        while data:
            data = data[os.write(self._fd, data) :]

    def _consume(self) -> None:
        reported = 0
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stopped or bool(self._important) or bool(self._normal)
                    )
                    if self._stopped:
                        return
                    record = (self._important if self._important else self._normal).popleft()
                    dropped = self.dropped
                try:
                    if dropped != reported:
                        self._write(
                            # i18n: ignore developer-only logging sink diagnostic
                            f"serve logging: dropped {dropped} records "
                            "(overflow/shutdown/sink failure)\n"
                        )
                        reported = dropped
                    self._write(self._formatter.format(record) + "\n")
                except Exception:  # logging must never recursively report a failed sink
                    with self._condition:
                        self.sink_errors += 1
                        self.dropped += 1
        finally:
            if self._fd is not None:
                os.close(self._fd)

    def close(self) -> None:
        # No join, flush, or sink close here: a consumer may be stuck forever in
        # os.write. Its private fd is closed by the worker if/when it returns.
        with self._condition:
            self._stopped = True
            self.dropped += len(self._normal) + len(self._important)
            self._normal.clear()
            self._important.clear()
            self._condition.notify()
        super().close()


def configure_serve_logging() -> ServeLogHandler:
    """Install before serve imports; repeat in lifespan after uvicorn config.

    Reuse the live handler across the two boundaries. Named uvicorn handlers
    must also be isolated: uvicorn installs direct stderr handlers of its own.
    Closed queues stay attached through final shutdown diagnostics (best effort
    means these can be dropped), never restoring a blocking handler on exit.
    Other CLI entry points continue using configure_console_logging unchanged.
    """
    root = logging.getLogger()
    handler = next(
        (h for h in root.handlers if isinstance(h, ServeLogHandler) and not h.stopped), None
    )
    if handler is None:
        handler = ServeLogHandler(sys.stderr)
    root.handlers = [handler]
    root.setLevel(_get_log_level())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        named = logging.getLogger(name)
        named.handlers = [handler]
        named.setLevel(logging.INFO)  # uvicorn's default, formerly set by dictConfig
        named.propagate = False
    return handler
