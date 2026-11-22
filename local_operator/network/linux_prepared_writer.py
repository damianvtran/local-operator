"""Dormant, default-OFF Linux mechanics for ONE already-sealed record.

No production caller may use this primitive yet. It neither grants authority nor
implements a durable fence: PeerLink queue/handshake/close handoff, exhaustive
mutator participation, and production concurrent reader-close ownership remain
unproved. macOS/Windows are deliberately refused, not inferred from Linux proof.
The caller owns codec sequencing and must never reseal/reuse an abandoned nonce.
"""

from __future__ import annotations

import math
import selectors
import socket
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass


@dataclass(frozen=True)
class AttemptState:
    """Closed mechanics snapshot, NOT a permission or durable authority record.

    A future coordinator must hold its real shared exclusion scope from snapshot
    creation through the one send. A callback returning truthy is not that scope.
    """

    revision: int
    pending: bool = False
    cancelled: bool = False

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 0:
            raise TypeError("revision must be an exact nonnegative int")
        if type(self.pending) is not bool or type(self.cancelled) is not bool:
            raise TypeError("pending and cancelled must be exact bools")


# This interface supplies exclusion only. No production authority store or
# coordinated mutator implementation is selected or installed by this module.
# It must support a fresh enter/exit for EVERY attempt (not a one-shot generator).
AttemptBoundary = AbstractContextManager[AttemptState]


@dataclass(frozen=True)
class WriteResult:
    """Kernel acceptance, not peer receipt; pre-revocation bytes are not recalled."""

    complete: bool
    accepted_bytes: int
    reason: str


class _Abandon(Exception):
    pass


class LinuxPreparedRecordWriter:
    """Own a duplicate, never the original descriptor; refuse concurrent records.

    Python's positive-timeout send can poll even with MSG_DONTWAIT. Conversely,
    timeout=0 alone cannot defend against another wrapper changing the shared
    kernel mode. BOTH are needed: duplicate timeout=0 skips Python's poll and
    MSG_DONTWAIT keeps each Linux syscall nonblocking despite kernel-mode races.
    This does not establish safety of a future product reader-close handoff.
    """

    def __init__(
        self,
        original_socket: socket.socket,
        *,
        linux_prepared_writer_enabled: bool = False,
    ) -> None:
        if type(linux_prepared_writer_enabled) is not bool:
            raise TypeError("linux_prepared_writer_enabled must be an exact bool")
        if not linux_prepared_writer_enabled:
            raise RuntimeError("Linux prepared writer is disabled")
        if sys.platform != "linux":
            raise RuntimeError("Linux prepared writer requires Linux")
        timeout = original_socket.gettimeout()
        if timeout is None or timeout <= 0:
            raise ValueError("original socket must have a positive timeout")

        self._lock = threading.Lock()
        self._active = False
        self._closed = False
        self._cancelled = False
        self._duplicate = original_socket.dup()
        try:
            self._duplicate.settimeout(0)
            self._wake_read, self._wake_write = socket.socketpair()
            try:
                self._wake_read.setblocking(False)
                self._wake_write.setblocking(False)
            except BaseException:
                self._wake_read.close()
                self._wake_write.close()
                raise
        except BaseException:
            self._duplicate.close()
            raise

    def _release_locked(self) -> None:
        """Only idle/finishing owner closes fds; a waiter never sees fd reuse."""
        self._duplicate.close()
        self._wake_read.close()
        self._wake_write.close()

    def _shutdown_locked(self) -> None:
        # A duplicate close alone would leave other endpoint handles usable with
        # a truncated record / skipped nonce. Abandonment poisons the shared link.
        try:
            self._duplicate.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # Already disconnected still needs the same owner-only release.

    def _stop(self, *, cancel: bool) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled = cancel or self._active
            if self._cancelled:
                self._shutdown_locked()
            if self._active:
                # Wake select without closing/reusing any fd owned by the attempt.
                try:
                    self._wake_write.send(b"x", socket.MSG_DONTWAIT)
                except BlockingIOError:
                    pass  # A queued wake already makes select return.
            else:
                self._release_locked()

    def cancel(self) -> None:
        """Cancel/poison the shared endpoint; active owner performs final close."""
        self._stop(cancel=True)

    def close(self) -> None:
        """Idle success closes only our handles; an active record is abandoned."""
        self._stop(cancel=False)

    def write_record(
        self,
        sealed_record: bytes,
        *,
        expected_revision: int,
        boundary: AttemptBoundary,
        deadline: float,
    ) -> WriteResult:
        """Send immutable ciphertext once, revalidating every partial/EAGAIN retry.

        Deadline is supplied by the caller in monotonic seconds, not a product
        timeout policy. Invalid input/concurrent calls are refused before record
        acceptance. After acceptance EVERY unsuccessful exit shuts down the link.
        No lifecycle lock spans readiness or external boundary acquisition.
        """
        if type(sealed_record) is not bytes or not sealed_record:
            raise TypeError("sealed_record must be exact nonempty bytes")
        if type(expected_revision) is not int or expected_revision < 0:
            raise TypeError("expected_revision must be an exact nonnegative int")
        if type(deadline) not in (float, int) or not math.isfinite(deadline):
            raise TypeError("deadline must be finite monotonic seconds")
        with self._lock:
            if self._closed:
                raise RuntimeError("writer is closed")
            if self._active:
                raise RuntimeError("another record owns the writer")
            self._active = True

        accepted = 0
        complete = False
        wait_needed = False
        try:
            view = memoryview(sealed_record)
            while accepted < len(view):
                # First attempt is immediate. Every continuation waits OUTSIDE
                # both locks; the boundary must be re-entered even after EAGAIN.
                if wait_needed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _Abandon("deadline")
                    # Linux's default selector uses epoll, not fd_set: valid
                    # duplicate AND wake descriptors may exceed select's ceiling.
                    # Scope its owned kernel handle to this wait, outside both
                    # exclusion scopes, including on timeout/EINTR/abandonment.
                    with selectors.DefaultSelector() as readiness:
                        readiness.register(self._wake_read, selectors.EVENT_READ)
                        readiness.register(self._duplicate, selectors.EVENT_WRITE)
                        events = readiness.select(remaining)
                    # Cancellation wins even when both handles become ready.
                    if any(key.fileobj is self._wake_read for key, _ in events):
                        raise _Abandon("cancelled")
                    if not events:
                        # An interrupted/empty wait is not evidence of expiry.
                        # Recompute the caller's absolute deadline, without a
                        # polling interval or another send on stale readiness.
                        continue
                with boundary as state:
                    if type(state) is not AttemptState:
                        raise TypeError("boundary must yield exact AttemptState")
                    with self._lock:
                        if self._cancelled or state.cancelled:
                            raise _Abandon("cancelled")
                        if state.pending:
                            raise _Abandon("pending")
                        if state.revision != expected_revision:
                            raise _Abandon("revision")
                        if time.monotonic() >= deadline:
                            raise _Abandon("deadline")
                        try:
                            sent = self._duplicate.send(view[accepted:], socket.MSG_DONTWAIT)
                        except (BlockingIOError, InterruptedError):
                            wait_needed = True
                            continue
                        if sent == 0:
                            raise _Abandon("zero-send")
                        accepted += sent
                        wait_needed = True
            complete = True
            return WriteResult(True, accepted, "complete")
        except _Abandon as exc:
            return WriteResult(False, accepted, str(exc))
        except OSError:
            return WriteResult(False, accepted, "socket-error")
        finally:
            with self._lock:
                self._active = False
                if not complete:
                    self._closed = True
                    self._shutdown_locked()
                if self._closed:
                    self._release_locked()
