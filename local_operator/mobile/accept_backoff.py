"""Pause accepting with a bounded backoff when the daemon is out of descriptors.

WHY THIS EXISTS. When the phone daemon reaches its open-file limit, CPython's
selector event loop does not stop trying to accept. `_accept_connection`
(asyncio/selector_events.py, unchanged from 3.12 through 3.14) handles
EMFILE/ENFILE/ENOBUFS/ENOMEM by logging, removing the reader and scheduling a
restart in 1 s, but it then CONTINUES its ``for _ in range(backlog)`` loop.
uvicorn listens with a backlog of 2048, so one readiness wakeup produces up to
2048 failed accepts, 2048 logged tracebacks and 2048 scheduled restarts, and
each batch of restarts re-arms the next storm. Measured on a rig at a forced
64-descriptor limit: 100 % of a core, ~12,600 failed accepts per second and
~6.7 MB/s of log. That is the state behind the "Asleep / Not answering" phone
report: the daemon is alive, burning a core, and serving nobody.

WHAT THIS DOES. `BackoffSelectorEventLoop` keeps the stdlib accept path for
every successful accept (handing the socket to the stdlib's own
`_accept_connection2`), and changes only what happens on a resource error: it
stops the batch, stops watching the listening socket, and retries after a
delay that starts at `INITIAL_DELAY_S` and doubles up to `MAX_DELAY_S`; the
first success resets it. One episode is logged once, through the loop's
exception handler with the stdlib's own message (so `lop mobile doctor`'s log
scan still counts it), and the recovery is logged once with its duration.

THE INVARIANTS:

* No connection is ever dropped. A paused listener accepts nothing, so pending
  connections stay in the kernel's backlog and are accepted when the pause
  ends; nothing is accepted-and-closed to shed load (libuv's "EMFILE trick"
  does that, deliberately not copied here).
* Accepting resumes within `MAX_DELAY_S` of descriptors becoming available.
* While exhausted, the listener costs at most one failed accept() per retry
  delay — at the ceiling, one syscall per second instead of a busy core.

HOW, AND WHAT IT LEANS ON. The override wraps the listening socket in a proxy
whose ``accept()`` turns a resource error into ``BlockingIOError``, which the
stdlib loop already treats as "nothing left to accept" and returns on. It thus
relies only on `_accept_connection`/`_start_serving` taking
``(protocol_factory, sock, *rest)`` and calling ``sock.accept()`` — stable since
3.7. If a future CPython changes that, the proxy is simply not consulted and
the stdlib behaviour returns; nothing breaks.

Windows is untouched: its default proactor loop accepts through IOCP, not this
code path, so `loop_factory` returns the platform default there.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

#: The errors CPython's accept path treats as "out of a system resource".
RESOURCE_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ENOMEM})
#: First retry: soon enough that a momentary spike (a burst of phone reconnects
#: briefly at the limit) costs well under a second of unavailability.
INITIAL_DELAY_S = 0.05
#: Ceiling: the stdlib's own intended pause (`asyncio.constants.ACCEPT_RETRY_DELAY`).
#: A failed retry is one accept() syscall, so the ceiling bounds recovery
#: latency rather than CPU: at one retry a second the cost is negligible.
MAX_DELAY_S = 1.0


class _GuardedListener:
    """The listening socket, with resource errors reported as "nothing to accept".

    Records what happened so the loop can decide, after the stdlib batch
    returns, whether to pause (``exhausted``) or reset the backoff (``accepted``).
    """

    __slots__ = ("_sock", "exhausted", "accepted")

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self.exhausted: OSError | None = None
        self.accepted = 0

    def accept(self) -> tuple[socket.socket, Any]:
        try:
            pair = self._sock.accept()
        except OSError as exc:
            if exc.errno in RESOURCE_ERRNOS:
                self.exhausted = exc
                raise BlockingIOError(errno.EAGAIN, "paused: out of descriptors") from exc
            raise
        self.accepted += 1
        return pair

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)


class _Pause:
    __slots__ = ("delay", "since", "retries")

    def __init__(self, now: float) -> None:
        self.delay = INITIAL_DELAY_S
        self.since = now
        self.retries = 0


class BackoffSelectorEventLoop(asyncio.SelectorEventLoop):
    """A selector loop whose listeners back off, rather than spin, on EMFILE."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Per listening socket (by fd): the pause in progress, if any.
        self._accept_pauses: dict[int, _Pause] = {}

    def _accept_connection(  # type: ignore[override]
        self, protocol_factory: Any, sock: socket.socket, *rest: Any, **kwargs: Any
    ) -> None:
        guarded = _GuardedListener(sock)
        # The stdlib accept batch, against the proxy (private API; see the module docstring).
        accept_batch: Any = super()._accept_connection  # type: ignore[misc]
        accept_batch(protocol_factory, guarded, *rest, **kwargs)
        fd = sock.fileno()
        pause = self._accept_pauses.get(fd)
        if guarded.exhausted is None:
            if pause is not None and guarded.accepted:
                self._accept_pauses.pop(fd, None)
                logger.warning(
                    "accepting connections again after %.1f s paused for lack of file "
                    "descriptors (%d retries)",
                    time.monotonic() - pause.since,
                    pause.retries,
                )
            return
        if pause is None:
            pause = self._accept_pauses[fd] = _Pause(time.monotonic())
            # Once per episode, with the stdlib's own message and exception, so
            # anything reading the log for this state (the doctor) still finds it.
            self.call_exception_handler(
                {
                    "message": "socket.accept() out of system resource",
                    "exception": guarded.exhausted,
                    "socket": sock,
                }
            )
        else:
            pause.retries += 1
            pause.delay = min(pause.delay * 2, MAX_DELAY_S)
        self._remove_reader(fd)  # type: ignore[attr-defined]  # private, stable since 3.4
        self.call_later(
            pause.delay, self._resume_accepting, fd, protocol_factory, sock, rest, kwargs
        )

    def _resume_accepting(
        self,
        fd: int,
        protocol_factory: Any,
        sock: socket.socket,
        rest: tuple[Any, ...],
        kwargs: Any,
    ) -> None:
        # The server may have closed while paused: re-arming a closed socket
        # would raise inside the loop (or watch a recycled fd number).
        if sock.fileno() != fd:
            self._accept_pauses.pop(fd, None)
            return
        self._start_serving(protocol_factory, sock, *rest, **kwargs)  # type: ignore[attr-defined]


def loop_factory() -> asyncio.AbstractEventLoop:
    """The daemon's event loop: the backoff loop on POSIX, the default elsewhere."""
    if sys.platform == "win32":
        return asyncio.new_event_loop()
    return BackoffSelectorEventLoop()
