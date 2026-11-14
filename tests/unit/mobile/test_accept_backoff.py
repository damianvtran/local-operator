"""At the descriptor limit the daemon's listener backs off instead of spinning.

The invariants under test (see `mobile/accept_backoff.py`): while accept()
fails with EMFILE the listener costs a bounded number of attempts and one
logged record per episode; a connection that arrives meanwhile is NOT dropped
and is accepted within `MAX_DELAY_S` of descriptors becoming available; a
server closed during a pause is not re-armed.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import sys
import time
from asyncio import trsock
from typing import Any

import pytest

from local_operator.mobile import accept_backoff

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="selector loop path only")


class _FlakyListener(socket.socket):
    """A real listening socket whose accept() reports EMFILE while ``exhausted``.

    With ``abort_next`` set, the next accept() plays a client that gave up
    before being accepted: it takes the connection off the backlog and reports
    ``ConnectionAbortedError``, so the following accept() sees EAGAIN.
    """

    exhausted = True
    abort_next = False
    calls = 0

    def accept(self):  # type: ignore[override]
        self.calls += 1
        if self.exhausted:
            raise OSError(errno.EMFILE, "Too many open files")
        if self.abort_next:
            self.abort_next = False
            conn, _addr = super().accept()
            conn.close()
            raise ConnectionAbortedError(errno.ECONNABORTED, "Software caused connection abort")
        return super().accept()


class _Proto(asyncio.Protocol):
    made: list[float] = []

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        _Proto.made.append(time.monotonic())


def _listener() -> _FlakyListener:
    sock = _FlakyListener(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(100)
    sock.setblocking(False)
    return sock


def _serve(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> asyncio.base_events.Server:
    _Proto.made = []
    return loop.run_until_complete(loop.create_server(_Proto, sock=sock, backlog=100))


def _connect(sock: socket.socket) -> socket.socket:
    client = socket.create_connection(sock.getsockname(), timeout=2)
    return client


def _run(loop: asyncio.AbstractEventLoop, seconds: float) -> None:
    loop.run_until_complete(asyncio.sleep(seconds))


def _handler_calls(loop: asyncio.AbstractEventLoop) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: calls.append(context))
    return calls


def test_the_stock_loop_spins_through_its_backlog_on_emfile():
    """WHY the override exists: one wakeup = one failure per backlog slot."""
    loop = asyncio.SelectorEventLoop()
    try:
        calls = _handler_calls(loop)
        sock = _listener()
        server = _serve(loop, sock)
        client = _connect(sock)
        _run(loop, 0.3)
        assert sock.calls >= 100 and len(calls) >= 100
        client.close()
        server.close()
        loop.run_until_complete(server.wait_closed())
    finally:
        loop.close()


def test_an_exhausted_listener_backs_off_and_logs_one_episode():
    loop = accept_backoff.BackoffSelectorEventLoop()
    try:
        calls = _handler_calls(loop)
        sock = _listener()
        server = _serve(loop, sock)
        client = _connect(sock)
        _run(loop, 1.6)
        # 0.05 + 0.1 + 0.2 + 0.4 + 0.8 s of pauses: at most ~6 attempts in 1.6 s.
        assert 1 <= sock.calls <= 7, sock.calls
        assert len(calls) == 1
        assert calls[0]["message"] == "socket.accept() out of system resource"
        assert calls[0]["exception"].errno == errno.EMFILE
        # Stdlib parity: handlers get the read-only wrapper, never the listener.
        assert isinstance(calls[0]["socket"], trsock.TransportSocket)
        assert _Proto.made == []  # nothing accepted, and nothing accepted-and-closed
        client.close()
        server.close()
        loop.run_until_complete(server.wait_closed())
    finally:
        loop.close()


def test_a_waiting_connection_is_accepted_once_descriptors_free(caplog):
    loop = accept_backoff.BackoffSelectorEventLoop()
    try:
        _handler_calls(loop)
        sock = _listener()
        server = _serve(loop, sock)
        client = _connect(sock)  # arrives while exhausted: waits in the backlog
        _run(loop, 3.0)  # long enough to reach the ceiling
        assert _Proto.made == []
        freed = time.monotonic()
        sock.exhausted = False
        with caplog.at_level(logging.WARNING, logger=accept_backoff.__name__):
            deadline = freed + accept_backoff.MAX_DELAY_S + 0.5
            while not _Proto.made and time.monotonic() < deadline:
                _run(loop, 0.02)
        assert _Proto.made, "the waiting connection was never accepted"
        assert _Proto.made[0] - freed <= accept_backoff.MAX_DELAY_S + 0.1
        assert any("accepting connections again" in r.getMessage() for r in caplog.records)
        assert loop._accept_pauses == {}  # type: ignore[attr-defined]
        client.close()
        server.close()
        loop.run_until_complete(server.wait_closed())
    finally:
        loop.close()


def test_a_resume_that_accepts_nothing_still_ends_the_episode(caplog):
    """Regression (#3954 review R1): a resume wakeup whose waiting client
    aborted accepts nothing; the pause must still end, or the next episode is
    never logged and the recovery line reports the wrong duration."""
    loop = accept_backoff.BackoffSelectorEventLoop()
    try:
        calls = _handler_calls(loop)
        sock = _listener()
        server = _serve(loop, sock)
        client = _connect(sock)
        _run(loop, 0.02)  # first episode: logged, pause pending
        assert len(calls) == 1
        assert loop._accept_pauses  # type: ignore[attr-defined]
        sock.exhausted = False
        sock.abort_next = True
        with caplog.at_level(logging.WARNING, logger=accept_backoff.__name__):
            _run(loop, accept_backoff.MAX_DELAY_S + 0.2)  # the resume fires
        assert not sock.abort_next, "the resume never reached accept()"
        assert loop._accept_pauses == {}  # type: ignore[attr-defined]
        assert not any("accepting connections again" in r.getMessage() for r in caplog.records)
        # The consequence the stale entry hid: a later episode is logged again.
        sock.exhausted = True
        later = _connect(sock)
        _run(loop, 0.02)
        assert len(calls) == 2, calls
        later.close()
        client.close()
        server.close()
        loop.run_until_complete(server.wait_closed())
    finally:
        loop.close()


def test_the_accept_proxy_forwards_every_other_attribute():
    """Only accept() is intercepted (#3954 review R2): if a future stdlib batch
    touches another socket attribute, it must reach the real socket instead
    of raising AttributeError on every wakeup."""
    sock = _listener()
    try:
        guarded = accept_backoff._GuardedListener(sock)
        assert guarded.fileno() == sock.fileno()
        assert guarded.getsockname() == sock.getsockname()
        assert guarded.family == sock.family
        with pytest.raises(BlockingIOError):
            guarded.accept()  # the resource error becomes "nothing to accept"
        assert guarded.exhausted is not None and guarded.exhausted.errno == errno.EMFILE
    finally:
        sock.close()


def test_a_server_closed_while_paused_is_not_rearmed():
    loop = accept_backoff.BackoffSelectorEventLoop()
    try:
        calls = _handler_calls(loop)
        sock = _listener()
        server = _serve(loop, sock)
        client = _connect(sock)
        _run(loop, 0.02)
        server.close()
        loop.run_until_complete(server.wait_closed())
        _run(loop, 1.5)  # every pending resume fires against a closed socket
        assert len(calls) == 1, calls  # no "Exception in callback" from a re-arm
        assert loop._accept_pauses == {}  # type: ignore[attr-defined]
        client.close()
    finally:
        loop.close()


def test_the_daemon_loop_factory_uses_the_backoff_loop():
    loop = accept_backoff.loop_factory()
    try:
        assert isinstance(loop, accept_backoff.BackoffSelectorEventLoop)
    finally:
        loop.close()
