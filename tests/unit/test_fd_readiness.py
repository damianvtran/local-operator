"""The readiness helpers answer correctly on a descriptor above FD_SETSIZE.

Regression for PR #4472's CI run 45338434283 (job 138256298160): deep in an
xdist worker a fresh socketpair lands above fd 1024 and ``select.select`` raises
``ValueError: filedescriptor out of range in select()``. Each test here moves
its socket above that ceiling with ``os.dup2`` and proves the raw call still
fails there (the control: without it a low-fd run would pass vacuously) while
the replacements give the same answers ``select.select`` gives at a low fd.
"""

from __future__ import annotations

import os
import resource
import select
import socket
from collections.abc import Iterator

import pytest

from local_operator.tui import host_theme
from tests import fd_readiness

#: Comfortably above select(2)'s FD_SETSIZE (1024 on Linux and macOS).
_HIGH_FD = 1500


def _require_low(fd: int) -> None:
    """Skip a low-fd semantics test in a process already past the ceiling.

    Those tests pin what the helpers do BELOW 1024; deep in a leaky worker the
    next descriptor may already be above it, and the precondition is simply
    unavailable there (the high-fd tests above cover that side).
    """
    if fd >= 1024:
        pytest.skip(f"no descriptor below 1024 available (got {fd})")


@pytest.fixture
def high_socketpair() -> Iterator[tuple[socket.socket, socket.socket]]:
    """A connected pair whose first end sits at fd ``_HIGH_FD``."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = _HIGH_FD + 16
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} cannot reach fd {_HIGH_FD}")
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    low, peer = socket.socketpair()
    try:
        os.dup2(low.fileno(), _HIGH_FD)
        high = socket.socket(fileno=_HIGH_FD)
        low.close()
        try:
            yield high, peer
        finally:
            high.close()
    finally:
        peer.close()
        low.close()
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_the_raw_call_really_fails_at_that_height(high_socketpair) -> None:
    high, _peer = high_socketpair
    assert high.fileno() >= 1024
    with pytest.raises(ValueError, match="out of range"):
        select.select([high], [], [], 0)


def test_readable_matches_select_semantics_above_the_ceiling(high_socketpair) -> None:
    high, peer = high_socketpair
    assert not fd_readiness.is_readable(high, 0)
    assert fd_readiness.readable([high], 0) == []
    peer.send(b"x")
    assert fd_readiness.is_readable(high, 5)
    assert fd_readiness.readable([high], 5) == [high]


def test_writable_matches_select_semantics_above_the_ceiling(high_socketpair) -> None:
    high, _peer = high_socketpair
    assert fd_readiness.is_writable(high, 0)
    high.setblocking(False)
    with pytest.raises(BlockingIOError):
        while True:
            high.send(b"x" * 65536)
    assert not fd_readiness.is_writable(high, 0.02)


def test_fifo_eof_keeps_selects_answer_below_the_ceiling(tmp_path) -> None:
    """Below 1024 the helper must answer exactly as ``select`` did.

    macOS reports a FIFO's EOF (last writer closed) only through ``select``;
    poll and kqueue stay silent. The osworld adapter's
    ``test_hard_reset_cancellation_ends_exact_probe_and_helper`` relies on that
    EOF, so a helper using a selector everywhere would change its meaning.
    """
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    try:
        _require_low(reader)
        assert not fd_readiness.is_readable(reader, 0)
        os.close(writer)
        writer = -1
        assert select.select([reader], [], [], 1)[0] == [reader]
        assert fd_readiness.is_readable(reader, 1)
        assert os.read(reader, 1) == b""
    finally:
        if writer >= 0:
            os.close(writer)
        os.close(reader)


def test_host_theme_probe_wait_survives_a_high_fd(high_socketpair) -> None:
    """``probe`` opens /dev/tty fresh; a high fd must read as data, not crash."""
    high, peer = high_socketpair
    assert not host_theme._readable_within(high.fileno(), 0)
    peer.send(b"\x1b[?62c")
    assert host_theme._readable_within(high.fileno(), 5)


def test_host_theme_low_fd_path_is_unchanged() -> None:
    reader, writer = socket.socketpair()
    with reader, writer:
        _require_low(reader.fileno())
        assert not host_theme._readable_within(reader.fileno(), 0)
        writer.send(b"x")
        assert host_theme._readable_within(reader.fileno(), 5)
