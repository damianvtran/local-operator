"""The readiness helpers answer correctly on a descriptor above FD_SETSIZE.

Regression for PR #4472's CI run 45338434283 (job 138256298160): deep in an
xdist worker a fresh socketpair lands above fd 1024 and ``select.select`` raises
``ValueError: filedescriptor out of range in select()``. Each test here moves
its socket above that ceiling with ``F_DUPFD`` and proves the raw call still
fails there (the control: without it a low-fd run would pass vacuously) while
the replacements give the same answers ``select.select`` gives at a low fd.
"""

from __future__ import annotations

import fcntl
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


def _dup_high(fd: int) -> int:
    """Duplicate ``fd`` onto the lowest FREE descriptor at or above ``_HIGH_FD``.

    Why ``F_DUPFD`` and not ``os.dup2(fd, _HIGH_FD)``: ``dup2`` silently closes
    whatever already occupies its target. Deep in an xdist worker fd 1500 is
    often live (PR #4519's merge-gate run 45368594814 saw fresh sockets at
    fd 1801), so a fixed ``dup2`` stole another object's descriptor: that owner
    and the test then fought over one fd (EINVAL/EBADF here, BrokenPipe in
    whoever owned it). ``F_DUPFD`` never closes anything; the caller must use
    the fd it returns, which may be above ``_HIGH_FD``.
    """
    return fcntl.fcntl(fd, fcntl.F_DUPFD, _HIGH_FD)


@pytest.fixture
def high_socketpair() -> Iterator[tuple[socket.socket, socket.socket]]:
    """A connected pair whose first end sits at a free fd >= ``_HIGH_FD``."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = _HIGH_FD + 16
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} cannot reach fd {_HIGH_FD}")
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    low, peer = socket.socketpair()
    try:
        high = socket.socket(fileno=_dup_high(low.fileno()))
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


def test_high_fd_dup_leaves_a_live_descriptor_at_the_target_intact() -> None:
    """Regression for run 45368594814: another owner already holds ``_HIGH_FD``.

    The pipe stands in for that owner. The old fixed ``dup2`` is the negative
    control: it closes the owner's read end, so the owner's write breaks. The
    helper the fixture uses must land elsewhere and leave the owner working.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = _HIGH_FD + 16
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} cannot reach fd {_HIGH_FD}")
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    try:
        for clobber in (True, False):
            reader, writer = os.pipe()
            try:
                held = _dup_high(reader)  # the "other owner", at or above _HIGH_FD
                os.close(reader)
                low, peer = socket.socketpair()
                try:
                    if clobber:
                        os.dup2(low.fileno(), held)
                        mine = held
                    else:
                        mine = _dup_high(low.fileno())
                    try:
                        if clobber:
                            with pytest.raises(BrokenPipeError):
                                os.write(writer, b"owner")
                        else:
                            assert mine != held and mine >= _HIGH_FD
                            os.write(writer, b"owner")
                            assert os.read(held, 5) == b"owner"
                    finally:
                        os.close(mine)
                finally:
                    low.close()
                    peer.close()
                    if not clobber:
                        os.close(held)
            finally:
                os.close(writer)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
