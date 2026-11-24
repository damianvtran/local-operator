"""Descriptor readiness checks that work at ANY descriptor number.

Why this exists: ``select.select`` builds an ``fd_set``, which holds descriptors
below ``FD_SETSIZE`` (1024 on Linux and macOS) and raises ``ValueError:
filedescriptor out of range in select()`` for anything higher. An xdist worker
runs thousands of tests in one process; when earlier tests leave descriptors
open, a later test's fresh ``socketpair()`` lands above 1024 and its readiness
assertion errors out before it checks anything. PR #4472's CI run 45338434283
(job 138256298160, shard ``test (3.13, 1)``, worker gw0) lost seven tests of
``tests/unit/network/test_linux_prepared_writer.py`` exactly this way.

So the helpers keep ``select.select`` -- byte-for-byte the call the tests
made before -- whenever every descriptor is below the ceiling, and switch to
``selectors.DefaultSelector`` (epoll on Linux, kqueue on macOS), which has no
ceiling, only above it. Why not the selector for everything: readiness is not
identical across mechanisms. Measured on macOS 2026-11-24: once the last writer
of a FIFO closes, ``select`` reports the read end readable (EOF) while
``poll`` and kqueue report nothing, and
``test_evaluator_dependencies.py::test_hard_reset_cancellation_ends_exact_probe_and_helper``
asserts exactly that EOF. Below the ceiling the old answer is therefore kept
exactly; above it the old call could only raise, so any answer is new. On Linux
epoll reports FIFO EOF as ``EPOLLHUP``, which ``selectors`` maps to readable.
Same split as ``local_operator.providers.auth_cli._readable_within``.

Semantics match ``bool(select.select(...)[i])``: a zero timeout is a
non-blocking poll, and a positive timeout waits at most that long. epoll
refuses regular files (``EPERM``); every caller passes a socket, pipe, FIFO or
tty, which is what readiness means anyway.

``tests/unit/test_no_select_select.py`` fails on any new in-process
``select.select(`` so the ceiling cannot creep back in.
"""

from __future__ import annotations

import select
import selectors
from collections.abc import Sequence
from typing import Any

#: select(2)'s FD_SETSIZE on Linux and macOS; CPython does not export it.
_SELECT_FD_LIMIT = 1024


def _fileno(fileobj: Any) -> int:
    return fileobj if isinstance(fileobj, int) else fileobj.fileno()


def _ready(fileobjs: Sequence[Any], events: int, timeout: float) -> list[Any]:
    """The subset of ``fileobjs`` ready for ``events`` within ``timeout`` seconds."""
    if all(_fileno(fileobj) < _SELECT_FD_LIMIT for fileobj in fileobjs):
        if events == selectors.EVENT_READ:
            return list(select.select(fileobjs, [], [], timeout)[0])
        return list(select.select([], fileobjs, [], timeout)[1])
    with selectors.DefaultSelector() as selector:
        for fileobj in fileobjs:
            selector.register(fileobj, events)
        return [key.fileobj for key, _mask in selector.select(timeout)]


def readable(fileobjs: Sequence[Any], timeout: float) -> list[Any]:
    """``select.select(fileobjs, [], [], timeout)[0]`` without the 1024 ceiling."""
    return _ready(fileobjs, selectors.EVENT_READ, timeout)


def is_readable(fileobj: Any, timeout: float) -> bool:
    """Whether one socket/pipe/FIFO/file object becomes readable within ``timeout``."""
    return bool(readable([fileobj], timeout))


def is_writable(fileobj: Any, timeout: float) -> bool:
    """Whether one socket/pipe/FIFO/file object becomes writable within ``timeout``."""
    return bool(_ready([fileobj], selectors.EVENT_WRITE, timeout))
