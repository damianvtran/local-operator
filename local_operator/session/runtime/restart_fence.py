"""Cross-process admission fencing for a member-initiated relay restart.

A relay and its session runtimes are different processes, so a threading lock
cannot order a new turn against the member's final idle check. This module uses
one permanent per-config-root lock inode instead: turn admission takes a SHARED
lock while it publishes its busy record; relay recovery takes an EXCLUSIVE lock
across the final registry read and the supervisor call. Shared admissions may
coexist. An exclusive holder makes later admissions wait until the restart
decision has completed (or the relay dies and the kernel releases its lock).

The lock file is never removed: unlinking a stale-looking inode would let two
holders serialize against different files. POSIX ``flock`` locks each open file
description, which also gives the required exclusion between threads in one
process. Windows has no compatible shared-lock primitive here; turn admission
continues without this fence and peer restart refuses rather than claiming the
same atomicity. The existing local-only service surface is unchanged.
"""

from __future__ import annotations

import asyncio
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from local_operator.paths import config_dir

FENCE_NAME = "relay-restart-admission.lock"


class RestartFenceError(RuntimeError):
    """The fence cannot prove the required admission ordering."""


class AdmissionConflict(RestartFenceError):
    """A shared admission is still settling inside the bounded wait."""


def _open(root: Path | None) -> int:
    if os.name != "posix":
        raise RestartFenceError("peer relay restart requires POSIX admission fencing")
    base = (root or config_dir()).expanduser().resolve()
    directory = base / "run"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / FENCE_NAME
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise RestartFenceError(f"unsafe relay restart admission fence: {path}")
        # The permanent inode is not itself authority; the kernel lock on this
        # description is. fchmod also repairs an older umask without changing
        # which inode every cooperating process opens.
        os.fchmod(descriptor, 0o600)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _flock(descriptor: int, operation: int) -> None:
    import fcntl

    fcntl.flock(descriptor, operation)


async def acquire_shared(root: Path | None) -> int | None:
    """Hold shared admission until :func:`release_shared`, without blocking the loop.

    The common path is one non-blocking syscall. An actual restart in progress is
    polled asynchronously rather than sent to a worker thread: cancellation then
    closes this coroutine's descriptor without stranding a lock in a thread.
    """
    if os.name != "posix":
        return None
    descriptor = _open(root)
    try:
        import fcntl

        while True:
            try:
                _flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                return descriptor
            except BlockingIOError:
                await asyncio.sleep(0.02)
    except BaseException:
        os.close(descriptor)
        raise


def release_shared(descriptor: int | None) -> None:
    """Close the description; process death has the same kernel release semantics."""
    if descriptor is not None:
        os.close(descriptor)


@contextmanager
def shared(root: Path | None) -> Iterator[None]:
    """Synchronous shared admission for the registry's busy-record write."""
    if os.name != "posix":
        yield
        return
    descriptor = _open(root)
    try:
        import fcntl

        _flock(descriptor, fcntl.LOCK_SH)
    except BaseException:
        os.close(descriptor)
        raise
    try:
        yield
    finally:
        os.close(descriptor)


@contextmanager
def exclusive(root: Path | None, *, timeout_s: float) -> Iterator[None]:
    """Exclude admissions through the final idle read and supervisor decision.

    A short bound lets an admission already inside its critical section publish
    busy and leave. Expiry is a busy outcome, never permission to restart.
    """
    descriptor = _open(root)
    import fcntl

    deadline = time.monotonic() + max(0.0, timeout_s)
    try:
        while True:
            try:
                _flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdmissionConflict("a turn admission is still settling") from None
                time.sleep(min(0.02, remaining))
    except BaseException:
        os.close(descriptor)
        raise
    try:
        yield
    finally:
        os.close(descriptor)
