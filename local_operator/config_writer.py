"""Bounded serialization for the EXISTING config read/compare/write primitives.

Atomic rename prevents torn bytes, not lost updates: two compliant writers can
read the same config and each replace it with a different subtree. This lock
spans the fresh read and mutation, is keyed by canonical CONFIG root (never the
shared install root), and nests only on the owning thread. Callbacks, provider
work, installation transitions and network calls belong outside this context.
External editors do not honor advisory locks; this is harness serialization,
not a claim that arbitrary editors participate in compare-and-swap.
"""

from __future__ import annotations

import errno
import functools
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, ParamSpec, TypeVar

DEFAULT_CONFIG_LOCK_WAIT_S = 5.0
_LOCK_RETRY_S = 0.02
P = ParamSpec("P")
T = TypeVar("T")


class ConfigWriteLockTimeout(TimeoutError):
    code = "config-write-busy"


class ConfigWriteLockError(OSError):
    code = "config-write-unsafe"


_registry_guard = threading.Lock()
_thread_locks: dict[Path, threading.RLock] = {}
_owned = threading.local()
_lock_fds: set[int] = set()


def _before_fork() -> None:
    _registry_guard.acquire()


def _after_fork_parent() -> None:
    _registry_guard.release()


def _after_fork() -> None:
    # A forked child must not inherit an in-process lock owned by a thread that
    # exists only in its parent. The parent's kernel lock remains authoritative.
    global _registry_guard, _thread_locks, _owned, _lock_fds
    # Close inherited references without unlocking the parent's shared open
    # description. Otherwise the child can keep a departed parent's lease alive.
    for fd in _lock_fds:
        os.close(fd)
    _registry_guard = threading.Lock()
    _thread_locks = {}
    _owned = threading.local()
    _lock_fds = set()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork, after_in_parent=_after_fork_parent, after_in_child=_after_fork
    )


def _try_lock(fd: int) -> bool:
    if os.name == "nt":  # pragma: no cover - exercised by the Windows probe lane
        import msvcrt

        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in (errno.EDEADLOCK, errno.EACCES, errno.EAGAIN):
                return False
            raise
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return False
        raise


def _unlock(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by the Windows probe lane
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def config_write_lock(
    config_dir: Path, *, wait: float = DEFAULT_CONFIG_LOCK_WAIT_S
) -> Iterator[None]:
    """Serialize one local transaction; ``wait=0`` performs one attempt.

    Reentrancy permits the current settings primitive to call ConfigManager's
    existing primitive without reacquiring a second file descriptor. Different
    threads cannot share ownership, and different profiles stay independent.
    """
    root = Path(config_dir).expanduser().resolve()
    budget = max(0.0, float(wait))
    deadline = time.monotonic() + budget
    held: dict[Path, int] = getattr(_owned, "roots", {})
    if held.get(root, 0):
        held[root] += 1
        try:
            yield
        finally:
            held[root] -= 1
        return
    with _registry_guard:
        gate = _thread_locks.setdefault(root, threading.RLock())
    if not gate.acquire(timeout=budget):
        raise ConfigWriteLockTimeout("configuration is being changed; retry shortly")
    fd: int | None = None
    acquired = False
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = root / ".config-write.lock"
        if path.is_symlink():
            raise ConfigWriteLockError("configuration writer lock must not be a symlink")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with _registry_guard:
            fd = os.open(path, flags, 0o600)
            _lock_fds.add(fd)
        while not acquired:
            acquired = _try_lock(fd)
            if acquired:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ConfigWriteLockTimeout("configuration is being changed; retry shortly")
            time.sleep(min(_LOCK_RETRY_S, remaining))
        held[root] = 1
        _owned.roots = held
        try:
            yield
        finally:
            held.pop(root, None)
    finally:
        try:
            if acquired and fd is not None:
                _unlock(fd)
        finally:
            if fd is not None:
                with _registry_guard:
                    _lock_fds.discard(fd)
                    os.close(fd)
            gate.release()


def serialized_config_write(function: Callable[P, T]) -> Callable[P, T]:
    """Protect existing manager-first primitives without changing their signature."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        manager: Any = args[0] if args else kwargs.get("manager")
        with config_write_lock(manager.config_dir):
            return function(*args, **kwargs)

    return wrapped


def serialized_config_path_write(function: Callable[P, T]) -> Callable[P, T]:
    """Protect the existing startup migration's config-directory-first boundary."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        root: Any = args[0] if args else kwargs.get("config_dir")
        with config_write_lock(root):
            return function(*args, **kwargs)

    return wrapped
