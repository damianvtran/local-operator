"""The Library's Linux journal watch: one inotify fd, shallow watches only.

WHY THIS EXISTS (measured on node2, 2026-11-17). The indexer used to hand
``<config>/sessions`` to watchdog with ``recursive=True``. On Linux that is one
inotify watch per DIRECTORY under the tree, and a session directory is not
shallow: it carries ``scratchpad/`` with whole rigs, evidence trees and copied
checkouts. One ``lop serve`` held 126,757 of the user's 126,814 watches
(``fs.inotify.max_user_watches``); systemd and every other inotify user on the
host then failed with "No space left on device" (ENOSPC). The depth bought
nothing: :class:`~local_operator.library.indexer._JournalHandler` only ever
reads ``parts[0]`` of an event path — WHICH session moved — and everything the
index reads lives at a session's ROOT (the journal or its compressed archive,
the roster sidecar ``subagent-roster.v1.json``, the stored title inside the
journal). Subagent journals are sibling session directories under
``sessions/``, not nested ones. So the watch set this module keeps is exactly:

* ``sessions/`` itself, non-recursive — a session directory created, renamed
  or removed;
* each ``sessions/<id>/``, non-recursive — a journal append, a roster or
  sidecar replaced. ``scratchpad/`` and anything else below a session root is
  never watched.

That is ``1 + <number of sessions>`` watches (481 for 480 sessions) instead of
one per directory in every scratchpad (17,501 directories on the same host).

WHY NOT watchdog FOR THIS. watchdog cannot express "these N directories, each
shallow" cheaply: every ``Observer.schedule`` is its own emitter THREAD, and
under watchdog 6.0.0 (the floor ``pyproject.toml`` allows) its own inotify
INSTANCE too — 480 sessions would exceed ``fs.inotify.max_user_instances``
(128 by default) and fail with EMFILE. One fd and one reader thread is the
shape inotify itself offers, so this speaks it directly through ``ctypes``
(the three syscalls are in every Linux libc). macOS and Windows keep watchdog:
FSEvents and ReadDirectoryChangesW watch a tree as ONE kernel object, so
recursion costs them nothing.

WHY ``IN_CLOEXEC``. ``lop serve`` re-execs itself in place, and watchdog opens
its inotify fd with plain ``inotify_init()`` (no close-on-exec). The same
node2 process carried 20 inotify fds, 17 of them each holding a full recursive
watch of ``sessions/`` from an earlier image, with no thread left to read or
close them (``/proc/<pid>/fdinfo``: ``flags: 00``). Opening this fd with
``IN_CLOEXEC`` means an exec drops it, and its watches with it.

FAILURE IS A FALLBACK, NOT A CRASH. Any ``OSError`` while arming the watch —
``EMFILE`` from ``inotify_init1``, ``ENOSPC`` from ``inotify_add_watch`` when
the user's watch budget is spent — closes the fd (handing back every watch
taken so far, so a failed start does not keep squatting on the budget) and is
raised to the indexer, which falls back to its timed reconcile. The same
budget errors on a LATER per-session add (a new conversation while the host
is at its limit) tear the watch down and report through ``on_failure`` once.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import logging
import os
import select
import struct
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# <sys/inotify.h>. Stable kernel ABI since 2.6.13 (IN_CLOEXEC/IN_NONBLOCK:
# 2.6.27), identical on every Linux architecture this ships to.
IN_MODIFY: Final = 0x00000002
IN_ATTRIB: Final = 0x00000004
IN_CLOSE_WRITE: Final = 0x00000008
IN_MOVED_FROM: Final = 0x00000040
IN_MOVED_TO: Final = 0x00000080
IN_CREATE: Final = 0x00000100
IN_DELETE: Final = 0x00000200
IN_DELETE_SELF: Final = 0x00000400
IN_MOVE_SELF: Final = 0x00000800
IN_Q_OVERFLOW: Final = 0x00004000
IN_IGNORED: Final = 0x00008000
IN_ONLYDIR: Final = 0x01000000
IN_ISDIR: Final = 0x40000000
IN_CLOEXEC: Final = 0o2000000
IN_NONBLOCK: Final = 0o4000

#: ``sessions/``: only directory-level membership changes matter here; a
#: plain file written straight into ``sessions/`` is not a session
#: (``_JournalHandler`` drops it too).
ROOT_MASK: Final = (
    IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_ONLYDIR
)
#: ``sessions/<id>/``: the same event classes watchdog delivered for files at a
#: session root. IN_MODIFY keeps a long-running turn indexed WHILE it streams
#: (the coalescer's ``MAX_COALESCE_S`` cap relies on events arriving); the
#: create/move pair catches ``os.replace`` of a sidecar or journal.
SESSION_MASK: Final = (
    IN_MODIFY
    | IN_ATTRIB
    | IN_CLOSE_WRITE
    | IN_CREATE
    | IN_DELETE
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_ONLYDIR
)

#: Errors that mean "the per-user inotify budget is spent", as opposed to a
#: path that vanished between ``scandir`` and ``inotify_add_watch``.
BUDGET_ERRNOS: Final = frozenset({errno.ENOSPC, errno.EMFILE, errno.ENFILE, errno.ENOMEM})

_EVENT: Final = struct.Struct("iIII")
#: Large enough for a burst of hundreds of events per read; the kernel never
#: splits one event across reads, so any size >= one max event is correct.
_READ_SIZE: Final = 64 * 1024


def supported() -> bool:
    """Whether this host can use the shallow inotify watch (Linux only)."""
    return sys.platform.startswith("linux")


class _Syscalls:
    """The three inotify syscalls. A seam: tests replace it to count watches."""

    def __init__(self) -> None:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        self._init1 = libc.inotify_init1
        self._init1.argtypes = [ctypes.c_int]
        self._init1.restype = ctypes.c_int
        self._add = libc.inotify_add_watch
        self._add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._add.restype = ctypes.c_int
        self._rm = libc.inotify_rm_watch
        self._rm.argtypes = [ctypes.c_int, ctypes.c_int]
        self._rm.restype = ctypes.c_int

    @staticmethod
    def _error(path: str | None = None) -> OSError:
        code = ctypes.get_errno()
        return OSError(code, os.strerror(code), path) if path else OSError(code, os.strerror(code))

    def init(self) -> int:
        fd = self._init1(IN_CLOEXEC | IN_NONBLOCK)
        if fd < 0:
            raise self._error()
        return int(fd)

    def add_watch(self, fd: int, path: str, mask: int) -> int:
        wd = self._add(fd, os.fsencode(path), mask)
        if wd < 0:
            raise self._error(path)
        return int(wd)

    def rm_watch(self, fd: int, wd: int) -> None:
        # EINVAL here means the kernel already dropped it (the directory went
        # away): nothing left to release.
        self._rm(fd, wd)

    def read(self, fd: int) -> bytes:
        try:
            return os.read(fd, _READ_SIZE)
        except BlockingIOError:
            return b""

    def close(self, fd: int) -> None:
        os.close(fd)


class SessionDirWatch:
    """Shallow watches on ``sessions/`` and each session directory, one fd.

    ``on_session(session_id)`` is called for every relevant event, from the
    reader thread; ``on_overflow()`` when the kernel queue overflowed (events
    were lost, so the caller should reconcile); ``on_failure(error)`` once, if
    the watch had to tear itself down (budget exhausted, ``sessions/`` gone).
    """

    def __init__(
        self,
        root: Path,
        on_session: Callable[[str], None],
        *,
        on_overflow: Callable[[], None] | None = None,
        on_failure: Callable[[OSError], None] | None = None,
        syscalls: Any = None,
    ) -> None:
        self.root = Path(root)
        self._on_session = on_session
        self._on_overflow = on_overflow
        self._on_failure = on_failure
        self._sys = syscalls if syscalls is not None else _Syscalls()
        self._fd: int | None = None
        self._root_wd: int | None = None
        self._wd_session: dict[int, str] = {}
        self._session_wd: dict[str, int] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._kill_r: int | None = None
        self._kill_w: int | None = None
        self._failed = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Arm every watch and start the reader. Raises ``OSError`` on failure.

        On any failure the fd is closed before raising, so the watches already
        added go back to the user's budget instead of leaking with it.
        """
        fd = self._sys.init()
        self._fd = fd
        try:
            self._root_wd = self._sys.add_watch(fd, str(self.root), ROOT_MASK)
            with os.scandir(self.root) as entries:
                names = sorted(e.name for e in entries if e.is_dir(follow_symlinks=False))
            for name in names:
                self._add_session(name)
        except BaseException:
            self._close_fd()
            raise
        self._kill_r, self._kill_w = os.pipe()  # close-on-exec by default (PEP 446)
        self._thread = threading.Thread(target=self._run, name="library-watch", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._kill_w is not None:
            try:
                os.write(self._kill_w, b"!")
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
        self._close_fd()
        for name in ("_kill_r", "_kill_w"):
            pipe_fd = getattr(self, name)
            setattr(self, name, None)
            if pipe_fd is not None:
                try:
                    os.close(pipe_fd)
                except OSError:
                    pass

    def join(self, timeout: float | None = None) -> None:
        """Duck-types watchdog's ``Observer.join`` for the indexer's shutdown path."""
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    def watch_count(self) -> int:
        """Watches this instance holds: ``sessions/`` plus one per session."""
        with self._lock:
            if self._fd is None:
                return 0
            return (1 if self._root_wd is not None else 0) + len(self._session_wd)

    def watched_paths(self) -> list[Path]:
        with self._lock:
            paths = [self.root] if self._root_wd is not None else []
            return paths + [self.root / name for name in sorted(self._session_wd)]

    # -- watches ------------------------------------------------------------

    def _add_session(self, name: str) -> bool:
        """Watch one session root. False when it vanished first; raises on budget."""
        fd = self._fd
        if fd is None:
            return False
        try:
            wd = self._sys.add_watch(fd, str(self.root / name), SESSION_MASK)
        except OSError as error:
            if error.errno in BUDGET_ERRNOS:
                raise
            # ENOENT/ENOTDIR: removed or replaced by a file between the listing
            # and the add. The removal event (or the reconcile) covers it.
            return False
        with self._lock:
            # A rename reuses a wd only after IN_IGNORED, which drops the old
            # mapping first; overwrite is still the safe order.
            self._wd_session[wd] = name
            self._session_wd[name] = wd
        return True

    def _drop_session(self, name: str, *, rm: bool) -> None:
        with self._lock:
            wd = self._session_wd.pop(name, None)
            if wd is not None:
                self._wd_session.pop(wd, None)
        if rm and wd is not None and self._fd is not None:
            # A directory MOVED out keeps its watch in the kernel (it follows
            # the inode); a deleted one is dropped by the kernel itself.
            self._sys.rm_watch(self._fd, wd)

    def _close_fd(self) -> None:
        with self._lock:
            fd, self._fd = self._fd, None
            self._root_wd = None
            self._wd_session.clear()
            self._session_wd.clear()
        if fd is not None:
            try:
                # Closing the inotify fd releases every watch on it at once.
                self._sys.close(fd)
            except OSError:
                pass

    def _fail(self, error: OSError) -> None:
        if self._failed:
            return
        self._failed = True
        self._stop.set()
        self._close_fd()
        if self._on_failure is not None:
            self._on_failure(error)

    # -- reader -------------------------------------------------------------

    def _run(self) -> None:
        poller = select.poll()
        fd = self._fd
        if fd is None or self._kill_r is None:
            return
        poller.register(fd, select.POLLIN)
        poller.register(self._kill_r, select.POLLIN)
        while not self._stop.is_set():
            try:
                ready = poller.poll()
            except InterruptedError:
                continue
            if self._stop.is_set():
                break
            if not any(ready_fd == fd for ready_fd, _ in ready):
                continue
            try:
                self.handle(self._sys.read(fd))
            except OSError as error:
                self._fail(error)
                break
            except Exception:  # noqa: BLE001 - a callback bug must not kill the watch
                logger.warning("library: journal watch event handling failed", exc_info=True)

    def handle(self, buffer: bytes) -> None:
        """Dispatch one ``read()`` worth of ``struct inotify_event`` records."""
        offset = 0
        while offset + _EVENT.size <= len(buffer):
            wd, mask, _cookie, length = _EVENT.unpack_from(buffer, offset)
            raw = buffer[offset + _EVENT.size : offset + _EVENT.size + length]
            offset += _EVENT.size + length
            name = os.fsdecode(raw.split(b"\0", 1)[0]) if length else ""
            self._event(wd, mask, name)
            if self._stop.is_set():
                return

    def _event(self, wd: int, mask: int, name: str) -> None:
        if mask & IN_Q_OVERFLOW:
            if self._on_overflow is not None:
                self._on_overflow()
            return
        if wd == self._root_wd:
            if mask & (IN_DELETE_SELF | IN_MOVE_SELF | IN_IGNORED):
                self._fail(OSError(errno.ENOENT, "sessions directory went away", str(self.root)))
                return
            if not name or not mask & IN_ISDIR:
                return
            if mask & (IN_CREATE | IN_MOVED_TO):
                try:
                    self._add_session(name)
                except OSError as error:
                    self._fail(error)
                    return
            elif mask & (IN_DELETE | IN_MOVED_FROM):
                self._drop_session(name, rm=bool(mask & IN_MOVED_FROM))
            # Created (scan what is already there — files may have landed
            # before the watch did) or removed (``_forget`` its assets).
            self._on_session(name)
            return
        with self._lock:
            session = self._wd_session.get(wd)
            if session is not None and mask & IN_IGNORED:
                # The kernel dropped this watch (directory deleted, or its
                # filesystem unmounted); the root's IN_DELETE carries the news.
                self._wd_session.pop(wd, None)
                if self._session_wd.get(session) == wd:
                    self._session_wd.pop(session, None)
                return
        if session is not None:
            self._on_session(session)
