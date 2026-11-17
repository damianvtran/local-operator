"""The Library's journal watch must stay shallow on Linux.

Regression for node2, 2026-11-17: a recursive watchdog watch of ``sessions/``
held 126,757 of the user's 126,814 inotify watches (one per scratchpad
directory) and every other inotify user on the host failed with ENOSPC. These
tests pin the replacement's contract: ``sessions/`` plus one watch per session
root, never a scratchpad, a new session picked up after start, and a spent
budget falling back to the timed reconcile with one warning.

The fake syscalls hand the watch a real pipe as its "inotify fd", so the
reader thread's ``poll``/``read`` loop is the production one and events are
fed as genuine ``struct inotify_event`` bytes.
"""

from __future__ import annotations

import errno
import logging
import os
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.library import indexer, session_watch

SESSIONS = ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc")


def _deep_sessions(root: Path, names: tuple[str, ...] = SESSIONS, depth: int = 6) -> Path:
    sessions = root / "sessions"
    for name in names:
        directory = sessions / name
        directory.mkdir(parents=True)
        (directory / "transcript.jsonl").write_text("")
        deep = directory / "scratchpad"
        for level in range(depth):
            deep = deep / f"rig{level}"
            (deep / "evidence").mkdir(parents=True)
    # A plain file at the sessions root is not a session and is never watched.
    (sessions / "index.json").write_text("{}")
    return sessions


def _event(wd: int, mask: int, name: str = "") -> bytes:
    raw = name.encode()
    if raw:
        raw += b"\0" * (16 - len(raw) % 16)
    return struct.pack("iIII", wd, mask, 0, len(raw)) + raw


class FakeSyscalls:
    """Records every watch; ``fail_after`` makes the Nth add raise ``ENOSPC``."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.added: list[tuple[str, int]] = []
        self.removed: list[int] = []
        self.closed: list[int] = []
        self.fail_after = fail_after
        self.read_fd, self.write_fd = os.pipe()

    def init(self) -> int:
        return self.read_fd

    def add_watch(self, fd: int, path: str, mask: int) -> int:
        if self.fail_after is not None and len(self.added) >= self.fail_after:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), path)
        self.added.append((path, mask))
        return len(self.added)  # wd 1 is sessions/, then one per session

    def wd_of(self, path: Path) -> int:
        return [p for p, _ in self.added].index(str(path)) + 1

    def rm_watch(self, fd: int, wd: int) -> None:
        self.removed.append(wd)

    def read(self, fd: int) -> bytes:
        return os.read(fd, 65536)

    def close(self, fd: int) -> None:
        self.closed.append(fd)
        os.close(fd)

    def feed(self, *events: bytes) -> None:
        os.write(self.write_fd, b"".join(events))


@pytest.fixture
def fake(monkeypatch) -> FakeSyscalls:
    syscalls = FakeSyscalls()
    monkeypatch.setattr(session_watch, "supported", lambda: True)
    monkeypatch.setattr(session_watch, "_Syscalls", lambda: syscalls)
    return syscalls


@pytest.fixture(autouse=True)
def _no_recursive_watchdog(monkeypatch):
    """Any watchdog schedule on the Linux path is the regression itself."""

    class RecordingObserver:
        def schedule(self, handler: Any, path: str, recursive: bool = False) -> None:
            raise AssertionError(f"watchdog scheduled {path!r} recursive={recursive}")

    try:
        import watchdog.observers as observers
    except ImportError:  # pragma: no cover - watchdog is a runtime dependency
        return
    monkeypatch.setattr(observers, "Observer", RecordingObserver)


def _until(predicate: Any, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _indexer(tmp_path: Path) -> tuple[indexer.LibraryIndexer, list[str]]:
    index = indexer.LibraryIndexer(tmp_path)
    dirty: list[str] = []
    index._mark_dirty = dirty.append  # type: ignore[method-assign]
    return index, dirty


def test_linux_watches_sessions_and_each_session_root_never_a_scratchpad(tmp_path, fake) -> None:
    sessions = _deep_sessions(tmp_path)
    index, _dirty = _indexer(tmp_path)
    index._start_observer()
    try:
        assert index.observer_failed is False
        paths = [path for path, _mask in fake.added]
        assert paths == [str(sessions)] + [str(sessions / name) for name in SESSIONS]
        assert not any("scratchpad" in path for path in paths)
        # The session watches are directory-only and carry no recursion of any
        # kind: inotify has none, so the set above IS the watch set.
        assert all(mask & session_watch.IN_ONLYDIR for _path, mask in fake.added)
        assert index._observer.watch_count() == 1 + len(SESSIONS)
    finally:
        index._stop_observer()
    assert fake.closed == [fake.read_fd]


def test_a_session_created_after_start_gets_its_watch_and_is_indexed(tmp_path, fake) -> None:
    sessions = _deep_sessions(tmp_path)
    index, dirty = _indexer(tmp_path)
    index._start_observer()
    try:
        late = "dddddddddddd"
        (sessions / late).mkdir()
        fake.feed(_event(1, session_watch.IN_CREATE | session_watch.IN_ISDIR, late))
        assert _until(lambda: late in dirty)
        assert (str(sessions / late), session_watch.SESSION_MASK) in fake.added

        # An append inside it now arrives through its own watch, as its id.
        dirty.clear()
        fake.feed(_event(fake.wd_of(sessions / late), session_watch.IN_MODIFY, "transcript.jsonl"))
        assert _until(lambda: dirty == [late])

        # A file landing at the sessions root is not a session.
        dirty.clear()
        fake.feed(_event(1, session_watch.IN_CREATE, "index.json.tmp"))
        fake.feed(_event(fake.wd_of(sessions / late), session_watch.IN_CLOSE_WRITE, "x"))
        assert _until(lambda: dirty == [late])

        # Removal: the session is still queued (its assets must leave), and a
        # moved-out directory's watch is handed back explicitly.
        dirty.clear()
        moved = SESSIONS[0]
        fake.feed(_event(1, session_watch.IN_MOVED_FROM | session_watch.IN_ISDIR, moved))
        assert _until(lambda: moved in dirty)
        assert fake.wd_of(sessions / moved) in fake.removed
        assert index._observer.watch_count() == 1 + len(SESSIONS)  # +late, -moved
    finally:
        index._stop_observer()


def test_enospc_at_start_falls_back_once_and_releases_the_fd(tmp_path, monkeypatch, caplog) -> None:
    _deep_sessions(tmp_path)
    syscalls = FakeSyscalls(fail_after=2)  # sessions/ and one session, then ENOSPC
    monkeypatch.setattr(session_watch, "supported", lambda: True)
    monkeypatch.setattr(session_watch, "_Syscalls", lambda: syscalls)
    index, _dirty = _indexer(tmp_path)
    with caplog.at_level(logging.WARNING, logger="local_operator.library.indexer"):
        index._start_observer()
        index._start_observer()  # a second failure must not warn again
    assert index.observer_failed is True
    assert index._observer is None
    # Closing the fd is what hands the watches it did take back to the user.
    assert syscalls.closed == [syscalls.read_fd] * 2  # one close per failed start
    warnings = [r for r in caplog.records if "inotify limit reached" in r.getMessage()]
    assert len(warnings) == 1
    os.close(syscalls.write_fd)


def test_enospc_on_a_later_session_tears_down_with_one_warning(tmp_path, fake, caplog) -> None:
    sessions = _deep_sessions(tmp_path)
    index, _dirty = _indexer(tmp_path)
    index._start_observer()
    try:
        fake.fail_after = len(fake.added)
        with caplog.at_level(logging.WARNING, logger="local_operator.library.indexer"):
            # Two new sessions in ONE read: the second must not warn again.
            events = []
            for name in ("eeeeeeeeeeee", "ffffffffffff"):
                (sessions / name).mkdir()
                events.append(_event(1, session_watch.IN_CREATE | session_watch.IN_ISDIR, name))
            fake.feed(*events)
            assert _until(lambda: index.observer_failed)
            assert _until(lambda: fake.closed == [fake.read_fd])
        warnings = [r for r in caplog.records if "inotify limit reached" in r.getMessage()]
        assert len(warnings) == 1
        assert index._observer.watch_count() == 0
    finally:
        index._stop_observer()


def test_a_kernel_queue_overflow_asks_for_a_reconcile(tmp_path, fake) -> None:
    _deep_sessions(tmp_path)
    index, _dirty = _indexer(tmp_path)
    woke = threading.Event()
    index.wake = woke.set  # type: ignore[method-assign]
    index._start_observer()
    try:
        fake.feed(_event(-1, session_watch.IN_Q_OVERFLOW))
        assert woke.wait(2.0)
    finally:
        index._stop_observer()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inotify is Linux-only")
def test_real_inotify_holds_one_watch_per_session_on_a_cloexec_fd(tmp_path) -> None:
    """The kernel's own count, read back from ``/proc/self/fdinfo``."""
    sessions = _deep_sessions(tmp_path)
    seen: list[str] = []
    watch = session_watch.SessionDirWatch(sessions, seen.append)
    watch.start()
    try:
        fd = watch._fd
        assert fd is not None
        info = Path(f"/proc/self/fdinfo/{fd}").read_text()
        assert sum(1 for line in info.splitlines() if line.startswith("inotify")) == 1 + len(
            SESSIONS
        )
        # Close-on-exec: an in-place re-exec of serve must not inherit it.
        flags = int(next(x for x in info.splitlines() if x.startswith("flags:")).split()[1], 8)
        assert flags & os.O_CLOEXEC

        late = sessions / "dddddddddddd"
        late.mkdir()
        assert _until(lambda: "dddddddddddd" in seen)
        seen.clear()
        (late / "transcript.jsonl").write_text("{}\n")
        assert _until(lambda: "dddddddddddd" in seen)
        # Deep writes are invisible by design: nothing below a session root.
        seen.clear()
        (sessions / SESSIONS[0] / "scratchpad" / "rig0" / "deep.txt").write_text("x")
        time.sleep(0.2)
        assert seen == []
        assert watch.watch_count() == 2 + len(SESSIONS)
    finally:
        watch.stop()
    assert watch.watch_count() == 0
