"""``wakes/lock.py``: one external writer of a session's wakes at a time.

The property under test is CROSS-PROCESS exclusion, so the load-bearing cell
drives a second interpreter rather than a second thread: a lock that only
serialises coroutines would pass a threaded test and lose the duplicate-row race
it exists for, because the writers that race are the CLI, the desktop server and
a second app.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from local_operator.wakes.lock import (
    LOCK_WAIT_S,
    WAKE_LOCK_NAME,
    WakeLockBusy,
    WakeWriteLock,
)

#: A holder in another interpreter: take the lock, say so, hold it past the
#: parent's attempt, then exit. Kept in-process as source so the test has no
#: fixture file to keep in sync with the module.
_HOLDER = """
import sys, time
sys.path.insert(0, {root!r})
from local_operator.wakes.lock import WakeWriteLock
with WakeWriteLock(__import__("pathlib").Path({session!r})):
    print("locked", flush=True)
    time.sleep({hold!r})
"""


def test_the_lock_file_is_the_session_sidecar_this_module_documents(tmp_path: Path) -> None:
    lock = WakeWriteLock(tmp_path)
    with lock:
        assert (tmp_path / WAKE_LOCK_NAME).exists()
    lock.release()  # idempotent: a `finally` after a failed acquire is safe


def test_a_second_holder_waits_and_then_times_out(tmp_path: Path) -> None:
    """The refusal, not a block: the waiter gives up at its deadline and the
    caller answers 503 rather than queueing a worker thread forever."""
    held = WakeWriteLock(tmp_path)
    held.acquire()
    try:
        with pytest.raises(WakeLockBusy) as busy:
            WakeWriteLock(tmp_path, timeout_s=0.05).acquire()
        assert "Retry in a moment" in str(busy.value)
    finally:
        held.release()

    # Released ⇒ the next holder gets it, which is what makes the refusal a
    # retryable answer rather than a permanent state.
    with WakeWriteLock(tmp_path, timeout_s=1.0):
        pass


def test_another_process_holds_the_lock_out(tmp_path: Path) -> None:
    """THE cross-process cell. A thread-only lock would pass every other test
    here and still lose the race this module exists to remove."""
    root = str(Path(__file__).resolve().parents[3])
    assert Path(root, "local_operator", "wakes", "lock.py").exists(), root
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER.format(root=root, session=str(tmp_path), hold=2.0)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"

        started = time.monotonic()
        with pytest.raises(WakeLockBusy):
            WakeWriteLock(tmp_path, timeout_s=0.4).acquire()
        assert time.monotonic() - started >= 0.4
    finally:
        holder.wait(timeout=30)

    with WakeWriteLock(tmp_path, timeout_s=2.0):
        pass


def test_release_allows_the_next_holder_immediately(tmp_path: Path) -> None:
    lock = WakeWriteLock(tmp_path)
    lock.acquire()
    lock.release()
    next_holder = WakeWriteLock(tmp_path, timeout_s=0.5)
    with next_holder:
        pass
    WakeWriteLock(tmp_path).release()  # never-held instance is a no-op


def test_the_wait_is_generous_enough_for_the_write_it_serialises() -> None:
    """Documented, not incidental: the write under the lock is a read plus an
    append, measured at ~0.5 s on a 25 MB transcript, so five concurrent arms
    legitimately queue for seconds. A short bound would turn contention into a
    false 503."""
    assert LOCK_WAIT_S >= 5.0


def test_two_threads_alternate_rather_than_overlap(tmp_path: Path) -> None:
    """The mutual exclusion itself, at thread granularity (the process
    granularity is the cell above): a critical section never overlaps."""
    inside = 0
    peak = 0
    guard = threading.Lock()

    def holder() -> None:
        nonlocal inside, peak
        with WakeWriteLock(tmp_path, timeout_s=10.0):
            with guard:
                inside += 1
                peak = max(peak, inside)
            time.sleep(0.05)
            with guard:
                inside -= 1

    threads = [threading.Thread(target=holder) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert peak == 1
