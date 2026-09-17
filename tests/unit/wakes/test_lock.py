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
    append, and the append parses the whole journal, so a 103 MB transcript with
    415k small entries was MEASURED at a 7.45 s hold (review round 2). Five
    concurrent arms legitimately queue; a short bound would turn contention into
    a false 503."""
    assert LOCK_WAIT_S >= 5.0


def test_a_lock_file_left_by_a_killed_holder_blocks_nobody(tmp_path: Path) -> None:
    """The stale lock, DRIVEN rather than reasoned about (review round 2, N5).

    A SIGKILLed holder leaves the FILE behind — nothing deletes it, and nothing
    should: the file is never written to, so there is nothing to clean. The whole
    design rests on the kernel dropping the flock at process death, which makes
    "a stale lock left by a crashed writer" a state this module does not have.
    Pinned because a future "tidy up the stale lock file" change would break it
    silently, and because the alternative (a lock file containing a pid to reap)
    is the design this deliberately is not.
    """
    root = str(Path(__file__).resolve().parents[3])
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER.format(root=root, session=str(tmp_path), hold=30.0)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        assert (tmp_path / WAKE_LOCK_NAME).exists()
        holder.kill()
        holder.wait(timeout=30)
    finally:
        if holder.poll() is None:
            holder.kill()

    # The file is still there; the lock is not. Taken immediately, not after the
    # holder's 30 s hold.
    assert (tmp_path / WAKE_LOCK_NAME).exists(), "the file is not supposed to be cleaned up"
    started = time.monotonic()
    with WakeWriteLock(tmp_path, timeout_s=1.0):
        pass
    assert time.monotonic() - started < 0.5


def test_a_directory_that_refuses_the_lock_file_is_a_refusal_not_a_crash(
    tmp_path: Path,
) -> None:
    """The lock's one non-contention failure mode (review round 2, R7; QA Q5).

    The lock file lives in the session directory, so a directory whose mode does
    not allow creating one used to raise ``PermissionError`` straight out of
    ``acquire()`` — past the caller's ``except WakeLockBusy`` and into an untyped
    500 where every other failure on this surface answers a sentence. Both halves
    of the mode change are covered: a directory that cannot be written, and one
    that is not there at all (the caller's existence check and this call are not
    atomic).
    """
    from local_operator.wakes.lock import WakeLockUnavailable

    refused = tmp_path / "readonly"
    refused.mkdir()
    refused.chmod(0o500)
    try:
        with pytest.raises(WakeLockUnavailable) as error:
            WakeWriteLock(refused, timeout_s=0.1).acquire()
        assert "does not accept a wake write" in str(error.value)
        assert "nothing was written" in str(error.value)
    finally:
        refused.chmod(0o700)

    with pytest.raises(WakeLockUnavailable):
        WakeWriteLock(tmp_path / "gone", timeout_s=0.1).acquire()


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
