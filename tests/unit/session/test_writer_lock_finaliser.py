"""A finaliser must never deadlock against the writer guard.

THE MEASURED HANG (CI run 40414178365, a 240 s shard stall in
``test_fork_snapshot``): ``Transcript.__init__`` -> ``acquire_writer`` took the
non-reentrant ``_WRITERS_GUARD`` and called ``_try_acquire_lock``, whose
``Path`` arithmetic allocated; the allocation ran the cyclic GC, the GC
finalised an unreferenced ``Transcript``, and its ``__del__`` called
``release_writer`` — which waited for the guard its own thread held. Forever.

These tests force that interleaving deterministically instead of waiting for GC
timing: a patched ``_lock_path`` drops the last reference (or runs
``gc.collect()`` on a cycle) while the guard is held. Each runs the
open in a daemon thread with a join timeout, so a regression FAILS here rather
than stalling the shard, and each swaps in a private guard and registry, so a
thread a regression leaves stuck holds only this test's lock.

The second half pins promptness: a deferred release is applied by the holder
as it lets go, with no later acquire needed — another process waiting on the
flock must not wait until this one exits.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.session import compression
from local_operator.session.transcript import Transcript

#: Long enough that a loaded CI runner never trips it on a working build (the
#: open is milliseconds), short enough that a regression fails fast.
_JOIN_S = 10.0


@pytest.fixture
def private_guard(monkeypatch: pytest.MonkeyPatch) -> threading.Lock:
    """A guard and registry of this test's own, so a deadlock cannot spread."""
    guard = threading.Lock()
    monkeypatch.setattr(compression, "_WRITERS_GUARD", guard)
    monkeypatch.setattr(compression, "_WRITERS", {})
    # ``raising=False``: the queue is new with the fix, and the reproduction
    # must still run (and fail by timing out) on a build that lacks it.
    monkeypatch.setattr(compression, "_DEFERRED_RELEASES", deque(), raising=False)
    return guard


def _hook_lock_path(
    monkeypatch: pytest.MonkeyPatch, target: Path, under_guard: Callable[[], object]
) -> list[bool]:
    """Run ``under_guard`` inside ``_lock_path`` for ``target``, once.

    ``_lock_path`` is exactly where the CI trace caught the GC: called by
    ``_try_acquire_lock`` while ``acquire_writer`` holds the guard. Returns a
    one-slot record of whether the hook fired, so a test can prove it reached
    the window rather than passing because nothing happened.
    """
    original = compression._lock_path
    fired: list[bool] = []

    def hooked(directory: Path) -> Path:
        if not fired and Path(directory) == target:
            fired.append(True)
            under_guard()
        return original(directory)

    monkeypatch.setattr(compression, "_lock_path", hooked)
    return fired


def _open_in_thread(directory: Path) -> tuple[threading.Thread, list[Any]]:
    """Construct a writer ``Transcript`` on a daemon thread; the result lands in the list."""
    opened: list[Any] = []

    def run() -> None:
        try:
            opened.append(Transcript(directory))
        except BaseException as exc:  # noqa: BLE001 — surfaced by the assertion
            opened.append(exc)

    worker = threading.Thread(target=run, name="writer-guard-open", daemon=True)
    worker.start()
    worker.join(_JOIN_S)
    return worker, opened


def _key(directory: Path) -> str:
    return os.path.realpath(os.fspath(directory))


def _kernel_lock_busy(directory: Path) -> bool:
    """Whether the flock is held, read straight from the kernel.

    A second open file description conflicts with this process's own hold, so
    this sees the registry's fd. Unlike ``lock_is_busy`` it never touches the
    guard, which is what a probe taken WHILE another thread holds it needs.
    """
    fd = os.open(os.fspath(directory / compression.LOCK_NAME), os.O_RDWR)
    try:
        if compression._flock(fd):
            compression._funlock(fd)
            return False
        return True
    finally:
        os.close(fd)


def test_a_transcript_dropped_under_the_guard_does_not_deadlock_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_guard: threading.Lock
) -> None:
    """The CI trace, by refcount: the last reference dies inside ``_lock_path``."""
    first_dir = tmp_path / "sessions" / "first0000001"
    second_dir = tmp_path / "sessions" / "second000001"
    holders = [Transcript(first_dir)]
    assert compression.lock_is_busy(first_dir)

    fired = _hook_lock_path(monkeypatch, second_dir, holders.clear)
    worker, opened = _open_in_thread(second_dir)

    assert not worker.is_alive(), (
        "opening a second transcript deadlocked: the dropped transcript's "
        "finaliser waited on the writer guard its own thread held"
    )
    assert fired, "the hook never ran, so the window was not exercised"
    assert len(opened) == 1 and isinstance(opened[0], Transcript), opened
    assert not private_guard.locked()
    # The deferred release was APPLIED as the open let go of the guard — not
    # parked until some later acquire. The kernel lock is free (another process
    # can take it) and the in-process row is gone.
    assert not compression.lock_is_busy(first_dir)
    assert _key(first_dir) not in compression._WRITERS
    assert _key(second_dir) in compression._WRITERS
    opened[0].close()


def test_a_cycle_collected_under_the_guard_does_not_deadlock_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_guard: threading.Lock
) -> None:
    """The CI trace, by the cyclic GC: a cycle-held transcript collected mid-acquire."""
    first_dir = tmp_path / "sessions" / "first0000002"
    second_dir = tmp_path / "sessions" / "second000002"
    was_enabled = gc.isenabled()
    # Automatic collection off for the setup, so the cycle is still alive when
    # the hook runs the collection itself, inside the guard.
    gc.disable()
    try:
        cycle: Any = Transcript(first_dir)
        cycle.self_reference = cycle
        del cycle
        assert compression.lock_is_busy(first_dir)

        fired = _hook_lock_path(monkeypatch, second_dir, gc.collect)
        worker, opened = _open_in_thread(second_dir)
    finally:
        if was_enabled:
            gc.enable()

    assert not worker.is_alive(), "the cyclic GC's finaliser deadlocked the open"
    assert fired
    assert len(opened) == 1 and isinstance(opened[0], Transcript), opened
    assert not compression.lock_is_busy(first_dir)
    assert _key(first_dir) not in compression._WRITERS
    opened[0].close()


def test_a_finaliser_on_another_thread_defers_and_the_holder_applies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_guard: threading.Lock
) -> None:
    """Cross-thread: the finaliser neither blocks nor loses its release.

    The guard keeps its cross-thread meaning — the release is not applied under
    another thread's hold — and the holder applies it the moment it lets go.
    """
    first_dir = tmp_path / "sessions" / "first0000003"
    second_dir = tmp_path / "sessions" / "second000003"
    holders = [Transcript(first_dir)]
    in_window = threading.Event()
    leave_window = threading.Event()

    def park() -> None:
        in_window.set()
        # Outlasts the dropper's join below: a park that timed out first would
        # free the guard and let a BLOCKED dropper finish just inside its join,
        # so a regression would fail one assertion late and misreport itself.
        # The ``finally`` releases the park as soon as the assertions are done.
        assert leave_window.wait(_JOIN_S * 3)

    _hook_lock_path(monkeypatch, second_dir, park)
    opener = threading.Thread(target=lambda: holders.append(Transcript(second_dir)), daemon=True)
    opener.start()
    try:
        assert in_window.wait(_JOIN_S)
        assert private_guard.locked()

        dropper = threading.Thread(target=lambda: holders.pop(0), daemon=True)
        dropper.start()
        dropper.join(_JOIN_S)
        assert not dropper.is_alive(), "the finaliser blocked on another thread's guard"

        # Deferred, not applied under the other thread's hold. A RAW kernel
        # probe: ``lock_is_busy`` settles the queue first, so it would wait for
        # the parked holder rather than report this in-between state.
        assert _kernel_lock_busy(first_dir)
        assert _key(first_dir) in compression._WRITERS
    finally:
        leave_window.set()
        opener.join(_JOIN_S)
    assert not opener.is_alive()

    # Applied as the opener let go: no further call into the registry needed.
    assert not compression.lock_is_busy(first_dir)
    assert _key(first_dir) not in compression._WRITERS
    for transcript in holders:
        transcript.close()


_TAKE_LOCK = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
print("taken", flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the probe takes a POSIX flock")
def test_a_deferred_release_frees_the_session_for_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_guard: threading.Lock
) -> None:
    """The point of promptness: a second PROCESS can take the lock at once.

    ``lock_is_busy`` answers from this process; this asks the kernel through a
    real second holder, which is who waits on a dropped transcript in practice
    (the compressor, another runtime).
    """
    first_dir = tmp_path / "sessions" / "first0000004"
    second_dir = tmp_path / "sessions" / "second000004"
    holders = [Transcript(first_dir)]
    _hook_lock_path(monkeypatch, second_dir, holders.clear)
    worker, opened = _open_in_thread(second_dir)
    assert not worker.is_alive()

    probe = subprocess.run(
        [sys.executable, "-c", _TAKE_LOCK, os.fspath(first_dir / compression.LOCK_NAME)],
        capture_output=True,
        text=True,
        timeout=_JOIN_S,
    )
    assert probe.returncode == 0 and probe.stdout.strip() == "taken", probe.stderr
    # And this process can take it again: a fresh kernel acquisition, not a
    # refcount bump on a stale row.
    assert compression.acquire_writer(first_dir) is True
    assert compression._WRITERS[_key(first_dir)][1] == 1
    compression.release_writer(first_dir)
    opened[0].close()


def test_explicit_close_still_releases_synchronously(
    tmp_path: Path, private_guard: threading.Lock
) -> None:
    """Only the finaliser defers: ``close`` frees the lock before it returns."""
    directory = tmp_path / "sessions" / "closed000001"
    transcript = Transcript(directory)
    assert compression.lock_is_busy(directory)
    transcript.close()
    assert not compression.lock_is_busy(directory)
    assert compression._WRITERS == {}
    transcript.close()  # idempotent


def _queue_a_deferred_release(directory: Path, guard: threading.Lock) -> None:
    """Leave ``directory``'s writer release QUEUED, the state a lost race leaves.

    A finaliser that fires while another thread holds the guard queues its key,
    and that holder's post-release drain only TRIES the guard; a third thread
    taking the guard in between leaves the key queued. That interleaving is a
    scheduling accident, so this builds its end state directly: the guard is
    held while the last reference dies on another thread, and the hold is
    dropped with ``guard.release()`` itself, which skips the drain.
    """
    holders = [Transcript(directory)]
    assert guard.acquire(timeout=_JOIN_S)
    try:
        dropper = threading.Thread(target=holders.clear, daemon=True)
        dropper.start()
        dropper.join(_JOIN_S)
        assert not dropper.is_alive(), "the finaliser blocked on another thread's guard"
    finally:
        guard.release()
    assert list(compression._DEFERRED_RELEASES) == [_key(directory)], "precondition: queued"


def _stale_session(tmp_path: Path, sid: str) -> tuple[Path, Path]:
    """A marked store with one session old enough for the default policy."""
    from local_operator.session import cleanup

    config = tmp_path / "config"
    sessions = config / "sessions"
    sessions.mkdir(parents=True)
    (sessions / cleanup.STORE_MARKER_NAME).write_text("{}", encoding="utf-8")
    directory = sessions / sid
    Transcript(directory).close()
    (directory / compression.PLAIN_FILENAME).write_text(
        '{"type":"message","payload":{"role":"user","content":"hello"}}\n', encoding="utf-8"
    )
    stamp = time.time() - 40 * 86400
    os.utime(directory / compression.PLAIN_FILENAME, (stamp, stamp))
    os.utime(directory, (stamp, stamp))
    return config, directory


def test_a_queued_release_does_not_make_compress_session_skip_the_session(
    tmp_path: Path, private_guard: threading.Lock
) -> None:
    """``compress_session`` reads the KERNEL lock, so it must settle the queue first.

    Without that, a writer this process already dropped reads as "session in
    use" — the CI failure in ``test_tail_miss_wire_matrix[archive]`` on #3011.
    """
    _, directory = _stale_session(tmp_path, "queued000001")
    _queue_a_deferred_release(directory, private_guard)

    outcome = compression.compress_session(directory)

    assert outcome.status == "compressed", outcome
    assert list(compression._DEFERRED_RELEASES) == []
    assert _key(directory) not in compression._WRITERS


def test_a_queued_release_does_not_read_as_busy_to_the_dry_run_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_guard: threading.Lock
) -> None:
    """The rehearsal's ``lock_is_busy`` probe settles the queue the same way."""
    config, directory = _stale_session(tmp_path, "queued000002")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    _queue_a_deferred_release(directory, private_guard)

    result = compression.run_compression(
        config, compression.CompressionPolicy(enabled=True, after_days=30), dry_run=True
    )

    assert [row.session for row in result.candidates] == [directory.name], result
    assert "session in use" not in result.skipped_by_reason
    assert list(compression._DEFERRED_RELEASES) == []


def test_a_finaliser_with_the_guard_free_releases_before_del_returns(
    tmp_path: Path, private_guard: threading.Lock
) -> None:
    """The ordinary case defers nothing: the release lands inside ``del`` itself.

    So a CI "session in use" after a dropped writer means the writer was NOT yet
    finalised (another reference was still alive), never that its release sat
    in the queue with the guard free.
    """
    directory = tmp_path / "sessions" / "freegc000001"
    holders = [Transcript(directory)]
    assert _kernel_lock_busy(directory)
    holders.clear()
    assert list(compression._DEFERRED_RELEASES) == []
    assert compression._WRITERS == {}
    assert not _kernel_lock_busy(directory)
