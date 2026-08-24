"""Cross-process sole-writer and generation-fenced release regressions."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

from local_operator.session_lease import (
    LEASE_NAME,
    SessionLease,
    SessionLeaseHeldError,
    acquire_session_lease,
)


def _contend(directory: str, start, release, results) -> None:
    start.wait()
    try:
        lease = acquire_session_lease(Path(directory))
    except SessionLeaseHeldError:
        results.put("held")
    else:
        results.put("won")
        # Keep the winning process alive until the sibling has probed it.
        release.wait(10)
        lease.release()


def test_two_processes_cannot_both_acquire_one_transcript(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "same"
    session_dir.mkdir(parents=True)
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    release = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(target=_contend, args=(str(session_dir), start, release, results))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [results.get(timeout=10) for _ in processes]
    release.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sorted(outcomes) == ["held", "won"]


def test_stale_release_cannot_unlink_successor_claim(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "same"
    first = acquire_session_lease(session_dir)
    path = session_dir / LEASE_NAME
    path.unlink()
    (session_dir / ".session.pid").unlink()
    successor = acquire_session_lease(session_dir)

    first.release()

    assert path.exists()
    assert successor.generation in path.read_text(encoding="utf-8")
    successor.release()


def test_proven_dead_owner_is_recovered(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "same"
    session_dir.mkdir(parents=True)
    stale = SessionLease(session_dir, "stale", 2_147_483_646)
    (session_dir / LEASE_NAME).write_text(
        '{"schema":1,"session_id":"same","generation":"stale","pid":2147483646}',
        encoding="utf-8",
    )

    current = acquire_session_lease(session_dir, pid=os.getpid())

    stale.release()
    assert (session_dir / LEASE_NAME).exists()
    current.release()
