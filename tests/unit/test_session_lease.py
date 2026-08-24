"""Cross-process sole-writer and generation-fenced release regressions."""

from __future__ import annotations

import multiprocessing
import os
import threading
from pathlib import Path

from local_operator import session_lease as lease_mod
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


def _recover_stale(directory: str, start, release, results) -> None:
    start.wait()
    try:
        lease = acquire_session_lease(Path(directory))
    except SessionLeaseHeldError:
        results.put("held")
    else:
        results.put("won")
        release.wait(10)
        lease.release()


def _write_stale_claim(session_dir: Path) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / LEASE_NAME).write_text(
        '{"schema":1,"session_id":"same","generation":"stale","pid":2147483646}',
        encoding="utf-8",
    )


def test_two_stale_recoverers_cannot_replace_a_fresh_successor(tmp_path: Path, monkeypatch) -> None:
    """Both contenders inspect the old claim before either may recover it."""
    session_dir = tmp_path / "sessions" / "same"
    _write_stale_claim(session_dir)
    barrier = threading.Barrier(2)
    original_read = lease_mod._read_claim
    inspected: set[int] = set()
    inspected_lock = threading.Lock()

    def synchronized_read(path: Path) -> tuple[str | None, int | None]:
        claim = original_read(path)
        ident = threading.get_ident()
        if claim == ("stale", 2_147_483_646):
            with inspected_lock:
                first_inspection = ident not in inspected
                inspected.add(ident)
            if first_inspection:
                barrier.wait(timeout=5)
        return claim

    monkeypatch.setattr(lease_mod, "_read_claim", synchronized_read)
    start = threading.Barrier(3)
    outcomes: list[SessionLease | SessionLeaseHeldError] = []

    def recover() -> None:
        start.wait(timeout=5)
        try:
            outcomes.append(acquire_session_lease(session_dir))
        except SessionLeaseHeldError as exc:
            outcomes.append(exc)

    contenders = [threading.Thread(target=recover) for _ in range(2)]
    for contender in contenders:
        contender.start()
    start.wait(timeout=5)
    for contender in contenders:
        contender.join(timeout=5)
        assert not contender.is_alive()

    winners = [outcome for outcome in outcomes if isinstance(outcome, SessionLease)]
    assert len(winners) == 1
    assert sum(isinstance(outcome, SessionLeaseHeldError) for outcome in outcomes) == 1
    assert winners[0].generation in (session_dir / LEASE_NAME).read_text(encoding="utf-8")
    winners[0].release()


def test_many_processes_recover_one_stale_claim_exactly_once(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "same"
    _write_stale_claim(session_dir)
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    release = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(target=_recover_stale, args=(str(session_dir), start, release, results))
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [results.get(timeout=10) for _ in processes]
    assert outcomes.count("won") == 1
    assert outcomes.count("held") == len(processes) - 1
    release.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0


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
    _write_stale_claim(session_dir)
    stale = SessionLease(session_dir, "stale", 2_147_483_646)

    current = acquire_session_lease(session_dir, pid=os.getpid())

    stale.release()
    assert (session_dir / LEASE_NAME).exists()
    current.release()
