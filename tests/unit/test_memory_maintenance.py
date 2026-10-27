"""Unattended memory maintenance: auto-compaction (B1) and capture expiry (B4).

Each test drives the REAL store and the real ``journal_plan``/``compact`` path;
only thresholds, the clock and disk-free are pinned, so a pass that is wrong
about when to run, what to drop or what to keep fails here rather than on the
operator's store.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from local_operator import memory_maintenance as mm
from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore
from local_operator.memory_capture import persist

DAY = 86400.0


def _store(config: Path, session: str = "a" * 12) -> MemoryStore:
    return MemoryStore(
        config, MemoryIdentity(session), config / "sessions" / session, origin="device-a"
    )


def _churn(store: MemoryStore, revisions: int = 6) -> str:
    """One fact edited ``revisions`` times: ``revisions`` reclaimable ops."""
    done = store.put("session", {"title": "Churn", "body": "revision 0"})
    for index in range(1, revisions + 1):
        done = store.put(
            "session",
            {"title": "Churn", "body": f"revision {index}"},
            entity=done["id"],
            expected_version=done["version"],
        )
    return done["id"]


def _prunes(config: Path) -> list[tuple[Any, ...]]:
    db = sqlite3.connect(config / "memory" / "memory.db")
    try:
        return db.execute("SELECT ops, reason FROM journal_prune ORDER BY at").fetchall()
    finally:
        db.close()


def _op_count(config: Path) -> int:
    db = sqlite3.connect(config / "memory" / "memory.db")
    try:
        return int(db.execute("SELECT count(*) FROM ops").fetchone()[0])
    finally:
        db.close()


@pytest.fixture
def low_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real threshold (5,000 ops) is exercised on a clone of the live store;
    # here a handful of revisions crosses it so the unit stays fast.
    monkeypatch.setattr(mm, "RECLAIMABLE_OPS_THRESHOLD", 3)


# ---------------------------------------------------------------------------
# B1: when the pass runs, and that it runs once
# ---------------------------------------------------------------------------


def test_the_supervisor_seat_compacts_once_and_a_second_tick_inside_6h_does_nothing(
    tmp_path: Path, low_threshold: None
) -> None:
    store = _store(tmp_path)
    entity = _churn(store)
    before = _op_count(tmp_path)
    head = store.show("session", entity)

    from local_operator.wakes import supervisor as sup

    async def tick() -> dict[str, Any]:
        seat = sup._MemoryMaintenanceSeat(tmp_path)
        seat.kick()
        assert seat._task is not None
        return await seat._task

    first = asyncio.run(tick())
    assert first["status"] == "ran", first
    assert first["compacted_ops"] == 6
    assert _prunes(tmp_path) == [(6, "auto")]
    assert _op_count(tmp_path) == before - 6
    # State-preserving: the same head is what every reader sees.
    assert store.show("session", entity)["version"] == head["version"]

    # More churn would cross the threshold again, but the 6 h interval holds.
    _churn(store)
    second = asyncio.run(tick())
    assert second["status"] == "not-due"
    assert _prunes(tmp_path) == [(6, "auto")]


def test_an_operator_pass_counts_toward_the_interval(tmp_path: Path, low_threshold: None) -> None:
    store = _store(tmp_path)
    _churn(store)
    plan = store.journal_plan()
    store.compact(expect=plan["fingerprint"])
    _churn(store)
    assert mm.run_pass(tmp_path)["status"] == "not-due"
    assert [reason for _, reason in _prunes(tmp_path)] == ["operator"]


def test_a_store_below_every_threshold_is_left_alone_and_rechecked_next_tick(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _churn(store, revisions=2)
    report = mm.run_pass(tmp_path)
    assert report["status"] == "idle"
    assert report["probe"]["reclaimable"] == 2
    assert _prunes(tmp_path) == []
    # Idle writes no stamp, so crossing a threshold is acted on at once.
    assert not (tmp_path / "memory" / mm.STAMP_NAME).exists()


def test_the_probe_counts_reclaimable_ops_without_creating_anything(tmp_path: Path) -> None:
    assert mm.probe(tmp_path) == {"exists": False}
    assert not (tmp_path / "memory").exists()
    store = _store(tmp_path)
    _churn(store, revisions=4)
    reading = mm.probe(tmp_path)
    assert reading["reclaimable"] == 4
    assert reading["triggers"] == []


def test_a_second_process_skips_while_one_holds_the_pass(
    tmp_path: Path, low_threshold: None
) -> None:
    import fcntl

    store = _store(tmp_path)
    _churn(store)
    with open(tmp_path / "memory" / mm.LOCK_NAME, "a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert mm.run_pass(tmp_path)["status"] == "busy"
    assert _prunes(tmp_path) == []


def test_memory_disabled_runs_nothing(tmp_path: Path, low_threshold: None) -> None:
    store = _store(tmp_path)
    _churn(store)
    (tmp_path / "config.yml").write_text(
        "values:\n  memory:\n    enabled: false\n", encoding="utf-8"
    )
    assert mm.run_pass(tmp_path)["status"] == "disabled"
    assert _prunes(tmp_path) == []


# ---------------------------------------------------------------------------
# B1: safety
# ---------------------------------------------------------------------------


def test_a_concurrent_writer_waits_for_the_pass_and_succeeds(
    tmp_path: Path, low_threshold: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _churn(store)
    planned = threading.Event()
    real = MemoryStore._journal_plan

    def slow_plan(self: MemoryStore, db: sqlite3.Connection) -> Any:
        result = real(self, db)
        if db.in_transaction:
            # Inside compact's BEGIN IMMEDIATE: hold the write lock about as long
            # as the measured live pass (1.28 s), under the 5 s busy_timeout.
            planned.set()
            time.sleep(1.2)
        return result

    monkeypatch.setattr(MemoryStore, "_journal_plan", slow_plan)
    outcome: dict[str, Any] = {}

    def writer() -> None:
        planned.wait(10)
        began = time.monotonic()
        try:
            outcome["done"] = _store(tmp_path, "b" * 12).put(
                "session", {"title": "Concurrent", "body": "written during the pass"}
            )
        except Exception as exc:  # noqa: BLE001 — the assertion reports it
            outcome["error"] = exc
        outcome["waited"] = time.monotonic() - began

    thread = threading.Thread(target=writer)
    thread.start()
    report = mm.run_pass(tmp_path)
    thread.join(15)
    assert report["status"] == "ran"
    assert "error" not in outcome, outcome
    assert outcome["done"]["status"] == "active"
    # It genuinely queued behind the pass rather than racing past it.
    assert outcome["waited"] >= 0.5
    assert _store(tmp_path, "b" * 12).search("Concurrent")


def test_low_disk_skips_vacuum_and_still_frees_pages(
    tmp_path: Path, low_threshold: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    for index in range(4):
        _churn(store, revisions=3)
    usage = namedtuple("usage", "total used free")
    # 512 MiB free: above 2x a tiny store, below the 1 GiB floor.
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda _p: usage(1 << 40, 0, 512 << 20))
    report = mm.run_pass(tmp_path)
    assert report["status"] == "ran"
    assert report["compacted_ops"] == 12
    assert report["vacuum_skipped_low_disk"] is True
    assert report["vacuumed"] is False
    db = sqlite3.connect(tmp_path / "memory" / "memory.db")
    try:
        # The DELETE returned pages to the freelist (reusable by every table),
        # and no VACUUM rewrote them away.
        assert int(db.execute("PRAGMA freelist_count").fetchone()[0]) > 0
    finally:
        db.close()


def test_ample_disk_vacuums(tmp_path: Path, low_threshold: None) -> None:
    store = _store(tmp_path)
    _churn(store)
    report = mm.run_pass(tmp_path)
    assert report["vacuumed"] is True
    assert report["vacuum_skipped_low_disk"] is False


def test_a_pass_past_its_timeout_is_abandoned_and_keeps_the_lock(
    tmp_path: Path, low_threshold: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _churn(store)
    release = threading.Event()
    monkeypatch.setattr(mm, "_run", lambda *a, **k: release.wait(10))
    assert mm.run_pass(tmp_path, timeout=0.2)["status"] == "timeout"
    # The abandoned worker still holds the pass: no second one overlaps it.
    assert mm.run_pass(tmp_path, timeout=0.2)["status"] == "busy"
    release.set()


def test_compact_refuses_an_unknown_reason(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _churn(store, revisions=1)
    plan = store.journal_plan()
    with pytest.raises(MemoryError, match="reason"):
        store.compact(expect=plan["fingerprint"], reason="cron")


# ---------------------------------------------------------------------------
# B4: capture expiry and session deletion
# ---------------------------------------------------------------------------


def _capture(store: MemoryStore, marker: str, body: str, at: float) -> None:
    rows = [{"title": body[:40], "body": body, "tags": ["decision"]}]
    assert persist(store, marker, rows, "ok", committed_at=at)[0] == "ok"


def _facts(config: Path) -> dict[str, dict[str, Any]]:
    db = sqlite3.connect(config / "memory" / "memory.db")
    try:
        return {
            json.loads(row[1])["body"]: json.loads(row[1])
            for row in db.execute("SELECT entity_id, fact FROM current")
        }
    finally:
        db.close()


def test_capture_stamps_a_14_day_deadline_refreshed_on_each_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = 1_790_000_000.25
    _capture(store, "m1", "Keep the stable prefix for caching", first)
    fact = _facts(tmp_path)["Keep the stable prefix for caching"]
    assert fact["expires_at"] == int(first) + 14 * 86400
    later = first + 3 * DAY
    _capture(store, "m2", "Keep the stable prefix for caching", later)
    fact = _facts(tmp_path)["Keep the stable prefix for caching"]
    assert fact["expires_at"] == int(later) + 14 * 86400
    # created_at is the first capture's; only the deadline moves.
    assert fact["created_at"] == first


def test_a_15_day_old_capture_is_hidden_then_tombstoned_and_a_deliberate_fact_is_untouched(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    aged = time.time() - 15 * DAY
    _capture(store, "old", "Captured decision about the release window", aged)
    _capture(store, "new", "Captured decision about the fresh window", time.time())
    deliberate = store.put(
        "session",
        {
            "title": "Deliberate window",
            "body": "Deliberate decision about the release window",
            "created_at": aged,
            "source_time": aged,
        },
    )
    # Hidden from search BEFORE the pass (the read path honours expires_at).
    found = {hit["title"] for hit in store.search("decision window")}
    assert "Captured decision about the release window"[:40] not in found
    assert "Deliberate window" in found

    report = mm.run_pass(tmp_path)
    assert report["expired"] == 1
    facts = _facts(tmp_path)
    assert "Captured decision about the release window" not in facts
    assert "Captured decision about the fresh window" in facts
    assert store.show("session", deliberate["id"])["status"] == "active"
    # The expiry is a real tombstone, which is what replicates to peers.
    db = sqlite3.connect(tmp_path / "memory" / "memory.db")
    try:
        assert db.execute("SELECT count(*) FROM ops WHERE kind='tombstone'").fetchone()[0] == 1
    finally:
        db.close()


def test_a_legacy_capture_without_a_deadline_expires_from_its_source_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(mm, "CAPTURE_TTL_S", 14 * 86400)
    aged = time.time() - 15 * DAY
    _capture(store, "legacy", "Captured before stamping existed", aged)
    # Rewrite the row as a pre-B4 capture: no expires_at.
    db = sqlite3.connect(tmp_path / "memory" / "memory.db")
    db.execute("UPDATE current SET fact=json_set(fact,'$.expires_at',json('null'))")
    db.commit()
    db.close()
    assert mm.run_pass(tmp_path)["expired"] == 1
    assert "Captured before stamping existed" not in _facts(tmp_path)


def test_promoting_a_capture_clears_its_deadline(tmp_path: Path) -> None:
    from local_operator.memory_cli import execute

    store = _store(tmp_path)
    _capture(store, "m1", "Captured then reviewed by the operator", time.time())
    entity = next(iter(_facts(tmp_path).values()))
    db = sqlite3.connect(tmp_path / "memory" / "memory.db")
    entity_id, version = db.execute("SELECT entity_id, op_id FROM current").fetchone()
    db.close()
    parser = __import__("argparse").ArgumentParser()
    from local_operator.memory_cli import configure_parser

    configure_parser(parser)
    args = parser.parse_args(
        [
            "edit",
            f"session/{entity_id}",
            "--revision",
            version,
            "--title",
            "Reviewed",
            "--session",
            "a" * 12,
        ]
    )
    execute(args, config=tmp_path, session_dir=tmp_path / "sessions" / ("a" * 12))
    fact = _facts(tmp_path)[entity["body"]]
    assert fact["status"] == "active"
    assert fact["expires_at"] is None


def test_delete_session_forgets_captures_and_keeps_deliberate_facts(tmp_path: Path) -> None:
    from local_operator.session.cleanup import delete_session, mark_store

    session = "a" * 12
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True)
    mark_store(sessions)
    directory = sessions / session
    directory.mkdir()
    (directory / "transcript.jsonl").write_text(
        json.dumps({"type": "message", "payload": {"role": "user", "content": "hi"}}) + "\n",
        encoding="utf-8",
    )
    store = _store(tmp_path, session)
    _capture(store, "m1", "Captured goal for the deleted session", time.time())
    _capture(store, "m2", "Captured decision for the deleted session", time.time())
    store.put("session", {"title": "Kept", "body": "Deliberate fact of the deleted session"})
    other = _store(tmp_path, "b" * 12)
    _capture(other, "m1", "Captured fact of a neighbour session", time.time())

    outcome = delete_session(tmp_path, session, actor="test")
    assert outcome.deleted is True
    facts = _facts(tmp_path)
    assert set(facts) == {
        "Deliberate fact of the deleted session",
        "Captured fact of a neighbour session",
    }
