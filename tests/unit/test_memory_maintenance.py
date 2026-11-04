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


# ---------------------------------------------------------------------------
# Review round 1: M1 (promotion/move), M2 (all heads), M3 (re-capture), m1
# ---------------------------------------------------------------------------


def _edit_through_tool(config: Path, session: str, path: str, version: str, new: str) -> Any:
    """The AGENT's edit path (``memory`` tool), not the CLI's."""
    from local_operator.harness.types import ToolContext
    from local_operator.tools.memory_tools import execute_memory

    context = ToolContext(
        session_id=session,
        memory_config_dir=str(config),
        session_dir=str(config / "sessions" / session),
    )
    result = asyncio.run(
        execute_memory("e", "edit", {"path": path, "old_text": version, "new_text": new}, context)
    )
    assert not result.is_error, result.content
    return json.loads(str(getattr(result.content[0], "text", "")))


def _only_row(config: Path) -> tuple[str, str, dict[str, Any]]:
    db = sqlite3.connect(config / "memory" / "memory.db")
    try:
        rows = db.execute("SELECT entity_id, op_id, fact, scope FROM current").fetchall()
    finally:
        db.close()
    assert len(rows) == 1, rows
    return rows[0][0], rows[0][1], json.loads(rows[0][2])


def _at_day_15(monkeypatch: pytest.MonkeyPatch, start: float) -> float:
    later = start + 15 * DAY
    monkeypatch.setattr(time, "time", lambda: later)
    return later


@pytest.mark.parametrize("status", ["active", "candidate"])
def test_an_agent_edit_through_the_tool_makes_a_capture_deliberate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """M1 + m1: an agent's edit (promotion OR a body correction that keeps
    ``candidate``) is deliberate: no carried deadline, no capture author, and it
    survives day 15 in search and in the pass."""
    session = "a" * 12
    start = time.time()
    _capture(_store(tmp_path), "m1", "Ship the context reduction POC this week", start)
    entity, version, fact = _only_row(tmp_path)
    edited = dict(fact, status=status, body="Ship the context reduction POC by Friday")
    _edit_through_tool(tmp_path, session, f"memory://session/{entity}", version, json.dumps(edited))
    _, _, now_fact = _only_row(tmp_path)
    assert now_fact["expires_at"] is None
    assert now_fact["source"]["author"] != "deterministic-capture"
    _at_day_15(monkeypatch, start)
    assert _store(tmp_path).search("context reduction POC")
    assert mm.run_pass(tmp_path)["expired"] == 0
    assert "Ship the context reduction POC by Friday" in _facts(tmp_path)


def test_a_capture_moved_to_global_survives_day_15(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: a move is deliberate; the moved fact keeps no deadline."""
    session = "a" * 12
    start = time.time()
    _capture(_store(tmp_path), "m1", "Damian prefers conventional commit messages", start)
    entity, version, _ = _only_row(tmp_path)
    _edit_through_tool(
        tmp_path, session, f"memory://session/{entity}", version, '{"move":"global"}'
    )
    _, _, moved = _only_row(tmp_path)
    assert moved["expires_at"] is None
    _at_day_15(monkeypatch, start)
    assert _store(tmp_path).search("conventional commit messages")
    assert mm.run_pass(tmp_path)["expired"] == 0


def _peer_correction(store: MemoryStore, entity: str, parent: dict[str, Any]) -> None:
    """A deliberate correction of ``parent`` from another device (concurrent head)."""
    import hashlib

    from local_operator.memory import _dump

    fact = dict(
        parent["fact"],
        status="active",
        body="Confirmed by the operator on the peer",
        source=dict(parent["fact"]["source"], author="main"),
        expires_at=None,
    )
    op = dict(
        parent,
        op_id="peer-correction-op",
        origin_device="device-b",
        seq=1,
        hlc=[parent["hlc"][0] + 1, 0, "device-b"],
        parents=[parent["op_id"]],
        fact=fact,
        body_sha=hashlib.sha256(_dump(fact).encode()).hexdigest(),
    )
    store.apply_ops([op])


@pytest.mark.parametrize("door", ["expiry", "session-delete"])
def test_a_capture_with_a_concurrent_deliberate_head_is_never_forgotten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, door: str
) -> None:
    """M2, the reviewer's repro: capture rev1, a local capture refresh rev2, and a
    peer's deliberate correction of rev1 -> two heads. Neither unattended door may
    tombstone the entity."""
    store = _store(tmp_path)
    old = time.time() - 20 * DAY
    _capture(store, "m1", "Captured goal that a peer confirmed", old)
    entity, _, _ = _only_row(tmp_path)
    with store.connection() as db:
        first = store._ops(db, entity)[0]
    _capture(store, "m2", "Captured goal that a peer confirmed", old + 1)
    _peer_correction(store, entity, first)
    with store.connection() as db:
        assert len(store._heads(store._ops(db, entity))) == 2
    if door == "expiry":
        report = mm.run_pass(tmp_path)
        assert report["expired"] == 0
    else:
        assert mm.forget_session_captures(tmp_path, "a" * 12) == 0
    with store.connection() as db:
        ops = store._ops(db, entity)
    assert not any(op["kind"] == "tombstone" for op in ops)
    assert any(op["op_id"] == "peer-correction-op" for op in ops)


def test_an_expired_capture_is_recaptured_but_an_explicit_forget_is_not(tmp_path: Path) -> None:
    """M3: a resumed session re-learns text that merely expired; text a person or
    an agent forgot stays forgotten."""
    store = _store(tmp_path)
    old = time.time() - 15 * DAY
    _capture(store, "m1", "Standing goal of the long lane", old)
    _capture(store, "m1b", "Text the operator deliberately forgot", old)
    forgotten = next(
        (eid, op)
        for eid, op, fact, _ in [
            (r[0], r[1], json.loads(r[2]), None)
            for r in sqlite3.connect(tmp_path / "memory" / "memory.db").execute(
                "SELECT entity_id, op_id, fact FROM current"
            )
        ]
        if fact["body"] == "Text the operator deliberately forgot"
    )
    store.put("session", {}, entity=forgotten[0], expected_version=forgotten[1], forget=True)
    assert mm.run_pass(tmp_path)["expired"] == 1
    assert _facts(tmp_path) == {}

    now = time.time()
    _capture(store, "m2", "Standing goal of the long lane", now)
    _capture(store, "m2b", "Text the operator deliberately forgot", now)
    facts = _facts(tmp_path)
    assert set(facts) == {"Standing goal of the long lane"}
    assert facts["Standing goal of the long lane"]["expires_at"] == int(now) + 14 * 86400
    assert _store(tmp_path).search("Standing goal long lane")
    # Deterministic: a second pass over the same text lands on the same entity.
    _capture(store, "m3", "Standing goal of the long lane", now + 60)
    assert len(_facts(tmp_path)) == 1


def test_the_expiry_step_stops_at_its_budget_and_leaves_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m2: expiry has its own wall-clock budget; leftovers wait for the next pass."""
    monkeypatch.setattr(mm, "EXPIRE_BUDGET_S", 0.0)
    old = time.time() - 15 * DAY
    for index in range(3):
        _capture(_store(tmp_path, f"{index:012x}"), "m1", f"Old capture number {index}", old)
    report = mm.run_pass(tmp_path)
    assert report["expired"] == 0
    assert "expiry_seconds" in report
    assert len(_facts(tmp_path)) == 3


def test_lop_memory_status_does_not_start_a_maintenance_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA Q1/Q2: a one-shot command never offers (or waits on) the pass."""
    from local_operator.memory_cli import configure_parser, execute

    calls: list[Any] = []
    monkeypatch.setattr(mm, "start_background", lambda *a: calls.append(a))
    monkeypatch.setattr(mm, "run_pass", lambda *a, **k: calls.append(a))
    _store(tmp_path).put("session", {"title": "t", "body": "a fact"})
    import argparse

    parser = argparse.ArgumentParser()
    configure_parser(parser)
    execute(parser.parse_args(["status"]), config=tmp_path)
    assert calls == []


# ---------------------------------------------------------------------------
# Review round 2: M3-r2 (a peer's expiry) and Q5 (attempt backoff)
# ---------------------------------------------------------------------------


def _sync(source: MemoryStore, target: MemoryStore, exclude: str) -> None:
    """Apply every op ``source`` holds from origins other than ``exclude`` (the
    target's own) to ``target``, in seq order. The pass mints the store's origin
    itself, so the source's origin is read from its rows, not assumed."""
    db = sqlite3.connect(source.path)
    try:
        rows = db.execute(
            "SELECT origin, data FROM ops WHERE origin!=? ORDER BY origin, seq", (exclude,)
        ).fetchall()
    finally:
        db.close()
    ops = [json.loads(data) for _, data in rows]
    target.apply_ops(ops, origins={origin for origin, _ in rows})


def _two_hosts(tmp_path: Path, *, b_has_session_dir: bool) -> tuple[MemoryStore, MemoryStore]:
    session = "a" * 12
    host_a = _store(tmp_path / "a", session)
    host_b = MemoryStore(
        tmp_path / "b",
        MemoryIdentity(session),
        tmp_path / "b" / "sessions" / session,
        origin="device-b",
    )
    old = time.time() - 15 * DAY
    _capture(host_a, "m1", "Standing goal of the long lane", old)
    if b_has_session_dir:
        (tmp_path / "b" / "sessions" / session).mkdir(parents=True)
    _sync(host_a, host_b, exclude="none")
    assert "Standing goal of the long lane" in _facts(tmp_path / "b")
    return host_a, host_b


def test_a_peers_expiry_syncs_back_and_the_capturing_host_can_recapture(tmp_path: Path) -> None:
    """M3-r2: B (which also holds the session) expires; A applies B's tombstone and
    must still be able to re-learn the text when the session resumes."""
    host_a, host_b = _two_hosts(tmp_path, b_has_session_dir=True)
    assert mm.run_pass(tmp_path / "b")["expired"] == 1
    _sync(host_b, host_a, exclude="device-a")
    assert _facts(tmp_path / "a") == {}

    now = time.time()
    _capture(host_a, "m2", "Standing goal of the long lane", now)
    assert set(_facts(tmp_path / "a")) == {"Standing goal of the long lane"}
    assert _store(tmp_path / "a").search("Standing goal long lane")
    # No local mark exists any more: the recognition comes from the synced op.
    db = sqlite3.connect(host_a.path)
    try:
        assert (
            db.execute("SELECT count(*) FROM meta WHERE key LIKE 'capture-expired:%'").fetchone()[0]
            == 0
        )
    finally:
        db.close()


def test_a_peer_never_expires_a_session_whose_directory_it_does_not_hold(tmp_path: Path) -> None:
    """M3-r2: only the host holding the session directory decides its expiry."""
    _two_hosts(tmp_path, b_has_session_dir=False)
    report = mm.run_pass(tmp_path / "b")
    assert report["expired"] == 0
    assert "Standing goal of the long lane" in _facts(tmp_path / "b")
    # ...while the owning host does expire it.
    assert mm.run_pass(tmp_path / "a")["expired"] == 1


def test_a_synced_deliberate_forget_still_keeps_the_text_out(tmp_path: Path) -> None:
    """M3-r2 control: a peer's EXPLICIT forget arriving by sync stays final."""
    host_a, host_b = _two_hosts(tmp_path, b_has_session_dir=True)
    db = sqlite3.connect(host_b.path)
    entity, version = db.execute("SELECT entity_id, op_id FROM current").fetchone()
    db.close()
    host_b.put("session", {}, entity=entity, expected_version=version, forget=True)
    _sync(host_b, host_a, exclude="device-a")
    _capture(host_a, "m2", "Standing goal of the long lane", time.time())
    assert _facts(tmp_path / "a") == {}


def test_a_cut_off_heavy_attempt_backs_off_instead_of_restarting(
    tmp_path: Path, low_threshold: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q5: the attempt is stamped BEFORE the plan, so a host killed mid-plan does
    not re-pay it on every start; after the backoff the work resumes."""
    store = _store(tmp_path)
    _churn(store)

    def killed(*_a: Any, **_k: Any) -> Any:
        raise KeyboardInterrupt  # stands in for the interpreter exiting mid-plan

    monkeypatch.setattr(MemoryStore, "journal_plan", killed)
    with pytest.raises(KeyboardInterrupt):
        mm._run(tmp_path, time.time(), time.monotonic() + 60, {"status": "ran"})
    monkeypatch.undo()
    monkeypatch.setattr(mm, "RECLAIMABLE_OPS_THRESHOLD", 3)
    assert mm.run_pass(tmp_path)["status"] == "backoff"
    assert _prunes(tmp_path) == []
    later = time.time() + mm.ATTEMPT_BACKOFF_S + 1
    assert mm.run_pass(tmp_path, now=later)["status"] == "ran"
    assert [reason for _, reason in _prunes(tmp_path)] == ["auto"]


def test_a_conflicted_capture_does_not_take_a_slot_in_the_expiry_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q6: with a cap of 1, the conflicted entity is skipped at selection and the
    clean capture behind it is expired in the same pass."""
    monkeypatch.setattr(mm, "EXPIRE_LIMIT", 1)
    store = _store(tmp_path)
    old = time.time() - 20 * DAY
    _capture(store, "m1", "AAA conflicted capture text", old)
    _capture(store, "m1b", "ZZZ clean capture text", old)
    db = sqlite3.connect(store.path)
    rows = {json.loads(f)["body"]: e for e, f in db.execute("SELECT entity_id, fact FROM current")}
    db.close()
    with store.connection() as db:
        first = store._ops(db, rows["AAA conflicted capture text"])[0]
    _capture(store, "m2", "AAA conflicted capture text", old + 1)
    _peer_correction(store, rows["AAA conflicted capture text"], first)
    assert mm.run_pass(tmp_path)["expired"] == 1
    assert "ZZZ clean capture text" not in _facts(tmp_path)


# ---------------------------------------------------------------------------
# Review round 3: M1-r3 (retention-reaped sessions) and m1 (clock skew)
# ---------------------------------------------------------------------------


def test_a_capture_this_host_wrote_expires_after_retention_reaps_its_session(
    tmp_path: Path,
) -> None:
    """M1-r3: retention removes the directory and leaves memory to the TTL, so
    the writing host must still expire the capture once the directory is gone."""
    import shutil

    session = "a" * 12
    # Production-shaped: no ``origin=`` override, so the origin is the install's
    # minted device id persisted in ``meta``, exactly what the pass compares.
    store = MemoryStore(tmp_path, MemoryIdentity(session), tmp_path / "sessions" / session)
    _capture(store, "m1", "Captured goal of a reaped delegated run", time.time() - 15 * DAY)
    directory = tmp_path / "sessions" / session
    assert directory.is_dir()
    shutil.rmtree(directory)  # what delegated_max_age's remove_session_dir does
    report = mm.run_pass(tmp_path)
    assert report["expired"] == 1
    assert _facts(tmp_path) == {}


def test_a_peer_that_neither_holds_the_session_nor_wrote_the_capture_never_expires_it(
    tmp_path: Path,
) -> None:
    """M1-r3 control: the origin rule widens expiry only to the WRITER."""
    _, host_b = _two_hosts(tmp_path, b_has_session_dir=False)
    report = mm.run_pass(tmp_path / "b")
    assert report["expired"] == 0
    assert "Standing goal of the long lane" in _facts(tmp_path / "b")


def test_a_future_dated_attempt_stamp_does_not_hold_passes_in_backoff(
    tmp_path: Path, low_threshold: None
) -> None:
    """Review round 3, m1: a clock stepped back must not freeze maintenance."""
    _churn(_store(tmp_path))
    (tmp_path / "memory" / mm.ATTEMPT_NAME).write_text(f"{time.time() + 3 * DAY}\n")
    assert mm.run_pass(tmp_path)["status"] == "ran"


# ---------------------------------------------------------------------------
# Rebase onto #3059 (capture eviction + recall telemetry) and review r5 m1
# ---------------------------------------------------------------------------


def _recall_rows(config: Path) -> set[str]:
    db = sqlite3.connect(config / "memory" / "memory.db")
    try:
        return {row[0] for row in db.execute("SELECT entity FROM recall_stats")}
    finally:
        db.close()


def test_an_expiry_tombstone_drops_the_facts_recall_telemetry(tmp_path: Path) -> None:
    """#3059 deletes ``recall_stats`` for a tombstoned entity in ``_reindex``; the
    expiry pass tombstones through ``forget_many``, so its rows go too."""
    store = _store(tmp_path)
    _capture(store, "m1", "Recalled once then left to go stale", time.time() - 15 * DAY)
    entity, _, _ = _only_row(tmp_path)
    store.note_recall([entity])
    assert _recall_rows(tmp_path) == {entity}
    assert mm.run_pass(tmp_path)["expired"] == 1
    assert _recall_rows(tmp_path) == set()


def test_eviction_and_expiry_tombstones_are_never_mistaken_for_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An EVICTION (#3059, budget) stays final for its text; an EXPIRY (this PR)
    is re-capturable and is never counted as an eviction."""
    from local_operator import memory_capture
    from local_operator.memory import is_expiry_tombstone

    store = _store(tmp_path)
    # The evicted row must still be INSIDE its TTL: a capture past its deadline is the
    # expiry sweep's to retire, never eviction's (PR #3057 review r6, m1 — eviction is
    # final, expiry is re-capturable, so evicting a stale row first loses the text).
    live = time.time() - 60
    _capture(store, "m1", "Oldest capture the budget will evict first", live)
    _capture(
        store, "stale", "Newer capture that will merely expire later on", time.time() - 15 * DAY
    )
    monkeypatch.setattr(memory_capture, "CAPTURE_SESSION_TOKENS", 20)
    _capture(store, "m2", "Live capture that makes the budget bind", live + 30)
    assert memory_capture.capture_status(store)["evicted"] == 1
    db = sqlite3.connect(tmp_path / "memory" / "memory.db")
    try:
        [evict] = [
            json.loads(r[0]) for r in db.execute("SELECT data FROM ops WHERE kind='tombstone'")
        ]
    finally:
        db.close()
    assert not is_expiry_tombstone(evict)
    monkeypatch.setattr(memory_capture, "CAPTURE_SESSION_TOKENS", 16_000)

    assert mm.run_pass(tmp_path)["expired"] == 1
    db = sqlite3.connect(tmp_path / "memory" / "memory.db")
    try:
        kinds = [
            is_expiry_tombstone(json.loads(r[0]))
            for r in db.execute("SELECT data FROM ops WHERE kind='tombstone'")
        ]
    finally:
        db.close()
    assert sorted(kinds) == [False, True]
    # Expiry is not an eviction in status.
    assert memory_capture.capture_status(store)["evicted"] == 1

    now = time.time()
    _capture(store, "m3", "Oldest capture the budget will evict first", now)
    _capture(store, "m4", "Newer capture that will merely expire later on", now)
    # The EVICTED text stays gone; the EXPIRED one is re-capturable, and the live row
    # the budget evicted first is the only thing that keeps it out of ``current``.
    assert set(_facts(tmp_path)) == {
        "Newer capture that will merely expire later on",
        "Live capture that makes the budget bind",
    }


def test_owed_ids_whose_op_compaction_dropped_are_pruned(tmp_path: Path) -> None:
    """Review r5 m1: an owed id whose op no longer exists is dropped on read and
    on save, so a still-legacy peer stops reporting it as pending."""
    from local_operator.network import memory as net

    store = _store(tmp_path)
    live = store.put("session", {"title": "Kept", "body": "still in the journal"})
    store.put("session", {"title": "Other", "body": "creates the store"})
    db = sqlite3.connect(tmp_path / "memory" / "memory.db")
    try:
        [held] = [
            r[0] for r in db.execute("SELECT op_id FROM ops WHERE entity_id=?", (live["id"],))
        ]
    finally:
        db.close()
    gone = "op-compaction-deleted"
    net._save_legacy_pending(store, "peer-legacy", [gone, held], False)
    assert net.legacy_pending(store, "peer-legacy") == {"ids": [held], "overflow": False}
    # Written before the prune existed: the read path drops it as well.
    with store.connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)",
            (net._legacy_key("peer-legacy"), json.dumps({"ids": [gone], "overflow": False})),
        )
        conn.commit()
    assert net.legacy_pending(store, "peer-legacy") == {"ids": [], "overflow": False}
    net._save_legacy_pending(store, "peer-legacy", [gone], False)
    with store.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM meta WHERE key=?", (net._legacy_key("peer-legacy"),)
        ).fetchone()
    assert row is None


def test_stale_ids_alone_do_not_force_a_full_resend(tmp_path: Path, monkeypatch) -> None:
    """PR #3057 review r6, n1: the id cap ran BEFORE stale ids were dropped.

    Compaction deletes ops, so a long-lived peer's withheld list accumulates ids this
    store no longer holds. Counted against ``LEGACY_WITHHELD_CAP`` they collapsed a
    list that already fit into ``overflow`` — a full redelivery refusal over ids
    nothing can ever redeliver, which is exactly what the prune exists to prevent.
    """
    from local_operator.network import memory as net

    store = _store(tmp_path)
    held = [
        store.put("session", {"title": f"Kept {n}", "body": f"owed body {n}"})["version"]
        for n in range(2)
    ]
    monkeypatch.setattr(net, "LEGACY_WITHHELD_CAP", 2)
    dead = [f"op-compaction-deleted-{n}" for n in range(5)]
    net._save_legacy_pending(store, "peer-legacy", [*dead, *held], False)
    assert net.legacy_pending(store, "peer-legacy") == {"ids": held, "overflow": False}
