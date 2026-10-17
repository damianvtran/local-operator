"""Bounded streaming, fingerprinting and the SQLite private-snapshot contract."""

from __future__ import annotations

import json
import sqlite3

from local_operator.harness_import import store


def test_iter_jsonl_skips_malformed_and_truncated_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps({"type": "ok", "id": "1"})
        + "\n"
        + "{not json\n"
        + json.dumps({"type": "ok", "id": "2"})
        + "\n"
        + '{"type":"truncated"'
    )
    rows = list(store.iter_jsonl(path))
    assert [row["id"] for _, row in rows] == ["1", "2"]


def test_iter_jsonl_skips_an_oversized_line(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    path.write_text("x" * 4096 + "\n" + json.dumps({"id": "kept"}) + "\n")
    monkeypatch.setattr(store, "MAX_LINE_BYTES", 128)
    stats: dict[str, int] = {}
    rows = list(store.iter_jsonl(path, stats=stats))
    assert [row["id"] for _, row in rows] == ["kept"]
    assert stats["skipped_lines"] == 1
    assert stats["oversized_lines"] == 1


def test_fingerprint_changes_on_append(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text("one\n")
    first = store.fingerprint(path)
    with path.open("a") as handle:
        handle.write("two\n")
    second = store.fingerprint(path)
    assert second["size"] == first["size"] + 4
    # A prefix that covers the whole old file necessarily changes; resume
    # compares the SAME leading span on both sides (service `_head_unchanged`).
    assert store.sha256_prefix(path, first["size"]) == first["sha256_prefix"]


def test_sqlite_snapshot_replays_a_live_wal(tmp_path):
    """The copy must be opened normally, not immutable: a live writer's WAL is
    uncheckpointed and ``immutable=1`` makes SQLite ignore it (zero threads)."""
    config = tmp_path / "config"
    root = tmp_path / "codex"
    root.mkdir()
    database = root / "state_5.sqlite"
    writer = sqlite3.connect(database)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        "CREATE TABLE threads (id TEXT, rollout_path TEXT, created_at INTEGER, "
        "updated_at INTEGER, cwd TEXT, title TEXT, archived INTEGER, "
        "cli_version TEXT, first_user_message TEXT, model TEXT, model_provider TEXT)"
    )
    writer.execute("INSERT INTO threads VALUES ('old', '', 1, 2, '/tmp', 't', 0, '', '', '', '')")
    writer.commit()
    writer.execute("INSERT INTO threads VALUES ('new', '', 2, 3, '/tmp', 't', 0, '', '', '', '')")
    writer.commit()
    assert (root / "state_5.sqlite-wal").stat().st_size > 0
    before = {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in root.iterdir()}
    try:
        with store.sqlite_snapshot(config, database) as snapshot:
            db = store.connect_snapshot(snapshot)
            try:
                ids = {row[0] for row in db.execute("SELECT id FROM threads")}
            finally:
                db.close()
        # Compare BEFORE closing the writer: closing a WAL connection
        # checkpoints (and removes) the sidecars, which is a source change this
        # test is not about.
        after = {
            path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in root.iterdir()
        }
    finally:
        writer.close()
    assert ids == {"old", "new"}
    assert after == before


def test_sqlite_snapshot_never_touches_the_source(tmp_path, monkeypatch):
    config = tmp_path / "config"
    root = tmp_path / "codex"
    root.mkdir()
    database = root / "state_5.sqlite"
    db = sqlite3.connect(database)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        "CREATE TABLE threads (id TEXT, rollout_path TEXT, created_at INTEGER, "
        "updated_at INTEGER, cwd TEXT, title TEXT, archived INTEGER, "
        "cli_version TEXT, first_user_message TEXT, model TEXT, model_provider TEXT)"
    )
    db.execute(
        "INSERT INTO threads VALUES "
        "('thread-1', '', 1, 2, '/tmp/x', 'title', 0, '0', 'first', 'm', 'p')"
    )
    db.commit()
    db.close()
    before = {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in root.iterdir()}
    with store.sqlite_snapshot(config, database) as snapshot:
        assert snapshot.path.exists()
        assert snapshot.path != database
    after = {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in root.iterdir()}
    assert after == before
    assert not (root / "state_5.sqlite-shm").exists()
    assert not (root / "state_5.sqlite-wal").exists()
