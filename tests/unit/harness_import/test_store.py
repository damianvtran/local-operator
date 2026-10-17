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
    rows = list(store.iter_jsonl(path))
    assert [row["id"] for _, row in rows] == ["kept"]


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
    before = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.iterdir()
    }
    with store.sqlite_snapshot(config, database) as snapshot:
        assert snapshot.path.exists()
        assert snapshot.path != database
    after = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.iterdir()
    }
    assert after == before
    assert not (root / "state_5.sqlite-shm").exists()
    assert not (root / "state_5.sqlite-wal").exists()
