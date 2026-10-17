"""Codex adapter: replacement history, event dedupe and thread rollback."""

from __future__ import annotations

import pytest

from local_operator.harness.types import CustomMessage
from local_operator.harness_import import store
from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.sources.codex import CodexAdapter
from local_operator.harness_import.types import ImportRequest
from local_operator.session.transcript import Transcript

SOURCE = "codex"


@pytest.mark.asyncio
async def test_codex_lists_threads_from_a_live_wal(isolated, monkeypatch):
    home, config = isolated
    root = home / ".codex"
    root.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(root))
    import sqlite3

    writer = sqlite3.connect(root / "state_5.sqlite")
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        "CREATE TABLE threads (id TEXT, rollout_path TEXT, created_at INTEGER, "
        "updated_at INTEGER, cwd TEXT, title TEXT, archived INTEGER, "
        "cli_version TEXT, first_user_message TEXT, model TEXT, model_provider TEXT)"
    )
    writer.execute(
        "INSERT INTO threads VALUES ('live-1', '', 1, 2, '/tmp', 'live', 0, '', '', '', '')"
    )
    writer.commit()
    try:
        service = HarnessImportService(config)
        sessions = await service.list_sessions(SOURCE)
        assert [row.source_id for row in sessions] == ["live-1"]
        assert sessions[0].availability == "metadata_only"
        info = service.adapters[SOURCE].detect()
        assert info.session_count == 1
        assert info.availability == "metadata_only"
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_codex_replacement_history_and_rollback(codex_store):
    home, config = codex_store
    service = HarnessImportService(config)
    sessions = await service.list_sessions(SOURCE)
    by_id = {row.source_id: row for row in sessions}
    assert set(by_id) == {"codex-synthetic", "codex-rollback"}
    assert all(row.availability == "full" for row in by_id.values())

    synthetic = await service.import_sessions(
        ImportRequest(sources=[SOURCE], session_ids={SOURCE: ["codex-synthetic"]})
    )
    row = synthetic.sessions[0]
    assert row.outcome == "imported"
    assert row.compactions == 1
    transcript = Transcript(config / "sessions" / row.target_session_id)
    raw = transcript.path.read_text()
    # The duplicate event_msg copy of a canonical assistant text is not imported.
    assert raw.count("first codex answer") == 1

    compaction = next(entry for entry in transcript.entries() if entry.type == "compaction")
    assert compaction.payload["summary"] == "CODEX SUMMARY"
    snapshot = compaction.payload["preserve_data"]["source_snapshot"]
    assert snapshot["window_id"] == 7
    assert snapshot["window_number"] == 3
    assert snapshot["replacement_history_count"] == 3
    anchor = next(
        entry
        for entry in transcript.entries()
        if entry.id == compaction.payload["first_kept_entry_id"]
    )
    assert anchor.payload["content"][0]["text"] == "kept codex question"

    history = transcript.build_llm_history()
    assert isinstance(history[0], CustomMessage)
    assert history[0].details["summary"] == "CODEX SUMMARY"
    assert [getattr(message, "role", "") for message in history[1:]] == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    rollback = await service.import_sessions(
        ImportRequest(sources=[SOURCE], session_ids={SOURCE: ["codex-rollback"]})
    )
    rollback_transcript = Transcript(config / "sessions" / rollback.sessions[0].target_session_id)
    rollback_raw = rollback_transcript.path.read_text()
    assert "rolled back question" not in rollback_raw
    assert "rolled back answer" not in rollback_raw
    assert "keep this question" in rollback_raw
    assert "after rollback question" in rollback_raw


def test_codex_unreadable_thread_index_is_surfaced(isolated, monkeypatch):
    home, config = isolated
    root = home / ".codex"
    root.mkdir()
    (root / "state_5.sqlite").write_bytes(b"not a sqlite database")
    monkeypatch.setenv("CODEX_HOME", str(root))
    adapter = CodexAdapter(config)
    assert adapter.list_sessions() == []
    info = adapter.detect()
    assert "thread index unreadable" in info.detail


def test_zst_size_is_checked_before_decompression(isolated, monkeypatch):
    home, config = isolated
    root = home / ".codex"
    root.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(root))
    sessions = root / "sessions" / "2026" / "01" / "01"
    sessions.mkdir(parents=True)
    body = sessions / "rollout-large.jsonl.zst"
    with body.open("wb") as handle:
        handle.truncate(store.MAX_SOURCE_BYTES + 1)
    adapter = CodexAdapter(config)
    rows = adapter.list_sessions()
    assert len(rows) == 1
    assert rows[0].availability == "unsupported"
    assert "import cap" in rows[0].warnings[0]
