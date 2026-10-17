"""Codex adapter: replacement history, event dedupe and thread rollback."""

from __future__ import annotations

import pytest

from local_operator.harness.types import CustomMessage
from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.types import ImportRequest
from local_operator.session.transcript import Transcript

SOURCE = "codex"


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
    rollback_transcript = Transcript(
        config / "sessions" / rollback.sessions[0].target_session_id
    )
    rollback_raw = rollback_transcript.path.read_text()
    assert "rolled back question" not in rollback_raw
    assert "rolled back answer" not in rollback_raw
    assert "keep this question" in rollback_raw
    assert "after rollback question" in rollback_raw
