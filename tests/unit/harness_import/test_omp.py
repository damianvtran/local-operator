"""omp adapter: active branch, compaction anchor, custom messages, injection."""

from __future__ import annotations

import json

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.types import ImportRequest
from local_operator.session.transcript import Transcript

SOURCE = "omp"
SESSION_ID = "omp-synthetic"


@pytest.mark.asyncio
async def test_omp_active_branch_and_compaction(omp_store):
    home, config = omp_store
    service = HarnessImportService(config)
    sessions = await service.list_sessions(SOURCE)
    assert [row.source_id for row in sessions] == [SESSION_ID]
    assert sessions[0].title == "Synthetic omp session"

    result = await service.import_sessions(
        ImportRequest(sources=[SOURCE], session_ids={SOURCE: [SESSION_ID]})
    )
    row = result.sessions[0]
    assert row.outcome == "imported"
    assert row.compactions == 1
    assert row.tool_calls == 1
    assert row.tool_results == 1

    transcript = Transcript(config / "sessions" / row.target_session_id)
    entries = transcript.entries()
    raw = transcript.path.read_text()
    assert "inactive branch text" not in raw
    assert "private omp reasoning" not in raw
    assert "injected omp note" in raw
    assert "[imported from omp]" in raw

    compaction = next(entry for entry in entries if entry.type == "compaction")
    assert compaction.payload["summary"] == "OMP SUMMARY"
    assert compaction.payload["tokens_before"] == 4321
    anchor = next(
        entry for entry in entries if entry.id == compaction.payload["first_kept_entry_id"]
    )
    assert anchor.payload["content"][0]["text"] == "kept omp question"

    history = transcript.build_llm_history()
    assert isinstance(history[0], CustomMessage)
    assert history[0].details["summary"] == "OMP SUMMARY"
    texts = []
    for message in history[1:]:
        if isinstance(message, Message) and message.role == "user":
            texts.append(message.text)
    assert texts == [
        "first omp question",
        "kept omp question",
        "[imported from omp] injected omp note",
    ]

    # Tool call and result stay adjacent in source order and keep their ids.
    roles = [entry.payload.get("role") for entry in entries if entry.type == "message"]
    assert roles[:4] == ["user", "assistant", "tool", "assistant"]
    tool_entry = next(entry for entry in entries if entry.payload.get("role") == "tool")
    assert tool_entry.payload["tool_call_id"] == "call-omp-1"

    title = json.loads((config / "sessions" / row.target_session_id / "title.json").read_text())
    assert title["text"] == "Synthetic omp session"


@pytest.mark.asyncio
async def test_omp_branch_leaf_is_metadata_row(omp_store):
    """A trailing metadata custom row is the newest leaf; the walk must not stop."""
    home, config = omp_store
    path = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["type"] == "custom"
    service = HarnessImportService(config)
    result = await service.import_sessions(
        ImportRequest(sources=[SOURCE], session_ids={SOURCE: [SESSION_ID]})
    )
    transcript = Transcript(config / "sessions" / result.sessions[0].target_session_id)
    assert any(
        entry.type == "message"
        and entry.payload.get("content", [{}])[0].get("text") == "kept omp answer"
        for entry in transcript.entries()
    )
