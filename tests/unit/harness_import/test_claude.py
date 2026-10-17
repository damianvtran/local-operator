"""Claude adapter: segments, boundaries, branch policy and replay semantics."""

from __future__ import annotations

import json

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.types import ImportRequest
from local_operator.session.transcript import Transcript

SOURCE = "claude"
SESSION_ID = "sess-claude"


@pytest.mark.asyncio
async def test_claude_two_boundaries_emit_every_segment(claude_store):
    home, config = claude_store
    service = HarnessImportService(config)
    sessions = await service.list_sessions(SOURCE)
    assert [row.source_id for row in sessions] == [SESSION_ID]
    assert sessions[0].compaction_count == 2
    assert sessions[0].title == "Synthetic Claude session"

    result = await service.import_sessions(
        ImportRequest(sources=[SOURCE], session_ids={SOURCE: [SESSION_ID]})
    )
    row = result.sessions[0]
    assert row.outcome == "imported"
    assert row.compactions == 2
    assert row.messages == 8
    assert row.tool_calls == 1
    assert row.tool_results == 1

    target = config / "sessions" / row.target_session_id
    transcript = Transcript(target)
    entries = transcript.entries()
    compactions = [entry for entry in entries if entry.type == "compaction"]
    assert len(compactions) == 2
    assert [entry.payload["summary"] for entry in compactions] == [
        "SUMMARY ONE: synthetic earlier work",
        "SUMMARY TWO: synthetic later work",
    ]
    assert [entry.payload["tokens_before"] for entry in compactions] == [1234, 2345]

    by_id = {entry.id: entry for entry in entries}
    latest_anchor = compactions[-1].payload["first_kept_entry_id"]
    anchor_entry = by_id[latest_anchor]
    assert anchor_entry.type == "message"
    assert anchor_entry.payload["role"] == "user"
    assert anchor_entry.payload["content"][0]["text"] == "third task"

    # Every pre-cut row stays on disk as the archive of what the summary replaced.
    raw = transcript.path.read_text()
    assert "hello world" in raw
    assert "SUMMARY TWO" in raw
    # Inactive branch, sidechain and thinking are never imported.
    assert "subagent-only text" not in raw
    assert "harness bookkeeping" not in raw
    assert "internal reasoning that must not be stored" not in raw

    # Source timestamps survive; no row carries import time.
    user_entry = next(
        entry
        for entry in entries
        if entry.type == "message"
        and str((entry.payload.get("content") or [{}])[0].get("text")) == "hello world"
    )
    assert user_entry.ts == pytest.approx(1767225601.0)

    history = transcript.build_llm_history()
    assert isinstance(history[0], CustomMessage)
    assert history[0].details["summary"] == "SUMMARY TWO: synthetic later work"
    assert [getattr(message, "role", "custom") for message in history] == [
        "custom",
        "user",
        "user",
        "user",
        "assistant",
    ]
    # Preserved user turns are replayed AFTER the summary, in source order,
    # before the kept window — that is what stops a summary from erasing the
    # user's own instructions.
    users = []
    for message in history[1:4]:
        assert isinstance(message, Message)
        users.append(message.text)
    assert users == [
        "hello world",
        "second task",
        "third task",
    ]

    # The latest compaction carries the user-authored turns from BOTH summarized
    # spans, not just the immediately preceding one.
    preserved = compactions[-1].payload["preserved_user_turns"]
    assert [turn["text"] for turn in preserved] == ["hello world", "second task"]
    assert compactions[-1].payload["preserved_turns_cap"] == 150_000

    title = json.loads((target / "title.json").read_text())
    assert title["text"] == "Synthetic Claude session"


@pytest.mark.asyncio
async def test_claude_legacy_summary_is_a_compaction(isolated):
    home, config = isolated
    root = home / ".claude" / "projects" / "encoded"
    root.mkdir(parents=True)
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "sessionId": "legacy",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"role": "user", "content": "legacy question"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "legacy",
            "timestamp": "2026-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "legacy answer"}],
            },
        },
        {
            "type": "summary",
            "uuid": "legacy-summary",
            "summary": "LEGACY SUMMARY",
            "leafUuid": "a1",
            "sessionId": "legacy",
            "timestamp": "2026-01-01T00:00:03Z",
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "la1",
            "sessionId": "legacy",
            "timestamp": "2026-01-01T00:00:04Z",
            "message": {"role": "user", "content": "after legacy summary"},
        },
    ]
    (root / "legacy.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    service = HarnessImportService(config)
    sessions = await service.list_sessions(SOURCE)
    assert len(sessions) == 1 and sessions[0].compaction_count == 1
    result = await service.import_sessions(
        ImportRequest(sources=[SOURCE], session_ids={SOURCE: ["legacy"]})
    )
    transcript = Transcript(config / "sessions" / result.sessions[0].target_session_id)
    compaction = next(entry for entry in transcript.entries() if entry.type == "compaction")
    assert compaction.payload["summary"] == "LEGACY SUMMARY"
    # leafUuid names the LAST SUMMARIZED message; the anchor is the first retained
    # one, never the leaf.
    anchor = next(
        entry
        for entry in transcript.entries()
        if entry.id == compaction.payload["first_kept_entry_id"]
    )
    assert anchor.payload["content"][0]["text"] == "after legacy summary"
