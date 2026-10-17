"""Idempotency, resume, user-edited targets and manifest capture policy."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from local_operator.harness_import.manifest import read_manifest
from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.store import sha256_file
from local_operator.harness_import.types import ImportRequest
from local_operator.memory_capture import import_capture_disabled
from local_operator.session.transcript import Transcript

SOURCE = "omp"
SESSION_ID = "omp-synthetic"


def _request() -> ImportRequest:
    return ImportRequest(sources=[SOURCE], session_ids={SOURCE: [SESSION_ID]})


@pytest.mark.asyncio
async def test_rerun_is_a_noop_and_source_is_untouched(omp_store):
    home, config = omp_store
    source = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    before = sha256_file(source)
    service = HarnessImportService(config)
    first = await service.import_sessions(_request())
    target = config / "sessions" / first.sessions[0].target_session_id
    transcript_before = transcript_hash(target)
    manifest = read_manifest(target)
    assert manifest and manifest["status"] == "complete"
    assert manifest["capture_memory"] is False
    assert import_capture_disabled(target) is True

    second = await service.import_sessions(_request())
    assert second.sessions[0].outcome == "skipped_already_imported"
    assert transcript_hash(target) == transcript_before
    assert sha256_file(source) == before


def transcript_hash(target) -> str:
    return hashlib.sha256((target / "transcript.jsonl").read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_grown_source_resumes_without_duplicates(omp_store):
    home, config = omp_store
    service = HarnessImportService(config)
    first = await service.import_sessions(_request())
    target = config / "sessions" / first.sessions[0].target_session_id
    before_count = len(Transcript(target).entries())

    path = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    appended = [
        {
            "type": "message",
            "id": "u-extra",
            "parentId": "meta1",
            "timestamp": "2026-01-03T00:00:11.000Z",
            "message": {
                "role": "user",
                "timestamp": "2026-01-03T00:00:11.000Z",
                "content": [{"type": "text", "text": "appended question"}],
            },
        },
        {
            "type": "message",
            "id": "a-extra",
            "parentId": "u-extra",
            "timestamp": "2026-01-03T00:00:12.000Z",
            "message": {
                "role": "assistant",
                "timestamp": "2026-01-03T00:00:12.000Z",
                "content": [{"type": "text", "text": "appended answer"}],
            },
        },
    ]
    with path.open("a") as handle:
        for record in appended:
            handle.write(json.dumps(record) + "\n")

    second = await service.import_sessions(_request())
    assert second.sessions[0].outcome == "resumed"
    entries = Transcript(target).entries()
    assert len(entries) == before_count + 2
    assert len({entry.id for entry in entries}) == len(entries)
    appended_rows = [
        entry
        for entry in entries
        if entry.payload.get("content", [{}])[0].get("text") == "appended question"
    ]
    assert len(appended_rows) == 1


@pytest.mark.asyncio
async def test_user_edited_target_gets_a_fresh_copy(omp_store):
    home, config = omp_store
    service = HarnessImportService(config)
    first = await service.import_sessions(_request())
    target = config / "sessions" / first.sessions[0].target_session_id
    with (target / "transcript.jsonl").open("a") as handle:
        handle.write("\n")

    second = await service.import_sessions(_request())
    row = second.sessions[0]
    assert row.outcome == "imported"
    assert row.target_session_id != first.sessions[0].target_session_id
    written = read_manifest(config / "sessions" / row.target_session_id)
    assert written is not None and written["status"] == "complete"


@pytest.mark.asyncio
async def test_rotated_source_creates_one_copy_then_skips(omp_store):
    home, config = omp_store
    path = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    service = HarnessImportService(config)
    first = await service.import_sessions(_request())
    original = first.sessions[0].target_session_id

    path.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "omp-synthetic",
                "timestamp": "2026-01-04T00:00:00.000Z",
                "cwd": "/tmp/synthetic-omp",
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "message",
                "id": "rotated-1",
                "parentId": None,
                "timestamp": "2026-01-04T00:00:01.000Z",
                "message": {
                    "role": "user",
                    "timestamp": "2026-01-04T00:00:01.000Z",
                    "content": [{"type": "text", "text": "rotated source"}],
                },
            }
        )
        + "\n"
    )
    second = await service.import_sessions(_request())
    assert second.sessions[0].outcome == "imported"
    copy = second.sessions[0].target_session_id
    assert copy == f"{original}-2"
    third = await service.import_sessions(_request())
    assert third.sessions[0].outcome == "skipped_already_imported"
    assert third.sessions[0].target_session_id == copy
    copied = read_manifest(config / "sessions" / copy)
    assert copied is not None
    assert copied["predecessor_target_session_id"] == original


@pytest.mark.asyncio
async def test_edited_target_creates_one_copy_then_skips(omp_store):
    home, config = omp_store
    service = HarnessImportService(config)
    first = await service.import_sessions(_request())
    original = first.sessions[0].target_session_id
    with (config / "sessions" / original / "transcript.jsonl").open("a") as handle:
        handle.write("\n")
    second = await service.import_sessions(_request())
    copy = second.sessions[0].target_session_id
    assert copy == f"{original}-2"
    third = await service.import_sessions(_request())
    assert third.sessions[0].outcome == "skipped_already_imported"
    assert third.sessions[0].target_session_id == copy
    copied = read_manifest(config / "sessions" / copy)
    assert copied is not None
    assert copied["predecessor_target_session_id"] == original


@pytest.mark.asyncio
async def test_branch_change_gets_a_new_target_and_predecessor(omp_store):
    home, config = omp_store
    path = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    service = HarnessImportService(config)
    first = await service.import_sessions(_request())
    original = first.sessions[0].target_session_id

    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "message",
                    "id": "branch-question",
                    "parentId": "u1",
                    "timestamp": "2026-01-03T00:00:20.000Z",
                    "message": {
                        "role": "user",
                        "timestamp": "2026-01-03T00:00:20.000Z",
                        "content": [{"type": "text", "text": "REWRITTEN BRANCH QUESTION"}],
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "message",
                    "id": "branch-answer",
                    "parentId": "branch-question",
                    "timestamp": "2026-01-03T00:00:21.000Z",
                    "message": {
                        "role": "assistant",
                        "timestamp": "2026-01-03T00:00:21.000Z",
                        "content": [{"type": "text", "text": "REWRITTEN BRANCH ANSWER"}],
                    },
                }
            )
            + "\n"
        )
    second = await service.import_sessions(_request())
    copy = second.sessions[0].target_session_id
    assert second.sessions[0].outcome == "imported"
    assert copy == f"{original}-2"
    assert (
        "REWRITTEN BRANCH QUESTION" in (config / "sessions" / copy / "transcript.jsonl").read_text()
    )
    assert (
        "REWRITTEN BRANCH QUESTION"
        not in (config / "sessions" / original / "transcript.jsonl").read_text()
    )
    copied = read_manifest(config / "sessions" / copy)
    assert copied is not None
    assert copied["predecessor_target_session_id"] == original
    third = await service.import_sessions(_request())
    assert third.sessions[0].outcome == "skipped_already_imported"
    assert third.sessions[0].target_session_id == copy


def _claude_records() -> list[dict[str, Any]]:
    return [
        {
            "type": "custom-title",
            "customTitle": "Synthetic rewind",
            "sessionId": "sess-claude",
        },
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "sessionId": "sess-claude",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "message": {"role": "user", "content": "first question"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "sess-claude",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "message": {
                "role": "assistant",
                "model": "claude-synthetic",
                "content": [{"type": "text", "text": "first answer"}],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "sessionId": "sess-claude",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "message": {"role": "user", "content": "second question"},
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "u2",
            "sessionId": "sess-claude",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "message": {
                "role": "assistant",
                "model": "claude-synthetic",
                "content": [{"type": "text", "text": "second answer"}],
            },
        },
    ]


@pytest.mark.asyncio
async def test_claude_rewind_creates_a_new_target(claude_store):
    home, config = claude_store
    path = home / ".claude" / "projects" / "encoded" / "sess-claude.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in _claude_records()) + "\n")
    service = HarnessImportService(config)
    request = ImportRequest(sources=["claude"], session_ids={"claude": ["sess-claude"]})
    first = await service.import_sessions(request)
    original = first.sessions[0].target_session_id

    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u3",
                    "parentUuid": "u1",
                    "sessionId": "sess-claude",
                    "timestamp": "2026-01-02T00:00:01.000Z",
                    "message": {"role": "user", "content": "REWRITTEN QUESTION"},
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a3",
                    "parentUuid": "u3",
                    "sessionId": "sess-claude",
                    "timestamp": "2026-01-02T00:00:02.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-synthetic",
                        "content": [{"type": "text", "text": "REWRITTEN ANSWER"}],
                    },
                }
            )
            + "\n"
        )
    second = await service.import_sessions(request)
    copy = second.sessions[0].target_session_id
    assert second.sessions[0].outcome == "imported"
    assert copy == f"{original}-2"
    old_raw = (config / "sessions" / original / "transcript.jsonl").read_text()
    new_raw = (config / "sessions" / copy / "transcript.jsonl").read_text()
    assert "second question" in old_raw and "second answer" in old_raw
    assert "REWRITTEN QUESTION" not in old_raw
    assert "REWRITTEN QUESTION" in new_raw
    assert "second question" not in new_raw
    copied = read_manifest(config / "sessions" / copy)
    assert copied is not None
    assert copied["predecessor_target_session_id"] == original
    third = await service.import_sessions(request)
    assert third.sessions[0].outcome == "skipped_already_imported"
    assert third.sessions[0].target_session_id == copy


@pytest.mark.asyncio
async def test_codex_rollback_creates_a_new_target(codex_store):
    home, config = codex_store
    path = home / ".codex" / "sessions" / "2026" / "01" / "01" / "rollout-rollback.jsonl"
    service = HarnessImportService(config)
    request = ImportRequest(sources=["codex"], session_ids={"codex": ["codex-rollback"]})
    first = await service.import_sessions(request)
    original = first.sessions[0].target_session_id

    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-01-02T00:00:09.000Z",
                    "type": "event_msg",
                    "payload": {"type": "thread_rolled_back", "num_turns": 1},
                }
            )
            + "\n"
        )
    second = await service.import_sessions(request)
    copy = second.sessions[0].target_session_id
    assert second.sessions[0].outcome == "imported"
    assert copy == f"{original}-2"
    old_raw = (config / "sessions" / original / "transcript.jsonl").read_text()
    new_raw = (config / "sessions" / copy / "transcript.jsonl").read_text()
    assert "after rollback question" in old_raw
    assert "after rollback question" not in new_raw
    assert "after rollback answer" not in new_raw
    copied = read_manifest(config / "sessions" / copy)
    assert copied is not None
    assert copied["predecessor_target_session_id"] == original
    entries = Transcript(config / "sessions" / copy).entries()
    message_rows = [
        entry
        for entry in entries
        if entry.type == "message" and entry.payload.get("role") in ("user", "assistant")
    ]
    assert copied["counts"]["messages"] == len(message_rows)
    third = await service.import_sessions(request)
    assert third.sessions[0].outcome == "skipped_already_imported"
    assert third.sessions[0].target_session_id == copy


@pytest.mark.asyncio
async def test_resume_does_not_name_itself_as_predecessor(omp_store):
    home, config = omp_store
    path = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    service = HarnessImportService(config)
    first = await service.import_sessions(_request())
    target = first.sessions[0].target_session_id
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "message",
                    "id": "linear-extra",
                    "parentId": "meta1",
                    "timestamp": "2026-01-04T00:00:00.000Z",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "linear extra"}],
                    },
                }
            )
            + "\n"
        )
    second = await service.import_sessions(_request())
    assert second.sessions[0].outcome == "resumed"
    assert second.sessions[0].target_session_id == target
    manifest = read_manifest(config / "sessions" / target)
    assert manifest is not None
    assert manifest["predecessor_target_session_id"] is None
