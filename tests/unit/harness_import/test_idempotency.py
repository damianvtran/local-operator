"""Idempotency, resume, user-edited targets and manifest capture policy."""

from __future__ import annotations

import hashlib
import json

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
    assert read_manifest(config / "sessions" / row.target_session_id)["status"] == "complete"
