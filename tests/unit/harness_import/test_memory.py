"""Memory import: scope classification, review artifact, ledger dedupe."""

from __future__ import annotations

import json

from local_operator.harness_import.memory import import_memory
from local_operator.harness_import.types import MemoryRecord
from local_operator.memory import MemoryIdentity, MemoryStore


def _record(source_id: str, body: str, *, hint: str, thread: str | None = None) -> MemoryRecord:
    return MemoryRecord(
        source_id=source_id,
        title=body[:60],
        body=body,
        tags=["synthetic"],
        created_ts=1700000000.0,
        source_time=1700000000.0,
        uri=f"harness-import:claude:{source_id}",
        status="active",
        scope_hint=hint,  # type: ignore[arg-type]
        thread_source_id=thread,
    )


def test_scope_classification_and_review_artifact(isolated):
    home, config = isolated
    records = [
        _record("p1", "I prefer synthetic jasmine tea.", hint="personal"),
        _record(
            "w1",
            "The synthetic migration is half done.",
            hint="work",
            thread="source-session",
        ),
        _record("w2", "Untethered synthetic work note.", hint="work"),
        _record("i1", "SYNTHETIC INSTRUCTION BODY", hint="instruction"),
    ]
    outcome = import_memory(
        config,
        "claude",
        records,
        target_by_thread={"source-session": "target-session"},
        target_by_cwd={},
        session_dirs={"target-session": config / "sessions" / "target-session"},
        importing_session_id="importing-session",
    )
    assert outcome.imported == 2
    assert outcome.skipped == 1
    assert outcome.instruction_path.endswith("imports/instructions/claude.md")
    assert outcome.review_path.endswith("imports/memory-review-claude.jsonl")

    global_hits = MemoryStore(
        config, MemoryIdentity(session_id="importing-session")
    ).search("synthetic jasmine tea", scope="global")
    assert any("jasmine" in hit["body"] for hit in global_hits)
    session_hits = MemoryStore(
        config,
        MemoryIdentity(session_id="target-session"),
        config / "sessions" / "target-session",
    ).search("synthetic migration", scope="session")
    assert any("migration" in hit["body"] for hit in session_hits)
    assert "SYNTHETIC INSTRUCTION BODY" not in json.dumps(global_hits)
    review = (config / "imports" / "memory-review-claude.jsonl").read_text()
    assert "Untethered synthetic work note" in review
    instructions = (config / "imports" / "instructions" / "claude.md").read_text()
    assert "SYNTHETIC INSTRUCTION BODY" in instructions

    # Ledger dedupe: a second pass imports nothing new.
    again = import_memory(
        config,
        "claude",
        records,
        target_by_thread={"source-session": "target-session"},
        target_by_cwd={},
        session_dirs={"target-session": config / "sessions" / "target-session"},
        importing_session_id="importing-session",
    )
    assert again.imported == 0
    assert again.skipped == 3


def test_work_memory_with_unique_cwd_target(isolated):
    home, config = isolated
    record = _record("w3", "Synthetic cwd-bound work note.", hint="work")
    record.cwd = "/tmp/synthetic-project"
    outcome = import_memory(
        config,
        "omp",
        [record],
        target_by_thread={},
        target_by_cwd={"/tmp/synthetic-project": ["target-cwd"]},
        session_dirs={"target-cwd": config / "sessions" / "target-cwd"},
        importing_session_id="importing-session",
    )
    assert outcome.imported == 1
    hits = MemoryStore(
        config,
        MemoryIdentity(session_id="target-cwd"),
        config / "sessions" / "target-cwd",
    ).search("cwd-bound synthetic", scope="session")
    assert hits


def test_work_memory_with_ambiguous_cwd_goes_to_review(isolated):
    home, config = isolated
    record = _record("w4", "Ambiguous synthetic work note.", hint="work")
    record.cwd = "/tmp/shared"
    outcome = import_memory(
        config,
        "omp",
        [record],
        target_by_thread={},
        target_by_cwd={"/tmp/shared": ["target-a", "target-b"]},
        session_dirs={},
        importing_session_id="importing-session",
    )
    assert outcome.imported == 0
    assert outcome.skipped == 1
    assert "Ambiguous synthetic work note" in (
        config / "imports" / "memory-review-omp.jsonl"
    ).read_text()
