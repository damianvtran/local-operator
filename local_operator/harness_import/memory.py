"""Foreign curated memory → scope-classified ``MemoryStore`` facts.

Memory is not "more transcript". Lop's memory guide forbids bulk-ingesting
transcripts, environment variables and credential paths, so this module writes
ONLY atomic, provenance-tagged facts through ``MemoryStore.put`` — never direct
SQLite — and defaults every risky shape to a review artifact.

Classification is deliberately conservative:

* ``personal`` hint (user-level memory files) is a candidate for the global
  bank, but ``MemoryStore``'s own ``scope_fit`` can still refuse work-state
  phrasing; a refusal goes to the review artifact and is counted, never
  silently promoted.
* ``work`` hint lands in the SESSION bank of exactly one imported conversation
  (thread id, else a unique cwd match). Several cwd matches require explicit
  selection, so the row goes to review instead of guessing.
* instruction files (``CLAUDE.md`` / ``AGENTS.md``) are NEVER auto-promoted:
  they are work-state-shaped and the design keeps them as a review artifact.

Dedupe is service-side: ``MemoryStore.put`` only dedupes active facts and
ignores ``created_at``, so a ledger keyed by (uri, source_time, body hash,
status) is the idempotency boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.compaction.tokens import count_text_tokens
from local_operator.harness_import import store
from local_operator.harness_import.redact import redact_text
from local_operator.harness_import.types import MemoryRecord
from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore

#: Whole serialized fact cap is 8 KiB; leave room for title/tags/source.
FACT_BODY_BUDGET = 6000
MAX_MEMORY_RECORDS = 5000


@dataclass
class MemoryOutcome:
    imported: int = 0
    skipped: int = 0
    review_path: str = ""
    instruction_path: str = ""
    warnings: list[str] = field(default_factory=list)
    projected_tokens: int = 0
    projected_global_tokens: int = 0


def instruction_path(config: Path, source: str) -> Path:
    return store.imports_root(config) / "instructions" / f"{source}.md"


def review_path(config: Path, source: str) -> Path:
    return store.imports_root(config) / f"memory-review-{source}.jsonl"


def ledger_path(config: Path, source: str) -> Path:
    return store.imports_root(config) / f"memory-{source}.json"


def _ledger_key(record: MemoryRecord, status: str, body: str) -> str:
    body = hashlib.sha256(body.encode()).hexdigest()
    return f"{record.uri}\0{record.source_time or 0}\0{body}\0{status}"


def _load_ledger(config: Path, source: str) -> dict[str, Any]:
    try:
        value = json.loads(ledger_path(config, source).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"v": 1, "keys": []}
    if not isinstance(value, dict) or not isinstance(value.get("keys"), list):
        return {"v": 1, "keys": []}
    return value


def _save_ledger(config: Path, source: str, ledger: dict[str, Any]) -> None:
    store.atomic_write_text(
        ledger_path(config, source), json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )


def collect_instructions(
    config: Path, source: str, records: Iterable[MemoryRecord]
) -> str:
    """Write instruction text to a review artifact; never touch live files."""
    parts: list[str] = []
    for record in records:
        text, _ = redact_text(record.body)
        parts.append(f"## {record.uri}\n\n{text.strip()}\n")
    if not parts:
        return ""
    path = instruction_path(config, source)
    store.atomic_write_text(path, "\n".join(parts))
    return str(path)


def _atomic_chunks(record: MemoryRecord) -> list[tuple[str, str]]:
    """Split one source fact on paragraph boundaries under the whole-fact cap."""
    if len(record.body.encode()) <= FACT_BODY_BUDGET:
        return [(record.title, record.body)]
    chunks: list[str] = []
    current = ""
    for paragraph in record.body.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph.strip()
        if len(candidate.encode()) <= FACT_BODY_BUDGET:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(paragraph.encode()) > FACT_BODY_BUDGET:
            chunks.append(paragraph[:FACT_BODY_BUDGET])
            paragraph = paragraph[FACT_BODY_BUDGET:]
        current = paragraph.strip()
    if current:
        chunks.append(current)
    return [
        (f"{record.title} ({index + 1})" if len(chunks) > 1 else record.title, chunk)
        for index, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def _target_for(
    record: MemoryRecord,
    *,
    target_by_thread: dict[str, str],
    target_by_cwd: dict[str, list[str]],
) -> str | None:
    if record.thread_source_id and record.thread_source_id in target_by_thread:
        return target_by_thread[record.thread_source_id]
    if record.cwd:
        matches = target_by_cwd.get(record.cwd, [])
        if len(matches) == 1:
            return matches[0]
    return None


def _fact(
    source: str,
    record: MemoryRecord,
    title: str,
    body: str,
    status: str,
) -> dict[str, Any]:
    clean_title, _ = redact_text(title[:160].strip())
    clean_body, _ = redact_text(body.strip())
    tags = ["imported", f"source:{source}", *record.tags]
    tags = [tag[:64] for tag in tags if tag][:12]
    return {
        "title": clean_title or clean_body[:160],
        "body": clean_body,
        "tags": tags,
        "status": status,
        "created_at": record.created_ts or 0.0,
        "source_time": record.source_time or record.created_ts or None,
        "source": {
            "kind": record.source_kind,
            "uri": f"harness-import:{source}:{record.source_id}"[:256],
            "author": "harness-import",
        },
    }


def import_memory(
    config: Path,
    source: str,
    records: Iterable[MemoryRecord],
    *,
    target_by_thread: dict[str, str],
    target_by_cwd: dict[str, list[str]],
    session_dirs: dict[str, Path],
    importing_session_id: str,
    include_instructions: bool = True,
    memory_status: str = "active",
) -> MemoryOutcome:
    """Write curated facts, return counts, and never fail the conversation import."""
    outcome = MemoryOutcome()
    rows = list(records)[:MAX_MEMORY_RECORDS]
    instructions = [row for row in rows if row.scope_hint == "instruction"]
    curated = [row for row in rows if row.scope_hint != "instruction"]
    if instructions and include_instructions:
        outcome.instruction_path = collect_instructions(config, source, instructions)
    ledger = _load_ledger(config, source)
    keys = set(str(key) for key in ledger.get("keys", []))
    review_rows: list[dict[str, Any]] = []

    for record in curated:
        outcome.projected_tokens += count_text_tokens(record.body) + count_text_tokens(record.title)
        for title, body in _atomic_chunks(record):
            status = "active"
            fact = _fact(source, record, title, body, status)
            if record.scope_hint == "personal":
                scope = "global"
                owner = "global"
                outcome.projected_global_tokens += count_text_tokens(body)
            else:
                target = _target_for(
                    record, target_by_thread=target_by_thread, target_by_cwd=target_by_cwd
                )
                if target is None:
                    review_rows.append(
                        {
                            "title": fact["title"],
                            "body": fact["body"],
                            "uri": record.uri,
                            "reason": "work_state_without_target_session",
                        }
                    )
                    outcome.skipped += 1
                    continue
                scope = "session"
                owner = target
                outcome.projected_global_tokens += 0
            key = _ledger_key(record, status, body)
            if key in keys:
                outcome.skipped += 1
                continue
            try:
                if scope == "global":
                    store_obj = MemoryStore(
                        config, MemoryIdentity(session_id=importing_session_id)
                    )
                else:
                    store_obj = MemoryStore(
                        config,
                        MemoryIdentity(session_id=owner),
                        session_dirs.get(owner),
                    )
                store_obj.put(scope, fact)
            except (MemoryError, ValueError) as exc:
                review_rows.append(
                    {
                        "title": fact["title"],
                        "body": fact["body"],
                        "uri": record.uri,
                        "reason": type(exc).__name__,
                    }
                )
                outcome.skipped += 1
                continue
            keys.add(key)
            outcome.imported += 1
            if memory_status == "candidate":
                outcome.warnings.append(
                    "memory_status=candidate is reserved for instruction-derived facts; "
                    "curated rows are active in their bank"
                )
    ledger["keys"] = sorted(keys)
    ledger["fingerprint"] = {"records": len(rows), "imported": outcome.imported}
    _save_ledger(config, source, ledger)
    if instructions and not include_instructions:
        outcome.skipped += len(instructions)
    if review_rows:
        path = review_path(config, source)
        store.atomic_write_text(
            path,
            "\n".join(json.dumps(row, ensure_ascii=False) for row in review_rows) + "\n",
        )
        outcome.review_path = str(path)
    return outcome


def project_memory(records: Iterable[MemoryRecord]) -> tuple[int, int]:
    """``(total_tokens, global_tokens)`` for preview; writes nothing."""
    total = 0
    global_tokens = 0
    for record in list(records)[:MAX_MEMORY_RECORDS]:
        cost = count_text_tokens(record.body) + count_text_tokens(record.title)
        total += cost
        if record.scope_hint == "personal":
            global_tokens += cost
    return total, global_tokens
