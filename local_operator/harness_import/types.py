"""Canonical import model shared by every source adapter and the mapper.

WHY a separate model instead of mapping each harness directly to ``Message``:
the three stores disagree about roles, block shapes, tool pairing, compaction
and branch semantics. Normalising ONCE here keeps ``mapping.py`` the single
place that knows lop's target row format, so a future adapter (or a fix to one
source) cannot fork the target contract.

These are plain dataclasses, not pydantic: they are internal, never serialised
to a provider, and a large source session holds hundreds of thousands of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SourceKind = Literal["claude", "codex", "omp"]

#: Stable order used by listings, previews and result summaries.
SOURCES: tuple[SourceKind, ...] = ("claude", "codex", "omp")

Availability = Literal["full", "metadata_only", "unsupported"]
SessionKind = Literal["conversation", "subagent"]
EventKind = Literal[
    "user",
    "assistant",
    "tool_result",
    "compaction",
    "model_change",
    "title",
    "reset",
    "branch_summary",
    "unknown",
]


@dataclass(slots=True)
class SourceInfo:
    """One detected harness. ``detected`` means a known root exists."""

    source: SourceKind
    label: str
    root: Path | None
    detected: bool
    session_count: int = 0
    #: Worst availability across the sessions found (``full`` when any body is
    #: readable). A source with only index metadata is still listed honestly.
    availability: Availability = "full"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "root": str(self.root) if self.root else "",
            "detected": self.detected,
            "session_count": self.session_count,
            "availability": self.availability,
            "detail": self.detail,
        }


@dataclass(slots=True)
class SourceSession:
    """A source conversation, before any body is parsed into events."""

    source: SourceKind
    source_id: str
    path: Path
    alternate_paths: list[Path] = field(default_factory=list)
    fork_parent_source_id: str | None = None
    title: str | None = None
    cwd: str | None = None
    created_ts: float = 0.0
    updated_ts: float = 0.0
    message_count: int | None = None
    compaction_count: int = 0
    parent_source_id: str | None = None
    kind: SessionKind = "conversation"
    availability: Availability = "full"
    warnings: list[str] = field(default_factory=list)
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "path": str(self.path),
            "alternate_paths": [str(p) for p in self.alternate_paths],
            "fork_parent_source_id": self.fork_parent_source_id,
            "title": self.title,
            "cwd": self.cwd,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "message_count": self.message_count,
            "compaction_count": self.compaction_count,
            "parent_source_id": self.parent_source_id,
            "kind": self.kind,
            "availability": self.availability,
            "warnings": list(self.warnings),
            "model": self.model,
        }


@dataclass(slots=True)
class CompactionInfo:
    """Source compaction, mapped to lop's compaction payload by ``mapping``.

    ``first_kept_source_id`` is the SOURCE event id the mapper must resolve to a
    target row id. A source that only knows the summarized span (Claude) leaves
    it ``None`` and gives ``summary_source_id`` so the mapper can anchor at the
    first message after the summary row in emitted order.
    """

    summary: str
    first_kept_source_id: str | None = None
    tokens_before: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    preserve_data: dict[str, Any] = field(default_factory=dict)
    preserved_user_turns: list[dict[str, str]] = field(default_factory=list)
    #: Source id of the summary/boundary row itself; becomes the target
    #: compaction row id and is never used as the kept-window anchor.
    summary_source_id: str | None = None
    #: ``replacement_history`` emission: the first message/tool event after the
    #: marker is synthetic retained history, and only it may anchor the cut.
    anchor_after_marker: bool = False


@dataclass(slots=True)
class SourceEvent:
    """One canonical event streamed out of an adapter."""

    source_event_id: str
    kind: EventKind
    parent_source_event_id: str | None = None
    ts: float = 0.0
    text: list[str] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)  # ImageContent
    thinking: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_result: dict[str, Any] | None = None
    compaction: CompactionInfo | None = None
    model: str | None = None
    raw_flags: dict[str, Any] = field(default_factory=dict)
    #: Codex replacement history rows are emitted AFTER the compaction marker
    #: but must never be anchor candidates before it.
    synthetic: str | None = None


@dataclass(slots=True)
class MemoryRecord:
    """One curated foreign memory fact, before scope classification."""

    source_id: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    created_ts: float = 0.0
    source_time: float | None = None
    uri: str = ""
    status: Literal["candidate", "active"] = "active"
    #: Adapter hint consumed by ``harness_import.memory``:
    #: ``personal`` = durable user-level fact, ``work`` = session/work state,
    #: ``instruction`` = CLAUDE.md/AGENTS.md-like text.
    scope_hint: Literal["personal", "work", "instruction"] = "work"
    cwd: str | None = None
    source_kind: str = "reported"
    #: Source session id this fact was mined from, when it is thread-bound.
    thread_source_id: str | None = None


@dataclass(slots=True)
class ImportRequest:
    """What to import. There is deliberately no arbitrary filesystem path."""

    sources: list[str] | None = None
    #: ``{source: [source_session_id, ...]}``; empty/None means every session.
    session_ids: dict[str, list[str]] = field(default_factory=dict)
    all: bool = False
    include_subagents: bool = False
    include_cleared_segments: bool = False
    include_memory: bool = True
    include_instructions: bool = True
    #: ``candidate`` only for explicit instruction-derived session facts.
    memory_status: Literal["active", "candidate"] = "active"
    max_sessions: int | None = None


@dataclass(slots=True)
class ImportWarning:
    source: str
    session_id: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "session_id": self.session_id,
            "code": self.code,
            "message": self.message,
        }


@dataclass(slots=True)
class ImportPreview:
    """Counts and projected targets; constructing one writes nothing."""

    sessions: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    total_sessions: int = 0
    total_messages: int = 0
    total_compactions: int = 0
    total_memory_records: int = 0
    projected_memory_tokens: int = 0
    projected_global_tokens: int = 0
    warnings: list[ImportWarning] = field(default_factory=list)
    instruction_review_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "sources": self.sources,
            "total_sessions": self.total_sessions,
            "total_messages": self.total_messages,
            "total_compactions": self.total_compactions,
            "total_memory_records": self.total_memory_records,
            "projected_memory_tokens": self.projected_memory_tokens,
            "projected_global_tokens": self.projected_global_tokens,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "instruction_review_paths": self.instruction_review_paths,
        }


@dataclass(slots=True)
class ImportProgress:
    """Bounded in-memory job state; polled by the desktop route."""

    job_id: str
    state: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
    phase: str = "queued"
    source: str = ""
    session_id: str = ""
    completed: int = 0
    total: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    error: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "phase": self.phase,
            "source": self.source,
            "session_id": self.session_id,
            "completed": self.completed,
            "total": self.total,
            "results": self.results,
            "warnings": self.warnings,
            "error": self.error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class SessionResult:
    source: str
    source_id: str
    target_session_id: str = ""
    outcome: Literal[
        "imported", "resumed", "skipped_already_imported", "metadata_only", "failed"
    ] = "imported"
    messages: int = 0
    tool_calls: int = 0
    tool_results: int = 0
    compactions: int = 0
    redactions: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "target_session_id": self.target_session_id,
            "outcome": self.outcome,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "compactions": self.compactions,
            "redactions": self.redactions,
            "skipped": self.skipped,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ImportResult:
    job_id: str
    state: Literal["done", "cancelled", "failed"] = "done"
    sessions: list[SessionResult] = field(default_factory=list)
    messages: int = 0
    compactions: int = 0
    memory_imported: int = 0
    memory_skipped: int = 0
    memory_review_path: str = ""
    instruction_review_path: str = ""
    redactions: int = 0
    warnings: list[ImportWarning] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "sessions": [row.to_dict() for row in self.sessions],
            "messages": self.messages,
            "compactions": self.compactions,
            "memory_imported": self.memory_imported,
            "memory_skipped": self.memory_skipped,
            "memory_review_path": self.memory_review_path,
            "instruction_review_path": self.instruction_review_path,
            "redactions": self.redactions,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "error": self.error,
        }
