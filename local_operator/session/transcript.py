"""Append-only JSONL transcript store.

The transcript is the session's durable memory: every LLM-visible message,
every compaction marker, and every host custom entry (wake schedules, skill
prompts, …) is one line in ``<dir>/transcript.jsonl``.

Entry shape::

    {"id": ..., "ts": ..., "type": "message" | "compaction" | "custom", "payload": {...}}

Replay semantics (ported from omp ``session-manager.ts``): the latest
compaction entry wins; ``build_llm_history`` returns a marker message for that
summary plus every entry from ``first_kept_entry_id`` onward — never anything
before it. Custom entries are ignored by replay, so bookkeeping like wake
schedules never enters LLM context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.harness.types import AgentMessage, CustomMessage, Message

logger = logging.getLogger(__name__)

TRANSCRIPT_FILENAME = "transcript.jsonl"

ENTRY_MESSAGE = "message"
ENTRY_COMPACTION = "compaction"
ENTRY_CUSTOM = "custom"

CUSTOM_KIND_MESSAGE = "message"
CUSTOM_KIND_CUSTOM = "custom"


@dataclass
class TranscriptEntry:
    """One line of the transcript. ``id`` is stable and referenced by
    compaction entries (``first_kept_entry_id``)."""

    id: str
    ts: float
    type: str  # ENTRY_MESSAGE | ENTRY_COMPACTION | ENTRY_CUSTOM
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"id": self.id, "ts": self.ts, "type": self.type, "payload": self.payload},
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(line: str) -> "TranscriptEntry | None":
        """Parse one line; malformed lines are dropped individually (a corrupt
        row must never destroy the whole replay)."""
        try:
            raw = json.loads(line)
            return TranscriptEntry(
                id=str(raw["id"]),
                ts=float(raw.get("ts", 0.0)),
                type=str(raw["type"]),
                payload=dict(raw.get("payload", {})),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("dropping malformed transcript line: %.120s", line)
            return None


class Transcript:
    """Append-only JSONL store, thread/async-safe via an asyncio lock.

    Message entries reuse the message's own ``id`` as the entry id so
    compaction can reference ``first_kept_entry_id`` without a second mapping.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / TRANSCRIPT_FILENAME
        self._lock = asyncio.Lock()
        self._entries: list[TranscriptEntry] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = TranscriptEntry.from_json(line)
                    if entry is not None:
                        self._entries.append(entry)

    # -- append -------------------------------------------------------------

    async def append_message(self, message: Message | CustomMessage) -> TranscriptEntry:
        """Append one LLM-visible message (or a custom transcript message).

        BOTH kinds persist the message's own ``id`` as the entry id (never
        mint a new one): compaction's ``first_kept_entry_id`` must be able to
        reference a custom entry that renders into LLM context.
        """
        if isinstance(message, Message):
            payload: dict[str, Any] = {"kind": CUSTOM_KIND_MESSAGE, **message.model_dump()}
        else:
            payload = {"kind": CUSTOM_KIND_CUSTOM, **message.model_dump()}
        return await self._append(ENTRY_MESSAGE, payload, message.id)

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        preserve_data: dict[str, Any] | None = None,
    ) -> TranscriptEntry:
        """Record a compaction marker. Replay treats the LATEST one as the
        boundary: summary marker + entries from ``first_kept_entry_id`` on.

        ``preserve_data`` carries strategy-specific replay payloads (e.g.
        ``{"snapcompact": Archive.model_dump()}``) that replay renders back
        into LLM context instead of plain text.
        """
        payload: dict[str, Any] = {
            "summary": summary,
            "first_kept_entry_id": first_kept_entry_id,
            "tokens_before": tokens_before,
        }
        if preserve_data is not None:
            payload["preserve_data"] = preserve_data
        return await self._append(ENTRY_COMPACTION, payload)

    async def append_custom(self, custom_type: str, details: dict[str, Any]) -> TranscriptEntry:
        """Append a host bookkeeping entry (wake schedules, checkpoints, …).
        Custom entries never enter LLM context."""
        return await self._append(ENTRY_CUSTOM, {"custom_type": custom_type, "details": details})

    async def _append(
        self, type: str, payload: dict[str, Any], entry_id: str | None = None
    ) -> TranscriptEntry:
        entry = TranscriptEntry(
            id=entry_id or uuid.uuid4().hex,
            ts=time.time(),
            type=type,
            payload=payload,
        )
        async with self._lock:
            self._entries.append(entry)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry.to_json() + "\n")
                handle.flush()
        return entry

    # -- replay -------------------------------------------------------------

    def entries(self) -> list[TranscriptEntry]:
        """All entries in append order (in-memory snapshot)."""
        return list(self._entries)

    def latest_custom(self, custom_type: str) -> dict[str, Any] | None:
        """Details of the newest custom entry of ``custom_type`` (backward
        scan, first hit wins — each change appends a full snapshot)."""
        for entry in reversed(self._entries):
            if entry.type == ENTRY_CUSTOM and entry.payload.get("custom_type") == custom_type:
                return dict(entry.payload.get("details", {}))
        return None

    def build_llm_history(self) -> list[AgentMessage]:
        """Replay the transcript into LLM-visible messages.

        Latest compaction entry wins: a marker for its summary followed by the
        entries from ``first_kept_entry_id`` onward. Without compaction, every
        message entry replays. Custom entries are ignored.
        """
        entries = self._entries
        compaction_index: int | None = None
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].type == ENTRY_COMPACTION:
                compaction_index = i
                break

        start = 0
        prefix: list[AgentMessage] = []
        if compaction_index is not None:
            compaction = entries[compaction_index]
            details: dict[str, Any] = {"summary": compaction.payload.get("summary", "")}
            preserve_data = compaction.payload.get("preserve_data")
            if preserve_data is not None:
                details["preserve_data"] = preserve_data
            prefix.append(
                CustomMessage(
                    custom_type="compaction_summary",
                    attribution="system",
                    details=details,
                )
            )
            first_kept_id = compaction.payload.get("first_kept_entry_id")
            # The first kept entry normally sits BEFORE the compaction marker
            # (messages are persisted as they happen; the marker comes last),
            # so scan the whole transcript.
            start = compaction_index + 1
            for i in range(len(entries)):
                if entries[i].id == first_kept_id:
                    start = i
                    break

        out: list[AgentMessage] = list(prefix)
        for entry in entries[start:]:
            if entry.type != ENTRY_MESSAGE:
                continue
            message = _entry_to_message(entry)
            if message is not None:
                out.append(message)
        return out

    # -- lifecycle ----------------------------------------------------------

    def flush(self) -> None:
        """Writes are flushed per append; provided for dispose() symmetry."""
        return None


def _entry_to_message(entry: TranscriptEntry) -> AgentMessage | None:
    """Rehydrate one message entry; malformed rows are dropped individually."""
    payload = dict(entry.payload)
    kind = payload.pop("kind", CUSTOM_KIND_MESSAGE)
    try:
        if kind == CUSTOM_KIND_CUSTOM:
            return CustomMessage.model_validate(payload)
        payload["id"] = entry.id  # the entry id IS the message id
        return Message.model_validate(payload)
    except Exception:
        logger.warning("dropping unparseable transcript message entry %s", entry.id)
        return None
