"""Append-only JSONL transcript store.

The transcript is the session's durable memory: every LLM-visible message,
every compaction marker, and every host custom entry (wake schedules, skill
prompts, …) is one line in ``<dir>/transcript.jsonl``.

Entry shape::

    {"id": ..., "ts": ..., "type": "message" | "compaction" | "custom" | "prune",
     "payload": {...}}

Replay semantics: the latest
compaction entry wins; ``build_llm_history`` returns a marker message for that
summary plus every entry from ``first_kept_entry_id`` onward — never anything
before it. Custom entries are ignored by replay, so bookkeeping like wake
schedules never enters LLM context.

Footprint. Two things keep the file from growing the way an unbounded harness
transcript does (the failure mode that fills a volume: 200 MB single files).

- **Slim rows.** Message payloads are written with ``exclude_defaults``, the
  entry id is not repeated inside the payload, and a ``raw_arguments`` string
  that round-trips to the structured ``arguments`` is dropped. Together those
  are ~26% of a real transcript, measured on a 66-entry deepseek run. Reading
  is unchanged: every omitted field is a pydantic default, so rows written by
  older builds still load.
- **Prune entries.** Compaction blanks superseded and useless tool outputs in
  the LIVE context; without a record of that the transcript keeps the original
  multi-kilobyte text and a resumed session pays for output the session it
  resumes had already thrown away. ``append_prune`` journals the blanking,
  replay applies it, and :meth:`Transcript.compact_file` folds the journal
  into the message rows so the dead bytes leave the disk too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    TextContent,
)

logger = logging.getLogger(__name__)

TRANSCRIPT_FILENAME = "transcript.jsonl"

ENTRY_MESSAGE = "message"
ENTRY_COMPACTION = "compaction"
ENTRY_CUSTOM = "custom"

CUSTOM_KIND_MESSAGE = "message"
CUSTOM_KIND_CUSTOM = "custom"

#: Journal entry recording that compaction blanked a tool result in the live
#: context. Replay applies it; :meth:`Transcript.compact_file` folds it away.
ENTRY_PRUNE = "prune"

#: Rewrite the file only once this many bytes are provably reclaimable. A
#: prune pass runs on most turns, and rewriting a multi-megabyte transcript
#: every turn would cost far more I/O than the blanking saves. 256 KiB makes
#: the rewrite amortized-free while still bounding a long session's file.
COMPACT_FILE_THRESHOLD_BYTES = 256 * 1024


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


def encode_message_payload(message: Message | CustomMessage) -> dict[str, Any]:
    """Serialize a message for one transcript row, omitting what a reader can
    reconstruct.

    Three redundancies, measured at 26% of a real 66-entry transcript:

    - ``exclude_defaults`` drops the ``"is_error":false``/``"usage":null``/
      ``"tool_calls":[]`` filler that ``model_dump`` writes on every row.
      Reading is unaffected because every omitted value IS the pydantic
      default, so rows from older builds and rows from this one both load.
    - The message ``id`` is already the entry id; repeating it inside the
      payload buys nothing and ``_entry_to_message`` overwrites it from the
      entry anyway.
    - ``raw_arguments`` is the provider's verbatim JSON argument string. When
      it round-trips to the same object as ``arguments`` it is a second,
      *escape-inflated* copy of data already on the row — on the measured
      transcript, 22.9 KB of escaped duplicate against 22.0 KB of structure.
      Dropped in that case only: wire clients fall back to
      ``json.dumps(arguments)``, which is the same call, and a string that
      does NOT round-trip (a model emitting non-canonical or repaired JSON) is
      kept verbatim because that is exactly where byte fidelity matters.
    """
    payload = message.model_dump(exclude_defaults=True, exclude={"id"})
    for call in payload.get("tool_calls") or ():
        raw = call.get("raw_arguments")
        if raw is None:
            continue
        try:
            redundant = json.loads(raw) == call.get("arguments", {})
        except (json.JSONDecodeError, TypeError):
            redundant = False
        if redundant:
            call.pop("raw_arguments")
    return payload


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
        kind = CUSTOM_KIND_MESSAGE if isinstance(message, Message) else CUSTOM_KIND_CUSTOM
        payload: dict[str, Any] = {"kind": kind, **encode_message_payload(message)}
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

    async def append_prune(self, target_entry_id: str, notice: str) -> TranscriptEntry:
        """Journal that compaction blanked the tool result at ``target_entry_id``.

        Without this the transcript and the live context disagree the moment
        pruning fires: the session drops a 12 KB tool output down to a short
        notice, then a resume replays the 12 KB back in and the resumed turn
        costs more tokens than the turn it resumed from. Journalling instead
        of rewriting the row keeps the store append-only on the hot path;
        :meth:`compact_file` folds the journal in later, off the hot path.
        """
        return await self._append(ENTRY_PRUNE, {"target": target_entry_id, "notice": notice})

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

    def usages_since_compaction(self) -> list[dict[str, Any]]:
        """Every message entry's ``usage`` payload recorded AFTER the newest
        compaction, oldest first (all of them when nothing compacted).

        Append order is the only place the "after" in that sentence exists.
        :meth:`build_llm_history` cannot answer it: it puts the compaction
        marker at the HEAD of what it returns and the kept window after it, so
        the replayed list carries pre-pass readings ahead of post-pass ones with
        nothing to tell them apart. The entries know, because they are in the
        order they happened.

        Why the distinction is worth an accessor: a kept message still carries
        the ``usage`` it had before the pass shrank the context, and a host that
        seeds a status readout from one reports a context that no longer exists
        (measured at 900k against a real 1.7k). But a session that compacted and
        then ran ten more turns has a perfectly good newest reading, and
        refusing that would be the opposite error. Entries after the marker are
        exactly the readings that survived the pass.

        PRUNING moves the boundary too, and it is the case that makes this more
        than a compaction concern. Blanking a tool result shrinks the live
        context exactly as a compaction pass does, but leaves no marker: the
        journal entry is folded away by :meth:`compact_file` and the message
        rows survive untouched. Measured against the real pruner: a reading of
        640_000 restored for a true context of 31_715, which a host installs as
        exact and hands to the compaction gate.

        So the boundary is POSITIONAL for both passes — the newest compaction
        marker or the newest pruned row, whichever is later — and that is not a
        stylistic choice. A filter that merely skipped pruned rows would be dead
        code: pruning only ever blanks ``role == "tool"`` messages
        (``compaction.pruning``) and ``usage`` is only ever set on the
        ASSISTANT message of a turn (``harness.loop``), so the two sets are
        disjoint by construction and no usage-carrying row is ever flagged. That
        version passed a test which pruned an assistant message — a state the
        production pruner cannot produce — and fixed nothing. What is wrong with
        a pre-prune reading is not the row it sits on, it is that it describes a
        context measured before the shrink, so position is the only thing that
        can express it.

        Both spellings of "pruned" are consulted for the same reason
        :meth:`compact_file` exists: the journal entry before a fold, the
        ``provider_payload`` flag on the row afterwards.

        Returns the raw payload dicts rather than ``Usage`` objects: this module
        is the persistence layer and does not own the harness's models, and the
        caller is already parsing them.
        """
        start = 0
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.type in (ENTRY_COMPACTION, ENTRY_PRUNE):
                # A journal entry sits at the moment the shrink HAPPENED, which
                # is what the boundary must be drawn on — not at the row it
                # targets. The targeted tool result may be hundreds of entries
                # older, and the readings in between were all measured before
                # the blanking and so describe the pre-shrink context. Seen in
                # the wild: one real transcript on this machine has its newest
                # prune at entry 88 with its newest usage at 82, and taking the
                # target's position instead restored two stale readings.
                start = index + 1
                break
            if entry.type == ENTRY_MESSAGE and (entry.payload.get("provider_payload") or {}).get(
                "pruned"
            ):
                # The FOLDED form of the same thing: `compact_file` materializes
                # the journal into the row and drops the entry, so after a fold
                # the flag on the row is the only evidence left. The row is a
                # tool result and carries no usage itself, so the boundary is
                # after it.
                start = index + 1
                break
        return [
            dict(entry.payload["usage"])
            for entry in self._entries[start:]
            if entry.type == ENTRY_MESSAGE and isinstance(entry.payload.get("usage"), dict)
        ]

    def pending_prunes(self) -> dict[str, str]:
        """``{target entry id: notice}`` for every un-folded prune entry.

        Later entries win, so a target blanked twice keeps the newest notice.
        """
        return {
            str(entry.payload.get("target")): str(entry.payload.get("notice", ""))
            for entry in self._entries
            if entry.type == ENTRY_PRUNE and entry.payload.get("target")
        }

    def build_llm_history(self) -> list[AgentMessage]:
        """Replay the transcript into LLM-visible messages.

        Latest compaction entry wins: a marker for its summary followed by the
        entries from ``first_kept_entry_id`` onward. Without compaction, every
        message entry replays. Custom entries are ignored.

        Prune entries are applied last, so a replayed history matches the
        pruned live context byte for byte rather than resurrecting the tool
        output compaction already decided was dead weight.
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
            # so scan the whole transcript. If the id no longer resolves (a
            # dropped malformed line, a converter-minted id), replaying from
            # compaction_index + 1 would point PAST the kept window and lose
            # every message compaction promised to preserve. Replaying too
            # much is recoverable at the next compaction; silent amnesia is
            # not, so fall back to the full history with an error.
            start = 0
            if first_kept_id is None:
                start = compaction_index + 1
            else:
                for i in range(len(entries)):
                    if entries[i].id == first_kept_id:
                        start = i
                        break
                else:
                    logger.error(
                        "first_kept_entry_id %s not found in transcript; " "replaying full history",
                        first_kept_id,
                    )

        prunes = self.pending_prunes()
        out: list[AgentMessage] = list(prefix)
        for entry in entries[start:]:
            if entry.type != ENTRY_MESSAGE:
                continue
            message = _entry_to_message(entry)
            if message is None:
                continue
            notice = prunes.get(entry.id)
            if notice is not None and isinstance(message, Message):
                _apply_prune(message, notice)
            out.append(message)
        return out

    # -- lifecycle ----------------------------------------------------------

    def reclaimable_bytes(self) -> int:
        """Bytes :meth:`compact_file` would free right now.

        The dead weight is the difference between each pruned row as written
        and the one-line notice that replaces it, plus the journal entries
        themselves.
        """
        prunes = self.pending_prunes()
        if not prunes:
            return 0
        total = 0
        for entry in self._entries:
            if entry.type == ENTRY_PRUNE:
                total += len(entry.to_json()) + 1
            elif entry.type == ENTRY_MESSAGE and entry.id in prunes:
                total += len(entry.to_json()) - len(
                    _pruned_entry(entry, prunes[entry.id]).to_json()
                )
        return max(total, 0)

    async def compact_file(self, min_reclaim_bytes: int = COMPACT_FILE_THRESHOLD_BYTES) -> int:
        """Fold the prune journal into the message rows and rewrite the file.

        This is the half of the footprint story that the in-context caps do
        not cover. A tool output the session already blanked still occupies
        its original kilobytes on disk, forever, in a file that only ever
        grows — the exact shape that let a sibling harness accumulate 233 MB
        single-file transcripts and fill a volume. Folding is semantically
        invisible: entry ids, order and types are unchanged, the journal
        entries disappear because their effect is now materialized, and
        :meth:`build_llm_history` produces the same messages either way.

        Returns the bytes reclaimed (0 when below ``min_reclaim_bytes``, so
        the caller can invoke it every turn without rewriting a large file
        for a few hundred bytes). Crash-safe: the new file is written beside
        the old one and moved over it with an atomic ``os.replace``, so an
        interrupted compaction leaves the original transcript intact.
        """
        async with self._lock:
            prunes = self.pending_prunes()
            if not prunes:
                return 0
            before = self.path.stat().st_size if self.path.exists() else 0
            folded: list[TranscriptEntry] = []
            for entry in self._entries:
                if entry.type == ENTRY_PRUNE:
                    continue
                if entry.type == ENTRY_MESSAGE and entry.id in prunes:
                    entry = _pruned_entry(entry, prunes[entry.id])
                folded.append(entry)
            payload = "".join(entry.to_json() + "\n" for entry in folded)
            reclaimed = before - len(payload.encode("utf-8"))
            if reclaimed < min_reclaim_bytes:
                return 0
            tmp = self.path.with_suffix(self.path.suffix + ".compact")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self.path)
            self._entries = folded
            return reclaimed

    def flush(self) -> None:
        """Writes are flushed per append; provided for dispose() symmetry."""
        return None


def _pruned_entry(entry: TranscriptEntry, notice: str) -> TranscriptEntry:
    """``entry`` with its content folded down to ``notice``.

    A fresh entry rather than a mutation: ``reclaimable_bytes`` measures the
    original against the folded form and must not destroy the original to do
    it. ``tool_calls`` are left alone — pruning only ever blanks tool RESULTS,
    and dropping an assistant message's calls would break call/result pairing
    for every provider.
    """
    payload = dict(entry.payload)
    payload["content"] = [TextContent(text=notice).model_dump()]
    provider_payload = dict(payload.get("provider_payload") or {})
    provider_payload["pruned"] = True
    payload["provider_payload"] = provider_payload
    return TranscriptEntry(id=entry.id, ts=entry.ts, type=entry.type, payload=payload)


def _entry_to_message(entry: TranscriptEntry) -> AgentMessage | None:
    """Rehydrate one message entry; malformed rows are dropped individually."""
    payload = dict(entry.payload)
    kind = payload.pop("kind", CUSTOM_KIND_MESSAGE)
    # The entry id IS the message id for BOTH kinds — the writer passes
    # ``message.id`` as the entry id and the encoder omits it from the
    # payload, so a custom entry that read its id from the payload would come
    # back with a converter-minted uuid and break the ``first_kept_entry_id``
    # reference the transcript exists to keep stable.
    payload["id"] = entry.id
    try:
        if kind == CUSTOM_KIND_CUSTOM:
            return CustomMessage.model_validate(payload)
        return Message.model_validate(payload)
    except Exception:
        logger.warning("dropping unparseable transcript message entry %s", entry.id)
        return None


def _apply_prune(message: Message, notice: str) -> None:
    """Blank ``message`` the way ``compaction.pruning._blank`` blanked it live.

    Deliberately mirrors that function's shape (notice as the whole content,
    ``provider_payload["pruned"] = True`` layered over the existing payload)
    so a replayed message is indistinguishable from one the running session
    pruned — including for the pruning pass itself, which skips anything
    already flagged and would otherwise re-blank it every turn.
    """
    message.content = [TextContent(text=notice)]
    message.provider_payload = {**(message.provider_payload or {}), "pruned": True}
