"""Canonical events → lop transcript rows.

ONE mapping module, deliberately. Every source adapter emits ``SourceEvent``;
this file is the only place that knows ``Message``/``CustomMessage``/
``TranscriptEntry`` shapes. Three rules are load-bearing and each exists
because the obvious implementation is wrong:

* **Source time, never import time.** Rows carry the event's own ``ts``; a row
  without one is filled at previous + 1 ms, never ``time.time()``.
* **Compaction is a boundary, not a deletion.** Pre-cut rows stay on disk as the
  archive of what the summary replaced; ``first_kept_entry_id`` points at the
  first RETAINED message. A missing anchor would make lop replay the whole
  history, so an unresolvable anchor falls back to the compaction row itself
  (summary-only replay) — never a dangling id.
* **A cut never splits a tool pair.** If the anchor lands on a tool result, the
  anchor moves back to the owning assistant row; a multi-call assistant turn is
  never split.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from local_operator.compaction.tokens import count_text_tokens
from local_operator.harness.types import (
    Content,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
)
from local_operator.harness_import.redact import redact_text, redact_value
from local_operator.harness_import.store import stable_entry_id
from local_operator.harness_import.types import CompactionInfo, SourceEvent
from local_operator.session.transcript import (
    CUSTOM_KIND_MESSAGE,
    ENTRY_COMPACTION,
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    TranscriptEntry,
)

#: The shipped write-side preserved-turn cap for a 1M window. 100k is lop's
#: legacy READ fallback and must not be written into a new imported row.
IMPORT_PRESERVED_TURN_CAP = 150_000
#: An inline image whose DECODED payload is larger than this is skipped with a
#: warning rather than externalised; the foreign store keeps the original.
#: Compare decoded bytes, never the base64 text length (~4/3 of the payload).
MAX_IMAGE_BYTES = 16 * 1024 * 1024


@dataclass
class MappedSession:
    rows: list[TranscriptEntry] = field(default_factory=list)
    messages: dict[str, Message] = field(default_factory=dict)
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "messages": 0,
            "tool_calls": 0,
            "tool_results": 0,
            "compactions": 0,
            "skipped": 0,
            "duplicate_results": 0,
            "redactions": 0,
            "thinking": 0,
        }
    )
    warnings: list[str] = field(default_factory=list)
    model_history: list[dict[str, Any]] = field(default_factory=list)
    compaction_anchors: dict[str, str] = field(default_factory=dict)
    active_leaf_source_id: str | None = None
    #: Source event ids from the active leaf back to the root, so a resume can
    #: distinguish a linear append (old leaf is an ancestor) from a branch
    #: switch (it is not) without re-reading or guessing.
    active_branch_source_ids: list[str] = field(default_factory=list)
    title: str | None = None
    title_user_set: bool = False
    first_user_text: str = ""


@dataclass
class _Emission:
    event: SourceEvent
    entry_id: str
    row: TranscriptEntry
    message: Message | None = None


def _clean_text_list(values: list[str], stats: dict[str, int]) -> list[str]:
    out: list[str] = []
    for value in values:
        clean, count = redact_text(value)
        stats["redactions"] += count
        if clean:
            out.append(clean)
    return out


def _clean_images(images: list[ImageContent], warnings: list[str]) -> list[ImageContent]:
    kept: list[ImageContent] = []
    for image in images:
        if not image.data:
            continue
        try:
            decoded_bytes = len(base64.b64decode(image.data, validate=False))
        except (ValueError, TypeError):
            warnings.append("image skipped: undecodable inline payload")
            continue
        if decoded_bytes > MAX_IMAGE_BYTES:
            warnings.append("image skipped: decoded payload exceeds the import cap")
            continue
        kept.append(image)
    return kept


def build_mapped_session(
    events: list[SourceEvent],
    *,
    source: str,
    source_session_id: str,
    default_created_ts: float = 0.0,
) -> MappedSession:
    """Map one session's canonical events to deterministic transcript rows."""
    mapped = MappedSession()
    stats = mapped.stats
    ordinal = 0
    last_ts = default_created_ts
    seen_tool_results: set[str] = set()

    # Parent links come from EVERY event, including silent metadata rows the
    # mapper does not emit: the branch walk must cross them.
    all_parents = {event.source_event_id: event.parent_source_event_id for event in events}
    emissions: list[_Emission] = []
    for event in events:
        ts = float(event.ts or 0.0)
        if ts <= 0:
            ts = last_ts + 0.001
        last_ts = max(last_ts, ts)
        event.source_event_id = event.source_event_id or f"anon:{ordinal}:{event.kind}:{ts}"
        if event.kind == "unknown":
            if event.raw_flags.get("silent"):
                continue
            stats["skipped"] += 1
            warning = event.raw_flags.get("warning")
            if isinstance(warning, str) and warning.strip():
                mapped.warnings.append(warning)
            continue
        if event.kind == "user":
            text = _clean_text_list(event.text, stats)
            images = _clean_images(event.images, mapped.warnings)
            if not text and not images:
                stats["skipped"] += 1
                continue
            entry_id = stable_entry_id(source, source_session_id, event.source_event_id, ordinal)
            ordinal += 1
            content: list[Content] = [TextContent(text=block) for block in text]
            content.extend(images)
            message = Message(role="user", content=content, id=entry_id)
            mapped.messages[entry_id] = message
            row = TranscriptEntry(entry_id, ts, ENTRY_MESSAGE, {"kind": CUSTOM_KIND_MESSAGE})
            emissions.append(_Emission(event, entry_id, row, message))
            stats["messages"] += 1
            if not mapped.first_user_text and text:
                mapped.first_user_text = text[0]
            continue
        if event.kind == "assistant":
            text = _clean_text_list(event.text, stats)
            stats["thinking"] += len(event.thinking)
            tool_calls: list[ToolCall] = []
            for call in event.tool_calls:
                clean_args, count = redact_value(call.get("arguments") or {})
                stats["redactions"] += count
                raw_arguments = call.get("raw_arguments")
                if isinstance(raw_arguments, str):
                    raw_arguments, count = redact_text(raw_arguments)
                    stats["redactions"] += count
                tool_calls.append(
                    ToolCall(
                        id=str(call.get("id") or ""),
                        name=str(call.get("name") or ""),
                        arguments=clean_args if isinstance(clean_args, dict) else {},
                        raw_arguments=raw_arguments,
                    )
                )
            images = _clean_images(event.images, mapped.warnings)
            if not text and not tool_calls and not images:
                stats["skipped"] += 1
                continue
            entry_id = stable_entry_id(source, source_session_id, event.source_event_id, ordinal)
            ordinal += 1
            assistant_content: list[Content] = [TextContent(text=block) for block in text]
            assistant_content.extend(images)
            message = Message(
                role="assistant",
                content=assistant_content,
                tool_calls=tool_calls,
                id=entry_id,
            )
            mapped.messages[entry_id] = message
            row = TranscriptEntry(entry_id, ts, ENTRY_MESSAGE, {"kind": CUSTOM_KIND_MESSAGE})
            emissions.append(_Emission(event, entry_id, row, message))
            stats["messages"] += 1
            stats["tool_calls"] += len(tool_calls)
            continue
        if event.kind == "tool_result":
            result = event.tool_result or {}
            call_id = str(result.get("tool_call_id") or "")
            if not call_id:
                stats["skipped"] += 1
                mapped.warnings.append("tool result without a call id skipped")
                continue
            if call_id in seen_tool_results:
                stats["skipped"] += 1
                stats["duplicate_results"] += 1
                mapped.warnings.append(
                    f"duplicate tool result for call {call_id} skipped (source anomaly)"
                )
                continue
            seen_tool_results.add(call_id)
            tool_text: list[str] = []
            for block in result.get("content") or []:
                clean, count = redact_text(str(block))
                stats["redactions"] += count
                tool_text.append(clean)
            entry_id = stable_entry_id(source, source_session_id, event.source_event_id, ordinal)
            ordinal += 1
            message = Message(
                role="tool",
                content=[TextContent(text=block) for block in tool_text],
                tool_call_id=call_id,
                tool_name=str(result.get("tool_name") or ""),
                is_error=bool(result.get("is_error")),
                id=entry_id,
            )
            mapped.messages[entry_id] = message
            row = TranscriptEntry(entry_id, ts, ENTRY_MESSAGE, {"kind": CUSTOM_KIND_MESSAGE})
            emissions.append(_Emission(event, entry_id, row, message))
            stats["messages"] += 1
            stats["tool_results"] += 1
            continue
        if event.kind == "compaction":
            info = event.compaction or CompactionInfo(
                summary="", summary_source_id=event.source_event_id
            )
            summary, count = redact_text(info.summary)
            stats["redactions"] += count
            if not summary.strip():
                stats["skipped"] += 1
                mapped.warnings.append("compaction without a summary skipped; messages kept")
                continue
            row_id = stable_entry_id(
                source,
                source_session_id,
                info.summary_source_id or event.source_event_id,
                ordinal,
            )
            ordinal += 1
            row = TranscriptEntry(
                row_id,
                ts,
                ENTRY_COMPACTION,
                {"summary": summary, "first_kept_entry_id": "", "tokens_before": 0},
            )
            emissions.append(_Emission(event, row_id, row, None))
            stats["compactions"] += 1
            continue
        if event.kind in ("model_change", "title", "branch_summary"):
            details: dict[str, Any] = {"source_event_id": event.source_event_id}
            custom_type = {
                "model_change": "import_model_change",
                "title": "import_title",
                "branch_summary": "import_branch_summary",
            }[event.kind]
            if event.model:
                details["model"] = event.model
                mapped.model_history.append({"ts": ts, "model": event.model})
            if event.text:
                text, count = redact_text(event.text[0])
                stats["redactions"] += count
                details["text"] = text
                if event.kind == "title":
                    mapped.title = text
                    mapped.title_user_set = bool(event.raw_flags.get("user_set"))
            details.update(
                {key: value for key, value in event.raw_flags.items() if key != "user_set"}
            )
            entry_id = stable_entry_id(source, source_session_id, event.source_event_id, ordinal)
            ordinal += 1
            row = TranscriptEntry(
                entry_id, ts, ENTRY_CUSTOM, {"custom_type": custom_type, "details": details}
            )
            emissions.append(_Emission(event, entry_id, row, None))
            continue
        stats["skipped"] += 1

    mapped.rows = [emission.row for emission in emissions]
    if mapped.active_leaf_source_id is None:
        mapped.active_leaf_source_id = next(
            (
                emission.event.source_event_id
                for emission in reversed(emissions)
                if emission.event.kind in ("user", "assistant", "tool_result")
            ),
            None,
        )
    parents = all_parents
    cursor = mapped.active_leaf_source_id
    seen: set[str] = set()
    while cursor and cursor not in seen:
        seen.add(cursor)
        mapped.active_branch_source_ids.append(cursor)
        cursor = parents.get(cursor)
    _resolve_compactions(emissions, mapped, source=source, source_session_id=source_session_id)
    return mapped


def _resolve_compactions(
    emissions: list[_Emission],
    mapped: MappedSession,
    *,
    source: str,
    source_session_id: str,
) -> None:
    del source_session_id
    row_index = {row.id: index for index, row in enumerate(mapped.rows)}
    source_to_row = {emission.event.source_event_id: emission.row.id for emission in emissions}
    message_indexes = [
        index
        for index, emission in enumerate(emissions)
        if emission.message is not None and emission.message.role in ("user", "assistant")
    ]
    compaction_emissions = [
        (index, emission)
        for index, emission in enumerate(emissions)
        if emission.event.kind == "compaction"
    ]
    user_emissions = [
        (index, emission)
        for index, emission in enumerate(emissions)
        if emission.event.kind == "user"
    ]
    latest_compaction: tuple[int, _Emission] | None = None

    for emit_index, emission in compaction_emissions:
        info = emission.event.compaction or CompactionInfo(summary="")
        anchor: str | None = None
        if info.first_kept_source_id:
            anchor = source_to_row.get(info.first_kept_source_id)
        if not anchor:
            for index in message_indexes:
                if index > emit_index:
                    anchor = emissions[index].row.id
                    break
        if not anchor:
            anchor = emission.row.id
            mapped.warnings.append("compaction has no retained message; replay is summary-only")
        anchor = _snap_anchor(anchor, mapped, row_index)
        payload = emission.row.payload
        if anchor is not None:
            payload["first_kept_entry_id"] = anchor
        else:
            payload["first_kept_entry_id"] = emission.row.id
            mapped.warnings.append(
                "compaction anchor could not be placed safely; replay is summary-only"
            )
        info.details.setdefault("source", source or "")
        snapshot = {
            **{key: value for key, value in info.details.items() if key != "source"},
            **info.preserve_data,
        }
        if snapshot:
            payload["preserve_data"] = {"source_snapshot": snapshot}
        payload["tokens_before"] = int(
            info.tokens_before or _estimate_before(emissions, emit_index)
        )
        mapped.compaction_anchors[emission.event.source_event_id] = payload["first_kept_entry_id"]
        latest_compaction = (emit_index, emission)

    if latest_compaction is None:
        return
    anchor_id = str(latest_compaction[1].row.payload.get("first_kept_entry_id") or "")
    anchor_index = row_index.get(anchor_id, len(mapped.rows))
    turns: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, emission in user_emissions:
        if index >= anchor_index:
            break
        text = " ".join(
            _redact_again(block) for block in emission.event.text if block.strip()
        ).strip()
        if not text or emission.row.id in seen:
            continue
        seen.add(emission.row.id)
        turns.append({"id": emission.row.id, "text": text})
    kept: list[dict[str, str]] = []
    tokens = 0
    for turn in reversed(turns):
        cost = count_text_tokens(turn["text"]) + 1
        if kept and tokens + cost > IMPORT_PRESERVED_TURN_CAP:
            break
        kept.append(turn)
        tokens += cost
    kept.reverse()
    dropped = len(turns) - len(kept)
    payload = latest_compaction[1].row.payload
    if kept:
        payload["preserved_user_turns"] = kept
        payload["preserved_turns_cap"] = IMPORT_PRESERVED_TURN_CAP
    if dropped:
        payload["preserved_turns_dropped"] = dropped
    payload["preserved_turns_cap"] = IMPORT_PRESERVED_TURN_CAP


def _redact_again(text: str) -> str:
    return redact_text(text)[0]


def _estimate_before(emissions: list[_Emission], index: int) -> int:
    total = 0
    for emission in emissions[:index]:
        for block in emission.event.text:
            total += count_text_tokens(block)
        for call in emission.event.tool_calls:
            total += count_text_tokens(str(call.get("arguments") or ""))
    return total


def _snap_anchor(anchor: str, mapped: MappedSession, row_index: dict[str, int]) -> str | None:
    """Never start a kept window on an orphan tool row or split a call group."""
    index = row_index.get(anchor)
    if index is None:
        return None
    rows = mapped.rows
    row = rows[index]
    if row.type != ENTRY_MESSAGE:
        return anchor
    message = mapped.messages.get(anchor)
    if message is None or message.role != "tool":
        return anchor
    call_id = message.tool_call_id or ""
    for previous in range(index - 1, -1, -1):
        candidate = mapped.messages.get(rows[previous].id)
        if candidate is None:
            continue
        if candidate.role == "assistant" and any(
            call.id == call_id for call in candidate.tool_calls
        ):
            return rows[previous].id
        if candidate.role == "user":
            break
    for following in range(index + 1, len(rows)):
        candidate = mapped.messages.get(rows[following].id)
        if candidate is not None and candidate.role in ("assistant", "user"):
            return rows[following].id
    return None
