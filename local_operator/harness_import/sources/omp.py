"""omp (oh-my-pi) adapter.

omp's JSONL is authoritative: ``session-index.db`` is stale and ``history.db``
only fills a few titles. Every file starts with a ``session`` header (a nested
subagent transcript carries ``parentSession``); messages, custom entries and
compaction rows hang off a ``parentId`` tree. We follow the ACTIVE branch — the
newest message leaf walked back to the root — and record inactive branches as
metadata, never as fake conversation.

Discovery uses omp's own resolver: ``PI_CODING_AGENT_DIR``, else
``PI_CONFIG_DIR`` (default ``~/.omp``) plus ``agent``, with
``OMP_PROFILE``/``PI_PROFILE`` selecting ``profiles/<name>/agent``. There is no
``OMP_HOME``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from local_operator.harness.types import ImageContent
from local_operator.harness_import import store
from local_operator.harness_import.sources.base import (
    SourceAdapter,
    json_list,
    json_object,
    json_text,
    parse_ts,
)
from local_operator.harness_import.types import (
    Availability,
    CompactionInfo,
    MemoryRecord,
    SourceEvent,
    SourceInfo,
    SourceSession,
)

_FILES_SKIPPED = {
    "session-index.db",
    "history.db",
}
_CUSTOM_SKIP = {"tool_execution_start", "tool_execution_update"}


def resolve_agent_dir() -> Path | None:
    explicit = os.environ.get("PI_CODING_AGENT_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    config_root = Path(os.environ.get("PI_CONFIG_DIR", "").strip() or "~/.omp").expanduser()
    profile = os.environ.get("OMP_PROFILE", "").strip() or os.environ.get("PI_PROFILE", "").strip()
    if profile:
        return config_root / "profiles" / profile / "agent"
    return config_root / "agent"


def _session_files(agent_dir: Path) -> Iterator[Path]:
    sessions = agent_dir / "sessions"
    if not sessions.is_dir():
        return
    for path in sorted(sessions.rglob("*.jsonl")):
        if path.name in _FILES_SKIPPED or path.name.startswith("."):
            continue
        if store.is_confined(sessions, path):
            yield path


def _read_header(path: Path) -> dict[str, Any] | None:
    for _, record in store.iter_jsonl(path):
        if record.get("type") == "session":
            return record
        # Metadata entries can precede the header in damaged files; keep looking
        # but stop after a bounded number of non-header rows.
        if _ > 64:
            return None
    return None


def _entry_id(record: dict[str, Any]) -> str:
    return str(record.get("id") or "")


def _metadata(path: Path) -> dict[str, Any]:
    """Light scan for listing: title, model, counts, timestamps, parent."""
    result = {
        "title": None,
        "title_source": None,
        "model": None,
        "messages": 0,
        "compactions": 0,
        "created": 0.0,
        "updated": 0.0,
        "parent": None,
        "header": {},
    }
    for ordinal, record in store.iter_jsonl(path):
        kind = record.get("type")
        if kind == "session" and not result["header"]:
            result["header"] = record
            result["created"] = parse_ts(record.get("timestamp"))
            result["parent"] = record.get("parentSession")
            result["updated"] = max(result["updated"], parse_ts(record.get("timestamp")))
            continue
        if kind in ("title", "title_change"):
            value = record.get("title")
            if isinstance(value, str) and value.strip():
                result["title"] = value.strip()
                result["title_source"] = record.get("source")
                result["updated"] = max(result["updated"], parse_ts(record.get("timestamp")))
            continue
        if kind == "model_change" and isinstance(record.get("model"), str):
            result["model"] = record["model"]
        if kind == "message":
            message = json_object(record.get("message"))
            role = message.get("role")
            if role in ("user", "assistant"):
                result["messages"] += 1
            if role == "toolResult":
                result["messages"] += 1
        elif kind == "custom_message":
            if record.get("attribution") == "user":
                result["messages"] += 1
        elif kind == "compaction":
            result["compactions"] += 1
        result["updated"] = max(result["updated"], parse_ts(record.get("timestamp")))
        if ordinal > 200_000:
            break
    return result


def _worst_availability(rows: list[SourceSession]) -> Availability:
    if any(row.availability == "full" for row in rows):
        return "full"
    if any(row.availability == "metadata_only" for row in rows):
        return "metadata_only"
    return "unsupported"


def encode_cwd(cwd: str) -> str:
    """omp's ``--private-tmp--`` form for an absolute cwd.

    Used for MATCHING rather than decoding: the encoding maps both ``/`` and a
    literal ``-`` to ``-``, so a decoded path can never be trusted to equal the
    original (``/tmp/synthetic-omp`` would decode as ``/tmp/synthetic/omp``).
    Encoding the selected sessions' real cwds avoids guessing.
    """
    return "--" + cwd.strip("/").replace("/", "-") + "--"


class OmpAdapter(SourceAdapter):
    source = "omp"
    label = "omp (oh-my-pi)"

    def detect(self) -> SourceInfo:
        root = resolve_agent_dir()
        detected = bool(root and (root / "sessions").is_dir())
        rows: list[SourceSession] = []
        if detected:
            try:
                rows = self.list_sessions()
            except OSError:
                rows = []
        return SourceInfo(
            source="omp",
            label=self.label,
            root=root,
            detected=detected,
            session_count=len(rows),
            availability=_worst_availability(rows),
            detail="" if detected else "no omp agent directory found",
        )

    def list_sessions(self, *, include_subagents: bool = False) -> list[SourceSession]:
        root = resolve_agent_dir()
        if root is None:
            return []
        rows: list[SourceSession] = []
        for path in _session_files(root):
            try:
                header = _read_header(path)
                if header is None:
                    continue
                meta = _metadata(path)
            except (OSError, ValueError) as exc:
                rows.append(
                    SourceSession(
                        source="omp",
                        source_id=f"unreadable:{path.name}",
                        path=path,
                        availability="unsupported",
                        warnings=[f"source unreadable: {type(exc).__name__}"],
                    )
                )
                continue
            parent = meta.get("parent")
            nested = bool(parent) or _is_nested(path, root)
            if nested and not include_subagents:
                continue
            source_id = str(header.get("id") or path.stem)
            rows.append(
                SourceSession(
                    source="omp",
                    source_id=source_id,
                    path=path,
                    title=meta.get("title"),
                    cwd=header.get("cwd") if isinstance(header.get("cwd"), str) else None,
                    created_ts=meta.get("created") or float(meta.get("updated") or 0.0),
                    updated_ts=meta.get("updated") or 0.0,
                    message_count=meta.get("messages"),
                    compaction_count=meta.get("compactions") or 0,
                    parent_source_id=str(parent) if parent else None,
                    kind="subagent" if nested else "conversation",
                    availability="full",
                    model=meta.get("model"),
                )
            )
        rows.sort(key=lambda row: row.updated_ts, reverse=True)
        return rows

    def preview_counts(self, session: SourceSession) -> tuple[int, int]:
        """Listing metadata only; preview must not build the mapped graph."""
        return int(session.message_count or 0), int(session.compaction_count or 0)

    def read_session(self, session: SourceSession) -> Iterator[SourceEvent]:
        stats: dict[str, int] = {}
        records: list[dict[str, Any]] = []
        for _, record in store.iter_jsonl(session.path, stats=stats):
            if record.get("type") == "session":
                continue
            records.append(record)
            if len(records) > 200_000:
                break
        if stats.get("skipped_lines"):
            yield SourceEvent(
                source_event_id="source:skipped-lines",
                kind="unknown",
                ts=session.updated_ts,
                raw_flags={
                    "warning": (
                        f"{stats['skipped_lines']} malformed or oversized source " "line(s) skipped"
                    )
                },
            )
        if not records:
            return

        # Hard reset: only the segment after the newest reset_boundary is live.
        reset_index = -1
        for index, record in enumerate(records):
            if record.get("type") == "reset_boundary":
                reset_index = index
        # Build the parent tree for the active segment.
        segment = records[reset_index + 1 :] if reset_index >= 0 else records
        parent_of = {
            _entry_id(record): record.get("parentId") for record in segment if _entry_id(record)
        }
        has_child = {str(parent) for parent in parent_of.values() if parent}
        # The newest leaf can be a metadata ``custom`` row (a todo nudge, a
        # session-exit marker). The ACTIVE BRANCH is still the chain through it:
        # choosing the newest MESSAGE leaf instead would stop at the last
        # message and silently drop every message whose only child is metadata.
        leaves = [
            record for record in segment if _entry_id(record) and _entry_id(record) not in has_child
        ]
        if not leaves:
            leaves = [record for record in segment if _entry_id(record)]
        if not leaves:
            return
        leaf = max(leaves, key=lambda record: parse_ts(record.get("timestamp")))
        path_ids: list[str] = []
        cursor = _entry_id(leaf)
        seen: set[str] = set()
        while cursor and cursor not in seen:
            seen.add(cursor)
            path_ids.append(cursor)
            cursor = str(parent_of.get(cursor) or "")
        path_ids.reverse()
        path_set = set(path_ids)
        active = [
            record
            for record in segment
            if _entry_id(record) in path_set or record.get("type") in ("title", "title_change")
        ]
        order = {entry_id: index for index, entry_id in enumerate(path_ids)}
        active.sort(key=lambda record: order.get(_entry_id(record), -1))

        previous_model: str | None = None
        for record in active:
            events = list(self._events(record))
            for event in events:
                if event.kind == "model_change" and event.model:
                    if event.model == previous_model:
                        continue
                    previous_model = event.model
                yield event
            if record.get("type") == "compaction" and record.get("id"):
                # A compaction row is emitted before the messages it retained,
                # in journal order; the mapper anchors on the source id.
                continue

    def _events(self, record: dict[str, Any]) -> Iterator[SourceEvent]:
        kind = record.get("type")
        ts = parse_ts(record.get("timestamp"))
        rid = _entry_id(record)
        if kind == "title" or kind == "title_change":
            title = record.get("title")
            if isinstance(title, str) and title.strip():
                yield SourceEvent(
                    source_event_id=rid or f"title:{ts}",
                    parent_source_event_id=record.get("parentId"),
                    ts=ts,
                    kind="title",
                    text=[title.strip()],
                    raw_flags={"user_set": record.get("source") == "user"},
                )
            return
        if kind == "model_change":
            model = record.get("model")
            if isinstance(model, str) and model:
                yield SourceEvent(
                    source_event_id=rid or f"model:{ts}",
                    parent_source_event_id=record.get("parentId"),
                    ts=ts,
                    kind="model_change",
                    model=model,
                )
            return
        if kind == "compaction":
            summary = record.get("summary") or record.get("shortSummary") or ""
            details = json_object(record.get("details"))
            first_kept = record.get("firstKeptEntryId")
            yield SourceEvent(
                source_event_id=rid or f"compaction:{ts}",
                parent_source_event_id=record.get("parentId"),
                ts=ts,
                kind="compaction",
                text=[],
                compaction=CompactionInfo(
                    summary=json_text(summary),
                    first_kept_source_id=str(first_kept) if first_kept else None,
                    tokens_before=int(record.get("tokensBefore") or 0),
                    details=details,
                    preserve_data=(
                        {"short_summary": record.get("shortSummary")}
                        if record.get("shortSummary")
                        else {}
                    ),
                    summary_source_id=rid or None,
                ),
                raw_flags={
                    "from_extension": bool(record.get("fromExtension")),
                    "warning": record.get("warning"),
                },
            )
            return
        if kind == "custom":
            # Structured metadata (todo nudges, session-exit markers) carries
            # no conversation text but DOES carry the parent link that keeps
            # the active-branch walk intact across it. Emit a silent event so
            # the mapper can see that link without counting it as a skip.
            if rid:
                yield SourceEvent(
                    source_event_id=rid,
                    parent_source_event_id=record.get("parentId"),
                    ts=ts,
                    kind="unknown",
                    raw_flags={"silent": True},
                )
            return
        if kind == "custom_message":
            if record.get("attribution") != "user":
                return
            text = json_text(record.get("content"))
            if text.strip():
                yield SourceEvent(
                    source_event_id=rid or f"custom:{ts}",
                    parent_source_event_id=record.get("parentId"),
                    ts=ts,
                    kind="user",
                    text=["[imported from omp] " + text],
                    raw_flags={"custom_type": record.get("customType"), "synthetic": True},
                )
            return
        if kind != "message":
            return
        message = json_object(record.get("message"))
        role = message.get("role")
        text_blocks: list[str] = []
        images: list[ImageContent] = []
        thinking: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in json_list(message.get("content")):
            if not isinstance(block, dict):
                if isinstance(block, str):
                    text_blocks.append(block)
                continue
            block_type = block.get("type")
            if block_type == "text":
                value = block.get("text")
                if isinstance(value, str) and value:
                    text_blocks.append(value)
            elif block_type == "thinking":
                value = block.get("thinking") or block.get("summary")
                if isinstance(value, str) and value:
                    thinking.append(value)
            elif block_type == "toolCall":
                call_id = block.get("id") or f"{rid}:call:{len(tool_calls)}"
                arguments = block.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {}
                raw = (
                    block.get("partialArgs") if isinstance(block.get("partialArgs"), str) else None
                )
                tool_calls.append(
                    {
                        "id": str(call_id),
                        "name": str(block.get("name") or "unknown"),
                        "arguments": arguments,
                        "raw_arguments": raw,
                    }
                )
            elif block_type == "image":
                data = block.get("data")
                if isinstance(data, str) and data:
                    images.append(
                        ImageContent(
                            data=data,
                            mime_type=str(
                                block.get("mimeType") or block.get("mime_type") or "image/png"
                            ),
                        )
                    )
        if role == "user":
            if text_blocks or images:
                yield SourceEvent(
                    source_event_id=rid or f"user:{ts}",
                    parent_source_event_id=record.get("parentId"),
                    ts=ts,
                    kind="user",
                    text=text_blocks,
                    images=images,
                    thinking=thinking,
                )
        elif role == "assistant":
            if text_blocks or tool_calls or images:
                yield SourceEvent(
                    source_event_id=rid or f"assistant:{ts}",
                    parent_source_event_id=record.get("parentId"),
                    ts=ts,
                    kind="assistant",
                    text=text_blocks,
                    images=images,
                    thinking=thinking,
                    tool_calls=tool_calls,
                )
        elif role == "toolResult":
            content: list[str] = []
            for block in json_list(message.get("content")):
                if isinstance(block, dict) and block.get("type") == "text":
                    content.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    content.append(block)
            yield SourceEvent(
                source_event_id=rid or f"tool:{ts}",
                parent_source_event_id=record.get("parentId"),
                ts=ts,
                kind="tool_result",
                tool_result={
                    "tool_call_id": str(
                        message.get("toolCallId") or message.get("tool_call_id") or ""
                    ),
                    "tool_name": str(message.get("toolName") or message.get("tool_name") or ""),
                    "content": content,
                    "is_error": bool(message.get("isError") or message.get("is_error")),
                },
            )

    def read_memory(self, sessions: Sequence[SourceSession] = ()) -> list[MemoryRecord]:
        """cwd-keyed ``memories/<encoded>/`` files; durable facts only.

        No real omp memory tree exists on this machine, so this path is exercised
        with synthetic fixtures. The scope classification is conservative: a
        file named ``learned.md`` is work state; only ``MEMORY.md`` bullets that
        already look like durable personal facts are candidates for global.
        """
        root = resolve_agent_dir()
        if root is None:
            return []
        memory_root = root / "memories"
        if not memory_root.is_dir():
            return []
        records: list[MemoryRecord] = []
        for path in sorted(memory_root.rglob("*.md")):
            if not store.is_confined(memory_root, path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            encoded = path.parent.name
            matching = [row.cwd for row in sessions if row.cwd and encode_cwd(row.cwd) == encoded]
            cwd = matching[0] if len(matching) == 1 else None
            for index, line in enumerate(text.splitlines()):
                stripped = line.strip()
                if not stripped.startswith(("- ", "* ")):
                    continue
                body = stripped[2:].strip()
                if not body:
                    continue
                records.append(
                    MemoryRecord(
                        source_id=f"{path.name}:{index}",
                        title=body[:150],
                        body=body,
                        tags=["omp", path.stem],
                        created_ts=path.stat().st_mtime,
                        source_time=None,
                        uri=str(path),
                        status="candidate",
                        scope_hint="work" if path.name == "learned.md" else "personal",
                        cwd=cwd,
                    )
                )
        return records


def _is_nested(path: Path, agent_dir: Path) -> bool:
    """A nested transcript lives below the encoded-cwd directory."""
    try:
        relative = path.relative_to(agent_dir / "sessions")
    except ValueError:
        return True
    return len(relative.parts) > 2
