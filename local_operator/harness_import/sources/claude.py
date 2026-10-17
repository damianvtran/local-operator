"""Claude Code adapter (projects/ and jobs/ transcripts).

The embedded record ``sessionId`` is authoritative: job copies can contain
records from more than one session, so grouping happens by that field rather
than by filename. A job copy is recorded as an alternate path and its job
``state.json`` supplies the fork parent.

**Compaction boundaries are continuations, not new sessions.** The real job
body has two ``compact_boundary`` rows with ``parentUuid: null``; one global
``parentUuid`` walk loses most of the conversation. We segment at every boundary,
emit every pre-boundary segment as archive rows before its compaction marker,
and walk each segment's own branch tree. The synthetic summary row's uuid is the
compaction ROW id; the kept window starts at the first retained message after it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from local_operator.harness.types import ImageContent
from local_operator.harness_import import store
from local_operator.harness_import.sources.base import (
    SourceAdapter,
    json_object,
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

#: Bounded per-file parse; a larger body is refused rather than silently cut.
MAX_RECORDS = 200_000


def resolve_root() -> Path:
    explicit = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    return Path(explicit).expanduser() if explicit else Path("~/.claude").expanduser()


def _is_boundary(record: dict[str, Any]) -> bool:
    if record.get("type") == "system" and record.get("subtype") == "compact_boundary":
        return True
    if record.get("type") == "summary":
        return True
    return bool(record.get("isCompactSummary"))


def _is_sidechain(record: dict[str, Any]) -> bool:
    return bool(record.get("isSidechain"))


def _is_meta(record: dict[str, Any]) -> bool:
    return bool(record.get("isMeta"))


def _uuid(record: dict[str, Any]) -> str:
    return str(record.get("uuid") or "")


def _blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = json_object(record.get("message"))
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _plain_text(record: dict[str, Any]) -> str:
    message = json_object(record.get("message"))
    content = message.get("content")
    if isinstance(content, str):
        return content
    out: list[str] = []
    for block in _blocks(record):
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            out.append(block["text"])
    return "\n".join(out)


def _files(root: Path) -> list[Path]:
    found: list[Path] = []
    projects = root / "projects"
    if projects.is_dir():
        for path in sorted(projects.rglob("*.jsonl")):
            if store.is_confined(projects, path):
                found.append(path)
    jobs = root / "jobs"
    if jobs.is_dir():
        for path in sorted(jobs.glob("*/tmp/*.jsonl")):
            if store.is_confined(jobs, path):
                found.append(path)
    return found


def _job_state(path: Path, root: Path) -> dict[str, Any]:
    """The job's own state.json, for fork provenance. Never auth/settings."""
    try:
        relative = path.relative_to(root / "jobs")
    except ValueError:
        return {}
    if len(relative.parts) < 2:
        return {}
    state_path = root / "jobs" / relative.parts[0] / "state.json"
    try:
        import json

        value = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _record_cost(record: dict[str, Any]) -> int:
    return 1 if record.get("type") in ("user", "assistant") else 0


def _scan_group(path: Path) -> dict[str, Any]:
    """Light scan of one file: session id, title, times, counts, cwd, model."""
    info: dict[str, Any] = {
        "session_id": None,
        "titles": {},
        "created": 0.0,
        "updated": 0.0,
        "messages": 0,
        "compactions": 0,
        "cwd": None,
        "model": None,
        "legacy_summaries": 0,
    }
    for ordinal, record in store.iter_jsonl(path):
        sid = record.get("sessionId")
        if isinstance(sid, str) and sid and info["session_id"] is None:
            info["session_id"] = sid
        ts = parse_ts(record.get("timestamp"))
        if ts:
            info["created"] = ts if not info["created"] else min(info["created"], ts)
            info["updated"] = max(info["updated"], ts)
        kind = record.get("type")
        if kind == "custom-title" and isinstance(record.get("customTitle"), str):
            info["titles"]["custom"] = record["customTitle"]
        elif kind == "ai-title" and isinstance(record.get("aiTitle"), str):
            info["titles"]["ai"] = record["aiTitle"]
        if kind in ("user", "assistant") and not _is_meta(record) and not _is_sidechain(record):
            info["messages"] += _record_cost(record)
            if isinstance(record.get("cwd"), str) and not info["cwd"]:
                info["cwd"] = record["cwd"]
            message = json_object(record.get("message"))
            if kind == "assistant" and isinstance(message.get("model"), str):
                info["model"] = message["model"]
        if kind == "system" and record.get("subtype") == "compact_boundary":
            info["compactions"] += 1
        elif kind == "summary":
            # A modern compaction has a boundary AND a summary row; counting
            # both would double the listing. Legacy stores have summaries only.
            info["legacy_summaries"] += 1
        if ordinal > MAX_RECORDS:
            break
    if not info["compactions"]:
        info["compactions"] = info["legacy_summaries"]
    return info


def _history_only(root: Path, seen: set[str]) -> list[SourceSession]:
    """Prompt-history rows with no body anywhere: listed, never imported."""
    history = root / "history.jsonl"
    if not history.exists():
        return []
    rows: dict[str, dict[str, Any]] = {}
    for _, record in store.iter_jsonl(history):
        sid = record.get("sessionId")
        display = record.get("display")
        if not isinstance(sid, str) or not sid or sid in seen:
            continue
        entry = rows.setdefault(sid, {"title": None, "ts": 0.0})
        if isinstance(display, str) and display.strip() and entry["title"] is None:
            entry["title"] = display.strip()
        entry["ts"] = max(entry["ts"], parse_ts(record.get("timestamp")))
    return [
        SourceSession(
            source="claude",
            source_id=sid,
            path=history,
            title=entry["title"],
            created_ts=entry["ts"],
            updated_ts=entry["ts"],
            availability="metadata_only",
            warnings=["conversation body unavailable; prompt history only"],
        )
        for sid, entry in rows.items()
    ]


def _worst_availability(rows: list[SourceSession]) -> Availability:
    if any(row.availability == "full" for row in rows):
        return "full"
    if any(row.availability == "metadata_only" for row in rows):
        return "metadata_only"
    return "unsupported"


def _unreadable(path: Path, exc: Exception) -> SourceSession:
    return SourceSession(
        source="claude",
        source_id=f"unreadable:{path.name}",
        path=path,
        availability="unsupported",
        warnings=[f"source unreadable: {type(exc).__name__}"],
    )


class ClaudeAdapter(SourceAdapter):
    source = "claude"
    label = "Claude Code"

    def detect(self) -> SourceInfo:
        root = resolve_root()
        sessions = root / "projects"
        jobs = root / "jobs"
        detected = sessions.is_dir() or jobs.is_dir()
        rows: list[SourceSession] = []
        if detected:
            try:
                rows = self.list_sessions()
            except OSError:
                rows = []
        return SourceInfo(
            source="claude",
            label=self.label,
            root=root,
            detected=detected,
            session_count=len(rows),
            availability=_worst_availability(rows),
            detail="" if detected else f"no Claude store at {root}",
        )

    def list_sessions(self, *, include_subagents: bool = False) -> list[SourceSession]:
        root = resolve_root()
        groups: dict[str, dict[str, Any]] = {}
        rows: list[SourceSession] = []
        for path in _files(root):
            try:
                scan = _scan_group(path)
            except (OSError, ValueError) as exc:
                rows.append(_unreadable(path, exc))
                continue
            sid = scan["session_id"]
            if not sid:
                continue
            group = groups.setdefault(
                sid,
                {
                    "canonical": None,
                    "alternates": [],
                    "scan": scan,
                    "job": {},
                },
            )
            job = _job_state(path, root)
            if path.is_relative_to(root / "projects"):
                group["canonical"] = path
            else:
                group["alternates"].append(path)
            if job:
                group["job"] = job
            current = group["scan"]
            current["created"] = min(
                filter(None, [current["created"], scan["created"]]), default=0.0
            )
            current["updated"] = max(current["updated"], scan["updated"])
            current["messages"] = max(current["messages"], scan["messages"])
            current["compactions"] = max(current["compactions"], scan["compactions"])
            current["titles"] = {**scan["titles"], **current["titles"]}
            current["cwd"] = current["cwd"] or scan["cwd"]
            current["model"] = current["model"] or scan["model"]
        for sid, group in groups.items():
            scan = group["scan"]
            canonical = group["canonical"]
            alternates = group["alternates"]
            job = group["job"]
            title = scan["titles"].get("custom") or scan["titles"].get("ai")
            primary = canonical or (alternates[0] if alternates else root)
            other_paths = [p for p in alternates if p != primary]
            rows.append(
                SourceSession(
                    source="claude",
                    source_id=sid,
                    path=primary,
                    alternate_paths=other_paths,
                    fork_parent_source_id=job.get("forkParentSessionId"),
                    title=title,
                    cwd=scan["cwd"] or job.get("cwd"),
                    created_ts=scan["created"] or parse_ts(job.get("createdAt")),
                    updated_ts=scan["updated"] or parse_ts(job.get("updatedAt")),
                    message_count=scan["messages"],
                    compaction_count=scan["compactions"],
                    availability="full",
                    model=scan["model"],
                )
            )
        rows.extend(_history_only(root, set(groups)))
        rows.sort(key=lambda row: row.updated_ts, reverse=True)
        if include_subagents:
            for row in rows:
                row.warnings.append(
                    "Claude sidechain/subagent inclusion is not supported; "
                    "sidechain records were skipped"
                )
        return rows

    def preview_counts(self, session: SourceSession) -> tuple[int, int]:
        """Listing metadata only; preview must not build the mapped graph."""
        return int(session.message_count or 0), int(session.compaction_count or 0)

    def read_session(self, session: SourceSession) -> Iterator[SourceEvent]:
        stats: dict[str, int] = {}
        records = _load_records(session, stats)
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
        segments, compactions = _segment(records)
        previous_model: str | None = None
        for index, segment in enumerate(segments):
            leaf = None
            if index < len(compactions):
                marker = compactions[index]
                logical = marker.get("boundary", {}).get("logicalParentUuid")
                if isinstance(logical, str):
                    leaf = logical
            path_ids = _active_path(segment, leaf)
            call_ids = {
                str(block.get("id"))
                for record in segment
                if _uuid(record) in path_ids and record.get("type") == "assistant"
                for block in _blocks(record)
                if block.get("type") == "tool_use" and block.get("id")
            }
            for record in segment:
                if _is_sidechain(record) or _is_meta(record):
                    continue
                rid = _uuid(record)
                kind = record.get("type")
                on_path = rid in path_ids
                if kind == "user":
                    blocks = _blocks(record)
                    results = [b for b in blocks if b.get("type") == "tool_result"]
                    if results:
                        for block in results:
                            call_id = str(block.get("tool_use_id") or "")
                            if call_id not in call_ids:
                                continue
                            yield _tool_result_event(
                                record, block, call_id, f"{rid}:result:{call_id}"
                            )
                        if not on_path:
                            continue
                    if not on_path:
                        continue
                    text = _plain_text(record)
                    images = _images(record)
                    if text.strip() or images:
                        yield SourceEvent(
                            source_event_id=rid or f"user:{index}:{ordinal(record)}",
                            parent_source_event_id=record.get("parentUuid"),
                            ts=parse_ts(record.get("timestamp")),
                            kind="user",
                            text=[text] if text else [],
                            images=images,
                        )
                elif kind == "assistant":
                    if not on_path:
                        continue
                    message = json_object(record.get("message"))
                    model = message.get("model")
                    if isinstance(model, str) and model and model != previous_model:
                        previous_model = model
                        yield SourceEvent(
                            source_event_id=f"{rid}:model",
                            parent_source_event_id=record.get("parentUuid"),
                            ts=parse_ts(record.get("timestamp")),
                            kind="model_change",
                            model=model,
                        )
                    text_blocks: list[str] = []
                    thinking: list[str] = []
                    tool_calls: list[dict[str, Any]] = []
                    for block in _blocks(record):
                        if block.get("type") == "text" and isinstance(block.get("text"), str):
                            text_blocks.append(block["text"])
                        elif block.get("type") == "thinking" and isinstance(
                            block.get("thinking"), str
                        ):
                            thinking.append(block["thinking"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": str(block.get("id") or ""),
                                    "name": str(block.get("name") or ""),
                                    "arguments": (
                                        block.get("input")
                                        if isinstance(block.get("input"), dict)
                                        else {}
                                    ),
                                }
                            )
                    if text_blocks or thinking or tool_calls:
                        yield SourceEvent(
                            source_event_id=rid or f"assistant:{index}:{ordinal(record)}",
                            parent_source_event_id=record.get("parentUuid"),
                            ts=parse_ts(record.get("timestamp")),
                            kind="assistant",
                            text=text_blocks,
                            thinking=thinking,
                            tool_calls=tool_calls,
                        )
            if index < len(compactions):
                event = _compaction_event(compactions[index])
                if event is not None:
                    yield event

        title = _title(records)
        if title:
            yield SourceEvent(
                source_event_id="title:custom" if title[1] else "title:ai",
                parent_source_event_id=None,
                ts=session.updated_ts,
                kind="title",
                text=[title[0]],
                raw_flags={"user_set": title[1]},
            )

    def read_memory(self, sessions: Sequence[SourceSession] = ()) -> list[MemoryRecord]:
        root = resolve_root()
        records: list[MemoryRecord] = []
        for name in ("CLAUDE.md", "CLAUDE.local.md"):
            path = root / name
            if path.is_file() and store.is_confined(root, path):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                records.append(
                    MemoryRecord(
                        source_id=name,
                        title=f"Claude instructions ({name})",
                        body=text,
                        tags=["claude", "instructions"],
                        created_ts=path.stat().st_mtime,
                        uri=str(path),
                        status="candidate",
                        scope_hint="instruction",
                    )
                )
        user_memory = root / "agent-memory"
        if user_memory.is_dir():
            records.extend(_memory_dir(user_memory, "personal", None))
        for session in sessions:
            if not session.cwd:
                continue
            cwd = Path(session.cwd).expanduser()
            for name, local in (
                (cwd / ".claude" / "agent-memory", False),
                (cwd / ".claude" / "agent-memory-local", True),
            ):
                if name.is_dir():
                    records.extend(_memory_dir(name, "work", session.source_id, local=local))
        legacy = root / "memory"
        if legacy.is_dir():
            records.extend(_memory_dir(legacy, "work", None))
        return records


def _memory_dir(
    root: Path,
    hint: Literal["personal", "work", "instruction"],
    thread: str | None,
    *,
    local: bool = False,
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for path in sorted(root.rglob("*.md")):
        if not store.is_confined(root, path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        heading = ""
        for index, line in enumerate(text.splitlines()):
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                continue
            if not stripped.startswith(("- ", "* ")):
                continue
            body = stripped[2:].strip()
            if not body:
                continue
            records.append(
                MemoryRecord(
                    source_id=f"{path}:{index}",
                    title=(heading or body)[:150],
                    body=body,
                    tags=["claude", "local" if local else "agent-memory"],
                    created_ts=path.stat().st_mtime,
                    uri=str(path),
                    status="candidate",
                    scope_hint=hint,
                    thread_source_id=thread,
                )
            )
            if len(records) >= 5000:
                return records
    return records


def _load_records(
    session: SourceSession, stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in (session.path, *session.alternate_paths):
        if not path.exists():
            continue
        for _, record in store.iter_jsonl(path, stats=stats):
            rid = _uuid(record)
            if rid and rid in seen:
                continue
            sid = record.get("sessionId")
            if sid not in (None, "", session.source_id) and not _is_boundary(record):
                continue
            if rid:
                seen.add(rid)
            records.append(record)
            if len(records) > MAX_RECORDS:
                return records
    return _include_branch_ancestors(records, session.source_id)


def _include_branch_ancestors(records: list[dict[str, Any]], sid: str) -> list[dict[str, Any]]:
    """Keep no-sessionId rows only when an included row descends from them."""
    parents = {_uuid(record): record.get("parentUuid") for record in records if _uuid(record)}
    included = {
        _uuid(record)
        for record in records
        if record.get("sessionId") == sid or _is_boundary(record)
    }
    cursor = list(included)
    while cursor:
        rid = cursor.pop()
        parent = parents.get(rid)
        if isinstance(parent, str) and parent and parent not in included:
            included.add(parent)
            cursor.append(parent)
    return [
        record
        for record in records
        if not _uuid(record) or _uuid(record) in included or _is_boundary(record)
    ]


def _segment(
    records: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    compactions: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    index = 0
    while index < len(records):
        record = records[index]
        if _is_boundary(record):
            segments.append(current)
            current = []
            summary = None
            if (
                record.get("type") == "system"
                and record.get("subtype") == "compact_boundary"
                and index + 1 < len(records)
                and records[index + 1].get("isCompactSummary")
            ):
                summary = records[index + 1]
                index += 1
            compactions.append(
                {
                    "boundary": record,
                    "summary": (
                        summary
                        if summary is not None
                        else (
                            record
                            if record.get("isCompactSummary") or record.get("type") == "summary"
                            else None
                        )
                    ),
                    "legacy": record.get("type") == "summary",
                }
            )
            index += 1
            continue
        current.append(record)
        index += 1
    segments.append(current)
    return segments, compactions


def _active_path(segment: list[dict[str, Any]], explicit_leaf: str | None) -> set[str]:
    by_id = {_uuid(record): record for record in segment if _uuid(record)}
    if explicit_leaf and explicit_leaf in by_id:
        leaf = by_id[explicit_leaf]
    else:
        has_child = {
            str(record.get("parentUuid"))
            for record in segment
            if isinstance(record.get("parentUuid"), str)
        }
        candidates = [
            record
            for record in segment
            if _uuid(record)
            and _uuid(record) not in has_child
            and record.get("type") in ("user", "assistant")
            and not _is_meta(record)
            and not _is_sidechain(record)
        ]
        if not candidates:
            candidates = [
                record
                for record in segment
                if _uuid(record)
                and record.get("type") in ("user", "assistant")
                and not _is_meta(record)
                and not _is_sidechain(record)
            ]
        if not candidates:
            return set()
        leaf = max(candidates, key=lambda record: parse_ts(record.get("timestamp")))
    path_ids: set[str] = set()
    cursor = leaf
    while cursor is not None and _uuid(cursor) and _uuid(cursor) not in path_ids:
        path_ids.add(_uuid(cursor))
        parent = cursor.get("parentUuid")
        cursor = by_id.get(parent) if isinstance(parent, str) else None
    return path_ids


def _images(record: dict[str, Any]) -> list[ImageContent]:
    images: list[ImageContent] = []
    for block in _blocks(record):
        if block.get("type") != "image":
            continue
        source = json_object(block.get("source"))
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            images.append(
                ImageContent(
                    data=source["data"],
                    mime_type=str(source.get("media_type") or "image/png"),
                )
            )
    return images


def _tool_result_event(
    record: dict[str, Any], block: dict[str, Any], call_id: str, source_event_id: str
) -> SourceEvent:
    content = block.get("content")
    texts: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text") or ""))
    return SourceEvent(
        source_event_id=source_event_id,
        parent_source_event_id=record.get("parentUuid"),
        ts=parse_ts(record.get("timestamp")),
        kind="tool_result",
        tool_result={
            "tool_call_id": call_id,
            "tool_name": "",
            "content": texts,
            "is_error": bool(block.get("is_error")),
        },
    )


def _compaction_event(compaction: dict[str, Any]) -> SourceEvent | None:
    boundary = json_object(compaction.get("boundary"))
    summary_record = compaction.get("summary") or {}
    if compaction.get("legacy"):
        summary = str(summary_record.get("summary") or "")
    else:
        summary = _plain_text(summary_record)
    metadata = json_object(boundary.get("compactMetadata"))
    summary_id = _uuid(summary_record) or _uuid(boundary)
    if not summary.strip():
        return None
    details: dict[str, Any] = {}
    for key in ("trigger", "postTokens", "cumulativeDroppedTokens", "durationMs"):
        if key in metadata:
            details[key] = metadata[key]
    for key in ("preservedSegment", "preservedMessages"):
        if isinstance(metadata.get(key), dict):
            details[key] = metadata[key]
    if boundary.get("logicalParentUuid"):
        details["logical_parent_uuid"] = boundary["logicalParentUuid"]
    if compaction.get("legacy"):
        details["leaf_uuid"] = summary_record.get("leafUuid")
        details["legacy_summary"] = True
    return SourceEvent(
        source_event_id=summary_id or "compaction",
        parent_source_event_id=boundary.get("parentUuid"),
        ts=parse_ts(summary_record.get("timestamp") or boundary.get("timestamp")),
        kind="compaction",
        compaction=CompactionInfo(
            summary=summary,
            first_kept_source_id=None,
            tokens_before=int(metadata.get("preTokens") or 0),
            details=details,
            summary_source_id=summary_id,
        ),
    )


def _title(records: list[dict[str, Any]]) -> tuple[str, bool] | None:
    custom: str | None = None
    ai: str | None = None
    for record in records:
        if record.get("type") == "custom-title" and isinstance(record.get("customTitle"), str):
            custom = record["customTitle"]
        elif record.get("type") == "ai-title" and isinstance(record.get("aiTitle"), str):
            ai = record["aiTitle"]
    if custom and custom.strip():
        return custom.strip(), True
    if ai and ai.strip():
        return ai.strip(), False
    return None


_ordinal_counter: dict[int, int] = {}


def ordinal(record: dict[str, Any]) -> int:
    """Stable-in-process ordinal for a record lacking a uuid."""
    key = id(record)
    value = _ordinal_counter.get(key, 0)
    _ordinal_counter[key] = value + 1
    return value
