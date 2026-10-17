"""Codex adapter (state_5.sqlite index + rollout JSONL bodies).

Most Codex conversations on this machine are INDEX-ONLY: ``threads`` has 1,203
rows but nearly every ``rollout_path`` is missing. They are listed with an
explicit ``metadata_only`` availability — never silently omitted and never
imported by inventing a conversation from the title.

SQLite is read through a private snapshot (see ``store.sqlite_snapshot``) so
opening a WAL database cannot create a ``-shm`` in the foreign root. Rollout
``.jsonl.zst`` is supported only when the runtime has ``compression.zstd``
(Python 3.14); on older Pythons those rows are listed ``unsupported`` rather
than shelling out to ``zstd``.

Codex compaction semantics differ from lop's: a ``compacted`` record may have
pruned its pre-cut messages and retained only ``replacement_history``. We import
the replacement messages AFTER the marker, and the anchor is the first message
row in that array — a leading compaction/encrypted item is not a message.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from local_operator.harness_import import store
from local_operator.harness_import.sources.base import (
    SourceAdapter,
    UnsupportedBody,
    json_list,
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

MAX_RECORDS = 200_000
#: Explicit thread columns only: never ``SELECT *`` from a foreign schema.
THREAD_COLUMNS = (
    "id",
    "rollout_path",
    "created_at",
    "updated_at",
    "cwd",
    "title",
    "archived",
    "cli_version",
    "first_user_message",
    "model",
    "model_provider",
)


def resolve_root() -> Path:
    explicit = os.environ.get("CODEX_HOME", "").strip()
    return Path(explicit).expanduser() if explicit else Path("~/.codex").expanduser()


def _database(root: Path) -> Path:
    return root / "state_5.sqlite"


def _archived_body(root: Path, source_id: str) -> Path | None:
    archived = root / "archived_sessions"
    if not archived.is_dir():
        return None
    for path in sorted(archived.rglob(f"*{source_id}*")):
        if path.suffix in (".jsonl", ".zst") and path.is_file():
            return path
    return None


def _resolve_body(root: Path, rollout: str | None, source_id: str) -> Path | None:
    if rollout:
        path = Path(rollout).expanduser()
        if path.exists():
            return path
    return _archived_body(root, source_id)


@contextmanager
def _text_handle(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".zst":
        try:
            from compression import zstd  # type: ignore
        except (ImportError, AttributeError) as exc:  # pragma: no cover - py<3.14
            raise UnsupportedBody("zstd rollout requires Python 3.14 compression.zstd") from exc
        try:
            opener = getattr(zstd, "open", None)
            if opener is not None:
                # ``compression.zstd.open`` is the TEXT API on 3.14;
                # ``ZstdFile`` is binary-only and rejects ``encoding``.
                with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[attr-defined]
                    yield handle
                return
            with zstd.ZstdFile(path, "rb") as raw:  # type: ignore[attr-defined]
                yield io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        except (TypeError, ValueError) as exc:
            raise UnsupportedBody(f"zstd rollout unreadable: {type(exc).__name__}") from exc
        return
    with path.open(encoding="utf-8", errors="replace") as handle:
        yield handle


def _source_id_from_body(path: Path) -> str:
    """The authoritative id is ``session_meta.payload.id``, never the filename.

    A real rollout filename ends in a UUID whose own hyphens make a
    ``rfind('-')`` slice return only the last group, and a synthetic name has
    no id at all. Reading the first session_meta is bounded by the wrapper.
    """
    try:
        with _text_handle(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict) or record.get("type") != "session_meta":
                    continue
                payload = json_object(record.get("payload"))
                value = payload.get("id")
                if isinstance(value, str) and value:
                    return value
    except (OSError, UnsupportedBody, TypeError, ValueError):
        return ""
    return ""


def _records(path: Path, stats: dict[str, int] | None = None) -> list[dict[str, Any]]:
    try:
        return _records_uncapped(path, stats)
    except (TypeError, ValueError) as exc:
        raise UnsupportedBody(f"rollout unreadable: {type(exc).__name__}") from exc


def _records_uncapped(path: Path, stats: dict[str, int] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if path.suffix != ".zst":
        # The plain-JSONL path uses the shared bounded reader: same 16 MiB
        # line and 512 MiB file caps as every other adapter.
        for _, record in store.iter_jsonl(path, stats=stats):
            out.append(record)
            if len(out) >= MAX_RECORDS:
                break
        return out
    decompressed = 0
    with _text_handle(path) as handle:
        for line in handle:
            if len(out) >= MAX_RECORDS:
                break
            decompressed += len(line.encode("utf-8"))
            if decompressed > store.MAX_SOURCE_BYTES:
                raise UnsupportedBody(
                    f"zstd rollout exceeds the {store.MAX_SOURCE_BYTES} byte "
                    "decompressed import cap"
                )
            if len(line) > store.MAX_LINE_BYTES:
                if stats is not None:
                    stats["skipped_lines"] = stats.get("skipped_lines", 0) + 1
                    stats["oversized_lines"] = stats.get("oversized_lines", 0) + 1
                continue
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                if stats is not None:
                    stats["skipped_lines"] = stats.get("skipped_lines", 0) + 1
                    stats["malformed_lines"] = stats.get("malformed_lines", 0) + 1
                continue
            if isinstance(record, dict):
                out.append(record)
    return out


def _text_content(content: Any) -> list[str]:
    out: list[str] = []
    for item in json_list(content):
        if not isinstance(item, dict):
            if isinstance(item, str) and item:
                out.append(item)
            continue
        if item.get("type") in ("input_text", "output_text", "text") and isinstance(
            item.get("text"), str
        ):
            out.append(item["text"])
    return out


def _tool_output(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                out.append(text if isinstance(text, str) else json.dumps(item, ensure_ascii=False))
        return out
    if value is None:
        return []
    return [json.dumps(value, ensure_ascii=False)]


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {"input": value}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    return {}


def _canonical_texts(records: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    users: set[str] = set()
    assistants: set[str] = set()
    calls: set[str] = set()
    for record in records:
        if record.get("type") != "response_item":
            continue
        payload = json_object(record.get("payload"))
        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role")
            text = "\n".join(_text_content(payload.get("content")))
            if role == "user" and text:
                users.add(text)
            elif role == "assistant" and text:
                assistants.add(text)
        elif kind in ("function_call", "custom_tool_call"):
            call_id = payload.get("call_id") or payload.get("id")
            if call_id:
                calls.add(str(call_id))
    return users, assistants, calls


def _rollback(events: list[SourceEvent], turns: int) -> int:
    """Drop the last ``turns`` user turns and everything after their start."""
    if turns <= 0:
        return 0
    user_indexes = [i for i, event in enumerate(events) if event.kind == "user"]
    if len(user_indexes) < turns:
        cut = 0
    else:
        cut = user_indexes[-turns]
    del events[cut:]
    return turns


def _worst_availability(rows: list[SourceSession]) -> Availability:
    if any(row.availability == "full" for row in rows):
        return "full"
    if any(row.availability == "metadata_only" for row in rows):
        return "metadata_only"
    return "unsupported"


class CodexAdapter(SourceAdapter):
    source = "codex"
    label = "Codex"

    def __init__(self, config: Path | None = None):
        self.config = Path(config) if config is not None else None
        self._thread_warning = ""

    def detect(self) -> SourceInfo:
        root = resolve_root()
        detected = root.is_dir() and (_database(root).exists() or (root / "sessions").is_dir())
        rows: list[SourceSession] = []
        if detected:
            try:
                rows = self.list_sessions()
            except (OSError, sqlite3.Error):
                rows = []
        detail = "" if detected else f"no Codex store at {root}"
        if detected and not rows and self._thread_warning:
            detail = self._thread_warning
        return SourceInfo(
            source="codex",
            label=self.label,
            root=root,
            detected=detected,
            session_count=len(rows),
            availability=_worst_availability(rows),
            detail=detail,
        )

    def _threads(self) -> tuple[list[dict[str, Any]], str]:
        self._thread_warning = ""
        root = resolve_root()
        database = _database(root)
        if self.config is None or not database.exists():
            return [], "thread index unavailable"
        try:
            with store.sqlite_snapshot(self.config, database) as snapshot:
                db = store.connect_snapshot(snapshot)
                try:
                    table = db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='threads'"
                    ).fetchone()
                    if not table:
                        self._thread_warning = "threads table missing"
                        return [], self._thread_warning
                    rows = db.execute(
                        "SELECT "
                        + ",".join(THREAD_COLUMNS)
                        + " FROM threads ORDER BY updated_at DESC"
                    ).fetchall()
                finally:
                    db.close()
        except (sqlite3.Error, OSError) as exc:
            self._thread_warning = f"thread index unreadable ({type(exc).__name__})"
            return [], self._thread_warning
        self._thread_warning = ""
        return [dict(zip(THREAD_COLUMNS, row, strict=False)) for row in rows], ""

    def list_sessions(self, *, include_subagents: bool = False) -> list[SourceSession]:
        del include_subagents  # Codex has no child-transcript files here
        root = resolve_root()
        threads, reason = self._threads()
        rows: dict[str, SourceSession] = {}
        for thread in threads:
            source_id = str(thread.get("id") or "")
            if not source_id:
                continue
            body = _resolve_body(root, thread.get("rollout_path"), source_id)
            availability: Availability = "full"
            warning = ""
            if body is None:
                availability = "metadata_only"
                warning = "rollout body unavailable; index metadata only"
            elif body.stat().st_size > store.MAX_SOURCE_BYTES:
                # Compressed size is enough for the guard: a .zst over the cap
                # is refused BEFORE decompression (the decompressed stream is
                # bounded again while reading).
                availability = "unsupported"
                warning = f"rollout exceeds the {store.MAX_SOURCE_BYTES} byte import cap"
            elif body.suffix == ".zst":
                try:
                    from compression import zstd  # type: ignore  # noqa: F401
                except (ImportError, AttributeError):
                    availability = "unsupported"
                    warning = "zstd rollout requires Python 3.14 compression.zstd"
            rows[source_id] = SourceSession(
                source="codex",
                source_id=source_id,
                path=body or Path(thread.get("rollout_path") or root),
                title=str(thread.get("title") or "") or None,
                cwd=str(thread.get("cwd") or "") or None,
                created_ts=parse_ts(thread.get("created_at")),
                updated_ts=parse_ts(thread.get("updated_at")),
                message_count=0,
                compaction_count=0,
                availability=availability,
                warnings=[item for item in (warning, reason) if item],
                model=str(thread.get("model") or "") or None,
            )
        # Rollout files not indexed by threads must still be listable.
        sessions_root = root / "sessions"
        if sessions_root.is_dir():
            for path in sorted(sessions_root.rglob("rollout-*.jsonl*")):
                if not store.is_confined(sessions_root, path):
                    continue
                if path.stat().st_size > store.MAX_SOURCE_BYTES:
                    key = f"unreadable:{path.name}"
                    rows[key] = SourceSession(
                        source="codex",
                        source_id=key,
                        path=path,
                        availability="unsupported",
                        warnings=[f"rollout exceeds the {store.MAX_SOURCE_BYTES} byte import cap"],
                    )
                    continue
                try:
                    source_id = _source_id_from_body(path)
                except (UnsupportedBody, OSError, TypeError, ValueError) as exc:
                    source_id = f"unreadable:{path.name}"
                    if source_id in rows:
                        continue
                    rows[source_id] = SourceSession(
                        source="codex",
                        source_id=source_id,
                        path=path,
                        availability="unsupported",
                        warnings=[f"rollout unreadable: {type(exc).__name__}"],
                    )
                    continue
                if not source_id:
                    key = f"unreadable:{path.name}"
                    rows[key] = SourceSession(
                        source="codex",
                        source_id=key,
                        path=path,
                        availability="unsupported",
                        warnings=["rollout unreadable or missing session_meta"],
                    )
                    continue
                if source_id in rows:
                    continue
                availability: Availability = "full"
                warnings: list[str] = []
                if path.stat().st_size > store.MAX_SOURCE_BYTES:
                    availability = "unsupported"
                    warnings.append("rollout exceeds the import size cap")
                rows[source_id] = SourceSession(
                    source="codex",
                    source_id=source_id,
                    path=path,
                    title=None,
                    created_ts=path.stat().st_mtime,
                    updated_ts=path.stat().st_mtime,
                    availability=availability,
                    warnings=warnings,
                )
        result = list(rows.values())
        result.sort(key=lambda row: row.updated_ts, reverse=True)
        del reason
        return result

    def preview_counts(self, session: SourceSession) -> tuple[int, int]:
        """Count readable message/tool rows and compactions without mapping."""
        messages = 0
        compactions = 0
        for record in _records(session.path):
            kind = record.get("type")
            payload = json_object(record.get("payload"))
            if kind == "compacted":
                compactions += 1
            elif kind == "response_item" and payload.get("type") in (
                "message",
                "function_call",
                "custom_tool_call",
                "function_call_output",
                "custom_tool_call_output",
            ):
                if payload.get("type") == "message" and payload.get("role") == "developer":
                    continue
                messages += 1
        return messages, compactions

    def read_session(self, session: SourceSession) -> Iterator[SourceEvent]:
        if session.availability != "full":
            raise UnsupportedBody(session.warnings[0] if session.warnings else "body unavailable")
        stats: dict[str, int] = {}
        records = _records(session.path, stats)
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
        canonical_users, canonical_assistants, canonical_calls = _canonical_texts(records)
        events: list[SourceEvent] = []
        pending: SourceEvent | None = None
        model: str | None = session.model
        tool_names: dict[str, str] = {}
        title = session.title

        def flush() -> None:
            nonlocal pending
            if pending is not None:
                events.append(pending)
                pending = None

        def assistant_event(
            payload: dict[str, Any], record: dict[str, Any], *, synthetic: str | None = None
        ) -> SourceEvent:
            return SourceEvent(
                source_event_id=str(payload.get("id") or f"assistant:{len(events)}"),
                parent_source_event_id=None,
                ts=parse_ts(record.get("timestamp")),
                kind="assistant",
                tool_calls=[],
                thinking=[],
                synthetic=synthetic,
            )

        for record in records:
            kind = record.get("type")
            payload = json_object(record.get("payload"))
            ts = parse_ts(record.get("timestamp"))
            if kind == "session_meta":
                model = model or payload.get("model_provider")
                continue
            if kind == "turn_context":
                next_model = payload.get("model")
                if isinstance(next_model, str) and next_model and next_model != model:
                    flush()
                    model = next_model
                    events.append(
                        SourceEvent(
                            source_event_id=f"model:{len(events)}",
                            parent_source_event_id=None,
                            ts=ts,
                            kind="model_change",
                            model=model,
                        )
                    )
                continue
            if kind == "response_item":
                item_type = payload.get("type")
                if item_type == "message":
                    role = payload.get("role")
                    text = _text_content(payload.get("content"))
                    if role == "developer":
                        continue
                    if role == "user":
                        flush()
                        if text:
                            events.append(
                                SourceEvent(
                                    source_event_id=str(payload.get("id") or f"user:{len(events)}"),
                                    parent_source_event_id=None,
                                    ts=ts,
                                    kind="user",
                                    text=["\n".join(text)],
                                )
                            )
                    elif role == "assistant":
                        if not text:
                            continue
                        if pending is None or pending.tool_calls:
                            flush()
                            pending = assistant_event(payload, record)
                        pending.text.extend(text)
                    continue
                if item_type in ("function_call", "custom_tool_call"):
                    call_id = str(payload.get("call_id") or payload.get("id") or "")
                    name = str(payload.get("name") or "")
                    if not call_id or not name:
                        continue
                    tool_names[call_id] = name
                    if pending is None:
                        pending = assistant_event(payload, record)
                    pending.tool_calls.append(
                        {
                            "id": call_id,
                            "name": name,
                            "arguments": _arguments(
                                payload.get("input")
                                if item_type == "custom_tool_call"
                                else payload.get("arguments")
                            ),
                        }
                    )
                    continue
                if item_type in ("function_call_output", "custom_tool_call_output"):
                    flush()
                    call_id = str(payload.get("call_id") or "")
                    if not call_id:
                        continue
                    events.append(
                        SourceEvent(
                            source_event_id=f"tool:{call_id}",
                            parent_source_event_id=None,
                            ts=ts,
                            kind="tool_result",
                            tool_result={
                                "tool_call_id": call_id,
                                "tool_name": tool_names.get(call_id, ""),
                                "content": _tool_output(payload.get("output")),
                                "is_error": False,
                            },
                        )
                    )
                    continue
                if item_type in ("web_search_call", "tool_search_call"):
                    flush()
                    call_id = str(payload.get("call_id") or payload.get("id") or "")
                    name = "web_search" if item_type == "web_search_call" else "tool_search"
                    if call_id:
                        tool_names[call_id] = name
                        events.append(
                            SourceEvent(
                                source_event_id=f"tool:{call_id}",
                                parent_source_event_id=None,
                                ts=ts,
                                kind="assistant",
                                tool_calls=[
                                    {
                                        "id": call_id,
                                        "name": name,
                                        "arguments": _arguments(
                                            payload.get("action") or payload.get("arguments")
                                        ),
                                    }
                                ],
                            )
                        )
                    continue
                if item_type == "reasoning":
                    if pending is None:
                        pending = assistant_event(payload, record)
                    pending.thinking.append("reasoning")
                    continue
                continue
            if kind == "event_msg":
                event_type = payload.get("type")
                if event_type == "thread_name_updated":
                    value = payload.get("thread_name") or payload.get("name")
                    if isinstance(value, str) and value.strip():
                        title = value.strip()
                        events.append(
                            SourceEvent(
                                source_event_id=f"title:{len(events)}",
                                parent_source_event_id=None,
                                ts=ts,
                                kind="title",
                                text=[title],
                            )
                        )
                    continue
                if event_type == "thread_rolled_back":
                    flush()
                    _rollback(events, int(payload.get("num_turns") or 0))
                    continue
                if event_type in ("user_message", "agent_message", "agent_reasoning"):
                    text = payload.get("message") or payload.get("text")
                    if not isinstance(text, str) or not text:
                        continue
                    if event_type == "user_message":
                        if text in canonical_users:
                            continue
                        flush()
                        events.append(
                            SourceEvent(
                                source_event_id=f"user:{len(events)}",
                                parent_source_event_id=None,
                                ts=ts,
                                kind="user",
                                text=[text],
                            )
                        )
                    elif event_type == "agent_message":
                        if text in canonical_assistants:
                            continue
                        if pending is None or pending.tool_calls:
                            flush()
                            pending = assistant_event(payload, record)
                        pending.text.append(text)
                    else:
                        if pending is None:
                            pending = assistant_event(payload, record)
                        pending.thinking.append("reasoning")
                    continue
                continue
            if kind == "compacted":
                flush()
                summary = str(payload.get("message") or "").strip()
                replacement = json_list(payload.get("replacement_history"))
                details: dict[str, Any] = {
                    "window_id": payload.get("window_id"),
                    "window_number": payload.get("window_number"),
                    "previous_window_id": payload.get("previous_window_id"),
                    "replacement_history_count": len(replacement),
                }
                encrypted = None
                for item in replacement:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "compaction" and isinstance(
                        item.get("encrypted_content"), str
                    ):
                        encrypted = item
                        break
                    if item.get("type") == "compaction_summary":
                        encrypted = item
                        break
                if encrypted is not None:
                    blob = json.dumps(encrypted, sort_keys=True).encode()
                    details["encrypted_item_sha256"] = hashlib.sha256(blob).hexdigest()
                    details["encrypted_item_bytes"] = len(blob)
                events.append(
                    SourceEvent(
                        source_event_id=f"compaction:{len(events)}",
                        parent_source_event_id=None,
                        ts=ts,
                        kind="compaction",
                        compaction=CompactionInfo(
                            summary=summary or "Context compacted by Codex.",
                            tokens_before=0,
                            details=details,
                        ),
                    )
                )
                for ordinal, item in enumerate(replacement):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") in ("compaction", "compaction_summary"):
                        continue
                    synthetic = "replacement_history"
                    item_type = item.get("type")
                    if item_type == "message":
                        role = item.get("role")
                        text = _text_content(item.get("content"))
                        if role == "user" and text:
                            events.append(
                                SourceEvent(
                                    source_event_id=f"replacement:{ordinal}",
                                    parent_source_event_id=None,
                                    ts=ts,
                                    kind="user",
                                    text=["\n".join(text)],
                                    synthetic=synthetic,
                                )
                            )
                        elif role == "assistant" and text:
                            events.append(
                                SourceEvent(
                                    source_event_id=f"replacement:{ordinal}",
                                    parent_source_event_id=None,
                                    ts=ts,
                                    kind="assistant",
                                    text=text,
                                    synthetic=synthetic,
                                )
                            )
                    elif item_type in ("function_call", "custom_tool_call"):
                        call_id = str(item.get("call_id") or item.get("id") or "")
                        name = str(item.get("name") or "")
                        if call_id and name:
                            tool_names[call_id] = name
                            events.append(
                                SourceEvent(
                                    source_event_id=f"replacement:{ordinal}",
                                    parent_source_event_id=None,
                                    ts=ts,
                                    kind="assistant",
                                    tool_calls=[
                                        {
                                            "id": call_id,
                                            "name": name,
                                            "arguments": _arguments(
                                                item.get("input")
                                                if item_type == "custom_tool_call"
                                                else item.get("arguments")
                                            ),
                                        }
                                    ],
                                    synthetic=synthetic,
                                )
                            )
                    elif item_type in ("function_call_output", "custom_tool_call_output"):
                        call_id = str(item.get("call_id") or "")
                        if call_id:
                            events.append(
                                SourceEvent(
                                    source_event_id=f"replacement:{ordinal}",
                                    parent_source_event_id=None,
                                    ts=ts,
                                    kind="tool_result",
                                    tool_result={
                                        "tool_call_id": call_id,
                                        "tool_name": tool_names.get(call_id, ""),
                                        "content": _tool_output(item.get("output")),
                                        "is_error": False,
                                    },
                                    synthetic=synthetic,
                                )
                            )
                continue
        flush()
        for event in events:
            yield event

    def read_memory(self, sessions: Sequence[SourceSession] = ()) -> list[MemoryRecord]:
        root = resolve_root()
        records: list[MemoryRecord] = []
        agents = root / "AGENTS.md"
        if agents.is_file() and store.is_confined(root, agents):
            try:
                text = agents.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if text:
                records.append(
                    MemoryRecord(
                        source_id="AGENTS.md",
                        title="Codex instructions (AGENTS.md)",
                        body=text,
                        tags=["codex", "instructions"],
                        created_ts=agents.stat().st_mtime,
                        uri=str(agents),
                        status="candidate",
                        scope_hint="instruction",
                    )
                )
        database = root / "memories_1.sqlite"
        if self.config is None or not database.exists():
            return records
        try:
            with store.sqlite_snapshot(self.config, database) as snapshot:
                db = store.connect_snapshot(snapshot)
                try:
                    table = db.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='stage1_outputs'"
                    ).fetchone()
                    if not table:
                        return records
                    rows = db.execute(
                        "SELECT raw_memory, rollout_summary, rollout_slug, "
                        "generated_at, usage_count FROM stage1_outputs "
                        "ORDER BY generated_at DESC LIMIT 5000"
                    ).fetchall()
                finally:
                    db.close()
        except (sqlite3.Error, OSError):
            return records
        for index, row in enumerate(rows):
            raw, summary, slug, generated_at, _usage = row
            text = str(raw or summary or "").strip()
            if not text:
                continue
            paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
            for offset, paragraph in enumerate(paragraphs[:8]):
                title = str(slug or "").strip() or paragraph.splitlines()[0][:150]
                records.append(
                    MemoryRecord(
                        source_id=f"stage1:{index}:{offset}",
                        title=title[:150],
                        body=paragraph,
                        tags=["codex"],
                        created_ts=parse_ts(generated_at),
                        source_time=parse_ts(generated_at) or None,
                        uri=str(database),
                        status="candidate",
                        scope_hint="work",
                        source_kind="summary",
                    )
                )
                if len(records) >= 5000:
                    return records
        return records
