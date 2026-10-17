"""Source-adapter protocol and shared timestamp/JSON helpers.

Adapters are READ-ONLY generators. ``list_sessions`` parses only enough of a
body to identify and rank it; ``read_session`` streams the body once. Neither
may write into the foreign root, and neither imports an HTTP/provider client.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from local_operator.harness_import.types import MemoryRecord, SourceEvent, SourceInfo, SourceSession


def parse_ts(value: object, *, fallback: float = 0.0) -> float:
    """ISO 8601 or epoch seconds/millis → Unix seconds; never import time."""
    if isinstance(value, bool) or value is None:
        return fallback
    if isinstance(value, (int, float)):
        number = float(value)
        if number <= 0:
            return fallback
        return number / 1000.0 if number > 10_000_000_000 else number
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        try:
            return parse_ts(float(text), fallback=fallback)
        except ValueError:
            pass
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    return fallback


def json_object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def json_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def json_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


class SourceAdapter(Protocol):
    source: str

    def detect(self) -> SourceInfo: ...

    def list_sessions(self, *, include_subagents: bool = False) -> list[SourceSession]: ...

    def read_session(self, session: SourceSession) -> Iterator[SourceEvent]: ...

    def read_memory(self, sessions: Sequence[SourceSession] = ()) -> list[MemoryRecord]: ...


class UnsupportedBody(RuntimeError):
    """The body format exists but this Python/runtime cannot decode it."""
