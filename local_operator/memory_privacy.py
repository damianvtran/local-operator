"""Narrow memory-call retained copies, BEFORE journal and diagnostic publication.

Execution keeps the original call so the memory guard can refuse it honestly.
Only history/events get a redacted copy. Raw provider payloads are a second
carrier, not an excuse to retain a credential after scrubbing typed arguments.
No general redaction policy changes for non-memory calls live here.
"""

from __future__ import annotations

import json
from typing import Any


def is_memory(name: str, arguments: Any) -> bool:
    return (
        name in ("read", "write", "edit")
        and isinstance(arguments, dict)
        and isinstance(arguments.get("path"), str)
        and arguments["path"].strip().lower().startswith("memory://")
    )


def guard_encoded(value: Any, depth: int = 0) -> None:
    from local_operator.memory import guard

    guard(value)
    # Numeric document/card values can occur in an invalid structured payload;
    # inspect its JSON spelling too, before another validator can echo it.
    if isinstance(value, (dict, list)):
        guard(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    if depth >= 4:
        return
    if isinstance(value, dict):
        for child in value.values():
            guard_encoded(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            guard_encoded(child, depth + 1)
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return
        guard_encoded(decoded, depth + 1)


def retained_arguments(
    name: str, arguments: Any, raw: str | None = None
) -> tuple[Any, str | None, bool]:
    if not is_memory(name, arguments):
        return arguments, raw, False
    from local_operator.memory import MemoryError, guard

    try:
        guard_encoded(arguments)
        if raw is not None:
            guard_encoded(raw)
    except MemoryError:
        path = arguments["path"]
        try:
            guard(path)
        except MemoryError:
            path = "memory://"
        safe = {"path": path, "memory_arguments_redacted": True}
        return safe, json.dumps(safe), True
    return arguments, raw, False


def retained_message(message: Any) -> Any:
    calls = getattr(message, "tool_calls", None)
    if not calls or not any(is_memory(call.name, call.arguments) for call in calls):
        return message
    from local_operator.memory import MemoryError

    changed = False
    kept = []
    for call in calls:
        args, raw, masked = retained_arguments(call.name, call.arguments, call.raw_arguments)
        kept.append(
            call.model_copy(update={"arguments": args, "raw_arguments": raw}) if masked else call
        )
        changed |= masked
    try:
        guard_encoded(getattr(message, "provider_payload", None))
    except MemoryError:
        changed = True
    if not changed:
        return message
    # A provider may retain its own original tool-call JSON, possibly repaired or
    # escaped differently. Drop this alternate carrier on sensitive memory rows
    # instead of pretending a textual substitution knows every provider schema.
    content = []
    for block in getattr(message, "content", ()):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            try:
                guard_encoded(text)
            except MemoryError:
                block = block.model_copy(update={"text": "[Sensitive memory content redacted]"})
        content.append(block)
    return message.model_copy(
        update={
            "tool_calls": kept,
            "content": content,
            "provider_payload": {"memory_arguments_redacted": True},
        }
    )
