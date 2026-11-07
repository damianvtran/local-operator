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
    return name == "memory" or (
        name in ("read", "write", "edit")
        and isinstance(arguments, dict)
        and isinstance(arguments.get("path"), str)
        and arguments["path"].strip().lower().startswith("memory:")
    )


def guard_encoded(value: Any, depth: int = 0, field: str | None = None) -> None:
    from local_operator.memory import guard

    # ``field`` follows the value down (a JSON string decoded below keeps the key that
    # held it): it is the context ``guard``'s opaque-run rule reads, and the name a
    # refusal reports, so a nested pass judges a value exactly as the top-level one did.
    guard(value, field=field)
    # Numeric document/card values can occur in an invalid structured payload;
    # inspect its JSON spelling too, before another validator can echo it.
    if isinstance(value, (dict, list)):
        guard(json.dumps(value, ensure_ascii=False, separators=(",", ":")), field=field)
    # No per-field opaque pass here any more: ``guard`` applies one rule to every
    # field (the store's own), so this pre-check and the store cannot disagree.
    if depth >= 4:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_field = f"{field}.{key}" if field else key
            guard_encoded(child, depth + 1, child_field if isinstance(child_field, str) else field)
    elif isinstance(value, list):
        for child in value:
            guard_encoded(child, depth + 1, field)
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return
        guard_encoded(decoded, depth + 1, field)


def _check(name: str, arguments: Any, raw: str | None) -> None:
    """Raise ``MemoryError`` when this memory call may not be retained verbatim."""
    from local_operator.memory import MemoryError, valid_memory_tool_arguments

    if name == "memory" and not valid_memory_tool_arguments(arguments):
        raise MemoryError("Invalid memory argument shape is not retained.")
    guard_encoded(arguments)
    if raw is not None:
        # A streamed call may end inside a JSON escape/token, before a value
        # can be inspected. Known memory calls fail CLOSED for retention;
        # the private call still carries the exact original execution input.
        # Decoding failure must not turn escaped IDs or opaque text into an
        # alternate journal/event/provider-payload carrier (R6/Q9).
        if raw.strip():
            try:
                decoded = json.loads(raw)
                if name == "memory" and not valid_memory_tool_arguments(decoded):
                    raise MemoryError("Invalid raw memory argument shape is not retained.")
            except (ValueError, TypeError):
                raise MemoryError("Unparseable memory arguments are not retained.") from None
        guard_encoded(raw)


#: Upper bound on spans masked in one call before retention gives up and redacts the
#: whole call. Each pass masks one span (offsets shift after a replacement), so this is
#: the number of distinct sensitive spans an explainable refusal may carry.
_MAX_MASKED_SPANS = 16


def _neutral(length: int) -> str:
    """A same-length stand-in for a masked span that can never act as context or value.

    ``[redacted]`` is the marker the shared shape table already treats as masked (its
    ``_NOT_ALREADY_MASKED`` guard), so ``key: [redacted]###`` is not re-read as a
    credential assignment, and ``redacted`` is in neither the secret-name nor the
    public-id vocabulary. Spans shorter than the marker are blanked with ``#``.
    """
    marker = "[redacted]"
    return marker + "#" * (length - len(marker)) if length >= len(marker) else "#" * length


def _masked(name: str, arguments: Any) -> dict[str, Any] | None:
    """The call with ONLY its offending spans replaced by ``[redacted:<kind>]``, or None.

    Why: a whole-call ``{"memory_arguments_redacted": true}`` left the agent's own
    transcript with nothing to correct, so after compaction it could neither see what
    tripped the guard nor recover the rest of the fact. Masking is attempted only where
    it is provably complete: a span in a TOP-LEVEL string argument, located in the
    original text, whose rule says the mask removes the whole credential. The masked
    copy must then pass the full retention check again; anything else (nested/decoded
    JSON, folded-only matches, partial masks, too many spans) returns None and the
    caller keeps the fail-closed whole-call redaction.
    """
    from local_operator.memory import MemoryError, SensitiveValueError

    # Only the ``memory`` tool's own arguments (``text``, ``i``...) are free text. A
    # ``read``/``write``/``edit`` ``memory://`` call's path is a locator: a masked locator
    # is a different address, so those keep the whole-call redaction.
    if name != "memory" or not isinstance(arguments, dict):
        return None
    # Residue is decided on the ORIGINAL text, never on the labelled copy: a label such
    # as ``[redacted:aws-access-key-id]`` ends in ``id``, and re-checking the labelled
    # copy let that word act as public-id context for the AWS secret beside it (agent
    # review round 1, B1). So: take EVERY span the guard reports, blank each with a
    # same-length neutral placeholder (``_neutral``: offsets stay in original coordinates),
    # re-check that neutral copy until it passes, and only then write the labels into
    # the original at the collected spans. The labelled copy is never judged.
    neutral = dict(arguments)
    spans: dict[str, list[tuple[int, int, str]]] = {}
    for _ in range(_MAX_MASKED_SPANS + 1):
        try:
            _check(name, neutral, None)
            break
        except SensitiveValueError as exc:
            if not exc.spans:
                return None
            for span in exc.spans:
                if (
                    span.start is None
                    or span.end is None
                    or not span.complete
                    or not isinstance(span.field, str)
                    or not isinstance(neutral.get(span.field), str)
                ):
                    return None
                text = neutral[span.field]
                neutral[span.field] = (
                    text[: span.start] + _neutral(span.end - span.start) + text[span.end :]
                )
                spans.setdefault(span.field, []).append((span.start, span.end, span.kind))
            if sum(len(found) for found in spans.values()) > _MAX_MASKED_SPANS:
                return None
        except MemoryError:
            return None
    else:
        return None
    masked = dict(arguments)
    for key, found in spans.items():
        merged: list[list[Any]] = []
        for start, end, kind in sorted(found):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end, kind])
        text = masked[key]
        for start, end, kind in reversed(merged):
            text = text[:start] + f"[redacted:{kind}]" + text[end:]
        masked[key] = text
    return masked


def _raw_is_sound(name: str, raw: str | None) -> bool:
    """Whether ``raw`` passes the fail-closed parse/shape checks ``_check`` applies.

    ``_check`` guards the typed arguments BEFORE it parses raw, so a sensitive span can
    be reported for a call whose raw is also unparseable; that call must still be
    redacted whole, never masked.
    """
    from local_operator.memory import valid_memory_tool_arguments

    if raw is None or not raw.strip():
        return True
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return False
    return name != "memory" or valid_memory_tool_arguments(decoded)


def refusal_message(name: str, arguments: Any, raw: str | None = None) -> str | None:
    """The value-free reason this memory call is refused, or None when it is not.

    ``SensitiveValueError`` names field, kind and character span so the agent can fix
    the call. Other refusals (unparseable/invalid shape, size cap) keep the generic
    text the loop always gave: their internal wording is about retention, not the fact.
    """
    from local_operator.memory import MemoryError, SensitiveValueError

    if not is_memory(name, arguments):
        return None
    try:
        _check(name, arguments, raw)
    except SensitiveValueError as exc:
        return str(exc)
    except MemoryError:
        return "Sensitive value refused. Use lop secret; memory may hold its NAME only."
    return None


def retained_arguments(
    name: str, arguments: Any, raw: str | None = None
) -> tuple[Any, str | None, bool]:
    if not is_memory(name, arguments):
        return arguments, raw, False
    from local_operator.memory import MemoryError, SensitiveValueError, guard

    try:
        _check(name, arguments, raw)
    except MemoryError as exc:
        if isinstance(exc, SensitiveValueError) and _raw_is_sound(name, raw):
            # Raw text is not masked span-by-span: re-encode the masked arguments
            # instead, so the retained raw is exactly the retained dict.
            masked = _masked(name, arguments)
            if masked is not None:
                return masked, json.dumps(masked, ensure_ascii=False), True
        safe: dict[str, Any] = {"memory_arguments_redacted": True}
        if name == "memory":
            # Malformed calls are recognised BEFORE validators can echo values. A
            # retained receipt may preserve only these enum-validated controls.
            if isinstance(arguments, dict):
                if arguments.get("op") in ("store", "recall"):
                    safe["op"] = arguments["op"]
        else:
            path = arguments["path"]
            try:
                guard(path)
            except MemoryError:
                path = "memory://"
            safe["path"] = path
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
