"""Near-miss tool-name detection and replay sanitisation.

Why this module exists
----------------------
A model on some routes occasionally streams a tool name with extra trailing
text glued on — measured live as ``bashLecture``, ``bashLectureLecture`` and
further repeats. The wire is clean: the harness only appends the name deltas,
so the garble is the model's own output, not a streaming artifact. What makes
it a problem is the replay: the garbled call and the harness's
``Tool not found: bashLecture`` result are written to the durable transcript
and re-sent on every later turn, and the model tends to copy the name it sees
in its own history. A live probe measured that contagion directly: with three
garbled calls in history the model garbled 4 of 6 fresh calls; with the same
history rewritten to ``bash`` it garbled 0 of 6.

The decision this implements is narrow by design. The call is still rejected
exactly as before — the harness never runs anything under a name the model did
not choose — and the durable transcript is unchanged. What changes is only the
copy of history actually sent to the model: a near-miss call is presented under
the real tool name with a neutral "not run" result, so the junk never reaches
the model a second time. The original row still exists on disk for audit; this
module only rewrites the wire view.

Matching rule
-------------
A candidate tool name is a lowercase identifier of at least three characters
that does not start with ``mcp__``. A garbled name matches when the real name
is its longest starting prefix and what follows is non-empty text that does
NOT start with another lowercase identifier character — ``bashLecture`` gives
``Lecture`` (uppercase start), ``bash-x`` gives ``-x``. Names that continue
as ordinary identifiers (``bash_file``, ``bash2``) are different tools, not
garbles, and never match. An exact name never matches.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from local_operator.harness.derived_ids import NAME_ID_SUFFIX
from local_operator.harness.types import Message, TextContent, ToolCall

logger = logging.getLogger(__name__)

__all__ = [
    "MALFORMED_TOOL_NAME_NOTICE",
    "near_miss_tool",
    "normalise_replay_tool_names",
]

#: Candidates must be plain lowercase identifiers to be considered.
_CANDIDATE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: A remainder starting with one of these is a longer identifier (``bash_file``,
#: ``bash2``) — a different tool, never a garble.
_IDENT_CHAR_RE = re.compile(r"^[a-z0-9_]")

#: ``(real name, remainder length)`` pairs already warned about, so one
#: garbled call costs one WARNING per process instead of one per request
#: build. The remainder itself is never stored — only its length, for the
#: same reason the log line never carries it.
_LOGGED_NEAR_MISSES: set[tuple[str, int]] = set()


def near_miss_tool(name: str, tools: Iterable[object]) -> str | None:
    """The real tool ``name`` garbled from, or ``None`` when it did not.

    ``tools`` is an iterable of tool objects (each with a ``name``) or of bare
    name strings. The longest matching candidate wins; an exact match is not
    a near-miss and returns ``None``.
    """
    if not isinstance(name, str) or not name:
        return None
    candidates: list[str] = []
    for tool in tools:
        tool_name = tool if isinstance(tool, str) else getattr(tool, "name", None)
        if not isinstance(tool_name, str):
            continue
        if len(tool_name) < 3 or tool_name.startswith("mcp__"):
            continue
        if not _CANDIDATE_RE.match(tool_name):
            continue
        candidates.append(tool_name)
    best: str | None = None
    for candidate in candidates:
        if name == candidate:
            return None
        if name.startswith(candidate):
            remainder = name[len(candidate) :]
            if remainder and not _IDENT_CHAR_RE.match(remainder):
                if best is None or len(candidate) > len(best):
                    best = candidate
    return best


#: The result text a sanitised replay carries in place of the original
#: ``Tool not found: <junk>`` line. The junk is never echoed: re-sending it in
#: any form is the contagion path. The real tool is named, always via
#: :func:`format_malformed_notice` for the substitution.
_NOT_RUN_TEMPLATE = (
    "Not run: the tool name in this call was malformed. Call `{tool}` with its arguments."
)

#: Prefix-free constant kept for callers that need the bare template.
MALFORMED_TOOL_NAME_NOTICE = _NOT_RUN_TEMPLATE


def format_malformed_notice(tool: str) -> str:
    """The neutral result text for a sanitised garbled call."""
    return _NOT_RUN_TEMPLATE.format(tool=tool)


def _norm_copy(message: Message, **update: object) -> Message:
    """A sanitised COPY of ``message`` with a derived id (see ``derived_ids``)."""
    update["id"] = f"{message.id}{NAME_ID_SUFFIX}"
    return message.model_copy(update=update)


def normalise_replay_tool_names(
    messages: list[Message], tool_names: Iterable[object]
) -> list[Message]:
    """The same messages, with garbled tool names rewritten for the wire only.

    Each assistant tool call whose name is a near-miss of a real tool is
    rewritten to the real name — arguments untouched — and its paired tool
    result (matched by ``tool_call_id`` and ``is_error``, never by text, so
    transcripts written before this fix are covered too) is replaced with
    the neutral not-run notice and its ``tool_name`` rewritten as well. A
    rewritten assistant copy carries ``provider_payload=None`` because the
    provider-native payload still holds the original garbled name. Messages
    with no near-miss pass through by identity; the function is idempotent
    and never mutates its input.

    Pairing is in order: a pending rewrite is set when its assistant call is
    seen and consumed by the next result with that id, so a call id reused
    later in the history keeps its own genuine result instead of being
    rewritten as though it were the earlier garble's. The pending entry is
    also consumed by a non-error result — the garbled call's only result is
    the harness's error, so that result needed no rewrite, and consuming
    keeps the pending set from leaking onto a later reuse.
    """
    out: list[Message] = []
    pending: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, Message):
            out.append(message)
            continue
        if message.role == "assistant" and message.tool_calls:
            calls: list[ToolCall] = []
            changed = False
            for call in message.tool_calls:
                real = near_miss_tool(call.name, tool_names)
                if real is None:
                    calls.append(call)
                    continue
                pending[call.id] = real
                remainder_len = len(call.name) - len(real)
                if (real, remainder_len) not in _LOGGED_NEAR_MISSES:
                    _LOGGED_NEAR_MISSES.add((real, remainder_len))
                    # The real name and the remainder LENGTH only — never the
                    # remainder itself, which is provider-controlled and may
                    # carry a credential. See loop.py's assembled-call warning
                    # for the same rule.
                    logger.warning(
                        "replay tool name was a near-miss of %s (junk remainder %d chars); "
                        "rewriting the name for the wire only",
                        real,
                        remainder_len,
                    )
                calls.append(call.model_copy(update={"name": real}))
                changed = True
            if changed:
                out.append(_norm_copy(message, tool_calls=calls, provider_payload=None))
            else:
                out.append(message)
        elif message.role == "tool" and message.tool_call_id in pending:
            real = pending.pop(message.tool_call_id)
            if not message.is_error:
                out.append(message)
                continue
            notice = format_malformed_notice(real)
            content = [
                TextContent(text=notice) if isinstance(block, TextContent) else block
                for block in message.content
            ]
            if not content:
                content = [TextContent(text=notice)]
            out.append(_norm_copy(message, content=content, tool_name=real, provider_payload=None))
        else:
            out.append(message)
    return out
