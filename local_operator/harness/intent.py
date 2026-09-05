"""The ``i`` intent field: schema injection, sanitisation, streaming scrape.

The working line used to narrate MECHANICS — "thinking", then "composing
bash", then "bash" — which tells the user which machine part is moving and
nothing about the work. What they need is the model's own one-line claim about
what it is doing, and the only component that knows that is the model. So we
ask for it: one extra property on every tool's wire schema, filled by the same
token stream that fills the real arguments, at the cost of no second model
call and a few tokens on a prompt-cached tools array.

Three rules hold this together, and each of them is load-bearing:

1. **The intent is never an argument.** It rides in the arguments dict because
   that is the only channel a tool call has, and it is lifted off before
   validation and before ``execute``. Every builtin params model is pydantic
   with ``extra="forbid"``, so a leaked ``i`` makes strict tools reject the
   call outright.
2. **A malformed intent must cost nothing.** It is narration; the work is the
   call. Anything that is not a usable string degrades to ``None`` and the
   call proceeds exactly as it would have before this file existed.
3. **It is model-controlled text reaching a terminal frame**, so it is
   sanitised with the same treatment tool names get on the approval prompt —
   control sequences stripped, whitespace collapsed, bounded.

Named ``i`` rather than ``intent`` deliberately: it is declared on every tool
in the tools array and emitted on every call, so the short name is paid for
once per prompt-cache prefix and once per call, and the description carries
the meaning the name does not.

The PHRASES built from the field live here too (:func:`tool_activity`,
:func:`batch_activity` and the two ``ACTIVITY_*`` words). Two surfaces narrate
a session's current step — the main conversation's working line and the
subagent panel's rows — and they read side by side in one frame, so the words
are derived once here rather than twice in the two renderers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from local_operator.ansi import sanitize_prompt_line

#: Wire name of the intent property. Matches oh-my-pi's ``INTENT_FIELD``.
INTENT_FIELD = "i"

#: Hard bound on a stored intent. NOT a display width — the renderer
#: ellipsises against the real row width, which it knows and this module does
#: not. This exists so a model that emits a kilobyte cannot park a kilobyte in
#: every event and every session dump.
INTENT_MAX_CHARS = 200

#: What the model is told the field is for, in the schema itself. oh-my-pi
#: ships the bare ``"concise intent"`` here and puts the real instruction in
#: its system prompt; we spend ~30 tokens (once, in the prompt-cached tools
#: array) on a self-sufficient description instead, because the failure this
#: field exists to fix IS the model describing the mechanism, and the schema
#: description is the text closest to the point of emission.
INTENT_DESCRIPTION = (
    "Concise intent: present participle, 2-6 words, no period, capitalized. "
    'What you are accomplishing, not the tool ("Auditing merged MRs", not "Running bash").'
)

#: The exact property injected into every tool schema. Compared by VALUE in
#: :func:`intent_is_injected` to tell our field apart from a tool that
#: genuinely owns a parameter called ``i`` — an MCP server's schema is its
#: own, and stealing its ``i`` would silently drop a real argument.
INTENT_PROPERTY: dict[str, Any] = {"type": "string", "description": INTENT_DESCRIPTION}

#: How far into a streaming argument buffer the partial-intent scrape looks.
#: ``i`` is injected as the FIRST property and a 2-6 word value closes within
#: a few tokens, so anything past this window is not a leading intent; the cap
#: is what keeps the scrape O(1) against a 14 KB ``write`` payload.
INTENT_SCAN_LIMIT = 512

#: Matches a CLOSED leading ``{"i": "..."}`` string in a partial JSON buffer.
#: Anchored at the start of the arguments on purpose: it makes a false
#: positive impossible (no depth tracking, no nested ``"i"`` key, no ``i``
#: inside another string value — a quote there is backslash-escaped and cannot
#: start this match) and it costs one failed match at character three for
#: every call whose first key is something else. An unterminated value simply
#: does not match, which is the wanted behaviour: a label that grows character
#: by character on a repainting row is worse than no label.
_LEADING_INTENT_RE = re.compile(r'\s*\{\s*"i"\s*:\s*"((?:[^"\\\x00-\x1f]|\\.)*)"')


def apply_intent_schema(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return ``parameters`` with the intent property declared.

    Injected at the few places an :class:`AgentTool` is built rather than
    added to each params model: a narration with holes is worse than none —
    the working line would flip register mid-turn — and a per-tool opt-in list
    drifts silently as tools are added.

    ``i`` goes FIRST in ``properties`` because models emit object keys in
    schema order, and a leading intent is what lets
    :func:`scan_streaming_intent` show the claim seconds (for a large
    ``write``, minutes) before the call is executable.

    Optional, never ``required``: under ``extra="forbid"`` a required field
    the model omitted would turn a missing narration into a failed call.

    A schema that already declares ``i`` is returned untouched, so a tool that
    owns that name keeps it (and :func:`intent_is_injected` then reports
    ``False``, which keeps the loop from lifting the value away from it).
    """
    schema: dict[str, Any] = dict(parameters) if parameters else {}
    properties = schema.get("properties")
    if isinstance(properties, dict) and INTENT_FIELD in properties:
        return schema
    existing: Mapping[str, Any] = properties if isinstance(properties, dict) else {}
    schema["properties"] = {INTENT_FIELD: dict(INTENT_PROPERTY), **existing}
    return schema


def intent_is_injected(parameters: Mapping[str, Any] | None) -> bool:
    """Whether ``parameters`` carries OUR intent property (vs. its own ``i``).

    Value equality against :data:`INTENT_PROPERTY` rather than a marker key:
    a marker would ride to the provider on every tool declaration, and some
    strict-schema modes reject unknown keywords. The description is long and
    specific enough that a foreign schema matching it by accident is not a
    case worth designing around.
    """
    if not parameters:
        return False
    properties = parameters.get("properties")
    return isinstance(properties, dict) and properties.get(INTENT_FIELD) == INTENT_PROPERTY


def sanitize_intent(value: Any) -> str | None:
    """Coerce a model-supplied intent to a safe single line, or ``None``.

    The type check is explicit because a value read out of a still-streaming
    JSON buffer has not been schema-validated: an object, number or boolean
    can arrive here. ``sanitize_prompt_line`` does the rest — control
    sequences that could repaint the frame are stripped, bidi/zero-width
    format characters are escaped to something visible, every whitespace run
    (newlines included) collapses to one space, and the result is capped.

    Returns ``None``, never ``""``: an empty string is a value the renderer
    would have to special-case, and "no intent" already has a spelling.
    """
    if not isinstance(value, str):
        return None
    return sanitize_prompt_line(value, limit=INTENT_MAX_CHARS) or None


def scan_streaming_intent(prefix: str) -> str | None:
    """Best-effort intent from a PARTIAL arguments buffer, or ``None``.

    Called while the model is still dictating, where the alternative is a row
    that says "composing write" for as long as the payload takes. Deliberately
    not a JSON parser: the buffer is by definition incomplete, so there is
    nothing to parse — this recognises one closed leading string and gives up
    on everything else.

    Never raises. A malformed escape sequence is a fragment we do not
    understand, not an error worth propagating into the model stream.
    """
    match = _LEADING_INTENT_RE.match(prefix)
    if match is None:
        return None
    try:
        # Re-use the JSON decoder for unescaping (\n, \uXXXX, \\) rather than
        # hand-rolling it; the group is a complete JSON string body, so this
        # is a parse of ~200 characters, not of the streaming payload.
        value = json.loads(f'"{match.group(1)}"')
    except ValueError:
        return None
    return sanitize_intent(value)


#: What a session is doing between one visible step and the next: a model call
#: is in flight and nothing is on the ledger to show for it — no tool running,
#: no prose streamed yet. The main conversation's working line calls that
#: "thinking" and the subagent rows say the same word, because a reader who
#: learns the vocabulary on one surface must not have to learn it again on the
#: other.
ACTIVITY_THINKING = "thinking"

#: Prose is ACTUALLY streaming: at least one text delta of the current message
#: has arrived and no tool call is running. Same word on both surfaces, same
#: reason. The trigger is the first text delta, never ``message_start``: the
#: loop yields ``message_start`` from a placeholder at the top of every provider
#: call, before the first token, and a turn that only emits tool calls never
#: streams a word of prose after it — a renderer that flips to this word on
#: ``message_start`` claims the model is writing for the whole of every call.
ACTIVITY_RESPONDING = "responding"


def tool_activity(display: str, intent: Any) -> str:
    """What a running tool call is DOING: its intent, or a named fallback.

    The intent is why this field exists — ``auditing merged MRs`` is what the
    model is up to, ``tool: bash`` is only how it is going about it — so the
    mechanism is what gets dropped when the model actually said something.

    ``display`` is the caller's already-presentable tool name, not the wire
    one: the TUI runs it through ``glyphs.display_name`` (an MCP tool's minted
    ``mcp__linear_create_issue`` reads as ``create_issue`` there) while the
    subagent relay, which has no display layer, passes the name as called.
    Keeping that choice with the caller is what lets both of them share this
    one phrase instead of growing a second one.
    """
    return sanitize_intent(intent) or f"running {display}"


def batch_activity(phrases: Sequence[str]) -> str:
    """One call's stated purpose; a plain COUNT once there are several.

    A batch drops the intents rather than concatenating or suffixing them:
    presenting one call's purpose as the whole batch's activity is a claim the
    rows above it immediately contradict, and the count is the one fact this
    phrase has that appears nowhere else in the frame.

    Empty means nothing is running, which this function has no word for — the
    caller's own fallback (``ACTIVITY_THINKING``) is the answer there.
    """
    if len(phrases) == 1:
        return phrases[0]
    return f"running {len(phrases)} tools"
