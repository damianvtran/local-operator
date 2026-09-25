"""Bound what one turn RE-SENDS, on the replay only, never in the transcript.

Why this exists
---------------
A tool payload is written to the transcript once and then re-sent on every later
turn of the session. Nothing bounded that replay, so a single pathological row
could sit in the prompt of every subsequent request. Measured across 300 real
transcripts in this operator's store (2026-09-20):

* tool RESULTS: 68,482 rows, p50 658 chars, p99 8,472 — the tools layer already
  caps what it keeps at ``TOOL_OUTPUT_LIMIT_CHARS`` (8 KiB, ``tools/builtin.py``),
  but 4% of rows leak past it, the largest observed single row being 300,014
  chars (~75k tokens re-sent on every later turn).
* tool-call ARGUMENTS: 69,038 calls, p50 328 chars, p99 9,652 — **no bound of any
  kind**, largest observed single call 130,491 chars. How much of the store they
  are is SAMPLING-RULE DEPENDENT, so each figure is quoted with its rule: `26.5%`
  of message bytes on a uniform random sample (four seeds, n=300 transcripts) and
  `40.6%` on the 300 largest transcripts by size. An earlier revision stated a
  flat `34%` with no rule named; that number is WITHDRAWN. The conclusion does
  not depend on the rule — arguments were unbounded under every one of them —
  but the number does.

So this is a BACKSTOP, not a latency lever: at the default bound it removes
**2.25% of replayed tokens** as a median — 3,231 median tokens saved over 119
real transcripts (baseline p50 141,459 replayed tokens), measured with the repo's
own ``estimate_messages_tokens``. Latency-wise the PR body's §2 table quotes
~26 ms for this row; that figure is that table's extrapolation of a LARGER saving
statistic than the median, not of the median itself — the §1 fitted 5.17 ms per
1k applied to the median gives 3,231 x 5.17/1e3 ≈ 17 ms. It earns its place by
closing two unbounded paths — one of them entirely uncapped — so that no single
row can dominate a session's every turn. Tightening the bound
is a one-line, measured, deferred decision (see the PR body's table) and it is
decidedly NOT taken by default: at 2,048 chars it would remove **22.41% of
replayed tokens** as a median (32,180 median tokens saved, p90 31.79%; ~263 ms is
the same §2 row's larger-statistic figure, against 32,180 x 5.17/1e3 ≈ 166 ms
from the median) at the cost of eliding the middle of most tool results and long
argument values, which is a lossy global behaviour change and belongs to the
operator rather than to this module's default.

The replay-only rule, and why it is not negotiable
--------------------------------------------------
Elision happens on the WIRE VIEW of a message and never on the message itself.
The stored transcript is the durable record; the in-memory copy is what the
compaction cache, the pruning pass and the next turn's render all read. So this
module returns COPIES and leaves every input message untouched, which is also why
it cannot be folded into the renderer: ``_default_convert_to_llm`` hands out the
transcript's own ``Message`` objects (``harness/render.py``, the
``out.append(message)`` arm), so eliding there would edit the durable record and
poison every reader of it.

A copy carries a DERIVED id, never the original's
-------------------------------------------------
``compaction/tokens.py`` memoizes ``estimate_tokens`` in a process-wide dict
keyed on ``message.id``. Two different contents behind one id means the second
one is answered with the first one's size — an estimate that is silently wrong
in whichever direction was cached, and that rides into the compaction trigger
and the local context estimate. The ``~rb`` suffix keeps the two apart. It is
deterministic, so the bounded view is byte-stable across turns and the provider's
prefix cache is not invalidated by it more than once.

What is bounded, and what is deliberately left alone
---------------------------------------------------
Bounded: every text block of a ``role="tool"`` message, and every string
anywhere in a tool call's argument payload — the EFFECTIVE payload, which is
``raw_arguments`` when that parses as an object, because the wire prefers those
bytes verbatim (``providers/clients.py::_replayable_tool_arguments_json``) and
bounding only ``arguments`` would leave the large string on the wire untouched.
Both fields are rewritten together so the two views cannot disagree.

Left alone: images (bounded by ``imaging.py`` and the compaction frame shed),
and the message list's shape — no message is ever removed or reordered,
because every provider rejects a tool call whose result is missing.

An ``is_error`` result IS bounded, and an earlier revision of this docstring
claimed the opposite ("an error's text is its signal, and it is short in
practice"). That claim contradicts both the code — ``_bound_result`` has no
``is_error`` arm — and this module's own test, which pins the bounded
behaviour. The decision is to keep the bound and say so here: a clipped error
still carries its ``is_error`` flag and the elision marker, so the signal
survives, and an error result is not privileged over any other replay payload
when the whole point is to bound what one turn re-sends. It is rare in
practice (measured: 1 of 821 error rows exceeded the bound), which is why the
wrong sentence went unnoticed rather than why it was right.
"""

from __future__ import annotations

import json
from typing import Any

from local_operator.harness.types import Content, Message, TextContent
from local_operator.text_bounds import clip_head_tail

__all__ = [
    "DEFAULT_REPLAY_BOUND_CHARS",
    "REPLAY_ARGUMENT_MARKER",
    "REPLAY_RESULT_MARKER",
    "bound_replay_payloads",
]

#: Largest single replayed payload, in characters. Chosen as the number the
#: tools layer ALREADY targets for one tool run (``TOOL_OUTPUT_LIMIT_CHARS``, 8
#: KiB) rather than as a new policy: the default therefore changes nothing for
#: the 96% of results that respect it and for every argument value under it, and
#: only enforces the intent the tool layer already had. See the module docstring
#: for what tighter values buy and what they cost.
DEFAULT_REPLAY_BOUND_CHARS = 8 * 1024

#: Written into a tool result where its middle was elided on replay. Distinct
#: from ``text_bounds.OUTPUT_TRUNCATION_MARKER`` on purpose: that one means "the
#: tool cut this, the full bytes are in a spill store you can expand by handle",
#: and it would be a lie here — these bytes live in the durable transcript, which
#: the model can neither see nor address. The recovery route is to re-run.
REPLAY_RESULT_MARKER = (
    "\n\n[... {elided} characters elided on replay; re-run the tool if you need them ...]\n\n"
)

#: Written inside a tool call's long argument value, for the same reason. The
#: call has already run, so the result beside it is the load-bearing record.
REPLAY_ARGUMENT_MARKER = (
    "[... {elided} characters elided on replay; this call already ran — see its result ...]"
)

#: Suffix that keeps a bounded copy's token-memo key distinct from the original's.
_ELIDED_ID_SUFFIX = "~rb"


def bound_replay_payloads(
    messages: list[Message],
    *,
    bound_chars: int = DEFAULT_REPLAY_BOUND_CHARS,
) -> list[Message]:
    """The same messages, with oversized replay payloads elided in copies.

    Returns a NEW list. An input message that needs no elision is passed through
    by identity, so the common turn allocates nothing and copies nothing; an
    input message that does need it is replaced by a copy, and the original is
    never mutated (see the module docstring).

    ``bound_chars <= 0`` disables the bound entirely, which is the escape hatch a
    caller wants when it is measuring the request as the transcript holds it. A
    ``bound_chars`` smaller than the marker sentence yields the marker plus one
    character from each end: the marker is the part that may not be elided, so
    such a bound is served by the smallest result that still states the loss.
    """
    if bound_chars <= 0:
        return list(messages)

    out: list[Message] = []
    for message in messages:
        if message.role == "tool":
            bounded = _bound_result(message, bound_chars)
        elif message.role == "assistant" and message.tool_calls:
            bounded = _bound_calls(message, bound_chars)
        else:
            bounded = message
        out.append(bounded)
    return out


def _bound_result(message: Message, bound_chars: int) -> Message:
    """``message`` with any oversized text block clipped, or unchanged if not."""
    content: list[Content] = []
    changed = False
    for block in message.content:
        if isinstance(block, TextContent) and len(block.text) > bound_chars:
            content.append(TextContent(text=_elide(block.text, bound_chars, REPLAY_RESULT_MARKER)))
            changed = True
        else:
            content.append(block)
    if not changed:
        return message
    return _elided_copy(message, content=content)


def _bound_calls(message: Message, bound_chars: int) -> Message:
    """``message`` with any oversized tool-call argument clipped, or unchanged."""
    calls = list(message.tool_calls)
    changed = False
    for index, call in enumerate(calls):
        bounded = _bounded_call_arguments(call.arguments, call.raw_arguments, bound_chars)
        if bounded is None:
            continue
        arguments, raw = bounded
        calls[index] = call.model_copy(update={"arguments": arguments, "raw_arguments": raw})
        changed = True
    if not changed:
        return message
    return _elided_copy(message, tool_calls=calls)


def _elide(text: str, bound_chars: int, template: str) -> str:
    """``text`` clipped with ``template`` across the gap, at most ``bound_chars`` long.

    The template is formatted BEFORE the clip, from the count it is about to
    report, so the result honours ``bound_chars`` rather than ``bound_chars`` plus
    however many digits that count needed; a second pass covers the one case
    where the count grew a digit and pushed the template longer than assumed.

    ``bound_chars`` is a floor the marker can override, deliberately: a bound
    smaller than the sentence that explains the loss could only be honoured by
    hiding the loss, which is the one thing this module must never do.
    """
    marker_len = len(template.format(elided=0))
    head, tail = clip_head_tail(text, max(1, bound_chars - marker_len))
    marker = template.format(elided=len(text) - len(head) - len(tail))
    if len(marker) > marker_len:  # the count needed more digits than assumed
        head, tail = clip_head_tail(text, max(1, bound_chars - len(marker)))
        marker = template.format(elided=len(text) - len(head) - len(tail))
    return head + marker + tail


def _bounded_call_arguments(
    arguments: dict[str, Any],
    raw_arguments: str | None,
    bound_chars: int,
) -> tuple[dict[str, Any], str | None] | None:
    """``(arguments, raw_arguments)`` bounded together, or None when untouched.

    The effective payload is ``raw_arguments`` parsed when that yields an object
    — those are the bytes the wire replays verbatim — and ``arguments``
    otherwise, including for the unparseable fragments a mid-call abort leaves
    behind (``_replayable_tool_arguments`` already salvages those). Both fields
    are rewritten from the bounded payload so they cannot disagree on the wire;
    ``raw_arguments`` stays None when it was None, letting the client re-encode.

    The length check comes FIRST, and it is not a micro-optimization: this runs
    for every call in the whole history on every request. A raw payload no longer
    than the bound cannot contain a string value longer than the bound (a value is
    always shorter than the JSON that carries it), so the common call is one
    comparison rather than a parse of the model's argument JSON. The claim that a
    short raw means a short effective payload rests on how the two fields are
    written together: the wire replays ``raw_arguments`` when they parse as an
    object, and the fragment an aborted call leaves behind is stored BESIDE the
    empty ``arguments`` the assembler could not fill. A hand-edited transcript
    that broke that pairing would be left as it was before this module existed,
    which is the failure direction to prefer.
    """
    if raw_arguments is not None and len(raw_arguments) <= bound_chars:
        return None
    effective = arguments
    if raw_arguments:
        try:
            parsed = json.loads(raw_arguments)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            effective = parsed
    bounded, changed = _bound_value(effective, bound_chars)
    if not changed:
        return None
    if not isinstance(bounded, dict):  # `_bound_value` preserves the input type
        return None
    return bounded, (json.dumps(bounded) if raw_arguments is not None else None)


def _bound_value(value: Any, bound_chars: int) -> tuple[Any, bool]:
    """``(value, changed)`` with every oversized string anywhere in it elided.

    Walks containers rather than only the top level: a call's payload nests its
    bulk as often as not (``{"edits": [{"new_text": ...}]}``), and a bound that
    only sees top-level keys would leave the largest argument unbounded while
    looking like it had handled it.
    """
    if isinstance(value, str):
        if len(value) <= bound_chars:
            return value, False
        return _elide(value, bound_chars, REPLAY_ARGUMENT_MARKER), True
    if isinstance(value, dict):
        changed = False
        bounded_dict: dict[Any, Any] = {}
        for key, item in value.items():
            new_item, item_changed = _bound_value(item, bound_chars)
            bounded_dict[key] = new_item
            changed = changed or item_changed
        return (bounded_dict, True) if changed else (value, False)
    if isinstance(value, list):
        changed = False
        bounded_list = []
        for item in value:
            new_item, item_changed = _bound_value(item, bound_chars)
            bounded_list.append(new_item)
            changed = changed or item_changed
        return (bounded_list, True) if changed else (value, False)
    return value, False


def _elided_copy(message: Message, **update: Any) -> Message:
    """A bounded COPY of ``message``, with an id the token memo cannot confuse.

    See the module docstring for why the id must differ. The rest of the message
    — role, tool pairing fields, usage, ``provider_payload`` — rides along
    untouched, so the wire still sees a well-formed call/result pair. The
    PAYLOAD does not: the fields this module rewrites (a result's ``content``, a
    call's ``arguments`` and ``raw_arguments``) are exactly the ones the wire
    reads, so a bounded assistant row deliberately loses the provider-native
    argument continuation. Pairing survives; content does not.
    """
    update["id"] = f"{message.id}{_ELIDED_ID_SUFFIX}"
    return message.model_copy(update=update)
