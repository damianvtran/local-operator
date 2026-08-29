"""Speculative compaction advisor (BETA) — a semantic second opinion on WHEN.

The shipped trigger is a SIZE trigger: compact once the context passes
``min(threshold_percent x window, threshold_tokens)``. It is correct about
*whether* the window is filling and says nothing about *where* the session is.
Measured over a real 8102-record session with seven compaction passes, the
active-task span (tokens since the last genuine user turn) at each pass was
0.3k / 46.9k / 48.8k / 30.0k / 19.8k / 123.4k / 49.1k against a
``keep_recent_tokens`` of 20k — five of seven passes cut INSIDE a live task,
summarizing away the first half of work the agent was still executing.

That mismatch, not the token bill, is why an earlier trigger is unsafe: firing
sooner on a recency-shaped cut severs MORE tasks, not fewer. So this module
pairs with :func:`local_operator.compaction.cutpoint.task_boundary_floor`,
which makes the CUT task-shaped, and adds a model judgement about the moment.

Why a model at all: 23 of that session's 69 user turns were continuations
("Continue", "Quota is back, keep going"), which are genuine user turns and
are NOT task boundaries. A purely local "preserve everything since the last
user turn" rule therefore anchors to the wrong place a third of the time,
which is exactly the discrimination a model reading the conversation can make
and a token counter cannot.

Design constraints this module is built around, each of which has cost
something before:

- **The advisor may only WIDEN what is preserved.** :func:`validate_hint`
  rejects any hint whose preserve window is narrower than the local
  ``task_boundary_floor``. That single rule is what makes a wrong or
  hallucinated hint harmless: the worst outcome is an early pass that keeps
  more history than it had to.
- **Reject, never repair.** A hint naming an entry id that is not in the
  candidate set is a hallucination, and a repaired hallucination is a
  hallucination with the evidence removed. Same for over-long ``reason``
  text, which reaches a user-visible receipt.
- **One trigger.** Nothing here decides to compact. The hint feeds
  ``should_compact(..., advisory_ok=True)`` and the effective keep-recent
  passed to ``find_cut_point``; see that function's docstring for why a
  second predicate was refused.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from local_operator.harness.types import AgentMessage, Message

from .cutpoint import task_boundary_floor
from .tokens import estimate_tokens

logger = logging.getLogger(__name__)

__all__ = [
    "ADVISOR_MAX_REASON_CHARS",
    "ADVISOR_SYSTEM_PROMPT",
    "CompactionHint",
    "build_advisor_prompt",
    "parse_hint",
    "validate_hint",
]

#: Cap on the advisor's ``reason``. It is rendered onto the compaction receipt
#: (``compaction_end``'s ``detail``), which is a one-line notice in a terminal
#: transcript, so an essay there is a layout bug. Over-length is REJECTED
#: rather than truncated: a model that ignored an explicit "one short clause"
#: instruction has ignored the format, and the rest of its JSON deserves the
#: same suspicion — the same posture ``naming.parse_title`` takes.
ADVISOR_MAX_REASON_CHARS = 120

#: How many trailing history entries are offered as cut candidates. The
#: advisor picks an anchor from this set and nothing else, so the set IS the
#: hallucination guard; it is bounded because the ids are printed into the
#: prompt and an unbounded list would grow the one thing this call is trying
#: to keep cheap.
ADVISOR_MAX_CANDIDATES = 40

#: The advisor's instructions.
#:
#: Named ``*_SYSTEM_PROMPT`` because it is the advisor's role definition, but
#: it is deliberately NOT sent as a system block. Measured on Anthropic with
#: ``scripts/measure_advisor_cache.py``: the system blocks sit in the cache
#: prefix AHEAD of the messages, so appending one more block diverges the
#: prefix and the request pays a full cache WRITE — 0% cache hit,
#: ``cache_write=14590`` against an identical conversation. Carried inside the
#: appended user turn instead, the same request read 96.1% from cache
#: (``cache_read=14024``, ``cache_write=568`` for the new turn alone). Since
#: the whole economic case for this feature is that the read is cached, the
#: placement is load-bearing: do not "tidy" this into ``system_blocks``.
ADVISOR_SYSTEM_PROMPT = """\
You are a compaction advisor for a long-running agent session. You are NOT \
answering the user and you are NOT continuing the work. You read the \
conversation above and answer one question: is this a good moment to compact \
the context, and how much of the recent history must survive intact?

Compaction replaces older history with a written summary. Everything from the \
anchor you choose to the end of the conversation is kept VERBATIM. Choosing an \
anchor too late destroys work the agent is still in the middle of; choosing one \
too early keeps so much that compaction reclaims nothing.

Pick the anchor at the start of the CURRENT unit of work — the request the \
agent is executing right now. Be careful with short continuation turns \
("continue", "go on", "quota is back"): those resume earlier work, so the task \
started BEFORE them and the anchor belongs at the request they are resuming, \
not at the continuation itself.

Answer with a single fenced JSON block and nothing else:

```json
{"preserve_from": "<entry id from the candidate list>",
 "compact_now": true,
 "confidence": 0.0,
 "reason": "one short clause, under 120 characters"}
```

- `preserve_from` MUST be copied exactly from the candidate list. Never invent \
an id.
- `compact_now` is true only when the agent is between units of work, or the \
current unit is young enough that a summary of everything before it loses \
nothing it still needs.
- `confidence` is your own 0.0-1.0 estimate. Be honest and low when the \
conversation does not make the task structure clear.
"""


@dataclass(frozen=True, slots=True)
class CompactionHint:
    """One validated piece of advice about a compaction pass.

    Frozen because it crosses a background-task boundary into the plan gate;
    a mutable hint would let one consumer's normalisation change what another
    reads. Only :func:`validate_hint` may produce one, so possessing a hint
    means it has already cleared every rejection rule.
    """

    #: Entry id of the first message that must survive verbatim. Always a
    #: member of the candidate set the advisor was shown.
    preserve_from_id: str
    #: Estimated tokens from ``preserve_from_id`` to the end of the history.
    #: Computed HERE, from the real messages, never taken from the model —
    #: the model is trusted to point at a boundary, not to count tokens.
    preserve_tokens: int
    #: Whether the advisor thinks a pass is due now. Honoured only above
    #: ``advisor_floor_tokens``; see ``should_compact``.
    compact_now: bool
    confidence: float
    reason: str
    #: Turn index the hint was produced at, so a hint that lands after the
    #: conversation has moved on can be recognised as stale and dropped. A
    #: late hint is nearly no hint (the posture ``session.naming`` takes for a
    #: late title).
    turn_index: int = 0


def _candidate_messages(messages: Sequence[AgentMessage]) -> list[Message]:
    """Trailing messages an advisor may anchor on, oldest first.

    Restricted to user and tool-free assistant messages because those are the
    positions ``cutpoint._is_valid_cut`` can actually honour; offering an
    anchor the cut point would have to snap away from invites advice that
    silently does not apply.
    """
    candidates: list[Message] = []
    for message in reversed(messages):
        if not isinstance(message, Message):
            continue
        if message.role == "user" or (message.role == "assistant" and not message.tool_calls):
            if not message.text:
                continue
            candidates.append(message)
            if len(candidates) >= ADVISOR_MAX_CANDIDATES:
                break
    candidates.reverse()
    return candidates


def _excerpt(text: str, limit: int = 160) -> str:
    """One-line excerpt for the candidate list."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "…"


def build_advisor_prompt(
    messages: Sequence[AgentMessage],
    *,
    context_tokens: int,
    threshold_tokens: int,
) -> str:
    """The user-role turn appended to the live conversation for one advisor call.

    Deliberately SMALL, and it carries the INSTRUCTIONS as well as the
    question. The conversation itself is not restated here — the call sends
    the live history as its message list (see ``Session.advise_compaction``),
    so repeating any of it would pay twice for the same tokens.

    The instructions ride here rather than in a system block because that is
    what the measurement said: a system block goes in front of the cached
    prefix and costs a full cache write (0% hit), while the same text as a
    trailing user turn leaves the prefix intact (96.1% hit). See
    :data:`ADVISOR_SYSTEM_PROMPT` and ``scripts/measure_advisor_cache.py``.
    Everything this function emits is therefore APPEND-ONLY relative to the
    conversation, which is the property the cache economics rest on.
    """
    candidates = _candidate_messages(messages)
    lines = [f"- {message.id} ({message.role}): {_excerpt(message.text)}" for message in candidates]
    listing = "\n".join(lines) if lines else "(no candidates)"
    return (
        f"{ADVISOR_SYSTEM_PROMPT}\n\n"
        "A compaction decision is pending for the conversation above.\n\n"
        f"Context size: {context_tokens:,} tokens.\n"
        f"Automatic compaction threshold: {threshold_tokens:,} tokens.\n\n"
        "Candidate anchors, oldest first — `preserve_from` must be one of these ids:\n"
        f"{listing}\n\n"
        "Answer with the fenced JSON block described in your instructions and nothing else."
    )


#: A fenced block, optionally tagged ``json``. The model is asked for exactly
#: one; a bare object is also accepted because that is the common near-miss
#: and it is unambiguous, while anything looser (prose containing braces)
#: would make the parser guess.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_RE = re.compile(r"^\s*(\{.*\})\s*$", re.DOTALL)


def parse_hint(raw: str) -> dict[str, Any] | None:
    """The advisor's JSON object, or ``None`` when the answer is unusable.

    Parsing only — every semantic rule lives in :func:`validate_hint`, so the
    rejection reasons stay in one place instead of being split across two
    functions that would each grow their own idea of a valid hint.
    """
    text = str(raw or "")
    match = _FENCE_RE.search(text) or _BARE_RE.match(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_hint(
    payload: dict[str, Any] | None,
    messages: Sequence[AgentMessage],
    *,
    genuine_user_ids: set[str] | None,
    min_confidence: float,
    keep_recent_tokens: int,
    floor_cap: int,
    turn_index: int = 0,
) -> CompactionHint | None:
    """A validated :class:`CompactionHint`, or ``None`` with the reason logged.

    Every rule here is a REJECTION rule. Nothing is repaired, clamped, or
    guessed, because the value of this feature is entirely in the cases where
    the model is right, and none of it is in salvaging the cases where it is
    wrong — a repaired hint is an unvalidated hint wearing a validated hint's
    type.

    The load-bearing rule is the last one: the preserve window the hint asks
    for must be at least ``max(keep_recent_tokens, task_boundary_floor(...))``.
    The advisor may only WIDEN what survives a pass. That is what bounds the
    blast radius of a hallucinating advisor to "an early compaction that kept
    more than it needed to", and it is why this function needs the messages
    rather than trusting the model's own token arithmetic.
    """
    if not payload:
        return None

    raw_id = payload.get("preserve_from")
    if not isinstance(raw_id, str) or not raw_id:
        logger.debug("compaction advisor: hint has no preserve_from")
        return None

    # Hallucination guard: the anchor must be one of the ids we SHOWED it.
    # Checking membership of the live history would not be enough — an id
    # from deep history is equally a sign the model is not reading the list.
    candidate_ids = {message.id for message in _candidate_messages(messages)}
    if raw_id not in candidate_ids:
        logger.debug("compaction advisor: rejected unknown preserve_from id %s", raw_id)
        return None

    raw_confidence = payload.get("confidence")
    if not isinstance(raw_confidence, (int, float)) or isinstance(raw_confidence, bool):
        logger.debug("compaction advisor: rejected non-numeric confidence")
        return None
    confidence = float(raw_confidence)
    if not 0.0 <= confidence <= 1.0 or confidence < min_confidence:
        logger.debug("compaction advisor: rejected confidence %.2f", confidence)
        return None

    reason = payload.get("reason")
    if reason is not None and not isinstance(reason, str):
        return None
    reason = " ".join((reason or "").split())
    if len(reason) > ADVISOR_MAX_REASON_CHARS:
        # Rejected rather than truncated: see ADVISOR_MAX_REASON_CHARS.
        logger.debug("compaction advisor: rejected over-long reason (%d chars)", len(reason))
        return None

    compact_now = payload.get("compact_now")
    if not isinstance(compact_now, bool):
        logger.debug("compaction advisor: rejected non-boolean compact_now")
        return None

    # The preserve window is measured from the REAL messages. A model asked to
    # count its own context tokens will confabulate a plausible number, and
    # that number would be the one guarding the cut.
    index = next(
        (
            position
            for position, message in enumerate(messages)
            if isinstance(message, Message) and message.id == raw_id
        ),
        None,
    )
    if index is None:
        return None
    preserve_tokens = sum(
        estimate_tokens(message) for message in messages[index:] if isinstance(message, Message)
    )

    local_floor = max(
        keep_recent_tokens,
        task_boundary_floor(messages, genuine_user_ids, cap=floor_cap),
    )
    if preserve_tokens < local_floor:
        # WIDEN-ONLY. A hint that would keep LESS than the local rules already
        # keep is not advice, it is damage — and it is the only way a bad hint
        # could sever a live task.
        logger.debug(
            "compaction advisor: rejected narrowing hint (%d < %d tokens)",
            preserve_tokens,
            local_floor,
        )
        return None

    return CompactionHint(
        preserve_from_id=raw_id,
        preserve_tokens=preserve_tokens,
        compact_now=compact_now,
        confidence=confidence,
        reason=reason,
        turn_index=turn_index,
    )
