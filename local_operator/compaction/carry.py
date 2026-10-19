"""The carried tier: asks still waiting and recent decisions, across a pass.

WHY THIS MODULE EXISTS. ``snapcompact`` foveates the archive — a few frames at
each chronological edge, the middle dropped — and even the text strategy
summarizes everything before the cut. The middle is exactly where a decision
from hours ago and an ask nobody has answered yet live, so a long session
loses both and reports the loss as two different symptoms: "it forgot what we
decided" and "it acted as if it were receiving the initial instruction again".
They are the same middle-loss seen from two ends (measured on session
``439818272d84``: 62.2M characters dropped across 93 passes).

What already exists is the verbatim tier for USER turns
(:func:`~local_operator.compaction.cutpoint.extract_preserved_user_turns`).
Asks must NOT ride that tier: it is token-capped with oldest-first eviction, so
newer chatter can push a still-open question out. This module carries them in a
separate tier bounded by COUNT, which is how "a genuinely blocked ask is never
the thing dropped" is enforced rather than hoped for.

TWO LAWS, and both are pins rather than preferences:

* **Live state every pass.** The writers re-read the ask log and the memory
  store on every pass; nothing is carried forward from the previous payload. An
  answered ask therefore disappears, and an expired ask cannot resurrect. A
  carried tier that accumulated would be worse than the bug it fixes.
* **One renderer, one payload contract.** The live commit and every replay
  reader call :func:`render_carried_tier` with the lists stored on the marker,
  so a live context and a resumed one agree message for message. A tier written
  but not re-injected changes nothing; a tier that works live but not on replay
  regresses silently on resume.

The rendered row is a ``role="user"`` ``Message`` for wire compatibility, but it
is stamped ``harness_injected`` — the existing provenance convention — so no
surface paints it as the operator's words and
``extract_preserved_user_turns`` refuses to re-harvest it across passes.

Nothing here imports ``session``, ``model``, ``providers`` or ``config``: the
compaction package is consumed by hosts that are forbidden those imports. The
ask and memory imports are lazy for the same reason.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_operator.harness.types import Message, TextContent

from .cutpoint import (
    CARRIED_TIER_ID_PREFIX,
    CARRIED_TIER_KEY,
    RENDERED_INJECTION_KEY,
    is_carried_tier as _is_carried_tier,
)
from .marker import replayed_user_message

logger = logging.getLogger(__name__)

__all__ = [
    "CARRIED_ASKS_KEY",
    "CARRIED_ASKS_LIMIT",
    "CARRIED_DECISIONS_KEY",
    "CARRIED_DECISIONS_LIMIT",
    "CARRIED_TIER_ID_PREFIX",
    "CARRIED_TIER_KEY",
    "is_carried_tier",
    "read_recent_decisions",
    "read_unresolved_asks",
    "render_carried_tier",
]

#: Marker payload key for the unresolved-ask list. ADDITIVE: a record without
#: it (every record written before this existed) replays with no carried asks,
#: which is exactly the old behaviour, and a writer with nothing to carry omits
#: the key rather than spelling an empty list.
CARRIED_ASKS_KEY = "carried_asks"

#: Marker payload key for the recent-decision list; same additive contract.
CARRIED_DECISIONS_KEY = "carried_decisions"

#: How many unresolved asks a pass may carry. COUNT-bounded on purpose: a
#: still-blocked ask must never lose a slot to token eviction. It sits well
#: above the queue's own live cap (``asks.policy.OPEN_ASK_CAP`` = 8), so it
#: never truncates a real backlog; it is a backstop against a corrupt log, not
#: a working limit.
CARRIED_ASKS_LIMIT = 50

#: How many decision facts a pass may carry, newest first.
#:
#: **EXPLICIT replacement is honoured; content-based supersession is not.**
#: A fact replaced via the documented ``replaces`` edge is excluded exactly as
#: memory search excludes it (one shared SQL predicate), so read the
#: replacement. The store does NOT derive that two facts say the same thing:
#: content is never compared, so a contradicted decision with no explicit edge
#: may still appear. Facts the store was TOLD are dead — a past ``expires_at``,
#: an ``expired`` status, a ``superseded`` status — are also excluded. Treat
#: the rest as "most recently written", never "in force".
CARRIED_DECISIONS_LIMIT = 20

#: The synthetic row's id IS this prefix (the cut-point predicate matches by
#: prefix so a historically suffixed copy is still recognised). Never
#: journalled: it is synthesized from the marker payload on both paths, so a
#: stored copy would be re-injected twice on resume (the defect that doubled
#: the elision notice, 6 -> 7,167 over ten cycles). ``_is_persistable_message``
#: excludes the prefix, and the live commit drops any old copy from the kept
#: window before re-seating the fresh one. The predicate and key live in
#: ``cutpoint`` because the CUT rules are what must refuse it.

#: Per-question clamp. The tier is count-bounded, never token-bounded; this
#: only stops ONE pathological question from dominating the block. Truncation
#: is not eviction: the ask still appears, with an explicit ellipsis.
_QUESTION_CHARS = 500

_ASK_HEADING = "<unresolved-asks>"
_ASK_CLOSING = "</unresolved-asks>"
_DECISION_HEADING = "<recent-decisions>"
_DECISION_CLOSING = "</recent-decisions>"

#: The opening line, phrased so a model reading it cannot mistake the block for
#: the operator's own words. It is also the wording ``is_harness_notice_text``
#: would need if a legacy copy ever lost its stamp.
_HEADER = (
    "[compaction carry] Harness-maintained records carried across compaction. "
    "These are not the user's words; treat them as a status list."
)


def _utc_stamp(created_at: Any) -> str:
    """A deterministic UTC stamp for an ask's ``created_at`` (epoch ms).

    Deterministic matters: the live commit and a replay render from the same
    payload and must produce byte-identical text. An empty or malformed stamp
    degrades to no timestamp rather than a guess.
    """
    if not isinstance(created_at, (int, float)) or created_at <= 0:
        return ""
    try:
        moment = datetime.fromtimestamp(float(created_at) / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return ""
    return moment.strftime("%Y-%m-%dT%H:%MZ")


def _ask_question(questions: Any) -> str:
    """One line of question text for an ask record, or ``""``.

    A multi-question ask is joined rather than truncated to its first question:
    the operator still owes an answer to every part. Whitespace is collapsed
    because the ask text may be a multi-line prompt, and this block is a list.
    """
    if not isinstance(questions, (list, tuple)):
        return ""
    parts: list[str] = []
    for question in questions:
        if not isinstance(question, Mapping):
            continue
        text = " ".join(str(question.get("question") or "").split())
        if text:
            parts.append(text)
    joined = " / ".join(parts)
    if len(joined) > _QUESTION_CHARS:
        return joined[:_QUESTION_CHARS].rstrip() + "…"
    return joined


def _ask_line(ask: Mapping[str, Any]) -> str:
    """One rendered ask row, or ``""`` when the payload has no question text."""
    question = " ".join(str(ask.get("question") or "").split())
    if not question:
        return ""
    ask_id = str(ask.get("id") or "")
    told = _utc_stamp(ask.get("created_at"))
    stamp = f" (asked {told})" if told else ""
    return f"- {ask_id}{stamp}: {question}"


def _decision_body(decision: Mapping[str, Any]) -> str:
    """One rendered decision body, whitespace-collapsed."""
    return " ".join(str(decision.get("body") or "").split())


def read_unresolved_asks(
    session_directory: Path | str | None,
    *,
    now_ms: int | None = None,
) -> list[dict[str, Any]]:
    """The asks still waiting on the operator, from the durable ask log.

    Rebuilt from the FILE on every pass — never carried forward — so an
    answered ask disappears and an expired ask cannot resurrect. The selection
    is the ask store's own fold (:func:`~local_operator.asks.store.open_asks`),
    not a second interpretation of the log: it already resolves answered,
    declined, dismissed and withdrawn rows, and ``open`` is exactly "queued and
    not past ``expires_at``".

    Missing or unreadable log -> carry nothing, never fail the pass. A pass
    that dies because bookkeeping was absent would turn a missing file into a
    lost conversation.

    Bounded by COUNT (newest first if a corrupt log ever exceeds the cap), not
    by tokens: an open ask is bounded by the queue's own cap, and the cap here
    is a backstop.
    """
    if not session_directory:
        return []
    try:
        from local_operator.asks import store

        events = store.read_events(session_directory)
        if not events:
            return []
        now = store.now_ms() if now_ms is None else now_ms
        records = store.open_asks(store.fold(events, now))
    except Exception:  # noqa: BLE001 - bookkeeping must never fail a pass
        logger.warning("carried asks: ask log read failed", exc_info=True)
        return []
    carried: list[dict[str, Any]] = []
    for record in records[-CARRIED_ASKS_LIMIT:]:
        ask_id = str(record.get("ask_id") or "")
        question = _ask_question(record.get("questions"))
        if not ask_id or not question:
            # An ask with no reconstructible question is not one the model can
            # act on; the fold already skips an ask whose ``queued`` row is
            # missing, and this covers a hand-edited row.
            continue
        created_at = record.get("created_at")
        carried.append(
            {
                "id": ask_id,
                "question": question,
                "created_at": int(created_at) if isinstance(created_at, (int, float)) else 0,
            }
        )
    return carried


def read_recent_decisions(
    store: Any, *, limit: int = CARRIED_DECISIONS_LIMIT
) -> list[dict[str, Any]]:
    """The newest decision facts, from the memory store handed in.

    ``store`` is a :class:`local_operator.memory.MemoryStore` built by the
    caller with the session's identity, so this module never re-derives scopes.
    The read itself is :meth:`MemoryStore.tagged`; this only projects the
    payload shape.

    **EXPLICIT REPLACEMENT, NOT CONTENT-BASED SUPERSESSION.** See
    :data:`CARRIED_DECISIONS_LIMIT`: :meth:`MemoryStore.tagged` hides a fact
    replaced through the documented ``replaces`` edge, as memory search does,
    and the store's dead statuses. It cannot derive that two facts say the same
    thing, so a contradicted decision with no explicit edge may still appear.

    Any failure -> carry nothing. Memory is an enhancement to a compaction
    pass, never a condition of one.
    """
    try:
        facts = store.tagged("decision", limit=limit)
    except Exception:  # noqa: BLE001 - a memory read must not fail a pass
        logger.warning("carried decisions: memory read failed", exc_info=True)
        return []
    carried: list[dict[str, Any]] = []
    for fact in facts[:limit]:
        body = " ".join(str(fact.get("body") or "").split())
        if not body:
            continue
        carried.append({"body": body, "tag": "decision"})
    return carried


#: Re-exported from :mod:`.cutpoint`, where the cut rules consume it.
is_carried_tier = _is_carried_tier


def render_carried_tier(
    carried_asks: Sequence[Mapping[str, Any]] | None,
    carried_decisions: Sequence[Mapping[str, Any]] | None,
) -> Message | None:
    """The single row a pass seats in front of the kept window, or ``None``.

    Called with the FRESH lists at the live commit and with the marker payload
    on replay. One renderer for both is the invariant: a live context and a
    resumed one must agree message for message, and two formatters is where
    that agreement silently rots.

    The row is stamped ``harness_injected`` — the existing provenance marker —
    so a later pass's
    :func:`~local_operator.compaction.cutpoint.extract_preserved_user_turns`
    refuses to harvest it as an operator turn.
    """
    asks = [ask for ask in (carried_asks or ()) if isinstance(ask, Mapping)]
    decisions = [
        decision for decision in (carried_decisions or ()) if isinstance(decision, Mapping)
    ]
    sections: list[list[str]] = []
    ask_rows: list[str] = []
    for ask in asks:
        line = _ask_line(ask)
        if line:
            ask_rows.append(line)
    if ask_rows:
        sections.append([_ASK_HEADING, *ask_rows, _ASK_CLOSING])
    decision_rows: list[str] = []
    for decision in decisions:
        body = _decision_body(decision)
        if body:
            decision_rows.append(f"- {body}")
    if decision_rows:
        sections.append([_DECISION_HEADING, *decision_rows, _DECISION_CLOSING])
    if not sections:
        return None
    lines = [_HEADER, ""]
    for index, section in enumerate(sections):
        if index:
            lines.append("")
        lines.extend(section)
    message = replayed_user_message(
        [TextContent(text="\n".join(lines))], CARRIED_TIER_ID_PREFIX
    )
    message.provider_payload = {RENDERED_INJECTION_KEY: True, CARRIED_TIER_KEY: True}
    return message
