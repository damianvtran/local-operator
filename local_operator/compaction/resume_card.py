"""The resume card: a short replay stand-in for the reprint (#3448).

WHY THIS MODULE EXISTS. A resumed session's context is rebuilt by
``replay_entries(mode="context")`` from four parts — the summary marker, every
preserved user turn verbatim, the carried tier and the whole kept window. The
summary stays small (~0.7k tokens on the measured sessions) but the preserved
block ratchets with every pass and the kept window is ``keep_recent_tokens``
deep, so a reset returns a long session to a floor that climbs over its life.
``compaction.resume_replay = "card"`` trades that reprint for this card plus a
small recent tail of the kept window.

WHAT THE CARD GIVES UP, stated plainly because it is the point of the setting:
the operator's verbatim turns stop surviving a resume. ``extract_preserved_user
_turns`` exists because a paraphrased constraint is how an agent later does the
forbidden thing; in card mode only the newest few operator turns ride the card
(clamped), and everything older reaches the model as the summary's paraphrase.
That is why the mode is opt-in and the default replay is untouched.

LIFECYCLE. The card is BUILT at the compaction commit from the messages the
pass saw and STORED on the marker payload (:data:`RESUME_CARD_PAYLOAD_KEY`) the
way the carried tier's lists are — it cannot be rebuilt at replay, because the
kept-window middle it summarizes is exactly what card replay omits. It is
RENDERED only by replay (never into the live context), by
:func:`render_resume_card`, and the row is a member of the carried family
(:data:`~local_operator.compaction.cutpoint.RESUME_CARD_ID`): stamped
``harness_injected`` so no later pass harvests it as an operator turn, never
journalled, never a cut boundary. Unlike the carried tier it IS summarized by
the next pass, because after a card resume it is the only trace of the
kept-window middle.

Deterministic and model-free on purpose: the card is built on the commit's
critical path and a second summarization call there would double the pass's
cost and failure surface for a transport change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from local_operator.harness.types import AgentMessage, Message, TextContent

from .cutpoint import (
    RENDERED_INJECTION_KEY,
    RESUME_CARD_ID,
    RESUME_CARD_KEY,
    is_carried_tier,
    is_rendered_injection,
)
from .marker import replayed_user_message

__all__ = [
    "RESUME_CARD_PAYLOAD_KEY",
    "RESUME_CARD_TAIL_TOKENS",
    "build_resume_card",
    "render_resume_card",
]

#: Marker payload key for the stored card. ADDITIVE: present only when the
#: writing pass ran in card mode, so a record without it replays in full —
#: which is every record written before this existed and every default-mode
#: record, and is what keeps the default replay byte-identical.
RESUME_CARD_PAYLOAD_KEY = "resume_card"

#: Token budget of the kept-window tail a card replay keeps verbatim after the
#: card. Small on purpose — the in-flight exchange, not a second kept window —
#: and snapped to a pairing-safe boundary
#: (:func:`~local_operator.compaction.cutpoint.resume_tail_start`).
RESUME_CARD_TAIL_TOKENS = 2_000

#: Newest genuine operator turns carried verbatim (clamped) as the objective.
#: A handful, not the block: the block is what the card replaces.
_OBJECTIVE_TURNS = 3
_OBJECTIVE_CHARS = 600
#: Newest tool calls listed with their outcomes.
_ACTIONS = 8
_CALL_CHARS = 160
_OUTCOME_CHARS = 200
_NEXT_STEP_CHARS = 600

_HEADER = (
    "[Resume card — compaction output, not an operator message. This session"
    " resumed in card mode: earlier operator turns are NOT replayed verbatim;"
    " the summary above paraphrases them.]"
)


def _clamp(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _call_text(call: Any) -> str:
    args = call.raw_arguments
    if args is None:
        args = ", ".join(f"{key}={value!r}" for key, value in (call.arguments or {}).items())
    return _clamp(f"{call.name}({args})", _CALL_CHARS)


def build_resume_card(messages: Sequence[AgentMessage]) -> dict[str, Any]:
    """The card's stored form, from the messages one pass saw (oldest first).

    ``messages`` is the summarized partition followed by the kept window. The
    stored form is plain data rather than rendered text so the renderer — the
    only place wording lives — can change without rewriting old records.
    """
    rows = [m for m in messages if isinstance(m, Message) and not is_carried_tier(m)]
    objective: list[str] = []
    for message in reversed(rows):
        if len(objective) >= _OBJECTIVE_TURNS:
            break
        if message.role != "user" or is_rendered_injection(message):
            continue
        text = _clamp(message.text, _OBJECTIVE_CHARS)
        if text:
            objective.append(text)
    objective.reverse()

    outcomes: dict[str, str] = {}
    for message in rows:
        if message.role == "tool" and message.tool_call_id:
            first = next((line for line in message.text.splitlines() if line.strip()), "")
            outcome = _clamp(first, _OUTCOME_CHARS) or "(no output)"
            outcomes[message.tool_call_id] = f"error: {outcome}" if message.is_error else outcome
    actions: list[dict[str, str]] = []
    for message in reversed(rows):
        if len(actions) >= _ACTIONS:
            break
        if message.role != "assistant":
            continue
        for call in reversed(message.tool_calls or ()):
            if len(actions) >= _ACTIONS:
                break
            actions.append(
                {"call": _call_text(call), "outcome": outcomes.get(call.id, "(no result yet)")}
            )
    actions.reverse()

    next_step = ""
    for message in reversed(rows):
        if message.role == "assistant" and message.text.strip():
            next_step = _clamp(message.text, _NEXT_STEP_CHARS)
            break
    return {"objective": objective, "actions": actions, "next_step": next_step}


def render_resume_card(
    card: Mapping[str, Any] | None, carried_asks: Sequence[Any] | None = None
) -> Message | None:
    """The single replay row for a stored card, or ``None`` for no/malformed card.

    Open items are COUNTED here, not listed: the carried tier still rides
    beside the card in card mode and is the one renderer for asks, so listing
    them twice would spend tokens the card exists to save.
    """
    if not isinstance(card, Mapping):
        return None
    lines = [_HEADER, ""]
    objective = [str(t) for t in card.get("objective") or () if str(t).strip()]
    lines.append("Objective — newest operator turns (verbatim, clamped):")
    if objective:
        lines.extend(f"- {turn}" for turn in objective)
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("Last actions → outcomes:")
    actions = [a for a in card.get("actions") or () if isinstance(a, Mapping)]
    if actions:
        lines.extend(f"- {a.get('call', '')} → {a.get('outcome', '')}" for a in actions)
    else:
        lines.append("- (none)")
    lines.append("")
    open_count = len([a for a in carried_asks or () if isinstance(a, Mapping)])
    lines.append(
        f"Open items: {open_count} unresolved ask(s), listed in the carried block."
        if open_count
        else "Open items: no unresolved asks recorded."
    )
    lines.append("")
    next_step = str(card.get("next_step") or "").strip()
    lines.append(f"Next step — last assistant message: {next_step or '(none)'}")
    message = replayed_user_message([TextContent(text="\n".join(lines))], RESUME_CARD_ID)
    message.provider_payload = {RENDERED_INJECTION_KEY: True, RESUME_CARD_KEY: True}
    return message
