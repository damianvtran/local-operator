"""H5b — the decide executor: only a trusted human answer drives ``/decide``.

WHY THIS FILE EXISTS, IN ONE SENTENCE. "Answered by the human" is
CLIENT-ASSERTED (§10.1) — the whole weight of the claim rests on this module
being reachable ONLY from a human answer surface, so it is written as harness
code called by the answer path, never as a model tool and never as a fallback
a model can trigger: the CoS is INSTRUCTED not to answer for the human, and
H5d's bridge tool deliberately carries NO decide authority.

THE TRAP THIS MODULE ENCODES. The ask surface has FOUR settle outcomes and
only TWO of them are decisions: Approve and Deny call ``/decide``; **Decline
("I won't answer now"), timeout and dismiss make NO server call** (§10.3,
#125/D8) — the request stays ``pending`` server-side. Mapping a decline onto a
denial would silently fabricate the human's answer, so the executor's input is
a closed set and everything else raises :class:`NotAHumanDecision` with zero
traffic.

BATCH: §10.2's select-and-approve is per-item explicit — the caller hands the
executor the items the human individually selected, never a "select all" the
code derives. The batch itself is one call (≤50, whole-refusal mirrored in the
client), results are per-item.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .asks_bridge import bridge_ask_ref
from .client import _require_ask_ref  # the SAME validator the decide path uses
from .client import TeamBridgeClient

#: The only two words that may become a server decision (§10.3):
#: approve → "approved"; deny → "denied".
HUMAN_DECISION_WORDS: dict[str, str] = {"approve": "approved", "deny": "denied"}

#: The settle outcomes that MUST NOT produce a call — named here so the
#: refusal reads as the rule, not as a missing branch.
NO_CALL_OUTCOMES = ("decline", "timeout", "dismiss", "defer", "snooze")


class NotAHumanDecision(ValueError):
    """The given outcome is not one of the two decision answers."""


@dataclass(frozen=True)
class HumanDecision:
    """One explicit human answer to one bridge ask.

    Constructed at the human answer surface (the ask panel's Approve/Deny
    action) — ``ask_id`` is the queue's minted ask instance, ``message_id`` the
    request's id. Immutable: an answer is not edited once taken, it is retaken
    through the surface.
    """

    message_id: str
    ask_id: str
    word: str  # "approve" | "deny"

    def receipt(self) -> dict[str, str]:
        return bridge_ask_ref(ask_id=self.ask_id, question_id=self.message_id)


def human_decision(*, message_id: str, ask_id: str, word: str) -> HumanDecision:
    """Validate and freeze one human answer word.

    Refusals carry the CONTRACT's own vocabulary so a caller bug is legible:
    ``decline``/``timeout``/``dismiss`` are stated outcomes that make NO call —
    passing one here is a wiring error, and it raises before any transport
    exists. Anything else unknown is equally refused (closed set).
    """
    normalized = str(word or "").strip().lower()
    if normalized in NO_CALL_OUTCOMES:
        raise NotAHumanDecision(
            f"{normalized!r} is a no-call outcome (§10.3): the request stays pending; "
            "only Approve/Deny drive /decide"
        )
    if normalized not in HUMAN_DECISION_WORDS:
        raise NotAHumanDecision(f"not a human decision word: {word!r} (closed set: approve | deny)")
    if not str(message_id or "").strip() or not str(ask_id or "").strip():
        raise NotAHumanDecision("a human decision names both the message id and the ask id")
    return HumanDecision(message_id=str(message_id), ask_id=str(ask_id), word=normalized)


async def decide_from_human_answer(
    client: TeamBridgeClient,
    decision: HumanDecision,
    *,
    tenant_id: str,
    note: str | None = None,
    expected_account_id: str | None = None,
    access: Any = None,
) -> dict[str, Any]:
    """Perform the question-bound decide for ONE human answer (§10.1).

    The receipt is the ask_ref pair — ``question_id`` IS the message id — and
    ``decided_via`` stays the client-asserted "bearer-only" the client
    already enforces; nothing here can call ``/decide`` without a
    :class:`HumanDecision`, which only :func:`human_decision` builds.
    """
    word = HUMAN_DECISION_WORDS[decision.word]
    receipt = _require_ask_ref(decision.receipt(), message_id=decision.message_id)
    return await client.decide(
        decision.message_id,
        tenant_id=tenant_id,
        decision=word,
        ask_ref=receipt,
        note=note,
        expected_account_id=expected_account_id,
        access=access,
    )


async def decide_batch_from_human_selection(
    client: TeamBridgeClient,
    decisions: Iterable[HumanDecision],
    *,
    tenant_id: str,
    expected_account_id: str | None = None,
    access: Any = None,
) -> dict[str, Any]:
    """One batch decide over the items the human EXPLICITLY selected (§10.2).

    The selection is the caller's evidence — each item was individually
    visible to the human (sender, summary, excerpt, expiry) before its
    Approve/Deny was taken; this function never expands, derives or defaults a
    selection, and the client's ≤50 whole-refusal applies unchanged.
    """
    items: list[dict[str, Any]] = []
    for decision in decisions:
        receipt = _require_ask_ref(decision.receipt(), message_id=decision.message_id)
        items.append(
            {
                "message_id": decision.message_id,
                "decision": HUMAN_DECISION_WORDS[decision.word],
                "ask_ref": receipt,
            }
        )
    if not items:
        raise NotAHumanDecision("a batch decide carries at least one explicitly selected item")
    return await client.decide_batch(
        items, tenant_id=tenant_id, expected_account_id=expected_account_id, access=access
    )


__all__ = [
    "HUMAN_DECISION_WORDS",
    "NO_CALL_OUTCOMES",
    "HumanDecision",
    "NotAHumanDecision",
    "decide_batch_from_human_selection",
    "decide_from_human_answer",
    "human_decision",
]
