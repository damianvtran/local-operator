"""Bridge requests as human-required asks (H5a).

WHY THIS FILE EXISTS. Contract §10.3 (#125, binding): a bridge `request` is a
message whose `request` block asks the RECIPIENT HUMAN for a decision, and its
ask "MUST be queued as ``human_required``" — clearance and classifiers may
assist wording/context but may never suppress the ask, bypass enqueueing,
auto-resolve it, or authorize a decide. This module is the client half of that
invariant:

- :func:`bridge_request_question` builds the ask-question mapping, with the
  carriage the rest of the system keys on: ``human_required`` (the gate
  short-circuit in ``Session._gate_ask`` never lets clearance touch it) and
  ``question_id`` (the request's message id — the binding the server DOES
  check when the decide receipt arrives, §10.1).
- :func:`enqueue_bridge_request_ask` queues it through the SHIPPED ask queue
  (``asks/queue``), so the ask is durable, countable and renderable by the
  surfaces that already exist — no second queue, no parallel bookkeeping.
- :func:`bridge_ask_ref` composes the decide receipt once the human answers:
  ``{ask_id, question_id}`` where ``ask_id`` is the queue's minted ask instance
  and ``question_id`` the message id.

THE STRING IS THE SURFACE. The question text is the contract's stated
composition: "sender + ask summary + body excerpt" (§10.3), options are the
two answer options Approve / Deny, and the excerpt is bounded CLIENT-side
because the ask log is a per-session file and a 64 KiB request body has no
business riding it — the full body stays on the server, the excerpt is a
reading aid. The excerpt bound is a constraint of this store, not a wire rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: The two answer options (§10.3: "options = Approve / Deny"). Ids are stable
#: identifiers the answer surfaces key on; labels are the §2.5 strings.
BRIDGE_ASK_OPTIONS: tuple[dict[str, str], ...] = (
    {"id": "approve", "label": "Approve"},
    {"id": "deny", "label": "Deny"},
)

#: Client-side excerpt bound for the ask question text (chars). The DECIDE
#: never travels through this text — the receipt carries ids — so this bound
#: trades only reading comfort against ask-log size.
EXCERPT_MAX_CHARS = 600


def bridge_request_question(
    *,
    message_id: str,
    sender: str,
    summary: str,
    excerpt: str,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One bridge request as the ask-question mapping the queue stores.

    ``message_id`` is REQUIRED and becomes both the per-question ``id`` stem
    and ``question_id`` — an ask whose question_id is empty could never mint a
    valid decide receipt (the server binds ``question_id == :messageid``), so
    it is refused here rather than discovered at answer time.

    ``scope`` is the ORIGINAL authority block
    (``team_bridge.authority.scope_from_runtime``) frozen AT ISSUANCE (S-3):
    it is what makes the recorded answer recoverable and non-forgeable later —
    recovery re-derives the binding from THIS block, never from the current
    runtime. A question issued WITHOUT it is deliberately non-authorizing for
    recovery (the durable record cannot say whose answer it was); the caller
    that owns a runtime must pass it.
    """
    if not str(message_id or "").strip():
        raise ValueError("a bridge ask needs the request's message id (question_id)")
    from local_operator.harness.message_types import (
        BRIDGE_UNTRUSTED_CLAUSE,
        neutralize_bridge_tag,
    )

    # S3-6: EVERY sender-derived line is flattened to ONE line and tag-
    # neutralised — a newline or a wrapper-tag lookalike inside the sender's
    # name/summary must not open a line of its own outside the framing. The
    # §11.1 clause comes FIRST, ahead of those words, so the frame is read
    # before the untrusted content it frames.
    sender_line = neutralize_bridge_tag(" ".join(str(sender or "").split()))
    summary_line = neutralize_bridge_tag(" ".join(str(summary or "").split()))
    plain = f"{sender_line} · {summary_line}".strip(" ·")
    bounded = " ".join(str(excerpt or "").split())[:EXCERPT_MAX_CHARS]
    if bounded:
        bounded = neutralize_bridge_tag(bounded)
    if plain or bounded:
        text = "\n".join(
            line for line in (BRIDGE_UNTRUSTED_CLAUSE, plain, bounded) if line
        )
    question = {
        "id": f"bridge-{message_id}",
        "question": text,
        "options": [dict(option) for option in BRIDGE_ASK_OPTIONS],
        "multi": False,
        "secret": False,
        "persist": True,
        "human_required": True,
        "question_id": str(message_id),
    }
    if isinstance(scope, Mapping):
        # Additive key ONLY when a runtime supplied the block: an ask without a
        # scope keeps the exact pre-scope shape (byte-compatible logs), and
        # the consumption/recovery paths treat its absence as non-authorizing.
        question["bridge"] = dict(scope)
    return question


def enqueue_bridge_request_ask(
    queue: Any,
    *,
    message_id: str,
    sender: str,
    summary: str,
    excerpt: str,
    timeout_raw: Any = None,
    scope: Mapping[str, Any] | None = None,
    issuance_key: str = "",
    fingerprint: str = "",
) -> dict[str, Any]:
    """Queue the request's ask via the SHIPPED queue; return its own receipt.

    Harness-side enqueue BY DESIGN: the bridge path raises this ask from code
    (the connector's request handling), not from the model, so no clearance
    fork is on the path at all — and the ``human_required`` mark additionally
    short-circuits the gate for any future path that would consult it. The
    returned receipt is the queue's unchanged ``{"ok": ..., ...}`` shape so
    callers handle success and refusal exactly as they do for model asks.
    ``scope`` rides through to the persisted question (S-3).

    When ``issuance_key``/``fingerprint`` are supplied (the bridge issuance
    path always does), the enqueue goes through the queue's KEYED-IDEMPOTENT
    door (S-10): the durable ask id derives from the preallocated issuance
    identity, so a crash retry or a live+cold duplicate resolves to the SAME
    ask. The generic API stays unchanged for every other caller.
    """

    question = bridge_request_question(
        message_id=message_id, sender=sender, summary=summary, excerpt=excerpt, scope=scope
    )
    keyed = getattr(queue, "enqueue_idempotent", None)
    if issuance_key and fingerprint and callable(keyed):
        return keyed(
            issuance_key=issuance_key,
            fingerprint=fingerprint,
            questions=[question],
            timeout=timeout_raw,
        )
    return queue.enqueue([question], timeout_raw)


def bridge_ask_ref(*, ask_id: str, question_id: str) -> dict[str, str]:
    """The decide receipt for one answered bridge ask (§10.1).

    ``ask_id`` is the queue's minted ask instance (``asks/store.py`` ``a-…``);
    ``question_id`` is the request's message id. Both travel because
    "answered by the human" is client-asserted and the pair is what the
    audit trail records.
    """
    return {"ask_id": str(ask_id), "question_id": str(question_id)}


def bridge_pending_count(records: Sequence[Mapping[str, Any]]) -> int:
    """How many OPEN asks are bridge requests — the ``request.badge`` input.

    A bridge ask is identified by its carriage (``question_id`` non-empty on
    at least one question), never by question-text sniffing, so the count
    cannot drift with copy. Counts OPEN records only: the badge says "N
    requests waiting", and a settled ask is not waiting.
    """
    count = 0
    for record in records:
        if record.get("status") != "open":
            continue
        questions = record.get("questions") or []
        if any(str(q.get("question_id") or "") for q in questions):
            count += 1
    return count


__all__ = [
    "BRIDGE_ASK_OPTIONS",
    "EXCERPT_MAX_CHARS",
    "bridge_ask_ref",
    "bridge_pending_count",
    "bridge_request_question",
    "enqueue_bridge_request_ask",
]
