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

THE STRING IS THE SURFACE. The question text is the human's question, and it
is composed for the HUMAN reading it: the first line carries WHO is asking and
WHAT they are asking for ("sender: body excerpt", §10.3), because the picker
bounds a question to one row (`MIN_QUESTION_ROWS`) and only the leading line
reliably survives that bound (D3, design round 2, PR #2378). A missing excerpt
is STATED (`(no preview available)`), never left as a bare identity: silence
asserts a fact the surface cannot know (D9, design round 3). The §11.1 clause
and the thread locator do NOT ride this text: the clause belongs to every
carrier the MODEL reads (the wake wrap, the bridge tool result, the ask
report's preamble, the timeout listing in `asks/render`), and the locator is
not a summary a human decides on. Options are the two answer options
Approve / Deny, and the excerpt is bounded CLIENT-side because the ask log is a
per-session file and a 64 KiB request body has no business riding it — the full
body stays on the server, the excerpt is a reading aid. The excerpt bound is a
constraint of this store, not a wire rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

# ONE spelling for ONE absence, shared with the transcript card: an empty excerpt
# is stated, never left as a bare identity. `card` owns the human-facing copy of
# this feature and imports nothing from here, so this direction carries no cycle.
from local_operator.team_bridge.card import NO_PREVIEW_TEXT

#: The two answer options (§10.3: "options = Approve / Deny"). Ids are stable
#: identifiers the answer surfaces key on; labels are the §2.5 strings.
BRIDGE_ASK_OPTIONS: tuple[dict[str, str], ...] = (
    {"id": "approve", "label": "Approve"},
    {"id": "deny", "label": "Deny"},
)

#: What the question's first line names when no validated member name reached
#: it. NEVER a raw account id: ``acc_…`` identifies nobody the member deciding
#: can recognise, and the decision card exists to say whose human is asking.
GENERIC_ASK_SENDER = "another member of your organization"

#: Client-side excerpt bound for the ask question text (chars). The DECIDE
#: never travels through this text — the receipt carries ids — so this bound
#: trades only reading comfort against ask-log size.
EXCERPT_MAX_CHARS = 600


def bridge_request_question(
    *,
    message_id: str,
    sender: str,
    excerpt: str,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One bridge request as the ask-question mapping the queue stores.

    ``message_id`` is REQUIRED and becomes both the per-question ``id`` stem
    and ``question_id`` — an ask whose question_id is empty could never mint a
    valid decide receipt (the server binds ``question_id == :messageid``), so
    it is refused here rather than discovered at answer time.

    ``sender`` is the asking member's DISPLAY IDENTITY, composed by the issuer
    from the display context it VALIDATED — ``<name> (<organization>)``, the
    same spelling ``TeamBridgeBlock._sender_name`` paints in the card's
    identity column. It is flattened and tag-neutralised like every other
    sender-derived line, and an empty value degrades to
    :data:`GENERIC_ASK_SENDER` rather than to an account id (D3). An empty
    ``excerpt`` composes to :data:`~local_operator.team_bridge.card.NO_PREVIEW_TEXT`
    rather than leaving the question bare (D9).

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
    from local_operator.harness.message_types import neutralize_bridge_tag

    # D3 (design round 2, PR #2378). The FIRST line is the line the reader needs,
    # because the decision card bounds a question to ONE row (`MIN_QUESTION_ROWS`
    # in `tui/widgets/ask_picker.py`) and marks the cut without choosing which
    # line survives: whatever leads is what the member deciding reads. Leading
    # with the §11.1 clause put model-facing vocabulary in front of the human and
    # pushed the sender and the request behind the ``…``, so the card a spend
    # decision was taken on said who had asked and what was being approved to
    # nobody. The clause is not lost by leaving this text — it rides unchanged on
    # every carrier a MODEL reads: the wake wrap (`session.py`, ahead of the
    # body), the bridge tool result, the ask report's own preamble
    # (`BRIDGE_ASK_REPORT_PREAMBLE`), and the timeout listing
    # (`asks/render._timeout_listing_line`), which names the request and the
    # clause itself instead of quoting this text back to the model.
    #
    # The composition is ONE line, so a newline smuggled in the sender or the
    # excerpt cannot open a line of its own: both are flattened, and a wrapper-tag
    # lookalike is neutralised, before they meet (S3-6/S3-7 — the sender's words
    # are untrusted even on a human surface, where the card's own cross-member
    # title is what marks them as another member's).
    sender_line = neutralize_bridge_tag(" ".join(str(sender or "").split()))
    who = sender_line or GENERIC_ASK_SENDER
    bounded = " ".join(str(excerpt or "").split())[:EXCERPT_MAX_CHARS]
    if bounded:
        bounded = neutralize_bridge_tag(bounded)
    # D9 (design round 3, PR #2378): the `who`-alone shape is UNREACHABLE by any
    # input. An empty excerpt states the absence instead — a decision card asking
    # a member to approve nothing, with no sign that the request text is simply
    # unavailable here, is the same failure class D3 was about (*who* AND *what*).
    text = f"{who}: {bounded or NO_PREVIEW_TEXT}"
    question = {
        "id": f"bridge-{message_id}",
        "question": text,
        "options": [dict(option) for option in BRIDGE_ASK_OPTIONS],
        "multi": False,
        "secret": False,
        # NO ``persist`` (round 13). ``persist`` is the CREDENTIAL promotion
        # route — ``AskQuestion.persist`` reads "with ``secret=true``, also save
        # the answer to the operator's encrypted long-term store" — and its only
        # reader is ``_report_secret_answers``' secret branch, which an
        # Approve/Deny question never enters. On this question it was inert AND
        # fatal to the surface: ``AskQuestion._shape`` refuses "persist only
        # applies to a secret question", so the row the queue stored could not
        # be validated by the ask card's models and a REAL bridge ask mounted no
        # card anywhere. This ask's durability is the ask LOG's job (the queue
        # persists every row), not this field's.
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
        message_id=message_id, sender=sender, excerpt=excerpt, scope=scope
    )
    keyed = getattr(queue, "enqueue_idempotent", None)
    if issuance_key and fingerprint and callable(keyed):
        # ``getattr`` erases the callable's type; the door is the queue's own
        # keyed-idempotent enqueue, whose receipt is the record shape below.
        return cast(Callable[..., dict[str, Any]], keyed)(
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
    "GENERIC_ASK_SENDER",
    "bridge_ask_ref",
    "bridge_pending_count",
    "bridge_request_question",
    "enqueue_bridge_request_ask",
]
