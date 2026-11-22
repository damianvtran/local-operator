"""The desktop's B4b hold reads and Deliver/Discard writes, over the LOCAL bridge store.

WHY A MODULE RATHER THAN CODE IN THE ROUTES (the ``desktop_approvals`` shape).
Three routes answer from one seam, so the list, the releasability rule and the
write cannot grow two policies. Operator decision a-3f46; B4b DESIGN §3, §7.1;
DECISIONS Amendment 1.

THE ROUTE NEVER CALLS THE HUB AND NEVER DELIVERS. It writes ONE signed,
write-once ``owner_releases`` row. The connector's next pass consumes it inside
screening (``screening._consume_owner_release``), re-verifying the signature
and the decision, and then the existing spine admits, reserves, posts the card
and wakes — or posts the reject. A second delivery path beside the connector's
would have to rebuild claims, reservations and the at-least-once rules
(DESIGN §3.1, option B, rejected).

WHO CAN CALL THIS. The router is gated by the desktop bearer, which reaches the
daemon through a descriptor, not the environment, so the chief of staff's
``bash`` cannot present it (DESIGN §4). The bridge tool has no release op and
there is no CLI write verb. On a ``file-only`` key host any process running as
this user can read the key and sign; and on ANY host same-user code that writes
the bridge store can forge an ordinary model allow without this path at all
(security review S-9). The card discloses the first (D3).

WHAT A ROW MAY EXPOSE: ids, timestamps, a coarse kind and closed codes. No body,
no label, no category, no screener reason — the renderer joins the owner-audit
page for sender and body (DESIGN §7.1).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge.screening import HOLD_CODES
from local_operator.team_bridge.store import (
    OWNER_RELEASE_LOCAL_TERMINAL,
    Store,
    StoreRefused,
    screening_hold_message_id,
)
from local_operator.team_bridge.types import (
    CONVERSATION_FEED_KIND,
    CONVERSATION_FEED_TYPE,
    is_conversation_message_id,
    parse_conversation_ref_value,
)

#: Hold codes whose delivery defect is REPAIRABLE, so NO bulk or age-based pass
#: may ever dispose of one.
#:
#: WHY THIS SET EXISTS AND WHO OWNS IT. A ``conversation_failed`` row has a local
#: screening receipt of ``allow`` with the body intact at the hub: it is NOT
#: undeliverable by construction, and the delivery-repair work may well make it
#: deliverable again. Terminalising one converts a fixable delivery defect into
#: permanent data loss with an audit row saying it was deliberate. There are 22
#: such rows on the operator's store today (2026-11-22), and they must survive
#: ``discard-all`` and the expiry pass untouched.
#:
#: THE TRAP THAT MAKES THIS SET NECESSARY: :func:`list_holds` maps any code
#: outside ``HOLD_CODES`` to the SERVED word ``unknown_hold``, and
#: ``conversation_failed`` is not in ``HOLD_CODES`` — so every one of those rows
#: is served as ``unknown_hold``. Classifying on the served value would read them
#: as "unknown code, no decision record" and eat exactly the rows this set
#: protects. Eligibility is therefore computed ONLY from the pre-mapping hold row
#: (``_conversation_rows``) or the stored decision, never from a serialised item.
#:
#: MEMBERSHIP IS THE DELIVERY-DEFECT PROGRAMME'S, NOT THIS MODULE'S. Adding a code
#: here only ever makes this module do LESS; a code may leave this set when that
#: owner proves the class is no longer repairable.
RETRYABLE_DELIVERY_CLASS = frozenset({"conversation_failed"})

#: The most rows ONE expiry pass will dispose of, so a large backlog cannot stall
#: a maintenance tick: the next tick continues where this one stopped, and the
#: store's insert-or-ignore makes a repeated pass over the same row a no-op.
EXPIRY_MAX_PER_PASS = 200

#: Hold codes the person is told about as "the screener was unsure / could not
#: answer"; anything else that is not releasable is "waiting" (infrastructure).
_SCREENING_KIND = frozenset(
    {
        "model_hold",
        "model_refused",
        "reply_truncated",
        "parse_failed",
        "empty_reply",
        "bounce_without_category",
    }
)


#: The ``decision.outcome`` words a list or receipt may show; any other stored
#: outcome (``refused:<code>``) is shown as ``refused``. ``pending`` is the
#: only non-final one: the local terminal is final for the person's purposes.
_OUTCOME_WORDS = frozenset({"pending", "admitted", "rejected", OWNER_RELEASE_LOCAL_TERMINAL})


class HoldRefused(Exception):
    """A closed refusal: ``code`` is the machine word, ``status`` the HTTP status."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


#: Domain tag of the discard receipt (:func:`discard_receipt`). Versioned so a
#: later change to the bound fields can never verify a receipt minted under
#: the old field list.
_DISCARD_RECEIPT_DOMAIN = b"lop.team_bridge.hold_discard_receipt.v1\n"


def discard_receipt(owner: Any, message_id: str, decision: dict[str, Any]) -> str:
    """An opaque binding of ONE local hold record, for a Discard with no body on screen.

    WHY IT EXISTS (operator report 2026-11-20, screenshot ``84249637…``). A held
    message whose Deliver window had passed AND whose body the owner-audit read
    could no longer serve ("Unknown sender", "could not load") was listed with
    ``can_discard`` true, so it held the Approvals badge at 1 — but the renderer
    can only send ``body_sha256`` from a body it has on screen, so neither
    button was armed and nothing could ever settle it. Measured read-only on the
    operator's store: ``bcmsg_5ac4…`` (``model_hold``, decided 2026-11-10 09:23,
    window 168 h, eligible, trusted, no release row).

    WHAT IT BINDS. The live owner key and enrollment generation, the bare
    message id, the conversation, and the decision row's own body hash,
    envelope digest, hold code and decision time. A re-screen that rewrites the
    decision, a re-enrollment, another owner or another message all change it,
    so a stale or mismatched receipt answers ``snapshot_stale`` and never
    settles the wrong message. It is NOT a secret and grants nothing on its own:
    the route still needs the desktop bearer and the captured-owner check, and
    the release is signed exactly as an ordinary Discard is.

    DISCARD ONLY. Deliver keeps requiring the hash of the body the person saw
    (S-11): a receipt says which record, never what the person read.
    """

    fields = [
        _owner_key(owner),
        int(owner.enrollment_generation),
        str(message_id),
        str(decision.get("conversation_id") or ""),
        str(decision.get("body_sha256") or ""),
        str(decision.get("envelope_digest") or ""),
        str(decision.get("hold_code") or ""),
        repr(float(decision.get("created_at") or 0.0)),
    ]
    payload = json.dumps(fields, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(_DISCARD_RECEIPT_DOMAIN + payload).hexdigest()


def _kind(code: str, releasable: bool) -> str:
    if code in hr.RELEASABLE_HOLD_CODES and releasable:
        return "owner_decision"
    if code in _SCREENING_KIND or code in hr.RELEASABLE_HOLD_CODES:
        return "screening"
    return "waiting"


def _conversation_rows(store: Store, owner: Any) -> dict[str, dict[str, Any]]:
    """One row per held CONVERSATION message, by bare id (the composite key's seq)."""

    rows: dict[str, dict[str, Any]] = {}
    for row in store.screening_hold_rows(owner):
        stored = str(row["message_id"])
        bare = screening_hold_message_id(stored)
        if not is_conversation_message_id(bare):
            continue
        seq: int | None = None
        conversation_id = ""
        prefix = f"{CONVERSATION_FEED_TYPE}:{CONVERSATION_FEED_KIND}="
        if stored.startswith(prefix):
            ref = parse_conversation_ref_value(stored[len(prefix) :])
            if ref is not None:
                seq = ref.conversation_seq
                conversation_id = ref.conversation_id
        prior = rows.get(bare)
        if prior is not None and float(prior["created_at"]) <= float(row["created_at"]):
            prior["seq"] = prior["seq"] if prior["seq"] is not None else seq
            prior["conversation_id"] = prior["conversation_id"] or conversation_id
            continue
        rows[bare] = {
            "message_id": bare,
            "conversation_id": conversation_id
            or (prior["conversation_id"] if prior is not None else ""),
            "seq": seq if seq is not None else (prior["seq"] if prior is not None else None),
            "hold_code": str(row["hold_code"]),
            "created_at": float(row["created_at"]),
        }
    return rows


def _terminal_reason(
    row: dict[str, Any],
    decision: dict[str, Any] | None,
    *,
    now: float,
    expiry_days: int,
) -> str | None:
    """WHY this hold can be terminally discarded with NO proof, or ``None``.

    ``row`` is the PRE-MAPPING hold row from :func:`_conversation_rows` (its
    ``hold_code`` is the real stored code — see :data:`RETRYABLE_DELIVERY_CLASS`
    for the trap of using the served one). ``decision`` is the stored
    ``work_scope_decisions`` row, if any.

    THE CLOSED ANSWER SET, and why each member is safe:

    ``deliver_window_passed`` — a decision EXISTS, is eligible, passes
    :func:`hold_release.trusted_decision`, and the Deliver window it opened has
    closed. Past that window ``decide`` refuses Deliver with ``deliver_expired``
    for ever, and a receipt-bound Discard already settles locally with no hub
    call at all, so accepting this with no proof grants NOTHING the receipt path
    did not already grant: it only removes the dependency on the renderer
    holding a receipt for a body it cannot load. Deliver stays impossible, which
    the unchanged ``deliver_expired`` refusal keeps enforcing.

    ``no_decision_record`` — no decision row was ever written AND the HOLD row's
    own ``created_at`` is past the Deliver window. Without a decision row
    ``decide`` refuses ``hold_not_releasable`` for ever, so nothing in this build
    can release or deliver the message and disposal is the only exit that
    exists. Age-bounded deliberately: an in-flight screener that is about to
    write a decision must not be raced out from under.

    ``payload_expired`` — the HOLD row is older than the configurable
    ``team_bridge.held_expiry_days`` retention threshold. The operator-requested
    backstop, and the only one that can settle a row whose decision exists but
    can never be acted on.

    AGE IS THE HOLD'S OWN ``created_at``, never ``held_at``: ``held_at`` is the
    DECISION time, which can be days later, and using it would make a row look
    younger than it is. ``cleared_at`` is NOT used for anything here: it means
    "this device retired the OS banner" (``hold_release.retire_review_notices``),
    not "the hold was settled", and a cleared decision can still be perfectly
    deliverable (trap pinned by ``test_cleared_but_deliverable_is_not_terminal``).
    """

    code = str(row["hold_code"])
    if code in RETRYABLE_DELIVERY_CLASS:
        return None
    age = now - float(row["created_at"])
    if decision is not None:
        if (
            decision.get("eligible")
            and str(decision.get("hold_code") or "") == code
            and hr.trusted_decision(decision)
            and now > float(decision.get("created_at") or 0.0) + hr.DELIVER_WINDOW_S
        ):
            return "deliver_window_passed"
    elif age > hr.DELIVER_WINDOW_S:
        return "no_decision_record"
    if expiry_days > 0 and age > float(expiry_days) * 86400.0:
        return "payload_expired"
    return None


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).isoformat()


def list_holds(config_dir: Path, owner: Any, *, now: float | None = None) -> dict[str, Any]:
    """``GET …/holds``: every conversation hold on this machine, metadata only."""

    store = Store(config_dir)
    now = time.time() if now is None else float(now)
    # ``enabled`` is "any release path is open": the work-scope class
    # (``hold_release``) or the owner review of cached answer holds
    # (``review_held``, default on). Each item's own class is gated below.
    enabled = hr.any_enabled(config_dir)
    review_on = hr.review_enabled(config_dir)
    work_scope_on = hr.enabled(config_dir)
    decisions = {str(d["message_id"]): d for d in store.work_scope_decisions(owner)}
    # The ONE terminal marker (``hold_terminals``): a row carrying one was already
    # disposed of, so it is not held, not listed and NOT counted — that is the
    # whole fix. Dropping it here (rather than per-surface) is what makes the
    # badge, the desktop list and ``lop team-bridge holds`` agree by construction.
    terminals = {str(t["message_id"]) for t in store.hold_terminals(owner)}
    expiry_days = hr.held_expiry_days(config_dir)
    items: list[dict[str, Any]] = []
    # Oldest hold first, the bare id breaking ties: two holds written in the
    # same instant must not swap places between refreshes (a list that reorders
    # under the person's pointer reads as rows jumping).
    for bare, row in sorted(
        _conversation_rows(store, owner).items(),
        key=lambda kv: (kv[1]["created_at"], kv[0]),
    ):
        if bare in terminals:
            # Disposed of: see the marker read above. Never re-served, whatever
            # the hold row still says — the retry loop may not have deleted it
            # yet, or may re-upsert it while a later attempt is in flight.
            continue
        code = row["hold_code"]
        decision = decisions.get(bare)
        eligible = bool(
            (review_on if code in hr.REVIEW_HOLD_CODES else work_scope_on)
            and code in hr.RELEASABLE_HOLD_CODES
            and decision is not None
            and str(decision.get("hold_code") or "") == code
            and decision.get("eligible")
            and hr.trusted_decision(decision)
        )
        held_at = float(decision["created_at"]) if decision is not None else row["created_at"]
        deliver_until = held_at + hr.DELIVER_WINDOW_S if eligible else None
        release = store.owner_release(owner, bare)
        can_discard = eligible
        conversation_id = row["conversation_id"] or (
            str(decision.get("conversation_id") or "") if decision else ""
        )
        can_deliver = bool(
            eligible
            and deliver_until is not None
            and now <= deliver_until
            and code not in hr.DISCARD_ONLY_HOLD_CODES
        )
        # PR #4358 R1-M1: ``decide`` refuses unless the DECISION's own
        # conversation equals the posted one, so a row whose decision names
        # another (or no) conversation is not something either button can
        # settle; counting it would be the stuck badge again.
        decision_bound = bool(
            decision is not None
            and conversation_id
            and str(decision.get("conversation_id") or "") == conversation_id
        )
        # Additive: WHY this row can be disposed of with NO proof at all, or
        # ``None``. Computed from the REAL hold row (``row``), never from
        # ``code``, which is the served word below (:func:`_terminal_reason`
        # explains the trap). A row that can still be delivered is never a
        # candidate, whatever its age.
        undeliverable_reason = (
            None
            if can_deliver
            else _terminal_reason(row, decision, now=now, expiry_days=expiry_days)
        )
        terminal_discard = bool(undeliverable_reason)
        items.append(
            {
                "message_id": bare,
                "conversation_id": conversation_id,
                "seq": row["seq"],
                "held_at": _iso(held_at),
                # When the message was FIRST held (the hold row's own time).
                # ``held_at`` is the decision time — the Deliver window's start —
                # and a review decision can be written days after the hold
                # (measured 2026-11-18: 17.9 d vs 8.1 d), so "how long has this
                # sat here" needs this one. Additive; ``lop team-bridge holds``
                # reads it for its age column.
                "first_held_at": _iso(row["created_at"]),
                "kind": _kind(code, eligible),
                # Closed codes only (``HOLD_CODES`` or ``unknown_hold``), like the read slice.
                "hold_code": code if code in HOLD_CODES else "unknown_hold",
                "releasable": eligible,
                "can_deliver": can_deliver,
                "can_discard": can_discard,
                # ONE predicate for "the person owes this an answer", so the
                # badge, the nav count and the list cannot disagree (the
                # renderer re-derived it and counted rows it could not act
                # on). A recorded decision — pending, applied or refused — is
                # settled from the person's side and never counted again.
                "actionable": bool(
                    release is None and decision_bound and (can_deliver or can_discard)
                ),
                # Present only where Discard is open and undecided: lets the
                # renderer discard a hold whose body can no longer be loaded
                # (see :func:`discard_receipt`). Never offered for Deliver.
                "discard_receipt": (
                    discard_receipt(owner, bare, decision)
                    if can_discard and release is None and decision_bound and decision is not None
                    else None
                ),
                # PR #4212 S-4: an aged hold that NO model read (provider outage,
                # hub unavailable) is a Deliver item too, but the owner must be able
                # to tell it from a screener's "unsure": its decision row names no
                # screener and no verdict. A closed boolean; the copy is the
                # renderer's.
                "unscreened": bool(
                    decision is not None
                    and code in hr.REVIEW_HOLD_CODES
                    and not str(decision.get("screener_model") or "")
                    and not str(decision.get("screener_verdict") or "")
                ),
                "deliver_until": _iso(deliver_until),
                "decision": _decision_view(release),
                # Additive, and the reason this change exists: True when Discard
                # may be sent with NO proof (no ``body_sha256``, no
                # ``hold_receipt``) because the row is provably undeliverable —
                # see :func:`_terminal_reason` for the closed rules. A renderer
                # arms Discard on this flag; Deliver is never armed by it.
                "terminal_discard": terminal_discard,
                # The closed code behind it (one of
                # ``store.HOLD_TERMINAL_DETAILS``), or null. Never free text, and
                # never derived from a failed body load, a 404 or a missing
                # record: only from rows actually read.
                "undeliverable_reason": undeliverable_reason,
            }
        )
    return {
        "enabled": enabled,
        "key_protection": hr.key_protection(config_dir),
        "items": items,
    }


def _outcome_word(release: dict[str, Any]) -> str:
    outcome = str(release.get("outcome") or "pending")
    return outcome if outcome in _OUTCOME_WORDS else "refused"


def _decision_view(release: dict[str, Any] | None) -> dict[str, Any] | None:
    if release is None:
        return None
    return {
        "word": release.get("word"),
        "decided_at": _iso(release.get("decided_at")),
        "outcome": _outcome_word(release),
        "key_id": release.get("key_id"),
    }


def decide(
    config_dir: Path,
    owner: Any,
    *,
    message_id: str,
    conversation_id: str,
    body_sha256: str | None,
    word: str,
    account_id: str,
    now: float | None = None,
    hold_receipt: str | None = None,
) -> dict[str, Any]:
    """``POST …/holds/{id}/deliver|discard``: sign and write ONE release row.

    Every check that can refuse runs BEFORE the signer, so a refused request
    never raises the operator prompt. Idempotent: the same word again returns
    the same row; the other word is ``already_decided``.
    """

    if word not in hr.WORDS:
        raise HoldRefused(422, "invalid_request")
    # Exactly one binding. A receipt names a RECORD, not bytes the person read,
    # so it can never carry a Deliver (S-11); checked before any store read.
    if (body_sha256 is None) == (hold_receipt is None) or (
        word == "deliver" and hold_receipt is not None
    ):
        raise HoldRefused(422, "invalid_request")
    if not hr.any_enabled(config_dir):
        raise HoldRefused(409, "hold_release_off")
    if not is_conversation_message_id(message_id):
        # The legacy pair lane (``bmsg_``) has its own ask/decide flow (DESIGN §1.2 rule 4).
        raise HoldRefused(409, "hold_not_releasable")
    store = Store(config_dir)
    now = time.time() if now is None else float(now)
    existing = store.owner_release(owner, message_id)
    if existing is not None:
        if existing["word"] != word:
            raise HoldRefused(409, "already_decided")
        return _receipt(existing)
    rows = _conversation_rows(store, owner)
    held = rows.get(message_id)
    decision = store.work_scope_decision(owner, message_id)
    if (
        held is None
        or held["hold_code"] not in hr.RELEASABLE_HOLD_CODES
        or decision is None
        or not decision.get("eligible")
        or not hr.trusted_decision(decision)
    ):
        # One code for "no hold", "not this kind" and "not eligible": the
        # renderer offers buttons only on ``releasable``, so a forced call gets
        # no detail about why (a closed allowlist, D1).
        raise HoldRefused(409, "hold_not_releasable")
    if not hr.enabled_for(config_dir, str(decision.get("hold_code") or "")):
        # The class's own setting, checked once the decision names its class.
        raise HoldRefused(409, "hold_release_off")
    if str(decision.get("conversation_id") or "") != conversation_id or (
        held["conversation_id"] and held["conversation_id"] != conversation_id
    ):
        raise HoldRefused(409, "snapshot_stale")
    if hold_receipt is not None:
        # Constant-time: the receipt is not a secret, but nothing here should
        # leak how much of a guessed binding matched.
        if not hmac.compare_digest(str(hold_receipt), discard_receipt(owner, message_id, decision)):
            raise HoldRefused(409, "snapshot_stale")
    elif str(decision["body_sha256"]) != str(body_sha256):
        # The person saw other bytes than the screener did (S-11).
        raise HoldRefused(409, "snapshot_stale")
    if word == "deliver" and now > float(decision["created_at"]) + hr.DELIVER_WINDOW_S:
        raise HoldRefused(409, "deliver_expired")
    if word == "deliver" and str(decision.get("hold_code") or "") in hr.DISCARD_ONLY_HOLD_CODES:
        # PR #4212 round 3 (S-13): every screener failed on THIS message while
        # others screened; nothing read it, so only Discard is offered.
        raise HoldRefused(409, "hold_not_releasable")
    message = hr.release_payload(
        owner_key=_owner_key(owner),
        generation=int(owner.enrollment_generation),
        conversation_id=conversation_id,
        message_id=message_id,
        body_sha256=str(decision["body_sha256"]),
        envelope_digest=str(decision["envelope_digest"]),
        hold_code=str(decision["hold_code"]),
        screener_provider=str(decision["screener_provider"]),
        screener_model=str(decision["screener_model"]),
        template_sha256=str(decision["template_sha256"]),
        categories=hr.decoded_categories(decision),
        screener_verdict=str(decision.get("screener_verdict") or ""),
        screener_hop=-1 if decision.get("screener_hop") is None else int(decision["screener_hop"]),
        screener_preferred=bool(decision.get("screener_preferred")),
        word=word,
        decided_at=now,
    )
    # THE LOCAL TERMINAL (follow-on to PR #4358). A receipt-bound Discard means
    # the renderer sent a hold receipt instead of a body hash: no body was
    # presented, so no hub confirmation of the record was obtained. That is the
    # ONLY signal. This server does not verify that the hub failed to serve the
    # body; any renderer holding the desktop bearer may choose the receipt
    # binding (#4371 security review S-1). That is safe because it is the
    # owner's own signed Discard, past the window, still honoured at the hub if
    # the envelope is ever served again. Past the EXISTING Deliver window
    # nothing else can ever settle such a row,
    # because the connector consumes a Discard only from an envelope the hub
    # serves. So that one case settles HERE, locally and atomically with an
    # audit row, instead of sitting ``pending`` forever. Deliberately narrow:
    # no new absence threshold (the decision time plus ``DELIVER_WINDOW_S`` is
    # the whole test), never a body-bound Discard (a presented body hash takes
    # the ordinary connector path), never inferred
    # from a missing record or a 404, and no hub call, delivery or sender
    # notice. Every owner/generation/decision/conversation check above runs
    # first, and the row is still signed like any Discard.
    deliver_until = float(decision["created_at"]) + hr.DELIVER_WINDOW_S
    settle_locally = word == "discard" and hold_receipt is not None and now > deliver_until
    try:
        signature, key_id = hr.sign_release(config_dir, message)
    except hr.ReleaseSigningUnavailable:
        raise HoldRefused(409, "release_signing_unavailable") from None
    release_row = {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "body_sha256": str(decision["body_sha256"]),
        "envelope_digest": str(decision["envelope_digest"]),
        "hold_code": str(decision["hold_code"]),
        "screener_provider": str(decision["screener_provider"]),
        "screener_model": str(decision["screener_model"]),
        "template_sha256": str(decision["template_sha256"]),
        "categories": str(decision["categories"]),
        "word": word,
        "decided_at": now,
        "account_id": account_id,
        "key_id": key_id,
        "signature": signature,
        "surface": "desktop",
        "screener_verdict": str(decision.get("screener_verdict") or ""),
        "screener_hop": (
            -1 if decision.get("screener_hop") is None else int(decision["screener_hop"])
        ),
        "screener_preferred": 1 if decision.get("screener_preferred") else 0,
    }
    try:
        if settle_locally:
            row = store.record_owner_release_local_terminal(
                owner,
                release_row,
                outcome=OWNER_RELEASE_LOCAL_TERMINAL,
                # A persisted audit word (``owner_release_local_terminals.reason``,
                # shipped in #4371), so it is kept as written. Read it as "no hub
                # confirmation was obtained", never "the hub was verified to have
                # failed" (S-1 above).
                reason="hub_unconfirmed_after_deliver_window",
                deliver_until=deliver_until,
                now=now,
            )
        else:
            row = store.record_owner_release(owner, release_row)
    except StoreRefused as failure:
        if failure.code == "already_decided":
            raise HoldRefused(409, "already_decided") from None
        raise HoldRefused(409, "state_unavailable") from None
    if settle_locally:
        # Closed words only, like every release line; nothing for the
        # connector to pick up, so the hold's backoff is left alone.
        hr.log_release(f"recorded word={word} outcome={OWNER_RELEASE_LOCAL_TERMINAL}")
        return _receipt(row)
    hr.log_release(f"recorded word={word}")
    # The next connector pass picks it up: make the hold due now rather than
    # after its doubling backoff (DESIGN §9.5).
    try:
        store.make_holds_due(owner, str(decision["hold_code"]), now=time.time())
    except StoreRefused:
        pass
    return _receipt(row)


def _evidence(
    store: Store,
    owner: Any,
    *,
    row: dict[str, Any],
    decision: dict[str, Any] | None,
    now: float,
    expiry_days: int,
) -> dict[str, Any]:
    """The corroborating facts an auditor needs, as booleans and numbers only.

    WHY: a disposition must not rest on intent alone. Every fact here was READ
    from a row, none is inferred, and the store refuses any value that is not a
    boolean or a number — so no body, sender, label or reason can reach the
    evidence column. ``allow_receipt`` is the ``screening_receipts`` concept
    itself (``verdict == 'allow'``, the same closed word the screener writes), and
    ``body_reservation`` is a real ``body_reservations`` row (``body_card_status``
    answers ``None`` when there is none).

    A read that fails records ``False`` rather than blocking the disposition: the
    row is being disposed of BECAUSE it cannot be acted on, so an unreadable
    supporting table is the expected state, not a stop sign. The fact stays
    honest — false means "not proven", never "proven absent".
    """

    deliver_until_passed = bool(
        decision is not None
        and now > float(decision.get("created_at") or 0.0) + hr.DELIVER_WINDOW_S
    )
    allow_receipt = False
    try:
        proof = store.content_proof_readonly(owner, [str(row["message_id"])])
        receipt = (proof.get(str(row["message_id"])) or {}).get("receipt") or {}
        allow_receipt = str(receipt.get("verdict") or "") == "allow"
    except Exception:  # noqa: BLE001 — evidence is a fact, never a gate
        allow_receipt = False
    body_reservation = False
    try:
        body_reservation = (
            store.body_card_status(owner, str(row["conversation_id"] or ""), str(row["message_id"]))
            is not None
        )
    except Exception:  # noqa: BLE001 — same law as above
        body_reservation = False
    return {
        "deliver_until_passed": deliver_until_passed,
        "allow_receipt": allow_receipt,
        "body_reservation": body_reservation,
        "hold_age_seconds": int(max(0.0, now - float(row["created_at"]))),
        "hold_expiry_days": int(expiry_days),
    }


def _notice_undelivered(
    store: Store,
    owner: Any,
    *,
    config_dir: Path,
    row: dict[str, Any],
    decision: dict[str, Any] | None,
    message_id: str,
) -> str:
    """At most ONE "not delivered" inbox item, and only with a RESOLVABLE sender.

    THREE GATES, all of them facts from rows:

    1. a ``deliveries`` row for this message carries both ``sender_account_id``
       and ``sender_endpoint_id`` (:meth:`Store.reply_target`). If it does not,
       nothing is sent and the state is ``none`` — never synthesise, guess or
       copy a sender from another row, and never add a name anywhere: the UI
       says "Unknown sender" and that is a renderer concern.
    2. a standing per-message notice was actually recorded (``notified_at``).
       The item this plane may retire is the one it raised; inventing a fresh
       lock-screen item for a message the person was never told about would be
       new noise, not a notice.
    3. a designated session exists to deliver into.

    CONTENT LAW, inherited verbatim from ``hold_release``'s per-message notice:
    no body, sender, conversation, category or label in the item, because the
    phone relay reads ``notifications.db`` and anything in it can reach a lock
    screen. It reuses the SHIPPED ``discarded`` copy at priority ``low`` — the
    same keyed update ``hold_release.clear_notice`` writes after a Discard — so
    no new translatable string is needed and the item reads as retired.

    Returns the closed ``notice_state`` to record: ``none`` | ``queued`` |
    ``sent``.
    """

    try:
        target = store.reply_target(owner, str(message_id))
    except Exception:  # noqa: BLE001 — no readable row is no resolvable sender
        return "none"
    if not target or not str(target.get("sender_account_id") or ""):
        return "none"
    if not str(target.get("sender_endpoint_id") or ""):
        return "none"
    if decision is None or decision.get("notified_at") is None:
        return "none"
    try:
        session_id = store.designated_session_for_readonly(owner)
    except Exception:  # noqa: BLE001
        session_id = None
    if not session_id:
        return "none"
    from local_operator.i18n.keys import cli_team_bridge_hold as _k
    from local_operator.i18n.text import tr
    from local_operator.notifications.store import NotificationStore

    t = tr("cli.team_bridge_hold")
    try:
        NotificationStore(config_dir).notify(
            session_id,
            title=t(_k.discarded_title()),
            body_md=t(_k.discarded_body()),
            priority="low",
            key=hr.notice_key(str(owner.endpoint_id), str(message_id)),
        )
    except Exception:  # noqa: BLE001 — chrome; the disposition already stands
        return "queued"
    try:
        # Frees the per-endpoint notice cap exactly as a Deliver/Discard does:
        # this item is retired, so it must stop counting as standing.
        store.mark_work_scope_cleared(owner, message_id, now=time.time())
    except Exception:  # noqa: BLE001 — the item is down; the cap entry is cosmetic
        pass
    return "sent"


def _dispose(
    store: Store,
    owner: Any,
    *,
    config_dir: Path,
    row: dict[str, Any],
    decision: dict[str, Any] | None,
    detail: str,
    now: float,
    expiry_days: int,
) -> dict[str, Any]:
    """Write ONE terminal marker (idempotent), then the at-most-one notice.

    No ``owner_releases`` row and no signature: a release is a signed word over a
    BODY (``hold_release.release_payload``) and this row has no loadable payload
    by definition, so inventing one would forge a record of a decision nobody
    made about bytes nobody saw. The marker is the local record; it grants
    nothing, and the connector is never told about it.
    """

    message_id = str(row["message_id"])
    evidence = _evidence(store, owner, row=row, decision=decision, now=now, expiry_days=expiry_days)
    terminal = store.record_hold_terminal(
        owner,
        message_id,
        row.get("conversation_id") or None,
        str(row["hold_code"]),
        reason="discarded_undeliverable",
        detail=detail,
        screener_verdict=str((decision or {}).get("screener_verdict") or ""),
        evidence=evidence,
        now=now,
    )
    if not terminal.get("already"):
        state = _notice_undelivered(
            store,
            owner,
            config_dir=config_dir,
            row=row,
            decision=decision,
            message_id=message_id,
        )
        if state != "none":
            store.mark_hold_terminal_notice(owner, message_id, notice_state=state)
            terminal["notice_state"] = state
    return terminal


def _conversation_decisions(store: Store, owner: Any) -> dict[str, dict[str, Any]]:
    """The stored work-scope decisions by bare id (the marker's own columns)."""

    return {str(d["message_id"]): d for d in store.work_scope_decisions(owner)}


def _disposal_candidates(
    config_dir: Path, owner: Any, *, store: Store, now: float
) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None, str]], int]:
    """Rows a bulk/age pass MAY dispose of, plus how many are already terminal.

    THE CLASSIFICATION IS :func:`list_holds`'S, deliberately: a second
    implementation of the same rules is how the badge and the pass would come to
    disagree. ``list_holds`` computes ``terminal_discard`` from the PRE-MAPPING
    hold row (see the trap in :data:`RETRYABLE_DELIVERY_CLASS`), so a
    ``conversation_failed`` row can never reach the list built here.

    The real hold row and the stored decision are read again for the columns the
    marker needs (the REAL hold code, the hold's own ``created_at``, the
    decision's verdict), never the served ones.
    """

    payload = list_holds(config_dir, owner, now=now)
    rows = _conversation_rows(store, owner)
    decisions = _conversation_decisions(store, owner)
    terminals = {str(t["message_id"]) for t in store.hold_terminals(owner)}
    already = len(terminals & set(rows))
    candidates: list[tuple[dict[str, Any], dict[str, Any] | None, str]] = []
    for item in payload["items"]:
        bare = str(item["message_id"])
        if not item.get("terminal_discard"):
            continue
        row = rows.get(bare)
        if row is None:
            continue
        candidates.append((row, decisions.get(bare), str(item.get("undeliverable_reason") or "")))
    return candidates, already


def discard_undeliverable_hold(
    config_dir: Path, owner: Any, *, message_id: str, now: float | None = None
) -> dict[str, Any]:
    """``POST …/holds/{id}/discard`` with NO proof, for a terminal-discardable row.

    Idempotent twice over: a repeat finds the marker and answers the SAME row
    (``already`` true, never ``already_decided`` — the stored word is ``discard``,
    so there is no second word to conflict with), and a message whose hold row has
    gone away answers "nothing left to dispose" rather than raising.

    ``hold_not_releasable`` (409) is the refusal for a row that is NOT provably
    undeliverable: identical code to the proof-bearing path, so a forced call
    learns nothing about why.
    """

    store = Store(config_dir)
    now = time.time() if now is None else float(now)
    expiry_days = hr.held_expiry_days(config_dir)
    existing = store.hold_terminal(owner, message_id)
    if existing is not None:
        return {
            "message_id": str(existing["message_id"]),
            "disposed": False,
            "already": True,
            "terminal": existing,
        }
    row = _conversation_rows(store, owner).get(message_id)
    if row is None:
        # Nothing left to dispose: the retry loop deleted the hold row (that is
        # how a hold ends when a delivery hands off). Not an error, and not a
        # marker written for a message this store no longer holds.
        return {"message_id": message_id, "disposed": False, "already": True, "terminal": None}
    decision = store.work_scope_decision(owner, message_id)
    reason = _terminal_reason(row, decision, now=now, expiry_days=expiry_days)
    if reason is None:
        raise HoldRefused(409, "hold_not_releasable")
    terminal = _dispose(
        store,
        owner,
        config_dir=config_dir,
        row=row,
        decision=decision,
        detail=reason,
        now=now,
        expiry_days=expiry_days,
    )
    return {
        "message_id": str(terminal["message_id"]),
        "disposed": not terminal.get("already"),
        "already": bool(terminal.get("already")),
        "terminal": terminal,
    }


def discard_all_undeliverable(
    config_dir: Path, owner: Any, *, now: float | None = None
) -> dict[str, int]:
    """Dispose of EVERY terminal-discardable row in one pass; closed counts.

    NEVER TOUCHES A DELIVERABLE ROW. ``list_holds`` already refuses to arm a row
    whose ``can_deliver`` is true, so a deliverable row is counted ``skipped``
    (it is still held — the person can act on it) and nothing is written for it.
    ``skipped`` is therefore "rows this pass left alone", not "errors".

    ``already`` counts markers standing over a hold row that is still in the
    store: the second call of the same batch reports ``discarded: 0``, which is
    what makes the batch a safe retry.
    """

    store = Store(config_dir)
    now = time.time() if now is None else float(now)
    expiry_days = hr.held_expiry_days(config_dir)
    candidates, already = _disposal_candidates(config_dir, owner, store=store, now=now)
    counts = {"discarded": 0, "already": already, "skipped": 0}
    candidate_ids = {str(row["message_id"]) for row, _, _ in candidates}
    for row, decision, reason in candidates:
        terminal = _dispose(
            store,
            owner,
            config_dir=config_dir,
            row=row,
            decision=decision,
            detail=reason,
            now=now,
            expiry_days=expiry_days,
        )
        if terminal.get("already"):
            counts["already"] += 1
        else:
            counts["discarded"] += 1
    for item in list_holds(config_dir, owner, now=now)["items"]:
        if str(item["message_id"]) not in candidate_ids:
            counts["skipped"] += 1
    return counts


def expire_undeliverable_holds(config_dir: Path, owner: Any, *, now: float | None = None) -> int:
    """Dispose of holds past ``team_bridge.held_expiry_days``; returns the count.

    THE BACKSTOP. ``discard_all_undeliverable`` settles what a person can see
    now; this bounds the table when nobody is looking, which is why it exists at
    all (an undeliverable row had no exit and could stand for ever).

    THREE GUARDS, each load-bearing:
    * ``held_expiry_days == 0`` disables the pass entirely;
    * a row in :data:`RETRYABLE_DELIVERY_CLASS` is never a candidate (the
      classification in ``list_holds`` already excludes it) — the delivery-repair
      work must be able to fix those rows;
    * a row whose ``can_deliver`` is true is never a candidate either.

    The AGE is the hold row's own ``created_at``, so a decision written days late
    cannot make an old hold look young. ``payload_expired`` is recorded as the
    detail even when the row would also qualify by the Deliver window, because
    the retention threshold is the authority THIS pass exercises — the evidence
    column still carries ``deliver_until_passed`` so an auditor sees both.

    BOUNDED and IDEMPOTENT: at most :data:`EXPIRY_MAX_PER_PASS` rows per call, and
    the store's insert-or-ignore makes a repeated pass over the same row a no-op
    (it reports ``already``, never a second write).
    """

    expiry_days = hr.held_expiry_days(config_dir)
    if expiry_days <= 0:
        return 0
    store = Store(config_dir)
    now = time.time() if now is None else float(now)
    threshold = now - float(expiry_days) * 86400.0
    candidates, _already = _disposal_candidates(config_dir, owner, store=store, now=now)
    disposed = 0
    for row, decision, reason in candidates:
        if disposed >= EXPIRY_MAX_PER_PASS:
            break
        if float(row["created_at"]) > threshold:
            continue
        terminal = _dispose(
            store,
            owner,
            config_dir=config_dir,
            row=row,
            decision=decision,
            detail="payload_expired",
            now=now,
            expiry_days=expiry_days,
        )
        if not terminal.get("already"):
            disposed += 1
    return disposed


def reopen_hold(config_dir: Path, owner: Any, *, message_id: str) -> bool:
    """Remove ONE terminal marker; ``True`` when one was removed.

    WHAT IT DOES NOT DO: it cannot undo a delivery that never happened, and it
    writes no release, no notice and no hub call. It only puts the hold back in
    front of the person — which is why it is the reversal verb and not a
    "restore" one.
    """

    return Store(config_dir).clear_hold_terminal(owner, message_id)


def _owner_key(owner: Any) -> str:
    from local_operator.team_bridge.store import _owner_key as key

    return key(owner)


def _receipt(row: dict[str, Any]) -> dict[str, Any]:
    word = str(row.get("word") or "")
    return {
        "message_id": row.get("message_id"),
        "state": "released" if word == "deliver" else "discarded",
        "decided_at": _iso(row.get("decided_at")),
        "key_id": row.get("key_id"),
        # A fingerprint of the signed record (never the signature itself), so a
        # renderer can tell two answers for the same message apart.
        "record_digest": hashlib.sha256(str(row.get("signature") or "").encode()).hexdigest()[:16],
        # Additive: what became of the record. ``pending`` means signed and
        # queued for the connector (not confirmed at the hub);
        # ``discarded_local_unconfirmed`` means settled on this machine only.
        "outcome": _outcome_word(row),
    }
