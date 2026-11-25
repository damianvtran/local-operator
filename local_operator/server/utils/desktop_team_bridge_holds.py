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
import re
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
    CONVERSATION_ID_PATTERN,
    is_conversation_message_id,
    parse_conversation_ref_value,
)

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

#: One desktop audit page, like the adjacent owner-audit lists (max 100).
#: Bounds signature checks and response size, not persistence or hold expiry.
_HISTORY_LIMIT = 100


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
        thread_root_id: str | None = None
        prefix = f"{CONVERSATION_FEED_TYPE}:{CONVERSATION_FEED_KIND}="
        if stored.startswith(prefix):
            ref = parse_conversation_ref_value(stored[len(prefix) :])
            if ref is not None:
                seq = ref.conversation_seq
                conversation_id = ref.conversation_id
                # ``validated()`` (inside the parser) already refused any
                # root that is not a ``bcmsg_`` id, so a damaged tag reads as
                # an unparsed ref here, never as a malformed root.
                thread_root_id = ref.thread_root_id
        prior = rows.get(bare)
        if prior is not None and float(prior["created_at"]) <= float(row["created_at"]):
            prior["seq"] = prior["seq"] if prior["seq"] is not None else seq
            prior["conversation_id"] = prior["conversation_id"] or conversation_id
            prior["thread_root_id"] = prior["thread_root_id"] or thread_root_id
            continue
        rows[bare] = {
            "message_id": bare,
            "conversation_id": conversation_id
            or (prior["conversation_id"] if prior is not None else ""),
            "seq": seq if seq is not None else (prior["seq"] if prior is not None else None),
            "thread_root_id": thread_root_id
            or (prior["thread_root_id"] if prior is not None else None),
            "hold_code": str(row["hold_code"]),
            "created_at": float(row["created_at"]),
        }
    return rows


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).isoformat()


def list_holds(config_dir: Path, owner: Any, *, now: float | None = None) -> dict[str, Any]:
    """``GET …/holds``: active conversation holds and recent signed decisions."""

    store = Store(config_dir)
    now = time.time() if now is None else float(now)
    # ``enabled`` is "any release path is open": the work-scope class
    # (``hold_release``) or the owner review of cached answer holds
    # (``review_held``, default on). Each item's own class is gated below.
    enabled = hr.any_enabled(config_dir)
    review_on = hr.review_enabled(config_dir)
    work_scope_on = hr.enabled(config_dir)
    decisions = {str(d["message_id"]): d for d in store.work_scope_decisions(owner)}
    items: list[dict[str, Any]] = []
    # Oldest hold first, the bare id breaking ties: two holds written in the
    # same instant must not swap places between refreshes (a list that reorders
    # under the person's pointer reads as rows jumping).
    for bare, row in sorted(
        _conversation_rows(store, owner).items(),
        key=lambda kv: (kv[1]["created_at"], kv[0]),
    ):
        code = row["hold_code"]
        decision = decisions.get(bare)
        eligible = bool(
            (review_on if code in hr.REVIEW_HOLD_CODES else work_scope_on)
            and code in hr.RELEASABLE_HOLD_CODES
            and decision is not None
            and str(decision.get("hold_code") or "") == code
            and decision.get("eligible")
            and hr.trusted_decision(decision)
            # A write-once conflict is offered only while its committed receipt
            # is still a screener allow a Deliver can commit over; re-read live
            # here as well as when its row was written (see ``owner_reviewable``).
            and hr.owner_reviewable(
                store, owner, bare, code, body_sha256=str(decision.get("body_sha256") or "")
            )
        )
        held_at = float(decision["created_at"]) if decision is not None else row["created_at"]
        deliver_until = held_at + hr.DELIVER_WINDOW_S if eligible else None
        release = store.owner_release(owner, bare)
        # EVERY eligible hold can be discarded (operator report 2026-11-24):
        # a write-once conflict's Discard settles locally in ``decide``
        # (``hold_release.LOCAL_DISCARD_HOLD_CODES``) instead of being refused.
        can_discard = eligible
        conversation_id = row["conversation_id"] or (
            str(decision.get("conversation_id") or "") if decision else ""
        )
        can_deliver = bool(
            eligible
            and deliver_until is not None
            and now <= deliver_until
            # Discard-only rows: S-13's code, and a conflict whose current
            # pass rejected (#4546 S-2; ``hold_release.deliver_refused``).
            and decision is not None
            and not hr.deliver_refused(decision)
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
        items.append(
            {
                "message_id": bare,
                "conversation_id": conversation_id,
                "seq": row["seq"],
                # The root a thread REPLY answers (``bcmsg_…``), or ``None`` for a
                # channel-level message or a hold whose ref did not carry one.
                # WHY (operator report 2026-11-24): the renderer joins sender and
                # body from the owner-audit messages page, and the hub excludes
                # thread replies from that page (agent-server
                # bridge_conversation_repository.go), so a held reply always
                # read "Unknown sender … could not be loaded". With the root the
                # renderer can read the reply from its thread instead.
                "thread_root_id": row["thread_root_id"],
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
            }
        )
    return {
        "enabled": enabled,
        "key_protection": hr.key_protection(config_dir),
        "items": items,
        "history": _hold_history(
            store,
            owner,
            live_decision_ids=tuple(item["message_id"] for item in items if item["decision"]),
        ),
    }


def _hold_history(
    store: Store, owner: Any, *, live_decision_ids: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Project the signed decision, never another delivery or discard capability.

    The signature binds the owner, conversation, message and decision, not the
    connector's outcome; the latter is a closed local-store status, not proof
    of hub delivery. The release does not bind a thread, sequence or original
    hold time. Do not reconstruct those from unverified refs or guess them.
    """

    releases = store.owner_release_history(
        owner, limit=_HISTORY_LIMIT, live_decision_ids=live_decision_ids
    )
    if not releases:
        return []
    spki, _ = hr.anchor_spki(store.config_dir)
    if spki is None:
        return []
    items: list[dict[str, Any]] = []
    for release in releases:
        bare = str(release["message_id"])
        conversation_id = str(release["conversation_id"])
        code = str(release["hold_code"])
        word = release["word"]
        outcome = _outcome_word(release)
        if (
            not is_conversation_message_id(bare)
            or re.fullmatch(CONVERSATION_ID_PATTERN, conversation_id) is None
            or code not in hr.RELEASABLE_HOLD_CODES
            # Either click can still be pending or be refused. Only a matching
            # signed word may be paired with a successful terminal outcome.
            or word not in ("deliver", "discard")
            or (outcome == "admitted" and word != "deliver")
            or (outcome in ("rejected", OWNER_RELEASE_LOCAL_TERMINAL) and word != "discard")
        ):
            continue
        try:
            if not hr.verify_row(release, spki):
                continue
            decided_at = _iso(release["decided_at"])
        except (ValueError, TypeError, OverflowError, OSError):
            # Damaged rows have no projection, even when other history is valid.
            continue
        items.append(
            {
                "message_id": bare,
                "conversation_id": conversation_id,
                "seq": None,
                "thread_root_id": None,
                "held_at": None,
                "first_held_at": None,
                "kind": "owner_decision",
                "hold_code": code,
                "releasable": False,
                "can_deliver": False,
                "can_discard": False,
                "actionable": False,
                "unscreened": not release["screener_model"] and not release["screener_verdict"],
                "deliver_until": None,
                "decision": {
                    "word": word,
                    "decided_at": decided_at,
                    "outcome": outcome,
                },
            }
        )
    return items


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
    if word == "deliver" and hr.deliver_refused(decision):
        # PR #4212 round 3 (S-13): every screener failed on THIS message while
        # others screened; nothing read it, so only Discard is offered. #4546
        # S-2: a conflict whose current pass rejected is Discard-only too.
        raise HoldRefused(409, "hold_not_releasable")
    if not hr.owner_reviewable(
        store,
        owner,
        message_id,
        str(decision.get("hold_code") or ""),
        body_sha256=str(decision["body_sha256"]),
    ):
        # The committed receipt is no longer one a Deliver can commit over
        # (a reject, an owner-release head, damage): nothing to sign.
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
    #
    # The second local terminal: a Discard on a ``LOCAL_DISCARD_HOLD_CODES``
    # hold (a write-once conflict), in or out of the window and whatever its
    # binding. The connector could never apply it (write-once refuses a reject
    # over the committed allow), so queueing it would be a word nothing
    # settles; it is recorded under its own audit ``reason``.
    deliver_until = float(decision["created_at"]) + hr.DELIVER_WINDOW_S
    local_code = word == "discard" and str(decision["hold_code"]) in hr.LOCAL_DISCARD_HOLD_CODES
    settle_locally = word == "discard" and (
        local_code or (hold_receipt is not None and now > deliver_until)
    )
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
                reason=(
                    "write_once_conflict_committed_allow"
                    if local_code
                    else "hub_unconfirmed_after_deliver_window"
                ),
                deliver_until=deliver_until,
                now=now,
            )
        else:
            row = store.record_owner_release(owner, release_row)
    except StoreRefused as failure:
        if failure.code == "already_decided":
            raise HoldRefused(409, "already_decided") from None
        if failure.code == "snapshot_stale":
            # #4570 security S-1: a re-drive replaced the decision row while
            # this word was being signed; the renderer re-reads it.
            raise HoldRefused(409, "snapshot_stale") from None
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
