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
import time
from pathlib import Path
from typing import Any

from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge.screening import HOLD_CODES
from local_operator.team_bridge.store import (
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


class HoldRefused(Exception):
    """A closed refusal: ``code`` is the machine word, ``status`` the HTTP status."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


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
    items: list[dict[str, Any]] = []
    for bare, row in sorted(
        _conversation_rows(store, owner).items(), key=lambda kv: kv[1]["created_at"]
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
        )
        held_at = float(decision["created_at"]) if decision is not None else row["created_at"]
        deliver_until = held_at + hr.DELIVER_WINDOW_S if eligible else None
        release = store.owner_release(owner, bare)
        items.append(
            {
                "message_id": bare,
                "conversation_id": row["conversation_id"]
                or (str(decision.get("conversation_id") or "") if decision else ""),
                "seq": row["seq"],
                "held_at": _iso(held_at),
                "kind": _kind(code, eligible),
                # Closed codes only (``HOLD_CODES`` or ``unknown_hold``), like the read slice.
                "hold_code": code if code in HOLD_CODES else "unknown_hold",
                "releasable": eligible,
                "can_deliver": bool(
                    eligible and deliver_until is not None and now <= deliver_until
                ),
                "can_discard": eligible,
                "deliver_until": _iso(deliver_until),
                "decision": _decision_view(release),
            }
        )
    return {
        "enabled": enabled,
        "key_protection": hr.key_protection(config_dir),
        "items": items,
    }


def _decision_view(release: dict[str, Any] | None) -> dict[str, Any] | None:
    if release is None:
        return None
    outcome = str(release.get("outcome") or "pending")
    return {
        "word": release.get("word"),
        "decided_at": _iso(release.get("decided_at")),
        "outcome": outcome if outcome in ("pending", "admitted", "rejected") else "refused",
        "key_id": release.get("key_id"),
    }


def decide(
    config_dir: Path,
    owner: Any,
    *,
    message_id: str,
    conversation_id: str,
    body_sha256: str,
    word: str,
    account_id: str,
    now: float | None = None,
) -> dict[str, Any]:
    """``POST …/holds/{id}/deliver|discard``: sign and write ONE release row.

    Every check that can refuse runs BEFORE the signer, so a refused request
    never raises the operator prompt. Idempotent: the same word again returns
    the same row; the other word is ``already_decided``.
    """

    if word not in hr.WORDS:
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
    if str(decision["body_sha256"]) != str(body_sha256):
        # The person saw other bytes than the screener did (S-11).
        raise HoldRefused(409, "snapshot_stale")
    if word == "deliver" and now > float(decision["created_at"]) + hr.DELIVER_WINDOW_S:
        raise HoldRefused(409, "deliver_expired")
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
    try:
        signature, key_id = hr.sign_release(config_dir, message)
    except hr.ReleaseSigningUnavailable:
        raise HoldRefused(409, "release_signing_unavailable") from None
    try:
        row = store.record_owner_release(
            owner,
            {
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
            },
        )
    except StoreRefused as failure:
        if failure.code == "already_decided":
            raise HoldRefused(409, "already_decided") from None
        raise HoldRefused(409, "state_unavailable") from None
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
    }
