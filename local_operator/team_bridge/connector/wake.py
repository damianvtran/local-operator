"""The wake spine (H4c): ``message_available`` → screen → ack → wake.

ONE inbound ``message_available`` event for the OWN endpoint moves through
this exact order, and the order is the safety property (contract §13.4):

1. **Screen before ANY wake** — the H3 pipeline (fresh single-turn tool-less
   one-shot; ``deliver|hold|bounce``; 40/h budget; HOLD on exhaustion).
   Nothing reaches an agent session unscreened (§13.9.1).
2. ``deliver`` → **ack ``read`` with the verdict attached** (the receipt binds
   ``body_sha256``), then wake **from the same bytes that were screened** —
   never a re-fetch (§11.4/S-2). The bytes' SHA-256 is re-checked against the
   receipt here, and a mismatch refuses delivery (re-hold, S-20).
3. ``bounce`` → ack ``blocked`` (verdict attached); ``hold`` → local
   quarantine, never delivered, never a wake. Held/bounced items never wake a
   runtime (`WakeErrand` is never used for delivery anywhere in this module —
   it deliberately delivers nothing).
4. Delivery goes to the DESIGNATED CoS session only, and only for an
   unstarted session as the documented engagement-rule exception (§5.6): a
   missing designation is badge-only (zero wakes), and another unstarted
   session is refused by the primitive itself.

CLAIM DISCIPLINE. One message is woken into ≤1 session at a time: a native
claim (H2/H3 ``Store.claim``, fresh lease token) is taken around the handoff
and conditionally released in ``finally``. The claim guards the WAKE HANDOFF;
the CoS session's own ack transitions (``working``/``done``) carry their own
ledger upstream.

PRIVACY. Nothing here logs, returns or persists message text, the screening
reason, or any excerpt beyond the bounded §11.4 payload — the payload is the
ONLY place body bytes leave this module, and it is capped and
provenance-marked.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Mapping

from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE
from local_operator.team_bridge.types import AckTarget

#: The excerpt window §11.4 pins: at least 1200 chars of the screened body
#: reach the CoS (a shorter body is sent whole), and at most 2400 — the
#: client-side cap on what of a large body enters model context.
EXCERPT_MIN_CHARS = 1200
EXCERPT_MAX_CHARS = 2400

#: How long one wake handoff may hold a message claim. Screening + ack +
#: delivery are bounded by the screening deadline; this lease only needs to
#: outlive them, and an expired lease is replaceable by a NEW token only
#: (the store enforces that), so a crash here costs one retry, not a wedge.
CLAIM_TTL_S = 900.0


def parse_event_ref(ref: str) -> tuple[str, str, str] | None:
    """``"message_available:message_id=bmsg_x"`` → ``(type, kind, value)``.

    The projection in ``pull.project_events_batch`` is the only writer of
    these strings; this parser stays as strict as that writing (exactly one
    ``:`` and one ``=``, non-empty parts) so a future/malformed ref is
    skipped rather than guessed at.
    """

    if not isinstance(ref, str) or ref.count(":") != 1 or "=" not in ref:
        return None
    event_type, rest = ref.split(":", 1)
    kind, value = rest.split("=", 1)
    if not event_type or not kind or not value:
        return None
    return event_type, kind, value


def build_wake_payload(
    item: Mapping[str, Any],
    *,
    message_id: str,
    body_sha256: str,
) -> str:
    """Render the BOUNDED §11.4 payload — the only body bytes that leave.

    Envelope identity is the SERVER-RESOLVED sender name/org; a free-text
    ``label`` (when the server carried one at all) rides a clearly-marked
    untrusted display line with newlines collapsed, per §11.4. The excerpt is
    cut to :data:`EXCERPT_MAX_CHARS`; the message id and ``body_sha256`` ride
    the wire so the CoS can ack/reply and verify a later take-up fetch.

    Deliberately a plain-text block, not markdown: every surface renders
    bridge content as INERT text (§13.9.6), and this text is the payload of
    the dedicated :data:`TEAM_BRIDGE_MESSAGE_TYPE` wake.
    """

    message = item.get("message") or {}
    sender = item.get("sender") or {}
    thread = item.get("thread") or {}
    delivery = item.get("delivery") or {}
    body = message.get("body") or {}
    text = body.get("text")
    if not isinstance(text, str):
        text = ""
    excerpt = text[:EXCERPT_MAX_CHARS]
    resolved_name = sender.get("name") or sender.get("account_id") or "an organization member"
    organization = sender.get("organization") or ""
    lines = [
        f"[{TEAM_BRIDGE_MESSAGE_TYPE}]",
        f"message_id: {message_id}",
        f"sender: {resolved_name}" + (f" ({organization})" if organization else ""),
        f"thread: {thread.get('id') or message.get('thread_id') or 'unknown'}",
        f"type: {message.get('type') or 'note'}",
    ]
    label = sender.get("label")
    if isinstance(label, str) and label.strip():
        flattened = " ".join(label.split())
        lines.append(f"sender_display (untrusted, not identity): {flattened}")
    if delivery.get("state"):
        lines.append(f"state: {delivery.get('state')}")
    if message.get("request_state"):
        lines.append(f"request pending: {message.get('request_state')}")
    if len(text) > EXCERPT_MAX_CHARS:
        lines.append(f"excerpt ({EXCERPT_MAX_CHARS} of {len(text)} chars, screened):")
    else:
        lines.append("body (screened):")
    lines.append(excerpt)
    lines.append(f"body_sha256: {body_sha256}")
    return "\n".join(lines)


def _body_text(item: Mapping[str, Any]) -> str | None:
    body = (item.get("message") or {}).get("body") or {}
    text = body.get("text")
    return text if isinstance(text, str) else None


async def screen_item(
    owner: Any,
    item: Mapping[str, Any],
    *,
    config_dir: Path,
    auth_store: Any,
    settings: Any = None,
) -> Any:
    """SEAM: run the H3 screening pipeline for ONE item (lazy heavy import).

    ``scheduler`` resolves a provider selection and the runner stack — heavy
    modules the connector must not import at module scope — so the import
    lives here. ``None`` (no configured default) is returned as-is; the
    caller treats it as a hold (fail-closed), never as an implicit deliver.
    """

    from local_operator.team_bridge import scheduler as _scheduler

    return await _scheduler.run_screening_cycle(
        config_dir=config_dir,
        owner=owner,
        items=[item],
        auth_store=auth_store,
        settings=settings,
    )


async def fetch_item(
    owner: Any,
    message_id: str,
    *,
    config_dir: Path,
    access: Any,
    auth_store: Any,
) -> Mapping[str, Any]:
    """Fetch ONE message's item envelope (``{message, sender, thread, delivery}``).

    A separate read after ``pull.message``'s validation read: the refs-only
    projection deliberately drops ``message.body``, and screening needs the
    bytes. Two GETs per event are the honest cost of not persisting bodies
    client-side (§13.9.10); a later lane may fold validation+bytes into one
    projection without changing this seam's contract.
    """

    from local_operator.team_bridge.client import TeamBridgeClient

    wire = TeamBridgeClient(
        config_dir,
        base_url=owner.canonical_base_url,
        credential_id=owner.credential_id,
        store=auth_store,
    )
    result = await wire.message(message_id, tenant_id=owner.tenant_id, access=access)
    item = result.get("result", result) if isinstance(result, Mapping) else None
    if not isinstance(item, Mapping) or not isinstance(item.get("message"), Mapping):
        from local_operator.team_bridge.pull import PullRefused

        raise PullRefused("malformed_message", "message read returned no item envelope")
    inner = item.get("message") or {}
    if inner.get("id") != message_id:
        from local_operator.team_bridge.pull import PullRefused

        raise PullRefused("mismatched_id", "server answered about a different message")
    return item


async def ack_read(
    owner: Any,
    message_id: str,
    target: "AckTarget",
    receipt: Any,
    *,
    config_dir: Path,
    access: Any,
) -> Any:
    """SEAM: record + send ONE ack with the EXACT screening tuple (H2 path)."""

    from local_operator.team_bridge import pull as _pull

    return await _pull.ack(owner, message_id, target, receipt, config_dir=config_dir, access=access)


async def deliver_wake(
    owner: Any,
    payload: str,
    *,
    session_id: str,
    config_dir: Path,
) -> str:
    """SEAM: hand the wake to the designated CoS session (lazy heavy import).

    Live target: the shipped ``deliver_peer_message`` with ``wake=True``.
    Cold target (no runtime): ``engage_runtime`` with a ``PeerMessageErrand``
    — the errand that CARRIES the message — because ``WakeErrand`` delivers
    nothing by design. The designated session is the one engagement-rule
    exception; the primitive still refuses any other unstarted target.
    """

    from local_operator.mobile.peer_send import (
        deliver_peer_message_outcome,
        session_has_durable_history,
    )

    sender = {"account_id": owner.account_id, "source": TEAM_BRIDGE_MESSAGE_TYPE}
    if session_has_durable_history(session_id):
        # Live, or cold WITH durable history: the primitive owns delivery and
        # answers with the TYPED outcome — no exception-type or message-string
        # inspection (the unengaged distinction is a state, not an error).
        outcome = await deliver_peer_message_outcome(
            None,
            session_id=session_id,
            text=payload,
            mode="mailbox",
            wake=True,
            sender=sender,
        )
        return str(outcome.detail)
    # Cold with NO durable history: exactly the designated-CoS engagement
    # exception. The typed history predicate — never an error string — selects
    # this branch, and the errand CARRIES the message (WakeErrand never does).
    from local_operator.session.runtime.launch import (
        PeerMessageErrand,
        engage_runtime,
    )

    outcome = await engage_runtime(
        session_id,
        "",
        PeerMessageErrand(text=payload, mode="mailbox", wake=True, sender=sender),
        config_dir=config_dir,
    )
    return f"engaged {session_id} ({getattr(outcome, 'state', 'started')})"


async def handle_message_available(
    owner: Any,
    message_id: str,
    *,
    config_dir: Path,
    auth_store: Any,
    access: Any,
    settings: Any = None,
) -> dict[str, Any]:
    """The spine for ONE message id: screen → ack → wake (or hold/bounce).

    Returns a metadata-only summary — no body, no excerpt, no screening
    reason (the reason is the H3 result's private accessor and stays there).
    """

    from local_operator.team_bridge.store import Store, StoreRefused

    store = Store(config_dir)
    designation = store.designated_session_for(owner)
    if designation is None:
        # §5.6: no designation = arrivals do not wake — badge only.
        return {"action": "badge_only", "message_id": message_id}

    token = f"connector:{uuid.uuid4()}"
    import time as _time

    try:
        store.claim(owner, message_id, designation, token, expiry=_time.time() + CLAIM_TTL_S)
    except StoreRefused as failure:
        # A live claim elsewhere (or a stale-token edge) is NOT this run's
        # message: skip quietly — the other claimant owns the handoff.
        return {"action": "claimed_elsewhere", "message_id": message_id, "code": failure.code}

    try:
        from local_operator.team_bridge import pull as _pull

        ref = await _pull.message(owner, message_id, config_dir=config_dir, access=access)
        if ref.message_id != message_id:
            # The validation read must answer about THIS message; a mismatch
            # is a hold, never a screen of the wrong bytes.
            return {"action": "hold", "message_id": message_id, "code": "mismatched_id"}
        item = await fetch_item(
            owner, message_id, config_dir=config_dir, access=access, auth_store=auth_store
        )
        result = await screen_item(
            owner, item, config_dir=config_dir, auth_store=auth_store, settings=settings
        )
        if result is None:
            return {"action": "hold", "message_id": message_id, "code": "screening_unavailable"}
        disposition = getattr(result, "disposition", None)
        if disposition == "hold":
            return {
                "action": "hold",
                "message_id": message_id,
                "code": getattr(result, "hold_code", None),
            }
        receipt = getattr(result, "receipt", None)
        if disposition == "bounce":
            if receipt is None:
                # Fail CLOSED under a NAMED refusal: a bounce without its
                # verdict receipt cannot ack `blocked` with the frozen tuple,
                # and a silent no-op would strand the item with no local
                # reason. A hold keeps it visible and retryable (H4-A2).
                return {
                    "action": "hold",
                    "message_id": message_id,
                    "code": "bounce_receipt_missing",
                }
            await ack_read(
                owner, message_id, "blocked", receipt, config_dir=config_dir, access=access
            )
            return {"action": "bounce", "message_id": message_id}
        if disposition != "deliver" or receipt is None:
            return {"action": "hold", "message_id": message_id, "code": "screening_unknown"}
        text = _body_text(item)
        if text is None:
            return {"action": "hold", "message_id": message_id, "code": "body_unavailable"}
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != receipt.body_sha256:
            # S-20: never wake from bytes other than the screened ones.
            return {"action": "hold", "message_id": message_id, "code": "body_hash_mismatch"}
        payload = build_wake_payload(item, message_id=message_id, body_sha256=digest)
        await ack_read(owner, message_id, "read", receipt, config_dir=config_dir, access=access)
        outcome = await deliver_wake(owner, payload, session_id=designation, config_dir=config_dir)
        return {
            "action": "delivered",
            "message_id": message_id,
            "session": designation,
            "detail": str(outcome)[:120],
        }
    finally:
        try:
            store.release_claim(owner, message_id, token)
        except StoreRefused:
            # Not the live holder anymore (expiry + successor, or generation
            # fence): releasing is the successor's business, never ours.
            pass
