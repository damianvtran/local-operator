"""Conversation delivery to an agent member (the local half).

A conversation send fans out as one ``message_available`` feed row per active
agent member whose endpoint consent allows it. That row carries refs only:
``conversation_id``, ``message_id``, ``conversation_seq`` and the member's
``mention`` bit. This module turns ONE such signal into a card:

1. **Metadata only.** Durable feed/hold refs contain IDs, sequence, mention
   and a closed sender kind, never names or bodies. Display names come only
   from the endpoint-authorized body response and are rendered after matching
   screening/hash/admission. Older servers omit them: render IDs and unknown
   provenance, never consult a person-gated view. Cards include tenant and
   conversation locators so the agent can read without guessing a channel.
2. **Screen, then admit.** Before any wake, the body is fetched through the
   endpoint-addressed classification endpoint, hashed against the server's
   binding hash, and run through the SAME H3 screening pipeline the pair lane
   uses. Only a matching ``allow`` receipt is admitted; a rejection is admitted
   as ``reject``. The admitted body becomes visible to the agent's own
   conversation read; a mutated body is never screened, admitted or rendered.
3. **Mailbox wake.** Delivery reuses the shipped peer MAILBOX path: the local
   session designated for the receiving endpoint's owner is woken. A running
   turn is never steered by channel data, and no OS notification is emitted.

The card renders the shipped ``BRIDGE_UNTRUSTED_CLAUSE`` verbatim and names the
served guidance version (``bridge-guidance-v2``); the triage rules themselves
(inform / course-correct / ask, never new work) live in that served text and are
deliberately not duplicated in the card.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE

from ..client import BridgeRefusal
from ..types import ConversationFeedRef
from .wake import (
    CLAIM_TTL_S,
    _producer_fence,
    deliver_wake,
    engagement_hold,
    screen_item,
)

logger = logging.getLogger(__name__)

#: The served guidance version the card names. Triage rules (inform /
#: course-correct / ask, no new work, scope) live in that served text; the card
#: deliberately does not duplicate them. Bump WITH the served version (§11.3).
GUIDANCE_VERSION = "bridge-guidance-v2"


@dataclass(frozen=True)
class ConversationCardMetadata:
    """What the card may say before (and without) a body — metadata only."""

    conversation_id: str
    tenant_id: str
    conversation_label: str
    sender_label: str
    sender_kind: str
    mention: bool
    message_id: str
    conversation_seq: int


def build_conversation_card(metadata: ConversationCardMetadata) -> str:
    """Render the METADATA-ONLY card.

    There is deliberately no body parameter: the card is a signal that a
    conversation message is (or is not) admitted, and the body reaches the
    agent only through its own conversation read after the admission gate.
    Keeping the type incapable of carrying a body is what keeps a future
    caller from accidentally rendering one.
    """

    mention_line = "mention: yes" if metadata.mention else "mention: no"
    label = _one_line(metadata.conversation_label) or metadata.conversation_id
    sender = _one_line(metadata.sender_label) or "unknown"
    lines = [
        f"Channel message · {label} · " f"from {sender} ({metadata.sender_kind}, same tenant)",
        # ONE home for the untrusted clause: the shipped constant, verbatim.
        BRIDGE_UNTRUSTED_CLAUSE,
        "",
        mention_line,
        f"tenant_id: {metadata.tenant_id}",
        f"conversation_id: {metadata.conversation_id}",
        f"message_id: {metadata.message_id}",
        f"conversation_seq: {metadata.conversation_seq}",
        f"guidance: {GUIDANCE_VERSION}",
    ]
    return "\n".join(lines)


def _one_line(value: Any, *, limit: int = 200) -> str:
    """Collapse every whitespace run (including CR/LF) and bound the result.

    A conversation name or sender label is server-resolved but not newline-free:
    allowing one through would inject lines ahead of the untrusted clause.
    """

    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _screening_item(
    *,
    owner: Any,
    conversation_id: str,
    message_id: str,
    body: str,
    body_hash: str,
    sender: Mapping[str, Any],
) -> dict[str, Any]:
    """The H3 envelope for a conversation message.

    ``type=note`` is the conversation domain's ordinary message type; the
    screening pipeline's caps and grammar are the same ones the pair lane
    uses. The recipient binding is THIS endpoint, so a body fetched under a
    different identity can never produce an admission.
    """

    return {
        "message": {
            "id": message_id,
            "thread_id": conversation_id,
            "tenant_id": owner.tenant_id,
            "type": "note",
            "body": {"text": body},
            "body_hash": body_hash,
            "recipient": {
                "account_id": owner.account_id,
                "endpoint_id": owner.endpoint_id,
            },
        },
        "sender": {
            "account_id": str(sender.get("account_id") or ""),
            "name": str(sender.get("display_label") or ""),
        },
        "thread": {"id": conversation_id, "tenant_id": owner.tenant_id},
    }


async def _fetch_body(
    client: Any,
    ref: ConversationFeedRef,
    owner: Any,
    access: Any,
) -> tuple[str, str, Mapping[str, Any]] | None:
    """Fetch once through the endpoint/member/consent/joined-seq authority.

    A missing route on an old server is a retryable hold, never permission to
    fall back to a person-only conversation GET or to an unadmitted history
    body. Optional display fields travel only in this transient response.
    """

    try:
        fetched = await client.conversation_body(
            ref.conversation_id,
            ref.message_id,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            access=access,
        )
    except BridgeRefusal as failure:
        if failure.http_status in (404, 405):
            return None
        raise
    if not isinstance(fetched, Mapping) or fetched.get("message_id") != ref.message_id:
        return None
    body = fetched.get("body")
    body_hash = fetched.get("body_hash")
    if not isinstance(body, str) or not isinstance(body_hash, str):
        return None
    return body, body_hash, fetched


def _card_metadata(
    owner: Any, ref: ConversationFeedRef, display: Mapping[str, Any]
) -> ConversationCardMetadata:
    """An admitted body's ephemeral labels, or explicit unknown provenance.

    Empty/missing additive fields are an older server's supported answer, not
    a reason to invent a name or read a human membership route. Present values
    must satisfy the closed shape; arbitrary server keys are never carried.
    """

    labels = []
    for field in ("conversation_name", "sender_label", "sender_kind"):
        value = display.get(field, "")
        if not isinstance(value, str):
            raise ValueError("invalid conversation display field")
        labels.append(value)
    name, sender, kind = labels
    if kind not in ("", "human", "agent"):
        raise ValueError("invalid conversation sender kind")
    return ConversationCardMetadata(
        conversation_id=ref.conversation_id,
        tenant_id=owner.tenant_id,
        conversation_label=name,
        sender_label=sender,
        sender_kind=kind or "unknown",
        mention=ref.mention,
        message_id=ref.message_id,
        conversation_seq=ref.conversation_seq,
    )


async def _wire_for(owner: Any, config_dir: Path, auth_store: Any) -> Any:
    """SEAM: the production client, built lazily like the pair pull seams."""

    from ..client import TeamBridgeClient

    return TeamBridgeClient(
        config_dir,
        base_url=owner.canonical_base_url,
        credential_id=owner.credential_id,
        store=auth_store,
    )


async def _deliver_card(
    owner: Any,
    metadata: ConversationCardMetadata,
    *,
    session_id: str,
    config_dir: Path,
) -> str:
    """Hand the metadata card to the designated session (shipped mailbox wake)."""

    card = build_conversation_card(metadata)
    return await deliver_wake(
        owner,
        card,
        session_id=session_id,
        config_dir=config_dir,
        sender_display={"sender_name": metadata.sender_label},
    )


def _clear_terminal_hold(owner: Any, ref: ConversationFeedRef, config_dir: Path) -> None:
    """A terminal outcome retires the durable retry row keyed by this ref."""

    from ..store import Store, StoreRefused

    try:
        Store(config_dir).delete_hold(owner, ref.encode())
    except StoreRefused:
        pass  # fenced owner: the next live generation owns its own rows


async def handle_conversation_available(
    owner: Any,
    ref: ConversationFeedRef,
    *,
    config_dir: Path,
    auth_store: Any,
    access: Any,
    settings: Any = None,
    wire: Any = None,
) -> dict[str, Any]:
    """Screen one conversation message and deliver its card (or hold/skip).

    Returns metadata only. ``action`` is one of ``delivered`` (admitted body),
    ``delivered_unadmitted`` (denied or unverifiable body), ``hold``,
    ``badge_only``, ``claimed_elsewhere`` or ``skipped`` (the message is no
    longer in the conversation's history).
    """

    from ..store import Store, StoreRefused

    message_id = ref.message_id
    store = Store(config_dir)
    fence_code = _producer_fence(store, owner, access, settings)
    if fence_code == "no_designation":
        # Consent without a session is badge-only: arrivals never wake.
        return {"action": "badge_only", "message_id": message_id}
    if fence_code:
        return {"action": "hold", "message_id": message_id, "code": fence_code}

    designation = store.designated_session_for(owner)
    if designation is None:  # the fence above guarantees this, kept for typing
        return {"action": "badge_only", "message_id": message_id}

    token = f"connector:{uuid.uuid4()}"
    try:
        store.claim(owner, message_id, designation, token, expiry=time.time() + CLAIM_TTL_S)
    except StoreRefused as failure:
        return {
            "action": "claimed_elsewhere",
            "message_id": message_id,
            "code": failure.code,
        }

    terminal = False
    try:
        fence_code = _producer_fence(store, owner, access, settings)
        if fence_code:
            return {"action": "hold", "message_id": message_id, "code": fence_code}
        # F8: a never-used designated chief of staff whose ACCEPTED consent does
        # not disclose engagement holds visibly before any body fetch, screen or
        # admit; the loop re-drives it on the bounded backoff until the owner
        # sends the session a message (see ``engagement_hold``).
        engage_code = engagement_hold(designation, config_dir)
        if engage_code:
            return {"action": "hold", "message_id": message_id, "code": engage_code}

        client = wire if wire is not None else await _wire_for(owner, config_dir, auth_store)
        # A durable ref is only a locator. No person-membership read or legacy
        # cached name can supply display provenance; even failure cards remain
        # IDs-only until the endpoint-authorized bytes pass screen/hash/admit.
        metadata = _card_metadata(owner, ref, {})
        sender: Mapping[str, Any] = {}

        fetched = await _fetch_body(client, ref, owner, access)
        if fetched is None:
            # Missing/old classification route or an unbound response: keep
            # the hold rather than treating a weaker history read as authority.
            return {"action": "hold", "message_id": message_id, "code": "body_unavailable"}
        body_text, body_hash, display = fetched
        try:
            served_digest = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            return {"action": "hold", "message_id": message_id, "code": "body_invalid"}
        if served_digest != body_hash:
            # The bytes fetched are not the bytes the server bound. Never
            # screen them, never admit them. The metadata card still wakes —
            # a wake without an admission is a metadata card, not an
            # instruction — and the body stays unreadable on the context path.
            detail = await _deliver_card(
                owner,
                metadata,
                session_id=designation,
                config_dir=config_dir,
            )
            terminal = True
            return {
                "action": "delivered_unadmitted",
                "message_id": message_id,
                "code": "body_hash_mismatch",
                "session": designation,
                "detail": detail,
            }

        item = _screening_item(
            owner=owner,
            conversation_id=ref.conversation_id,
            message_id=message_id,
            body=body_text,
            body_hash=body_hash,
            sender=sender,
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
                "code": getattr(result, "hold_code", None) or "screening_hold",
            }
        receipt = getattr(result, "receipt", None)
        if disposition == "bounce":
            if receipt is None:
                return {
                    "action": "hold",
                    "message_id": message_id,
                    "code": "bounce_receipt_missing",
                }
            await client.conversation_admit(
                ref.conversation_id,
                message_id,
                verdict="reject",
                body_hash=body_hash,
                tenant_id=owner.tenant_id,
                endpoint_id=owner.endpoint_id,
                access=access,
            )
            detail = await _deliver_card(
                owner,
                metadata,
                session_id=designation,
                config_dir=config_dir,
            )
            terminal = True
            return {
                "action": "delivered_unadmitted",
                "message_id": message_id,
                "code": "screening_bounce",
                "session": designation,
                "detail": detail,
            }
        if disposition != "deliver" or receipt is None:
            return {"action": "hold", "message_id": message_id, "code": "screening_unknown"}
        if receipt.body_sha256 != served_digest:
            # S-20's rule, conversation half: never wake from bytes other than
            # the ones the screener saw.
            return {"action": "hold", "message_id": message_id, "code": "body_hash_mismatch"}
        try:
            admitted_metadata = _card_metadata(owner, ref, display)
        except ValueError:
            return {"action": "hold", "message_id": message_id, "code": "sender_malformed"}
        await client.conversation_admit(
            ref.conversation_id,
            message_id,
            verdict="allow",
            body_hash=body_hash,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            access=access,
        )
        detail = await _deliver_card(
            owner,
            admitted_metadata,
            session_id=designation,
            config_dir=config_dir,
        )
        terminal = True
        return {
            "action": "delivered",
            "message_id": message_id,
            "conversation_id": ref.conversation_id,
            "conversation_seq": ref.conversation_seq,
            "mention": ref.mention,
            "session": designation,
            "detail": detail,
        }
    except StoreRefused as failure:
        return {"action": "hold", "message_id": message_id, "code": failure.code}
    except Exception:  # noqa: BLE001 — a connector turn must not die on one item
        # The failure text is never surfaced: it can carry provider or wire
        # detail. The hold code keeps the item visible and retryable.
        logger.warning("bridge: conversation delivery held", exc_info=True)
        return {"action": "hold", "message_id": message_id, "code": "conversation_failed"}
    finally:
        if terminal:
            _clear_terminal_hold(owner, ref, config_dir)
        try:
            store.release_claim(owner, message_id, token)
        except StoreRefused:
            pass


__all__ = [
    "GUIDANCE_VERSION",
    "ConversationCardMetadata",
    "build_conversation_card",
    "handle_conversation_available",
]
