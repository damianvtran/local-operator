"""Conversation delivery to an agent member (the local half).

A conversation send fans out as one ``message_available`` feed row per active
agent member whose endpoint consent allows it. That row carries refs only:
``conversation_id``, ``message_id``, ``conversation_seq`` and the member's
``mention`` bit. This module turns ONE such signal into a card:

1. **Metadata only.** The feed plus ONE bounded history page (``after_seq``
   from the member's last read, never the pair inbox) gives the channel name,
   the sender's display label and dynamic kind, the mention state, the message
   id and its conversation seq. The card carries those fields and **never the
   body**; the body is never persisted anywhere.
2. **Screen, then admit.** Before any wake, the body is fetched through the
   endpoint-addressed classification endpoint, hashed against the server's
   binding hash, and run through the SAME H3 screening pipeline the pair lane
   uses. Only a matching ``allow`` receipt is admitted; a rejection is admitted
   as ``reject``. The admitted body becomes visible to the agent's own
   conversation read; a mutated body is never screened, admitted or rendered.
3. **Wake or steer.** Delivery reuses the shipped peer MAILBOX path: the local
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

from ..types import ConversationFeedRef
from .wake import CLAIM_TTL_S, _producer_fence, deliver_wake, screen_item

logger = logging.getLogger(__name__)

#: One bounded catch-up page. The server caps a conversation page at 100; 50
#: keeps a single delivery bounded while still covering an ordinary gap.
HISTORY_PAGE_LIMIT = 50

#: The served guidance version the card names. Triage rules (inform /
#: course-correct / ask, no new work, scope) live in that served text; the card
#: deliberately does not duplicate them. Bump WITH the served version (§11.3).
GUIDANCE_VERSION = "bridge-guidance-v2"

_SENDER_KINDS = frozenset({"human", "agent"})


@dataclass(frozen=True)
class ConversationCardMetadata:
    """What the card may say before (and without) a body — metadata only."""

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
    lines = [
        f"Channel message · {metadata.conversation_label} · "
        f"from {metadata.sender_label} ({metadata.sender_kind}, same tenant)",
        # ONE home for the untrusted clause: the shipped constant, verbatim.
        BRIDGE_UNTRUSTED_CLAUSE,
        "",
        mention_line,
        f"message_id: {metadata.message_id}",
        f"conversation_seq: {metadata.conversation_seq}",
        f"guidance: {GUIDANCE_VERSION}",
    ]
    return "\n".join(lines)


def _conversation_label(view: Mapping[str, Any]) -> str:
    """A human-facing conversation name; the kind decides the marker, not a guess."""

    kind = view.get("kind")
    raw_name = view.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) else ""
    if kind == "channel":
        return name if name.startswith("#") else f"#{name or 'channel'}"
    if kind == "dm":
        return "DM"
    if kind == "group":
        return f"group {name}" if name else "group"
    return name or "conversation"


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


def _find_message(page: Mapping[str, Any], ref: ConversationFeedRef) -> Mapping[str, Any] | None:
    """Locate THIS message in one bounded page; never fall back to the pair inbox."""

    messages = page.get("messages")
    if not isinstance(messages, list):
        return None
    for raw in messages:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("id") == ref.message_id:
            return raw
    return None


def _int_field(value: Any) -> int | None:
    """A non-negative integer or ``None`` — never a coerced bool/float."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


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
    ``delivered_metadata_only`` (denied or unverifiable body), ``hold``,
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

        client = wire if wire is not None else await _wire_for(owner, config_dir, auth_store)
        view = await client.conversation(
            ref.conversation_id, tenant_id=owner.tenant_id, access=access
        )
        if not isinstance(view, Mapping) or view.get("id") != ref.conversation_id:
            return {"action": "hold", "message_id": message_id, "code": "conversation_malformed"}
        last_read_seq = _int_field(view.get("last_read_seq", 0))
        if last_read_seq is None:
            return {"action": "hold", "message_id": message_id, "code": "conversation_malformed"}

        # ONE bounded page with ``after_seq``. Starting at ``seq - 1``
        # guarantees the target itself is in the page even when the member's
        # gap is larger than the page bound; for a one-message gap that is
        # exactly ``last_read_seq``. The pair inbox is never consulted for
        # conversation history.
        after_seq = max(last_read_seq, ref.conversation_seq - 1)
        page = await client.conversation_messages(
            ref.conversation_id,
            tenant_id=owner.tenant_id,
            limit=HISTORY_PAGE_LIMIT,
            after_seq=after_seq,
            endpoint_id=owner.endpoint_id,
            access=access,
        )
        if not isinstance(page, Mapping):
            return {"action": "hold", "message_id": message_id, "code": "history_malformed"}
        if not isinstance(page.get("messages"), list):
            return {"action": "hold", "message_id": message_id, "code": "history_malformed"}
        message = _find_message(page, ref)
        if message is None:
            # A signal for a message the history no longer carries (deleted or
            # never visible to this endpoint). Nothing to card; do not guess,
            # and retire any retry row — a later retry cannot conjure it.
            terminal = True
            return {"action": "skipped", "message_id": message_id}
        if _int_field(message.get("seq")) != ref.conversation_seq:
            return {"action": "hold", "message_id": message_id, "code": "seq_mismatch"}

        sender = message.get("sender")
        if not isinstance(sender, Mapping):
            return {"action": "hold", "message_id": message_id, "code": "sender_missing"}
        sender_label = sender.get("display_label")
        sender_kind = sender.get("kind")
        sender_account = sender.get("account_id")
        if (
            not isinstance(sender_label, str)
            or not sender_label.strip()
            or not isinstance(sender_kind, str)
            or sender_kind not in _SENDER_KINDS
            or not isinstance(sender_account, str)
            or not sender_account
        ):
            # A card cannot render provenance it cannot verify; a hardcoded
            # kind or an invented label is worse than a retryable hold.
            return {"action": "hold", "message_id": message_id, "code": "sender_malformed"}

        metadata = ConversationCardMetadata(
            conversation_label=_conversation_label(view),
            sender_label=" ".join(sender_label.split()),
            sender_kind=str(sender_kind),
            mention=ref.mention,
            message_id=message_id,
            conversation_seq=ref.conversation_seq,
        )

        fetched = await client.conversation_body(
            ref.conversation_id,
            message_id,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            access=access,
        )
        if not isinstance(fetched, Mapping) or fetched.get("message_id") != message_id:
            return {"action": "hold", "message_id": message_id, "code": "body_malformed"}
        body_text = fetched.get("body")
        body_hash = fetched.get("body_hash")
        if not isinstance(body_text, str) or not isinstance(body_hash, str):
            return {"action": "hold", "message_id": message_id, "code": "body_malformed"}
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
                "action": "delivered_metadata_only",
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
                "action": "delivered_metadata_only",
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
            metadata,
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
    "HISTORY_PAGE_LIMIT",
    "ConversationCardMetadata",
    "build_conversation_card",
    "handle_conversation_available",
]
