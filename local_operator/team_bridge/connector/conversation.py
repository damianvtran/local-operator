"""Conversation delivery to an agent member (the local half).

A conversation send fans out as one ``message_available`` feed row per active
agent member whose endpoint consent allows it. That row carries refs only:
``conversation_id``, ``message_id``, ``conversation_seq`` and the member's
``mention`` bit. This module turns ONE such signal into a card:

1. **Metadata only.** The feed subject carries the server-resolved display
   name, sender label and dynamic kind, so the card needs neither the
   person-gated conversation view nor a human read cursor. A legacy row
   without those fields falls back to a bounded target window; the pair inbox
   is never consulted. The card carries metadata and **never the body**; the
   body is never persisted anywhere.
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE

from ..attachments import (
    AttachmentDescriptor,
    AttachmentRejected,
    attachments_hash,
    parse_descriptors,
    verify_and_bound,
    withheld_marker,
)
from ..client import SCREENING_TEMPLATE_V3_SHA256, BridgeRefusal
from ..types import ConversationFeedRef
from .wake import CLAIM_TTL_S, _producer_fence, deliver_wake, screen_item

logger = logging.getLogger(__name__)

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
    #: The thread root when this message is a REPLY; ``None`` for a
    #: channel-level message. An id, never content.
    thread_root_id: str | None = None
    #: How many images the message carries — a count, never bytes or ids.
    attachments: int = 0
    #: The explicit refusal line when those images were NOT admitted (D5),
    #: so an agent is told images exist rather than silently missing them.
    attachments_note: str | None = None


def build_conversation_card(metadata: ConversationCardMetadata) -> str:
    """Render the METADATA-ONLY card.

    There is deliberately no body parameter: the card is a signal that a
    conversation message is (or is not) admitted, and the body reaches the
    agent only through its own conversation read after the admission gate.
    Keeping the type incapable of carrying a body is what keeps a future
    caller from accidentally rendering one.
    """

    mention_line = "mention: yes" if metadata.mention else "mention: no"
    label = _one_line(metadata.conversation_label) or "conversation"
    sender = _one_line(metadata.sender_label) or "member"
    kind = "Thread reply" if metadata.thread_root_id else "Channel message"
    lines = [
        f"{kind} · {label} · " f"from {sender} ({metadata.sender_kind}, same tenant)",
        # ONE home for the untrusted clause: the shipped constant, verbatim.
        BRIDGE_UNTRUSTED_CLAUSE,
        "",
        mention_line,
        f"message_id: {metadata.message_id}",
        f"conversation_seq: {metadata.conversation_seq}",
    ]
    if metadata.attachments:
        # A count and how to see them: images reach the agent only through
        # channel_history after an allow that screened them (never this card).
        lines.append(f"attachments: {metadata.attachments} image(s)")
        if metadata.attachments_note:
            lines.append(_one_line(metadata.attachments_note))
        else:
            lines.append("view: bridge channel_history returns the admitted images")
    if metadata.thread_root_id:
        # Thread context: where the reply lives and how to answer IN it, so an
        # agent coordinating in a thread does not answer at channel level.
        lines += [
            f"thread_root_id: {metadata.thread_root_id}",
            "reply in thread: bridge channel_post with this thread_root_id; "
            "read the thread with channel_thread",
        ]
    lines.append(f"guidance: {GUIDANCE_VERSION}")
    return "\n".join(lines)


def _one_line(value: Any, *, limit: int = 200) -> str:
    """Collapse every whitespace run (including CR/LF) and bound the result.

    A conversation name or sender label is server-resolved but not newline-free:
    allowing one through would inject lines ahead of the untrusted clause.
    """

    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


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


async def _target_message(
    client: Any,
    ref: ConversationFeedRef,
    owner: Any,
    access: Any,
) -> Mapping[str, Any] | None:
    """The message at THIS endpoint's cursor, fetched newest-first and bounded.

    ``before_seq = seq + 1`` makes the target the FIRST row when it exists: the
    server orders newest-first and returns ``limit`` rows, so the target cannot
    be pushed out by newer messages (the old ``after_seq`` window could be).
    The cursor is the feed's own ``conversation_seq`` for this endpoint, never
    the person-gated human read row.
    """

    page = await client.conversation_messages(
        ref.conversation_id,
        tenant_id=owner.tenant_id,
        limit=1,
        before_seq=ref.conversation_seq + 1,
        endpoint_id=owner.endpoint_id,
        access=access,
    )
    if not isinstance(page, Mapping) or not isinstance(page.get("messages"), list):
        return None
    message = _find_message(page, ref)
    if message is None:
        return None
    if _int_field(message.get("seq")) != ref.conversation_seq:
        return None
    return message


@dataclass(frozen=True)
class _Fetched:
    """The classification read: body, its hash, and the bound image set.

    ``attachments`` is ``None`` when the hub's descriptors were malformed
    (the caller holds) and ``()`` for a text-only message or an older hub.
    """

    body: str
    body_hash: str
    attachments: tuple[AttachmentDescriptor, ...] | None = ()
    attachments_hash: str | None = None


async def _body_from_history(
    client: Any,
    ref: ConversationFeedRef,
    owner: Any,
    access: Any,
) -> _Fetched | None:
    """Fallback for a server without the classification route (or a 404 route).

    Returns the screened bytes ONLY from the target window; an absent target, a
    null body (unadmitted) or a missing hash is ``None`` and the caller keeps
    the hold. It is never a reason to call the message gone.
    """

    message = await _target_message(client, ref, owner, access)
    if message is None:
        return None
    if message.get("attachment_count") or message.get("attachments"):
        # An image message must be screened through the classification route
        # (which carries the binding hash); a history row cannot prove the set.
        return None
    body = message.get("body")
    body_hash = message.get("body_hash")
    if not isinstance(body, str) or not isinstance(body_hash, str):
        return None
    # The legacy fallback never screens images: a history row reaching here is
    # from a hub without the classification route, which predates attachments.
    return _Fetched(body=body, body_hash=body_hash)


async def _fetch_body(
    client: Any,
    ref: ConversationFeedRef,
    owner: Any,
    access: Any,
) -> _Fetched | None:
    """The classification fetch, endpoint-addressed, with a legacy fallback.

    The primary read is ``GET .../messages/{message_id}/body`` — it addresses
    the message directly, so no history window can hide the target. History is
    consulted ONLY when that route is unavailable (an older server answering
    404/405); a transport/refusal of any other kind propagates to the handler's
    bounded hold.
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
        if failure.http_status in (404, 405) or failure.code in (
            "not_found",
            "method_not_allowed",
        ):
            return await _body_from_history(client, ref, owner, access)
        raise
    if not isinstance(fetched, Mapping) or fetched.get("message_id") != ref.message_id:
        return None
    body = fetched.get("body")
    body_hash = fetched.get("body_hash")
    if not isinstance(body, str) or not isinstance(body_hash, str):
        return None
    # Additive (image attachments): an older hub sends neither field and the
    # message is text-only exactly as before.
    served_attachments_hash = fetched.get("attachments_hash")
    return _Fetched(
        body=body,
        body_hash=body_hash,
        attachments=parse_descriptors(fetched.get("attachments")),
        attachments_hash=(
            served_attachments_hash if isinstance(served_attachments_hash, str) else None
        ),
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


_WITHHELD_REASONS = {
    # D5's exact wording for the vision case; the others name their cause the
    # same way so an agent is never left guessing why it cannot see an image.
    "unavailable": "[{n} image {noun} withheld: no longer available]",
    "unverified": "[{n} image {noun} withheld: could not be verified]",
}


def _withheld_note(count: int, reason: str) -> str:
    if reason == "no_vision":
        return withheld_marker(count)
    noun = "attachment" if count == 1 else "attachments"
    return _WITHHELD_REASONS[reason].format(n=count, noun=noun)


def _screener_sees_images(config_dir: Path) -> bool:
    """SEAM: whether the configured SCREENING model accepts image input (D5).

    Read from the same canonical selection the screening runner is built from,
    so the decision to send pixels and the model that receives them cannot
    disagree. Unknown/unusable selection reads as "no vision": fail closed.
    """

    from ..model_boundary import resolve_default_selection

    selection = resolve_default_selection(config_dir)
    if selection is None:
        return False
    return bool(getattr(selection.spec, "supports_images", False))


async def _screen_images(
    client: Any,
    ref: ConversationFeedRef,
    owner: Any,
    access: Any,
    descriptors: tuple[AttachmentDescriptor, ...],
) -> list[Any]:
    """Fetch each image with ``purpose=screen``, verify and bound it.

    These bytes exist ONLY to be classified: they ride the private carrier into
    the isolated screening dispatch and are dropped with it. They are never
    the bytes an agent sees — that is a separate ``purpose=context`` fetch
    after allow (manager decision D1) — and never written anywhere.
    Raises :class:`AttachmentRejected` (verification) or ``BridgeRefusal``
    (transport); the caller maps both to closed codes.
    """

    from ..screening import screening_image

    images = []
    for descriptor in descriptors:
        data, _mime = await client.fetch_attachment(
            ref.conversation_id,
            ref.message_id,
            descriptor.id,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            purpose="screen",
            access=access,
        )
        verified = await verify_and_bound(data, descriptor)
        images.append(
            screening_image(
                descriptor.mime,
                descriptor.width,
                descriptor.height,
                verified.private_payload(),
                verified.wire_mime,
            )
        )
    return images


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

        client = wire if wire is not None else await _wire_for(owner, config_dir, auth_store)
        # GAP CURSOR: this endpoint's own cursor is the feed row's
        # ``conversation_seq``. The person-gated view's ``last_read_seq`` is a
        # HUMAN cursor and is never read here; the body route addresses the
        # message by id, so a gap cannot hide it.
        if ref.has_card_metadata():
            # The NORMAL path for a conversation delivery. The feed carries the
            # server-resolved display provenance, so the person-gated
            # conversation view is never called: an account that owns an active
            # agent member but has not joined as a human can still wake.
            metadata = ConversationCardMetadata(
                conversation_label=ref.conversation_name or "conversation",
                sender_label=ref.sender_label or "",
                sender_kind=ref.sender_kind or "",
                mention=ref.mention,
                message_id=message_id,
                conversation_seq=ref.conversation_seq,
                thread_root_id=ref.thread_root_id,
            )
            sender: Mapping[str, Any] = {
                "account_id": "",
                "display_label": ref.sender_label,
            }
        else:
            # LEGACY/pair-shaped row (no card metadata): keep the old fallback
            # for a server that has not grown the additive fields. This path
            # may legitimately refuse for an agent-only membership; it is not
            # the delivery path for the new lane.
            try:
                view = await client.conversation(
                    ref.conversation_id, tenant_id=owner.tenant_id, access=access
                )
            except BridgeRefusal as failure:
                return {
                    "action": "hold",
                    "message_id": message_id,
                    "code": f"conversation_view_refused:{failure.code}",
                }
            if not isinstance(view, Mapping) or view.get("id") != ref.conversation_id:
                return {
                    "action": "hold",
                    "message_id": message_id,
                    "code": "conversation_malformed",
                }
            message = await _target_message(client, ref, owner, access)
            if message is None:
                # Absent from its own target window: keep the hold, never
                # declare it gone (a newer-first page can be wrong; this one
                # cannot hide the target behind later messages, but a
                # transient read still must retry).
                return {"action": "hold", "message_id": message_id, "code": "history_missing"}
            raw_sender = message.get("sender")
            if not isinstance(raw_sender, Mapping):
                return {"action": "hold", "message_id": message_id, "code": "sender_missing"}
            sender = dict(raw_sender)
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
                sender_label=sender_label,
                sender_kind=sender_kind,
                mention=ref.mention,
                message_id=message_id,
                conversation_seq=ref.conversation_seq,
                thread_root_id=ref.thread_root_id,
            )

        fetched = await _fetch_body(client, ref, owner, access)
        if fetched is None:
            # The target could not be read as a bound body (absent from the
            # fallback window, or null on the context path). Keep the hold so
            # the retry path can re-drive it.
            return {"action": "hold", "message_id": message_id, "code": "body_unavailable"}
        body_text, body_hash = fetched.body, fetched.body_hash
        if fetched.attachments is None:
            # Descriptors the hub sent but this build cannot validate: never
            # screen a guessed subset of the images; keep it retryable.
            return {"action": "hold", "message_id": message_id, "code": "attachments_malformed"}
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

        # IMAGES. Screened WITH the text only when every image is verified
        # and the screening model can see them; otherwise the text is
        # screened alone and the images stay withheld (no attachments_hash
        # is ever admitted for them), with an explicit line on the card.
        descriptors = fetched.attachments
        screened_images: Any = None
        attachments_note: str | None = None
        if descriptors:
            metadata = replace(metadata, attachments=len(descriptors))
            withheld_reason: str | None = None
            if not all(d.fetchable for d in descriptors):
                withheld_reason = "unavailable"
            elif fetched.attachments_hash != attachments_hash(descriptors):
                # The set the hub describes is not the set it bound: these
                # pixels can never be admitted (the image body_hash rule).
                withheld_reason = "unverified"
            elif not _screener_sees_images(config_dir):
                withheld_reason = "no_vision"
            else:
                try:
                    images = await _screen_images(client, ref, owner, access, descriptors)
                except AttachmentRejected:
                    withheld_reason = "unverified"
                else:
                    from ..screening import ScreeningAttachments

                    screened_images = ScreeningAttachments(
                        attachments_hash=str(fetched.attachments_hash),
                        images=tuple(images),
                    )
            if withheld_reason is not None:
                attachments_note = _withheld_note(len(descriptors), withheld_reason)
                metadata = replace(metadata, attachments_note=attachments_note)

        item = _screening_item(
            owner=owner,
            conversation_id=ref.conversation_id,
            message_id=message_id,
            body=body_text,
            body_hash=body_hash,
            sender=sender,
        )
        if screened_images is not None:
            item["attachments"] = screened_images
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
        admitted_images = (
            screened_images is not None
            and getattr(result, "attachments_hash", None) == screened_images.attachments_hash
        )
        if screened_images is not None and not admitted_images:
            # The verdict does not carry the image set we sent: something
            # between dispatch and here lost the binding. Never admit pixels
            # on it — hold and re-screen.
            return {"action": "hold", "message_id": message_id, "code": "attachments_unbound"}
        await client.conversation_admit(
            ref.conversation_id,
            message_id,
            verdict="allow",
            body_hash=body_hash,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            # The hub requires the v3 hash with an image allow: it is the only
            # template whose wrapper presented these images to the screener.
            attachments_hash=screened_images.attachments_hash if admitted_images else None,
            template_sha256=SCREENING_TEMPLATE_V3_SHA256 if admitted_images else None,
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
    "ConversationCardMetadata",
    "build_conversation_card",
    "handle_conversation_available",
]
