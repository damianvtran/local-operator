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
    #: The root this message answers, when it is a thread reply. Carried from
    #: the durable ref's tagged slot; `None` for a channel-level message.
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
    label = _one_line(metadata.conversation_label) or metadata.conversation_id
    sender = _one_line(metadata.sender_label) or "unknown"
    kind = "Thread reply" if metadata.thread_root_id else "Channel message"
    lines = [
        f"{kind} · {label} · " f"from {sender} ({metadata.sender_kind}, same tenant)",
        # ONE home for the untrusted clause: the shipped constant, verbatim.
        BRIDGE_UNTRUSTED_CLAUSE,
        "",
        mention_line,
        f"tenant_id: {metadata.tenant_id}",
        f"conversation_id: {metadata.conversation_id}",
        f"message_id: {metadata.message_id}",
        f"conversation_seq: {metadata.conversation_seq}",
    ]
    if metadata.attachments:
        # A count and how to see them: images reach the agent only through
        # channel_history after an allow that screened them (never this card).
        lines.append(f"attachments: {metadata.attachments} image(s)")
        if metadata.attachments_note:
            lines.append(_one_line(metadata.attachments_note))
        elif metadata.thread_root_id:
            # The channel timeline EXCLUDES thread replies (hub threads), so
            # a reply's images are reachable only through its thread (R1-02).
            lines.append("view: bridge channel_thread with this thread_root_id returns the images")
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
    #: The transient classification response: the ONLY carrier of main's
    #: additive display fields. Never stored — the card builder reads it in
    #: this call and drops it.
    display: Mapping[str, Any]
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
    return _Fetched(body=body, body_hash=body_hash, display=message)


async def _fetch_body(
    client: Any,
    ref: ConversationFeedRef,
    owner: Any,
    access: Any,
) -> _Fetched | None:
    """The classification fetch, endpoint-addressed, with a legacy fallback.
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
    # Additive (image attachments): an older hub sends neither field and the
    # message is text-only exactly as before. ``display`` carries main's
    # transient label mapping alongside the image binding: both are facts of
    # the SAME authorized read, so one carrier keeps them from drifting apart.
    served_attachments_hash = fetched.get("attachments_hash")
    return _Fetched(
        body=body,
        body_hash=body_hash,
        display=fetched,
        attachments=parse_descriptors(fetched.get("attachments")),
        attachments_hash=(
            served_attachments_hash if isinstance(served_attachments_hash, str) else None
        ),
    )


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
        thread_root_id=ref.thread_root_id,
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


#: Closed outcomes of the reservation emit. A string constant rather than a
#: bool so the caller can name WHY it held, and so the loop's backoff row
#: carries a code an operator can act on.
RESERVATION_RECORDED = "recorded"

#: A replay of an admission whose reservation already exists. NOT an error:
#: the hub re-delivers after a crash, and the second pass must consume the
#: message WITHOUT posting a second card.
RESERVATION_ALREADY_RECORDED = "already_recorded"


def _screening_fingerprint(owner: Any, receipt_row: Mapping[str, Any]) -> str:
    """The digest hydration recomputes, over the SAME tuple the store keys on.

    WHY the OWNER is in the pre-image: ``consume_screening_receipt`` scopes its
    read by owner namespace first (store.py:3970-4010), so a digest that omits
    it would match a different owner's identical receipt fields — every
    reservation would then fail its admission check AFTER the card was already
    posted, and the body would never hydrate. Two owners with byte-identical
    receipts must therefore produce different fingerprints, which is exactly
    what the test asserts.

    WHY the other five: ``body_sha256`` alone is not enough (the same bytes
    under a new template/model snapshot a different admission), so the
    fingerprint carries the policy identity with it.
    """

    from local_operator.team_bridge.store import _owner_key

    return hashlib.sha256(
        "\x1f".join(
            [_owner_key(owner)]
            + [
                str(receipt_row.get(field) or "")
                for field in (
                    "body_sha256",
                    "template_sha256",
                    "model",
                    "admission_revision",
                    "envelope_digest",
                )
            ]
        ).encode("utf-8")
    ).hexdigest()


def _record_body_reservation(
    store: Any,
    owner: Any,
    ref: ConversationFeedRef,
    *,
    body_hash: str,
    designation: str,
) -> str:
    """Record the CONTENTLESS reservation for one admitted message.

    Called only after the hub accepted the message, and before any card: a card
    without a reservation would wake the agent for a body nothing will deliver.

    The arguments are named scalars on purpose. The body text and the display
    labels are still in the CALLER's scope (screening needed them), and the
    guarantee that they cannot reach the row is threefold: they are never
    passed here, the table has no column for them, and the store validates
    every written value against a closed grammar.

    Returns ``RESERVATION_RECORDED`` or a closed hold code; the caller must not
    post a card on a non-recorded answer, and must leave the message retryable.
    """

    receipt_row = store.receipt_for_message(owner, ref.message_id)
    if receipt_row is None:
        # No trusted receipt means the producer epoch never wrote one; the
        # admission cannot be re-proved later, so the message stays pending.
        return "receipt_unavailable"
    if str(receipt_row.get("body_sha256") or "") != body_hash:
        return "body_hash_mismatch"
    consent = store.owner_consent_receipt(owner) or {}
    try:
        created = store.record_body_reservation(
            owner,
            conversation_id=ref.conversation_id,
            message_id=ref.message_id,
            conversation_seq=ref.conversation_seq,
            mention=ref.mention,
            body_sha256=body_hash,
            screening_fingerprint=_screening_fingerprint(owner, receipt_row),
            policy_epoch=int(receipt_row.get("policy_epoch") or 0),
            consent_version=consent.get("consent_version"),
            designated_session=designation,
            now=time.time(),
        )
    except Exception:  # noqa: BLE001 — closed structural code, never content
        logger.warning("bridge: body reservation refused code=store_refused")
        return "reservation_refused"
    # STRUCTURAL ONLY: whether this call created the row is the one fact worth
    # logging — a duplicate emit is normal after a crash, and no id or hash
    # belongs in a log line about it.
    logger.info("bridge: body reservation created=%s", created)
    return RESERVATION_RECORDED if created else RESERVATION_ALREADY_RECORDED


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


async def _images_admitted(
    client: Any,
    ref: ConversationFeedRef,
    owner: Any,
    access: Any,
    descriptors: tuple[AttachmentDescriptor, ...],
) -> bool:
    """Whether the hub already admitted THIS image set for this endpoint.

    WHY (review R1-01): after an image allow, the hub closes ``purpose=screen``
    for the (message, endpoint) for good (decision D1). A retry that re-ran the
    screen fetch after a failure between admit and card — a lost admit reply, a
    crash, a refused wake — got that 404 on every attempt and the card never
    went out. The hub's admission is the authority, so it is asked directly: a
    ``purpose=context`` read succeeds only under an allow bound to the current
    ``attachments_hash``. The bytes are discarded unread — the probe proves
    admission, it never feeds a model — and only a 404 means "not admitted";
    any other refusal propagates so the handler keeps its bounded hold.
    """

    try:
        await client.fetch_attachment(
            ref.conversation_id,
            ref.message_id,
            descriptors[0].id,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            purpose="context",
            access=access,
        )
    except BridgeRefusal as failure:
        if _is_not_found(failure):
            return False
        raise
    return True


def _is_not_found(failure: BridgeRefusal) -> bool:
    """THE one "the hub has no such fetchable attachment" predicate (R2-05).

    D1 answers every miss with a uniform 404 ``attachment_not_found``; the
    code is checked too so a proxy that rewrites the status cannot turn a
    closed route into a retryable failure in one branch and not the other.
    """

    return failure.http_status == 404 or failure.code == "attachment_not_found"


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
    ``rejected`` (a confirmed moderation refusal: terminal, and it posts NO card
    and wakes nobody), ``hold``, ``badge_only``, ``claimed_elsewhere`` or
    ``skipped`` (the message is no longer in the conversation's history).
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
        # cached name can supply display provenance. Failed admission must not
        # post even a metadata-only card: that would still wake the recipient.
        sender: Mapping[str, Any] = {}

        fetched = await _fetch_body(client, ref, owner, access)
        if fetched is None:
            # Missing/old classification route or an unbound response: keep
            # the hold rather than treating a weaker history read as authority.
            return {"action": "hold", "message_id": message_id, "code": "body_unavailable"}
        body_text, body_hash, display = fetched.body, fetched.body_hash, fetched.display
        if fetched.attachments is None:
            # Descriptors the hub sent but this build cannot validate: never
            # screen a guessed subset of the images; keep it retryable.
            return {"action": "hold", "message_id": message_id, "code": "attachments_malformed"}
        # Display labels are EPHEMERAL: they come from this authorized read and
        # are dropped with it (#2856 keeps them out of the durable ref). Built
        # here, once, because the image block annotates the same object.
        try:
            metadata = _card_metadata(owner, ref, display)
        except ValueError:
            return {"action": "hold", "message_id": message_id, "code": "sender_malformed"}
        try:
            served_digest = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            return {"action": "hold", "message_id": message_id, "code": "body_invalid"}
        if served_digest != body_hash:
            # Neither a bad body nor an unscreened metadata notification may
            # spend a recipient turn. Keep the exact locator pending so a later
            # authorized fetch can recover; this is not terminal consumption.
            return {
                "action": "hold",
                "message_id": message_id,
                "code": "body_hash_mismatch",
            }

        # IMAGES. Screened WITH the text only when every image is verified
        # and the screening model can see them; otherwise the text is
        # screened alone and the images stay withheld (no attachments_hash
        # is ever admitted for them), with an explicit line on the card.
        descriptors = fetched.attachments
        screened_images: Any = None
        attachments_note: str | None = None
        #: Set when a retry finds the verdict already concluded (R1-01):
        #: ``images_admitted`` / ``text_admitted`` deliver the card,
        #: ``rejected`` (our receipt says reject) and ``unadmitted`` (no local
        #: receipt, hub grants no images) deliver the unadmitted card — none
        #: re-screens.
        recovery: str | None = None
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
                # RETRY AFTER A VERDICT (review R1-01). D1 closes the screen
                # route for good once this endpoint has ANY terminal verdict,
                # so a retry after a failure between admit and card (a lost
                # admit reply, a crash, a refused wake) must never depend on
                # re-screening. The hub's admission and this machine's own
                # receipt for these exact body bytes decide what was already
                # concluded; only a message with neither is screened.
                prior = store.receipt_for_message(owner, message_id)
                if prior is not None and prior.get("body_sha256") != served_digest:
                    prior = None
                if prior is not None and prior.get("verdict") == "allow":
                    if await _images_admitted(client, ref, owner, access, descriptors):
                        recovery = "images_admitted"
                if recovery is None:
                    try:
                        images = await _screen_images(client, ref, owner, access, descriptors)
                    except AttachmentRejected:
                        withheld_reason = "unverified"
                    except BridgeRefusal as failure:
                        if not _is_not_found(failure):
                            raise
                        # Screen closed. That USUALLY means a terminal verdict
                        # for this endpoint (D1), but the hub answers the same
                        # uniform not-found when no verdict exists at all — an
                        # unbound row, missing blob, or a message before this
                        # member joined (review R3-02). So it is not proof of
                        # a verdict, only that these pixels cannot be screened.
                        if prior is None:
                            # Nothing local to trust (e.g. the store was reset
                            # after a verdict, or the hub cannot serve the
                            # bytes). The hub's context predicate is the only
                            # authority for the images. When it is closed too,
                            # the card goes out METADATA-ONLY with an explicit
                            # withheld line (R2-04, R3-01): no body and no
                            # image, the hub still gates both. The text is
                            # deliberately NOT screened alone here: if the hub
                            # already holds a reject or an image allow, a fresh
                            # text-only admit is a verdict conflict and would
                            # bring back the hold-until-dropped R2-04 removed;
                            # and with no verdict at all, a body-less wake is
                            # the fail-closed outcome (R3-02).
                            if await _images_admitted(client, ref, owner, access, descriptors):
                                recovery = "images_admitted"
                            else:
                                recovery = "unadmitted"
                                withheld_reason = "unverified"
                        elif prior.get("verdict") == "reject":
                            recovery = "rejected"
                        else:
                            # An allow the hub bound text-only: the images
                            # stay withheld, the card still goes out.
                            recovery = "text_admitted"
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

        if recovery is not None:
            detail = await _deliver_card(
                owner, metadata, session_id=designation, config_dir=config_dir
            )
            terminal = True
            if recovery in ("rejected", "unadmitted"):
                return {
                    "action": "delivered_unadmitted",
                    "message_id": message_id,
                    "code": (
                        "screening_bounce" if recovery == "rejected" else "admission_unverified"
                    ),
                    "session": designation,
                    "detail": detail,
                }
            return {
                "action": "delivered",
                "message_id": message_id,
                "conversation_id": ref.conversation_id,
                "conversation_seq": ref.conversation_seq,
                "mention": ref.mention,
                "session": designation,
                "detail": detail,
                "recovered": recovery,
            }

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
            # A confirmed reject retires the notification, not the moderation
            # boundary. Posting a metadata-only card here would still wake the
            # CoS for rejected input, even without exposing any body bytes.
            terminal = True
            return {
                "action": "rejected",
                "message_id": message_id,
                "code": "screening_bounce",
            }
        if disposition != "deliver" or receipt is None:
            return {"action": "hold", "message_id": message_id, "code": "screening_unknown"}
        if receipt.body_sha256 != served_digest:
            # S-20's rule, conversation half: never wake from bytes other than
            # the ones the screener saw.
            return {"action": "hold", "message_id": message_id, "code": "body_hash_mismatch"}
        admitted_hash: str | None = None
        if screened_images is not None:
            if getattr(result, "attachments_hash", None) == screened_images.attachments_hash:
                admitted_hash = screened_images.attachments_hash
        if screened_images is not None and admitted_hash is None:
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
            attachments_hash=admitted_hash,
            template_sha256=SCREENING_TEMPLATE_V3_SHA256 if admitted_hash else None,
            access=access,
        )
        # THE ADMISSION POINT (seam design, Revision 5): the hub has accepted
        # this message for this endpoint/member/consent tuple, so one
        # designated session now owes it a body. Recorded BEFORE the card —
        # a card with no reservation wakes an agent for a body that will never
        # arrive — and a failure here is a RETRYABLE hold, never a delivery.
        recorded = _record_body_reservation(
            store,
            owner,
            ref,
            body_hash=body_hash,
            designation=designation,
        )
        if recorded == RESERVATION_ALREADY_RECORDED:
            # A replay (crash-after-record, or a hub re-delivery) must not post
            # a SECOND card: the row is the dedupe witness, and a duplicate
            # wake costs the agent a turn for a message it already answered.
            # RESIDUAL, stated rather than hidden: a crash in the window
            # between recording and the card leaves a pending row with no
            # wake; the row survives and stays claimable, and closing that
            # window needs a card-delivery marker the claiming slice owns.
            return {
                "action": "duplicate",
                "message_id": message_id,
                "conversation_id": ref.conversation_id,
                "reservation": "already_recorded",
            }
        if recorded != RESERVATION_RECORDED:
            return {"action": "hold", "message_id": message_id, "code": recorded}
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
            # Which template screened (R1-07): a version string and a public
            # template hash — never content.
            "screened_template": getattr(result, "rendered_version", None),
            "screened_template_sha256": getattr(result, "rendered_template_sha256", None),
        }
    except StoreRefused as failure:
        return {"action": "hold", "message_id": message_id, "code": failure.code}
    except Exception:  # noqa: BLE001 — a connector turn must not die on one item
        # This is the first connector error sink: an exception or traceback
        # can retain fetched body/provider echoes even when its message is not
        # explicitly interpolated. Emit only a closed structural code here.
        logger.warning("bridge: conversation delivery held code=conversation_failed")
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
