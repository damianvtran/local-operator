"""Typed, parse-don't-validate records for the team-bridge metadata path (H2).

WHAT LIVES HERE. The values the store may persist and the wire projections
that produce them. Two hard rules, both from the frozen contract (§13.9's
"operational metadata … no body cache" and the H2 handoff's privacy list):

1. **No content.** Nothing here holds a body, excerpt, summary, sender name or
   email, raw model reason, token or HTTP error detail. Projections WHITELIST
   the metadata fields; everything else on a server item is dropped by
   construction, not by a redaction pass.
2. **Distinct types for distinct cursors.** ``InboxCursor`` is an OPAQUE
   server string; ``FeedSeq`` is an integer. They are never interchangeable
   (H2 handoff: "never interchange them") — the constructor functions here
   are the only sanctioned way to mint either, and they refuse the other's
   shape, so a swap is a type error at the call site rather than a silent
   misordered resume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, NewType

__all__ = [
    "ACK_TARGETS",
    "AckTarget",
    "FeedSeq",
    "InboxCursor",
    "InboxPage",
    "MessageRef",
    "OwnerSnapshot",
    "ScreeningReceipt",
    "TeamBridgeTypeError",
    "feed_seq",
    "inbox_cursor",
    "project_inbox_page",
    "project_message_ref",
]

#: §6.2's closed ack states for ONE recipient advancing their own delivery:
#: unread→read (with an allow verdict) or unread/held→blocked (with a reject
#: verdict). working/done are NOT H2 acks (no automatic progression exists).
ACK_TARGETS = ("read", "blocked")
AckTarget = Literal["read", "blocked"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_VERDICTS = ("allow", "reject")
_DELIVERY_STATES = ("unread", "read", "working", "done", "blocked")


class TeamBridgeTypeError(ValueError):
    """A record violated its contract shape. Fail closed: callers refuse."""


# ---------------------------------------------------------------------------
# Identity and cursors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerSnapshot:
    """The committed local owner: NON-secret, immutable, exact.

    Deliberately NOT the credential: the token stays in the in-memory
    ``OAuthAccess`` and is never copied here, into SQLite, or into any JSON
    (H2 handoff; contract §13.9). ``credential_id`` is the exact row H1's
    forced-refresh pin already uses, so a same-account re-login with a NEW
    credential row cannot silently continue under this owner — that needs an
    explicit rebinding through the enrollment seam.
    """

    credential_id: int
    account_id: str
    tenant_id: str
    endpoint_id: str
    canonical_base_url: str
    enrollment_generation: int

    def validated(self) -> "OwnerSnapshot":
        """Refuse an incomplete owner. No fresh authority is ever inferred."""

        for name in ("account_id", "tenant_id", "endpoint_id", "canonical_base_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TeamBridgeTypeError(f"owner.{name} must be a non-empty string")
        if not self.canonical_base_url.startswith(("http://", "https://")):
            raise TeamBridgeTypeError("owner.canonical_base_url must be an absolute URL")
        if not isinstance(self.credential_id, int) or isinstance(self.credential_id, bool):
            raise TeamBridgeTypeError("owner.credential_id must be an int")
        if (
            not isinstance(self.enrollment_generation, int)
            or isinstance(self.enrollment_generation, bool)
            or self.enrollment_generation < 1
        ):
            raise TeamBridgeTypeError("owner.enrollment_generation must be >= 1")
        return self


#: An opaque server-issued inbox page cursor. Deliberately NOT an int: the
#: server cursor can reset/regress on resync, so ordering rules differ from
#: ``FeedSeq`` and a swap must not type-check.
InboxCursor = NewType("InboxCursor", str)

#: The monotonic feed position (§6.7). Deliberately NOT a string.
FeedSeq = NewType("FeedSeq", int)


def inbox_cursor(value: Any) -> InboxCursor:
    """Mint an ``InboxCursor``; refuse non-strings and empty strings."""

    if not isinstance(value, str) or not value:
        raise TeamBridgeTypeError("inbox cursor must be a non-empty string")
    return InboxCursor(value)


def feed_seq(value: Any) -> FeedSeq:
    """Mint a ``FeedSeq``; refuse bools (ints in Python) and negatives."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TeamBridgeTypeError("feed seq must be a non-negative int")
    return FeedSeq(value)


# ---------------------------------------------------------------------------
# Screening evidence and wire projections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreeningReceipt:
    """Typed evidence from the (future) screener — never a permission to act.

    The tuple is the write-once verdict the server compares (§6.2, §7.2): a
    repeat must send the EXACT same values, and an existing differing verdict
    is a 409 to reconcile, never to overwrite. ``categories`` comparison is
    order-irrelevant server-side (``bridge_ack_service.go`` treats it as a
    set), so :meth:`tuple_key` mirrors that with a ``frozenset``.
    """

    verdict: str
    model: str
    prompt_version: str
    template_sha256: str
    body_sha256: str
    categories: tuple[str, ...] = ()

    def validated(self) -> "ScreeningReceipt":
        if self.verdict not in _VERDICTS:
            raise TeamBridgeTypeError(f"screening verdict {self.verdict!r} not in {_VERDICTS}")
        for name in ("model", "prompt_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TeamBridgeTypeError(f"screening.{name} must be a non-empty string")
        for name in ("template_sha256", "body_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_HEX.match(value):
                raise TeamBridgeTypeError(f"screening.{name} must be a lowercase sha256 hex")
        if not all(isinstance(c, str) and c for c in self.categories):
            raise TeamBridgeTypeError("screening.categories entries must be non-empty strings")
        return self

    def tuple_key(self) -> tuple[Any, ...]:
        """The comparison form: categories order-irrelevant, like the server."""

        return (
            self.verdict,
            self.model,
            self.prompt_version,
            self.template_sha256,
            self.body_sha256,
            frozenset(self.categories),
        )

    def to_json(self) -> dict[str, Any]:
        """The EXACT §6.2 request block. No extra keys are ever added."""

        return {
            "verdict": self.verdict,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "template_sha256": self.template_sha256,
            "body_sha256": self.body_sha256,
            "categories": list(self.categories),
        }


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TeamBridgeTypeError(f"{where} must be a mapping")
    return value


def _require_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeamBridgeTypeError(f"{where} must be a non-empty string")
    return value


def _optional_id(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class MessageRef:
    """The metadata mirror row for ONE inbox item — never its content.

    Mirrors exactly the handoff's ``deliveries`` fact list: ids, hashes,
    observed state, request bookkeeping. The projection below is the ONLY
    constructor from wire data, and it whitelists: a body, sender display
    block, ``events`` object or arbitrary sidecar on the server item is
    dropped rather than stored.
    """

    message_id: str
    thread_id: str
    tenant_id: str
    message_type: str | None
    body_hash: str | None
    seq: int | None
    origin: str | None
    delivery_state: str
    state_at: str | None
    sender_account_id: str | None
    sender_endpoint_id: str | None
    recipient_account_id: str | None
    recipient_endpoint_id: str | None
    request_state: str | None = None
    request_expires_at: str | None = None
    request_ref: str | None = None

    def validated(self) -> "MessageRef":
        if self.delivery_state not in _DELIVERY_STATES:
            raise TeamBridgeTypeError(
                f"delivery state {self.delivery_state!r} not in {_DELIVERY_STATES}"
            )
        if self.body_hash is not None and not _SHA256_HEX.match(self.body_hash):
            raise TeamBridgeTypeError("body_hash must be a lowercase sha256 hex when present")
        return self


@dataclass(frozen=True)
class InboxPage:
    """One typed inbox page: projections + the server's next-page cursor."""

    items: tuple[MessageRef, ...] = field(default_factory=tuple)
    next_cursor: InboxCursor | None = None
    counts: Mapping[str, int] | None = None


def _project_item(item: Mapping[str, Any]) -> MessageRef:
    """Whitelist ONE server item (``{message, delivery, thread, sender}``).

    §6.2's inbox response nests these; the single-message read shares the
    shape. Required: message id + thread id + tenant id + delivery state.
    Everything else is optional metadata, and anything NOT named below —
    including ``message.body`` — is deliberately never read.
    """

    message = _require_mapping(item.get("message"), "item.message")
    delivery = _require_mapping(item.get("delivery", {}), "item.delivery")
    message_id = _require_id(message.get("id") or message.get("_id"), "item.message.id")
    thread_id = _require_id(message.get("thread_id") or message.get("threadId"), "thread_id")
    tenant_id = _require_id(message.get("tenant_id") or message.get("tenantId"), "tenant_id")
    state = delivery.get("state")
    if not isinstance(state, str) or state not in _DELIVERY_STATES:
        raise TeamBridgeTypeError(f"item.delivery.state {state!r} not in {_DELIVERY_STATES}")
    raw_sender = item.get("sender")
    sender: Mapping[str, Any] = raw_sender if isinstance(raw_sender, Mapping) else {}
    raw_recipient = message.get("recipient")
    recipient: Mapping[str, Any] = (
        _require_mapping(raw_recipient, "message.recipient")
        if isinstance(raw_recipient, Mapping)
        else {}
    )
    raw_request = message.get("request")
    request: Mapping[str, Any] = raw_request if isinstance(raw_request, Mapping) else {}
    seq = message.get("seq")
    body_hash = message.get("body_hash") if isinstance(message.get("body_hash"), str) else None
    origin = message.get("origin") if isinstance(message.get("origin"), str) else None
    message_type = message.get("type") if isinstance(message.get("type"), str) else None
    if isinstance(seq, bool) or not isinstance(seq, int):
        seq = None
    return MessageRef(
        message_id=message_id,
        thread_id=thread_id,
        tenant_id=tenant_id,
        message_type=message_type,
        body_hash=body_hash,
        seq=seq,
        origin=origin,
        delivery_state=state,
        state_at=_optional_id(delivery.get("state_at") or delivery.get("stateAt")),
        sender_account_id=_optional_id(sender.get("account_id") or sender.get("accountId")),
        sender_endpoint_id=_optional_id(sender.get("endpoint_id") or sender.get("endpointId")),
        recipient_account_id=_optional_id(
            recipient.get("account_id") or recipient.get("accountId")
        ),
        recipient_endpoint_id=_optional_id(
            recipient.get("endpoint_id") or recipient.get("endpointId")
        ),
        request_state=(request.get("state") if isinstance(request.get("state"), str) else None),
        request_expires_at=(
            request.get("expires_at") if isinstance(request.get("expires_at"), str) else None
        ),
        request_ref=(
            request.get("request_ref") if isinstance(request.get("request_ref"), str) else None
        ),
    ).validated()


def project_inbox_page(result: Mapping[str, Any]) -> InboxPage:
    """The §6.2 inbox result → typed page. Refuses a non-list ``items``.

    A malformed page is a REFUSAL, not a partial adoption: half a page merged
    under a checkpoint that claims it is complete is exactly the bug the
    checkpoint exists to prevent.
    """

    result = _require_mapping(result, "inbox result")
    items = result.get("items")
    if not isinstance(items, list):
        raise TeamBridgeTypeError("inbox result.items must be a list")
    next_cursor = result.get("next_cursor")
    cursor = inbox_cursor(next_cursor) if isinstance(next_cursor, str) and next_cursor else None
    counts = result.get("counts")
    clean_counts: dict[str, int] | None = None
    if isinstance(counts, Mapping):
        clean_counts = {
            str(key): value
            for key, value in counts.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    return InboxPage(
        items=tuple(_project_item(item) for item in items if isinstance(item, Mapping)),
        next_cursor=cursor,
        counts=clean_counts,
    )


def project_message_ref(result: Mapping[str, Any]) -> MessageRef:
    """The single-message read: the same nested shape as ONE inbox item."""

    return _project_item(_require_mapping(result, "message result"))
