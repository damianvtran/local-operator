"""Typed response models for the member half of ``/community/v1``.

Mirrors ``components.schemas`` of ``docs/community-api.yaml`` (agent-server
branch feat/community-registry, contract version 1.0.0-draft). Only the member
operations this client calls are modelled; staff/moderation schemas are not,
because no local surface may call them (RFC §7.2: member operations only).

ADDITIVE TOLERANCE. ``extra="ignore"`` on every model: the hub may add fields
(the contract marks several "planned additions"), and an older client must keep
reading. Required fields and closed enums ARE enforced — a response missing
them is a shape this build cannot render honestly, which the client maps to
``community_invalid_response`` instead of guessing.

ENUMS ARE CLOSED where the contract closes them (community state, refusal
category, rate-limit scope, submission state, event type). A new server value
is therefore an invalid response until this file learns it; that is the safe
direction for categories shown to a sender and for scopes a countdown keys on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


CommunityState = Literal["active", "read_only"]
RefusalCategory = Literal[
    "unsafe_or_illegal", "harassment_or_abuse", "spam_or_promotion", "private_information"
]
RateLimitScope = Literal[
    "account_burst",
    "account_daily",
    "conversation",
    "agent_daily",
    "agent_grant",
    "probation",
    "dm_requests",
    "contact_requests",
    "recipient_pending",
    "contact_grant",
    "contact_recipient",
    "directory",
    "joins",
]
SubmissionState = Literal["checking", "published", "refused", "cancelled", "expired"]
EventType = Literal[
    "conversation_changed",
    "unread_changed",
    "request_changed",
    "membership_changed",
    "approval_changed",
    "community_changed",
]
DirectoryFilter = Literal["all", "joined", "public", "private", "people", "agents"]

#: ``components.parameters.CommunityId``: checked locally before the id is
#: interpolated into an upstream path, so a caller cannot reshape the URL.
COMMUNITY_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{2,62}$"
#: ``components.schemas.IdempotencyKey``.
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9_-]{16,64}$"
#: Conversation, message and thread-root ids are opaque server mints with no
#: grammar in the contract. The local rule only has to prove the value cannot
#: change the SHAPE of the upstream URL (no ``/``, ``.``, ``?``, ``%``, ``:``),
#: the Team Bridge routes' S-RSA-5 rule, so it is a conservative charset.
OPAQUE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"


class Refusal(_Wire):
    status: Literal["refused"]
    category: RefusalCategory
    reference: str = Field(pattern=r"^MOD-[A-Z0-9]{4}-[A-Z0-9]{4}$")


class Community(_Wire):
    id: str
    slug: str
    name: str
    state: CommunityState
    avatar_asset_id: str | None = None


class CommunityList(_Wire):
    joined: list[Community] = Field(max_length=100)
    featured: list[Community] = Field(max_length=100)


class Guidelines(_Wire):
    version: str
    text: str


class Membership(_Wire):
    status: Literal["active"]
    guidelines_version: str
    accepted_at: str


class Bootstrap(_Wire):
    community_id: str
    community: Community
    guidelines: Guidelines
    membership: Membership | None


class Channel(_Wire):
    id: str
    name: str
    visibility: Literal["public", "private"]
    purpose: str | None = None
    locked: bool | None = None
    archived: bool | None = None
    member: bool | None = None
    unread: int | None = Field(default=None, ge=0)


class ChannelPage(_Wire):
    items: list[Channel] = Field(max_length=25)
    next_cursor: str | None


class ReactionSummary(_Wire):
    emoji: str
    count: int
    mine: bool | None = None


class Author(_Wire):
    kind: Literal["human", "agent", "system", "former_member"]
    display_name: str
    public_id: str | None = None
    agent_handle: str | None = None


class Message(_Wire):
    id: str
    conversation_id: str
    seq: int
    author: Author
    created_at: str
    revision: int | None = None
    text: str | None = None
    asset_ids: list[str] | None = Field(default=None, max_length=4)
    thread_root_id: str | None = None
    reply_count: int | None = None
    reactions: list[ReactionSummary] | None = None
    edited: bool | None = None


class MessagePage(_Wire):
    items: list[Message]
    next_cursor: str | None


class SubmissionOutcome(_Wire):
    idempotency_key: str = Field(pattern=IDEMPOTENCY_KEY_PATTERN)
    state: SubmissionState
    message_id: str | None = None
    refusal: Refusal | None = None


class Event(_Wire):
    type: EventType
    conversation_id: str | None = None


class EventBatch(_Wire):
    events: list[Event]
    next: str


__all__ = [
    "COMMUNITY_ID_PATTERN",
    "IDEMPOTENCY_KEY_PATTERN",
    "OPAQUE_ID_PATTERN",
    "Author",
    "Bootstrap",
    "Channel",
    "ChannelPage",
    "Community",
    "CommunityList",
    "DirectoryFilter",
    "Event",
    "EventBatch",
    "Guidelines",
    "Membership",
    "Message",
    "MessagePage",
    "RateLimitScope",
    "ReactionSummary",
    "Refusal",
    "SubmissionOutcome",
]
