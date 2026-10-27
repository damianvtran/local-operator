"""Wire shapes for the desktop's Team Bridge OWNER AUDIT ("{CoS}'s messages").

WHAT THIS IS. The agent owner's read-only view of every conversation their own
agent (the designated chief of staff) is or was in: DMs with humans and agents,
groups and channels, including messages the agent's screening REJECTED or never
screened (agent-server #221, docs/TEAM-BRIDGE-CONVERSATIONS.md §16; the frozen
contract is the owner-audit contract v1-v3 + FROZEN). These models are what
``/v1/desktop/team-bridge/owner-audit/...`` answers inside the usual
``CRUDResponse`` envelope, and the renderer (local-operator-ui
``src/shared/desktop-contract.ts``) mirrors them field for field.

HUMAN-ONLY. Every body here bypassed the agent's screening on purpose: screening
guards the agent's CONTEXT against injection, it is not a confidentiality
boundary against the agent's owner. So none of this may reach a model — see the
section comment on ``TeamBridgeClient.owner_audit_conversations`` and
``tests/unit/team_bridge/test_owner_audit_isolation.py``.

THE SHAPES ARE THE HUB'S, PASSED THROUGH. The route validates the hub's answer
against these models (a shape this build cannot read is the family's 502
``malformed_page`` rather than a half-rendered page) and then forwards exactly
the fields the hub SENT (``exclude_unset``), so an omitted ``left_seq`` stays
omitted instead of turning into an explicit ``null`` that reads as state.
``extra="allow"`` keeps the hub's additive-field rule (§6.6): a field the hub
adds later reaches the renderer without a core release.

VOCABULARIES ARE DOCUMENTED, NOT ENFORCED. ``agent_delivery_status`` and the
member ``state`` are ``str`` rather than ``Literal``: an enum value the hub adds
later must degrade to "unknown status" in the renderer, not fail the whole page.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _HubShape(BaseModel):
    """Hub-owned response: unknown fields pass through (the additive rule)."""

    model_config = ConfigDict(extra="allow")


class OwnerAuditMember(_HubShape):
    """One conversation member (the hub's ``BridgeConversationParticipantView``).

    On a CLOSED audit window members added after the agent's departure are
    already hidden by the hub (contract v3 N3).
    """

    account_id: str
    #: ``human`` | ``agent``.
    kind: str
    display_label: str = ""
    role: str = ""
    state: str = ""
    joined_seq: int | None = None
    notify: str | None = None


class OwnerAuditViewer(_HubShape):
    """The owner-audit DISCLOSURE every member sees (contract v2 M5).

    One entry per agent member whose owner can audit this conversation, e.g.
    render "Visible to {display_label} ({agent}'s owner)".
    """

    account_id: str
    display_label: str = ""
    #: Always ``agent_owner`` today.
    reason: str = "agent_owner"
    #: ``a:<endpoint>``: the agent member this viewer owns.
    agent_member_key: str = ""


class OwnerAuditAgent(_HubShape):
    """The audited agent member and its window ``[joined_seq, left_seq)``.

    ``left_seq`` absent means the window is open (the agent is still a member).
    """

    account_id: str
    display_label: str = ""
    #: ``active`` | ``left`` | ``removed``.
    state: str
    joined_seq: int
    left_seq: int | None = None


class OwnerAuditConversation(_HubShape):
    """One audit row (the hub's ``BridgeConversationOwnerAuditView``).

    DELIBERATELY NOT the member conversation row: an audit has no reader, so
    there is no ``last_read_seq``/``unread_count``/``notify`` — the renderer
    must raise no unread badge or notification from this list (contract UI
    rule: agent channels never notify humans). For a closed window
    ``latest_seq``/``last_message_at`` describe the last in-window message.
    """

    id: str
    tenant_id: str
    #: ``dm`` | ``group`` | ``channel``.
    kind: str
    name: str | None = None
    topic: str | None = None
    visibility: str = ""
    status: str = ""
    #: ``humans`` | ``agents`` for the derived builtin channels.
    builtin: str | None = None
    last_message_at: str | None = None
    latest_seq: int = 0
    last_thread_root_id: str | None = None
    members: list[OwnerAuditMember] = []
    member_count: int = 0
    viewers: list[OwnerAuditViewer] | None = None
    agent: OwnerAuditAgent


class OwnerAuditConversationPage(_HubShape):
    """``GET /v1/desktop/team-bridge/owner-audit/conversations``."""

    conversations: list[OwnerAuditConversation]
    #: Opaque; pass back as ``cursor`` for the next page.
    next_cursor: str | None = None
    has_more: bool = False


class OwnerAuditSender(_HubShape):
    account_id: str
    #: ``human`` | ``agent``.
    kind: str
    display_label: str = ""


class OwnerAuditAttachment(_HubShape):
    """One attachment descriptor; bytes via the owner-audit attachment route.

    ``mime`` is the hub's sniffed type. A non-image (generic file) is served by
    the byte route as ``application/octet-stream`` and should be offered as a
    download, never rendered inline. ``state`` ``expired`` means the bytes were
    purged (render a placeholder).
    """

    id: str
    mime: str = ""
    size: int = 0
    width: int | None = None
    height: int | None = None
    sha256: str = ""
    state: str = ""
    expires_at: str | None = None


class OwnerAuditMessage(_HubShape):
    """One audited message (the hub's ``BridgeConversationMessageView``).

    ``body`` is the FULL text whatever the agent's verdict was; render held and
    rejected bodies as plaintext, collapsed by default, with no link previews or
    remote images (contract v2 m4). ``agent_delivery_status`` is ``admitted`` |
    ``rejected`` | ``not_screened`` and is ABSENT on messages the audited agent
    itself sent (v3 N4); ``agent_verdict_at`` accompanies admitted/rejected.
    """

    id: str
    conversation_id: str
    seq: int
    sender: OwnerAuditSender
    body: str | None = None
    body_hash: str | None = None
    #: ``human`` | ``agent`` (who authored it).
    origin: str = ""
    agent_label: str | None = None
    created_at: str
    thread_root_id: str | None = None
    reply_count: int | None = None
    last_reply_at: str | None = None
    reply_participants: list[str] | None = None
    attachments: list[OwnerAuditAttachment] | None = None
    attachment_count: int | None = None
    agent_delivery_status: str | None = None
    agent_verdict_at: str | None = None


class OwnerAuditMessagePage(_HubShape):
    """``GET .../owner-audit/conversations/{id}/messages``: newest-first."""

    messages: list[OwnerAuditMessage]
    #: Pass back as ``before_seq`` for the next (older) page.
    next_before_seq: int | None = None
    has_more: bool = False


class OwnerAuditThread(_HubShape):
    """``GET .../owner-audit/conversations/{id}/messages/{root}/replies``: oldest-first."""

    root: OwnerAuditMessage
    replies: list[OwnerAuditMessage]
    #: Pass back as ``after_seq`` for the next page.
    next_after_seq: int | None = None
    has_more: bool = False


def owner_audit_projection(model: type[_HubShape], payload: Any) -> dict[str, Any]:
    """Validate ``payload`` as ``model``; return exactly the fields the hub sent.

    Raises ``pydantic.ValidationError`` on a shape this build cannot read; the
    route maps that to ``malformed_page``.
    """

    return model.model_validate(payload).model_dump(mode="json", exclude_unset=True)


__all__ = [
    "OwnerAuditAgent",
    "OwnerAuditAttachment",
    "OwnerAuditConversation",
    "OwnerAuditConversationPage",
    "OwnerAuditMember",
    "OwnerAuditMessage",
    "OwnerAuditMessagePage",
    "OwnerAuditSender",
    "OwnerAuditThread",
    "OwnerAuditViewer",
    "owner_audit_projection",
]
