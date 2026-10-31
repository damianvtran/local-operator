"""H5d — the bridge agent tool (the CoS half; NO decide authority).

WHAT THE COS MAY DO with a delivered bridge message, per contract:

- **take_up** — fetch the FULL body and verify it against the ``body_sha256``
  the wake payload carried (S-20). The wake excerpt is a reading aid; the tool
  is where the agent gets the complete screened text, and the digest is what
  proves the fetched message IS the screened one. A mismatch is a typed HOLD:
  nothing is acked, nothing is handed to the model.
- **ack** — ``working`` claims the message for THIS session (the §13.6 local
  claim/lease, so a second session cannot also wake it); ``done`` releases
  that claim (a claim releases on ack done, §13.6/N2). Claim tokens are held
  by the runtime for the session's lifetime; a lost token refuses rather than
  stealing a successor's claim.
- **send / reply** — outbound notes/questions/requests through the durable
  outbox (write-intent-first, §6.2). ``reply`` is the §13.5 pairing: thread_id
  AND to always together. ``state=sent`` means server acceptance into the
  legacy peer inbox, not conversation delivery or recipient agent execution.
- **NO decide** — the ``decide`` op does not exist here BY DESIGN: only H5b's
  answer path (a trusted human Approve/Deny) may call ``/decide`` (§10.3,
  #125). The schema omits it and the executor refuses the word if a model
  invents it, pointing at the human ask queue.

THE CHANNEL OPS work the conversation domain AS THIS SESSION'S AGENT:

- **channels** — one page of the owner account's conversations. Tenant-visible
  channels the owner has not joined come back with ``discoverable=true``,
  ``member=false`` and a member count, so an agent can find a channel to join.
  It lists LIVE conversations only: an archived one has left the server's
  default list (agent-server ``status=open``), so an agent cannot see or post to
  it. That is the product intent. There is deliberately no ``status`` parameter:
  measured against ``scripts/bench_context_budget.py``, publishing one optional
  field costs ~22 billed tokens on both the main and the child arm, and the
  child arm's ratchet had 2 tokens of headroom when this was decided, so the
  field cannot ship without first paying for it elsewhere. The desktop
  ``/conversations?status=archived`` route is the human's view of archives.
- **channel_create** — a group or channel created by the agent identity. The
  creator agent member is added automatically (the server requires it); the
  human owner is added ONLY when named explicitly, because an agent-created
  channel is the agent's own space, not a silent copy of its owner's.
- **channel_post** — append one message as the agent. The body is the caller's
  text; one ``client_ref`` is minted per call, so a retry inside the client
  reuses it and a second call is a second message. ``thread_root_id`` posts it
  as a REPLY in that message's thread instead of at channel level — how agents
  coordinate without flooding the channel (e.g. one QA-claim thread per item
  in #agents). Threads are one level: the root must be a channel-level message.

  ``in_reply_to_message_id`` binds the exact admitted inbound message to this
  conversation BEFORE lookup, budget or network I/O. Thread roots do not bind
  replies. After actual bridge input dispatch in a run, supply this binding or
  declare ``new_post: true`` intentionally. That declaration is model-supplied
  intent, NOT semantic isolation, trusted authority, or duplicate suppression.

  ``to: {"name": …}`` DIRECTS the post at one member, and the name is resolved,
  never guessed: trimmed, case-insensitive, whole DISPLAY name only (no handle,
  no email, no substring), against THIS conversation's ACTIVE members — the
  scope is the room the message lands in, so a colleague who exists in the
  organization but is not in this conversation cannot be resolved to. Exactly
  one match becomes that member's mention row on the wire (``kind`` +
  ``account_id``), so the name itself never leaves the machine; zero matches
  and several matches both refuse, the latter listing the candidates. The
  attempt is written to ``recipient_resolutions.jsonl`` (requested name,
  resolved id or null, matched count, conversation id) BEFORE the post, and on
  a refusal too, so a misdelivery claim is reconstructable afterwards.
- **channel_thread** — one thread, oldest-first: the root (with its reply
  count) and a bounded page of replies. Same screening rule as history — an
  unadmitted reply is metadata with ``body`` null.
- **channel_history** — a bounded page of messages. The server admits bodies
  per (message, endpoint) screening; an agent sees the METADATA (sender, seq,
  hash) of messages it has not admitted, with ``body`` null, and only the
  admitted bodies. There is intentionally no local body cache here. The one
  exception to ``body: null`` is a message THIS account authored: the hub never
  delivers it back for admission, so history and thread fetch its text per read
  (hash-verified, never stored) so a sender can read what it wrote.

- **react** — set an emoji on an admitted conversation message (or remove it
  with ``remove=true``). This is a screened agent WRITE, never an ack; the hub
  refuses a message this endpoint has not admitted. Reactions never wake agents.

IDENTITY is never a model argument: the actor endpoint and provenance session
come from ``runtime.owner.endpoint_id`` and ``runtime.session_id``. Every send
carries that session id as origin provenance (the server stores it and never
returns it); a runtime with no endpoint refuses every channel op rather than
acting as a human. Neither message bodies nor session ids are ever logged here.

A DISPLAY NAME (``to: {"name": …}``) is resolved, never guessed, and WHICH SET
it resolves against is the difference between the two lanes:
``channel_post`` resolves inside that conversation's active members (it carries
a conversation id), while ``send``/``reply`` resolve inside the owner's own
``send`` allowlist intersected with the bridge directory (they carry no
conversation id at all, so the conversation rule cannot apply). Both run the
same resolver, both demand exactly one match, and both record the attempt in
``recipient_resolutions.jsonl`` before anything is sent — with the scope word in
the row, so a later misdelivery claim is answerable for either lane.

REFUSALS, one line each: ``not_permitted`` for ``decide``; ``invalid_request``
for unknown ops, malformed ``channel_id``, bad history bounds/limits, invalid
member shapes, and a display name that resolves to no member, to several, or to
one whose bridge is not enabled; ``no_agent_endpoint`` when a channel op has no
owner endpoint; ``members_failed`` when the conversation read that supplies the
candidates does not answer; ``directory_failed`` when the directory/allowlist
pair that a ``send``/``reply`` name resolves against does not answer;
``audit_unavailable`` when the resolution could not be recorded (nothing is sent,
posted or queued — a directed delivery that cannot be accounted for is refused);
``*_failed`` when the transport does not confirm.

THE RUNTIME IS BUILD-TIME GATED: a session with no bridge runtime — unsigned,
disabled, or with no live enrollment generation — never advertises this tool at
all (the same createIf shape ``wake`` uses). No probing, no second gating
convention. EVERY writer session of the enrolled owner now carries the tool: the
collaboration ops act on that owner's live binding, and the mailbox /
owner-shared reads are gated per call by the strict authority fence
(`not_designated` / conversation-level refusals), which is why runtime presence
is NOT a test for "this is the chief-of-staff session" (review round 1).

WHICH SESSIONS MAY ACT, and what the designation no longer gates, is
`OUTBOUND_OPS`: every bridge-connected writer session of the enrolled owner may
`directory`/`dm`/`channel_create`/`channel_post`, and read or react in a room its
own AGENT is a member of (`channel_history`, `channel_thread`, `react` are
endpoint-addressed, so the hub gates them on the agent's membership, never the
owner's). Kept behind the strict fence: the inbound mailbox (take_up / ack /
send / reply), the owner's human conversation list (`channels`), and the
owner-shared reads (`shared_*`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.harness.types import (
    AgentTool,
    Content,
    ImageContent,
    TextContent,
    ToolContext,
    ToolResult,
)

from . import reply_guard
from .client import TeamBridgeClient
from .outbox import Outbox, send_from_outbox
from .recipient_resolution import (
    ResolutionAuditUnavailable,
    members_from_allowlist,
    members_from_conversation,
    record_resolution,
    resolve_for_delivery,
)
from .types import _CONVERSATION_MESSAGE_ID as _CANONICAL_MESSAGE_ID
from .types import (
    CONVERSATION_ID_PATTERN,
    RECIPIENT_NAME_SHAPE,
    SCOPE_ALLOWLIST,
    SCOPE_CONVERSATION,
    SEND_RECIPIENT_DESCRIPTION,
    InvalidSendRecipient,
    NameResolution,
    RecipientNameRefused,
    is_conversation_message_id,
    validate_recipient_shape,
    validate_send_recipient,
)

#: Ops the tool accepts. ``decide`` is deliberately absent (see the module
#: docstring); the executor still names it in its refusal so a model that
#: tries the word gets the right sentence back.
CHANNEL_OPS: tuple[str, ...] = (
    "channels",
    "channel_create",
    "channel_post",
    "channel_history",
    "react",
    "channel_thread",
    "dm",
    "shared_conversations",
    "shared_history",
)
#: ``directory`` is read-only METADATA about the organization (who is on the
#: team, each person's chief-of-staff endpoint and its current name). It acts
#: as nobody, so unlike the channel ops it needs no agent endpoint.
BRIDGE_OPS: tuple[str, ...] = ("take_up", "ack", "send", "reply", "directory", *CHANNEL_OPS)
#: The COLLABORATION surface: the ops the user story needs from ANY bridge-
#: connected session of the enrolled owner ("two people's article-search manager
#: sessions coordinating on campaigns"). Only these take the outbound fence
#: (every binding/consent check kept, designation relaxed, because a colleague's
#: task session is not the account's inbox). Everything else — the mailbox ops
#: (take_up/ack/send/reply) and the owner-shared reads — keeps the STRICT fence
#: exactly as it shipped: widening the inbox would change authority that was
#: reviewed without anyone asking for it, and a session that may not read the
#: owner's rooms must not gain that by posting in one.
OUTBOUND_OPS: tuple[str, ...] = (
    "directory",
    "dm",
    "channel_create",
    "channel_post",
    # READS OF A ROOM THE AGENT IS IN. All three are ENDPOINT-addressed, so the
    # hub gates them on this agent's own membership — never on the owner's human
    # membership — which is what lets a task session READ the reply to the DM it
    # opened (review round 1 MAJOR 2: without these, `dm` was write-only and the
    # reply could only be read from the designated session). `channels` stays
    # OUT on purpose: that list is the owner's human conversation list, and a
    # task session has no business enumerating the owner's rooms.
    "channel_history",
    "channel_thread",
    "react",
)
#: The fetch-shaped ops: admitted while paused/read-only, and approval tier
#: ``read``. ONE list so the admission gate and the approval tier agree.
READ_OPS: tuple[str, ...] = (
    "take_up",
    "channels",
    "channel_history",
    "channel_thread",
    "directory",
    "shared_conversations",
    "shared_history",
)

#: The server caps a conversation page at 100 (contract §10.1); bounding here
#: turns an over-wide ask into a typed refusal instead of a server 400.
CONVERSATION_PAGE_MAX = 100

#: Display label for the auto-added creator member. The label is display-only
#: and taken from the member row by the server; this one is what another
#: member sees until the channel's members relabel it.
CHANNEL_CREATE_DEFAULT_LABEL = "agent"

#: The most images ONE channel_history call puts into context. A 100-message
#: page could otherwise carry 400 images (each up to 1 MiB after bounding) —
#: a context bill no single read should run up. The rest are named in text
#: so the agent can page a narrower window to see them.
HISTORY_IMAGES_MAX = 8

#: Private outcome key for the fetched images. Popped by ``_execute`` before
#: the outcome becomes ``ToolResult.details`` — details persist with the
#: transcript row, and image bytes must exist only in the content blocks.
_IMAGES_KEY = "_images"

#: Local claim lifetime for ``ack working`` — generous enough for a CoS turn,
#: short enough that a crashed session's claim frees itself.
CLAIM_TTL_S = 3600.0


@dataclass
class BridgeRuntime:
    """Everything the tool needs, constructed by the session that owns it.

    Session-scoped ON PURPOSE: ``claims`` maps message id -> the lease token
    this session minted, so ``ack done`` releases the claim this session took
    and nothing else (a token is never re-derived — store.release_claim
    compares the exact value).
    """

    client: TeamBridgeClient
    config_dir: Path | str
    tenant_id: str
    owner: Any  # store.OwnerSnapshot — typed loosely to keep imports light
    session_id: str
    owner_key: str = ""
    claims: dict[str, str] = field(default_factory=dict)
    #: The session's admission state — settled ONLY by an authenticated status
    #: answer at a BOUNDARY (boot, or an op-edge that landed while not
    #: admitted). Construction always starts "pending": the persisted gates
    #: prove WHO (owner/designation/consent), never CURRENT eligibility, and a
    #: cached UI record is not authority. "disabled" is not process-permanent:
    #: a later boundary may re-admit it once the local gates hold (§36).
    admission: str = "pending"
    admission_note: str = ""
    # Run-local dispatch evidence, never inferred from persisted conversation
    # text. Remains true after owed cards clear; the session resets at boundaries.
    bridge_input_presented: bool = False
    #: The ONE shared in-flight admission task (S-2): callers JOIN it instead
    #: of queueing their own fetch, and it is cancelled/discarded when the
    #: bound authority is revoked. Lazily created (no loop-bound default).
    _admission_task: Any = None

    def key(self) -> str:
        if self.owner_key:
            return self.owner_key
        return json.dumps(
            [
                getattr(self.owner, "canonical_base_url", ""),
                getattr(self.owner, "account_id", ""),
                getattr(self.owner, "tenant_id", ""),
            ]
        )


class ChannelMemberParams(BaseModel):
    """One invited member. An agent names its endpoint; a human must not."""

    model_config = ConfigDict(extra="forbid")

    account_id: str
    kind: Literal["human", "agent"] = "human"
    endpoint_id: str = ""
    display_label: str = ""


def _closed_recipient_schema(schema: dict[str, Any]) -> None:
    # Derive explicit properties from Pydantic's Literal-key schema instead of
    # requiring every provider to implement propertyNames. This preserves the
    # closed key set without a second hand-maintained list or duplicate models.
    # The field is a UNION (object | bare id | null), so the object branch is one
    # of several at the top level; the string and null arms fail the ``type``
    # test and are left exactly as published. The pop is deliberately UNGUARDED:
    # if a serializer ever publishes the object branch without ``propertyNames``
    # this must fail loudly HERE (a KeyError while the schema is built —
    # ``BridgeToolParams.model_json_schema()``, i.e. session start) rather than
    # quietly leave a recipient object open to extra keys — that silence is the
    # bug class the recipient validation exists to prevent.
    for branch in schema.get("anyOf", [schema]):
        if branch.get("type") == "object":
            names = branch.pop("propertyNames")["enum"]
            value = branch["additionalProperties"]
            branch["properties"] = {name: value for name in names}
            branch["additionalProperties"] = False


#: The recipient OBJECT shape as a type: exactly one of the three keys, each a
#: non-empty string. Cardinality and the closed key set live HERE (not on the
#: field) so they still apply now that the field also accepts a bare id.
#:
#: ``name`` is in the closed set because ``channel_post`` accepts it, and the
#: schema is ONE object for every op. It is not therefore addressable: the
#: per-op check decides. ``send``/``reply`` refuse it before admission
#: (``validate_send_recipient`` — they carry no conversation to scope a
#: resolution to); ``channel_post`` resolves it against that conversation's
#: active members (``resolve_for_delivery``). Advertising it at the schema and
#: refusing it per op is deliberate: a model should be able to choose the right
#: op by reading the tool, and the description says which op resolves a name.
_SendRecipientObject = Annotated[
    dict[Literal["endpoint_id", "account_id", "name"], Annotated[str, Field(min_length=1)]],
    Field(min_length=1, max_length=1),
]


class BridgeToolParams(BaseModel):
    # Keep implementation guidance in comments, not the model docstring: Pydantic
    # publishes that prose in every tool call's schema. extra="forbid" already
    # communicates the closed top-level contract without restating it as prose.
    model_config = ConfigDict(extra="forbid", title="Bridge")

    op: Literal[
        "take_up",
        "ack",
        "send",
        "reply",
        "channels",
        "channel_create",
        "channel_post",
        "channel_history",
        "react",
        "channel_thread",
        "directory",
        "dm",
        "shared_conversations",
        "shared_history",
    ]
    message_id: str = ""
    body_sha256: str = ""
    state: Literal["working", "done"] | None = None
    # Literal keys + cardinality publish the same closed one-of as two named
    # recipient models, without duplicating object definitions, titles and
    # descriptions in every prompt. Pydantic and JSON Schema both reject unknown
    # keys; the shared pre-admission/client guard additionally rejects padding.
    #
    # A BARE ID is also accepted: ``"acc_123"`` is one recipient in one field,
    # the shape a model writes first, and it is unambiguous. The description is
    # the schema's own string, assembled from the same constants the refusals
    # teach (:data:`SEND_RECIPIENT_DESCRIPTION`), so the published advice cannot
    # drift from what ``validate_send_recipient`` accepts. Names stay refused:
    # only an id (bare or paired) is a destination.
    to: _SendRecipientObject | Annotated[str, Field(min_length=1)] | None = Field(
        default=None,
        description=SEND_RECIPIENT_DESCRIPTION,
        json_schema_extra=_closed_recipient_schema,
    )
    type: Literal["note", "question", "request"] = "note"
    text: str = ""
    thread_id: str | None = None
    expires_at: str | None = None
    # -- channel ops --
    channel_id: str = ""
    kind: Literal["group", "channel"] | None = None
    name: str = ""
    topic: str = ""
    visibility: Literal["tenant", "private"] | None = None
    members: list[ChannelMemberParams] = Field(default_factory=list)
    body: str = ""
    # A reaction is a write, not a read or an acknowledgement. The actor is the
    # enrolled endpoint; the model supplies only the message and emoji.
    emoji: str = ""
    remove: bool = False
    limit: int = 50
    cursor: str | None = None
    before_seq: int | None = None
    after_seq: int | None = None
    #: channel_post: reply in this root message's thread; channel_thread: the
    #: thread to read. A conversation message id (bcmsg_…).
    thread_root_id: str = ""
    in_reply_to_message_id: str = Field(
        default="",
        description="Exact admitted inbound bcmsg_ being answered; distinct from thread_root_id.",
    )
    new_post: bool = Field(
        default=False,
        description=(
            "Intentionally start a new post instead of replying to bridge input. "
            "An intent declaration only, not semantic isolation or outbound authority."
        ),
    )
    # dm: the person (account_id) or agent (endpoint_id, from `directory`).
    # Kept as plain strings so the shared pair-lane validator never reads them
    # as a `send` recipient; `to` reaches these same two fields through
    # `_dm_recipient`, which transcribes and refuses disagreements rather than
    # resolving a destination a second time.
    account_id: str = ""
    endpoint_id: str = ""
    # dm to an account: "agent" (default) reaches that person's chief of
    # staff, "human" the person themself.
    target: Literal["agent", "human"] = "agent"

    @field_validator("to")
    @classmethod
    def _normalize_recipient(cls, value: Any) -> dict[str, str] | None:
        """Hand on THE validator's canonical pair, never a second reading.

        The annotation accepts a bare id because the schema must ADVERTISE it, but
        a raw string leaving this model would reach a serializer as one — the exact
        class of silent reinterpretation ``validate_send_recipient`` exists to stop.
        Delegating (rather than re-deriving the mapping here) is what keeps ONE
        rule for the executor, the wire client and every other caller.

        The SHAPE validator, not the send one: a display name is a legal shape for
        the op that can resolve it, and this model is shared by every op. The
        op-specific refusal is the executor's, where the op is known.
        """
        return None if value is None else validate_recipient_shape(value)


def _error(message: str, *, code: str = "invalid_request") -> dict[str, Any]:
    return {"ok": False, "code": code, "error": message}


#: A refusal's ``field``/``rule`` is a server identifier, never prose. Bound and
#: flatten it anyway: this text is billed into the agent's context and persisted
#: in the transcript, so an upstream answer that smuggled newlines or a paragraph
#: into ``details`` must not be able to shape the tool result around it.
_REFUSAL_TOKEN_MAX = 64


def _refusal_token(value: Any) -> str:
    """One coded value as a single short token, or ``""`` when it is not one."""

    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:_REFUSAL_TOKEN_MAX]


def _failure_text(failure: BaseException) -> str:
    """A bridge failure as its sentence PLUS the coded detail the hub sends.

    WHY THIS EXISTS (operator incident, 2026-10-29). Every messaging 400 from
    the hub carries ``details: {"field": ..., "rule": ...}`` and a sentence that
    literally tells the reader to "see details for the field" (agent-server
    ``internal/errors/messages.go`` — the phrase is deliberate; the details ARE
    the actionable half). This tool wrapped only ``str(failure)``, so that half
    was discarded and the sentence pointed at nothing: an agent that opened a DM
    was refused for ``actor_endpoint_id``/``unsupported_kind``, the reader saw
    "This messaging request is not valid. See details for the field.", and the
    cause took hours to find. A failure the hub explains now carries its
    explanation.

    WHAT TRAVELS, AND WHAT NEVER DOES. Only ``code`` (the refusal's stable
    machine code) and ``field``/``rule`` from its ``details``. Never a body, a
    token, an account id, an endpoint id or a message body: those are caller and
    tenant data, and this sentence reaches the model, the transcript and the
    screen.

    WHEN THE FOOTER APPEARS AT ALL is the second half of the rule: only when the
    hub ANSWERED (a refusal carries ``http_status``) or named a
    ``(field, rule)`` pair. A transport failure, a local store refusal or any
    other exception reads EXACTLY as it did before — the footer is absent, never
    an empty bracket, because there is no server answer to quote and the tool's
    own ``code`` field already says what failed.

    THAT PROXY IS NOT EXACT, and the limit is stated rather than assumed away
    (review rounds 1–2): the client also SYNTHESIZES refusals carrying an
    ``http_status`` with no exchange at all, and ONE of them is reachable from
    these sites — ``_guarded_access``'s 409 ``account_mismatch``, which ``dm``'s
    create and post can raise because both pass ``expected_account_id``. It
    therefore gains a footer with no server answer, which is INTENDED and safe:
    the footer is a code and never content, and the sentence beside it is the
    client's own (it tells the user to sign in again). The others
    (``_require_body_within``'s 413 ``payload_too_large``, the avatar paths' 409s)
    are not reachable here today. If a future local refusal must stay
    byte-identical, key the footer on a server-derived marker on the refusal
    rather than on the status alone.

    The keys are read from the exception, not from a parsed body: every refusal
    shape that reaches here (:class:`~.client.BridgeRefusal` and its subclasses,
    re-coded refusals included) already carries them as attributes, which keeps
    this a content-free projection rather than a second decoder.
    """

    field_rule: list[str] = []
    details = getattr(failure, "details", None)
    if isinstance(details, Mapping):
        for key in ("field", "rule"):
            value = _refusal_token(details.get(key))
            if value:
                field_rule.append(f"{key}={value}")
    # A field/rule pair is the hub's own explanation; an ``http_status`` is the
    # hub having answered at all (a 400 with only a code still diagnoses more
    # than the sentence alone). Neither means there is nothing to add.
    if not field_rule and getattr(failure, "http_status", None) is None:
        return str(failure)
    pairs: list[str] = []
    code = _refusal_token(getattr(failure, "code", ""))
    if code:
        pairs.append(f"code={code}")
    pairs.extend(field_rule)
    if not pairs:
        return str(failure)
    return f"{failure} [refusal {' '.join(pairs)}]"


#: The conversation id family, shared with the desktop routes' S-RSA-5 guard:
#: a model-supplied id is interpolated upstream, so a value that could reshape
#: the path (``#``, ``..``, slashes) is refused BEFORE any client call. The
#: pattern lives in ``team_bridge.types`` so the two entry points cannot drift.
_CONVERSATION_ID = re.compile(CONVERSATION_ID_PATTERN)
#: A conversation MESSAGE id is interpolated into the replies path the same
#: way, so it is held to the minted grammar before any client call. THE
#: package's object, not a copy: this module used to compile its own literal
#: requiring eight characters while ``types`` allows one, so the desktop route
#: and this tool disagreed about which roots exist. Aliasing a compiled pattern
#: is safe — ``re.Pattern`` is immutable and nothing here reassigns or
#: identity-compares it.
_CONVERSATION_MESSAGE_ID = _CANONICAL_MESSAGE_ID


def _resolve_reply(
    runtime: "BridgeRuntime", args: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Fill a ``reply``'s ``to``/``thread_id`` from the delivery it answers (D4).

    Returns ``(args, None)`` to proceed or ``(args, refusal)``. WHY: ``reply``
    once demanded ``to`` and never read ``message_id``, although the owner's
    ``deliveries`` row already holds the sender and thread of every message it
    received; and a ``bcmsg_`` id was pair-sent, so a DM answer landed in the
    peer inbox and never in the DM. Constraints:

    * the destination comes from a row THIS owner received (owner-keyed,
      tenant-matched, generation-fenced) — an unknown or foreign id refuses,
      and an explicit ``to``/``thread_id`` that the row contradicts, or cannot
      vouch for, refuses rather than being trusted;
    * a conversation message never pair-sends: the refusal names the
      ``channel_post`` call instead of routing it, because posting into a
      conversation humans read is a visible act the agent makes explicitly;
    * no ``message_id`` keeps the legacy ``to`` + ``thread_id`` shape as is;
    * local reads only, before admission, so a refusal schedules no status
      read; refusals quote no caller-supplied value (``validate_send_recipient``).
    """

    message_id = str(args.get("message_id") or "").strip()
    if not message_id:
        return args, None
    from .store import Store, StoreRefused

    store = Store(Path(runtime.config_dir))
    try:
        if is_conversation_message_id(message_id):
            conversation_id = store.conversation_of_reserved_message(runtime.owner, message_id)
            # Kept short: a refusal is billed into the agent's context. The
            # reservation has no thread-root column, so the thread comes from
            # the card the agent already holds.
            where = (
                f"channel_id={conversation_id}"
                if conversation_id
                else "channel_id=<its conversation_id>"
            )
            return args, _error(
                f"conversation message: answer with bridge channel_post {where}, "
                "plus thread_root_id if its card names one; reply never reaches a conversation",
                code="conversation_message",
            )
        row = store.reply_target(runtime.owner, message_id)
    except StoreRefused as failure:
        return args, _error(f"could not read the received message: {failure}", code=failure.code)
    if row is None:
        return args, _error(
            "no received message has that message_id; reply answers only a delivery "
            "this agent received",
            code="unknown_message",
        )
    sender = {
        key: str(row[f"sender_{key}"])
        for key in ("endpoint_id", "account_id")
        if row.get(f"sender_{key}")
    }
    if not sender:
        return args, _error("the received message names no sender", code="unknown_message")
    explicit = args.get("to")
    if explicit is None:
        # The pair lane resolves the sender; see the note on the name branch below.
        to = {"endpoint_id": sender["endpoint_id"]} if "endpoint_id" in sender else sender
    elif isinstance(explicit, dict) and "name" in explicit:
        # A NAME has no destination yet: it is resolved AFTER the admission fence
        # (the directory and the owner's own allowlist are wire reads, and this
        # function must not make one before admission). The executor calls this
        # function again on the resolved pair, so the sender check below is
        # applied a moment later rather than skipped — the guard is relocated,
        # never dropped.
        return args, None
    else:
        try:
            to = validate_send_recipient(explicit)
        except InvalidSendRecipient as failure:
            return args, _error(str(failure), code=failure.code)
        key, value = next(iter(to.items()))
        if sender.get(key) != value:
            return args, _error(
                "to is not the sender of that message; omit to and reply resolves it",
                code="reply_conflict",
            )
    thread_id = args.get("thread_id")
    if thread_id is not None and thread_id != row["thread_id"]:
        return args, _error(
            "thread_id is not that message's thread; omit it and reply resolves it",
            code="reply_conflict",
        )
    return {**args, "to": to, "thread_id": row["thread_id"]}, None


async def _resolve_named_recipient(
    runtime: "BridgeRuntime", requested: str
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    """Resolve a display name to ONE addressable pair for ``send``/``reply``.

    Returns ``(pair, None)`` to proceed or ``(None, refusal)``.

    SCOPE, and why it is not the conversation rule: these ops carry no
    conversation id, so there is no room to scope a name to. The scope is the
    owner's REACHABLE set — the bridge directory intersected with the owner's own
    send-allowlist (``members_from_allowlist``, one function, shared) — which is
    narrower than the organization directory on purpose: a name may only resolve
    to somebody this owner has explicitly permitted, and an empty allowlist
    resolves nobody rather than falling back to the directory.

    The audit row is the same row ``channel_post`` writes, under
    ``scope=allowlist``, with the TENANT as its id: the allowlist is the owner's
    per organization, and a reconstruction has to know which one it was read
    from. Both reads happen after the admission gate (they are wire calls), and a
    read that fails is still recorded — as an ``unavailable`` attempt — so the
    one path that never resolved anything does not become the one path with no
    record.
    """

    scope_id = str(getattr(runtime, "tenant_id", "") or "")
    try:
        directory = await runtime.client.directory(runtime.tenant_id)
        allowlist = await runtime.client.allowlist(tenant_id=runtime.tenant_id)
    except Exception as failure:
        try:
            record_resolution(
                config_dir=runtime.config_dir,
                scope=SCOPE_ALLOWLIST,
                scope_id=scope_id,
                resolution=NameResolution(requested=requested.strip(), matches=()),
                outcome="unavailable",
            )
        except ResolutionAuditUnavailable as unaudited:
            return None, _error(str(unaudited), code=unaudited.code)
        return None, _error(
            f"could not read the directory and allowlist a name resolves against: {failure}",
            code="directory_failed",
        )
    try:
        member = resolve_for_delivery(
            config_dir=runtime.config_dir,
            scope=SCOPE_ALLOWLIST,
            scope_id=scope_id,
            requested_name=requested,
            candidates=members_from_allowlist(directory, allowlist),
        )
    except RecipientNameRefused as failure:
        return None, _error(str(failure), code=failure.code)
    except ResolutionAuditUnavailable as failure:
        return None, _error(str(failure), code=failure.code)
    return {"account_id": member.account_id}, None


# -- directory + direct messages ----------------------------------------------
#
# WHY THESE EXIST (operator incident, 2026-10-28). A teammate's chief of staff,
# asked to pass feedback to the operator's Aida, answered "I don't have a route
# to it ... forward this thread". The hub already published everything needed —
# GET /messaging/directory carries each member's chief-of-staff endpoint
# (``agent.endpoint_id``, ``agent.name``, ``agent.state``) — but no tool op read
# it, so an agent could not discover a teammate's agent, and ``send`` reaches
# only the legacy pair inbox, which no Team Bridge conversation view shows.
#
# THE LANE. ``dm`` uses the SAME conversation DM the desktop opens (POST
# /messaging/conversations ``kind=dm`` find-or-create, then the ordinary
# conversation send), acting AS this session's agent (``actor_endpoint_id``).
# So the message is visible in both people's Team Bridge UI, it is an auditable
# conversation row, and the recipient's connector delivers it to their
# designated chief-of-staff session like any conversation message — the
# recipient's screening, owner consent, receive policy and mute rules apply
# server-side unchanged (an agent target its owner's policy does not admit
# stays ``invited``; nothing auto-joins anyone).
#
# NOTHING IS CACHED. Names and endpoint ids are fetched per call and never
# written to the store (store.py's no-name-cache / no-body rules): a renamed
# agent or a re-enrolled endpoint is correct on the next call by construction.

#: The directory's own cap (agent-server ``bridgeDirectoryMaxMembers``); the hub
#: has no cursor, so the tool pages that bounded answer locally.
DIRECTORY_MAX = 200


def _directory_rows(result: Any, owner: Any) -> list[dict[str, Any]]:
    """Project the hub directory into METADATA rows for the model.

    Only identity and reachability: account id, display name, role, bridge
    state, and the chief-of-staff agent (endpoint id, CURRENT name, state). The
    hub's profile/avatar fields are dropped: they cost context and carry nothing
    an agent needs to address someone. The caller's own rows are marked
    ``you: true`` so the agent never DMs itself by accident.
    """

    members = result.get("members") if isinstance(result, dict) else None
    if not isinstance(members, list):
        return []
    own_account = str(getattr(owner, "account_id", "") or "")
    own_endpoint = str(getattr(owner, "endpoint_id", "") or "")
    rows: list[dict[str, Any]] = []
    for member in members[:DIRECTORY_MAX]:
        if not isinstance(member, dict) or not isinstance(member.get("account_id"), str):
            continue
        row: dict[str, Any] = {
            "account_id": member["account_id"],
            "name": str(member.get("display_label") or ""),
            "role": str(member.get("role") or ""),
            "bridge_enabled": member.get("bridge_enabled") is True,
            "bridge_state": str(member.get("bridge_state") or "none"),
        }
        if member["account_id"] == own_account:
            row["you"] = True
        agent = member.get("agent")
        if isinstance(agent, dict) and isinstance(agent.get("endpoint_id"), str):
            row["chief_of_staff"] = {
                "endpoint_id": agent["endpoint_id"],
                "name": str(agent.get("name") or ""),
                "state": str(agent.get("state") or ""),
                "reachable": agent.get("state") == "active",
                **({"you": True} if agent["endpoint_id"] == own_endpoint else {}),
            }
        rows.append(row)
    return rows


async def _directory(runtime: "BridgeRuntime", params: "BridgeToolParams") -> dict[str, Any]:
    if not 1 <= int(params.limit) <= CONVERSATION_PAGE_MAX:
        return _error(f"directory limit must be 1..{CONVERSATION_PAGE_MAX}")
    start = 0
    if params.cursor:
        # ``str.isdigit`` is true for characters ``int()`` rejects (superscripts,
        # circled digits), so the bounds check and the conversion must agree —
        # otherwise a malformed cursor escapes as an exception instead of the
        # typed refusal every neighbouring arm returns (review round 1 MINOR 5).
        if not params.cursor.isdecimal():
            return _error("directory cursor is the next_cursor a previous page returned")
        start = int(params.cursor)
    try:
        result = await runtime.client.directory(
            runtime.tenant_id, expected_account_id=runtime.owner.account_id
        )
    except Exception as failure:
        return _error(
            f"could not read the team directory: {_failure_text(failure)}", code="directory_failed"
        )
    rows = _directory_rows(result, runtime.owner)
    page = rows[start : start + int(params.limit)]
    next_start = start + len(page)
    payload: dict[str, Any] = {
        "members": page,
        "count": len(rows),
        "hub_limit": DIRECTORY_MAX,
        "possibly_truncated": len(rows) == DIRECTORY_MAX,
    }
    if next_start < len(rows):
        payload["next_cursor"] = str(next_start)
    return {
        "ok": True,
        "state": "ok",
        "members": page,
        "next_cursor": payload.get("next_cursor"),
        # Display names are member-chosen text: data, never instructions.
        "content": f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(payload, ensure_ascii=False)}",
    }


def _bridge_word(row: dict[str, Any]) -> str:
    """``off`` for an opted-out / never-enrolled member, else the state itself."""

    state = str(row.get("bridge_state") or "none")
    return "off" if state in ("disabled", "none") else state


def _recipient_label(recipient: dict[str, str]) -> str:
    """The pair recipient as its own id, already validated (closed one-key shape)."""

    return next(iter(recipient.values()), "the recipient")


def _not_reachable(reason: str, message: str) -> dict[str, Any]:
    out = _error(message, code="not_reachable")
    out["reason"] = reason
    return out


def _dm_recipient(
    params: "BridgeToolParams",
) -> tuple["BridgeToolParams", dict[str, Any] | None]:
    """Fold dm's documented ``to`` recipient onto ``account_id``/``endpoint_id``.

    WHY ``to`` IS ACCEPTED (operator incident, 2026-10-29). ``to`` is the shared
    recipient field: its published description is op-agnostic ("Who receives
    it: …"), ``send`` and ``reply`` address through it — and a dm refused it
    outright, so a form the schema offered never reached the hub, and a model
    that addressed a DM the way it addresses a send got a client-side refusal
    naming a shape it had just been shown. Unfolding ``to`` onto the two fields
    dm already carries leaves ONE way to name one recipient.

    NOT A SECOND RESOLVER. ``to`` arrives settled in the SHARED shape —
    ``BridgeToolParams`` delegates to ``validate_recipient_shape``, so cardinality,
    the closed key set and padding are already decided before this function runs —
    and it only TRANSCRIBES. The PER-OP half deliberately lives here, where the op
    is known: a display name is a legal SHAPE because ``channel_post`` (a room's
    active members) and ``send``/``reply`` (the owner's own send allowlist) both
    resolve one, and ``dm`` resolves against neither — its recipients are the
    directory's people and their chiefs of staff — which is why that case is
    answered below rather than in a second shape rule. :func:`_resolve_dm_target`
    keeps owning every decision (the fresh directory read, reachability,
    ``self_target``, the unknown-endpoint and not-a-member refusals) unchanged,
    whichever spelling arrived.

    WHAT REFUSES HERE. Only what the pair itself cannot settle, in the
    resolver's own words: a ``to`` that names a DISPLAY NAME (the shared shape
    validator admits ``{"name": …}`` because a room post and a pair send both
    resolve one; ``dm`` resolves against neither, so it refuses and points at
    ``op=directory`` rather than letting the name be silently dropped beside an
    id), an id standing beside a ``to`` that names the OTHER key ("exactly one of
    account_id or endpoint_id"), or the same key naming two different ids. None
    is guessed at. A fourth branch refuses a ``to`` that is not the canonical
    one-key dict at all, which ``BridgeToolParams`` cannot produce — its validator
    returns that dict or raises — so the branch exists to keep the type checker
    and the runtime on the same value and to refuse rather than guess if that ever
    changes; it reuses the same sentence. ``target`` is untouched, so
    ``to={"endpoint_id": …}`` beside ``target="human"`` hits the resolver's
    existing "endpoint_id addresses an agent" refusal rather than a new one.
    """

    # THE DECLARED TYPE IS ``dict | str | None`` because the field also accepts a
    # bare id; the validator above has already canonicalised every accepted value
    # to the one-key dict. Narrowing the LOCAL (rather than casting, or blanking
    # the checker) is what keeps the type checker and the runtime reading the
    # SAME value: a str reaching here is refused, never guessed at.
    recipient = params.to
    if recipient is None:
        return params, None
    if not isinstance(recipient, dict):
        return params, _error("dm needs exactly one of account_id or endpoint_id")
    key = next(iter(recipient))
    if key not in ("endpoint_id", "account_id"):
        return params, _error(
            "dm takes an account_id or endpoint_id, not a display name: read "
            "op=directory for the person's account id or their agent's endpoint_id"
        )
    value = recipient[key]
    other = "account_id" if key == "endpoint_id" else "endpoint_id"
    if getattr(params, other).strip():
        return params, _error("dm needs exactly one of account_id or endpoint_id")
    current = getattr(params, key).strip()
    if current and current != value:
        return params, _error(f"dm to and {key} disagree; name the recipient once")
    # A COPY: the caller's params stay exactly as the model wrote them, and the
    # resolver never receives a ``to`` it would have to reinterpret.
    return params.model_copy(update={key: value, "to": None}), None


async def _resolve_dm_target(
    runtime: "BridgeRuntime", params: "BridgeToolParams"
) -> tuple[dict[str, str] | None, str, dict[str, Any] | None]:
    """``(wire member, display name, refusal)`` for a dm's recipient.

    Resolved from a FRESH directory read, never from the model's word: an
    ``endpoint_id`` must be some member's CURRENT chief-of-staff endpoint (so a
    stale or invented id is refused before any write), and an ``account_id``
    with ``target=agent`` maps to that member's current agent. Reachability is
    the directory's own fact: an endpoint that is not ``active`` (paused,
    disabled, or a member who never enrolled) is ``not_reachable`` with the
    reason. A human target needs the person's bridge enabled for the same
    reason: an opted-out person has not chosen to be messaged here.
    """

    account_id = params.account_id.strip()
    endpoint_id = params.endpoint_id.strip()
    if bool(account_id) == bool(endpoint_id):
        return None, "", _error("dm needs exactly one of account_id or endpoint_id")
    if endpoint_id and params.target != "agent":
        return None, "", _error("endpoint_id addresses an agent; a human needs account_id")
    try:
        directory = await runtime.client.directory(
            runtime.tenant_id, expected_account_id=runtime.owner.account_id
        )
    except Exception as failure:
        return (
            None,
            "",
            _error(
                f"could not read the team directory: {_failure_text(failure)}", code="dm_failed"
            ),
        )
    rows = _directory_rows(directory, runtime.owner)
    own_account = str(getattr(runtime.owner, "account_id", "") or "")
    if endpoint_id:
        row = next(
            (r for r in rows if (r.get("chief_of_staff") or {}).get("endpoint_id") == endpoint_id),
            None,
        )
        if row is None:
            return (
                None,
                "",
                _not_reachable(
                    "unknown_endpoint",
                    "no teammate's chief of staff has that endpoint_id; read op=directory again",
                ),
            )
        kind = "agent"
    else:
        row = next((r for r in rows if r["account_id"] == account_id), None)
        if row is None:
            return (
                None,
                "",
                _not_reachable(
                    "not_a_member", "no active member of this organization has that account_id"
                ),
            )
        kind = params.target
    if row["account_id"] == own_account:
        # A DM has two identities; one's own person or agent is not a teammate
        # (the hub refuses it too, rule=self_target). Say so before any write.
        return (
            None,
            "",
            _error("that is you or your own owner; dm reaches other members", code="self_target"),
        )
    if kind == "human":
        if not row["bridge_enabled"]:
            return (
                None,
                "",
                _not_reachable(
                    f"bridge_{row['bridge_state']}",
                    f"{row['name'] or 'that person'} has Team Bridge {_bridge_word(row)}",
                ),
            )
        return {"account_id": row["account_id"], "kind": "human"}, row["name"], None
    agent = row.get("chief_of_staff")
    if not agent:
        return (
            None,
            "",
            _not_reachable(
                f"no_agent_{row['bridge_state']}",
                f"{row['name'] or 'that person'} has no reachable chief of staff "
                f"(bridge {row['bridge_state']}); dm them with target=human instead",
            ),
        )
    if not agent["reachable"]:
        return (
            None,
            "",
            _not_reachable(
                f"agent_{agent['state'] or 'unavailable'}",
                f"{agent['name'] or 'their chief of staff'} is {agent['state'] or 'unavailable'}; "
                "try later or dm the person with target=human",
            ),
        )
    return (
        {"account_id": row["account_id"], "kind": "agent", "endpoint_id": agent["endpoint_id"]},
        f"{agent['name']} ({row['name']})" if agent["name"] else row["name"],
        None,
    )


async def _direct_message(
    runtime: "BridgeRuntime", params: "BridgeToolParams", endpoint_id: str
) -> dict[str, Any]:
    if params.thread_id or params.channel_id or params.members:
        return _error(
            "dm takes account_id, endpoint_id or to; thread_id, channel_id and "
            "members belong to the pair-inbox and room ops"
        )
    params, refused = _dm_recipient(params)
    if refused is not None:
        return refused
    if params.body and params.text:
        return _error("dm takes body or text, not both")
    text = params.body or params.text
    if not text.strip():
        return _error("dm needs non-empty body")
    thread_root_id = params.thread_root_id.strip()
    if thread_root_id and _CONVERSATION_MESSAGE_ID.fullmatch(thread_root_id) is None:
        return _error("dm thread_root_id must be a message id (bcmsg_…)")
    target, label, refusal = await _resolve_dm_target(runtime, params)
    if refusal is not None or target is None:
        return refusal or _error("dm target unresolved")
    creator = {
        "account_id": str(getattr(runtime.owner, "account_id", "") or ""),
        "kind": "agent",
        "endpoint_id": endpoint_id,
    }
    try:
        # Find-or-create: the hub keys a DM on its two sorted member keys, so
        # both directions (A's agent -> B's agent, B's -> A's) land in ONE DM.
        created = await runtime.client.create_conversation(
            {"tenant_id": runtime.tenant_id, "kind": "dm", "members": [creator, target]},
            actor_endpoint_id=endpoint_id,
            origin_session_id=runtime.session_id,
            expected_account_id=runtime.owner.account_id,
        )
    except Exception as failure:
        code = getattr(failure, "code", "")
        if code in ("agent_not_permitted", "member_not_found"):
            return _not_reachable(
                str(code), f"the hub refused that recipient: {_failure_text(failure)}"
            )
        return _error(
            f"the direct message was not opened: {_failure_text(failure)}", code="dm_failed"
        )
    if not isinstance(created, dict):
        return _error("the hub returned no DM view", code="dm_failed")
    nested = created.get("conversation")
    view: dict[str, Any] = nested if isinstance(nested, dict) else created
    conversation_id = str(view.get("id") or "")
    if _CONVERSATION_ID.fullmatch(conversation_id) is None:
        return _error("the hub opened no usable DM id", code="dm_failed")
    # LOOP/COST GUARD, DM HALF (review round 1, m1). A DM is an ordinary
    # conversation with both agents as members, so under the wake contract each
    # `dm` wakes the other agent exactly like a channel post — two agents trading
    # DMs is the same runaway the guard exists for, and an older hub enforces no
    # cap at all. Same limits, same structured refusal as `channel_post`, keyed
    # on the DM's own conversation id; reserved AFTER every local refusal (a
    # malformed or unreachable target never spends budget) and BEFORE the write
    # (a concurrent post of this endpoint sees the slot).
    try:
        slot = reply_guard.reserve_post(
            runtime.config_dir, endpoint_id=endpoint_id, conversation_id=conversation_id
        )
    except reply_guard.ReplyGuardUnavailable:
        return _error(
            "the reply guard could not read its state; not posting",
            code=reply_guard.CODE_UNAVAILABLE,
        )
    if isinstance(slot, reply_guard.Refusal):
        return slot.as_result()
    try:
        result = await runtime.client.conversation_send(
            conversation_id,
            client_ref=str(uuid.uuid4()),
            text=text,
            tenant_id=runtime.tenant_id,
            actor_endpoint_id=endpoint_id,
            origin_session_id=runtime.session_id,
            thread_root_id=thread_root_id or None,
            expected_account_id=runtime.owner.account_id,
        )
    except Exception as failure:
        # The hub did not take it: the budget slot goes back (the run counter
        # does not — see ``reply_guard.release``).
        reply_guard.release(
            runtime.config_dir, slot, endpoint_id=endpoint_id, conversation_id=conversation_id
        )
        return _error(
            f"the DM {conversation_id} opened but the post did not confirm: "
            f"{_failure_text(failure)}",
            code="post_failed",
        )
    receipt = _post_receipt(
        conversation_id, result, thread_root_id=thread_root_id, lead=f"direct message to {label} in"
    )
    receipt["destination_type"] = "conversation_dm"
    # The target's membership state is the server's admission answer: an
    # ``invited`` agent will not be woken until its owner accepts, and the
    # sender must know that rather than assume delivery.
    members = view.get("members") if isinstance(view, dict) else None
    if target["kind"] == "agent" and isinstance(members, list):
        state = next(
            (
                str(m.get("state") or "")
                for m in members
                if isinstance(m, dict)
                and m.get("kind") == "agent"
                and m.get("account_id") == target["account_id"]
            ),
            "",
        )
        if state:
            receipt["recipient_state"] = state
            if state != "active":
                receipt[
                    "content"
                ] += f"; recipient agent is {state} (its owner must accept before it is woken)"
    return receipt


# -- owner-shared conversations (the chief-of-staff read grant) --------------
#
# WHY. The operator asked their own Aida to draft replies to a question in a
# Team Bridge DM and a group Aida was not in; it could not read either ("An
# agent cannot join this conversation"), and owner-audit covers only rooms the
# owner's agent is ALREADY a member of. The fix is an OPERATOR-CONSENTED read:
# the human grants their own chief of staff read access to ONE conversation
# (agent-server ``PUT …/cos-read``, a human act from the desktop; never from
# this tool), and the agent then reads it here.
#
# THE SHAPE, AND WHY PER-CONVERSATION. A blanket "read all my conversations"
# toggle grows silently with every new conversation; a per-conversation grant
# discloses exactly what the human chose, is revocable per room, and the hub
# refuses it on a human-only (agent_capable=false) conversation, keeping §6.1's
# promise that human-only chat never reaches an agent.
#
# SAFETY. Reads only: no op here writes, and nothing can post as the human (the
# post ops all act as the agent endpoint). The hub derives the human from the
# signed-in account and checks the endpoint is that account's own chief of
# staff with an active grant; every refusal is one ``conversation_not_found``.
# Every page is an audit event on the hub (``cos_read``). Bodies are the HUMAN
# view — never screened for this agent — so they are wrapped as untrusted data
# exactly like history, and never cached locally. The desktop bearer and OAuth
# token never enter the model: the call rides the same client the other ops use.


def _shared_read_fence(runtime: "BridgeRuntime") -> dict[str, Any] | None:
    """A human's share belongs to their actual CoS, never another task session.

    Outbound collaboration does not imply permission to read the owner's rooms.
    Require both the durable CoS identity and the existing designated-session
    authority fence; neither a display name nor a model-supplied endpoint grants
    this access.
    """
    from local_operator.aida.state import is_aida_session

    from .authority import authority_fence

    refusal = authority_fence(runtime)
    if refusal is not None:
        return _error(refusal["error"], code=refusal["code"])
    if not is_aida_session(runtime.config_dir, runtime.session_id):
        return _error(
            "owner-shared reads belong only to the chief of staff", code="not_chief_of_staff"
        )
    return None


async def _shared_conversations(
    runtime: "BridgeRuntime", params: "BridgeToolParams", endpoint_id: str
) -> dict[str, Any]:
    refused = _shared_read_fence(runtime)
    if refused:
        return refused
    if not 1 <= int(params.limit) <= CONVERSATION_PAGE_MAX:
        return _error(f"shared_conversations limit must be 1..{CONVERSATION_PAGE_MAX}")
    try:
        result = await runtime.client.cos_read_conversations(
            tenant_id=runtime.tenant_id,
            actor_endpoint_id=endpoint_id,
            expected_account_id=runtime.owner.account_id,
            limit=int(params.limit),
            cursor=params.cursor or None,
        )
    except Exception as failure:
        return _error(
            f"could not list shared conversations: {_failure_text(failure)}", code="list_failed"
        )
    return {
        "ok": True,
        "state": "ok",
        "content": f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(result, ensure_ascii=False)}",
    }


async def _shared_history(
    runtime: "BridgeRuntime", params: "BridgeToolParams", endpoint_id: str
) -> dict[str, Any]:
    refused = _shared_read_fence(runtime)
    if refused:
        return refused
    channel_id = params.channel_id.strip()
    if _CONVERSATION_ID.fullmatch(channel_id) is None:
        return _error("shared_history needs a conversation id (bcv_…)")
    thread_root_id = params.thread_root_id.strip()
    if thread_root_id and _CONVERSATION_MESSAGE_ID.fullmatch(thread_root_id) is None:
        return _error("shared_history thread_root_id must be a message id (bcmsg_…)")
    if params.before_seq is not None and params.after_seq is not None:
        return _error("shared_history takes at most one of before_seq/after_seq")
    if thread_root_id and params.before_seq is not None:
        return _error("a thread reads oldest-first: page with after_seq")
    if not 1 <= int(params.limit) <= CONVERSATION_PAGE_MAX:
        return _error(f"shared_history limit must be 1..{CONVERSATION_PAGE_MAX}")
    try:
        result = await runtime.client.cos_read_messages(
            channel_id,
            tenant_id=runtime.tenant_id,
            actor_endpoint_id=endpoint_id,
            expected_account_id=runtime.owner.account_id,
            thread_root_id=thread_root_id or None,
            limit=int(params.limit),
            before_seq=params.before_seq,
            after_seq=params.after_seq,
        )
    except Exception as failure:
        if getattr(failure, "code", "") == "conversation_not_found":
            return _error(
                "not shared with you: ask your owner to share this conversation with their "
                "chief of staff in Team Bridge (it cannot be granted from here)",
                code="not_shared",
            )
        return _error(
            f"could not read the shared conversation: {_failure_text(failure)}",
            code="history_failed",
        )
    return {
        "ok": True,
        "state": "ok",
        "content": f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(result, ensure_ascii=False)}",
    }


def _posted_message(result: Any) -> tuple[str, int | None]:
    """``(message_id, seq)`` from a conversation-send answer.

    WHY THIS EXISTS. ``POST /messaging/conversations/{id}/messages`` answers
    ``{msg, result: <BridgeConversationMessageView>}`` (agent-server
    ``SendHandlerEcho``), and the client unwraps ``result`` — so the message
    view IS the dict: ``{"id": "bcmsg_…", "seq": 7, …}``. The tool used to
    read ``result["message"]["id"]``, a wrapper the hub never sends, so every
    real post reported ``message ?`` and a sender could not confirm or thread
    on its own message (operator report, 2026-10-28). The nested shape is
    still accepted so a proxy that wraps the view keeps working; anything
    else yields an empty id rather than a guess.
    """

    if not isinstance(result, dict):
        return "", None
    nested = result.get("message")
    view: dict[str, Any] = nested if isinstance(nested, dict) else result
    message_id = view.get("id")
    seq = view.get("seq")
    return (
        message_id if isinstance(message_id, str) else "",
        seq if isinstance(seq, int) and not isinstance(seq, bool) else None,
    )


def _post_receipt(
    conversation_id: str,
    result: Any,
    *,
    thread_root_id: str = "",
    lead: str = "posted to",
    mention_account_id: str = "",
) -> dict[str, Any]:
    """The ONE success receipt for a conversation write (channel_post, dm).

    Carries ``message_id`` and ``seq`` as fields so the sender can confirm the
    post and pass the id back as ``thread_root_id``. A missing id is reported
    honestly as unconfirmed rather than rendered as ``?``.

    ``mention_account_id`` is set only when the write DIRECTED the message at a
    member (a name resolved to exactly one): the receipt then carries
    ``resolved_account_id`` and says which account the post reached. Absent for
    a plain post and for ``dm``, whose destination is the conversation itself —
    the key means "this write named a member", so omitting it when nothing was
    named is the honest shape, not an empty string pretending to be an id.
    """

    message_id, seq = _posted_message(result)
    content = f"{lead} {conversation_id}: " + (
        f"message {message_id}" if message_id else "accepted, but the hub returned no message id"
    )
    if thread_root_id:
        content += f" (reply in thread {thread_root_id})"
    if seq is not None:
        content += f" at seq {seq}"
    if mention_account_id:
        content += f" (mentioning {mention_account_id})"
    receipt: dict[str, Any] = {
        "ok": True,
        "state": "sent",
        "conversation_id": conversation_id,
        "message_id": message_id,
        "seq": seq,
        "destination_type": "conversation",
        "conversation_history": True,
        "receipt_stage": "server_accepted",
        "execution_confirmed": False,
        "content": content,
    }
    if mention_account_id:
        receipt["resolved_account_id"] = mention_account_id
    return receipt


def _full_body_text(result: Any) -> str | None:
    message = (result or {}).get("message") or {}
    body = message.get("body") or {}
    text = body.get("text")
    return text if isinstance(text, str) else None


def body_digest(text: str) -> str:
    """The same construction the wake path stamps (§11.4/§13.4, S-20):
    sha256 hex over the UTF-8 body text (``connector/wake.py`` builds it the
    same way; if that ever gains a helper, this delegates to it rather than
    drifting)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _channel_members_with_creator(
    members: list[ChannelMemberParams], owner: Any
) -> list[dict[str, str]]:
    """The wire ``members[]`` with THIS agent always present.

    The server refuses an actor create whose creator agent is not named
    (``rule=actor_required``) because it will not invent the one member whose
    identity the act is. So the tool adds the creator when the model omitted
    it; an explicit creator entry is preserved as-is (its display label is a
    request fact, not ours to overwrite). The human owner is NOT synthesized:
    an agent-created channel has no human member unless one was listed.
    """

    account_id = str(getattr(owner, "account_id", "") or "")
    endpoint_id = str(getattr(owner, "endpoint_id", "") or "").strip()
    out: list[dict[str, str]] = []
    creator_present = False
    for member in members:
        # WHITESPACE IS ABSENT: the validators already strip for their checks,
        # so the serializer must too, or "   " rides the wire as an endpoint
        # the server will try (and fail) to bind.
        member_endpoint = member.endpoint_id.strip()
        entry: dict[str, str] = {"account_id": member.account_id, "kind": member.kind}
        if member_endpoint:
            entry["endpoint_id"] = member_endpoint
        if member.display_label:
            entry["display_label"] = member.display_label
        if (
            member.kind == "agent"
            and member.account_id == account_id
            and member_endpoint == endpoint_id
        ):
            creator_present = True
        out.append(entry)
    if not creator_present:
        out.append(
            {
                "account_id": account_id,
                "kind": "agent",
                "endpoint_id": endpoint_id,
                "display_label": CHANNEL_CREATE_DEFAULT_LABEL,
            }
        )
    return out


def _validate_reply_destination(
    runtime: BridgeRuntime, args: dict[str, Any]
) -> dict[str, Any] | None:
    """Bind a declared reply before admission I/O, member lookup or budget.

    A thread root is presentation, not evidence of which inbound delivery is
    being answered. The explicit new-post escape hatch states intent only: a
    model can misdeclare it, so this is not a semantic noninterference claim.
    """
    from .store import Store, StoreRefused

    message_id = args.get("in_reply_to_message_id", "")
    new_post = args.get("new_post", False)
    if not isinstance(message_id, str) or not isinstance(new_post, bool):
        return _error("reply binding needs a message id string and boolean new_post")
    message_id = message_id.strip()
    if message_id and new_post:
        return _error(
            "new_post contradicts in_reply_to_message_id; nothing was sent",
            code="reply_binding_conflict",
        )
    if not message_id:
        if runtime.bridge_input_presented and not new_post:
            return _error(
                "this run received bridge input: supply in_reply_to_message_id "
                "or intentionally declare new_post; thread_root_id is not reply binding",
                code="reply_binding_required",
            )
        return None
    if _CONVERSATION_MESSAGE_ID.fullmatch(message_id) is None:
        return _error("in_reply_to_message_id must be an admitted message id (bcmsg_…)")
    try:
        row = Store(Path(runtime.config_dir)).admitted_inbound_reservation_readonly(
            runtime.owner,
            in_reply_to_message_id=message_id,
            designated_session=runtime.session_id,
        )
    except (StoreRefused, OSError, UnicodeDecodeError):
        return _error(
            "reply binding could not be verified; nothing was sent",
            code="reply_binding_unavailable",
        )
    if row is None:
        return _error(
            "no unique current admitted reply binding; nothing was sent",
            code="reply_binding_unavailable",
        )
    channel_id = args.get("channel_id", "")
    if not isinstance(channel_id, str):
        return _error("channel_post needs a conversation id (bcv_…)")
    if row["conversation_id"] != channel_id.strip():
        return _error(
            "in_reply_to_message_id belongs to a different conversation; nothing was sent",
            code="reply_destination_mismatch",
        )
    return None


async def execute_bridge(runtime: BridgeRuntime, args: dict[str, Any]) -> dict[str, Any]:
    """Run one tool op. Returns a JSON-ish dict; the AgentTool wrapper turns it
    into a ``ToolResult`` (and the honest error shape into ``is_error``)."""

    op = str(args.get("op") or "")
    # The decide word gets its OWN refusal, before ANY gate, so an endpoint-less
    # runtime that invents it still hears the §10.3 sentence rather than a
    # channel-identity refusal (the schema also omits it).
    if op == "decide":
        return _error(
            "decisions are made by the human through the ask queue — this tool has "
            "no decide authority (contract §10.3, #125)",
            code="not_permitted",
        )
    if op and op not in BRIDGE_OPS:
        return _error(f"unknown bridge op {op!r}", code="invalid_request")

    # CHANNEL IDENTITY, before the gates: these ops act as an AGENT member,
    # and a runtime with no owner endpoint has no agent identity to act as.
    # This is a refusal, never a fallback to the human lane — and it only
    # touches runtime state, so it precedes the network-free fence safely.
    endpoint_id = str(getattr(runtime.owner, "endpoint_id", "") or "").strip()
    if op in CHANNEL_OPS and not endpoint_id:
        return _error(
            "channel operations act as this session's enrolled agent, and this "
            "runtime has no endpoint — re-enroll before using channels",
            code="no_agent_endpoint",
        )

    # LOCAL AUTHORITY FENCE, FIRST (S-1): enabled / unique committed owner /
    # live non-empty generation / consent / exact base-account-tenant-endpoint-
    # credential-row binding / tenant consistency / native designation. It runs
    # BEFORE anything may be scheduled: a drifted or disabled runtime performs
    # NO status fetch (no admit_soon), NO op and NO mutation — typed refusal.
    from .host import (
        admission_refusal,
        admit_soon,
        outbound_tool_fence,
        owner_fence_refusal,
    )

    # WHICH FENCE — see OUTBOUND_OPS. The collaboration ops take the relaxed
    # fence; the mailbox ops and the owner-shared reads keep the strict one, so
    # a colleague's task session can reach teammates without ever acquiring the
    # owner's inbox or their shared conversations.
    fence = outbound_tool_fence(runtime) if op in OUTBOUND_OPS else owner_fence_refusal(runtime)
    if fence is not None:
        return _error(fence["error"], code=fence["code"])
    if op == "channel_post":
        refused = _validate_reply_destination(runtime, args)
        if refused is not None:
            return refused
    # Refuse unsupported destinations before even scheduling admission I/O.
    # The local authority fence keeps precedence; no invalid destination can
    # create an outbox intent, send a message, or trigger a status request.
    #
    # A NAME destination is the one shape the SHAPE check cannot finish here:
    # turning a display name into an account id needs the bridge directory and
    # the owner's own allowlist, which are wire reads and therefore sit after the
    # admission gate. The shape is still checked now (cheaply, before any I/O),
    # and the pair lane's reading of it is enforced at both ends: a name is
    # resolved below, an id/endpoint is validated exactly where it always was.
    recipient: dict[str, str] = {}
    named_recipient = ""
    if op == "reply":
        # Resolution reads only the local store, so it sits between the fence
        # and the recipient check (R9): an unresolvable reply refuses here,
        # before admission could schedule any status read.
        args, refused = _resolve_reply(runtime, args)
        if refused is not None:
            return refused
    if op in ("send", "reply"):
        try:
            shape = validate_recipient_shape(args.get("to"))
        except InvalidSendRecipient as failure:
            return _error(str(failure), code=failure.code)
        if "name" in shape:
            named_recipient = shape["name"]
        else:
            recipient = shape
    # ADMISSION GATE, second: only a SETTLED "admitted" may act. A non-admitted
    # state refuses THIS op typed and schedules the same single-flight boundary
    # read (fire-and-forget — never waits, never posts). ``take_up`` is a FETCH
    # (S-9): the paused/read-only postures refuse its MUTATING neighbours but
    # admit the read, while disabled/signin/pending refuse both directions. The
    # channel reads (list/history) are fetches for the same reason; create and
    # post are mutations.
    is_read = op in READ_OPS
    refusal = admission_refusal(runtime, is_read=is_read)
    if refusal is not None:
        admit_soon(runtime)
        return _error(refusal["error"], code=refusal["code"])

    if named_recipient:
        # A name becomes an address HERE: after the admission gate (its reads are
        # wire calls) and before anything is recorded or sent. The args carry the
        # resolved PAIR from this point on, so every downstream step — the params
        # model, the reply's own sender check, the durable outbox intent, the
        # wire body — sees exactly what an explicit account id would have given
        # it. Nothing downstream has to know a name was ever involved.
        resolved_to, unresolved = await _resolve_named_recipient(runtime, named_recipient)
        if unresolved is not None or resolved_to is None:
            # The helper returns a pair or a refusal. A None PAIR with no refusal
            # would be a bug in it, and an `assert` here would be stripped under
            # `-O` into an outbox intent whose `to` is None — so the unreachable
            # case is a typed refusal, the same way every other unknown is.
            return unresolved or _error(
                "that recipient name did not resolve to an address; nothing was sent"
            )
        args = {**args, "to": resolved_to}
        recipient = resolved_to
        if op == "reply":
            # The pair exists, so the pair lane's own local check runs now — the
            # answer must address the sender of the message it answers.
            args, refused = _resolve_reply(runtime, args)
            if refused is not None:
                return refused

    try:
        params = BridgeToolParams(**args)
    except Exception as failure:  # pydantic ValidationError, kept dependency-free
        return _error(f"bridge args invalid: {failure}", code="invalid_request")

    if params.op == "take_up":
        if not params.message_id or not params.body_sha256:
            return _error("take_up needs message_id and body_sha256 (S-20)")
        if is_conversation_message_id(params.message_id):
            # A ``bcmsg_`` id lives in the conversation store; the pair
            # read below would answer "does not exist" and the model would
            # conclude the message vanished. The conversation lane admits its
            # body through the connector, and the agent reads it from history.
            return _error(
                "conversation messages (bcmsg_ ids) are not taken up here: read them with "
                "channel_history or channel_thread",
                code="conversation_message",
            )
        try:
            result = await runtime.client.message(params.message_id, tenant_id=runtime.tenant_id)
        except Exception as failure:  # BridgeRefusal/BridgeUnreachable -> model-readable
            return _error(
                f"could not fetch the message: {_failure_text(failure)}", code="fetch_failed"
            )
        text = _full_body_text(result)
        if text is None:
            return _error(
                "the message carried no readable body text; refusing to hand over "
                "anything unverified",
                code="malformed_message",
            )
        if body_digest(text) != params.body_sha256:
            # S-20: this is NOT the screened message (or not the same bytes).
            # Hold: no ack, no hand-over.
            return _error(
                "body hash mismatch — this is not the screened message; held for "
                "re-check, nothing was acked",
                code="body_hash_mismatch",
            )
        return {"ok": True, "message_id": params.message_id, "text": text}

    if params.op == "ack":
        if not params.message_id or params.state is None:
            return _error("ack needs message_id and state (working|done)")
        import time as _time

        from .store import Store, StoreRefused

        store = Store(Path(runtime.config_dir))
        if params.state == "working":
            token = str(uuid.uuid4())
            try:
                store.claim(
                    runtime.owner,
                    params.message_id,
                    runtime.session_id,
                    token,
                    _time.time() + CLAIM_TTL_S,
                )
            except StoreRefused as failure:
                if failure.code == "claimed":
                    return _error(
                        "another session holds this message; do not duplicate the work",
                        code="claimed",
                    )
                return _error(f"claim refused: {failure}", code=failure.code)
            runtime.claims[params.message_id] = token
            return {"ok": True, "state": "working", "message_id": params.message_id}
        token = runtime.claims.pop(params.message_id, None)
        if token is None:
            return _error(
                "no claim held by this session for that message; 'done' releases only "
                "this session's own claim",
                code="not_holder",
            )
        try:
            store.release_claim(runtime.owner, params.message_id, token)
        except StoreRefused as failure:
            return _error(f"release refused: {failure}", code=failure.code)
        return {"ok": True, "state": "done", "message_id": params.message_id}

    if params.op in ("send", "reply"):
        if not params.text.strip():
            return _error("send/reply need non-empty text")
        if params.op == "reply" and not (params.thread_id or "").strip():
            return _error(
                "a reply names thread_id AND to together (§13.5 — thread_id never travels alone)"
            )
        outbox = Outbox(runtime.config_dir)
        intent = outbox.record_send(
            runtime.key(),
            to=recipient,
            type=params.type,
            text=params.text,
            thread_id=params.thread_id,
            expires_at=params.expires_at,
        )
        try:
            result = await send_from_outbox(
                runtime.client, outbox, intent, tenant_id=runtime.tenant_id
            )
        except Exception as failure:
            return _error(
                f"the send did not confirm: {_failure_text(failure)}", code="send_unconfirmed"
            )
        message = (result or {}).get("message") or {}
        # Keep the legacy fields/wire semantics, but make their limited meaning
        # machine-readable in the actual result (including an idempotent reply).
        # Acceptance is only into the peer inbox, never conversation history or
        # proof of execution: even a read ACK precedes the local wake handoff.
        # These are result-only constants, not request/body fields or a new
        # outbox schema. Errors above must never acquire this success receipt.
        return {
            "ok": True,
            "message_id": str(message.get("id") or ""),
            "state": "sent",
            "destination_type": "peer_inbox",
            "receipt_stage": "server_accepted",
            "conversation_history": False,
            "execution_confirmed": False,
            # The model reads this JSON receipt verbatim (``_execute``), so the
            # lane boundary is said in words too: a pair send is invisible in
            # every Team Bridge conversation view, and agents kept mistaking it
            # for a DM (2026-10-28). One path per intent: dm / channel_post.
            "note": (
                f"Accepted into {_recipient_label(recipient)}'s peer inbox (legacy pair "
                "lane). NOT posted in any conversation and not shown in Team Bridge "
                "conversations. For a visible direct message use op=dm; for a room use "
                "channel_post."
            ),
        }

    if params.op == "channels":
        if not 1 <= int(params.limit) <= CONVERSATION_PAGE_MAX:
            return _error(f"channels limit must be 1..{CONVERSATION_PAGE_MAX}")
        try:
            # No ``status``: the server's default is ``open``, so archived
            # conversations are intentionally invisible to the agent (module
            # docstring, "channels", for why it is not a published option).
            result = await runtime.client.conversations(
                tenant_id=runtime.tenant_id,
                limit=int(params.limit),
                cursor=params.cursor or None,
            )
        except Exception as failure:
            return _error(
                f"could not list conversations: {_failure_text(failure)}", code="list_failed"
            )
        # Pass-through, deliberately: ``discoverable``/``member``/``member_count``
        # are the server's answer for a caller who may not be a human member, and
        # re-deriving them here would be a second, wrong authorization opinion.
        return {
            "ok": True,
            "state": "ok",
            "content": json.dumps(result, ensure_ascii=False),
        }

    if params.op == "channel_create":
        if params.kind is None:
            return _error("channel_create needs kind (group or channel)")
        if any(not member.account_id.strip() for member in params.members):
            return _error("every channel member needs an account_id")
        if any(
            member.kind == "agent" and not member.endpoint_id.strip() for member in params.members
        ):
            return _error("an agent member needs its endpoint_id")
        # The server keys a member by endpoint when present, ignoring kind, so a
        # human entry carrying the actor endpoint would collide with (and be
        # promoted into) the creator agent row. Refuse it at the boundary.
        if any(member.kind == "human" and member.endpoint_id.strip() for member in params.members):
            return _error("a human member must not carry an endpoint_id")
        body: dict[str, Any] = {
            "tenant_id": runtime.tenant_id,
            "kind": params.kind,
            "members": _channel_members_with_creator(params.members, runtime.owner),
        }
        if params.name.strip():
            body["name"] = params.name.strip()
        if params.topic.strip():
            body["topic"] = params.topic.strip()
        if params.visibility is not None:
            body["visibility"] = params.visibility
        try:
            result = await runtime.client.create_conversation(
                body,
                actor_endpoint_id=endpoint_id,
                origin_session_id=runtime.session_id,
            )
        except Exception as failure:
            return _error(
                f"the channel was not created: {_failure_text(failure)}", code="create_failed"
            )
        view = result if isinstance(result, dict) else {}
        nested = view.get("conversation")
        if isinstance(nested, dict):
            view = nested
        summary = {
            key: view[key]
            for key in ("id", "kind", "name", "topic", "visibility", "member_count")
            if key in view
        }
        return {
            "ok": True,
            "state": "created",
            "conversation_id": str(view.get("id") or ""),
            "content": json.dumps(summary, ensure_ascii=False),
        }

    if params.op == "directory":
        return await _directory(runtime, params)

    if params.op == "dm":
        return await _direct_message(runtime, params, endpoint_id)

    if params.op == "shared_conversations":
        return await _shared_conversations(runtime, params, endpoint_id)

    if params.op == "shared_history":
        return await _shared_history(runtime, params, endpoint_id)

    if params.op == "channel_post":
        channel_id = params.channel_id.strip()
        if _CONVERSATION_ID.fullmatch(channel_id) is None:
            return _error("channel_post needs a conversation id (bcv_…)")
        if not params.body.strip():
            return _error("channel_post needs non-empty body")
        thread_root_id = params.thread_root_id.strip()
        if thread_root_id and _CONVERSATION_MESSAGE_ID.fullmatch(thread_root_id) is None:
            return _error("channel_post thread_root_id must be a message id (bcmsg_…)")
        # WHO this post is directed at, resolved BEFORE anything is posted. The
        # candidate set is THIS conversation's active members — the least
        # privilege rule of this slice: a name from elsewhere in the organization
        # is not a candidate, so it cannot resolve and cannot be delivered to.
        # A resolved name never reaches the wire as a name; only its ids do.
        mentions: list[dict[str, str]] = []
        recipient_account_id = ""
        if params.to:
            # The ANNOTATION on ``params.to`` is the published union (object | bare
            # id), which is what the type checker sees; the field validator has
            # already canonicalised the value, and re-stating the shape here is
            # the SAME rule read once more rather than a second one. It also
            # keeps a bare id from being treated as a name by accident.
            recipient = validate_recipient_shape(params.to)
            if "name" not in recipient:
                # An id is not a name: the directory read that produces ids is
                # the caller's, and a name is the form a human actually has.
                return _error(
                    "channel_post addresses a member by display name "
                    f"({RECIPIENT_NAME_SHAPE}); endpoint_id/account_id belong to send/reply"
                )
            try:
                conversation = await runtime.client.conversation(
                    channel_id, tenant_id=runtime.tenant_id
                )
            except Exception as failure:
                # No candidate set exists, so no resolution ran — but the ATTEMPT
                # did, and every attempt is recorded. The outcome word is
                # ``unavailable`` rather than a derived ``no_match``, because
                # nobody was consulted; ``no_match`` would say the members were
                # read and none matched, which is a different fact.
                try:
                    record_resolution(
                        config_dir=runtime.config_dir,
                        scope=SCOPE_CONVERSATION,
                        scope_id=channel_id,
                        resolution=NameResolution(requested=recipient["name"].strip(), matches=()),
                        outcome="unavailable",
                    )
                except ResolutionAuditUnavailable as unaudited:
                    return _error(str(unaudited), code=unaudited.code)
                return _error(
                    f"could not read this conversation's members: {failure}",
                    code="members_failed",
                )
            try:
                member = resolve_for_delivery(
                    config_dir=runtime.config_dir,
                    scope=SCOPE_CONVERSATION,
                    scope_id=channel_id,
                    requested_name=recipient["name"],
                    candidates=members_from_conversation(conversation),
                )
            except RecipientNameRefused as failure:
                # The attempt is ALREADY recorded (resolve_for_delivery writes the
                # row before the match is acted on), so a refusal leaves the same
                # reconstructable record a delivery does — this is where a later
                # "it went to the wrong colleague" claim gets answered from.
                return _error(str(failure), code=failure.code)
            except ResolutionAuditUnavailable as failure:
                return _error(str(failure), code=failure.code)
            mentions = [member.mention()]
            recipient_account_id = member.account_id
        # Member resolution awaited remote I/O after the early binding check.
        # Revocation during that wait must refuse before even spending budget.
        # Keep this cheap early refusal, then repeat at each actual pre-HTTP
        # boundary below: OAuth/refresh may await after the tool's handoff.
        fence = outbound_tool_fence(runtime)
        if fence is not None:
            return _error(fence["error"], code=fence["code"])
        refused = _validate_reply_destination(runtime, args)
        if refused is not None:
            return refused
        # LOOP/COST GUARD (wake contract): every conversation message now wakes
        # its agent members, so two agents answering every wake would trade
        # posts for ever. Reserved LAST among the local checks (a malformed or
        # unresolvable call never spends budget) and BEFORE the wire write (a
        # concurrent post of the same endpoint sees the slot). Fail closed: a
        # guard that cannot read its state refuses rather than going silent.
        slot = None

        class DispatchRefused(Exception):
            def __init__(self, result: dict[str, Any]) -> None:
                self.result = result

        def pre_http_guard() -> None:
            nonlocal slot
            fence = outbound_tool_fence(runtime)
            refusal = (
                _error(fence["error"], code=fence["code"])
                if fence is not None
                else _validate_reply_destination(runtime, args)
            )
            if refusal is not None:
                raise DispatchRefused(refusal)
            # Reserve only when the first physical attempt is authorized.
            # Releasing an early slot would still consume the run counter;
            # retries instead revalidate and reuse the same slot/client_ref.
            if slot is None:
                try:
                    reserved = reply_guard.reserve_post(
                        runtime.config_dir, endpoint_id=endpoint_id, conversation_id=channel_id
                    )
                except reply_guard.ReplyGuardUnavailable:
                    raise DispatchRefused(
                        _error(
                            "the reply guard could not read its state; not posting",
                            code=reply_guard.CODE_UNAVAILABLE,
                        )
                    ) from None
                if isinstance(reserved, reply_guard.Refusal):
                    raise DispatchRefused(reserved.as_result())
                slot = reserved

        # ONE ref per CALL, minted before dispatch: the client's own 401 refresh
        # retry re-sends this exact body, so that retry is idempotent on the
        # server; a second tool call mints a second ref and is a second message.
        client_ref = str(uuid.uuid4())
        try:
            result = await runtime.client.conversation_send(
                channel_id,
                client_ref=client_ref,
                text=params.body,
                tenant_id=runtime.tenant_id,
                actor_endpoint_id=endpoint_id,
                origin_session_id=runtime.session_id,
                thread_root_id=thread_root_id or None,
                mentions=mentions or None,
                pre_http_guard=pre_http_guard,
            )
        except DispatchRefused as failure:
            if slot is not None:
                reply_guard.release(
                    runtime.config_dir, slot, endpoint_id=endpoint_id, conversation_id=channel_id
                )
            return failure.result
        except Exception as failure:
            # The hub did not take it: the budget slot goes back. (The run
            # counter does not — see ``reply_guard.release``.)
            if slot is not None:
                reply_guard.release(
                    runtime.config_dir, slot, endpoint_id=endpoint_id, conversation_id=channel_id
                )
            return _error(f"the post did not confirm: {_failure_text(failure)}", code="post_failed")
        # ONE receipt shape for every conversation write (main's `_post_receipt`,
        # shared with `dm`), EXTENDED with the resolved recipient rather than
        # rebuilt here: a second construction site is how the two lanes drift.
        # ``mention_account_id`` is the id the post was actually directed at, so
        # the confirmation names a fact instead of restating the caller's ask.
        return _post_receipt(
            channel_id,
            result,
            thread_root_id=thread_root_id,
            mention_account_id=recipient_account_id,
        )

    if params.op == "react":
        channel_id = params.channel_id.strip()
        if _CONVERSATION_ID.fullmatch(channel_id) is None:
            return _error("react needs a conversation id (bcv_…)")
        if _CONVERSATION_MESSAGE_ID.fullmatch(params.message_id) is None:
            return _error("react needs a conversation message id (bcmsg_…)")
        emoji = unicodedata.normalize("NFC", params.emoji)
        if not emoji or len(emoji.encode("utf-8")) > 128:
            return _error("react needs one emoji, at most 128 UTF-8 bytes")
        # No body fetch, local cache, or implicit admission here: the server
        # requires this exact endpoint's prior allow verdict before a reaction
        # can act on a message. It is a normal write at both admission and tool
        # approval boundaries, never an ack-shaped bypass.
        try:
            result = await runtime.client.conversation_react(
                channel_id,
                params.message_id,
                emoji,
                active=not params.remove,
                tenant_id=runtime.tenant_id,
                actor_endpoint_id=endpoint_id,
            )
        except Exception as failure:
            return _error(
                f"the reaction did not confirm: {_failure_text(failure)}", code="react_failed"
            )
        return {
            "ok": True,
            "state": "removed" if params.remove else "reacted",
            "message_id": params.message_id,
            # Labels in reaction summaries are member-authored external data,
            # just like a history sender label: keep the untrusted framing.
            "content": f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(result, ensure_ascii=False)}",
        }

    if params.op == "channel_history":
        channel_id = params.channel_id.strip()
        if _CONVERSATION_ID.fullmatch(channel_id) is None:
            return _error("channel_history needs a conversation id (bcv_…)")
        if params.before_seq is not None and params.after_seq is not None:
            return _error(
                "channel_history takes at most one of before_seq/after_seq "
                "(before is exclusive, after inclusive; never combined)"
            )
        if not 1 <= int(params.limit) <= CONVERSATION_PAGE_MAX:
            return _error(f"channel_history limit must be 1..{CONVERSATION_PAGE_MAX}")
        try:
            result = await runtime.client.conversation_messages(
                channel_id,
                tenant_id=runtime.tenant_id,
                limit=int(params.limit),
                before_seq=params.before_seq,
                after_seq=params.after_seq,
                endpoint_id=endpoint_id,
            )
        except Exception as failure:
            return _error(
                f"could not read the channel history: {_failure_text(failure)}",
                code="history_failed",
            )
        # Bodies are external content even when the sender is a teammate's
        # agent: the same untrusted wrap as take_up. Unadmitted messages carry
        # ``body: null`` — metadata only — and stay that way (no local cache).
        if isinstance(result, dict) and isinstance(result.get("messages"), list):
            restored = await _restore_own_bodies(
                runtime, channel_id, result["messages"], endpoint_id
            )
        else:
            restored = set()
        images = await _history_images(runtime, channel_id, result, endpoint_id, restored)
        return {
            "ok": True,
            "state": "ok",
            "content": f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(result, ensure_ascii=False)}",
            _IMAGES_KEY: images,
        }

    if params.op == "channel_thread":
        channel_id = params.channel_id.strip()
        if _CONVERSATION_ID.fullmatch(channel_id) is None:
            return _error("channel_thread needs a conversation id (bcv_…)")
        thread_root_id = params.thread_root_id.strip()
        if _CONVERSATION_MESSAGE_ID.fullmatch(thread_root_id) is None:
            return _error("channel_thread needs thread_root_id, a message id (bcmsg_…)")
        if params.before_seq is not None:
            return _error("channel_thread reads oldest-first: page with after_seq (inclusive)")
        if not 1 <= int(params.limit) <= CONVERSATION_PAGE_MAX:
            return _error(f"channel_thread limit must be 1..{CONVERSATION_PAGE_MAX}")
        try:
            result = await runtime.client.conversation_replies(
                channel_id,
                thread_root_id,
                tenant_id=runtime.tenant_id,
                limit=int(params.limit),
                after_seq=params.after_seq,
                endpoint_id=endpoint_id,
            )
        except Exception as failure:
            return _error(
                f"could not read the thread: {_failure_text(failure)}", code="thread_failed"
            )
        # Same untrusted wrap as channel_history: a reply body is external
        # content, and an unadmitted one is already null from the server.
        # Images too (R1-02): the channel timeline excludes replies, so this
        # is the ONLY read through which a reply's admitted images can reach
        # the agent — same caption, same fresh context fetch, same budget.
        thread_rows: list[Any] = []
        if isinstance(result, dict):
            root = result.get("root")
            if isinstance(root, dict):
                thread_rows.append(root)
            replies = result.get("replies")
            if isinstance(replies, list):
                thread_rows.extend(replies)
        restored = await _restore_own_bodies(runtime, channel_id, thread_rows, endpoint_id)
        images = await _history_images(
            runtime, channel_id, {"messages": thread_rows}, endpoint_id, restored
        )
        return {
            "ok": True,
            "state": "ok",
            "content": f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(result, ensure_ascii=False)}",
            _IMAGES_KEY: images,
        }

    # The decide/unknown refusals above run before the gates; this later return
    # is the schema-valid op that lost its branch — a programmer error, not a
    # model input.
    return _error(
        f"unknown bridge op {params.op!r}",
        code="invalid_request",
    )


def _screener_sees_images(runtime: BridgeRuntime) -> bool:
    """Whether this machine's screening model accepts images (the connector's rule)."""

    from .connector.conversation import _screener_sees_images as sees

    try:
        return sees(Path(runtime.config_dir))
    except Exception:  # noqa: BLE001 — unknown reads as "no vision", never a crash
        return False


def _image_provenance(channel_id: str, message: dict[str, Any], lines: list[str]) -> str:
    """The caption leading one message's images: clause first, then provenance.

    Every image is sender-controlled content exactly like a body — a picture
    can carry instructions as easily as text — so the ONE untrusted clause
    leads it, and the provenance (who, where, which attachment, how big) lets
    the agent say what it is looking at. No hashes and no filenames.
    """

    raw_sender = message.get("sender")
    sender: dict[str, Any] = raw_sender if isinstance(raw_sender, dict) else {}
    label = str(sender.get("display_label") or "member").replace("\n", " ")[:200]
    kind = str(sender.get("kind") or "unknown")[:16]
    head = (
        f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n"
        f"Images from message {message.get('id')} in {channel_id} "
        f"(seq {message.get('seq')}) from {label} ({kind}):"
    )
    return "\n".join([head, *lines])


#: Most own-message bodies one history/thread read will fetch. The page itself
#: is bounded by ``CONVERSATION_PAGE_MAX``; this keeps a page of the agent's own
#: posts from becoming a hundred sequential reads.
OWN_BODY_FETCH_MAX = 25


async def _restore_own_bodies(
    runtime: BridgeRuntime, channel_id: str, rows: list[Any], endpoint_id: str
) -> set[str]:
    """Fill ``body`` on the agent rows THIS account authored, in place.

    Returns the ids it filled, so the image caption can tell a restored own
    post (never screened) from an admitted one.

    WHY (the sender could not read its own write): the hub serves a history body
    only for a message THIS endpoint admitted, and an endpoint admits a message
    by screening it as an INBOUND delivery (``connector/conversation.py``, the
    only caller of ``conversation_admit``). A message this account posted is
    never delivered back to it, so it is never admitted and reads back
    ``body: null`` with only its hash, forever.

    WHAT THIS DOES NOT DO: it does not weaken screening. The ``/body`` route the
    connector uses for classification is asked ONLY for rows whose
    ``sender.account_id`` is the runtime's own account AND ``sender.kind`` is
    ``agent``: text this account's agent wrote is the owner's own authority, and
    everyone else's stays ``null`` until the connector screens it. No local copy
    is made: the text is fetched per read, handed to the model and dropped.

    DELIBERATE, ACCOUNT-WIDE TRUST: the hub's history ``sender`` carries no
    endpoint id, so this cannot tell THIS endpoint's posts from another agent
    endpoint of the same account (which IS delivered here and screened, and so
    could be held or rejected). Such a row is treated as the owner's own agent.
    Narrowing it to this endpoint needs ``sender.endpoint_id`` from the hub.
    The ``agent`` kind keeps the account's HUMAN rows (always screened for
    this agent) out of it. At most :data:`OWN_BODY_FETCH_MAX` rows are filled per
    read, newest first; older own rows stay ``null``.
    A row is filled only when the served body hashes to the row's own
    ``body_hash`` (sha256 over the UTF-8 text, as the connector verifies); any
    failure — transport, refusal, mismatch — leaves the row exactly as the hub
    sent it.
    """

    import asyncio

    own = getattr(runtime.owner, "account_id", "")
    if not own:
        return set()
    wanted = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("body") is None
        and isinstance(row.get("body_hash"), str)
        and isinstance(row.get("id"), str)
        and _CONVERSATION_MESSAGE_ID.fullmatch(row["id"]) is not None
        and isinstance(row.get("sender"), dict)
        and row["sender"].get("account_id") == own
        and row["sender"].get("kind") == "agent"
    ][:OWN_BODY_FETCH_MAX]
    if not wanted:
        return set()
    gate = asyncio.Semaphore(5)

    async def fill(row: dict[str, Any]) -> None:
        async with gate:
            try:
                served = await runtime.client.conversation_body(
                    channel_id,
                    row["id"],
                    tenant_id=runtime.tenant_id,
                    endpoint_id=endpoint_id,
                )
            except Exception:  # noqa: BLE001 — the hub's null stands; never a crash
                return
        if not isinstance(served, dict) or served.get("message_id") != row["id"]:
            return
        text = served.get("body")
        try:
            verified = isinstance(text, str) and body_digest(text) == row["body_hash"]
        except UnicodeError:  # a lone surrogate cannot be hashed: not the screened text
            return
        if verified:
            row["body"] = text
            filled.add(row["id"])

    filled: set[str] = set()
    await asyncio.gather(*(fill(row) for row in wanted))
    return filled


async def _history_images(
    runtime: BridgeRuntime,
    channel_id: str,
    page: Any,
    endpoint_id: str,
    restored: set[str] | None = None,
) -> list[tuple[str, list[tuple[bytes, str]]]]:
    """Per admitted message with images: ``(caption, [(bytes, mime), ...])``.

    WHY A FRESH FETCH. The hub lists attachment descriptors to an agent only
    after an ``allow`` bound to the message's ``attachments_hash``, and serves
    ``purpose=context`` bytes under that same predicate (decision D1). These
    bytes are fetched HERE, at read time, every time: never from the
    connector's screening fetch, never cached in the team-bridge store,
    sqlite, disk or a log. A refusal (text-only allow, reject, expired) is a
    line in the caption, never a silent gap.
    """

    from .attachments import (
        AttachmentRejected,
        parse_descriptors,
        verify_and_bound,
        withheld_marker,
    )

    if not isinstance(page, dict) or not isinstance(page.get("messages"), list):
        return []
    out: list[tuple[str, list[tuple[bytes, str]]]] = []
    budget = HISTORY_IMAGES_MAX
    for message in page["messages"]:
        if not isinstance(message, dict) or not isinstance(message.get("id"), str):
            continue
        descriptors = parse_descriptors(message.get("attachments"))
        count = message.get("attachment_count")
        if not descriptors:
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                # Images exist but were not admitted to this agent's context
                # (screened text-only, rejected, or pending). When the text
                # WAS admitted and this machine's screener has no vision, say
                # exactly that (D5) — it is the one cause the user can fix.
                noun = "attachment" if count == 1 else "attachments"
                marker = f"[{count} image {noun} withheld: not admitted by screening]"
                if (
                    message.get("body") is not None
                    and message["id"] not in (restored or ())
                    and not _screener_sees_images(runtime)
                ):
                    marker = withheld_marker(count)
                out.append((_image_provenance(channel_id, message, [marker]), []))
            elif descriptors is None:
                out.append(
                    (
                        _image_provenance(
                            channel_id, message, ["[images withheld: could not be verified]"]
                        ),
                        [],
                    )
                )
            continue
        lines: list[str] = []
        blocks: list[tuple[bytes, str]] = []
        for index, descriptor in enumerate(descriptors, start=1):
            meta = (
                f"{index}. attachment {descriptor.id}: {descriptor.mime} "
                f"{descriptor.width}x{descriptor.height}, {descriptor.size} bytes"
            )
            if not descriptor.fetchable:
                lines.append(f"{meta} — [image expired]")
                continue
            if budget <= 0:
                lines.append(f"{meta} — [not shown: image limit per read reached; page narrower]")
                continue
            try:
                data, _mime = await runtime.client.fetch_attachment(
                    channel_id,
                    message["id"],
                    descriptor.id,
                    tenant_id=runtime.tenant_id,
                    endpoint_id=endpoint_id,
                    purpose="context",
                )
                verified = await verify_and_bound(data, descriptor)
            except AttachmentRejected:
                lines.append(f"{meta} — [image withheld: could not be verified]")
                continue
            except Exception:  # noqa: BLE001 — BridgeRefusal/unreachable: a line, not a crash
                lines.append(f"{meta} — [image withheld: not available to this agent]")
                continue
            budget -= 1
            lines.append(meta)
            blocks.append((verified.private_payload(), verified.wire_mime))
        out.append((_image_provenance(channel_id, message, lines), blocks))
    return out


def build_bridge_tool(context: ToolContext) -> AgentTool | None:
    """createIf: absent without a bridge runtime (unsigned / not designated).

    The runtime is stashed by the session that owns one (``bridge_runtime``,
    declared on ``ToolContext``); no probing beyond that one declared field.
    """
    runtime = getattr(context, "bridge_runtime", None)
    if runtime is None:
        return None

    async def _execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any,
        on_update: Any,
        context: ToolContext,
    ) -> ToolResult:
        outcome = await execute_bridge(runtime, args)
        # Bytes never ride ``details`` (persisted with the transcript row).
        images = outcome.pop(_IMAGES_KEY, None) or []
        if outcome.get("ok"):
            peer_receipt = args.get("op") in ("send", "reply")
            if peer_receipt:
                # Providers receive TextContent, not necessarily details. Keep
                # the same receipt in both so "sent" cannot hide the inbox-only
                # acceptance boundary from a model. The executor's receipt is
                # closed metadata; never serialize the request or server body.
                content = json.dumps(outcome, ensure_ascii=False)
            elif outcome.get("content") is not None:
                # Read-shaped channel results carry their own model-facing
                # rendering (JSON page, or the untrusted-wrapped history).
                content = str(outcome["content"])
            elif outcome.get("text") is not None:
                # §11.1: the take_up result marks the body as external and
                # untrusted, exactly as the wake wrap does — the ONE clause
                # constant (D6), so a copy edit cannot silently un-mark this hop.
                content = (
                    f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n"
                    f"message {outcome.get('message_id')}\n\n{outcome['text']}"
                )
            else:
                content = f"bridge {args.get('op')}: {outcome.get('state', 'ok')}"
            # N1: the served §11.3 guidance is ADVISORY — appended when a
            # status snapshot carried one; its absence degrades advice only.
            client = getattr(runtime, "client", None)
            getter = getattr(client, "guidance_note", None)
            note = ""
            if callable(getter):
                try:
                    note = str(getter() or "")
                except Exception:  # noqa: BLE001 — advice only, never a failure
                    note = ""
            blocks: list[Content] = [TextContent(text=content)]
            if note:
                if peer_receipt:
                    # Preserve existing advice without appending prose to the
                    # receipt's JSON block, which must parse on its own.
                    blocks.append(TextContent(text=f"[bridge guidance — advisory]\n{note}"))
                else:
                    content = f"{content}\n\n[bridge guidance — advisory]\n{note}"
                    blocks = [TextContent(text=content)]
            # Caption-then-image, the shape every image tool result uses
            # (``tools.builtin._image``): a text-only consumer still reads what
            # each image is, and a session whose model cannot take images has
            # each block replaced by the harness's own omission notice
            # (``session._without_images``) while the caption remains.
            for caption, pictures in images:
                blocks.append(TextContent(text=caption))
                blocks.extend(
                    ImageContent(data=base64.b64encode(data).decode("ascii"), mime_type=mime)
                    for data, mime in pictures
                )
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="bridge",
                content=blocks,
                details=outcome,
            )
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="bridge",
            content=[TextContent(text=str(outcome.get("error") or "refused"))],
            details=outcome,
            is_error=True,
        )

    return AgentTool(
        name="bridge",
        label="Team Bridge",
        description=(
            "Team Bridge as this session's agent. directory: teammates and their "
            "chiefs of staff. dm: message a person or their agent directly (a "
            "relay request means dm, not 'forward this'); channel_post: rooms; "
            "send/reply: only to answer a pair-inbox (bmsg_) delivery. "
            "shared_*: conversations your owner shared with you. Cannot approve "
            "or deny. More: read tool://bridge."
        ),
        parameters=BridgeToolParams.model_json_schema(),
        approval_tier="write",
        call_approval_tier=lambda args: (
            "read" if str(args.get("op")) in (*READ_OPS, "ack") else "write"
        ),
        concurrency="exclusive",
        interruptible=True,
        execute=_execute,
    )


__all__ = [
    "BRIDGE_OPS",
    "READ_OPS",
    "CHANNEL_OPS",
    "CLAIM_TTL_S",
    "CONVERSATION_PAGE_MAX",
    "BridgeRuntime",
    "BridgeToolParams",
    "ChannelMemberParams",
    "body_digest",
    "build_bridge_tool",
    "execute_bridge",
]
