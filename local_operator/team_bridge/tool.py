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
- **channel_thread** — one thread, oldest-first: the root (with its reply
  count) and a bounded page of replies. Same screening rule as history — an
  unadmitted reply is metadata with ``body`` null.
- **channel_history** — a bounded page of messages. The server admits bodies
  per (message, endpoint) screening; an agent sees the METADATA (sender, seq,
  hash) of messages it has not admitted, with ``body`` null, and only the
  admitted bodies. There is intentionally no local body cache here.

- **react** — set an emoji on an admitted conversation message (or remove it
  with ``remove=true``). This is a screened agent WRITE, never an ack; the hub
  refuses a message this endpoint has not admitted. Reactions never wake agents.

IDENTITY is never a model argument: the actor endpoint and provenance session
come from ``runtime.owner.endpoint_id`` and ``runtime.session_id``. Every send
carries that session id as origin provenance (the server stores it and never
returns it); a runtime with no endpoint refuses every channel op rather than
acting as a human. Neither message bodies nor session ids are ever logged here.

REFUSALS, one line each: ``not_permitted`` for ``decide``; ``invalid_request``
for unknown ops, malformed ``channel_id``, bad history bounds/limits, and
invalid member shapes; ``no_agent_endpoint`` when a channel op has no owner
endpoint; ``*_failed`` when the transport does not confirm.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.harness.types import (
    AgentTool,
    Content,
    ImageContent,
    TextContent,
    ToolContext,
    ToolResult,
)

from .client import TeamBridgeClient
from .outbox import Outbox, send_from_outbox
from .types import _CONVERSATION_MESSAGE_ID as _CANONICAL_MESSAGE_ID
from .types import (
    CONVERSATION_ID_PATTERN,
    InvalidSendRecipient,
    is_conversation_message_id,
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
    for branch in schema.get("anyOf", [schema]):
        if branch.get("type") == "object":
            names = branch.pop("propertyNames")["enum"]
            value = branch["additionalProperties"]
            branch["properties"] = {name: value for name in names}
            branch["additionalProperties"] = False


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
    to: dict[Literal["endpoint_id", "account_id"], Annotated[str, Field(min_length=1)]] | None = (
        Field(default=None, min_length=1, max_length=1, json_schema_extra=_closed_recipient_schema)
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
    # dm: the person (account_id) or agent (endpoint_id, from `directory`).
    # Plain strings, not the `to` recipient object, so a dm can never be
    # mistaken for a pair-lane `send` destination by the shared validator.
    account_id: str = ""
    endpoint_id: str = ""
    # dm to an account: "agent" (default) reaches that person's chief of
    # staff, "human" the person themself.
    target: Literal["agent", "human"] = "agent"


def _error(message: str, *, code: str = "invalid_request") -> dict[str, Any]:
    return {"ok": False, "code": code, "error": message}


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
        # The endpoint is the precise addressee; the account is the fallback
        # the pair contract also accepts (exactly one key on the wire).
        to = {"endpoint_id": sender["endpoint_id"]} if "endpoint_id" in sender else sender
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
        return _error(f"could not read the team directory: {failure}", code="directory_failed")
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
        return (None, "", _error(f"could not read the team directory: {failure}", code="dm_failed"))
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
    if params.to or params.thread_id or params.channel_id or params.members:
        return _error("dm takes account_id or endpoint_id, not pair/room destinations")
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
            return _not_reachable(str(code), f"the hub refused that recipient: {failure}")
        return _error(f"the direct message was not opened: {failure}", code="dm_failed")
    if not isinstance(created, dict):
        return _error("the hub returned no DM view", code="dm_failed")
    view = created.get("conversation") if isinstance(created.get("conversation"), dict) else created
    conversation_id = str(view.get("id") or "") if isinstance(view, dict) else ""
    if _CONVERSATION_ID.fullmatch(conversation_id) is None:
        return _error("the hub opened no usable DM id", code="dm_failed")
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
        return _error(
            f"the DM {conversation_id} opened but the post did not confirm: {failure}",
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
        return _error(f"could not list shared conversations: {failure}", code="list_failed")
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
        return _error(f"could not read the shared conversation: {failure}", code="history_failed")
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
    view = result.get("message") if isinstance(result.get("message"), dict) else result
    message_id = view.get("id")
    seq = view.get("seq")
    return (
        message_id if isinstance(message_id, str) else "",
        seq if isinstance(seq, int) and not isinstance(seq, bool) else None,
    )


def _post_receipt(
    conversation_id: str, result: Any, *, thread_root_id: str = "", lead: str = "posted to"
) -> dict[str, Any]:
    """The ONE success receipt for a conversation write (channel_post, dm).

    Carries ``message_id`` and ``seq`` as fields so the sender can confirm the
    post and pass the id back as ``thread_root_id``. A missing id is reported
    honestly as unconfirmed rather than rendered as ``?``.
    """

    message_id, seq = _posted_message(result)
    content = f"{lead} {conversation_id}: " + (
        f"message {message_id}" if message_id else "accepted, but the hub returned no message id"
    )
    if thread_root_id:
        content += f" (reply in thread {thread_root_id})"
    if seq is not None:
        content += f" at seq {seq}"
    return {
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
    # Refuse unsupported destinations before even scheduling admission I/O.
    # The local authority fence keeps precedence; no invalid destination can
    # create an outbox intent, send a message, or trigger a status request.
    recipient: dict[str, str] = {}
    if op == "reply":
        # Resolution reads only the local store, so it sits between the fence
        # and the recipient check (R9): an unresolvable reply refuses here,
        # before admission could schedule any status read.
        args, refused = _resolve_reply(runtime, args)
        if refused is not None:
            return refused
    if op in ("send", "reply"):
        try:
            recipient = validate_send_recipient(args.get("to"))
        except InvalidSendRecipient as failure:
            return _error(str(failure), code=failure.code)
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
            return _error(f"could not fetch the message: {failure}", code="fetch_failed")
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
            return _error(f"the send did not confirm: {failure}", code="send_unconfirmed")
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
            return _error(f"could not list conversations: {failure}", code="list_failed")
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
            return _error(f"the channel was not created: {failure}", code="create_failed")
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
            )
        except Exception as failure:
            return _error(f"the post did not confirm: {failure}", code="post_failed")
        return _post_receipt(channel_id, result, thread_root_id=thread_root_id)

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
            return _error(f"the reaction did not confirm: {failure}", code="react_failed")
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
            return _error(f"could not read the channel history: {failure}", code="history_failed")
        # Bodies are external content even when the sender is a teammate's
        # agent: the same untrusted wrap as take_up. Unadmitted messages carry
        # ``body: null`` — metadata only — and stay that way (no local cache).
        images = await _history_images(runtime, channel_id, result, endpoint_id)
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
            return _error(f"could not read the thread: {failure}", code="thread_failed")
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
        images = await _history_images(runtime, channel_id, {"messages": thread_rows}, endpoint_id)
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


async def _history_images(
    runtime: BridgeRuntime, channel_id: str, page: Any, endpoint_id: str
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
                if message.get("body") is not None and not _screener_sees_images(runtime):
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
