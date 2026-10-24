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

THE RUNTIME BUNDLE is declared on ``ToolContext`` (``bridge_runtime``): a
session that has no bridge runtime — unsigned, disabled, or not the designated
session — never advertises this tool at all (the same createIf shape ``wake``
uses). No probing, no second gating convention.
"""

from __future__ import annotations

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
)
BRIDGE_OPS: tuple[str, ...] = ("take_up", "ack", "send", "reply", *CHANNEL_OPS)

#: The server caps a conversation page at 100 (contract §10.1); bounding here
#: turns an over-wide ask into a typed refusal instead of a server 400.
CONVERSATION_PAGE_MAX = 100

#: Display label for the auto-added creator member. The label is display-only
#: and taken from the member row by the server; this one is what another
#: member sees until the channel's members relabel it.
CHANNEL_CREATE_DEFAULT_LABEL = "agent"

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
    from .host import admission_refusal, admit_soon, owner_fence_refusal

    fence = owner_fence_refusal(runtime)
    if fence is not None:
        return _error(fence["error"], code=fence["code"])
    # Refuse unsupported destinations before even scheduling admission I/O.
    # The local authority fence keeps precedence; no invalid destination can
    # create an outbox intent, send a message, or trigger a status request.
    recipient: dict[str, str] = {}
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
    is_read = op in ("take_up", "channels", "channel_history", "channel_thread")
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
        }

    if params.op == "channels":
        if not 1 <= int(params.limit) <= CONVERSATION_PAGE_MAX:
            return _error(f"channels limit must be 1..{CONVERSATION_PAGE_MAX}")
        try:
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
        message = result.get("message") if isinstance(result, dict) else None
        if not isinstance(message, dict):
            message = {}
        message_id = str(message.get("id") or "")
        seq = message.get("seq")
        content = f"posted to {channel_id}: message {message_id or '?'}"
        if thread_root_id:
            content += f" (reply in thread {thread_root_id})"
        if isinstance(seq, int):
            content += f" at seq {seq}"
        return {"ok": True, "message_id": message_id, "state": "sent", "content": content}

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
        return {
            "ok": True,
            "state": "ok",
            "content": f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(result, ensure_ascii=False)}",
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
        return {
            "ok": True,
            "state": "ok",
            "content": f"{BRIDGE_UNTRUSTED_CLAUSE}\n\n{json.dumps(result, ensure_ascii=False)}",
        }

    # The decide/unknown refusals above run before the gates; this later return
    # is the schema-valid op that lost its branch — a programmer error, not a
    # model input.
    return _error(
        f"unknown bridge op {params.op!r}",
        code="invalid_request",
    )


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
                    blocks = [
                        TextContent(text=f"{content}\n\n[bridge guidance — advisory]\n{note}")
                    ]
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
            "Team Bridge as this session's agent: take_up/ack, send/reply, and "
            "channel ops (channels, channel_create, channel_post, channel_history, "
            "channel_thread, react). Cannot approve or deny."
        ),
        parameters=BridgeToolParams.model_json_schema(),
        approval_tier="write",
        call_approval_tier=lambda args: (
            "read"
            if str(args.get("op"))
            in ("take_up", "ack", "channels", "channel_history", "channel_thread")
            else "write"
        ),
        concurrency="exclusive",
        interruptible=True,
        execute=_execute,
    )


__all__ = [
    "BRIDGE_OPS",
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
