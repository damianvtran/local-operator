"""Typed, parse-don't-validate records for the team-bridge metadata path (H2).

WHAT LIVES HERE. The values the store may persist and the wire projections
that produce them. Two hard rules, both from the frozen contract (§13.9's
"operational metadata … no body cache" and the H2 handoff's privacy list):

1. **No content.** Nothing here holds a body, excerpt, summary, sender name or
   email, raw model reason, token or HTTP error detail. Projections WHITELIST
   the metadata fields; everything else on a server item is dropped by
   construction, not by a redaction pass. The ONE exception is
   :class:`MemberCandidate`, which carries a member's ``display_label`` —
   because a name -> member resolution cannot be done without the name it is
   matching. It is a TRANSIENT in-memory candidate set handed to
   :func:`resolve_member_by_name`, it is never a durable record of this module,
   and the delivery-side half that does persist an attempt
   (``team_bridge.recipient_resolution``) states its own narrower data set there.
2. **Distinct types for distinct cursors.** ``InboxCursor`` is an OPAQUE
   server string; ``FeedSeq`` is an integer. They are never interchangeable
   (H2 handoff: "never interchange them") — the constructor functions here
   are the only sanctioned way to mint either, and they refuse the other's
   shape, so a swap is a type error at the call site rather than a silent
   misordered resume.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, NewType

# WHY the two imports differ from the branch's original pair: ``quote`` /
# ``unquote`` left with #2856 (durable refs no longer carry display labels, so
# nothing here percent-encodes them); ``UUID`` arrived with #2828's canonical
# account shape and is still required by ``is_account_id``.
from uuid import UUID

__all__ = [
    "ATTACHMENT_ID_PATTERN",
    "CONVERSATION_ID_PATTERN",
    "InvalidSendRecipient",
    "RECIPIENT_ACCOUNT_SHAPE",
    "RECIPIENT_ENDPOINT_SHAPE",
    "RECIPIENT_KEYS",
    "RECIPIENT_NAME_SHAPE",
    "RECIPIENT_TEACHING",
    "SCOPE_ALLOWLIST",
    "SCOPE_CONVERSATION",
    "SEND_RECIPIENT_DESCRIPTION",
    "MemberCandidate",
    "NAMED_CANDIDATES_MAX",
    "NameResolution",
    "RecipientNameRefused",
    "ResolutionScope",
    "is_account_id",
    "resolve_member_by_name",
    "validate_member_mention",
    "validate_recipient_shape",
    "validate_send_recipient",
    "SCREENING_CATEGORIES",
    "ACK_TARGETS",
    "AckTarget",
    "ConversationFeedRef",
    "FeedSeq",
    "InboxCursor",
    "InboxPage",
    "MessageRef",
    "OwnerSnapshot",
    "ScreeningReceipt",
    "TeamBridgeTypeError",
    "feed_seq",
    "inbox_cursor",
    "is_conversation_message_id",
    "parse_conversation_ref_value",
    "project_events_batch",
    "project_inbox_page",
    "project_message_ref",
]

#: The minted conversation-id family: ``bcv_`` + 8..64 lowercase base36 chars.
#: ONE definition shared by the desktop route guard (S-RSA-5) and the agent
#: tool: both interpolate a caller-supplied id into an upstream path, so a
#: second copy is a drift risk, not a style choice. Kept here (not in the
#: route module) because this module has no httpx/server imports.
CONVERSATION_ID_PATTERN = r"^bcv_[0-9a-z]{8,64}$"

#: The hub-minted image attachment id (``batt_`` + a UUID). Shared by the
#: client, the desktop routes and the agent tool for the same reason as the
#: conversation pattern: each interpolates the id into an upstream path, and
#: the grammar's job is to prove it cannot reshape that path (no ``/``, ``.``,
#: ``%`` or ``?``), not to re-implement the hub's mint.
ATTACHMENT_ID_PATTERN = r"^batt_[0-9A-Za-z-]{8,64}$"

#: The conversation-message id family (``bcmsg_`` + lowercase base36), and the
#: kind token the feed ref uses to carry a conversation delivery. Both are
#: shared by the projection and the parser so a ref cannot be written in one
#: spelling and read in another.
_CONVERSATION_MESSAGE_ID = re.compile(r"^bcmsg_[0-9a-z]{1,64}$")
CONVERSATION_FEED_KIND = "conversation_id"
CONVERSATION_FEED_TYPE = "message_available"

#: §6.2's closed ack states for ONE recipient advancing their own delivery:
#: unread→read (with an allow verdict) or unread/held→blocked (with a reject
#: verdict). working/done are NOT H2 acks (no automatic progression exists).
ACK_TARGETS = ("read", "blocked")
AckTarget = Literal["read", "blocked"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_DECIMAL = re.compile(r"^(0|[1-9][0-9]*)$")
#: The frozen server DTO's closed event-type set (bridge.go BridgeEventItem).
_EVENT_TYPES = (
    "message_available",
    "state_changed",
    "request_decided",
    "request_revoked",
    "request_expired",
)
#: The pair subject-ref kinds a feed event may carry (bridge.go
#: BridgeEventIDs). Conversation deliveries are projected SEPARATELY below:
#: their subject is a conversation message plus the member's mention bit, not
#: a pair message the inbox can resolve.
_EVENT_ID_KINDS = ("message_id", "request_id", "thread_id")
#: The frozen §7.2 delivery-state set, defined ONCE at its source.
DELIVERY_STATES = ("unread", "read", "working", "done", "blocked")
_VERDICTS = ("allow", "reject")


#: The closed screening-category vocabulary (§8.3). Defined HERE — the leaf
#: module both the pipeline and the store already import — so the screener's
#: parser and the persisted-proof validator cannot drift apart. A new category
#: is added once, here, and both sides accept it; an unknown string is damaged
#: proof, never re-emitted.
SCREENING_CATEGORIES = (
    "prompt_injection",
    "social_engineering",
    "harassment",
    "malicious_link",
    "spam",
    # §3.3 scope/prying classes (bridge-screen-v2).
    "personal_info_request",
    "out_of_scope_work",
    "new_work_request",
)


class TeamBridgeTypeError(ValueError):
    """A record violated its contract shape. Fail closed: callers refuse."""


class InvalidSendRecipient(TeamBridgeTypeError):
    """A caller supplied an unsupported destination; never persist or send it."""

    code = "invalid_request"


#: The two accepted recipient OBJECT shapes, spelled once. Every refusal that
#: teaches the contract and the tool's ``to`` description (below) interpolate
#: these, so what a model is TOLD cannot drift from what the validator accepts.
RECIPIENT_ENDPOINT_SHAPE = '{"endpoint_id": "<id>"}'
RECIPIENT_ACCOUNT_SHAPE = '{"account_id": "<id>"}'

#: The teaching tail on a shape refusal. A refusal that only said "unsupported
#: field" left the caller with no next move, so this names BOTH accepted shapes,
#: the bare-string form and where the ids come from. It deliberately says nothing
#: about the VALUE or KEY it refused (see the validator's docstring): the words
#: "name"/"label" do not appear, so no caller-controlled key is echoed back.
RECIPIENT_TEACHING = (
    f"a recipient is exactly one of {RECIPIENT_ENDPOINT_SHAPE} or "
    f"{RECIPIENT_ACCOUNT_SHAPE}, or the bare account id string; those ids come "
    f"from the bridge directory, which lists every addressable member"
)

#: The closed recipient KEY set for one destination. Two are ADDRESSES (an id the
#: server already knows); ``name`` is a RESOLUTION REQUEST — it is never a wire
#: value, it is what a caller asks the local resolver to turn into an address.
#: One tuple, so "exactly one of them" is one rule instead of three.
RECIPIENT_KEYS = ("endpoint_id", "account_id", "name")

#: The one accepted display-name shape. Spelled once, like its id siblings, so the
#: refusal a caller reads and the value the schema advertises cannot drift.
RECIPIENT_NAME_SHAPE = '{"name": "<display name>"}'

#: Model-facing help for the ``to`` parameter, published in the tool schema
#: (``tool.BridgeToolParams.to``). Built from the same shapes the refusals teach,
#: so the published advice cannot drift from the contract. It is billed on every
#: tool call, so it states the two object shapes, the bare form, where the ids
#: come from, and the ONE per-op difference that decides which op a model should
#: reach for (``scripts/bench_context_budget.py``).
SEND_RECIPIENT_DESCRIPTION = (
    f"Who receives it: {RECIPIENT_ACCOUNT_SHAPE} or {RECIPIENT_ENDPOINT_SHAPE}, "
    f"the bare account id, or {RECIPIENT_NAME_SHAPE} — trimmed, case-insensitive, "
    f"WHOLE display name only (never a handle, email or substring). A name resolves "
    f"in that op's own scope: channel_post inside the conversation's active members, "
    f"send/reply inside your own send allowlist. One match delivers; none or several "
    f"refuse. Ids come from the bridge directory."
)


def validate_send_recipient(value: Any) -> dict[str, str]:
    """Return the closed pair-inbox destination without coercion or field loss.

    A nested ``to.kind`` once survived tool validation but disappeared when the
    client rebuilt the HTTP body. That made an unsupported delivery request look
    successful. Both the tool (BEFORE its durable outbox) and the wire client
    use this check, so neither can silently reinterpret a destination. This is
    the existing pair-inbox contract, not conversation routing or a human fallback.
    Refusals deliberately do not quote caller-controlled keys or values.

    A DISPLAY NAME never reaches this function from the tool: the tool resolves one
    (``channel_post`` inside the conversation, ``send``/``reply`` inside the
    owner's allowlist) and passes on the resolved id, so the pair body it builds
    here only ever carries an address. A ``{"name": ...}`` that arrives anyway —
    a direct client caller, a replayed intent, a bug — refuses here, because this
    layer cannot resolve it and must not forward a string the server would read as
    an account id.

    A BARE STRING is an ACCOUNT id (``"acc_123"`` → ``{"account_id": "acc_123"}``):
    one string is one recipient in one field, so there is nothing to reinterpret
    and no second field that can be lost — the same guarantee the object form
    gives, for the shape a model reaches for first. Names are still NOT resolved
    HERE: the pair lane carries no conversation, so it has no least-privilege
    candidate set to resolve against, and a ``{"name": ...}`` refuses with a
    sentence naming the op that CAN resolve one. (``channel_post`` resolves it —
    see :func:`resolve_member_by_name`; the shape rule is shared, the op that may
    use it is not.) The string checks are the SAME ones the object values get,
    applied to the only value there is, plus ONE extra: a bare string carrying
    INTERNAL whitespace refuses with the same teaching text, because a display
    name is a guess at an id (no id contains a space) and accepting "Ada Lovelace"
    as an account would forward the mistake to the server instead of correcting
    it. The OBJECT form is an explicit claim that its value already IS an id, so
    its pre-existing contract — any non-empty unpadded string — is deliberately
    unchanged here.
    """
    recipient = validate_recipient_shape(value)
    if "name" in recipient:
        raise InvalidSendRecipient(
            # This is the WIRE guard, not the op policy: the tool resolves a
            # display-label recipient (channel_post inside the room, send/reply
            # inside the owner's allowlist) and hands this layer an id. A name
            # reaching HERE means a caller skipped that resolution, and the pair
            # body has nowhere to put one — so it refuses rather than forwarding a
            # string the server would read as an account id.
            "this wire carries an id, not a resolved recipient; "
            + RECIPIENT_TEACHING
        )
    return recipient


def validate_recipient_shape(value: Any) -> dict[str, str]:
    """Return the closed one-of destination AS ASKED, without coercion.

    The SHAPE rule every op shares: exactly one recipient, exactly one of
    :data:`RECIPIENT_KEYS`, no extra keys. ``name`` passes HERE and is then
    refused by :func:`validate_send_recipient` (no conversation to scope it to)
    or resolved by :func:`resolve_member_by_name` (``channel_post``). Splitting
    the shape from the per-op permission is what keeps ONE answer to "which
    keys exist": a second copy of this rule at a call site is how a shape that
    survived one reader quietly became a different shape at the next.

    Refusals quote no caller-controlled key or value, exactly as the send form
    does — the teaching tail names the accepted shapes and the directory instead.
    """
    if isinstance(value, str):
        if any(character.isspace() for character in value.strip()):
            raise InvalidSendRecipient(RECIPIENT_TEACHING)
        key, identifier = "account_id", value
    else:
        if not isinstance(value, Mapping) or len(value) != 1:
            raise InvalidSendRecipient(RECIPIENT_TEACHING)
        key = next(iter(value))
        if key not in RECIPIENT_KEYS:
            raise InvalidSendRecipient(
                "recipient contains an unsupported field; " + RECIPIENT_TEACHING
            )
        identifier = value[key]
    if key == "name":
        # A name is TRIMMED by the resolver, so padding here is not an error:
        # refusing it would make " Ada Lovelace " unresolvable for the one form
        # whose whole job is to be human-spelled. Whitespace-only never reaches
        # the resolver with anything to match.
        if not isinstance(identifier, str) or not identifier.strip():
            raise InvalidSendRecipient("a recipient name must be a non-empty display name")
        return {"name": identifier}
    if not isinstance(identifier, str) or not identifier or identifier != identifier.strip():
        raise InvalidSendRecipient("send/reply recipient must be a non-empty unpadded string")
    return {key: identifier}


# ---------------------------------------------------------------------------
# Name -> member resolution (the ONE resolver; both call sites use it)
# ---------------------------------------------------------------------------
#
# WHY THIS IS A RESOLUTION REQUEST AND NOT A DESTINATION. Measured on this
# machine: agents addressed colleagues as ``to: "Ada Lovelace"`` and every call
# was refused, because the tool accepts only opaque ids — five allow cases, seven
# calls, seven refusals, zero messages on the wire. A name is how a HUMAN is
# named, so refusing the shape outright made the feature unusable for people.
# The fix is not to guess: it is to RESOLVE the name against a caller-supplied
# candidate set and deliver only on an EXACTLY-ONE match. A wrong answer here
# delivers to the wrong human, which no later receipt repairs.

#: The member kinds the hub addresses. A mention row carries one, so a candidate
#: whose row has neither is not addressable and never becomes a candidate.
MEMBER_KINDS = ("human", "agent")


@dataclass(frozen=True)
class MemberCandidate:
    """One member a display name may resolve to: the address and its display name.

    ``display_label`` is what a human sees in the member list (server-side, from
    the member row) and is the ONLY field a name is matched against. ``kind`` and
    ``endpoint_id`` ride along because a directed post's mention row needs them.

    ``bridge_enabled`` is the DIRECTORY's verdict for this account at the moment
    the candidate set was built — the third guard of the ``send``/``reply``
    scope, where presence in the allowlist is not the same fact as being able to
    receive, and a resolution to a switched-off bridge is a message that goes
    nowhere. It defaults ``True`` because the conversation scope's rule is
    presence in the room, not this flag.
    """

    account_id: str
    display_label: str
    kind: str = ""
    endpoint_id: str = ""
    bridge_enabled: bool = True

    def mention(self) -> dict[str, str]:
        """The wire ``mentions[]`` row for THIS member — never the asked-for name.

        One method rather than a literal at each call site: a mention rebuilt
        from a name somewhere else is the silent-loss class this file exists to
        close (a directed message delivered to nobody looks exactly like an
        undirected one). A member with no addressable kind refuses HERE rather
        than building a row the transport would reject: the conversation
        projection filters those out already, so this is the guard that keeps a
        future candidate source from silently reintroducing them.
        """

        if self.kind not in MEMBER_KINDS:
            raise TeamBridgeTypeError("only a human or agent member can be mentioned")
        row = {"kind": self.kind, "account_id": self.account_id}
        if self.endpoint_id:
            row["endpoint_id"] = self.endpoint_id
        return row


@dataclass(frozen=True)
class ResolutionScope:
    """WHERE a name was resolved, as a noun and an audit word.

    The sentence a caller reads and the row the audit keeps both need the scope,
    and PR2 wires ``send``/``reply`` to :data:`SCOPE_ALLOWLIST` without editing a
    single refusal: the resolver composes them from HERE, so a second scope cannot
    arrive with a second spelling of "no member by that name".
    """

    #: The audit vocabulary — the row's ``scope`` value (NOT ``applies_to``:
    #: that was this field's working name and it never reached the log).
    name: str
    #: The clause a refusal uses, e.g. "in this conversation".
    where: str
    #: What a caller should do about an ambiguous match, e.g. "a more specific label".
    disambiguator: str
    #: Appended to a ZERO-match refusal, when "nobody by that name" has more than
    #: one reason and the caller cannot tell which. A model that cannot tell them
    #: apart retries uselessly instead of telling its human something true, which
    #: is why the reason is spelled out rather than left to be inferred.
    miss_hint: str = ""


#: ``channel_post``: the conversation's own active members (least privilege — a
#: name is only resolvable to somebody who is IN the room the message lands in).
SCOPE_CONVERSATION = ResolutionScope(
    name="conversation",
    where="in this conversation",
    disambiguator="a more specific label, or add the member to this conversation",
)

#: ``send``/``reply``: the owner's REACHABLE set — the bridge directory
#: intersected with the owner's own send-allowlist, never directory-wide. A name
#: that resolves to nobody names the reasons it might not — "not on this team",
#: "their bridge is off", or "your allowlist does not name them" — because those
#: are different facts with different next moves for the human, and a model that
#: cannot tell them apart retries uselessly instead of saying something true.
SCOPE_ALLOWLIST = ResolutionScope(
    name="allowlist",
    where="in your bridge allowlist",
    disambiguator="a more specific label, or an explicit account id",
    miss_hint=(
        " — either they are not on this team, or their bridge is not enabled, or "
        "your allowlist does not name them; ask the human which, and an explicit "
        "account id still sends"
    ),
)


@dataclass(frozen=True)
class NameResolution:
    """The record of ONE resolution attempt — resolved, refused or ambiguous.

    Carried by :class:`RecipientNameRefused` so the refusal path can write the
    same audit row a delivery does: requested name, matched count, candidates and
    the resolved id (null when nothing resolved).
    """

    requested: str
    matches: tuple[MemberCandidate, ...]

    @property
    def matched_count(self) -> int:
        return len(self.matches)

    @property
    def resolved(self) -> MemberCandidate | None:
        """The one match, or ``None`` — never the first of several."""

        return self.matches[0] if len(self.matches) == 1 else None

    def candidates_named(self) -> tuple[str, ...]:
        """``label (account_id)`` per NAMED candidate, bounded like the audit row.

        A label is MEMBER-authored text (like a history sender label), so it is
        flattened and bounded here: a newline in a label must not be able to
        restructure the sentence a model reads. The LIST is bounded too, and for
        the same class of reason one step out: the legacy placeholder label
        ``Member`` is shared by many participants of one conversation, so an
        unbounded join turned one refusal into a thousand characters of ids a
        model pays for and cannot use. :meth:`unnamed_count` says how many were
        left out, so the refusal stays exact about N even when it names few.
        """

        return tuple(
            f"{_flatten_label(member.display_label)} ({member.account_id})"
            for member in self.matches[:NAMED_CANDIDATES_MAX]
        )

    def unnamed_count(self) -> int:
        """How many matches the refusal does NOT name (0 when it names them all)."""

        return max(0, self.matched_count - NAMED_CANDIDATES_MAX)


#: How many candidates ONE refusal names. The COUNT is always exact; the list is
#: bounded so a conversation whose members share a placeholder label cannot turn
#: a disambiguation sentence into a page of ids (the audit row keeps its own
#: bound for the same reason, and only the log may be wider than the sentence).
NAMED_CANDIDATES_MAX = 5


class RecipientNameRefused(InvalidSendRecipient):
    """A display name did not resolve to exactly one member; nothing was sent.

    Carries the :class:`NameResolution` so the CALLER can record the attempt
    before it refuses: the audit row must exist for a refusal exactly as it does
    for a delivery, or a later misdelivery claim cannot be reconstructed. It is
    an :class:`InvalidSendRecipient` (``code = "invalid_request"``) because the
    caller's request is what did not resolve — never the transport.
    """

    def __init__(self, message: str, resolution: NameResolution) -> None:
        super().__init__(message)
        self.resolution = resolution


def _flatten_label(label: str) -> str:
    """One line, bounded: the same treatment a history sender label gets."""

    return " ".join(str(label).split())[:80]


def resolve_member_by_name(
    requested_name: Any,
    candidates: Iterable[MemberCandidate],
    *,
    scope: ResolutionScope = SCOPE_CONVERSATION,
) -> MemberCandidate:
    """Resolve a DISPLAY NAME to exactly one candidate member, or refuse.

    THE MATCHING RULE, stated once (the tool schema says the same in fewer
    words): the requested name is TRIMMED, compared CASE-INSENSITIVELY
    (``casefold``, so a locale cannot split a name), and matched against the
    member's DISPLAY NAME ONLY — never a handle, an email, an endpoint label or
    a PARTIAL/SUBSTRING of a label. ``"Ada"`` never resolves to ``"Ada
    Lovelace"``: a prefix is a guess about which Ada, and guessing is the one
    thing this function may not do.

    THREE OUTCOMES, and no fourth:

    * exactly one candidate matches -> that member, and the caller may deliver;
    * none -> :class:`RecipientNameRefused` saying no active member by that name
      exists in the scope (a caller with a name from ANOTHER context learns the
      scope is the limit, not that the person does not exist);
    * several -> the same refusal with the candidate list (labels and ids) and
      the name of a disambiguator. The first row is never taken, and no
      preference exists between a human and an agent: preferring one is a guess
      dressed as a rule.

    ``candidates`` is ALWAYS an argument, never fetched here: the scope belongs
    to the CALLER (a conversation's active members for ``channel_post``; the
    owner's allowlist for ``send``/``reply``), and a resolver that went looking
    for its own candidate set would be able to resolve outside the scope it was
    handed.
    """

    if not isinstance(requested_name, str) or not requested_name.strip():
        raise RecipientNameRefused(
            "a recipient name must be a non-empty display name",
            NameResolution(requested="", matches=()),
        )
    requested = requested_name.strip()
    wanted = requested.casefold()
    matches = tuple(
        member
        for member in candidates
        if isinstance(member, MemberCandidate) and member.display_label.strip().casefold() == wanted
    )
    resolution = NameResolution(requested=requested, matches=matches)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RecipientNameRefused(
            f"no active member by that name {scope.where}{scope.miss_hint}", resolution
        )
    named = ", ".join(resolution.candidates_named())
    more = resolution.unnamed_count()
    if more:
        named += f", and {more} more"
    raise RecipientNameRefused(
        f"that name matches {len(matches)} members {scope.where}: {named} — ask for "
        f"{scope.disambiguator}",
        resolution,
    )


def validate_member_mention(value: Any) -> dict[str, str]:
    """Return one closed ``{kind, account_id[, endpoint_id]}`` mention row.

    Shared by the tool (BEFORE dispatch) and the wire client (when it rebuilds
    the body) for the same reason :func:`validate_send_recipient` is: a shape
    that survives tool validation must not vanish at serialization. A mention
    that is dropped turns a DIRECTED message into an undirected one, and the two
    look identical in the server's answer — the silent-success class, again.

    The shape is the hub's own Send-policy mention row, already accepted by the
    scheduled-send body (``desktop_team_bridge.ScheduledMentionIn``), so an
    immediate send speaks the vocabulary the plan already uses. ``endpoint_id``
    is optional (an agent member has one; a human member does not).
    """

    if not isinstance(value, Mapping):
        raise InvalidSendRecipient("a mention row is an object with kind and account_id")
    keys = set(value)
    if keys - {"kind", "account_id", "endpoint_id"}:
        raise InvalidSendRecipient("a mention row carries only kind, account_id, endpoint_id")
    kind = value.get("kind")
    if kind not in MEMBER_KINDS:
        raise InvalidSendRecipient(f"a mention kind is one of {MEMBER_KINDS!r}")
    row = {"kind": str(kind), "account_id": _mention_id(value.get("account_id"), "account_id")}
    if "endpoint_id" in keys:
        row["endpoint_id"] = _mention_id(value.get("endpoint_id"), "endpoint_id")
    return row


def _mention_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InvalidSendRecipient(f"a mention {field_name} must be a non-empty unpadded string")
    return value


def is_account_id(value: Any) -> bool:
    """Accept legacy account IDs and the account service's canonical UUID shape.

    Keep this identity guard in the dependency-light types module: desktop read
    queries and conversation path guards must agree without importing an HTTP
    client at server boot. UUID parsing plus canonical-form comparison follows
    the Radient client's identifier guard; no braced, compact, URN, whitespace,
    or path/query spellings are admitted. Case is accepted, not normalized:
    authorization still compares the caller's actual identity, unchanged.
    """
    if not isinstance(value, str):
        return False
    if re.fullmatch(r"acc_[A-Za-z0-9_-]{1,64}", value) is not None:
        return True
    try:
        return value.lower() == str(UUID(value))
    except ValueError:
        return False


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


def feed_seq(value: int | str) -> FeedSeq:
    """Mint a FeedSeq from either canonical wire form, normalized to int.

    The frozen server answers the events `cursor` as a decimal STRING
    (bridge.go BridgeEventsResponse: `Cursor string json:"cursor" example:"42"`)
    while the store orders on an INTEGER; both canonical forms are accepted
    here and anything else — a bool, a negative, a leading-zero form, an
    opaque page cursor — is refused, so the two cursor species can never
    silently interchange.
    """

    if isinstance(value, bool):
        raise TeamBridgeTypeError("FeedSeq must be an int or canonical decimal string")
    if isinstance(value, int):
        if value >= 0:
            return FeedSeq(value)
    elif isinstance(value, str) and _CANONICAL_DECIMAL.match(value):
        return FeedSeq(int(value))
    raise TeamBridgeTypeError(f"FeedSeq {value!r} is not a non-negative integer or decimal string")


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
class ConversationFeedRef:
    """Contentless durable locator, never a cache of display names.

    New refs contain IDs, sequence, mention and the closed sender-kind enum.
    Migration-only aliases distinguish old queue rows, never authority. Old
    four/seven-part refs remain readable, but legacy labels are discarded
    on read. Names are fetched ephemerally through the endpoint-authorized body
    route and only rendered after screening/hash/admission. The store's v7
    migration removes old name-bearing queue keys transactionally; this parser
    also accepts older in-flight refs. Transcripts are never rewritten.
    """

    conversation_id: str
    message_id: str
    conversation_seq: int
    mention: bool
    sender_kind: str | None = None
    # Migration-only row identity. Never copied from wire ids or used for
    # message/body/admit/claim authority; preserves old pending aliases exactly.
    legacy_alias: str | None = None
    #: Set only when the message is a thread REPLY (agent-server threads): the
    #: root it answers. Persisted as a TAGGED slot (``/thread_root=<bcmsg_…>``)
    #: rather than a positional part, because a positional part would collide
    #: with the old seven-part name/label/kind layout. Optional and additive:
    #: a ref written before threads parses exactly as it did, and a reader that
    #: does not know the tag ignores it.
    thread_root_id: str | None = None

    def validated(self) -> "ConversationFeedRef":
        """Refuse a ref the parser could not read back (never guess)."""

        if re.fullmatch(CONVERSATION_ID_PATTERN, self.conversation_id) is None:
            raise TeamBridgeTypeError("conversation_id is not a bcv_ id")
        if _CONVERSATION_MESSAGE_ID.fullmatch(self.message_id) is None:
            raise TeamBridgeTypeError("message_id is not a bcmsg_ id")
        if (
            not isinstance(self.conversation_seq, int)
            or isinstance(self.conversation_seq, bool)
            or self.conversation_seq < 1
        ):
            raise TeamBridgeTypeError("conversation_seq must be a positive integer")
        if not isinstance(self.mention, bool):
            raise TeamBridgeTypeError("mention must be a boolean")
        if self.sender_kind is not None and self.sender_kind not in ("human", "agent"):
            raise TeamBridgeTypeError("sender_kind must be human or agent")
        if self.legacy_alias is not None:
            if (
                self.sender_kind is None
                or not isinstance(self.legacy_alias, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.legacy_alias) is None
            ):
                raise TeamBridgeTypeError("invalid legacy conversation alias")
        if self.thread_root_id is not None and (
            _CONVERSATION_MESSAGE_ID.fullmatch(self.thread_root_id) is None
        ):
            raise TeamBridgeTypeError("thread_root_id is not a bcmsg_ id")
        return self

    def encode(self) -> str:
        """The ONE persisted ref spelling, shared with :func:`parse_conversation_ref_value`."""

        base = (
            f"{CONVERSATION_FEED_TYPE}:{CONVERSATION_FEED_KIND}="
            f"{self.conversation_id}/{self.message_id}/{self.conversation_seq}/"
            f"{1 if self.mention else 0}"
        )
        encoded = base if self.sender_kind is None else f"{base}/{self.sender_kind}"
        if self.legacy_alias is not None:
            self.validated()
            encoded += f"/legacy_alias={self.legacy_alias}"
        if self.thread_root_id is not None:
            self.validated()
            encoded += f"/thread_root={self.thread_root_id}"
        return encoded


def parse_conversation_ref_value(value: str) -> ConversationFeedRef | None:
    """Decode a conversation feed ref value; ``None`` means malformed (skip, never guess).

    Four parts is the original locator; five adds the closed sender kind.
    Six adds a migration-only legacy_alias=<sha256> slot. Seven is the old
    name/label/kind layout: discard both name slots and derive a stable alias
    from the EXACT full old key's UTF-8 bytes. This preserves distinct pending
    states even when names were the only difference. It is a name-derived
    operational digest, NOT anonymization, authority or forensic erasure.
    """

    if not isinstance(value, str):
        return None
    parts = value.split("/")
    if len(parts) not in (4, 5, 6, 7):
        return None
    conversation_id, message_id, seq_text, mention_text = parts[:4]
    if _CANONICAL_DECIMAL.fullmatch(seq_text) is None:
        return None
    if mention_text not in ("0", "1"):
        return None
    # Tagged vs positional: a segment carrying ``=`` is a NAME=value tag and is
    # never read as a positional slot. That is what makes the shape extensible —
    # ``thread_root=`` (and anything a later release adds) rides beside the
    # existing parts instead of changing what a 6- or 7-part ref means. An
    # unknown tag is IGNORED, never rejected: a reader that predates a tag must
    # still decode the ref rather than dropping a durable queue row.
    positional: list[str] = []
    tags: dict[str, str] = {}
    seen_tag = False
    for segment in parts[4:]:
        if "=" in segment:
            name, _, tagged = segment.partition("=")
            tags[name] = tagged
            seen_tag = True
            continue
        if seen_tag:
            # A POSITIONAL after a tag is the ambiguity the old parser refused
            # (``legacy_alias=<hash>/human``): the tag would have to mean two
            # different things depending on order. Tags may follow positionals,
            # never precede one.
            return None
        positional.append(segment)

    kind: str | None = None
    if "legacy_alias" in tags:
        # A KNOWN tag with an invalid value is malformed, not "no alias": the
        # empty/mis-sized alias exists precisely to keep two rows distinct.
        if re.fullmatch(r"[0-9a-f]{64}", tags["legacy_alias"]) is None:
            return None
    legacy_alias = tags.get("legacy_alias") or None
    if len(positional) == 3:
        # The OLD seven-part layout: name, label, kind. The names are discarded
        # (they are a name cache and a privacy regression) and the alias is
        # derived from the EXACT full old key's bytes, so two pending rows that
        # differed only by display name stay distinct.
        if tags:
            return None
        kind = positional[2] or None
        original = f"{CONVERSATION_FEED_TYPE}:{CONVERSATION_FEED_KIND}={value}"
        try:
            legacy_alias = hashlib.sha256(original.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            return None
    elif len(positional) == 1:
        kind = positional[0] or None
    elif len(positional) > 1:
        return None

    if "thread_root" in tags:
        # A KNOWN tag with an EMPTY value is a refusal, like an empty
        # ``legacy_alias=``: "present but blank" is a malformed writer, not a
        # ref that happens to have no thread root.
        if not tags["thread_root"]:
            return None
        # And a thread root only exists on a ref that states its sender kind.
        # ``validated()`` cannot express this — kind is optional on every other
        # shape — so the grammar states it here: 4-form + ``/thread_root=``
        # would otherwise parse and round-trip while meaning "reply from
        # nobody", which no writer produces.
        if len(positional) != 1:
            return None
    if "thread_root" in tags and "legacy_alias" in tags:
        # REFUSED ON PURPOSE. A reply is written by the threads path, which
        # never mints migration aliases; a ref carrying both is a combination
        # no released writer can produce. Accepting it would quietly widen the
        # grammar to every tag combination, and a widened grammar is how a
        # reader starts honouring shapes nobody ever meant to persist.
        return None
    thread_root_id = tags.get("thread_root") or None
    try:
        return ConversationFeedRef(
            conversation_id=conversation_id,
            message_id=message_id,
            conversation_seq=int(seq_text),
            mention=mention_text == "1",
            sender_kind=kind,
            legacy_alias=legacy_alias,
            thread_root_id=thread_root_id,
        ).validated()
    except TeamBridgeTypeError:
        return None


def is_conversation_message_id(value: str) -> bool:
    """Whether ``value`` is a BARE conversation-message id (``bcmsg_…``).

    The connector's hold retry needs this distinction: a bare ``bcmsg_`` id
    cannot name its conversation, so it must never be re-driven through the
    pair spine (which would ask the hub for a pair message that does not
    exist). Shares the ONE id grammar the ref validator uses.
    """

    return isinstance(value, str) and _CONVERSATION_MESSAGE_ID.fullmatch(value) is not None


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
        if self.delivery_state not in DELIVERY_STATES:
            raise TeamBridgeTypeError(
                f"delivery state {self.delivery_state!r} not in {DELIVERY_STATES}"
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
    raw_delivery = item.get("delivery")
    if raw_delivery is None:
        # A sender reading their own message gets delivery:null (bridge_read.go
        # BridgeMessageItem). That is NOT a recipient view and carries no
        # delivery state, so H2 refuses it rather than filing a phantom row.
        raise TeamBridgeTypeError("item.delivery is null: not a recipient view")
    delivery = _require_mapping(raw_delivery, "item.delivery")
    raw_thread = item.get("thread")
    thread: Mapping[str, Any] = raw_thread if isinstance(raw_thread, Mapping) else {}
    message_id = _require_id(message.get("id") or message.get("_id"), "item.message.id")
    # The frozen server DTO has NO tenant_id on the message: the tenant rides
    # the thread block (bridge_read.go: BridgeThreadDetail.TenantID), so the
    # verified thread value is the normalization source — never an invented
    # field on the message.
    thread_id = _require_id(
        message.get("thread_id") or message.get("threadId") or thread.get("id"),
        "thread_id",
    )
    tenant_id = _require_id(
        message.get("tenant_id") or message.get("tenantId") or thread.get("tenant_id"),
        "tenant_id",
    )
    state = delivery.get("state")
    if not isinstance(state, str) or state not in DELIVERY_STATES:
        raise TeamBridgeTypeError(f"item.delivery.state {state!r} not in {DELIVERY_STATES}")
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
    if not all(isinstance(entry, Mapping) for entry in items):
        # One malformed entry refuses the WHOLE page: adopting the valid
        # subset while recording a checkpoint either loses backlog (claimed
        # complete) or poisons the cursor. Nothing is persisted before this
        # projection returns, so refusal means zero rows and zero checkpoint
        # movement (H2-Q3).
        raise TeamBridgeTypeError("inbox result.items entries must all be mappings")
    raw_cursor = result.get("next_cursor")
    if raw_cursor is None:
        cursor = None
    elif isinstance(raw_cursor, str) and raw_cursor:
        cursor = inbox_cursor(raw_cursor)
    else:
        # A present-but-invalid cursor must refuse: silently reading it as
        # "terminal" is exactly the backlog-loss bug (H2-Q3).
        raise TeamBridgeTypeError("inbox result.next_cursor must be null or an opaque string")
    counts = result.get("counts")
    clean_counts: dict[str, int] | None = None
    if isinstance(counts, Mapping):
        clean_counts = {
            str(key): value
            for key, value in counts.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    return InboxPage(
        items=tuple(_project_item(item) for item in items),
        next_cursor=cursor,
        counts=clean_counts,
    )


def project_message_ref(result: Mapping[str, Any]) -> MessageRef:
    """The single-message read: the same nested shape as ONE inbox item."""

    return _project_item(_require_mapping(result, "message result"))


def project_events_batch(result: Mapping[str, Any]) -> tuple[FeedSeq, tuple[str, ...], bool]:
    """The §6.7 events answer → (cursor, refs, resync flag). Strict, refs-only.

    The frozen DTO answers `cursor` as a canonical decimal STRING and
    `events` as `{type, ids{message_id?,request_id?,thread_id?}, at}`
    objects that "carry refs only — never message text" (bridge.go). This
    projection keeps that promise client-side: each event becomes ONE
    canonical ref string built ONLY from whitelisted scalar ids, the type is
    checked against the closed set, and every other key (a future server
    addition, or a garbled payload) is never read, stringified or persisted.
    An empty event list is the resync signal.
    """

    result = _require_mapping(result, "events result")
    events = result.get("events")
    if not isinstance(events, list):
        raise TeamBridgeTypeError("events result.events must be a list")
    refs: list[str] = []
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            raise TeamBridgeTypeError(f"events[{index}] must be a mapping")
        event_type = raw.get("type")
        if not isinstance(event_type, str) or event_type not in _EVENT_TYPES:
            raise TeamBridgeTypeError(
                f"events[{index}].type {event_type!r} is not in the closed set"
            )
        ids = raw.get("ids")
        if not isinstance(ids, Mapping):
            raise TeamBridgeTypeError(f"events[{index}].ids must be a mapping")
        if "conversation_id" in ids:
            # THE CONVERSATION SUBJECT (design D6). It is a distinct lane, not
            # a pair message with extra keys: route it as its own ref so the
            # connector can never hand it to the pair inbox reader. An empty
            # value is malformed, never a fallback to the pair lane.
            if event_type != CONVERSATION_FEED_TYPE:
                raise TeamBridgeTypeError(
                    f"events[{index}] carries conversation_id on {event_type!r}"
                )
            conversation_id = ids.get("conversation_id")
            message_id = ids.get("message_id")
            if not isinstance(conversation_id, str) or not isinstance(message_id, str):
                raise TeamBridgeTypeError(f"events[{index}].ids conversation refs must be strings")
            raw_seq = ids.get("conversation_seq")
            if isinstance(raw_seq, bool) or not isinstance(raw_seq, (int, str)):
                raise TeamBridgeTypeError(
                    f"events[{index}].ids.conversation_seq must be an integer"
                )
            seq_text = str(raw_seq)
            if _CANONICAL_DECIMAL.fullmatch(seq_text) is None:
                raise TeamBridgeTypeError(
                    f"events[{index}].ids.conversation_seq must be canonical decimal"
                )
            mention = ids.get("mention", False)
            if not isinstance(mention, bool):
                raise TeamBridgeTypeError(f"events[{index}].ids.mention must be a boolean")
            # The server omits ``mention`` when false (Go's ``omitempty`` on a
            # bool), so an absent bit is a NON-mention delivery, not a
            # malformed one.
            # Wire labels are deliberately not read: feed/hold refs survive
            # restart and must not become a persistent name cache. Only the
            # closed provenance enum may accompany the locator.
            kind = ids.get("sender_kind")
            if kind is not None and kind not in ("human", "agent"):
                raise TeamBridgeTypeError(f"events[{index}].ids.sender_kind is invalid")
            at = raw.get("at")
            if not isinstance(at, str) or not at:
                raise TeamBridgeTypeError(f"events[{index}].at must be a non-empty string")
            refs.append(
                ConversationFeedRef(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    conversation_seq=int(seq_text),
                    mention=mention,
                    sender_kind=kind,
                )
                .validated()
                .encode()
            )
            continue
        subject_kind: str | None = None
        subject: str | None = None
        for kind in _EVENT_ID_KINDS:
            value = ids.get(kind)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise TeamBridgeTypeError(f"events[{index}].ids.{kind} must be a string")
            if subject is None:
                subject_kind, subject = kind, value
        if subject is None:
            raise TeamBridgeTypeError(f"events[{index}].ids carries no subject ref")
        at = raw.get("at")
        if not isinstance(at, str) or not at:
            raise TeamBridgeTypeError(f"events[{index}].at must be a non-empty string")
        refs.append(f"{event_type}:{subject_kind}={subject}")
    raw_cursor = result.get("cursor")
    if raw_cursor is None:
        raise TeamBridgeTypeError("events result.cursor is required on a 200 answer")
    cursor = feed_seq(raw_cursor)
    return cursor, tuple(refs), not refs
