"""The mesh's vocabulary: record shapes, capability names, op tables, refusals.

WHY THIS MODULE IS STDLIB-ONLY. ``local_operator/cli.py`` imports the network
package to register ``lop network``, so every ``lop`` invocation including
``--version`` pays whatever this module imports. Nothing here may reach
``cryptography`` (the handshake), ``socket``/``threading`` (the relay) or
``yaml`` (the config store): the same import-light contract
``session/runtime/types.py`` carries, and the reason its constants are imported
from it rather than re-spelled here.

TWO VERSIONS, DELIBERATELY SEPARATE.
:data:`MESH_PROTOCOL_VERSION` is the LINK protocol — the handshake, the
transcript and the record framing in ``wire.py``/``handshake.py``. It moves only
when one of those three changes. ``PROTOCOL_VERSION`` (imported, never
re-spelled) is the SESSION control protocol the local runtimes speak; the relay
carries it through as ``session_protocol`` and never interprets a session frame,
so a mixed-version fleet degrades per op through the existing unknown-op rule
rather than at the link. Adding a session-level op therefore moves neither
number, and adding a link feature moves neither either — that is what the
capability strings below are for.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # the crypto module is function-local at runtime; this is a type only
    from local_operator.network.wire import LinkKeys

# The session control protocol, imported rather than copied: a second literal
# here is a second thing to forget to bump, and this module is already on the
# CLI startup path with that module for other reasons (it is stdlib-only).
from local_operator.session.runtime.types import PROTOCOL_VERSION

#: The LINK protocol: the handshake frames, the transcript construction and the
#: AEAD record framing. Bumped for a change to any of those three and for
#: nothing else — a new op is a new string in a dispatch table, and an old peer
#: answers ``error: unknown op`` exactly as it does on the session plane.
MESH_PROTOCOL_VERSION = 1

#: Directory (under the config root) holding one record per live mesh RELAY.
#:
#: A FOURTH namespace for the reason ``run/serve`` is a second one: every reader
#: of ``run/mobile`` treats each file there as a SESSION, and ``SessionRecord.kind``
#: is a ``Literal`` those readers pass through unvalidated, so a relay record
#: dropped in beside them would surface as a phantom session with an empty
#: ``session_id`` and no error anywhere. A relay record carries ``network_id``,
#: ``device_id``, ``epoch`` and endpoints — facts a session record must not carry
#: and a session reader would misread. It is also not a ``run/serve`` record:
#: that answers "which install is serving HTTP", this answers "which install is
#: on the mesh, as whom, and on which networks". One install can have either,
#: both, or neither.
#:
#: It lives HERE rather than beside ``RUN_DIRNAME``/``SERVE_RUN_DIRNAME``/
#: ``HOST_RUN_DIRNAME`` in ``session/runtime/types.py`` because this slice may
#: not edit that module; the registry path is parameterised by dirname, so the
#: namespace works identically from here. See ``store.py`` for the accessors.
PEERS_RUN_DIRNAME = "run/peers"

# ---------------------------------------------------------------------------
# Capabilities and roles
# ---------------------------------------------------------------------------

#: The ONE capability vocabulary (convergence round, authoritative): ten
#: names, no synonyms. ``broker:request``, ``broker:grant`` and ``member:admin``
#: were draft names and are not implemented.
CAPABILITIES: frozenset[str] = frozenset(
    {
        "list",
        "view",
        "prompt",
        "steer",
        "stop",
        "slash",
        "delete",
        "move",
        "broker_credential",
        "admin",
    }
)

#: What an invite's ``--role`` grants. Resolved AT ADMISSION and stored on the
#: member row rather than derived at read time, so a later change to this table
#: cannot retroactively widen an existing member's authority — the failure a
#: derived-at-read-time design gets wrong.
ROLE_CAPABILITIES: dict[str, frozenset[str]] = {
    # Read-only is FIRST-CLASS, not a degenerate drive: "let my laptop see the
    # fleet" is a common and much safer ask than "let it drive everything".
    "read": frozenset({"list", "view"}),
    "drive": frozenset({"list", "view", "prompt", "steer", "stop", "slash"}),
    "admin": CAPABILITIES,
}

ROLES: tuple[str, ...] = ("read", "drive", "admin")

#: Roles an invite may name. Deliberately includes ``admin``: an operator
#: pairing their own second machine needs it, and the human SAS step is what
#: makes the grant deliberate.
INVITE_ROLES: tuple[str, ...] = ("read", "drive", "admin")


def capabilities_for_role(role: str) -> frozenset[str]:
    """The capability set a role grants, or :class:`KeyError` for an unknown role.

    Raising rather than defaulting is the point: an unknown role silently
    resolving to ``read`` or to ``admin`` is a grant nobody decided.
    """
    if role not in ROLE_CAPABILITIES:
        raise KeyError(f"unknown role {role!r}; known roles are {', '.join(ROLES)}")
    return ROLE_CAPABILITIES[role]


# ---------------------------------------------------------------------------
# The session row's ``pending`` vocabulary
# ---------------------------------------------------------------------------

#: WHAT A PERSON IS BEING WAITED ON, and the ONE spelling of it.
#:
#: ``SessionRecord.pending`` owns this vocabulary (``approval`` / ``ask`` /
#: ``None``; see ``session/runtime/types.py``): ``lop sessions`` prints the value
#: raw in its NEEDS column and the sidebar maps it to "Approval needed"/"Answer
#: needed", so a second spelling for one fact shows up as a column nobody can
#: read. The federated row carries the SAME strings (§9.2), and the stored half
#: of a catalogue derives its claim from the attention store's ``unseen`` flag —
#: an unread completion IS the operator being awaited, and it is not an
#: approval, so that half publishes :data:`NEEDS_ASK`.
NEEDS_APPROVAL = "approval"
NEEDS_ASK = "ask"


def normalise_pending(value: object) -> str | None:
    """The ONE reader of a ``pending`` value, whichever producer wrote it.

    A STRING IS THE CONTRACT and ``None`` is "no claim", never ``False``: the
    field is read by the federated listing, the sidebar and the picker, and a
    truthy NON-string reaching any of them is how ``lop sessions --all-peers``
    died with ``TypeError: object of type 'bool' has no len()`` (QA round 7,
    Q-R7-1) — the crashing row carried ``True``, because a producer had answered
    "is there an unread completion" in the field that asks "what is needed".

    A BOOLEAN is therefore TRANSLATED rather than echoed: the only build that
    ever wrote one wrote it from that same ``unseen`` flag, so ``True`` reads as
    :data:`NEEDS_ASK` — the claim it meant — while ``"True"`` (the
    stringification a reviewer flagged) can no longer be produced. ``False``,
    ``None``, an empty string and anything that is neither a string nor a bool
    are all "no claim".
    """
    if isinstance(value, bool):
        return NEEDS_ASK if value else None
    if not isinstance(value, str):
        return None
    return value.strip() or None


# ---------------------------------------------------------------------------
# Op vocabularies
# ---------------------------------------------------------------------------

#: The peer-scope ops: relay → relay over an authenticated MEMBER link. The
#: session-plane ops travelled by ``net_forward`` are ``ControlOp`` values and
#: are deliberately absent here (see :data:`INNER_OP_CAPABILITY`).
#:
#: ``net_stream`` is the ONE name this slice adds beyond the design's vocabulary,
#: and it is a CARRIER rather than a new capability surface: ``net_forward``
#: carries one frame and returns its reply, which cannot express a viewer
#: connection (a welcome, then a continuous stream of events with the viewer's
#: frames interleaved). Its capability row is ``view`` — the act of opening is a
#: read — and every frame written down it is resolved through
#: :data:`INNER_OP_CAPABILITY` exactly as ``net_forward``'s inner frame is, so it
#: is not a way round the capability model.
NET_OPS: tuple[str, ...] = (
    "net_reconcile",
    "net_catalog",
    "net_member_list",
    "net_epoch",
    "net_leave",
    "net_panic",
    "net_trust",
    "net_identity_rotate",
    "net_forward",
    "net_stream",
    "net_sync",
    "net_broker",
    "net_session_lifecycle",
    "net_session_move",
    "net_session_create",
    "net_session_engage",
    "net_session_stop",
    "net_bye",
    "ping",
)

#: The pairing ceremony's ops. They are NOT in :data:`OP_CAPABILITY` and this is
#: deliberate: a pair-phase link has no member row yet, so there is nothing to
#: authorise against — its authorisation IS the invite validation plus the two
#: human confirmations. Dispatch refuses them outside phase ``pair``, and a test
#: asserts the two directions of that rule.
NET_PAIR_OPS: tuple[str, ...] = ("net_pair_ready", "net_pair_abort", "net_pair_result")

#: The relay's LOCAL control-socket ops (viewer → relay, authorised by the
#: control key of the peers record). These never reach a peer link: a name from
#: this tuple appearing in :data:`OP_CAPABILITY` is a bug, not a gap, and the
#: totality test asserts both directions.
#:
#: THE ``peer_*`` NAMES ARE THIS SLICE'S, and they are deliberately not ``net_*``:
#: the local surface has its own vocabulary so a reader can tell from the frame
#: alone which boundary it crossed (the ``stream_*`` precedent), and a viewer
#: asking ITS OWN relay to ask a peer is a local act. ``peer_session_create`` /
#: ``_engage`` / ``_stop`` are the client half of the design's three peer ops;
#: ``peer_session_rows`` and ``peer_session_facts`` are the catalogue and the
#: resolver's live read.
LOCAL_OPS: tuple[str, ...] = (
    "net_status",
    "net_ls",
    "net_show",
    "net_init",
    "net_rename",
    "net_rm",
    "net_invite",
    "net_join",
    "net_member_rm",
    "net_peer_ls",
    "net_disconnect",
    "net_panic_local",
    "net_trust_local",
    "net_log",
    "net_doctor",
    "stream_open",
    "stream_send",
    "stream_close",
    "peer_session_rows",
    "peer_session_facts",
    "peer_session_create",
    "peer_session_engage",
    "peer_session_stop",
    # P0 PLUMBING (mesh build plan §0 finding 3): the LOCAL half of a capability
    # grant. A member row's capabilities are resolved at admission and every
    # device keeps its OWN copy (``relay.adopt_members`` rule 2), so widening what
    # a peer may do HERE is a write to this device's row and nothing else — a
    # local act with a local name, never a wire op.
    "net_member_caps",
    # The three verbs the later slices serve (§1.3). Declared here, in P0, so the
    # totality rule ("a LOCAL_OPS name never appears in OP_CAPABILITY") already
    # holds for them and no slice has to edit this tuple: the names are this
    # tree's local vocabulary, deliberately not ``net_*`` (``relay.py``'s
    # control-handler comment gives the rule).
    "session_move",
    "session_sync",
    "session_lifecycle",
    # The credential broker's leg 1 (runtime → its own relay, §2.3).
    "credential_grant",
    "credential_report",
    "credential_placement",
)

#: The phases ``net_session_move`` carries (§1.3 of the build plan). Declared
#: next to the op tables because the AUTHORISER reads them: ``status``, ``ready``
#: and ``done`` must still reach a device that has already TOMBSTONED the id (the
#: §6.5 recovery — the destination asks the source what happened), and ``invite``
#: reaches the DESTINATION, which by definition does not own the id yet. See
#: ``authorizer.Authorizer._session_scope`` for the carve-outs.
MOVE_PHASES: tuple[str, ...] = ("status", "prepare", "ready", "done", "invite")

#: Move phases the SOURCE must answer for an id it has already handed away. A
#: ``prepare`` is deliberately absent: preparing a session this device no longer
#: holds is exactly the double-writer INV-1 forbids, so the scope rule refuses it.
MOVE_PHASES_AFTER_HANDOFF: frozenset[str] = frozenset({"status", "ready", "done"})

#: Capabilities ``lop network member grant`` may add to a peer's LOCAL row.
#: ``admin`` is absent ON PURPOSE: admin is a role granted by an admin invite and a
#: human SAS step (``INVITE_ROLES``), and a one-line CLI verb that could mint one
#: would be a way round that ceremony. ``broker_credential`` IS grantable — it is
#: what ``credential share`` writes on the owner (§2.2) — but only by a device
#: whose own row holds ``admin``, which is the "stays admin-only" rule: the
#: capability is never in a non-admin ROLE, and only an admin may hand it out.
GRANTABLE_CAPABILITIES: frozenset[str] = frozenset(
    {"list", "view", "prompt", "steer", "stop", "slash", "delete", "move", "broker_credential"}
)

#: What each capability lets the peer DO, in words, for ``member grant/revoke``'s
#: human output. A capability name alone ("broker_credential") does not tell the
#: operator what they just allowed.
CAPABILITY_WORDS: dict[str, str] = {
    "list": "see this device's sessions",
    "view": "watch a session here",
    "prompt": "send prompts to a session here",
    "steer": "steer a running turn here",
    "stop": "stop a session here",
    "slash": "run slash commands in a session here",
    "delete": "archive or delete a session here",
    "move": "move sessions to or from this device",
    "broker_credential": "borrow this device's logins",
    "admin": "administer the network",
}

#: The capability each peer-scope op requires. ``None`` means "an authenticated
#: member at the current epoch, no capability needed" and exists for exactly one
#: op — ``net_bye``, the graceful teardown, which every member may always send.
OP_CAPABILITY: dict[str, str | None] = {
    "net_reconcile": "list",
    "net_catalog": "list",
    "net_member_list": "list",
    "net_epoch": "admin",
    # ``net_leave`` is ``list``: a member announcing its own departure is the
    # cheapest honest signal there is, and gating it higher would leave a
    # read-only member unable to leave cleanly.
    "net_leave": "list",
    # ``net_panic`` is ``list`` on purpose: any member that DETECTS a compromise
    # must be able to raise the alarm. The DoS that enables is bounded (it forces
    # a re-admit, it leaks and destroys nothing) and it is audited with the
    # sender's id; a suppressed alarm costs the network, a false one costs an
    # operator action.
    "net_panic": "list",
    # ``net_trust`` is ``admin``: the convergence round's one capability
    # vocabulary has no ``trust`` member, so re-admitting a network after a panic
    # is an administrative act rather than a capability of its own.
    "net_trust": "admin",
    "net_identity_rotate": "admin",
    "net_forward": None,  # resolved to the INNER op's capability; see effective_op
    # ``net_stream`` needs ``view`` to OPEN a pipe, and every frame written down
    # that pipe is resolved through INNER_OP_CAPABILITY at the receiving relay —
    # so a read-only member may open a stream and may not prompt through it.
    "net_stream": "view",
    "net_sync": "view",
    "net_broker": "broker_credential",
    "net_session_lifecycle": "delete",
    "net_session_move": "move",
    # The three the mobility design adds (§2.2): creating a session IS a prompt
    # (it admits one), warming a cold one is a read, and the kill switch is its
    # own capability.
    "net_session_create": "prompt",
    "net_session_engage": "view",
    "net_session_stop": "stop",
    "net_bye": None,
    "ping": "list",
}

#: The capability each SESSION-plane op needs when it arrives wrapped in
#: ``net_forward``. Every ``ControlOp`` name must appear here, which is what
#: makes "a new op cannot be added without deciding who may call it" true; the
#: totality test fails by name.
#:
#: Two of these are judgement calls the design left open, resolved toward the
#: CONSERVATIVE end and recorded here so a reviewer can disagree with a reason:
#: ``variables`` (the live eval-kernel namespace) and ``complete_aside`` (an
#: authoritative off-record provider request) are ``prompt``-level, because both
#: spend the peer's context or mutate its live state; ``resume_session`` (rebind
#: the runtime to another transcript) and ``new_conversation`` are ``prompt`` and
#: ``stop`` respectively, because a peer must not be able to silently repoint a
#: session that is mid-turn — a viewer that owns the session tree enough to
#: resume another conversation is driving it, so ``prompt``/``stop`` bound the
#: damage without inventing a capability the spine does not name.
INNER_OP_CAPABILITY: dict[str, str] = {
    # prompt — "start/continue a turn, answer an approval or an ask"
    "prompt": "prompt",
    "approval_answer": "prompt",
    "ask_answer": "prompt",
    "set_model": "prompt",
    "set_effort": "prompt",
    "new_conversation": "prompt",
    "complete_aside": "prompt",
    "peer_message": "prompt",
    "variables": "prompt",
    # steer
    "steer": "steer",
    "recall_steer": "steer",
    # stop — "abort, cancel, stop (the kill switch), retire_if_pristine"
    "abort": "stop",
    "cancel": "stop",
    "stop": "stop",
    "retire_if_pristine": "stop",
    "resume_session": "stop",
    # slash
    "slash": "slash",
    # view — reading the session
    "snapshot": "view",
    "watch": "view",
    "unwatch": "view",
    # list — liveness only
    "ping": "list",
    # THE OPS A REAL VIEWER SENDS THAT ARE NOT IN ``ControlOp`` (mesh slice V).
    # The totality test above keys on the ``ControlOp`` literal, but the runtime's
    # dispatch serves more than that literal names — the payload ops an
    # ``AttachedSession`` issues on every bind and every routed slash. Without a
    # row here the relay refused each one ("has no capability decision"), so a
    # remote viewer could prompt but could not page history, sync its canonical
    # state, or run one routed slash command (`/rename`, `/model`, `/goal` all
    # travel as ``slash_result``). Measured over two real relays with a real peer
    # Session. Each is the NARROWEST capability already covering its effect, and
    # an op NOT listed here is still refused — ``_accept_stream_frame`` answers
    # ``unknown_op`` for a missing row (relay.py, "has no capability decision").
    #
    # Every op here acts on the session the STREAM is bound to: the frame carries
    # no session id of its own, and the peer relay's dial is to that one runtime's
    # socket (``_session_scope`` authorised the id at ``stream_open``).
    #
    # view — reads, and presence hints; none mutates the conversation.
    "frontend_sync": "view",  # the canonical state the welcome already carries
    "history_page": "view",  # older transcript pages; `view` already streams them live
    "job_trajectory": "view",  # a background job's own transcript page, read-only
    "watch_job": "view",  # subscribe to a job's events; no effect on the job
    "unwatch_job": "view",  # the matching unsubscribe
    "viewer_watch": "view",  # "someone is displaying this" — residency hint only
    "desktop_watch": "view",  # the desktop's presence lease — same hint, other surface
    "event_mute": "view",  # stop sending THIS connection deltas; affects nobody else
    "acknowledge_attention": "view",  # clear an unread-completion mark this viewer rendered
    # slash — the authoritative slash seam. The owner's own dispatch decides what
    # each command does and refuses the terminal-only ones
    # (serving.py ``run_slash_authoritative`` / ``slash``: "terminal-only here").
    "slash_result": "slash",
    # prompt — session-scoped WRITES. The threshold is design §3.3: a member that
    # may prompt can already make the agent (which has a shell) do each of these.
    #
    # fork_snapshot copies THE STREAM'S OWN session into a new id on the OWNER's
    # store and returns only ``{fork_id, parent_id, busy, incomplete}``
    # (session.py:6012 ``Session.fork_snapshot``; server.py:5917 refuses any frame
    # field but ``message``, so no other session or path can be named). It exposes
    # no content the stream's `view` does not already carry, and the fork stays on
    # the owner, readable only through that owner's own authorisation.
    "fork_snapshot": "prompt",
    # credential / mcp_credentials are WRITE-ONLY toward the owner: they carry a
    # value IN and return only names and outcomes, never material. Proof:
    # session/credential_ops.py:59/:66 (``list``/``names`` → key names and a
    # source label), :81 (``persist`` → key + outcome sentence), the ``store`` arm
    # (→ key + replaced); mcp/credentials.py:129-130 (``store_credentials`` →
    # ``saved_ids``/``failed_ids``/``code``); serving.py ``credential_op``'s
    # docstring ("The value is never logged, never journalled, and never
    # returned"). Borrowing a token FROM the owner is a different act and needs
    # ``broker_credential`` (the credentials slice); nothing here reads one out.
    "credential": "prompt",
    "mcp_credentials": "prompt",
    # record_shell appends a `! cmd` receipt row to the transcript (a turn-shaped
    # write); adopt_aside adds an aside's turns to the conversation.
    "record_shell": "prompt",
    "adopt_aside": "prompt",
    # register_secret_redaction accepts ONE value and adds it to the owner's
    # redaction set — it only makes the owner scrub more; it returns ``True`` and
    # nothing is read back (serving.py:1205 ``register_secret_redaction``: "never a
    # credential, never an announcement, never a log line").
    "register_secret_redaction": "prompt",
    # stop — ends work or ends the runtime.
    "cancel_subagents": "stop",  # the second-Esc path: cancels this session's children
    # retire_now / refresh_if_idle CANNOT INTERRUPT WORK IN FLIGHT: both go
    # through ``RuntimeServer._retire_for`` (server.py:5540), which asks the
    # handle's own idle predicate (``may_refresh``) first and answers
    # ``kept: <reason>`` for a busy runtime (server.py:5569) — a wrong "retire"
    # costs a cold start, never a turn. ``retire_now {exclusive}`` additionally
    # refuses while another viewer is attached.
    "retire_now": "stop",
    "refresh_if_idle": "stop",
    # DELIBERATELY ABSENT, and therefore refused: ``operator_challenge``. It
    # produces material a surface uses to SIGN as the operator, and no capability
    # in the transport's set grants a peer that authority.
}


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class MeshRefusal(Exception):
    """A refusal that carries a machine ``code`` and a sentence for the operator.

    EVERY refusal path refuses CLOSED and names its reason in the operator's
    language: the code is what the audit log and the tests key on, the sentence
    is what a person reads. The two are separate because the sentence is often
    NOT what the peer is told — an authorisation refusal tells the peer only that
    it was refused, while the local audit record keeps the real cause.
    """

    def __init__(self, code: str, sentence: str) -> None:
        super().__init__(sentence)
        self.code = code
        self.sentence = sentence

    def __str__(self) -> str:  # pragma: no cover - Exception's own repr is enough
        return self.sentence


class Refusal(MeshRefusal):
    """The authoriser's refusal (``authorizer.Authorizer.check``)."""


class HandshakeRefusal(MeshRefusal):
    """A handshake step refused. The socket closes with no reply frame."""


class PairingRefusal(MeshRefusal):
    """A pairing step refused (invite validation, SAS, admission)."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

MemberKind = Literal["device", "pool"]
MemberLifecycle = Literal["active", "provisioning", "draining", "expired"]
TrustState = Literal["active", "untrusted", "disconnected"]
InviteState = Literal["minted", "redeemed", "consumed"]
LinkPhase = Literal["member", "reconcile", "pair"]
Outcome = Literal["ok", "refused", "failed", "partial", "admitted", "aborted"]

#: The three trust states as a runtime set, mirrored by the reader below — which
#: is the ONE place a string becomes a :data:`TrustState`, whether it arrived in
#: another device's frame or on an operator's command line. The set is exported so
#: a future caller can ask "is this one of them" without restating the list.
TRUST_STATES: frozenset[str] = frozenset({"active", "untrusted", "disconnected"})


def trust_state(value: object) -> TrustState:
    """``value`` as a :data:`TrustState`, or :class:`MeshRefusal` naming it.

    One reader means one refusal code and one sentence instead of one per caller,
    and it is deliberately EXACT rather than coercing: a trust value that cannot
    be read must never be quietly promoted to ``"active"``, which is the one
    direction where mis-reading it would matter. The refusal is ``bad_trust``,
    the code every caller that used to validate this by hand already raised, so
    moving the check here changes no contract — it removes the copies.
    """
    if value == "active":
        return "active"
    if value == "untrusted":
        return "untrusted"
    if value == "disconnected":
        return "disconnected"
    raise MeshRefusal("bad_trust", f"unknown trust state {value!r}")


@dataclass
class MemberRecord:
    """One device's membership in one network.

    A REMOVED MEMBER IS A TOMBSTONE, never a deleted row: ``removed_at`` is set,
    ``lifecycle`` is ``expired``, and the id can never be re-added. Tombstones
    make revocation auditable and make a re-pair a genuinely new identity; the
    ``removed_ids`` list on the network record is what actually enforces the
    burn, so pruning the row after the retention window cannot un-burn the id.
    """

    device_id: str
    #: The member's Ed25519 public key, base64url — the thing every later
    #: handshake is verified against. An id is a name, never authority.
    public_key: str = ""
    name: str = ""
    kind: MemberKind = "device"
    lifecycle: MemberLifecycle = "active"
    role: str = "drive"
    #: Resolved at admission and stored, never derived at read time.
    capabilities: list[str] = field(default_factory=list)
    added_at: float = field(default_factory=time.time)
    added_by: str = ""
    #: ``self`` (this device created the network), ``invite``, or ``rotation``.
    added_via: str = "invite"
    endpoints: list[str] = field(default_factory=list)
    last_seen_at: float | None = None
    last_seen_instance: str = ""
    #: How many times a second live process has claimed this device id outside
    #: the restart grace window — the copied-key detector's evidence.
    duplicate_count: int = 0
    suspect: bool = False
    #: Device ids this row was known by before a key rotation, kept for a bounded
    #: window so a link that authed at the old id is not cut mid-turn.
    previous_ids: list[str] = field(default_factory=list)
    #: The rotation STATEMENT that produced this row (``identity.rotation_statement``,
    #: signed by the old key), or ``{}`` when the row was never rotated. It is the
    #: continuity proof, and it rides on the row because the row is the only place a
    #: peer that never saw the ``net_identity_rotate`` frame can get it: the member
    #: table is a snapshot, and a table delivered after a rotation otherwise shows a
    #: device as a new id with no way to tell that from an impostor claiming a
    #: member's name with a new key (QA round 16, Q16-1). Additive: a build that does
    #: not know the field drops it on parse (``_known``) and simply cannot retire the
    #: superseded row, which is where this codebase was before it existed.
    rotation_proof: dict[str, Any] = field(default_factory=dict)
    rotated_at: float | None = None
    removed_at: float | None = None
    removed_by: str | None = None

    @property
    def active(self) -> bool:
        """A live member: not removed, and not expired by any other route."""
        return self.removed_at is None and self.lifecycle != "expired"

    def has(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> MemberRecord:
        fields = _known(MemberRecord, data)
        # A peer's JSON can carry ``null`` or a list where this field expects an
        # object, and a row that fails to parse is a member this device forgets — so
        # an unusable proof degrades to "no proof", never to a raise.
        if not isinstance(fields.get("rotation_proof"), dict):
            fields["rotation_proof"] = {}
        return MemberRecord(**fields)


@dataclass
class PendingPairing:
    """An inbound pairing waiting for the INVITER's human to confirm the code.

    WHY THIS EXISTS ON DISK. The inviter's second hand (design §5.3: "B
    transcribes, A compares") happens on a device whose relay is usually a
    launchd daemon with no terminal. The code therefore has to be somewhere a
    human can reach it: this record is written the instant the transcript is
    fixed, the relay prints it when it HAS a terminal, and otherwise the operator
    answers with ``lop network confirm``, which reads this record and writes the
    decision beside it.

    ``sas`` is in here and nowhere else. It is a value derived from THIS device's
    own handshake and it is what the human compares — so it lives in a 0600 file
    under a 0700 directory for the seconds the ceremony lasts, is never sent to
    the peer in either direction (§5.3), and never reaches the audit log (the
    never-log list covers it). The record is deleted with the decision.
    """

    invite_id: str
    network_id: str
    network_name: str
    joiner_device_id: str
    joiner_name: str
    #: THIS device's derivation, and the value its human must see on the other
    #: device's screen.
    sas: str
    fingerprint: str
    #: What the joiner's human typed, from the ``net_pair_ready`` frame. Kept so
    #: the inviter's human compares two values rather than trusting one.
    transcribed: str = ""
    peer_addr: str = ""
    issued_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    #: The rendered question, so every surface (the relay's log, `lop network
    #: confirm`, `--json`) shows the SAME words. Rendering it twice is how two
    #: prompts drift apart.
    prompt: str = ""
    schema: int = 1

    def seconds_left(self, now: float | None = None) -> float:
        return max(0.0, self.expires_at - (time.time() if now is None else now))

    def is_open(self, now: float | None = None) -> bool:
        return self.seconds_left(now) > 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> PendingPairing:
        return PendingPairing(**_known(PendingPairing, data))


@dataclass
class PairDecision:
    """The inviter's human answer to a :class:`PendingPairing`.

    ``admit`` is only ever written when BOTH transcriptions matched this device's
    derivation — the joiner's (in the frame) and the inviter's human (here) — so a
    single mistyped digit anywhere ends the ceremony, and the invite is consumed
    either way.
    """

    invite_id: str
    decision: Literal["admit", "decline"]
    matched: bool = False
    reason: str = ""
    answered_by: str = "human"
    answered_at: float = field(default_factory=time.time)
    schema: int = 1

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> PairDecision:
        return PairDecision(**_known(PairDecision, data))


@dataclass
class InviteRecord:
    """One minted invite, and its single-use state.

    ``state`` is on disk, so a relay restart mid-pairing cannot be used to replay
    an invite: ``redeemed`` is written the instant a valid redemption arrives
    (before any human sees anything) and ``consumed`` when the ceremony ends in
    EITHER outcome.
    """

    invite_id: str
    minted_at: float = field(default_factory=time.time)
    #: The epoch the token was minted at. A rotation during the invite's life
    #: invalidates it (``invite_epoch_stale``), and the comparison needs the minted
    #: epoch stored rather than re-derived from the token, which the inviter no
    #: longer has by the time it admits.
    epoch: int = 0
    #: A DURATION, not an absolute expiry, so no cross-host clock comparison ever
    #: enters pairing: the inviter enforces freshness against its own clock.
    ttl_s: float = 600.0
    role: str = "drive"
    capabilities: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    state: InviteState = "minted"
    #: OPTIONAL device binding (convergence round): when set, only that device id
    #: may redeem the token. Empty means unbound — the ordinary invitation case.
    device_id: str = ""
    redeemed_by: str = ""
    redeemed_at: float | None = None
    outcome: str = ""
    #: How many CODE guesses this invite has spent: incremented by
    #: :func:`local_operator.network.invite.release` when a ceremony failed on a code
    #: that was compared and disagreed. A delay does not spend one (no guess was made),
    #: and neither does anything that consumes the token outright — a decline or a
    #: device-id conflict ends it, and ``consume`` does not touch this field.
    #: :func:`local_operator.network.invite.failures_exhausted` reads it against
    #: :data:`local_operator.network.invite.PAIRING_MAX_FORGIVEN_FAILURES`, and it lives
    #: on the record so a relay restart cannot hand the same token a fresh budget.
    attempts: int = 0

    @property
    def expires_at(self) -> float:
        """The expiry as the MINTING device's clock sees it — local convenience.

        Never transmitted (the envelope carries ``ttl_s``) and never compared
        across hosts; it exists so a prompt or a listing can say "10 minutes".
        """
        return self.minted_at + self.ttl_s

    def is_fresh(self, now: float) -> bool:
        return now < self.expires_at

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> InviteRecord:
        return InviteRecord(**_known(InviteRecord, data))


@dataclass
class NetworkRecord:
    """One network on this device: membership, epoch, trust, invites.

    THE SECRET IS NOT HERE. It lives in a sibling ``<network_id>.secrets.json``
    so that the record — which ``lop network show --json`` dumps, which the
    desktop route returns, and which a future syncer copies — has no field a
    surface has to remember to redact. That inversion is the same one the secret
    store enforces: no surface returns a value to the model.
    """

    network_id: str
    name: str
    epoch: int = 1
    schema: int = 1
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    #: Increments on every write and is carried in epoch broadcasts, so a
    #: receiver can tell "I already have this" from "this is newer". It is NOT a
    #: Lamport clock for the member list — the epoch's ``min(device_id)`` rule is
    #: what makes concurrent rotations converge.
    sequence: int = 0
    trust: TrustState = "active"
    untrusted_reason: str = ""
    self_device_id: str = ""
    self_role: str = "admin"
    self_capabilities: list[str] = field(default_factory=list)
    #: ``{address, port, advertised: [...]}`` — what this device publishes.
    listen: dict[str, Any] = field(default_factory=dict)
    #: epoch → the device id that initiated the rotation. The tie-break for
    #: concurrent rotations reads this, and it is also the rotation lock.
    rotations: dict[str, str] = field(default_factory=dict)
    members: list[MemberRecord] = field(default_factory=list)
    #: A join in progress: ``{invite_id, device_id, public_key, name, role,
    #: started_at}``. Kept so a crash mid-ceremony leaves a visible pending row
    #: rather than a half-written member.
    pending: list[dict[str, Any]] = field(default_factory=list)
    invites: list[InviteRecord] = field(default_factory=list)
    #: Ids burned forever by a removal, kept even after the tombstone row is
    #: pruned — this list is what actually prevents a re-admission.
    removed_ids: list[str] = field(default_factory=list)
    #: Refused-by-peers evidence: a device that was removed while offline learns
    #: it on its next dial, and says so rather than showing a healthy network it
    #: cannot reach.
    stale: str = ""
    #: Unix seconds before which a second LOCAL rotation is refused
    #: (``rotation_in_progress``).
    rotation_lock_until: float = 0.0

    def member(self, device_id: str) -> MemberRecord | None:
        for row in self.members:
            if row.device_id == device_id or device_id in row.previous_ids:
                return row
        return None

    def active_members(self) -> list[MemberRecord]:
        return [row for row in self.members if row.active]

    def invite(self, invite_id: str) -> InviteRecord | None:
        for row in self.invites:
            if row.invite_id == invite_id:
                return row
        return None

    def self_member(self) -> MemberRecord | None:
        return self.member(self.self_device_id)

    def is_burned(self, device_id: str) -> bool:
        return device_id in self.removed_ids

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["members"] = [row.to_json() for row in self.members]
        payload["invites"] = [row.to_json() for row in self.invites]
        return payload

    @staticmethod
    def from_json(data: dict[str, Any]) -> NetworkRecord:
        fields = _known(NetworkRecord, data)
        fields["members"] = [
            MemberRecord.from_json(row)
            for row in data.get("members") or []
            if isinstance(row, dict)
        ]
        fields["invites"] = [
            InviteRecord.from_json(row)
            for row in data.get("invites") or []
            if isinstance(row, dict)
        ]
        # A JSON object with non-string keys (``{"7": "d_…"}``) is what this
        # round-trips to, and it must stay a dict: the epoch lookup is by string.
        rotations = data.get("rotations")
        if isinstance(rotations, dict):
            fields["rotations"] = {str(key): str(value) for key, value in rotations.items()}
        return NetworkRecord(**fields)


@dataclass
class SecretState:
    """The two retained epoch secrets. Never more than two, by construction.

    ``current`` authenticates member links; ``previous`` is accepted ONLY for a
    ``reconcile``-phase handshake, so a stolen old secret is worth one rotation
    generation and nothing else. A third is dropped on rotation.
    """

    network_id: str
    epoch: int
    secret: str = ""  # base64url(32), current epoch
    previous_epoch: int | None = None
    previous_secret: str = ""
    schema: int = 1

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "network_id": self.network_id,
            "current": {"epoch": self.epoch, "secret": self.secret},
        }
        if self.previous_epoch is not None and self.previous_secret:
            payload["previous"] = {
                "epoch": self.previous_epoch,
                "secret": self.previous_secret,
            }
        return payload

    def rotate(self, new_secret: str, new_epoch: int) -> None:
        """Adopt a new epoch, keeping exactly one generation of history."""
        self.previous_epoch = self.epoch
        self.previous_secret = self.secret
        self.epoch = new_epoch
        self.secret = new_secret

    @staticmethod
    def from_json(data: dict[str, Any]) -> SecretState:
        current = data.get("current") or {}
        previous = data.get("previous") or {}
        if not isinstance(current, dict) or not isinstance(previous, dict):
            raise ValueError("secrets file: 'current'/'previous' must be objects")
        return SecretState(
            network_id=str(data.get("network_id") or ""),
            epoch=int(current.get("epoch") or 0),
            secret=str(current.get("secret") or ""),
            previous_epoch=int(previous["epoch"]) if previous.get("epoch") is not None else None,
            previous_secret=str(previous.get("secret") or ""),
            schema=int(data.get("schema") or 1),
        )


@dataclass
class PeerRecord:
    """The relay's discovery record, published under ``run/peers``.

    The file is keyed by pid and mode 0600 inside a 0700 directory, exactly like
    a session record, so the ``control_key`` it carries is protected by the
    account and the loopback control socket needs no credential of its own.

    TWO ABSENCES, AS PROPERTIES: it never contains the device private key, and
    it never contains a network secret. Its docstring says so because "a record
    that is dumped by every status command" is the natural place for someone to
    add a helpful field.
    """

    pid: int
    #: A constant, present so a reader that globbed the wrong directory sees it.
    kind: str = "relay"
    protocol: int = MESH_PROTOCOL_VERSION
    #: What the local session runtimes speak — passed through, never interpreted.
    session_protocol: int = PROTOCOL_VERSION
    device_id: str = ""
    #: Operator-set, cosmetic, never authority.
    device_name: str = ""
    #: Per-process: minted at relay start and bound into the handshake transcript.
    instance_id: str = ""
    control_port: int = 0
    control_key: str = ""
    #: ``{address, port, advertised: [...]}``
    listen: dict[str, Any] = field(default_factory=dict)
    networks: list[dict[str, Any]] = field(default_factory=list)
    links: int = 0
    #: Link feature strings (``mesh-net-v1``, ``credential-broker-v1``, …).
    capabilities: list[str] = field(default_factory=list)
    version: str = ""
    source_ref: str = ""
    install_root: str = ""
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> PeerRecord:
        # Tolerate unknown keys: a record written by a newer binary must not break
        # an older reader mid-upgrade, the same contract every record here keeps.
        return PeerRecord(**_known(PeerRecord, data))


def _known(cls: type[Any], data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys this build does not know, so a newer peer's record still parses.

    Shared rather than repeated because it is the forward-compatibility contract
    of every record in this module, and four copies of it are four places to get
    the ``pid``/``heartbeat_at`` requirement subtly wrong.
    """
    known = set(cls.__dataclass_fields__)
    return {key: value for key, value in data.items() if key in known}


# ---------------------------------------------------------------------------
# Link-scoped types (the authoriser's inputs and outputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkContext:
    """Everything the authoriser is allowed to know about one link.

    Frozen and self-contained so a refusal can be reasoned about from the
    context alone: the epoch the link authed at, the capabilities resolved from
    the MEMBER ROW at admission (not recomputed from the role now), and the phase
    that decides which ops exist at all.
    """

    link_id: str
    device_id: str
    instance_id: str
    network_id: str
    epoch: int
    capabilities: frozenset[str]
    phase: LinkPhase
    peer_addr: str = ""


@dataclass(frozen=True)
class Granted:
    """The authoriser's answer: what was admitted, for the audit record."""

    action: str
    session_id: str | None = None
    capability: str | None = None


@dataclass
class HandshakeResult:
    """What both roles end up holding after ``welcome``.

    ``sas`` is derived and never transmitted; ``transcript`` is kept because the
    fingerprint a human may compare is a function of it, and ``keys`` is the
    only thing that can decrypt a record on this link.
    """

    role: Literal["dialer", "listener"]
    peer_device_id: str
    peer_instance_id: str
    peer_public_key: str
    network_id: str
    epoch: int
    phase: LinkPhase
    link_id: str
    #: ``wire.LinkKeys``. Typed through ``TYPE_CHECKING`` so this module stays free
    #: of the ``cryptography`` import while still being precise at a call site.
    keys: LinkKeys
    sas: str
    transcript_hash: str
    session_protocol: int = PROTOCOL_VERSION
    peer_build: dict[str, Any] = field(default_factory=dict)
    peer_capabilities: list[str] = field(default_factory=list)
    network_name: str = ""
