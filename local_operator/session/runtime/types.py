"""Wire and discovery types for the session runtime.

These moved here from :mod:`local_operator.mobile.types` because they were
never about the phone. A record, a heartbeat, a client-kind and an attach cap
describe **one session made reachable over a loopback control socket** — the
phone daemon is one client of that, an attach terminal is another, and wakes
and background automations are the next. Keeping them under ``mobile/`` made
every non-phone consumer import a package named for a front end it does not
use, and made the phone look like the owner of a mechanism it merely borrows.

Stdlib-only and import-light by contract: the runtime publishes its record on the
CLI startup path, so anything imported here is paid by every ``lop``
invocation including ``--version``. ``tests/unit/test_import_graph.py`` pins
that. In particular this module must never reach asyncio, pydantic, or
:mod:`local_operator.session.session`.

``local_operator.mobile.types`` re-exports every name below, so the phone
stack's imports keep working unchanged (§8.3 of the design).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

#: Bumped on any breaking change to control frames or web payloads. The
#: runtime and daemon always ship together; the phone UI learns the
#: version in its bootstrap payload and can warn on a stale cached bundle.
#:
#: v2 (attach + reaping) added the ``watch``/``unwatch`` ops, the auth
#: frame's optional ``client`` field, and multi-connection runtimes. The
#: bump is load-bearing for ATTACH specifically: an old (v1) runtime
#: treats any authenticated dial as THE daemon and evicts the real one, so
#: an attach client must refuse to dial a record whose ``protocol`` is < 2
#: rather than silently breaking the owner's phone bridge. The record's
#: version field is the only pre-dial gate — the socket itself speaks the
#: same frame shapes either side of the bump.
#:
#: v4 (full-TUI attach) is ADDITIVE: an attach client's auth frame may carry
#: ``"events": true`` to subscribe to the owner's raw ``AgentEvent`` relay,
#: and the ``recall_steer`` op lets a follower unsend a queued steer. A v3 attach
#: client that omits the flag gets exactly the v3 behaviour (projection
#: frames only), and daemon connections never see the new frames, so the
#: phone path is byte-identical across the bump.
# v5 adds an attach-only canonical frontend state channel. Phone/daemon
# connections still receive projection frames only; the new capability is
# negotiated explicitly by full TUI clients.
#
# It lives HERE rather than in mobile/types.py only because SessionRecord
# defaults ``protocol`` to it: splitting them would make the mobile re-export
# shim import this module and this module import the shim. The move is not a
# bump — the value is unchanged and no frame shape moved with it.
PROTOCOL_VERSION = 5

# Additive attach metadata: a desktop proxy is not itself a person watching.
# Clients must negotiate this before connecting or an old owner would count
# their background socket as a terminal and suppress its fallback notification.
DESKTOP_WATCH_CAPABILITY = "desktop-watch-v1"
DESKTOP_WATCH_LEASE_S = 45.0

#: Which side of the owner relationship a control connection speaks for.
#: ``daemon`` (the default when the auth frame omits ``client``) may rebind
#: the owner's conversation; ``attach`` is a follower terminal that may
#: watch and steer but never rebind. Absent-means-daemon keeps an OLD
#: daemon dialing a NEW runtime on the same class it always had.
ClientKind = Literal["daemon", "attach"]

#: Whether the human driving a control connection sits at THIS machine.
#:
#: Some operations are only meaningful where the user is: an OAuth grant opens
#: a browser tab and writes a credential into this machine's ``auth.db``, so
#: running one for a phone would pop a tab nobody is looking at and store a
#: grant the phone's owner cannot use.
#:
#: This cannot be inferred from inside the runtime, which is exactly the bug
#: that produced it: ``/mcp reauth`` refused every routed invocation on the
#: theory that a routed command comes from elsewhere, when the control socket
#: binds ``127.0.0.1`` only and every client is therefore local. So locality is
#: DECLARED by the client in its auth frame and defaults to ``local``: today
#: every dialer reaches the runtime over loopback and is local by construction.
#: A relay that forwards a remote device's commands is the case that must
#: declare ``remote``, and until one exists this stays a one-value union in
#: practice while giving that relay a seam that does not require re-deciding
#: the question at each call site.
ClientLocality = Literal["local", "remote"]

#: How many concurrent attach (viewer terminal) connections one runtime
#: accepts before evicting the least-recently-seen one. Connection close is
#: detected anyway (the reader loop drops the registry entry); the cap is
#: defense against leaked-but-open sockets — half-open TCP with no FIN —
#: which liveness detection cannot see.
ATTACH_MAX_CLIENTS = 4

#: ``SlashResult`` ``data.type`` values that carry an ACTION for the invoking
#: terminal: the owner attached a team or an agent profile, and the invoker is
#: expected to submit ``data["request"]`` as a user turn of its own.
#:
#: This exists because that expectation used to be implicit, and an older
#: viewer that did not hold it dropped the request in total silence: the
#: runtime attached the team, returned "sending to <team>. <manager> is
#: coordinating.", and the pre-#624 renderer printed the line and had no
#: consumer for ``data.request``. No user row, no turn, no error.
#:
#: So a client that renders these DECLARES them in its auth frame's
#: ``slash_consumers``; an undeclared (older) client means the RUNTIME admits
#: the request itself. Absent-means-old, exactly like ``ClientKind`` above.
#: The list is the single source of truth for both sides of that seam, and
#: ``tests/unit/tui/test_noop_consumers.py`` fails CI if a producer emits a
#: ``request``-carrying receipt whose type is missing here.
# Goal submissions follow the same ownership rule: old/mobile clients leave
# admission to the owner; current terminals consume the receipt exactly once.
SLASH_ACTION_RECEIPTS: tuple[str, ...] = ("team_attached", "agent_attached", "goal_set")


def runtime_must_complete(receipt_type: Any, consumers: Any) -> bool:
    """Whether the OWNER must submit this receipt's request itself.

    The one rule, in one place: a client completes a receipt only if it
    DECLARED that type, so ``type not in declared`` — never ``declared is
    None``. Both ``None`` (a client built before the field) and ``[]``
    (declared, consumes nothing) mean undeclared and therefore admit here.

    Extracted because the predicate has two hosts by contract — a session is
    owned either by a detached runtime or by a TUI, and both must answer
    identically. Hand-duplicating it left the two copies free to drift, where
    a drift toward ``declared is None`` silently double-submits on one host
    only (review round 1, NIT-2). Living beside ``SLASH_ACTION_RECEIPTS``
    keeps the rule next to the list it is applied to; this module is
    import-light by contract and this adds no imports.
    """
    if receipt_type not in SLASH_ACTION_RECEIPTS:
        return False
    return receipt_type not in (consumers or ())


# ---------------------------------------------------------------------------
# Discovery record
# ---------------------------------------------------------------------------

#: Directory (under the config root) holding one record per live session.
#:
#: The name is now misleading and is kept anyway, deliberately. It is not a
#: mobile-only directory — every live session publishes here, and `lop
#: sessions`, `lop send`, attach and the daemon all read it. Renaming it would
#: buy nothing but a tidier string and would cost a mixed-version split-brain:
#: during any upgrade window an old binary writes and scans ``run/mobile``
#: while a new one uses the new name, so neither can see the other's sessions
#: — `lop sessions` goes half-blind, `lop send` cannot resolve a peer, and the
#: daemon drops live sessions off the phone. There is no migration that avoids
#: it, because the two binaries genuinely coexist in running processes. The
#: literal is a wire constant; treat it as one.
RUN_DIRNAME = "run/mobile"

#: How often a runtime rewrites its record's ``heartbeat_at``. The daemon
#: treats a record as wedged (not merely quiet) after ``HEARTBEAT_TIMEOUT_S``.
HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_TIMEOUT_S = 45.0

#: Subagent roster statuses that count as a RUNNING trajectory — one agent loop
#: that can independently make model calls right now.
#:
#: Lives here, beside the record fields it defines the meaning of, because two
#: modules must agree on it and a divergence is invisible: the runtime publishes
#: ``subagents_running`` from this set (``serving.ServingSessionHandle``) and
#: ``/info`` tallies this session's own tree from it (``info.collect``). If they
#: drifted, the fleet total and the tree drawn directly beneath it on the same
#: card would disagree, which is the one error that section must never make.
#:
#: ``queued`` is excluded and counted separately: a delegated child waiting for
#: a capacity slot is not spending anything. This module stays stdlib-only, so
#: the constant costs nothing to import on the CLI startup path.
RUNNING_SUBAGENT_STATUSES = frozenset({"running", "starting", "pausing"})


@dataclass
class SessionRecord:
    """The discovery record one ``lop`` process publishes for one session.

    Lives at ``~/.local-operator/run/mobile/<pid>.json`` — keyed by pid
    because a process hosts exactly one interactive session at a time, so the
    pid is the natural uniqueness token and ``kill -9`` leaves exactly one
    stale file to reap.

    ``control_key`` is the whole authorization story of the control socket:
    the record is mode 0600 under a 0700 directory, so anything that can read
    the key is already the owning account. The daemon never transmits it
    further — the phone never learns it.
    """

    pid: int
    kind: Literal["tui", "exec", "daemon"]
    session_id: str
    conversation_name: str
    cwd: str
    model_label: str
    control_port: int
    control_key: str
    protocol: int = PROTOCOL_VERSION
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)
    capabilities: list[str] = field(default_factory=list)

    # -- live state ---------------------------------------------------------
    # Purely ADDITIVE, and PROTOCOL_VERSION deliberately does NOT move for
    # them. Nothing is required to read these: an older reader drops unknown
    # keys in ``from_json`` and behaves exactly as it did, and a newer reader
    # sees the dataclass defaults for a record an older runtime wrote. Bumping
    # the protocol would instead make every older peer refuse a record it can
    # in fact use — the compatibility cost of a field nobody has to read is
    # zero, and the version is the one thing that would make it non-zero.

    #: A turn is running right now. The picker's liveness marker, and the
    #: difference between a session that is working and one merely resident.
    busy: bool = False
    #: This session has run at least one REAL turn (a user prompt, a wake
    #: delivery, a resume catch-up — anything through ``_run_turn_pipeline``).
    #: ``False`` marks the window after ``/new`` when the record is already
    #: published but the owner is still composing their first prompt, so a
    #: peer broadcast or an exact-address wake/steer must not drive a turn
    #: into it. One-way per session identity: once a real turn has run it
    #: stays True. Two paths other than a turn set it — a TUI ``/new``
    #: rebind re-seeds it for the NEW identity (see
    #: ``RuntimeServer.reset_record_started``), and a boot that RESUMED a
    #: history-bearing conversation seeds True at record construction (see
    #: ``RuntimeServer.__init__``) so the idle session is peer-visible
    #: before any turn runs in the new process. Every heartbeat/republish
    #: carries it forward so it is never reset by a later write.
    #: Deserialization overrides the default for pre-field records —
    #: see ``from_json``.
    started: bool = False
    #: No front end is attached. A working session with nobody watching is
    #: exactly what this release makes possible, so it is worth naming.
    detached: bool = False
    #: This session is WAITING FOR A PERSON: ``"approval"``, ``"ask"``, or
    #: None. A parked gate holds the runtime resident for up to a day, so the
    #: cost has to be findable — this field is what puts it in `lop sessions`
    #: and sorts it first in the picker.
    pending: str | None = None

    # -- build stamp --------------------------------------------------------
    # Same additive contract as the live-state block above, and for the same
    # reason: PROTOCOL_VERSION deliberately does not move for a field nobody
    # is required to read.
    #
    # The record IS the version channel between a viewer and a runtime. An
    # attach client reads it before dialing (``find_runtime_record``) and holds
    # it at bind, so one comparison there is complete — a runtime's build
    # cannot change while the process lives.

    #: What build this runtime is running (``update.installed_version()``).
    #: ``""`` means a runtime older than this field, which by construction is
    #: older than any terminal that can read it. A viewer compares this with
    #: its own build to NAME skew rather than fail silently under it.
    version: str = ""
    #: The git ref of that install when ``lop-update`` recorded one; ``""``
    #: for PyPI/pipx/editable installs. Needed because same-version rebuilds
    #: are this host's common drift — see ``update.BuildStamp``.
    source_ref: str = ""

    # -- agent trajectories -------------------------------------------------
    # Same additive contract as the live-state and build-stamp blocks above,
    # and for the same reason: PROTOCOL_VERSION deliberately does NOT move.
    # It gates SOCKET FRAME compatibility and is read as a pre-dial CAPABILITY
    # ASSERTION by peers already running — ``attach_client`` refuses below 2,
    # ``session_factory`` and the TUI's takeover path below 4, ``remote``'s
    # canonical attach below 5. Those readers take a HIGHER number as a promise
    # that every frame through v5 is understood. Two JSON integers that touch
    # no frame do not make that promise different, so bumping would spend the
    # one number that carries it on a field nobody has to read, and would leave
    # nothing to distinguish a build that genuinely changed the frames. A
    # future reader that must REQUIRE these fields negotiates through
    # ``capabilities``, which is the seam for exactly that (see
    # ``FRONTEND_CAPABILITY``, gated alongside the version rather than by it).
    #
    # ``None`` (not 0) is the "this build does not report" signal, and the
    # distinction is load-bearing: a runtime predating these fields has not
    # told us it has no subagents, and a reader that collapsed the two would
    # publish a confident total that is silently missing terms. See
    # ``SessionsInfo.subagents_unreported``.

    #: Subagent trajectories spending tokens right now: roster entries whose
    #: status is running/starting/pausing. ``SubagentComms.nodes()`` returns
    #: the COMPLETE roster including nested descendants (nested launches land
    #: in the root session's single records map tagged with their true
    #: parent), so this is a flat count over one read and must never be summed
    #: with a recursive child walk.
    subagents_running: int | None = None
    #: Delegated but still waiting for a capacity slot. Kept separate from
    #: ``subagents_running`` because a queued child is not spending anything,
    #: and folding it in would inflate "what is running right now".
    subagents_queued: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "SessionRecord":
        # Tolerate unknown keys (a NEWER binary's record read by an older
        # daemon mid-upgrade): forward-compat here is what lets a restart
        # rolling-upgrade the daemon without the phone losing sessions.
        #
        # ``started`` is the one field whose ABSENT key carries meaning: this
        # binary always serializes it (``to_json`` is ``asdict``), so a record
        # without the key was written by a PRE-field binary — and a pre-field
        # session had no composer gate at all, so to it "started" can only
        # mean True. Reading absent-as-True restores old-peer behaviour
        # exactly: broadcasts still reach a working old runtime (it keeps
        # heartbeating its key-less record until it restarts, which for
        # daemon/cmux sessions is days, not an upgrade window), and an exact
        # send dials it rather than spooling into an inbox only a boot-time
        # drain ever reads. The honest cost is the mirror image: an OLD
        # binary's fresh ``/new`` composer stays wakeable — but that is
        # precisely the old binary's own behaviour, unfixable from here until
        # it restarts, so True is the only default that does not penalise the
        # working old sessions for a bug they do not have. The dataclass
        # default stays ``False`` because a CONSTRUCTED record is a fresh
        # this-binary session (the composer window the field exists for);
        # only a deserialized key-less record is assumed pre-field.
        known = {f for f in SessionRecord.__dataclass_fields__}
        fields = {k: v for k, v in data.items() if k in known}
        if "started" not in data:
            fields["started"] = True
        return SessionRecord(**fields)
