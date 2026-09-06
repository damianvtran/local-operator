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
SLASH_ACTION_RECEIPTS: tuple[str, ...] = ("team_attached", "agent_attached")


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
    # attach client reads it before dialing (``find_owner_record``) and holds
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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "SessionRecord":
        # Tolerate unknown keys (a NEWER binary's record read by an older
        # daemon mid-upgrade): forward-compat here is what lets a restart
        # rolling-upgrade the daemon without the phone losing sessions.
        known = {f for f in SessionRecord.__dataclass_fields__}
        return SessionRecord(**{k: v for k, v in data.items() if k in known})
