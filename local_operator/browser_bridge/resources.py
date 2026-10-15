"""Private session browser obligations, never a PID/age-based orphan registry.

The transcript's execution lease owns writes. A process disappearing proves
nothing about task completion: only finish() records terminal intent. Capabilities
stay in a mode-0600 sidecar, and public inventories deliberately omit them.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.backend import BridgeClient, BridgeError
from local_operator.browser_bridge.protocol import (
    OWNERSHIP_MIN_EXTENSION_VERSION,
    ErrorCode,
    extension_older,
)

RESOURCE_NAME = ".browser-resource.json"

#: The record's spelling of each ownership host. ``HOST_BRIDGE`` names the
#: EXTENSION host — the same word its surface tokens use (``bridge:<tab>:<nonce>``),
#: because the record and the prefix have to agree or a resumed session would
#: select the wrong lane. The model-facing COPY calls that host "the extension"
#: (see ``browser_bridge.backend.HOST_EXTENSION``), which is a naming difference
#: between the wire and the prose, not a second host.
HOST_BRIDGE = "bridge"
HOST_UI = "ui"

#: Longest crash-replay chain kept per lane. The chain exists so a host whose
#: scope is stamped with an OLDER generation can still be reconciled
#: (`previous_generations`). It is trimmed to the NEWEST entries only because a
#: lane that never acknowledges (an extension that stays unpaired across many
#: resumes) would otherwise grow the durable record without bound. Overflow fails
#: CLOSED: a lane unreachable for more resumes than this is refused as stale
#: through the existing "resume the scope" copy, never silently accepted.
MAX_REPLAY_GENERATIONS = 16

#: The lanes that can hold a scope for a session; each acknowledges on its own.
_REPLAY_LANES = (HOST_UI, HOST_BRIDGE)


def surface_host(surface_id: str) -> str:
    """The host named by a surface token's prefix, or "" when it names none.

    One helper rather than a prefix test per caller: the record, the tool's
    ownership gate and the lane all have to agree about which host a handle
    belongs to, and three copies of `startswith("ui:")` is three chances to
    disagree with the one prefix the tokens actually carry.
    """
    for name in (HOST_UI, HOST_BRIDGE):
        if surface_id.startswith(f"{name}:"):
            return name
    return ""


class OwnershipDiscovery(Protocol):
    """The one discovery question the ownership layer asks, per host.

    It is NOT a state read: what the ownership mode needs is the peer's
    (version, protocol) pair, and the two hosts answer it from different fields
    — the extension from ``extension_version``/``extension_proto`` plus a proven
    connection, the UI host from its ``app_version`` and ``proto`` (it has no
    second process to be disconnected from). ``None`` means "cannot tell", and
    every caller must read it that way.
    """

    def peer_identity(self) -> tuple[str, int] | tuple[str, int, str] | None: ...


@dataclass(frozen=True)
class OwnershipHost:
    """One host's ownership lane: its transport, plus its discovery record.

    Selected by the record's ``host`` field so a session resumed on a UI surface
    does not send ``owner_*`` to a daemon it was never talking to.
    """

    name: str
    #: A FACTORY, not a client: constructing one up front would open no socket
    #: but would bind the module-level class at import time, which is exactly
    #: what keeps a monkeypatched `BridgeClient` (how the whole browser test
    #: suite fakes the wire) from being used.
    client: Callable[[], Any]
    discovery: OwnershipDiscovery


class _BridgeDiscovery:
    """The extension's peer facts, from the daemon's own published link state."""

    def peer_identity(self) -> tuple[str, int] | tuple[str, int, str] | None:
        try:
            current = state_store.read()
        except Exception:  # noqa: BLE001 - discovery must never raise at a call site
            return None
        if current is None or not current.extension_connected:
            return None
        return (current.extension_version, current.extension_proto)


class _UiDiscovery:
    """The UI host's peer facts: its app version and its bridge protocol.

    No connection flag is required — the file only exists while the app is
    running, and its views are its own children — so any readable record is a
    real peer. That is what gives ``ownership_mode()`` a meaningful answer on
    this host instead of ``None``, which elsewhere means "must ask".
    """

    def peer_identity(self) -> tuple[str, int] | tuple[str, int, str] | None:
        try:
            from local_operator.ui_browser import state as ui_state

            current = ui_state.read()
        except Exception:  # noqa: BLE001 - discovery must never raise at a call site
            return None
        if current is None:
            return None
        # Only invalidates a cached protocol/ownership verdict. Actual rebind still
        # requires the session's durable proof; a UI restart cannot adopt a tab
        # from its URL, number or discovery generation.
        if current.instance_id is not None:
            return (current.app_version, current.proto, current.instance_id)
        return (current.app_version, current.proto)


def _bridge_client() -> Any:
    # Resolved through the module global at CALL time, not captured: the browser
    # test suite monkeypatches ``resources.BridgeClient`` to fake the wire.
    return BridgeClient()


def _ui_client() -> Any:
    from local_operator.ui_browser.backend import UiHostClient

    return UiHostClient()


def ownership_host(name: str) -> OwnershipHost:
    """The lane for one host name; the bridge is the fallback for an empty one.

    Defaulting to the bridge is the fail-safe the record's own compatibility rule
    needs: a record written before the ``host`` field existed names no host, and
    every such record belongs to a session that was talking to the daemon.
    """
    if name == HOST_UI:
        return OwnershipHost(HOST_UI, _ui_client, _UiDiscovery())
    return OwnershipHost(HOST_BRIDGE, _bridge_client, _BridgeDiscovery())


class BrowserOwnershipError(RuntimeError):
    """An expected ownership/compatibility refusal, safe to show without a stack."""


#: The refusal a peer that cannot RECONCILE ownership gets, in the two cases
#: where that is the honest answer: a genuine protocol incompatibility, and a
#: pre-ownership extension that is nonetheless being asked to reconcile a durable
#: obligation it has no verbs for. Verbatim what this path said before the change,
#: because in both cases updating really is the remedy.
_OWNERSHIP_REQUIRES_UPDATE_MESSAGE = (
    "Browser ownership recovery requires an updated Local Operator extension. "
    "Update the extension, reconnect it, then retry; no new tab was allocated."
)

#: The refusal a CURRENT extension gets when `owner_recover` answers with the
#: bare `internal` shape. That shape has two producers and the version tells them
#: apart: below the ownership floor it is a pre-ownership release, and above it
#: the worker has stopped answering. Telling the second one to update was the
#: defect — the live 0.1.10 ships `owner_*` (as does the 0.1.9 tree) and the
#: store had nothing newer to offer — so the remedy named here is the one that
#: actually clears a wedge.
_EXTENSION_STOPPED_ANSWERING_MESSAGE = (
    "the browser extension stopped answering while this session's tab ownership was "
    "being recovered. Ask the user to toggle the Local Operator extension OFF then ON "
    "in chrome://extensions (pairing is preserved), then retry."
)

#: The same two refusals, per host. They cannot be one sentence with a noun
#: swapped in: both name a PROCESS to go and fix, and naming the extension to the
#: user of the desktop app (or the reverse) sends them somewhere that cannot help.
_UI_REQUIRES_UPDATE_MESSAGE = (
    "Browser tab ownership recovery requires an updated Local Operator desktop app. "
    "Update the app, then retry; no new tab was allocated."
)
_UI_STOPPED_ANSWERING_MESSAGE = (
    "the Local Operator desktop app's browser host stopped answering while this "
    "session's tab ownership was being recovered. Ask the user to restart the desktop "
    "app (tab handles and pending site decisions are lost with it), then retry."
)


def _ownership_requires_update_message(host: str) -> str:
    if host == HOST_UI:
        return _UI_REQUIRES_UPDATE_MESSAGE
    return _OWNERSHIP_REQUIRES_UPDATE_MESSAGE


def _peer_stopped_answering_message(host: str) -> str:
    if host == HOST_UI:
        return _UI_STOPPED_ANSWERING_MESSAGE
    return _EXTENSION_STOPPED_ANSWERING_MESSAGE


def _host_noun(host: str) -> str:
    """The subject of a refusal that names who must reconcile: a PROCESS name.

    Kept separate from the two whole-sentence builders above because this one is
    spliced into a sentence that is otherwise host-neutral.
    """
    return "browser extension" if host != HOST_UI else "the Local Operator desktop app"


@dataclass(frozen=True)
class BrowserCleanupResult:
    state: str
    detail: str = ""


class BrowserResource:
    def __init__(self, directory: Path, session_id: str, *, host: str = "") -> None:
        self.directory = directory
        self.session_id = session_id
        self.path = directory / RESOURCE_NAME
        self.lock = asyncio.Lock()
        self.record: dict[str, Any] = {}
        self.generation = ""
        self.previous_generation = ""
        #: Lanes whose peer has been reconciled in THIS run (see `recovered`).
        #: Reconciliation is per HOST — the app and the extension each keep their
        #: own scope and fence — so one boolean for the whole resource let the
        #: app's acknowledgement stand in for the extension's. Initialised
        #: directly (not through the `recovered` setter) because the setter
        #: resolves the current lane, which needs `host`, assigned below.
        self._recovered_lanes: set[str] = set()
        #: WHICH host owns this session's surface: "" (not learned yet), "ui" or
        #: "bridge". Written into the record on the first successful `open`, so a
        #: resumed session selects the lane it opened on. Everything the lane
        #: touches — the `owner_*` transport and the discovery read that feeds
        #: `ownership_mode` — is selected from it.
        #:
        #: It is the lane's fallback, not its authority: while the record HOLDS a
        #: capability, its prefix outranks this flag (`_obligation_host`), so a
        #: lane left naming the host of a tab that has already closed cannot
        #: address the next allocation's cleanup to it. See `_lane`.
        #:
        #: "" deliberately means the BRIDGE: a record written before this field
        #: existed belongs to a session that was talking to the daemon, so an old
        #: record behaves exactly as it did before (fail-safe, not fail-open).
        self.host = host
        # Whether the attached extension can reconcile ownership at all.
        #
        #   * None  = not learned yet (or the peer changed since it was).
        #   * True  = `owner_recover` answered; the `owner_*` lifecycle is real
        #     on this link and every obligation must go through it.
        #   * False = a PROVEN pre-ownership peer; the obligation verbs do not
        #     exist, so the record is reconciled capability-only (see
        #     `_degraded_recover`).
        #
        # Sticky per peer but REFRESHABLE: `_ownership_mode` invalidates it when
        # the daemon reports a different (version, proto), so a user who updates
        # the extension mid-session gets ownership back without restarting.
        self.ownership: bool | None = None
        # The peer identity `self.ownership` was learned against, AND the lane
        # it was learned on. See `_peer_identity` for why the pair, not just the
        # version. The lane name is part of the cache key: two hosts publish
        # DIFFERENT identity shapes (the extension an extension version, the app
        # an app version), so comparing a UI identity to a peer learned on the
        # bridge lane is a category error that reads as an incarnation change and
        # forces a spurious reconcile. `refresh_identity` re-scopes both on a
        # lane move without treating it as a peer change.
        self._ownership_peer: tuple[str, int] | tuple[str, int, str] | None = None
        self._ownership_peer_lane: str = ""
        self._lease_at_creation = self._lease_generation()
        # Whether the execution lease DEFINES this owner's identity. Only then
        # is a lease change a resume that must fence us; an owner whose identity
        # came from the record is authorized by the record plus, for the CLI
        # path, the lease it holds for exclusivity rather than for naming.
        self._identity_from_lease = False

    @property
    def execution_generation(self) -> str:
        """A pure identity lookup; stale owners are rejected INSIDE finalization.

        Never mints: an identity read must not become a write, or merely asking
        who owns the tab would rotate the answer.
        """
        if self.generation:
            return self.generation
        if self._lease_at_creation:
            return self._lease_at_creation
        try:
            return str(self._load().get("generation", ""))
        except (BrowserOwnershipError, OSError, ValueError):
            return ""

    def _obligation_host(self) -> str:
        """The host that owns the capability this record is HOLDING, or "".

        This is the source of truth for lane selection, and the reason is the
        failure it removes: a session that closed its bridge tab and then opened
        a fresh one on the app recorded `host: "ui"` beside a `ui:` handle, but
        the in-memory lane flag still said `bridge` from the closed tab, so
        `finish` sent `owner_finish` to the daemon, the app's allocation stayed
        live, and the record lost the handle it was claiming to clean up (QA
        Q-5 / review R6-R7).

        The HANDLE decides, in the order the record itself treats as current:
        the surface the session is actually holding, then the evidence slot it
        could not prove. `unresolved_surface_id` is deliberately second rather
        than ignored: it is still a claim on a host that has to hear about it.
        """
        for field in ("surface_id", "unresolved_surface_id"):
            host = surface_host(str(self.record.get(field, "")))
            if host:
                return host
        return ""

    @property
    def recovered(self) -> bool:
        """Whether the CURRENT lane's peer has been reconciled in this run.

        Why this is per lane and not one flag: an execution-lease change (a resume
        of the same durable session) leaves every host holding a scope stamped
        with the OLD generation. Each host must be told, once, via `owner_recover`
        carrying `previous_generation(s)` — the only verb that advances a scope.
        The extension fences every other verb with strict generation equality, so
        a plain `open` against an unreconciled extension is refused as stale. With
        a single boolean, reconciling the app first marked the extension
        reconciled too; a later explicit `backend=extension` open then skipped its
        recovery and was refused on a perfectly legitimate resume.

        Kept as a property so every existing reader and writer (`recover`,
        `_degraded_recover`, `refresh_identity`, the tool's allocation reset)
        keeps its spelling and now acts on the lane it is talking to.
        """
        return self._lane().name in self._recovered_lanes

    @recovered.setter
    def recovered(self, value: bool) -> None:
        lane = self._lane().name
        if value:
            self._recovered_lanes.add(lane)
        else:
            self._recovered_lanes.discard(lane)

    def _retarget_lane(self, host: str) -> None:
        """Move the lane to `host` and drop every verdict cached for the old one.

        The cached `ownership` verdict and the peer identity it was learned
        against belong to a PEER, not to the session, so carrying them across a
        host change would let a link with no ownership lifecycle decide the
        verdict for a different transport with one. Both are cleared here and
        re-learned lazily through `ownership_mode()`.
        """
        if host and host != self.host:
            self.ownership = None
            self._ownership_peer = None
            self._ownership_peer_lane = ""
            self.host = host

    def _lane(self) -> OwnershipHost:
        """The ownership lane for this session: transport + discovery, per host.

        THE HELD CAPABILITY COMES FIRST, then the learned lane flag, then the
        record's `host` field. The order is the whole fix for a session whose
        allocation moved between hosts: the tab the record holds is the only
        thing that names the host which must receive its cleanup, and every
        operation that reaches a host — `params()`, `client()`, recover, finish,
        teardown — resolves its transport through here, so one rule covers them
        all instead of each call site picking a host for itself.
        """
        return ownership_host(
            self._obligation_host() or self.host or str(self.record.get("host", ""))
        )

    def select_host(self, host: str) -> None:
        """Bind this session's surface to a host, once it is known.

        Called by the tool before the lane runs, from the pinned surface prefix
        when there is one and from the availability probe when there is not.

        A HELD CAPABILITY PINS THE LANE; nothing else does. The earlier rule —
        "an already-known host wins" — kept a lane on a host whose tab was
        already closed, so a fresh allocation on the other host was recorded but
        never used as the lane (QA Q-5). Nothing is in flight in that state, so
        there is no transport to keep stable and a probe answering with the host
        that actually serves the session is right.

        While a capability IS held, the probe may still AGREE with it — a
        resumed session's pinned prefix is read from the handle itself, so the
        two are the same answer and recording it keeps `host` usable in failure
        copy — but it may not CONTRADICT it: that surface's transport is pinned
        for the life of the surface, and a probe answering differently must not
        move an obligation onto a host that never held the tab.
        """
        if not host:
            return
        held = self._obligation_host()
        if held and held != host:
            return
        self.host = host

    def pinned_host(self) -> str:
        """The host this session's durable state pins it to, or "" for none.

        The RECORD is the durable form of the pin `select_host` documents, and
        on a RESUMED session it is the only one there is: the surface prefix in
        `state.surface_id` is empty until the lane adopts it from this same
        record, which happens well after the gate has to decide. Reading only
        the availability probes there does not merely guess — a probe-selected
        host lands in `self.host`, which `_lane()` reads ahead of
        `self.record["host"]` and behind only a HELD handle — so it overrides
        the record's `host` FIELD and silently moves a resumed session onto
        whichever host happened to be up. What it cannot override is the record's
        HANDLE: that prefix decides every operation that reaches a host,
        including this selection, so the tab's own host still wins.

        Two spellings of one fact, and the HANDLE is read first:

        * `surface_id` names its own host in its prefix, and it is the surface
          the session is actually HOLDING, so it outranks the field beside it.
          The writer no longer produces the disagreement (`remember` retargets the
          lane onto the host that allocated the handle), but the READER still has
          to resolve it: a record left by an earlier client, or by a lane move
          that never completed, still carries `host: "bridge"` beside a `ui:`
          handle. Reading the field first sends
          `owner_recover` to the daemon for a tab that lives in the app, the
          daemon answers `unresolved` with no tab, `recover()` moves the handle
          into `unresolved_surface_id`, and the app's live tab is left
          stranded — where the handle keeps the lane on the host that owns it.
        * `record["host"]` is the fallback for a record with NO handle, and
          only while that record owes a reconciliation (`OBLIGATION_FIELDS`).
          That is the state `recover()` leaves behind when it cannot prove a
          handle, and the session still owes `owner_*` an answer there: the
          field is the only thing naming the host that holds it.
        * A record with NEITHER an obligation nor a handle pins NOTHING. A
          `close` clears `surface_id` and deliberately leaves `host` behind, so
          a settled session goes on naming the host it used last; that is not a
          transport to keep stable, because no surface is in flight, and it
          must not govern the next `open` — pinning there is what makes a
          resumed session whose app is down refuse to open at all instead of
          using the host that answers. A record written before the `host` field
          existed is still served by the first rule: its handle, defaulted
          through `ownership_host("")`, names the bridge, which is where that
          session was talking.

        Read from `self.record` when the lane has run and from the file
        otherwise, because the tool's gate asks this BEFORE `initialize()` —
        the only thing that populates `record`. An absent file, a foreign one
        (`_load` raises) and a malformed one all answer "": the typed refusal
        for a record that cannot be read belongs to `initialize()`, which
        renders it as the actionable failure it is, not to a selection helper
        that would otherwise raise it from inside the gate.
        """
        record = self.record
        if not record:
            try:
                record = self._load()
            except (BrowserOwnershipError, OSError, ValueError):
                return ""
        surface = str(record.get("surface_id", ""))
        host = surface_host(surface)
        if host:
            return host
        # A denied fresh navigation can owe an approval without a minted tab.
        # Its saved transport pins only that pending flow; it never grants tab
        # ownership, origin access, or permission to adopt a foreign capability.
        pending_host = str(record.get("pending_access_host", ""))
        if pending_host in (HOST_UI, HOST_BRIDGE):
            return pending_host
        return self._obligation_only_host(record)

    @classmethod
    def _obligation_only_host(cls, record: Mapping[str, Any]) -> str:
        """The `host` a record names ONLY because it still owes that host a cleanup.

        Neither a held handle nor a pending approval: the record holds no tab on
        that host and no approval is waiting on it. What remains is the debt itself
        (a `terminal` intent left by `finish`, a retention, `resumed_scope`,
        `release_pause`, unevidenced `unresolved_surface_id`), which only that
        host's `owner_recover` can discharge. It pins the lane so the cleanup
        reaches its owner, but it must not also decide where a LATER fresh `open`
        lands: the tool uses this to reconcile the old host first and then
        re-decide the fresh open from the caller's explicit `backend`.
        """
        host = str(record.get("host", ""))
        if host and cls._record_owes_reconciliation(record):
            return host
        return ""

    def pending_access_pin(self) -> str:
        """The host a no-tab site approval is still waiting on, or "".

        Routing metadata only (see `pinned_host`): it never grants a tab, an origin or
        permission to adopt anything. Exposed so the dispatcher can refuse an explicit
        `backend` that contradicts it, instead of opening on the other host while
        `request_access`/`await_access`/`cancel_access` keep going to this one.
        Reads the loaded record, or the file when the lane has not run yet.
        """
        record = self.record
        if not record:
            try:
                record = self._load()
            except (BrowserOwnershipError, OSError, ValueError):
                return ""
        pending = str(record.get("pending_access_host", ""))
        return pending if pending in (HOST_UI, HOST_BRIDGE) else ""

    def obligation_pin_host(self) -> str:
        """The host pinning this session by an obligation alone, or "".

        "" whenever a handle or a pending approval names the host instead, so those
        pins keep their existing precedence. Reads the loaded record, or the file
        when the lane has not run yet (the tool asks before `initialize()`).
        """
        record = self.record
        if not record:
            try:
                record = self._load()
            except (BrowserOwnershipError, OSError, ValueError):
                return ""
        if surface_host(str(record.get("surface_id", ""))):
            return ""
        if str(record.get("pending_access_host", "")) in (HOST_UI, HOST_BRIDGE):
            return ""
        return self._obligation_only_host(record)

    async def move_lane_after_obligation(self, host: str) -> bool:
        """Re-decide a FRESH open's lane once the host that was owed has been reconciled.

        The tool keeps the lane on an obligation-only pin for the first recovery, so
        the debt (a finished scope's `terminal`, a retention, `release_pause`...)
        reaches the host that owns it; dropping it, or letting an explicit `backend`
        win straight away, would strand that host's cleanup. This runs after that
        recovery, under the caller's lock, and answers whether the open may go on:

        * a LIVE handle (`surface_id`) still names a host -> True without moving: the
          dispatcher owns that refusal (a hint cannot contradict a held tab), as before;
        * a debt is STILL owed -> False: the old host did not discharge it, and
          sending the destination an `open` anyway would dispatch to a host that
          is mismatched with the session's own record;
        * otherwise the lane moves to `host` and that peer is reconciled exactly
          once (`ensure_recovered` is per lane) BEFORE anything allocates there.

        Only `surface_id` counts as "held" here, NOT `_obligation_host()`: that one
        also answers for `unresolved_surface_id`, which is private EVIDENCE the old
        host could not prove after an authority loss, never a capability this session
        holds. The dispatcher only ever sees the live handle, so treating evidence as
        held made this return True and delegate a refusal the dispatcher cannot make:
        the explicit open then reached an unreconciled destination (review R1 / QA Q1,
        written entirely by product paths). The evidence is preserved, not cleared:
        nothing but the old host's own proof can discharge it, so a hint that needs
        the other host keeps refusing until `backend` is dropped.
        """
        if surface_host(str(self.record.get("surface_id", ""))):
            return True
        if self.obligation_pin_host():
            return False
        self.select_host(host)
        await self.ensure_recovered()
        return True

    def client(self) -> Any:
        """A client for this session's host, built fresh per call (as before).

        Public because the ownership lane lives in the tool (`builtin.py`) and
        must drive the SAME transport the resource's own finalizer does: two
        selections would be two answers to "which host owns this tab".
        """
        return self._lane().client()

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        if not isinstance(value, dict) or value.get("session_id") != self.session_id:
            raise BrowserOwnershipError("browser resource ownership is unresolved")
        return value

    def _lease_generation(self) -> str:
        from local_operator.session_lease import _read_claim

        # One parser for the claim, shared with the lease and the resume path:
        # the claim now carries the writer's birth fields as well as its
        # generation, and a second reader beside that parser is how two callers
        # come to disagree about who owns a transcript. This one wants only the
        # generation — a browser resource is fenced on it, never on the pid.
        return _read_claim(self.directory / ".execution-lease").generation or ""

    def initialize(self) -> None:
        if self.generation:
            self.assert_current()
            return
        self.record = self._load()
        self.previous_generation = str(self.record.get("generation", ""))
        if self._lease_at_creation and self._lease_generation() != self._lease_at_creation:
            raise BrowserOwnershipError("browser host execution lease changed; no action taken")
        # Identity is DURABLE PER SESSION, never per BrowserResource instance.
        #
        # Factory-built sessions inherit the exclusive execution lease's
        # generation, so a resume is a genuinely new execution and the old one
        # is correctly fenced. In-process children take neither path: they use
        # `claim_session`, not `acquire_session_lease`, so there is no lease to
        # read. Minting a per-instance token there made a SECOND Session over
        # the same directory revoke the FIRST one's authority over a tab it was
        # still holding — the incumbent's finalizer returned `unresolved` while
        # the newcomer inherited its surface_id, stranding a tab in the pool
        # that nobody could close. That is the exact leak this module exists to
        # remove, so the unleased case ADOPTS the stored identity instead: one
        # session id means one owner, whichever object is asking.
        self.generation = (
            self._lease_at_creation or self.previous_generation or secrets.token_urlsafe(24)
        )
        self._identity_from_lease = bool(self._lease_at_creation)
        generations = self._extend_chain(
            self.record.get("bridge_generations", []), (self.previous_generation, self.generation)
        )
        # A lane that has acknowledged before keeps its OWN chain, which must hear
        # of this run's generations too; a lane that never has falls back to the
        # shared chain above (see `lane_generations`).
        lanes = {
            lane: self._extend_chain(chain, (self.previous_generation, self.generation))
            for lane, chain in dict(self.record.get("lane_generations", {})).items()
        }
        if lanes:
            self.record["lane_generations"] = lanes
        self.record.update(
            bridge_generations=generations,
            session_id=self.session_id,
            generation=self.generation,
            proof=self.record.get("proof") or secrets.token_urlsafe(32),
            allocation_id=self.record.get("allocation_id") or secrets.token_urlsafe(24),
            state=self.record.get("state", "closed"),
        )
        if self._is_new_execution_over_settled_scope():
            self.record.pop("terminal", None)
            # The extension keeps its OWN copy of the terminal intent and used
            # to clear it only when the generation string changed — which the
            # unleased path deliberately never does, so retiring the record
            # alone moved the refusal one layer down instead of removing it.
            # DURABLE, and cleared only by an acknowledged recover: a run that
            # dies between here and the bridge would otherwise leave the
            # extension holding a terminal no later run could ever retire.
            self.record["resumed_scope"] = True
            if self.record.get("retention") == "paused scope":
                self.record["release_pause"] = True
        self._save()

    def _is_new_execution_over_settled_scope(self) -> bool:
        """May this owner retire the terminal intent the record already carries?

        The question a resume has to answer is "is this a NEW RUN of the
        session", and the honest evidence for that is the record itself, never
        a pid and never a lease.

        Keying this on holding a lease was wrong in both directions, and the
        wrong one shipped: in-process children use ``claim_session``, not
        ``acquire_session_lease``, so for every subagent the branch was dead
        and ``terminal`` became permanent. ``allocate`` refuses on terminal, so
        a finalized child could never browse again on any later run — and
        ``hub op='resume'`` relaunches children on their own directory, which
        made that an ordinary flow, not a corner. Worse, a scope that ended
        holding a tab for a pending approval could no longer be finished, so
        the tab was stranded: the leak class this module exists to remove.

        Two facts make the record sufficient. ``terminal`` is written only by
        ``finish``, so its presence means the previous scope SETTLED and has no
        live owner left to fence. And ``initialize`` runs only on a
        freshly-constructed resource — the instance that called ``finish``
        returns early from it — so reaching here over a settled record IS a
        later execution. That keeps B1 intact: a live incumbent has no terminal
        recorded, so a concurrent second instance still cannot retire anything.

        A STRANDED scope (the close failed, so the tab is still out there) is
        retired too, and that is deliberate rather than an oversight: the
        resumed owner is alive again and is the party responsible for that tab,
        which is precisely the route out that the crash-shape refusal names
        — resume the session and let the owner close it. It costs the operator
        no evidence, because ``cleanup_exact`` reaches a record through
        ``adopt``, never through this method, and a row only stops being a
        cleanup candidate while a live owner is actually holding it.
        """
        return bool(self.record.get("terminal"))

    def adopt(self, generation: str) -> None:
        """Take on an EXISTING record's identity without minting a new one.

        Operator recovery acts on the row the operator selected, so it must not
        renumber it: the CLI holds the lease for exclusivity, and the lease's
        own freshly-minted generation is not this owner's name.
        """
        self.record = self._load()
        if self.record.get("generation") != generation:
            raise BrowserOwnershipError("browser owner generation is stale; no action taken")
        self.generation = generation
        self.previous_generation = generation
        self._identity_from_lease = False

    def assert_current(self) -> None:
        current = self._load()
        lease = self._lease_generation() if self._identity_from_lease else ""
        if (lease and lease != self.generation) or current.get("generation") != self.generation:
            raise BrowserOwnershipError("browser owner generation is stale; no action taken")

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".browser-resource-", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(self.record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def _extend_chain(chain: Any, candidates: tuple[str, ...]) -> list[str]:
        """`chain` plus any new `candidates`, newest-last, bounded to the newest."""
        extended = [str(item) for item in chain]
        for candidate in candidates:
            if candidate and candidate not in extended:
                extended.append(candidate)
        return extended[-MAX_REPLAY_GENERATIONS:]

    def lane_generations(self, lane: str | None = None) -> list[str]:
        """The crash-replay chain owed to ONE lane's peer.

        Reconciliation is per host, so the generations a host still needs are too:
        `owner_recover` is the only verb that consults them, and the strict fence
        accepts a stale scope only if it equals the current generation, the
        immediate predecessor or a member of this list. A lane that has
        acknowledged keeps only what it was told; a lane that never has (no entry,
        or a record written before per-lane chains existed) is owed the full
        shared chain. Without that split, one host's ACK collapsed the chain and
        the other host, still stamped with an older generation, was refused as
        stale on a legitimate resume.
        """
        name = lane or self._lane().name
        per_lane = self.record.get("lane_generations", {})
        chain = per_lane[name] if name in per_lane else self.record.get("bridge_generations", [])
        return list(chain)

    def _acknowledge_lane_chain(self) -> None:
        """Record that the CURRENT lane's peer is now stamped with this generation.

        Trims only this lane's view. The shared chain (what a never-acknowledged
        lane is owed) collapses only once EVERY lane has acknowledged, which is
        the point at which nothing older can still be held by any host.
        """
        per_lane = dict(self.record.get("lane_generations", {}))
        per_lane[self._lane().name] = [self.generation]
        self.record["lane_generations"] = per_lane
        if all(per_lane.get(lane) == [self.generation] for lane in _REPLAY_LANES):
            self.record["bridge_generations"] = [self.generation]

    def params(self) -> dict[str, Any]:
        self.initialize()
        return {
            "requester": f"session:{self.session_id}",
            "owner_proof": str(self.record["proof"]),
            "owner_generation": self.generation,
            "previous_generation": self.previous_generation,
            "previous_generations": self.lane_generations(),
            "allocation_id": str(self.record["allocation_id"]),
            # This owner is a NEW RUN over a scope that already settled. The
            # generation is reused (B1), so it is the only evidence the
            # extension has that a resume happened; without it the two sides
            # disagree about what a resume is and `open` is refused forever.
            "resumed_scope": bool(self.record.get("resumed_scope")),
        }

    #: Record fields whose presence means a DURABLE obligation exists that only
    #: the `owner_*` lifecycle can reconcile: a tab this record still holds, a
    #: capability it could not prove, a settled scope, retention, an
    #: unacknowledged resume, or a paused scope awaiting release.
    #:
    #: Deliberately FIELDS rather than "does it look allocated": `initialize`
    #: mints `allocation_id` unconditionally, so testing that would be dead code
    #: that always answered "yes, there is an obligation".
    OBLIGATION_FIELDS = (
        "surface_id",
        "unresolved_surface_id",
        "terminal",
        "retention",
        "resumed_scope",
        "release_pause",
    )

    @classmethod
    def _record_owes_reconciliation(cls, record: Mapping[str, Any]) -> bool:
        """Whether a RECORD — not necessarily this instance's — owes one.

        Takes the record as an argument because `pinned_host()` runs before
        `initialize()` has populated `self.record`: the gate asks it about the
        record it just read from the file, and the answer has to be the same one
        `has_durable_obligation()` gives once the lane has loaded that file.
        """
        return any(record.get(field) for field in cls.OBLIGATION_FIELDS)

    def has_durable_obligation(self) -> bool:
        """Whether the record carries something only `owner_*` can reconcile.

        The discriminator between the two bare-`internal` cases: with no
        obligation in flight, a pre-ownership peer can be served by the
        capability-only path, so the tool degrades instead of dying. With one,
        the session genuinely owes the extension a reconciliation it cannot
        perform, and the honest failure stays.
        """
        return self._record_owes_reconciliation(self.record)

    def _peer_identity(self) -> tuple[str, int] | tuple[str, int, str] | None:
        """The peer's (version, proto[, UI instance]), or None when unknown.

        Read from the DISCOVERY FILE rather than a socket: the host publishes
        these from the same live state `/health` serves, and this is consulted on
        the `finish`/`retain`/`release` paths too, which must stay cheap and must
        not depend on a dial. `None` — no host, nothing attached, or an
        unreadable file — means "cannot tell", and every caller must treat it
        that way: unknown is never "old".

        WHICH file depends on the lane: the extension answers from
        ``extension_version``/``extension_proto`` and requires a proven link, the
        UI host from its ``app_version`` and ``proto``. Reading the daemon's file
        for a UI surface returned ``None`` forever, which is the reason the UI
        host could never be classified and its capability-only path stayed
        unreachable.
        """
        return self._lane().discovery.peer_identity()

    def refresh_identity(self) -> None:
        """Invalidate the cached verdict when the peer identity actually CHANGED.

        Discovery is never proof, and an unknown discovery is not proof of a
        changed incarnation either. ``_peer_identity`` returns ``None`` for
        "cannot tell" (the record is gone or unreadable), so WITHIN ONE LANE a
        ``None`` result must NOT be compared against a learned peer: a host that
        flaps A -> None -> A would otherwise invalidate twice and force a needless
        re-recovery each time. Reset only when a RESOLVED peer differs from the
        learned one; a transient ``None`` leaves the learned verdict intact.

        That rule is per-lane and deliberately NOT absolute. Two other outcomes
        are intended and are the opposite of "no churn": this method still CLEARS
        ``recovered`` (forcing a reconcile) on a resolved same-lane peer change,
        which is exactly the CR1 defect's cure; and it drops the cached
        ``ownership`` verdict on a LANE move without touching any lane's
        reconciliation state (see the lane-scope branch below). "Does not churn
        on None" governs the same-lane unresolvable case only.

        The cache is LANE-SCOPED. ``_peer_identity`` reads the CURRENT lane's
        discovery while ``_ownership_peer`` may have been learned on the other
        lane (a `bridge` peer versus a `ui` peer are different shapes entirely),
        and `_retarget_lane` can move the lane without clearing the learned flag.
        Comparing across lanes would misread a lane move as an incarnation change
        and issue a spurious ``owner_recover`` before an ordinary finish. So when
        the resolved lane differs from the one the cache was learned on, the
        verdict is dropped and the peer is RE-SCOPED to the new lane WITHOUT
        being treated as a change (re-learn lazily). Only a change WITHIN the same
        lane resets that lane's ``recovered``.

        Whether a lane move owes a recovery is decided by the per-lane
        ``recovered`` state, not by this branch: a lane already reconciled (or
        that just served this run's successful allocation, see ``remember``) owes
        nothing, and a lane never reconciled in this run owes exactly one.
        """
        lane = self._lane().name
        if lane != self._ownership_peer_lane:
            self._ownership_peer_lane = lane
            self._ownership_peer = self._peer_identity()
            self.ownership = None
            return
        peer = self._peer_identity()
        if peer is not None and peer != self._ownership_peer:
            self._ownership_peer = peer
            self.ownership = None
            self.recovered = False

    async def ensure_recovered(self) -> None:
        """Refresh/recover under the caller's resource lock, without minting proof.

        The refresh runs FIRST, so an incarnation change (which clears
        ``recovered`` inside ``refresh_identity``) is what decides — never a
        stale flag consulted before the fingerprint (review R1 CR1). This is the
        ONE place the ordinary action and the finalizer both go through, so
        there is no second spelling of "do we owe an automatic recovery".
        """
        self.refresh_identity()
        if self.recovered:
            return
        await self.recover()

    def ownership_mode(self) -> bool | None:
        """The cached ownership verdict, invalidated when the PEER changed.

        Sticky per peer, because re-probing on every call would re-issue
        `owner_recover` — and, on a legacy link, re-pay a failed round-trip — on
        every command, including the plain `close` that settles the scope.
        Refreshable, because an extension update mid-session changes the
        reported (version, proto) pair, which invalidates the verdict and lets
        the next call re-probe: the user gets ownership back without restarting
        anything.

        Returns None when it has not been learned for this peer yet — callers
        must treat that as "must ask", never as "unavailable".
        """
        self.refresh_identity()
        return self.ownership

    def _peer_is_pre_ownership(self) -> bool:
        """Whether the peer is PROVABLY older than the ownership floor.

        False when the version is unknown or unparseable, and that direction is
        load-bearing: "cannot tell" must not be read as "old", or an
        unreadable discovery file would silently demote a current extension to
        the capability-only path and hide a real wedge.
        """
        peer = self._ownership_peer
        if peer is None:
            return False
        if self._lane().name == HOST_UI:
            # The ownership floor is an EXTENSION version and the UI host reports
            # an APP version, so comparing them is a category error — and in the
            # wrong direction: an app at 0.1.x would be read as a pre-ownership
            # extension. A UI host that answers the wire at all ships the whole
            # method set, so it can never be read as old.
            return False
        version = peer[0]
        return extension_older(version, OWNERSHIP_MIN_EXTENSION_VERSION)

    async def recover(self) -> dict[str, Any]:
        # Already learned that this peer has no ownership lifecycle: do NOT
        # re-issue the verb it cannot answer. `ownership_mode` refreshes the
        # verdict through the peer identity, so a mid-session extension update
        # leaves this branch and probes normally.
        if self.ownership_mode() is False:
            return self._degraded_recover()
        try:
            result = await self.client().call("owner_recover", self.params())
        except BridgeError as exc:
            # An ownership-AWARE extension refuses with the typed OWNER_REFUSED;
            # anything else from an `owner_*` method means the extension does
            # not implement ownership at all. Keying on the typed code rather
            # than on the old side's wording is what stops a reworded message
            # from silently degrading this into a raw internal error.
            #
            # The narrowing matters as much as the mapping: TWO kinds of
            # data-carrying INTERNAL now reach this branch and NEITHER may be
            # reported as a version mismatch, because both mean "the extension
            # is wedged" — the one thing the old message told the operator not
            # to look for.
            #
            #   * a DAEMON-side timeout, which carries `timeout_s`; and
            #   * an EXTENSION-side per-call deadline, which carries `stalled`
            #     (settle.ts's `deadline()`), i.e. exactly the wedged worker
            #     this PR exists for — `owner_recover` reaches it through
            #     `withOwnership` → `withSessionMutation` → `scopes()`, so the
            #     recovery command would otherwise tell the operator to update
            #     a perfectly current extension from inside the incident.
            #
            # Only a genuinely old extension returns `internal` with NEITHER
            # key — and even that is a guess rather than proof, which is why this
            # branch is worth reading carefully before adding to it. `worker.ts`'s
            # catch-all emits `internal` with an EMPTY `data` for any non-
            # BridgeCommandError thrown inside a handler, and `deadline()`
            # rethrows the underlying rejection verbatim, so a current extension
            # can produce this exact shape too (review R2-4 reproduced one: a
            # genuine `chrome.storage` rejection with the message "Access to
            # storage is not allowed from this context."). The two are
            # indistinguishable on the wire, so this branch is a best-effort
            # mapping of the LEGACY case and the message it renders is the least
            # wrong answer available, not a diagnosis.
            #
            # Keying on the absence of both discriminators is still right: it is
            # what keeps a future third producer of `data`-carrying INTERNAL from
            # silently re-creating the misdiagnosis, and a new `data` key belongs
            # in this predicate.
            #
            # THE BARE SHAPE HAS TWO PRODUCERS, and the peer's reported version
            # is what tells them apart — this is the whole reason
            # `OWNERSHIP_MIN_EXTENSION_VERSION` exists:
            #
            #   * a PRE-OWNERSHIP release (<= 0.1.8) does not implement
            #     `owner_recover` at all, so the method is simply unknown and
            #     the catch-all emits this shape. Telling that user to update is
            #     an instruction the store may be unable to satisfy, and the
            #     extension is otherwise perfectly usable — so with NO durable
            #     obligation in flight we degrade to the capability-only path
            #     and keep working.
            #   * a CURRENT release (>= 0.1.9, which DOES ship `owner_*`) gets
            #     here only when something inside the handler threw a bare
            #     error — the wedged worker. It must be reported as a wedge,
            #     naming the one remedy that clears it, NOT as a version skew.
            #
            # A durable obligation overrides the first case: the session owes
            # the extension a reconciliation the peer cannot perform, so the
            # honest failure stands rather than a silent downgrade that would
            # leave a tab stranded.
            if exc.code is ErrorCode.PROTO_MISMATCH:
                # A genuine incompatibility: the peer's proto falls outside
                # `MIN_SUPPORTED_PROTO..PROTO_VERSION`. Unchanged copy — here
                # updating really is the remedy, and the popup's `#incompatible`
                # card is the user-facing half of the same verdict.
                raise BrowserOwnershipError(
                    _ownership_requires_update_message(self._lane().name)
                ) from exc
            if exc.code is ErrorCode.OWNER_REFUSED:
                # A typed refusal from an ownership-AWARE peer. The peer now also
                # says WHICH kind via an additive `data.refusal` discriminator, so
                # the two are told apart without substring-matching the message:
                #
                #   * `stale`   — our own generation/predecessor is behind; the
                #     remedy is a resume, the ordinary same-session case.
                #   * `foreign` — the proof/session/shape is not ours at all (a
                #     different owner, a tab that belongs to another allocation).
                #     Never retried as if it were our own staleness.
                #
                # The distinction is CARRIED, not inferred: an older peer that
                # predates the key omits it, which reads as unknown and keeps the
                # prior bare-refusal behaviour (the failure still stands).
                refusal = str(exc.data.get("refusal") or "")
                if refusal == "stale":
                    raise BrowserOwnershipError(
                        "browser owner generation is stale; resume this session's scope "
                        "before browsing"
                    ) from exc
                if refusal == "foreign":
                    raise BrowserOwnershipError(
                        "this browser scope belongs to a different owner; no action taken"
                    ) from exc
                raise
            if exc.code is ErrorCode.INTERNAL and (
                "timeout_s" not in exc.data and not exc.data.get("stalled")
            ):
                if not self._peer_is_pre_ownership():
                    # A CURRENT extension (or one we cannot identify, which must
                    # not be read as old) that cannot answer a verb it ships
                    # means its worker has stopped answering. The remedy is the
                    # toggle, never an update the store may not be able to serve.
                    raise BrowserOwnershipError(
                        _peer_stopped_answering_message(self._lane().name)
                    ) from exc
                if not self.has_durable_obligation():
                    # Nothing to reconcile, so the extension's missing
                    # lifecycle costs this session nothing: keep working in
                    # capability-only mode instead of demanding an update.
                    self.ownership = False
                    self.assert_current()
                    return self._degraded_recover()
                # A durable obligation exists and only `owner_*` can reconcile
                # it, so the honest failure stands — a silent downgrade here
                # would leave a tab stranded.
                raise BrowserOwnershipError(
                    _ownership_requires_update_message(self._lane().name)
                ) from exc
            raise
        if result.get("ownership_version") != 1:
            # The peer answers `owner_recover` but describes a lifecycle this
            # runtime does not know. Not the legacy case (that one cannot answer
            # at all), so no degradation is safe: the two sides would disagree
            # about what an obligation is.
            raise BrowserOwnershipError(
                f"{_host_noun(self._lane().name)} needs ownership-recovery support; update it "
                "first"
            )
        self.ownership = True
        self.assert_current()
        if self.record.get("release_pause") and result.get("state") != "unresolved":
            await self.client().call("owner_release", self.params())
            self.assert_current()
            self.record.pop("release_pause", None)
            self.record["retention"] = ""
        if result.get("state") == "unresolved" and self.record.get("surface_id"):
            # A full browser restart destroys session-storage authority. Keep
            # the old capability only as private evidence, never adopt by tab ID.
            self.record["unresolved_surface_id"] = self.record["surface_id"]
        self.record["surface_id"] = str(result.get("tab", ""))
        self.record["state"] = (
            "cleanup_pending"
            if self.record.get("terminal")
            else str(result.get("state", "unresolved"))
        )
        if result.get("state") != "unresolved":
            # Acknowledged CAS collapses THIS lane's crash-replay chain. Before
            # this receipt every attempted generation remains a possible peer
            # state, and the OTHER lane's receipt is still outstanding.
            self._acknowledge_lane_chain()
        # The extension has now retired its copy, so the obligation is
        # discharged. Held until the ACK rather than cleared when it was
        # written: an unacknowledged resume must be replayed, and leaving it
        # set would make a LATER finish's terminal clearable by a recover that
        # is no longer a resume at all.
        self.record.pop("resumed_scope", None)
        # A successful reconcile means this link really does implement the
        # lifecycle, so any `unavailable` marker written by an earlier degraded
        # pass under a different peer is retired with it.
        self.record.pop("ownership", None)
        self.recovered = True
        self._save()
        return result

    def _degraded_recover(self) -> dict[str, Any]:
        """Capability-only recovery for a link with no ownership lifecycle.

        Reached when a PROVEN pre-ownership extension refuses `owner_recover`
        and no durable obligation is in flight. It resolves whatever this record
        already holds and writes a REDACTED ``ownership: "unavailable"`` marker
        so diagnostics can tell "worked, in legacy mode" from "ownership
        proven" — the distinction support needs and that a bare state string
        cannot make.

        It deliberately reconciles nothing: there is nothing the extension could
        answer, and the record's own capability is the only thing that survives.
        """
        self.assert_current()
        surface = str(self.record.get("surface_id", ""))
        self.record["ownership"] = "unavailable"
        self.record["state"] = (
            "cleanup_pending" if self.record.get("terminal") else ("owned" if surface else "closed")
        )
        self.recovered = True
        self._save()
        return {"state": self.record["state"], "tab": surface, "ownership_version": 0}

    def remember(
        self, surface_id: str, *, state: str | None = None, host: str | None = None
    ) -> None:
        """Record the surface AND, on the first successful open, its host.

        The host is written here rather than at open time so the record cannot
        name a host for a surface that never materialised: `remember` is the one
        call that follows a real open. It is only ever SET, never cleared, so a
        later `close` (which remembers an empty surface) cannot erase the lane a
        resumed session still needs to reach its record's owner.

        A NON-EMPTY handle also RETARGETS the lane to the host that actually
        allocated it (`host`, which every caller passes truthfully from the
        transport that just answered). That is not the same as the record's
        `host` field growing a new value: the field is what a RESUMED session
        reads, while the lane is what this live session's `finish`, `recover`
        and teardown will dial. Leaving the lane on the host of a tab that has
        already closed is exactly the defect QA Q-5 reproduced — the app's
        allocation stayed live while `owner_finish` was sent to the extension.
        """
        self.assert_current()
        if host:
            # Only a fresh allocation may move the lane: an empty surface (a
            # close) keeps the lane so the record's owner is still reachable, and
            # a capability held on another host keeps it too (see `select_host`).
            if surface_id:
                self._retarget_lane(host)
                # The host that just SERVED an open under this run's generation
                # holds a scope stamped with it (a strict-fence peer refuses the
                # open otherwise), so it owes no reconciliation before the
                # finish. Without this, per-lane state would make the lane move
                # after a fresh allocation send a spurious `owner_recover`.
                self.recovered = True
            else:
                self.select_host(host)
        if self.host:
            self.record["host"] = self.host
        self.record["surface_id"] = surface_id
        self.record["state"] = (
            "cleanup_pending"
            if self.record.get("terminal")
            else state or ("owned" if surface_id else "closed")
        )
        self._save()

    def allocate(self) -> None:
        self.assert_current()
        if self.record.get("terminal"):
            raise BrowserOwnershipError("browser scope has ended; resume it before allocating")
        if self.record.get("state") in ("closed", "unresolved"):
            self.record["allocation_id"] = secrets.token_urlsafe(24)
        self.record["state"] = "allocating"
        self._save()

    async def finish(self, generation: str, outcome: str) -> BrowserCleanupResult:
        try:
            self.initialize()
        except (RuntimeError, OSError, ValueError):
            # A stale child still owes its terminal event. Refusal must be a
            # result, not an exception that bypasses the runner's publication.
            return BrowserCleanupResult(
                "unresolved", "owner changed or unreadable; no action taken"
            )
        if generation != self.generation:
            return BrowserCleanupResult("unresolved", "stale generation; no action taken")
        # Persist intent BEFORE waiting for a possibly in-flight tool. A crash
        # or timeout leaves an exact obligation, not guessed terminal evidence.
        self.assert_current()
        outcome = str(self.record.get("terminal") or outcome)
        self.record["terminal"] = outcome
        self.record["state"] = "cleanup_pending"
        self._save()
        try:
            async with self.lock:
                self.assert_current()
                await self.ensure_recovered()
                # A link with no ownership lifecycle settles through the plain
                # `close` verb instead. Without this the fallback that keeps the
                # tool working would strand the very tab it opened: `owner_finish`
                # does not exist on that peer, so the read would fail and the tab
                # would live on with nobody able to close it.
                if self.ownership_mode() is False:
                    return await self.finish_degraded()
                if self.record.get("unresolved_surface_id") and not self.record.get("surface_id"):
                    return BrowserCleanupResult(
                        "unresolved",
                        "browser ownership could not be proven after restart; no tab adopted",
                    )
                if self.record.get("retention"):
                    await self.client().call(
                        "owner_retain", {**self.params(), "reason": self.record["retention"]}
                    )
                result = await self.client().call(
                    "owner_finish", {**self.params(), "outcome": outcome}
                )
                self.assert_current()
                state = str(result.get("state", "unresolved"))
                self.record["state"] = state
                if state == "closed":
                    self.record["surface_id"] = ""
                    if self.record.get("unresolved_surface_id"):
                        state = self.record["state"] = "unresolved"
                self._save()
                return BrowserCleanupResult(state)
        except Exception as exc:
            # The bridge's own message names the command that diagnoses the
            # failure; the class name is the fallback for a type that carries
            # no message, matching how the CLI renders its outer handler.
            return BrowserCleanupResult("pending", str(exc) or type(exc).__name__)

    async def finish_degraded(self) -> BrowserCleanupResult:
        """Settle the scope with a plain `close`, for a peer with no `owner_*`.

        LOCK-FREE on purpose: `finish` reaches this while already holding
        ``self.lock``, and the session tool's release path calls it from inside
        the same lock. Taking the lock here would deadlock on that path.

        The close names the RECORDED surface capability, never a numeric tab id
        and never the unresolved-evidence slot (both of those are the
        "never adopt by tab id" stance) — and it carries the IDENTITY PARAMS,
        which is not decoration: a released extension refuses a `close` on a
        surface that carries an ``allocationId`` when ``owner_proof`` is absent
        (``owner_refused`` / "owner-aware client required"). Sending the tab
        capability alone therefore made this fallback exactly the thing it
        exists to prevent — a degradation path that strands the tab it cannot
        reconcile (review R1-2) — on any peer the classification guessed wrong
        about. So the obligation verbs are forked; the identity params are not,
        which is what this docstring claimed before the code did it.

        ONE retention decision, stated here because the record has to agree with
        the copy: a pre-ownership extension cannot enforce a retention, so a
        retention recorded by the degraded `retain` is a STATEMENT OF INTENT the
        extension never accepted. Once this method has closed the tab, that
        intent must not outlive the tab it described — leaving it set made
        `cleanup_disposition` refuse a settled row as "retained: … the owning
        session must release it", i.e. a row that is neither cleanable nor true
        (review R1-4). It is cleared only on a SETTLED close; a failed close
        keeps it, because there the tab really is still out there.
        """
        if self.record.get("unresolved_surface_id") and not self.record.get("surface_id"):
            return BrowserCleanupResult(
                "unresolved",
                "browser ownership could not be proven after restart; no tab adopted",
            )
        surface = str(self.record.get("surface_id", ""))
        if surface:
            try:
                # `{"tab": …, **identity}` matches the shape `builtin.py`'s own
                # bridge `close` already sends, so there is one spelling of an
                # owner-bearing close in the codebase rather than two.
                await self.client().call("close", {"tab": surface, **self.params()})
            except BridgeError as exc:
                # A failed close is a result here for the same reason it is in
                # `finish`: the tab is genuinely still out there and the record
                # must say so instead of reporting a settled scope.
                self.record["state"] = "pending"
                self._save()
                return BrowserCleanupResult("pending", str(exc) or type(exc).__name__)
        self.assert_current()
        self.record["surface_id"] = ""
        self.record["state"] = "closed"
        # See the docstring: an unenforceable retention must not outlive the tab
        # it described, or the settled row reads as "retained".
        self.record["retention"] = ""
        self._save()
        return BrowserCleanupResult("closed")


#: States in which a tab is stranded and an operator may recover it. Both are
#: reachable: ``finish`` writes ``cleanup_pending`` as its durable intent, then
#: OVERWRITES it with the bridge's own reply, so a failed ``chrome.tabs.remove``
#: persists ``pending``. Listing one and accepting the other made the single
#: genuinely-stranded state render as "not a candidate" while cleanup would have
#: taken it. One tuple, read by both, so they cannot drift apart again.
_RECOVERABLE_STATES = ("cleanup_pending", "pending")


def cleanup_disposition(value: dict[str, Any]) -> tuple[bool, str]:
    """``(eligible, reason)`` for one record — the ONE eligibility rule.

    The reason names what would make the row actionable, because "you copied
    the wrong generation" and "a human is finishing a login in that tab" are
    opposite situations and the operator cannot otherwise tell which they are
    in. Guessing wrong pushes them to retry against a protected tab, which is
    the behaviour the fence exists to discourage.
    """
    if value.get("retention"):
        return False, f"retained: {value['retention']} — the owning session must release it"
    if not value.get("terminal"):
        # The crash shape: killed before finish_browser_scope, so no terminal
        # intent exists and none may be inferred from the process being gone.
        # Refusing is correct, but refusing SILENTLY dead-ends the operator, so
        # name the route that actually works: the owner closes its own tab.
        return False, (
            "no terminal intent recorded (owner did not finish) — "
            "resume it with 'lop --resume <session>' and let it close the tab"
        )
    if value.get("state") not in _RECOVERABLE_STATES:
        return False, f"state is '{value.get('state', 'unresolved')}'; nothing is stranded"
    return True, ""


def read_inventory(directory: Path) -> list[dict[str, Any]]:
    """Redacted local evidence; unknown/live/retained never becomes eligible."""
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"*/{RESOURCE_NAME}")):
        try:
            value = json.loads(path.read_text())
            eligible, reason = cleanup_disposition(value)
            rows.append(
                {
                    "session_id": path.parent.name,
                    "generation": value.get("generation", ""),
                    "state": value.get("state", "unresolved"),
                    "terminal": value.get("terminal", ""),
                    "retention": value.get("retention", ""),
                    # REDACTED, and deliberately so: this marker names the MODE,
                    # never a capability. Published because "it worked, in
                    # legacy mode" and "its ownership is proven" are otherwise
                    # indistinguishable in a support conversation, and the
                    # operator cannot ask the right question without the
                    # difference. Empty for every record that has not degraded.
                    "ownership": value.get("ownership", ""),
                    "cleanup_candidate": eligible,
                    "blocked_reason": reason,
                }
            )
        except (OSError, ValueError, TypeError):
            rows.append(
                {
                    "session_id": path.parent.name,
                    "state": "unresolved",
                    "cleanup_candidate": False,
                    "blocked_reason": "record is unreadable",
                }
            )
    return rows


def _session_config_root(directory: Path) -> Path | None:
    """The config root when ``directory`` is ``<config>/sessions/<id>``, else ``None``.

    CHECKED rather than assumed (review round 3, NIT 1): the guess this replaces was
    ``directory.parent.parent`` for any path at all, which silently produced an unrelated
    root for a caller that passed something else. A path this build cannot place in the
    session layout is not a session, and ``cleanup_exact`` says what it does about that.
    """
    parent = Path(directory).parent
    return parent.parent if parent.name == "sessions" else None


async def cleanup_exact(
    directory: Path,
    generation: str,
    *,
    config_dir: Path | None = None,
) -> BrowserCleanupResult:
    """Explicit operator recovery of ONE terminal generation, never a sweep.

    The execution lease rejects live/uncertain owners. A dead PID only permits
    acquiring that lease; matching durable terminal intent is still mandatory.

    ``config_dir`` IS REQUESTED, NOT DERIVED (review round 3, NIT 1). The move guard used to
    take ``directory.parent.parent`` positionally, so a caller passing anything but
    ``<config>/sessions/<id>`` got a root that belonged to no config — no guard and no
    error, the one shape in which a safety check fails open. A caller that knows its store
    passes it (``cli`` does); one that does not gets the layout CHECKED, and a directory
    this build cannot place under ``<config>/sessions/`` consults no guard because nothing
    can be handing over a path that is not a session's.
    """
    from local_operator.session_lease import acquire_session_lease

    value = BrowserResource(directory, directory.name)._load()
    if not value:
        return BrowserCleanupResult("unresolved", "no ownership record for that session")
    if value.get("generation") != generation:
        return BrowserCleanupResult(
            "unresolved",
            "generation does not match the current record — "
            "re-run 'lop browser tabs' and copy the current generation",
        )
    eligible, reason = cleanup_disposition(value)
    if not eligible:
        return BrowserCleanupResult("unresolved", reason)
    # THE HANDOFF GUARD, CONSULTED HERE BECAUSE THIS IS THE ONE OTHER LEASE ACQUIRER
    # (review round 2, MINOR 1). Every product opener goes through
    # ``session_factory._prepare``, which refuses a session whose move is in flight; this
    # path takes the lease directly, so a cleanup that lands between the move's lease
    # re-check and its delete could become a second writer on a transcript that device is
    # about to hand over. ``.browser-resource.json`` is all this holder writes, which is
    # why the reviewer rated it a minor — but the refusal costs one sentence and the
    # window is not worth keeping.
    from local_operator.session.placement import handoff_guard_refusal

    root = Path(config_dir) if config_dir is not None else _session_config_root(directory)
    if root is not None:
        moving = handoff_guard_refusal(root, directory.name)
        if moving:
            return BrowserCleanupResult("unresolved", moving)
    lease = acquire_session_lease(directory)
    try:
        # Constructed INSIDE the lease so it reads the generation the lease just
        # established. Building it outside made a CLI-path resource mint its own
        # token and persist it, rotating the very identifier the listing had
        # just told the operator to copy: their retry then failed as "stale"
        # through no fault of theirs, on every attempt.
        resource = BrowserResource(directory, directory.name)
        current = resource._load()
        if current != value:
            return BrowserCleanupResult("unresolved", "ownership changed; no action taken")
        resource.adopt(str(value["generation"]))
        return await asyncio.wait_for(
            resource.finish(resource.generation, str(value["terminal"])), 5.0
        )
    except TimeoutError:
        return BrowserCleanupResult("pending", "browser cleanup deadline exceeded")
    finally:
        lease.release()
