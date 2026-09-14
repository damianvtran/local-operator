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


class OwnershipDiscovery(Protocol):
    """The one discovery question the ownership layer asks, per host.

    It is NOT a state read: what the ownership mode needs is the peer's
    (version, protocol) pair, and the two hosts answer it from different fields
    — the extension from ``extension_version``/``extension_proto`` plus a proven
    connection, the UI host from its ``app_version`` and ``proto`` (it has no
    second process to be disconnected from). ``None`` means "cannot tell", and
    every caller must read it that way.
    """

    def peer_identity(self) -> tuple[str, int] | None: ...


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

    def peer_identity(self) -> tuple[str, int] | None:
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

    def peer_identity(self) -> tuple[str, int] | None:
        try:
            from local_operator.ui_browser import state as ui_state

            current = ui_state.read()
        except Exception:  # noqa: BLE001 - discovery must never raise at a call site
            return None
        if current is None:
            return None
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
        self.recovered = False
        #: WHICH host owns this session's surface: "" (not learned yet), "ui" or
        #: "bridge". Written into the record on the first successful `open`, so a
        #: resumed session selects the lane it opened on. Everything the lane
        #: touches — the `owner_*` transport and the discovery read that feeds
        #: `ownership_mode` — is selected from it.
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
        # The peer identity `self.ownership` was learned against. See
        # `_peer_identity` for why the pair, not just the version.
        self._ownership_peer: tuple[str, int] | None = None
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

    def _lane(self) -> OwnershipHost:
        """The ownership lane for this session: transport + discovery, per host."""
        return ownership_host(self.host or str(self.record.get("host", "")))

    def select_host(self, host: str) -> None:
        """Bind this session's surface to a host, once it is known.

        Called by the tool before the lane runs, from the pinned surface prefix
        when there is one and from the availability probe when there is not. An
        already-known host wins over the argument: a surface's transport is
        pinned for its whole life, so a later action cannot silently move an
        established session to a different lane.
        """
        if not self.host and host:
            self.host = host

    def pinned_host(self) -> str:
        """The host this session's durable state pins it to, or "" for none.

        The RECORD is the durable form of the pin `select_host` documents, and
        on a RESUMED session it is the only one there is: the surface prefix in
        `state.surface_id` is empty until the lane adopts it from this same
        record, which happens well after the gate has to decide. Reading only
        the availability probes there does not merely guess — `_lane()` prefers
        `self.host` over `self.record["host"]`, so a probe-selected host
        OVERRIDES the record and silently moves a resumed session onto whichever
        host happened to be up, including off the host that owns its tab.

        Two spellings of one fact, and the HANDLE is read first:

        * `surface_id` names its own host in its prefix, and it is the surface
          the session is actually HOLDING, so it outranks the field beside it.
          The two can disagree, because `remember("ui:…", host="ui")` passes
          the truthful host while `select_host` keeps an established lane's
          host: a session that has been talking to the daemon records
          `host: "bridge"` beside a `ui:` handle. Reading the field first sends
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
        for name in (HOST_UI, HOST_BRIDGE):
            if surface.startswith(f"{name}:"):
                return name
        host = str(record.get("host", ""))
        if host and self._record_owes_reconciliation(record):
            return host
        return ""

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

        generation, _pid = _read_claim(self.directory / ".execution-lease")
        return generation or ""

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
        generations = list(self.record.get("bridge_generations", []))
        for candidate in (self.previous_generation, self.generation):
            if candidate and candidate not in generations:
                generations.append(candidate)
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

    def params(self) -> dict[str, Any]:
        self.initialize()
        return {
            "requester": f"session:{self.session_id}",
            "owner_proof": str(self.record["proof"]),
            "owner_generation": self.generation,
            "previous_generation": self.previous_generation,
            "previous_generations": self.record["bridge_generations"],
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

    def _peer_identity(self) -> tuple[str, int] | None:
        """The peer's (version, proto), or None when it cannot be told.

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
        peer = self._peer_identity()
        if peer != self._ownership_peer:
            self._ownership_peer = peer
            self.ownership = None
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
        version, _proto = peer
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
            # Acknowledged CAS collapses the crash-replay chain. Before this
            # receipt every attempted generation remains a possible peer state.
            self.record["bridge_generations"] = [self.generation]
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
        """
        self.assert_current()
        if host:
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
                if not self.recovered:
                    await self.recover()
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


async def cleanup_exact(directory: Path, generation: str) -> BrowserCleanupResult:
    """Explicit operator recovery of ONE terminal generation, never a sweep.

    The execution lease rejects live/uncertain owners. A dead PID only permits
    acquiring that lease; matching durable terminal intent is still mandatory.
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
