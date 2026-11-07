"""The borrower's half: ask the owner, hold the bearer, never write it down.

TWO ROLES IN ONE MODULE, because they are one conversation and splitting them
would put the frame shape in two places:

* **leg 1 (runtime → its own relay)** — :meth:`MeshCredentialClient.for_this_device`
  builds a client that dials THIS device's relay over its loopback control socket
  with ``credential_grant``. The runtime is a CLIENT here, not a dispatchee, which
  is why no ``ControlOp`` is added anywhere: the local op travels the relay's own
  control vocabulary (P0 declared the three names in ``types.LOCAL_OPS``).
* **leg 2 (relay → owner relay)** — :meth:`MeshCredentialClient.for_relay` builds
  one that reaches the owning device and runs ``net_broker``.

WHAT THIS MODULE GUARANTEES, and the mechanism for each:

* **The borrower makes ZERO token POSTs.** There is no refresh call anywhere in
  this file, and there is no refresh token to call with: the owner returns an
  access token and the requester stores it in memory under
  :class:`GrantCache`. A downstream owner-offline is a refusal, never a fallback
  to "refresh it here" — there is nothing here to refresh with.
* **No bearer touches a disk.** :class:`GrantCache` is a dict on one object. The
  only durable state this module writes is :class:`PlacementState`, whose writer
  refuses a token field (``state.py``) — so "the bearer stayed in memory" is
  enforced by the file's own schema, not by discipline.
* **A refusal is cached, so a "no" is not re-asked per provider call.** The
  device-local observation document is consulted BEFORE any dial; that is the
  whole retry-storm defence, and it is why ``state.py`` is a file rather than a
  dict.
* **Latency is bounded by the local broker, not the owner.** ``deadline_s`` is
  deliberately BELOW the secret broker's own ``REQUEST_TIMEOUT_S = 10 s``
  (``secrets/broker.py``): a borrow must never be able to hang a turn longer than
  the local secret store already would.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Collection

from local_operator.network.credentials import mcp_state
from local_operator.network.credentials.messages import render_broker_error
from local_operator.network.credentials.placement import (
    PlacementDocument,
    merge_from_peer,
    placement_for_store,
    placement_path,
    placement_state_path,
)
from local_operator.network.credentials.state import PlacementState
from local_operator.network.credentials.types import (
    BROKER_ERROR_TTL_MS,
    MAX_PEER_RETRY_AFTER_MS,
    NO_RETRY,
    PEER_BROKER_OP,
    QUOTA_REASON_POOL_DEPLETED,
    BrokerError,
    CredentialPlacementEntry,
    Grant,
    credential_key_for_mcp,
    credential_key_for_provider,
    is_device_bound_provider,
    is_mcp_key,
)

logger = logging.getLogger("local_operator.network.credentials.client")

#: The outcomes that mean "the owner did not answer" and nothing more. Only these may
#: be bridged by a grant this process already holds (:meth:`_held_or`).
_HOLDABLE_REFUSALS: frozenset[str] = frozenset({"owner_offline", "owner_unreachable"})

#: :meth:`MeshCredentialClient._hold_admission`'s verdict for "this device is no
#: longer a live member". Not a broker code: each caller answers it with the
#: network-wide ``_invalidate_membership_scope``, never a per-key deny.
_MEMBERSHIP_LOST = "<membership-lost>"

#: Outcomes that are not an owner answer about permission either, so they neither
#: serve nor drop a held grant: a local broker fault, a busy owner, a quota rest.
_TRANSIENT_REFUSALS: frozenset[str] = frozenset(
    {"internal", "rate_limited", "quota_blocked", "unsupported"}
)

#: The borrower's whole budget for one borrow, in seconds. Below the local secret
#: broker's 10 s (see this module's docstring) and far below the owner-side
#: deadline registered for ``net_broker`` (75 s): a borrower that gave up first
#: would report a bare timeout for a request the owner was about to answer, and a
#: borrower that waited as long as the owner would hold a provider call open for
#: the provider refresh's full budget.
CLIENT_DEADLINE_S = 8.0

#: How long before its expiry a cached grant is re-asked, in milliseconds. LARGER
#: than the owner's own ``OAUTH_REFRESH_SKEW_MS`` (60 s) on purpose: the requester
#: must never be the component that discovers an expiry, so it asks early enough
#: that the owner's refresh (if any) happens on the owner's clock.
REASK_MARGIN_MS = 120_000

#: The TTL of an ``unsupported`` observation, i.e. "this peer's build cannot do
#: this". Session-scoped in effect: it cannot change while the build does not, so
#: it is cached until the process exits rather than re-asked per call.
UNSUPPORTED_OBSERVATION_MS = 300_000


class GrantCache:
    """In-memory grants on the borrower. Deliberately not persistable.

    The single most important property of this class is what it does NOT have: a
    serializer. There is no ``to_json``, no path and no pickle hook, so a future
    caller cannot accidentally make a borrowed bearer durable — the same
    structural argument the placement documents make in the other direction. A
    restart loses every grant and re-asks, which is correct: the grant was
    ``min(token_exp, now + 15 min)`` anyway.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._grants: dict[tuple[str, str], Grant] = {}
        self._policy_fences: dict[tuple[str, str], Any] = {}
        # What a hot provider read (``MeshCredentialClient._serve_cached``) must
        # find unchanged before it may hand a grant out without dialling the owner:
        # ``(stored_at monotonic, lifecycle fence, selection inputs)``. Kept beside
        # the grant and dropped with it at every eviction site, so a slot can never
        # outlive the evidence that licensed reusing it. Absent = never reusable
        # (a put without a fence, i.e. an uncovered owner or an older caller).
        self._reuse: dict[tuple[str, str], tuple[float, Any, Any]] = {}

    def get(self, key: str, session_id: str) -> Grant | None:
        """The cached grant when it is still usable, else ``None``.

        ``REASK_MARGIN_MS`` before expiry, the grant is treated as absent: the
        caller re-asks and the owner decides whether that needs a refresh.
        """
        with self._lock:
            grant = self._grants.get((key, session_id))
        if grant is None:
            return None
        now_ms = time.time() * 1000.0
        if now_ms >= grant.grant_expires_at_ms - REASK_MARGIN_MS:
            return None
        return grant

    def peek(self, key: str, session_id: str) -> Grant | None:
        """The cached grant for ``(key, session_id)``, WITHOUT the re-ask margin.

        AN ATTRIBUTION READ, NOT A USABILITY READ. When a provider has just refused
        a bearer, the grant that carried it may be inside ``REASK_MARGIN_MS`` — the
        window where :meth:`get` reads as absent — and that grant is exactly the one
        a report must name. The reads that serve a bearer without dialling (the
        ``github`` gate in ``request_grant_sync``, the provider hot read
        ``_serve_cached``, MCP's ``_live_bearer``) use :meth:`get`, so they never
        hand out a grant inside the margin.

        ONE CALLER DOES SERVE FROM HERE: ``_held_or``, the owner-unreachable path.
        It applies its own ``Grant.usable_at`` check, which has no margin, so it
        serves a grant marked ``held_offline`` right up to the grant's expiry.
        """
        with self._lock:
            return self._grants.get((key, session_id))

    def put(
        self,
        key: str,
        session_id: str,
        grant: Grant,
        *,
        policy_fence: Any = "",
        lifecycle_fence: Any = None,
        selection: Any = None,
    ) -> None:
        with self._lock:
            self._grants[(key, session_id)] = grant
            self._policy_fences[(key, session_id)] = policy_fence
            if lifecycle_fence:
                self._reuse[(key, session_id)] = (time.monotonic(), lifecycle_fence, selection)
            else:
                self._reuse.pop((key, session_id), None)

    def reuse_stamp(self, key: str, session_id: str) -> tuple[float, Any, Any] | None:
        """``(stored_at, lifecycle_fence, selection)`` recorded with the slot, or ``None``.

        ``stored_at`` is ``time.monotonic()`` on THIS device: the reuse age is a
        borrower-local bound, so neither the owner's clock nor a wall-clock step
        can stretch it.
        """
        with self._lock:
            return self._reuse.get((key, session_id))

    def policy_stamp(self, key: str, session_id: str) -> Any:
        with self._lock:
            return self._policy_fences.get((key, session_id))

    def drop_borrowed(self, self_device: str) -> None:
        """One client owns one root/network; local-owned sentinels are unrelated."""
        with self._lock:
            for pair, grant in list(self._grants.items()):
                if grant.credential_ref.owner_device != self_device:
                    del self._grants[pair]
                    self._policy_fences.pop(pair, None)
                    self._reuse.pop(pair, None)

    def invalidate_policy(self, key: str, session_id: str, fence: str) -> bool:
        """Drop ALL old slots, not just the session that first hears the refusal.

        Compare opaque generations, never the remote owner's clock. A fresh
        authorized grant can coexist with this watermark without resurrecting
        grants issued before a restriction another process persisted.
        """
        with self._lock:
            stale = [
                pair
                for pair in self._grants
                if pair[0] == key and self._policy_fences.get(pair, "") != fence
            ]
            requested = (key, session_id) in stale
            for pair in stale:
                del self._grants[pair]
                self._policy_fences.pop(pair, None)
                self._reuse.pop(pair, None)
            return requested

    def drop(self, key: str, session_id: str = "") -> None:
        """Forget one grant, or every session's grant for ``key`` when unnamed."""
        with self._lock:
            if session_id:
                self._grants.pop((key, session_id), None)
                self._policy_fences.pop((key, session_id), None)
                self._reuse.pop((key, session_id), None)
                return
            for pair in [p for p in self._grants if p[0] == key]:
                del self._grants[pair]
                self._policy_fences.pop(pair, None)
                self._reuse.pop(pair, None)

    def find_by_token(self, key: str, bearer: str) -> tuple[str, str] | None:
        """The ``(key, session_id)`` a bearer belongs to, for attributing a failure.

        The failover driver hands ``rotate_sibling`` the bearer that failed, and the
        only way to know whether that bearer was BORROWED (rather than a local row's)
        is to compare it against what this process is holding. Comparing the token
        itself is deliberate: it is exact where an id is not, and neither the token
        nor its comparison leaves this process.
        """
        if not bearer:
            return None
        with self._lock:
            items = list(self._grants.items())
        for pair, grant in items:
            if pair[0] == key and grant.access_token == bearer:
                return pair
        return None

    def keys(self) -> list[str]:
        """Every key this process is holding a grant for, sorted and de-duplicated.

        The reconcile pass needs the SET OF KEYS to re-check against placement; it
        never needs the grants themselves, so no accessor returns one. Reading the
        keys is not serving a bearer — ``get``/``peek`` remain the only readers of
        a grant, and neither is called here.
        """
        with self._lock:
            return sorted({key for key, _session in self._grants})

    def clear(self) -> None:
        with self._lock:
            self._grants.clear()
            self._policy_fences.clear()
            self._reuse.clear()


class MeshCredentialClient:
    """Asks the device that owns a credential for a short-lived bearer."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        self_device: str = "",
        network_id: str = "",
        placement: PlacementDocument | None = None,
        state: PlacementState | None = None,
        relay: Any = None,
        deadline_s: float = CLIENT_DEADLINE_S,
    ) -> None:
        self.root = root
        self.self_device = self_device
        self.network_id = network_id
        self._placement = placement
        #: ``(mtime_ns, size)`` of the file ``_placement`` was read from, so a change
        #: on disk is noticed with one ``stat`` instead of a parse per call.
        self._placement_stamp: tuple[int, int] | None = self._stat_placement()
        self._placement_lock = threading.Lock()
        self._state: PlacementState | None = state
        #: ``(mtime_ns, size)`` of the observation file ``_state`` reflects, so a
        #: clearing written by ANOTHER process is noticed with one ``stat``: the
        #: pull runs on the relay, and the session serving a stale refusal from its
        #: own memory is a different process. See :attr:`state`.
        self._state_stamp: tuple[int, int] | None = self._stat_state()
        self._relay = relay
        self.deadline_s = deadline_s
        self.grants = GrantCache()
        # A validated store remains established even after every cached lease is
        # evicted. This is deny-only knowledge, scoped to canonical identity,
        # never permission or a claim about another process's unobserved history.
        self._held_policy_established: set[tuple[Path, str, str]] = set()
        self._state_lock = threading.Lock()
        #: The sync engine used by the CLI's direct fallback. Built lazily; the
        #: runtime never reaches it (it asks its own relay, leg 1).
        self._engine: Any = None

    # -- construction -------------------------------------------------------

    @classmethod
    def for_this_device(
        cls,
        root: Path | None = None,
        network_id: str = "",
        *,
        include_bridged: bool = False,
        relay: Any = None,
    ) -> MeshCredentialClient | None:
        """A runtime-side client, or ``None`` when this device may borrow nothing.

        ``None`` is the zero-peer answer. ``include_bridged`` is only for session
        construction: an already-admitted active remote member permits an empty
        client whose document can later converge. Status readers keep the strict
        default, and every bearer request still checks current holder permission.
        No member/keypair is created and no dial occurs on either construction path.
        """
        found = placement_for_store(root)
        if found is None:
            if not include_bridged:
                return None
            found_network, document = _empty_document_for_member(root)
            if document is None:
                return None
        else:
            found_network, document = found
        from local_operator.network.identity import load as load_identity

        identity = load_identity(root)
        if identity is None:
            return None
        device = identity.device_id
        return cls(
            root=root,
            self_device=device,
            network_id=network_id or found_network,
            placement=document,
            state=PlacementState.load(network_id or found_network, root),
            # The CLI's direct fallback passes a relay it built for ONE dial (see
            # ``pull_copy_sync``); every runtime caller keeps the default ``None``,
            # where leg 1 goes through the control socket instead.
            relay=relay,
        )

    @classmethod
    def for_relay(cls, server: Any) -> MeshCredentialClient | None:
        """A relay-side client — the leg-1 server and the leg-2 dialer.

        MORE TOLERANT THAN ``for_this_device``, deliberately. A relay serving a
        network whose members include another device builds a client even when this
        device holds no placement document yet, because an EMPTY document is exactly
        the state a device is in before someone shares a credential with it — and a
        client that refused to exist then could never learn about the share. The
        bootstrap is :meth:`pull_placement`, which asks every other active member.

        The session-construction path keeps the strict predicate
        (``placement_for_store`` via ``build_auth_store``), so tolerating this here
        does not weaken the 0-peer guarantee there.
        """
        root = getattr(server, "root", None)
        found = placement_for_store(root)
        if found is not None:
            found_network, document = found
        else:
            found_network, document = _empty_document_for_member(root)
            if document is None:
                return None
        self_device = str(getattr(getattr(server, "identity", None), "device_id", ""))
        return cls(
            root=root,
            self_device=self_device,
            network_id=found_network,
            placement=document,
            state=PlacementState.load(found_network, root),
            relay=server,
        )

    # -- placement queries --------------------------------------------------

    @property
    def placement(self) -> PlacementDocument | None:
        """The placement document AS IT IS ON DISK, re-read when the file changes.

        A copy loaded once at construction is what let a relay keep answering from
        a sharing list the CLI (another process) had already changed, and then save
        that stale copy back over the change (review round 1, F2). One ``stat`` per
        read keeps this off the provider hot path's cost sheet.
        """
        if not self.network_id:
            return self._placement
        stamp = self._stat_placement()
        if stamp == self._placement_stamp:
            return self._placement
        with self._placement_lock:
            if stamp != self._placement_stamp:
                self._placement = (
                    PlacementDocument.load(self.network_id, self.root, self_device=self.self_device)
                    if stamp is not None
                    else None
                )
                # Changed/deleted placement invalidates permission and generation;
                # never serve a cached bearer across a withdrawal.
                self.grants.clear()
                self._placement_stamp = stamp
        return self._placement

    def _stat_placement(self) -> tuple[int, int] | None:
        if not self.network_id:
            return None
        try:
            found = placement_path(self.network_id, self.root).stat()
        except OSError:
            return None
        return (found.st_mtime_ns, found.st_size)

    def _stat_state(self) -> tuple[int, int] | None:
        """``(mtime_ns, size)`` of this device's observation file, or ``None``.

        The twin of :meth:`_stat_placement`, for :attr:`state` — same reason, same
        cost (one ``stat`` per read, no parse unless it moved).
        """
        if not self.network_id:
            return None
        try:
            found = placement_state_path(self.network_id, self.root).stat()
        except OSError:
            return None
        return (found.st_mtime_ns, found.st_size)

    def entry(self, key: str) -> CredentialPlacementEntry | None:
        return self.placement.entry(key) if self.placement is not None else None

    def owner_of(self, key: str) -> str:
        return self.placement.owner_of(key) if self.placement is not None else ""

    def is_holder(self, key: str) -> bool:
        return bool(self.placement and self.placement.is_holder(key, self.self_device))

    def should_borrow(self, key: str) -> bool:
        """Whether a borrow is the right next rung for ``key`` on this device.

        Requires BOTH "someone else owns it" and "I am a holder". Being a holder is
        what authorises the request, so a device that is not one must not send it —
        the owner would refuse, and the refusal would be cached against a key this
        device never had a claim to.
        """
        if self.placement is None:
            return False
        entry = self.placement.entry(key)
        if entry is None:
            return False
        return entry.owner_device != self.self_device and entry.is_holder(self.self_device)

    def owner_label(self, key: str) -> str:
        entry = self.entry(key)
        if entry is None:
            return "the owner device"
        return entry.owner_device_name or entry.owner_device

    def owner_last_seen_s(self, device: str) -> float | None:
        """Seconds since the owner was last observed, or ``None`` for never.

        Read from the member table rather than from the link, because the whole
        point of this number is the case where there IS no link.
        """
        if not device:
            return None
        # THE LAST GRANT THIS DEVICE WAS SERVED BY THE OWNER COUNTS AS A SIGHTING
        # (QA round 1, Q6). ``member.last_seen_at`` is only stamped by a rotation, so on
        # every real row it was ``None`` and an owner that had lent a grant minutes
        # earlier was reported "last seen never". A served grant is the strongest
        # evidence this module has that the owner was up, and it is already durable
        # (``PlacementState.last_grant_at``), so the newer of the two is the answer.
        stamps: list[float] = []
        with suppress(Exception):
            granted = self.state.last_grant_from(device)
            if granted:
                stamps.append(granted)
        try:
            from local_operator.network import store

            for record in store.list_networks(self.root):
                member = record.member(device)
                if member is not None and member.last_seen_at:
                    stamps.append(float(member.last_seen_at))
        except Exception:  # noqa: BLE001 — a missing observation is not a failure
            pass
        if not stamps:
            return None
        return max(0.0, time.time() - max(stamps))

    # -- refusal cache ------------------------------------------------------

    @property
    def state(self) -> PlacementState:
        """This device's observations, RE-READ when the file changes.

        The same pattern, for the same reason, as :attr:`placement` above — and the
        reason now runs in both directions. A refusal this process cached has to
        stop being served once ANOTHER process clears it: the pull runs on the
        relay, and the clearing a merge writes (``placement.py``) reaches the
        session that must not serve the stale "does not share" only through this
        file. A copy loaded once was exactly the shadow the clearing exists to
        remove.
        """
        with self._state_lock:
            if self._state is None:
                self._state = PlacementState.load(self.network_id, self.root)
                self._state_stamp = self._stat_state()
                return self._state
            stamp = self._stat_state()
            if stamp is not None and stamp != self._state_stamp:
                self._state = PlacementState.load(self.network_id, self.root)
                self._state_stamp = stamp
            return self._state

    def cached_refusal(self, key: str, *, now: float | None = None) -> BrokerError | None:
        """A refusal this device already has, or ``None``. Consulted BEFORE a dial."""
        status = self.state.status(key, now=now)
        if not status or status == "active":
            return None
        ttl = BROKER_ERROR_TTL_MS.get(status, 0)
        if ttl == NO_RETRY:
            ttl = UNSUPPORTED_OBSERVATION_MS
        floor = False
        row = self.state.observation(key, now=now)
        if row is None:
            return None
        # ``reason`` is deliberately NOT read back here. The observation document
        # carries the field for a diagnostic to read, but the sentence a person sees
        # is rendered at the point of use from the CODE (``messages.py``) — and a
        # borrower that echoed a stored string would be presenting one device's
        # prose as another's, which is the failure ``render_broker_error`` exists to
        # prevent. The message is filled in by ``_render`` below.
        if ttl <= 0:
            # A ZERO IN THE TABLE MEANS "NO HARD-CODED TTL — SEE WHAT THE OWNER
            # SAID" (design round 1, D1; corrected in review round 2, M2). Only the
            # remainder the owner itself stated may be counted down
            # (``owner_retry_after_ms``, recorded by ``_remember``): the row's own
            # ``retry_after_ms`` is the CACHE's lifetime, which for a refusal with
            # no stated remainder is this device's own default — rendering it back
            # as "the owner's remainder" claimed a fact about another device's cap
            # that nobody stated. No stated remainder ⇒ 0 ⇒ the sentence keeps
            # "for now", exactly as the fresh one does.
            owner_ms = int(row.get("owner_retry_after_ms") or 0)
            # A remembered clamped value is still a floor while it counts down: the
            # owner's reset lies beyond it, so "at least" stays true (#3108 r2 m1).
            floor = owner_ms >= MAX_PEER_RETRY_AFTER_MS
            if owner_ms:
                stamp = float(row.get("observed_at") or 0.0)
                current = time.time() if now is None else now
                elapsed_ms = int(max(0.0, current - stamp) * 1000.0)
                ttl = max(0, owner_ms - elapsed_ms)
        return BrokerError(
            code=status,
            key=key,
            owner_device=str(row.get("owner_device") or self.owner_of(key)),
            owner_device_name=self.owner_label(key),
            retry_after_ms=ttl,
            retry_after_is_floor=floor,
            # A CACHED ``quota_blocked`` IS ALWAYS THE OWNER'S OWN STATE (#3108 review
            # r2-m1): the holder-exclusion answer is never remembered (``_from_reply``
            # / ``_detail_from_owner`` return it uncached), so whatever reached the
            # cache was the owner's pool. Rebuilt without the marker, a 0-remainder
            # cached refusal read as the holder's exclusion again and a concurrent
            # ask rendered the borrower's own cooldowns for it.
            reason=QUOTA_REASON_POOL_DEPLETED if status == "quota_blocked" else "",
        )

    def _remember(self, key: str, error: BrokerError) -> None:
        if error.code in _HOLDABLE_REFUSALS and self.state.held_policy_refusal(
            key, error.owner_device or self.owner_of(key)
        ):
            # Connectivity cannot erase an authenticated refusal, including after
            # its retry-cache TTL. Publish only the final held/policy decision.
            return
        ttl = error.cache_ttl_ms
        if ttl == NO_RETRY:
            ttl = UNSUPPORTED_OBSERVATION_MS
        self.state.observe(
            key,
            error.code,
            reason="",
            owner_device=error.owner_device or self.owner_of(key),
            retry_after_ms=ttl,
            # WHAT THE OWNER SAID, IF ANYTHING (review round 2, M2): ``ttl`` above
            # is this device's cache lifetime and may be its own default; only a
            # value that travelled on the refusal may ever be shown back as a
            # countdown, so the two are recorded separately.
            owner_retry_after_ms=error.retry_after_ms,
        )
        from local_operator.network.credentials.effective import note_resolution

        # A transport observation is not yet the resolution: this process may
        # still use the owner's valid held lease. Publishing here made the real
        # session host emit a false outage before that decision was made.
        if error.code not in _HOLDABLE_REFUSALS:
            note_resolution(self.root, key, error)
        self._remember_evidence(key, error)
        self._save_state()

    def forget_refusal(self, key: str) -> None:
        """Forget THIS device's cached "no" for ``key``, so the next ask dials.

        WHAT THIS IS FOR, and what it is NOT. It removes one observation from
        ``placement.state.json`` — the document that records what THIS device last
        learned about a key. It does NOT re-enable a borrow that the owner refuses:
        a provider borrow is re-resolved with the owner on every request, so a
        repaired login is picked up by the next turn anyway, and a withdrawn share is
        refused again on the next dial whether or not this memo exists.

        The two things the observation DOES change, which is why a repair clears it:

        * ``request_grant_sync`` short-circuits on a cached refusal of ONE code,
          ``quota_blocked`` (the owner-stated backpressure). A quota verdict whose
          window has passed must not keep answering for an owner that would serve now;
        * every status surface reads this document. Left in place it keeps reporting
          the failure the operator has just fixed, which is the "why is it still
          broken" the repair path exists to end.

        CALLED ONLY WHERE A REPAIR IS PLAUSIBLE — the failure that makes it so is a
        bearer the provider refused (``MeshAwareAuthStore.rotate_sibling`` and
        ``mcp_bearer``'s 401 arm), never an ordinary resolve: clearing it per call
        would rebuild the ask-per-call the cache exists to prevent. No new state is
        written and nothing is cached: the next borrow asks the owner, which decides
        from its current login and its current grants. A device still not permitted
        to ask stays refused, because that refusal comes back from the owner.

        A POLICY FENCE IS NOT PART OF THIS. ``PlacementState.clear`` keeps the
        ``held_fence*`` fields (``HELD_FENCE_FIELDS``), because a deny's recorded
        diagnosis outlives the retry cache by design — clearing a transient
        ``owner_offline`` must not erase it. The client does nothing extra to
        preserve them; it relies on that one rule rather than restating it.
        """
        try:
            self.state.clear(key)
        except Exception:  # noqa: BLE001 — a cache is best-effort; never fail a turn
            return
        self._save_state()

    def _bounded(self, grant: Grant) -> Grant:
        """``grant`` with its expiry capped at THIS device's ``now + grant_ttl_s``.

        The revoke receipt promises "a grant already lent is dropped within
        ``grant_ttl_s``", and until now only the OWNER enforced it (``min(token_exp,
        now + ttl)`` in ``owner.py``) — so an owner clock far ahead, or an owner that
        simply sent ``10**18``, got a bearer cached as live for years (review round 4,
        R4-m1). The borrower enforces its own half. Capping at the borrower's clock
        can only make a grant END EARLIER, which costs one re-ask, never a longer life.
        """
        from local_operator.network.credentials import grant_ttl_s

        ceiling_ms = int(time.time() * 1000) + int(grant_ttl_s(self.root) * 1000)
        if grant.grant_expires_at_ms > ceiling_ms:
            grant.grant_expires_at_ms = ceiling_ms
        return grant

    def _remember_grant(self, key: str, grant: Grant) -> None:
        from local_operator.network.credentials.effective import note_resolution

        note_resolution(self.root, key, grant)
        self.state.note_grant(key, grant.grant_id, owner_device=grant.credential_ref.owner_device)
        # THE DURABLE HALF OF A SUCCESSFUL BORROW. The grant itself lives only in
        # memory (``GrantCache``), so without this row a restarted session has no
        # way to say "this server works because <owner> signed in" — it would read
        # as a server nobody has ever signed in to. Writing ``synced`` also
        # REPLACES an older ``expired``/``unauthorized``, which is the "clear the
        # entry on success" half: the failure it described is no longer true.
        mcp_state.record(
            self.network_id,
            key,
            state="synced",
            owner_device=grant.credential_ref.owner_device,
            owner_name=grant.credential_ref.owner_device_name,
            root=self.root,
        )
        self._save_state()

    def _remember_evidence(self, key: str, error: BrokerError) -> None:
        """Record WHY a refusal happened, in the durable, value-free vocabulary.

        A broker code this build does not classify records NOTHING: it overwrote a
        precise older answer with a vague new one otherwise, and the status surface
        treats absent evidence as "ask the stores", which is honest.
        """
        state = mcp_state.state_for_code(error.code)
        if not state:
            return
        if state == "revoked" and not self._has_prior_claim(key):
            # A COLD "not a holder" IS NOT A REVOCATION (requirement-6 inversion).
            # A server that was just mirrored, or one whose share has not arrived,
            # answers ``not_a_holder`` before any dial because ``should_borrow`` is
            # false. Recording ``revoked`` there made the status say access had
            # been withdrawn for a server nobody ever shared — the exact opposite
            # of ``not_yet_synced`` / ``needs_operator_login``. Only a device that
            # HAD a claim (a cached grant or an existing evidence row) can lose it.
            return
        mcp_state.record(
            self.network_id,
            key,
            state=state,
            code=error.code,
            owner_device=error.owner_device or self.owner_of(key),
            owner_name=error.owner_device_name or self.owner_label(key),
            root=self.root,
        )

    def _has_prior_claim(self, key: str) -> bool:
        """Whether this device ever held a claim on ``key``: a grant or evidence."""
        if self.grants.peek(key, "") is not None:
            return True
        try:
            return bool(mcp_state.load(self.network_id, self.root).get(key))
        except Exception:  # noqa: BLE001 — unknown is not a claim
            return False

    def _save_state(self) -> None:
        try:
            self.state.save()
        except OSError:
            # A cache that cannot be written is still a cache: the observation lives
            # in this object for the life of the process, and failing a borrow because
            # the refusal cache was not persisted would trade a real capability for a
            # memo. Read-only config dirs are legitimate (a container, a test root).
            pass
        else:
            with self._state_lock:
                self._state_stamp = self._stat_state()

    # -- leg 1: the runtime asks its own relay ------------------------------

    def request_grant_sync(
        self,
        key: str,
        *,
        session_id: str = "",
        model_id: str = "",
        force_refresh: bool = False,
        provider: str = "",
        affinity: tuple[int, int] | None = None,
        demote_credential_ids: Collection[int] = (),
    ) -> Grant | BrokerError:
        """Borrow a bearer for ``key``. Blocking; call from a thread.

        ``affinity`` is this session's warm-account anchor ``(owner row id, row
        created_at)`` and ``demote_credential_ids`` the owner rows this session just
        saw fail server-side; both come from ``MeshAwareAuthStore`` and only steer
        the owner's choice (``owner.grant`` step 2e).

        THE ORDER IS THE CONTRACT: a live grant, then a cached refusal, then the
        dial. Checking the cache before the grant would serve a stale "no" over a
        bearer this process is holding; checking it after the dial would defeat it.
        """
        label = provider or key
        # A grant TTL is not an upstream credential generation (0385927ab94), so a
        # live grant alone never licenses skipping the owner: rotation/logout must
        # not hide behind it. ``_serve_cached`` reuses one only while the OWNER's
        # announced lifecycle generation for this key is unchanged since before the
        # dial that fetched it, for at most ``grant_ttl_s/2``; an uncovered owner
        # (no lifecycle stamps, link without ``credential-lifecycle-v1``) still
        # re-resolves on every request. Refusal observations are for diagnosis, not
        # a login-negative cache: the owner may have signed in since the last
        # request. Forced refresh therefore always reaches the sole refresh owner.
        if not self.should_borrow(key):
            self.grants.drop(key)
            return BrokerError(
                code="not_a_holder",
                key=key,
                owner_device=self.owner_of(key),
                owner_device_name=self.owner_label(key),
                message="",
            )
        # Quota is owner-stated backpressure, not a missing-login negative
        # cache. Respect its interval except for an explicit refresh.
        if not force_refresh:
            # Installation grants are freshly minted, repository-narrowed forge
            # tokens, NOT a provider login generation. Preserve their established
            # helper TTL window; current holder permission was checked above.
            # Provider OAuth/API access must still re-resolve its sole owner.
            from local_operator.network.credentials.github import is_github_key

            if is_github_key(key):
                cached = self.grants.get(key, session_id)
                if cached is not None:
                    return cached
            else:
                reused = self._serve_cached(
                    key,
                    session_id,
                    model_id=model_id,
                    selection=_selection_fields(affinity, demote_credential_ids),
                )
                if reused is not None:
                    return reused
            refusal = self.cached_refusal(key)
            if refusal is not None and refusal.code == "quota_blocked":
                return self._render(refusal, key, label)
        # THE ROWS THIS DEVICE SAW ANSWER A USAGE LIMIT, hidden from this ask only
        # (see ``PlacementState.cool_down``). An explicit refresh still carries
        # them: a forced refresh is a repair of the bearer, not a request to be
        # served by a row the provider said is spent.
        # What this device last heard about ``key``, read BEFORE the dial overwrites
        # it: a non-transport refusal recorded by ANOTHER process (another session,
        # the relay's own pull) is an owner answer this process's memory never saw,
        # and it must keep a held grant from being served (:meth:`_held_or`).
        prior_status = self.state.status(key)
        request_stamp = self._capture_request(key)
        resting = self.state.cooldowns(key, model_id, owner_device=self.owner_of(key))
        excluded = sorted({int(row["credential_id"]) for row in resting})
        try:
            from local_operator.network import store
            from local_operator.network.relay import control_request

            record = store.find_own_relay(self.root)
            if record is None:
                return self._held_or(
                    self._offline(key, label),
                    key,
                    session_id,
                    force_refresh=force_refresh,
                    prior_status=prior_status,
                )
            extra: dict[str, Any] = {}
            if excluded:
                # Only when there is something to hide, so the frame an older relay
                # or owner reads is byte-identical to the one it always read.
                extra["exclude_credential_ids"] = excluded
            # The same only-when-present rule for the selection inputs.
            extra.update(_selection_fields(affinity, demote_credential_ids))
            reply = control_request(
                record,
                "credential_grant",
                timeout=self.deadline_s,
                # ``credential_key``, never ``key``: the control socket's FIRST frame
                # already uses ``key`` for the control key (``control_request`` sends
                # it), and a second field of the same name in the op frame is a
                # collision waiting for a reader to misread.
                credential_key=key,
                provider=label,
                session_id=session_id,
                model_id=model_id,
                force_refresh=bool(force_refresh),
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 — a broker bug must not fail a turn
            return BrokerError(
                code="internal",
                key=key,
                owner_device=self.owner_of(key),
                owner_device_name=self.owner_label(key),
                message=exc.__class__.__name__,
            )
        outcome = self._from_reply(
            reply,
            key,
            label,
            session_id,
            resting=resting,
            request_stamp=request_stamp,
            selection=_selection_fields(affinity, demote_credential_ids),
        )
        if isinstance(outcome, BrokerError):
            return self._held_or(
                outcome, key, session_id, force_refresh=force_refresh, prior_status=prior_status
            )
        return outcome

    def _policy(self) -> Any:
        from local_operator.network.credentials.held_policy import HeldPolicyStore

        return HeldPolicyStore(self.network_id, self.self_device, self.root)

    def _placement_authority(self, key: str) -> Any:
        entry = self.entry(key)
        holder = entry.holder(self.self_device) if entry else None
        if entry is None or holder is None:
            return None
        return (
            entry.owner_device,
            entry.provider,
            entry.kind,
            entry.doc_rev,
            holder.scope,
            holder.granted_at,
            holder.granted_by,
        )

    def _capture_request(self, key: str) -> Any:
        from local_operator.network.credentials.held_policy import PolicyUnknown

        try:
            policy = self._policy()
            identity = (policy.root, policy.network_id, policy.recipient)
            # Evaluate genesis eligibility UNDER the store's shared locks: a
            # concurrent capture may have validated this identity while we wait.
            with policy.transaction(
                initialize=lambda: identity not in self._held_policy_established
            ) as document:
                self._held_policy_established.add(identity)
                if document["membership_denied"] and self._self_membership_active():
                    # This only permits a later loss to advance its deny counter;
                    # it NEVER restamps an old grant or resets that counter.
                    document["membership_denied"] = False
                return (
                    policy.stamp(document, self.owner_of(key), key),
                    self._placement_authority(key),
                    # Taken BEFORE the dial, like the policy stamp: a logout or
                    # rotation announced while the owner is answering must leave
                    # the stored fence older than the next read's, so the grant
                    # this dial returns is never reused across that change.
                    self._lifecycle_fence(key),
                )
        except PolicyUnknown:
            return None

    def _lifecycle_fence(self, key: str) -> frozenset[tuple[str, tuple[int, int], str]]:
        """The owner's announced lifecycle for ``key``, as a comparable snapshot.

        Every ``(scope, (login_chain_revision, material_generation), state)`` this
        device has observed for ``(owner_of(key), key)``. ``material_generation``
        moves on the owner's login, logout and row rotation (``lifecycle.py``), and
        a withdraw sets a restrictive state, so an unchanged fence is the evidence
        that the upstream credential a cached grant came from is still the current
        one — the evidence whose absence made 0385927ab94 stop reusing grants.

        EMPTY MEANS "NOT COVERED" AND DISABLES REUSE: no observation (an owner whose
        logins carry no lifecycle stamp, a link that never negotiated
        ``credential-lifecycle-v1``), an unprovable file (``needs_baseline``) or a
        pending restriction. Those keys keep re-resolving on every request.
        """
        from local_operator.network.credentials import lifecycle

        owner = self.owner_of(key)
        if not owner:
            return frozenset()
        try:
            life = lifecycle.LifecycleStore(self.network_id, self.root)
            if life.restriction_pending():
                return frozenset()
            document = life.load()
        except Exception:  # noqa: BLE001 — unknown evidence is no evidence
            return frozenset()
        if document.needs_baseline:
            return frozenset()
        return frozenset(
            (scope, high_water, document.observation_states.get((owner, row_key, scope), ""))
            for (row_owner, row_key, scope), high_water in document.high_water.items()
            if row_owner == owner and row_key == key
        )

    def _hold_admission(self, policy: Any, document: Any, key: str) -> tuple[str, Any]:
        """``(verdict, current policy stamp)`` for serving a grant without the owner.

        THE ONE ADMISSIBILITY CHECK both no-dial serves run inside the held-policy
        transaction: the owner-unreachable hold (:meth:`_held_or`) and the hot read
        (:meth:`_serve_cached`). Sharing it is what keeps the two from drifting —
        a fence added to one is a fence on both. ``verdict`` is ``""`` when the
        grant may be considered, :data:`_MEMBERSHIP_LOST`, or a refusal code.
        """
        stamp = policy.stamp(document, self.owner_of(key), key)
        restriction = self._lifecycle_restriction(key)
        # The final membership read MUST use the same invalidation helper as the
        # first, before any successful hold is visible.
        if not self._self_membership_active():
            return _MEMBERSHIP_LOST, stamp
        if not self.should_borrow(key) or restriction:
            return restriction or "not_a_holder", stamp
        return "", stamp

    def _serve_cached(
        self, key: str, session_id: str, *, model_id: str, selection: dict[str, Any]
    ) -> Grant | BrokerError | None:
        """A held provider grant reused WITHOUT dialling the owner, or ``None`` to dial.

        WHY: every provider resolve used to dial both relay legs, one owner round
        trip per model call, and an outage cost each call the full failed dial.
        Approved by the operator 2026-11-07 (C5): reuse bounds revocation on an
        honest borrower to about one placement/lifecycle tick (15 s), at most
        ``grant_ttl_s/2`` when an update is lost — inside the ``grant_ttl_s``
        that ``mesh-credentials.md`` §3.7 already promises.

        CONSTRAINTS. This serves only an access token the owner already sent: no
        refresh token, no token POST, no new authority. ``None`` (dial, today's
        behaviour) unless ALL hold:

        * a provider key — MCP, github and device-bound keys keep their own paths;
        * the slot is outside the re-ask margin (:meth:`GrantCache.get`), younger
          than ``grant_ttl_s/2`` (the soft refresh that keeps a later
          ``held_offline`` window at least that long), and was stored with a
          non-empty lifecycle fence;
        * the selection inputs (``affinity``/``demote_credential_ids``) equal the
          dial's, so a new demotion is never masked;
        * the lent row is not cooling for ``model_id`` — another process's 429
          reaches this one only through ``placement.state.json``;
        * :meth:`_hold_admission` passes and the policy stamp is current;
        * the lifecycle fence equals the one taken before the fetching dial.

        A lost membership is answered here, exactly as both other paths would;
        every other failed check falls through to the dial, whose outcome is
        today's. Revocation needs nothing here: ``placement()`` clears the cache
        when the document changes, and ``should_borrow`` ran just before.
        """
        from local_operator.network.credentials import grant_ttl_s
        from local_operator.network.credentials.effective import note_resolution
        from local_operator.network.credentials.held_policy import PolicyUnknown

        if is_mcp_key(key) or is_device_bound_provider(key):
            return None
        held = self.grants.get(key, session_id)
        reuse = self.grants.reuse_stamp(key, session_id)
        if held is None or reuse is None:
            return None
        stored_at, fence, stored_selection = reuse
        if time.monotonic() - stored_at >= grant_ttl_s(self.root) / 2:
            return None
        if stored_selection != selection:
            return None
        owner = self.owner_of(key)
        if held.credential_ref.owner_device != owner or any(
            int(row.get("credential_id") or 0) == held.credential_ref.credential_id
            for row in self.state.cooldowns(key, model_id, owner_device=owner)
        ):
            return None
        try:
            policy = self._policy()
            with policy.transaction() as document:
                verdict, stamp = self._hold_admission(policy, document, key)
                if verdict == _MEMBERSHIP_LOST:
                    lost = self._invalidate_membership_scope(key, document)
                    note_resolution(self.root, key, lost)
                    return lost
                if verdict:
                    self.grants.drop(key)
                    return None
                if self.grants.invalidate_policy(key, session_id, stamp):
                    return None
                if self._lifecycle_fence(key) != fence:
                    # Dial, but KEEP the slot: if the owner is unreachable the
                    # dial ends in ``_held_or``, whose ``held_offline`` contract
                    # predates this read and must not shrink because of it. A
                    # successful dial replaces the slot (and its fence).
                    return None
                if self.grants.peek(key, session_id) is not held:
                    return None  # replaced or dropped by a concurrent resolve
        except PolicyUnknown:
            return None
        return held

    def _lifecycle_restriction(self, key: str) -> str:
        from local_operator.network.credentials import lifecycle

        life = lifecycle.LifecycleStore(self.network_id, self.root)
        document = life.load()
        if life.restriction_pending() or document.needs_baseline:
            return "consent_unverifiable"
        if any(
            owner == self.owner_of(key)
            and row_key == key
            and state in lifecycle._RESTRICTIVE_STATES
            for (owner, row_key, _scope), state in document.observation_states.items()
        ):
            return "consent_withdrawn"
        return ""

    def _policy_error(self, key: str, code: str) -> BrokerError:
        return self._render(
            BrokerError(
                code=code,
                key=key,
                owner_device=self.owner_of(key),
                owner_device_name=self.owner_label(key),
            ),
            key,
            key,
        )

    def _invalidate_membership_scope(self, key: str, document: Any = None) -> BrokerError:
        """The SAME persistent deny operation at both membership checkpoints.

        A network generation fences OTHER clients/processes too. Cache eviction
        is namespace-wide but leaves locally owned contexts intact. No retry
        snapshot, rejoin or transport success can license those old stamps.
        """
        policy = self._policy()
        if document is None:
            with policy.transaction() as row:
                policy.deny_network(row)
                self.grants.drop_borrowed(self.self_device)
        else:
            policy.deny_network(document)
            self.grants.drop_borrowed(self.self_device)
        return self._policy_error(key, "not_a_holder")

    def _adopt_online(
        self,
        grant: Grant,
        key: str,
        session_id: str,
        request_stamp: Any,
        *,
        cache: bool = True,
        selection: Any = None,
    ) -> Grant | BrokerError:
        """CAS before ANY positive cache/status/publication, on BOTH relay legs.

        Only an owner-verified response for this dispatch generation can tag its
        new slot. A late reply never adopts a deny discovered on completion.
        No network call or placement mutation runs under this transaction.
        """
        from local_operator.network.credentials.effective import note_resolution
        from local_operator.network.credentials.held_policy import PolicyUnknown

        denied = None
        try:
            policy = self._policy()
            with policy.transaction() as document:
                if not self._self_membership_active():
                    denied = self._invalidate_membership_scope(key, document)
                elif (
                    request_stamp is None
                    or request_stamp[0] != policy.stamp(document, self.owner_of(key), key)
                    or request_stamp[1] != self._placement_authority(key)
                ):
                    denied = self._policy_error(
                        key,
                        policy.code(document, self.owner_of(key), key) or "consent_unverifiable",
                    )
                elif (
                    not self.should_borrow(key)
                    or grant.credential_ref.owner_device != request_stamp[0].owner
                    or self.owner_of(key) != request_stamp[0].owner
                    or not grant.usable_at()
                ):
                    denied = self._policy_error(key, "not_a_holder")
                elif restriction := self._lifecycle_restriction(key):
                    denied = self._policy_error(key, restriction)
                else:
                    # Fence -> cache lock order. Success changes only the retry
                    # diagnosis; durable deny generations NEVER reset on success.
                    if cache:
                        # The pre-dial lifecycle fence and the selection inputs
                        # are what ``_serve_cached`` must find unchanged later.
                        self.grants.put(
                            key,
                            session_id,
                            grant,
                            policy_fence=request_stamp[0],
                            lifecycle_fence=(
                                request_stamp[2]
                                if selection is not None and len(request_stamp) > 2
                                else None
                            ),
                            selection=selection,
                        )
                    self._remember_grant(key, grant)
                    return grant
        except PolicyUnknown:
            denied = self._policy_error(key, "consent_unverifiable")
        self.grants.drop(key)
        assert denied is not None
        note_resolution(self.root, key, denied)
        return denied

    def _held_or(
        self,
        refusal: BrokerError,
        key: str,
        session_id: str,
        *,
        force_refresh: bool,
        prior_status: str = "",
    ) -> Grant | BrokerError:
        """A CURRENT immutable stamp + original owner lease, never a retry-cache vote."""
        from local_operator.network.credentials.effective import note_resolution
        from local_operator.network.credentials.github import is_github_key
        from local_operator.network.credentials.held_policy import PolicyUnknown

        if is_mcp_key(key) or is_device_bound_provider(key) or is_github_key(key):
            note_resolution(self.root, key, refusal)
            return refusal
        if refusal.code not in _HOLDABLE_REFUSALS:
            if refusal.code not in _TRANSIENT_REFUSALS:
                self.grants.drop(key)
            return refusal
        outcome: Grant | BrokerError = refusal
        try:
            if not self._self_membership_active():
                outcome = self._invalidate_membership_scope(key)
            else:
                policy = self._policy()
                with policy.transaction() as document:
                    verdict, stamp = self._hold_admission(policy, document, key)
                    if verdict == _MEMBERSHIP_LOST:
                        outcome = self._invalidate_membership_scope(key, document)
                    elif verdict:
                        code = verdict
                        policy.deny(document, self.owner_of(key), key, code)
                        self.grants.drop(key)
                        outcome = self._policy_error(key, code)
                    else:
                        stale = self.grants.invalidate_policy(key, session_id, stamp)
                        held = self.grants.peek(key, session_id)
                        if stale or held is None:
                            code = policy.code(document, self.owner_of(key), key)
                            if code:
                                outcome = self._policy_error(key, code)
                        elif not held.usable_at():
                            self.grants.drop(key, session_id)
                        elif not force_refresh:
                            outcome = dataclasses.replace(held, held_offline=True)
                            # Positive publication is part of this linearized
                            # read, not a callback after releasing the deny lock.
                            note_resolution(self.root, key, outcome)
                            return outcome
        except PolicyUnknown:
            self.grants.drop(key)
            outcome = self._policy_error(key, "consent_unverifiable")
        # Commit a discovered deny BEFORE publication. The successful path never
        # publishes an intermediate owner_unavailable or changes grant clocks.
        note_resolution(self.root, key, outcome)
        return outcome

    def _resting_refusal(self, key: str, label: str, resting: list[dict[str, Any]]) -> BrokerError:
        """``quota_blocked`` built from this device's own cooldowns. Not cached.

        Not cached in the refusal file on purpose: the cooldowns already ARE the
        record, and a cached refusal would outlive a cooldown the operator clears
        (or one that ends) by its own TTL rather than by the fact.
        """
        now = time.time()
        remaining_ms = max(0, int((min(float(r["until"]) for r in resting) - now) * 1000))
        accounts = tuple(
            dict.fromkeys(str(r.get("label") or "") for r in resting if r.get("label"))
        )
        error = BrokerError(
            code="quota_blocked",
            key=key,
            owner_device=self.owner_of(key),
            owner_device_name=self.owner_label(key),
            retry_after_ms=remaining_ms,
        )
        error.message = render_broker_error(
            error,
            key=key,
            provider=label,
            owner_name=self.owner_label(key),
            last_seen_s=self.owner_last_seen_s(error.owner_device),
            accounts=accounts,
        )
        return error

    # -- leg 2: the relay asks the owner ------------------------------------

    def serve_grant(
        self,
        key: str,
        *,
        provider: str = "",
        session_id: str = "",
        model_id: str = "",
        force_refresh: bool = False,
        exclude_credential_ids: list[int] | None = None,
        affinity: tuple[int, int] | None = None,
        demote_credential_ids: Collection[int] = (),
    ) -> dict[str, Any]:
        """Answer a leg-1 ``credential_grant``. Returns a ``detail`` object.

        THE OWNER IS ASKED DIRECTLY WHENEVER THIS DEVICE CAN REACH IT ITSELF
        (``_dial_owner``), and the runtime's own relay otherwise. Both paths end at
        the same handler on the same device, so this is a routing choice and not a
        second protocol. Direct is preferred because a broker request may sit
        through a provider refresh bounded at 60 s (build plan §0 finding 4): one
        hop fewer is one fewer link whose reader is occupied for that long, and if
        the other hop is DOWN the borrow still succeeds.
        """
        label = provider or key
        # Leg 2 shares the repair contract: missing-login observations cannot
        # hide a new login; only owner-stated quota backpressure may short circuit.
        # Validate current permission before trusting even that observation.
        refusal = (
            self.cached_refusal(key) if self.should_borrow(key) and not force_refresh else None
        )
        if refusal is not None and refusal.code == "quota_blocked":
            return self._detail_error(self._render(refusal, key, label))
        if not self.should_borrow(key):
            return self._detail_error(
                BrokerError(
                    code="not_a_holder",
                    key=key,
                    owner_device=self.owner_of(key),
                    owner_device_name=self.owner_label(key),
                )
            )
        owner = self.owner_of(key)
        request_stamp = self._capture_request(key)
        frame = {
            "op": PEER_BROKER_OP,
            "kind": "grant",
            "network_id": self.network_id,
            "from_device": self.self_device,
            "from_device_name": self._self_label(),
            "key": key,
            "provider": label,
            "for_session": session_id,
            "model_id": model_id,
            "force_refresh": bool(force_refresh),
        }
        if exclude_credential_ids:
            # The borrower's resting rows (``PlacementState.cool_down``). Present only
            # when non-empty, so the frame an older owner reads is unchanged — and an
            # older owner simply ignores it, which costs one more 429, never a wrong
            # account (it is a narrowing, never a widening).
            frame["exclude_credential_ids"] = list(exclude_credential_ids)
        frame.update(_selection_fields(affinity, demote_credential_ids))
        direct = self._ask_owner_directly(owner, frame)
        if direct is not None:
            return self._detail_from_owner(
                direct,
                key,
                label,
                excluded=bool(exclude_credential_ids),
                request_stamp=request_stamp,
            )
        return self._detail_error(self._render(self._offline(key, label, owner=owner), key, label))

    def fetch_catalogue_metadata(
        self, key: str, *, provider: str = "openai", ttl: int = 60, validator: str = ""
    ) -> dict[str, Any]:
        """Slice B's metadata seam; no token, inference grant, or cache mutation.

        Call from a worker thread. Consumers must keep scope/incarnation/lifecycle
        and TTL together and recheck permission before using cached metadata.
        """
        from local_operator.network.credentials.catalogue import (
            CAPABILITY,
            ERROR_CODES,
            validate_response,
        )
        from local_operator.network.types import MeshRefusal

        if key != "openai" or provider != "openai" or type(ttl) is not int or not 1 <= ttl <= 300:
            raise ValueError("unsupported catalogue request")
        if not self.should_borrow(key):
            raise MeshRefusal("not_a_holder", "catalogue metadata is not shared")
        owner = self.owner_of(key)
        if self._relay is None:
            from local_operator.network import store
            from local_operator.network.relay import control_request

            record = store.find_own_relay(self.root)
            if record is None:
                raise MeshRefusal("owner_offline", "catalogue relay unavailable")
            reply = control_request(
                record,
                "credential_catalogue",
                timeout=self.deadline_s,
                credential_key=key,
                provider=provider,
                ttl=ttl,
                validator=validator,
            )
        else:
            reply = self._ask_owner_directly(
                owner,
                {
                    "op": PEER_BROKER_OP,
                    "kind": "catalogue",
                    "network_id": self.network_id,
                    "from_device": self.self_device,
                    "key": key,
                    "provider": provider,
                    "ttl": ttl,
                    "validator": validator,
                },
                required_capability=CAPABILITY,
            )
        detail = reply.get("detail") if isinstance(reply, dict) else None
        if not isinstance(detail, dict):
            raise MeshRefusal("catalogue_unavailable", "catalogue metadata unavailable")
        if detail.get("kind") == "error":
            # Never forward an untrusted peer's free-form error text or auth data.
            code = detail.get("code")
            raise MeshRefusal(
                code if code in ERROR_CODES else "catalogue_unavailable",
                "catalogue metadata unavailable",
            )
        if not self.should_borrow(key) or self.owner_of(key) != owner:
            raise MeshRefusal("not_a_holder", "catalogue sharing changed")
        return validate_response(detail, network=self.network_id, owner=owner, key=key)

    def _ask_owner_directly(
        self, owner: str, frame: dict[str, Any], *, required_capability: str = ""
    ) -> dict[str, Any] | None:
        """Run ``net_broker`` over a link to ``owner``, dialling if there is none.

        ``None`` means "this device could not get an answer from the owner", which
        the caller turns into ``owner_offline``. A broker REFUSAL is not ``None`` —
        an ``assumed`` refusal is an answer, and conflating the two is how a
        "you are not a holder" becomes "the device is asleep".
        """
        if not owner:
            return None
        # ``_ensure_link_with_reason``, which returns ``(link, reason)``. The first
        # version called ``_ensure_link`` — which returns a bare link — and unpacked
        # it as a pair, so EVERY leg-2 ask on a real relay raised ``TypeError`` and
        # answered ``internal``, including the owner-unreachable case that must
        # answer ``owner_offline`` (found building the running-relay test F7 asked for).
        ensure = getattr(self._relay, "_ensure_link_with_reason", None)
        next_req = getattr(self._relay, "_next_relay_req", None)
        if ensure is None or next_req is None:
            # The relay's one implementation of "reach this peer, dialling if
            # needed" — with its own endpoint probe, its own budget and its own
            # audit records. Re-implementing a dialer here would be a second
            # dialer with a second idea of which address of a member row to use,
            # which is the bug that function's probe exists to prevent. A relay
            # without it is a build mismatch, and it fails LOUDLY rather than by
            # quietly reporting the owner offline.
            return {
                "op": "ack",
                "detail": {
                    "kind": "error",
                    "code": "internal",
                    "message": "this relay build cannot dial peers for credential brokering",
                },
            }
        try:
            link, _reason = ensure(owner)
        except Exception as exc:  # noqa: BLE001 — reported as the owner being unreachable
            return {
                "op": "ack",
                "detail": {"kind": "error", "code": "internal", "message": exc.__class__.__name__},
            }
        if link is None:
            return None
        if required_capability and (
            required_capability not in getattr(link, "local_capabilities", ())
            or required_capability not in getattr(link, "capabilities", ())
        ):
            return {"op": "ack", "detail": {"kind": "error", "code": "catalogue_unsupported"}}
        # EVERY LEG-2 FRAME CARRIES A ``req`` MINTED FROM THE RELAY'S OWN COUNTER (QA
        # round 1, Q2). ``PeerLink.request`` registers a waiter only for a frame that
        # has one; without it the frame is SENT, the call returns ``None`` at once, and
        # the owner's answer arrives as a stray reply nobody is waiting for. That is
        # how every borrow, pull and report on the shipped build read as "owner
        # offline" while the owner really did refresh and serve the grant. The
        # relay's counter, not a private one, because the waiter table is keyed by
        # ``(link, req)`` and a second counter on the same link could collide with the
        # relay's own requests (``mobility``/``sync`` mint theirs the same way).
        frame = {**frame, "req": next_req()}
        try:
            reply = link.request(frame)
        except Exception:  # noqa: BLE001 — the own-link refusal is a bug, not a state
            return None
        return reply if isinstance(reply, dict) else None

    # -- reports ------------------------------------------------------------

    def report_sync(
        self,
        key: str,
        *,
        kind: str,
        session_id: str = "",
        model_id: str = "",
        retry_after_ms: int = 0,
        block_scope: str = "",
    ) -> None:
        """Tell the owner a borrowed bearer failed. BLOCKING, best effort.

        ``kind`` is what the PROVIDER said, never a verdict about the credential:
        ``quota`` for a usage limit, ``invalid`` for an ``invalid_grant``,
        ``unauthorized`` for a plain 401 and ``unavailable`` for a provider fault
        (5xx/529). ``block_scope`` carries the family a scoped verdict named, so the
        owner never has to widen it (review round 1, F3). The
        owner decides what that means for its own row, and the whole point of the
        split is that this device cannot decide it — see ``owner.py``'s report arm
        for why routing this into ``rotate_sibling`` would log the operator out of
        every device.

        THE ROW THAT WAS LENT TRAVELS WITH THE REPORT (audit Q5 #4). The owner used
        to resolve a report's target as the provider's FIRST row, so on a multi-row
        owner a quota block or a provoked refresh could land on a row this device
        was never lent. The cached grant's ``credential_ref.credential_id`` is read
        BEFORE the cache is dropped and echoed as ``credential_id``; the owner
        validates it against its own id space and falls back to the first-row pick
        only when the field is absent (an older peer, or no grant left to attribute).
        """
        if not self.should_borrow(key):
            return
        credential_id = self._reported_credential_id(key, session_id)
        if kind == "quota":
            self._cool_lent_row(
                key,
                session_id=session_id,
                credential_id=credential_id,
                model_id=model_id,
                retry_after_ms=retry_after_ms,
                block_scope=block_scope,
            )
        self.grants.drop(key)
        try:
            from local_operator.network import store
            from local_operator.network.relay import control_request

            record = store.find_own_relay(self.root)
            if record is None:
                return
            control_request(
                record,
                "credential_report",
                timeout=self.deadline_s,
                credential_key=key,
                kind=kind,
                session_id=session_id,
                model_id=model_id,
                retry_after_ms=int(retry_after_ms),
                block_scope=block_scope,
                credential_id=credential_id,
            )
        except Exception:  # noqa: BLE001 — a failed report must not raise into a turn
            return

    def _cool_lent_row(
        self,
        key: str,
        *,
        session_id: str,
        credential_id: int,
        model_id: str,
        retry_after_ms: int,
        block_scope: str,
    ) -> None:
        """Rest the LENT row for the window a local row would rest. Best effort.

        The borrower-side half of the local rule (``AuthStore.rotate_sibling``):
        same window, same family scoping, read from
        ``auth_store.failure_block_window``. Read BEFORE the grant cache is dropped,
        like the report's ``credential_id``, because the lent row's id and label
        are only known from that grant. Without an id there is nothing to hide —
        the owner's own first-row pick then stands, exactly as before.
        """
        if credential_id <= 0:
            return
        from local_operator.providers.auth_store import failure_block_window
        from local_operator.session.credential_binding import identity_label_for

        scope, window_ms = failure_block_window(
            retry_after_ms, usage_limited=True, model_id=model_id
        )
        if block_scope:
            # The scope the caller actually observed (``configure``'s Fable-only
            # cap) wins over the one derived from the model, as on the owner.
            scope = block_scope
        label = ""
        for sid in dict.fromkeys((session_id, "")):
            grant = self.grants.peek(key, sid)
            if isinstance(grant, Grant):
                label = identity_label_for(grant.identity)
                break
        try:
            self.state.cool_down(
                key,
                owner_device=self.owner_of(key),
                credential_id=credential_id,
                block_scope=scope,
                until=time.time() + window_ms / 1000.0,
                label=label,
            )
            self._save_state()
        except Exception:  # noqa: BLE001 — a bookkeeping failure must not raise into a turn
            logger.debug("could not record a borrowed row's cooldown", exc_info=True)

    def _reported_credential_id(self, key: str, session_id: str) -> int:
        """The owner's row id from the grant this device was last lent for ``key``.

        A ``peek``, not a ``get``: the failing bearer's grant may be inside the
        re-ask margin, and attributing the report to the row that was actually lent
        matters most exactly then. ``0`` means "nothing to attribute" — no grant was
        cached for this ``(key, session)``, or it was cached before grants carried
        an id at all.
        """
        for sid in dict.fromkeys((session_id, "")):
            grant = self.grants.peek(key, sid)
            if isinstance(grant, Grant) and grant.credential_ref.credential_id:
                return int(grant.credential_ref.credential_id)
        return 0

    def serve_report(self, key: str, *, kind: str, **fields: Any) -> dict[str, Any]:
        """Answer a leg-1 ``credential_report`` (relay side)."""
        owner = self.owner_of(key)
        frame = {
            "op": PEER_BROKER_OP,
            "kind": "report",
            "network_id": self.network_id,
            "from_device": self.self_device,
            "from_device_name": self._self_label(),
            "key": key,
            "failure": kind,
            # ``session_id`` NEVER TRAVELS ON THE PEER FRAME (QA round 1, Q7). The
            # transport's chokepoint reads a top-level ``session_id`` as a claim that
            # the op acts on a session THE OWNER owns (``authorizer._session_scope``)
            # and refuses it before the broker runs, so every real report was
            # dropped at the owner's door. The borrower's session is carried as
            # ``for_session`` — the same name the grant frame uses, and the one the
            # owner's report arm audits.
            **{k: v for k, v in fields.items() if v not in (None, "") and k != "session_id"},
        }
        self.grants.drop(key)
        reply = self._ask_owner_directly(owner, frame)
        detail = (reply or {}).get("detail")
        if not isinstance(detail, dict):
            return {"kind": "error", "code": "owner_offline", "key": key}
        return detail

    def pull_placement(self, only_device: str = "") -> dict[str, Any]:
        """Ask every other active member for its placement document, and merge.

        ``only_device`` narrows the fan-out to ONE member, for the cadence path:
        a member that holds keys owned by one peer needs that peer's document to
        converge, and asking every member on every tick would put a per-member
        round trip on a 15 s floor. The full sweep stays the explicit verb.

        WHY EVERY MEMBER RATHER THAN THE KNOWN OWNERS: a share done for a key this
        device has never heard of leaves no entry here to name the owner, so
        asking only the owners we already know about can never discover the FIRST
        share — which is the whole point of the verb. The cost is one bounded
        ``net_broker`` frame per member, and it is paid on an explicit act (the
        ``credential_placement`` local op or ``lop network credentials``), never on
        the provider path.

        The merge is ``PlacementDocument.merge``, so a member can only ever tell us
        about the rows IT owns; a document that claims a third device's credential
        is dropped there rather than here.

        ``newly_borrowable`` IS THE REPLY'S SECOND ANSWER (audit Q5 #5): the keys
        this device could NOT reach before the pull and can reach after it. It is
        the only honest signal for "a share just arrived that a running session
        cannot see" — ``build_auth_store`` decides plain-vs-mesh at session
        construction, so a key that was not borrowable before this pull cannot have
        been wired into any session that was running while it landed.
        """
        if self.placement is None or not self.network_id:
            # A MISSING DOCUMENT IS ITSELF A LOSS (design §4: access ends at
            # reconcile, not at token expiry). The device may have been removed from
            # the network while this process was running; the grants in memory must
            # not outlive that, so the same reconcile runs here before answering.
            revoked = self.revoke_local_grants()
            missing: dict[str, Any] = {
                "kind": "ack",
                "key": "",
                "changed": [],
                "owners": 0,
                "newly_borrowable": [],
            }
            if revoked:
                missing["revoked"] = revoked
            return missing
        from local_operator.network.types import MeshRefusal

        before_borrowable = set(self._borrowable_now())
        # THE KEYS THAT COULD LOSE ACCESS ON THIS PULL, captured BEFORE the merge:
        # reading the placement property after a change CLEARS every cached grant
        # (the withdrawal rule), and a cleared cache can no longer name what was
        # withdrawn. These names are what the evidence document records.
        held_before = self.grants.keys()
        changed: list[str] = []
        #: ``{device, reason}`` for every member whose document was NOT merged, so a
        #: skipped member is named in the reply rather than silently missing.
        skipped: list[dict[str, str]] = []
        #: Members that answered the pull with a refusal (see the loop below).
        refused_by: list[str] = []
        asked = 0
        members = self._other_active_members()
        if only_device:
            members = [device for device in members if device == only_device]
        from local_operator.network import wire

        for device in members:
            asked += 1
            # A PEER THAT DOES NOT ADVERTISE THE PUSH CAPABILITY MUST NOT BE SENT
            # A PLACEMENT REQUEST EITHER (lifecycle-only peer test): the handshake
            # already says it cannot handle one, so asking would put a frame on
            # the wire that a previous-release peer has no handler for. The link
            # is established first only to READ its capabilities; a failed dial
            # falls through to ``_ask_owner_directly``'s own offline handling.
            ensure = getattr(self._relay, "_ensure_link_with_reason", None)
            if ensure is not None:
                try:
                    link, _why = ensure(device)
                except Exception:  # noqa: BLE001 — an unreachable peer is reported below
                    link = None
                caps = tuple(getattr(link, "capabilities", ()) or ()) if link is not None else ()
                if link is not None and wire.CREDENTIAL_PLACEMENT_PUSH_V1 not in caps:
                    skipped.append({"device": device, "reason": "peer_lacks_placement_push"})
                    continue
            reply = self._ask_owner_directly(
                device,
                {
                    "op": PEER_BROKER_OP,
                    "kind": "placement",
                    "want": "pull",
                    "network_id": self.network_id,
                    "from_device": self.self_device,
                    "from_device_name": self._self_label(),
                },
            )
            detail = (reply or {}).get("detail")
            document = detail.get("document") if isinstance(detail, dict) else None
            if (
                isinstance(reply, dict)
                and not isinstance(document, dict)
                and (
                    reply.get("op") == "error"
                    or (isinstance(detail, dict) and detail.get("kind") == "error")
                )
            ):
                # THE OWNER ANSWERED AND SAID NO (its authoriser's codeless
                # ``op: error``, or its by-name broker refusal) — a statement by the
                # peer, unlike ``None`` (no answer). Reported apart from ``skipped``
                # because it is not a merge failure: the listing must not warn about
                # an unreadable document, but the cadence must know to back off
                # (``sync._pull_past_stale_self_view``).
                refused_by.append(device)
                continue
            if isinstance(document, dict):
                # AN ANSWERED PULL ENDS A PARKED REFUSAL (review round 1, n1), whoever
                # asked: the explicit listing and onboarding's in-run refresh run this
                # too, and a park left standing after the owner served its document
                # would only delay the cadence's copies for no reason.
                from local_operator.network.credentials import sync as sync_mod

                sync_mod.forget_owner_refusal(self._relay, device)
                # ``device`` is the member this device DIALLED, whose key the
                # handshake verified — the transport's identity, not a frame field.
                # Merged into the FILE under its lock: saving an in-memory copy is
                # what wrote a revoked holder back over a CLI revoke (F2).
                try:
                    changed.extend(
                        merge_from_peer(
                            self.network_id,
                            document,
                            from_device=device,
                            self_device=self.self_device,
                            root=self.root,
                        )
                    )
                except (OSError, MeshRefusal):
                    # ``MeshRefusal`` is the placement lock's bounded timeout (another
                    # writer held it for 10 s). It is NOT an ``OSError``, and letting it
                    # out aborted the loop, so one contended merge left every later
                    # member's document unmerged (review round 2, m3). Skipping the
                    # member is the honest outcome: its document is merged on the next
                    # pull, and nothing was written for it this time.
                    skipped.append({"device": device, "reason": "busy"})
                    continue
                except Exception:  # noqa: BLE001 — one member's bytes must not stop the rest
                    # ONE MEMBER'S DOCUMENT IS THAT MEMBER'S PROBLEM (review round 4,
                    # R4-M1). The document is a PAIRED peer's bytes, and a field this
                    # build could not read (a 309-digit ``doc_rev`` raised
                    # ``OverflowError`` before the boundary helper was total) escaped
                    # the loop — so any invited device could stop every borrower from
                    # merging anyone else's document with one bad field. The boundary
                    # helper no longer raises; this is the loop's OWN guarantee, so the
                    # next unforeseen shape refuses that one document by name instead
                    # of silently skipping every member after it. The merge writes
                    # under the lock and only after parsing, so nothing was written.
                    skipped.append({"device": device, "reason": "malformed_document"})
                    continue
        revoked = self.revoke_local_grants(held_before)
        after_borrowable = self._borrowable_now()
        result: dict[str, Any] = {
            "kind": "ack",
            "key": "",
            "changed": sorted(set(changed)),
            "revoked": revoked,
            "owners": asked,
            "newly_borrowable": sorted(
                key
                for key in set(changed)
                if key in after_borrowable and key not in before_borrowable
            ),
        }
        if skipped:
            result["skipped"] = skipped
        if refused_by:
            result["refused_by"] = refused_by
        return result

    def _borrowable_now(self) -> dict[str, Any]:
        """``{key: entry}`` for what this device may borrow, per the file on disk.

        One read of :attr:`placement` (which re-stats), so a pull's before/after
        snapshots see the same reloaded document every other reader does. An entry
        owned by this device is excluded: owning is not borrowing.
        """
        document = self.placement
        if document is None:
            return {}
        return {
            key: entry
            for key, entry in document.entries.items()
            if entry.owner_device != self.self_device and entry.is_holder(self.self_device)
        }

    # -- the node's half of a refresh delegation (design §3.4/§3.7d) --------

    def refresh_take(self, key: str, *, provider: str = "") -> dict[str, Any]:
        """Ask the owner for the delegated material, and install the statement.

        Three properties, each of which is why this is not a second ``grant``:

        * **gated on the CAPABILITY, never on the answer.** An owner on the
          pre-delegation build answers the unknown kind ``internal``, which this
          module reports as ``owner_unreachable``-ish noise; a node would then
          retry forever and the fleet would look flaky (§3.7b, §8.5). So the
          advertised string is required before the frame is sent, and its absence
          answers ``unsupported`` — what an older peer's silence already gets.
        * **the statement is verified and INSTALLED here**, not carried in memory:
          :func:`refresh.install` re-verifies against the operator anchor and
          writes the document the pre-POST fence reads, so a restart cannot lose a
          live delegation while the material stays behind (§3.7d).
        * **the material lands in THIS device's own row** with ``mesh_origin``
          provenance, which is what makes the ordinary refresh path apply — "the
          node's row is a normal row" (§3.7e) is true only because this is how the
          row is written.
        """
        from local_operator.network.credentials import refresh as refresh_mod
        from local_operator.network.credentials.types import (
            PEER_BROKER_OP,
            REFRESH_CAP_STRING,
        )

        owner = self.owner_of(key)
        entry = self.entry(key)
        label = provider or (entry.provider if entry else "") or key
        if not owner:
            return self._refresh_refusal(key, "not_a_holder", "no owner is placed for this key")
        reply = self._ask_owner_directly(
            owner,
            {"op": PEER_BROKER_OP, "kind": "refresh_take", "key": key},
            required_capability=REFRESH_CAP_STRING,
        )
        if reply is None:
            return self._refresh_refusal(
                key, "owner_offline", f"{self.owner_label(key) or owner} did not answer"
            )
        detail = reply.get("detail") if isinstance(reply, dict) else None
        if not isinstance(detail, dict):
            return self._refresh_refusal(key, "internal", "the answer carried no detail")
        if detail.get("kind") == "error" or detail.get("ok") is False:
            return self._refresh_refusal(
                key,
                str(detail.get("code") or "internal"),
                str(detail.get("message") or ""),
            )
        statement = str(detail.get("statement") or "")
        token = detail.get("refresh")
        if not statement or not isinstance(token, str) or not token:
            return self._refresh_refusal(
                key, "grant_invalid", "the owner's answer carried no usable material"
            )
        try:
            installed = refresh_mod.install(
                self.network_id,
                self.root,
                recipient=self.self_device,
                statement=statement,
            )
        except (refresh_mod.RefreshRefused, refresh_mod.RefreshUnknown) as exc:
            return self._refresh_refusal(key, "grant_invalid", str(exc))
        # THE ROW'S PROVENANCE COMES FROM THE VERIFIED STATEMENT (R2-2), never from
        # the answer's envelope: ``detail["grant_id"]`` is unsigned, so an answer
        # that omitted or renamed it would land a real delegated row that reads as
        # a sync copy ever after — and a copy is precisely what the fence does not
        # guard. `install` returned the verified grant it wrote, so the row and the
        # document agree by construction.
        verified = refresh_mod.read_statement_grant(
            str(installed.get("statement") or ""), root=self.root
        )
        if verified is None:
            return self._refresh_refusal(
                key, "grant_invalid", "the installed delegation could not be re-read"
            )
        row_written = self._write_delegated_row(
            key,
            owner=owner,
            provider=str(detail.get("provider") or label),
            grant_id=verified.grant_id,
            epoch=int(installed.get("epoch") or 0),
            data=detail,
        )
        if not row_written:
            return self._refresh_refusal(
                key, "internal", "the delegated material could not be recorded on this device"
            )
        self.install_refresh_fence()
        self._audit_refresh(
            "credential.refresh_taken",
            credential_key=key,
            act=self.self_device,
            sub=owner,
            grant_id=verified.grant_id,
            epoch=int(installed.get("epoch") or 0),
        )
        return {
            "ok": True,
            "key": key,
            "grant_id": verified.grant_id,
            "epoch": int(installed.get("epoch") or 0),
            "code": "",
            "message": "",
        }

    def _refresh_refusal(self, key: str, code: str, message: str) -> dict[str, Any]:
        return {"ok": False, "key": key, "code": code, "message": message, "grant_id": ""}

    def _write_delegated_row(
        self,
        key: str,
        *,
        owner: str,
        provider: str,
        grant_id: str,
        epoch: int,
        data: dict[str, Any],
    ) -> bool:
        """Write the node's own row for one delegated key, provenance-marked (§3.7d).

        The provenance is ``sync.MESH_ORIGIN_KEY``'s shape — ``{owner_device, key,
        grant_id, epoch}`` — because that is the shape the lifecycle and reconcile
        readers already understand, and because the fence finds a delegated row by
        it: a row with no ``mesh_origin`` is an ordinary local login and is never
        fenced.

        Written through ``upsert_credential`` like any other login, so the store
        owns the dedupe key and the payload's shape; the transient markers
        (``grant_dead_at``, ``refresh_send_unconfirmed``) are dropped for the
        reason ``sync`` drops them — they describe a LOCAL interaction.
        """
        from local_operator.network.credentials import sync as sync_mod

        payload: dict[str, Any] = {
            "access": data.get("access"),
            "refresh": data.get("refresh"),
            "expires": data.get("expires"),
            "token_type": data.get("token_type") or "Bearer",
            "login_incarnation": str(data.get("login_incarnation") or ""),
            sync_mod.MESH_ORIGIN_KEY: {
                "owner_device": owner,
                "key": key,
                "grant_id": grant_id,
                "epoch": int(epoch),
            },
        }
        if data.get("email"):
            payload["email"] = data.get("email")
        if data.get("account_id"):
            payload["account_id"] = data.get("account_id")
        try:
            store = self._auth_store()
            store.upsert_credential(provider, payload)
        except Exception:  # noqa: BLE001 — a write failure is a refusal, never a crash
            return False
        return True

    def refresh_fence(self, credential_id: int, data: dict[str, Any]) -> str | None:
        """The node's PRE-POST fence (§3.4): ``None`` allows the POST, a sentence refuses.

        Runs immediately before the store arms its send marker, on EVERY attempt —
        not once when the material arrived — and refuses at the first failure, in
        the design's order:

        1. this device's membership is active;
        2. the signed delegation verifies for this device, this key, inside its
           window, with budget left (``refresh.active`` re-verifies against the
           anchor every time; §6 item 3 is encoded as a REFUSAL: an anchor this
           device cannot read verifies nothing, so nothing is presented);
        2b. the network deny stamp (``deny_network`` / ``membership_denied``);
        3. the per-key deny watermark, and ``PolicyUnknown`` is a refusal;
        4. the local lease (unchanged, and taken by the store after this returns).

        A row with no ``mesh_origin`` is an ordinary local login: the fence answers
        ``None`` so the shipped path is untouched (T10/T14).
        """
        from local_operator.network.credentials import refresh as refresh_mod
        from local_operator.network.credentials.held_policy import (
            NONRESTRICTIVE,
            PolicyUnknown,
        )

        origin = (data or {}).get("mesh_origin")
        if not isinstance(origin, dict):
            return None
        key = str(origin.get("key") or "")
        owner = str(origin.get("owner_device") or "")
        if not key or not owner:
            return None
        if not str(origin.get("grant_id") or ""):
            # A COPY's provenance (``sync`` writes owner+key, no grant) is this
            # device's OWN row with its own refresh material: it refreshes on the
            # ordinary path and this fence must not touch it. Only a DELEGATED row
            # carries a grant id, because only a delegation hands over a token.
            return None
        if not self._self_membership_active():
            return (
                "this device is no longer an active member of the network, so the delegated "
                "refresh was not presented"
            )
        try:
            grant = refresh_mod.active(
                self.network_id,
                self.root,
                recipient=self.self_device,
                key=key,
                operator_spki=refresh_mod.anchor_spki_for(self.root),
            )
        except refresh_mod.RefreshUnknown:
            return (
                "the delegation for this login could not be proved on this device, so the "
                "refresh token was not presented"
            )
        if grant is None:
            return (
                "no signed refresh delegation is live for this login, so the refresh token "
                "was not presented"
            )
        if grant.owner_device != owner:
            return (
                "the delegation on this device names a different owner, so the refresh token "
                "was not presented"
            )
        try:
            policy = self._policy()
            # ``initialize=True`` is genesis on a MISSING file only — a deny-only
            # document with no entries, which cannot grant anything. A node that
            # pulled a placement document already has one; a corrupt or
            # uninspectable file still raises PolicyUnknown and still refuses, so
            # "missing evidence is never permission" is unchanged.
            with policy.transaction(initialize=True) as document:
                if document.get("membership_denied"):
                    return (
                        "this device was denied membership of the network, so the delegated "
                        "refresh was not presented"
                    )
                code = policy.code(document, owner, key)
        except PolicyUnknown:
            return (
                "the deny record on this device could not be read, so the delegated refresh "
                "was not presented"
            )
        if code and code not in NONRESTRICTIVE:
            return (
                f"refreshing {key!r} was withdrawn ({code}), so the delegated refresh was not "
                "presented"
            )
        return None

    # -- the outcome half: a POST happened, or the family moved ---------------

    def refresh_posted(self, credential_id: int, data: dict[str, Any]) -> None:
        """Record one successful POST against the delegation's budget (R1-3).

        ``max_refreshes`` was decorative while nothing incremented it: the
        document's ``refreshes_used`` stayed at zero and ``refresh.active`` could
        never refuse on budget. This is the caller, and it is the store's own
        success path — the one place that knows the exchange was accepted.
        """
        from local_operator.network.credentials import refresh as refresh_mod

        origin = (data or {}).get("mesh_origin")
        if not isinstance(origin, dict) or not str(origin.get("grant_id") or ""):
            return
        try:
            document = refresh_mod.load(self.network_id, self.root, recipient=self.self_device)
        except refresh_mod.RefreshUnknown:
            return
        if document is None:
            return
        # ``record_post`` increments the stored epoch itself; the argument is the
        # floor, so passing the current epoch + 1 makes the caller's intent explicit
        # rather than relying on that internal increment.
        refresh_mod.record_post(
            self.network_id,
            self.root,
            recipient=self.self_device,
            epoch=int(document.get("epoch") or 0) + 1,
        )

    def refresh_invalid_grant(self, credential_id: int, data: dict[str, Any]) -> dict[str, Any]:
        """§3.5's node-side stop (R1-5): wipe, fail the delegation, report, never retry.

        Reached ONLY when the store's own race guard passed — i.e. the token the
        provider refused was this device's CURRENT token, so no sibling on this node
        moved the family and the move therefore happened outside the device. That
        is the design's exact condition, and it is why this is not "any 401".

        The three local effects happen before the report, in this order, so a crash
        mid-report cannot leave a retryable delegation behind: the row's material is
        wiped, the document is failed with ``family_moved`` (``active`` is the only
        state that may POST), and only then is the owner told. The return value
        distinguishes the two facts a caller needs — that this device STOOD DOWN,
        and whether the owner actually heard about it — because an owner that cannot
        tell "stopped" from "unreachable" acts on the wrong one.
        """
        from local_operator.network.credentials import refresh as refresh_mod

        origin = (data or {}).get("mesh_origin")
        key = str((origin or {}).get("key") or "")
        grant_id = str((origin or {}).get("grant_id") or "")
        # THE GUARD ITS SIBLINGS ALREADY HAD (R2-1). The store calls this watch for
        # EVERY row once a fence is installed, and a fence is installed on any
        # device that holds a delegation document or a delegated row — so without
        # this check an ordinary local login lost both its secrets and gained a
        # family-moved marker that told the operator to sign in again. "Cannot
        # tell" resolves to STOP (§3.5); it never resolves to destroying a login
        # this delegation was never about.
        #
        # Two conditions, and the second is the one that makes the marker
        # meaningful: the row must be delegation-sourced AND its grant must be the
        # one the document currently holds — otherwise this node has moved on (a
        # new delegation, or none), and stopping would be a stop for a family it
        # no longer owns.
        if not self._owns_delegation(origin, key, grant_id):
            return {"stopped": False, "reported": False, "reason": "not_delegated"}
        self._auth_store().wipe_delegated_material(credential_id)
        try:
            refresh_mod.stop(
                self.network_id,
                self.root,
                recipient=self.self_device,
                state="failed",
                cause="family_moved",
            )
        except refresh_mod.RefreshUnknown:
            pass
        self._audit_refresh(
            "credential.refresh_family_moved",
            credential_key=key,
            act=self.self_device,
            sub=self.owner_of(key),
            grant_id=grant_id,
            cause="family_moved",
        )
        if key:
            report = self.report_family_moved(key)
        else:
            report = {"reported": False, "reason": "no_key"}
        return {"stopped": True, **report}

    def _owns_delegation(self, origin: Any, key: str, grant_id: str) -> bool:
        """Whether this row is the delegation the document on disk currently holds.

        ONE predicate for the watch's two destructive arms, so a future third
        caller cannot re-derive it and forget half. ``grant_id`` is what separates a
        delegated row from a sync COPY (``sync`` writes ``mesh_origin`` without a
        grant), which is the same discriminator the fence uses.
        """
        from local_operator.network.credentials import refresh as refresh_mod

        if not key or not grant_id:
            return False
        try:
            document = refresh_mod.load(self.network_id, self.root, recipient=self.self_device)
        except refresh_mod.RefreshUnknown:
            # Unprovable evidence is not permission to destroy: the fence already
            # refuses every POST in this state, so there is nothing here to stop.
            return False
        if document is None:
            return False
        try:
            held = refresh_mod.read_statement_grant(
                str(document.get("statement") or ""), root=self.root
            )
        except Exception:  # noqa: BLE001 — an unreadable statement is not this grant
            return False
        return held is not None and held.grant_id == grant_id and held.key == key

    def report_family_moved(self, key: str) -> dict[str, Any]:
        """Tell the owner the family moved, and say whether it was told.

        ``stopped`` is a statement about THIS device and is true before the call;
        ``reported``/``reason`` are a statement about the link. The design's §3.5
        step 3 exists so the owner stops advertising and refreshing a dead family —
        it does not exist to make the node's stop conditional on the owner hearing
        it, which is why an unreachable owner does not re-enable anything here.
        """
        detail = self.serve_report(key, kind="grant_dead", cause="family_moved")
        reported = isinstance(detail, dict) and detail.get("kind") != "error"
        reason = "reported" if reported else str((detail or {}).get("code") or "owner_unreachable")
        return {"reported": reported, "reason": reason}

    def install_refresh_fence(self) -> bool:
        """Install :meth:`refresh_fence` into this device's auth store.

        A fence is installed whenever a delegation MIGHT be in force — a readable
        document, an unprovable one, or (R1-2) a delegation-sourced ROW even when the
        document is gone. Only a device with neither gets no fence at all, which is
        what keeps the mixed-fleet and 0-peer path byte-identical (§7, T14).
        """
        try:
            target = self._auth_store()
        except Exception:  # noqa: BLE001 — no store root here: nothing to fence
            return False
        installer = getattr(target, "set_refresh_fence", None)
        if installer is None:  # pragma: no cover — a store build without the seam
            return False
        if not self._delegation_may_exist(target):
            return False
        installer(self.refresh_fence, watch=self)
        return True

    def _delegation_may_exist(self, store: Any) -> bool:
        """Whether a fence is needed at all — document OR a delegated ROW (R1-2).

        The document alone is not the question. §3.5's "cannot tell" case is a
        DELETED document beside a row that still holds the material, and the first
        version asked only about the file, so the fence was never installed and the
        store's photo-free path posted. The rows are the durable evidence; the file
        is an optimisation.
        """
        from local_operator.network.credentials import refresh as refresh_mod

        try:
            if refresh_mod.load(self.network_id, self.root, recipient=self.self_device) is not None:
                return True
        except refresh_mod.RefreshUnknown:
            return True
        return self._delegated_rows(store)

    def _delegated_rows(self, store: Any) -> bool:
        """Whether this store holds any delegation-sourced row. Unknown => True."""
        query = getattr(store, "delegated_credential_ids", None)
        if query is None:  # pragma: no cover — a store build without the seam
            return True
        try:
            ids = query()
        except Exception:  # noqa: BLE001 — cannot tell: fence (fail closed)
            return True
        if ids is None:
            return True
        return bool(ids)

    def _audit_refresh(self, event: str, **fields: Any) -> None:
        """Emit one non-material delegation record, if this client has an audit sink."""
        sink = getattr(self, "audit", None)
        write = getattr(sink, "write", None)
        if write is None:
            return
        try:
            write(event, **fields)
        except Exception:  # noqa: BLE001 — an audit fault must not fail a refresh
            return

    def _auth_store(self) -> Any:
        """This root's credential database, explicitly — see :meth:`_sync_engine`."""
        from local_operator.providers.auth_store import AuthStore

        if self.root is None:
            # Only reachable if a caller builds the factory without a root; the
            # engine passes it conditionally, but the type is ``Path | None``.
            raise RuntimeError("no config root for the credential store")
        return AuthStore(db_path=self.root / "auth.db", config_dir=self.root)

    def _sync_engine(self) -> Any:
        """The engine the direct fallback applies through. Built once, lazily."""
        if self._engine is None:
            from local_operator.network.credentials import sync as sync_mod

            # THE STORE IS ROOTED EXPLICITLY. ``SyncEngine``'s default store derives
            # its database from the AMBIENT config dir (``AuthStore`` defaults
            # ``db_path`` to ``default_db_path()``), so a CLI working under an
            # injected root would apply the copy into the operator's own store — or,
            # in a test, into whichever store happens to exist outside the root.
            self._engine = sync_mod.SyncEngine(
                self._relay,
                root=self.root,
                self_device=self.self_device,
                self_device_name=self._self_label(),
                auth_store_factory=self._auth_store if self.root is not None else None,
            )
        return self._engine

    def pull_copy_sync(self, key: str) -> dict[str, Any]:
        """Pull ONE copy from its owner over ``net_broker kind=copy``, directly.

        THE CLI'S FALLBACK PATH, and why it exists: a branch CLI talking to an
        older relay (or any relay without the local ``credential_sync_pull`` op)
        must still repair a member. The client built for this call holds a relay
        object for ONE dial (``for_this_device(..., relay=...)``), asks the owner
        itself, and applies the reply through the SAME ``SyncEngine.apply_copy_reply``
        the tick uses — same digest check, same monotonicity rule, same evidence.

        NO DIAL when it cannot help: not a holder, a key this device owns, or a
        broker-only key all answer locally. The receipt is ``{ok, state, code}``,
        the same three fields ``pull_now`` answers, and no value crosses it.
        """
        from local_operator.network.credentials import mcp_state
        from local_operator.network.credentials import sync as sync_mod

        placement = self.placement
        if placement is None:
            # No document is a placement fact, not a question for the owner.
            return {
                "ok": False,
                "state": "revoked" if self._has_prior_claim(key) else "unknown",
                "code": "not_a_holder",
            }
        entry = placement.entry(key)
        if entry is None or not entry.is_holder(self.self_device):
            # A placement fact, not something to ask the owner about. The state is
            # ``revoked`` only for a device that had a claim; a cold device gets
            # ``unknown`` and the status surface reads placement for itself.
            return {
                "ok": False,
                "state": "revoked" if self._has_prior_claim(key) else "unknown",
                "code": "not_a_holder",
            }
        owner = str(entry.owner_device)
        if not owner or owner == self.self_device:
            return {"ok": False, "state": "synced", "code": "local_owner"}
        if not sync_mod.copies_by_class(str(entry.kind)):
            return {"ok": False, "state": "unavailable", "code": "not_copyable"}
        state = sync_mod.SyncState.load(placement.network_id, root=self.root)
        held = state.applied_for(key) or {}
        held_gen = int(held.get("gen") or 0)
        reply = self._ask_owner_directly(
            owner,
            {
                "op": PEER_BROKER_OP,
                "kind": "copy",
                "network_id": placement.network_id,
                "from_device": self.self_device,
                "from_device_name": self._self_label(),
                "key": key,
                "gen": 0,
                "held": held_gen,
            },
        )
        if reply is None:
            mcp_state.record(
                placement.network_id,
                key,
                state="unavailable",
                code="owner_offline",
                owner_device=owner,
                owner_name=str(entry.owner_device_name or ""),
                root=self.root,
            )
            return {"ok": False, "state": "unavailable", "code": "owner_offline"}
        detail = reply.get("detail") if isinstance(reply, dict) else None
        outcome = self._sync_engine().apply_copy_reply(key, detail, owner=owner, held_gen=held_gen)
        if outcome in ("applied", "in_sync"):
            return {"ok": True, "state": "synced", "code": ""}
        if outcome.startswith("refused:"):
            code = outcome.split(":", 1)[1]
            mapped = mcp_state.state_for_code(code) or "unavailable"
            if mapped == "revoked" and not self._has_prior_claim(key):
                return {"ok": False, "state": "unknown", "code": code}
            mcp_state.record(
                placement.network_id,
                key,
                state=mapped,
                code=code,
                owner_device=owner,
                owner_name=str(entry.owner_device_name or ""),
                root=self.root,
            )
            return {"ok": False, "state": mapped, "code": code}
        return {"ok": False, "state": "unknown", "code": "no_answer"}

    def revoke_local_grants(self, keys: list[str] | None = None) -> list[str]:
        """Drop every cached grant this device may no longer hold. Returns the keys.

        TWO REASONS A GRANT DIES HERE, and both are reconcile facts rather than
        token facts (design §4):

        * **This device is no longer a holder** for the key — the owner revoked the
          share, the entry moved to another owner, or the placement document is
          gone. ``should_borrow`` is the same predicate every request checks, so
          this cannot disagree with what a fresh borrow would answer.
        * **This device is no longer a live member** of the network — removed, or
          the record itself is gone. The handshake would refuse on the next
          contact anyway; dropping now ends access at reconcile instead of leaving
          the bearer usable until its TTL.

        THE EVIDENCE IS THE OTHER HALF. A dropped grant with no record reads to
        ``/mcp`` as "never signed in", which sends the operator looking for a
        sign-in that used to work. ``revoked`` says what happened and asks for a
        new share, which is the only remedy that can work.

        ``keys`` is the caller's pre-merge snapshot: reading ``self.placement`` after
        a changed document clears the cache as a side effect, so the names to
        reconcile must be captured before that read. ``None`` means "whatever the
        cache holds now".
        """
        candidates = list(self.grants.keys()) if keys is None else list(dict.fromkeys(keys))
        if not candidates:
            return []
        membership_gone = not self._self_membership_active()
        dropped: list[str] = []
        for key in candidates:
            if not membership_gone and self.should_borrow(key):
                continue
            owner = self.owner_of(key)
            owner_name = self.owner_label(key)
            self.grants.drop(key)
            mcp_state.record(
                self.network_id,
                key,
                state="revoked",
                code="member_not_active" if membership_gone else "not_a_holder",
                owner_device=owner,
                owner_name=owner_name,
                root=self.root,
            )
            dropped.append(key)
        return dropped

    def _self_membership_active(self) -> bool:
        """Whether this device is still a live member of its network.

        No record, an unreadable store, a missing self row or an inactive one all
        answer ``False``: the question is permission to keep using a bearer, and
        every failure mode of the proof must fail closed.
        """
        if not self.network_id:
            return True
        try:
            from local_operator.network import store

            for record in store.list_networks(self.root):
                if record.network_id != self.network_id:
                    continue
                member = record.self_member()
                return bool(member is not None and member.active)
            return False
        except Exception:  # noqa: BLE001 — a store we cannot read proves nothing
            return False

    def _other_active_members(self) -> list[str]:
        try:
            from local_operator.network import store

            rows: list[str] = []
            for record in store.list_networks(self.root):
                if self.network_id and record.network_id != self.network_id:
                    continue
                for member in record.active_members():
                    if member.device_id and member.device_id != self.self_device:
                        rows.append(member.device_id)
            return sorted(set(rows))
        except Exception:  # noqa: BLE001 — an unreadable store means nobody to ask
            return []

    def close(self) -> None:
        """Drop every cached bearer. Called when a session tears down."""
        self.grants.clear()

    # -- shared plumbing ----------------------------------------------------

    def _self_label(self) -> str:
        try:
            from local_operator.network.identity import load as load_identity

            identity = load_identity(self.root)
            return identity.name if identity is not None else ""
        except Exception:  # noqa: BLE001 — a label is decoration
            return ""

    def _offline(self, key: str, label: str, *, owner: str = "") -> BrokerError:
        device = owner or self.owner_of(key)
        error = BrokerError(
            code="owner_offline",
            key=key,
            owner_device=device,
            owner_device_name=self.owner_label(key),
            retry_after_ms=BROKER_ERROR_TTL_MS["owner_offline"],
        )
        # Publication belongs to the final borrower decision in _held_or;
        # the relay cannot know whether that borrower holds a valid lease.
        return error

    def _render(self, error: BrokerError, key: str, label: str) -> BrokerError:
        """Fill in the person-facing sentence. The one place messages.py is reached.

        THE CATALOGUE WINS OVER THE OWNER'S WORDS, ALWAYS — there is no code for which
        the owner's own string stands in place of a sentence from here. ``owner._refuse``
        puts a DIAGNOSTIC on the wire (it names the requesting device by id and carries
        no remedy) and says as much in its own comment — "the SENTENCE the operator
        reads is rendered where the operator is" — but this method used to fill in a
        sentence only when the message was EMPTY, so the diagnostic was what the
        operator read. Measured on the two-device rig (audit round 2): after
        ``credential revoke`` the borrower showed
        "damians-MacBook-Pro does not share 'zai' with d_1d2a4f5aae0d880affc36653e3659260"
        where the design's §4.6 sentence (owner named by name, the share remedy, no
        device id) was the contract.

        A second arm was the same defect one step over (review round 1, R2): an
        unclassified code — a NEWER owner refusing for a reason this build cannot
        name, e.g. a broker ``internal`` carrying ``TimeoutError`` — took an early
        return too, so the operator read the bare class name with no owner and no
        remedy, contradicting this module's own contract. The owner's words are not
        THROWN AWAY for that case; they become the parenthetical detail
        :func:`messages._generic` interpolates, which is where a diagnostic belongs.
        """
        error.message = render_broker_error(
            error,
            key=key,
            provider=label,
            owner_name=self.owner_label(key),
            last_seen_s=self.owner_last_seen_s(error.owner_device or self.owner_of(key)),
        )
        return error

    def _from_reply(
        self,
        reply: Any,
        key: str,
        label: str,
        session_id: str,
        *,
        resting: list[dict[str, Any]] | None = None,
        request_stamp: Any = None,
        selection: Any = None,
    ) -> Grant | BrokerError:
        """Turn a leg-1 control reply into a grant or a refusal.

        ``resting`` is the cooldown set this ask excluded. When the owner answers
        that the exclusion left nothing (the owner's ``peer_excluded`` arm, marked
        ``reason: holder_excluded``; an older owner's unmarked ``quota_blocked`` with
        no remainder, see :attr:`BrokerError.is_holder_exclusion`), the answer is this
        device's own rest, rendered from its cooldowns and NOT remembered (review
        round 1, m1): caching it would record a refusal (or a ``withdrawn`` receipt)
        that outlives the cooldowns it describes, and the cooldowns already are the
        record. An owner-measured ``quota_blocked`` (its own block, or a pool spent by
        usage, marked ``pool_depleted`` whether or not its reopening is known) is the
        owner's state and is cached as before.
        """
        if not isinstance(reply, dict):
            return self._render(self._offline(key, label), key, label)
        detail = reply.get("detail")
        if not isinstance(detail, dict):
            message = str(reply.get("message") or "")
            # An ``error`` frame at the LOCAL boundary is the relay's own refusal
            # (a build without the broker, a bad frame) — never a transport fact.
            return self._render(
                BrokerError(
                    code=str(reply.get("code") or "internal"),
                    key=key,
                    owner_device=self.owner_of(key),
                    owner_device_name=self.owner_label(key),
                    message=message,
                ),
                key,
                label,
            )
        if detail.get("kind") == "grant":
            grant = self._bounded(Grant.from_detail(detail))
            return self._adopt_online(grant, key, session_id, request_stamp, selection=selection)
        refused = BrokerError.from_detail(detail)
        if resting and refused.is_holder_exclusion:
            return self._resting_refusal(key, label, resting)
        error = self._render(refused, key, label)
        self._remember(key, error)
        return error

    def _detail_from_owner(
        self,
        reply: dict[str, Any],
        key: str,
        label: str,
        *,
        excluded: bool = False,
        request_stamp: Any = None,
    ) -> dict[str, Any]:
        """A leg-2 reply as a detail object, with the cache updated.

        The transport's own refusal (``{"op": "error", ...}``) carries no code, only
        the owner relay's sentence, so it is classified by :func:`_transport_refusal_code`.
        """
        if reply.get("op") == "error":
            sentence = str(reply.get("message") or "")
            code = _transport_refusal_code(sentence)
            error = BrokerError(
                code=code,
                key=key,
                owner_device=self.owner_of(key),
                owner_device_name=self.owner_label(key),
                # Only a refusal this module has no sentence for keeps the owner's own
                # words (rendered through ``messages._generic``); every classified one is
                # rendered here, so the operator reads the remedy, not the relay's code.
                message=sentence if code == "internal" else "",
            )
            self._remember(key, error)
            return self._detail_error(self._render(error, key, label))
        detail = reply.get("detail")
        if not isinstance(detail, dict):
            error = BrokerError(code="internal", key=key, message="the owner answered no detail")
            return self._detail_error(self._render(error, key, label))
        if detail.get("kind") == "grant":
            grant = Grant.from_detail(detail)
            result = self._adopt_online(grant, key, "", request_stamp, cache=False)
            return self._detail_error(result) if isinstance(result, BrokerError) else detail
        error = self._render(BrokerError.from_detail(detail), key, label)
        if excluded and error.is_holder_exclusion:
            # The answer to the HOLDER'S exclusion, not an owner-side state (review
            # round 1, m1): the holder's cooldowns are the record, and caching this
            # here — the relay shares the session's observation file — would make
            # the next ask short-circuit on a refusal that outlives them.
            return self._detail_error(error)
        self._remember(key, error)
        return self._detail_error(error)

    def _detail_error(self, error: BrokerError) -> dict[str, Any]:
        return error.to_detail()

    # -- async helpers used by the auth store -------------------------------

    async def grant_async(
        self,
        key: str,
        *,
        session_id: str = "",
        model_id: str = "",
        force_refresh: bool = False,
        provider: str = "",
        affinity: tuple[int, int] | None = None,
        demote_credential_ids: Collection[int] = (),
    ) -> Grant | BrokerError:
        """``request_grant_sync`` off the event loop.

        The dial is blocking (a control socket, a peer link handshake), so it runs
        in a worker thread. The FUTURE-level coalescing the design asks for lives in
        the caller (``store.py``), because that is where several provider calls in
        one turn share one grant — this method's job is only to not block a loop.
        """
        return await asyncio.to_thread(
            self.request_grant_sync,
            key,
            session_id=session_id,
            model_id=model_id,
            force_refresh=force_refresh,
            provider=provider,
            affinity=affinity,
            demote_credential_ids=tuple(demote_credential_ids),
        )


def _selection_fields(
    affinity: tuple[int, int] | None, demote_credential_ids: Collection[int]
) -> dict[str, Any]:
    """The optional lend-selection fields, present ONLY when they carry something.

    The ``exclude_credential_ids`` rule (#3066): a frame with nothing to say is
    byte-identical to the one an older relay or owner always read, and an older
    owner that receives them ignores them, which is today's lend.
    """
    fields: dict[str, Any] = {}
    if affinity is not None:
        fields["affinity"] = {
            "account_id": int(affinity[0]),
            "account_created_at": int(affinity[1]),
        }
    if demote_credential_ids:
        fields["demote_credential_ids"] = sorted({int(i) for i in demote_credential_ids})
    return fields


def key_for(*, provider: str = "", mcp_url: str = "") -> str:
    """The placement key for a provider or an MCP server URL. One spelling."""
    return credential_key_for_mcp(mcp_url) if mcp_url else credential_key_for_provider(provider)


def _transport_refusal_code(sentence: str) -> str:
    """The broker code for the owner relay's own refusal of a ``net_broker`` frame.

    WHY A CLASSIFIER AND NOT ONE CODE (QA round 1, Q4). Every transport refusal used to
    read as ``unsupported`` ("runs a build that cannot lend credentials"), cached for
    five minutes. The commonest one is not that at all: ``credential revoke`` also
    takes ``broker_credential`` off the device's member row, so the owner's
    chokepoint refuses a revoked borrower before the broker's by-name
    ``not_a_holder`` can run — and the borrower was told the owner's build was too old.

    The transport deliberately sends a SENTENCE and no code (``wire.refusal_frame``),
    so the sentence is what there is. Each marker below is a fixed phrase from this
    package's own relay (``authorizer.check``, ``relay.dispatch``,
    ``relay._dispatch_slow``, ``relay.not_implemented_peer_op``), and
    ``tests/unit/network/test_credentials_real_link.py`` drives the revoked case over
    a real link so a reworded refusal fails there rather than here. Anything
    unrecognised is ``internal``: the operator reads the owner's own words, and it is
    re-asked after a minute rather than written off for five.
    """
    from local_operator.network.credentials.types import BROKER_CAPABILITY

    lowered = sentence.lower()
    if f"{BROKER_CAPABILITY!r}" in sentence or BROKER_CAPABILITY in sentence:
        # The owner holds no share for this device any more (or never did): the
        # same answer, by name, the broker itself gives a device that is not a holder.
        return "not_a_holder"
    if "not implemented" in lowered or "not an operation this build dispatches" in lowered:
        return "unsupported"
    if "relay is stopping" in lowered:
        # THE OWNER IS GOING AWAY (review round 3, F2). ``RelayServer.stop`` shuts its
        # slow-op pool ~50 ms before it closes links, so a borrower asking inside that
        # gap reaches a live link and gets the relay's own first-person sentence
        # ("this device's relay is stopping"). As ``internal`` it was cached and shown
        # to the borrower in the OWNER's voice; to the borrower it means exactly what
        # a closed link a moment later would: the owner is offline.
        return "owner_offline"
    if "already running" in lowered:
        return "rate_limited"
    if "did not finish within" in lowered:
        return "refresh_failed"
    return "internal"


def _empty_document_for_member(root: Path | None) -> tuple[str, PlacementDocument | None]:
    """An empty placement document when this device is in a network with a peer.

    ``None`` when this device is in no network at all (the 0-peer case) or is the
    only member of one — a relay that has nobody to ask has nobody to borrow from,
    so it builds no client and registers no brokering surface.
    """
    try:
        from local_operator.network import store

        for record in store.list_networks(root):
            others = [m for m in record.active_members() if m.device_id != record.self_device_id]
            if others:
                return record.network_id, PlacementDocument(record.network_id, root=root)
    except Exception:  # noqa: BLE001 — no store means no network means no client
        return "", None
    return "", None
