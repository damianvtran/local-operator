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
import threading
import time
from pathlib import Path
from typing import Any

from local_operator.network.credentials.messages import render_broker_error
from local_operator.network.credentials.placement import (
    PlacementDocument,
    placement_for_store,
)
from local_operator.network.credentials.state import PlacementState
from local_operator.network.credentials.types import (
    BROKER_ERROR_TTL_MS,
    NO_RETRY,
    PEER_BROKER_OP,
    BrokerError,
    CredentialPlacementEntry,
    Grant,
    credential_key_for_mcp,
    credential_key_for_provider,
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

    def put(self, key: str, session_id: str, grant: Grant) -> None:
        with self._lock:
            self._grants[(key, session_id)] = grant

    def drop(self, key: str, session_id: str = "") -> None:
        """Forget one grant, or every session's grant for ``key`` when unnamed."""
        with self._lock:
            if session_id:
                self._grants.pop((key, session_id), None)
                return
            for pair in [p for p in self._grants if p[0] == key]:
                del self._grants[pair]

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

    def clear(self) -> None:
        with self._lock:
            self._grants.clear()


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
        self.placement = placement
        self._state: PlacementState | None = state
        self._relay = relay
        self.deadline_s = deadline_s
        self.grants = GrantCache()
        self._state_lock = threading.Lock()

    # -- construction -------------------------------------------------------

    @classmethod
    def for_this_device(
        cls, root: Path | None = None, network_id: str = ""
    ) -> MeshCredentialClient | None:
        """A runtime-side client, or ``None`` when this device may borrow nothing.

        ``None`` is the 0-peer answer and the whole reason ``build_auth_store`` can
        promise byte-identical behaviour without a network: no placement that names
        another device's credential as borrowable means no client exists, so no
        code path below this line is reachable.
        """
        found = placement_for_store(root)
        if found is None:
            return None
        found_network, document = found
        from local_operator.network.identity import load as load_identity

        identity = load_identity(root)
        device = identity.device_id if identity is not None else ""
        return cls(
            root=root,
            self_device=device,
            network_id=network_id or found_network,
            placement=document,
            state=PlacementState.load(network_id or found_network, root),
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
        try:
            from local_operator.network import store

            for record in store.list_networks(self.root):
                member = record.member(device)
                if member is None or not member.last_seen_at:
                    continue
                return max(0.0, time.time() - float(member.last_seen_at))
        except Exception:  # noqa: BLE001 — a missing observation is not a failure
            return None
        return None

    # -- refusal cache ------------------------------------------------------

    @property
    def state(self) -> PlacementState:
        with self._state_lock:
            if self._state is None:
                self._state = PlacementState.load(self.network_id, self.root)
            return self._state

    def cached_refusal(self, key: str) -> BrokerError | None:
        """A refusal this device already has, or ``None``. Consulted BEFORE a dial."""
        status = self.state.status(key)
        if not status or status == "active":
            return None
        ttl = BROKER_ERROR_TTL_MS.get(status, 0)
        if ttl == NO_RETRY:
            ttl = UNSUPPORTED_OBSERVATION_MS
        row = self.state.observation(key)
        if row is None:
            return None
        # ``reason`` is deliberately NOT read back here. The observation document
        # carries the field for a diagnostic to read, but the sentence a person sees
        # is rendered at the point of use from the CODE (``messages.py``) — and a
        # borrower that echoed a stored string would be presenting one device's
        # prose as another's, which is the failure ``render_broker_error`` exists to
        # prevent. The message is filled in by ``_render`` below.
        return BrokerError(
            code=status,
            key=key,
            owner_device=str(row.get("owner_device") or self.owner_of(key)),
            owner_device_name=self.owner_label(key),
            retry_after_ms=ttl,
        )

    def _remember(self, key: str, error: BrokerError) -> None:
        ttl = error.cache_ttl_ms
        if ttl == NO_RETRY:
            ttl = UNSUPPORTED_OBSERVATION_MS
        self.state.observe(
            key,
            error.code,
            reason="",
            owner_device=error.owner_device or self.owner_of(key),
            retry_after_ms=ttl,
        )
        self._save_state()

    def _remember_grant(self, key: str, grant: Grant) -> None:
        self.state.note_grant(key, grant.grant_id, owner_device=grant.credential_ref.owner_device)
        self._save_state()

    def _save_state(self) -> None:
        try:
            self.state.save()
        except OSError:
            # A cache that cannot be written is still a cache: the observation lives
            # in this object for the life of the process, and failing a borrow because
            # the refusal cache was not persisted would trade a real capability for a
            # memo. Read-only config dirs are legitimate (a container, a test root).
            pass

    # -- leg 1: the runtime asks its own relay ------------------------------

    def request_grant_sync(
        self,
        key: str,
        *,
        session_id: str = "",
        model_id: str = "",
        force_refresh: bool = False,
        provider: str = "",
    ) -> Grant | BrokerError:
        """Borrow a bearer for ``key``. Blocking; call from a thread.

        THE ORDER IS THE CONTRACT: a live grant, then a cached refusal, then the
        dial. Checking the cache before the grant would serve a stale "no" over a
        bearer this process is holding; checking it after the dial would defeat it.
        """
        label = provider or key
        cached = self.grants.get(key, session_id)
        if cached is not None:
            return cached
        refusal = self.cached_refusal(key)
        if refusal is not None:
            return self._render(refusal, key, label)
        if not self.should_borrow(key):
            return BrokerError(
                code="not_a_holder",
                key=key,
                owner_device=self.owner_of(key),
                owner_device_name=self.owner_label(key),
                message="",
            )
        try:
            from local_operator.network import store
            from local_operator.network.relay import control_request

            record = store.find_own_relay(self.root)
            if record is None:
                return self._offline(key, label)
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
            )
        except Exception as exc:  # noqa: BLE001 — a broker bug must not fail a turn
            return BrokerError(
                code="internal",
                key=key,
                owner_device=self.owner_of(key),
                owner_device_name=self.owner_label(key),
                message=exc.__class__.__name__,
            )
        return self._from_reply(reply, key, label, session_id)

    # -- leg 2: the relay asks the owner ------------------------------------

    def serve_grant(
        self,
        key: str,
        *,
        provider: str = "",
        session_id: str = "",
        model_id: str = "",
        force_refresh: bool = False,
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
        refusal = self.cached_refusal(key)
        if refusal is not None:
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
        direct = self._ask_owner_directly(owner, frame)
        if direct is not None:
            return self._detail_from_owner(direct, key, label)
        return self._detail_error(self._render(self._offline(key, label, owner=owner), key, label))

    def _ask_owner_directly(self, owner: str, frame: dict[str, Any]) -> dict[str, Any] | None:
        """Run ``net_broker`` over a link to ``owner``, dialling if there is none.

        ``None`` means "this device could not get an answer from the owner", which
        the caller turns into ``owner_offline``. A broker REFUSAL is not ``None`` —
        an ``assumed`` refusal is an answer, and conflating the two is how a
        "you are not a holder" becomes "the device is asleep".
        """
        if not owner:
            return None
        ensure = getattr(self._relay, "_ensure_link", None)
        if ensure is None:
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
    ) -> None:
        """Tell the owner a borrowed bearer failed. BLOCKING, best effort.

        ``kind`` is what the PROVIDER said, never a verdict about the credential:
        ``quota`` for a 429, ``invalid`` for a 401 or an ``invalid_grant``. The
        owner decides what that means for its own row, and the whole point of the
        split is that this device cannot decide it — see ``owner.py``'s report arm
        for why routing this into ``rotate_sibling`` would log the operator out of
        every device.
        """
        if not self.should_borrow(key):
            return
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
            )
        except Exception:  # noqa: BLE001 — a failed report must not raise into a turn
            return

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
            **{k: v for k, v in fields.items() if v not in (None, "")},
        }
        self.grants.drop(key)
        reply = self._ask_owner_directly(owner, frame)
        detail = (reply or {}).get("detail")
        if not isinstance(detail, dict):
            return {"kind": "error", "code": "owner_offline", "key": key}
        return detail

    def pull_placement(self) -> dict[str, Any]:
        """Ask every other active member for its placement document, and merge.

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
        """
        if self.placement is None:
            return {"kind": "ack", "key": "", "changed": [], "owners": 0}
        changed: list[str] = []
        asked = 0
        for device in self._other_active_members():
            asked += 1
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
            if isinstance(document, dict):
                changed.extend(self.placement.merge(document, from_device=device))
        if changed:
            try:
                self.placement.save()
            except OSError:
                pass
        return {
            "kind": "ack",
            "key": "",
            "changed": sorted(set(changed)),
            "owners": asked,
        }

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
        return BrokerError(
            code="owner_offline",
            key=key,
            owner_device=device,
            owner_device_name=self.owner_label(key),
            retry_after_ms=BROKER_ERROR_TTL_MS["owner_offline"],
        )

    def _render(self, error: BrokerError, key: str, label: str) -> BrokerError:
        """Fill in the person-facing sentence. The one place messages.py is reached."""
        if not error.message:
            error.message = render_broker_error(
                error,
                key=key,
                provider=label,
                owner_name=self.owner_label(key),
                last_seen_s=self.owner_last_seen_s(error.owner_device or self.owner_of(key)),
            )
        return error

    def _from_reply(self, reply: Any, key: str, label: str, session_id: str) -> Grant | BrokerError:
        """Turn a leg-1 control reply into a grant or a refusal."""
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
            grant = Grant.from_detail(detail)
            self.grants.put(key, session_id, grant)
            self._remember_grant(key, grant)
            return grant
        error = self._render(BrokerError.from_detail(detail), key, label)
        self._remember(key, error)
        return error

    def _detail_from_owner(self, reply: dict[str, Any], key: str, label: str) -> dict[str, Any]:
        """A leg-2 reply as a detail object, with the cache updated.

        The transport's own refusal (``{"op": "error", ...}``) has no code and is
        mapped to ``unsupported``: it is what an older peer answers to an unknown
        op, and the design's requester behaviour for that is to degrade to today's
        no-credential path for the session rather than retry.
        """
        if reply.get("op") == "error":
            error = BrokerError(
                code="unsupported",
                key=key,
                owner_device=self.owner_of(key),
                owner_device_name=self.owner_label(key),
                message=str(reply.get("message") or ""),
            )
            self._remember(key, error)
            return self._detail_error(self._render(error, key, label))
        detail = reply.get("detail")
        if not isinstance(detail, dict):
            error = BrokerError(code="internal", key=key, message="the owner answered no detail")
            return self._detail_error(self._render(error, key, label))
        if detail.get("kind") == "grant":
            grant = Grant.from_detail(detail)
            self._remember_grant(key, grant)
            return detail
        error = self._render(BrokerError.from_detail(detail), key, label)
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
        )


def key_for(*, provider: str = "", mcp_url: str = "") -> str:
    """The placement key for a provider or an MCP server URL. One spelling."""
    return credential_key_for_mcp(mcp_url) if mcp_url else credential_key_for_provider(provider)


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
