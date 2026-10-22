"""H3 — the serialized per-account screening cycle (A5, mandatory in H3).

Two pieces:

- :func:`select_next_screening_target` — ordering only: senders in a stable
  rotation starting AFTER the persisted ``last_sender``, request-type
  messages ahead of notes within each sender's turn. Canonical DTO shapes:
  the sender lives at ``message.sender.account_id`` and the type at
  ``message.type`` (QA: a flat ``item.type`` read selected a note over a
  request); a tolerated flat fallback exists ONLY for already-normalized
  fixtures.
- :class:`AccountScreeningCycle` — the THIN serialized turn: one conditional
  native lease claim (so two workers/processes can never process the same
  account turn), a durable rotation cursor, ONE target dispatched through
  the PUBLIC async screener, release in ``finally``. No daemon, no startup
  integration, no passive inbox loop, no fanout, no new quotas: this is the
  frozen H3 mandate, deliberately NOT deferred to H4. Scheduling metadata is
  accounts/senders/lease only — message bodies stay ephemeral in memory.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

if TYPE_CHECKING:
    from local_operator.team_bridge.screening import ScreeningPolicy

__all__ = [
    "AccountScreeningCycle",
    "canonical_sender",
    "canonical_type",
    "select_next_screening_target",
]


#: The turn token currently held by the CALLING context (set by the cycle,
#: read by the public screening entry): dispatch inside a claimed cycle must
#: reuse that exact token instead of claiming a second, competing turn.
_CURRENT_TURN: ContextVar[tuple[str, str] | None] = ContextVar("team_bridge_turn", default=None)


def current_turn() -> tuple[str, str] | None:
    """(account_scope, token) of the turn this context already holds."""

    return _CURRENT_TURN.get()


@contextmanager
def turn_binding(scope: str, token: str):
    """Bind an already-claimed native turn for the enclosed dispatch."""

    reset = _CURRENT_TURN.set((scope, token))
    try:
        yield
    finally:
        _CURRENT_TURN.reset(reset)


def _message_block(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    message = item.get("message")
    return message if isinstance(message, Mapping) else None


def canonical_type(item: Mapping[str, Any]) -> str:
    """The DTO's message type: canonical nested first, flat fallback second."""

    message = _message_block(item)
    if message is not None:
        nested = message.get("type")
        if isinstance(nested, str) and nested:
            return nested
    flat = item.get("type")
    return flat if isinstance(flat, str) else ""


def canonical_sender(item: Mapping[str, Any]) -> str:
    """The sender account id from the canonical H2 DTO.

    The server DTO carries the sender as a SIBLING of ``message``
    (``item['sender']['account_id']``, QA's probe fixture); a
    message-nested sender is tolerated as a legacy shape. A malformed
    sender must not share a rotation slot with a real one.
    """

    for location in (item, _message_block(item)):
        if location is None:
            continue
        sender = location.get("sender")
        if isinstance(sender, Mapping):
            account = sender.get("account_id")
            if isinstance(account, str) and account:
                return account
    return ""


def select_next_screening_target(
    items: Sequence[Mapping[str, Any]], *, last_sender: str | None = None
) -> Mapping[str, Any] | None:
    """The NEXT eligible item to screen, or None — at most ONE per turn."""

    if not items:
        return None
    order: list[str] = []
    for item in items:
        sender = canonical_sender(item)
        if sender not in order:
            order.append(sender)
    start = 0
    if last_sender in order:
        start = order.index(last_sender) + 1
    for offset in range(len(order)):
        sender = order[(start + offset) % len(order)]
        candidates = [item for item in items if canonical_sender(item) == sender]
        requests = [item for item in candidates if canonical_type(item) == "request"]
        others = [item for item in candidates if canonical_type(item) != "request"]
        return requests[0] if requests else others[0]
    return None


class AccountScreeningCycle:
    """One account's serialized screening turn (THIN — no daemon, no poll).

    ``run_once`` claims the account's native lease (conditional update in one
    transaction; the minted UNFORGEABLE token, not a reusable holder label,
    identifies this turn), selects the ONE next target with the durable
    cursor, advances the cursor before dispatch, binds the turn for the
    enclosed dispatch (so the public entry reuses IT rather than claiming a
    second competing turn) and releases in ``finally`` on success, failure
    AND cancellation. Eligibility (breaker/budget/held retry) is decided by
    the caller before items arrive — no invented quotas, no bypass.
    """

    def __init__(self, *, store: Any, owner: Any, holder: str, ttl: float = 90.0) -> None:
        self._store = store
        self._owner = owner
        self._holder = holder
        self._ttl = ttl

    async def run_once(
        self,
        items: Sequence[Mapping[str, Any]],
        screen: Callable[[Mapping[str, Any]], Awaitable[Any]],
    ) -> Any:
        """One serialized turn; None when the lease is held or nothing is due."""

        turn = self._store.claim_screen_turn(
            self._owner, holder_prefix=self._holder, now=time.time(), ttl=self._ttl
        )
        if turn is None:
            return None
        # WHOLE-DURATION turn lifetime (R8/Q15): the keeper covers the
        # COMPLETE cycle — fetch/preparation, the callback and cleanup — not
        # only the inference inside the screener. It renews the exact
        # claimed turn while the work runs and, on ownership loss, CANCELS
        # the work and joins it; the successor's turn is never released.
        lost = {"flag": False}
        work = asyncio.ensure_future(self._run_body(items, screen, turn))
        keeper = asyncio.ensure_future(self._keep_turn(turn, work, lost))
        try:
            result = await work
        except asyncio.CancelledError:
            if lost["flag"]:
                # Ownership moved mid-cycle: no result, no receipt, and the
                # successor keeps the lease.
                return None
            raise
        else:
            if lost["flag"]:
                # A callback that SUPPRESSED the cancellation can return a
                # value while ownership is already lost (R9/Q3-6): refuse
                # the result — no receipt, no successor release.
                return None
            return result
        finally:
            keeper.cancel()
            try:
                await keeper
            except BaseException:  # noqa: BLE001 — the keeper is already done
                pass
            if not lost["flag"]:
                self._store.release_screen_turn(self._owner, turn=turn)

    async def _run_body(
        self,
        items: Sequence[Mapping[str, Any]],
        screen: Callable[[Mapping[str, Any]], Awaitable[Any]],
        turn: str,
    ) -> Any:
        schedule = self._store.screen_schedule(self._owner) or {}
        target = select_next_screening_target(items, last_sender=schedule.get("last_sender"))
        if target is None:
            return None
        self._store.set_screen_cursor(self._owner, canonical_sender(target))
        with turn_binding(self._store._account_scope(self._owner), turn):
            return await screen(target)

    async def _keep_turn(
        self, turn: str, work: "asyncio.Future[Any]", lost: dict[str, bool]
    ) -> None:
        """Renew the claimed turn while the cycle body runs; on loss, cancel
        the body (fencing) and flag the loss for the caller."""

        while True:
            await asyncio.sleep(20.0)
            try:
                ok = self._store.renew_screen_turn(
                    self._owner,
                    turn=turn,
                    now=time.time(),
                    ttl=max(self._ttl, 90.0),
                )
            except Exception:  # noqa: BLE001 — a refusal is treated as loss
                ok = False
            if not ok:
                lost["flag"] = True
                work.cancel()
                return


def default_status_fetcher(
    *,
    config_dir: Path,
    owner: Any,
    transport: Any = None,
) -> Callable[[], Awaitable[Any]]:
    """The PRODUCTION SDK-status fetcher (R8/Q3).

    Calls the H1 client's own ``status`` with the owner's PINNED OAuth
    access — the ordinary SDK/auth path, no parallel transport and no model-
    key cascade. The RAW nested DTO is returned untouched; the caller
    normalizes it strictly and re-verifies the owner after the await.
    ``transport`` is injected only by proof rigs.
    """

    _require_owner_identity(owner)  # fail closed on a missing canonical hub

    async def _fetch() -> Any:
        from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
        from local_operator.team_bridge.store import Store

        client = TeamBridgeClient(
            config_dir,
            base_url=owner.canonical_base_url,
            credential_id=owner.credential_id,
            transport=transport,
        )
        # R9/Q3-3: the exact intended AuthStore row and canonical hub are
        # compared to the native initiating owner BEFORE the GET — a bearer
        # the owner did not enroll never travels.
        before = await client.current_access()
        _require_access_matches_owner(before, owner)
        result = await client.status(access=before)
        # ... and AGAIN after the await: a same-credential_id account swap
        # during the read is a change even while the native g1 stays live.
        after = await client.current_access()
        _require_access_matches_owner(after, owner)
        if (after.credential_id, after.account_id, after.api_endpoint) != (
            before.credential_id,
            before.account_id,
            before.api_endpoint,
        ):
            raise BridgeRefusal(
                "bridge_owner_moved", "the signed-in account changed during the status read"
            )
        if not Store(config_dir).current_generation_is_live(owner):
            raise BridgeRefusal(
                "bridge_owner_moved", "the native owner generation moved during the status read"
            )
        if not isinstance(result, Mapping):
            from local_operator.team_bridge.client import BridgeRefusal

            raise BridgeRefusal("bridge_owner_mismatch", "the status read returned no mapping")
        return VerifiedStatus(_VERIFIED_TOKEN, result)

    return _fetch


def _require_owner_identity(owner: Any) -> None:
    """The owner must carry a canonical hub for any identity comparison."""

    hub = str(getattr(owner, "canonical_base_url", "") or "").rstrip("/")
    if not hub:
        from local_operator.team_bridge.client import BridgeRefusal

        raise BridgeRefusal("bridge_owner_mismatch", "the enrolled owner has no canonical hub")


def _require_access_matches_owner(access: Any, owner: Any) -> None:
    """Credential/account/canonical-hub equality; tokens are never touched."""

    from local_operator.team_bridge.client import BridgeRefusal

    endpoint = (getattr(access, "api_endpoint", None) or "").rstrip("/")
    if (
        getattr(access, "credential_id", None) != owner.credential_id
        or getattr(access, "account_id", None) != owner.account_id
        or endpoint != str(owner.canonical_base_url).rstrip("/")
    ):
        raise BridgeRefusal(
            "bridge_owner_mismatch", "the signed-in credential is not the enrolled owner"
        )


async def _verify_live_identity(owner: Any, auth_store: Any, config_dir: Any) -> None:
    """Refuse unless the SUPPORTED AuthStore still resolves the exact enrolled
    identity (R10/A).

    The capture's own fields prove what WAS verified at mint time; this read
    proves what the live store serves NOW. Token values are never compared,
    logged or hashed — only credential/account/canonical-hub identity. A
    revoked row (logout) or a same-row account switch refuses; a legitimate
    token refresh under the SAME identity passes.
    """

    from local_operator.team_bridge.client import (
        BridgeRefusal,
        BridgeSignInRequired,
        TeamBridgeClient,
    )
    from local_operator.team_bridge.store import StoreRefused

    client = TeamBridgeClient(
        config_dir,
        base_url=owner.canonical_base_url,
        credential_id=owner.credential_id,
        store=auth_store,
    )
    try:
        access = await client.current_access()
    except BridgeSignInRequired as failure:
        raise StoreRefused(
            "policy_stale", "the signed-in Radient account is not available"
        ) from failure
    except BridgeRefusal as failure:
        raise StoreRefused("policy_stale", "the signed-in credential was refused") from failure
    if (
        access.credential_id != owner.credential_id
        or access.account_id != owner.account_id
        or (access.api_endpoint or "").rstrip("/") != str(owner.canonical_base_url).rstrip("/")
    ):
        raise StoreRefused("policy_stale", "the signed-in identity is not the enrolled owner")


def _owner_frame_digest(owner: Any) -> str:
    """The initiating owner frame digest — reused here so a cached status is
    bound to the exact owner that cached it (R8/Q3)."""

    from local_operator.team_bridge.store import _owner_frame_digest as _digest

    return _digest(owner)


@dataclasses.dataclass(frozen=True)
class AcceptedLaw:
    """The normalized outcome of one lawful status read: selector identity
    plus the consent/guidance FLOOR metadata the verified path binds (the
    served texts themselves are never retained)."""

    version: str
    template_hash: str
    consent_served_version: str
    guidance_version: str
    guidance_sha: str
    consent_sha: str
    # The hub's accepted consent-version set (``consent.accepted_versions``),
    # empty when the hub does not send one. See ``_consent_receipt_is_current``.
    consent_accepted_versions: tuple[str, ...] = ()


def _stage_text_floor(kind: str, version: Any, text: Any) -> tuple[str, str]:
    """Stage one served text as contentless candidate metadata (R11/3).

    Shape, boundedness and cap only — NO floor read and NO write here. The
    version-rank/immutable-bytes comparison and the durable write belong to
    ONE atomic native transaction (`Store.advance_text_floors`) AFTER the
    whole origin/identity/currency law has passed, so a refused candidate —
    or an unverified origin upstream — can never advance, or partially
    poison, any floor.
    """

    import hashlib as _hashlib

    from local_operator.team_bridge.client import TEXT_CAP_BYTES, version_number
    from local_operator.team_bridge.store import StoreRefused

    if not isinstance(version, str) or not isinstance(text, str):
        raise StoreRefused("policy_stale", f"{kind} text has an unsupported shape")
    if version_number(version) is None:
        raise StoreRefused("policy_stale", f"{kind} version does not parse")
    encoded = text.encode("utf-8")
    if len(encoded) > TEXT_CAP_BYTES:
        raise StoreRefused("policy_stale", f"{kind} text exceeds the acceptance cap")
    return version, _hashlib.sha256(encoded).hexdigest()


def _consent_accepted_versions(consent: "Mapping[str, Any]") -> tuple[str, ...]:
    """The hub's accepted consent-version set from ``consent.accepted_versions``.

    Absent or malformed reads as EMPTY, never as "anything goes": with no
    well-formed set, currency falls back to the exact served-version match
    this client always enforced. Only non-empty strings survive.
    """

    raw = consent.get("accepted_versions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    if not all(isinstance(v, str) and v for v in raw):
        return ()
    return tuple(raw)


def _consent_receipt_is_current(
    receipt_version: Any, served_version: str, accepted: "tuple[str, ...]"
) -> bool:
    """Is the committed consent receipt still CURRENT under this hub's law?

    WHY A SET (agent-server PR #156): the hub serves ``bridge_consent_v2`` for
    NEW consents while every ``bridge_consent_v1`` endpoint stays valid — the
    server never re-consents them. Requiring receipt == served version held
    every v1 member ``policy_stale`` the moment v2 was served, so the hub now
    also sends ``consent.accepted_versions``. A receipt is current when it is
    the served version, or when it is EXACTLY a member of a set that also
    contains the served version (a set that omits the version it serves is
    incoherent and grants nothing).

    Exact string membership only — never string or numeric ordering: a
    version is an identifier, and "v10" > "v9" style comparisons would
    accept versions the hub never named. Anything not listed fails closed.
    """

    if not isinstance(receipt_version, str) or not receipt_version:
        return False
    if receipt_version == served_version:
        return True
    return served_version in accepted and receipt_version in accepted


def _normalize_accepted_status(status: "Mapping[str, Any]", owner: Any, store: Any) -> AcceptedLaw:
    """Strictly normalize the RAW SDK status DTO against the FROZEN server law.

    The frozen Go DTO is ``{api_version:"1", available, tenants:[{tenant_id,
    tenant_name, eligible, endpoint:null|{id,label,state}, counts}],
    guidance:{version,text}, screening:{version,template}, consent:{version,
    text}}``. The endpoint lives ONLY on the caller's OWN tenant row and must
    be ``owner.endpoint_id`` with state exactly ``active``; ``available``
    means SOME organization. Served guidance/consent texts pass the DURABLE
    contentless floors; the screening selector passes the local-template
    comparison; consent CURRENCY checks the committed receipt. Every miss is
    a typed ``policy_stale`` refusal before publication, client creation or
    spend.
    """

    from local_operator.team_bridge.client import (
        _api_supported,
        _screening_is_trusted,
        screening_template_sha256,
    )
    from local_operator.team_bridge.store import StoreRefused

    if not isinstance(status, Mapping):
        raise StoreRefused("policy_stale", "accepted status is not a mapping")
    if not _api_supported(status):
        raise StoreRefused("policy_stale", "unsupported api_version in status")
    if status.get("available") is not True:
        raise StoreRefused("policy_stale", "messaging is not available for this account")
    tenants = status.get("tenants")
    if not isinstance(tenants, Sequence) or isinstance(tenants, (str, bytes)):
        raise StoreRefused("policy_stale", "tenants missing from status")
    rows = [
        entry
        for entry in tenants
        if isinstance(entry, Mapping) and entry.get("tenant_id") == owner.tenant_id
    ]
    if len(rows) != 1:
        raise StoreRefused("policy_stale", "owner tenant row missing or duplicated")
    row = rows[0]
    if row.get("eligible") is not True:
        raise StoreRefused("policy_stale", "owner organization is not eligible")
    endpoint = row.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise StoreRefused("policy_stale", "owner endpoint absent from the tenant row")
    if endpoint.get("id") != owner.endpoint_id:
        raise StoreRefused("policy_stale", "status endpoint is not the enrolled owner endpoint")
    if endpoint.get("state") != "active":
        raise StoreRefused("policy_stale", "owner endpoint is not active")
    screening = status.get("screening")
    if not isinstance(screening, Mapping):
        raise StoreRefused("policy_stale", "status carries no screening selector")
    version = screening.get("version")
    template = screening.get("template")
    if not _screening_is_trusted(version, template):
        raise StoreRefused("policy_stale", "screening selector does not match the local template")
    guidance = status.get("guidance")
    if not isinstance(guidance, Mapping):
        raise StoreRefused("policy_stale", "status carries no guidance section")
    guidance_version, guidance_sha = _stage_text_floor(
        "guidance", guidance.get("version"), guidance.get("text")
    )
    consent = status.get("consent")
    if not isinstance(consent, Mapping):
        raise StoreRefused("policy_stale", "status carries no consent section")
    consent_version, consent_sha = _stage_text_floor(
        "consent", consent.get("version"), consent.get("text")
    )
    # CONSENT CURRENCY: the served version is the CURRENT server version, NOT
    # proof of acceptance. Acceptance is the COMMITTED native receipt;
    # currency is that receipt's version being the served one or in the hub's
    # accepted set (_consent_receipt_is_current).
    accepted_versions = _consent_accepted_versions(consent)
    receipt = store.owner_consent_receipt(owner)
    if receipt is None:
        raise StoreRefused("policy_stale", "no committed consent receipt for the owner")
    if not _consent_receipt_is_current(
        receipt["consent_version"], consent_version, accepted_versions
    ):
        raise StoreRefused(
            "policy_stale", "committed consent is not current for the served version"
        )
    import hashlib as _hashlib

    digest = _hashlib.sha256(str(template).encode("utf-8")).hexdigest()
    if digest != screening_template_sha256(version):
        raise StoreRefused(
            "policy_stale", "screening template bytes do not match the local selector"
        )
    return AcceptedLaw(
        version=str(version),
        template_hash=digest,
        consent_served_version=consent_version,
        guidance_version=guidance_version,
        guidance_sha=guidance_sha,
        consent_sha=consent_sha,
        consent_accepted_versions=accepted_versions,
    )


_CAPTURE_TOKEN = object()


class AcceptedCapture:
    """An INTERNALLY minted proof that a live status read passed the full law.

    R9/Q3-4: a caller-supplied mapping — even one carrying a matching
    ``owner_frame`` string — cannot certify its own provenance, and QA
    forged exactly such a negative into a DELIVER. Only this class, minted
    by the verified fetch path with the module-private token, may authorise
    reuse, and only while its owner frame, selector AND freshness still
    hold. ``str(...)`` digests, public dict frames and any Mapping are
    refused where a capture is required.
    """

    __slots__ = (
        "_locked",
        "version",
        "template_hash",
        "owner_frame",
        "captured_at",
        "credential_id",
        "account_id",
        "hub",
        "owner_generation",
        "consent_version",
        "consent_ref",
        "consent_served_version",
        "consent_accepted_versions",
        "guidance_version",
        "guidance_sha",
        "consent_sha",
        "source",
    )

    def __init__(
        self,
        token: Any,
        *,
        version: str,
        template_hash: str,
        owner_frame: str,
        captured_at: float,
        credential_id: Any = None,
        account_id: Any = None,
        hub: Any = None,
        owner_generation: Any = None,
        consent_version: Any = None,
        consent_ref: Any = None,
        consent_served_version: Any = None,
        guidance_version: Any = None,
        guidance_sha: Any = None,
        consent_sha: Any = None,
        source: Any = None,
        consent_accepted_versions: "tuple[str, ...]" = (),
    ) -> None:
        if token is not _CAPTURE_TOKEN:
            raise ValueError("AcceptedCapture is minted only by the verified status path")
        self.version = version
        self.template_hash = template_hash
        self.owner_frame = owner_frame
        self.captured_at = captured_at
        # R10/A: the FULL authenticated identity the capture was minted
        # under. The abbreviated lease frame deliberately omits account/hub
        # (leases carry a separate account scope); a CAPTURE has no such
        # second binding, so it must carry the complete namespace itself —
        # otherwise a same-row account switch would reuse it.
        self.credential_id = credential_id
        self.account_id = account_id
        self.hub = hub
        self.owner_generation = owner_generation
        self.consent_version = consent_version
        self.consent_ref = consent_ref
        self.consent_served_version = consent_served_version
        self.consent_accepted_versions = consent_accepted_versions
        self.guidance_version = guidance_version
        self.guidance_sha = guidance_sha
        self.consent_sha = consent_sha
        self.source = source
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("AcceptedCapture is immutable once minted")
        object.__setattr__(self, name, value)

    # R12/Q2: there is deliberately NO production copy affordance. The
    # supported object is strictly immutable after mint (`__setattr__`
    # raises once locked), so no supported API can refresh `captured_at`
    # and re-assert the age gate. Negative-arm fixtures clone via
    # `object.__new__` + `object.__setattr__`, which bypasses the guard BY
    # DESIGN and therefore demonstrates nothing beyond what the guard
    # already blocks; the age/freshness provenance is anchored by the
    # `_CAPTURE_TOKEN` mint and the strict non-future/300 s re-validation.


_VERIFIED_TOKEN = object()


class VerifiedStatus:
    """A status payload carrying the SEALED internal receipt of the real SDK
    verification (R11/2).

    ``isinstance`` of a publicly constructible class is not provenance: a
    caller could wrap a lawful raw DTO and gain authority. The constructor is
    therefore token-sealed — only :func:`default_status_fetcher`, after its
    actual before/after OAuth pin, native owner/generation and canonical-hub
    checks, may issue one — and the cycle REFUSES any fetch result that is
    not a sealed instance BEFORE normalizing, minting a capture, advancing a
    text floor or creating a client. The pure normalizer stays available for
    adversarial raw-data tests; provenance and data validation are separate
    concerns.
    """

    __slots__ = ("status",)

    def __init__(self, token: Any, status: "Mapping[str, Any]") -> None:
        if token is not _VERIFIED_TOKEN:
            raise ValueError("VerifiedStatus is issued only by the verified status path")
        self.status = status


def _full_owner_frame(owner: Any) -> str:
    """The COMPLETE owner frame for capture bindings (identity + hub + gen).

    Deliberately distinct from the abbreviated lease-token frame, which
    omits account/hub by design (R10/A).
    """

    material = (
        f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
        f"|{owner.endpoint_id}|{owner.canonical_base_url}|{owner.enrollment_generation}"
    )
    import hashlib as _hashlib

    return _hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


_STATUS_CAPTURE_TTL = 300.0


@dataclasses.dataclass(frozen=True)
class _CyclePolicy:
    """The thin composition's policy view (R6/Q3): values come from the
    TRUSTED resolver + accepted-status constants, never from message data
    or caller strings. Capture via PolicySnapshot reads these fields."""

    screening_version: str
    template_sha256: str
    # Invariant with the protocol's attribute type: a narrower "str" is NOT
    # assignable to the read-write "str | None" protocol member.
    default_model: str | None
    admission_revision: str
    approved_serving_identities: tuple[str, ...]
    provider: str


async def run_screening_cycle(
    *,
    config_dir: "Path",
    owner: Any,
    items: "Sequence[Mapping[str, Any]]",
    auth_store: Any,
    settings: Any = None,
    holder: str = "team-bridge-cycle",
    ttl: float = 120.0,
    accepted_capture: "AcceptedCapture | None" = None,
    status_fetcher: "Callable[[], Awaitable[Any]] | None" = None,
) -> Any:
    """The SHIPPED THIN production composition (R6/Q3).

    accepted-status/config-default PUBLISHER -> durable account cycle ->
    lazy production factory -> PUBLIC screening core, as ONE ordinary
    callable. This is the production caller a normal consumer uses; it is
    not rig-only machinery and adds no daemon, poll, connector or UI. The
    policy is published from the SAME trusted resolver the factory uses
    (CAS against the current epoch: a stale publisher refuses rather than
    rewinding current authority), and the cycle serializes the account.
    Each item's runner is built fresh because the wire guard binds MESSAGE
    identity; the cycle's turn is reused by the public entry via
    ``turn_binding`` instead of claiming a second competing turn.
    """

    from local_operator.team_bridge.model_boundary import (
        make_screening_runner,
        resolve_default_selection,
    )
    from local_operator.team_bridge.screening import screen_message_async
    from local_operator.team_bridge.store import Store, StoreRefused

    selection = resolve_default_selection(config_dir)
    if selection is None:
        return None
    # ACCEPTED STATUS (R8/Q3): the owner-pinned SDK status is received
    # through the normal accepted-status seam — an injected fetcher (the
    # production one calls the H1 client's own ``status`` with the PINNED
    # OAuth access) or an explicitly CACHED mapping that carries its own
    # provenance. The RAW nested DTO is normalized STRICTLY; the LOCAL
    # embedded template hash is the only template authority. A missing
    # source, unknown version, mismatched bytes, disabled/ineligible
    # tenant/endpoint, or an unprovenanced cache is a TYPED REFUSAL — never
    # a default, never an arbitrary flat mapping, never a dispatch.
    from local_operator.team_bridge.client import screening_template_sha256

    store = Store(config_dir)
    capture = accepted_capture
    if capture is not None and not isinstance(capture, AcceptedCapture):
        # R9/Q3-4: a caller-supplied mapping or a public frame string is an
        # ASSERTION, not provenance — QA forged exactly such a negative into
        # a DELIVER, so it is refused here as not-authority.
        raise StoreRefused("policy_stale", "an unverified status mapping is not authority")
    if capture is not None:
        # R10/A: reuse is authority ONLY while its COMPLETE identity still
        # holds — full namespace, selector, consent currency and floors —
        # AND the live supported AuthStore still resolves to the SAME
        # credential/account/hub. A same-row account switch or a logout
        # therefore refuses HERE, before any client/POST/spend, even on an
        # unbilled semantic-cache hit.
        now = time.time()
        stale = (
            not isinstance(capture.captured_at, (int, float))
            or isinstance(capture.captured_at, bool)
            or capture.captured_at != capture.captured_at  # NaN
            or capture.captured_at > now
            or (now - capture.captured_at) > _STATUS_CAPTURE_TTL
            or capture.owner_frame != _full_owner_frame(owner)
            or capture.credential_id != owner.credential_id
            or capture.account_id != owner.account_id
            or capture.hub != owner.canonical_base_url
            or capture.owner_generation != owner.enrollment_generation
            or screening_template_sha256(capture.version) is None
            or capture.template_hash != screening_template_sha256(capture.version)
        )
        if not stale:
            receipt_now = store.owner_consent_receipt(owner)
            floors_ok = True
            for kind, ver, sha in (
                ("guidance", capture.guidance_version, capture.guidance_sha),
                ("consent", capture.consent_version, capture.consent_sha),
            ):
                if not isinstance(ver, str) or not isinstance(sha, str):
                    floors_ok = False
                    break
                floor = store.text_floor(owner, kind)
                # PRESENT and matching: an absent floor is a refusal, not
                # accepted absence (the capture must still be backed by the
                # durable acceptance it claims).
                if floor is None or floor != (ver, sha):
                    floors_ok = False
                    break
            if (
                receipt_now is None
                or not _consent_receipt_is_current(
                    receipt_now["consent_version"],
                    capture.consent_served_version,
                    capture.consent_accepted_versions,
                )
                or receipt_now.get("consent_ref") != capture.consent_ref
                or not floors_ok
            ):
                stale = True
        if not stale:
            await _verify_live_identity(owner, auth_store, config_dir)
        else:
            capture = None
    if capture is None:
        if status_fetcher is None:
            # None is NOT authority: without a status source there is
            # nothing to accept, so nothing is published and nothing
            # dispatches.
            raise StoreRefused("policy_stale", "no verified status source")
        fetched = await status_fetcher()
        if not isinstance(fetched, VerifiedStatus):
            # R11/2: a public class shape is NOT provenance — an unverified
            # callback (raw Mapping or a forged wrapper) refuses HERE, before
            # normalization, floor advance, client creation or spend.
            raise StoreRefused("policy_stale", "unverified status source")
        raw = fetched.status
        if not isinstance(raw, Mapping):
            raise StoreRefused("policy_stale", "status fetch returned no mapping")
        law = _normalize_accepted_status(raw, owner, store)
        if not store.current_generation_is_live(owner):
            raise StoreRefused("policy_stale", "owner generation lost during the status read")
        # The FULL law has passed (origin seal, identity, UNIQUE eligible
        # tenant/endpoint, pinned selector, committed-consent currency):
        # advance BOTH accepted floors in ONE atomic transaction — a refused
        # candidate leaves every floor untouched.
        receipt_pre = store.owner_consent_receipt(owner)
        if receipt_pre is None:
            raise StoreRefused("policy_stale", "no committed consent receipt for the owner")
        if not _consent_receipt_is_current(
            receipt_pre["consent_version"],
            law.consent_served_version,
            law.consent_accepted_versions,
        ) or not receipt_pre.get("consent_ref"):
            raise StoreRefused("policy_stale", "consent receipt is not the served version")
        store.advance_text_floors(
            owner,
            [
                ("guidance", law.guidance_version, law.guidance_sha),
                ("consent", law.consent_served_version, law.consent_sha),
            ],
        )
        # R12/O1: re-verify the receipt AFTER the floor advance and before
        # mint/dispatch. A receipt rotated between the law check and the
        # advance would otherwise mint a capture whose served-vs-current
        # consent versions disagree — its future reuse would refuse, but
        # the CURRENT run would already have proceeded on stale consent.
        receipt = store.owner_consent_receipt(owner)
        if (
            receipt is None
            or receipt_pre["consent_version"] != receipt["consent_version"]
            or receipt_pre.get("consent_ref") != receipt.get("consent_ref")
            or not _consent_receipt_is_current(
                receipt["consent_version"],
                law.consent_served_version,
                law.consent_accepted_versions,
            )
        ):
            raise StoreRefused("policy_stale", "consent receipt changed across the floor advance")
        capture = AcceptedCapture(
            _CAPTURE_TOKEN,
            version=law.version,
            template_hash=law.template_hash,
            owner_frame=_full_owner_frame(owner),
            captured_at=time.time(),
            credential_id=owner.credential_id,
            account_id=owner.account_id,
            hub=owner.canonical_base_url,
            owner_generation=owner.enrollment_generation,
            consent_version=receipt["consent_version"],
            consent_ref=receipt["consent_ref"],
            consent_served_version=law.consent_served_version,
            consent_accepted_versions=law.consent_accepted_versions,
            guidance_version=law.guidance_version,
            guidance_sha=law.guidance_sha,
            consent_sha=law.consent_sha,
            source="sdk-verified",
        )
    if capture is None:  # both branches above either set it or raised
        raise StoreRefused("policy_stale", "no verified status source")
    version, template_hash = capture.version, capture.template_hash
    # OWNER PIN (R8/Q3): the awaited fetch may have raced an AuthStore
    # credential-row switch; the owner captured BEFORE the await must still
    # be the live, resolved owner, or nothing publishes and nothing
    # dispatches under the old identity.
    live = store.resolve_owner(
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        canonical_base_url=owner.canonical_base_url,
    )
    if live != owner or not store.current_generation_is_live(owner):
        raise StoreRefused("policy_stale", "owner identity moved while the status was awaited")
    # ONE capture: provider/model/approvals come from the SAME resolver the
    # factory uses, and the canonical revision DIGEST binds version, hash,
    # provider, model and approved map — so a same-version different-SHA
    # status, or a same-model provider/map change, advances the authority
    # and can never reuse an old proof (R7/Q3).
    revision = (
        "accepted:"
        + hashlib.sha256(
            (
                f"{version}|{template_hash}|{selection.provider}|{selection.model_id}|"
                + ",".join(sorted(selection.approved_identities))
            ).encode("utf-8")
        ).hexdigest()[:32]
    )
    current = store.current_screening_policy(owner)
    expected = (
        version,
        template_hash,
        selection.model_id,
        revision,
    )
    row = (
        None
        if current is None
        else (
            current["version"],
            current["template_sha256"],
            current["default_model"],
            current["revision"],
        )
    )
    if row != expected:
        try:
            store.publish_screening_policy(
                owner,
                version=version,
                template_sha256=template_hash,
                default_model=selection.model_id,
                revision=revision,
                expected_epoch=None if current is None else int(current["epoch"]),
                create_only=current is None,
            )
        except StoreRefused:
            # A competing publisher moved authority; screen nothing rather
            # than admit against a stale view (the cycle's own caller holds
            # the eligibility decision).
            return None
    policy = _CyclePolicy(
        screening_version=version,
        template_sha256=template_hash,
        default_model=selection.model_id,
        admission_revision=revision,
        approved_serving_identities=tuple(sorted(selection.approved_identities)),
        provider=selection.provider,
    )
    cycle = AccountScreeningCycle(store=store, owner=owner, holder=holder, ttl=ttl)

    async def _screen(item: "Mapping[str, Any]") -> Any:
        message_block = item.get("message")
        message_id = str(message_block.get("id")) if isinstance(message_block, Mapping) else ""
        runner = make_screening_runner(
            config_dir=config_dir,
            store=store,
            owner=owner,
            message_id=message_id,
            auth_store=auth_store,
            settings=settings,
            # The SAME capture this composition resolved — never a second
            # resolve that could certify a different config (R8/Q3).
            selection=selection,
        )
        if runner is None:
            return None
        return await screen_message_async(
            owner,
            item,
            config_dir=config_dir,
            # _CyclePolicy is READ-ONLY (frozen): the protocol's attribute
            # members are writable, so the read-only side is cast rather
            # than loosened — the pipeline only ever reads these fields and
            # re-captures them into its own immutable snapshot.
            policy=cast("ScreeningPolicy", policy),
            runner=runner,
        )

    return await cycle.run_once(items, _screen)
