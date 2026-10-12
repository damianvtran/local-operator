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
            return await work
        except asyncio.CancelledError:
            if lost["flag"]:
                # Ownership moved mid-cycle: no result, no receipt, and the
                # successor keeps the lease.
                return None
            raise
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
) -> Callable[[], Awaitable[Mapping[str, Any]]]:
    """The PRODUCTION SDK-status fetcher (R8/Q3).

    Calls the H1 client's own ``status`` with the owner's PINNED OAuth
    access — the ordinary SDK/auth path, no parallel transport and no model-
    key cascade. The RAW nested DTO is returned untouched; the caller
    normalizes it strictly and re-verifies the owner after the await.
    ``transport`` is injected only by proof rigs.
    """

    async def _fetch() -> Mapping[str, Any]:
        from local_operator.team_bridge.client import TeamBridgeClient

        client = TeamBridgeClient(
            config_dir,
            base_url=owner.canonical_base_url,
            credential_id=owner.credential_id,
            transport=transport,
        )
        access = await client.current_access()
        result = await client.status(access=access)
        return result if isinstance(result, Mapping) else {}

    return _fetch


def _owner_frame_digest(owner: Any) -> str:
    """The initiating owner frame digest — reused here so a cached status is
    bound to the exact owner that cached it (R8/Q3)."""

    from local_operator.team_bridge.store import _owner_frame_digest as _digest

    return _digest(owner)


def _normalize_accepted_status(status: "Mapping[str, Any]", owner: Any) -> tuple[str, str]:
    """Strictly normalize the RAW SDK status DTO (R8/Q3).

    The real response nests the selector under ``screening`` and carries
    eligibility as ``available``/``tenants``/``endpoint``; the LOCAL
    embedded template hash is the only template authority — server bytes are
    HASHED and compared, never executed. Every missing, unknown, disabled or
    mismatched shape is a typed refusal (``policy_stale``), never a default
    and never an arbitrary flat mapping.
    """

    from local_operator.team_bridge.screening import (
        SCREENING_TEMPLATE_V1_SHA256,
        SCREENING_VERSION_V1,
    )
    from local_operator.team_bridge.store import StoreRefused

    if not isinstance(status, Mapping):
        raise StoreRefused("policy_stale", "accepted status is not a mapping")
    if not status.get("available"):
        raise StoreRefused("policy_stale", "messaging is not available for this account")
    screening = status.get("screening")
    if not isinstance(screening, Mapping):
        raise StoreRefused("policy_stale", "status carries no screening selector")
    version = screening.get("version")
    template = screening.get("template")
    if not isinstance(version, str) or not version:
        raise StoreRefused("policy_stale", "screening version missing")
    if isinstance(template, str):
        template_bytes = template.encode("utf-8")
    elif isinstance(template, (bytes, bytearray)):
        template_bytes = bytes(template)
    else:
        raise StoreRefused("policy_stale", "screening template missing")
    if version != SCREENING_VERSION_V1:
        raise StoreRefused("policy_stale", "unknown screening version")
    digest = hashlib.sha256(template_bytes).hexdigest()
    if digest != SCREENING_TEMPLATE_V1_SHA256:
        raise StoreRefused(
            "policy_stale", "screening template bytes do not match the local selector"
        )
    # Consent/API versions: when the server sends them they must be strings
    # (a None/absent pair is not trusted as authority, but the VALUES are
    # verified upstream by the caller's own compatibility contract).
    for key in ("consent_version", "api_version"):
        if key in status and not isinstance(status.get(key), str):
            raise StoreRefused("policy_stale", f"{key} has an unsupported shape")
    tenants = status.get("tenants")
    tenant_ids: set[str] = set()
    if isinstance(tenants, Sequence) and not isinstance(tenants, (str, bytes)):
        for entry in tenants:
            if isinstance(entry, str):
                tenant_ids.add(entry)
            elif isinstance(entry, Mapping):
                tid = entry.get("tenant_id") or entry.get("id")
                if isinstance(tid, str):
                    tenant_ids.add(tid)
    if str(owner.tenant_id) not in tenant_ids:
        raise StoreRefused("policy_stale", "owner tenant is not an eligible tenant")
    endpoint = status.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise StoreRefused("policy_stale", "status carries no endpoint")
    if str(endpoint.get("id") or "") != str(owner.endpoint_id):
        raise StoreRefused("policy_stale", "status endpoint does not match the owner")
    if not (endpoint.get("active") is True or endpoint.get("state") == "active"):
        raise StoreRefused("policy_stale", "endpoint is not active")
    return version, digest


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
    accepted_status: "Mapping[str, Any] | None" = None,
    status_fetcher: "Callable[[], Awaitable[Mapping[str, Any]]] | None" = None,
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
    store = Store(config_dir)
    if status_fetcher is not None:
        fetched = await status_fetcher()
        if not isinstance(fetched, Mapping):
            raise StoreRefused("policy_stale", "status fetch returned no mapping")
        status = fetched
    elif accepted_status is not None:
        cached = accepted_status
        if not isinstance(cached, Mapping):
            raise StoreRefused("policy_stale", "accepted status is not a mapping")
        if cached.get("owner_frame") != _owner_frame_digest(owner):
            raise StoreRefused("policy_stale", "cached status provenance does not match the owner")
        status = cached
    else:
        # None is NOT authority (R8/Q3): without a status source there is
        # nothing to accept, so nothing is published and nothing dispatches.
        raise StoreRefused("policy_stale", "no accepted status source")
    version, template_hash = _normalize_accepted_status(status, owner)
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
