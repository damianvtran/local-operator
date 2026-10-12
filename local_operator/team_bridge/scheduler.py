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

import dataclasses
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
        try:
            schedule = self._store.screen_schedule(self._owner) or {}
            target = select_next_screening_target(items, last_sender=schedule.get("last_sender"))
            if target is None:
                return None
            self._store.set_screen_cursor(self._owner, canonical_sender(target))
            with turn_binding(self._store._account_scope(self._owner), turn):
                return await screen(target)
        finally:
            self._store.release_screen_turn(self._owner, turn=turn)


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
    from local_operator.team_bridge.screening import (
        SCREENING_TEMPLATE_V1_SHA256,
        SCREENING_VERSION_V1,
        screen_message_async,
    )
    from local_operator.team_bridge.store import Store, StoreRefused

    selection = resolve_default_selection(config_dir)
    if selection is None:
        return None
    store = Store(config_dir)
    revision = "accepted-status-v1"
    current = store.current_screening_policy(owner)
    expected = (
        SCREENING_VERSION_V1,
        SCREENING_TEMPLATE_V1_SHA256,
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
                version=SCREENING_VERSION_V1,
                template_sha256=SCREENING_TEMPLATE_V1_SHA256,
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
        screening_version=SCREENING_VERSION_V1,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
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
