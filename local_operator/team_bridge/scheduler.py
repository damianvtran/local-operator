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

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Mapping

__all__ = [
    "AccountScreeningCycle",
    "canonical_sender",
    "canonical_type",
    "select_next_screening_target",
]


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
    """One account's serialized screening turn.

    ``run_once`` claims the account's native lease (conditional update, one
    transaction — two racing processes can never both win), selects the ONE
    next target with the persisted cursor, ADVANCES the cursor before dispatch
    (so a crash cannot starve the same sender), awaits the public async
    screener and releases the lease in ``finally`` on success, failure AND
    cancellation. Eligibility (breaker/budget/held retry) is decided by the
    caller before items reach here — this class never invents quotas and never
    bypasses a hold.
    """

    def __init__(self, *, store: Any, owner: Any, holder: str, ttl: float = 30.0) -> None:
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

        if not self._store.claim_screen_turn(
            self._owner, holder=self._holder, now=time.time(), ttl=self._ttl
        ):
            return None
        try:
            schedule = self._store.screen_schedule(self._owner) or {}
            target = select_next_screening_target(items, last_sender=schedule.get("last_sender"))
            if target is None:
                return None
            self._store.set_screen_cursor(self._owner, canonical_sender(target))
            return await screen(target)
        finally:
            self._store.release_screen_turn(self._owner, holder=self._holder)
