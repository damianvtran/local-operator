"""H3 — the serialized per-account screening selector (A5, mandatory in H3).

The frozen contract requires fair, ONE-AT-A-TIME screening per account:
round-robin across senders with request-type messages ahead of notes within
each sender's turn. A parallel screening amplifier (N messages screened
concurrently, or one eager sender draining the queue) is exactly what this
prevents — the selector hands back the SINGLE next item each cycle.

Eligibility (breaker open, budget exhausted, holds) is decided by the
pipeline/store; the selector only orders ELIGIBLE items, so it never invents
new quotas and never bypasses a hold. ``last_sender`` carries the previous
cycle's sender so the rotation is persistent across calls without extra
state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

__all__ = ["select_next_screening_target"]


def _sender_of(item: Mapping[str, Any]) -> str:
    sender = item.get("sender")
    if isinstance(sender, Mapping):
        account = sender.get("account_id")
        if isinstance(account, str) and account:
            return account
    # A malformed sender must not share a rotation slot with a real one.
    return ""


def select_next_screening_target(
    items: Sequence[Mapping[str, Any]], *, last_sender: str | None = None
) -> Mapping[str, Any] | None:
    """The NEXT item to screen, or None when the account has nothing to do.

    Order: senders in a stable rotation starting AFTER ``last_sender``;
    within the chosen sender, requests (``type == "request"``) before any
    other type, preserving arrival order inside each group.
    """

    if not items:
        return None
    order: list[str] = []
    for item in items:
        sender = _sender_of(item)
        if sender not in order:
            order.append(sender)
    start = 0
    if last_sender in order:
        start = order.index(last_sender) + 1
    for offset in range(len(order)):
        sender = order[(start + offset) % len(order)]
        candidates = [item for item in items if _sender_of(item) == sender]
        requests = [item for item in candidates if item.get("type") == "request"]
        others = [item for item in candidates if item.get("type") != "request"]
        return requests[0] if requests else others[0]
    return None
