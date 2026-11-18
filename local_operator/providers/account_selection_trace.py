"""Observational account-selection evidence for an owner's grant audit.

This module is stdlib-only, including when imported by the broker vocabulary.
It neither orders accounts nor reads credentials/usage. Existing selector branches
record their result into a task-local collector only while a lend opts in. A
row's ordinal is within its winning eligible tier, NEVER a global best across
OAuth, login-key, environment and fallback tiers. No candidate labels are kept.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Protocol, TypeVar

BASES = frozenset(
    {
        "affinity",
        "affinity_fallback",
        "request_pick",
        "request_fallback",
        "reset_rank",
        "headroom_rank",
        "fair_hash",
        "round_robin",
        "demotion_preference",
    }
)


class _Row(Protocol):
    provider: str
    id: int
    created_at: int


R = TypeVar("R", bound=_Row)


def _key(row: _Row) -> tuple[str, int, int]:
    # Refresh may re-read the row into a different Python object. Its creation
    # stamp, not object identity or a mutable current-account pointer, binds it.
    return row.provider, row.id, row.created_at


@dataclass(frozen=True)
class SelectionExplanation:
    """Owner-local display metadata; excluded from wire, equality and keys.

    ``reason`` is a closed scalar ``basis:ordinal`` or ``unranked``. Audit's
    existing sanitizer bounds strings; do not widen it to accept nested payloads.
    ``label`` may be an email, intentionally the existing owner's display label,
    never a new identity dictionary, candidate list or credential material.
    """

    label: str | None = None
    reason: str = "unranked"

    def audit_reason(self) -> str:
        """Reject arbitrary strings even if a future producer bypasses the trace.

        An ordinal cannot exceed Python's maximum list length, so the encoded
        value is naturally bounded without a new routing cap or setting.
        """
        if not isinstance(self.reason, str):
            return "unranked"
        basis, separator, rank = self.reason.partition(":")
        if (
            basis in BASES
            and separator
            and rank.isascii()
            and rank.isdecimal()
            and len(rank) <= len(str(sys.maxsize))
            and 0 < int(rank) <= sys.maxsize
        ):
            return f"{basis}:{int(rank)}"
        return "unranked"


@dataclass
class _Trace:
    ranks: dict[tuple[str, int, int], tuple[str, int]] = field(default_factory=dict)


_current: ContextVar[_Trace | None] = ContextVar("account_selection_audit_trace", default=None)


@contextmanager
def capture_selection(*, enabled: bool = True) -> Iterator[None]:
    """One observation; nested scopes and task cancellation restore the parent."""
    if not enabled:
        yield
        return
    token = _current.set(_Trace())
    try:
        yield
    finally:
        _current.reset(token)


def selection_active() -> bool:
    """An owner opted in; each getter must still own its own mutable collector."""
    return _current.get() is not None


def record_order(rows: list[R], basis: str, *, fallback_basis: str | None = None) -> list[R]:
    """Return the EXACT list supplied, recording no new routing decision."""
    trace = _current.get()
    if trace is not None and basis in BASES:
        for ordinal, row in enumerate(rows, 1):
            # Only the first row is the honoured anchor/request pick. If it
            # fails, do not imply the served sibling was itself that anchor.
            why = (
                fallback_basis
                if ordinal > 1 and fallback_basis is not None and fallback_basis in BASES
                else basis
            )
            trace.ranks[_key(row)] = why, ordinal
    return rows


def record_demotion(rows: list[R]) -> list[R]:
    """Observe the final stable partition, without querying demotions again."""
    trace = _current.get()
    if trace is not None:
        for ordinal, row in enumerate(rows, 1):
            previous = trace.ranks.get(_key(row))
            if previous is not None:
                basis, before = previous
                trace.ranks[_key(row)] = (
                    "demotion_preference" if before != ordinal else basis,
                    ordinal,
                )
    return rows


def selected_reason(row: _Row | None) -> str | None:
    """None means no observer; unranked means an observed non-ranking path."""
    trace = _current.get()
    if trace is None:
        return None
    pick = trace.ranks.get(_key(row)) if row is not None else None
    return f"{pick[0]}:{pick[1]}" if pick is not None else "unranked"
