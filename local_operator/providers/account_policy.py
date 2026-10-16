"""Account preference within an already reachable provider credential tier.

No credential, network or route state belongs here. Cached reset forecasts are
preferences, not reservations: all relevant measured caps still gate a pick,
and missing evidence stays neutral instead of starving a newly added account.
"""

from __future__ import annotations

import logging
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, cast

from local_operator.config import DEFAULT_ACCOUNT_SELECTION_POLICY

if TYPE_CHECKING:
    from local_operator.providers.usage import UsageReport

AccountSelectionPolicy = Literal["reset-soonest", "headroom", "fair"]
ACCOUNT_SELECTION_POLICIES = ("reset-soonest", "headroom", "fair")
# A fixed-minimum bucket absorbs vendor second/report-relative jitter without
# letting a chain of near-neighbours turn minutes of difference into one tie.
RESET_TOLERANCE_MS = 60_000
RESET_REPORT_MAX_AGE_MS = 30 * 60_000
_WEEK_MS = 7 * 24 * 60 * 60_000
_FIVE_HOUR_MS = 5 * 60 * 60_000
logger = logging.getLogger(__name__)


def effective_account_selection_policy(
    settings: Mapping[str, Any] | None,
) -> AccountSelectionPolicy:
    """Resolve presence before defaults, including strict legacy compatibility.

    Materializing the canonical default before checking legacy presence would
    silently change old true/false configurations. Invalid canonical values
    deliberately select the shipped default, not an unrelated legacy override.
    """
    retry = settings.get("retry", {}) if isinstance(settings, Mapping) else {}
    if not isinstance(retry, Mapping):
        retry = {}
    for key in ("accountSelectionPolicy", "account_selection_policy"):
        if key in retry:
            value = retry[key]
            if isinstance(value, str) and value in ACCOUNT_SELECTION_POLICIES:
                return cast(AccountSelectionPolicy, value)
            logger.warning("Invalid retry.accountSelectionPolicy; using reset-soonest")
            return DEFAULT_ACCOUNT_SELECTION_POLICY
    for key in ("usageAwareAccountPick", "usage_aware_account_pick"):
        if key in retry:
            value = retry[key]
            if type(value) is bool:
                return "headroom" if value else "fair"
            logger.warning("Invalid legacy account-pick boolean; using reset-soonest")
            return DEFAULT_ACCOUNT_SELECTION_POLICY
    return DEFAULT_ACCOUNT_SELECTION_POLICY


@dataclass(frozen=True)
class ResetCandidate:
    """Non-secret ranking evidence; deadlines never imply account eligibility."""

    account_id: int
    state: Literal["healthy", "reserve", "depleted", "unknown"] = "unknown"
    weekly_ms: int | None = None
    five_hour_ms: int | None = None


def _finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reset_candidate(
    account_id: int,
    report: UsageReport | None,
    model_id: str,
    now_ms: int,
    *,
    reserve_percent: float = 10.0,
    max_age_ms: int = RESET_REPORT_MAX_AGE_MS,
) -> ResetCandidate:
    """Classify all measured overlaps, then extract only fresh useful resets.

    Older reports can conservatively retain depletion until every relevant
    measured window is still future, but may never win a reset forecast. Paid
    balances retain the existing usage_health contract, not invented periods.
    """
    from local_operator.providers.usage import usage_health

    if report is None or not _finite_number(report.fetched_at):
        return ResetCandidate(account_id)
    if report.fetched_at <= 0 or report.fetched_at > now_ms:
        return ResetCandidate(account_id)
    relevant = [
        limit
        for limit in report.limits
        if not limit.paid_fallback
        and (limit.shared or not limit.tier or limit.tier.lower() in model_id.lower())
    ]
    # A past reset makes that old amount uninformative. Do not extrapolate a
    # fresh window from its previous utilization, even in a fresh report.
    current = [
        (
            replace(limit, resets_at_ms=None)
            if limit.resets_at_ms is not None and not _finite_number(limit.resets_at_ms)
            else limit
        )
        for limit in relevant
        if limit.resets_at_ms is None
        or not _finite_number(limit.resets_at_ms)
        or limit.resets_at_ms > now_ms
    ]
    fresh = now_ms - report.fetched_at <= max_age_ms
    if not fresh:
        measured = [limit for limit in current if limit.amount.fraction() is not None]
        if (
            not measured
            or len(current) != len(relevant)
            or any(limit.resets_at_ms is None for limit in measured)
            or any(limit.paid_usable() for limit in report.limits)
        ):
            return ResetCandidate(account_id)
    # Keep the existing included-plan/explicit-paid alternative semantics. This
    # is not permission to activate extra usage or promote a credential tier.
    safe_report = replace(
        report, limits=current + [limit for limit in report.limits if limit.paid_fallback]
    )
    health = usage_health(safe_report, model_id, reserve_percent=reserve_percent, now_ms=now_ms)
    if not fresh:
        return ResetCandidate(account_id, "depleted" if health.state == "depleted" else "unknown")
    weekly: list[int] = []
    short: list[int] = []
    for limit in current:
        deadline = limit.resets_at_ms
        if not _finite_number(deadline) or deadline <= now_ms:
            continue
        # These are normalized parser window identifiers, never localized UI
        # labels. Unsupported periods remain unknown rather than being guessed
        # from primary/secondary position or a translated allowance title.
        window = limit.window.strip().lower()
        period = (
            _WEEK_MS
            if window in ("7 day", "7 days", "7d", "weekly")
            else _FIVE_HOUR_MS if window in ("5 hour", "5 hours", "5h") else None
        )
        if period is None or deadline - now_ms > period + RESET_TOLERANCE_MS:
            continue
        (weekly if period == _WEEK_MS else short).append(int(deadline))
    return ResetCandidate(
        account_id,
        health.state,
        min(weekly) if weekly else None,
        min(short) if short else None,
    )


def _spread(
    rows: Sequence[ResetCandidate],
    session_id: str | None,
    tie_order: Mapping[int, int] | None = None,
) -> list[ResetCandidate]:
    if tie_order is not None:
        return sorted(rows, key=lambda row: tie_order[row.account_id])
    ordered = sorted(rows, key=lambda row: row.account_id)
    if not ordered:
        return []
    start = zlib.crc32((session_id or "").encode("utf-8")) % len(ordered)
    return ordered[start:] + ordered[:start]


def _buckets(rows: Sequence[ResetCandidate], field: str) -> list[list[ResetCandidate]]:
    known = sorted(
        (row for row in rows if getattr(row, field) is not None),
        key=lambda row: (getattr(row, field), row.account_id),
    )
    unknown = [row for row in rows if getattr(row, field) is None]
    if not known:
        return [unknown] if unknown else []
    result: list[list[ResetCandidate]] = []
    while known:
        minimum = getattr(known[0], field)
        split = next(
            (
                i
                for i, row in enumerate(known)
                if getattr(row, field) - minimum > RESET_TOLERANCE_MS
            ),
            len(known),
        )
        result.append(known[:split])
        known = known[split:]
    # Unknown evidence is neutral beside the earliest forecast, not a perpetual
    # tail. It cannot supply a fictitious secondary deadline to win this tie.
    result[0].extend(unknown)
    return result


def reset_ranked_ids(
    rows: Sequence[ResetCandidate],
    session_id: str | None,
    *,
    tie_order: Mapping[int, int] | None = None,
) -> list[int]:
    """Return healthy/unknown first, reserve last; depleted rows cannot win.

    The caller owns credential tiers, hard blocks, exclusions, cooldowns and
    all-demoted recovery. A valid compatible success receipt is applied there
    before this preference, never encoded as a synthetic reset timestamp here.
    """
    ordered: list[ResetCandidate] = []
    for states in (("healthy", "unknown"), ("reserve",)):
        eligible = [row for row in rows if row.state in states]
        for weekly in _buckets(eligible, "weekly_ms"):
            # Weekly-unknown rows form a neutral tie pool rather than inheriting
            # the known rows' weekly forecast. Spread them at the best 5h level.
            for short in _buckets(weekly, "five_hour_ms"):
                ordered.extend(_spread(short, session_id, tie_order))
    return [row.account_id for row in ordered]
