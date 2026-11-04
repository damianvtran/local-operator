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
    #: When a ``depleted`` verdict stops holding, epoch ms — set ONLY when a FRESH
    #: report proves it: the LATEST reset among the windows that are themselves
    #: depleted (every one of them must reopen, so the earliest would be a guess
    #: an account can still refuse). ``None`` for every other state, for a stale
    #: report (usage may have grown since, so its resets are a lower bound) and
    #: for a depleted window with no stated reset. Read by the mesh broker's
    #: ``quota_blocked`` remainder (``AuthStore.quota_eligibility``); never by
    #: ranking, which keeps ordering on the two reset fields above.
    reopens_at_ms: int | None = None


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
    """Classify all measured overlaps, then extract the still-true resets.

    Older reports can conservatively retain depletion until every relevant
    measured window is still future; a non-depleted older report stays
    "unknown" but keeps its still-future reset deadlines for ranking (never a
    reopening). Paid balances retain the existing usage_health contract, not
    invented periods.
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
    weekly, short = _reset_deadlines(current, now_ms, now_ms if fresh else int(report.fetched_at))
    if not fresh:
        # STALE DEADLINES STILL RANK (2026-11-04 incident). With the shipped
        # ``retry.usageAwareFallback=false`` the only Anthropic evidence is the
        # warmer's row, refreshed only while a TUI warmer runs or someone opens
        # /usage, so headless sessions routinely read it past ``max_age_ms``.
        # Returning a deadline-less "unknown" then put every non-depleted account
        # in ONE neutral bucket and crc32(session) % N spread new sessions: an
        # account resetting 13h LATER took ~1/3 of them (255/270 measured
        # receipts were exactly the hash pick). A window's reset instant is fixed
        # when the window opens and cannot move until it elapses, so age makes
        # the UTILIZATION unreliable, not the deadline: a finite deadline still
        # in the future, and within its period of the report's own fetch, is
        # still true now. The gates above keep everything that is NOT provable:
        # an elapsed window (``current`` shrank) or a reset-less measured window
        # leaves the whole report neutral, and a paid balance can move on its
        # own. The verdict stays conservative — never "healthy" (it may have
        # filled since; a reactive 429 still fails over), and never a stated
        # reopening, since a stale depletion's resets are only a lower bound.
        state = "depleted" if health.state == "depleted" else "unknown"
        if state == "depleted":
            return ResetCandidate(account_id, state)
        return ResetCandidate(
            account_id,
            state,
            min(weekly) if weekly else None,
            min(short) if short else None,
        )
    return ResetCandidate(
        account_id,
        health.state,
        min(weekly) if weekly else None,
        min(short) if short else None,
        (
            _proven_reopening(safe_report, current, model_id, reserve_percent, now_ms)
            if health.state == "depleted"
            else None
        ),
    )


def _reset_deadlines(
    current: Sequence[Any], now_ms: int, anchor_ms: int
) -> tuple[list[int], list[int]]:
    """Weekly and five-hour deadlines that are finite, future and plausible.

    ``anchor_ms`` is the instant the period sanity bound is measured from: now
    for a fresh report (unchanged), the report's own ``fetched_at`` for a stale
    one. A window opened at most one period before its fetch, so a deadline more
    than ``period + RESET_TOLERANCE_MS`` past the fetch is garbled, however far
    from now it is. That same bound is what limits how stale a report may be and
    still rank: once the age exceeds the period (+ tolerance), every plausible
    deadline it carried is in the past and drops out — no second age constant.
    """
    weekly: list[int] = []
    short: list[int] = []
    for limit in current:
        deadline = limit.resets_at_ms
        if not _finite_number(deadline) or deadline <= now_ms:
            continue
        period = _window_period_ms(limit)
        if period is None or deadline - anchor_ms > period + RESET_TOLERANCE_MS:
            continue
        (weekly if period == _WEEK_MS else short).append(int(deadline))
    return weekly, short


def _proven_reopening(
    report: UsageReport,
    current: Sequence[Any],
    model_id: str,
    reserve_percent: float,
    now_ms: int,
) -> int | None:
    """The latest reset among the individually depleted windows, or ``None``.

    Each window is judged by the SAME reduction the verdict used
    (:func:`usage_health` over that one window), so this adds no second idea of
    "spent". A depleted window with no finite future reset makes the reopening
    unknowable, and so does a verdict no single window explains: both answer
    ``None`` rather than a time nobody stated.
    """
    from local_operator.providers.usage import usage_health

    deadlines: list[int] = []
    for limit in current:
        alone = replace(report, limits=[limit])
        if usage_health(alone, model_id, reserve_percent=reserve_percent, now_ms=now_ms).state != (
            "depleted"
        ):
            continue
        deadline = limit.resets_at_ms
        if not _finite_number(deadline) or deadline <= now_ms:
            return None
        # THE RANKING PATH'S SANITY WINDOW, applied here too (#3108 review m2): a
        # reset further away than its own window's period (plus the tolerance) is
        # garbled evidence, and it must not become the owner's retry figure. An
        # unrecognised window has no period to judge by, so it proves nothing.
        period = _window_period_ms(limit)
        if period is None or deadline - now_ms > period + RESET_TOLERANCE_MS:
            return None
        deadlines.append(int(deadline))
    return max(deadlines) if deadlines else None


def _window_period_ms(limit: Any) -> int | None:
    """The period a normalized window id names, or ``None`` for an unsupported one.

    These are parser window identifiers, never localized UI labels; anything else
    stays unknown rather than being guessed from position or a translated title.
    """
    window = str(getattr(limit, "window", "") or "").strip().lower()
    if window in ("7 day", "7 days", "7d", "weekly"):
        return _WEEK_MS
    if window in ("5 hour", "5 hours", "5h"):
        return _FIVE_HOUR_MS
    return None


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
