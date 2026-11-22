"""Turn-end ECONOMIC compaction: compact when the pass pays for itself.

WHY. The size trigger fires at ``min(0.8 * window, 400k)`` and the idle pass
(``compaction/idle.py``) fires only once a session has parked with no armed
wake inside its cache TTL. An always-on, high-wake session — the chief of
staff is the measured case — satisfies neither in a useful way: its armed
wakes keep the cache warm forever (idle gate ``wake_keeps_cache_warm``; 0.35%
of its 570 measured request gaps exceeded an hour), so it never parks, and it
reaches the size trigger only after hundreds of requests each re-reading
~250-400k of context. Every one of those reads is billed, and every cold
rewrite (restart, route change) re-writes the whole thing.

THE RULE is the economic-order-quantity balance between a pass's one-off cost
and the per-request carrying cost of the context it would remove. Over a cycle
in which the context grows from the residual ``R`` to ``C`` at ``g`` tokens
per request, the cost per request is::

    (p_r + f * p_w) * (C + R) / 2  +  (R * p_w + pass_cost) * g / (C - R)

with ``p_r``/``p_w`` the cache read/write prices per token and ``f`` the
fraction of requests that miss the cache anyway (restarts, route changes).
That is minimized at a gain of::

    delta* = sqrt(2 * (R * p_w + pass_cost) * g / (p_r + f * p_w))

so a pass is due when ``C - R >= delta*``. At that gain the pass pays for
itself within the first half of the cycle it starts (``delta* / 2g``
requests). It is the operator's "compact when the savings over the coming
requests exceed the pass's cost", made exact.

COLD SHORTCUT. When the next request will miss the cache anyway (its expected
arrival is past the cache expiry), the pass's own rewrite REPLACES the
rewrite that request would pay, so the only remaining cost is ``pass_cost``
and a pass is due on any gain worth having (``>= G_min``).

ANTI-SAWTOOTH. Every pass is bounded on three sides, so an always-on session
cannot fall into compact-regrow-compact:

- the projected residual must be at most ``rho`` of the context
  (:data:`ECONOMIC_MAX_RESIDUAL_RATIO`): a pass that keeps most of what it
  started with buys a full rewrite for little relief;
- the context must have grown ``G_min`` past the larger of the last pass's
  MEASURED after-size and the last economic decline, and the pass must remove
  at least ``G_min`` itself;
- after a low-yield pass (measured residual ratio above ``rho``) the growth
  requirement doubles per consecutive low-yield pass, capped at ``T_eff / 2``,
  and resets on a pass that lands within ``rho``.

Hence at most one economic pass per ``G_min`` of growth, and no two
consecutive turn ends can both compact. The session re-checks the residual
EXACTLY (a real cut plan) before committing, mirroring idle's cheap-arm /
exact-fire split.

This module is PURE: no clock, no session, no I/O. ``Session`` gathers the
inputs (usage rows, prices, transcript markers) and calls
:func:`economic_compaction_due`; tests drive the same function with tables.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "COMPACTION_ECONOMIC_DECLINED_TYPE",
    "COMPACTION_ECONOMIC_PASS_TYPE",
    "ECONOMIC_ARCHIVE_ALLOWANCE_TOKENS",
    "ECONOMIC_COLD_READ_FRACTION",
    "ECONOMIC_MAX_RESIDUAL_RATIO",
    "ECONOMIC_MIN_SAMPLES",
    "ECONOMIC_RATE_WINDOW",
    "EconomicDecision",
    "HIGH_WAKE_MIN_REQUESTS",
    "HIGH_WAKE_WINDOW_MS",
    "UsageStats",
    "backoff_growth",
    "delta_star",
    "economic_compaction_due",
    "is_high_wake_rate",
    "low_yield_streak",
    "usage_stats",
]

#: ``rho``: the largest projected residual / context ratio a discretionary pass
#: may leave. The fleet's measured residual/before median is 0.38 (snapcompact)
#: and 0.33 (LLM summary) over n=504 passes in 30 sessions; the chief of staff's
#: preserved-turn-heavy passes landed at 0.62-0.84. 0.6 sits above every
#: healthy pass and below every one of those low-yield ones.
ECONOMIC_MAX_RESIDUAL_RATIO = 0.6

#: Usage rows the rates (growth, cold-event fraction, gap) are measured over:
#: ~7 h of the chief of staff's measured 28 requests/h. Long enough that one
#: burst does not decide the rate, short enough to follow a change of cadence.
ECONOMIC_RATE_WINDOW = 200

#: Minimum rows before the cold-event fraction ``f`` is trusted; below it ``f``
#: is 0, which only makes the warm rule more conservative (a larger delta*).
ECONOMIC_MIN_SAMPLES = 50

#: A request is a COLD event when it read less than this fraction of its own
#: context from the cache. A warm request reads everything but the newest
#: message (~99%); a cold one reads only the system+tools prefix (the chief of
#: staff's cold requests read 34,947 of ~250-400k, i.e. <15%). One half
#: separates the two populations with a wide margin either side.
ECONOMIC_COLD_READ_FRACTION = 0.5

#: CHEAP admission hint only, never an upper bound or final authorization.
#: The chief of staff's measured marker was 31,378 ruler tokens. The exact
#: recheck prepares/reuses the actual snapcompact replay (including frames),
#: or reserves the text strategy's configured generation bound, plus carry.
ECONOMIC_ARCHIVE_ALLOWANCE_TOKENS = 32_000

#: HIGH-WAKE classification: at least this many turn requests in the trailing
#: :data:`HIGH_WAKE_WINDOW_MS`. 90 in 6 h is 15 requests/h, the fleet's p75
#: (14.67/h) of 3,048 active 6-hour session windows over 14 days of
#: ``analytics.db calls`` (median 6.17, p90 27.0, max 111.5; 429 top-level
#: sessions; measured 2026-11-21). The chief of staff's median window is 26.3/h.
#: The design's first proposal (6/h) is the fleet MEDIAN and would have tagged
#: half of all active windows, i.e. ordinary busy tool loops, as high-wake.
HIGH_WAKE_MIN_REQUESTS = 90
HIGH_WAKE_WINDOW_MS = 6 * 3600 * 1000

#: Transcript custom types (``append_custom``; never LLM context), the durable
#: half of the anti-thrash rule exactly like ``compaction_idle_*``: a restarted
#: runtime re-derives the growth baseline from them, not from process memory.
COMPACTION_ECONOMIC_PASS_TYPE = "compaction_economic_pass"
COMPACTION_ECONOMIC_DECLINED_TYPE = "compaction_economic_declined"


@dataclass(frozen=True, slots=True)
class EconomicDecision:
    """The predicate's verdict plus the numbers it judged, for logs and journals."""

    due: bool
    #: ``"due"`` or the first failed gate's machine token.
    reason: str
    context_tokens: int = 0
    projected_residual: int = 0
    delta_star: int = 0
    growth_tokens: int = 0
    growth_needed_tokens: int = 0
    min_gain_tokens: int = 0
    #: Requests until the pass has paid for itself at the warm rate.
    payback_requests: float = 0.0
    cold_next: bool = False


@dataclass(frozen=True, slots=True)
class UsageStats:
    """Rates measured over a session's recent usage rows."""

    #: Rows considered (at most :data:`ECONOMIC_RATE_WINDOW`).
    samples: int = 0
    #: Mean context growth per request since the newest shrink, floored at 1.
    growth_per_request: int = 1
    #: Fraction of requests that missed the cache (``f``); 0 below n_min.
    cold_event_rate: float = 0.0
    #: Median gap between consecutive requests, ms (0 when unknown).
    median_gap_ms: int = 0
    #: Smallest positive cache read among cold requests — the system+tools
    #: prefix every request shares (0 when no cold request read anything).
    prefix_tokens: int = 0


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def usage_stats(rows: Sequence[Mapping[str, Any]], *, since_shrink: int = 0) -> UsageStats:
    """Measure ``g``, ``f``, the median gap and the shared prefix.

    ``rows`` are usage payloads oldest first (at most the rate window);
    ``since_shrink`` is how many of the NEWEST rows were recorded after the
    newest compaction/prune, the only ones whose growth is meaningful (a
    pre-pass row still carries the size the pass removed).
    """
    sized = [row for row in rows if _int(row.get("context_tokens")) > 0]
    samples = len(sized)
    cold = 0
    prefix = 0
    for row in sized:
        context = _int(row.get("context_tokens"))
        read = _int(row.get("cache_read_tokens"))
        if read < ECONOMIC_COLD_READ_FRACTION * context:
            cold += 1
            if read > 0:
                prefix = read if prefix == 0 else min(prefix, read)
    rate = cold / samples if samples >= ECONOMIC_MIN_SAMPLES else 0.0
    tail = sized[-since_shrink:] if since_shrink > 0 else []
    growth = 1
    if len(tail) >= 2:
        first = _int(tail[0].get("context_tokens"))
        last = _int(tail[-1].get("context_tokens"))
        growth = max(1, (last - first) // (len(tail) - 1))
    stamps = [_int(row.get("at_ms")) for row in sized if _int(row.get("at_ms")) > 0]
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]) if b >= a)
    median_gap = gaps[len(gaps) // 2] if gaps else 0
    return UsageStats(
        samples=samples,
        growth_per_request=growth,
        cold_event_rate=rate,
        median_gap_ms=median_gap,
        prefix_tokens=prefix,
    )


def is_high_wake_rate(stamps_ms: Sequence[int], *, now_ms: int) -> bool:
    """Whether at least :data:`HIGH_WAKE_MIN_REQUESTS` requests fell in the window."""
    floor = now_ms - HIGH_WAKE_WINDOW_MS
    return sum(1 for stamp in stamps_ms if floor <= stamp <= now_ms) >= HIGH_WAKE_MIN_REQUESTS


def delta_star(
    *,
    residual: int,
    growth_per_request: int,
    p_read: float,
    p_write: float,
    cold_event_rate: float = 0.0,
    pass_cost: float = 0.0,
) -> int:
    """The EOQ gain ``sqrt(2 * (R * p_w + pass_cost) * g / (p_r + f * p_w))``."""
    carrying = p_read + max(0.0, cold_event_rate) * p_write
    if carrying <= 0:
        return 0
    setup = max(0, residual) * p_write + max(0.0, pass_cost)
    return int(math.sqrt(2.0 * setup * max(1, growth_per_request) / carrying))


def backoff_growth(min_gain: int, streak: int, t_eff: int) -> int:
    """Growth required after ``streak`` consecutive low-yield passes.

    ``G_min * 2**streak``, capped at ``T_eff / 2`` (never below ``G_min``):
    doubling makes a session whose passes keep landing high wait
    geometrically longer, and the cap keeps the economic pass meaningfully
    ahead of the size trigger it is meant to precede.
    """
    base = max(1, int(min_gain))
    if streak <= 0:
        return base
    ceiling = max(base, int(t_eff) // 2) if t_eff > 0 else base * (2 ** min(streak, 16))
    return min(ceiling, base * (2 ** min(streak, 16)))


def low_yield_streak(yields: Sequence[tuple[int, int]], *, ratio: float) -> int:
    """Consecutive NEWEST passes whose measured ``after / before`` exceeded ``ratio``.

    ``yields`` is ``(before, after)`` per pass, newest first, in provider
    tokens. A pair with a non-positive ``before`` cannot be judged and stops
    the count (never treated as low-yield on a guess).
    """
    streak = 0
    for before, after in yields:
        if before <= 0 or after <= 0:
            break
        if after / before <= ratio:
            break
        streak += 1
    return streak


def economic_compaction_due(
    *,
    enabled: bool,
    strategy: str,
    is_subagent: bool,
    priced: bool,
    context_tokens: int,
    min_context_tokens: int,
    projected_residual: int | None,
    baseline_after_tokens: int | None,
    declined_at_tokens: int | None,
    growth_per_request: int,
    p_read: float,
    p_write: float,
    cold_event_rate: float,
    cold_next: bool,
    pass_cost: float,
    min_gain_tokens: int,
    max_residual_ratio: float,
    low_yield_streak: int,
    t_eff: int,
) -> EconomicDecision:
    """Whether a turn-end economic pass is due. Gates, in order:

    1. ``enabled``, ``strategy != "off"``, not a subagent (a child's context
       dies with its job), and the route PRICED: unknown prices fail closed,
       leaving only the size and idle triggers.
    2. ``C >= min_context_tokens`` (the caller's floor; see Session).
    3. Growth since ``max(A_M, D)`` at least ``G_min``, doubled per
       consecutive low-yield pass (:func:`backoff_growth`). ``None`` baseline
       (a pass not yet measured) is zero growth.
    4. A residual estimate exists and ``R / C <= rho``.
    5. Gain ``C - R >= G_min``.
    6. EOQ: ``C - R >= delta*``, or the cold shortcut (``cold_next``) where
       the pass's dollar saving must merely exceed ``pass_cost``.
    """
    context = int(context_tokens)
    g_min = max(1, int(min_gain_tokens))

    def verdict(reason: str, **extra: Any) -> EconomicDecision:
        return EconomicDecision(
            due=reason == "due",
            reason=reason,
            context_tokens=context,
            min_gain_tokens=g_min,
            cold_next=bool(cold_next),
            **extra,
        )

    if not enabled or strategy == "off":
        return verdict("disabled")
    if is_subagent:
        return verdict("subagent")
    if not priced or p_read <= 0 or p_write <= 0:
        return verdict("unpriced")
    if context < max(0, int(min_context_tokens)):
        return verdict("below_floor")
    needed = backoff_growth(g_min, low_yield_streak, t_eff)
    if baseline_after_tokens is None:
        return verdict("no_growth_baseline", growth_needed_tokens=needed)
    growth = context - max(int(baseline_after_tokens), int(declined_at_tokens or 0))
    if growth < needed:
        return verdict("insufficient_growth", growth_tokens=growth, growth_needed_tokens=needed)
    if projected_residual is None or projected_residual <= 0:
        return verdict("no_residual_estimate", growth_tokens=growth, growth_needed_tokens=needed)
    residual = int(projected_residual)
    common = {
        "projected_residual": residual,
        "growth_tokens": growth,
        "growth_needed_tokens": needed,
    }
    if context <= 0 or residual / context > max_residual_ratio:
        return verdict("residual_ratio", **common)
    gain = context - residual
    if gain < g_min:
        return verdict("min_gain", **common)
    star = delta_star(
        residual=residual,
        growth_per_request=growth_per_request,
        p_read=p_read,
        p_write=p_write,
        cold_event_rate=cold_event_rate,
        pass_cost=pass_cost,
    )
    payback = star / (2.0 * max(1, growth_per_request))
    if cold_next:
        # The next request rewrites C at p_w anyway; after a pass it rewrites
        # R instead. The pass is worth it once that saving beats its own cost.
        if gain * p_write <= max(0.0, pass_cost):
            return verdict("below_eoq", delta_star=star, payback_requests=payback, **common)
        return verdict("due", delta_star=star, payback_requests=payback, **common)
    if gain < star:
        return verdict("below_eoq", delta_star=star, payback_requests=payback, **common)
    return verdict("due", delta_star=star, payback_requests=payback, **common)
