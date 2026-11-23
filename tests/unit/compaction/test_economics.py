"""Pure tests for the turn-end economic compaction predicate.

Numbers are the chief of staff's measured economics (design §1.4): cache read
$0.20/M, 1 h cache write $8/M (p_w/p_r = 40), ~525 tokens of growth per
request, cold-event rate f = 0.025, median request gap 16 s / p95 460 s.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from local_operator.compaction import economics
from local_operator.compaction.economics import (
    ECONOMIC_MAX_RESIDUAL_RATIO,
    backoff_growth,
    delta_star,
    economic_compaction_due,
    is_high_wake_rate,
    low_yield_streak,
    usage_stats,
)

P_READ = 0.20e-6
P_WRITE = 8.0e-6
P_INPUT = 4.0e-6
G = 525
F = 0.025
G_MIN = 60_000
T_EFF = 400_000


#: A small context whose gain sits under delta*: R/C = 0.6, gain 24k.
SMALL: dict[str, Any] = {
    "context_tokens": 60_000,
    "projected_residual": 36_000,
    "min_gain_tokens": 10_000,
    "baseline_after_tokens": 40_000,
}


def due(**overrides: Any) -> economics.EconomicDecision:
    base: dict[str, Any] = {
        "enabled": True,
        "strategy": "snapcompact",
        "is_subagent": False,
        "priced": True,
        "context_tokens": 330_000,
        "min_context_tokens": 0,
        "projected_residual": 115_000,
        "baseline_after_tokens": 115_000,
        "declined_at_tokens": None,
        "growth_per_request": G,
        "p_read": P_READ,
        "p_write": P_WRITE,
        "cold_event_rate": F,
        "cold_next": False,
        "pass_cost": 0.0,
        "min_gain_tokens": G_MIN,
        "max_residual_ratio": ECONOMIC_MAX_RESIDUAL_RATIO,
        "low_yield_streak": 0,
        "t_eff": T_EFF,
    }
    base.update(overrides)
    return economic_compaction_due(**base)


@pytest.mark.parametrize(
    ("residual", "expected"),
    [(247_000, 72_000), (115_000, 49_000)],
)
def test_delta_star_matches_the_design_table(residual, expected):
    star = delta_star(
        residual=residual, growth_per_request=G, p_read=P_READ, p_write=P_WRITE, cold_event_rate=F
    )
    assert abs(star - expected) < 1_000, star


def test_delta_star_grows_with_pass_cost_and_shrinks_with_cold_rate():
    plain = delta_star(residual=115_000, growth_per_request=G, p_read=P_READ, p_write=P_WRITE)
    costly = delta_star(
        residual=115_000, growth_per_request=G, p_read=P_READ, p_write=P_WRITE, pass_cost=1.0
    )
    colder = delta_star(
        residual=115_000, growth_per_request=G, p_read=P_READ, p_write=P_WRITE, cold_event_rate=0.2
    )
    assert costly > plain > colder > 0


def test_due_at_aida_shape_with_capped_residual():
    decision = due()
    assert decision.due and decision.reason == "due"
    assert decision.delta_star == pytest.approx(49_000, abs=1_000)
    # The invariant: at delta* the pass pays back in delta*/(2g) requests.
    assert decision.payback_requests == pytest.approx(decision.delta_star / (2 * G))


def test_below_eoq_when_gain_is_under_delta_star():
    # residual 36k (ratio 0.6), delta* ~27.5k, gain 24k < delta* (>= a small G_min)
    decision = due(**SMALL)
    assert not decision.due and decision.reason == "below_eoq"


def test_cold_shortcut_fires_on_min_gain_alone():
    assert due(**SMALL).reason == "below_eoq"
    cold = due(cold_next=True, **SMALL)
    assert cold.due and cold.cold_next


def test_cold_shortcut_still_refuses_a_pass_costing_more_than_it_saves():
    # gain 24k * $8/M = $0.19 of avoided rewrite; a $1 pass is a loss.
    assert due(pass_cost=1.0, cold_next=True, **SMALL).reason == "below_eoq"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"enabled": False}, "disabled"),
        ({"strategy": "off"}, "disabled"),
        ({"is_subagent": True}, "subagent"),
        ({"priced": False}, "unpriced"),
        ({"p_read": 0.0}, "unpriced"),
        ({"min_context_tokens": 400_000}, "below_floor"),
        ({"baseline_after_tokens": None}, "no_growth_baseline"),
        ({"baseline_after_tokens": 300_000}, "insufficient_growth"),
        ({"declined_at_tokens": 300_000}, "insufficient_growth"),
        ({"projected_residual": None}, "no_residual_estimate"),
        ({"projected_residual": 250_000}, "residual_ratio"),
        (
            {"context_tokens": 100_000, "projected_residual": 50_000, "baseline_after_tokens": 0},
            "min_gain",
        ),
        # 180k of growth: enough for G_min, not for the 2-pass backoff (200k cap)
        ({"low_yield_streak": 2, "baseline_after_tokens": 150_000}, "insufficient_growth"),
    ],
)
def test_every_gate_names_its_reason(overrides, reason):
    decision = due(**overrides)
    assert not decision.due
    assert decision.reason == reason


def test_backoff_doubles_per_low_yield_pass_and_caps_at_half_trigger():
    assert backoff_growth(G_MIN, 0, T_EFF) == 60_000
    assert backoff_growth(G_MIN, 1, T_EFF) == 120_000
    assert backoff_growth(G_MIN, 2, T_EFF) == 200_000  # capped at T_eff / 2
    assert backoff_growth(G_MIN, 9, T_EFF) == 200_000


def test_low_yield_streak_counts_newest_consecutive_and_resets():
    rho = ECONOMIC_MAX_RESIDUAL_RATIO
    assert low_yield_streak([], ratio=rho) == 0
    assert low_yield_streak([(400_000, 330_000), (400_000, 250_000)], ratio=rho) == 2
    # a good pass (0.3) newest resets the streak regardless of older ones
    assert low_yield_streak([(400_000, 120_000), (400_000, 330_000)], ratio=rho) == 0
    # unmeasurable pairs never extend a streak on a guess
    assert low_yield_streak([(0, 0), (400_000, 330_000)], ratio=rho) == 0


def test_usage_stats_measures_growth_cold_rate_gap_and_prefix():
    rows = []
    stamp = 1_000_000
    for index in range(60):
        context = 200_000 + 500 * index
        cold = index % 20 == 0
        rows.append(
            {
                "context_tokens": context,
                "cache_read_tokens": 35_000 if cold else context - 400,
                "at_ms": stamp,
            }
        )
        stamp += 16_000
    stats = usage_stats(rows, since_shrink=40)
    assert stats.samples == 60
    assert stats.growth_per_request == 500
    assert stats.cold_event_rate == pytest.approx(3 / 60)
    assert stats.median_gap_ms == 16_000
    assert stats.prefix_tokens == 35_000
    # below ECONOMIC_MIN_SAMPLES the cold rate is not trusted
    assert usage_stats(rows[:10], since_shrink=10).cold_event_rate == 0.0


def test_high_wake_rate_threshold_is_ninety_requests_in_six_hours():
    now = 10 * 3600 * 1000
    window = economics.HIGH_WAKE_WINDOW_MS
    inside = [now - window + 1 + i for i in range(economics.HIGH_WAKE_MIN_REQUESTS)]
    assert is_high_wake_rate(inside, now_ms=now)
    assert not is_high_wake_rate(inside[1:], now_ms=now)
    assert not is_high_wake_rate([now - window - 1] * 200, now_ms=now)


# -- the 2,000-request high-wake simulation ---------------------------------


def _gap_s(rng: random.Random) -> float:
    # lognormal with median 16 s and p95 460 s (sigma = ln(460/16) / 1.645)
    return math.exp(math.log(16.0) + rng.gauss(0.0, math.log(460 / 16) / 1.645))


def _simulate(
    *,
    seed: int,
    residual_of: Any,
    economic: bool,
    requests: int = 2_000,
    start: int = 150_000,
) -> dict[str, Any]:
    """One always-on session: +g per request, size trigger at T_eff, restarts at f.

    The cheap residual estimate is the last pass's MEASURED after-size (the
    session's own rule); a cheap "due" is re-checked on the EXACT residual,
    and a failed re-check journals a decline at the current size, exactly as
    ``Session._maybe_economic_compact`` does.
    """
    rng = random.Random(seed)
    context = start
    baseline = 0
    declined: int | None = None
    yields: list[tuple[int, int]] = []  # newest first
    dollars = 0.0
    cold_pending = True
    passes: list[tuple[int, int, int, str]] = []  # (request, before, after, kind)
    growth_total = 0
    for request in range(requests):
        grow = max(0, int(rng.gauss(G, G / 2)))
        context += grow
        growth_total += grow
        restart = rng.random() < F
        cold = cold_pending or restart
        dollars += context * (P_WRITE if cold else P_READ)
        cold_pending = False
        if context >= T_EFF:
            after = residual_of(rng, context)
            passes.append((request, context, after, "size"))
            yields.insert(0, (context, after))
            baseline, declined, context, cold_pending = after, None, after, True
            continue
        if not economic:
            continue
        common = {
            "enabled": True,
            "strategy": "snapcompact",
            "is_subagent": False,
            "priced": True,
            "context_tokens": context,
            "min_context_tokens": 0,
            "baseline_after_tokens": baseline,
            "declined_at_tokens": declined,
            "growth_per_request": G,
            "p_read": P_READ,
            "p_write": P_WRITE,
            "cold_event_rate": F,
            "cold_next": _gap_s(rng) > 3600,
            "pass_cost": 0.0,
            "min_gain_tokens": G_MIN,
            "max_residual_ratio": ECONOMIC_MAX_RESIDUAL_RATIO,
            "low_yield_streak": low_yield_streak(yields, ratio=ECONOMIC_MAX_RESIDUAL_RATIO),
            "t_eff": T_EFF,
        }
        cheap = baseline or 50_000
        if not economic_compaction_due(projected_residual=cheap, **common).due:
            continue
        exact = residual_of(rng, context)
        verdict = economic_compaction_due(projected_residual=exact, **common)
        if not verdict.due:
            declined = context
            continue
        passes.append((request, context, exact, "economic"))
        yields.insert(0, (context, exact))
        baseline, declined, context, cold_pending = exact, None, exact, True
    return {"passes": passes, "dollars": dollars, "growth": growth_total}


def _drawn_ratio(rng: random.Random, context: int) -> int:
    return int(context * rng.choice((0.4, 0.62, 0.84)))


@pytest.mark.parametrize("seed", range(5))
def test_no_sawtooth_high_wake(seed):
    run = _simulate(seed=seed, residual_of=_drawn_ratio, economic=True)
    economic = [p for p in run["passes"] if p[3] == "economic"]
    assert economic, "the economic rule never fired in 2,000 high-wake requests"
    # (1) at most one economic pass per G_min of growth
    assert len(economic) <= run["growth"] // G_MIN
    # (2) every economic pass follows >= G_min of growth past the previous
    # pass's measured after-size, so two consecutive turn ends never both
    # compact (growth per request is ~525 << G_min).
    ordered = run["passes"]
    for first, second in zip(ordered, ordered[1:]):
        if second[3] == "economic":
            assert second[1] - first[2] >= G_MIN, (first, second)
            assert second[0] - first[0] > 1, (first, second)
    # (3) never an economic pass whose residual ratio exceeds rho
    for _request, before, after, _kind in economic:
        assert after / before <= ECONOMIC_MAX_RESIDUAL_RATIO
    # (4) after a low-yield pass, the next economic pass waited for 2 * G_min
    for previous, nxt in zip(ordered, ordered[1:]):
        if nxt[3] != "economic":
            continue
        if previous[2] / previous[1] > ECONOMIC_MAX_RESIDUAL_RATIO:
            assert nxt[1] - previous[2] >= 2 * G_MIN, (previous, nxt)


@pytest.mark.parametrize("seed", range(5))
def test_economic_rule_beats_size_trigger_alone_when_residual_is_small(seed):
    # The capped high-wake residual: ~115k regardless of C, i.e. R <= 0.5 C at fire.
    def capped(rng: random.Random, _context: int) -> int:
        return 115_000 + int(rng.gauss(0, 5_000))

    economic = _simulate(seed=seed, residual_of=capped, economic=True)
    size_only = _simulate(seed=seed, residual_of=capped, economic=False)
    fired = [p for p in economic["passes"] if p[3] == "economic"]
    assert fired
    assert economic["dollars"] < size_only["dollars"], (economic["dollars"], size_only["dollars"])
