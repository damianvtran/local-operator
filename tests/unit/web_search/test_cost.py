"""Cost accounting for web search: pricing and the session ledger."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from local_operator.web_search.cost import (
    SEARCH_SPEND,
    SearchSpendLedger,
    deepseek_is_peak_hour,
    estimate_search_cost,
)
from local_operator.web_search.models import SearchCost, SearchUsage


def _at(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def test_deepseek_peak_windows_follow_utc_weekday_rules() -> None:
    # 2026-09-16 is a Wednesday.
    assert deepseek_is_peak_hour(_at(2026, 9, 16, 2)) is True
    assert deepseek_is_peak_hour(_at(2026, 9, 16, 7)) is True
    assert deepseek_is_peak_hour(_at(2026, 9, 16, 0)) is False
    assert deepseek_is_peak_hour(_at(2026, 9, 16, 5)) is False
    assert deepseek_is_peak_hour(_at(2026, 9, 16, 23)) is False
    # 2026-09-19 is a Saturday: off-peak all day, including a peak weekday hour.
    assert deepseek_is_peak_hour(_at(2026, 9, 19, 2)) is False


def test_deepseek_cost_is_token_priced_and_halves_off_peak() -> None:
    usage = SearchUsage(
        input_tokens=15_000,
        cache_read_tokens=10_000,
        output_tokens=800,
    )

    peak = estimate_search_cost("deepseek", usage, moment=_at(2026, 9, 16, 2))
    off_peak = estimate_search_cost("deepseek", usage, moment=_at(2026, 9, 16, 12))

    # 5,000 cache-miss input at $0.30/1M, 10,000 cache reads at $0.006/1M,
    # 800 output at $1.20/1M.
    expected_peak = 5_000 * 0.30e-6 + 10_000 * 0.006e-6 + 800 * 1.20e-6
    assert peak.usd == pytest.approx(expected_peak, abs=1e-9)
    assert off_peak.usd == pytest.approx(expected_peak / 2, abs=1e-9)
    assert "estimate" in peak.basis and "peak" in peak.basis
    assert "off-peak" in off_peak.basis
    assert peak.priced_from_usage is True


def test_cache_reads_are_not_charged_twice() -> None:
    """A cache read is cheaper input, not extra input.

    Charging 10k cache reads at both the miss rate and the hit rate is the easy
    arithmetic slip here, and it inflates every enriched DeepSeek search.
    """
    usage = SearchUsage(input_tokens=10_000, cache_read_tokens=10_000, output_tokens=0)
    cost = estimate_search_cost("deepseek", usage, moment=_at(2026, 9, 16, 2))

    assert cost.usd == pytest.approx(10_000 * 0.006e-6, abs=1e-9)


def test_free_tiers_are_zero_and_paid_paths_use_published_rates() -> None:
    assert estimate_search_cost("duckduckgo").usd == 0.0
    assert estimate_search_cost("searxng").usd == 0.0
    # Keyless Tavily is free even though Tavily bills per credit when keyed.
    assert estimate_search_cost("tavily", SearchUsage(keyless=True)).usd == 0.0
    keyed = estimate_search_cost("tavily", SearchUsage(keyless=False))
    assert keyed.usd == pytest.approx(0.008)
    assert estimate_search_cost("perplexity", SearchUsage(keyless=True)).usd == 0.0
    assert estimate_search_cost("brave").usd == pytest.approx(0.004)
    assert estimate_search_cost("serpapi").usd == pytest.approx(0.015)


def test_an_unknown_provider_is_unpriced_not_free() -> None:
    """ "No published rate" must never render as $0.00."""
    cost = estimate_search_cost("some-future-provider")

    assert cost.usd is None
    assert "no published rate" in cost.basis


def test_ledger_keeps_sessions_apart_and_totals_by_provider() -> None:
    ledger = SearchSpendLedger()
    ledger.record("s1", "deepseek", SearchCost(usd=0.004, basis="tokens at list price (estimate)"))
    ledger.record("s1", "tavily", SearchCost(usd=0.0, basis="free (keyless tier)"))
    ledger.record("s2", "brave", SearchCost(usd=0.004, basis="published per-search rate"))

    first = ledger.session("s1")
    assert first.searches == 2
    assert first.usd == pytest.approx(0.004)
    assert first.by_provider["deepseek"].searches == 1
    assert first.by_provider["tavily"].usd == 0.0

    second = ledger.session("s2")
    assert second.searches == 1
    assert second.usd == pytest.approx(0.004)

    overall = ledger.overall()
    assert overall.searches == 3
    assert overall.usd == pytest.approx(0.008)
    assert set(overall.by_provider) == {"deepseek", "tavily", "brave"}
    assert overall.as_dict()["by_provider"][0]["provider"] in {"deepseek", "tavily", "brave"}


def test_ledger_counts_unpriced_searches_separately() -> None:
    ledger = SearchSpendLedger()
    ledger.record("s1", "mystery", SearchCost(usd=None, basis="no published rate"))
    ledger.record("s1", "duckduckgo", SearchCost(usd=0.0, basis="free"))

    totals = ledger.session("s1")
    assert totals.searches == 2
    assert totals.priced_searches == 1
    assert totals.unpriced_searches == 1
    assert totals.by_provider["mystery"].unpriced_searches == 1


def test_module_ledger_is_shared_state_with_a_reset_for_tests() -> None:
    SEARCH_SPEND.record("shared", "duckduckgo", SearchCost(usd=0.0, basis="free"))
    assert SEARCH_SPEND.session("shared").searches == 1
    SEARCH_SPEND.reset()
    assert SEARCH_SPEND.session("shared").searches == 0
