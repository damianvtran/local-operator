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

    # The Anthropic wire EXCLUDES cache reads from input_tokens, so all 15,000
    # inputs bill at the miss rate and the 10,000 reads bill at the hit rate.
    expected_peak = 15_000 * 0.30e-6 + 10_000 * 0.006e-6 + 800 * 1.20e-6
    assert peak.usd == pytest.approx(expected_peak, abs=1e-9)
    assert off_peak.usd == pytest.approx(expected_peak / 2, abs=1e-9)
    assert "estimate" in peak.basis and "peak" in peak.basis
    assert "off-peak" in off_peak.basis
    assert peak.priced_from_usage is True


def test_anthropic_wire_cache_reads_are_added_not_subtracted() -> None:
    """The search route is Anthropic-wire: reads are OUTSIDE ``input_tokens``.

    The OpenAI-shaped reading (subtract the reads from the input count) charged
    nothing for fresh input on a cache-dominated turn -- and the enriched path is
    exactly that shape, since the replayed pages arrive as cache reads. This pins
    the convention so the subtraction cannot come back.
    """
    usage = SearchUsage(input_tokens=200, cache_read_tokens=10_000, output_tokens=0)
    cost = estimate_search_cost("deepseek", usage, moment=_at(2026, 9, 16, 2))

    assert cost.usd == pytest.approx(200 * 0.30e-6 + 10_000 * 0.006e-6, abs=1e-9)
    # Both buckets are charged: a subtraction would bill only the reads.
    assert cost.usd is not None
    assert cost.usd > 10_000 * 0.006e-6


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


# -- keyed Sonar is billed, and must never be priced as the free tier -------


def test_a_keyed_sonar_request_is_priced_and_never_free() -> None:
    """MAJOR-1: the anonymous tier is free; keyed traffic must not land there.

    The keyed response reported no ``usage``, so ``estimate_search_cost`` fell
    through to the anonymous branch and booked a billed Sonar call at $0.0000.
    """
    from local_operator.web_search.models import SearchUsage

    priced = estimate_search_cost(
        "perplexity", SearchUsage(input_tokens=14_500, output_tokens=776, keyless=False)
    )
    assert priced.usd is not None and priced.usd > 0.005
    # The request fee is $5/1,000 plus $1/1M tokens: both halves are in there.
    assert priced.usd == pytest.approx(0.005 + 15_276 * 1e-6, abs=1e-9)
    assert priced.priced_from_usage is True

    # The API answered without a usage block: the fee is real, the tokens are
    # unknown rather than zero, so the row is a FLOOR and not a confident total.
    floored = estimate_search_cost("perplexity", SearchUsage(keyless=False))
    assert floored.usd == pytest.approx(0.005)
    assert floored.priced_from_usage is False

    # The free tier is the ANONYMOUS one, and the tier flag -- not the absence of
    # usage -- is what says so. A ``None`` usage cannot distinguish the free tier
    # from a keyed call whose usage was lost, so it is UNPRICED: guessing "free"
    # is the claim this whole path was fixed for (round-2 review R2-MINOR-1).
    anonymous = estimate_search_cost("perplexity", SearchUsage(keyless=True))
    assert anonymous.usd == 0.0

    unknown = estimate_search_cost("perplexity", None)
    assert unknown.usd is None, "no usage and no tier flag is unknown, not free"


def test_the_search_routes_peak_constants_agree_with_the_registry_row() -> None:
    """The one remaining duplication of DeepSeek's rates, pinned to its source.

    This route keeps its own per-token constants because it is Anthropic-wire and
    prices its own buckets (cache reads sit OUTSIDE ``input_tokens`` there, the
    opposite of the chat route), so folding it onto the registry row would import
    the other wire's arithmetic. The constants must still be the SAME published
    peak rates as ``deepseek_models["deepseek-flash"]`` divided by 1e6 — this is
    the test that makes that duplication safe, because the day they drift the
    search ledger starts quoting a price the model picker does not.
    """
    from local_operator.model.registry import deepseek_models
    from local_operator.web_search import cost as search_cost

    row = deepseek_models["deepseek-flash"]
    assert row.cache_reads_price is not None
    assert search_cost.DEEPSEEK_PEAK_INPUT_USD_PER_TOKEN == pytest.approx(row.input_price / 1e6)
    assert search_cost.DEEPSEEK_PEAK_OUTPUT_USD_PER_TOKEN == pytest.approx(row.output_price / 1e6)
    assert search_cost.DEEPSEEK_PEAK_CACHE_HIT_USD_PER_TOKEN == pytest.approx(
        row.cache_reads_price / 1e6
    )


def test_the_search_route_evaluates_the_shared_schedule_not_its_own_copy() -> None:
    """Both routes must answer "is this peak" identically, hour by hour.

    The search route's predicate now DELEGATES to ``model/tariff``; this walks a
    week of hours and asserts the two agree, so the ownership is observable
    rather than merely claimed in a docstring.
    """
    from local_operator.model import tariff

    for day_offset in range(7):
        for hour in range(24):
            moment = _at(2026, 9, 14 + day_offset, hour)
            assert deepseek_is_peak_hour(moment) is tariff.is_peak(
                tariff.DEEPSEEK_TOU, moment
            ), moment


# -- the keyless MCP tiers: free where documented, never guessed -------------


def test_the_keyless_mcp_tiers_are_free_and_the_keyed_paths_are_not() -> None:
    """A keyless flag is a FREE TIER, not a missing price -- and not the reverse."""
    for provider in ("exa", "parallel"):
        keyless = estimate_search_cost(provider, SearchUsage(keyless=True))
        assert keyless.usd == 0.0, provider
        assert keyless.basis == "free (keyless tier)"

    # Exa's REST API keeps its published rate; Parallel's keyed Search API has no
    # rate we can verify, so it is UNPRICED -- which must never render as $0.00.
    assert estimate_search_cost("exa", SearchUsage(keyless=False)).usd == pytest.approx(0.005)
    unpriced = estimate_search_cost("parallel", SearchUsage(keyless=False))
    assert unpriced.usd is None
    assert unpriced.basis == "no published rate"

    # No usage block and no tier flag. Exa has a published per-search rate, so
    # the estimator prices ONE search at list (its long-standing rule for a
    # rate-bearing provider); Parallel has no verifiable rate, so it stays
    # unpriced. Neither case may be reported as free.
    assert estimate_search_cost("exa", None).usd == pytest.approx(0.005)
    assert estimate_search_cost("parallel", None).usd is None
    assert "free" not in estimate_search_cost("parallel", None).basis


def test_a_session_can_hold_one_free_and_one_paid_exa_leg_honestly() -> None:
    """One provider id, two cost bases: the ledger must not blend them."""
    ledger = SearchSpendLedger()
    ledger.record("s1", "exa", estimate_search_cost("exa", SearchUsage(keyless=True)))
    ledger.record("s1", "exa", estimate_search_cost("exa", SearchUsage(keyless=False)))

    totals = ledger.session("s1")
    row = totals.by_provider["exa"]

    assert row.free_operations == 1 and row.paid_operations == 1
    assert row.free_usd == 0.0
    assert row.paid_usd == pytest.approx(0.005)
    assert row.usd == pytest.approx(0.005)
    assert len(row.bases) == 2
