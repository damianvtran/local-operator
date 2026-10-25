"""A cache-saving claim must not credit a prefix no provider ever wrote."""

import pytest

from local_operator.harness.types import Message, TextContent
from scripts.bench_aged_clearing import ArmResult, BreakpointCache, distribution


def _rows(n):
    return [Message.assistant(f"m{i}") for i in range(n)]


def test_first_request_has_no_cached_system_credit():
    cache = BreakpointCache()
    rows = _rows(1)
    assert cache.price(rows, [10], 100, 150_000) == 110 * 1.25
    assert cache.price(rows, [10], 100, 150_000) == 110 * 0.1


def test_only_previously_written_tail_is_read_not_the_unchanged_cut():
    cache = BreakpointCache()
    rows = _rows(50)
    cache.price(rows, [10] * 50, 100, 150_000)
    # Clearing message 20 leaves a large unchanged prefix, but the only
    # conversation breakpoint the old request wrote was message 49. There
    # is no explicit write at 19 to read: only the system head can hit.
    replacement = Message(role="tool", content=[TextContent(text="stub")])
    changed = rows[:20] + [replacement] + rows[21:]
    assert cache.price(changed, [10] * 50, 100, 150_000) == 100 * 0.1 + 500 * 1.25


def test_tail_lookback_cannot_find_a_write_more_than_twenty_blocks_back():
    cache = BreakpointCache()
    rows = _rows(10)
    cache.price(rows, [10] * 10, 100, 150_000)
    extended = rows + _rows(21)
    assert cache.price(extended, [10] * 31, 100, 150_000) == 100 * 0.1 + 310 * 1.25


def test_previous_tail_inside_lookback_is_reused():
    cache = BreakpointCache()
    rows = _rows(10)
    cache.price(rows, [10] * 10, 100, 150_000)
    extended = rows + _rows(2)
    assert cache.price(extended, [10] * 12, 100, 150_000) == 200 * 0.1 + 20 * 1.25


def test_long_context_hint_selects_two_times_write_rate():
    cache = BreakpointCache()
    cache.price(_rows(10), [10] * 10, 100, 150)
    # All new identities: only system hits. The previous receipt exceeds the
    # test threshold, so the new suffix is written at 1h/2x, not 5m/1.25x.
    assert cache.price(_rows(10), [10] * 10, 100, 150) == 100 * 0.1 + 100 * 2


def test_distribution_exposes_short_session_regressions_despite_aggregate_saving():
    short = (
        ArmResult(requests=50, anthropic_cost=100),
        ArmResult(requests=50, anthropic_cost=110, clear_events=1),
    )
    long = (
        ArmResult(requests=200, anthropic_cost=1000),
        ArmResult(requests=200, anthropic_cost=500, clear_events=1),
    )
    result = distribution([short, long])
    assert result["regressions"] == 1
    assert result["buckets"]["41-80"]["cost_delta"] == pytest.approx(0.1)
    assert result["quantiles"]["max"] == pytest.approx(0.1)
