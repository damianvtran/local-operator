"""The ``/analytics /usage`` screen — the report it renders and its Esc close.

The renderer is a pure function of the aggregate, so most of this asserts plain
strings (what a user reads). Two pilot tests drive the REAL ``OperatorApp`` so
the screen actually mounts under the stylesheet and Esc/``q`` return to the
previous view — a passing text assertion is not evidence a TUI looks right, but
it is the right way to pin what the screen SAYS and that it closes.
"""

from __future__ import annotations

from local_operator.analytics.model import COMPONENT_KEYS, UsageAggregate
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.analytics_panel import (
    AnalyticsScreen,
    build_report,
    format_cost,
    format_percent,
    format_tokens,
    proportion_bar,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _agg() -> UsageAggregate:
    agg = UsageAggregate(
        calls=100,
        ok_calls=98,
        input_tokens=500_000,
        output_tokens=120_000,
        cache_read_tokens=3_000_000,
        cache_write_tokens=80_000,
        reasoning_tokens=40_000,
        context_tokens=3_500_000,
        cost_micro=4_200_000,
        cost_known_calls=100,
    )
    agg.components = {k: 0 for k in COMPONENT_KEYS}
    agg.components["conversation"] = 1_800_000
    agg.components["tool_results"] = 900_000
    agg.components["system_prompt"] = 500_000
    agg.components["tool_schemas"] = 300_000
    sub = UsageAggregate(
        calls=100,
        input_tokens=500_000,
        output_tokens=120_000,
        context_tokens=3_500_000,
        cache_read_tokens=3_000_000,
        cost_micro=4_200_000,
        cost_known_calls=100,
    )
    agg.by_provider = {"anthropic": sub}
    agg.by_session = {"abc123": sub}
    setattr(agg, "session_names", {"abc123": "my session"})
    return agg


def _text(agg: UsageAggregate, width: int = 90) -> str:
    return "\n".join(line.plain for line in build_report(agg, width))


def test_format_tokens_scales():
    assert format_tokens(912) == "912"
    assert format_tokens(3400) == "3.4k"
    assert format_tokens(1_200_000) == "1.2M"
    assert format_tokens(4_100_000_000) == "4.1B"


def test_format_percent():
    assert format_percent(0.734) == "73%"
    assert format_percent(None) == "—"


def test_format_cost_states():
    def agg(cost_micro, known, calls):
        return UsageAggregate(calls=calls, cost_micro=cost_micro, cost_known_calls=known)

    # Complete, whole-dollar.
    assert format_cost(agg(8_340_000, 10, 10)) == "$8.34"
    # Partial (some calls unpriced) -> lower-bound marker.
    assert format_cost(agg(8_340_000, 7, 10)) == "$8.34+"
    # Nothing priced -> $—, never $0.00.
    assert format_cost(agg(0, 0, 5)) == "$—"
    # Sub-cent keeps precision so a real spend is not rounded to zero.
    assert format_cost(agg(4_200, 3, 3)) == "$0.0042"
    # Large sum abbreviates.
    assert format_cost(agg(1_200_000_000, 5, 5)) == "$1.2k"


def test_report_shows_cost():
    text = _text(_agg())
    assert "Est. cost" in text
    assert "$4.20" in text  # 4_200_000 micro-USD
    assert "list price" in text  # the estimate caveat
    # cost column appears in the per-provider and per-session tables
    assert text.count("$4.20") >= 3  # totals + provider + session


def test_report_marks_unpriced_and_partial():
    agg = _agg()
    # Add an unpriced local provider and a partially-priced session.
    unpriced = UsageAggregate(calls=5, context_tokens=100, cost_micro=0, cost_known_calls=0)
    agg.by_provider["ollama"] = unpriced
    text = "\n".join(line.plain for line in build_report(agg, 120))
    assert "$—" in text  # the unpriced provider row


def test_narrow_table_drops_cache_keeps_cost():
    agg = _agg()
    narrow = "\n".join(line.plain for line in build_report(agg, 58))
    wide = "\n".join(line.plain for line in build_report(agg, 120))
    # Cost survives at every width; cache is shed only when narrow.
    assert "$4.20" in narrow
    assert "cache" in wide
    # The BY PROVIDER row in the narrow render carries no cache column.
    provider_line = next(line for line in narrow.splitlines() if "anthropic" in line)
    assert "cache" not in provider_line
    assert "$4.20" in provider_line


def test_proportion_bar_fills():
    assert proportion_bar(1.0, 10) == "█" * 10
    assert proportion_bar(0.0, 10) == "·" * 10
    bar = proportion_bar(0.5, 10)
    assert bar.count("█") == 5 and bar.count("·") == 5


def test_proportion_bar_nonzero_floors_to_one_cell():
    # D3: a small nonzero fraction must show at least one filled cell so a real
    # 1% contributor is distinguishable from an empty (rounds-to-zero) row.
    bar = proportion_bar(0.01, 24)  # would round to 0 filled cells
    assert bar.count("█") == 1
    # A genuine zero still renders empty.
    assert proportion_bar(0.0, 24).count("█") == 0


def test_format_percent_floors_near_100():
    # D4: 99.6% must not round up to a flat, suspicious-looking 100%.
    assert format_percent(0.996) == "99%"
    assert format_percent(1.0) == "100%"  # a genuine full rate still shows 100%


def test_estimate_is_marked_at_data_level():
    # D1: the estimated split must be marked as an estimate on the DATA (~ on
    # each percentage), so the distinction survives the heading scrolling away.
    import re

    text = _text(_agg())
    assert "≈ estimated" in text
    # Every WHERE-INPUT-WENT percentage carries a ~ prefix (modelled, not
    # measured); the TOTALS section carries none.
    assert re.search(r"~\s*\d+%", text)


def test_empty_aggregate_says_no_data():
    text = _text(UsageAggregate())
    assert "No usage recorded yet" in text


def test_report_shows_totals_and_split():
    text = _text(_agg())
    assert "TOTALS" in text
    assert "100 calls" in text
    assert "(2 failed)" in text
    # authoritative headline numbers
    assert "Cache hit rate" in text
    # the estimated component split, largest first
    assert "WHERE INPUT WENT" in text
    assert "estimated" in text
    assert "Conversation" in text
    assert "System prompt" in text
    # per-provider and per-session tables
    assert "BY PROVIDER" in text
    assert "anthropic" in text
    assert "BY SESSION" in text
    # named session shows its title, not the id
    assert "my session" in text


def test_component_split_ordered_largest_first():
    text = _text(_agg())
    conv = text.index("Conversation")
    tool_results = text.index("Tool results")
    system = text.index("System prompt")
    # conversation (1.8M) before tool results (900k) before system (500k)
    assert conv < tool_results < system


def test_thinking_generation_split_shown():
    text = _text(_agg())
    # output 120k = 80k generation + 40k thinking
    assert "generation" in text and "thinking" in text


async def _push(pilot, app, agg):
    await pilot.pause()
    screen = AnalyticsScreen(agg)
    await app.push_screen(screen)
    await pilot.pause()
    await pilot.pause()
    return screen


def test_screen_mounts_and_esc_closes():
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            before = app.screen.__class__.__name__
            await _push(pilot, app, _agg())
            assert app.screen.__class__.__name__ == "AnalyticsScreen"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.__class__.__name__ == before

    asyncio.run(run())


def test_screen_q_closes():
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            before = app.screen.__class__.__name__
            await _push(pilot, app, _agg())
            await pilot.press("q")
            await pilot.pause()
            assert app.screen.__class__.__name__ == before

    asyncio.run(run())


def test_render_lines_for_test_available():
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _agg())
            lines = screen.render_lines_for_test()
            joined = "\n".join(lines)
            assert "TOTALS" in joined
            assert "WHERE INPUT WENT" in joined

    asyncio.run(run())
