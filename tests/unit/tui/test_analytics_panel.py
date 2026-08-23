"""The ``/analytics /usage`` screen — the report it renders and its Esc close.

The renderer is a pure function of the aggregate, so most of this asserts plain
strings (what a user reads). Two pilot tests drive the REAL ``OperatorApp`` so
the screen actually mounts under the stylesheet and Esc/``q`` return to the
previous view — a passing text assertion is not evidence a TUI looks right, but
it is the right way to pin what the screen SAYS and that it closes.
"""

from __future__ import annotations

from local_operator.analytics.model import COMPONENT_KEYS, UsageAggregate, UsagePeriod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.analytics_panel import (
    METRIC_COST,
    METRIC_TOKENS,
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
    agg.components["images"] = 400_000
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


def test_cost_marker_legend_shown_only_when_marks_present():
    # D1: a legend explains + and $— — but only when one is actually on screen.
    partial = _agg()
    partial.by_provider["ollama"] = UsageAggregate(
        calls=5, context_tokens=100, cost_micro=0, cost_known_calls=0
    )
    text_with_marks = "\n".join(line.plain for line in build_report(partial, 120))
    assert "lower bound" in text_with_marks
    assert "no published price" in text_with_marks

    # A fully-priced run (no + or $—) draws no legend.
    clean = _agg()  # all cost_known_calls == calls, no unpriced provider
    text_clean = "\n".join(line.plain for line in build_report(clean, 120))
    assert "lower bound" not in text_clean


def test_tables_share_a_name_column():
    # D2: BY PROVIDER and BY SESSION align — the tokens column starts at the
    # same offset in both, because they share one name_col.
    agg = _agg()
    lines = [line.plain for line in build_report(agg, 120)]
    text = "\n".join(lines)
    prov = next(li for li in text.splitlines() if "anthropic" in li)
    sess = next(li for li in text.splitlines() if "my session" in li)
    assert prov.index(" tokens") == sess.index(" tokens")


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
    # Section headers are title-case (not all-caps — the app uses that nowhere),
    # marked with the accent bar.
    assert "Totals" in text
    assert "100 calls" in text
    assert "(2 failed)" in text
    # authoritative headline numbers
    assert "Cache hit rate" in text
    # NESTED input breakdown: the flat "Input" row is gone; the full context
    # read is the parent and fresh/cache-read/cache-write are its sub-rows. The
    # small "Fresh (uncached)" figure must be unmistakably the uncached slice,
    # never labelled as "Input" (which read as a total) or "user input".
    assert "Context read" in text
    assert "Fresh (uncached)" in text
    assert "Cache read" in text
    assert "Cache write" in text
    # The old flat label must not survive as a standalone Totals row.
    assert not any(line.strip().startswith("Input ") for line in text.splitlines())
    # the estimated component split, largest first
    assert "Where input went" in text
    assert "estimated" in text
    assert "Conversation" in text
    assert "System prompt" in text
    # image-vs-text split surfaces once image tokens are present
    assert "Images (est.)" in text
    # per-provider and per-session tables
    assert "By provider" in text
    assert "anthropic" in text
    assert "By session" in text
    # named session shows its title, not the id
    assert "my session" in text


def test_section_headers_are_not_all_caps():
    # Guard the deliberate choice: the app uses no all-caps headers, so the
    # analytics sections must not regress to them.
    text = _text(_agg())
    for caps in ("TOTALS", "WHERE INPUT WENT", "BY PROVIDER", "BY SESSION"):
        assert caps not in text
    # And each section carries the accent delineation mark.
    assert "▌ Totals" in text
    assert "▌ By provider" in text


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


async def _push(pilot, app, agg, *, daily=None, monthly=None):
    await pilot.pause()
    screen = AnalyticsScreen(agg, daily=daily, monthly=monthly)
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
            assert "Totals" in joined
            assert "Where input went" in joined

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Historical time-series bar charts (the daily/monthly rollup extension).
# ---------------------------------------------------------------------------


def _daily() -> list[UsagePeriod]:
    return [
        UsagePeriod(
            period="2026-08-21",
            model="",
            context_tokens=900_000,
            output_tokens=300_000,
            cost_micro=3_410_000,
            cost_known_calls=4,
            calls=4,
        ),
        UsagePeriod(
            period="2026-08-22",
            model="",
            context_tokens=300_000,
            output_tokens=100_000,
            cost_micro=980_000,
            cost_known_calls=2,
            calls=3,  # one unpriced → floor
        ),
        UsagePeriod(
            period="2026-08-23",
            model="",
            context_tokens=600_000,
            output_tokens=200_000,
            cost_micro=0,
            cost_known_calls=0,
            calls=2,  # fully unpriced → floor, cost $—
        ),
    ]


def _monthly() -> list[UsagePeriod]:
    return [
        UsagePeriod(
            period="2026-07",
            model="",
            context_tokens=5_000_000,
            output_tokens=2_000_000,
            cost_micro=42_000_000,
            cost_known_calls=40,
            calls=50,
        ),
        UsagePeriod(
            period="2026-08",
            model="",
            context_tokens=1_800_000,
            output_tokens=600_000,
            cost_micro=4_390_000,
            cost_known_calls=6,
            calls=9,
        ),
    ]


def _fully_unpriced() -> list[UsagePeriod]:
    # Every bucket has cost_known_calls == 0 → cost_is_known False everywhere.
    # This is the local-model-only run (D1): cost mode must show clean $— rows
    # with NO ≥ mark, never the self-contradictory ≥ $—.
    return [
        UsagePeriod(
            period="2026-08-21",
            model="",
            context_tokens=900_000,
            output_tokens=300_000,
            cost_micro=0,
            cost_known_calls=0,
            calls=3,
        ),
        UsagePeriod(
            period="2026-08-22",
            model="",
            context_tokens=300_000,
            output_tokens=100_000,
            cost_micro=0,
            cost_known_calls=0,
            calls=2,
        ),
    ]


def test_daily_chart_renders_labels_and_bars():
    text = "\n".join(
        line.plain for line in build_report(_agg(), 90, daily=_daily(), monthly=_monthly())
    )
    # Section headers and human bucket labels appear. The daily title counts
    # DAYS WITH USAGE, not a calendar window (D3).
    assert "3 days with usage" in text
    assert "Monthly" in text
    assert "Aug 21" in text
    assert "Jul 2026" in text
    # Bars are drawn.
    assert "█" in text


def test_daily_title_singularizes_one_day():
    # D4: a single-day series reads "1 day with usage", not "1 days".
    one = [_daily()[0]]
    text = "\n".join(line.plain for line in build_report(_agg(), 90, daily=one))
    assert "1 day with usage" in text
    assert "1 days" not in text


def test_monthly_meta_self_describes_metric():
    # D5/U2: the Monthly section states its metric and the toggle too, so a
    # reader parked on it is not left guessing $ vs tokens.
    text = "\n".join(line.plain for line in build_report(_agg(), 90, monthly=_monthly()))
    assert "by calendar month · cost · t → tokens" in text


def test_daily_chart_default_metric_is_cost():
    # Default metric is cost, and the header advertises the toggle to tokens.
    text = "\n".join(line.plain for line in build_report(_agg(), 90, daily=_daily()))
    assert "cost · t → tokens" in text
    assert "$3.41" in text


def test_chart_tokens_metric_shows_token_cells():
    text = "\n".join(
        line.plain for line in build_report(_agg(), 90, daily=_daily(), metric=METRIC_TOKENS)
    )
    assert "tokens · t → cost" in text
    # The Aug 21 bucket's billed total (900k context + 300k output) → 1.2M.
    assert "1.2M" in text


def test_chart_floor_mark_only_on_mixed_priced_bucket():
    # D1/D2: Aug 21 fully priced → plain $3.41 (no ≥). Aug 22 mixes priced +
    # unpriced → a GENUINE lower bound, so ≥ $0.980 — with NO trailing + (the ≥
    # is the single lower-bound signal). Aug 23 is fully unpriced → clean $—
    # with NO ≥ (≥ $— would be "≥ unknown").
    cost_text = "\n".join(
        line.plain for line in build_report(_agg(), 90, daily=_daily(), metric=METRIC_COST)
    )
    assert "≥ $0.980" in cost_text  # mixed bucket: mark present, no +
    assert "$0.980+" not in cost_text  # D2: trailing + stripped under the mark
    assert "$3.41" in cost_text  # fully-priced bucket: no mark
    assert "$—" in cost_text  # fully-unpriced bucket
    assert "≥ $—" not in cost_text  # D1: the contradiction must NOT appear
    token_text = "\n".join(
        line.plain for line in build_report(_agg(), 90, daily=_daily(), metric=METRIC_TOKENS)
    )
    # No floor mark in tokens mode: tokens are always known.
    assert "≥" not in token_text


def test_fully_unpriced_cost_chart_has_no_floor_mark():
    # D1 (the MAJOR): the local-model-only default cost view. Every row is $—
    # over an empty track — and crucially NOT a single ≥ anywhere, which would
    # be the "≥ unknown" wall the finding called out.
    cost_text = "\n".join(
        line.plain for line in build_report(_agg(), 90, daily=_fully_unpriced(), metric=METRIC_COST)
    )
    assert "$—" in cost_text
    assert "≥" not in cost_text
    # In tokens mode the same buckets show real bars (tokens are known).
    token_text = "\n".join(
        line.plain
        for line in build_report(_agg(), 90, daily=_fully_unpriced(), metric=METRIC_TOKENS)
    )
    assert "█" in token_text


def test_daily_meta_shows_window_total():
    # C2: series_totals surfaces as the daily section's window summary, ahead of
    # the metric clause, in the active metric.
    wt = UsagePeriod(
        period="",
        model="",
        context_tokens=1_800_000,
        output_tokens=600_000,
        cost_micro=4_390_000,
        cost_known_calls=6,
        calls=9,
    )
    cost_text = "\n".join(
        line.plain
        for line in build_report(_agg(), 90, daily=_daily(), window_totals=wt, metric=METRIC_COST)
    )
    # $4.39+ (partial) · cost · t → tokens — the window total leads the meta.
    assert "$4.39+ · cost · t → tokens" in cost_text
    token_text = "\n".join(
        line.plain
        for line in build_report(_agg(), 90, daily=_daily(), window_totals=wt, metric=METRIC_TOKENS)
    )
    # In tokens mode the window summary is a token count.
    assert "2.4M tokens · tokens · t → cost" in token_text


def test_chart_empty_series_shows_note_not_bars():
    text = "\n".join(line.plain for line in build_report(_agg(), 90, daily=[], monthly=[]))
    assert "no daily usage recorded yet" in text
    assert "no monthly usage recorded yet" in text


def test_no_series_omits_chart_sections():
    # A caller with no rollups (daily=None) gets exactly the original report —
    # no chart headers at all.
    text = "\n".join(line.plain for line in build_report(_agg(), 90))
    assert "Last" not in text
    assert "Monthly" not in text


def test_toggle_key_flips_metric_in_real_app():
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _agg(), daily=_daily(), monthly=_monthly())
            joined = "\n".join(screen.render_lines_for_test())
            assert "cost · t → tokens" in joined  # starts in cost mode
            await pilot.press("t")
            await pilot.pause()
            joined2 = "\n".join(screen.render_lines_for_test())
            assert "tokens · t → cost" in joined2  # flipped to tokens
            await pilot.press("t")
            await pilot.pause()
            joined3 = "\n".join(screen.render_lines_for_test())
            assert "cost · t → tokens" in joined3  # flipped back

    asyncio.run(run())


def test_pinned_title_carries_active_metric_and_flips():
    # U1/U4: the pinned title states the active metric so `t` gives visible
    # feedback even when the charts are scrolled off. It flips with the toggle
    # and is absent when there are no charts.
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _agg(), daily=_daily(), monthly=_monthly())
            title0 = screen._title_text().plain
            assert "bars: cost" in title0
            await pilot.press("t")
            await pilot.pause()
            title1 = screen._title_text().plain
            assert "bars: tokens" in title1
            # The actual pinned Static was repainted by the toggle handler, not
            # just the recomputed text — read what the widget is rendering.
            # ``Static.render`` returns a RenderableType union (rich ``Text``,
            # Textual ``Content``, ``str``, …) whose members do not share a
            # ``.plain`` attribute pyright can narrow. ``str(...)`` is defined
            # on every member and yields the plain rendered text, so it type-
            # checks against the whole union while still asserting what the
            # widget actually paints.
            assert "bars: tokens" in str(screen._title.render())
            await pilot.press("escape")
            await pilot.pause()
            # No charts → no metric suffix in the title (nothing to toggle).
            screen2 = await _push(pilot, app, _agg())
            assert "bars:" not in screen2._title_text().plain

    asyncio.run(run())


def test_toggle_hint_shown_only_with_series():
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            # With non-empty series: the footer advertises the t toggle.
            screen = await _push(pilot, app, _agg(), daily=_daily())
            hint = screen._hint_text(scrollable=False).plain
            assert "t cost/tokens" in hint
            await pilot.press("escape")
            await pilot.pause()
            # Without series: no toggle hint (the key would be a no-op).
            screen2 = await _push(pilot, app, _agg())
            hint2 = screen2._hint_text(scrollable=False).plain
            assert "t cost/tokens" not in hint2
            await pilot.press("escape")
            await pilot.pause()
            # Empty series ([] not None): still no toggle — there are no bars.
            screen3 = await _push(pilot, app, _agg(), daily=[], monthly=[])
            hint3 = screen3._hint_text(scrollable=False).plain
            assert "t cost/tokens" not in hint3

    asyncio.run(run())
