"""The ``/analytics /usage`` screen — the report it renders and its Esc close.

The renderer is a pure function of the aggregate, so most of this asserts plain
strings (what a user reads). Two pilot tests drive the REAL ``OperatorApp`` so
the screen actually mounts under the stylesheet and Esc/``q`` return to the
previous view — a passing text assertion is not evidence a TUI looks right, but
it is the right way to pin what the screen SAYS and that it closes.
"""

from __future__ import annotations

import copy
import re

from rich.cells import cell_len

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    UsageAggregate,
    UsagePeriod,
    build_session_forest,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.analytics_panel import (
    _MAX_NAME_COL,
    _MIN_NAME_COL,
    _ROW_CURSOR,
    _WIDE_TABLE_MIN,
    METRIC_COST,
    METRIC_TOKENS,
    AnalyticsScreen,
    _forest_labels,
    _forest_rows,
    _row_overhead,
    _row_prefix,
    build_report,
    format_cost,
    format_percent,
    format_tokens,
    proportion_bar,
)
from local_operator.tui.widgets.tool_card import truncate_cells
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


def test_fresh_is_uncached_slice_on_openai_shaped_usage():
    # OpenAI-shaped: cache is already folded into input, so context == input.
    # Fresh must be context − cache_read − cache_write (20k), NOT input (100k).
    agg = UsageAggregate(
        calls=10,
        ok_calls=10,
        input_tokens=100_000,
        output_tokens=5_000,
        cache_read_tokens=80_000,
        cache_write_tokens=0,
        context_tokens=100_000,
        cost_micro=1_000_000,
        cost_known_calls=10,
    )
    assert agg.fresh_tokens == 20_000
    text = _text(agg)
    assert "20k" in text
    # The Fresh row itself must not show the full 100k as its value.
    fresh_line = next(line for line in text.splitlines() if "Fresh (uncached)" in line)
    assert "20k" in fresh_line
    assert "100k" not in fresh_line
    # The three children partition Context read (20k + 80k + 0 == 100k).
    assert agg.fresh_tokens + agg.cache_read_tokens + agg.cache_write_tokens == (agg.context_tokens)
    context_line = next(line for line in text.splitlines() if "Context read" in line)
    assert "20k fresh" in context_line
    assert "80k cached" in context_line
    assert "0 written" in context_line


def test_fresh_equals_input_on_anthropic_shaped_usage():
    # Anthropic: context = input + cache_read + cache_write, so fresh == input.
    # The shared ``_agg()`` fixture is close but does not partition (500k + 3M
    # + 80k ≠ 3.5M), so this case is built to actually sum.
    agg = UsageAggregate(
        calls=100,
        ok_calls=100,
        input_tokens=387_000,
        output_tokens=210_000,
        cache_read_tokens=3_000_000,
        cache_write_tokens=113_000,
        context_tokens=3_500_000,
        cost_micro=1_000_000,
        cost_known_calls=100,
    )
    assert agg.fresh_tokens == agg.input_tokens == 387_000
    assert agg.fresh_tokens + agg.cache_read_tokens + agg.cache_write_tokens == agg.context_tokens
    text = _text(agg)
    fresh_line = next(line for line in text.splitlines() if "Fresh (uncached)" in line)
    assert "387k" in fresh_line
    context_line = next(line for line in text.splitlines() if "Context read" in line)
    assert "387k fresh" in context_line
    assert "3M cached" in context_line
    assert "113k written" in context_line


def test_totals_value_cells_share_a_note_gutter():
    # D2: a short value (``3M``) must not pull its note left of a longer
    # sibling (``387k`` / ``500k``). Notes share one column after the pad.
    text = _text(_agg())
    rows = [
        next(line for line in text.splitlines() if needle in line)
        for needle in ("Fresh (uncached)", "Cache read", "Cache write")
    ]
    notes = (
        "new input, billed at full rate",
        "input served from cache",
        "new input written to cache",
    )
    starts = [row.index(note) for row, note in zip(rows, notes)]
    assert len(set(starts)) == 1, rows


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


def _sub(cost_micro: int, calls: int = 1) -> UsageAggregate:
    """A minimal per-session aggregate carrying a distinguishable cost."""
    return UsageAggregate(
        calls=calls,
        input_tokens=1_000,
        output_tokens=500,
        context_tokens=1_500,
        cost_micro=cost_micro,
        cost_known_calls=calls,
    )


def test_sessions_sharing_a_rendered_label_each_get_their_own_row():
    """N sessions with one label render N rows, not one (round-1 F1).

    The table used to be a dict keyed by the RENDERED label, so a collision did
    not merge rows — it dropped every row but the last, taking its tokens, calls
    and cost off the screen. On the operator's ledger that hid 355 sessions and
    $3,913. The label is display text; the row's identity is the session id.
    """
    agg = _agg()
    spend = {"aa11bb22cc33": 900_000, "dd44ee55ff66": 500_000, "1122334455aa": 250_000}
    agg.by_session = {sid: _sub(cost) for sid, cost in spend.items()}
    # One shared name, exactly what the backfill mints for sibling subagents.
    shared = "reviewer · Article-search-svc schema review"
    setattr(agg, "session_names", {sid: shared for sid in spend})

    lines = build_report(agg, width=140)
    body = "\n".join(line.plain for line in lines)
    session_block = body.split("By session", 1)[1]

    # Every session is on screen, with ITS OWN cost — not one merged row.
    for cost in spend.values():
        assert f"${cost / 1_000_000:.2f}" in session_block, session_block
    # ...and each row is separately addressable rather than three identical ones.
    rows = [ln for ln in session_block.splitlines() if "tokens" in ln]
    assert len(rows) == len(spend)
    assert len({ln.strip() for ln in rows}) == len(spend), rows


def test_a_unique_session_label_is_left_exactly_as_it_was():
    """The disambiguator is paid only where it buys something."""
    agg = _agg()
    agg.by_session = {"aa11bb22cc33": _sub(900_000), "dd44ee55ff66": _sub(500_000)}
    setattr(
        agg,
        "session_names",
        {"aa11bb22cc33": "Fix the analytics rollup", "dd44ee55ff66": "Rename the sessions"},
    )
    body = "\n".join(line.plain for line in build_report(agg, width=140))
    assert "Fix the analytics rollup " in body
    assert "·" not in body.split("By session", 1)[1].split("Fix the analytics rollup")[1][:40]


def _long_named_agg(count: int = 12) -> UsageAggregate:
    """An aggregate whose session names are longer than any name column.

    The shape design review D3 measured on the real ledger: once subagent
    naming lands, composed ``<role> · <parent title>`` labels are the MAJORITY
    of rows rather than the exception, so a name column that only pads is
    ragged for most of the table instead of a few rows.
    """
    agg = _agg()
    agg.by_session = {}
    names = {}
    for index in range(count):
        sub = UsageAggregate(
            calls=10 + index,
            input_tokens=500_000,
            output_tokens=120_000,
            context_tokens=3_500_000,
            cost_micro=4_200_000 + index,
            cost_known_calls=10 + index,
        )
        sid = f"{index:012x}"
        agg.by_session[sid] = sub
        names[sid] = f"reviewer · Article-search-svc schema review number {index}"
    setattr(agg, "session_names", names)
    return agg


def test_the_number_columns_line_up_however_long_the_names_are():
    """Design D3: the name column must TRUNCATE, not only pad.

    ``{name:<{name_col}}`` is a pad and nothing else, so a name wider than the
    column pushed tokens/cost/calls right by however much it overran. The cost
    column is the one thing this screen exists to let you scan straight down,
    and on the real ledger 78% of rows overflowed at 100 columns.
    """
    agg = _long_named_agg()
    for width in (71, 100, 140):
        # A section is ONE ``Text`` carrying embedded newlines, so the rows have
        # to be split back out of the rendered block rather than read off the
        # returned list.
        text = "\n".join(line.plain for line in build_report(agg, width))
        block = text.split("By session", 1)[-1]
        rows = [li for li in block.splitlines() if "calls" in li and " tokens" in li]
        assert len(rows) == 12, (width, rows)
        offsets = {li.index(" tokens") for li in rows}
        assert len(offsets) == 1, (width, sorted(offsets), rows[:3])


def test_a_wide_frame_spends_its_width_on_the_name():
    """Design D2: the budget follows the frame.

    At 140 columns the panel offers 48 cells, so a 43-character title must
    arrive whole; at 71 it offers 30 and the same title is condensed. Composing
    against a fixed 32 left 34 cells of dead right gutter on the wide frame.
    """
    agg = _agg()
    title = "coder · Fix subagent effort levels per model"
    agg.by_session = {"abc123": next(iter(agg.by_session.values()))}
    setattr(agg, "session_names", {"abc123": title})

    wide = "\n".join(line.plain for line in build_report(agg, 140))
    narrow = "\n".join(line.plain for line in build_report(agg, 71))
    assert title in wide, "a 43-char title fits the wide frame's 48-cell column"
    assert title not in narrow
    narrow_row = next(li for li in narrow.splitlines() if "coder · Fix" in li)
    assert "…" in narrow_row, "a cut name must say it was cut"


def test_no_content_width_overruns_its_box_and_clips_a_column():
    """Design D8: no width band may lose a column off the right edge.

    The bug this pins was a CLIFF, not a slope. ``name_cap`` was
    ``30 if width < 96 else min(48, width - 40)``, so one extra cell of frame
    bought 18 cells of label while the rest of the row still needed 51-55 —
    the widest row jumped to 103 against a 96-cell box and terminal widths
    114-120 clipped ``% cache`` off every row, silently, because a row ending
    in ``cach`` still looks like a row.

    So this sweeps the BAND rather than sampling it. The sibling test above
    checks 71 and 140 and is blind to everything between them, which is exactly
    how the regression shipped: sampling cannot catch a breakpoint it does not
    happen to land on. Both cost profiles are swept because the cost column is
    sized to the widest figure present, so the overhead the name budget has to
    clear is itself a function of the data — a fixed allowance is wrong for one
    of them whichever constant is chosen.

    The sweep starts where the name budget is actually free to move. Below
    ``_MIN_NAME_COL`` + the row overhead the FLOOR wins by design — a frame too
    narrow to hold both a readable label and every column keeps the label and
    lets the row overrun, which is the pre-existing narrow-width behaviour the
    review confirmed as out of scope and which is byte-identical across this
    fix. This pins the band the budget controls; it is not a claim that a
    60-cell card fits.

    SEVERAL sessions, deliberately. One row alone does not reproduce: the name
    column is sized to the widest label present, so a lone long name is cut to
    the cap and the row lands exactly ON the box edge. It takes a second row
    holding the column at the full cap for the overrun to become visible —
    which is the ordinary shape of this table and was the shape of the ledger
    the review bisected.

    The sweep covers DATA PROFILES as well as widths, which is design review
    D11's structural lesson rather than an embellishment. The first version of
    this test swept every width in the band and still shipped a real defect,
    because it swept them all at ``calls=100``: the row overhead is a function
    of the DATA, so a width sweep at a single data profile pins one slice
    through a two-dimensional space and is blind along the other axis. Both
    axes that size a row are varied here — the cost figure (2-8 cells) and the
    call count (3-6 digits) — with magnitudes taken from the operator's real
    ledger, whose largest session ran 25,445 calls and whose ``anthropic``
    provider row ran 317,977. Against the hard-coded 4-cell calls allowance
    those two profiles fail at 6 and 12 widths respectively while ``calls=100``
    passes, which is exactly the blind spot that let D11 through a green test
    and an exported frame at once.
    """
    names = {
        "abc123def456": "Toggleable Sidebar for Session Switching in the TUI",
        "bbb222ccc333": "coder · Improve Support for Local Model Provider",
        "ccc333ddd444": "short one",
    }
    # (label, cost_micro, calls). 100 is the ordinary small-session case; the
    # other two are the ledger's real largest session and largest provider row.
    profiles = (
        ("cheap", 4_200_000, 100),
        ("expensive", 3_433_960_000, 100),
        ("cheap/5-digit-calls", 4_200_000, 25_445),
        ("expensive/5-digit-calls", 3_433_960_000, 25_445),
        ("cheap/6-digit-calls", 4_200_000, 317_977),
        ("expensive/6-digit-calls", 3_433_960_000, 317_977),
    )
    for label, cost_micro, calls in profiles:

        def _build(width: int, cost_micro: int = cost_micro, calls: int = calls) -> UsageAggregate:
            agg = _agg()
            base = next(iter(agg.by_session.values()))
            agg.by_session = {}
            for sid in names:
                scope = copy.copy(base)
                scope.cost_micro = cost_micro
                scope.calls = calls
                scope.cost_known_calls = calls
                agg.by_session[sid] = scope
            setattr(agg, "session_names", dict(names))
            return agg

        floor = _MIN_NAME_COL + _row_overhead(
            list(_build(_WIDE_TABLE_MIN).by_session.items()), _WIDE_TABLE_MIN
        )
        for width in range(floor, 161):
            agg = _build(width)

            text = "\n".join(line.plain for line in build_report(agg, width))
            rows = [
                li.rstrip()
                for li in text.split("By session", 1)[-1].splitlines()
                if " tokens" in li
            ]
            widest = max(cell_len(li) for li in rows)
            assert widest <= width, (
                f"{label} ledger at content width {width}: widest painted row is "
                f"{widest} cells, so {widest - width} cells fall off the box and the "
                f"rightmost column is clipped — {rows[0]!r}"
            )
            # State the consequence, not just the arithmetic. ``build_report``
            # composes rows without cropping — the crop happens when the widget
            # paints them into its content box — so a row that is too wide is
            # not visibly damaged in this text and asserting on its tail here
            # would prove nothing. Applying the box width is what turns the
            # cell count into the thing the reader actually loses: at 96 the
            # old rule painted 99 cells and the row ended in ``cach``.
            assert truncate_cells(rows[0], width).endswith(" cache"), (
                f"{label} ledger at content width {width}: the % cache column is "
                f"cut off when the row is painted into the box — "
                f"{truncate_cells(rows[0], width)!r}"
            )


def test_report_fits_the_box_the_scroll_container_actually_paints():
    """Design D8, on the SCREEN rather than in the arithmetic.

    The sweep above is a pure-function check: it compares composed row widths
    against the width ``build_report`` was handed. That is necessary and not
    sufficient, and the gap between the two is exactly where the first attempt
    at this fix went wrong. ``#analytics-scroll`` sets ``scrollbar-gutter:
    stable``, so the container reserves a column whether or not the bar is
    drawn and paints into one cell LESS than the card. A report composed
    against the full card width therefore passed every arithmetic assertion
    while the rendered frame still ended in ``cach`` — the defect was one layer
    below the one being measured.

    So this drives the real ``OperatorApp`` and asserts against
    ``scrollable_content_region``, the box Textual actually paints into. It
    covers both frames the review asked for; 114 is the first width that
    clipped and 120 is the ordinary terminal in the band.

    The session rows carry the ledger's real call magnitude (design review D11).
    At the fixture's original ``calls=100`` this test passed at both widths while
    the operator's own data clipped at both — the two widths this test exists to
    protect. A row's width is a function of its data, so a test that pins the
    geometry has to carry data of the size that occurs.
    """
    import asyncio

    async def run():
        for width in (114, 120):
            agg = _agg()
            base = next(iter(agg.by_session.values()))
            agg.by_session = {}
            for sid, calls in (
                ("abc123def456", 317_977),
                ("bbb222ccc333", 25_445),
                ("ccc333ddd444", 96),
            ):
                scope = copy.copy(base)
                scope.calls = calls
                scope.cost_known_calls = calls
                agg.by_session[sid] = scope
            setattr(
                agg,
                "session_names",
                {
                    "abc123def456": "Toggleable Sidebar for Session Switching in the TUI",
                    "bbb222ccc333": "coder · Improve Support for Local Model Provider",
                    "ccc333ddd444": "short one",
                },
            )
            app = OperatorApp(lambda: _factory(FakeSession()))
            async with app.run_test(size=(width, 40)) as pilot:
                screen = await _push(pilot, app, agg)
                painted = screen._scroll.scrollable_content_region.width
                assert screen._card_width() <= painted, (
                    f"at {width} columns the report is composed against "
                    f"{screen._card_width()} cells but the scroll container paints "
                    f"into {painted} — the rightmost column is cut"
                )
                text = "\n".join(line.plain for line in build_report(agg, screen._card_width()))
                rows = [
                    line.rstrip()
                    for line in text.split("By session", 1)[-1].splitlines()
                    if " tokens" in line
                ]
                for row in rows:
                    assert truncate_cells(row, painted).endswith(" cache"), (
                        f"at {width} columns the % cache column is cut off when the "
                        f"row is painted: {truncate_cells(row, painted)!r}"
                    )

    asyncio.run(run())


def test_calls_column_is_sized_by_the_data_not_by_a_constant():
    """A 6-digit call count widens the calls column instead of overrunning it.

    Design review D11. ``_row_overhead`` budgeted the calls column at a literal
    4 cells while ``_group_section`` painted it with a ``:>4`` PAD \u2014 and a pad
    grows rather than truncating, so a wider number pushed every column to its
    right off the box. The operator's ledger has ``anthropic`` at 317,977 calls
    and eight sessions past 9,999, which cost the ``% cache`` column on the 13
    most expensive rows of both tables across terminals 104-123.

    Two independent things are asserted, because the defect was a DISAGREEMENT
    between them rather than a fault in either alone:

    * the budget knows how wide the column will be (``_row_overhead`` grows with
      the digits), and
    * the paint agrees, so the column is aligned and the row still fits.

    The alignment half matters on its own: a per-row pad left ``317977 calls``
    and ``16 calls`` with their labels at different offsets, which defeats
    reading the column straight down.
    """
    small = _scoped(calls=16)
    large = _scoped(calls=317_977)

    budget_small = _row_overhead([("ollama", small)], 120)
    budget_large = _row_overhead([("anthropic", large)], 120)
    assert budget_large - budget_small == len("317977") - len("9999"), (
        "the row overhead must grow with the width of the call count; it read "
        f"{budget_small} for 16 calls and {budget_large} for 317,977"
    )

    # Mixed magnitudes in ONE table: the column is sized to the widest count,
    # every row is padded to that same width, and nothing overruns the box.
    agg = _agg()
    base = next(iter(agg.by_session.values()))
    agg.by_session = {}
    names = {}
    for sid, name, calls in (
        ("abc123def456", "Toggleable Sidebar for Session Switching in the TUI", 317_977),
        ("bbb222ccc333", "coder \u00b7 Improve Support for Local Model Provider", 25_445),
        ("ccc333ddd444", "short one", 16),
    ):
        scope = copy.copy(base)
        scope.calls = calls
        scope.cost_known_calls = calls
        agg.by_session[sid] = scope
        names[sid] = name
    setattr(agg, "session_names", names)

    for width in range(100, 141):
        text = "\n".join(line.plain for line in build_report(agg, width))
        rows = [
            li.rstrip() for li in text.split("By session", 1)[-1].splitlines() if " tokens" in li
        ]
        widest = max(cell_len(li) for li in rows)
        assert widest <= width, (
            f"at content width {width} a 6-digit call count pushes the row to "
            f"{widest} cells, clipping {widest - width} off the box: {rows[0]!r}"
        )
        offsets = {li.index(" calls") for li in rows}
        assert len(offsets) == 1, (
            "every row's ' calls' label must sit at the same offset for the column "
            f"to be readable; at width {width} they sit at {sorted(offsets)}"
        )


def _scoped(*, calls: int) -> UsageAggregate:
    """One per-provider/per-session scope with a given call count."""
    return UsageAggregate(
        calls=calls,
        input_tokens=500_000,
        output_tokens=120_000,
        context_tokens=3_500_000,
        cache_read_tokens=3_000_000,
        cost_micro=4_200_000,
        cost_known_calls=calls,
    )


def _nested_aggregate() -> UsageAggregate:
    """A root with two subagents and a grandchild, plus an unrelated session."""

    def scope(micro, calls=1):
        return UsageAggregate(
            calls=calls,
            ok_calls=calls,
            context_tokens=micro,
            cost_micro=micro,
            cost_known_calls=calls,
        )

    agg = scope(15_000_000, calls=5)
    agg.by_session = {
        "rootsession": scope(1_000_000),
        "kid1session": scope(2_000_000),
        "kid2session": scope(3_000_000),
        "grandkidses": scope(4_000_000),
        "solosession": scope(5_000_000),
    }
    setattr(agg, "session_names", {"rootsession": "Review and merge open PRs"})
    setattr(
        agg,
        "session_parents",
        {"kid1session": "rootsession", "kid2session": "rootsession", "grandkidses": "kid1session"},
    )
    return agg


def _is_child_row(row: str) -> bool:
    """A child row is marked by the └ glyph, not merely by indentation.

    Design D4 added the glyph because 21% of ROOTS are unnamed 12-hex ids, so an
    unnamed root and a child differed only by two spaces and a colour — a
    distinction that vanishes under NO_COLOR. Tests classify rows the same way
    the reader does, so this stays true if the indent width ever changes.
    """
    return row.lstrip().startswith("└")


def _session_rows(text: list[str]) -> list[str]:
    start = next(i for i, line in enumerate(text) if "By session" in line)
    rows = []
    for line in text[start].split("\n")[1:]:
        if line.strip():
            rows.append(line)
    return rows


def _all_expanded(aggregate: UsageAggregate) -> set[str]:
    """Every session id with children, i.e. the fully-expanded table.

    The table is COLLAPSED by default (that is the point of the feature), so a
    test about how a CHILD row renders has to say which rows are open. Derived
    from the forest rather than hard-coded per fixture, so a fixture gaining a
    child does not silently stop being covered.
    """
    parents = getattr(aggregate, "session_parents", {}) or {}
    return {
        parent
        for child, parent in parents.items()
        if child != parent and child in aggregate.by_session and parent in aggregate.by_session
    }


def _row_name(row: str) -> str:
    """The name a reader sees on ``row``, stripped of everything around it.

    Rows lead with the cursor gutter, the nesting indent and glyph, and the
    disclosure marker, and expandable rows trail a ``+N subagents`` count. All of
    those are chrome around the name; a test asking "what does this row call
    itself" must see through them, and doing it in ONE place is what stops each
    test growing its own slightly different parser.
    """
    name = row.strip().lstrip("❯").strip().lstrip("└").strip()
    name = name.lstrip("▸▾").strip()
    # The name field ends at the run of spaces padding it out to ``name_col``.
    name = name.split("   ")[0].rstrip()
    return re.sub(r" \+\d+ subagents?$", "", name)


def test_session_table_shows_roots_with_tree_totals_and_indents_children():
    # Expanded explicitly: the table now hides children until a row is opened,
    # and this test is about how the hierarchy renders once it IS open.
    aggregate = _nested_aggregate()
    lines = [line.plain for line in build_report(aggregate, 120, expanded=_all_expanded(aggregate))]
    rows = _session_rows(lines)
    # Two top-level rows only; the three children moved under their root.
    top = [r for r in rows if not _is_child_row(r)]
    assert len(top) == 2
    root = next(r for r in rows if "Review and merge open PRs" in r)
    assert not _is_child_row(root)
    assert "$10.00" in root  # own $1 + kid1 $2 + kid2 $3 + grandkid $4
    # Children are present, indented, and reachable — not deleted.
    assert any(_is_child_row(r) and "kid1session" in r for r in rows)
    assert any(_is_child_row(r) and "grandkidses" in r for r in rows)
    # Depth still reads as depth: the grandchild is one indent step deeper.
    grandkid = next(r for r in rows if "grandkidses" in r)
    kid = next(r for r in rows if "kid1session" in r)
    assert grandkid.index("└") == kid.index("└") + 2
    # And the section says the roots include their subagents.
    assert "totals include subagents" in "\n".join(lines)


def test_session_column_still_sums_to_the_headline_total():
    """THE invariant (design §7 risk 1): a rolled-up column that still lists
    children inflates the operator's real table by $8,077. Only ROOT rows may
    carry a tree total, and they must add to the total printed above them."""
    aggregate = _nested_aggregate()
    lines = [line.plain for line in build_report(aggregate, 120)]
    rows = _session_rows(lines)
    top = [r for r in rows if not _is_child_row(r)]
    total = 0.0
    for row in top:
        money = next(part for part in row.split() if part.startswith("$"))
        total += float(money.strip("+").lstrip("$"))
    assert abs(total - aggregate.cost_usd) < 0.01


def test_a_flat_ledger_renders_exactly_as_before():
    """No parent edges (an old ledger, or a machine that never ran subagents)
    means every session is a root and the table is byte-identical to today's."""
    aggregate = _nested_aggregate()
    setattr(aggregate, "session_parents", {})
    rows = _session_rows([line.plain for line in build_report(aggregate, 120)])
    assert len(rows) == 5
    assert not any(_is_child_row(r) for r in rows)
    assert "totals include subagents" not in rows[0]


def test_nesting_does_not_cost_the_cache_column_on_a_narrow_frame():
    """Risk 4: indentation must not push the table past _WIDE_TABLE_MIN and
    shed the cache column that a wide frame keeps."""
    aggregate = _nested_aggregate()
    wide = _session_rows([line.plain for line in build_report(aggregate, 120)])
    narrow = _session_rows([line.plain for line in build_report(aggregate, 80)])
    assert all("cache" in r for r in wide)
    assert all("cache" in r for r in narrow)
    # Below the threshold cache sheds for every row equally, root and child.
    tight = _session_rows([line.plain for line in build_report(aggregate, 60)])
    assert not any("cache" in r for r in tight)
    assert all("$" in r for r in tight)  # the cost column survives, as designed


def test_legend_is_drawn_for_a_plus_that_only_the_rollup_produces():
    """A parent priced in full whose CHILD is unpriced draws a ``+`` on the root
    row that no individual session shows. The footnote must follow it."""
    agg = UsageAggregate(
        calls=3, ok_calls=3, context_tokens=3, cost_micro=1_000_000, cost_known_calls=2
    )
    agg.by_session = {
        "rootsession": UsageAggregate(
            calls=2, ok_calls=2, context_tokens=2, cost_micro=1_000_000, cost_known_calls=2
        ),
        "kid1session": UsageAggregate(
            calls=1, ok_calls=1, context_tokens=1, cost_micro=0, cost_known_calls=0
        ),
    }
    setattr(agg, "session_names", {})
    setattr(agg, "session_parents", {"kid1session": "rootsession"})
    text = "\n".join(line.plain for line in build_report(agg, 120))
    assert "$1.00+" in text
    assert "lower bound" in text


# -- convergence: the nesting and the label budget must both hold --------------
# These pin the INTERACTION between two changes that landed from different
# branches on the same table (#716 nested the rows; #717 budgeted their labels
# and measured the columns). Each was verified against a fixture that did not
# exercise the other, and the failure mode is invisible to both: a nested label
# composed to the full budget and THEN given its ``└ `` prefix is over the
# column it is padded into, so the paint cuts it back off without a marker.


def _deep_named_aggregate(name: str = "Toggleable Sidebar for Session Switching in the TUI"):
    """A root and a child that both carry LONG names, so the budget actually binds.

    The nesting fixtures above name only the root, which is the real-ledger shape
    but cannot see a prefix overrunning the budget — an unnamed child renders a
    12-hex id that fits at any width.
    """

    def scope(micro, calls=1):
        return UsageAggregate(
            calls=calls,
            ok_calls=calls,
            context_tokens=micro,
            cost_micro=micro,
            cost_known_calls=calls,
        )

    agg = scope(6_000_000, calls=4)
    agg.by_session = {
        "rootsession": scope(1_000_000),
        "kid1session": scope(2_000_000),
        "grandkidses": scope(3_000_000),
    }
    setattr(
        agg,
        "session_names",
        {"rootsession": name, "kid1session": f"reviewer · {name}", "grandkidses": f"qa · {name}"},
    )
    setattr(agg, "session_parents", {"kid1session": "rootsession", "grandkidses": "kid1session"})
    return agg


def test_a_nested_label_pays_for_its_own_indent():
    """The child's ``└ `` prefix comes OUT of the label budget, not on top of it.

    Asserted on ``prefix + label`` against the budget directly, because the
    symptom is NOT a row that overruns the frame: ``name_col`` is sized to the
    widest label present, so an over-budget nested label simply widens the
    column and every row still fits. What is lost is a CHARACTER — the label was
    composed to the full budget, the prefix pushed it past the column, and the
    paint's truncation takes the tail back off without a marker. Measuring the
    rendered row therefore cannot see this; measuring the label against the
    budget it was promised can.

    Verified to fail against the mutant (``budget = name_cap``): at a 48-cell cap
    a depth-2 label renders 50 cells.
    """
    # The aggregate is irrelevant here — only the id and the depth decide a
    # label's budget — but it is a real one so the row shape matches the paint.
    scope = UsageAggregate(calls=1, ok_calls=1, context_tokens=1, cost_micro=1, cost_known_calls=1)
    structure = [("rootsession", 0, scope), ("kid1session", 1, scope), ("grandkidses", 2, scope)]
    long_name = "Toggleable Sidebar for Session Switching in the TUI"
    names = {
        "rootsession": long_name,
        "kid1session": f"reviewer · {long_name}",
        "grandkidses": f"qa · {long_name}",
    }
    for cap in range(_MIN_NAME_COL, _MAX_NAME_COL + 1):
        labels = _forest_labels(structure, names, cap)
        for sid, depth, _ in structure:
            rendered = _row_prefix(depth) + labels[sid]
            assert cell_len(rendered) <= cap, (
                f"cap {cap}, depth {depth}: {rendered!r} is {cell_len(rendered)} cells — "
                "the indent was not paid for out of the budget"
            )


def test_no_session_row_is_cut_without_a_marker_at_any_width():
    """Every name the reader sees is either COMPLETE or ends in ``…``.

    The third state is the defect: a name that is a strict prefix of the real
    one with no marker, which reads as a complete title and is not. Nested rows
    are the case that matters here, because their prefix is what pushes a label
    over the column that then cuts it.
    """
    aggregate = _deep_named_aggregate()
    known = set((getattr(aggregate, "session_names", {}) or {}).values())
    expanded = _all_expanded(aggregate)
    for width in range(96, 161):
        # Swept in BOTH states: the disclosure marker and the ``+N subagents``
        # count are charged to the same label budget the nesting prefix is, so
        # each state composes a different label from the same name and either
        # could be the one that overruns.
        for opened in ((), expanded):
            lines = [line.plain for line in build_report(aggregate, width, expanded=opened)]
            block = "\n".join(lines).split("By session", 1)[-1]
            for row in (r.rstrip() for r in block.splitlines() if " tokens" in r):
                name = _row_name(row)
                if name.endswith("…"):
                    continue  # a MARKED cut is exactly what this fix produces
                assert name in known, (
                    f"width {width}: {name!r} is neither a complete name nor marked as cut "
                    f"— it is a silent fragment"
                )


def test_the_budget_is_measured_over_subtree_totals_not_own_spend():
    """A root row paints its SUBTREE total, so that is what the columns must fit.

    Budgeting against ``by_session``'s own aggregates understates the cost and
    calls columns by however much the children add, which is the D8/D11 clipping
    one rollup later.
    """
    aggregate = _deep_named_aggregate()
    own = _row_overhead(list(aggregate.by_session.items()), 120)
    # what the table actually paints: the root carries 1+2+3 == 6, not 1
    rolled = _row_overhead([("rootsession", aggregate)], 120)
    assert rolled >= own, "the subtree total cannot need LESS room than one part"
    text = "\n".join(line.plain for line in build_report(aggregate, 120))
    rows = [r.rstrip() for r in text.split("By session", 1)[-1].splitlines() if " tokens" in r]
    assert all(truncate_cells(r, 120) == r for r in rows)


def test_two_children_of_one_parent_never_render_the_same_label():
    """Sibling subagents compose byte-identical names; the rows must still differ.

    ``<role> · <parent title>`` is the real composed shape, so every sibling
    delegated under one parent with one role collides by construction — the
    operator's ledger has parents with 46, 29 and 24 such children. Nesting does
    not excuse the collision: the indent says "child of the row above", not
    "a different session from the one below".
    """

    def scope(micro):
        return UsageAggregate(
            calls=1, ok_calls=1, context_tokens=micro, cost_micro=micro, cost_known_calls=1
        )

    agg = scope(4_000_000)
    kids = ["aa01000000c1", "aa02000000c2", "aa03000000c3"]
    agg.by_session = {"rootsession": scope(1_000_000), **{k: scope(1_000_000) for k in kids}}
    shared = "reviewer · Toggleable Sidebar for Session Switching"
    setattr(agg, "session_names", {"rootsession": "Root", **{k: shared for k in kids}})
    setattr(agg, "session_parents", {k: "rootsession" for k in kids})

    text = "\n".join(line.plain for line in build_report(agg, 120, expanded=_all_expanded(agg)))
    rows = [r.rstrip() for r in text.split("By session", 1)[-1].splitlines() if " tokens" in r]
    names = [_row_name(r) for r in rows if _is_child_row(r)]
    assert len(names) == len(kids), names
    assert len(set(names)) == len(names), f"sibling rows collide: {names}"


# ---------------------------------------------------------------------------
# Collapsible session rows.
#
# The report against v0.52.15 was "my screen is freezing on /analytics ... each
# session shows hundreds of rows of subagent cost". On the operator's ledger the
# table painted 2,240 rows of which 1,645 were subagents nobody asked to see,
# and the whole report was rebuilt on EVERY resize event — a terminal drag emits
# a burst of them, which is what wedged the terminal. These pin the fix: the
# default view is roots only, expanding is per-row and keyed by session id, and
# the state survives the rebuilds that used to be free to discard it.
# ---------------------------------------------------------------------------


def test_children_are_hidden_until_their_root_is_expanded():
    """THE fix: the default table is roots only, and nothing is lost by it."""
    aggregate = _nested_aggregate()
    rows = _session_rows([line.plain for line in build_report(aggregate, 120)])
    assert len(rows) == 2, rows  # the root and the unrelated session
    assert not any(_is_child_row(r) for r in rows)
    # Nothing was DELETED — the same report expanded reaches every session.
    opened = _session_rows(
        [line.plain for line in build_report(aggregate, 120, expanded=_all_expanded(aggregate))]
    )
    assert len(opened) == 5, opened


def test_a_collapsed_root_still_sums_to_the_headline_total():
    """The load-bearing invariant, restated for the collapsed view.

    Roots already carry their subtree's spend (``build_session_forest`` rolls it
    up), so hiding the children cannot change what the column adds to. This is
    the property the whole rollup exists to protect, and a collapse that broke it
    would be a worse defect than the one being fixed.
    """
    aggregate = _nested_aggregate()
    rows = _session_rows([line.plain for line in build_report(aggregate, 120)])
    total = 0.0
    for row in rows:
        money = next(part for part in row.split() if part.startswith("$"))
        total += float(money.strip("+").lstrip("$"))
    assert abs(total - aggregate.cost_usd) < 0.01


def test_only_an_expandable_row_carries_a_disclosure_glyph():
    """A childless row must not look interactive: 492 of 595 real roots are one.

    A glyph on a row whose Enter does nothing is the "never advertise a dead
    control" rule (D5/U1/U4) in its per-row form.
    """
    aggregate = _nested_aggregate()
    rows = _session_rows([line.plain for line in build_report(aggregate, 120)])
    parent = next(r for r in rows if "Review and merge open PRs" in r)
    childless = next(r for r in rows if "solosession" in r)
    assert "▸" in parent
    assert "▸" not in childless and "▾" not in childless
    # The glyph flips to point DOWN once the row is open, so the frame says which
    # state it is in rather than only which action is available.
    opened = _session_rows(
        [line.plain for line in build_report(aggregate, 120, expanded={"rootsession"})]
    )
    parent_open = next(r for r in opened if "Review and merge open PRs" in r)
    assert "▾" in parent_open and "▸" not in parent_open


def test_an_expandable_row_says_how_much_is_hidden_under_it():
    """The count is the information the collapse removes, handed back.

    Without it an expensive leaf and an expensive session with three subagents
    under it are indistinguishable — which is the distinction the reader opened
    the screen to make.
    """
    aggregate = _nested_aggregate()
    rows = _session_rows([line.plain for line in build_report(aggregate, 120)])
    parent = next(r for r in rows if "Review and merge open PRs" in r)
    # Three descendants: two children and one grandchild. The SUBTREE, not the
    # direct children, because that is what expanding eventually reveals.
    assert "+3 subagents" in parent
    assert "subagent" not in next(r for r in rows if "solosession" in r)


def test_one_hidden_row_is_singular():
    """A count is prose the user reads; "+1 subagents" is a visible defect."""
    agg = UsageAggregate(calls=2, ok_calls=2, context_tokens=2, cost_micro=2, cost_known_calls=2)
    scope = UsageAggregate(calls=1, ok_calls=1, context_tokens=1, cost_micro=1, cost_known_calls=1)
    agg.by_session = {"rootsession": scope, "kid1session": scope}
    setattr(agg, "session_names", {"rootsession": "Root"})
    setattr(agg, "session_parents", {"kid1session": "rootsession"})
    rows = _session_rows([line.plain for line in build_report(agg, 120)])
    assert "+1 subagent " in next(r for r in rows if "Root" in r) + " "


def test_a_flat_ledger_pays_nothing_for_a_gutter_it_cannot_use():
    """No subagents anywhere means the table renders as it always has.

    The disclosure column is charged to every row so the names stay in one
    column, which is only defensible when SOMETHING can be expanded. A permanent
    two-cell blank advertising an absent affordance is the same defect as a dead
    hint.
    """
    aggregate = _nested_aggregate()
    setattr(aggregate, "session_parents", {})
    rows = _session_rows([line.plain for line in build_report(aggregate, 120)])
    assert len(rows) == 5
    for row in rows:
        # Two leading cells for the cursor gutter and then straight into the
        # name — no third pair of blanks where the glyph would sit.
        assert not row.startswith("    "), row
    assert "totals include subagents" not in "\n".join(rows)


def test_the_rollup_note_survives_the_collapse():
    """A root's figure exceeds its own spend in the DEFAULT view too.

    The meta used to be derived from whether any indented row was on screen,
    which with the table collapsed is never — dropping the one sentence that
    explains the inflated figure exactly when the reader most needs it.
    """
    text = "\n".join(line.plain for line in build_report(_nested_aggregate(), 120))
    assert "totals include subagents" in text


def test_expanding_one_root_does_not_expand_its_sibling():
    """Expansion is per row, not a global mode."""
    aggregate = _nested_aggregate()
    rows = _session_rows(
        [line.plain for line in build_report(aggregate, 120, expanded={"rootsession"})]
    )
    # The root's two direct children appear; the grandchild does NOT, because its
    # own parent (kid1session) is still closed.
    assert any(_is_child_row(r) and "kid1session" in r for r in rows)
    assert not any("grandkidses" in r for r in rows)


def test_the_label_budget_pays_for_the_marker_and_the_count():
    """Whatever a row prints beside its name is width the name cannot have.

    Same rule the nesting prefix already follows, and the same failure if it is
    skipped: a label composed to the full budget and THEN given a marker is over
    the column, so the paint cuts the tail off with no ``…`` to say it did.
    """
    scope = UsageAggregate(calls=1, ok_calls=1, context_tokens=1, cost_micro=1, cost_known_calls=1)
    long_name = "Toggleable Sidebar for Session Switching in the TUI"
    names = {"rootsession": long_name, "kid1session": f"reviewer · {long_name}"}
    structure = [("rootsession", 0, scope), ("kid1session", 1, scope)]
    for cap in range(_MIN_NAME_COL, _MAX_NAME_COL + 1):
        # A root with a marker and a count; the child with neither.
        reserved = {"rootsession": cell_len("▸ ") + cell_len(" +1 subagent"), "kid1session": 0}
        labels = _forest_labels(structure, names, cap, reserved=reserved)
        rendered = "▸ " + labels["rootsession"] + " +1 subagent"
        assert cell_len(rendered) <= cap, f"cap {cap}: {rendered!r} is {cell_len(rendered)} cells"


def test_collapsing_leaves_far_fewer_rows_to_compose():
    """The structural reason the screen got fast, asserted as a fact not a clock.

    A wall-clock bound here would be a bet on machine load. The row count is what
    the paint cost is proportional to and it is deterministic, so it is the
    honest assertion — the measured timings live on the PR.
    """
    aggregate = _nested_aggregate()
    forest = build_session_forest(
        aggregate.by_session, getattr(aggregate, "session_parents", {}) or {}
    )
    assert len(_forest_rows(forest)) == 2
    assert len(_forest_rows(forest, _all_expanded(aggregate))) == 5


# -- the interactive screen ---------------------------------------------------


def _nested_screen_agg() -> UsageAggregate:
    """``_nested_aggregate`` with the fields the SCREEN needs to mount."""
    agg = _nested_aggregate()
    agg.components = {k: 0 for k in COMPONENT_KEYS}
    agg.components["conversation"] = 1_000
    agg.by_provider = {"anthropic": copy.deepcopy(agg)}
    return agg


def test_enter_expands_the_row_the_cursor_is_on():
    """The user's words: "hits enter to expand that row".

    No priming press: the cursor is placed at MOUNT (review D2), so the FIRST
    ``enter`` does the thing the hint advertises instead of arming the next one.
    """
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            assert "kid1session" not in "\n".join(screen.render_lines_for_test())
            await pilot.press("enter")
            await pilot.pause()
            body = "\n".join(screen.render_lines_for_test())
            assert "kid1session" in body
            # And Enter again closes it: one key, both directions.
            await pilot.press("enter")
            await pilot.pause()
            assert "kid1session" not in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def test_the_cursor_is_on_screen_from_the_first_frame():
    """A press must do what it says on its FIRST try, not arm the next one.

    The hint says ``enter expand`` on the opening frame, so a cursor has to
    exist there (review D2) — otherwise the advertised key takes two presses.
    Placed without scrolling, so the caret costs the reader none of the totals
    block they opened the screen to read.
    """
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            assert screen._cursor == screen._layout.session_rows[0].session_id
            assert screen._scroll.scroll_offset.y == 0, "placing the cursor scrolled the report"
            assert _ROW_CURSOR in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def test_the_cursor_clamps_at_both_ends():
    """A full-page surface clamps; wrapping would throw the reader out of the
    section they are working in (the documented ``/settings`` exception)."""
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            for _ in range(8):
                await pilot.press("down")
                await pilot.pause()
            assert screen._cursor == screen._layout.session_rows[-1].session_id
            for _ in range(8):
                await pilot.press("up")
                await pilot.pause()
            assert screen._cursor == screen._layout.session_rows[0].session_id

    asyncio.run(run())


def test_enter_on_a_childless_row_does_nothing():
    """No glyph, no action — the row is not a control."""
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            # solosession is the most expensive row, so it sorts first.
            await pilot.press("down")
            await pilot.pause()
            row = next(r for r in screen._layout.session_rows if r.session_id == "solosession")
            assert not row.expandable
            screen._cursor = "solosession"
            before = screen.render_lines_for_test()
            await pilot.press("enter")
            await pilot.pause()
            assert screen.render_lines_for_test() == before

    asyncio.run(run())


def test_left_collapses_an_open_row_and_then_steps_out_to_the_parent():
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            screen._cursor = "rootsession"
            await pilot.press("right")
            await pilot.pause()
            assert "kid1session" in "\n".join(screen.render_lines_for_test())
            # Standing on the CHILD, left steps out rather than doing nothing.
            screen._cursor = "kid1session"
            await pilot.press("left")
            await pilot.pause()
            assert screen._cursor == "rootsession"
            # And now left closes the row it stepped out to.
            await pilot.press("left")
            await pilot.pause()
            assert "kid1session" not in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def test_expansion_survives_the_metric_toggle():
    """``t`` rebuilds the whole report; it must not silently close what the user
    opened. State is keyed by session ID precisely so a rebuild can find it."""
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg(), daily=_daily())
            screen._cursor = "rootsession"
            await pilot.press("enter")
            await pilot.pause()
            assert "kid1session" in "\n".join(screen.render_lines_for_test())
            await pilot.press("t")
            await pilot.pause()
            assert "kid1session" in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def test_expansion_survives_a_resize():
    """A resize recomposes every label at a new width, so anything remembered by
    LABEL would be forgotten here. This is the regression that keying by id
    prevents, and the resize path is also the one that skips rebuilds."""
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            screen._cursor = "rootsession"
            await pilot.press("enter")
            await pilot.pause()
            assert "kid1session" in "\n".join(screen.render_lines_for_test())
            await pilot.resize_terminal(150, 40)
            await pilot.pause()
            await pilot.pause()
            assert "kid1session" in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def test_a_resize_that_does_not_change_the_card_width_does_not_rebuild():
    """The freeze itself. A drag emits a burst of resize events and each rebuild
    was 0.5-0.6 s on a real ledger; the card is a floored 90% of the terminal, so
    most of those events cannot change a single cell of output.

    Asserted structurally (did the renderer run?) rather than on a clock, which
    would be a bet on machine load."""
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            calls = 0
            original = screen._report_lines

            def counted():
                nonlocal calls
                calls += 1
                return original()

            setattr(screen, "_report_lines", counted)
            width_before = screen._card_width()
            # 110 -> 111 keeps the floored card width identical.
            await pilot.resize_terminal(111, 40)
            await pilot.pause()
            assert screen._card_width() == width_before, "fixture no longer shares a width band"
            assert calls == 0, "a resize that changes nothing rebuilt the whole report"
            # A resize that DOES cross a band still repaints, or the frame lies.
            await pilot.resize_terminal(150, 40)
            await pilot.pause()
            assert calls >= 1

    asyncio.run(run())


def test_e_expands_everything_and_then_collapses_everything():
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            await pilot.press("e")
            await pilot.pause()
            body = "\n".join(screen.render_lines_for_test())
            # Every session, grandchild included, is reachable in one press.
            assert "grandkidses" in body and "kid2session" in body
            await pilot.press("e")
            await pilot.pause()
            assert "grandkidses" not in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def test_e_expands_all_when_only_some_rows_are_open():
    """ "Expand all" must not mean "collapse" because one row happened to be open.

    Found by driving the real screen: with a single row expanded, ``e`` closed
    everything — the opposite of what a reader pressing it means. The test is
    whether everything is ALREADY open, not whether anything is.
    """
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            screen._cursor = "rootsession"
            await pilot.press("enter")  # one row open, the grandchild still not
            await pilot.pause()
            assert "grandkidses" not in "\n".join(screen.render_lines_for_test())
            await pilot.press("e")
            await pilot.pause()
            assert "grandkidses" in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def test_the_hint_advertises_the_expand_keys_only_when_a_row_can_expand():
    """D5: never advertise a dead control."""
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            assert "enter expand" in screen._hint_text(scrollable=True).plain
            flat = _nested_screen_agg()
            setattr(flat, "session_parents", {})
            screen2 = await _push(pilot, app, flat)
            assert "enter expand" not in screen2._hint_text(scrollable=True).plain

    asyncio.run(run())


def test_the_cursor_is_scrolled_into_view_when_it_moves():
    """The cursor may be left off screen by the wheel, so a key that ACTS on it
    has to reveal it first — otherwise it writes to a row the reader cannot see
    and the frame appears not to change."""
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 12)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            # Bring the table into view first. Above it the arrows line-scroll
            # (there is no row to address up there — R1/D1/U1), so the cursor
            # contract this test is about only applies once rows are on screen.
            await pilot.press("end")
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            index = screen._cursor_index()
            assert index is not None
            line = screen._layout.session_first_line + index
            top = screen._scroll.scroll_offset.y
            height = screen._scroll.size.height
            assert (
                top <= line < top + height
            ), f"cursor line {line} is outside the viewport {top}..{top + height}"

    asyncio.run(run())


def test_the_row_keys_work_on_a_report_too_tall_to_fit():
    """Focus sits on the inner ``VerticalScroll``, which eats the arrows while it
    can still scroll — so a non-priority binding reaches the screen only at the
    ends of the travel.

    Pinned at two heights because the symptom is invisible at one: the cursor
    moved fine at 110x40, where the body fits, and never at 110x20, where it does
    not. A single-height test would have passed against the broken binding.
    """
    import asyncio

    async def run():
        for height in (40, 20, 12):
            app = OperatorApp(lambda: _factory(FakeSession()))
            async with app.run_test(size=(110, height)) as pilot:
                screen = await _push(pilot, app, _nested_screen_agg())
                await pilot.press("down")
                await pilot.pause()
                assert (
                    screen._cursor is not None
                ), f"the arrow never reached the screen at h={height}"

    asyncio.run(run())


def _tall_report_agg(roots: int = 40) -> UsageAggregate:
    """A report shaped like the operator's real ledger: a table far BELOW the fold.

    The defect these tests pin is only visible when the session table starts
    below the viewport — on the real ledger it begins ~57 body lines down, behind
    the totals block, both bar charts and the input attribution. The small
    ``_nested_screen_agg`` fixture puts the table 17 lines down, which fits a
    40-row terminal whole and hid the bug from every existing test.
    """

    def scope(micro, calls=1):
        return UsageAggregate(
            calls=calls,
            ok_calls=calls,
            context_tokens=micro,
            cost_micro=micro,
            cost_known_calls=calls,
        )

    agg = scope(1_000_000 * (roots + 2), calls=roots + 2)
    by_session = {f"root{i:08d}": scope(1_000_000 * (roots - i)) for i in range(roots)}
    # One expandable root so the disclosure gutter and the expand hints are live,
    # exactly as on the real ledger where 103 of 595 roots have children.
    by_session["kid00000001"] = scope(500_000)
    agg.by_session = by_session
    setattr(agg, "session_parents", {"kid00000001": "root00000000"})
    setattr(agg, "session_names", {f"root{i:08d}": f"Session number {i}" for i in range(roots)})
    agg.components = {k: 0 for k in COMPONENT_KEYS}
    agg.components["conversation"] = 1_000
    agg.by_provider = {"anthropic": copy.deepcopy(agg)}
    return agg


def test_the_top_of_the_report_is_reachable_with_the_arrow_keys():
    """R1/R2/D1/U1/U2 — four review streams, one root cause.

    ``↑``/``↓`` became ``priority=True`` row-moves, and the cursor only ever
    addresses SESSION-TABLE rows. On the real ledger that table starts ~57 body
    lines down, so every press re-pinned the cursor to table row 0 and dragged
    the viewport back to it: measured at 120x40 on the operator's ledger,
    ``end`` then 400 ``up`` presses left the viewport stuck at y=56 forever
    while the hint on that same frame read ``↑↓ row``.

    The invariant: from the BOTTOM, arrows alone must walk the viewport all the
    way to y=0, one line per press where there is no row to move. Asserted as
    reachability rather than as a per-press delta because the cursor legitimately
    consumes presses while it crosses the table.
    """
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 24)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            scroll = screen._scroll
            assert scroll.max_scroll_y > 0, "fixture is not taller than the viewport"
            assert (
                screen._layout.session_first_line > scroll.size.height
            ), "fixture's table is not below the fold, so it cannot pin this defect"

            await pilot.press("end")
            await pilot.pause()
            bottom = scroll.scroll_offset.y
            assert bottom > 0

            # One ``up`` must never be a large backward jump (U2b: 570 lines).
            await pilot.press("up")
            await pilot.pause()
            assert bottom - scroll.scroll_offset.y <= 1, (
                f"one 'up' moved {bottom - scroll.scroll_offset.y} lines — the cursor re-home "
                "is teleporting the viewport"
            )

            # Generous budget: the cursor eats one press per table row on the way
            # through, and the assertion is reachability, not press count.
            for _ in range(bottom + len(screen._layout.session_rows) + 10):
                if scroll.scroll_offset.y <= 0:
                    break
                await pilot.press("up")
            await pilot.pause()
            floor = scroll.scroll_offset.y
            assert floor == 0, f"the top of the report is unreachable by arrow (floor at y={floor})"

    asyncio.run(run())


def test_the_hint_fits_one_line_at_every_width():
    """U3 — the hint grew 36→56 cells but its ``Static`` is ONE content line.

    ``#analytics-hint`` is ``height: 2`` with ``padding-top: 1``, so anything
    wider than the box wraps to a second line that is never painted. At 60x20
    the user lost ``· e all``; at 50x16 they lost ``enter expand · e all`` and the
    visible copy ended on a dangling ``·`` — exactly where the table is most
    cramped, the expand keys stopped being documented while the ``▸`` glyphs
    stayed on screen inviting the press.

    Pinned at the two widths the review measured plus the ones that must not
    regress, and the assertion is against the widget's REAL content box rather
    than an assumed width.
    """
    import asyncio

    async def run():
        for width, height in ((50, 16), (60, 20), (70, 24), (110, 24), (160, 40)):
            app = OperatorApp(lambda: _factory(FakeSession()))
            async with app.run_test(size=(width, height)) as pilot:
                screen = await _push(pilot, app, _tall_report_agg())
                box = screen._hint.content_size.width
                text = screen._hint_text(scrollable=True).plain
                assert cell_len(text) <= box, (
                    f"hint is {cell_len(text)} cells in a {box}-cell box at {width}x{height}: "
                    f"{text!r} would wrap to an unpainted second line"
                )
                # Degrade by dropping whole clauses, never by losing the tail.
                assert not text.rstrip().endswith(
                    "·"
                ), f"hint ends on a dangling separator: {text!r}"
                # The keys a reader cannot guess must survive every tier.
                assert "enter" in text and "e all" in text, (
                    f"the expand keys are undocumented at {width}x{height} while the ▸ glyphs "
                    f"remain on screen: {text!r}"
                )

    asyncio.run(run())


def test_the_hint_says_row_even_on_a_subagent_free_ledger():
    """R2 — the hint read ``↑↓ scroll`` where the keys did row-jumps.

    ``_move_cursor`` never consults expandability, so the cursor moves on every
    ledger that has session rows. Gating the wording on *expandability* made the
    hint describe a control the screen did not have.
    """
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 24)) as pilot:
            flat = _tall_report_agg()
            setattr(flat, "session_parents", {})
            screen = await _push(pilot, app, flat)
            text = screen._hint_text(scrollable=True).plain
            assert "↑↓ row" in text, text
            assert "↑↓ scroll" not in text, text
            # and no dead controls advertised: nothing can expand here.
            assert "expand" not in text and "e all" not in text, text

    asyncio.run(run())


def test_collapsing_keeps_the_cursor_on_the_visible_ancestor():
    """U2c — collapse left ``_cursor`` naming a HIDDEN row.

    ``_cursor_index()`` then returned ``None`` and the next arrow re-homed to
    table row 0, teleporting the viewport ~200 lines on the real ledger. The
    ancestry needed to land on the root that swallowed the row is already in the
    layout, so the cursor follows it there instead.
    """
    import asyncio

    async def run():
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            await pilot.press("enter")  # open the root the cursor mounted on
            await pilot.pause()
            await pilot.press("down")  # step onto its child
            await pilot.pause()
            child = screen._cursor_row()
            assert child is not None and child.depth == 1

            # ``e`` with only SOME rows open expands all (the documented
            # contract); the second press is the collapse-all that hides the
            # child the cursor is standing on.
            await pilot.press("e")
            await pilot.pause()
            await pilot.press("e")
            await pilot.pause()
            assert screen._cursor_index() is not None, "the cursor was orphaned on a hidden row"
            assert screen._cursor_row().depth == 0  # type: ignore[union-attr]
            assert screen._cursor == "rootsession", screen._cursor

    asyncio.run(run())
