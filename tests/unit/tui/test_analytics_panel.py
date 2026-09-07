"""The ``/analytics /usage`` screen — the report it renders and its Esc close.

The renderer is a pure function of the aggregate, so most of this asserts plain
strings (what a user reads). Two pilot tests drive the REAL ``OperatorApp`` so
the screen actually mounts under the stylesheet and Esc/``q`` return to the
previous view — a passing text assertion is not evidence a TUI looks right, but
it is the right way to pin what the screen SAYS and that it closes.
"""

from __future__ import annotations

import copy

from rich.cells import cell_len

from local_operator.analytics.model import COMPONENT_KEYS, UsageAggregate, UsagePeriod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.analytics_panel import (
    _MIN_NAME_COL,
    _WIDE_TABLE_MIN,
    METRIC_COST,
    METRIC_TOKENS,
    AnalyticsScreen,
    _row_overhead,
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


def test_session_table_shows_roots_with_tree_totals_and_indents_children():
    lines = [line.plain for line in build_report(_nested_aggregate(), 120)]
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
