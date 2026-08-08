"""The `/usage` popup — rendering, scrolling, and the numbers it must state.

The rows are a pure function of ``(reports, width, now)``, so almost everything
here runs without an app. The three tests carried over from the old transcript
table (a remaining-only balance, unit labels, a fraction-only window) are the
regressions that surface still has to pass: the popup replaced the renderer, so
it inherits the renderer's contract.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container

from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.tui.widgets.usage_panel import (
    BAR_UNKNOWN,
    UsagePanel,
    binding_limit,
    build_usage_body,
    collect_stats,
    format_amount,
    format_countdown,
    usage_bar,
)

#: A wide-enough panel that nothing truncates, so a failure is about content.
WIDTH = 76


def _lines(reports, width: int = WIDTH, now: float = 0.0) -> list[str]:
    return [line.plain for line in build_usage_body(reports, width, now)]


def _report(*limits, provider: str = "anthropic", notes: str | None = None, identity=None):
    return UsageReport(provider=provider, limits=list(limits), notes=notes, identity=identity)


def _percent(limit_id: str, label: str, percent: float, **kwargs) -> UsageLimit:
    return UsageLimit(
        id=limit_id,
        label=label,
        amount=UsageAmount(
            used=percent,
            limit=100.0,
            remaining=100.0 - percent,
            used_fraction=percent / 100.0,
            unit="percent",
        ),
        **kwargs,
    )


# -- inherited renderer contract -------------------------------------------
def test_a_remaining_only_balance_renders_its_number() -> None:
    """Both account-balance fetchers report `remaining` with no `used` — neither
    vendor gives a limit to derive spend from. A renderer that printed a value
    only when `used` was set left a row labelled "Balance" that never said how
    much, and for DeepSeek no digit anywhere on screen."""
    lines = _lines(
        [
            _report(
                UsageLimit(
                    id="kimi:balance",
                    label="Balance (USD)",
                    amount=UsageAmount(remaining=12.5, unit="usd"),
                    window="lifetime",
                ),
                provider="kimi",
                notes="voucher $2.50 + cash $10.00",
            ),
            _report(
                UsageLimit(
                    id="deepseek:balance:cny",
                    label="Balance (CNY)",
                    # No UNIT_LABELS entry for CNY: the number must still print,
                    # just without a currency it did not earn.
                    amount=UsageAmount(remaining=70.0, unit="unknown"),
                    window="lifetime",
                ),
                provider="deepseek",
            ),
        ]
    )
    joined = "\n".join(lines)
    assert "12.50 USD left" in joined, lines
    assert "70 left" in joined, lines
    assert "voucher $2.50 + cash $10.00" in joined, lines


def test_amounts_print_the_unit_label_not_the_raw_key() -> None:
    """`UNIT_LABELS` exists so a row reads `519.86 USD` / `30%` rather than
    `519.86 usd` / `30 percent`."""
    lines = _lines(
        [
            _report(
                UsageLimit(
                    id="openrouter:spend",
                    label="Spend (no limit set)",
                    amount=UsageAmount(used=519.855, unit="usd"),
                    window="lifetime",
                ),
                _percent("x:pct", "Session", 30.0, window="5 hour"),
                provider="openrouter",
            )
        ]
    )
    joined = "\n".join(lines)
    assert "519.86 USD" in joined, lines
    assert "30%" in joined, lines
    assert "usd" not in joined and "percent" not in joined, lines


def test_a_fraction_only_window_still_states_its_percentage() -> None:
    """A bar shows the proportion and cannot be read off precisely, so the
    number is stated too."""
    lines = _lines(
        [
            _report(
                UsageLimit(
                    id="openai:primary",
                    label="Primary",
                    amount=UsageAmount(used_fraction=0.4, unit="percent"),
                    window="5 hour",
                ),
                provider="openai",
            )
        ]
    )
    assert any("40% used" in line for line in lines), lines


def test_a_money_row_states_the_cap_it_is_drawn_against() -> None:
    """`$12.00` is meaningless without the limit; a percentage already carries
    its own denominator, so only real units get the pair."""
    assert (
        format_amount(UsageAmount(used=12.0, limit=200.0, unit="usd")) == "12.00 USD / 200.00 USD"
    )
    assert format_amount(UsageAmount(used=100.0, limit=100.0, unit="percent")) == "100%"


# -- the popup's own behaviour ---------------------------------------------
def test_the_provider_header_names_the_window_that_binds() -> None:
    """The question a usage view is opened to answer is "can I keep working".
    `omp` lists every window at equal weight and leaves it to be found."""
    report = _report(
        _percent("anthropic:5h", "5 hour", 2.0, window="5 hour", shared=True),
        _percent("anthropic:7d", "7 day", 100.0, window="7 day", shared=True),
    )
    header = _lines([report])[0]
    assert "anthropic" in header
    assert "7 day 100%" in header, header


def test_an_exhausted_tier_row_never_outranks_a_shared_window() -> None:
    """A per-model cap at 100% stops one model family; a shared window at 80% is
    what throttles every request. Calling the tier row binding would tell a user
    to stop working when they only needed to switch model."""
    report = _report(
        _percent("anthropic:7d", "7 day", 80.0, shared=True),
        _percent("anthropic:7d:fable", "7 day (Fable)", 100.0, tier="fable"),
    )
    binding = binding_limit(report)
    assert binding is not None
    assert binding.id == "anthropic:7d"


def test_a_tier_row_is_indented_under_the_windows_it_is_subordinate_to() -> None:
    """Account-wide and per-model caps rendered identically is how a 100% Fable
    cap reads as a dead account."""
    lines = _lines(
        [
            _report(
                _percent("anthropic:7d", "7 day", 50.0, shared=True),
                _percent("anthropic:7d:fable", "7 day (Fable)", 0.0, tier="fable"),
            )
        ]
    )
    # The limit ROWS, not the provider header (which also names the binding
    # window): every row starts with the status mark.
    rows = [line for line in lines if line.startswith(("●", "○"))]
    shared_row = next(line for line in rows if "Fable" not in line)
    tier_row = next(line for line in rows if "Fable" in line)
    assert tier_row.index("7 day") > shared_row.index("7 day"), (shared_row, tier_row)


def test_a_window_that_reported_nothing_is_dots_not_an_empty_bar() -> None:
    """An empty bar says "you have used none of it", which is a claim. None was
    made."""
    assert usage_bar(None, 8) == BAR_UNKNOWN * 8
    assert set(usage_bar(0.0, 8)) == {"░"}


def test_a_barely_used_window_still_fills_a_cell() -> None:
    """Rounding 1% of a 24-cell bar to zero draws an empty bar for an account
    that HAS started spending."""
    assert usage_bar(0.01, 24).startswith("█")
    # And the converse: only a genuinely full window fills the last cell, so
    # 99.6% cannot be misread as exhausted.
    assert usage_bar(0.996, 24).endswith("░")
    assert set(usage_bar(1.0, 24)) == {"█"}


def test_the_countdown_says_two_units_at_most() -> None:
    """`3d11h47m` answers "wait or switch model" no better than `3d11h`, and
    costs the width the reset column has to fit in."""
    assert format_countdown(3 * 86_400_000 + 11 * 3_600_000 + 47 * 60_000) == "3d11h"
    assert format_countdown(3 * 3_600_000 + 24 * 60_000) == "3h24m"
    assert format_countdown(45 * 60_000) == "45m"
    assert format_countdown(0) == ""
    assert format_countdown(None) == ""


def test_a_row_states_how_long_until_its_window_rolls_over() -> None:
    limit = _percent("anthropic:5h", "5 hour", 2.0, window="5 hour", shared=True)
    limit.resets_at_ms = 3 * 3_600_000
    lines = _lines([_report(limit)], now=0.0)
    assert any("resets in 3h" in line for line in lines), lines


def test_a_window_that_already_reset_shows_no_countdown() -> None:
    """A negative countdown is worse than none: it reports a deadline that has
    already passed as though it were still ahead."""
    limit = _percent("anthropic:5h", "5 hour", 2.0, shared=True)
    limit.resets_at_ms = 1_000
    lines = _lines([_report(limit)], now=2_000.0)
    assert not any("resets in" in line for line in lines), lines


def test_the_stats_line_counts_windows_rather_than_providers() -> None:
    """One provider with an exhausted weekly cap and three healthy ones is a
    different situation from four providers each at 90%, and a provider count
    cannot tell them apart."""
    stats = collect_stats(
        [
            _report(
                _percent("a:5h", "5 hour", 2.0, shared=True),
                _percent("a:7d", "7 day", 100.0, shared=True),
                _percent("a:7d:x", "7 day (X)", 92.0, tier="x"),
                UsageLimit(id="a:?", label="Unknown", amount=UsageAmount(unit="unknown")),
            )
        ]
    )
    assert stats.windows == 4
    assert stats.exhausted == 1
    assert stats.warning == 1
    assert stats.unknown == 1
    assert stats.describe() == "4 windows · 1 exhausted · 1 near limit · 1 not reported"


def test_columns_line_up_across_providers() -> None:
    """Measured over every row rather than per provider, so the whole panel
    reads as one table instead of two that happen to be stacked."""
    lines = _lines(
        [
            _report(_percent("a:5h", "5 hour", 10.0, shared=True), provider="anthropic"),
            _report(
                _percent("o:primary", "A much longer window name", 20.0, shared=True),
                provider="openai",
            ),
        ]
    )
    bars = [line.index("█") for line in lines if "█" in line]
    assert len(bars) == 2 and bars[0] == bars[1], lines


def test_a_provider_with_no_windows_says_so_rather_than_rendering_blank() -> None:
    lines = _lines([_report(provider="kimi")])
    assert any("no windows reported" in line for line in lines), lines


def test_a_narrow_panel_drops_the_bar_before_the_numbers() -> None:
    """The bar is the only element that can shrink without losing information —
    a percentage is exact and a countdown is words."""
    lines = _lines([_report(_percent("a:5h", "5 hour", 42.0, shared=True))], width=34)
    row = next(line for line in lines if "5 hour" in line and line.startswith("●"))
    assert "42%" in row, row
    assert len(row) <= 34, row


# -- scrolling and chrome (needs a mounted panel for the screen geometry) ----
class _PanelHost(App[None]):
    """The panel alone, in a host that mirrors the app's (a hugging container).

    Not the full ``OperatorApp``: the scroll budget is a function of the screen
    size and nothing else here, and booting a session to test paging would make
    these tests slow for no extra coverage.
    """

    def compose(self) -> ComposeResult:
        with Container(id="usage-host"):
            yield UsagePanel()


@asynccontextmanager
async def _panel_app(size: tuple[int, int] = (80, 24)):
    app = _PanelHost()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        yield app.query_one(UsagePanel)


def _many_reports(count: int = 12):
    """More rows than any terminal in the tests can show at once."""
    return [
        _report(
            _percent(f"p{index}:5h", "5 hour", 10.0, shared=True),
            _percent(f"p{index}:7d", "7 day", 20.0, shared=True),
            provider=f"provider-{index}",
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_a_long_report_scrolls_instead_of_being_truncated() -> None:
    """A quota view that hid the provider you were looking for would send the
    user back to the CLI, which is the whole reason the popup exists."""
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        first = panel.render_lines_for_test()
        panel.action_scroll_page(1)
        second = panel.render_lines_for_test()
    assert panel.view_offset > 0
    assert first != second


@pytest.mark.asyncio
async def test_scrolling_clamps_at_both_ends_rather_than_wrapping() -> None:
    """A Down that silently returns to the top of a long report looks like the
    panel reset itself — the same rule the model picker's paging follows."""
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        panel.action_scroll_rows(-5)
        assert panel.view_offset == 0
        panel.action_scroll_end()
        bottom = panel.view_offset
        assert bottom > 0
        panel.action_scroll_rows(5)
        assert panel.view_offset == bottom
        panel.action_scroll_home()
        assert panel.view_offset == 0


@pytest.mark.asyncio
async def test_the_title_and_hints_stay_pinned_while_the_body_scrolls() -> None:
    """A user scrolling a long report must never lose the way out."""
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        top = panel.render_lines_for_test()
        panel.action_scroll_page(1)
        bottom = panel.render_lines_for_test()
    assert top[0].startswith("Usage")
    assert bottom[0].startswith("Usage")
    assert "esc close" in top[-1] and "esc close" in bottom[-1]


@pytest.mark.asyncio
async def test_the_scroll_position_is_reported_only_when_there_is_overflow() -> None:
    """A "1 of 1" on a report that fits is noise; its absence is the signal that
    nothing is hidden."""
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        scrolled = "\n".join(panel.render_lines_for_test())
        panel.show_reports([_report(_percent("a:5h", "5 hour", 5.0, shared=True))])
        short = "\n".join(panel.render_lines_for_test())
    assert " of " in scrolled
    assert " of " not in short
    # The scroll hint follows the same rule: a key that does nothing teaches the
    # user to distrust the others.
    assert "↑↓ scroll" in scrolled
    assert "↑↓ scroll" not in short


@pytest.mark.asyncio
async def test_an_empty_result_names_both_reasons_it_could_be_empty() -> None:
    """ "No endpoint" and "an endpoint you cannot reach" look identical in an
    empty panel, and only the second is something the user can act on."""
    async with _panel_app() as panel:
        panel.show_reports([])
        text = "\n".join(panel.render_lines_for_test())
    assert "no quota endpoint" in text
    assert "no credential" in text


@pytest.mark.asyncio
async def test_the_panel_says_it_is_fetching_before_the_first_result() -> None:
    """The fetch crosses the network once per logged-in provider; a command with
    no visible effect for two seconds reads as one that did not run."""
    async with _panel_app() as panel:
        panel.start_fetch("")
        assert "fetching…" in "\n".join(panel.render_lines_for_test())
