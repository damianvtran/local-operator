"""The `/usage` popup — rendering, scrolling, and the numbers it must state.

The rows are a pure function of ``(reports, width, now)``, so almost everything
here runs without an app. The three tests carried over from the old transcript
table (a remaining-only balance, unit labels, a fraction-only window) are the
regressions that surface still has to pass: the popup replaced the renderer, so
it inherits the renderer's contract.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager

import pytest
from rich.cells import cell_len
from rich.style import Style
from textual.app import App, ComposeResult
from textual.containers import Container

from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.usage_panel import (
    BAR_UNKNOWN,
    PANEL_MAX_WIDTH,
    PANEL_PADDING_ROWS,
    PANEL_WIDTH_MARGIN,
    UsagePanel,
    binding_limit,
    build_usage_body,
    collect_stats,
    format_age,
    format_amount,
    format_countdown,
    usage_bar,
)

# The boot app's fake session, reused rather than re-declared: the frame tests
# below need the REAL app (the lightweight host declares no CSS_PATH, so the
# card's padding and fill — half of what "centred" means — are not in its
# frames), and what they need from a session is only that one starts.
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: A wide-enough panel that nothing truncates, so a failure is about content.
WIDTH = 76


def _lines(
    reports, width: int = WIDTH, now: float = 0.0, header_ms: float | None = None
) -> list[str]:
    return [line.plain for line in build_usage_body(reports, width, now, header_ms).lines]


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


def _wide_reports(count: int = 12):
    """Reports whose composed rows FILL the rightmost columns — long labels,
    reset countdowns, and long identities, unlike ``_many_reports``' short
    blank-tailed rows.

    This is the fixture M1's regression turns on: the content-aware pad can only
    be exercised by rows that actually put a character in cols ``w-1``/``w-2``,
    which the short-label fixture never does. Anthropic/OpenAI reports carry both
    a countdown and an identity, so this is the realistic shape, not a corner.
    """
    return [
        _report(
            _percent(
                f"p{index}:5h",
                "5-hour session window",
                95.0,
                resets_at_ms=5 * 3600 * 1000,
            ),
            _percent(
                f"p{index}:7d",
                "7-day rolling window",
                60.0,
                resets_at_ms=48 * 3600 * 1000,
            ),
            provider=f"provider-with-a-long-name-{index}",
            identity=f"someone-with-a-long-email-{index}@example.com",
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
async def test_show_reports_keeps_the_reader_where_they_were() -> None:
    """A refresh replaces numbers the user is already reading; snapping their
    place is jostling, not a new open. `_repaint` still clamps if the new body
    is shorter than the offset they held."""
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        panel.action_scroll_page(1)
        offset = panel.view_offset
        assert offset > 0
        panel.show_reports(_many_reports())
        assert panel.view_offset == offset
        panel.show_reports([_report(_percent("a:5h", "5 hour", 5.0, shared=True))])
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
async def test_a_skipped_hint_never_leaves_a_leading_separator() -> None:
    """The hint row's separators ride the PREVIOUS segment, so when the
    scroll hint is skipped (nothing to scroll) on a row with no stat prefix,
    the first visible hint does not start with a dangling ``·``."""
    async with _panel_app() as panel:
        panel.show_reports([])  # empty: no stat prefix, so row.plain is empty
        hint_line = panel.render_lines_for_test()[-1]
    assert not hint_line.lstrip().startswith("·"), hint_line
    assert hint_line == "r refresh · esc close"
    # And the separator still joins hints when scroll IS offered. The stats
    # tally joins the first hint with the SAME " · " glyph as the hints join
    # each other (a consistent join, never a leading one) — pinned here so a
    # regression that reintroduces a leading "·" or drops the join is caught.
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())  # scrollable → ↑↓ offered
        scrolled = panel.render_lines_for_test()[-1]
    assert " ↑↓ scroll · r refresh · esc close" in scrolled
    assert not scrolled.lstrip().startswith("·"), scrolled
    # Populated-but-fits (no scroll) also joins stats to the first hint.
    async with _panel_app() as panel:
        panel.show_reports([_report(_percent("a:5h", "5 hour", 5.0, shared=True))])
        fits = panel.render_lines_for_test()[-1]
    assert " · r refresh · esc close" in fits
    assert not fits.lstrip().startswith("·"), fits


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


@pytest.mark.asyncio
async def test_the_card_reserves_its_padding_rows_so_nothing_is_clipped() -> None:
    """The widget pins its own height, and Textual sizes border-box: pinning
    the ROW COUNT alone would hand the gutter the last two rows and cut the
    hint row off the bottom of the card. The pinned height is therefore the
    rows PLUS the padding.

    The resulting CONTENT box is asserted against the real stylesheet in
    ``test_app_pilot`` — this host deliberately loads no CSS, so the padding
    it must survive only exists there.
    """
    async with _panel_app() as panel:
        panel.show_reports([_report(_percent("a:5h", "5 hour", 5.0, shared=True))])
        rows = panel.render_lines_for_test()
        assert panel.styles.height is not None
        assert panel.styles.height.value == len(rows) + PANEL_PADDING_ROWS


@pytest.mark.asyncio
async def test_a_report_block_separates_its_heading_from_its_meters() -> None:
    """Heading → (note) → blank → meters. Without the blank, the identity
    line, the account note and the first meter render as three equal rows and
    the block reads as an undifferentiated wall."""
    lines = _lines(
        [
            _report(
                _percent("a:5h", "5 hour", 5.0, shared=True),
                notes="extra usage disabled",
            )
        ]
    )
    assert lines[0].startswith("anthropic")
    assert lines[1].strip() == "extra usage disabled"
    assert lines[2] == ""  # the breathing row
    assert "5 hour" in lines[3]


def test_an_unavailable_account_keeps_its_identity_and_last_known_numbers() -> None:
    """A maxed-out probe must not drop the block — the login is still real."""
    report = _report(
        _percent("a:5h", "5 hour", 72.0, shared=True),
        identity="damian@gominerva.com",
    )
    report.usage_unavailable = True
    report.consecutive_failures = 5
    report.fetched_at = 0
    lines = _lines([report], now=60_000)
    assert any("damian@gominerva.com" in line for line in lines)
    assert any("usage unavailable" in line for line in lines)
    assert any("72%" in line for line in lines)


def test_unavailable_with_last_known_meters_keeps_binding_on_the_heading() -> None:
    """Heading stays identity + binding; the new fact lives only on the note.

    Putting ``usage unavailable`` on both rows duplicates the status and
    clips the binding window on the designed 72-col card. Gominerva's
    last-known hierarchy is the model: heading is the window, note is
    the honesty. With no meters the heading may still say unavailable.
    """
    now = 40 * 60_000
    with_meters = _report(
        _percent("c:7d", "7 day", 33.0, shared=True, resets_at_ms=now + 60 * 3600 * 1000),
        identity="damian@pergamonhq.com",
    )
    with_meters.usage_unavailable = True
    with_meters.consecutive_failures = 5
    with_meters.fetched_at = 1  # any past stamp; age is now - fetched_at
    heading, note, *_rest = _lines([with_meters], width=72, now=now)
    assert "damian@pergamonhq.com" in heading
    assert "7 day 33%" in heading
    assert "usage unavailable" not in heading
    assert "…" not in heading
    assert "usage unavailable" in note
    assert "last known" in note

    no_meters = _report(identity="new@example.com")
    no_meters.usage_unavailable = True
    no_meters.consecutive_failures = 5
    no_heading = _lines([no_meters])[0]
    assert "usage unavailable" in no_heading


def test_a_stale_account_keeps_its_numbers_and_says_last_known() -> None:
    report = _report(
        _percent("a:7d", "7 day", 40.0, shared=True),
        identity="damian@radienthq.com",
    )
    report.consecutive_failures = 2
    report.fetched_at = 0
    lines = _lines([report], now=5 * 60_000)
    assert any("damian@radienthq.com" in line for line in lines)
    assert any("last known" in line for line in lines)
    assert any("40%" in line for line in lines)


def test_an_exhausted_200_is_not_labelled_unavailable() -> None:
    """100% weekly from a live 200 is quota, not a fetch failure."""
    report = _report(
        _percent("a:7d", "7 day", 100.0, shared=True),
        identity="damianvtran@gmail.com",
    )
    lines = _lines([report])
    joined = "\n".join(lines)
    assert "damianvtran@gmail.com" in joined
    assert "100%" in joined
    assert "usage unavailable" not in joined
    assert "last known" not in joined


@pytest.mark.asyncio
async def test_the_footer_is_separated_from_the_last_meter() -> None:
    """The tally/keys footer is chrome. Flush against the last meter it reads
    as one more row of the report."""
    async with _panel_app() as panel:
        panel.show_reports([_report(_percent("a:5h", "5 hour", 5.0, shared=True))])
        rows = panel.render_lines_for_test()
    assert rows[-1].startswith("1 window")  # the footer
    assert rows[-2] == ""  # the quiet row above it


#: The position marker, matched as a whole row. It is the first row of the
#: card's bottom meta, so the tests below find the end of the report by finding
#: it (and fall back to the key hints when there is nothing to scroll).
_MARKER = re.compile(r"^showing \d+ of \d+$")


def _report_rows(rows: list[str]) -> list[str]:
    """The report's own rows: the rule to the last row that says something.

    The quiet ground above the meta is not part of the report — it is the
    card's bottom margin, and how much of it there is depends on where the
    block boundary fell.
    """
    meta = next((index for index, row in enumerate(rows) if _MARKER.match(row)), len(rows) - 1)
    report = rows[2:meta]
    while report and not report[-1]:
        report.pop()
    return report


@pytest.mark.asyncio
async def test_the_card_ends_on_quiet_ground_then_its_meta_in_both_states() -> None:
    """One grammar for the bottom of the card, scrolled or not.

    The position and the key hints are the same KIND of row — statements about
    the list rather than entries in it — so they travel together at the bottom
    with the quiet ground above the pair, and the scrolled card differs from the
    one that fits by exactly one meta row. Pinned because the two other
    arrangements are both reachable by one plausible edit and both read wrong:
    the counter under the spacer glues it to the keys with the gap in the middle
    of the chrome, and a card whose slack lands between the counter and the keys
    punches a hole through the meta that reads as a failure to render.
    """
    async with _panel_app() as panel:
        panel.show_reports([_report(_percent("a:5h", "5 hour", 5.0, shared=True))])
        fits = panel.render_lines_for_test()
        panel.show_reports(_many_reports())
        scrolled = panel.render_lines_for_test()

    assert fits[-2] == "", fits
    assert _MARKER.match(scrolled[-2]), scrolled
    assert scrolled[-3] == "", scrolled
    assert _report_rows(scrolled)[-1].strip(), scrolled


@pytest.mark.asyncio
async def test_no_window_ends_on_a_provider_heading_with_no_meters_under_it() -> None:
    """A block boundary is where the window stops, at every scroll position.

    A budget that cut wherever it ran out left ``provider-2  ·  7 day 20%`` as the
    last row of the report with nothing under it — a provider announced and then
    no numbers, which is the one thing the card exists to state. Verified at 80x24
    because that is the size where the budget (12 rows) lands mid-block.

    The card's HEIGHT is pinned across the sweep in the same pass: the rows the
    boundary gives back become quiet ground above the meta rather than a shorter
    card, because a card that changed height per keystroke would walk up and down
    the screen while being read (it is centred on its own height).
    """
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        panel.action_scroll_home()
        heights: set[int] = set()
        offsets = 0
        while True:
            rows = panel.render_lines_for_test()
            report = _report_rows(rows)
            heights.add(len(rows))
            offsets += 1
            assert report, (panel.view_offset, rows)
            assert not report[-1].startswith("provider-"), (panel.view_offset, rows)
            before = panel.view_offset
            panel.action_scroll_rows(1)
            if panel.view_offset == before:
                break

    assert offsets > 10, "premise: this size scrolls far enough to cut mid-block"
    assert heights == {max(heights)}, f"the card changed height while scrolling: {heights}"


class _Wheel:
    """The only thing the scroll handlers use from a Textual event."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_the_mouse_wheel_scrolls_the_report_and_clamps() -> None:
    """The card is read, not picked from, so the wheel moves the WINDOW. It
    clamps at both ends for the same reason the keys do: a quota list has a
    top and a bottom that mean something."""
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        assert panel.view_offset == 0
        panel.on_mouse_scroll_down(_Wheel())
        assert panel.view_offset == 1
        for _ in range(500):
            panel.on_mouse_scroll_down(_Wheel())
        bottom = panel.view_offset
        assert bottom > 1
        panel.on_mouse_scroll_down(_Wheel())
        assert panel.view_offset == bottom  # clamped, never wrapped
        for _ in range(500):
            panel.on_mouse_scroll_up(_Wheel())
        assert panel.view_offset == 0


@pytest.mark.asyncio
async def test_the_wheel_is_stopped_so_the_transcript_behind_stays_put() -> None:
    """The card floats over the conversation; an un-stopped wheel would move
    both surfaces for one gesture."""
    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        down, up = _Wheel(), _Wheel()
        panel.on_mouse_scroll_down(down)
        panel.on_mouse_scroll_up(up)
    assert down.stopped and up.stopped


# -- the draggable scrollbar -------------------------------------------------
# The test host loads no CSS, so the panel's gutter is zero (padding lives only
# in the real stylesheet) and content coordinates equal widget coordinates —
# `get_content_offset_capture` subtracts a zero gutter. The bar therefore lands
# at content column ``_body_content_width()`` and content rows counted from the
# title at row 0, which is exactly what the widget's own hit-test derives.
def _mouse(cls, x: int, y: int):
    """A synthesized Textual mouse event at widget coordinates ``(x, y)``."""
    return cls(
        widget=None,
        x=x,
        y=y,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
    )


def _bar_column(panel: UsagePanel) -> int:
    """The reserved gutter column the scrollbar paints into."""
    return panel._body_content_width()


def _bar_glyphs(panel: UsagePanel) -> str:
    """The rightmost cell of every composed row — the scrollbar column's glyphs.

    Read off the composed strings rather than the paint so the assertion is
    about what a reader sees in that column, track and thumb together.
    """
    from local_operator.tui.widgets.usage_panel import SCROLLBAR_THUMB, SCROLLBAR_TRACK

    column = _bar_column(panel)
    glyphs = ""
    for row in panel.render_lines_for_test():
        if len(row) > column and row[column] in (SCROLLBAR_TRACK, SCROLLBAR_THUMB):
            glyphs += row[column]
    return glyphs


@pytest.mark.asyncio
async def test_the_scrollbar_is_absent_until_the_body_actually_scrolls() -> None:
    """A bar on a report that fits is the same lie a "1 of 1" position is: it
    offers a drag that does nothing. The gutter is still reserved (numbers must
    not reflow — see the reflow test), but no track or thumb is drawn."""
    from local_operator.tui.widgets.usage_panel import SCROLLBAR_THUMB, SCROLLBAR_TRACK

    async with _panel_app() as panel:
        panel.show_reports([_report(_percent("a:5h", "5 hour", 5.0, shared=True))])
        assert _bar_glyphs(panel) == ""  # not scrollable → no bar
        panel.show_reports(_many_reports())
        glyphs = _bar_glyphs(panel)
    assert SCROLLBAR_THUMB in glyphs  # a proportional thumb
    assert SCROLLBAR_TRACK in glyphs  # over a longer track


@pytest.mark.asyncio
async def test_dragging_the_thumb_moves_the_offset_with_the_pointer() -> None:
    """A grab on the thumb then a drag down the track increases the offset
    monotonically and reaches the very bottom at the track's foot — the whole
    point of a draggable bar. Release ends the drag without moving the offset."""
    from textual import events

    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        budget = panel._body_budget()
        total = len(panel._body().lines)
        first_row, count = panel._body_region(budget)
        thumb_top, _ = panel._scrollbar_thumb(total, budget)
        column = _bar_column(panel)

        panel.on_mouse_down(_mouse(events.MouseDown, column, first_row + thumb_top))
        assert panel._dragging

        offsets = []
        for y in range(first_row, first_row + count):
            panel.on_mouse_move(_mouse(events.MouseMove, column, y))
            offsets.append(panel.view_offset)

        panel.on_mouse_up(_mouse(events.MouseUp, column, first_row + count - 1))

    assert offsets == sorted(offsets)  # monotonic non-decreasing with pointer y
    assert offsets[0] == 0 and offsets[-1] == panel._max_offset()  # top → bottom
    assert not panel._dragging  # released


@pytest.mark.asyncio
async def test_clicking_the_track_below_the_thumb_scrolls_down() -> None:
    """A click on the bare track past the thumb jumps toward that position, the
    page-style affordance every scrollbar offers alongside the drag."""
    from textual import events

    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        assert panel.view_offset == 0
        budget = panel._body_budget()
        total = len(panel._body().lines)
        first_row, count = panel._body_region(budget)
        thumb_top, thumb_len = panel._scrollbar_thumb(total, budget)
        column = _bar_column(panel)
        # A row on the track below the thumb, still inside the viewport.
        target = min(count - 1, thumb_top + thumb_len + 2)
        panel.on_mouse_down(_mouse(events.MouseDown, column, first_row + target))
        moved = panel.view_offset
        panel.on_mouse_up(_mouse(events.MouseUp, column, first_row + target))
    assert moved > 0


@pytest.mark.asyncio
async def test_the_scrollbar_gutter_never_reflows_the_number_columns() -> None:
    """Requirement 1: the report's right-aligned numbers land at the same
    column whether the bar is present (scrolled) or absent (fits), because the
    gutter is reserved in both states rather than stolen from the numbers when
    the bar appears."""

    def meter(rows: list[str]) -> str:
        # The meter row (leading ●), not the header, whose binding summary also
        # prints a percent — the meter is the row with the right-aligned column.
        return next(row for row in rows if row.startswith("●") and "10%" in row)

    async with _panel_app() as panel:
        panel.show_reports(_many_reports())
        scrolled = meter(panel.render_lines_for_test())
        panel.show_reports([_report(_percent("a:5h", "5 hour", 10.0, shared=True))])
        fits = meter(panel.render_lines_for_test())
    assert scrolled.index("10%") == fits.index("10%"), (scrolled, fits)


# -- the grab must not leave a selection armed (needs the REAL app) ----------
# The lightweight ``_PanelHost`` above is deliberately NOT used here: the bug is
# in the interaction between the panel's grab and ``Screen._forward_event``,
# which arms a text selection on the MouseDown before the panel's handler runs
# because the panel is a selectable ``Static``. That machinery only exists on a
# real ``Screen``, so these tests drive the full ``OperatorApp`` (as
# ``test_app_pilot`` does) and push events through ``screen._forward_event`` —
# the same path a real mouse press takes — rather than calling the panel's
# handler directly, which would skip the selection-arming under test.
def _bar_screen_x(panel: UsagePanel) -> int:
    """The absolute screen column the scrollbar paints into.

    Composed from the panel's region plus its live gutter plus the body content
    width, never hardcoded: the card is positioned by an offset inside a hugging
    host inside the screen inset, and its padding differs between the real
    stylesheet and the test host.
    """
    return panel.region.x + panel.styles.gutter.left + panel._body_content_width()


def _body_screen_y0(panel: UsagePanel) -> int:
    """Absolute screen row of composed body row 0 (the title cell)."""
    return panel.region.y + panel.styles.gutter.top


def _screen_mouse(cls, x: int, y: int):
    """A Textual mouse event carrying SCREEN coordinates, for ``_forward_event``.

    ``x``/``y`` and ``screen_x``/``screen_y`` are equal here because the event is
    fed to the screen, which translates to widget-relative coordinates itself.
    """
    return cls(
        widget=None,
        x=x,
        y=y,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=x,
        screen_y=y,
    )


async def _grab_via_screen(panel: UsagePanel, app: OperatorApp, pilot, dx: int):
    """Press ``dx`` cells from the bar's screen x (on the thumb row) via the
    screen, drag to the track's foot, and return the observed grab/selection
    state plus whether the offset advanced.

    Each event is followed by ``pilot.pause()`` because ``_forward_event`` POSTS
    the mouse event to the panel's message pump — ``on_mouse_down`` runs on the
    next pump cycle, not synchronously — so a check before the pause would read
    the pre-handler state and spuriously fail.
    """
    from textual import events

    panel.set_view_offset(0)
    app.screen.clear_selection()
    panel._dragging = False

    budget = panel._body_budget()
    total = len(panel._body().lines)
    first_row, count = panel._body_region(budget)
    thumb_top, _ = panel._scrollbar_thumb(total, budget)
    sx = _bar_screen_x(panel) + dx
    sy = _body_screen_y0(panel) + first_row + thumb_top

    app.screen._forward_event(_screen_mouse(events.MouseDown, sx, sy))
    await pilot.pause()
    dragging = panel._dragging
    captured = app.mouse_captured
    off0 = panel.view_offset

    # Drag to the track's foot; the base move handler would extend a leaked
    # selection here, which is exactly what must NOT happen.
    app.screen._forward_event(
        _screen_mouse(events.MouseMove, sx, _body_screen_y0(panel) + first_row + count - 1)
    )
    await pilot.pause()
    off1 = panel.view_offset
    selecting = app.screen._selecting
    selections = dict(app.screen.selections)

    app.screen._forward_event(_screen_mouse(events.MouseUp, sx, sy))
    await pilot.pause()
    return dragging, captured, off0, off1, selecting, selections


@pytest.mark.asyncio
@pytest.mark.parametrize("dx", [0, -1, -2])
async def test_a_scrollbar_grab_never_leaks_a_selection(dx: int) -> None:
    """The reported bug: pressing on (dx=0) or just left of (dx=-1,-2) the bar
    must grab the thumb and scroll, WITHOUT the base screen arming a text
    selection that its move handler then extends into the messy highlight.

    Driven through the real screen so the selection-arming that caused the bug
    actually runs; the near-miss cases (dx<0) also exercise the forgiving pad.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.show_reports(_many_reports())
        await pilot.pause()

        dragging, captured, off0, off1, selecting, selections = await _grab_via_screen(
            panel, app, pilot, dx
        )

    assert dragging, dx  # the panel took the grab
    assert captured is panel, (dx, captured)  # and captured the mouse
    assert not selecting, (dx, selecting)  # no selection in flight
    assert selections == {}, (dx, selections)  # nothing selected
    assert off1 > off0, (dx, off0, off1)  # the drag actually scrolled


def _pad_rows(panel: UsagePanel) -> tuple[int | None, int | None, str]:
    """``(a body row whose pad cells hold CONTENT, a row whose pad is BLANK, the
    content row's tail)`` for the current window, or ``None`` where absent.

    Reads the same composed window rows the painter overlays the bar onto, so a
    test can aim a press at a row that provably fills cols ``w-1``/``w-2`` rather
    than trusting a fixture to do so — the exact gap M1/m1 flag.
    """
    from local_operator.tui.widgets.usage_panel import SCROLLBAR_GRAB_PAD

    budget = panel._body_budget()
    window, _ = panel._window_rows(panel._body(), budget)
    bar = panel._body_content_width()
    content = blank = None
    tail = ""
    for row, line in enumerate(window):
        cells = line.plain[bar - SCROLLBAR_GRAB_PAD : bar]
        if cells.strip():
            if content is None:
                content, tail = row, cells
        elif blank is None:
            blank = row
    return content, blank, tail


@pytest.mark.asyncio
async def test_the_pad_is_content_aware_over_the_unsafe_geometry() -> None:
    """M1: this panel composes rows flush to the bar with NO empty gutter, so the
    pad columns can hold real data. A near-miss must grab only over a BLANK tail;
    over a row that fills those columns it must fall through and select, so the
    forgiveness never steals a visible character.

    Uses ``_wide_reports`` (long labels + reset countdowns + identities) so the
    unsafe geometry actually exists, and ASSERTS a row with content in the pad
    band is present — the fixture cannot silently regress to blank tails.
    """
    from textual import events

    from local_operator.tui.widgets.usage_panel import SCROLLBAR_GRAB_PAD

    # A narrow terminal packs the meters flush to the bar (M1 reproduced at 40-44
    # cols); the wide fixture keeps content there at any width.
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(44, 20)) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.show_reports(_wide_reports())
        await pilot.pause()

        content_row, blank_row, tail = _pad_rows(panel)
        # The fixture MUST exercise the unsafe geometry or the test is vacuous.
        assert content_row is not None, "fixture has no content in the pad band"
        assert tail.strip(), tail

        first_row, _ = panel._body_region(panel._body_budget())
        bar_x = _bar_screen_x(panel)
        y0 = _body_screen_y0(panel)

        async def press(sx: int, sy: int) -> tuple[bool, bool]:
            panel.set_view_offset(0)
            app.screen.clear_selection()
            panel._dragging = False
            app.screen._forward_event(_screen_mouse(events.MouseDown, sx, sy))
            await pilot.pause()
            grabbed = panel._dragging
            armed = app.screen._select_state is not None
            app.screen._forward_event(_screen_mouse(events.MouseUp, sx, sy))
            await pilot.pause()
            return grabbed, armed

        # Near-miss (both pad cells) over a CONTENT row: no grab, selection arms —
        # the visible character is NOT stolen.
        for dx in (-1, -SCROLLBAR_GRAB_PAD):
            grabbed, armed = await press(bar_x + dx, y0 + first_row + content_row)
            assert not grabbed, (dx, "grabbed over content")
            assert armed, (dx, "content press did not arm a selection")

        # The EXACT bar column always grabs, even on a content row.
        grabbed, armed = await press(bar_x, y0 + first_row + content_row)
        assert grabbed and not armed

        # Near-miss over a BLANK tail: forgiveness still works — grab, no select.
        if blank_row is not None:
            for dx in (-1, -SCROLLBAR_GRAB_PAD):
                grabbed, armed = await press(bar_x + dx, y0 + first_row + blank_row)
                assert grabbed, (dx, "blank-tail near-miss did not grab")
                assert not armed, (dx, "blank-tail grab armed a selection")


@pytest.mark.asyncio
async def test_a_content_click_left_of_the_pad_still_selects() -> None:
    """The pad is a narrow band: a press ``SCROLLBAR_GRAB_PAD + 1`` cells left of
    the bar is ordinary content and must still arm a selection, so widening the
    grab target cannot swallow a real text drag.

    Driven with ``_wide_reports`` so cols at and left of the boundary actually
    hold content (m1: the short-label fixture left them blank, making the
    boundary assertion vacuous). Asserts the pressed cell is non-blank first.
    """
    from textual import events

    from local_operator.tui.widgets.usage_panel import SCROLLBAR_GRAB_PAD

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(44, 20)) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.show_reports(_wide_reports())
        await pilot.pause()

        content_row, _, _ = _pad_rows(panel)
        assert content_row is not None, "fixture has no content in the pad band"

        panel.set_view_offset(0)
        app.screen.clear_selection()
        first_row, _ = panel._body_region(panel._body_budget())
        window, _ = panel._window_rows(panel._body(), panel._body_budget())
        bar = panel._body_content_width()
        col = bar - (SCROLLBAR_GRAB_PAD + 1)
        # The boundary is only meaningful if the pressed cell holds a character.
        assert window[content_row].plain[col : col + 1].strip(), "boundary cell is blank"
        sx = _bar_screen_x(panel) - (SCROLLBAR_GRAB_PAD + 1)
        sy = _body_screen_y0(panel) + first_row + content_row
        app.screen._forward_event(_screen_mouse(events.MouseDown, sx, sy))
        await pilot.pause()
        grabbed = panel._dragging
        armed = app.screen._select_state is not None
        app.screen._forward_event(_screen_mouse(events.MouseUp, sx, sy))
        await pilot.pause()

    assert not grabbed  # the panel did NOT treat content as a grab
    assert armed  # the base screen armed a selection, as for any content press


@pytest.mark.asyncio
async def test_a_press_on_a_non_scrollable_panel_is_not_a_grab() -> None:
    """A report that fits reserves the gutter but paints no bar; a press in the
    gutter column must fall through to the base handler (no grab, no capture),
    the same non-affordance the "bar absent until it scrolls" test asserts."""
    from textual import events

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.show_reports([_report(_percent("a:5h", "5 hour", 5.0, shared=True))])
        await pilot.pause()

        first_row, _ = panel._body_region(panel._body_budget())
        sx = _bar_screen_x(panel)
        sy = _body_screen_y0(panel) + first_row
        app.screen._forward_event(_screen_mouse(events.MouseDown, sx, sy))
        await pilot.pause()
        grabbed = panel._dragging
        captured = app.mouse_captured
        app.screen._forward_event(_screen_mouse(events.MouseUp, sx, sy))
        await pilot.pause()

    assert not grabbed
    assert captured is not panel


# -- the card ON the screen (the real app, so the fill is in the frame) -------
def _painted_span(app: OperatorApp, row: int, fill: str) -> tuple[int, int, int]:
    """``(cells left of the card, cells right of it, its painted width)``.

    Measured off the composed strip rather than off ``panel.region``: the card is
    positioned by an offset inside a hugging host inside the screen's own inset,
    and a region compared against the wrong one of those three boxes is how the
    card was reported as off-centre when the paint is symmetric. The fill is the
    card's own background, which is what a reader sees the edges of.
    """
    strip = app.screen._compositor.render_strips()[row]
    column = first = last = 0
    for segment in strip:
        colour = segment.style.bgcolor if segment.style else None
        if colour is not None and colour.triplet is not None and colour.triplet.hex == fill:
            first = first or column
            last = column + len(segment.text) - 1
        column += len(segment.text)
    return first, column - 1 - last, last - first + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (121, 40), (100, 30), (101, 30), (80, 24), (81, 24)])
async def test_the_card_is_centred_on_the_terminal_at_odd_and_even_widths(
    size: tuple[int, int],
) -> None:
    """Equal ground either side of the card, to the cell the parity allows.

    The centring is arithmetic in the widget (the host hugs the card, so
    ``align: center middle`` is not available), and arithmetic against the
    screen's CONTENT box while the card is painted inside the screen's inset —
    two boxes that differ by the inset on each side. They cancel only because the
    inset is symmetric, which is an assumption worth a frame test rather than a
    comment: this reads the painted edges, so it fails for a real off-centre card
    however the widget arrived at its offset.
    """
    width, _ = size
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.display = True
        panel.show_reports(_many_reports())
        for _ in range(4):
            await pilot.pause()
        fill = app.get_css_variables()["lo-overlay"]
        card = panel.region
        left, right, painted = _painted_span(app, card.y + 2, fill)

    assert painted == card.width, (painted, card)
    slack = width - painted
    assert (left, right) == (slack // 2, slack - slack // 2), (left, right, painted)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (100, 30), (120, 24), (80, 24), (80, 16), (80, 14)])
async def test_a_scrolled_card_never_covers_the_input_prompt(
    size: tuple[int, int],
) -> None:
    """D19 was visible only after the card had enough data to scroll.

    Drive the REAL app and compare the painted widgets' absolute regions. The
    old screen-centred placement overlapped the editor by two rows even though
    the body budget was correct; at 80x24 its footer rendered as
    ``❯  24 windows · ↑↓ scroll`` on the prompt's own row.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.display = True
        panel.show_reports(_many_reports())
        for _ in range(4):
            await pilot.pause()
        editor = app.query_one(Editor)

    assert not panel.region.overlaps(editor.region), (size, panel.region, editor.region)
    assert panel.region.bottom <= editor.region.y


@pytest.mark.asyncio
async def test_narrow_error_and_empty_states_keep_their_retry_and_close_receipts() -> None:
    """Provider prose is unbounded; at 50x20 it must lose its tail rather than
    wrap through or clip the footer actions that recover and close the panel."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(50, 20)) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.show_error("provider failed: " + "network timeout " * 20)
        for _ in range(3):
            await pilot.pause()
        error_rows = panel.render_lines_for_test()
        assert all(cell_len(row) <= panel._content_width() for row in error_rows)
        assert "r refresh" in error_rows[-1] and "esc close" in error_rows[-1]
        assert not panel.region.overlaps(app.query_one(Editor).region)

        panel.show_reports([])
        for _ in range(3):
            await pilot.pause()
        empty_rows = panel.render_lines_for_test()
        assert all(cell_len(row) <= panel._content_width() for row in empty_rows)
        assert "r refresh" in empty_rows[-1] and "esc close" in empty_rows[-1]
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())

    assert "r refresh" in painted and "esc close" in painted, painted


@pytest.mark.asyncio
async def test_compact_usage_keeps_provider_identity_and_operational_actions() -> None:
    """At End on 50x20, decorative air must yield before the provider heading
    for the final meters and the refresh/close actions."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(50, 20)) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.show_reports(_many_reports())
        panel.action_scroll_end()
        for _ in range(3):
            await pilot.pause()
        rows = panel.render_lines_for_test()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())

    assert any("provider-11" in row for row in rows), rows
    assert any("5 hour" in row for row in rows), rows
    assert any("7 day" in row for row in rows), rows
    assert all(cell_len(row) <= panel._content_width() for row in rows)
    assert "r refresh" in rows[-1] and "esc close" in rows[-1], rows[-1]
    assert "provider-11" in painted, painted


@pytest.mark.asyncio
async def test_open_usage_reflows_when_the_bottom_band_grows(monkeypatch) -> None:
    """A dock mutation does not resize an overlay, so the app must explicitly
    re-measure an already-open tall card after todos appear."""
    from local_operator.tui.widgets import todo_panel as todo_panel_module

    # ``todo_items`` now returns PHASES (``{"name", "items"}``), not a flat item
    # list — the panel was made phase-aware. This test only needs the band to
    # grow when todos appear, so the stub wraps its items in one implicit phase,
    # matching what a flat ``init`` produces (design §3.2). Returning bare item
    # dicts here would parse as zero-item phases and the band would never grow,
    # which is the shape mismatch that surfaced as ``assert 13 < 13``.
    todos: list[dict[str, str]] = []
    monkeypatch.setattr(
        todo_panel_module,
        "todo_items",
        lambda _session_id: (
            [{"name": todo_panel_module._IMPLICIT_PHASE, "items": list(todos)}] if todos else []
        ),
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.show_reports(_many_reports())
        for _ in range(4):
            await pilot.pause()
        editor = app.query_one(Editor)
        before = panel.region
        assert not before.overlaps(editor.region)

        todos.extend({"text": f"new live todo {index}", "status": "pending"} for index in range(5))
        app._refresh_band()
        for _ in range(6):
            await pilot.pause()
        after = panel.region
        editor_region = editor.region

    assert after.height < before.height, (before, after)
    assert not after.overlaps(editor_region), (after, editor_region)
    assert after.bottom <= editor_region.y


@pytest.mark.asyncio
async def test_a_scrolled_footer_keeps_the_affordance_over_the_tally() -> None:
    """When both cannot fit, the way OUT beats the summary of what is hidden.

    The tally describes what the report contains; ``↑↓ scroll`` is the only
    thing on screen saying the remainder is reachable at all. Dropping the
    affordance and keeping the tally leaves a footer that reports a total the
    reader has no stated way to get to.

    Multi-account reports are what made this bite rather than theorise: the
    body grew from 16 rows to 23, so the hidden remainder at a normal terminal
    height became whole providers instead of a short tail.
    """
    # 56 cells fits both; 52 is the first width at which one must go.
    async with _panel_app(size=(56, 20)) as panel:
        panel.show_reports(_many_reports())
        both = panel.render_lines_for_test()[-1]
    assert "windows" in both and "↑↓ scroll" in both, both

    async with _panel_app(size=(52, 20)) as panel:
        panel.show_reports(_many_reports())
        footer = panel.render_lines_for_test()[-1]

    assert "↑↓ scroll" in footer, footer
    assert "esc close" in footer, footer
    # The tally is what yields, not the way out.
    assert "windows" not in footer, footer


# -- cached-first open --------------------------------------------------------
@pytest.mark.asyncio
async def test_show_cached_paints_the_reports_with_their_age_and_a_refreshing_mark() -> None:
    """When the shared cache already holds a row, `/usage` must not say
    "fetching…" and hide an answer that is on hand — it paints the reports at
    once, states their age, and marks that a refresh is running behind them."""
    import time as _time

    async with _panel_app() as panel:
        panel.start_fetch("")
        fetched_ms = _time.time() * 1000 - 120_000  # 2 minutes old
        panel.show_cached(
            [_report(_percent("a:5h", "5 hour", 5.0, shared=True))], now_ms=fetched_ms
        )
        text = "\n".join(panel.render_lines_for_test())
        assert "fetching…" not in text
        assert "refreshing…" in text
        assert "2m ago" in text
        assert "anthropic" in text


@pytest.mark.asyncio
async def test_show_reports_clears_the_refreshing_mark() -> None:
    """The fetch's result replaces the cached view; the `refreshing…` mark goes."""
    import time as _time

    async with _panel_app() as panel:
        panel.start_fetch("")
        panel.show_cached(
            [_report(_percent("a:5h", "5 hour", 5.0, shared=True))],
            now_ms=_time.time() * 1000,
        )
        panel.show_reports([_report(_percent("a:5h", "5 hour", 50.0, shared=True))])
        text = "\n".join(panel.render_lines_for_test())
        assert "refreshing…" not in text
        assert "50%" in text


@pytest.mark.asyncio
async def test_settle_refresh_keeps_the_cached_numbers_when_the_fetch_fails() -> None:
    """A failed refresh must not blank the panel: the cached numbers stay, the
    `refreshing…` mark goes, and the age in the title already says how stale
    they are."""
    import time as _time

    async with _panel_app() as panel:
        panel.start_fetch("")
        panel.show_cached(
            [_report(_percent("a:5h", "5 hour", 5.0, shared=True))],
            now_ms=_time.time() * 1000,
        )
        panel.settle_refresh()
        text = "\n".join(panel.render_lines_for_test())
        assert "refreshing…" not in text
        assert "anthropic" in text
        assert "5%" in text


@pytest.mark.asyncio
async def test_a_cold_open_still_says_fetching() -> None:
    """With nothing cached the panel shows its loading state — the cached-first
    path must not accidentally render an empty report set."""
    async with _panel_app() as panel:
        panel.start_fetch("")
        assert "fetching…" in "\n".join(panel.render_lines_for_test())


@pytest.mark.asyncio
async def test_a_failed_refresh_says_so_instead_of_just_dropping_the_mark() -> None:
    """`r` came back empty-handed: the numbers stay, and a pinned note says the
    refresh failed — the mark silently vanishing is not an answer to a key the
    user just pressed."""
    import time as _time

    async with _panel_app() as panel:
        panel.start_fetch("")
        panel.show_cached(
            [_report(_percent("a:5h", "5 hour", 5.0, shared=True))],
            now_ms=_time.time() * 1000,
        )
        panel.settle_refresh(failed=True)
        text = "\n".join(panel.render_lines_for_test())
        assert "refresh failed — showing last known numbers" in text
        assert "refreshing…" not in text
        assert "5 hour" in text  # the numbers survived


@pytest.mark.asyncio
async def test_the_title_row_truncates_instead_of_wrapping() -> None:
    """The title gained suffixes (target, age, refreshing…) that can outrun a
    narrow card. An untruncated Text WRAPS — an extra visual row the pinned
    height never counted, which clipped the footer (and its `r refresh`
    receipt) off the card exactly in the stale/narrow state where the refresh
    key matters most."""
    import time as _time

    async with _panel_app(size=(36, 24)) as panel:
        panel.start_fetch("openrouter")
        panel.show_cached(
            [_report(_percent("a:5h", "5 hour", 5.0, shared=True), provider="openrouter")],
            now_ms=_time.time() * 1000 - 12 * 3600_000,  # "12h ago"
        )
        title = panel._compose_rows()[0]
        assert cell_len(title.plain) <= panel.panel_width(), title.plain


@pytest.mark.asyncio
async def test_panel_width_caps_on_a_laptop_and_holds_on_eighty_cols() -> None:
    """Extra terminal width used to be thrown away at 76, so a laptop /usage
    card sat as a skinny column with short bars. The cap is a measure (104),
    not a fraction: 80-col stays ``80 - margin``, a 140-col laptop grows to
    the cap rather than filling the sheet."""
    async with _panel_app(size=(80, 24)) as panel:
        assert panel.panel_width() == 80 - PANEL_WIDTH_MARGIN
        assert panel.panel_width() == 76
    async with _panel_app(size=(140, 34)) as panel:
        assert panel.panel_width() == PANEL_MAX_WIDTH
        assert panel.panel_width() == 104


# -- header freshness vs per-account staleness -------------------------------
#
# Reported against v0.44.38: the title read `2h ago` while five of six accounts
# had been refreshed under two minutes earlier, and `r` did not move it. The
# header took the OLDEST `fetched_at`, so a single account sitting in its
# per-account backoff pinned the age of the whole set — and because a forced
# re-probe that misses again keeps the previous report object (and its old
# stamp), no number of refreshes could unstick it. These tests hold the two
# halves of the contract that replaced it: the title states when the set was
# last confirmed, and the individual staleness stays on the account it belongs
# to.


def _ink(style) -> str:  # noqa: ANN001 — Style | str, as Rich hands it back
    """Hex of a span's foreground, matching ``test_band_panels._ink``."""
    resolved = Style.parse(style) if isinstance(style, str) else style
    assert resolved.color is not None and resolved.color.triplet is not None
    return resolved.color.triplet.hex.lower()


def _aged(report, fetched_at: float):
    """A report stamped at ``fetched_at``, since ``_report`` builds live ones."""
    report.fetched_at = int(fetched_at)
    return report


def test_one_stuck_account_does_not_pin_the_header_to_its_age() -> None:
    """The bug, at the helper the title is computed from.

    Five accounts confirmed 1.8 minutes ago and one serving 169-minute-old
    last-good: the header must describe the confirmation, not the straggler.
    """
    now = 200 * 60_000.0
    fresh = [
        _aged(
            _report(_percent(f"a:5h:{index}", "5 hour", 20.0, shared=True), identity=f"a{index}@x"),
            now - 1.8 * 60_000,
        )
        for index in range(5)
    ]
    stuck = _aged(
        _report(
            _percent("kimi:7d", "7 day", 64.0, shared=True),
            provider="kimi",
            identity="cred:8",
        ),
        now - 169 * 60_000,
    )
    stuck.consecutive_failures = 1

    fetched_ms = OperatorApp._usage_data_fetched_ms([*fresh, stuck])

    assert fetched_ms == max(report.fetched_at for report in fresh)
    # What the title actually renders, not just the millisecond arithmetic.
    assert format_age(now - fetched_ms) == "1m ago"


def test_the_stuck_account_still_carries_its_own_last_known_note() -> None:
    """The header got less specific, so the per-account honesty must not.

    A fresh title next to a silently stale row would be a worse lie than the
    one this change removes; the note is what keeps the individual age visible.
    """
    now = 200 * 60_000.0
    fresh = _aged(
        _report(_percent("a:5h", "5 hour", 20.0, shared=True), identity="a@x"),
        now - 1.8 * 60_000,
    )
    stuck = _aged(
        _report(
            _percent("kimi:7d", "7 day", 64.0, shared=True),
            provider="kimi",
            identity="cred:8",
        ),
        now - 169 * 60_000,
    )
    stuck.consecutive_failures = 1

    reports = [fresh, stuck]
    header_ms = OperatorApp._usage_data_fetched_ms(reports)
    lines = _lines(reports, now=now, header_ms=header_ms)
    title_age = format_age(now - header_ms)

    assert title_age == "1m ago"
    assert any("last known 2h ago" in line for line in lines), lines
    # The healthy account is not labelled by its neighbour's failure.
    assert sum("last known" in line for line in lines) == 1, lines


def test_a_wholly_stale_set_still_reports_its_real_age() -> None:
    """Freshness measured from the newest stamp is not freshness invented.

    With nothing refreshed there is no recent confirmation to report, so the
    title must still say how old the numbers are rather than ``just now``.
    """
    now = 200 * 60_000.0
    stale = [
        _aged(
            _report(_percent(f"a:5h:{index}", "5 hour", 20.0, shared=True), identity=f"a{index}@x"),
            now - (150 + index) * 60_000,
        )
        for index in range(3)
    ]

    fetched_ms = OperatorApp._usage_data_fetched_ms(stale)

    assert fetched_ms == max(report.fetched_at for report in stale)
    assert format_age(now - fetched_ms) == "2h ago"


def test_reports_without_a_usable_stamp_fall_back_to_the_wall_clock() -> None:
    """The cold path: no stamp anywhere means "as of now", not an epoch age."""
    import time as _time

    unstamped = [_aged(_report(_percent("a:5h", "5 hour", 20.0, shared=True)), 0)]

    assert OperatorApp._usage_data_fetched_ms([]) == pytest.approx(_time.time() * 1000, abs=5_000)
    assert OperatorApp._usage_data_fetched_ms(unstamped) == pytest.approx(
        _time.time() * 1000, abs=5_000
    )


# -- what the header stops speaking for must say so itself --------------------
#
# Round 1 review (R1/R2) and design review (D1/D2/D3). Taking the newest stamp
# makes the title a statement about the SET, which is only honest while every
# block the title does not describe is marked. These pin the three ways that
# marking was reachable-but-absent: a stale row with no failure streak, a
# synthetic stamp from a probe that never succeeded, and a layout free to
# discard the note while keeping the meter it qualifies.


def test_a_silently_stale_row_is_marked_even_with_no_failure_streak() -> None:
    """R1: the note's trigger is "the title does not speak for this row".

    An expired cache row round-trips through ``include_expired=True`` with
    ``consecutive_failures == 0`` and ``usage_unavailable`` false, and the
    lease-loser branch returns the stale payload verbatim. Keyed on the
    counters alone, such a row renders completely unmarked beside a fresh
    sibling under a ``just now`` header — stale numbers with nothing on screen
    saying so.
    """
    now = 200 * 60_000.0
    fresh = _aged(
        _report(_percent("a:5h", "5 hour", 20.0, shared=True), identity="a@x"),
        now - 20_000,
    )
    silent = _aged(
        _report(
            _percent("kimi:5h", "5 hour", 64.0, shared=True),
            provider="kimi",
            identity="cred:8",
        ),
        now - 180 * 60_000,
    )
    # The distinguishing fact: no failure counters are set on this row at all.
    assert silent.consecutive_failures == 0
    assert silent.usage_unavailable is False

    reports = [fresh, silent]
    header_ms = OperatorApp._usage_data_fetched_ms(reports)
    lines = _lines(reports, now=now, header_ms=header_ms)

    assert format_age(now - header_ms) == "just now"
    assert any("last known 3h ago" in line for line in lines), lines
    # And the fresh sibling is still not labelled.
    assert sum("last known" in line for line in lines) == 1, lines


def test_a_row_level_with_the_header_is_not_marked_stale() -> None:
    """The other side of R1: the mark must mean something.

    Reports fetched in the same round differ by milliseconds, and below a
    minute ``format_age`` renders both the title and the row as ``just now``.
    A note there would contradict a title that agrees with it.
    """
    now = 200 * 60_000.0
    reports = [
        _aged(
            _report(_percent(f"a:5h:{index}", "5 hour", 20.0, shared=True), identity=f"a{index}@x"),
            now - 1.8 * 60_000 - index * 40,  # same round, milliseconds apart
        )
        for index in range(3)
    ]
    header_ms = OperatorApp._usage_data_fetched_ms(reports)

    assert not any("last known" in line for line in _lines(reports, now=now, header_ms=header_ms))


def test_a_failed_probe_stamp_never_becomes_the_header() -> None:
    """R2: ``max`` must count confirmations, not the clock reading of a miss.

    ``_mark_account_failure`` stamps a never-successful account's stub with
    ``now_ms`` — the moment the probe FAILED — on a report carrying no limits.
    Counted, it lets the title read ``just now`` sourced from an account that
    has never once reported a number, which one new login during an outage is
    enough to trigger.
    """
    now = 200 * 60_000.0
    last_good = [
        _aged(
            _report(_percent(f"a:5h:{index}", "5 hour", 20.0, shared=True), identity=f"a{index}@x"),
            now - 125 * 60_000,
        )
        for index in range(3)
    ]
    for report in last_good:
        report.consecutive_failures = 2

    stub = _report(provider="kimi", identity="cred:9")  # no limits: never succeeded
    stub.fetched_at = int(now)  # the time of the FAILED probe
    stub.consecutive_failures = 1

    honest = OperatorApp._usage_data_fetched_ms(last_good)
    with_stub = OperatorApp._usage_data_fetched_ms([*last_good, stub])

    assert format_age(now - honest) == "2h ago"
    # The stub must not move the header off the true age of the real numbers.
    assert with_stub == honest, format_age(now - with_stub)


def test_a_confirmed_account_outranks_a_last_good_one_of_the_same_age() -> None:
    """The confirmation filter is about provenance, not recency.

    A last-good report keeps the stamp of its last SUCCESS, which is a real
    reading but not a new one. When anything was actually confirmed, that is
    what the title reports.
    """
    now = 200 * 60_000.0
    confirmed = _aged(
        _report(_percent("a:5h", "5 hour", 20.0, shared=True), identity="a@x"),
        now - 30 * 60_000,
    )
    newer_but_failing = _aged(
        _report(_percent("k:5h", "5 hour", 64.0, shared=True), provider="kimi", identity="cred:8"),
        now - 10 * 60_000,
    )
    newer_but_failing.consecutive_failures = 1

    fetched_ms = OperatorApp._usage_data_fetched_ms([confirmed, newer_but_failing])

    assert fetched_ms == confirmed.fetched_at
    assert format_age(now - fetched_ms) == "30m ago"


def test_the_stale_note_survives_compaction_with_the_meters_it_qualifies() -> None:
    """D1: a short pane may not keep the stale meter and drop its warning.

    ``_window_rows`` compacts an over-tall block to the heading plus the rows
    whose index is a cut point. Before this, the account note was not one, so a
    short pane rendered a 169-minute-old meter with no staleness note anywhere
    on screen, under a title reading ``1m ago``.
    """
    now = 200 * 60_000.0
    stuck = _aged(
        _report(
            _percent("kimi:7d", "7 day", 64.0, shared=True),
            provider="kimi",
            identity="cred:8",
        ),
        now - 169 * 60_000,
    )
    stuck.consecutive_failures = 1

    body = build_usage_body([stuck], WIDTH, now, now - 1.8 * 60_000)
    note_index = next(index for index, line in enumerate(body.lines) if "last known" in line.plain)

    # The cut set is what `_window_rows` retains; the note has to be in it.
    assert note_index + 1 in body.cuts, sorted(body.cuts)


def test_a_degraded_account_is_marked_without_reading_the_note() -> None:
    """D2: the block's highest-contrast elements must not say "healthy".

    A stale account kept the same success-green status dot as a live one, so
    the only thing marking it was a dim sentence in the panel's decoration
    colour. The mark drops to the dim ramp instead; the bar keeps its quota
    tint, because the fill still measures what it measures.
    """
    now = 200 * 60_000.0
    healthy = _aged(
        _report(_percent("a:5h", "5 hour", 20.0, shared=True), identity="a@x"),
        now - 1.8 * 60_000,
    )
    stuck = _aged(
        _report(_percent("k:5h", "5 hour", 64.0, shared=True), provider="kimi", identity="cred:8"),
        now - 169 * 60_000,
    )
    stuck.consecutive_failures = 1

    body = build_usage_body([healthy, stuck], WIDTH, now, healthy.fetched_at)
    meters = [line for line in body.lines if "5 hour" in line.plain and "●" in line.plain]
    healthy_mark, stale_mark = (_ink(line.spans[0].style) for line in meters)

    assert healthy_mark != stale_mark
    assert stale_mark == theme_mod.semantic_color("dim").lower()
    # The note itself is promoted off the decoration ramp (D2).
    note = next(line for line in body.lines if "last known" in line.plain)
    assert _ink(note.style) == theme_mod.semantic_color("warning").lower()


@pytest.mark.asyncio
async def test_the_title_names_how_many_accounts_it_does_not_speak_for() -> None:
    """D3: a bare age never said WHAT was that age.

    The suffix is pinned chrome, so unlike the per-account note it cannot be
    scrolled past or dropped by the row budget — which is what makes a short
    pane honest even when the block's own note is off screen.
    """
    now = 200 * 60_000.0
    fresh = _aged(
        _report(_percent("a:5h", "5 hour", 20.0, shared=True), identity="a@x"),
        now - 1.8 * 60_000,
    )
    stuck = _aged(
        _report(_percent("k:5h", "5 hour", 64.0, shared=True), provider="kimi", identity="cred:8"),
        now - 169 * 60_000,
    )
    stuck.consecutive_failures = 1

    async with _panel_app() as panel:
        panel.set_clock(now)
        panel.show_reports([fresh, stuck], now_ms=fresh.fetched_at)
        title = panel.render_lines_for_test()[0]
        # A healthy set says nothing extra — the suffix is an exception report.
        panel.show_reports([fresh], now_ms=fresh.fetched_at)
        healthy_title = panel.render_lines_for_test()[0]

    assert "1m ago" in title and "1 stale" in title, title
    assert "stale" not in healthy_title, healthy_title


def test_the_unavailable_note_keeps_its_age_on_a_narrow_card() -> None:
    """D4: shorten the note under pressure rather than clipping its age.

    ``usage unavailable — last known 2h ago`` is 37 cells and truncates from
    the right, so a narrow card kept the label and dropped the age — the half
    a reader can act on.
    """
    now = 200 * 60_000.0
    report = _aged(
        _report(_percent("k:7d", "7 day", 64.0, shared=True), provider="kimi", identity="cred:8"),
        now - 120 * 60_000,
    )
    report.usage_unavailable = True
    report.consecutive_failures = 5

    narrow = next(line for line in _lines([report], width=30, now=now) if "unavailable" in line)
    wide = next(line for line in _lines([report], width=76, now=now) if "unavailable" in line)

    assert "2h ago" in narrow, narrow
    assert "…" not in narrow, narrow
    # The full sentence is still preferred wherever it fits.
    assert wide.strip() == "usage unavailable — last known 2h ago", wide
