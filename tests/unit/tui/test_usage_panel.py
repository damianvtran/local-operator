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
from textual.app import App, ComposeResult
from textual.containers import Container

from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.usage_panel import (
    BAR_UNKNOWN,
    PANEL_PADDING_ROWS,
    UsagePanel,
    binding_limit,
    build_usage_body,
    collect_stats,
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


def _lines(reports, width: int = WIDTH, now: float = 0.0) -> list[str]:
    return [line.plain for line in build_usage_body(reports, width, now).lines]


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
