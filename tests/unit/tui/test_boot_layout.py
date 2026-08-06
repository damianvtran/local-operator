"""Boot layout — the input is a centred card until the session has content.

Two things are worth machine-checking here, and they are not the same thing:

- the SWITCH: one class on the Screen carries both layouts, and it rides the
  same condition as the welcome splash. If those two can disagree the app shows
  a centred boot card under a populated transcript, so the flip is pinned in
  both directions (first conversation block, then ``/clear``).
- the GEOMETRY: the card is clamped and centred, the status band stays its last
  row, and no rendered row is ever wider than the terminal — including at 16 and
  20 cells, where the clamp has to degrade to "as wide as there is room for"
  rather than hold its floor and overflow.

The frame is read from the compositor rather than from widget sizes: a size
field can be stale and a region can be off-screen, while the composed strips are
what the terminal is actually sent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len

from local_operator.tui import theme as theme_mod
from local_operator.tui.app import (
    BOOT_CARD_CLASS,
    BOOT_CARD_MIN_INSET,
    BOOT_LAYOUT_CLASS,
    OperatorApp,
)
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import GAP_CLASS, NoticeBlock, TranscriptView
from local_operator.tui.widgets.welcome import WelcomeView

TCSS = Path(__file__).parent.parent.parent.parent / "local_operator" / "tui" / "local_operator.tcss"

#: (width, height) pairs the app is actually used at, plus the two absurd ones
#: the minimalism suite measures. 16x10 is below every clamp and tier floor in
#: the design, which is exactly why it is here.
SIZES = [(16, 10), (20, 12), (40, 20), (80, 24), (200, 40)]


class FakeSession:
    """Minimal SessionProtocol stand-in: enough to boot and take one prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    @property
    def session_id(self) -> str:
        return "sess"

    @property
    def agent_id(self) -> str:
        return "agent"

    @property
    def is_streaming(self) -> bool:
        return False

    @property
    def model_label(self) -> str:
        return "test/model"

    @property
    def conversation_name(self) -> str:
        return ""

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        return text

    async def complete_once(self, system: str, prompt: str) -> str:
        return ""

    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str) -> None:
        pass

    def abort(self, reason: str = "interrupted") -> None:
        pass

    def subscribe(self, handler: Any) -> Any:
        return lambda: None

    async def dispose(self) -> None:
        pass


async def _factory(session: FakeSession) -> FakeSession:
    return session


def _make_app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


async def _settle(pilot, ticks: int = 24) -> None:  # type: ignore[no-untyped-def]
    """Pause until the boot frame stops moving, not for a fixed count.

    Two things settle here and both are timing-dependent: the splash's poll timer
    (see test_snapshot._settle) and the composition's vertical reserve, which is
    measured off a laid-out frame and re-measured until it agrees with itself
    (OperatorApp._sync_boot_composition). A fixed pause count races both, and the
    race changes the frame's geometry rather than only its segmentation.
    """
    welcome = pilot.app.query_one(WelcomeView)
    previous: tuple[bool, int] | None = None
    for _ in range(ticks):
        await pilot.pause()
        current = _reserve(pilot.app)
        if welcome._timer is None and current == previous:
            break
        previous = current
    await pilot.pause()


def _reserve(app: OperatorApp) -> tuple[bool, int]:
    """The composition's chrome: the ground row above the card, and the lift."""
    dock = app.query_one("#input-dock")
    return dock.has_class(GAP_CLASS), dock.styles.padding.bottom


def _rows(app: OperatorApp) -> list[str]:
    """The composed frame, one string per terminal row."""
    return [strip.text for strip in app.screen._compositor.render_strips()]


def _clamp() -> tuple[int, int, int]:
    """The card's clamp — proportion, floor and cap — read from the sheet.

    Parsed rather than duplicated as a constant: the stylesheet is the only place
    the numbers live, so a change to any of the three lands here as an arithmetic
    failure instead of quietly agreeing with a stale copy. Read off the
    ``.boot-card`` selector, because the clamp is CONDITIONAL — a plain
    ``Screen.boot`` panel is the full-width bar the base rule gives it, and the
    app applies the class only where the resolved width leaves a real margin.
    """
    rule = re.search(
        r"^Screen\.boot\.boot-card #input-shell\s*\{([^}]*)\}", TCSS.read_text(), re.MULTILINE
    )
    assert rule is not None, "the boot card's clamp rule is gone from the stylesheet"
    body = rule.group(1)
    percent = int(re.search(r"width:\s*(\d+)%", body).group(1))  # type: ignore[union-attr]
    floor = int(re.search(r"min-width:\s*(\d+)", body).group(1))  # type: ignore[union-attr]
    cap = int(re.search(r"max-width:\s*(\d+)\s*;", body).group(1))  # type: ignore[union-attr]
    return percent, floor, cap


def _expected_card_width(terminal_width: int) -> int:
    """The width the panel actually renders at, clamp AND threshold.

    ``min(box, cap, max(floor, proportion))`` when that leaves at least
    ``BOOT_CARD_MIN_INSET`` cells of ground, else the full box: an inset of one to
    three cells is not a card, so the app does not ask for one (see
    ``OperatorApp._sync_boot_card``).
    """
    percent, floor, cap = _clamp()
    box = terminal_width - 2  # the screen's one-cell left and right inset
    card = min(box, cap, max(floor, box * percent // 100))
    return card if box - card >= BOOT_CARD_MIN_INSET else box


@pytest.mark.asyncio
async def test_layout_flips_on_the_first_conversation_block_and_back_on_clear() -> None:
    """One condition drives the splash and the layout, in both directions."""
    app = _make_app()
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        shell = app.query_one("#input-shell")

        # Boot: class on, splash visible, card clamped and centred.
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)
        assert app.query_one(WelcomeView).display is True
        assert shell.region.width == _expected_card_width(100) == 75
        left = shell.region.x - 1  # minus the screen's own inset
        right = 98 - shell.region.width - left
        assert abs(left - right) <= 1, (left, right)

        # First conversation block: docked, full width, splash retired.
        app.query_one(Editor).text = "hello"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert not app.screen.has_class(BOOT_LAYOUT_CLASS)
        assert app.query_one(WelcomeView).display is False
        assert shell.region.width == 98, "the docked panel spans the content box"
        assert shell.region.x == 1

        # /clear puts both back — the same mechanism, the other direction.
        app._clear_transcript()
        await pilot.pause()
        await pilot.pause()
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)
        assert app.query_one(WelcomeView).display is True
        assert shell.region.width == 75


@pytest.mark.asyncio
async def test_the_splash_stays_attached_to_the_card_when_the_terminal_has_room() -> None:
    """The pair travels together: exactly ONE row between them, whatever the slack.

    Pinned at 40 rows rather than 28 on purpose: at 28 the block fills the region it
    is given, so any vertical rule at all produces the same frame and the test would
    pass with the placement deleted. The slack only exists to be misplaced on a tall
    terminal, which is where rows opening up between the splash and the input turn
    the composition back into a logo adrift over a bar.

    Where the slack GOES is the centring test below; this is the invariant that
    survived it — the separator is one row, and the row above the block is empty
    because the block starts where the slack ends, at either end of it.
    """
    app = _make_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        welcome = app.query_one(WelcomeView).region
        region = app.query_one(TranscriptView).content_region
        dock = app.query_one("#input-dock").region
        assert region.height - welcome.height >= 1, "premise: this size has slack to place"
        # The separator is the dock's own top margin, which sits OUTSIDE its region:
        # the block ends one row above the dock, and the card is the dock's first row.
        assert welcome.bottom == dock.y - 1
        assert app.query_one("#input-shell").region.y == dock.y
        rows = _rows(app)
        assert not rows[welcome.y - 1].strip(), "slack above"
        assert rows[welcome.y].strip(), "the block starts where the slack ends"


@pytest.mark.asyncio
async def test_the_card_is_a_bounded_fill_and_not_a_box() -> None:
    """What makes the boot input read as a card is the surface step, bounded.

    The mandate allows no border, rule or second fill to draw the edge, so the
    edge has to BE the fill's edge: one elevation step inside the clamp, ground
    immediately outside it, and no line character anywhere near it.
    """
    app = _make_app()
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        card = app.query_one("#input-shell").region
        surface = theme_mod.semantic_color("surface")
        ground = theme_mod.semantic_color("bg")

        def bg(x: int, y: int) -> str:
            """The composed background at one cell, as a `$lo-*` token value."""
            return app.screen.get_style_at(x, y).bgcolor.triplet.hex.lower()

        for y in range(card.y, card.bottom):
            assert bg(card.x, y) == surface, (y, "the card's first cell")
            assert bg(card.right - 1, y) == surface, (y, "the card's last cell")
            assert bg(card.x - 1, y) == ground, (y, "ground outside the card")
            assert bg(card.right, y) == ground, (y, "ground outside the card")
        # And the row above it is ground too: the fill starts at the card, not at
        # some rule drawn over the transcript.
        assert bg(card.x, card.y - 1) == ground

        rows = _rows(app)
        for glyph in "─│┌┐└┘━┃╭╮╰╯▏▕":
            assert glyph not in "".join(rows), glyph


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_no_rendered_row_exceeds_the_terminal_in_either_layout(
    size: tuple[int, int],
) -> None:
    """The clamp degrades instead of overflowing, in both layouts.

    ``min-width: 75`` would hold its floor on a 20-cell terminal if
    ``max-width: 100%`` did not beat it, and every row of the card would then be
    55 cells wider than the screen.
    """
    width, height = size
    app = _make_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _settle(pilot)

        for label in ("boot", "content"):
            rows = _rows(app)
            assert len(rows) == height, label
            for index, row in enumerate(rows):
                assert cell_len(row) <= width, (label, index, repr(row))
            # The input panel is the one widget that may never be pushed off the
            # screen: it is what the user types into, and a clipped one is a dead
            # app that still paints.
            card = app.query_one("#input-shell").region
            assert card.width <= width - 2, (label, card)
            assert card.right <= width, (label, card)
            assert card.bottom <= height, (label, card)
            if label == "boot":
                assert card.width == _expected_card_width(width)
                app.query_one(Editor).text = "hello"
                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()


@pytest.mark.asyncio
async def test_the_status_band_stays_the_cards_last_row_in_both_layouts() -> None:
    """The band reads as the app's footer, so it travels with the input rather
    than stranding itself at the bottom of the screen when the panel narrows."""
    app = _make_app()
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        for label in ("boot", "content"):
            card = app.query_one("#input-shell").region
            band = app.query_one("#status-band").region
            row = app.query_one("#input-row").region
            assert band.bottom == card.bottom - 1, label  # the card's padding row
            assert band.y > row.y, label
            assert card.x <= band.x and band.right <= card.right, label
            if label == "boot":
                app.query_one(Editor).text = "hi"
                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()


@pytest.mark.asyncio
async def test_the_command_picker_opens_inside_the_boot_card() -> None:
    """The picker mounts between the input row and the band in both layouts, so
    in the boot layout it has to live inside the clamp — and the card growing
    around it must not push the input off the bottom of the screen."""
    app = _make_app()
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/"
        await pilot.pause()
        await pilot.pause()

        picker = app.query_one(Editor).picker
        assert picker.is_open()
        card = app.query_one("#input-shell").region
        assert app.screen.has_class(BOOT_LAYOUT_CLASS), "a picker is not content"
        assert picker.region.width <= card.width
        assert card.x <= picker.region.x and picker.region.right <= card.right
        assert card.bottom <= 28, card
        for row in _rows(app):
            assert cell_len(row) <= 100, repr(row)
        # The splash yields the rows the taller card needs rather than the card
        # overrunning the screen: the transcript region shrank, so the block
        # rebuilt smaller inside it.
        welcome = app.query_one(WelcomeView)
        assert welcome.region.bottom <= app.query_one("#input-dock").region.y
        assert welcome.region.height <= app.query_one(TranscriptView).content_region.height


@pytest.mark.asyncio
async def test_the_card_is_a_card_or_a_bar_and_never_a_sliver() -> None:
    """Swept one column at a time, because the defect only existed in a band.

    ``max(75, 70%)`` with no lower guard put 1 to 3 cells of ground beside the
    panel between 78 and 84 columns — at 80, the commonest terminal, 2 on the left
    and 3 on the right. A borderless fill offset by less than the app's own gutter
    does not read as a card; it reads as a full-width bar that is misaligned, and
    there is no edge for the eye to attribute the offset to. So every width is one
    of exactly two things: a bar that meets both walls of the content box, or a
    card with a real margin either side.

    One app RESIZED rather than sixty booted: the resize is also the event the
    threshold is decided on, so this exercises the path a user actually takes when
    they drag their terminal across the band.
    """
    app = _make_app()
    async with app.run_test(size=(72, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        for width in list(range(72, 131)) + [160, 190, 200]:
            await pilot.resize_terminal(width, 28)
            await _settle(pilot)
            card = app.query_one("#input-shell").region
            left = card.x - 1  # minus the screen's own one-cell inset
            right = (width - 1) - card.right
            if not app.screen.has_class(BOOT_CARD_CLASS):
                assert (left, right) == (0, 0), (width, "not a card, so it is the honest bar")
            else:
                assert left >= BOOT_CARD_MIN_INSET // 2, (width, left, right)
                assert right >= BOOT_CARD_MIN_INSET // 2, (width, left, right)
                assert abs(left - right) <= 1, (width, "centred, give or take an odd cell")
            assert card.width == _expected_card_width(width), width
            for row in _rows(app):
                assert cell_len(row) <= width, (width, repr(row))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(190, 48), (160, 48), (120, 40)])
async def test_a_wide_terminal_gets_a_card_and_not_a_wide_bar(size: tuple[int, int]) -> None:
    """The proportion needs a ceiling as much as a floor.

    Unbounded, ``70%`` resolves to 131 cells at 190 columns, which is a bar again:
    what makes a borderless surface read as a card is the ground around it, and a
    fill that wide has none to speak of. The cap also has to leave the card clearly
    wider than the block above it, or the composition inverts and the input starts
    reading as a caption to the splash.
    """
    _percent, _floor, cap = _clamp()
    app = _make_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _settle(pilot)
        card = app.query_one("#input-shell").region
        assert card.width <= cap
        rows = _rows(app)
        widest_above = max(
            (len(row.rstrip()) - (len(row) - len(row.lstrip())) for row in rows[: card.y]),
            default=0,
        )
        assert card.width > widest_above, (card.width, widest_above)


@pytest.mark.asyncio
@pytest.mark.parametrize("notices", [1, 2, 3])
async def test_notices_under_the_splash_never_scroll_the_region(notices: int) -> None:
    """The splash shares its region, so it may only budget for what is LEFT.

    Budgeting the whole region overdrew it by exactly the siblings' rows, and the
    boot layout bottom-aligns the column — so what scrolled out of sight was the top
    of the logo, with a scrollbar thumb appearing beside it. Both triggers are
    ordinary: the ``/clear`` receipt is one row, and a failing MCP server is another
    each. Measured at 96x28, where the block wants every row the region has.
    """
    app = _make_app()
    async with app.run_test(size=(96, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        for index in range(notices):
            app._system_notice(f"MCP srv{index} failed: command not found", "error")
        await _settle(pilot)

        transcript = app.query_one(TranscriptView)
        assert transcript.scroll_offset.y == 0, "the top of the block is what scrolls away"
        assert transcript.show_vertical_scrollbar is False
        assert transcript.virtual_size.height <= transcript.size.height
        welcome = app.query_one(WelcomeView)
        region = transcript.content_region
        assert welcome.region.y >= region.y, "the block starts inside the region"
        assert welcome.region.bottom <= region.bottom
        # And the splash still owns the rows it did not give away.
        assert welcome.region.height > 0


@pytest.mark.asyncio
async def test_the_mark_survives_two_failed_servers_at_the_tightest_size() -> None:
    """96x28 is where the block wants the whole region, so it is where the budget
    shows: two failing servers cost three rows (a row each plus the separator), and
    the ladder spends the version number rather than the product's own mark."""
    app = _make_app()
    async with app.run_test(size=(96, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app._system_notice("MCP one failed: command not found", "error")
        app._system_notice("MCP two failed: command not found", "error")
        await _settle(pilot)
        frame = "\n".join(_rows(app))
        assert "l o c a l   o p e r a t o r" in frame, "the wordmark"
        assert "▄█████▄" in frame, "the mark's first row — the one that scrolled away"
        assert frame.count("failed: command not found") == 2


@pytest.mark.asyncio
async def test_the_first_block_under_a_visible_splash_opens_with_one_blank_row() -> None:
    """A receipt flush against ``ctrl+d  quit`` reads as a line that fell out of the
    block, not as the answer to what the user just did. One blank row, from the
    app's one vertical separator class — and left-aligned, because a centred notice
    would be a second alignment convention.
    """
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app._clear_transcript()  # its receipt is the one-row block
        await _settle(pilot)

        blocks = app.query_one(TranscriptView).blocks()
        assert len(blocks) == 1 and isinstance(blocks[0], NoticeBlock)
        assert blocks[0].has_class(GAP_CLASS)
        rows = _rows(app)
        receipt = blocks[0].region
        assert "transcript cleared" in rows[receipt.y]
        assert not rows[receipt.y - 1].strip(), "one blank row between the two blocks"
        welcome = app.query_one(WelcomeView)
        assert rows[welcome.region.bottom - 1].strip(), "and the splash's last row is drawn"
        assert receipt.y == welcome.region.bottom + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(190, 48), (160, 48), (120, 40)])
async def test_the_composition_is_centred_when_the_rows_are_there(size: tuple[int, int]) -> None:
    """Splash, separator and card are ONE block, centred in the screen.

    Resting the pair on the bottom of the screen left the upper two thirds of a
    48-row terminal empty. The slack is split instead — above the splash and below
    the card — and the card keeps one row of ground above it so the hints are not
    flush against the fill.
    """
    width, height = size
    app = _make_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _settle(pilot)
        region = app.query_one(TranscriptView).content_region
        welcome = app.query_one(WelcomeView).region
        card = app.query_one("#input-shell").region
        above = welcome.y - region.y
        below = (height - 1) - card.bottom  # the screen's own inset is not slack
        assert above >= 1, "premise: this size has rows to spare"
        assert abs(above - below) <= 1, (above, below)
        rows = _rows(app)
        assert not rows[card.y - 1].strip(), "a ground row above the card, not a fill row"
        assert rows[welcome.bottom - 1].strip(), "and the splash ends where it ends"
        assert welcome.bottom == card.y - 1, "one row, not two"


@pytest.mark.asyncio
async def test_a_short_terminal_keeps_resting_the_splash_on_the_card() -> None:
    """Centring is CONDITIONAL, and 96x28 is why.

    Every row the composition reserves comes out of the splash's budget, and the
    splash pays in whole sections: reserving rows to centre a block that already
    fills the region would trade the mark for air. With nothing spare, the pair
    rests on the card exactly as it did before — which is the same graceful answer
    as the docked bar at 40 columns.
    """
    app = _make_app()
    async with app.run_test(size=(96, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        assert _reserve(app) == (False, 0)
        welcome = app.query_one(WelcomeView).region
        assert welcome.bottom == app.query_one("#input-dock").region.y, "rests on the card"
        assert "▄█████▄" in "\n".join(_rows(app)), "and it kept the mark"


@pytest.mark.asyncio
async def test_the_conversation_layout_reserves_nothing_and_clear_puts_it_back() -> None:
    """The centred composition is the EMPTY state's, and only the empty state's.

    A reserve left behind after the first block would be a hole under a populated
    transcript, and the lift lives on the dock — which the conversation layout still
    docks full-width at the bottom of the screen.
    """
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        assert _reserve(app) != (False, 0), "premise: this size centres"

        app.query_one(Editor).text = "hello"
        await pilot.press("enter")
        await _settle(pilot)
        assert _reserve(app) == (False, 0)
        shell = app.query_one("#input-shell").region
        assert shell.width == 118 and shell.x == 1, "full-width bar"
        assert shell.bottom == 39, "docked against the screen's bottom inset"

        app._clear_transcript()
        await _settle(pilot)
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)
        assert _reserve(app) != (False, 0), "and the centring comes back"
