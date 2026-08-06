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
from local_operator.tui.app import BOOT_LAYOUT_CLASS, OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import TranscriptView
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
    """Pause until the splash's poll timer retires (see test_snapshot._settle).

    A fixed pause count races that timer, and the race changes the frame's
    segmentation rather than its characters — invisible here, but the same flake
    the snapshot suite had to fix.
    """
    welcome = pilot.app.query_one(WelcomeView)
    for _ in range(ticks):
        await pilot.pause()
        if welcome._timer is None:
            break
    await pilot.pause()


def _rows(app: OperatorApp) -> list[str]:
    """The composed frame, one string per terminal row."""
    return [strip.text for strip in app.screen._compositor.render_strips()]


def _clamp() -> tuple[float, int]:
    """The card's clamp, read from the sheet that implements it.

    Parsed rather than duplicated as a constant: the stylesheet is the only place
    the numbers live, so a change to the proportion or the floor lands here as an
    arithmetic failure instead of quietly agreeing with a stale copy. The
    ``max-width`` is asserted, not read — it carries no number, it is what makes
    the floor yield on a terminal narrower than 75 cells.
    """
    rule = re.search(r"^Screen\.boot #input-shell\s*\{([^}]*)\}", TCSS.read_text(), re.MULTILINE)
    assert rule is not None, "the boot card's clamp rule is gone from the stylesheet"
    body = rule.group(1)
    fraction = int(re.search(r"width:\s*(\d+)%", body).group(1)) / 100  # type: ignore[union-attr]
    floor = int(re.search(r"min-width:\s*(\d+)", body).group(1))  # type: ignore[union-attr]
    assert "max-width: 100%;" in body, "without this the floor overflows a narrow terminal"
    return fraction, floor


def _expected_card_width(terminal_width: int) -> int:
    """``max(floor, fraction of the content box)``, never wider than the box."""
    fraction, floor = _clamp()
    content = terminal_width - 2  # the screen's one-cell left and right inset
    return min(content, max(floor, int(content * fraction)))


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
async def test_the_splash_rests_on_the_card_when_the_terminal_has_room() -> None:
    """Every spare row goes ABOVE the block, none between it and the card.

    Pinned at 40 rows rather than 28 on purpose: at 28 the block fills the region
    it is given, so any vertical rule at all produces the same frame and the test
    would pass with the alignment deleted. The slack only exists to be misplaced
    on a tall terminal, which is where a gap between the splash and the input
    turns the composition back into a logo adrift over a bar.
    """
    app = _make_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        welcome = app.query_one(WelcomeView).region
        region = app.query_one(TranscriptView).content_region
        assert region.height - welcome.height >= 8, "premise: this size has real slack"
        assert welcome.bottom == app.query_one("#input-dock").region.y
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
