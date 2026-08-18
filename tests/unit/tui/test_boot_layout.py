"""Boot layout — the input is a centred card until the session has content.

Three things are worth machine-checking here, and they are not the same thing:

- the SWITCH: one class on the Screen carries both layouts, and it rides the
  same condition as the welcome splash. If those two can disagree the app shows
  a centred boot card under a populated transcript, so the flip is pinned in
  both directions (first conversation block, then ``/clear``).
- the GEOMETRY: the card is clamped and centred, the status band stays its last
  row, and no rendered row is ever wider than the terminal — including at 16 and
  20 cells, where the clamp has to degrade to "as wide as there is room for"
  rather than hold its floor and overflow.
- the STILLNESS: the frame the user stares at while the session connects holds
  perfectly still except for the mark's own glow, and holds still ENTIRELY once
  animation is gated off. Two things had to be pinned here, and neither was a
  design choice anyone made. The flicker was the editor's blinking caret
  inverting a letter of the placeholder twice a second. The other was
  structural: the composition measured itself off the frame it had just changed,
  so the splash fell from the top of the screen and the card rose from the
  bottom until they met, once per painted frame. The first frame is now the
  settled frame, and a test walks every painted frame to say so.

The frame is read from the compositor rather than from widget sizes: a size
field can be stale and a region can be off-screen, while the composed strips are
what the terminal is actually sent.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.text import Text
from textual.css.query import NoMatches
from textual.screen import Screen

from local_operator.harness.types import AgentMessage, ImageContent
from local_operator.session.naming import ConversationName
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import (
    BOOT_CARD_CLASS,
    BOOT_CARD_MIN_INSET,
    BOOT_LAYOUT_CLASS,
    Band,
    OperatorApp,
)
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import GAP_CLASS, NoticeBlock, TranscriptView
from local_operator.tui.widgets.welcome import (
    LOGO_MARK,
    TIPS,
    WORDMARK_SPACED,
    WelcomeView,
    app_version,
)
from tests.unit.tui.conftest import caret_cells, composer_cells

TCSS = Path(__file__).parent.parent.parent.parent / "local_operator" / "tui" / "local_operator.tcss"

#: (width, height) pairs the app is actually used at, plus the two absurd ones
#: the minimalism suite measures. 16x10 is below every clamp and tier floor in
#: the design, which is exactly why it is here.
SIZES = [(16, 10), (20, 12), (40, 20), (80, 24), (200, 40)]

#: Enough of the composer's placeholder to find its row, minus the trailing
#: ellipsis — a truncated placeholder is still the row this looks for.
PLACEHOLDER_HEAD = "Message Local Operator"


class FakeSession:
    """Minimal SessionProtocol stand-in: enough to boot and take one prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.asides: list[list[Any]] = []
        self.adopted: list[list[Any]] = []

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
    def model(self) -> Any:
        return None

    def set_model(self, model: Any) -> None:
        pass

    @property
    def goal(self) -> str:
        return ""

    def set_goal(self, text: str) -> str:
        return text

    async def seed_history(self, messages: list[Any]) -> None:
        pass

    @property
    def conversation_name(self) -> str:
        return self.conversation_name_state.text

    @property
    def conversation_name_state(self) -> ConversationName:
        # The real holder, created on first read: `user_set` precedence (a
        # human rename outranks every generated title, forever) is behaviour
        # the TUI reads before it spends a re-title call, so a fake that
        # reimplemented it as a bare string would hide a regression in it.
        state = getattr(self, "_name_state", None)
        if state is None:
            state = self._name_state = ConversationName()
        return state

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        return self.conversation_name_state.set(text, user_set=user_set)

    async def complete_once(self, system: str, prompt: str) -> str:
        return ""

    def history(self) -> list[AgentMessage]:
        return []

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        pass

    def set_approval_handler(self, handler: object | None) -> None:
        # The TUI installs its own approval gate on boot (the stdin gate
        # deadlocks under a full-screen app); fakes only need to accept it.
        self.approval_handler = handler

    def set_ask_handler(self, handler: object | None) -> None:
        # The TUI installs the `ask` tool's picker surface on boot, and that
        # install is what makes the tool exist; fakes only need to accept it.
        self.ask_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        pass

    def subscribe(self, handler: Any) -> Any:
        return lambda: None

    async def dispose(self) -> None:
        pass

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> str:
        # Recorded, not answered: the aside's no-trace contract is proven
        # against the real Session in tests/unit/session/test_aside.py. Here
        # the only thing that must hold is that the app can call it.
        self.asides.append(list(turns))
        return ""

    async def adopt_aside(self, messages: list[Any]) -> None:
        self.adopted.append(list(messages))

    async def compact_now(self) -> CompactionOutcome:
        # No history to compact: this fake never carries a conversation, which
        # is the state a real session answers with the same refusal.
        return CompactionOutcome(
            ran=False, reason="nothing_to_compact", detail="nothing to compact"
        )


async def _factory(session: FakeSession) -> FakeSession:
    return session


def _make_app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


async def _settle(pilot, ticks: int = 24) -> None:  # type: ignore[no-untyped-def]
    """Pause until the boot frame stops moving, not for a fixed count.

    The COMPOSITION settles before the first paint and never moves again (see
    test_the_boot_frame_paints_once_and_never_converges_into_place), but the
    splash's poll timer does not: the model label lands a fraction of a second
    in, the block can change height with it, and the composition is recomputed
    from the message that reports it. The poll retiring with an unchanged reserve
    is the settled edge; a fixed pause count races it, and the race changes the
    frame's geometry rather than only its segmentation.
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


def _styled_rows(app: OperatorApp) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    """The composed frame WITH its segment styles.

    :func:`_rows` answers "what does it say"; a repaint that only recolours a
    cell — a blinking caret inverting a letter, the mark glowing — is
    invisible to it. Stillness is a claim about the bytes the terminal receives,
    so the styles have to be in the comparison.
    """
    return [
        (strip.text, tuple((str(segment.style), segment.text) for segment in strip._segments))
        for strip in app.screen._compositor.render_strips()
    ]


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
            bgcolor = app.screen.get_style_at(x, y).bgcolor
            assert bgcolor is not None and bgcolor.triplet is not None
            return bgcolor.triplet.hex.lower()

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
async def test_notices_spend_the_mark_before_existing_information() -> None:
    """Two startup failures consume rows from the splash's region.

    The old refund ladder spent version and tip to preserve the mark, then
    restored them only when an even shorter box dropped the mark. The strict
    ladder keeps everything the user could already read and spends the largest
    decoration first: wordmark, version, tip and keys survive; the mark yields.
    """
    app = _make_app()
    async with app.run_test(size=(96, 28)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app._system_notice("MCP one failed: command not found", "error")
        app._system_notice("MCP two failed: command not found", "error")
        await _settle(pilot)
        frame = "\n".join(_rows(app))
        assert "l o c a l   o p e r a t o r" in frame
        assert "▄█████▄" not in frame
        # Derived, not pinned: the boot frame prints the INSTALLED version, so a
        # literal here turns every release into a failing test and teaches the
        # next person to edit the assertion rather than read it.
        #
        # Two assertions because the derived one alone is a tautology - it
        # compares the frame against the same function that rendered it, and
        # would still pass if `app_version()` started returning junk. The regex
        # is the independent half: whatever is shown has to LOOK like a version.
        assert re.search(r"v\d+\.\d+\.\d+", frame), frame
        assert f"v{app_version()}" in frame
        assert "/help" in frame
        assert "/resume picks up a recent session where you left off" in frame
        assert frame.count("failed: command not found") == 2


@pytest.mark.asyncio
async def test_more_terminal_height_never_removes_welcome_content() -> None:
    """D18 lives in the coupling between the terminal and the widget's budget.

    A pure builder test cannot catch the app handing that builder a surprising
    region. Walk every height around the classic 24-row terminal in the REAL app
    and read the composited frame: once a section appears, it never disappears.
    """
    previous: frozenset[str] = frozenset()
    for height in range(14, 34):
        app = _make_app()
        async with app.run_test(size=(80, height)) as pilot:
            await pilot.pause()
            await _settle(pilot)
            frame = "\n".join(_rows(app))
        current = frozenset(
            name
            for name, visible in (
                ("keys", "/help" in frame),
                ("tip", TIPS[0] in frame),
                ("version", "v0." in frame),
                ("wordmark", WORDMARK_SPACED in frame),
                ("mark", LOGO_MARK[0] in frame),
            )
            if visible
        )
        assert previous <= current, (height, previous - current, previous, current)
        previous = current


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


#: The one column where the splash's degradation ladder has a step between a
#: cell and the next one: the block is 9 rows drawn at 21 cells and 19 rows drawn
#: at 22. Measuring the splash a cell wider than it renders is invisible in the
#: middle of a tier and decides the whole composition here, which is why the
#: sweep is pinned at this width rather than at a comfortable one.
LADDER_EDGE_WIDTH = 25


@pytest.mark.asyncio
@pytest.mark.parametrize("height", list(range(20, 51, 3)))
async def test_the_composition_measures_the_splash_at_the_width_it_renders(
    height: int,
) -> None:
    """The block the app centres for is the block the layout engine draws.

    The composition's whole claim is that it can answer the layout engine's own
    question one step ahead of it, so the two can never disagree. They did: the
    width handed to ``spare_rows`` subtracted the transcript's padding but not
    its permanently reserved scrollbar column, so the splash was measured one
    cell wide. At 25 columns that cell is a tier edge — 19 rows measured for a
    block that renders 9 — and the frame came out with every one of the missing
    ten rows piled above the splash: at 25x30, 12 rows of ground above and 1
    below; at 25x50, 22 and 11; at 25x20 the reserve overshot the other way (0
    above, 3 below).

    Swept over heights rather than pinned at one, because the error is in a WIDTH
    and shows up as a mis-split of whatever slack the height provides — a single
    size would pin one arbitrary point of a wrong line.
    """
    app = _make_app()
    async with app.run_test(size=(LADDER_EDGE_WIDTH, height)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        transcript = app.query_one(TranscriptView)
        region = transcript.content_region
        welcome = app.query_one(WelcomeView).region
        card = app.query_one("#input-shell").region

        assert (
            welcome.width == region.width - transcript.scrollbar_size_vertical
        ), "premise: the reserved scrollbar column is not in the transcript's gutter"
        above = welcome.y - region.y
        below = (height - 1) - card.bottom  # the screen's own inset is not slack
        assert above >= 1, "premise: this size has rows to spare"
        assert abs(above - below) <= 1, (above, below, welcome.height)


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


@pytest.mark.asyncio
async def test_the_band_is_refitted_when_the_card_hands_back_the_width() -> None:
    """The band's row belongs to the box it is painted in, across the handover.

    The status band is the input panel's last row, so the boot card's clamp is its
    clamp too: while the splash is up it is fitted to ~97 cells of a 150-column
    terminal, and the first substantive prompt hands it the full 145 back. That
    hand-back is a class change on the Screen, not a terminal resize, so nothing
    told the band — and the frames straight after the opening submit, which are
    exactly the frames a user is watching when they press Enter, kept the card's
    row: a basename cwd, no effort segment, and a stub of the name the prompt had
    just earned.

    The band answers to its OWN ``Resize`` now (``Band.BoxChanged``), so this
    holds for any cause rather than for this one; the assertion is the general
    one, that what is painted is what the current box fits.
    """
    app = _make_app()
    async with app.run_test(size=(150, 24)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        band = app.query_one("#status-band", Band)
        status = app._status
        assert status is not None
        clamped = band.content_size.width
        assert clamped < 145, "premise: the boot card has the band clamped"

        app.query_one(Editor).text = "add todo guardrails to the operator loop"
        await pilot.press("enter")
        await _settle(pilot)

        assert band.content_size.width > clamped, "premise: the card gave the width back"
        painted = band.content
        assert isinstance(painted, Text), "the band is painted as a rich Text"
        assert (
            painted.plain == status._render(band.content_size.width).plain
        ), "the band is still painting the row it fitted to the boot card's box"
        # And the name the prompt just earned is on the row it is painted on.
        assert "Add todo guardrails" in painted.plain


#: Sizes where the composition has rows to spare, so the reserve is non-zero and
#: a staged one would have somewhere to travel from. 100x30 is here as the
#: control: it reserves nothing, which is why the regression this pins hid for so
#: long — every existing test ran at a size that could not show it.
REFLOW_SIZES = [(190, 48), (120, 40), (100, 50), (100, 30)]


def _watch_painted_frames(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, int]]:
    """Collect ``(splash top, splash rows, card top)`` for every PAINTED frame.

    ``Screen._compositor_refresh`` is the hook because it is the call that hands a
    frame to ``App._display``: Textual exposes no public "a frame was painted"
    signal, and widget geometry read at any other moment is a frame nobody saw.
    Frames with the splash hidden are skipped — the conversation layout is not the
    composition under test, and it holds no splash to move.
    """
    frames: list[tuple[int, int, int]] = []
    painted = Screen._compositor_refresh

    def record(self: "Screen[object]") -> None:
        painted(self)
        try:
            welcome = self.app.query_one(WelcomeView)
            card = self.app.query_one("#input-shell").region
        except NoMatches:
            return
        if welcome.display:
            frames.append((welcome.region.y, welcome.region.height, card.y))

    monkeypatch.setattr(Screen, "_compositor_refresh", record)
    return frames


def _composition(app: OperatorApp) -> tuple[int, int, int]:
    """The same triple as :func:`_watch_painted_frames`, read from the live app."""
    welcome = app.query_one(WelcomeView).region
    return welcome.y, welcome.height, app.query_one("#input-shell").region.y


@pytest.mark.asyncio
@pytest.mark.parametrize("size", REFLOW_SIZES)
async def test_the_boot_frame_paints_once_and_never_converges_into_place(
    monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    """The FIRST frame the terminal is sent is the settled frame.

    This is the "split that comes together from the top and bottom" the boot
    screen used to do, and it was never a declared animation: the composition was
    measured off a laid-out frame and re-measured one refresh later, but
    ``call_after_refresh`` resumes BEFORE the compositor has re-arranged. So each
    pass read the previous frame's splash offset against the padding it had just
    written, double-counted its own reserve, and overshot. At 190x48 the mark's
    top row walked 20, 3, 16, 6, 13, 8, 12, 9, 11, 10 while the input card walked
    42, 26, 39, 29, 36, 31, 35, 32, 34, 33 — the logo falling from the top and the
    card rising from the bottom until they met, over ten painted frames.

    Every painted frame is sampled, not just the first and the last, because the
    failure is the WALK: an assertion over the endpoints alone would pass a
    composition that bounced and happened to return.
    """
    frames = _watch_painted_frames(monkeypatch)
    app = _make_app()
    async with app.run_test(size=size) as pilot:
        await _settle(pilot)
        settled = _composition(app)

    assert frames, "premise: the boot screen painted at all"
    assert set(frames) == {
        settled
    }, f"the boot composition moved while the user watched: {sorted(set(frames))}"


@pytest.mark.asyncio
async def test_clear_puts_the_composition_back_in_a_single_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/clear`` re-enters the boot layout, and it lands centred on the first try.

    The same defect had a second home here. The clear hook restores the splash and
    resolves the composition, and the "transcript cleared" receipt is appended
    AFTER it — so the reserve was computed for a region that did not yet contain
    the receipt's rows, and the splash settled a row late once the frame was
    already up. The receipt goes through ``_append_block`` for exactly that
    reason; this is the test that says so.
    """
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(pilot)
        app.query_one(Editor).text = "hello"
        await pilot.press("enter")
        await _settle(pilot)

        frames = _watch_painted_frames(monkeypatch)
        app._clear_transcript()
        await _settle(pilot)
        settled = _composition(app)

    assert frames, "premise: the restored splash painted at all"
    assert set(frames) == {settled}, f"the composition moved after /clear: {sorted(set(frames))}"


# --- what MOVES on the boot frame ---------------------------------------------


@pytest.mark.asyncio
async def test_the_boot_composer_draws_its_caret_beside_the_placeholder() -> None:
    """The caret is on the frame AND the placeholder stays PROSE.

    Two earlier contracts in this file each gave up one of those. The first
    demanded the placeholder's leading cell stay inverted, which rendered
    `▉essage Local Operator…` — a block measuring 13.76:1 against the panel,
    roughly 2.6x the mark's own 3.71-5.35:1, parked on a word (D-05). The
    second dropped the caret entirely while the buffer was empty, which is the
    state a first-time user meets the app in: clicking the field changed
    nothing on the frame, so there was no way to tell that the next keystroke
    would land in it.

    Neither trade was necessary. The collision is a CELL collision, so the
    caret takes a cell of its own and the copy starts one column later: a
    solid block at the head of the field with `Message Local Operator…`
    unbroken beside it.

    Sampled across four stock blink periods, so the no-strobe half of the
    contract still fails loudly if blinking ever comes back.
    """
    app = _make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app.query_one(Editor).focus()
        await pilot.pause()
        assert app.query_one(Editor).cursor_blink is False

        samples = {tuple(composer_cells(app))}
        for _ in range(8):
            await asyncio.sleep(0.25)
            await pilot.pause()
            samples.add(tuple(composer_cells(app)))
        assert len(samples) == 1, f"the composer row changed between frames: {samples}"

        cells = composer_cells(app)
        # One caret, and it is on a BLANK cell — a caret carrying a letter is
        # the caret sitting on the copy again.
        assert caret_cells(cells) == [" "], "the empty composer is not showing a caret"

        # The copy survives as words, in ONE colour: a partially restyled run
        # would mean something is still painting over a character.
        placeholder = [(text, fg) for text, fg, _ in cells if PLACEHOLDER_HEAD in text]
        assert placeholder, f"the placeholder is broken into pieces: {cells}"
        assert placeholder[0][1] == theme_mod.semantic_color("dim").lower()

        # ...and the second affordance is on too: focus BRIGHTENS the chevron
        # (D23), in the neutral ramp rather than the accent — green is reserved
        # for "a turn is live" and this splash has no turn running (D5).
        chevron = [fg for text, fg, _ in cells if "❯" in text]
        assert chevron == [theme_mod.semantic_color("fg").lower()]
        assert chevron != [theme_mod.semantic_color("accent").lower()]


@pytest.mark.asyncio
async def test_the_caret_appears_solid_as_soon_as_the_buffer_has_content() -> None:
    """The other half of the rule: suppressed on the placeholder, present the
    instant there is anything to point at — at the END of the buffer, which is
    where a chat composer's caret lives, and inside it after a cursor move.

    Non-blinking is re-checked here rather than assumed: the caret the user
    actually meets is this one, and a blink reintroduced for typed text would be
    invisible to the boot-frame test above.
    """
    app = _make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app.query_one(Editor).focus()
        await pilot.pause()

        await pilot.press("h", "e", "l", "l", "o")
        await pilot.pause()
        assert caret_cells(composer_cells(app)) == [" "], "no caret at the insertion point"

        await pilot.press("left")
        await pilot.pause()
        assert caret_cells(composer_cells(app)) == ["o"], "no caret inside the text"

        # Solid, not blinking: four stock blink periods, one rendering.
        samples = set()
        for _ in range(8):
            await asyncio.sleep(0.25)
            await pilot.pause()
            samples.add(tuple(composer_cells(app)))
        assert len(samples) == 1, "the caret blinked once there was text"


@pytest.mark.asyncio
async def test_the_splash_holds_completely_still_under_the_animation_gate() -> None:
    """With animation off (this suite's autouse fixture), the whole boot frame
    is byte-identical over two seconds — text AND styles.

    Both moving parts are covered at once here: the caret, which is static
    unconditionally, and the mark's pulse, which the gate suppresses. The SVG
    goldens are captured from exactly this state, so anything that moves here
    turns a snapshot into a coin flip.
    """
    app = _make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app.query_one(Editor).focus()
        await pilot.pause()

        before = _styled_rows(app)
        for _ in range(4):
            await asyncio.sleep(0.5)
            await pilot.pause()
            assert _styled_rows(app) == before
