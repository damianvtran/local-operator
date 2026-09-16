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
import contextlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.text import Text
from textual.css.query import NoMatches
from textual.screen import Screen

from local_operator.harness.types import (
    AgentMessage,
    AskOption,
    AskQuestion,
    ImageContent,
)
from local_operator.session.naming import ConversationName
from local_operator.session.protocol import CompactionOutcome, RuntimeLocality
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import (
    BOOT_CARD_CLASS,
    BOOT_CARD_MIN_INSET,
    BOOT_LAYOUT_CLASS,
    Band,
    OperatorApp,
)
from local_operator.tui.session_catalog import SidebarSettings
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import (
    BOOT_COLUMN_CLASS,
    GAP_CLASS,
    SPINE_INDENT,
    NoticeBlock,
    TranscriptView,
    UserBlock,
)
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

#: A session name long enough that the band's row has a tail to clip. Used where
#: the claim is about the row the terminal is sent rather than about the name.
NAME = "Fix sidebar reconnect on session switch"


class FakeSession:
    """Minimal SessionProtocol stand-in: enough to boot and take one prompt."""

    # Runtime role (SessionProtocol). This fake stands in for an OWNER:
    # it carries no attached runtime, which is what the absent legacy
    # `is_remote` meant.
    owns_runtime = True
    outcome_is_synchronous = True
    runtime_locality: RuntimeLocality = "this-process"

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

    @property
    def effective_model(self) -> Any:
        # The fake never falls back, so selection and effective agree.
        return self.model

    @property
    def effective_model_label(self) -> str:
        return self.model_label

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        pass

    @property
    def goal(self) -> str:
        return ""

    def set_goal(self, text: str) -> str:
        return text

    @property
    def variables(self) -> Any:
        """Memory-only store for ``/credential``. Created on first use so
        tests that never touch credentials pay nothing for the property."""
        store = getattr(self, "_variables", None)
        if store is None:
            from local_operator.variables import VariableStore

            store = self._variables = VariableStore(cwd="/tmp", env={})
        return store

    async def variables_op(
        self, action: str, key: str = "", value: str = "", value_type: str = ""
    ) -> dict[str, Any]:
        """The REAL verb table against this fake's (empty) kernel registry.

        ``SessionProtocol`` declares code memory for every session shape and the
        desktop route reaches it BY NAME through the bridge's facade, so a double
        without it does not type as a session at all — the drift the declaration
        exists to catch rather than a test-only nuisance.

        The fake owns no interpreter, so the table answers exactly what a real
        session whose runtime has never run a cell answers: observed/absent for a
        read, ``no_kernel`` for a write. Delegating rather than hand-writing that
        envelope keeps ONE copy of the frozen shape in the tree, so the double
        cannot certify a branch the real session does not have.
        """
        from local_operator.session.variable_ops import run_variable_verb

        return await run_variable_verb(
            f"fake-{id(self):x}",
            action,
            key,
            value,
            value_type,
            redact=getattr(getattr(self, "variables", None), "redact", None),
        )

    async def credential_op(self, action: str, key: str = "", value: str = "") -> dict[str, Any]:
        """The REAL verb table against this fake's store, not a stub of it.

        ``SessionProtocol`` declares this verb for every session shape, and
        the TUI's submit seam probes it BY NAME — a double lacking it
        silently degrades every credential gesture driven through it to
        "this session cannot hold credentials", and a fake that swallows the
        verb is how #891 passed four review streams on an unreachable path.
        The canonical delegation rationale lives on
        ``test_app_pilot.FakeSession.credential_op``.
        """
        from local_operator.session.credential_ops import run_credential_verb

        return await run_credential_verb(
            self.variables, getattr(self, "journal_credential_change", None), action, key, value
        )

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

    def queued_steering(self) -> list[Any]:
        return []

    def steer_message(self, message: Any) -> None:
        pass

    def recall_steering(self, message: Any) -> bool:
        return False

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

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

    def running_subagents(self) -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

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

    The CLOSED-layout form: with no sidebar docked the lane IS the screen's
    content box. The docked form, and the box it resolves in, are
    :func:`_panel_width_in` and :func:`_lane_width`.
    """
    return _panel_width_in(terminal_width - 2)  # the screen's one-cell inset each side


def _panel_width_in(box: int) -> int:
    """Width the panel renders at inside a composer lane of ``box`` cells.

    ``min(box, cap, max(floor, proportion))`` when that leaves at least
    ``BOOT_CARD_MIN_INSET`` cells of ground, else the full box: an inset of one to
    three cells is not a card, so the app does not ask for one (see
    ``OperatorApp._sync_boot_card``). The clamp is resolved in the LANE rather
    than in the terminal because that is the box the dock is laid out in and the
    box the sheet resolves the percentage against — the two disagreed only while
    the decision was taken on the terminal's width.
    """
    percent, floor, cap = _clamp()
    card = min(box, cap, max(floor, box * percent // 100))
    return card if box - card >= BOOT_CARD_MIN_INSET else box


def _lane_width(app: OperatorApp, terminal_width: int) -> int:
    """Cells the composer is laid out in: the content box minus a DOCKED sidebar.

    A RESTATEMENT of the app's rule, not an independent check of it — it reads the
    same two inputs ``OperatorApp._boot_lane_width`` reads (the sidebar's resolved
    stylesheet width and the workspace's overlay class) and so cannot disagree with
    that function on its own. What carries the weight in the tests below is the
    assertion that uses this number against a REAL widget — ``lane ==
    dock.width``, read off ``#input-dock``'s own region — which is the layout
    engine's answer, not a second copy of the app's arithmetic. This helper exists
    to state the rule once where the tests can read it, not to be a second opinion.

    The stylesheet width rather than ``#session-sidebar``'s region width, and that
    is deliberate rather than convenient: region is the PREVIOUS frame's geometry
    until layout settles (QA round 1, Q2), which is the stale read the app's own
    comment on this pass rejects. An overlay drawer displaces nothing, so it is
    not subtracted — the one branch the tests' real-region check does pin.
    """
    box = terminal_width - 2
    sidebar = app._session_sidebar
    sidebar_width = sidebar.styles.width
    if sidebar.display and sidebar_width is not None:
        if not app.query_one("#session-workspace").has_class("sidebar-overlay"):
            box = max(0, box - int(sidebar_width.value))
    return box


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


#: Widths the boot-notice column is asserted at. The parametrization is the
#: POINT of these two tests, not thoroughness for its own sake: the offset
#: defect they guard CANCELLED at exactly the width the older single-width test
#: ran at (120). `_sync_boot_column_width` centred the block in `box - gutter`
#: while the stylesheet centres the card in `box`, so the two agreed only where
#: `box - card` happened to be EVEN — and a real invariant was pinned at one of
#: the widths that hides its own violation.
#:
#: Measured on the unfixed tree: one cell of drift at 86/90/100/110 (`box - card`
#: odd), zero at 120/160 (even). Both parities are therefore represented and
#: must stay — dropping the odd ones disarms the guard, dropping the even ones
#: stops it proving the fix did not simply move the error.
#:
#: 86 is the lowest carded width at which the defect APPEARS, which is why the
#: list starts there. The first carded width is 85 (`box=83`, `card=75`,
#: `d=8 == BOOT_CARD_MIN_INSET`; 84 with `d=7` is the last uncarded one), but
#: 85's `box - card` is even, so it sits on the parity that hides the bug.
#: Below the threshold the app deliberately leaves the notice on the full-width
#: spine, where there is no card column to share; a narrower terminal is
#: covered by the spine tests instead, not by these.
BOOT_NOTICE_WIDTHS = (86, 90, 100, 110, 120, 160)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_width", (*BOOT_NOTICE_WIDTHS, 190, 240))
@pytest.mark.parametrize("sidebar_open", (False, True))
async def test_a_notice_under_the_splash_sits_on_the_card_not_the_spine(
    terminal_width: int,
    sidebar_open: bool,
) -> None:
    """A boot notice shares the card's COLUMN, and keeps one left edge in it.

    Two separate claims, both of which were once wrong at some width.

    The BLOCK joins the centred composition: left at `1fr` it drew flush against
    the terminal's left edge while the splash and the card sat centred. So it
    takes the card's width and the card's exact `x` — asserted here at six
    widths rather than one, because the offset that places it used to be
    computed against a different box than the card's own and drifted a cell at
    four of them.

    The composition is scoped to the LANE, and a docked sidebar narrows it. Where
    the lane has no room for a card — a 33-cell drawer leaves 65 of 100 columns —
    the panel is the full-width bar of its lane and there is no card column to
    share: the notice is a spine block, the same degradation a terminal too narrow
    for a card already gets. Both regimes are asserted here; what is never allowed
    in either is a block wider than the screen it is drawn on.

    The TEXT inside it is left-aligned on the hanging indent, NOT centred. Rows
    centred on their own widths made the ink's left edge a function of sentence
    length — a stack of three notices drew four ragged edges, the same "diamond"
    `welcome.py::_center_blocks` rejected for the splash above it. One column,
    landing on the composer's own text column below.
    """
    app = _make_app()
    async with app.run_test(size=(terminal_width, 36)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app._set_sidebar_open(sidebar_open)
        app._system_notice("MCP cloudflare failed: needs authorization", "error")
        await _settle(pilot)
        card = app.query_one("#input-shell").region
        notice = app.query_one(NoticeBlock).region
        # The clamp resolves in the LANE the dock is laid out in, docked or not.
        lane = _lane_width(app, terminal_width)
        assert card.width == _panel_width_in(lane), (terminal_width, lane, card.width)
        if app.screen.has_class(BOOT_CARD_CLASS):
            assert notice.width == card.width, (notice.width, card.width)
            # The BLOCK shares the card's column, not only its width.
            assert notice.x == card.x, (terminal_width, notice.x, card.x)
        else:
            # No card at this lane: the notice is back on the spine, and the one
            # thing that must hold there is that it is drawn on the screen at all
            # (it is `1fr` of the transcript's content box, so nothing else can
            # make it overflow — but the base of this test measured 73 cells of
            # block at x=35 on a 100-cell terminal, right edge 108).
            assert notice.x + notice.width <= terminal_width, (notice, terminal_width)
        # The narrow drawer intentionally covers the transcript, not the dock.
        # Assert its geometry above, but do not mistake occlusion for alignment.
        if app.query_one("#session-workspace").has_class("sidebar-overlay"):
            return
        # And its text starts on ONE column: the glyph field's width in from the
        # block's own left edge, exactly as a spine notice does.
        line = _rows(app)[notice.y]
        span = line[notice.x : notice.x + notice.width]
        left = len(span) - len(span.lstrip())
        assert left == SPINE_INDENT, (terminal_width, left, span)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_width", (*BOOT_NOTICE_WIDTHS, 190, 240))
@pytest.mark.parametrize("sidebar_open", (False, True))
async def test_a_boot_notice_starts_on_the_composers_own_text_column(
    terminal_width: int,
    sidebar_open: bool,
) -> None:
    """The notice's sentence begins where the user's typing begins.

    This is the invariant that makes the notice read as part of the composition
    rather than as a stray fragment above it, and nothing pinned it before: the
    block's placement and the composer's placement are computed in different
    places (`_sync_boot_column_width` against the card, the composer by the
    stylesheet), so they can drift apart without either looking wrong alone.

    Scoped to the CARDED lane, which is where the claim has meaning: the card's
    own column and the composer's text column are the same column by
    construction. With the card withheld — a docked sidebar leaves 65 of 100
    columns, under the 75-cell floor the card would need — the notice is a spine
    block, and a spine block's indent sits one cell right of the composer's text
    column. That difference is pre-existing and the app's narrow-terminal shape
    (measured on the base at 70, 80 and 84 columns with no sidebar: sentence at
    6, composer text at 5), so it is asserted as "stays on the screen" rather
    than as column equality.

    Measured against the editor's CONTENT box rather than `chevron.x + 2`: the
    editor carries its own one-cell left padding, so the chevron-relative form
    is off by one and would pin the wrong column.
    """
    app = _make_app()
    async with app.run_test(size=(terminal_width, 36)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app._set_sidebar_open(sidebar_open)
        app._system_notice(
            "this session is running 0.51.0@ad6db35 \u2192 0.51.5@ad6db35 \u2014 it will "
            "switch to the new version when it is next idle.",
            "note",
        )
        await _settle(pilot)
        notice = app.query_one(NoticeBlock).region
        sentence_x = notice.x + NoticeBlock.GLYPH_COLS
        if not app.screen.has_class(BOOT_CARD_CLASS):
            assert notice.x + notice.width <= terminal_width, (notice, terminal_width)
            return
        composer_text_x = app.query_one(Editor).content_region.x
        assert sentence_x == composer_text_x, (
            terminal_width,
            sentence_x,
            composer_text_x,
        )
        if app.query_one("#session-workspace").has_class("sidebar-overlay"):
            return
        # Every wrapped continuation lands on that same column — the property a
        # per-row centre destroyed, where a 4-word orphan floated 34 cells right
        # of its own first row at 160 columns.
        rows = _rows(app)
        for row_index in range(notice.y + 1, notice.y + notice.height):
            span = rows[row_index][notice.x : notice.x + notice.width]
            if not span.strip():
                continue
            assert notice.x + (len(span) - len(span.lstrip())) == sentence_x, span


@pytest.mark.asyncio
async def test_boot_notice_tracks_sidebar_toggles_and_resizes(
    monkeypatch,
) -> None:
    """Sidebar toggles do not paint a stale column; resizing keeps the same lane.

    Sample every compositor frame on drawer toggles, rather than accepting a
    fix that only converges after paint. Resize checks cover the settled layout
    across the overlay boundary and the existing composer's width floor.
    """
    app = _make_app()
    async with app.run_test(size=(190, 36)) as pilot:
        await _settle(pilot)
        app._system_notice("this session is running an older version", "note")
        await _settle(pilot)
        block = app.query_one(NoticeBlock)
        frames: list[tuple[int, int, int, int, bool]] = []
        painted = Screen._compositor_refresh

        def record(screen: "Screen[object]") -> None:
            painted(screen)
            notice = block.region
            card = app.query_one("#input-shell").region
            frames.append(
                (
                    notice.x,
                    notice.width,
                    card.x,
                    card.width,
                    app.screen.has_class(BOOT_CARD_CLASS),
                )
            )

        def consistent(frame: tuple[int, int, int, int, bool], width: int) -> bool:
            """Is this painted frame the column the live layout asks for?

            Carded, the notice takes the card's column exactly. With the card
            withheld (a docked drawer leaving under the floor) it is a spine
            block, and the claim that survives is the one the base violated: the
            block is drawn ON the screen. Sampled per frame rather than only on
            the settled one, because a stale column is a wrong frame even when the
            next paint corrects it.
            """
            nx, nw, cx, cw, carded = frame
            if carded:
                return nx == cx and nw == cw
            return nx + nw <= width

        monkeypatch.setattr(Screen, "_compositor_refresh", record)
        previous_width = 190
        for width, sidebar_open in (
            (190, True),
            (190, False),
            (190, True),
            (240, True),
            (100, True),
            (90, True),
            (120, True),
            (120, False),
            (190, False),
        ):
            frames.clear()
            app._set_sidebar_open(sidebar_open)
            await pilot.resize_terminal(width, 36)
            await _settle(pilot)
            assert frames, "premise: the changed layout painted"
            checked = frames if width == previous_width else frames[-2:]
            assert all(consistent(frame, width) for frame in checked), checked
            previous_width = width


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((100, 30), (160, 40)))
@pytest.mark.parametrize("conversation_started", (False, True))
@pytest.mark.parametrize("sidebar_open", (False, True))
async def test_the_docked_composer_band_is_measured_against_its_lane(
    size: tuple[int, int],
    conversation_started: bool,
    sidebar_open: bool,
) -> None:
    """The band's box is the width the shell HAS, docked or not, cold or live.

    The defect this pins, measured on the base at 100x30 with the drawer docked on
    an EMPTY session: `#input-dock` resized (98 -> 65) while `#input-shell` did
    not, so the shell stayed 75 cells wide at x=34 — right edge 109 on a 100-cell
    terminal — and `#status-band`, which is the shell's own child, inherited the
    same phantom box: region `[35, 26, 73, 2]`, a 72-cell box where 62 remained.
    The row was painted clipped at the screen edge with no ellipsis, on the one
    row the reconnection sentence lives on.

    Root cause, and why the numbers are asserted rather than the ink: the app
    decided the boot card from the TERMINAL's width while the sheet resolved the
    clamp against the DOCK's content box, and `min-width: 75` is an absolute floor.
    A docked drawer therefore made the two disagree by 33 cells, and nothing
    text-only could see it — the band still produced a perfectly good row for the
    box it believed it had.

    Both conversation states are asserted against ONE expression, which is the
    second half of the defect: a harness that measures the band on an empty
    session used to read a 72-cell box where a seeded one read 62 (the drawer
    leaves the same 65-cell lane either way).
    """
    terminal_width, _height = size
    app = _make_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _settle(pilot)
        if conversation_started:
            app._append_block(UserBlock("hello"))
            await _settle(pilot)
        app._set_sidebar_open(sidebar_open)
        await _settle(pilot)

        dock = app.query_one("#input-dock").region
        shell = app.query_one("#input-shell").region
        band_widget = app.query_one("#status-band")
        band = band_widget.region

        # The lane the app measured IS the lane the dock got, and — while the boot
        # card is up — the clamp resolves inside it. That agreement is the fix:
        # the two disagreed on the base whenever a drawer was docked.
        lane = _lane_width(app, terminal_width)
        assert lane == dock.width, (lane, dock.width)
        if app.screen.has_class(BOOT_CARD_CLASS):
            assert shell.width == _panel_width_in(lane), (shell.width, lane)
        else:
            assert shell.width == lane, (
                "no card: the panel is the full width of its lane",
                shell.width,
                lane,
            )

        # The shell never leaves its own lane, in either direction.
        assert shell.x >= dock.x, (shell, dock)
        assert shell.x + shell.width <= dock.x + dock.width, (shell, dock)

        # The band is the shell's child, one cell of `#input-shell {padding: 1}`
        # in from each edge, and one more cell of its own right padding. Its
        # region is what the fit ladder measured against; its content box is the
        # width the row is drawn in — 62 on the docked 100x30 frame, the number
        # the base reported as 72.
        assert band.width == shell.width - 2, (band, shell)
        assert band_widget.content_region.width == shell.width - 3, (
            band_widget.content_region,
            shell,
        )
        assert band.x + band.width <= terminal_width, (band, terminal_width)

        # And the row the band holds LANDS IN ITS LANE, which is the sufficient
        # form of the claim this geometry exists to make, and the one the symptom
        # needs: the band right-aligns the name's ink to the right edge of the box
        # it was fitted to, so a box that runs past the LANE edge is cropped
        # mid-word by the compositor — on the base, `... retry    Fix sidebar
        # reconnec…` fitted to a 72-cell box at x=35 could only land 65 of its
        # cells, and the painted row read `... retry    Fix sidebar  ` with no
        # ellipsis anywhere. The bound is the LANE and not the screen edge on
        # purpose (QA round 1, Q1): with the drawer docked on the right, every
        # cell of that row was inside the 100-cell terminal — `content.x + held <=
        # terminal_width` is SATISFIED by a row that is already cropped — because
        # the crop happens at the main lane, not at the screen.
        status = app._status
        assert status is not None
        held = cell_len(status.render_text(band_widget.content_region.width).plain.rstrip())
        assert band_widget.content_region.x + held <= dock.x + dock.width, (
            band_widget.content_region,
            held,
            dock,
        )


@pytest.mark.asyncio
async def test_a_docked_cold_session_paints_the_connection_row_whole() -> None:
    """The reported frame: 100x30, drawer docked, empty session, failed reconnect.

    `#status-band` is a child of `#input-shell`, so the phantom box of the docked
    boot card reached the connection row itself — the one row in the app that
    carries "Reconnect failed · Select again to retry". The base fitted that row
    to 72 cells inside a form 65 cells wide, at x=35 on a 100-cell terminal: right
    edge 108, and the name cut mid-word at the screen edge with no ellipsis.

    Asserted on the numbers, because the text-only view of this surface was green
    throughout: the band produced a perfectly well-formed row for the box it
    believed it had. The text is asserted too, but as the user-visible claim it is
    — the sentence intact on the row the terminal is sent.
    """
    session = FakeSession()
    session.set_conversation_name(NAME)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app._sidebar_settings = SidebarSettings(False, "left")
        await pilot.press("ctrl+b")
        await _settle(pilot)
        source = SessionInteraction(session)
        source.display_only = True
        source.connection_error = "the runtime is not responding"
        app._interaction = source
        app._interactions[id(session)] = source
        status = app._status
        assert status is not None
        status.update(conversation_name=NAME)
        app._show_sidebar_connection(source)
        await _settle(pilot)

        band_widget = app.query_one("#status-band")
        band = band_widget.region
        content = band_widget.content_region
        lane = _lane_width(app, 100)
        assert lane == 65, ("premise: the docked drawer leaves the lane this narrow", lane)
        assert app.query_one("#input-shell").region.width == _panel_width_in(lane) == lane
        assert band.width == lane - 2, band
        assert content.width == lane - 3, content
        assert band.x + band.width <= 100, band

        # The row the band holds for its own box must land inside its LANE, not
        # merely inside the screen — 35 + 72 cells did not fit the lane on the
        # base, which is the whole defect, and a screen-edge bound is satisfied by
        # exactly that row whenever the drawer is docked on the right (QA round 1,
        # Q1). `#input-dock` is the lane's own widget, so this is the bound the
        # compositor enforces.
        dock = app.query_one("#input-dock").region
        held = status.render_text(content.width).plain
        ink_right = content.x + cell_len(held.rstrip())
        assert ink_right <= dock.x + dock.width, (content, held, ink_right, dock)

        # And the row the terminal is actually sent still carries the sentence:
        # this is the surface the session has been fixing, and it must not be
        # traded for a geometry that merely measures well.
        painted = _rows(app)[content.y][content.x : content.x + content.width]
        assert "Reconnect failed · Select again to retry" in painted, painted


@pytest.mark.asyncio
async def test_the_boot_column_follows_the_card_not_the_moment_it_was_written() -> None:
    """The centring class is reconciled against the LIVE card, not fixed at creation.

    The card threshold is dynamic, so a class set once at append goes stale twice:
    a notice created with the card up stays centred after a resize drops the card
    (centred text in a full-width block, not the spine), and the splash retiring
    removes the card for good with no resize to reconcile it. The fix makes the
    class follow the card on every boot-layout sync. Asserted on the class and
    the drawn row, the two halves of the staleness.
    """
    app = _make_app()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        app._system_notice("MCP cloudflare failed: needs authorization", "error")
        await _settle(pilot)
        notice_block = app.query_one(NoticeBlock)
        assert notice_block.has_class(BOOT_COLUMN_CLASS)

        # The conversation starting retires the splash and the card with it.
        app._append_block(UserBlock("hello"))
        await _settle(pilot)
        assert not app.screen.has_class(BOOT_LAYOUT_CLASS)
        assert not app.screen.has_class(BOOT_CARD_CLASS)
        assert not notice_block.has_class(BOOT_COLUMN_CLASS)
        # The width written for the card goes WITH it: left set, the notice would
        # stay boot-card narrow (82) with dead space to its right for the
        # session. It reflows to the full transcript — far wider than the card.
        await pilot.pause()
        assert notice_block.region.width > _expected_card_width(120), notice_block.region.width
        # And a resize while the transcript is populated does not bring the card
        # — or the centring, or the narrow width — back: a wide terminal with
        # content is a bar.
        await pilot.resize_terminal(140, 40)
        await _settle(pilot)
        assert not app.screen.has_class(BOOT_CARD_CLASS)
        assert not notice_block.has_class(BOOT_COLUMN_CLASS)
        assert notice_block.region.width > _expected_card_width(140), notice_block.region.width


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


@pytest.mark.asyncio
async def test_a_late_splash_resize_does_not_crash_a_torn_down_app() -> None:
    """Quitting during boot must not raise out of the message pump.

    The splash resizes asynchronously: ``WelcomeView._poll`` posts
    ``BlockResized`` once the model label is known, and the model label lands
    when the session factory resolves. A quit that beats the factory home
    therefore leaves that message queued against a screen stack that is already
    gone, and every boot-layout pass reaches ``self.screen``, which RAISES
    ``ScreenStackError`` on an empty stack instead of returning ``None``.

    Observed as a real crash on exit, from the pump rather than from anything
    the user did: ``on_welcome_view_block_resized`` -> ``_sync_boot_layout`` ->
    ``_sync_boot_card`` -> ``ScreenStackError: No screens on stack``.

    Driven by emptying the stack and dispatching the handler exactly as the
    pump would, because reproducing the wall-clock ordering by racing a real
    quit against a real poll is the coin flip that hid the bug in the first
    place. The state is what matters, and the state is "this message arrived
    with no screens left".
    """
    app = _make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        screen = app.screen
        card_up_before = screen.has_class(BOOT_CARD_CLASS)

        # Exactly the teardown state: a BlockResized is in the pump's hand and
        # the screen stack has already been emptied.
        stack = app._screen_stacks[app._current_mode]
        stack.clear()
        assert not app.screen_stack, "the race under test needs an empty stack"

        app.on_welcome_view_block_resized(WelcomeView.BlockResized())

        # The guard skips the pass; it does not swallow a failure. Put the
        # screen back and the identical message reconciles the layout as
        # before, so what was added is a teardown check and not a mute.
        stack.append(screen)
        app.on_welcome_view_block_resized(WelcomeView.BlockResized())
        assert app.screen.has_class(BOOT_CARD_CLASS) == card_up_before


# -- a live prompt stands the card down (#168) --------------------------------


async def _raise_ask(app: OperatorApp, pilot) -> asyncio.Task:  # type: ignore[no-untyped-def]
    """Park a real `ask` on the dock and wait for the card to lay out."""
    question = AskQuestion(
        id="rollout",
        question="Which rollout should the stale-row migration take?",
        options=[
            AskOption(label="Drop the rows", description="nothing reads the column"),
            AskOption(label="Backfill from the audit log", description="keeps history"),
        ],
        recommended=1,
    )
    task = asyncio.create_task(app.request_user_choice([question]))
    for _ in range(10):
        await pilot.pause()
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 40), (120, 36), (100, 30)])
async def test_a_live_prompt_is_never_wider_than_the_composer_it_is_docked_to(
    size: tuple[int, int],
) -> None:
    """#168: the clamp reached ``#input-shell`` and not the prompt above it.

    ``AskPickerScreen`` is ``width: 1fr`` and has been since the picker moved
    into the dock, so a boot layout restored under a live question clamped the
    composer to the card while the question kept the full width. Measured on the
    pre-fix tree at 160 columns: ``card=156 shell=98``, the question hanging 58
    cells past the input it belongs to.

    Asserted against the COMPOSER rather than against a width constant: the
    property is that the question and the input it is docked to share an edge,
    whatever the clamp resolves to at this size.
    """
    width, _ = size
    app = _make_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _settle(pilot)
        assert app.screen.has_class(BOOT_CARD_CLASS), "this size must get a card at all"

        task = await _raise_ask(app, pilot)
        shell = app.query_one("#input-shell")
        card = app.query_one(AskPickerScreen)

        # Ctrl+L is the confirmed path, but the bug is reachable through any
        # call that restores the splash under a mounted prompt.
        await pilot.press("ctrl+l")
        for _ in range(10):
            await pilot.pause()

        assert app.screen.has_class(BOOT_LAYOUT_CLASS), "the splash is back"
        assert card.region.right == shell.region.right, (card.region, shell.region)
        assert card.region.width == shell.region.width, (card.region, shell.region)
        # And nothing gained the ability to overflow the terminal doing it.
        for index, row in enumerate(_rows(app)):
            assert cell_len(row) <= width, (index, repr(row))

        task.cancel()
        with contextlib.suppress(BaseException):
            await task


@pytest.mark.asyncio
async def test_the_boot_card_comes_back_once_the_question_is_answered() -> None:
    """The stand-down is only safe if it is temporary.

    A card suppressed while a prompt is live and never restored would be a
    permanently full-width boot layout — the splash composition silently lost to
    the first `ask` of the session. The condition is re-derived from the host on
    every pass (``_prompt_is_live``) precisely so it cannot latch.
    """
    app = _make_app()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        clamped = app.query_one("#input-shell").region.width
        assert app.screen.has_class(BOOT_CARD_CLASS)

        task = await _raise_ask(app, pilot)
        assert not app.screen.has_class(BOOT_CARD_CLASS), "a live question stands it down"
        assert app.query_one("#input-shell").region.width > clamped

        await pilot.press("enter")
        for _ in range(12):
            await pilot.pause()

        assert app.screen.has_class(BOOT_CARD_CLASS), "answered, so the composition returns"
        assert app.query_one("#input-shell").region.width == clamped
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, timeout=2)
