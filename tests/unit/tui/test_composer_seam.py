"""The seam between the conversation and the composer.

One row of ground, and it is the TRANSCRIPT's row: the sheet gives
``TranscriptView`` a bottom padding cell the way it already gives it a top one,
so the conversation's last line never rests on the input panel's fill. Reported
from the field with the last two rows of a turn (``✕ name 'CompactionOutcome'
is not defined`` / ``✕ interrupted``) sitting directly on the composer, which
made the ledger and the input read as one slab.

These tests are about a ONE-ROW difference, so they measure the composed frame
rather than a style value wherever they can: a padding declaration proves the
rule was written, not that the row reached the terminal. Each one pins a state
the app actually reaches, because the way this regresses is a fix that is right
in one frame:

- every last-block kind (prose, tool row, notice, prompt, working line), since
  they carry different ``SPACING_KIND``s and only the LAST one meets the dock
- a conversation grown one block at a time into the seam, where a row that
  appears and disappears would be worse than no row
- scrolled to the end, which is where a padding row outside the scrollable
  region would be clipped away
- the ``/btw`` aside, whose card is deliberately FLUSH on the composer, and the
  frame after Esc, where the row has to come back
- the boot splash, which owns its own vertical composition and must not gain a
  stray row
- the dock band up, where the transcript's row is the band's air too and the
  join must not double
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.aside_panel import AsidePanel
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
from local_operator.tui.widgets.welcome import WelcomeView
from tests.unit.tui.test_aside import AsideSession
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The two frames the seam was measured in: the everyday wide terminal and the
#: narrow one where the conversation reaches the dock soonest.
SIZES = [(120, 40), (60, 40)]

PROSE = (
    "Rebuilt the parser and re-ran the suite; both tails are green now, and the "
    "compaction cutpoint no longer walks off the end of the ledger."
)


def _frame(app: OperatorApp) -> list[str]:
    """The composed frame as plain text, one entry per terminal row."""
    return [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]


def _blank_rows_above(app: OperatorApp, selector: str) -> int:
    """Blank rows immediately above ``selector``'s top edge on the FRAME.

    The measurement the user makes with their eyes, and the only one that
    survives a rule that is declared but never painted.
    """
    frame = _frame(app)
    index = app.query_one(selector).region.y - 1
    count = 0
    while index >= 0 and not frame[index].strip():
        count += 1
        index -= 1
    return count


def _seam(app: OperatorApp) -> int:
    """Ground rows between the conversation and the composer panel."""
    return _blank_rows_above(app, "#input-shell")


def _last_painted_row(app: OperatorApp, above: str = "#input-shell") -> str:
    """The last row with ink above ``above`` — the conversation's last line."""
    frame = _frame(app)
    index = app.query_one(above).region.y - 1
    while index >= 0 and not frame[index].strip():
        index -= 1
    return frame[index] if index >= 0 else ""


def _prose(text: str = PROSE) -> AssistantBlock:
    block = AssistantBlock()
    block.update_text(text)
    return block


def _card(name: str = "bash", error: bool = False) -> ToolCard:
    card = ToolCard(f"c-{name}", name, {"command": "pytest tests/unit -q"})
    if error:
        card.mark_failed("permission denied")
    else:
        card.mark_done("66 passed")
    return card


#: The last block of the conversation, per kind, with the ink its row carries.
#: Every one of these has a different ``SPACING_KIND``, which is what decides
#: the gaps BETWEEN blocks — and is deliberately not what decides this one.
LAST_BLOCK: dict[str, tuple[Any, str]] = {
    "prose": (lambda: _prose("Both tails are green now — the cutpoint holds."), "cutpoint holds"),
    "tool": (lambda: _card("grep", error=True), "grep"),
    "notice": (lambda: NoticeBlock("interrupted", "warning"), "interrupted"),
    "user": (lambda: UserBlock("now do the other one"), "now do the other one"),
}


async def _settle(pilot: Any, ticks: int = 4) -> None:
    for _ in range(ticks):
        await pilot.pause()


async def _fill(pilot: Any, app: OperatorApp, turns: int) -> None:
    """A conversation tall enough to reach the dock, in the app's own shapes."""
    app.query_one(Editor).cursor_blink = False
    await _settle(pilot)
    for n in range(turns):
        app._append_block(UserBlock(f"turn {n}: port the loop to the new harness"))
        app._append_block(_prose())
        app._append_block(_card(f"bash{n}"))
    await _settle(pilot)


def _app(session: Any | None = None) -> tuple[OperatorApp, Any]:
    session = session if session is not None else FakeSession()
    return OperatorApp(lambda: _factory(session)), session


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("last", sorted(LAST_BLOCK))
async def test_one_row_of_ground_under_every_kind_of_last_block(
    size: tuple[int, int], last: str
) -> None:
    """Whatever ended the turn, it does not rest on the composer.

    The owner named prose, tool calls and thinking; the kinds differ in
    ``SPACING_KIND``, in height and in whether they carry their own fill (a
    failed tool row is a tinted slab, and flush it read as the composer's own
    top border), so each is measured on the frame rather than argued about.
    """
    build, ink = LAST_BLOCK[last]
    app, _session = _app()
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        app._append_block(build())
        await _settle(pilot)
        transcript = app.query_one(TranscriptView)

        assert transcript.max_scroll_y > 0, "the case that matters is a scrollable transcript"
        assert transcript.scroll_offset.y >= transcript.max_scroll_y, "an append lands at the end"
        assert _seam(app) == 1, _frame(app)[-8:]
        assert ink in _last_painted_row(app), _frame(app)[-8:]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_the_working_line_keeps_the_row_while_the_agent_runs(
    size: tuple[int, int],
) -> None:
    """The common case while a turn is live: the pinned transient line is last.

    It is the one block that is lifted again at turn end, so a seam that came
    from the BLOCK rather than from the container would blink here.
    """
    app, _session = _app()
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        app._start_working_block()
        await _settle(pilot)
        assert _seam(app) == 1, _frame(app)[-8:]

        app._dismiss_working_block()
        await _settle(pilot)
        assert _seam(app) == 1, _frame(app)[-8:]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_the_row_is_stable_as_content_grows_into_it(size: tuple[int, int]) -> None:
    """Consecutive frames, one block at a time, through the moment the
    conversation stops fitting.

    Two failures at once. A row that only exists while there is slack goes to
    zero the moment content fills the frame — the reported bug. A row outside
    the scrollable region is clipped once the transcript scrolls, which is the
    same bug arrived at from the other side: ``scroll_end`` would land the last
    CONTENT row against the dock. Textual keeps a container's padding inside
    its scrollable region, and this is the test that says so.
    """
    app, _session = _app()
    async with app.run_test(size=size) as pilot:
        app.query_one(Editor).cursor_blink = False
        await _settle(pilot)
        transcript = app.query_one(TranscriptView)
        seams: list[tuple[int, int, int]] = []
        for n in range(size[1]):
            app._append_block(NoticeBlock(f"row {n:02d}", "info"))
            await _settle(pilot, 2)
            seams.append((n, transcript.max_scroll_y, _seam(app)))

        # Before the frame is full the seam is slack and shrinking; from the
        # first overflowing frame on it is exactly one row, on every frame.
        overflowing = [(n, scroll, seam) for n, scroll, seam in seams if scroll > 0]
        assert overflowing, seams
        assert [seam for _n, _scroll, seam in overflowing] == [1] * len(overflowing), seams
        # And it never collapses at any point in the sequence, full or not.
        assert min(seam for _n, _scroll, seam in seams) == 1, seams


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_the_aside_card_still_rests_flush_on_the_composer(size: tuple[int, int]) -> None:
    """The one join in the stack that is deliberately tight stays tight.

    ``AsidePanel`` has no bottom padding row, and that was measured: with one,
    the card-to-composer seam was the widest interval on screen. The card and
    the composer are two halves of one unit, so the transcript's trailing row
    must not reappear between them — which is why the aside's reservation
    REPLACES that padding instead of adding to it.
    """
    app, _session = _app(AsideSession())
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        app.query_one(Editor).load_text("/btw why is the loop slow?")
        await pilot.press("enter")
        await _settle(pilot, 8)

        panel = app.query_one(AsidePanel)
        assert panel.is_open
        transcript = app.query_one(TranscriptView)
        assert _seam(app) == 0, _frame(app)[-10:]
        assert "esc" in _last_painted_row(app), "the card's own footer is the row above the input"
        # The card's rows are reserved out of the transcript, so the reservation
        # is the card's height and not the sheet's one row.
        assert transcript.styles.padding.bottom == panel.region.height


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_closing_the_aside_hands_the_row_back(size: tuple[int, int]) -> None:
    """Esc returns the seam to one row, from the SHEET rather than a constant.

    ``_close_aside`` used to write ``0`` back, which was invisible while the
    sheet said 0 too and ate the row the moment it said 1.
    """
    app, _session = _app(AsideSession())
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        app._append_block(NoticeBlock("interrupted", "warning"))
        await _settle(pilot)
        app.query_one(Editor).load_text("/btw why is the loop slow?")
        await pilot.press("enter")
        await _settle(pilot, 8)
        await pilot.press("escape")
        await _settle(pilot, 8)

        assert not app.query_one(AsidePanel).is_open
        assert app.query_one(TranscriptView).styles.padding.bottom == 1
        assert _seam(app) == 1, _frame(app)[-8:]
        assert "interrupted" in _last_painted_row(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [*SIZES, (80, 24), (60, 20)])
async def test_the_boot_splash_gains_no_row(size: tuple[int, int]) -> None:
    """The splash rests ON the card, and the seam rule must not lift it.

    The boot layout owns its whole vertical composition — the splash is
    bottom-aligned against the docked card and the app reserves the slack below
    it — so a trailing row from the transcript could only ever be a row the
    splash's own degradation ladder has to pay for. ``Screen.boot
    TranscriptView`` takes it back, and this pins BOTH halves: the declared
    padding (0) and the painted frame.

    ONE row above the panel, at every size, and it is the splash's own: measured
    on the frame before this rule existed at 60x20, 80x24, 60x40 and 120x40, all
    1. That is why the number is asserted exactly rather than as an upper bound —
    an upper bound would pass just as happily if the boot layout lost its row.
    """
    app, _session = _app()
    async with app.run_test(size=size) as pilot:
        app.query_one(Editor).cursor_blink = False
        await _settle(pilot, 14)

        assert app.screen.has_class("boot")
        assert app.query_one(WelcomeView).display
        assert app.query_one(TranscriptView).styles.padding.bottom == 0
        assert _seam(app) == 1, _frame(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_the_band_join_does_not_double(size: tuple[int, int]) -> None:
    """With a panel in the dock band, every join in the column is one row.

    The band's slot used to carry its own row ABOVE itself, which stacked with
    the transcript's trailing row into a two-row interval — the widest on
    screen, and the exact "both halves pushed a padding row into the join"
    failure the aside card's missing bottom row was fixed for. Each slot now
    owns the ground BELOW it instead, so the transcript's row is the band's air
    and the band's row is the composer's.
    """
    from local_operator.tools import builtin

    app, session = _app()
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        app._append_block(NoticeBlock("interrupted", "warning"))
        await _settle(pilot)
        builtin.TODO_STORE[session.session_id] = [
            {"text": "wire the band", "status": "done"},
            {"text": "capture frames", "status": "pending"},
        ]
        app._refresh_band()
        await _settle(pilot, 6)
        transcript = app.query_one(TranscriptView)
        transcript.scroll_end(animate=False)
        await _settle(pilot, 6)

        try:
            assert app.query_one("#todo-panel").display, "the band has to be up to measure it"
            assert _blank_rows_above(app, "#todo-body") == 1, _frame(app)[-12:]
            assert _seam(app) == 1, _frame(app)[-12:]
        finally:
            builtin.TODO_STORE.pop(session.session_id, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_the_subagent_page_ends_in_ground_too(size: tuple[int, int]) -> None:
    """The page takes the transcript's REGION, so it inherits its contract.

    Its last row is the exit hint, and flush on the composer the two chrome
    lines fused: `esc back to conversation · read-only` directly above `❯
    Read-only — press esc to reply` read as one four-row panel rather than as a
    page over an input. The row cannot come from the body's own
    ``TranscriptView`` padding — the hint is BELOW the body — so it is the
    page container's, which is the same "one owner per join" rule the dock
    column follows.
    """
    from tests.unit.tui.test_subagent_view import (
        TRAJECTORY,
        FakeSession as PageSession,
        _async_factory,
        _job_with,
        _open,
    )

    app = OperatorApp(_async_factory(PageSession()))
    async with app.run_test(size=size) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        await _settle(pilot)

        assert view.is_mounted
        assert _seam(app) == 1, _frame(app)[-8:]
        assert "esc" in _last_painted_row(app), _frame(app)[-8:]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_the_aside_rests_on_whatever_the_dock_puts_first(size: tuple[int, int]) -> None:
    """With a band up the card's flush seam is against the BAND, on purpose.

    The card is placed on the dock (``overlay.stack_on_dock``, gap 0), so what
    it rests on is whatever the dock puts at its top — the composer when the
    band is empty, the band's slab when it is not. Pinned as a deliberate 0
    because it LOOKS like the two-row-join bug this suite is otherwise about,
    and it is the opposite: before the band's padding row moved below its slot,
    this was the one state where the card had a stray row under it, so the same
    card read as seated on the composer and loose above the band.

    What makes 0 legible here is the elevation step rather than a gap — the card
    is one background step off the band, which is this kit's separator — so the
    fills are asserted too. Flush against the SAME fill would be the real
    failure (that is why the band's own join with the composer is a row).
    """
    from local_operator.tools import builtin

    app, session = _app(AsideSession())
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        builtin.TODO_STORE[session.session_id] = [{"text": "wire the band", "status": "pending"}]
        app._refresh_band()
        await _settle(pilot, 6)
        app.query_one(Editor).load_text("/btw why is the loop slow?")
        await pilot.press("enter")
        await _settle(pilot, 8)

        try:
            card = app.query_one(AsidePanel)
            assert card.is_open
            assert app.query_one("#todo-panel").display
            # The card rests ON the band, and the band still ends in its own row.
            assert _blank_rows_above(app, "#todo-panel") == 0, _frame(app)[-12:]
            assert _seam(app) == 1, _frame(app)[-12:]
            # And the seam is readable because the fills differ by a step.
            assert card.styles.background != app.query_one("#todo-body").styles.background
        finally:
            builtin.TODO_STORE.pop(session.session_id, None)
