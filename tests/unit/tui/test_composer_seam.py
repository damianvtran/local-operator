"""The seam between the conversation and the composer, and the dock's own fill.

One row of ground, and it is the TRANSCRIPT's row: the sheet gives
``TranscriptView`` a bottom padding cell the way it already gives it a top one,
so the conversation's last line never rests on the input panel's fill. Reported
from the field with the last two rows of a turn (``✕ name 'CompactionOutcome'
is not defined`` / ``✕ interrupted``) sitting directly on the composer, which
made the ledger and the input read as one slab.

The rule that row belongs to has two halves, and the second was reported from
the field later ("there's a gap between the list and the composer, there should
be a solid fill in between"): GROUND SEPARATES THE CONVERSATION FROM THE DOCK,
NOT THE DOCK'S PANELS FROM THE COMPOSER. A docked panel is chrome; the blank
row under it is the dock's ``$lo-surface``, not the screen's ``$lo-bg``. Both
halves are pinned here, because a fix for either one alone is a regression of
the other — fill everything and the conversation rests on the input; fill
nothing and the dock is cut in three.

These tests are about a ONE-ROW difference, so they measure the composed frame
rather than a style value wherever they can: a padding declaration proves the
rule was written, not that the row reached the terminal, and a blank row proves
nothing at all about whose surface it is (see ``_fill_at``). Each one pins a
state the app actually reaches, because the way this regresses is a fix that is
right in one frame:

- every last-block kind (prose, tool row, notice, prompt, working line), since
  they carry different ``SPACING_KIND``s and only the LAST one meets the dock
- a conversation grown one block at a time into the seam, where a row that
  appears and disappears would be worse than no row
- scrolled to the end, which is where a padding row outside the scrollable
  region would be clipped away
- the ``/btw`` aside, whose card is deliberately FLUSH on the composer, and the
  frame after Esc, where the row has to come back
- the boot splash, which owns its own vertical composition and must not gain a
  stray row — and whose band is zero rows, which is what makes the dock fill
  safe to declare unconditionally
- the dock band up, where the transcript's row is the band's air too and the
  join must not double, with every slot the band can hold (todo, subagent,
  both) driven through the fill rule so one cannot fuse while another floats
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
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_aside import AsideSession

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


def _fill_at(app: OperatorApp, y: int, x: int) -> str:
    """The background actually PAINTED at one cell, as ``#rrggbb``.

    The counterpart to ``_blank_rows_above``: that one answers "is this row
    empty", this one answers "whose surface is it". A blank row proves nothing
    on its own — the whole dock seam bug was a row that was correctly blank and
    wrongly dark — and a ``styles.background`` read proves the declaration, not
    the pixel, so this walks the composed strip the way the terminal does.
    """
    strip = app.screen._compositor.render_strips()[y]
    cursor = 0
    for segment in strip:
        cursor += len(segment.text)
        if cursor > x:
            style = segment.style
            if style is None or style.bgcolor is None:
                return "none"
            return style.bgcolor.get_truecolor().hex.lower()
    return "none"


def _ink_column(app: OperatorApp, y: int) -> int:
    """Column of the first painted glyph on row ``y``, or -1 if the row is blank.

    The dock has ONE left rail and every glyph in it lands on that column; this
    is how a row that missed it is caught, since a widget one cell out still
    measures a plausible region and still fills correctly.
    """
    row = _frame(app)[y]
    stripped = row.lstrip()
    return len(row) - len(stripped) if stripped else -1


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


def _app(session: FakeSession | None = None) -> tuple[OperatorApp, FakeSession]:
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

    This asserted a ZERO-blank seam and now asserts one, so it owes the
    original decision an argument. The measurement behind the zero was real:
    with a bottom padding row the hint-to-``❯`` interval is 2 rows, against 1
    everywhere else in the stack. What it got wrong is where that interval
    LIVES. The fill changes between those two rows — the card contributes its
    row in ``$lo-overlay``, the composer contributes its own in ``$lo-surface``
    — so neither SURFACE holds a 2-row interval, and a surface is the unit a
    reader groups by. What the zero cost, measured at 120x40: the row telling
    the user how to leave was the only row in the card with no air under it,
    and the card's fill terminated on an inked line.

    So "flush" is unchanged, and is now asserted where it is actually claimed —
    on the REGIONS. The card's bottom edge is still the composer's top edge,
    with none of the transcript between them; the blank row the frame shows is
    therefore the card's own last row, not the transcript's returning one.

    The elevation step is what makes that row affordable, which is why the
    fills are asserted here and not only in the band test: flatten the card
    onto the composer's fill and the two padding rows fall inside ONE fill and
    become a real hole. This test is the guard on that coupling.
    """
    app, _session = _app(AsideSession())
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        app.query_one(Editor).load_text("/btw why is the loop slow?")
        await pilot.press("enter")
        await _settle(pilot, 8)

        panel = app.query_one(AsidePanel)
        assert panel.is_open
        shell = app.query_one("#input-shell")
        # Flush, on the regions: no transcript ground survives between them, so
        # the single blank row below is interior to the card.
        assert panel.region.bottom == shell.region.y, (panel.region, shell.region)
        assert _seam(app) == 1, _frame(app)[-10:]
        assert "esc" in _last_painted_row(app), "the card's own footer is the row above the input"
        # The row is only affordable because the two surfaces differ.
        assert panel.styles.background != shell.styles.background


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
    from tests.unit.tui.test_subagent_view import TRAJECTORY
    from tests.unit.tui.test_subagent_view import FakeSession as PageSession
    from tests.unit.tui.test_subagent_view import _async_factory, _job_with, _open

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
    band is empty, the band's slab when it is not. Asserted on the REGIONS,
    which is where that claim lives; the frame's blank row is the card's own
    bottom padding and is interior to it, exactly as it is over the composer.

    This is also the state design finding D1 called a blocker, reading the card
    as "floating five rows above the composer". The geometry was never the
    fault: band, card and ``#input-shell`` all measure the same x and width,
    and the card's bottom edge IS the band's top edge (measured at 120 and 60,
    both pinned here). What floated was the FILL — ``#band`` inherited the
    screen ground, so a trench of ``$lo-bg`` sat between the card above and the
    composer below and read as the transcript showing through. ``Screen.aside
    #band`` puts the band on the composer's fill, so band and composer are one
    panel and the card is one elevated slab resting on it. Both halves are
    pinned, because dropping either one brings the floating card back.
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
            todo = app.query_one("#todo-panel")
            assert todo.display
            # The card rests ON the band: no ground between the two regions, so
            # the one blank row above the band is the card's own last row.
            assert card.region.bottom == todo.region.y, (card.region, todo.region)
            assert _blank_rows_above(app, "#todo-panel") == 1, _frame(app)[-12:]
            # And the band still ends in its own row before the composer.
            assert _seam(app) == 1, _frame(app)[-12:]
            # The band is the composer's panel, not the transcript's ground …
            assert (
                app.query_one("#band").styles.background
                == app.query_one("#input-shell").styles.background
            )
            # … and the card is one elevation step off it, which is the seam.
            assert card.styles.background != app.query_one("#todo-body").styles.background
        finally:
            builtin.TODO_STORE.pop(session.session_id, None)


#: The docked compositions the fill rule has to hold across. A panel that fused
#: with the composer while its sibling still floated would be worse than the
#: uniform gap this replaced, so every slot the band can hold is driven here.
DOCKED = {
    "todo": (True, False),
    "subagent": (False, True),
    "both": (True, True),
}


async def _dock(pilot: Any, app: OperatorApp, session: Any, todos: bool, subagents: bool) -> None:
    """Populate the band the way the app does, then let it settle and scroll."""
    from local_operator.tools import builtin
    from tests.unit.tui.test_band_panels import _fake_jobs, _Job

    if todos:
        builtin.TODO_STORE[session.session_id] = [
            {"text": "wire the band", "status": "done"},
            {"text": "capture frames", "status": "pending"},
        ]
    if subagents:
        session.jobs = _fake_jobs(_Job("sub-1", "IngestAuditor"))
    app._refresh_band()
    await _settle(pilot, 6)
    app.query_one(TranscriptView).scroll_end(animate=False)
    await _settle(pilot, 6)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("docked", sorted(DOCKED))
async def test_the_dock_is_one_fill_from_the_band_down_to_the_status_row(
    size: tuple[int, int], docked: str
) -> None:
    """Ground separates the CONVERSATION from the dock, not the dock's panels
    from the composer.

    Reported from the field as "there's a gap between the list and the composer,
    there should be a solid fill in between". Measured at 120x40 with five todos
    up: the body painted ``$lo-surface`` over columns 1-119, the slot's own
    padding row came back ``$lo-bg`` over the full 0-120 — wider than either
    slab it sat between, because a transparent slot falls all the way through
    ``#band`` to the Screen — and the composer resumed ``$lo-surface`` one row
    later. A dark row between two lighter ones cut the dock in three.

    Asserted as PAINT rather than as a style value, and against the composer's
    own fill rather than a literal, so the rule survives a theme change: every
    row from the band's first to the status band's last is the same surface the
    input is, and the row above the band is not. That last clause is the other
    half of the rule and the reason this cannot be satisfied by filling
    everything — ``test_one_row_of_ground_under_every_kind_of_last_block``
    pins the transcript's row from the other side.
    """
    from local_operator.tools import builtin

    todos, subagents = DOCKED[docked]
    app, session = _app()
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        await _dock(pilot, app, session, todos, subagents)

        try:
            band = app.query_one("#band")
            shell = app.query_one("#input-shell")
            assert band.region.height > 0, "nothing docked — there is no seam to measure"
            assert band.region.bottom == shell.region.y, (band.region, shell.region)
            # One cell inside the dock's own column, which both the band and the
            # shell are inset to; column 0 is the Screen's gutter in both.
            x = shell.region.x + 1
            composer = _fill_at(app, shell.region.y, x)
            for y in range(band.region.y, shell.region.bottom):
                assert _fill_at(app, y, x) == composer, (
                    y,
                    _fill_at(app, y, x),
                    composer,
                    _frame(app)[-14:],
                )
            # …and the conversation above it is NOT on that fill.
            assert _fill_at(app, band.region.y - 1, x) != composer, _frame(app)[-14:]
        finally:
            builtin.TODO_STORE.pop(session.session_id, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("docked", sorted(DOCKED))
async def test_the_row_under_each_panel_survives_as_the_dock_s_own_fill(
    size: tuple[int, int], docked: str
) -> None:
    """The slot's blank row is kept and repainted, not deleted.

    The two ways this regresses are opposite and both plausible: dropping
    ``.band-slot``'s padding to butt the slabs together (which crowds the dock
    — two same-fill slabs touching read as one crowded panel), or painting the
    row ``$lo-bg`` again (which is the trench). So the row is pinned as BLANK
    and as the composer's surface at the same time; either regression fails one
    clause.
    """
    from local_operator.tools import builtin

    todos, subagents = DOCKED[docked]
    app, session = _app()
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        await _dock(pilot, app, session, todos, subagents)

        try:
            shell = app.query_one("#input-shell")
            x = shell.region.x + 1
            seam = shell.region.y - 1  # the last slot's padding row
            assert not _frame(app)[seam].strip(), _frame(app)[-14:]
            assert _fill_at(app, seam, x) == _fill_at(app, shell.region.y, x), _frame(app)[-14:]
            # The blank row is a row, not a hole: exactly one, as everywhere
            # else in the column.
            assert _seam(app) == 1, _frame(app)[-14:]
        finally:
            builtin.TODO_STORE.pop(session.session_id, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [*SIZES, (80, 24)])
async def test_an_empty_band_paints_nothing_on_the_splash(size: tuple[int, int]) -> None:
    """The fill is unconditional; the BAND is not.

    An idle session and the boot splash have both panels hidden, so the band is
    zero rows and there is nothing for its surface to reach. This is what makes
    an unconditional fill safe where the earlier ``Screen.aside``-scoped one was
    thought necessary — pinned because a later `height: 1` or a panel that
    stopped hiding itself would put a bare surface strip into the splash's
    composition with no content on it.
    """
    app, _session = _app()
    async with app.run_test(size=size) as pilot:
        await _settle(pilot, 6)

        assert app.query_one(WelcomeView).display, "not on the splash"
        band = app.query_one("#band")
        assert band.region.height == 0, band.region
        assert not app.query_one("#todo-panel").display
        assert not app.query_one("#subagent-panel").display
        assert _seam(app) == 1, _frame(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("docked", sorted(DOCKED))
async def test_every_docked_panel_puts_its_content_on_the_dock_s_one_rail(
    size: tuple[int, int], docked: str
) -> None:
    """A panel's rows ARE the panel: same fill, same inset, same column.

    The other face of the fill rule, and the one that says where a panel's
    content goes rather than what is behind it. The subagent panel wore
    ``.band-body`` on its header only, so its rows sat on bare ground one cell
    left of every other glyph in the column — a filled caption over a floating
    list, four rows above a todo panel that was one solid slab (design round
    12, D1/D5).

    Asserted with the ASIDE CLOSED, deliberately: ``Screen.aside #band`` used
    to fill the whole band, so with a ``/btw`` card open those rows were
    already on a surface and the defect disappeared from exactly the frame most
    likely to be inspected. The fill half of that is now unconditional and the
    inset never came from the band at all.
    """
    from local_operator.tools import builtin

    todos, subagents = DOCKED[docked]
    app, session = _app()
    async with app.run_test(size=size) as pilot:
        await _fill(pilot, app, turns=6)
        await _dock(pilot, app, session, todos, subagents)

        try:
            assert not app.query_one(AsidePanel).is_open, "the aside hides this defect"
            band = app.query_one("#band")
            shell = app.query_one("#input-shell")
            rail = _ink_column(app, app.query_one("#prompt-chevron").region.y)
            assert rail >= 0, _frame(app)[-14:]
            columns = {
                y: _ink_column(app, y)
                for y in range(band.region.y, shell.region.bottom)
                if _ink_column(app, y) >= 0
            }
            assert len(columns) >= 3, columns  # header, a row, the chevron
            assert set(columns.values()) == {rail}, (columns, _frame(app)[-14:])
        finally:
            builtin.TODO_STORE.pop(session.session_id, None)
