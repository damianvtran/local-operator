"""A lane change must re-fit every ledger row, not only the ones it resizes.

The operator's second report on this surface — "some lines will be full width and
some won't be, and moving the mouse over them updates them" — is a tool ledger
whose rows stop short of the lane's right edge, with the right-aligned status
(``✓ 0.5s``) stranded mid-row, while the row's own background slab spans the
whole lane. So the widget is full width and only the AUTHORED content is narrow:
the row was built at an older lane width and nothing ever re-fitted it.

The row's re-fit notification is ``Resize``, and there is one frame in which it
reaches nobody who is already on screen:

* A widget's layout request reaches the Screen ASYNCHRONOUSLY —
  ``Widget._check_refresh`` posts ``messages.Layout``, and ``Screen._on_layout``
  is what fills ``_layout_widgets``. Until that message is processed,
  ``_layout_widgets`` is EMPTY, so ``Screen._refresh_layout(scroll=True)`` —
  the pass a pending wheel scroll triggers (``Screen._on_timer_update``) —
  takes ``Compositor.reflow_visible``, the visible-only arrangement. A lane
  change landing in that window is serviced by that pass, and it resizes only
  the widgets newly EXPOSED by it: every already-visible row keeps the width it
  was authored at while the compositor paints it at the new one.
* ``reflow_visible`` sets ``_full_map_invalidated``. The lazy ``full_map`` read
  that follows (any ``Widget.size``/``region`` lookup for a widget the visible
  map does not hold, plus the app's own measuring code) re-arranges at the NEW
  size and stores that as ``_full_map``, so the corrective full reflow compares
  against it, finds nothing changed and sends ``Resize`` to nobody. The narrow
  content is then permanent until something else calls ``refresh_row`` — a
  hover, which is the operator's one-row-at-a-time cure.

``TranscriptView._refit_ledger_lane`` is the fix, and these tests pin it the way
the operator met it: the sidebar opened, a wheel notch, the sidebar closed, and
then an assertion on the FRAME for every visible ledger row, against the width
the reconciler laid that row out at (``outer_size``, the field the paint and the
slab both use). ``region``/``size`` are deliberately not the expectation: they
read the compositor's map, and in the frame under test that map already answers
with the NEW lane — which is why a check for "the row was built at its region
width" cannot see this defect.

The sidebar's toggle is called SYNCHRONOUSLY (``action_toggle_sidebar``) rather
than pressed, because a real key press pumps the message queue and lets the
``Layout`` message land before the scroll pass — which is precisely the window
these tests exist to exercise.
"""

from __future__ import annotations

import pytest
from rich.text import Text

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: Frame the tests run at. Wide enough that the docked sidebar is a real lane
#: change (at 130 columns the lane is 126 closed and 79 open) and tall enough
#: that a 40-row ledger scrolls.
FRAME = (130, 36)

#: Taller than the frame, so there are rows above the viewport to scroll to.
LEDGER_ROWS = 40

#: The word the prose assertions locate their rows by. Inside EVERY wrapped row
#: of :data:`PROMPT`, so a painted-line search finds all of them (or, when the
#: block is stale, all FOUR of the narrow build instead of the one wide one).
PROSE_MARKER = "ZLOTY"

#: A prompt that is one row in a 126-cell lane and four in a 79-cell one — the
#: same shape the design and QA streams measured on a wrapping `UserBlock`.
PROMPT = " ".join([f"{PROSE_MARKER} alpha beta gamma delta epsilon"] * 8)


def _app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


def _marker(index: int) -> str:
    """A unique substring of one row's summary, for locating it in the frame."""
    return f"row{index}_"


def _card(index: int) -> ToolCard:
    return ToolCard(f"c{index}", "bash", {"command": f"echo {_marker(index)} " + "x" * 30})


def _ledger(view: TranscriptView) -> list[ToolCard]:
    """The mounted tool rows. Only this suite's own cards are asserted on."""
    return [
        block
        for block in view.blocks()
        if isinstance(block, ToolCard) and block.LEDGER_ROW and block.parent is view
    ]


def _painted_line(app: OperatorApp, marker: str) -> str:
    """The painted screen line carrying ``marker``.

    Located by CONTENT rather than by the block's ``region.y``: this defect is
    precisely the compositor's map and the frame it painted disagreeing, so the
    instrument must not read the map to find the pixels.
    """
    for strip in app.screen._compositor.render_strips():
        if marker in strip.text:
            return strip.text
    raise AssertionError(f"no painted line carries {marker!r}")


def _painted_right_edge(app: OperatorApp, block: ToolCard, marker: str) -> int:
    """Cell one past the row's last inked cell in the frame, inside its own box.

    The measurement is confined to the row's own cells because the transcript's
    scrollbar thumb paints beside it: read across the whole line, the thumb's
    ink would be attributed to the row.
    """
    line = _painted_line(app, marker)
    box = block.region.x + block.outer_size.width
    return len(line[:box].rstrip())


def _expected_right_edge(block: ToolCard) -> int:
    """Where the row's last ink cell belongs: its own line built at its laid-out width.

    A build is a pure function of the row's state and the width it is given, so
    this is the line the row SHOULD be painting — the assertion is "the frame
    shows a build at the width the reconciler gave you", which is what a tear
    breaks.
    """
    return block.region.x + len(block._build_row(block.outer_size.width).plain)


async def _settle(pilot) -> None:
    """Wait for the frame a reader is left with (a mount paints, layout follows)."""
    await pilot.pause()
    await pilot.pause()


async def _seed_ledger(pilot, view: TranscriptView) -> None:
    for i in range(LEDGER_ROWS):
        card = _card(i)
        view.append_block(card)
        card.mark_done("done")
    await _settle(pilot)


async def _scroll_frame(app: OperatorApp, pilot) -> None:
    """A wheel notch, honoured by the visible-only layout pass.

    ``_refresh_layout`` is called directly rather than waiting for the update
    timer, because the pass is the frame under test and a test cannot depend on
    which of two queued messages the loop runs first.
    """
    view = app.query_one(TranscriptView)
    view.scroll_relative(y=-6, animate=False)
    await pilot.pause()
    app.screen._refresh_layout(scroll=True)
    await pilot.pause()
    assert app.screen._compositor._visible_map is not None, (
        "the scroll pass did not take the visible-only branch, so this test is not "
        "exercising the frame it was written for"
    )


async def _toggle_sidebar_behind_a_scroll(app: OperatorApp, pilot) -> None:
    """Move the lane while a scroll pass is still pending — the operator's race.

    Three steps, each asserted so a Textual change that closes this window fails
    here loudly instead of leaving a green test that reproduces nothing:

    1. ``action_toggle_sidebar`` runs synchronously, so the ``Layout`` message it
       queues has NOT been processed and ``_layout_widgets`` is empty.
    2. The pass a pending scroll triggers is therefore the visible-only one, and
       it resizes only the widgets newly exposed by it.
    3. The lazy full arrangement — what any ``size``/``region`` read of a widget
       the visible map does not hold performs — stores the new geometry as
       ``_full_map``, so the corrective full reflow finds nothing changed.
    """
    assert app.screen._layout_widgets == {}, "a layout request was already processed"
    app.action_toggle_sidebar()
    app.screen._refresh_layout(scroll=True)
    app.screen._compositor.full_map
    await _settle(pilot)


def _assert_every_row_fits_its_own_lane(app: OperatorApp, view: TranscriptView) -> None:
    """The settled frame: no row may paint a line narrower than the lane it is in."""
    lane = view.scrollable_content_region.width
    rows = _ledger(view)
    assert rows, "no ledger rows were mounted"
    # The funnel's rule is the whole ledger, on screen or not: a row scrolled out
    # of view is already correct when the reader reveals it.
    assert {b._built_width for b in rows} == {
        lane
    }, "some ledger row is still authored at an older lane width"
    edges = []
    for i, block in enumerate(rows):
        # The truth is the width the reconciler laid the row out at -- never
        # `region`/`size`, which read the compositor map that already answers
        # with the new lane in the frame under test.
        assert block.outer_size.width == lane, (
            f"{block.tool_call_id} is laid out at {block.outer_size.width}, "
            f"not the {lane}-cell lane"
        )
        try:
            right = _painted_right_edge(app, block, _marker(i))
        except AssertionError:
            continue  # scrolled out of the frame; the width checks above still hold
        assert right == _expected_right_edge(block), (
            f"{block.tool_call_id}'s painted line stops at cell {right} instead of "
            f"{_expected_right_edge(block)}: its content is authored at an older lane width"
        )
        edges.append(right)
    assert len(edges) >= 2, "too few ledger rows were on the painted frame to compare"
    # The operator's own words: "some lines will be full width and some won't".
    assert len(set(edges)) == 1, f"ledger rows paint at different widths: {sorted(set(edges))}"


@pytest.mark.asyncio
async def test_closing_the_sidebar_behind_a_scroll_refits_every_row() -> None:
    """The operator's frame: the lane grows and the rows keep the width they had."""
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        await _seed_ledger(pilot, view)

        await pilot.press("ctrl+b")  # sidebar open: the lane narrows
        await _settle(pilot)
        narrowed = view.scrollable_content_region.width
        assert {b._built_width for b in _ledger(view)} == {narrowed}
        await _scroll_frame(app, pilot)

        await _toggle_sidebar_behind_a_scroll(app, pilot)
        assert view.scrollable_content_region.width > narrowed, "the sidebar did not close"
        _assert_every_row_fits_its_own_lane(app, view)


@pytest.mark.asyncio
async def test_opening_the_sidebar_behind_a_scroll_refits_every_row() -> None:
    """The mirror: a lane that shrinks leaves rows overflowing it, not stopping short."""
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        await _seed_ledger(pilot, view)
        widened = view.scrollable_content_region.width
        await _scroll_frame(app, pilot)

        await _toggle_sidebar_behind_a_scroll(app, pilot)
        assert view.scrollable_content_region.width < widened, "the sidebar did not open"
        _assert_every_row_fits_its_own_lane(app, view)


@pytest.mark.asyncio
async def test_a_lane_that_did_not_move_repaints_no_row() -> None:
    """The funnel is keyed on the MEASUREMENT, so it must cost nothing when it holds.

    A scroll frame is a layout pass like any other and every append moves this
    container's extent, so a funnel that repainted unconditionally would be an
    O(rows) rebuild per frame on a long ledger. The lane is compared as an
    equality for exactly that reason.
    """
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        await _seed_ledger(pilot, view)

        calls = 0
        original = ToolCard.refresh_row

        def counting(self: ToolCard, width: int | None = None) -> None:
            nonlocal calls
            calls += 1
            original(self, width)

        ToolCard.refresh_row = counting  # type: ignore[method-assign]
        try:
            # A layout pass with the lane where it already was, then a settle with
            # nothing changed at all: neither may touch a row.
            await _scroll_frame(app, pilot)
            await _settle(pilot)
            assert calls == 0, "a lane that did not move repainted the ledger"

            # And the case the funnel is FOR, to prove the instrument counts at all.
            await pilot.press("ctrl+b")
            await _settle(pilot)
            assert calls > 0, "the funnel ignored a lane that did move"
        finally:
            ToolCard.refresh_row = original  # type: ignore[method-assign]


def test_a_published_lane_outranks_the_cards_own_ladder() -> None:
    """Threading: the container's number is used verbatim, and `None` changes nothing.

    A row built DETACHED has no map, no container and no parent to answer with,
    so without the threaded lane it falls to ``FALLBACK_WIDTH`` (80) — which is
    the second wrong width, not a harmless default.
    """
    card = ToolCard("t1", "bash", {"command": "echo hi-" + "x" * 20})
    assert card.fit_width(64) == 64
    assert card.fit_width(None) == 0  # nothing knows anything; the ladder still says so
    assert card.fit_width(0) == 0

    card._refresh_row(64)
    applied = card.renderable
    assert isinstance(applied, Text)
    assert applied.plain == card._build_row(64).plain
    assert applied.plain != card._build_row(80).plain


def _painted_lines(app: OperatorApp) -> list[str]:
    """Every painted strip's text, in frame order."""
    return [strip.text for strip in app.screen._compositor.render_strips()]


def _painted_prose_rows(app: OperatorApp, block: UserBlock, expected: list[str]) -> list[str]:
    """The block's painted rows, in its own column, clipped to the authored length.

    Sliced at ``block.region.x`` because the transcript's own padding is not
    part of what the block authored, and CLIPPED to ``expected``'s own length so
    the scrollbar thumb painted beside one row is not read as that row's ink.
    The comparison is against the block's FRESH build, so what this asserts is
    "these cells carry the characters this block authors at this lane", which is
    exactly what a stale fold breaks. A row count that does not match returns
    the raw lines, so the caller's equality fails with a readable diff.
    """
    painted = [line for line in _painted_lines(app) if PROSE_MARKER in line]
    if len(painted) != len(expected):
        return painted
    pad = block.region.x
    return [line[pad : pad + len(exp)].rstrip() for line, exp in zip(painted, expected)]


def _fresh_prose_rows(block: UserBlock) -> list[str]:
    """The rows the block authors at the lane it is now in (a fresh fold)."""
    rendered = block._build()
    assert isinstance(rendered, Text)
    return [line.rstrip() for line in rendered.plain.splitlines() if PROSE_MARKER in line]


@pytest.mark.asyncio
async def test_a_wrapping_prompt_behind_the_scroll_window_refits_to_the_lane() -> None:
    """The missed ``Resize`` is not ledger-specific: the prose next to it tears too.

    Three review streams hit this in the same frame — the ledger rows repaired
    and the prompt directly under them still wrapped for the old lane, ~50 cells
    short of the tool rows it sits between. The fix walks every block that
    authors its rows at a width (:meth:`TranscriptView._refit_authored_blocks`),
    so the operator's sequence has to leave the prompt at the lane it is in.
    """
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        await _seed_ledger(pilot, view)
        prompt = UserBlock(PROMPT)
        view.append_block(prompt)
        # A few rows of ledger after it, so the prompt is ON SCREEN both before
        # and after the wheel notch — the reader's position in the operator's
        # own frame is the foot of the transcript, and prose below the viewport
        # would be re-fitted by being EXPOSED rather than by this walk, which is
        # a different code path and would prove nothing here.
        for extra in range(6):
            tail_card = _card(LEDGER_ROWS + extra)
            view.append_block(tail_card)
            tail_card.mark_done("done")
        await _settle(pilot)
        view.scroll_end(animate=False)
        await _settle(pilot)
        wide_rows = _fresh_prose_rows(prompt)
        assert (
            _painted_prose_rows(app, prompt, wide_rows) == wide_rows
        ), "the prompt is not on the painted frame at rest, so this test proves nothing"

        # The pending-scroll window, armed at the foot of the transcript.
        await _scroll_frame(app, pilot)
        assert _painted_prose_rows(
            app, prompt, wide_rows
        ), "the prompt left the frame with the scroll"

        await pilot.press("ctrl+b")  # sidebar open: the lane narrows
        await _settle(pilot)
        narrow_lane = view.scrollable_content_region.width
        assert prompt._built_width == narrow_lane, "the fixture never wrapped at the narrow lane"
        narrow_rows = _fresh_prose_rows(prompt)
        assert len(narrow_rows) > len(
            wide_rows
        ), "the fixture does not fold differently at the two lanes"
        assert (
            _painted_prose_rows(app, prompt, narrow_rows) == narrow_rows
        ), "the narrow fold is not the one on screen, so the sequence below proves nothing"
        await _scroll_frame(app, pilot)

        await _toggle_sidebar_behind_a_scroll(app, pilot)
        lane = view.scrollable_content_region.width
        assert lane > narrow_lane, "the sidebar did not close"
        assert (
            prompt._built_width == lane
        ), f"the prompt is still authored at {prompt._built_width} in a {lane}-cell lane"
        fresh = _fresh_prose_rows(prompt)
        assert _painted_prose_rows(app, prompt, fresh) == fresh, (
            "the painted prompt is not the build the block authors at this lane: it is "
            "still showing the narrow fold"
        )


@pytest.mark.asyncio
async def test_a_lane_change_rebuilds_an_authored_block_once() -> None:
    """Both triggers can reach one block in one lane change; the guard makes it one build.

    A lane change delivers the ordinary ``Resize`` to the blocks the compositor
    does reach and the container's lane walk to every block that authors a
    width, so one block can be notified twice for one rebuild. The width
    equality in :meth:`TranscriptBlock.refit_width` is what makes the second
    notification free; without it the walk would double the cost of the path
    that already worked.
    """
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        prompt = UserBlock(PROMPT)
        view.append_block(prompt)
        await _settle(pilot)

        builds = 0
        original = UserBlock.set_content

        def counting(self: UserBlock, renderable, *, layout: bool = True) -> None:
            nonlocal builds
            builds += 1
            original(self, renderable, layout=layout)

        UserBlock.set_content = counting  # type: ignore[method-assign]
        try:
            await pilot.press("ctrl+b")
            await _settle(pilot)
            assert view.scrollable_content_region.width < FRAME[0], "the lane did not move"
        finally:
            UserBlock.set_content = original  # type: ignore[method-assign]
        assert builds == 1, f"one lane change built the prompt {builds} times"


@pytest.mark.asyncio
async def test_a_gutter_only_lane_move_leaves_one_right_edge() -> None:
    """A scrollbar-only lane move is outside the ``changed`` gate — and must not tear.

    ``scrollable_content_region.width`` also moves when the vertical thumb
    appears or leaves without this container's own size moving, and that does
    not reach ``_size_updated``'s ``changed`` gate, so the lane funnel does not
    run (R3, review round 1; measured: the thumb leaving moves the lane 126→127
    with ZERO funnel calls, and every row stays laid out at 126 — uniformly).

    The symptom this file exists for is rows at DIFFERENT widths in one frame,
    so the assertion is agreement rather than equality with the lane: reaching
    the lane here would need the funnel on every layout pass, including the
    passes that moved nothing, and reading ``scrollable_content_region`` on one
    of those forces the deferred full arrangement (see the funnel's docstring).
    """
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        await _seed_ledger(pilot, view)
        assert view.show_vertical_scrollbar, "no thumb to remove: the fixture does not overflow"

        lane_before = view.scrollable_content_region.width
        view.styles.overflow_y = "hidden"
        await _settle(pilot)
        assert (
            view.scrollable_content_region.width != lane_before
        ), "the gutter did not move the lane"

        rows = _ledger(view)
        assert {b._built_width for b in rows} == {
            lane_before
        }, "a gutter-only move re-authored some rows and not others"
        edges = []
        for i, block in enumerate(rows):
            try:
                edges.append(_painted_right_edge(app, block, _marker(i)))
            except AssertionError:
                continue  # scrolled out of the frame
        assert len(edges) >= 2, "too few rows on the painted frame to compare"
        assert len(set(edges)) == 1, f"rows paint at different widths: {sorted(set(edges))}"
