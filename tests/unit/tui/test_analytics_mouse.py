"""The ``/analytics`` session table driven by a POINTER.

Separate from ``test_analytics_panel.py`` (what the report says, and its
keyboard) for the reason ``test_copy_picker_mouse.py`` is separate from its
own sibling: this file asks whether the surface behaves when it is DRIVEN
rather than merely drawn. Everything here runs against the real ``OperatorApp``
under the production stylesheet — the lightweight hosts elsewhere in this suite
declare no ``CSS_PATH``, so a card sized by percentage rules is not sized at
all under one, and a mouse coordinate against an unsized card means nothing.

The hit-test geometry these tests pin was MEASURED before it was written, and
two of the measurements are why the guards exist:

* ``_body`` is ``height: auto`` inside the scroll container, so its region
  tracks the scroll offset (measured: ``region.y`` 8 -> 4 for an offset of 3).
  The body line under a screen row is therefore a plain subtraction.
* the body OVERHANGS the viewport by everything scrolled out of sight
  (measured: 27 rows on a 40-row frame), and ``body.region`` contains the hint
  line below the card. A hit-test guarded by the body alone resolves clipped
  coordinates to real rows, which is what ``test_a_click_below_the_viewport_
  is_not_a_row`` pins.
"""

from __future__ import annotations

import asyncio

from textual import events

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen
from tests.unit.tui.test_analytics_panel import (
    _nested_screen_agg,
    _push,
    _tall_report_agg,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


def _row_y(screen: AnalyticsScreen, index: int) -> int:
    """Screen row of visible session row ``index``.

    Deliberately NOT a call to the screen's own ``_row_at`` inverted: a test
    that aims with the code under test cannot catch that code aiming wrongly.
    This is the arithmetic spelled from the widget geometry — the viewport
    plus the scroll offset, since the body's region lags a scroll by a frame —
    so the two agree only if both are right.
    """
    scroll = screen._scroll
    line = screen._layout.session_first_line + index
    return scroll.scrollable_content_region.y + line - int(scroll.scroll_offset.y)


def _row_x(screen: AnalyticsScreen) -> int:
    return screen._body.region.x + 6


def _move(screen: AnalyticsScreen, x: int, y: int) -> events.MouseMove:
    return events.MouseMove(
        widget=screen._body,
        x=0,
        y=0,
        delta_x=0,
        delta_y=0,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=x,
        screen_y=y,
    )


def _click(screen: AnalyticsScreen, x: int, y: int, button: int = 1) -> events.Click:
    return events.Click(
        widget=screen._body,
        x=0,
        y=0,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=x,
        screen_y=y,
        chain=1,
    )


def _hover_index(screen: AnalyticsScreen) -> int | None:
    """The hovered row as an INDEX, resolved against the painted layout.

    The screen stores a session id (an index slides when a row above it
    expands), and the tests are about which row the reader sees highlighted.
    """
    if screen._hover is None:
        return None
    return next(
        (i for i, row in enumerate(screen._layout.session_rows) if row.session_id == screen._hover),
        None,
    )


def _first_row(screen: AnalyticsScreen, *, expandable: bool) -> int:
    return next(
        i for i, row in enumerate(screen._layout.session_rows) if row.expandable is expandable
    )


def _show_table(screen: AnalyticsScreen) -> None:
    """Scroll the session table into view, whatever the terminal height.

    Scrolled directly rather than by counting ``pagedown`` presses: how many
    pages reach the table depends on the height, which varies across these
    tests.
    """
    screen._scroll.scroll_to(y=max(0, screen._layout.session_first_line - 1), animate=False)


# -- clicking -----------------------------------------------------------------


def test_clicking_an_expandable_row_expands_it():
    """The operator's ask: "make it so you can also click to expand"."""

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            _show_table(screen)
            await pilot.pause()
            index = _first_row(screen, expandable=True)
            row = screen._layout.session_rows[index]
            assert "kid1session" not in "\n".join(screen.render_lines_for_test())

            await pilot.click(screen, offset=(_row_x(screen), _row_y(screen, index)))
            await pilot.pause()
            assert "kid1session" in "\n".join(screen.render_lines_for_test())
            assert row.session_id in screen._expanded

            # And clicking it again closes it: one gesture, both directions,
            # exactly like the ``enter`` it mirrors.
            await pilot.click(screen, offset=(_row_x(screen), _row_y(screen, index)))
            await pilot.pause()
            assert "kid1session" not in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def test_a_click_moves_the_keyboard_cursor_to_the_clicked_row():
    """The two selection models must not visibly disagree after an ACT.

    Hover alone may sit apart from the caret — that is what makes hover a
    preview — but once a click has expanded row N, an ``enter`` that collapsed
    some other row (because the caret never left it) would be the screen
    holding two different opinions about which row is current.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            index = _first_row(screen, expandable=True)
            target = screen._layout.session_rows[index].session_id
            screen._cursor = screen._layout.session_rows[-1].session_id
            screen._repaint()

            await pilot.click(screen, offset=(_row_x(screen), _row_y(screen, index)))
            await pilot.pause()
            assert screen._cursor == target

            # The caret really does act there now: ``enter`` closes the row the
            # click opened rather than some row the mouse never visited.
            await pilot.press("enter")
            await pilot.pause()
            assert target not in screen._expanded

    asyncio.run(run())


def test_clicking_a_childless_row_is_a_clean_no_op():
    """492 of the operator's 595 roots are childless: this is the MAJORITY click.

    "Toggling nothing" is not enough. Moving the caret there would scroll the
    viewport for a gesture the user experienced as doing nothing, so the row
    count, the cursor, the viewport and the painted frame must all be untouched.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            _show_table(screen)
            await pilot.pause()
            index = _first_row(screen, expandable=False)
            cursor_before = screen._cursor
            offset_before = screen._scroll.scroll_offset.y
            frame_before = "\n".join(screen.render_lines_for_test())
            expanded_before = set(screen._expanded)

            await pilot.click(screen, offset=(_row_x(screen), _row_y(screen, index)))
            await pilot.pause()

            assert screen._cursor == cursor_before, "a dead click moved the keyboard cursor"
            assert screen._scroll.scroll_offset.y == offset_before, "a dead click moved the view"
            assert "\n".join(screen.render_lines_for_test()) == frame_before
            assert screen._expanded == expanded_before

    asyncio.run(run())


def test_clicking_a_depth_one_child_toggles_that_child():
    """A child with children of its own is expandable like any other row.

    The nesting is not two kinds of row with two kinds of click; depth only
    changes the indent.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            _show_table(screen)
            await pilot.pause()
            # Open the root so its child is on screen to be clicked.
            root = _first_row(screen, expandable=True)
            await pilot.click(screen, offset=(_row_x(screen), _row_y(screen, root)))
            await pilot.pause()

            child = next(
                i
                for i, row in enumerate(screen._layout.session_rows)
                if row.depth == 1 and row.expandable
            )
            child_id = screen._layout.session_rows[child].session_id
            assert "grandkidses" not in "\n".join(screen.render_lines_for_test())

            await pilot.click(screen, offset=(_row_x(screen), _row_y(screen, child)))
            await pilot.pause()
            assert child_id in screen._expanded
            assert "grandkidses" in "\n".join(screen.render_lines_for_test())

    asyncio.run(run())


def _painted_y(app, needle: str) -> int:
    """Screen row that actually PAINTS ``needle``, from the compositor.

    Read out of the finished frame rather than re-derived from the geometry the
    hit-test uses: a test that aims with the same arithmetic it is checking
    cannot catch that arithmetic aiming wrongly (both move together). This is
    the independent reading — where the row IS, as the terminal shows it.
    """
    strips = app.screen._compositor.render_strips()
    for y, strip in enumerate(strips):
        if needle in "".join(segment.text for segment in strip):
            return y
    raise AssertionError(f"{needle!r} is not painted anywhere")


def test_the_hit_test_aims_where_the_row_is_painted():
    """A click on the pixel a row is PAINTED on must hit that row.

    This is the test the arithmetic-derived ones cannot be: they compute the
    target from the same geometry ``_row_at`` reads, so a wrong basis moves
    both and nothing trips. Here the coordinate comes from the compositor, so
    the hit-test and the paint agree only if both are right. It exists because
    two mutations survived the rest of the suite — pointing the hit-test at the
    scroll region instead of the viewport, and off the compositor's row — both
    of which this catches.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 24)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()

            # Aim at a row by where it actually PAINTED, not by re-deriving it.
            index = _first_row(screen, expandable=True)
            target = screen._layout.session_rows[index].session_id
            text = screen.render_lines_for_test()[screen._layout.session_first_line + index]
            y = _painted_y(app, text.strip()[:16])

            await pilot.click(screen, offset=(_row_x(screen), y))
            await pilot.pause()
            assert (
                target in screen._expanded
            ), "the click landed on a different row than the one it was painted on"

    asyncio.run(run())


def test_a_right_click_does_not_toggle_a_row():
    """A context-menu click must not act on its way to being ignored.

    The button is tested BEFORE any state changes, which is why this asserts the
    expansion set rather than merely that nothing crashed.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            _show_table(screen)
            await pilot.pause()
            index = _first_row(screen, expandable=True)
            before = set(screen._expanded)
            screen.on_click(_click(screen, _row_x(screen), _row_y(screen, index), button=3))
            await pilot.pause()
            assert screen._expanded == before

    asyncio.run(run())


def test_a_click_below_the_viewport_is_not_a_row():
    """The body overhangs the viewport, so ``body.region`` alone is not a guard.

    Measured on this fixture: the body extends well past the scroll viewport
    and its region CONTAINS the hint line under the card. Clicking there must
    not resolve to whatever table row happens to sit at that body line.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 24)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            body = screen._body.region
            hint_y = screen._hint.region.y
            # The precondition this test exists for: the coordinate really is
            # inside the body and outside the viewport, so the guard is being
            # exercised rather than trivially satisfied.
            assert body.contains(_row_x(screen), hint_y), "fixture no longer reproduces overhang"
            assert not screen._scroll.scrollable_content_region.contains(_row_x(screen), hint_y)

            before = set(screen._expanded)
            assert screen._row_at(_click(screen, _row_x(screen), hint_y)) is None
            screen.on_click(_click(screen, _row_x(screen), hint_y))
            await pilot.pause()
            assert screen._expanded == before

    asyncio.run(run())


# -- hovering -----------------------------------------------------------------


def test_the_hover_follows_the_pointer_between_rows():
    """The highlight names the row under the pointer, and only that row."""

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()

            for index in (0, 1, 2):
                await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, index)))
                await pilot.pause()
                assert _hover_index(screen) == index

    asyncio.run(run())


def test_the_hover_clears_when_the_pointer_leaves_the_rows():
    """A highlight left painted after the pointer is gone is a lie about aim."""

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 0)))
            await pilot.pause()
            assert _hover_index(screen) == 0

            # Onto the hint line, which is outside the scroll viewport.
            await pilot.hover(screen, offset=(_row_x(screen), screen._hint.region.y))
            await pilot.pause()
            assert screen._hover is None
            assert str(screen.styles.pointer) == "default"

    asyncio.run(run())


def test_the_pointer_is_a_hand_only_over_an_expandable_row():
    """A hand over a dead row promises a click it does not keep."""

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            _show_table(screen)
            await pilot.pause()

            await pilot.hover(
                screen, offset=(_row_x(screen), _row_y(screen, _first_row(screen, expandable=True)))
            )
            await pilot.pause()
            assert str(screen.styles.pointer) == "pointer"

            childless = _first_row(screen, expandable=False)
            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, childless)))
            await pilot.pause()
            # Still highlighted — it IS the row under the pointer — but no hand.
            assert _hover_index(screen) == childless
            assert str(screen.styles.pointer) == "default"

    asyncio.run(run())


def test_the_totals_block_above_the_table_takes_no_hover():
    """Only table rows are live. The report above them is text, not targets."""

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            await pilot.pause()
            # Frame 1, table not scrolled to: aim at the totals block near the
            # top of the body.
            await pilot.hover(screen, offset=(_row_x(screen), screen._body.region.y + 2))
            await pilot.pause()
            assert screen._hover is None
            assert str(screen.styles.pointer) == "default"

    asyncio.run(run())


def test_the_hover_and_the_cursor_can_sit_on_different_rows():
    """Two selection models, two marks, and they must not merge.

    If hover painted like the caret the reader would see two cursors and could
    not tell which row ``enter`` acts on.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            screen._cursor = screen._layout.session_rows[2].session_id
            screen._repaint()
            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 0)))
            await pilot.pause()

            assert _hover_index(screen) == 0
            assert screen._cursor_index() == 2
            # The caret is on the CURSOR row and nowhere else: the hovered row
            # is marked by its ground, not by a second caret.
            lines = screen.render_lines_for_test()
            first = screen._layout.session_first_line
            assert "❯" in lines[first + 2]
            assert "❯" not in lines[first]

    asyncio.run(run())


def test_the_hover_does_not_reflow_the_row():
    """No reflow on interaction — three review rounds of #873 pinned this.

    The highlight is a background span, so the row's TEXT is identical hovered
    and not. A hover that inserted a marker would shift the row under a moving
    pointer, which is the one place a layout shift is least forgivable. The
    colour half is pinned too: the hovered row's segments must carry a
    background, because a highlight that costs the text nothing and paints
    nothing is not a highlight.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            before = screen.render_lines_for_test()

            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 1)))
            await pilot.pause()
            after = screen.render_lines_for_test()

            assert _hover_index(screen) == 1, "the fixture did not actually hover a row"
            assert after == before, "the hover changed the table's text, not just its colour"

            # And the paint DID change on exactly that row: the hover ground was
            # laid over it. The panel background itself has a bgcolor, so this
            # must name the tint specifically rather than merely "a background".
            from local_operator.tui import theme as theme_mod

            tint = theme_mod.semantic_color("tint-select")
            y = _painted_y(app, after[screen._layout.session_first_line + 1].strip()[:16])
            strip = app.screen._compositor.render_strips()[y]
            hexes = {
                s.style.bgcolor.triplet.hex
                for s in strip
                if s.style is not None
                and s.style.bgcolor is not None
                and s.style.bgcolor.triplet is not None
            }
            assert tint in hexes, f"the hovered row did not paint the tint-select ground; {hexes}"

    asyncio.run(run())


def test_hover_on_the_cursor_row_is_a_distinct_state():
    """Hover + caret on the same row paints a THIRD ground, not either alone.

    The pointer landing on the row the keyboard is already on is its own
    visible state: ``tint-select-hi`` against the hover's ``tint-select``, so a
    reader can tell "the mouse is on the caret row" from "the mouse is on some
    other row" at a glance.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            cursor = screen._cursor_index()
            assert cursor is not None

            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, cursor)))
            await pilot.pause()
            assert _hover_index(screen) == cursor, "hover and cursor did not coincide"

            y = _painted_y(
                app,
                screen.render_lines_for_test()[screen._layout.session_first_line + cursor].strip()[
                    :16
                ],
            )
            strip = app.screen._compositor.render_strips()[y]
            grounds = {
                segment.style.bgcolor
                for segment in strip
                if segment.style is not None and segment.style.bgcolor is not None
            }
            from local_operator.tui import theme as theme_mod

            hi = theme_mod.semantic_color("tint-select-hi")
            lo = theme_mod.semantic_color("tint-select")
            hexes = {g.triplet.hex for g in grounds if g is not None and g.triplet is not None}
            assert (
                hi in hexes
            ), f"the cursor row under the pointer did not step up to tint-select-hi; {hexes}"
            assert lo not in hexes, "the coincide state painted the plain hover ground"

    asyncio.run(run())


# -- the wheel gotcha ---------------------------------------------------------


def test_the_wheel_moves_the_highlight_under_a_resting_pointer():
    """THE WHEEL CASE. A real terminal sends NO ``MouseMove`` while only the
    wheel turns, so the rows slide under a pointer that never moved and the
    highlight would stay painted on a row the pointer is no longer over.

    The pointer is placed once and then NEVER moved again: every subsequent
    event is a wheel notch. The highlight must track the row that is now under
    that fixed coordinate.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 24)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()

            x, y = _row_x(screen), _row_y(screen, 3)
            await pilot.hover(screen, offset=(x, y))
            await pilot.pause()
            before = _hover_index(screen)
            assert before is not None
            hovered_before = screen._hover

            # Wheel notches through ``App.on_event``, the path a terminal uses.
            # Textual stops the event on the container while it can still
            # scroll, so these never reach a screen-level handler at all — the
            # highlight can only follow via the scroll watcher.
            offset_before = screen._scroll.scroll_offset.y
            for _ in range(3):
                await app.on_event(
                    events.MouseScrollDown(
                        widget=None,
                        x=x,
                        y=y,
                        delta_x=0,
                        delta_y=1,
                        button=0,
                        shift=False,
                        meta=False,
                        ctrl=False,
                        screen_x=x,
                        screen_y=y,
                    )
                )
                await pilot.pause()
            await pilot.pause()

            moved = screen._scroll.scroll_offset.y - offset_before
            assert moved > 0, "the wheel did not scroll, so this test proves nothing"

            # The rows moved UP by ``moved`` lines under a fixed pointer, so the
            # row now under it is ``before + moved``. Asserted as the exact row
            # rather than merely "it changed": a highlight that cleared, or one
            # that jumped somewhere arbitrary, would both pass a weaker check.
            assert _hover_index(screen) == before + moved
            assert screen._hover != hovered_before

    asyncio.run(run())


def test_scrolling_the_rows_out_from_under_the_pointer_clears_the_highlight():
    """Scrolled far enough, the pointer is over the report ABOVE the table.

    The highlight must go out with them rather than staying painted on the last
    row it saw — the failure ``copy_picker`` records as the highlight surviving
    six notches off the pane.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 24)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            x, y = _row_x(screen), _row_y(screen, 1)
            await pilot.hover(screen, offset=(x, y))
            await pilot.pause()
            assert _hover_index(screen) is not None

            # Back to the top of the report: the table is now far below, and the
            # pointer's coordinate is over the totals block.
            screen._scroll.scroll_to(y=0, animate=False)
            await pilot.pause()
            await pilot.pause()
            assert screen._hover is None
            assert str(screen.styles.pointer) == "default"

    asyncio.run(run())


def test_expanding_a_row_re_resolves_the_highlight_under_a_still_pointer():
    """A click inserts rows BELOW the pointer, moving the table under it.

    The gesture that changes the row count is itself a way to move rows under a
    resting pointer, so the highlight has to be re-resolved from it too — and
    the row under the pointer after an expand is not the row that was there.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg(kids=30))
            _show_table(screen)
            await pilot.pause()
            root = _first_row(screen, expandable=True)

            # Rest the pointer a few rows BELOW the expandable root, so opening
            # it pushes different sessions under the fixed coordinate.
            x, y = _row_x(screen), _row_y(screen, root + 2)
            await pilot.hover(screen, offset=(x, y))
            await pilot.pause()
            before_id = screen._hover
            assert before_id is not None

            # Expand with the KEYBOARD so the pointer genuinely never moves.
            screen._cursor = screen._layout.session_rows[root].session_id
            screen._repaint()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            assert _hover_index(screen) == root + 2, "the highlight did not follow the coordinate"
            assert screen._hover != before_id, "30 rows were inserted above it and it did not move"

    asyncio.run(run())
