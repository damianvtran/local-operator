"""The ``/analytics`` session table driven by a POINTER.

Separate from ``test_analytics_panel.py`` (what the report says, and its
keyboard) for the reason ``test_copy_picker_mouse.py`` is separate from its
own sibling: this file asks whether the surface behaves when it is DRIVEN
rather than merely drawn. Everything here runs against the real ``OperatorApp``
under the production stylesheet — the lightweight hosts elsewhere in this suite
declare no ``CSS_PATH``, so a card sized by percentage rules is not sized at
all under one, and a mouse coordinate against an unsized card means nothing.

The hit-test geometry these tests pin was MEASURED before it was written, and
one of the measurements is why the guards exist:

* the body and the viewport are now the SAME widget (``ReportView``), so a row's
  screen position is the viewport plus the scroll offset with nothing in
  between: ``region.y`` cannot lag a scroll because there are no longer two
  regions to disagree. Before that they were a ``Static`` inside a
  ``VerticalScroll``, its region tracked the scroll offset a frame late
  (measured: ``region.y`` 8 -> 4 for an offset of 3) and it OVERHANGED the
  viewport by everything scrolled out of sight (measured: 27 rows on a 40-row
  frame). A hit-test guarded by the body region alone therefore resolved clipped
  coordinates to real rows, and ``test_a_click_below_the_viewport_is_not_a_row``
  still asserts the guard that rejects them — one widget does not make the guard
  unnecessary, it just makes it the only thing standing between the pointer and
  the row arithmetic, which that test's precondition pins.
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


def _press(screen: AnalyticsScreen, x: int, y: int) -> events.MouseDown:
    """Mouse button DOWN at a screen coordinate, as the screen sees it."""
    return events.MouseDown(
        widget=screen._body,
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


def _drag_to(screen: AnalyticsScreen, x: int, y: int) -> events.MouseMove:
    """A move with the button HELD — the event a drag is made of.

    ``button=1`` rather than ``_move``'s 0, because the screen's selection
    machinery only grows a selection while a button is down; a hover move is a
    different event as far as this path is concerned.

    ``x``/``y`` carry the SCREEN coordinate, like ``_press``/``_release`` and
    unlike ``_move``'s 0,0: the screen resolves the selection's content offset by
    hit-testing the pointer position it is given (``get_widget_and_offset_at``),
    so a zeroed position resolves no offset at all — the drag then has a start
    and no end, which Textual reads as "select to the end of the report". A
    silently unbounded copy is exactly the kind of thing a test asserting
    "the row is in the copied text" cannot see.
    """
    return events.MouseMove(
        widget=screen._body,
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


def _release(screen: AnalyticsScreen, x: int, y: int) -> events.MouseUp:
    return events.MouseUp(
        widget=screen._body,
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
    """A coordinate BELOW the viewport is not a row, whatever the row maths says.

    The defect this pins came with the body-inside-a-scroll-container shape: the
    body overhung the viewport by everything scrolled out of sight and its region
    contained the hint line under the card, so a hit-test that trusted the body
    resolved a clipped coordinate to a real table row.

    The body and the viewport are one widget now, which is what makes the
    precondition below the interesting half of this test: the row arithmetic on
    its own — screen y, minus the viewport's top, plus the scroll offset, minus
    the table's first line — DOES name a real row at this coordinate. So the
    viewport guard is the only thing rejecting the click, and this test would
    pass on a hit-test that had lost it only if the fixture stopped aiming below
    the frame. Asserting that is the difference between exercising the guard and
    asserting something that was already true.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 24)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            viewport = screen._scroll.scrollable_content_region
            x, hint_y = _row_x(screen), screen._hint.region.y
            # The coordinate is outside the viewport...
            assert not viewport.contains(x, hint_y), "fixture no longer aims below the frame"
            # ...and the arithmetic the guard protects would land on a real row.
            line = hint_y - viewport.y + int(screen._scroll.scroll_offset.y)
            index = line - screen._layout.session_first_line
            assert 0 <= index < len(screen._layout.session_rows), (
                "the unguarded arithmetic no longer resolves a row, so this test "
                "would pass without the viewport guard"
            )

            before = set(screen._expanded)
            assert screen._row_at(_click(screen, x, hint_y)) is None
            screen.on_click(_click(screen, x, hint_y))
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


def test_expand_all_re_resolves_the_highlight_without_moving_the_viewport():
    """``e`` moves rows under a resting pointer EVEN WHEN THE VIEWPORT DOES NOT.

    This is the case the wheel test cannot be and QA's round-1 expand-all row
    (M5c) did not cover: their geometry scrolled the viewport, so the ``scroll_y``
    watch fired and re-resolved the highlight for them. Here the cursor row stays
    visible across ``e`` — mount parks the cursor on table row 0, expanding root 0
    inserts its children BELOW it, and the reveal inside ``action_toggle_all`` is
    therefore a no-op — so ``scroll_y`` never notifies and the ONLY thing that can
    re-resolve the highlight is the action's own deferred ``_viewport_moved``.
    Without it the highlight stays painted on a row the pointer is not over: the
    wheel-gotcha defect via a keypress (review R1).

    The ``scroll_offset`` assertion is the pin, not a comment. A test that let the
    viewport move would pass against the broken code exactly like M5c did, because
    the watch would do the re-resolve the action failed to schedule.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg(kids=30))
            _show_table(screen)
            await pilot.pause()

            scroll = screen._scroll
            # Precondition: the cursor is on table row 0 and ON SCREEN, so the
            # reveal inside ``action_toggle_all`` has nothing to scroll and no
            # ``scroll_y`` notification can fire.
            assert screen._cursor == screen._layout.session_rows[0].session_id
            assert screen._cursor_on_screen(), "fixture: cursor row is not visible"

            # Rest the pointer two rows below the expandable root. Expand-all
            # inserts 30 children at indices 1..30, so the row under this FIXED
            # coordinate changes from a root to a child.
            x, y = _row_x(screen), _row_y(screen, 2)
            await pilot.hover(screen, offset=(x, y))
            await pilot.pause()
            before_id = screen._hover
            assert before_id == screen._layout.session_rows[2].session_id
            assert screen._layout.session_rows[2].depth == 0, "fixture: index 2 is not a root"
            scroll_y_before = scroll.scroll_offset.y

            await pilot.press("e")
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()

            # THE PIN: the viewport did not move, so no watch fired and nothing
            # but the action's own deferred re-resolve could have moved the
            # highlight. If this assertion ever fails the test has silently
            # degraded into the M5c case and no longer pins R1.
            assert scroll.scroll_offset.y == scroll_y_before, (
                "the viewport moved, so this no longer pins the no-scroll path — "
                "the scroll_y watch would re-resolve the highlight on its own"
            )
            # The highlight followed the COORDINATE into the inserted block.
            assert _hover_index(screen) == 2, "the highlight did not follow the coordinate"
            assert screen._hover != before_id, "30 rows were inserted above it and it did not move"
            assert screen._layout.session_rows[2].depth == 1, "fixture: index 2 is not a child"

    asyncio.run(run())


def test_collapse_all_re_resolves_the_highlight_without_moving_the_viewport():
    """The other direction of R1: collapsing every row also moves them.

    Staged so the pointer rests on a CHILD that collapse-all removes: the row
    under the fixed coordinate changes from a child back to a root, and — as in
    the expand direction — the cursor row stays visible so ``scroll_y`` never
    notifies. The re-resolve can only come from ``action_toggle_all`` itself.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg(kids=30))
            _show_table(screen)
            await pilot.pause()

            scroll = screen._scroll
            # Expand everything first, so the second ``e`` is a collapse-all.
            await pilot.press("e")
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()
            assert len(screen._layout.session_rows) > 41, "fixture: expand-all did not expand"
            assert screen._cursor_on_screen(), "fixture: cursor row is not visible after expand"

            x, y = _row_x(screen), _row_y(screen, 2)
            await pilot.hover(screen, offset=(x, y))
            await pilot.pause()
            before_id = screen._hover
            assert screen._layout.session_rows[2].depth == 1, "fixture: index 2 is not a child"
            assert before_id == screen._layout.session_rows[2].session_id
            scroll_y_before = scroll.scroll_offset.y

            await pilot.press("e")
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()

            assert (
                scroll.scroll_offset.y == scroll_y_before
            ), "the viewport moved, so this no longer pins the no-scroll path"
            assert _hover_index(screen) == 2, "the highlight did not follow the coordinate"
            assert screen._hover != before_id, "the children vanished and it did not move"
            assert screen._layout.session_rows[2].depth == 0, "fixture: index 2 is not a root"

    asyncio.run(run())


def test_the_empty_scrollbar_gutter_takes_no_hover_and_no_click():
    """The stable gutter is RESERVED space, not a row — even when it is empty.

    ``#analytics-scroll`` sets ``scrollbar-gutter: stable``, so the column is
    there whether or not a bar is drawn. On a frame whose body does NOT overflow
    there is no bar widget to capture events, so gutter coordinates really reach
    the screen — and the viewport guard in ``_row_at`` is the only thing standing
    between them and a row. Review R2 measured exactly that: under a
    ``scroll.region.contains`` mutant a gutter hover highlights a row and a
    gutter click on the root line TOGGLES it. The shipped guard is correct; this
    is the pin that was missing.

    The preconditions ARE the test. Without asserting that the frame does not
    overflow and that the gutter column sits inside ``scroll.region`` but outside
    the viewport, a passing run cannot distinguish "the guard rejected it" from
    "nothing was ever delivered".
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _nested_screen_agg())
            _show_table(screen)
            await pilot.pause()

            scroll = screen._scroll
            viewport = scroll.scrollable_content_region
            gutter_x = viewport.x + viewport.width
            row_y = _row_y(screen, 0)

            # Precondition 1: the body does NOT overflow, so no bar is drawn and
            # the gutter is empty — events there really reach the screen.
            assert (
                scroll.max_scroll_y == 0
            ), "fixture overflows: a drawn bar would capture the gutter"
            # Precondition 2: the gutter column is inside the container's region
            # (what the R2 mutant guards with) but outside the viewport (what the
            # shipped guard uses), so the guard is genuinely exercised rather
            # than trivially satisfied.
            assert scroll.region.contains(gutter_x, row_y), "gutter is not inside scroll.region"
            assert not viewport.contains(gutter_x, row_y), "gutter is inside the viewport"
            # Precondition 3: the coordinate is over a visible, EXPANDABLE row,
            # so a click there would toggle it under the mutant.
            assert screen._layout.session_rows[0].expandable, "fixture: row 0 is not expandable"

            await pilot.hover(screen, offset=(gutter_x, row_y))
            await pilot.pause()
            assert screen._hover is None, "the empty gutter took a hover"
            assert str(screen.styles.pointer) == "default", "the gutter promised a click"

            before = set(screen._expanded)
            await pilot.click(screen, offset=(gutter_x, row_y))
            await pilot.pause()
            assert screen._expanded == before, "the empty gutter took a click"

    asyncio.run(run())


# -- text selection -----------------------------------------------------------


def test_a_drag_across_the_rows_selects_them_and_copies_them():
    """The drag/ctrl+c affordance the widget swap could silently have taken.

    A line-API widget returns no ``Text``/``Content`` from ``_render()``, so
    ``Widget.get_selection`` extracts nothing; and because the compositor
    recovers the pointer's CONTENT offset from an ``"offset"`` style meta on the
    rendered segments, a strip returned without one makes the drag resolve a
    start and no end — textually "select everything from here to the end". Both
    halves are asserted here because both were broken at once: the selection
    must PAINT (a drag that highlights nothing reads as "the drag did nothing")
    and ``ctrl+c`` must hand over those rows' text, which is the only way a
    session name or a figure leaves this read-only screen.

    The report is the ``Static`` body's replacement, so this is a regression
    pin rather than a new feature: on the tree before the swap a drag over these
    same three rows paints them and copies their text.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(120, 45)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            first = 1
            x = _row_x(screen)
            top = _row_y(screen, first)
            bottom = _row_y(screen, first + 2)
            expected = screen.render_lines_for_test()

            def painted() -> dict[int, tuple[tuple[str, str | None], ...]]:
                """Row -> (text, background) per segment: what the eye actually sees.

                Iterated rather than indexed: ``Strip.__getitem__`` is a CROP, not
                a sequence access, so ``strip[0]`` is not the first segment.
                """
                out: dict[int, tuple[tuple[str, str | None], ...]] = {}
                for index, strip in enumerate(app.screen._compositor.render_strips()):
                    out[index] = tuple(
                        (segment.text, str(segment.style.bgcolor) if segment.style else None)
                        for segment in strip
                        if segment.text
                    )
                return out

            before = painted()
            expanded_before = set(screen._expanded)

            screen._forward_event(_press(screen, x, top))
            await pilot.pause()
            for step in range(1, 4):
                screen._forward_event(
                    _drag_to(screen, x + 20 * step, top + (bottom - top) * step // 3)
                )
                await pilot.pause()
            screen._forward_event(_release(screen, x + 60, bottom))
            await pilot.pause()

            selected = app.screen.get_selected_text()
            assert selected, "the drag selected nothing — the body paints no selection"
            # The drag starts mid-row, so the FIRST row is only partially covered;
            # the row in the middle is fully inside it, so its whole composed text
            # must be in the copy. Asserted on the text the user reads, taken from
            # the same renderer the tests always read the report through.
            rows = [expected[screen._layout.session_first_line + i] for i in (1, 2, 3)]
            assert (
                rows[1].strip() in selected
            ), f"the dragged row is not in the copied text: {selected[:120]!r}"
            assert rows[0].strip()[-24:] in selected, "the first dragged row's tail is missing"
            assert selected.count("\n") >= 2, f"fewer than three rows copied: {selected!r}"
            # ...and no more than the rows the drag covered: an unresolved end
            # offset reads as "select to the end of the report" (~40 lines here),
            # which every other assertion in this test would happily accept.
            assert (
                selected.count("\n") <= 4
            ), f"the drag selected past the rows it covered: {selected.count(chr(10))} lines"

            dragged = {screen._scroll.scrollable_content_region.y + first - 1 + i for i in range(4)}
            changed = {index for index, bg in painted().items() if before[index] != bg}
            assert (
                changed & dragged
            ), f"the drag painted no selection on the rows it covered: changed={sorted(changed)}"

            # The gesture must not be read as a row action: no expansion.
            assert screen._expanded == expanded_before, "a drag toggled a row"

            # And the paint goes away with the selection, so the highlight is the
            # selection and not a side effect of moving the pointer.
            app.screen.clear_selection()
            await pilot.pause()
            after_clear = painted()
            assert all(
                after_clear[index] == before[index] for index in dragged
            ), "clearing the selection did not repaint the rows it had highlighted"

    asyncio.run(run())


# -- paint == copy ------------------------------------------------------------


def _selection_ground(app: OperatorApp) -> str:
    """The background the selection band paints with, as the strips report it."""
    return str(app.screen.get_component_rich_style("screen--selection").bgcolor)


def _painted_by_selection(app: OperatorApp) -> dict[int, str]:
    """Screen row -> the text the SELECTION ground is painted on, per row.

    Read off the compositor's own strips, so it is the frame the reader sees
    rather than what the widget intended: the concatenated text of the segments
    whose background is the selection ground.
    """
    ground = _selection_ground(app)
    rows: dict[int, str] = {}
    for index, strip in enumerate(app.screen._compositor.render_strips()):
        text = "".join(
            segment.text
            for segment in strip
            if segment.text and segment.style and str(segment.style.bgcolor) == ground
        )
        if text:
            rows[index] = text
    return rows


def test_the_selection_band_paints_the_characters_ctrl_c_copies():
    """Paint and copy must describe the SAME characters — wide glyphs included.

    The offsets a drag produces are CHARACTER offsets: ``Strip.apply_offsets``
    advances ``x`` by ``len(segment.text)``, the compositor counts characters
    into the segment the pointer landed on, and ``Selection.extract`` (what
    ctrl+c ends up with) slices the plain string with the same numbers. Reading
    them as cell columns and converting again is the identity only while every
    glyph is one cell wide — true of the operator's real ledger and of every
    other fixture here, which is exactly why the drift hid: on a line whose
    prefix carries a 2-cell glyph the band highlights a different range than the
    copy. A session title is free text, so a wide-glyph label is reachable even
    though the ledger the PR was measured on has none.
    """

    async def run():
        app = _app()
        agg = _tall_report_agg()
        # The first table row (the priciest root). Renamed rather than added so
        # the row keeps its place, its widths and its sort order.
        # setattr, not attribute access: the fixture hangs the labels on the
        # aggregate the same way (`analytics_panel` reads them with `getattr`), so
        # this is the shape the report actually consumes.
        names = dict(getattr(agg, "session_names", {}))
        names["root00000000"] = "日本語セッション"
        setattr(agg, "session_names", names)
        async with app.run_test(size=(120, 45)) as pilot:
            screen = await _push(pilot, app, agg)
            _show_table(screen)
            await pilot.pause()
            y = _row_y(screen, 0)
            row = screen.render_lines_for_test()[screen._layout.session_first_line]
            assert "語" in row, f"fixture row has no wide glyph: {row!r}"
            start = _row_x(screen) + 2
            screen._forward_event(_press(screen, start, y))
            await pilot.pause()
            for step in (1, 2, 3):
                screen._forward_event(_drag_to(screen, start + 10 * step, y))
                await pilot.pause()
            screen._forward_event(_release(screen, start + 30, y))
            await pilot.pause()

            copied = app.screen.get_selected_text()
            assert copied, "the drag selected nothing"
            painted = _painted_by_selection(app).get(y, "")
            assert painted, "the drag painted no selection band"
            assert (
                "語" in copied or "語" in painted
            ), f"the drag missed the wide glyph, so this proves nothing: {copied!r}"
            assert painted == copied, (
                f"the band highlights {painted!r} while ctrl+c copies {copied!r} — "
                f"the frame describes different characters than the clipboard"
            )

    asyncio.run(run())


def test_a_selection_from_the_report_top_bands_only_the_selected_rows():
    """A drag that starts on the report's FIRST line must band only its own rows.

    ``/analytics`` opens at the report top, so body line 0 is the line under the
    reader's pointer — and a drag upward is normalised to start there, so this is
    the first gesture the screen offers. The selection style is applied per line
    here (this widget knows the line index; a single-line visual does not), but
    the generic visual path re-resolves the selection against the one line it is
    handed, which reads body line 0's span for EVERY strip: with a selection
    starting at line 0 the ground then covered all 28 visible rows / 1394 cells
    where 4 rows were selected (measured, 120x45). The frame is what lies —
    the copy was always right — so the copy cannot be the guard here.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(120, 45)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            screen._scroll.scroll_to(y=0, animate=False)
            await pilot.pause()
            content = screen._scroll.scrollable_content_region
            x = _row_x(screen)
            screen._forward_event(_press(screen, x, content.y))
            await pilot.pause()
            for step in (1, 2, 3):
                screen._forward_event(_drag_to(screen, x + 40, content.y + step))
                await pilot.pause()
            screen._forward_event(_release(screen, x + 40, content.y + 3))
            await pilot.pause()

            assert app.screen.get_selected_text(), "the drag selected nothing"
            selected_rows = set(range(content.y, content.y + 4))
            banded = set(_painted_by_selection(app))
            assert banded, "no row carries the selection ground, so this proves nothing"
            assert banded <= selected_rows, (
                f"a selection from body line 0 banded rows outside it: "
                f"{sorted(banded - selected_rows)} (banded {sorted(banded)})"
            )

    asyncio.run(run())
