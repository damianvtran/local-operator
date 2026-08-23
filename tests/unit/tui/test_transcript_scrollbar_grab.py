"""Near-miss grab on the 1-cell transcript scrollbar (TranscriptScreen).

The transcript's vertical scrollbar is deliberately **1 cell wide** so the
reserved gutter never reads as a right-hand border (see the `TranscriptView`
rules in ``local_operator.tcss``, D4/D21/D27). A 1-cell mouse target is easy to
miss by a column, and a miss used to land on selectable content: Textual's
``Screen._forward_event`` armed a text selection there, and dragging toward an
edge tripped selection auto-scroll — the reported "it scrolls while I drag but
leaves a messy highlight" bug.

``TranscriptScreen._forward_event`` now redirects a ``MouseDown`` that lands one
cell left of the bar (inside the thumb's vertical extent) onto the scrollbar
column BEFORE the base class can arm a selection, so the near-miss becomes a
real thumb grab. These tests pin that behaviour directly against the real
``OperatorApp`` — the redirect, the drag-scrolls-without-highlighting path, the
untouched true-thumb grab, and the two guards that keep the pad from stealing
ordinary content clicks or firing when there is no scrollbar.

Why events are posted rather than driven through ``pilot.mouse_down``: the bar
sits at ``x == screen.width`` (the gutter is the last column), which
``Pilot._post_mouse_events`` rejects as out of the visible region. The
production path is ``Screen._forward_event`` regardless, so calling it directly
exercises exactly the code under test. ``app.mouse_position`` is set before the
down because ``App.capture_mouse`` records it as the grab origin (scrollbar.py
``_on_mouse_capture``), and a zero origin yields a falsy ``grabbed`` offset that
short-circuits ``ScrollBar._on_mouse_move``.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.geometry import Offset
from textual.scrollbar import ScrollBar

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import RichBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _mouse_down(x: int, y: int) -> events.MouseDown:
    """A left-button ``MouseDown`` at screen ``(x, y)`` with no widget bound.

    ``Screen._forward_event`` resolves the target from the coordinate, so the
    ``widget`` is left ``None`` exactly as a real terminal event arrives.
    """
    return events.MouseDown(
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
        style=None,
        widget=None,
    )


def _fill(view: TranscriptView, n: int = 60) -> None:
    """Overflow the transcript so its vertical scrollbar is shown."""
    for i in range(n):
        view.append_block(RichBlock(f"line {i:02d} lorem ipsum dolor sit amet consectetur"))


def _scrollbar_region(app: OperatorApp, view: TranscriptView):
    """The compositor region of ``view``'s vertical scrollbar (never hardcoded).

    ``x`` depends on the terminal size and the reserved gutter, so tests compute
    it here rather than assuming 78 — the whole point of the redirect is that the
    bar's column is wherever the layout put it.

    Tests below drive the bar with ``scroll_y == 0`` so the thumb is pinned to
    the TOP of the track; the grab then targets ``region.y`` (the thumb's first
    cell) rather than a fixed mid-track offset. This is deliberate: the
    transcript's track height is not stable across apps created in one process
    (welcome/dock layout state leaks between ``run_test`` instances, shrinking
    the view by a few rows), so a fixed offset like ``region.y + 5`` can land in
    the empty track BELOW a short thumb — a page-scroll, not a grab. The thumb's
    top cell is always present whatever the track height.
    """
    return app.screen._compositor.find_widget(view.vertical_scrollbar).region


@pytest.mark.asyncio
async def test_near_miss_does_not_arm_selection_and_grabs_the_bar() -> None:
    """A mousedown one cell left of the bar grabs the thumb, no selection armed.

    The direct regression assertion: ``_select_state`` stays ``None`` and
    ``_selecting`` stays ``False`` (so no auto-scroll highlight can follow), and
    the ScrollBar captures the mouse — i.e. the near-miss became a real grab.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        _fill(view)
        await pilot.pause()
        # Thumb pinned to the top of the track (see _scrollbar_region on why a
        # fixed mid-track offset is unstable across in-process apps).
        view.scroll_y = 0
        await pilot.pause()

        region = _scrollbar_region(app, view)
        # The bar stays exactly 1 cell — the fix widens the INPUT hit area only,
        # never the reserved gutter (D4/D21/D27).
        assert view.scrollbar_size_vertical == 1

        near_miss_x = region.x - 1
        thumb_y = region.y  # the thumb's top cell
        app.mouse_position = Offset(near_miss_x, thumb_y)
        app.screen._forward_event(_mouse_down(near_miss_x, thumb_y))
        await pilot.pause()

        assert app.screen._select_state is None
        assert app.screen._selecting is False
        assert type(app.mouse_captured).__name__ == "ScrollBar"


@pytest.mark.asyncio
async def test_near_miss_drag_scrolls_without_highlighting() -> None:
    """After the redirected grab, dragging scrolls the transcript and never
    leaves a selection — the exact symptom the bug produced is gone."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        _fill(view)
        await pilot.pause()
        view.scroll_y = 0
        await pilot.pause()

        region = _scrollbar_region(app, view)
        near_miss_x = region.x - 1
        thumb_y = region.y
        app.mouse_position = Offset(near_miss_x, thumb_y)
        app.screen._forward_event(_mouse_down(near_miss_x, thumb_y))
        await pilot.pause()

        captured = app.mouse_captured
        assert isinstance(captured, ScrollBar)

        start_y = view.scroll_y
        # animate=not supports_smooth_scrolling: force the non-animated path so
        # the drag's ScrollTo applies synchronously within the test.
        app.supports_smooth_scrolling = True
        move = events.MouseMove(
            x=region.x,
            y=thumb_y + 8,
            delta_x=0,
            delta_y=8,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=region.x,
            screen_y=thumb_y + 8,
            style=None,
            widget=captured,
        )
        await captured._on_mouse_move(move)
        for _ in range(4):
            await pilot.pause()

        assert view.scroll_y > start_y
        assert app.screen.selections == {}


@pytest.mark.asyncio
async def test_true_thumb_click_still_grabs() -> None:
    """A mousedown ON the scrollbar column is unaffected by the redirect.

    Guards against the redirect breaking the direct path it is supposed to leave
    alone: a click at exactly ``region.x`` must still capture the ScrollBar and
    arm no selection.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        _fill(view)
        await pilot.pause()
        view.scroll_y = 0
        await pilot.pause()

        region = _scrollbar_region(app, view)
        thumb_y = region.y
        app.mouse_position = Offset(region.x, thumb_y)
        app.screen._forward_event(_mouse_down(region.x, thumb_y))
        await pilot.pause()

        assert app.screen._select_state is None
        assert type(app.mouse_captured).__name__ == "ScrollBar"


@pytest.mark.asyncio
async def test_two_cells_left_still_selects_content() -> None:
    """The pad is exactly one column: a mousedown two cells left of the bar is
    ordinary content and still arms a text selection.

    This is the guard that the forgiveness cannot swallow real clicks — raising
    ``SCROLLBAR_GRAB_PAD`` without a fresh geometry check would break it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        _fill(view)
        await pilot.pause()
        view.scroll_y = 0
        await pilot.pause()

        region = _scrollbar_region(app, view)
        content_x = region.x - 2
        content_y = region.y
        app.mouse_position = Offset(content_x, content_y)
        app.screen._forward_event(_mouse_down(content_x, content_y))
        await pilot.pause()

        # Content selection armed, scrollbar NOT captured.
        assert app.screen._select_state is not None
        assert type(app.mouse_captured).__name__ != "ScrollBar"


@pytest.mark.asyncio
async def test_short_transcript_passes_through() -> None:
    """With no overflow there is no scrollbar, so the redirect never fires and a
    mousedown near the right edge is left completely untouched."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        # A single short block: nothing to scroll.
        view.append_block(RichBlock("just one short line"))
        await pilot.pause()

        assert view.show_vertical_scrollbar is False

        # Near the right edge, where the bar would be if it existed.
        edge_x = view.region.right - 1
        edge_y = view.region.y + 1
        app.mouse_position = Offset(edge_x, edge_y)
        app.screen._forward_event(_mouse_down(edge_x, edge_y))
        await pilot.pause()

        # No grab: the loop body was skipped and the event passed through
        # unchanged (no ScrollBar to capture).
        assert type(app.mouse_captured).__name__ != "ScrollBar"
