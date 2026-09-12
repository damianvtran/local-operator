"""One viewport move writes ONE complete frame: the ``/analytics`` scroll invariant.

The operator's report: on a scroll the screen "seems that not the entire frame
gets updated at once and there's an update that passes down one line at a time
through the TUI". That is exactly what was happening, and it had TWO causes in
one place — the pagination keys.

1. ``action_page_up``/``action_page_down``/``action_scroll_home``/
   ``action_scroll_end`` called Textual's ``scroll_page_*``/``scroll_home``/
   ``scroll_end`` with their default ``animate=True``. The body is a line-API
   ``ReportView`` whose ``render_line(y)`` serves ``scroll_offset.y + y``:
   shifting the offset by ONE line changes EVERY viewport row, so each eased step
   of the animation was a full-viewport rewrite.
2. The four bindings were NOT ``priority=True``, unlike the arrow keys beside
   them. Focus sits on the scroller, whose ``ScrollView`` base has its own
   ``pageup``/``pagedown``/``home``/``end`` bindings, so a real key press went to
   ``Widget.action_page_down`` (again an ``animate=True`` default) and the screen
   actions above were unreachable on any report tall enough to scroll. Animating
   them alone would have changed nothing a user could press.

Measured at 120x45 on ``_tall_report_agg()`` (body region 102x29,
``max_scroll_y`` 31), one key press, before either half of the fix: ``pagedown``
wrote **20 compositor frames, every one of them all 29 body rows**, with the
offset walking 3 -> 5 -> 7 -> ... -> 29 one line at a time; ``pageup`` 20,
``end`` 28, ``home`` 25. After it: one frame per key, covering all 29 rows and
painted from the destination offset.

**These tests assert the work, not the clock** — the same discipline as
``test_analytics_repaint.py`` beside them, for the same reason (AGENTS.md
"Calibrate ceilings from CI": a wall-clock or CPU ceiling calibrated on this
laptop is a bet on machine load, and this screen already has a CPU-bound rebuild
path that makes such a bound useless). The invariant here is structural and
holds at any speed:

* one viewport-move gesture writes exactly ONE report frame;
* that frame covers the WHOLE body viewport, row for row — "fewer frames" is not
  the claim, "a complete frame" is, so a fix that dropped the animation but left
  the body painting partially would fail here;
* the offset is already at its target IN that frame (nothing is left easing);
* no animation is left registered on the body afterwards.

The instrument is the compositor's own ``render_update``, because the body is a
single widget: a per-row or per-line counter would see one dirty widget either
way and could not tell 1 frame from 20. Frames written outside the body's region
(the surrounding card, the hint row) are ignored — they are present on both the
fixed and the unfixed tree and are not what the gesture costs.

Mutation check (scratch: the file run against ``origin/main``'s tree with
``PYTHONPATH`` pointed at it) — 8 of the 9 cases fail there, the four action-path
ones and the four key-path ones, reporting 17-28 report frames each. The ninth is
the wheel cell, which passes on both trees by design and whose own mutation
check is documented there.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from textual import events
from textual._compositor import ChopsUpdate, Compositor

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.analytics_panel import _SCROLLBAR_GUTTER
from tests.unit.tui.test_analytics_panel import _push, _tall_report_agg
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: (gesture name, the action method, the key a real terminal sends)
_GESTURES = [
    ("page_down", "action_page_down", "pagedown"),
    ("page_up", "action_page_up", "pageup"),
    ("home", "action_scroll_home", "home"),
    ("end", "action_scroll_end", "end"),
]


def _app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


def _viewport_rows(scroll: Any) -> set[int]:
    """The body's viewport, as absolute screen rows.

    ``ReportView`` is a line API: ``render_line(y)`` serves ``scroll_offset.y + y``,
    so the compositor's chop rows and the widget's screen rows share one
    coordinate space and can be compared directly.
    """
    return set(range(scroll.region.y, scroll.region.y + scroll.region.height))


class _BodyFrames:
    """Every compositor frame written during the block, split body vs scrollbar.

    ``render_update`` is the single call that turns a dirty widget into an
    update, so counting its invocations counts frames. A frame counts as a
    report frame when its rows fall inside the body viewport AND its column span
    reaches left of the scrollbar column: the thumb repaint arrives as its own
    one-column chop, and counting that as a second viewport write would make
    every correct gesture look doubled.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, scroll: Any) -> None:
        self.scroll = scroll
        self.viewport = _viewport_rows(scroll)
        # ``_SCROLLBAR_GUTTER`` is the width the body reserves for its thumb
        # (``_card_width`` subtracts the same constant), so the last column of
        # the body region is chrome, not report.
        self.scrollbar_column = scroll.region.x + scroll.region.width - _SCROLLBAR_GUTTER
        self.frames: list[dict[str, Any]] = []
        original = Compositor.render_update

        def patched(comp, full=False, screen_stack=None, simplify=False):
            out = original(comp, full=full, screen_stack=screen_stack, simplify=simplify)
            record: dict[str, Any] = {"kind": "other"}
            if isinstance(out, ChopsUpdate):
                spans = list(out.spans)
                record = {
                    "kind": "chops",
                    "rows": {y for y, _, _ in spans},
                    "left": min(x1 for _, x1, _ in spans),
                    "right": max(x2 for _, _, x2 in spans),
                    "cells": sum(max(0, x2 - x1) for _, x1, x2 in spans),
                    # Taken AT PAINT TIME: the offset the frame was painted from,
                    # so "the destination landed in this frame" is a fact about
                    # the frame rather than about whatever the offset became
                    # later.
                    "scroll_y": float(self.scroll.scroll_offset.y),
                }
            self.frames.append(record)
            return out

        monkeypatch.setattr(Compositor, "render_update", patched)

    def _chops_inside_the_viewport(self) -> list[dict[str, Any]]:
        return [f for f in self.frames if f["kind"] == "chops" and f["rows"] <= self.viewport]

    def content(self) -> list[dict[str, Any]]:
        """Frames that painted report rows, rather than the scrollbar thumb."""
        return [f for f in self._chops_inside_the_viewport() if f["left"] < self.scrollbar_column]

    def scrollbar(self) -> list[dict[str, Any]]:
        """Frames confined to the body's scrollbar gutter."""
        return [f for f in self._chops_inside_the_viewport() if f["left"] >= self.scrollbar_column]

    def assert_no_partial_repaint(self) -> None:
        """Any extra body chop must be the thumb, never part of the report.

        This is the half of "fewer frames" that matters: the fix must not trade
        an animated walk for a partial paint. A chop inside the viewport that is
        not the whole viewport and not confined to the gutter is exactly a torn
        frame, however few of them there are.
        """
        for frame in self.scrollbar():
            assert (
                self.scrollbar_column <= frame["left"]
                and frame["right"] <= self.scrollbar_column + _SCROLLBAR_GUTTER
            ), (
                "a body frame repainted part of the report instead of the whole "
                f"viewport: columns {frame['left']}..{frame['right']} of a body region "
                f"ending at column {self.scrollbar_column}. All frames: {self.describe()}"
            )

    def body(self) -> list[dict[str, Any]]:
        return [f for f in self.frames if f["kind"] == "chops" and f["rows"] <= self.viewport]

    def describe(self) -> str:
        """The frames, as a failure message a reader can act on."""
        return "; ".join(
            f"{f['kind']} rows={sorted(f['rows'])[:3]}... scroll_y={f.get('scroll_y')}"
            for f in self.frames
        )


def _target(scroll: Any, gesture: str, before: float) -> float:
    """Where the gesture must land, spelled from the geometry not the code."""
    page = scroll.scrollable_content_region.height
    if gesture == "page_down":
        return float(min(before + page, scroll.max_scroll_y))
    if gesture == "page_up":
        return float(max(0, before - page))
    if gesture == "home":
        return 0.0
    return float(scroll.max_scroll_y)


def _start_offset(scroll: Any, gesture: str) -> float:
    """A start offset with the whole travel ahead of the gesture.

    Opposite ends, so the ``home`` and ``end`` cases are real moves: a no-op
    gesture would write no frame and the assertions below would be vacuous,
    which is the failure mode AGENTS.md records for a cell that passes because
    the thing it checks never happened.
    """
    return float(scroll.max_scroll_y) if gesture in ("page_up", "home") else 0.0


@pytest.mark.parametrize("gesture, action, _key", _GESTURES)
def test_a_viewport_move_writes_one_complete_frame(gesture, action, _key, monkeypatch):
    """The gesture's whole cost is ONE frame, and it covers the whole viewport."""

    async def run():
        app = _app()
        async with app.run_test(size=(120, 45)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            scroll = screen._scroll
            assert scroll.max_scroll_y > 0, "fixture is not taller than the viewport"
            assert scroll.size.height > 1, "a one-row viewport cannot show a partial frame"

            scroll.scroll_to(y=_start_offset(scroll, gesture), animate=False)
            await pilot.pause()
            await pilot.pause()
            before = float(scroll.scroll_offset.y)
            # The fixture's starting state, asserted rather than assumed: on the
            # unfixed tree a leftover easing frame would otherwise be counted as
            # the gesture's, and the test would report the wrong cause.
            assert before == _start_offset(scroll, gesture), (
                "the setup scroll had not landed, so this frame count would not be " "the gesture's"
            )
            assert scroll.size.height < scroll.virtual_size.height, "body does not scroll"

            frames = _BodyFrames(monkeypatch, scroll)
            getattr(screen, action)()
            # Both, in this order: ``wait_for_scheduled_animations`` drives any
            # animation the gesture scheduled to COMPLETION without donating
            # wall-clock time, and the idle pause lets the resulting paint land.
            # Neither alone is enough — the settle helper does not itself force a
            # refresh after an unanimated move, and the idle pause would leave an
            # animation mid-flight and under-count its frames.
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

            frames.assert_no_partial_repaint()
            content = frames.content()
            assert len(content) == 1, (
                f"{gesture} wrote {len(content)} report frames, not one — the viewport "
                f"move is being eased, and every eased step repaints every row. "
                f"All frames: {frames.describe()}"
            )
            written = content[0]["rows"]
            assert written == frames.viewport, (
                f"the single frame did not cover the whole viewport: wrote "
                f"{sorted(written)} of {sorted(frames.viewport)}"
            )
            expected = _target(scroll, gesture, before)
            assert content[0]["scroll_y"] == expected, (
                f"the frame was painted from y={content[0]['scroll_y']} rather than the "
                f"destination y={expected}: the move is still easing"
            )
            assert content[0]["cells"] == scroll.region.width * len(
                frames.viewport
            ), "the single frame did not paint the whole body width"
            assert float(scroll.scroll_offset.y) == expected, "the offset did not land"
            assert not app.animator.is_being_animated(
                scroll, "scroll_y"
            ), "an animation is still in flight on scroll_y after a settled move"

    asyncio.run(run())


@pytest.mark.parametrize("gesture, _action, key", _GESTURES)
def test_the_key_binding_writes_the_same_single_frame(gesture, _action, key, monkeypatch):
    """The BINDING path writes one frame too — the same claim via ``pilot.press``.

    Driven by the KEY rather than the action method, because the key is what the
    operator presses: ``pagedown`` reaches the action through a ``Binding``, and
    a fix applied to the method is only half the surface if the binding is ever
    given its own route. The frame assertions are the action-path ones, so both
    routes are pinned to the same behaviour.

    Note what is NOT asserted here. "The offset has reached its destination after
    one ``pilot.pause()``" looks like the sharpest form of "nothing is easing",
    and it is not: Textual's idle wait drives the animation's ticks to completion
    as fast as the loop will take them, so ONE pause already lands the target on
    the unfixed tree (measured: the 20-28-frame animations complete inside a
    single pause). A test written that way passes either way — decoration. What
    separates the trees is that the unfixed one paints the whole viewport once
    per eased step, which is the frame count below.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(120, 45)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            scroll = screen._scroll
            assert scroll.max_scroll_y > 0, "fixture is not taller than the viewport"

            scroll.scroll_to(y=_start_offset(scroll, gesture), animate=False)
            await pilot.pause()
            await pilot.pause()
            before = float(scroll.scroll_offset.y)
            assert before == _start_offset(scroll, gesture), "the setup scroll had not landed"

            frames = _BodyFrames(monkeypatch, scroll)
            await pilot.press(key)
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

            frames.assert_no_partial_repaint()
            content = frames.content()
            assert len(content) == 1, (
                f"the {key} binding wrote {len(content)} report frames, not one. "
                f"All frames: {frames.describe()}"
            )
            expected = _target(scroll, gesture, before)
            assert content[0]["rows"] == frames.viewport, (
                f"the single frame did not cover the whole viewport: wrote "
                f"{sorted(content[0]['rows'])} of {sorted(frames.viewport)}"
            )
            assert content[0]["scroll_y"] == expected, (
                f"the frame was painted from y={content[0]['scroll_y']} rather than the "
                f"destination y={expected}: the move is still easing"
            )
            assert float(scroll.scroll_offset.y) == expected, "the offset did not land"
            assert not app.animator.is_being_animated(scroll, "scroll_y")
            registered = [k for k in app.animator._animations if k[0] == id(scroll)]
            assert registered == [], (
                "a viewport move left an animation registered for the body on a screen "
                f"whose every mover is supposed to be unanimated: {registered}"
            )

    asyncio.run(run())


def test_a_wheel_notch_writes_one_complete_frame(monkeypatch):
    """The wheel stays atomic: one notch, one complete body frame, no easing.

    Folded in with the four bindings above because they are ONE claim — no
    viewport move on this screen is animated. The wheel cells already held
    before the fix (``_on_mouse_scroll_down`` passes ``animate=False``), so this
    test does not discriminate the tearing fix; it is the guard against a
    screen-level wheel handler being added later with an animation, which is the
    shape `AGENTS.md` records as the trap on ``/settings``.

    Driven through ``App.on_event`` with a real ``MouseScrollDown``, the path a
    terminal uses: Textual stops the event on the container while it can still
    scroll, so this never reaches a screen-level handler and the frame count is
    the container's own behaviour, which is exactly what must stay unanimated.

    Mutation check (scratch, not committed): patching the container's wheel
    handler to ``scroll_to(y=self.scroll_y + 12, animate=True)`` — a notch that
    moves a distance and eases it — writes **5 report frames** where this cell
    asserts 1, so the cell can go red. It cannot detect a ONE-row animated step:
    the animator's first tick already lands a single row, measured at 1 frame with
    ``animate=True``, so the cell makes no claim about that shape.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(120, 45)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            scroll = screen._scroll
            assert scroll.max_scroll_y > 0, "fixture is not taller than the viewport"

            scroll.scroll_to(y=0, animate=False)
            await pilot.pause()
            await pilot.pause()
            assert float(scroll.scroll_offset.y) == 0.0, "the setup scroll had not landed"
            x, y = scroll.region.x + 4, scroll.region.y + 4

            frames = _BodyFrames(monkeypatch, scroll)
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
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

            frames.assert_no_partial_repaint()
            content = frames.content()
            assert len(content) == 1, (
                f"one wheel notch wrote {len(content)} report frames, not one. "
                f"All frames: {frames.describe()}"
            )
            assert content[0]["rows"] == frames.viewport, (
                "the wheel frame did not cover the whole viewport: wrote "
                f"{sorted(content[0]['rows'])} of {sorted(frames.viewport)}"
            )
            assert content[0]["scroll_y"] > 0, "the wheel did not scroll, so this proves nothing"
            assert not app.animator.is_being_animated(scroll, "scroll_y")

    asyncio.run(run())
