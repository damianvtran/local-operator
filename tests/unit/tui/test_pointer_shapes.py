"""Pointer shapes over clickable surfaces: the hand arrives with the hover.

Textual 8.2.8 owns the whole mechanism — a ``pointer`` CSS rule (or inline
style), ``Screen.update_pointer_shape()`` walking the hovered widget's
ancestors on every mouse move, and ``App._set_pointer_shape`` emitting OSC 22,
which ghostty implements and an older terminal ignores harmlessly. The app
simply never told it which surfaces were clickable, so every hover kept the
default arrow; the user reads that as "the cursor doesn't work".

These tests drive the REAL mouse-move path through ``pilot.hover`` (not a
direct style read) so what they assert is what the terminal would be told:
``Screen._pointer_shape`` after the move. Tool cards carry a static rule (the
whole row is the click target); picker rows set the inline rule from the row
index the move landed on, which is why the picker also has to give the shape
BACK when the pointer sits on a non-row (the overflow count) — a shape that
never resets is a cursor the user stops trusting.
"""

from __future__ import annotations

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import CommandPicker
from local_operator.tui.widgets.toast import Toast
from local_operator.tui.widgets.tool_card import ToolCard
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_command_picker import PickerHarnessApp


@pytest.mark.asyncio
async def test_hovering_a_tool_row_sets_the_hand_pointer() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        card = ToolCard("call-1", "bash", {"command": "ls"}, None)
        card.mark_done("done")
        app._append_block(card)
        await pilot.pause()

        landed = await pilot.hover(card)
        assert landed, "hover missed the card"
        await pilot.pause()
        assert app.screen._pointer_shape == "pointer"

        # The transcript's own ground row (above the card) gives the arrow
        # back: leaving a clickable row must reset the shape, or the hand
        # stops meaning anything.
        transcript = app._transcript_view()
        landed = await pilot.hover(transcript, offset=(5, 0))
        assert landed
        await pilot.pause()
        assert app.screen._pointer_shape == "default"


@pytest.mark.asyncio
async def test_picker_rows_take_and_release_the_hand_pointer() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        picker = app.editor.picker
        assert picker.is_open()
        assert isinstance(picker, CommandPicker)
        assert picker.region.height >= 9  # 8 suggestion rows + the overflow count

        # A suggestion row (row 0): the hand.
        landed = await pilot.hover(picker, offset=(10, 0))
        assert landed
        await pilot.pause()
        assert app.screen._pointer_shape == "pointer", "row hover kept the arrow"

        # The overflow-count row (last row of the picker) is not a click
        # target: the inline rule falls back and so does the shape.
        landed = await pilot.hover(picker, offset=(10, picker.region.height - 1))
        assert landed
        await pilot.pause()
        assert (
            app.screen._pointer_shape == "default"
        ), f"non-row area kept the hand (inline rule now {picker.styles.pointer!r})"

        # Closing the picker under the still pointer must also release the
        # hand; no mouse move follows a programmatic/keyboard close.
        landed = await pilot.hover(picker, offset=(10, 0))
        assert landed
        await pilot.pause()
        assert app.screen._pointer_shape == "pointer"
        picker.close()
        await pilot.pause()
        assert app.screen._pointer_shape == "default"


@pytest.mark.asyncio
async def test_dismissing_a_toast_releases_the_hand_pointer() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        toast = app.query_one(Toast)
        toast.show("Saved", duration_ms=60_000)
        await pilot.pause()

        landed = await pilot.hover(toast)
        assert landed
        await pilot.pause()
        assert app.screen._pointer_shape == "pointer"

        toast.dismiss_toast()
        await pilot.pause()
        assert app.screen._pointer_shape == "default"
