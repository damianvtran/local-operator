"""A scheduled scroll is not complete just because the event loop was idle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest

from local_operator.tui.widgets.subagent_view import SubagentView
from tests.unit.tui.test_subagent_view import _wait_for_scroll_animation


class ScrollPilot:
    """Forced ordering model, not a claim about the historical CI scheduler.

    Down's handler is still queued while an up animation produces another frame.
    These are rows, matching Textual's scroll offset, not pixel coordinates.
    """

    def __init__(self) -> None:
        self.direction = "up"
        self.target = 54
        self.body = SimpleNamespace(
            scroll_y=54.0,
            scroll_target_y=54.0,
            scroll_offset=SimpleNamespace(y=54),
            max_scroll_y=54,
            virtual_size=SimpleNamespace(height=79),
            app=SimpleNamespace(animator=SimpleNamespace(is_being_animated=lambda *_: self.active)),
            note_user_scroll=lambda: None,
            scroll_page_up=lambda: self.schedule("up"),
            scroll_page_down=lambda: self.schedule("down"),
        )
        self.active = False

    def schedule(self, direction: str) -> None:
        self.direction = direction
        self.target = max(0, min(54, self.body.scroll_y + (25 if direction == "down" else -25)))
        self.active = True

    async def click(self, direction: str) -> None:
        # Execute the actual production arrow handler, with just the animated
        # body replaced by the deterministic scheduler below (no app repetition).
        SubagentView._scroll_body(
            cast(SubagentView, SimpleNamespace(_body=self.body)), down=direction == "down"
        )

    async def pause(self) -> None:
        # An idle/pause need not dispatch the newly queued scroll or complete
        # the existing animation: recreate the reported 38 > 42 predicate.
        self.body.scroll_y = 42 if self.direction == "up" else 38
        self.body.scroll_offset.y = int(self.body.scroll_y)
        self.body.scroll_target_y = 29

    async def wait_for_scheduled_animations(self) -> None:
        await asyncio.sleep(0)
        self.body.scroll_y = self.body.scroll_target_y = self.target
        self.body.scroll_offset.y = self.target
        self.active = False


@pytest.mark.asyncio
async def test_forced_order_reproduces_original_direction_failure() -> None:
    pilot = ScrollPilot()
    await pilot.click("up")
    await pilot.pause()
    lifted = pilot.body.scroll_offset.y
    await pilot.click("down")
    await pilot.pause()
    assert (pilot.body.scroll_offset.y, lifted) == (38, 42)
    with pytest.raises(AssertionError):
        assert pilot.body.scroll_offset.y > lifted


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "noop", "inverted"])
async def test_completion_wait_keeps_both_direction_contracts(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SubagentView._scroll_body
    if mode != "normal":

        def broken_handler(view: SubagentView, *, down: bool) -> None:
            if down and mode == "noop":
                return
            original(view, down=False if down else down)

        monkeypatch.setattr(SubagentView, "_scroll_body", broken_handler)
    pilot = ScrollPilot()
    tail = pilot.body.scroll_offset.y
    await pilot.click("up")
    await _wait_for_scroll_animation(pilot, pilot.body)
    lifted = pilot.body.scroll_offset.y
    assert lifted < tail
    await pilot.click("down")
    await _wait_for_scroll_animation(pilot, pilot.body)
    if mode == "normal":
        assert pilot.body.scroll_offset.y > lifted
    else:
        with pytest.raises(AssertionError):
            assert pilot.body.scroll_offset.y > lifted


@pytest.mark.asyncio
async def test_completion_wait_is_bounded() -> None:
    pilot = ScrollPilot()

    async def never_complete() -> None:
        await asyncio.Event().wait()

    pilot.wait_for_scheduled_animations = never_complete
    with pytest.raises(AssertionError, match="scroll animation never completed"):
        await _wait_for_scroll_animation(pilot, pilot.body, timeout=0.01)
