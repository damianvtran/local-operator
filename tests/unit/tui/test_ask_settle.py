"""Taking the ``ask`` picker down without taking the wrong screen with it.

``_settle_ask_picker`` is the teardown path: a stop, an app quit, or a cancelled
tool call resolves the question's future with whatever the user had answered and
then has to get the modal off the screen, because a modal left up for a call that
no longer exists holds the keyboard hostage.

It did that with ``pop_screen()``, which pops whatever the stack ENDS with rather
than the screen whose future it just settled. With anything mounted above the
question — a palette, a picker the user opened while the agent waited on them —
that dismissed the wrong screen and left the settled picker mounted, which is
both halves of the bug it was written to prevent.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.tui.app import OperatorApp

from .test_app_pilot import FakeSession, _factory


class Overlay(ModalScreen[None]):
    """Stands in for whatever the user put on top of the question."""

    def compose(self) -> ComposeResult:
        yield Static("on top")


def _question() -> AskQuestion:
    return AskQuestion(
        id="stale",
        question="What should happen to the stale rows?",
        options=[
            AskOption(label="Drop them", description="nothing reads the column"),
            AskOption(label="Backfill", description="slower, keeps history"),
        ],
    )


@pytest.mark.asyncio
async def test_settling_the_picker_dismisses_it_when_it_is_on_top() -> None:
    """The ordinary case, unchanged: the question is the top screen and comes
    off, and the parked tool call gets its answer rather than hanging."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        await pilot.pause()
        picker = app._ask_screen
        assert picker is not None and app.screen is picker

        app._settle_ask_picker()
        await pilot.pause()

        await asyncio.wait_for(asked, 2)
        assert picker not in app.screen_stack
        assert app.screen is not picker


@pytest.mark.asyncio
async def test_settling_the_picker_leaves_the_screen_above_it_alone() -> None:
    """The regression: with another modal above the question, ``pop_screen``
    dismissed THAT screen — the user's — and left the picker mounted on an
    already-settled future, still holding the keyboard."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        await pilot.pause()
        picker = app._ask_screen
        assert picker is not None

        overlay = Overlay()
        app.push_screen(overlay)
        await pilot.pause()
        assert app.screen is overlay

        app._settle_ask_picker()
        await pilot.pause()

        await asyncio.wait_for(asked, 2)
        # The screen the user is actually looking at survived...
        assert app.screen is overlay
        assert overlay in app.screen_stack
        # ...and the picker is off the stack and off the DOM, not merely settled.
        assert picker not in app.screen_stack
        assert not picker.is_attached
