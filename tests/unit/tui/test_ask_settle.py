"""Taking the ``ask`` card down without disturbing anything else on screen.

``_settle_ask_picker`` is the teardown path: a stop, an app quit, or a cancelled
tool call resolves the question's future with whatever the user had answered and
then has to get the card off screen, because a question left up for a call that
no longer exists holds the keyboard hostage.

The card is anchored in the dock now rather than pushed as a modal screen, which
retires the specific bug this file was written for (``pop_screen()`` popping
whatever the stack ENDED with, dismissing the user's own modal and leaving the
settled picker mounted). The INVARIANT survives the move and is what these tests
still pin: settling answers the parked tool call, takes the card off the DOM, and
touches nothing the user opened over it.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor

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
async def test_settling_the_picker_takes_the_card_down_and_answers_the_call() -> None:
    """The ordinary case: the card is in the dock, comes off, and the parked
    tool call gets its answer rather than hanging."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(4):
            await pilot.pause()
        picker = app._ask_screen
        assert picker is not None
        # Anchored in the dock, not covering the screen: the conversation the
        # question is about stays visible behind it.
        assert picker.is_attached
        assert picker.parent is app.query_one("#prompt-host")

        app._settle_ask_picker()
        for _ in range(4):
            await pilot.pause()

        await asyncio.wait_for(asked, 2)
        assert not picker.is_attached
        # And the dock gives the rows back rather than holding an empty slot.
        assert not app.query_one("#prompt-host").display
        # Focus lands back in the composer, so the next keystroke goes somewhere.
        assert isinstance(app.screen.focused, Editor)


@pytest.mark.asyncio
async def test_settling_the_picker_leaves_the_screen_above_it_alone() -> None:
    """Settling must not disturb a screen the user opened over the question.

    The original defect was ``pop_screen()`` taking whatever the stack ENDED
    with: with a palette or picker opened while the agent waited, that dismissed
    the user's screen and left the settled card mounted on an already-resolved
    future. Anchoring the card in the dock removes the mechanism, and this pins
    the property so a future revision cannot reintroduce it by reaching for the
    screen stack again.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(4):
            await pilot.pause()
        picker = app._ask_screen
        assert picker is not None

        overlay = Overlay()
        app.push_screen(overlay)
        for _ in range(3):
            await pilot.pause()
        assert app.screen is overlay

        app._settle_ask_picker()
        for _ in range(4):
            await pilot.pause()

        await asyncio.wait_for(asked, 2)
        # The screen the user is actually looking at survived...
        assert app.screen is overlay
        assert overlay in app.screen_stack
        # ...and the card is off the DOM, not merely settled.
        assert not picker.is_attached
