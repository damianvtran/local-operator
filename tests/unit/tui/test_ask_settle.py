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
from local_operator.tui.widgets.approval import ApprovalPrompt
from local_operator.tui.widgets.ask_picker import AskPickerScreen
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


@pytest.mark.asyncio
async def test_answering_the_last_question_takes_the_card_down() -> None:
    """The ORDINARY path, which is the one that had no teardown at all.

    The card resolves its own future from `settle`, so answering satisfied the
    awaiting tool call and the `finally` had nothing to do — leaving the widget
    mounted, the dock still holding its row, and focus still on a question
    nobody was waiting for. The composer could not be typed into afterwards,
    which is the same "the keys go nowhere" failure the anchoring exists to fix,
    arriving one step later.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(10):
            await pilot.pause(0.05)
        assert app.query(AskPickerScreen), "no card was raised"

        await pilot.press("enter")
        assert await asyncio.wait_for(asked, 2) == {"stale": ["Drop them"]}
        for _ in range(10):
            await pilot.pause(0.05)

        # Nothing left behind: no widget, no reserved row, and the caret is back
        # where the user types.
        assert not app.query(AskPickerScreen)
        assert not app.query_one("#prompt-host").display
        assert isinstance(app.screen.focused, Editor)


@pytest.mark.asyncio
async def test_an_overlapping_ask_survives_an_answered_approval() -> None:
    """Two prompts can be live at once, and answering one must not bury the other.

    Approvals serialize against `_approval`, but `request_user_choice` mounts
    into the same host with no interlock, so an `ask` and an approval genuinely
    overlap. Hiding the host unconditionally as one card left took the OTHER
    question off screen while it was still attached and still awaited: a turn
    parked on an answer the user could no longer see or reach, which is the
    hang class this whole module exists to remove (F1, agent review round 1).
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            await pilot.pause(0.02)
            if getattr(session, "approval_handler", None) is not None:
                break
        app._set_approve_all(False)
        app._approvals_default_auto = False

        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(10):
            await pilot.pause(0.02)
        handler = getattr(session, "approval_handler", None)
        assert handler is not None, "the app never installed its approval gate"
        approving = asyncio.ensure_future(handler("bash", "run: make"))
        for _ in range(20):
            await pilot.pause(0.02)
        assert app.query(AskPickerScreen), "the ask card is not up"

        # Answer the APPROVAL and leave the ask standing.
        await pilot.press("y")
        assert await asyncio.wait_for(approving, 2) is True
        for _ in range(20):
            await pilot.pause(0.02)

        # The surviving question is still visible, still holds the keyboard...
        assert app.query_one("#prompt-host").display, "the live ask was hidden"
        assert app.query(AskPickerScreen), "the live ask was unmounted"
        assert isinstance(app.screen.focused, AskPickerScreen)
        # ...and is still answerable.
        await pilot.press("enter")
        assert await asyncio.wait_for(asked, 2) == {"stale": ["Drop them"]}


@pytest.mark.asyncio
async def test_escape_answers_the_prompt_the_user_is_looking_at() -> None:
    """With two prompts live, Escape must reach the one on screen.

    The picker takes Escape as "skip this question", but that branch matched on
    a picker merely EXISTING and returned before the approval was considered.
    In the ask + approval overlap the approval is the card actually painted and
    focused, and the picker is scrolled off — so one Escape settled the
    invisible question and left a focused `rm -rf` approval unanswered with the
    turn still running (F5, agent review round 4).

    That branch is also the only implementation of the approval's advertised
    `esc deny`: `ApprovalPrompt.action_cancel` raises `SkipAction` by design so
    the key reaches the app.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            await pilot.pause(0.02)
            if getattr(session, "approval_handler", None) is not None:
                break
        app._set_approve_all(False)
        app._approvals_default_auto = False
        # `FakeSession` reports `is_streaming` from this attribute; set via
        # `setattr` because the fake declares it dynamically.
        setattr(session, "streaming", True)

        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(12):
            await pilot.pause(0.02)
        handler = getattr(session, "approval_handler", None)
        assert handler is not None
        approving = asyncio.ensure_future(handler("bash", "run: rm -rf /Users/x/data"))
        for _ in range(20):
            await pilot.pause(0.02)
        assert isinstance(app.screen.focused, ApprovalPrompt), "the approval is not the live card"

        await pilot.press("escape")
        # The card the user was looking at is the one that answered.
        assert await asyncio.wait_for(approving, 2) is False
        assert session.aborts == ["interrupted"]

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass
