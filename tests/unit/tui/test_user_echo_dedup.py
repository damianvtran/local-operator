"""A user message painted by the TUI is painted once, however late its receipt.

The report: a message steered mid-turn shows up twice in the transcript. The
TUI paints the ``UserBlock`` optimistically at submit, and the session later
emits a user ``MessageStartEvent`` when the steering queue is drained — the
mobile→TUI direction of keeping the two surfaces in step. The app de-duped
that echo by scanning the LAST THREE transcript blocks for a matching
``UserBlock``, which holds for a ``prompt()`` (its event fires while the echo
is still the tail) and almost never for a steer: a steer is by definition
delivered at a later tool boundary, and every tool card mounted in between
pushes the echo out of the window. Three blocks was one interrupted bash batch
away from a duplicate.

The fix is an explicit registry instead of a guess from the tail: every echo
the TUI paints for a message the session will later announce is recorded in
``_pending_user_echoes``, and ``on_user_message_start`` consumes a matching
entry instead of painting. No entry means the message came from another front
end, which is the case the handler exists to paint.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

import pytest

from local_operator.harness.types import ImageContent
from local_operator.tui.app import LOOP_PROMPT, OperatorApp
from local_operator.tui.events import SteeringDelivered, UserMessageStart
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock

from .test_app_pilot import FakeSession, _factory


class _Streaming(FakeSession):
    """A fake that is mid-turn, so a submit is STEERED rather than prompted."""

    def __init__(self) -> None:
        super().__init__()
        self.steers: list[str] = []

    @property
    def is_streaming(self) -> bool:
        return True

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.steers.append(text)


def _user_blocks(app: OperatorApp, text: str) -> list[UserBlock]:
    """Every user row carrying ``text``, in transcript order."""
    return [
        block
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock) and block.text() == text
    ]


async def _submit(pilot: Any, app: OperatorApp, text: str) -> None:
    """Type ``text`` into the composer and send it, once the app can accept it."""
    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    assert app._session is not None, "the session never booted"
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_steered_message_is_not_painted_again_by_its_own_delivery() -> None:
    """The reported duplicate, reproduced block for block.

    The steer's echo is painted at submit; the interrupted batch then mounts
    further blocks; only THEN does the drain's user MessageStartEvent reach the
    app. With the tail-window de-dup this painted a second row — the window is
    three blocks and the echo is four back — and the registry must not.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    steer = "also cover this in regression testing"
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, steer)
        assert session.steers == [steer]
        assert len(_user_blocks(app, steer)) == 1

        # The queued row settles, then the interrupted batch's cards land
        # between the echo and the delivery event — the exact shape of the
        # report. Notices stand in for the cards: any block pushes the echo
        # out of a fixed tail window, which is the property under test.
        app.post_message(SteeringDelivered(1))
        await pilot.pause()
        app._append_block(NoticeBlock("stand-in for an interrupted tool card", "info"))
        app._append_block(NoticeBlock("stand-in for a second tool card", "info"))
        await pilot.pause()

        app.post_message(UserMessageStart(steer, 0))
        await pilot.pause()

        assert len(_user_blocks(app, steer)) == 1, "the delivery echoed an already-painted row"


@pytest.mark.asyncio
async def test_two_steers_drained_together_neither_repaints() -> None:
    """Two messages queued against one boundary are announced FIFO — A then B
    — while the NEWEST user row on screen is B. A de-dup that only compares
    against the newest user row mismatches on A's event and repaints it; the
    registry matches each event to its own entry regardless of what was
    painted after it."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "first steer")
        await _submit(pilot, app, "second steer")
        assert session.steers == ["first steer", "second steer"]

        app.post_message(SteeringDelivered(2))
        await pilot.pause()
        app.post_message(UserMessageStart("first steer", 0))
        app.post_message(UserMessageStart("second steer", 0))
        await pilot.pause()

        assert len(_user_blocks(app, "first steer")) == 1
        assert len(_user_blocks(app, "second steer")) == 1


@pytest.mark.asyncio
async def test_a_message_from_another_front_end_still_paints() -> None:
    """No pending echo means the message is NEW here — the phone direction."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app.post_message(UserMessageStart("sent from the phone", 1))
        await pilot.pause()
        blocks = _user_blocks(app, "sent from the phone")
        assert len(blocks) == 1, "a foreign prompt must appear in this transcript"


@pytest.mark.asyncio
async def test_two_identical_foreign_prompts_paint_twice() -> None:
    """Sending the same words twice is two messages, and the old tail-window
    scan silently swallowed the second one. The registry only suppresses rows
    this TUI itself painted, so both must show."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app.post_message(UserMessageStart("try again", 0))
        await pilot.pause()
        app.post_message(UserMessageStart("try again", 0))
        await pilot.pause()
        assert len(_user_blocks(app, "try again")) == 2


@pytest.mark.asyncio
async def test_a_prompt_echo_is_consumed_by_its_own_event() -> None:
    """The plain prompt() path: echo at submit, event on the turn's first
    append. One row before the event, one row after."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "a normal prompt")
        assert session.prompts == ["a normal prompt"]
        assert len(_user_blocks(app, "a normal prompt")) == 1

        app.post_message(UserMessageStart("a normal prompt", 0))
        await pilot.pause()
        assert len(_user_blocks(app, "a normal prompt")) == 1


@pytest.mark.asyncio
async def test_a_session_swap_drops_the_echoes_waiting_on_the_old_queue() -> None:
    """A steer queued against the old session dies with it, so its pending
    entry must go too — left standing, it would swallow the first identical
    prompt of the NEXT conversation."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "same words, new session")
        assert app._pending_user_echoes == ["same words, new session"]

        app._session_factory = lambda: _factory(FakeSession())  # type: ignore[assignment]
        await app._reload_session()
        for _ in range(4):
            await pilot.pause()
        assert app._pending_user_echoes == []

        app.post_message(UserMessageStart("same words, new session", 0))
        await pilot.pause()
        assert len(_user_blocks(app, "same words, new session")) == 1


@pytest.mark.asyncio
async def test_a_failed_prompt_does_not_swallow_the_next_identical_one() -> None:
    """A prompt that raised never announced itself, so its entry must be
    dropped on the way out — left standing it would silently consume the
    event of the NEXT identical prompt, which this time really is new."""

    class _Failing(FakeSession):
        async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
            raise RuntimeError("boom")

    app = OperatorApp(lambda: _factory(_Failing()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "same words twice")
        assert app._pending_user_echoes == [], "the failed prompt's entry must be dropped"

        assert len(_user_blocks(app, "same words twice")) == 1, "the submit's own row"

        # The same words arrive as a real message (another front end, or a
        # later working retry): the event must PAINT a second row. A stale
        # entry would have consumed it and left the screen silent about a
        # message the session did receive.
        app.post_message(UserMessageStart("same words twice", 0))
        await pilot.pause()
        assert len(_user_blocks(app, "same words twice")) == 2


@pytest.mark.asyncio
async def test_a_pending_echo_survives_clear_and_is_consumed_by_its_event() -> None:
    """`/clear` empties the screen, not the engine's queue: a steer pending
    across the clear still has its event coming, and the entry must survive
    to suppress the repaint. The block is gone; the promise is not."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "survives the clear")
        assert app._pending_user_echoes == ["survives the clear"]

        app._clear_transcript()
        await pilot.pause()
        assert app._pending_user_echoes == [
            "survives the clear"
        ], "the clear removed the row but not the event that is still coming"

        app.post_message(UserMessageStart("survives the clear", 0))
        await pilot.pause()
        assert app._pending_user_echoes == []
        assert _user_blocks(app, "survives the clear") == [], "consumed, not repainted"


@pytest.mark.asyncio
async def test_the_loop_prompt_never_gets_a_user_row() -> None:
    """`/loop` deliberately paints a NOTICE per iteration, not a user row —
    LOOP_PROMPT is app-authored chrome. Its prompt() still announces a user
    MessageStartEvent, so without a registered echo the event painted the
    loudest mark in the transcript for words the user never typed."""
    session = FakeSession()
    session.set_goal("reach the standing goal")  # `/loop` refuses without one
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/loop 1")
        for _ in range(10):
            await pilot.pause()

        app.post_message(UserMessageStart(LOOP_PROMPT, 0))
        await pilot.pause()
        assert _user_blocks(app, LOOP_PROMPT) == []
