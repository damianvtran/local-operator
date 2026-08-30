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

Issue #228 is the second half: the registry originally matched by TEXT, so a
DISTINCT message whose words collided with a pending echo (repeated "yes" /
"continue" sent from the phone while a TUI echo of the same words was still
outstanding) consumed that entry and never painted — the message reached the
model and the transcript stayed silent about it. Entries now carry the message
id the app itself minted and handed to the session, so an id this surface never
registered paints however familiar the words. An entry with no id (a session
whose ``prompt`` predates the ``message_id`` seam) keeps the text match, which
is what the ``FakeSession`` cases below exercise.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

import pytest

from local_operator.harness.types import ImageContent
from local_operator.tui.app import (
    DEFERRED_SENT_STEER_NOTICE,
    DEFERRED_STEER_NOTICE,
    LOOP_PROMPT,
    SENT_STEER_NOTICE,
    OperatorApp,
)
from local_operator.tui.events import SteeringDelivered, TurnEnded, UserMessageStart
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

    def steer_message(self, message: Any) -> None:
        # The app queues via `steer_message` now; record the text the old
        # `steer` override did and let the base fake hold the object.
        self.steers.append(message.text)
        super().steer_message(message)


class _IdAware(FakeSession):
    """A fake with the production ``message_id`` seam on ``prompt``.

    ``FakeSession`` deliberately predates it, which is how the legacy text
    fallback stays covered; this one is the shape the real ``Session`` has, so
    the app mints a correlation id and hands it over. ``prompt_ids`` records
    what it received, which is the id the real session would then announce.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_ids: list[str | None] = []

    async def prompt(  # type: ignore[override]
        self,
        text: str,
        images: Sequence[ImageContent] | None = None,
        *,
        message_id: str | None = None,
    ) -> None:
        self.prompt_ids.append(message_id)
        await super().prompt(text, images)


def _echo_ids(app: OperatorApp) -> list[str]:
    """The message ids of the pending echo entries, in registration order."""
    return [entry.message_id for entry in app._pending_user_echoes]


def _echo_texts(app: OperatorApp) -> list[str]:
    """The texts of the pending echo entries, in registration order."""
    return [entry.text for entry in app._pending_user_echoes]


def _steer_id(session: FakeSession, text: str) -> str:
    """The id of the queued steering Message carrying ``text``.

    The session announces a drained steer with the very object the app queued,
    so this is the id the real ``MessageStartEvent`` would carry — the tests
    below post it rather than an empty id, which is what makes them exercise
    the id path instead of the legacy fallback.
    """
    for message in session.queued_steering():
        if getattr(message, "text", None) == text:
            return str(message.id)
    raise AssertionError(f"no queued steer carrying {text!r}")


def _user_blocks(app: OperatorApp, text: str) -> list[UserBlock]:
    """Every user row carrying ``text``, in transcript order."""
    return [
        block
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock) and block.text() == text
    ]


def _notice_texts(app: OperatorApp) -> list[str]:
    """Every notice row's text, in transcript order."""
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
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

        app.post_message(UserMessageStart(steer, 0, _steer_id(session, steer)))
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

        first_id = _steer_id(session, "first steer")
        second_id = _steer_id(session, "second steer")
        app.post_message(SteeringDelivered(2))
        await pilot.pause()
        app.post_message(UserMessageStart("first steer", 0, first_id))
        app.post_message(UserMessageStart("second steer", 0, second_id))
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
        assert _echo_texts(app) == ["same words, new session"]

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
        assert _echo_texts(app) == ["survives the clear"]
        steer_id = _steer_id(session, "survives the clear")

        app._clear_transcript()
        await pilot.pause()
        assert _echo_texts(app) == [
            "survives the clear"
        ], "the clear removed the row but not the event that is still coming"

        app.post_message(UserMessageStart("survives the clear", 0, steer_id))
        await pilot.pause()
        assert app._pending_user_echoes == []
        assert _user_blocks(app, "survives the clear") == [], "consumed, not repainted"


@pytest.mark.asyncio
async def test_a_deferred_steer_is_settled_not_repainted_by_the_next_turns_echo() -> None:
    """The one state where the registry and the receipt machinery overlap: a
    steer whose turn ended before any drain defers its row ("still queued —
    sends with your next message"), and its registry entry must SURVIVE the
    turn end — the message really is still in the engine's queue. When the
    next turn's drain announces it, the deferred row settles to `sent` and
    the event is consumed by the entry, not painted as a second row."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "deferred across the turn")
        assert _echo_texts(app) == ["deferred across the turn"]
        steer_id = _steer_id(session, "deferred across the turn")

        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        assert DEFERRED_STEER_NOTICE in _notice_texts(app)
        assert _echo_texts(app) == [
            "deferred across the turn"
        ], "the turn ended, but the message is still queued — the entry stays"

        # The next turn's drain: the receipt settles the deferred row, and the
        # announcement consumes the entry instead of repainting the message.
        app.post_message(SteeringDelivered(1))
        app.post_message(UserMessageStart("deferred across the turn", 0, steer_id))
        await pilot.pause()
        texts = _notice_texts(app)
        # A row deferred ACROSS the turn takes the cross-turn settle: it went
        # with the message this echo is about (issue #160, D5).
        assert DEFERRED_SENT_STEER_NOTICE in texts and DEFERRED_STEER_NOTICE not in texts
        assert SENT_STEER_NOTICE not in texts
        assert len(_user_blocks(app, "deferred across the turn")) == 1


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


# --- issue #228: a distinct message whose words collide must still paint ------


@pytest.mark.asyncio
async def test_a_foreign_prompt_with_the_words_of_a_pending_steer_still_paints() -> None:
    """Issue #228, reproduced: the collision the text match swallowed.

    A steer is queued from the TUI ("continue") and its echo is pending. The
    PHONE then sends its own "continue", which the session announces with a
    DIFFERENT message id. Matching by text consumed the steer's entry and
    painted nothing, so the TUI showed one row where the conversation had two
    messages — the model saw both and the transcript denied one of them. The id
    is what tells them apart.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "continue")
        assert len(_user_blocks(app, "continue")) == 1, "the TUI's own echo"
        steer_id = _steer_id(session, "continue")

        # The phone's message: same words, its own identity.
        app.post_message(UserMessageStart("continue", 0, "phone-message-id"))
        await pilot.pause()
        assert (
            len(_user_blocks(app, "continue")) == 2
        ), "the phone's distinct message must paint; matching by text swallowed it"
        assert _echo_ids(app) == [steer_id], "the steer's entry is still waiting on its own event"

        # And the true duplicate is still suppressed EXACTLY once: the steer's
        # own delivery finds its entry and repaints nothing.
        app.post_message(SteeringDelivered(1))
        app.post_message(UserMessageStart("continue", 0, steer_id))
        await pilot.pause()
        assert len(_user_blocks(app, "continue")) == 2, "the steer's own event must not repaint"
        assert app._pending_user_echoes == []


@pytest.mark.asyncio
async def test_the_prompt_path_registers_the_id_it_handed_the_session() -> None:
    """The prompt half of the same contract, end to end.

    The app mints the correlation id and pushes it into ``prompt(message_id=)``
    — it cannot read one back, because ``prompt`` returns only when the whole
    turn is over, long after the announcement. The id the session received is
    therefore the id its ``MessageStartEvent`` carries, and it must be the id
    sitting in the registry.
    """
    session = _IdAware()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "a prompt with an id")
        for _ in range(10):
            await pilot.pause()

        assert session.prompts == ["a prompt with an id"]
        assert session.prompt_ids and session.prompt_ids[0], "the app must supply a message_id"
        assert _echo_ids(app) == [session.prompt_ids[0]]

        # A colliding message from elsewhere paints...
        app.post_message(UserMessageStart("a prompt with an id", 0, "some-other-id"))
        await pilot.pause()
        assert len(_user_blocks(app, "a prompt with an id")) == 2

        # ...and this prompt's own announcement still suppresses its repaint.
        app.post_message(UserMessageStart("a prompt with an id", 0, session.prompt_ids[0] or ""))
        await pilot.pause()
        assert len(_user_blocks(app, "a prompt with an id")) == 2
        assert app._pending_user_echoes == []


@pytest.mark.asyncio
async def test_a_session_without_the_message_id_seam_keeps_the_text_match() -> None:
    """``FakeSession.prompt`` has no ``message_id`` keyword, so the app must
    NOT mint an id it cannot hand over: an entry keyed by an id the session
    will never announce could match nothing, and every prompt would paint
    twice. Such an entry registers id-less and matches by text, exactly as
    before — the compatibility path older and third-party sessions ride."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "no id seam here")
        for _ in range(10):
            await pilot.pause()
        assert _echo_ids(app) == [""], "no id may be minted for a session that cannot take one"

        # The session mints its own id, so the announcement carries one the app
        # has never seen — and the text fallback is what suppresses the repaint.
        app.post_message(UserMessageStart("no id seam here", 0, "session-minted-id"))
        await pilot.pause()
        assert len(_user_blocks(app, "no id seam here")) == 1
        assert app._pending_user_echoes == []


@pytest.mark.asyncio
async def test_an_id_carrying_entry_is_never_consumed_by_a_text_collision() -> None:
    """The fallback must not undo the fix. An entry that HAS an id has an exact
    event coming, so an id-less announcement carrying the same words must paint
    rather than spend it — otherwise the steer's own delivery would later find
    its entry gone and paint the duplicate #227 removed."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "yes")
        steer_id = _steer_id(session, "yes")

        # An announcement with no id at all (a reduced event producer): it is
        # not this steer's, so it paints and leaves the entry standing.
        app.post_message(UserMessageStart("yes", 0))
        await pilot.pause()
        assert len(_user_blocks(app, "yes")) == 2
        assert _echo_ids(app) == [steer_id]

        app.post_message(UserMessageStart("yes", 0, steer_id))
        await pilot.pause()
        assert len(_user_blocks(app, "yes")) == 2, "the steer's own delivery must not repaint"


@pytest.mark.asyncio
async def test_a_recalled_steer_takes_only_its_own_entry() -> None:
    """Esc unsends a queued steer, so its entry must go with the row it
    described. By ID: with two steers carrying the same words, a by-text
    removal could take the sibling's entry and leave the recalled one standing
    — the survivor would then swallow the resend's echo."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "same words")
        await _submit(pilot, app, "same words")
        first_id = _steer_id(session, "same words")
        ids = _echo_ids(app)
        assert len(ids) == 2 and ids[0] == first_id

        # Esc recalls the NEWEST steer; its entry, and only its entry, goes.
        await pilot.press("escape")
        await pilot.pause()
        assert _echo_ids(app) == [first_id], "the recall took the wrong steer's entry"
