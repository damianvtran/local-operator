"""A queued mid-turn message says so, and then says it was delivered.

The report: sending a message while the agent is working prints "queued — sends
when this step finishes", and that row never changes. After the message has
actually been delivered the transcript still shows the promise, so the only
evidence it arrived is the agent eventually answering it.

The fix is a state transition rather than a second notice. ``steer()`` is
fire-and-forget by design — it drops a message on a queue the loop empties at
its next boundary — so the session emits ``SteeringDeliveredEvent`` at the
moment it actually drains that queue, and the app RESTATES the row it already
painted. One statement that became true, instead of a stale promise with a
correction underneath it.

Both halves are pinned here: the engine only speaks when it really took
something (``tests/unit/session`` would be the wrong home for the app half, and
a UI test alone would pass against an event nothing emits).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from local_operator.harness.types import ModelSpec, SteeringDeliveredEvent
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tui.app import (
    DEFERRED_SENT_STEER_NOTICE,
    DEFERRED_STEER_NOTICE,
    QUEUED_STEER_NOTICE,
    SENT_STEER_NOTICE,
    OperatorApp,
)
from local_operator.tui.events import SteeringDelivered, TurnEnded
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    NoticeKind,
    TranscriptView,
)

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
        # `steer` override did and let the base fake hold the object so Esc
        # can recall it.
        self.steers.append(message.text)
        super().steer_message(message)


def _notice_blocks(app: OperatorApp) -> list[NoticeBlock]:
    """Every notice row, in transcript order.

    The BLOCKS, not their text, for the tests that have to follow one specific
    row across several deliveries: two rows can carry the same string, and
    #151's whole failure is a row that never changes — which a text list cannot
    attribute to the row it belongs to.
    """
    return [
        block for block in app.query_one(TranscriptView).blocks() if isinstance(block, NoticeBlock)
    ]


def _notice_texts(app: OperatorApp) -> list[str]:
    """Every notice row's text, in transcript order."""
    return [block._text for block in _notice_blocks(app)]


async def _settled_notice_height(cols: int, text: str, kind: NoticeKind) -> int:
    """The row count ``text`` renders to at ``cols`` columns, read once, settled.

    One block in a FRESH app, rather than one block measured before and after a
    restate. Two things make that the reliable protocol:

    * The pre-restate reading of a restated block is racy — the region is not
      always settled after a fixed number of pauses, so it comes back one row
      short intermittently (review round 1, F2, measured at 1 in 12 at 52
      columns). Here there is no baseline reading to slip: each string is
      rendered independently and compared as a number.
    * A second notice mounted beside the first is given a gap row by adaptive
      spacing when its neighbour is multi-row, so two blocks in one app differ
      in ``virtual_region`` for reasons that have nothing to do with the copy.

    The height is taken from the block's OWN wrap of its own text at its own
    measured width, not from ``virtual_region``. That is what makes it
    deterministic: ``virtual_region`` is assigned by the compositor, so reading
    it races the layout pass no matter how many frames are waited — polling it
    for stability merely trades a fixed pause count for an unsettled plateau,
    which is the same flake one step removed (observed at a width that moved
    between runs). ``_rows`` is a pure function of the text and the body width,
    it is the very computation that decides the row count, and it is what
    ``_build`` itself calls, so it answers the question the test is actually
    asking: how many rows does this string occupy here.

    The width is still read from the mounted block, so the transcript's real
    padding and scrollbar are in the number rather than assumed.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(cols, 24)) as pilot:
        await pilot.pause()
        block = NoticeBlock(text, kind)
        app._append_block(block)
        await pilot.pause()
        await pilot.pause()
        # Mirrors `NoticeBlock._build`: the block reserves a fixed glyph field
        # and wraps the remainder, so the body is the block's width less that
        # gutter. Read from the live block rather than recomputed from `cols`.
        width = max((block.size.width or 80) - 2, 12)
        body = max(width - block.GLYPH_COLS, 8)
        return len(block._rows(body))


async def _submit(pilot: Any, app: OperatorApp, text: str) -> None:
    """Type ``text`` into the composer and send it, once the app can accept it.

    Waits for the SESSION first. The app paints before its session exists (the
    factory is awaited in a boot worker), and a submit that lands in that window
    is answered with "session is still starting…" and never reaches `steer` — so
    a test that pressed enter on a fixed frame count was racing the boot rather
    than testing anything. Waiting on the condition instead of on a sleep keeps
    it deterministic on a loaded machine; the same race is what costs
    `test_app_pilot`'s first test on `main`.

    Focus is then set explicitly, because `enter` goes to whatever holds it.
    """
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
async def test_a_queued_message_says_queued_until_it_is_delivered() -> None:
    """The row starts as the promise, and only the delivery settles it."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "also check the images")

        assert session.steers == ["also check the images"]
        assert QUEUED_STEER_NOTICE in _notice_texts(app)
        assert SENT_STEER_NOTICE not in _notice_texts(app), "nothing has been delivered yet"

        app.post_message(SteeringDelivered(1))
        await pilot.pause()

        texts = _notice_texts(app)
        assert SENT_STEER_NOTICE in texts
        # UPDATED, not appended: the stale promise must be gone, and the
        # transcript must not have grown a row to say so.
        assert QUEUED_STEER_NOTICE not in texts
        assert texts.count(SENT_STEER_NOTICE) == 1


@pytest.mark.asyncio
async def test_every_message_queued_against_one_boundary_settles_together() -> None:
    """Three messages typed during one tool call are three promises kept at once.

    The queue is drained WHOLE at a single boundary, so settling only the newest
    row would leave the earlier ones lying about their own delivery.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for text in ("one", "two", "three"):
            await _submit(pilot, app, text)

        assert _notice_texts(app).count(QUEUED_STEER_NOTICE) == 3

        app.post_message(SteeringDelivered(3))
        await pilot.pause()

        texts = _notice_texts(app)
        assert texts.count(SENT_STEER_NOTICE) == 3
        assert QUEUED_STEER_NOTICE not in texts


@pytest.mark.asyncio
async def test_a_delivery_with_no_queued_row_changes_nothing() -> None:
    """A late or duplicated delivery must not restate an unrelated notice.

    The app settles rows it is HOLDING references to, so an event arriving when
    it holds none is a no-op — not a hunt through the transcript for something
    that looks queued.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._append_block(NoticeBlock("compacting context…", "info"))
        await pilot.pause()
        before = _notice_texts(app)

        app.post_message(SteeringDelivered(1))
        await pilot.pause()

        assert _notice_texts(app) == before


@pytest.mark.asyncio
async def test_clearing_the_transcript_drops_the_rows_it_removed() -> None:
    """`/clear` removes the rows, so a later delivery has nothing to settle.

    Without this the app would hold references to widgets no longer mounted and
    "settle" them where nobody can see, while any row still on screen kept its
    promise.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "queued before the clear")
        assert app._queued_steer_notices

        app._clear_transcript()
        await pilot.pause()

        assert app._queued_steer_notices == []
        app.post_message(SteeringDelivered(1))
        await pilot.pause()
        assert SENT_STEER_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_an_interrupted_turn_retires_the_promise_it_can_no_longer_keep() -> None:
    """Ctrl+C ends the turn before any boundary, so the row must stop promising.

    The receipt only fires when the engine actually drains the queue. A turn
    that is aborted never reaches a boundary, so without this the row goes on
    promising delivery for the rest of the session — the same defect the receipt
    was added to fix, reached by the abort path. It is worse than doing nothing:
    the message really is still queued, so the NEXT turn's drain would settle a
    row the user stopped caring about minutes earlier.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "steered into a turn about to be stopped")
        assert QUEUED_STEER_NOTICE in _notice_texts(app)

        app.post_message(TurnEnded(True, None))
        await pilot.pause()

        texts = _notice_texts(app)
        assert DEFERRED_STEER_NOTICE in texts
        assert QUEUED_STEER_NOTICE not in texts
        assert SENT_STEER_NOTICE not in texts, "nothing was delivered"
        assert app._queued_steer_notices == []
        # Retired from the CURRENT turn's list, but still held: the message is
        # genuinely still in the engine's queue, so the row is still waiting on
        # a delivery that has not happened yet (issue #151).
        assert len(app._deferred_steer_notices) == 1
        # The message is still in the engine's queue, so the row must not claim
        # it was lost — the user would retype and the agent would get it twice.
        assert "not sent" not in DEFERRED_STEER_NOTICE
        assert "still queued" in DEFERRED_STEER_NOTICE
        # And it must not claim the user will send that message (design round 2,
        # D6). A wake that lands while idle, a peer `lop send` and a background
        # job result each open their own turn and drain the queue, so the turn
        # this row is waiting on need not be one the user started. The promise
        # and the settle share the deictic for that reason; `your` in either is
        # the same false authorship claim one state apart.
        assert "your" not in DEFERRED_STEER_NOTICE
        assert "your" not in DEFERRED_SENT_STEER_NOTICE


@pytest.mark.asyncio
async def test_a_clean_turn_that_never_drained_says_the_message_is_still_queued() -> None:
    """A turn can end cleanly without ever reaching a boundary to drain at.

    The model answers with no further tool calls after the steer lands, so no
    injection boundary runs, `agent_end` arrives with `aborted=False` and no
    error, and the queue is untouched. The message really will go — at the next
    turn's first boundary — so the row must claim neither a delivery nor a loss.

    Left alone it would settle minutes later against whatever the user is
    looking at by then, which is the same disconnect the interrupted case has.
    Every turn end reconciles its held rows: a row still held when a turn ends
    is by definition one that turn did not deliver.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "a steer the model answers without tools")
        assert QUEUED_STEER_NOTICE in _notice_texts(app)

        app.post_message(TurnEnded(False, None))
        await pilot.pause()

        texts = _notice_texts(app)
        assert DEFERRED_STEER_NOTICE in texts
        assert QUEUED_STEER_NOTICE not in texts
        # Not the delivered row: nothing was sent. The waiting row is the same
        # one an interrupted turn shows, deliberately — from the user's side the
        # two are one fact (the message is queued and goes with their next
        # message) and the action they take is identical.
        assert SENT_STEER_NOTICE not in texts
        assert app._queued_steer_notices == []
        assert len(app._deferred_steer_notices) == 1, "the row still awaits its delivery"


@pytest.mark.asyncio
async def test_a_deferred_row_is_settled_by_the_delivery_it_was_waiting_for() -> None:
    """Issue #151: the one promise in the set that was never kept on screen.

    `still queued — sends with your next message` is TRUE when it is painted and
    the delivery really does happen — at the next turn's first boundary. The app
    used to drop its reference to the row at the moment it painted that text, so
    the delivery arrived with nothing left to settle: the user sent their next
    message, watched the agent act on the steered instruction, and the row above
    still read `still queued`.

    Driven end to end through the transition the user actually makes: steer,
    turn ends undelivered, next turn's boundary drains the queue.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "and use the staging credentials")

        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        assert DEFERRED_STEER_NOTICE in _notice_texts(app)

        # The next turn reaches its first boundary and the engine takes the
        # message that has been waiting since before the interrupt.
        app.post_message(SteeringDelivered(1))
        await pilot.pause()

        texts = _notice_texts(app)
        # The DEFERRED settle, not the shared one: this row promised to go with
        # the user's next message and it did, which is the one fact the
        # same-turn string cannot carry (issue #160, D5).
        assert texts.count(DEFERRED_SENT_STEER_NOTICE) == 1
        assert SENT_STEER_NOTICE not in texts
        assert DEFERRED_STEER_NOTICE not in texts, "the kept promise still reads as waiting"
        assert QUEUED_STEER_NOTICE not in texts
        assert app._deferred_steer_notices == []


@pytest.mark.asyncio
async def test_a_deferred_row_settles_before_one_queued_against_the_new_turn() -> None:
    """The two holders are one FIFO, because the engine's queue is one FIFO.

    A row carried over from an ended turn is OLDER than anything steered into
    the turn now running, and nothing reordered the engine's queue — it was
    never drained. So a delivery that takes one message takes the carried-over
    one, and settling the newer row instead would claim a delivery for a message
    still sitting behind it.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "the older instruction")
        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        await _submit(pilot, app, "the newer instruction")

        blocks = _notice_blocks(app)
        older = next(block for block in blocks if block._text == DEFERRED_STEER_NOTICE)
        newer = next(block for block in blocks if block._text == QUEUED_STEER_NOTICE)

        # The boundary took ONE message.
        app.post_message(SteeringDelivered(1))
        await pilot.pause()

        assert older._text == DEFERRED_SENT_STEER_NOTICE, "the older message was the one that went"
        assert newer._text == QUEUED_STEER_NOTICE, "the newer row must keep promising"
        assert app._deferred_steer_notices == []
        assert app._queued_steer_notices == [newer]


@pytest.mark.asyncio
async def test_a_surviving_deferred_row_stays_deferred_and_is_not_restated_again() -> None:
    """A PARTIAL settle must not migrate the survivor into the queued list.

    The two lists are concatenated for ordering only, and splitting the
    survivors back by arithmetic on `taken` instead of by membership silently
    puts a leftover deferred row into `_queued_steer_notices`. Nothing is
    visibly wrong at that moment — and then the next turn end finds it there
    and restates it a second time, which is exactly the redundant rebuild and
    gap re-measure the two-list split exists to prevent.

    Reached through the count guard rather than a contrived call: `count`
    crosses a thread boundary from a producer this app does not own, an
    unusable value degrades to ONE delivery, and one delivery against two held
    rows is the partial case. Review round 1, F1 \u2014 the arithmetic version of
    this split passed all fifteen other tests in this file.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for text in ("the first instruction", "the second instruction"):
            await _submit(pilot, app, text)

        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        assert len(app._deferred_steer_notices) == 2

        older, younger = app._deferred_steer_notices

        # An unusable `count` degrades to one delivery, so ONE of the two goes.
        garbled = SteeringDelivered(1)
        garbled.count = "not a number"  # type: ignore[assignment]
        app.post_message(garbled)
        await pilot.pause()

        assert older._text == DEFERRED_SENT_STEER_NOTICE
        assert younger._text == DEFERRED_STEER_NOTICE
        # The survivor is still DEFERRED. Landing in the queued list instead
        # would be invisible here and wrong on the next turn end, below.
        assert app._deferred_steer_notices == [younger]
        assert app._queued_steer_notices == []

        restatements: list[str] = []
        original = type(younger).restate

        def _record(self: NoticeBlock, text: str, kind: str) -> None:
            if self is younger:
                restatements.append(text)
            original(self, text, kind)  # type: ignore[arg-type]

        with patch.object(type(younger), "restate", _record):
            app.post_message(TurnEnded(False, None))
            await pilot.pause()

        assert restatements == [], "the survivor was retired a second time"
        assert younger._text == DEFERRED_STEER_NOTICE


@pytest.mark.asyncio
async def test_a_second_turn_end_does_not_restate_a_row_it_already_retired() -> None:
    """Rows move OUT of the queued list when they are retired, once.

    Holding them in the same list would make every later turn end rewrite every
    row still waiting — N rebuilds and N gap re-measurements per turn, for text
    that does not change, for the rest of the session.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "waiting on a later turn")

        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        row = next(block for block in _notice_blocks(app) if block._text == DEFERRED_STEER_NOTICE)
        restatements: list[str] = []
        original = type(row).restate

        def _record(self: NoticeBlock, text: str, kind: str) -> None:
            if self is row:
                restatements.append(text)
            original(self, text, kind)  # type: ignore[arg-type]

        with patch.object(type(row), "restate", _record):
            for _ in range(3):
                app.post_message(TurnEnded(False, None))
                await pilot.pause()

        assert restatements == [], "a retired row was rewritten by a later turn end"
        assert row._text == DEFERRED_STEER_NOTICE


@pytest.mark.asyncio
async def test_a_session_swap_drops_the_rows_waiting_on_the_old_queue() -> None:
    """The distinction the holding must preserve: swap vs turn ended.

    A carried-over row is waiting on a message in THIS session's steering queue.
    A swap tears that session down and empties the transcript, so the message is
    gone with it and the row is off screen — the replacement session's first
    delivery is a different conversation's message, and settling these rows
    against it would be a receipt for something that never happened.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "queued before the swap")
        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        assert app._deferred_steer_notices, "the row should be held before the swap"

        app._session_factory = lambda: _factory(_Streaming())  # type: ignore[assignment]
        await app._reload_session()
        for _ in range(4):
            await pilot.pause()

        assert app._deferred_steer_notices == []
        assert app._queued_steer_notices == []
        app.post_message(SteeringDelivered(1))
        await pilot.pause()
        assert SENT_STEER_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_clearing_the_transcript_drops_the_rows_waiting_on_a_later_turn() -> None:
    """`/clear` removes carried-over rows for the same reason it removes queued ones.

    The widgets are no longer in the transcript, so a later delivery would
    "settle" rows nobody can see. The message itself is unaffected — `/clear`
    empties the screen, not the engine's queue — so it still arrives, it just
    has no row left to report it on.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "queued before the clear")
        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        assert app._deferred_steer_notices

        app._clear_transcript()
        await pilot.pause()

        assert app._deferred_steer_notices == []
        app.post_message(SteeringDelivered(1))
        await pilot.pause()
        assert SENT_STEER_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_an_unusable_delivery_count_does_not_take_the_app_down() -> None:
    """`count` crosses a thread boundary from a producer this app does not own.

    "A receipt must never take the app down" is the posture the whole handler is
    written to, and a bare `int()` on a field from elsewhere was the one line
    that could break it. A nonsense value degrades to the safe reading — one
    delivery, because the event means at least one message went.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "one")
        await _submit(pilot, app, "two")

        # `float("inf")` specifically, alongside the obvious nonsense: the
        # first version of this guard named `TypeError`/`ValueError` and this
        # value raises `OverflowError`, which killed the app in the message
        # loop — the one gap in a guard written because the producer is not
        # ours. Enumerating failure modes is how such a gap reappears.
        for bad in ("not a number", float("inf"), float("nan"), None, [1], -3, 0):
            message = SteeringDelivered(1)
            message.count = bad  # type: ignore[assignment]
            app.post_message(message)
            await pilot.pause()
            assert app.is_running, f"the app died on count={bad!r}"

        texts = _notice_texts(app)
        # Each unusable value degraded to ONE delivery, so the two rows settled
        # one at a time rather than all at once or not at all.
        assert texts.count(SENT_STEER_NOTICE) == 2
        assert QUEUED_STEER_NOTICE not in texts


@pytest.mark.asyncio
async def test_only_the_messages_that_went_are_settled() -> None:
    """A message that raced into the queue after the drain keeps its promise.

    `_drain_steering` awaits a disk append per message, so a message steered
    during that await lands after the loop has exited and is NOT in the batch.
    It is genuinely still queued and goes at the next boundary, so settling its
    row on this delivery would claim something that has not happened. FIFO, so
    "the first `count` rows" and "the messages that went" are the same set.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for text in ("first", "second raced in late"):
            await _submit(pilot, app, text)
        assert _notice_texts(app).count(QUEUED_STEER_NOTICE) == 2

        # The drain took ONE of them.
        app.post_message(SteeringDelivered(1))
        await pilot.pause()

        texts = _notice_texts(app)
        assert texts.count(SENT_STEER_NOTICE) == 1
        assert texts.count(QUEUED_STEER_NOTICE) == 1, "the undelivered row still promises"
        assert len(app._queued_steer_notices) == 1

        # The next boundary takes the straggler.
        app.post_message(SteeringDelivered(1))
        await pilot.pause()

        assert _notice_texts(app).count(SENT_STEER_NOTICE) == 2
        assert QUEUED_STEER_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_the_session_announces_the_drain_that_actually_took_messages(
    tmp_path: Path,
) -> None:
    """The engine half: one event per DRAIN, and silence when nothing was queued.

    ``_drain_steering`` is called at every tool and message boundary, so an
    event per call would be noise; the receipt exists only where a promise was
    actually kept. Driven through the real Session because the emission point —
    after persistence, inside the drain — is the fact under test.
    """

    async def _stream(request: Any, signal: Any = None):  # pragma: no cover - never called
        if False:
            yield None

    session = Session(
        model=ModelSpec(provider="anthropic", model_id="sonnet", context_window=200_000),
        stream_fn=_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["system"],
    )
    events: list[Any] = []
    session.subscribe(events.append)

    # Nothing queued: the boundary passes in silence.
    assert await session._drain_steering() == []
    assert [e for e in events if isinstance(e, SteeringDeliveredEvent)] == []

    session.steer("first")
    session.steer("second")
    drained = await session._drain_steering()

    # `getattr` rather than `.text`: the queue's element type is the union
    # `AgentMessage`, and only the `Message` arm declares `text`.
    assert [getattr(message, "text", "") for message in drained] == ["first", "second"]
    delivered = [e for e in events if isinstance(e, SteeringDeliveredEvent)]
    assert len(delivered) == 1, "one receipt per drain, not one per message"
    assert delivered[0].count == 2
    # Persisted before the receipt: the row claims the message is in the
    # conversation, which is only true once it is on disk.
    assert [entry.payload.get("content") for entry in session._transcript.entries()]


@pytest.mark.asyncio
async def test_an_interrupt_spends_the_loud_ink_once() -> None:
    """Ctrl+C prints ONE amber row, not two saying nearly the same thing.

    The standalone `interrupted` notice already fires on this path whenever no
    tool was in flight, which is the common case. A settled queued row in the
    same weight put two amber `!` rows next to each other, one almost a
    substring of the other, which reads as the app stuttering — and spends the
    loudest ink in the palette twice on one event. This codebase already fought
    that battle from the other side: `on_turn_ended` suppresses the standalone
    notice when tool cards carry the interrupt mark.

    The queued row stays at `note`, which is also the honest weight: the state
    did not get worse. The message is still queued and still going, so the row
    has no business getting louder.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    # A NARROW frame on purpose: the property under test (exactly one alarm
    # row) must hold at a width where the settled row is under pressure. The
    # settled row fits on one line down to 53 columns and wraps onto its hanging
    # indent at 52, so 60 exercises a narrow terminal while keeping the
    # settled-row assertion below readable as a single string.
    #
    # The alarm-count assertion does NOT depend on that threshold — it has been
    # verified to hold from 30 to 120 columns — so a copy change that moves the
    # wrap point costs at most the substring assertion, not the property this
    # test exists for.
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "steered into a turn about to be stopped")

        app.post_message(TurnEnded(True, None))
        await pilot.pause()

        rows = [
            strip.text.rstrip()
            for strip in app.screen._compositor.render_strips()
            if strip.text.strip()
        ]
        # The whole point of the fix: ONE amber row for one interrupt, not two.
        # No filtering of the status band is needed here — the band's own
        # `! auto-approve always` is rendered inside the band's chrome and does
        # not begin a strip with `!`, so this already sees transcript rows only.
        # (An earlier version filtered it out defensively; the filter never
        # removed anything at any width, so it only made the test look like it
        # was guarding something it was not.)
        alarms = [row for row in rows if row.lstrip().startswith("!")]
        assert len(alarms) == 1, alarms
        assert "interrupted" in alarms[0]
        # And the settled row is present, quiet, and says the message survives.
        assert any(DEFERRED_STEER_NOTICE in row for row in rows), rows


@pytest.mark.asyncio
async def test_a_dead_sessions_delivery_cannot_settle_the_new_sessions_row() -> None:
    """Issue #160, F3: the receipt is guarded by SESSION, not only by turn.

    `/reload` disposes the outgoing controller, but an event it had ALREADY
    dispatched is a Textual message sitting in the app's queue, and
    unsubscribing cannot recall one. It is handled after the swap cleared the
    held lists — and a user quick enough to steer into the replacement session
    has a row held again by then, so the dying session's drain settled a row
    about a message it knows nothing about.

    Driven by racing dispose deliberately, because nothing else reaches it: the
    outgoing controller's handler is invoked while it is still subscribed (the
    engine emitting on its way down) and the resulting message is left to be
    handled after the swap, which is exactly the real ordering.
    """
    outgoing = _Streaming()
    app = OperatorApp(lambda: _factory(outgoing))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        # The controller is installed by the boot WORKER, so waiting on the
        # condition rather than on a frame count is what keeps this
        # deterministic on a loaded machine — the same race `_submit` documents.
        for _ in range(200):
            if app._controller is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        assert app._controller is not None, "the session never booted"
        dying_controller = app._controller

        # The swap.
        replacement = _Streaming()
        app._session_factory = lambda: _factory(replacement)  # type: ignore[assignment]
        await app._reload_session()
        for _ in range(4):
            await pilot.pause()
        assert app._controller is not dying_controller, "the swap must install a new controller"

        # The user steers into the REPLACEMENT session, so a row is held again.
        await _submit(pilot, app, "an instruction for the new session")
        assert QUEUED_STEER_NOTICE in _notice_texts(app)
        (new_row,) = app._queued_steer_notices

        # NOW the dying session's drain is handled. This is the ordering the
        # race produces and the reason the emptiness check cannot catch it: the
        # outgoing controller emitted on its way down, its message sat in the
        # app's queue behind the swap, and by the time it is handled the lists
        # are non-empty again — holding a row about a DIFFERENT conversation's
        # message. Dispatched through the dead controller's own handler so the
        # message carries exactly the stamp the production path gives it.
        dying_controller._handle_steering_delivered(SteeringDeliveredEvent(count=1))
        await pilot.pause()
        await pilot.pause()

        # The dead session's receipt must not settle it: that message went into
        # a conversation the user cannot see, and this row is still promising a
        # delivery that has not happened.
        assert new_row._text == QUEUED_STEER_NOTICE
        assert SENT_STEER_NOTICE not in _notice_texts(app)
        assert DEFERRED_SENT_STEER_NOTICE not in _notice_texts(app)
        assert app._queued_steer_notices == [new_row]

        # And the LIVE session's own delivery still settles it, so the guard
        # refuses the stale event rather than the event type.
        assert app._controller is not None
        app._controller._handle_steering_delivered(SteeringDeliveredEvent(count=1))
        await pilot.pause()
        assert new_row._text == SENT_STEER_NOTICE


@pytest.mark.asyncio
async def test_the_cross_turn_settle_does_not_change_the_rows_height() -> None:
    """Issue #160, D1/D2: settling a deferred row must not reflow the transcript.

    A notice's height is a step function of its wrap points, so a settle that
    shortens the text can shorten the ROW — pulling everything below it up at a
    moment the user did not act, and, with the transcript scrolled up, leaving
    `scroll_offset` on a viewport that now shows different text. Measured on
    base, the shared 27-character settle shrank the deferred row at every width
    from 28 to 52 columns (`scripts/steer_receipt_transitions.py`).

    `DEFERRED_SENT_STEER_NOTICE` is sized to match, which is the whole reason it
    is 43 characters. This pins that: a word added or dropped from either string
    moves a wrap point at some width and silently brings the jump back, and the
    character counts alone cannot tell you — the round-1 candidate was chosen by
    counting and turned out to be the same length as the string it replaced.

    Every width from 20 up, because the crossovers are not where arithmetic on
    the string lengths puts them: the transcript's own padding and scrollbar are
    in the block's usable width and not in the raw character count. The binding
    constraint is WORD SHAPE rather than length — all three receipt strings in
    play are 43 characters, and `sent — it rode along with the message below`
    still reflows at 22 and 24 because its tail breaks differently.

    Each string is measured in its OWN app, read once through
    `_settled_notice_height`, rather than by reading one block before and after
    a restate. That protocol was racy (review round 1, F2): the pre-restate
    reading is taken while the block's region may not have settled, so it came
    back one row short about once in twelve at 52 columns — and over 33 widths
    that made a red run likely, reporting a copy regression that did not exist.
    The post-restate reading was never wrong; only the baseline slipped.
    """
    for cols in range(20, 91):
        deferred = await _settled_notice_height(cols, DEFERRED_STEER_NOTICE, "note")
        settled = await _settled_notice_height(cols, DEFERRED_SENT_STEER_NOTICE, "success")
        # Naming both strings, because the failure a future reader will hit is
        # "somebody edited the copy", and the message has to point at that
        # rather than at the transcript.
        assert deferred == settled, (
            f"at {cols} columns the receipt strings render to different heights "
            f"({deferred} vs {settled} rows), so the cross-turn settle reflows "
            f"the transcript. The copy was edited and its wrap point moved:\n"
            f"  deferred={DEFERRED_STEER_NOTICE!r}\n"
            f"  settled ={DEFERRED_SENT_STEER_NOTICE!r}\n"
            f"Re-run scripts/steer_receipt_candidates.py to pick a replacement."
        )


@pytest.mark.asyncio
async def test_a_settling_deferred_row_leaves_a_scrolled_up_reader_alone() -> None:
    """The symptom D2 describes, driven end to end rather than measured statically.

    With the transcript scrolled up mid-history, a shrinking row drops
    `virtual_size` while `scroll_offset` stays put, so the same offset shows
    different text and the viewport appears to scroll itself. 52 columns is the
    width at which base was worst (measured: virtual_size 51 -> 49, 8 of 24
    viewport rows changed with nothing the user did).

    The assertion is on the FRAME, not on the string: what the user is promised
    is that nothing below the receipt moves, and a future copy change that
    reintroduces the reflow has to fail here even if it keeps every word this
    test could otherwise look for.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(52, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "and use the staging credentials")
        # History below the receipt, so there is something for a reflow to move
        # and somewhere for the reader to scroll away from the tail.
        for index in range(20):
            app._append_block(NoticeBlock(f"history row {index}", "info"))
        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        await pilot.pause()

        view = app.query_one(TranscriptView)
        view.scroll_to(y=8, animate=False, immediate=True)
        await pilot.pause()
        await pilot.pause()

        before_extent = view.virtual_size.height
        before_offset = view.scroll_offset.y
        before_rows = [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]

        app.post_message(SteeringDelivered(1))
        await pilot.pause()
        await pilot.pause()

        assert view.virtual_size.height == before_extent, "the settle moved the scrollable extent"
        assert view.scroll_offset.y == before_offset
        changed = [
            (index, before, after)
            for index, (before, after) in enumerate(
                zip(
                    before_rows,
                    [strip.text.rstrip() for strip in app.screen._compositor.render_strips()],
                )
            )
            if before != after
        ]
        # EXACTLY the receipt's own rows change. Anything else is content the
        # user did not touch moving under them, which is the report.
        assert all(
            DEFERRED_STEER_NOTICE.split(" — ")[0] in before
            or DEFERRED_SENT_STEER_NOTICE.split(" — ")[0] in after
            or "message" in before
            or "message" in after
            for _, before, after in changed
        ), changed


@pytest.mark.asyncio
async def test_a_takeover_still_settles_the_rows_it_deliberately_kept() -> None:
    """Review round 1, F3: the guard's premise is "the rows went", not "the
    controller changed", and the two come apart on takeover.

    `_adopt_takeover_session` rotates the transport when the remote facade wins
    the transcript lease. Unlike `/reload` it is NOT a conversation change: the
    transcript, the held steer rows and the engine's queue all survive, and its
    docstring says so. So a drain already in flight across that swap is about a
    row this app is still holding, and dropping it leaves the receipt stuck on
    `queued` after the message really went — the exact stale promise #151 and
    #157 exist to prevent, reintroduced by a guard that over-generalised.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for _ in range(200):
            if app._controller is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        assert app._controller is not None, "the session never booted"
        outgoing_controller = app._controller

        await _submit(pilot, app, "steered just before the lease changed hands")
        assert QUEUED_STEER_NOTICE in _notice_texts(app)
        (row,) = app._queued_steer_notices

        # The takeover: a new session object, same conversation, transcript and
        # held rows deliberately kept.
        await app._adopt_takeover_session(_Streaming())
        await pilot.pause()
        assert app._controller is not outgoing_controller
        assert app._queued_steer_notices == [row], "takeover must keep the held rows"

        # The drain that was already in flight when the lease changed hands.
        outgoing_controller._handle_steering_delivered(SteeringDeliveredEvent(count=1))
        await pilot.pause()
        await pilot.pause()

        assert row._text == SENT_STEER_NOTICE, (
            "a takeover keeps the conversation, so the receipt for a message "
            "that really went must still settle its row"
        )
