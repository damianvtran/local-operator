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

from pathlib import Path
from typing import Any, Sequence

import pytest

from local_operator.harness.types import ImageContent, ModelSpec, SteeringDeliveredEvent
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tui.app import (
    QUEUED_STEER_NOTICE,
    SENT_STEER_NOTICE,
    STOPPED_STEER_NOTICE,
    OperatorApp,
)
from local_operator.tui.events import SteeringDelivered, TurnEnded
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

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


def _notice_texts(app: OperatorApp) -> list[str]:
    """Every notice row's text, in transcript order."""
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


async def _submit(pilot: Any, app: OperatorApp, text: str) -> None:
    app.query_one(Editor).text = text
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
        assert STOPPED_STEER_NOTICE in texts
        assert QUEUED_STEER_NOTICE not in texts
        assert SENT_STEER_NOTICE not in texts, "nothing was delivered"
        assert app._queued_steer_notices == []


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
