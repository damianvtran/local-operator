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
from typing import Any, Sequence

import pytest

from local_operator.harness.types import ImageContent, ModelSpec, SteeringDeliveredEvent
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tui.app import (
    DEFERRED_STEER_NOTICE,
    QUEUED_STEER_NOTICE,
    SENT_STEER_NOTICE,
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
        # The message is still in the engine's queue, so the row must not claim
        # it was lost — the user would retype and the agent would get it twice.
        assert "not sent" not in DEFERRED_STEER_NOTICE
        assert "still queued" in DEFERRED_STEER_NOTICE


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
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "steered into a turn about to be stopped")

        app.post_message(TurnEnded(True, None))
        await pilot.pause()

        rows = [
            strip.text.rstrip()
            for strip in app.screen._compositor.render_strips()
            if strip.text.strip()
        ]
        alarms = [row for row in rows if row.lstrip().startswith("!")]
        # The band's `! auto-approve always` lives on the status line and is not
        # a transcript row; the interrupt notice is the only one in the ledger.
        transcript_alarms = [row for row in alarms if "auto-approve" not in row]
        assert len(transcript_alarms) == 1, transcript_alarms
        assert "interrupted" in transcript_alarms[0]
        # And the settled row is present, quiet, and says the message survives.
        assert any(DEFERRED_STEER_NOTICE in row for row in rows), rows
