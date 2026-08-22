"""Esc lifts the newest still-queued mid-turn steer back into the composer.

The report: a message sent while the agent is working is queued ("queued —
sends when this step finishes"), and the only way to change one's mind was to
wait for it to land and steer again — or retype it. Esc is the app's cancel
key, so Esc with a queued steer now UNSENDS the newest one: the message leaves
the engine's steering queue, its rows (the user row, its images, the queued
receipt) leave the transcript, and the composer holds the text ready for an
immediate edit-and-resend.

The pairing is by IDENTITY: the app hands the session the very ``Message`` it
queued (``steer_message``), so a recall can name exactly the object the
composer is about to hold — equal-but-distinct messages, older steers, and
wake deliveries that ride the same queue are never what a recall removes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message, ModelSpec
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
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock

from .test_app_pilot import FakeSession, _factory


class _Streaming(FakeSession):
    """A fake that is mid-turn, so a submit is STEERED rather than prompted."""

    @property
    def is_streaming(self) -> bool:
        return True


def _notice_texts(app: OperatorApp) -> list[str]:
    """Every notice row's text, in transcript order."""
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _user_texts(app: OperatorApp) -> list[str]:
    """Every user row's text, in transcript order."""
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock)
    ]


async def _boot(pilot: Any, app: OperatorApp) -> Editor:
    """Wait for the session and focus the composer, as the app's own tests do."""
    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    assert app._session is not None, "the session never booted"
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    return editor


async def _submit(pilot: Any, editor: Editor, text: str) -> None:
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_esc_recalls_the_newest_queued_steer_into_the_composer() -> None:
    """One press: the queue loses the message, the composer gains the text."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "use 0.75 for the direct API")
        assert len(session.queued_steering()) == 1
        assert QUEUED_STEER_NOTICE in _notice_texts(app)
        assert "use 0.75 for the direct API" in _user_texts(app)

        await pilot.press("escape")
        await pilot.pause()

        # Unsent: the engine will never see the message.
        assert session.queued_steering() == []
        # The composer holds the text, cursor-ready for a resend.
        assert editor.text == "use 0.75 for the direct API"
        # The transcript lost the steer's rows: no promise, no user row.
        assert QUEUED_STEER_NOTICE not in _notice_texts(app)
        assert DEFERRED_STEER_NOTICE not in _notice_texts(app)
        assert SENT_STEER_NOTICE not in _notice_texts(app)
        assert _user_texts(app) == []
        # And the history does not offer the unsent line as a past prompt.
        assert editor._history == []


@pytest.mark.asyncio
async def test_a_resend_after_recall_is_a_fresh_steer() -> None:
    """Enter on the recalled text re-queues it; nothing double-sends."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "first wording")
        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "first wording"

        editor.text = "second wording"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert [m.text for m in session.queued_steering()] == ["second wording"]
        assert _user_texts(app) == ["second wording"]
        assert _notice_texts(app).count(QUEUED_STEER_NOTICE) == 1


@pytest.mark.asyncio
async def test_only_the_newest_queued_steer_is_recalled() -> None:
    """Two queued steers: one Esc takes the newest and leaves the older one."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "older steer")
        await _submit(pilot, editor, "newer steer")
        assert len(session.queued_steering()) == 2

        await pilot.press("escape")
        await pilot.pause()

        assert editor.text == "newer steer"
        assert [m.text for m in session.queued_steering()] == ["older steer"]
        assert _user_texts(app) == ["older steer"]
        assert _notice_texts(app).count(QUEUED_STEER_NOTICE) == 1


@pytest.mark.asyncio
async def test_a_delivered_steer_is_not_recallable() -> None:
    """Once the engine drains the queue, Esc has nothing to take back."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "already delivered")
        app.post_message(SteeringDelivered(1))
        await pilot.pause()
        assert SENT_STEER_NOTICE in _notice_texts(app)

        await pilot.press("escape")
        await pilot.pause()

        # The composer is untouched: the message was sent, Esc is a stop.
        assert editor.text == ""
        # The delivered rows stay exactly where they were.
        assert SENT_STEER_NOTICE in _notice_texts(app)
        assert "already delivered" in _user_texts(app)


@pytest.mark.asyncio
async def test_a_recalled_draft_keeps_its_original_screenshot(tmp_path: Path) -> None:
    """The recall hands back the ORIGINAL bytes, not the transcript's blur.

    The transcript's ImageBlocks keep a downscaled copy of the pixels; a
    recall rebuilt from them would resend a blur of what the user pasted.
    The held entry carries the submit-time attachment map instead, so the
    recalled draft's markers resolve to the very bytes that were queued.
    """
    import base64

    from PIL import Image
    from textual import events

    path = tmp_path / "shot.png"
    Image.new("RGB", (64, 32), (30, 30, 40)).save(path)
    original = base64.b64encode(path.read_bytes()).decode()

    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.insert("check this ")
        app.post_message(events.Paste(str(path)))
        await pilot.pause()
        await pilot.pause()
        assert "[Image #1" in editor.text
        await pilot.press("enter")
        await pilot.pause()
        assert len(session.queued_steering()) == 1

        await pilot.press("escape")
        await pilot.pause()

        assert "[Image #1" in editor.text
        attachments = editor.attachments()
        assert list(attachments) == [1]
        assert attachments[1].image.data == original, "the original bytes ride the recall"
        # And the resend resolves the marker to that image.
        assert [image.data for image in editor.referenced_images()] == [original]


@pytest.mark.asyncio
async def test_recall_works_after_the_turn_has_ended() -> None:
    """The deferred state: the turn stopped, the steer is still queued.

    This is the moment the report describes — the user stopped the turn to
    steer, and the queued message is exactly what they want back. The rows
    now read `still queued`, and Esc must lift the message out of the queue
    and the deferred row off the transcript.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "steer into a dying turn")
        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        assert DEFERRED_STEER_NOTICE in _notice_texts(app)

        await pilot.press("escape")
        await pilot.pause()

        assert editor.text == "steer into a dying turn"
        assert session.queued_steering() == []
        assert DEFERRED_STEER_NOTICE not in _notice_texts(app)
        assert _user_texts(app) == []


@pytest.mark.asyncio
async def test_recall_declines_over_a_half_typed_draft() -> None:
    """The cancel key never throws away what the user is typing."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "queued steer")
        editor.text = "half typed"
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        # The draft survives, the steer stays queued, the rows stay up.
        assert editor.text == "half typed"
        assert [m.text for m in session.queued_steering()] == ["queued steer"]
        assert QUEUED_STEER_NOTICE in _notice_texts(app)
        # And the decline is not silent: the one row names the obstacle and
        # the recovery (design round 1, D1).
        assert any("esc again to recall" in text for text in _notice_texts(app))

        # The advertised recovery: clear the buffer, Esc again — the steer is
        # recalled, and the decline row that advertised the recall retires
        # with the steer's own rows (design round 2, D4).
        editor.text = ""
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "queued steer"
        assert session.queued_steering() == []
        assert not any("esc again to recall" in text for text in _notice_texts(app))


@pytest.mark.asyncio
async def test_recall_does_not_steal_the_stop_escalation_ladder() -> None:
    """With children running, the first Esc offers the wider stop and returns.

    The recall must not run on that press: the ladder's contract is that the
    first press reports and the second acts, and a recall in between would
    leave the second press recalling instead of stopping the children.
    """
    session = _Streaming()
    session.running_children = 1
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "queued steer")

        await pilot.press("escape")
        await pilot.pause()
        # The offer press reports on children and recalls nothing.
        assert editor.text == ""
        assert [m.text for m in session.queued_steering()] == ["queued steer"]

        await pilot.press("escape")
        await pilot.pause()
        # The escalation press stops the children and STILL recalls nothing:
        # both presses of the ladder are the children's contract.
        assert session.subagent_cancels
        assert editor.text == ""
        assert [m.text for m in session.queued_steering()] == ["queued steer"]

        # Once the children are gone, the next press is an ordinary stop —
        # and recalls the steer.
        session.running_children = 0
        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "queued steer"
        assert session.queued_steering() == []


@pytest.mark.asyncio
async def test_recall_pops_the_prompt_history_entry() -> None:
    """Up-arrow after a recall must not offer the unsent line as a past prompt."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "an earlier real prompt")
        # A turn ends so the next submit is a fresh turn, not a steer.
        app.post_message(TurnEnded(False, None))
        await pilot.pause()
        session._streaming = False  # type: ignore[attr-defined]
        await _submit(pilot, editor, "queued steer")
        session._streaming = True  # type: ignore[attr-defined]
        app.post_message(TurnEnded(False, None))
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "queued steer"

        # Navigate: Up shows the older real prompt, never the unsent line.
        editor.focus()
        await pilot.pause()
        editor.text = ""
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert editor.text == "an earlier real prompt"
        await pilot.press("up")
        await pilot.pause()
        assert editor.text == "an earlier real prompt", "the unsent line is not offered"


@pytest.mark.asyncio
async def test_the_session_recall_is_identity_scoped(tmp_path: Path) -> None:
    """recall_steering removes exactly the object handed to it, in order.

    Driven on the real Session: the identity semantics are the contract the
    TUI's block pairing rests on, so they are pinned where the queue lives,
    not against a fake.
    """

    async def _stream(request: Any, signal: Any = None):  # pragma: no cover
        if False:
            yield None

    session = Session(
        model=ModelSpec(provider="anthropic", model_id="sonnet", context_window=200_000),
        stream_fn=_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["system"],
    )
    first = Message.user("same text")
    second = Message.user("same text")
    session.steer_message(first)
    session.steer_message(second)

    # Equal-but-distinct: recalling the SECOND leaves the first in place.
    assert session.recall_steering(second) is True
    held = session.queued_steering()
    # `getattr` rather than `.text`: the queue's element type is the union
    # `AgentMessage`, and only the `Message` arm declares `text`.
    assert [getattr(m, "text", "") for m in held] == ["same text"]
    assert held[0] is first
    # The snapshot preserves FIFO order and identity.
    assert session.queued_steering()[0] is first
    # A second recall of the same object finds nothing; the other still goes.
    assert session.recall_steering(second) is False
    assert session.recall_steering(first) is True
    assert session.queued_steering() == []
    # And a recall never disturbs the drain: what remains is what goes.
    session.steer_message(first)
    drained = await session._drain_steering()
    assert [getattr(m, "text", "") for m in drained] == ["same text"]
