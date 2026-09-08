"""Esc lifts the newest still-queued mid-turn steer back into the composer.

The report: a message sent while the agent is working is queued ("queued —
sends when this step finishes"), and the only way to change one's mind was to
wait for it to land and steer again — or retype it. Esc is the app's cancel
key, so Esc with a queued steer now UNSENDS the newest one: the message leaves
the engine's steering queue, its rows (the user row, its images, the queued
receipt) leave the transcript, and the composer holds the text ready for an
immediate edit-and-resend.

The pairing is by MESSAGE ID, with object identity kept as a fast path. The
app hands the session the very ``Message`` it queued (``steer_message``), and
the in-process ``Session`` gives that object back — but a ``RemoteSession``
rebuilds its queue snapshot out of serialized frontend state, so on every
daemon-attached session the objects differ and only the id survives. The id is
what the recall already crosses the process boundary on (``command_id``), so
it is the seam's real identity; older steers and wake deliveries that ride the
same queue are still never what a recall removes.
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
    RECALL_AMBIGUOUS_NOTICE,
    RECALL_DECLINE_NOTICE,
    RECALL_UNCONFIRMED_NOTICE,
    SENT_STEER_NOTICE,
    OperatorApp,
)
from local_operator.tui.events import SteeringDelivered, TurnEnded
from local_operator.tui.widgets.editor import Attachment, Editor
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
        # The map holds both payload shapes now; an image paste must still put
        # an `Attachment` in it, not merely something marker-shaped.
        recalled = attachments[1]
        assert isinstance(recalled, Attachment)
        assert recalled.image.data == original, "the original bytes ride the recall"
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


class _RemoteLikeStreaming(_Streaming):
    """A mid-turn fake with ``RemoteSession``'s queued_steering semantics.

    The in-process ``Session`` drains and re-puts the very objects it was
    handed, so a snapshot's entries ARE the app's messages.
    ``RemoteSession.queued_steering`` cannot do that: the queue lives in
    another process and reaches this one as serialized frontend state, so it
    rebuilds a fresh ``Message`` per item on every call. Equal ids, never the
    same object — which is what the TUI's pointer-identity match silently
    failed on for every daemon-attached session.

    Modelled rather than driven through a real socket because the property
    under test is the app's matching rule, not the transport: what a fake can
    get wrong here is only whether it rebuilds, and it does.
    """

    def steer_message(self, message: Any) -> None:
        # Stored as the WIRE shape (a dict), so no Message object survives in
        # this fake for the app to accidentally match by identity.
        self._steering_queue.append({"id": message.id, "text": message.text})

    def queued_steering(self) -> list[Any]:
        return [Message.user(item["text"], id=item["id"]) for item in self._steering_queue]

    def recall_steering(self, message: Any) -> bool:
        wanted = str(getattr(message, "id", "") or "")
        for index, item in enumerate(self._steering_queue):
            if item["id"] == wanted:
                del self._steering_queue[index]
                return True
        return False


class _IdLessStreaming(_Streaming):
    """A follower whose OWNER is too old to put ``id`` on its queued-steer rows.

    ``RemoteSession.queued_steering`` substitutes ``UNIDENTIFIED_STEER_ID`` for
    such an item, so every id-less entry arrives under one key. Modelled with
    the real substitution rather than a hand-written literal, so a rename of
    the constant moves this fake with it.
    """

    def steer_message(self, message: Any) -> None:
        self._steering_queue.append({"text": message.text})  # no id on the wire

    def queued_steering(self) -> list[Any]:
        from local_operator.session.remote import UNIDENTIFIED_STEER_ID

        return [
            Message.user(
                str(item.get("text", "") or ""),
                id=str(item.get("id", "") or UNIDENTIFIED_STEER_ID),
            )
            for item in self._steering_queue
        ]

    def recall_steering(self, message: Any) -> bool:
        ids = {str(item.get("id", "") or "") for item in self._steering_queue}
        return str(getattr(message, "id", "") or "") in ids


@pytest.mark.asyncio
async def test_esc_recalls_a_steer_from_a_session_that_rebuilds_its_queue() -> None:
    """The daemon case: equal-but-distinct messages must still be recallable.

    Pointer identity was a single-process assumption. Every session attached
    to a ``kind=daemon`` runtime goes through ``RemoteSession``, whose
    ``queued_steering`` rebuilds its entries, so the match found nothing and
    Esc-recall was a total, SILENT no-op there — the message stayed queued and
    was delivered anyway, while the user believed they had taken it back.
    """
    session = _RemoteLikeStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "Ok after you did that, it seems to work now")
        assert len(session.queued_steering()) == 1
        # The premise of the test: nothing in the snapshot IS the held object.
        held = app._held_steer_blocks[-1][0]
        snapshot = session.queued_steering()
        assert not any(item is held for item in snapshot), "the fake must rebuild"
        assert snapshot[0].id == held.id, "...while preserving the id"

        await pilot.press("escape")
        await pilot.pause()

        # Unsent: the message left the queue, so it cannot ride a later
        # boundary and be delivered a second time.
        assert session.queued_steering() == []
        assert editor.text == "Ok after you did that, it seems to work now"
        assert QUEUED_STEER_NOTICE not in _notice_texts(app)
        assert _user_texts(app) == []


@pytest.mark.asyncio
async def test_an_ambiguous_queue_id_declines_instead_of_guessing() -> None:
    """An id that names two queue entries is not an identity, so it must not match.

    ``RemoteSession.queued_steering`` substitutes the literal ``remote-steer``
    for a wire item carrying no id, and any wire that repeats an id has the
    same shape: one key, several messages. Matching on it would unsend one
    message while handing the composer another one's text. The recall declines
    instead — the steers stay queued and ride the next boundary, exactly as if
    Esc had not been pressed, which is the same "cost the user nothing" rule
    the read-only and half-typed-draft declines follow.

    The app's OWN ids are minted per submit and therefore unique, so this
    guards the queue the app is told about rather than one it can produce; a
    duplicated id is modelled here directly because that is the only way the
    condition is reachable from outside.
    """
    session = _RemoteLikeStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "older steer")
        await _submit(pilot, editor, "newer steer")
        # The wire reports the newest steer's id TWICE — one key, two messages.
        newest_id = app._held_steer_blocks[-1][0].id
        for item in session._steering_queue:
            item["id"] = newest_id
        assert [m.id for m in session.queued_steering()] == [newest_id, newest_id]

        await pilot.press("escape")
        await pilot.pause()

        # Nothing lifted and nothing unsent: an ambiguous match is no match.
        assert editor.text == ""
        assert [m.text for m in session.queued_steering()] == ["older steer", "newer steer"]
        assert _user_texts(app) == ["older steer", "newer steer"]
        # ...and it SAYS so. A press that changes nothing on screen is the
        # dropped-keystroke reading D1 filed the decline row against; this case
        # needs its own row because the composer is not the obstacle.
        assert RECALL_AMBIGUOUS_NOTICE in _notice_texts(app)
        assert RECALL_DECLINE_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_a_rejected_remote_recall_warns_instead_of_double_sending() -> None:
    """The owner refused: the composer holds text the session already sent.

    A follower's recall is optimistic — the composer takes the text and the
    rows go before the owner answers. When the drain won the race the owner
    answers ``that steering message is no longer queued``, and the message is
    both delivered AND drafted here. Pressing Enter would send it twice, which
    is the reported double-send, so the app says so rather than leaving the
    user to infer it from the agent acting on one instruction twice.
    """
    session = _RemoteLikeStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "use 0.75 for the direct API")
        message_id = app._held_steer_blocks[-1][0].id

        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "use 0.75 for the direct API"

        # The owner's rejection arrives after the press has already returned.
        app._on_recall_rejected(session, message_id)
        await pilot.pause()

        assert RECALL_UNCONFIRMED_NOTICE in _notice_texts(app)
        # The draft is NOT thrown away: discarding what the user may want to
        # edit is the loss `action_stop` forbids. The row warns; the user decides.
        assert editor.text == "use 0.75 for the direct API"


@pytest.mark.asyncio
async def test_an_unrecallable_newest_steer_never_falls_back_to_an_older_one() -> None:
    """The scan STOPS at an unnameable newest entry; it does not substitute.

    Round-1 review BLOCKER-1. The guard refused the ambiguous newest candidate
    but let the `for` keep walking backwards, so the next-older entry — whose
    id happened to be unique — was recalled instead: the press unsent and
    lifted a message the user never pointed at, while the one they did mean to
    take back stayed queued and was delivered. That is strictly worse than the
    silent no-op it replaced, and it is the precise harm the guard was written
    to prevent, one entry over.
    """
    session = _RemoteLikeStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "OLD: use the staging bucket")
        await _submit(pilot, editor, "NEW: actually stop and revert")
        # A third queue entry duplicates the NEWEST id, so the newest held
        # entry is unnameable while the OLDER one is still perfectly unique.
        newest_id = app._held_steer_blocks[-1][0].id
        session._steering_queue.append({"id": newest_id, "text": "wake delivery"})
        older_id = app._held_steer_blocks[0][0].id
        assert [m.id for m in session.queued_steering()].count(older_id) == 1

        await pilot.press("escape")
        await pilot.pause()

        # The OLDER steer is what a fallback would have taken. It must not.
        assert editor.text == "", "no message may be lifted when the newest is unnameable"
        assert [m.text for m in session.queued_steering()] == [
            "OLD: use the staging bucket",
            "NEW: actually stop and revert",
            "wake delivery",
        ], "nothing may be unsent"
        assert _user_texts(app) == ["OLD: use the staging bucket", "NEW: actually stop and revert"]
        assert RECALL_AMBIGUOUS_NOTICE in _notice_texts(app)


@pytest.mark.asyncio
async def test_an_id_less_queue_entry_is_answered_rather_than_ignored() -> None:
    """`UNIDENTIFIED_STEER_ID` names every id-less entry, so it names none.

    Round-1 review MAJOR-3. The guard keyed on the HELD id — always an
    app-minted uuid4 — which can never equal the placeholder a `RemoteSession`
    substitutes for a wire item with no id. So the case the guard's own
    docstring was written around never reached it, and the press was a silent
    dropped keystroke: exactly the D1 failure the PR claimed to have closed.
    """
    session = _IdLessStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "please stop")
        from local_operator.session.remote import UNIDENTIFIED_STEER_ID

        assert [m.id for m in session.queued_steering()] == [UNIDENTIFIED_STEER_ID]
        assert app._held_steer_blocks[-1][0].id != UNIDENTIFIED_STEER_ID, "ids cannot be compared"

        await pilot.press("escape")
        await pilot.pause()

        # Declining is right; being SILENT about it is the defect.
        assert editor.text == ""
        assert [m.text for m in session.queued_steering()] == ["please stop"]
        assert RECALL_AMBIGUOUS_NOTICE in _notice_texts(app)


@pytest.mark.asyncio
async def test_a_stale_recall_refusal_never_paints_on_another_conversation() -> None:
    """A late ack belongs to the conversation that issued it, or to nothing.

    Round-1 review BLOCKER-2. The refusal crosses a socket with a 15 s ack
    timeout, so `/clear`, `/new`, `/resume`, a sidebar switch or a takeover can
    all land first. A row reading "that steer was sent" on a conversation that
    never sent it is worse than silence: it is the double-send warning aimed at
    text the user never queued.
    """
    first = _RemoteLikeStreaming()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "use 0.75 for the direct API")
        message_id = app._held_steer_blocks[-1][0].id
        await pilot.press("escape")
        await pilot.pause()

        # The user moves to a DIFFERENT conversation before the owner answers.
        second = _RemoteLikeStreaming()
        app._adopt_session(second, replay_history=False)
        await pilot.pause()
        assert app._session is second

        # The first conversation's refusal finally arrives.
        app._on_recall_rejected(first, message_id)
        await pilot.pause()

        assert RECALL_UNCONFIRMED_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_the_double_send_warning_survives_the_next_escape() -> None:
    """The warning outlives the reflex press, because its risk is unresolved.

    Design round 1, D1. Written through the Esc ladder's single slot, the
    warning was replaced by the NEXT press's decline row — and that row invites
    the exact wrong action ("esc again to recall") over a composer holding the
    duplicate. Pressing Esc again after reading that something went wrong is
    the reflex; the row has to be a transcript fact, not a ladder state.
    """
    session = _RemoteLikeStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        # TWO steers: with only one queued the second press finds nothing and
        # returns before touching the slot, which is why this hid in testing.
        await _submit(pilot, editor, "first steer")
        await _submit(pilot, editor, "use 0.75 for the direct API")
        message_id = app._held_steer_blocks[-1][0].id

        await pilot.press("escape")
        await pilot.pause()
        app._on_recall_rejected(session, message_id)
        await pilot.pause()
        assert RECALL_UNCONFIRMED_NOTICE in _notice_texts(app)

        # The reflex press. The composer is dirty, so this is the DECLINE path.
        await pilot.press("escape")
        await pilot.pause()

        assert RECALL_DECLINE_NOTICE in _notice_texts(app), "the decline still speaks"
        assert RECALL_UNCONFIRMED_NOTICE in _notice_texts(app), "and the warning survives it"
        assert editor.text == "use 0.75 for the direct API"


@pytest.mark.asyncio
async def test_a_successful_recall_retires_the_ambiguity_row() -> None:
    """`it is still queued` must not outlive the recall that unsends the steer.

    Design round 3 D6 / review round 2 MINOR-2. The ambiguity row asserts a
    PRESENT-TENSE fact and lives in the Esc ladder's single slot, but the
    successful-recall path retired that slot only when it held the decline
    row. Both routes into ambiguity clear on their own — a stale replicated
    ``frontend_state`` that stops doubling an id once the socket pump catches
    up, and an id-less entry draining at a boundary — so the very next press
    succeeds and leaves a row standing that says the steer is queued while it
    is sitting in the composer.

    That is the same stale-row class the decline row is retired for, on a
    surface whose whole argument is that stale promises get retired.
    """
    session = _RemoteLikeStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "use 0.75 for the direct API")
        # A stale snapshot doubles the id, so this press cannot name the steer.
        held_id = app._held_steer_blocks[-1][0].id
        session._steering_queue.append({"id": held_id, "text": "a stale duplicate"})

        await pilot.press("escape")
        await pilot.pause()
        assert RECALL_AMBIGUOUS_NOTICE in _notice_texts(app)
        assert editor.text == "", "nothing was recalled while the id was ambiguous"

        # The pump catches up: the duplicate is gone and the id names one entry.
        session._steering_queue = [
            item for item in session._steering_queue if item["text"] != "a stale duplicate"
        ]
        await pilot.press("escape")
        await pilot.pause()

        # The recall worked...
        assert editor.text == "use 0.75 for the direct API"
        assert session.queued_steering() == []
        # ...so the row claiming it is still queued must be gone with it.
        remaining = _notice_texts(app)
        assert (
            RECALL_AMBIGUOUS_NOTICE not in remaining
        ), f"the ambiguity row outlived the recall that made it false: {remaining}"
