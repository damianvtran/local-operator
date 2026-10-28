"""An accepted steer must outlive the runtime that accepted it.

THE DEFECT (operator report, 2026-10-28, session ``1cfce255afc3``). A message sent
while a turn is running is admitted as a steer and answered 200
``{"status":"admitted"}``, but the steering queue is IN MEMORY and the durable
transcript row is only written at the next tool boundary
(``Session._drain_steering``). A runtime replaced before that boundary takes the
queue with it: no transcript row, no error to the producer, no durable notice.
Two command ids that both got 200 appeared nowhere but the frontend checkpoint
row, and the session's epoch changed 35 s later.

So the serving handle spools every accepted steer for a successor BEFORE it
acknowledges — ``inbox.jsonl``, the vehicle the update window already uses — the
live turn drops that copy once its own row is durable, and the successor replays
it exactly once, keyed on the producer command id (``producer_command_id``).
These cells drive the real ``Session`` behind the production handle and the real
boot drain (``process._drain_inbox_into``), never a re-implementation of either.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ImageContent, NoticeEvent, StreamEvent
from local_operator.session.errors import RuntimeRetiring
from local_operator.session.runtime import process as process_mod
from local_operator.session.runtime.inbox import (
    SOURCE_USER,
    InboxLine,
    append_inbox,
    peek_inbox,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, last_user_text, text_turn

#: The id the live incident carried, kept verbatim so a reader can match this
#: cell against the transcript the report quotes.
LOST_COMMAND_ID = "157224d3-8892-47b2-88c5-64e95c1e611f"

STEER_TEXT = "hey can you try to send Damian a message"


class _ParkedStream(ScriptedStream):
    """``ScriptedStream`` that parks the provider call asking ``park_on``.

    The call is identified by its QUESTION, not by its index: a runtime's first
    prompt also spends a provider call on naming the conversation (the handle's
    own worker), so an index-based gate parks the naming call and lets the turn
    it was meant to hold fail on an exhausted tape. Parked inside the TURN's
    call, the stream holds ``_is_streaming`` for exactly as long as the test
    needs: a steer only exists as a queued message while some turn is running.
    """

    def __init__(self, turns: Sequence[Sequence[StreamEvent]], *, park_on: str) -> None:
        super().__init__(turns)
        self.park_on = park_on
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def __call__(self, request: Any, signal: Any = None) -> AsyncIterator[StreamEvent]:
        inner = super().__call__(request, signal)
        parks = last_user_text(request) == self.park_on

        async def gen() -> AsyncIterator[StreamEvent]:
            if parks:
                self.entered.set()
                await self.release.wait()
            async for event in inner:
                yield event

        return gen()


def _message_rows(directory: Path) -> list[dict[str, Any]]:
    """Every durable MESSAGE row, read from disk — the authority a restart reads."""
    path = directory / "transcript.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "message":
            rows.append(record)
    return rows


def _rows_produced_by(directory: Path, command_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in _message_rows(directory)
        if row.get("payload", {}).get("producer_command_id") == command_id
    ]


async def _live_turn_with_a_queued_steer(
    root: Path, command_id: str
) -> tuple[Any, ServingSessionHandle, _ParkedStream, Path]:
    """A real session with a turn parked mid-provider and a steer queued on it.

    The turn is NEVER released: this is the runtime that will not consume its
    queue again. It stays reachable only because an in-process test cannot
    SIGKILL itself — nothing in it runs another tool boundary, which is the whole
    of the fact the defect turns on. ``dispose()`` is deliberately not called
    before the successor boots: that path rejects queued steers, and the incident
    is the one where it never ran.
    """
    directory = root / "session"
    stream = _ParkedStream([text_turn("working")], park_on="start the work")
    session = build_session(directory, stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    # The conversation-naming worker is a paid side call over the same stream,
    # and it is nothing these cells assert about — stubbed the way the drain
    # cells stub it, so the tape's one turn belongs to the turn under test.
    handle._maybe_name_conversation = lambda text: None
    assert await handle.prompt("start the work", command_id="prompt-1") == "prompt admitted"
    await asyncio.wait_for(stream.entered.wait(), 5)
    assert session._turn_lock.locked(), "premise: a turn is running, so a steer queues"
    assert await handle.steer(STEER_TEXT, command_id=command_id) == "steering queued"
    return session, handle, stream, directory


async def _boot_successor(
    root: Path,
) -> tuple[Any, ServingSessionHandle, ScriptedStream]:
    """A fresh runtime over the same directory, through the real boot drain.

    The tape carries spare turns on purpose: a runtime's first prompt also spends
    a provider call naming the conversation (``Session``'s naming worker), so a
    tape sized to the replay alone runs out on the call that is not the replay.
    """
    stream = ScriptedStream([text_turn(f"replayed {index}") for index in range(4)])
    session = build_session(root / "session", stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    # Same naming stub as the live runtime, so a replay's tape is spent on the
    # replay rather than on a title.
    handle._maybe_name_conversation = lambda text: None
    await process_mod._drain_inbox_into(handle)
    return session, handle, stream


async def _stop_quietly(handle: ServingSessionHandle) -> None:
    with suppress(Exception):
        await asyncio.wait_for(handle.dispose(), 10)


@pytest.mark.asyncio
async def test_an_accepted_steer_survives_the_runtime_that_accepted_it(tmp_path: Path) -> None:
    """The reproduction: an acked steer must be findable by the successor.

    On the unfixed tree the successor's boot drain finds an empty inbox and an
    empty steering queue — the message the producer was told was queued is gone
    with the runtime that acked it.
    """
    _, first_handle, _, directory = await _live_turn_with_a_queued_steer(tmp_path, LOST_COMMAND_ID)
    assert (
        _rows_produced_by(directory, LOST_COMMAND_ID) == []
    ), "premise: the steer is not a durable row yet — it is waiting for a boundary"

    _, successor, _ = await _boot_successor(tmp_path)
    try:
        rows = _rows_produced_by(directory, LOST_COMMAND_ID)
        assert len(rows) == 1, (
            "the accepted steer reached no durable row and no steering queue: the "
            "producer was told 'steering queued' and the successor found nothing"
        )
        payload = rows[0]["payload"]
        assert payload.get("role") == "user", payload
        assert STEER_TEXT in json.dumps(payload), payload
    finally:
        await _stop_quietly(successor)
        await _stop_quietly(first_handle)


@pytest.mark.asyncio
async def test_the_live_turn_drops_the_spooled_copy_once_the_row_is_durable(
    tmp_path: Path,
) -> None:
    """Req 4, normal operation: no double delivery after the live turn drains it.

    The successor's drain runs BEFORE its socket listens, so a spool row left
    behind by a live turn that already admitted the steer would be delivered a
    second time as a fresh prompt. The drain that writes the row is therefore the
    one that removes the spool copy — called directly here because that is
    exactly the call the loop makes at a tool boundary.
    """
    _, handle, stream, directory = await _live_turn_with_a_queued_steer(tmp_path, "steer-live-1")
    try:
        spooled = peek_inbox(directory)
        assert [row.command_id for row in spooled] == ["steer-live-1"], spooled
        assert spooled[0].source == SOURCE_USER, spooled[0]

        delivered = await handle._session._drain_steering()  # type: ignore[attr-defined]
        assert [m.text for m in delivered] == [STEER_TEXT]
        assert peek_inbox(directory) == [], "the spool kept a copy the transcript now owns"
        assert len(_rows_produced_by(directory, "steer-live-1")) == 1
    finally:
        stream.release.set()
        await _stop_quietly(handle)


@pytest.mark.asyncio
async def test_a_resend_of_the_replayed_command_id_is_not_a_second_copy(tmp_path: Path) -> None:
    """Req 2: exactly-once is keyed on the producer command id, not on the file.

    A producer that reconnects and sends the SAME message again (its ack was
    lost, or its receipt journal replays it) is answered from the durable index
    and reaches no second row.
    """
    _, first_handle, _, directory = await _live_turn_with_a_queued_steer(tmp_path, LOST_COMMAND_ID)
    _, successor, _ = await _boot_successor(tmp_path)
    try:
        assert len(_rows_produced_by(directory, LOST_COMMAND_ID)) == 1

        assert await successor.steer(STEER_TEXT, command_id=LOST_COMMAND_ID) == "already admitted"
        assert (
            len(_rows_produced_by(directory, LOST_COMMAND_ID)) == 1
        ), "the resend duplicated the row"
    finally:
        await _stop_quietly(successor)
        await _stop_quietly(first_handle)


@pytest.mark.asyncio
async def test_a_lost_spool_removal_still_delivers_the_message_once(tmp_path: Path) -> None:
    """Req 2/4: a crash between the append and the removal must not duplicate.

    The removal is the fast path, not the authority: the producer identity on the
    transcript row is. Putting the row back stands in for the removal that never
    ran.
    """
    _, first_handle, _, directory = await _live_turn_with_a_queued_steer(tmp_path, LOST_COMMAND_ID)
    _, successor, _ = await _boot_successor(tmp_path)
    try:
        assert len(_rows_produced_by(directory, LOST_COMMAND_ID)) == 1

        assert append_inbox(
            directory,
            InboxLine(
                text=STEER_TEXT,
                sender={},
                mode="mailbox",
                wake=True,
                source=SOURCE_USER,
                command_id=LOST_COMMAND_ID,
            ),
        ), "premise: the spool now holds the copy a lost removal would have left"

        _, third, _ = await _boot_successor(tmp_path)
        try:
            assert (
                len(_rows_produced_by(directory, LOST_COMMAND_ID)) == 1
            ), "a stale spool row delivered the message a second time"
        finally:
            await _stop_quietly(third)
    finally:
        await _stop_quietly(successor)
        await _stop_quietly(first_handle)


@pytest.mark.asyncio
async def test_an_attachment_steer_is_refused_rather_than_accepted_and_dropped(
    tmp_path: Path,
) -> None:
    """Req 3: the one message an inbox row cannot carry fails visibly.

    An ``InboxLine`` is text, so an image/audio/document steer cannot be held for
    a successor. Accepting it would reintroduce the silent loss this change
    exists to end, so it takes the typed refusal the update window already uses
    for the same reason — and the identity stays retryable.
    """
    _, handle, stream, directory = await _live_turn_with_a_queued_steer(tmp_path, "steer-image-0")
    try:
        with pytest.raises(RuntimeRetiring):
            await handle.steer(
                "look at this",
                images=[ImageContent(data="AAAA", mime_type="image/png")],
                command_id="steer-image-1",
            )
        assert [row.command_id for row in peek_inbox(directory)] == [
            "steer-image-0"
        ], "the refused steer left a spool row"
        assert _rows_produced_by(directory, "steer-image-1") == []

        # The identity a refusal releases must be reusable, or the producer's
        # retry without the attachment would answer "already admitted".
        assert await handle.steer("look at this", command_id="steer-image-1") == "steering queued"
    finally:
        stream.release.set()
        await _stop_quietly(handle)


@pytest.mark.asyncio
async def test_a_rejected_steer_is_not_called_unsaved_while_its_copy_is_spooled(
    tmp_path: Path,
) -> None:
    """The "not saved" notice follows the spool, not the queue.

    A steer rejected at disposal used to be gone: the queue died and the notice
    was true. With the spooled copy still on disk it is held for the next
    runtime, so the same sentence would be false — the one claim this whole
    change exists to stop making. A steer that is in NO spool keeps the notice,
    which is the honest answer for the TUI's own handle.
    """
    _, handle, stream, directory = await _live_turn_with_a_queued_steer(tmp_path, "steer-reject-1")
    session = handle._session
    notices: list[NoticeEvent] = []

    def observe(event: Any) -> None:
        if isinstance(event, NoticeEvent):
            notices.append(event)

    session.subscribe(observe)  # type: ignore[attr-defined]
    try:
        await session._reject_steering(  # type: ignore[attr-defined]
            "steer-reject-1", "session closed before durable admission"
        )
        assert [
            notice.text for notice in notices if "not saved" in notice.text
        ] == [], "a steer held on the spool was reported as not saved"
        assert [row.command_id for row in peek_inbox(directory)] == [
            "steer-reject-1"
        ], "the rejection removed the durable copy"

        await session._reject_steering("never-spooled", "no copy")  # type: ignore[attr-defined]
        assert any(
            "never-spooled" in notice.text and "not saved" in notice.text for notice in notices
        ), "a steer with no durable copy lost its refusal"
    finally:
        stream.release.set()
        await _stop_quietly(handle)
