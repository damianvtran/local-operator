"""The session-side composition of patience: cancel-on-reply, hidden delivery.

These build a REAL ``Session`` over a scratch root (the same construction the
resume path performs) and drive the seams production drives:

- a human/peer inbound cancels the pending wait (and a WAKE DELIVERY does not —
  wake deliveries arrive under ``attribution="user"``, the trap §8.2.3 names);
- a fire reaching the session with a reply in between retires silently;
- a fresh fire delivers HIDDEN — the turn runs, the text reaches the model, and
  NO ``wake_delivered`` receipt is emitted;
- the class is read at the moment of delivery, so a switch to reactive stops a
  fire without a restart;
- the turn-end flush stamps ``armed_after`` on rows armed this turn;
- the switch's immediate cleanup cancels this session's pending rows.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from local_operator.harness.types import ModelSpec
from local_operator.harness.wake_types import DueWake, WakeSchedule
from local_operator.resume import write_session_attachment
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, text_turn

MODEL = ModelSpec(provider="test", model_id="patience-model", context_window=100_000)
SESSION_ID = "pat000000001"


def make_session(
    root: Path,
    session_id: str = SESSION_ID,
    *,
    agent: str = "aida",
    agent_registry=None,
) -> Session:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    write_session_attachment(directory, team="", agent=agent, goal="")
    return Session(
        model=MODEL,
        model_source="config",
        stream_fn=ScriptedStream([text_turn("ok") for _ in range(6)]),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        session_id=session_id,
        agent_registry=agent_registry,
        cwd=str(directory.parent),
    )


def pat_row(**overrides: Any) -> WakeSchedule:
    now = int(time.time() * 1000)
    base: dict[str, Any] = dict(
        id="patience-1",
        message="",
        next_due_at=now + 300_000,
        created_at=now,
        kind="patience",
        hidden=True,
        episode_id="patience-1",
        attempt=1,
        armed_at=now,
        armed_after="",
        note="ask the operator about the build",
    )
    base.update(overrides)
    return WakeSchedule(**base)


@pytest.mark.asyncio
async def test_a_user_prompt_cancels_pending_patience(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    try:
        await session._wake.update([pat_row()])
        assert [r.id for r in session._wake.schedules] == ["patience-1"]

        await session.prompt("hello there")

        assert [r.id for r in session._wake.schedules] == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_peer_message_cancels_pending_patience(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    try:
        await session._wake.update([pat_row()])
        await session.receive_peer_message("status?", mode="mailbox", wake=False, sender=None)
        assert [r.id for r in session._wake.schedules] == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_fire_delivers_hidden_with_no_receipt(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    events: list[object] = []
    session.subscribe(events.append)
    try:
        row = pat_row(next_due_at=int(time.time() * 1000) - 1)
        await session._deliver_wake(DueWake(schedule=row, occurrence=1))
        # Drain the spawned turn so the transcript write lands.
        for _ in range(200):
            if any(
                getattr(getattr(m, "payload", None), "get", lambda *_: None)("custom_type", "")
                == "wake_prompt"
                for m in session._transcript.entries()
                if isinstance(getattr(m, "payload", None), dict)
            ):
                break
            await asyncio.sleep(0.01)

        receipts = [e for e in events if getattr(e, "type", None) == "wake_delivered"]
        assert receipts == []
        customs = [
            m.payload
            for m in session._transcript.entries()
            if isinstance(getattr(m, "payload", None), dict)
            and m.payload.get("custom_type") == "wake_prompt"
        ]
        assert len(customs) == 1
        details = customs[0]["details"]
        assert details["hidden"] is True and details["kind"] == "patience"
        assert details["attempt"] == 1
        assert "internal timer note" in details["text"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_reply_in_between_retires_the_fire_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watermark, at the last possible moment.

    The reply landed (cross-runtime, or a cancel that raced the persist), so
    the fire must NOT run a turn even though the scheduler handed it over. The
    row is armed in the FUTURE so the session's own pump cannot race the
    manual delivery under test — with a row already due, the pump would fire
    it in the background and the assertions below would be about a different
    call path; removal from the list is the pump's advance, pinned in
    ``tests/unit/wakes``.
    """
    from local_operator.harness.types import Message

    session = make_session(tmp_path)
    spawned = AsyncMock()
    monkeypatch.setattr(session, "_prompt_messages", spawned)
    try:
        row = pat_row(armed_at=int(time.time() * 1000) - 1000)
        await session._wake.update([row])
        await session._transcript.append_messages([Message.user("the reply", id="m-reply")])

        await session._deliver_wake(DueWake(schedule=row, occurrence=1))
        await asyncio.sleep(0.01)

        assert not spawned.called
        assert not any(
            isinstance(getattr(m, "payload", None), dict)
            and m.payload.get("custom_type") == "wake_prompt"
            for m in session._transcript.entries()
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_wake_delivery_does_not_count_as_a_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attribution trap (§8.2.3): wake deliveries ride ``attribution="user"``.

    A visible wake fired and was delivered to the model as a user turn; that is
    NOT a reply, so a pending patience wait must still deliver afterwards.
    """
    from local_operator.harness.types import CustomMessage
    from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE

    session = make_session(tmp_path)
    spawned = AsyncMock()
    monkeypatch.setattr(session, "_prompt_messages", spawned)
    try:
        now = int(time.time() * 1000)
        row = pat_row(armed_at=now - 2000, next_due_at=1)
        await session._wake.update([row])
        await session._transcript.append_messages(
            [
                CustomMessage(
                    custom_type=WAKE_PROMPT_MESSAGE_TYPE,
                    attribution="user",
                    details={"text": "scheduled wake u1", "wake_catchup": False},
                )
            ]
        )

        await session._deliver_wake(DueWake(schedule=row, occurrence=1))

        assert spawned.called
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_reactive_class_stops_the_fire_at_delivery_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.action_class import set_registered_action_class
    from local_operator.agents import AgentRegistry

    registry = AgentRegistry(tmp_path)
    session = make_session(tmp_path, agent_registry=registry)
    spawned = AsyncMock()
    monkeypatch.setattr(session, "_prompt_messages", spawned)
    try:
        set_registered_action_class(registry, "aida", "reactive")  # the switch
        row = pat_row()  # armed in the future: see the stale-fire test's note
        await session._wake.update([row])

        await session._deliver_wake(DueWake(schedule=row, occurrence=1))
        await asyncio.sleep(0.01)

        # The switch landed without a restart: no turn, nothing appended.
        assert not spawned.called
        assert not any(
            isinstance(getattr(m, "payload", None), dict)
            and m.payload.get("custom_type") == "wake_prompt"
            for m in session._transcript.entries()
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_armed_after_flushes_at_turn_end(tmp_path: Path) -> None:
    """The default target: this turn's own output, stamped when the turn ends.

    Driven through ``_prompt_messages`` (the wake path) rather than
    ``prompt()`` on purpose: a HUMAN prompt first cancels pending waits — which
    is the other half of the contract, pinned above — so the canonical arm
    case is an agent turn that arms mid-turn and flushes at its end.
    """
    from local_operator.harness.types import Message

    session = make_session(tmp_path)
    try:
        await session._wake.update([pat_row(armed_after="")])
        session._note_patience_armed("patience-1")

        await session._prompt_messages([Message.user("internal drive", id="m-drive")])

        row = next(r for r in session._wake.schedules if r.id == "patience-1")
        assert row.armed_after.startswith("message:")
        # And the pointer names a real message in the journal.
        ids = [m.id for m in session._transcript.entries()]
        assert row.armed_after.split(":", 1)[1] in ids
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_cleanup_after_class_switch_cancels_this_sessions_rows(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    try:
        await session._wake.update([pat_row()])
        outcome = await session.cleanup_after_class_switch("aida")
        assert outcome["patience_cancelled"] == ["patience-1"]
        assert [r.id for r in session._wake.schedules] == []

        # A switch of a DIFFERENT profile leaves this session alone.
        await session._wake.update([pat_row()])
        outcome = await session.cleanup_after_class_switch("somebody-else")
        assert outcome["patience_cancelled"] == []
        assert [r.id for r in session._wake.schedules] == ["patience-1"]
    finally:
        await session.dispose()
