"""ADR 0009 slice 1b, the session half: origin and ``needs_user`` at run end.

WHY (operator, 2026-11-17, quoted in the mobile repo's ADR 0009): the phone
should hear about work started on it, and the desktop about everything else,
unless the user is away and something needs them. That needs two facts no
record carried before: which SURFACE drove a run (origin), and whether the run
ended on an ask it opened (``needs_user``). Both are stamped on the
``AgentEndEvent`` and published on the ``completions`` row.

THE RIG IS THE REAL ONE: ``ScriptedStream`` + a real ``Session`` + the real
``AttentionStore`` (``test_attention_notify``'s rig). The ask queue is a stub
whose records carry ``created_at`` on the session's own wall clock, which is
the one clock ``AskQueue`` stamps from in production.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AgentEndEvent, CustomMessage
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.session.session import (
    INPUT_SURFACE_KEY,
    _input_surface_of,
    spooled_input_surface,
)
from tests.unit.session.test_attention_notify import _complete_stream, _wait_published
from tests.unit.session.test_session import ScriptedStream, make_session


@pytest.fixture(autouse=True)
def _wakes_open_their_own_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wake right after an operator prompt is PARKED into the next prompt
    (``WAKE_OPERATOR_QUIET_S``); these tests need it to open its own run, the
    run with no human input whose origin must be inherited."""
    from local_operator.session import session as session_module

    monkeypatch.setattr(session_module, "WAKE_OPERATOR_QUIET_S", 0)


class _Queue:
    """``AskQueue.open_records``'s shape: the open records, ``created_at`` in ms."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def open_now(self, ask_id: str) -> None:
        self.records.append(
            {"ask_id": ask_id, "status": "open", "created_at": int(time.time() * 1000)}
        )

    def open_records(self, now_ms: int | None = None) -> list[dict[str, Any]]:
        return [dict(record) for record in self.records]

    def install(self, session: Any) -> None:
        """Answer the session's open-ask read, leaving the real queue alone."""
        session._open_ask_records = self.open_records


def _ends(events: list[Any]) -> list[AgentEndEvent]:
    return [event for event in events if getattr(event, "type", None) == "agent_end"]


def _stream(turns: int) -> ScriptedStream:
    from local_operator.harness.types import StreamEndEvent, StreamTextDelta

    return ScriptedStream(
        [[StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")] for _ in range(turns)]
    )


async def _wake(session: Any, stream: ScriptedStream, wake_id: str) -> None:
    """A real wake delivery: no human input, so the run must inherit."""
    before = len(stream.requests)
    # The previous run's row, so the wait below is for THIS run's publication
    # (``_attention_run_settled`` can still be true from the run before).
    before_token = (getattr(session, "_attention", None) or {}).get("completion_token")
    schedule = WakeSchedule(id=wake_id, message="check", next_due_at=0, created_at=0)
    await session._deliver_wake(
        DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
    )
    from tests.unit.session.test_session import wait_for

    await wait_for(lambda: len(stream.requests) > before)
    await wait_for(
        lambda: session._attention_run_settled
        and not session._is_streaming
        and (session._attention or {}).get("completion_token") not in (None, before_token)
    )


# -- origin -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_direct_prompt_stamps_tui_on_the_row_the_end_and_the_store(
    tmp_path: Path,
) -> None:
    """The in-process owner (m1): ``Session.prompt`` defaults to ``tui``."""
    session = make_session(tmp_path, _complete_stream())
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.prompt("Work")
        state = await session.refresh_attention()
        assert _ends(events)[-1].origin == "tui"
        assert state["origin"] == "tui"
        assert state["needs_user"] is False
        user = [m for m in session._context.messages if getattr(m, "role", None) == "user"][-1]
        assert _input_surface_of(user) == "tui"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_mobile_prompt_stamps_mobile_and_a_later_tui_prompt_flips_it(
    tmp_path: Path,
) -> None:
    """Origin is per run: the phone-driven conversation, typed at the in-process
    owner, flips to ``tui``; it never inherits ``mobile`` (m1)."""
    session = make_session(tmp_path, _stream(2))
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.prompt("from the phone", input_surface="mobile")
        assert (await session.refresh_attention())["origin"] == "mobile"
        await session.prompt("now at the desk")
        assert (await session.refresh_attention())["origin"] == "tui"
        assert [end.origin for end in _ends(events)] == ["mobile", "tui"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_wake_run_inherits_the_last_driver(tmp_path: Path) -> None:
    stream = _stream(2)
    session = make_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.prompt("from the phone", input_surface="mobile")
        await _wake(session, stream, "w1")
        assert [end.origin for end in _ends(events)] == ["mobile", "mobile"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_legacy_conversation_reads_unknown(tmp_path: Path) -> None:
    """No stamped input anywhere (every conversation before ADR 0009): a wake run
    has no driver to inherit, so it is ``unknown`` and routes as non-mobile."""
    stream = _stream(1)
    session = make_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await _wake(session, stream, "w1")
        assert _ends(events)[-1].origin == "unknown"
        assert (await session.refresh_attention())["origin"] == "unknown"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_resumed_conversation_rebuilds_its_driver_from_the_transcript(
    tmp_path: Path,
) -> None:
    stream = _stream(1)
    session = make_session(tmp_path, stream)
    try:
        await session.prompt("from the phone", input_surface="mobile")
    finally:
        await session.dispose()
    resumed_stream = _stream(1)
    resumed = make_session(tmp_path, resumed_stream)
    events: list[Any] = []
    resumed.subscribe(events.append)
    try:
        await _wake(resumed, resumed_stream, "w2")
        assert _ends(events)[-1].origin == "mobile"
    finally:
        await resumed.dispose()


@pytest.mark.asyncio
async def test_a_steer_is_stamped_with_its_own_surface(tmp_path: Path) -> None:
    """A phone steer into a desk-started run flips that run (``_note_run_input``
    reads the stamp off the drained steering row)."""
    session = make_session(tmp_path, _stream(0))
    session.steer("adjust", input_surface="mobile")
    queued = session._steering_queue.get_nowait()
    assert queued.provider_payload[INPUT_SURFACE_KEY] == "mobile"
    session._run_input_surface = "desktop"
    session._note_run_input_surface(queued)
    assert session._run_origin() == "mobile"


@pytest.mark.asyncio
async def test_a_harness_injection_is_not_human_input(tmp_path: Path) -> None:
    session = make_session(tmp_path, _stream(0))
    session.steer("judge says continue", harness_injected=True)
    queued = session._steering_queue.get_nowait()
    assert _input_surface_of(queued) is None


@pytest.mark.asyncio
async def test_a_queued_ask_answer_carries_the_answering_surface(tmp_path: Path) -> None:
    """R10: the control server notes the answer's surface; the ``ask_response``
    row that opens the next run picks it up."""
    session = make_session(tmp_path, _stream(0))
    session.note_answer_surface("ask-1", "mobile")
    row = CustomMessage(
        custom_type="ask_response",
        attribution="user",
        details={"ask_id": "ask-1", "status": "answered"},
    )
    session._note_run_input_surface(row)
    assert session._run_input_surface == "mobile"
    assert "ask-1" not in session._ask_answer_surfaces


def test_a_spooled_line_keeps_its_surface_and_a_legacy_line_reads_unknown() -> None:
    class _Line:
        def __init__(self, sender: dict[str, Any]) -> None:
            self.sender = sender

    assert spooled_input_surface(_Line({"input_surface": "mobile"})) == "mobile"
    assert spooled_input_surface(_Line({"remote_origin": True})) == "unknown"
    assert spooled_input_surface(_Line({"input_surface": "forged"})) == "unknown"


# -- needs_user -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_needs_user_stamps_once_per_ask_and_a_later_new_ask_stamps_again(
    tmp_path: Path,
) -> None:
    """The ADR's R2-1 rows, on real run ends:

    - a wake run that opens an ask ends with ``needs_user`` true (and
      ``notify`` false: it bypasses notify);
    - a SECOND wake run ending with the same ask still open does not stamp it;
    - a later run that opens a NEW ask does.
    """
    stream = _stream(3)
    session = make_session(tmp_path, stream)
    queue = _Queue()
    queue.install(session)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        original = session._run_needs_user

        def open_during_first(now: float | None = None) -> bool:
            if not queue.records:
                queue.open_now("ask-1")
            return original(now)

        session._run_needs_user = open_during_first  # type: ignore[method-assign]
        await _wake(session, stream, "w1")
        session._run_needs_user = original  # type: ignore[method-assign]
        first = _ends(events)[-1]
        assert first.needs_user is True
        assert first.notify is False, "a quiet wake: needs_user is computed apart from notify"
        state = await session.refresh_attention()
        assert state["needs_user"] is True and state["notify"] is False

        await _wake(session, stream, "w2")
        assert _ends(events)[-1].needs_user is False, "same ask still open: not again"
        assert (await session.refresh_attention())["needs_user"] is False

        def open_second(now: float | None = None) -> bool:
            if len(queue.records) == 1:
                queue.open_now("ask-2")
            return original(now)

        session._run_needs_user = open_second  # type: ignore[method-assign]
        await _wake(session, stream, "w3")
        assert _ends(events)[-1].needs_user is True, "a new ask stamps again"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_bridge_ask_raised_while_idle_is_announced_by_the_next_run_once(
    tmp_path: Path,
) -> None:
    """R3-1: a Team Bridge request ask is raised by the harness BETWEEN runs, so
    "opened during this run" would never announce it. The window opens at the
    previous run's end, which covers it exactly once."""
    stream = _stream(3)
    session = make_session(tmp_path, stream)
    queue = _Queue()
    queue.install(session)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.prompt("hello")
        assert _ends(events)[-1].needs_user is False
        time.sleep(0.002)
        queue.open_now("bridge-ask")  # raised while idle, before the next run
        await _wake(session, stream, "w1")
        assert _ends(events)[-1].needs_user is True
        await _wake(session, stream, "w2")
        assert _ends(events)[-1].needs_user is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_ask_restored_from_an_earlier_process_is_not_re_announced(
    tmp_path: Path,
) -> None:
    stream = _stream(1)
    queue = _Queue()
    queue.open_now("old-ask")
    time.sleep(0.002)
    session = make_session(tmp_path, stream)
    queue.install(session)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await _wake(session, stream, "w1")
        assert _ends(events)[-1].needs_user is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_marker_carries_origin_and_needs_user(tmp_path: Path) -> None:
    """Durable for republish and for the resume's driver rebuild."""
    from local_operator.session.attention import ATTENTION_CUSTOM_TYPE

    stream = _complete_stream()
    session = make_session(tmp_path, stream)
    try:
        await session.prompt("from the phone", input_surface="mobile")
        await _wait_published(session, stream)
        marker = session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
        assert marker["origin"] == "mobile"
        assert marker["needs_user"] is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_quiet_bridge_wake_with_a_new_ask_still_publishes_for_the_phone(
    tmp_path: Path,
) -> None:
    """R3-1 end to end: a Team Bridge request ask raised while idle, then the
    card's wake ending in ``no_reply``. A quiet end publishes nothing
    (``test_attention_notify::test_a_bridge_card_wake_ending_in_no_reply_publishes_nothing``
    is the no-ask control), but an ask opened in the run's window is not quiet
    for the phone: the row is published with ``needs_user`` and ``notify`` false,
    so the desktop's completion banner stays silent as the quiet end asked."""
    from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE
    from local_operator.harness.types import StreamEndEvent, StreamToolCallDelta
    from tests.unit.session.test_session import wait_for

    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="q1", name="no_reply", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ]
        ]
    )
    session = make_session(tmp_path, stream)
    queue = _Queue()
    queue.install(session)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        time.sleep(0.002)
        queue.open_now("bridge-request")
        await session.receive_peer_message(
            "Channel message · #agents · from Ada (human, same tenant)\nmention: no\n"
            "expect: weigh relevance; no_reply if nothing is needed",
            mode="mailbox",
            wake=True,
            sender={"account_id": "acc_1", "source": TEAM_BRIDGE_MESSAGE_TYPE},
            custom_type=TEAM_BRIDGE_MESSAGE_TYPE,
        )
        await wait_for(lambda: bool(stream.requests))
        await wait_for(lambda: session._attention_run_settled)
        await wait_for(lambda: not session.is_streaming and not session._turn_lock.locked())
        end = _ends(events)[-1]
        assert end.notify is False and end.needs_user is True
        state = await session.refresh_attention()
        assert state["completion_token"] is not None, "the phone's row exists"
        assert state["needs_user"] is True
        assert state["notify"] is False, "the desktop completion banner stays quiet"
    finally:
        await session.dispose()
