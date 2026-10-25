"""A child's unprompted hub note and its own settle wake the parent ONCE.

Measured motivation (see ``harness.comms.HUB_NOTE_WAKE_GRACE_S``): 60%
of child->parent notes were followed by the same job's settle within 120 s,
and each note cut the parent's parked ``wait`` short — one full-context model
call to read the note and re-issue ``wait``, then a second wake for the result.

Every test here drives a REAL ``Session`` and its real ``SubagentComms``, with
scripted job runners standing in for children (no provider), so the wake path
is the one production runs: ``reply_to_parent`` -> ``queue_aside_coalesced`` ->
``_PeerArrival.mark`` -> the ``wait`` tool's park loop.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from local_operator.harness import comms as comms_module
from local_operator.harness.message_types import HUB_MESSAGE_TYPE
from local_operator.harness.types import (
    CustomMessage,
    Message,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
)
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for


def _grace(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    # Each test states the window it is about rather than inheriting the default.
    monkeypatch.setattr(comms_module, "HUB_NOTE_WAKE_GRACE_S", seconds)


def _wait_call(job_id: str, call_id: str = "w1", wait_ms: int = 600_000) -> list[StreamEvent]:
    return [
        StreamToolCallDelta(
            index=0,
            id=call_id,
            name="wait",
            argument_delta=json.dumps({"job_id": job_id, "wait_ms": wait_ms}),
        ),
        StreamEndEvent(stop_reason="toolUse"),
    ]


def _tool_texts(messages: list[Any]) -> list[str]:
    return [m.text for m in messages if isinstance(m, Message) and m.role == "tool"]


def _wire(messages: list[Any]) -> str:
    return "\n".join(getattr(m, "text", "") or "" for m in messages)


async def _no_leaks(session: Any) -> None:
    await session.dispose()
    assert not session._background_tasks, "dispose must reap every deferred-mark timer"


@pytest.mark.asyncio
async def test_note_then_settle_wakes_the_parked_wait_once_with_the_result(tmp_path, monkeypatch):
    """(1) note, settle 2 s later: ONE wait return carrying the result; the
    note rides the very next request."""
    _grace(monkeypatch, 20.0)
    release = asyncio.Event()
    turns: list[list[StreamEvent]] = []
    stream = ScriptedStream(turns)
    session = make_session(tmp_path, stream)
    comms = session.subagent_comms

    async def child(job_id, signal, report_progress):
        await release.wait()
        comms.reply_to_parent(job_id, "report: all three fixtures pass")
        await asyncio.sleep(2.0)
        return "final: fixtures pass, PR ready"

    job_id = session.jobs.register("task", "reviewer", child)
    turns.append(_wait_call(job_id))
    turns.append([StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")])

    prompt = asyncio.ensure_future(session.prompt("await the reviewer"))
    await wait_for(lambda: len(stream.requests) == 1)
    await asyncio.sleep(0.05)
    started = time.perf_counter()
    release.set()
    await asyncio.wait_for(prompt, timeout=10)
    elapsed = time.perf_counter() - started

    assert len(stream.requests) == 2, "the note must not buy its own model call"
    assert 1.8 < elapsed < 5.0, elapsed
    follow_up = stream.requests[1].messages
    [wait_text] = _tool_texts(follow_up)
    assert "still running" not in wait_text
    assert "final: fixtures pass, PR ready" in wait_text
    assert "report: all three fixtures pass" in _wire(follow_up), "the note must not be lost"
    assert session._peer_arrival.arrivals() == {HUB_MESSAGE_TYPE: 1}, "marked exactly once"
    await _no_leaks(session)


@pytest.mark.asyncio
async def test_note_without_settle_interrupts_the_wait_after_the_grace(tmp_path, monkeypatch):
    """(2) a genuine mid-job note still arrives — at most ``grace`` late."""
    _grace(monkeypatch, 0.5)
    gate = asyncio.Event()
    turns: list[list[StreamEvent]] = []
    stream = ScriptedStream(turns)
    session = make_session(tmp_path, stream)
    comms = session.subagent_comms
    noted = asyncio.Event()

    async def child(job_id, signal, report_progress):
        await noted.wait()
        comms.reply_to_parent(job_id, "I am blocked: no staging credentials")
        await gate.wait()
        return "late"

    job_id = session.jobs.register("task", "deployer", child)
    turns.append(_wait_call(job_id))
    turns.append([StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")])

    prompt = asyncio.ensure_future(session.prompt("await the deploy"))
    await wait_for(lambda: len(stream.requests) == 1)
    await asyncio.sleep(0.05)
    started = time.perf_counter()
    noted.set()
    await asyncio.wait_for(prompt, timeout=5)
    elapsed = time.perf_counter() - started

    assert 0.4 < elapsed < 2.0, f"the note must wait out the grace, no longer: {elapsed:.2f}s"
    follow_up = stream.requests[1].messages
    [wait_text] = _tool_texts(follow_up)
    assert "still running" in wait_text and "hub message" in wait_text
    assert "I am blocked: no staging credentials" in _wire(follow_up)
    row = session.jobs.get(job_id)
    assert row is not None and row.status == "running"
    gate.set()
    await _no_leaks(session)


@pytest.mark.asyncio
async def test_an_answer_to_an_armed_question_is_never_delayed(tmp_path, monkeypatch):
    """(3) the armed path resolves the asker's future synchronously and does
    not go through the deferred mark at all."""
    _grace(monkeypatch, 30.0)
    gate = asyncio.Event()
    session = make_session(tmp_path, ScriptedStream([]))
    comms = session.subagent_comms

    async def child(job_id, signal, report_progress):
        await gate.wait()
        return "done"

    job_id = session.jobs.register("task", "parser", child)
    comms.record_launch(job_id, "parser")

    class _Child:
        def __init__(self) -> None:
            self.asides: list[Any] = []

        def queue_aside(self, thunk):
            self.asides.append(thunk)

        def steer_message(self, message):  # pragma: no cover - unused here
            pass

        def subscribe(self, handler):
            return lambda: None

    fake = _Child()
    comms.attach(job_id, fake, tmp_path)
    session._is_streaming = True  # a busy parent: the case that WOULD defer a note
    try:
        question = asyncio.ensure_future(comms.ask(job_id, "are you stuck?", 5_000))
        await wait_for(lambda: bool(fake.asides))
        [thunk] = fake.asides
        thunk()  # the child's boundary materializes the question: now armed
        started = time.perf_counter()
        assert comms.reply_to_parent(job_id, "not stuck") == "answered the parent's question"
        reply = await asyncio.wait_for(question, timeout=1)
        assert time.perf_counter() - started < 0.5
        assert reply.text == "not stuck"
        assert not session._background_tasks, "an answer must not arm a deferred mark"
    finally:
        session._is_streaming = False
        gate.set()
    await _no_leaks(session)


@pytest.mark.asyncio
async def test_dispose_inside_the_window_leaks_nothing_and_loses_nothing(tmp_path, monkeypatch):
    """(4) dispose cancels the timer, still counts the arrival, and the note
    stays queued for the boundary."""
    _grace(monkeypatch, 30.0)
    gate = asyncio.Event()
    session = make_session(tmp_path, ScriptedStream([]))
    comms = session.subagent_comms

    async def child(job_id, signal, report_progress):
        await gate.wait()
        return "done"

    job_id = session.jobs.register("task", "worker", child)
    session._is_streaming = True
    comms.reply_to_parent(job_id, "halfway there")
    session._is_streaming = False
    assert session._peer_arrival.count() == 0, "a busy parent's wake is deferred"
    assert len(session._background_tasks) == 1
    assert len(session._aside_thunks) == 1, "the note itself is queued immediately"

    await session.dispose()
    assert not session._background_tasks
    assert session._peer_arrival.arrivals() == {HUB_MESSAGE_TYPE: 1}, "cancel still marks"
    [thunk] = session._aside_thunks
    assert callable(thunk)
    note = thunk()
    assert isinstance(note, CustomMessage)
    assert "halfway there" in note.details["body"]
    gate.set()


@pytest.mark.asyncio
async def test_a_zero_window_is_todays_behaviour_exactly(tmp_path, monkeypatch):
    """(5) knob = 0: immediate mark, plain aside, no timer — mid-turn too."""
    _grace(monkeypatch, 0.0)
    gate = asyncio.Event()
    turns: list[list[StreamEvent]] = []
    stream = ScriptedStream(turns)
    session = make_session(tmp_path, stream)
    comms = session.subagent_comms
    noted = asyncio.Event()

    async def child(job_id, signal, report_progress):
        await noted.wait()
        comms.reply_to_parent(job_id, "report")
        await asyncio.sleep(1.0)
        return "final"

    job_id = session.jobs.register("task", "reviewer", child)
    turns.append(_wait_call(job_id, "w1"))
    turns.append(_wait_call(job_id, "w2"))
    turns.append([StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")])

    prompt = asyncio.ensure_future(session.prompt("await"))
    await wait_for(lambda: len(stream.requests) == 1)
    await asyncio.sleep(0.05)
    started = time.perf_counter()
    noted.set()
    await wait_for(lambda: len(stream.requests) == 2, timeout=2)
    assert time.perf_counter() - started < 0.5, "the note interrupted the wait at once"
    [first] = _tool_texts(stream.requests[1].messages)
    assert "still running" in first
    await asyncio.wait_for(prompt, timeout=5)
    assert len(stream.requests) == 3, "today: note and settle are two wakes, two calls"
    gate.set()
    await _no_leaks(session)


@pytest.mark.asyncio
async def test_an_idle_parent_is_marked_at_once(tmp_path, monkeypatch):
    """An aside does not open a turn on an idle parent today; nothing is parked
    to cut short, so deferral would only add latency. Unchanged."""
    _grace(monkeypatch, 30.0)
    gate = asyncio.Event()
    session = make_session(tmp_path, ScriptedStream([]))

    async def child(job_id, signal, report_progress):
        await gate.wait()
        return "done"

    job_id = session.jobs.register("task", "worker", child)
    session.subagent_comms.reply_to_parent(job_id, "fyi")
    assert session._peer_arrival.count() == 1
    assert not session._background_tasks
    gate.set()
    await _no_leaks(session)


@pytest.mark.asyncio
async def test_three_child_fan_out_costs_fewer_parent_calls(tmp_path, monkeypatch):
    """Acceptance: the parent's model calls for 3 children that each note and
    then settle within 5 s, counted from the ACTUAL run, before vs after."""

    async def run(grace: float) -> int:
        _grace(monkeypatch, grace)
        turns: list[list[StreamEvent]] = []

        class Stream(ScriptedStream):
            # The parent re-issues `wait` on whatever is still running, which is
            # the behaviour the transcripts show; a script cannot know in
            # advance how many wakes the run will take, so it answers live.
            def __call__(self, request, signal):
                self.requests.append(request)
                rows = {j: session.jobs.get(j) for j in ids}
                running = [j for j, row in rows.items() if row and row.status == "running"]
                undelivered = [j for j, row in rows.items() if row and not row.consumed]
                events: list[StreamEvent]
                if running or undelivered:
                    target = running or undelivered
                    events = _wait_call(target[0], f"w{len(self.requests)}")
                    events[0] = StreamToolCallDelta(
                        index=0,
                        id=f"w{len(self.requests)}",
                        name="wait",
                        argument_delta=json.dumps({"job_id": target, "wait_ms": 600_000}),
                    )
                else:
                    events = [StreamTextDelta(delta="all done"), StreamEndEvent(stop_reason="stop")]

                async def gen():
                    for event in events:
                        yield event

                return gen()

        stream = Stream(turns)
        session = make_session(tmp_path / f"g{grace}", stream)
        comms = session.subagent_comms
        start = asyncio.Event()

        def child_for(delay_note: float, delay_settle: float):
            async def child(job_id, signal, report_progress):
                await start.wait()
                await asyncio.sleep(delay_note)
                comms.reply_to_parent(job_id, f"report from {job_id}")
                await asyncio.sleep(delay_settle)
                return f"final from {job_id}"

            return child

        ids = [
            session.jobs.register("task", f"c{i}", child_for(0.3 + i * 1.5, 1.0 + i))
            for i in range(3)
        ]
        prompt = asyncio.ensure_future(session.prompt("fan out"))
        await wait_for(lambda: len(stream.requests) == 1)
        start.set()
        await asyncio.wait_for(prompt, timeout=30)
        await wait_for(lambda: not session._background_tasks, timeout=5)
        calls = len(stream.requests)
        await _no_leaks(session)
        return calls

    before = await run(0.0)
    after = await run(20.0)
    print(f"\nfan-out parent model calls: before={before} after={after}")
    assert after < before, (before, after)
    assert after == 4, "one initial call + one per settle; no call is spent on a note"
    assert before == 7, "one initial call + a note wake and a settle wake per child"
