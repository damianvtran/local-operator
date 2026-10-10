"""End-to-end: a steer admitted while no turn is live opens its own turn.

The operator's report (2026-10-09): "I tried to steer this session and it
didn't respond to my messages until a subagent completed, it stayed idle with
running subagents." Probe-measured on the base this file was written against,
two holes produced it:

* ``Session.steer`` / ``steer_message`` only QUEUED. Every other delivery path
  re-checks ``_is_streaming`` and has an idle arm that opens a turn
  (``receive_peer_message``, wakes, monitors); the composer steer path had
  none, so a steer admitted while idle sat on the steering queue until an
  unrelated wake opened a turn whose boundary drain took it (probe result:
  qsize stayed 1 and zero provider calls over 5 s).
* A steer queued while a turn streamed past its LAST drain boundary was
  re-queued by the turn-end flush (``_flush_parked_deliveries`` hands over
  ``CustomMessage``s; a plain steer keeps its place) and nothing re-checked
  the queue afterwards (probe result: after turn end, calls=1, qsize=1, the
  row stranded until the next unrelated turn).

Both are pinned here against ASSEMBLED sessions: the real ``Session``, the
real agent loop, the real ``wait`` tool where one is involved, and only the
provider scripted. The wake's cost contract is asserted explicitly, because
the historical wake shape (an empty-initial run draining only from its second
iteration) spends one paid no-op call before the steer's own -- every cell
that pins a wake asserts the provider call it bought CARRIES the steer.

The busy arms are pinned as unchanged: a steer against a live turn still
folds at the next boundary without opening a rival turn, and the peer/mailbox
delivery arms keep their own (already correct) idle and no-wake shapes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEvent,
    AgentTool,
    MessageStartEvent,
    SteeringDeliveredEvent,
    TextContent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.session.session import Session
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    text_turn,
    tool_call_turn,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

BOUND_S = 90.0

#: How long a cell waits for a wake the runtime OWES. The healthy path is one
#: event-loop iteration of local work (no network, no subprocess), measured in
#: milliseconds in the probe; the bound exists for a loaded CI runner, not
#: because the work is slow.
WAKE_WAIT_S = 10.0


def _session(config_dir: Path, name: str, stream: ScriptedStream, *, tools: Any = ()) -> Session:
    return build_session(config_dir / "sessions" / name, stream, cwd=config_dir, tools=tools)


async def _poll_until(predicate: Any, timeout_s: float = WAKE_WAIT_S) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def _until(predicate: Any, *, what: str, timeout_s: float = WAKE_WAIT_S) -> None:
    if not await _poll_until(predicate, timeout_s):
        raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


def _request_texts(request: Any) -> list[str]:
    texts: list[str] = []
    for message in request.messages:
        for block in message.content or []:
            text = getattr(block, "text", "") or (
                block.get("text") if isinstance(block, dict) else ""
            )
            if text:
                texts.append(text)
    return texts


def _calls_carrying(stream: ScriptedStream, needle: str, *, after: int = 0) -> list[int]:
    """1-based indices of provider calls (later than ``after``) carrying ``needle``."""
    return [
        index
        for index, request in enumerate(stream.requests, start=1)
        if index > after and any(needle in text for text in _request_texts(request))
    ]


def _context_count(session: Session, needle: str) -> int:
    """User-role context messages whose text carries ``needle``."""
    count = 0
    for message in session._context.messages:
        if getattr(message, "role", None) != "user":
            continue
        for block in getattr(message, "content", None) or []:
            text = getattr(block, "text", "") or (
                block.get("text") if isinstance(block, dict) else ""
            )
            if needle in (text or ""):
                count += 1
                break
    return count


def _transcript_count(session: Session, needle: str) -> int:
    """Persisted message rows whose text carries ``needle``."""
    count = 0
    for entry in session._transcript.entries():
        if entry.type != "message" or entry.payload.get("kind") != "message":
            continue
        for block in entry.payload.get("content", []):
            if needle in str(block.get("text") or ""):
                count += 1
                break
    return count


def _steer_events(
    events: list[AgentEvent], needle: str
) -> tuple[list[SteeringDeliveredEvent], list[MessageStartEvent]]:
    delivered = [event for event in events if isinstance(event, SteeringDeliveredEvent)]
    announced = [
        event
        for event in events
        if isinstance(event, MessageStartEvent)
        and needle in str(getattr(event.message, "text", "") or "")
    ]
    return delivered, announced


@pytest.mark.asyncio
async def test_an_idle_steer_opens_its_own_turn_and_delivers_exactly_once(
    headless_tui_env: Path,
) -> None:
    """The headline cell: idle session, one op-steer, one new provider call.

    The call must be the STEER's (it carries the text) rather than the empty
    warm-up the pre-fix wake shape spent first, and the row must land exactly
    once in context and on disk, with the queue empty afterwards.
    """
    stream = ScriptedStream([text_turn("first answer"), text_turn("steered answer")])
    session = _session(headless_tui_env, "idle-steer", stream)
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    try:
        with bounded(BOUND_S, "an idle steer opening its own turn"):
            await session.prompt("say something")
            await _until(lambda: not session._is_streaming, what="the first turn to end")
            assert len(stream.requests) == 1

            started = time.perf_counter()
            session.steer("STEER-ONE")
            await _until(
                lambda: len(stream.requests) == 2 and session._steering_queue.qsize() == 0,
                what="the steer's wake turn to run and drain",
            )
            elapsed = time.perf_counter() - started

            # Promptly, and at the cost of ONE call, which carries the steer:
            # an opening drain that still spent the historical empty warm-up
            # would show two calls with the text only on the second.
            assert elapsed < 2.0, f"the wake took {elapsed:.2f}s"
            assert len(stream.requests) == 2
            assert _calls_carrying(stream, "STEER-ONE", after=1) == [2]
            delivered, announced = _steer_events(events, "STEER-ONE")
            assert len(delivered) == 1 and delivered[0].count == 1
            assert len(announced) == 1
            assert _context_count(session, "STEER-ONE") == 1
            assert _transcript_count(session, "STEER-ONE") == 1
            assert stream.exhausted_at is None
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_running_subagent_does_not_gate_the_idle_wake(headless_tui_env: Path) -> None:
    """The operator's exact scenario: running children, an idle session, a steer.

    The children are real ``task`` rows in the session's job manager (the same
    predicate the TUI's stop ladder counts), and they must not gate the wake --
    the steer is the user's new intent, not something the children own.
    """
    stream = ScriptedStream([text_turn("first answer"), text_turn("steered answer")])
    session = _session(headless_tui_env, "idle-steer-with-child", stream)
    job_id = session.jobs.register("task", "coder", _forever_job)
    try:
        with bounded(BOUND_S, "a steer beside a running subagent"):
            await session.prompt("say something")
            await _until(lambda: not session._is_streaming, what="the first turn to end")
            assert session.running_subagents() == 1

            session.steer("STEER-CHILD")
            await _until(
                lambda: len(stream.requests) == 2 and session._steering_queue.qsize() == 0,
                what="the steer's wake turn to run and drain",
            )
            assert _calls_carrying(stream, "STEER-CHILD", after=1) == [2]
            assert _context_count(session, "STEER-CHILD") == 1
            assert session.running_subagents() == 1, "the wake must not disturb the children"
            assert stream.exhausted_at is None
    finally:
        await session.jobs.cancel(job_id)
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_tail_parked_steer_is_woken_by_the_turn_end_flush(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A steer queued past a turn's last drain gets its own follow-up turn.

    The interleaving is FORCED (a held ``_mark_code_requests_dirty`` in the
    tail -- the thread hop that held the window open on CI), not waited for:
    while the turn is held the steer is queued (busy latch, no wake), the turn
    is released, and the turn-end flush must open the steer's consumer.
    """
    stream = ScriptedStream([text_turn("turn done"), text_turn("follow-up answer")])
    session = _session(headless_tui_env, "tail-parked", stream)
    in_tail = asyncio.Event()
    release = asyncio.Event()
    armed = [True]
    original = Session._mark_code_requests_dirty

    async def held_mark(self: Session) -> None:
        if armed[0]:
            armed[0] = False
            in_tail.set()
            await release.wait()
        return await original(self)

    monkeypatch.setattr(Session, "_mark_code_requests_dirty", held_mark)
    try:
        with bounded(BOUND_S, "a steer parked in a turn's tail"):
            task = asyncio.ensure_future(session.prompt("go"))
            await asyncio.wait_for(in_tail.wait(), BOUND_S)
            assert session._is_streaming, "the turn must still be live in its tail"
            session.steer("STEER-TAIL")
            assert session._steering_queue.qsize() == 1
            assert len(stream.requests) == 1, "a live turn must not open a rival wake"
            release.set()
            await asyncio.wait_for(task, BOUND_S)
            assert not session._is_streaming

            await _until(
                lambda: len(stream.requests) == 2 and session._steering_queue.qsize() == 0,
                what="the follow-up turn to run and drain the park",
            )
            assert _calls_carrying(stream, "STEER-TAIL", after=1) == [2]
            assert _context_count(session, "STEER-TAIL") == 1
            assert _transcript_count(session, "STEER-TAIL") == 1, "one row, not a re-append"
            assert stream.exhausted_at is None
    finally:
        release.set()
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_busy_steer_still_folds_at_the_next_boundary(
    headless_tui_env: Path,
) -> None:
    """The busy arm is unchanged: a live turn takes the steer; no rival turn opens.

    The tool is non-interruptible, so the steer waits for it (the established
    fold) and rides the turn's next request. A wake opened beside a live turn
    would show a THIRD call -- and exhaust the two-turn tape.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any,
        on_update: Any,
        context: Any,
    ) -> ToolResult:
        started.set()
        await release.wait()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
        )

    echo = AgentTool(
        name="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
    )
    stream = ScriptedStream(
        [
            tool_call_turn(text="working", tool_name="echo", tool_call_id="c1", arguments={}),
            text_turn("adjusted"),
        ]
    )
    session = _session(headless_tui_env, "busy-fold", stream, tools=[echo])
    try:
        with bounded(BOUND_S, "a busy steer folding at the boundary"):
            task = asyncio.ensure_future(session.prompt("start"))
            await asyncio.wait_for(started.wait(), BOUND_S)
            session.steer("STEER-BUSY")
            assert len(stream.requests) == 1
            release.set()
            await asyncio.wait_for(task, BOUND_S)

            assert len(stream.requests) == 2
            assert _calls_carrying(stream, "STEER-BUSY", after=1) == [2]
            assert _context_count(session, "STEER-BUSY") == 1
            assert _transcript_count(session, "STEER-BUSY") == 1
            assert session._steering_queue.qsize() == 0
            assert stream.exhausted_at is None
    finally:
        release.set()
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_steer_interrupts_a_parked_wait_and_is_consumed(
    headless_tui_env: Path,
) -> None:
    """A turn parked in ``wait`` on a running job hears the steer within seconds.

    The wait's budget is ten minutes and the job never settles, so nothing but
    the steering interrupt can end the block: the tool must return its
    still-running shape, the job must keep running, and the steer must ride
    the very next provider call.
    """
    from local_operator.tools.builtin import build_wait_tool

    directory = headless_tui_env / "sessions" / "steer-wait"
    # The model's first turn is appended once the job id exists (the wait tool
    # needs a REAL row to block on); the tape is indexed by call, so building
    # the session around a stream whose first turn is filled in below is safe.
    stream = ScriptedStream([text_turn("resumed after the steer")])
    session = build_session(directory, stream, cwd=headless_tui_env)
    # The runtime builds the tool inventory; this cell wires the ONE builtin
    # under test, from the same factory the registry uses.
    wait_tool = build_wait_tool(session._build_tool_context())
    assert wait_tool is not None
    session._tools = [wait_tool]
    job_id = session.jobs.register("task", "long-build", _forever_job)
    stream.turns.insert(
        0,
        tool_call_turn(
            text="waiting on the build",
            tool_name="wait",
            tool_call_id="wait-1",
            arguments={"job_id": job_id, "wait_ms": 600_000},
        ),
    )
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    task: asyncio.Task[Any] | None = None
    try:
        with bounded(BOUND_S, "a steer interrupting a parked wait"):
            task = asyncio.ensure_future(session.prompt("wait for the build"))
            await _until(
                lambda: any(
                    isinstance(event, ToolExecutionStartEvent) and event.tool_name == "wait"
                    for event in events
                ),
                what="the wait tool to start",
            )
            started = time.perf_counter()
            session.steer("STEER-WAIT")
            await asyncio.wait_for(task, BOUND_S)
            elapsed = time.perf_counter() - started

            assert elapsed < 30.0, f"the wait heard the steer only after {elapsed:.1f}s"
            assert len(stream.requests) == 2
            second = _request_texts(stream.requests[1])
            assert any("STEER-WAIT" in text for text in second), "the steer never reached the model"
            assert any(
                "still running" in text for text in second
            ), "the wait's still-running shape never reached the model"
            assert _context_count(session, "STEER-WAIT") == 1
            assert _transcript_count(session, "STEER-WAIT") == 1
            row = session.jobs.get(job_id)
            assert row is not None and row.status == "running"
            assert stream.exhausted_at is None
    finally:
        await session.jobs.cancel(job_id)
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_steer_ends_a_parked_ask_and_is_consumed(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sibling passive wait: a blocking ``ask`` parked on a person.

    The ask tool is ``interruptible`` by design (design §4: "a stop or a
    steering message arriving while the question is up has to be able to end
    the call"), so a steer must end the parked question the way it ends a
    parked ``wait`` -- and be delivered on the next request rather than
    waiting behind the unanswered prompt. The handler never answers, so only
    the steering interrupt can finish the turn.
    """
    from local_operator.asks import policy

    monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="asked",
                tool_name="ask",
                tool_call_id="ask-1",
                arguments={
                    "questions": [
                        {
                            "id": "q1",
                            "question": "Deploy or roll back?",
                            "options": [
                                {"label": "Deploy", "description": "ships the head"},
                                {"label": "Roll back", "description": "returns to the tag"},
                            ],
                        }
                    ]
                },
            ),
            text_turn("resumed after the steer"),
        ]
    )
    session = _session(headless_tui_env, "steer-ask", stream)
    session._has_ui = False
    never = asyncio.Event()

    async def never_answers(questions: Any) -> None:
        await never.wait()

    session.set_ask_handler(never_answers)
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    try:
        with bounded(BOUND_S, "a steer ending a parked ask"):
            task = asyncio.ensure_future(session.prompt("decide the release"))
            await _until(
                lambda: any(
                    isinstance(event, ToolExecutionStartEvent) and event.tool_name == "ask"
                    for event in events
                ),
                what="the ask tool to park",
            )
            started = time.perf_counter()
            session.steer("STEER-ASK")
            await asyncio.wait_for(task, BOUND_S)
            elapsed = time.perf_counter() - started

            assert elapsed < 30.0, f"the parked ask heard the steer only after {elapsed:.1f}s"
            assert len(stream.requests) == 2
            second = _request_texts(stream.requests[1])
            assert any("STEER-ASK" in text for text in second), "the steer never reached the model"
            assert any(
                "interrupted by steering" in text for text in second
            ), "the cancelled ask's result never reached the model"
            assert _context_count(session, "STEER-ASK") == 1
            assert _transcript_count(session, "STEER-ASK") == 1
            assert stream.exhausted_at is None
    finally:
        never.set()
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_the_peer_and_mailbox_arms_keep_their_shapes(headless_tui_env: Path) -> None:
    """The delivery arms beside the steer are unchanged.

    * ``mode="steer"`` while idle opens its own turn (its idle arm predates
      this change and must keep working);
    * ``mode="mailbox", wake=False`` while idle stays record-only: the row is
      persisted and NO turn opens -- the steer wake must not have widened this
      arm into one that spends calls.
    """
    stream = ScriptedStream([text_turn("peer turn")])
    session = _session(headless_tui_env, "peer-arms", stream)
    try:
        with bounded(BOUND_S, "the peer arms beside the steer"):
            detail = await session.receive_peer_message(
                "PEER-STEER", mode="steer", sender={"pid": 41}
            )
            assert detail
            await _until(lambda: len(stream.requests) == 1, what="the peer steer's turn")
            assert any("PEER-STEER" in text for text in _request_texts(stream.requests[0]))
            await _until(lambda: not session._is_streaming, what="the peer turn to end")

            await session.receive_peer_message("MAILBOX-QUIET", mode="mailbox", wake=False)
            # Give any wrongly-spawned turn every chance to show itself.
            await asyncio.sleep(0.3)
            assert len(stream.requests) == 1, "a no-wake mailbox delivery must open no turn"
            assert any(
                "MAILBOX-QUIET" in str(entry.payload)
                for entry in session._transcript.entries()
                if entry.type == "message"
            ), "the quiet row must still be durable"
            assert stream.exhausted_at is None
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_an_idle_steer_recalled_before_its_wake_runs_costs_nothing(
    headless_tui_env: Path,
) -> None:
    """The guard's whole point: a wake whose queue emptied first retires silently.

    Recall is synchronous, so no ordering luck is involved: the steer is in
    the queue when the wake is spawned and out of it before the wake can run.
    The wake must find nothing, open nothing, and spend NO provider call.
    """
    stream = ScriptedStream([text_turn("first answer")])
    session = _session(headless_tui_env, "recalled", stream)
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    try:
        with bounded(BOUND_S, "a recalled steer retiring its wake"):
            await session.prompt("say something")
            await _until(lambda: not session._is_streaming, what="the first turn to end")
            assert len(stream.requests) == 1

            session.steer("STEER-RECALL")
            queued = session.queued_steering()
            assert len(queued) == 1
            assert session.recall_steering(queued[0]) is True

            # Let the spawned wake run and retire.
            await asyncio.sleep(0.3)
            assert session._steering_queue.qsize() == 0
            assert len(stream.requests) == 1, "an empty wake must cost no provider call"
            assert [event for event in events if isinstance(event, SteeringDeliveredEvent)] == []
            assert stream.exhausted_at is None
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_two_wakes_for_one_steer_open_exactly_one_turn(headless_tui_env: Path) -> None:
    """Arrivals race for the lock; the queue still gets ONE consumer.

    ``steer`` spawns its own wake, and a second wake for the same row is
    requested here by hand. Whichever acquires the lock first drains the steer;
    the other's under-lock guard must find the queue empty and retire -- one
    provider call, one row, no double injection.
    """
    stream = ScriptedStream([text_turn("first answer"), text_turn("steered answer")])
    session = _session(headless_tui_env, "double-wake", stream)
    try:
        with bounded(BOUND_S, "two wakes racing for one steer"):
            await session.prompt("say something")
            await _until(lambda: not session._is_streaming, what="the first turn to end")

            session.steer("STEER-RACE")
            session._ensure_steering_wake()
            await _until(
                lambda: len(stream.requests) == 2 and session._steering_queue.qsize() == 0,
                what="exactly one wake to drain the steer",
            )
            # One more beat for a second wake to reveal itself if the guard failed.
            await asyncio.sleep(0.3)
            assert len(stream.requests) == 2, "a second wake ran against an empty queue"
            assert _calls_carrying(stream, "STEER-RACE", after=1) == [2]
            assert _context_count(session, "STEER-RACE") == 1
            assert _transcript_count(session, "STEER-RACE") == 1
            assert stream.exhausted_at is None
    finally:
        await dispose_quietly(session)


async def _forever_job(job_id: str, signal: Any, report_progress: Any) -> str:
    """A job that never settles: the state every parked wait sits in."""
    await asyncio.sleep(3600)
    return "never"
