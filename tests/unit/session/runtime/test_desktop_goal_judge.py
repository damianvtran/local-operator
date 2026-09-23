"""The runtime's judge wiring: the trigger, the queue, the stamp, the guards.

What this file is for, and why it uses a REAL ``Session`` for the interesting
half: the edge-triggered judge is only correct if it is wired to the RIGHT event
(the runtime's own session subscription, not the prompt drain, which documents
its own gap) and if its continuation is admitted through the ORDINARY queue
rather than a second concurrency path. Both of those are properties of the
composition, so the two tests that matter drive a real session through a real
handle and read the durable rows and the provider requests back.

The guards — ownership, a running loop, the boot re-arm — are driven through the
handle with the runtime suite's own ``FakeSession`` double, because what they
assert is WHICH of them fires and not what a turn does.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    ChatRequest,
    Message,
    MessageStartEvent,
    ModelSpec,
    NoticeEvent,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.goal import GoalState
from local_operator.session.goal_judge import (
    GOAL_JUDGE_FAILURES,
    MAX_GOAL_CONTINUATIONS,
    STALLED_BREAKER_NOTICE,
    STALLED_CAP_NOTICE,
    goal_continuation_prompt,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.unit.session.runtime.test_serving import FakeSession

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

JUDGE_CONTINUE = "VERDICT: CONTINUE\nThere is more to do"
JUDGE_ACHIEVED = "VERDICT: ACHIEVED\nAll the work is done"
#: An answer with no readable verdict: the strike the breaker counts.
JUDGE_UNREADABLE = "I think it is probably done, maybe?"

#: How the judge's own request is recognised in the stream. Matched on the
#: JUDGE PROMPT's opening words, never on a request count: a count makes every
#: assertion below sensitive to how many turns the fixture happens to script.
JUDGE_MARKER = "judging whether a standing goal"


async def wait_for(predicate, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


class JudgeStream:
    """A scripted provider stream that knows which call is the judge's.

    Judge answers come from ``verdicts``, one per judge call, and every other
    request is an ordinary turn that ends. That split is what lets a test say
    "two verdicts, three turns" without counting array indices.
    """

    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = list(verdicts)
        self.requests: list[ChatRequest] = []
        self.judge_calls: list[str] = []
        self.turn_calls = 0
        #: When set, the judge's call parks on it — the deterministic way to hold
        #: a verdict open while another message arrives.
        self.judge_gate: asyncio.Event | None = None
        self.judge_started = asyncio.Event()

    def __call__(self, request: ChatRequest, signal: Any = None):
        self.requests.append(request)
        texts = [(message.text or "") for message in request.messages]
        is_judge = any(JUDGE_MARKER in text for text in texts)
        if is_judge:
            self.judge_calls.append("")
        else:
            self.turn_calls += 1
        answer = self.verdicts.pop(0) if (is_judge and self.verdicts) else ""

        async def gen():
            if is_judge:
                self.judge_started.set()
                if self.judge_gate is not None:
                    await self.judge_gate.wait()
                if answer:
                    yield StreamTextDelta(delta=answer)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_owner_session(directory, stream) -> Session:
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    # Naming is a SECOND provider call this file is not about: an unnamed
    # conversation would spend a request on a title (and a scripted verdict on
    # nothing), so the fixture arrives named.
    session.set_conversation_name("judge fixture", user_set=True)
    return session


def make_handle(session: Session) -> ServingSessionHandle:
    """A handle with the runtime's own boot registration installed.

    ``subscribe`` is what installs the turn-end trigger — the runtime does it in
    its prologue with no client attached. Skipping it is exactly how a test would
    certify a judge that never fires.
    """
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    handle.subscribe(lambda: None)
    return handle


def _user_messages(session: Session) -> list[Message]:
    """The user rows, TYPED as ``Message``.

    ``session.history()`` is the ``AgentMessage`` union, so reading ``.text`` or
    ``.provider_payload`` off a row is a type error until the row is narrowed —
    the same narrowing the e2e cells make inline, kept here because four cells
    need it.
    """
    return [
        row
        for row in session.history()
        if isinstance(row, Message) and getattr(row, "role", "") == "user"
    ]


def _user_rows(session: Session) -> list[str]:
    """Every user-role row the session's own context carries, in order."""
    return [
        getattr(row, "text", "") for row in session.history() if getattr(row, "role", "") == "user"
    ]


@pytest.mark.asyncio
async def test_an_agent_end_admits_a_continuation_through_the_prompt_queue(tmp_path):
    """The whole chain: judge → CONTINUE → one queued turn → judge → ACHIEVED.

    The continuation is asserted through the durable rows and the provider
    requests, because "it was admitted as a task through the FIFO" is only
    provable by the turn having actually run.
    """
    stream = JudgeStream([JUDGE_CONTINUE, JUDGE_ACHIEVED])
    session = make_owner_session(tmp_path / "sess", stream)
    assert session.arm_goal("Ship it") == "Ship it"
    handle = make_handle(session)
    try:
        await handle.prompt("Ship it", command_id="c1", wait_complete=True)
        await wait_for(lambda: session.goal_status == "done")
        # TWO judge calls — one per verdict — which is the assertion that the
        # admitted continuation's own turn end did NOT start a second judge: the
        # chain judges what it admitted, in its own loop.
        assert len(stream.judge_calls) == 2
        # ...and exactly two ordinary turns: the user's own, and the one
        # continuation the judge admitted through the queue.
        assert stream.turn_calls == 2
        assert _user_rows(session) == ["Ship it", goal_continuation_prompt("Ship it")]
        # ACHIEVED settled the goal through the SAME call `/goal --done` makes.
        assert session.goal_status == "done"
        assert session.goal == "Ship it"
        history = session.history_view()
        assert [entry["status"] for entry in history] == ["done"]
        assert history[0]["reason"] == "All the work is done"
        assert session.goal_judge is not None
        assert session.goal_judge["state"] == "done"
        assert session.goal_judge["verdict"] == "achieved"
        # The continuation row is durable and carries the STRUCTURAL stamp, so a
        # viewer that never heard of the recogniser still knows not to paint it.
        continuation = [
            row
            for row in _user_messages(session)
            if row.text == goal_continuation_prompt("Ship it")
        ]
        assert len(continuation) == 1
        assert continuation[0].provider_payload == {"harness_injected": True}
    finally:
        await handle.dispose()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_continuation_runs_after_a_message_admitted_during_the_judge(tmp_path):
    """Steering races the judge by construction, and the FIFO decides the order.

    A message arriving while the verdict is being computed is appended to the
    same queue, so it is drained BEFORE the continuation the judge is about to
    admit (arrival order); the verdict for the context it read is then dropped by
    the staleness guard, and the re-judge settles the goal against the context
    that now exists.
    """
    # THREE verdicts: the first is dropped by the staleness guard (a turn this
    # judge did not admit ended while it was in flight), the re-judge continues,
    # and the third call — on the continuation — is the one that settles.
    stream = JudgeStream([JUDGE_CONTINUE, JUDGE_CONTINUE, JUDGE_ACHIEVED])
    session = make_owner_session(tmp_path / "sess", stream)
    session.arm_goal("Ship it")
    handle = make_handle(session)
    stream.judge_gate = asyncio.Event()
    try:
        await handle.prompt("Ship it", command_id="c1", wait_complete=True)
        await stream.judge_started.wait()
        # The user speaks while the judge is mid-call.
        await handle.prompt("actually, do X first", command_id="c2", wait_complete=True)
        # ...and only now does the judge answer.
        stream.judge_gate.set()
        await wait_for(lambda: session.goal_status == "done")
        rows = _user_rows(session)
        assert rows.index("actually, do X first") < rows.index(goal_continuation_prompt("Ship it"))
    finally:
        await handle.dispose()
        await session.dispose()


@pytest.mark.asyncio
async def test_an_error_turn_sets_waiting_and_admits_nothing(tmp_path):
    stream = JudgeStream([JUDGE_ACHIEVED])
    session = make_owner_session(tmp_path / "sess", stream)
    session.arm_goal("Ship it")
    handle = make_handle(session)
    try:
        session._emit_nowait(AgentEndEvent(messages=[], error="provider exploded", generation=1))
        await asyncio.sleep(0.05)
        assert session.goal_judge is not None
        assert session.goal_judge["state"] == "waiting"
        assert stream.judge_calls == []
        assert stream.requests == []
        # The goal stays ACTIVE: an error is not a release, and the next turn end
        # re-arms the judge.
        assert session.goal_status == "active"
    finally:
        await handle.dispose()
        await session.dispose()


@pytest.mark.asyncio
async def test_an_injected_continuation_row_is_stamped_and_never_announced(tmp_path):
    """The stamp and the announce decision, exercised on the runtime's own path."""
    from local_operator.harness.rows import is_harness_chrome

    session = make_owner_session(tmp_path / "sess", JudgeStream([JUDGE_ACHIEVED]))
    seen: list[str] = []

    def watch(event: Any) -> None:
        if isinstance(event, MessageStartEvent) and getattr(event.message, "role", "") == "user":
            seen.append(getattr(event.message, "text", ""))

    session.subscribe(watch)
    try:
        await session.prompt(
            goal_continuation_prompt("Ship it"), harness_injected=True, message_id="m1"
        )
        rows = _user_messages(session)
        assert len(rows) == 1
        assert rows[0].provider_payload == {"harness_injected": True}
        assert is_harness_chrome(rows[0].text)
        # The row is durable and NOT announced: a front end that cannot recognise
        # chrome is never asked to paint it either.
        assert seen == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_ordinary_prompt_carries_no_stamp_and_is_announced(tmp_path):
    """The other half: the default must not quietly mark every user row."""
    session = make_owner_session(tmp_path / "sess", JudgeStream([JUDGE_ACHIEVED]))
    seen: list[str] = []

    def watch(event: Any) -> None:
        if isinstance(event, MessageStartEvent) and getattr(event.message, "role", "") == "user":
            seen.append(getattr(event.message, "text", ""))

    session.subscribe(watch)
    try:
        await session.prompt("just talk to me", message_id="m2")
        rows = _user_messages(session)
        assert len(rows) == 1
        assert not rows[0].provider_payload
        assert seen == ["just talk to me"]
    finally:
        await session.dispose()


class GoalDouble(FakeSession):
    """The runtime suite's session double, plus the judged-goal record.

    Delegating to a REAL ``GoalState`` — the same call ``test_goal_submission``
    makes — so the guards below are asserted against state that actually moves
    rather than against a stub that agrees with whatever the code asks.
    """

    def __init__(self, *, locality: str = "this-process") -> None:
        super().__init__()
        self.goal_state = GoalState()
        self.complete_aside_calls: list[str] = []
        #: The judge's answer for the next aside. One field rather than a scripted
        #: queue: the two arms below stall on their FIRST judge call, so a queue
        #: would only be carrying a fiction.
        self.verdict = JUDGE_ACHIEVED
        self.runtime_locality = locality  # type: ignore[misc]

    @property
    def goal(self) -> str:
        return self.goal_state.text

    @property
    def goal_status(self) -> str:
        return self.goal_state.status

    @property
    def goal_token(self) -> str:
        return self.goal_state.token

    @property
    def goal_turn_serial(self) -> int:
        return 1

    @property
    def goal_judge_state(self) -> Any:
        return self.goal_state.judge

    def note_goal_judge(self, **changes: Any) -> None:
        for name, value in changes.items():
            setattr(self.goal_state.judge, name, value)

    def mark_goal_done(self, reason: str = "") -> Any:
        return self.goal_state.mark_done(reason)

    async def complete_aside(self, turns: list[Any], **kwargs: Any) -> str:
        self.complete_aside_calls.append(turns[0].text)
        return self.verdict


def _double_handle(session: GoalDouble) -> ServingSessionHandle:
    return ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")


@pytest.mark.asyncio
async def test_a_follower_never_judges():
    """§3.2: a follower TUI, the phone and a cold viewer must never spend."""
    follower = GoalDouble(locality="this-machine")
    follower.goal_state.arm("Ship it")
    handle = _double_handle(follower)
    try:
        handle._maybe_judge_goal(AgentEndEvent(messages=[], generation=1))
        await asyncio.sleep(0.05)
        assert follower.complete_aside_calls == []
        assert follower.prompt_calls == []
        # The ownership check declines BEFORE the driver is even constructed,
        # which is the strongest form of "a follower never judges".
        assert handle._goal_judge is None
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_running_loop_owns_the_verdict():
    session = GoalDouble()
    session.goal_state.arm("Ship it")
    handle = _double_handle(session)

    class _RunningDriver:
        running = True

        async def cancel(self) -> None:
            return None

    handle._goal_loop = _RunningDriver()
    try:
        handle._maybe_judge_goal(AgentEndEvent(messages=[], generation=1))
        await asyncio.sleep(0.05)
        assert session.complete_aside_calls == []
        assert session.prompt_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_the_boot_rearm_judges_a_continuing_record_and_leaves_waiting_alone():
    continuing = GoalDouble()
    continuing.goal_state.arm("Ship it")
    continuing.goal_state.judge.state = "continuing"
    handle = _double_handle(continuing)
    try:
        handle.rearm_goal_judge()
        await wait_for(lambda: bool(continuing.complete_aside_calls))
        assert len(continuing.complete_aside_calls) == 1
        assert continuing.goal_status == "done"
    finally:
        await handle.dispose()

    waiting = GoalDouble()
    waiting.goal_state.arm("Ship it")
    waiting.goal_state.judge.state = "waiting"
    other = _double_handle(waiting)
    try:
        other.rearm_goal_judge()
        await asyncio.sleep(0.05)
        # RULINGS R3: a `waiting` goal is NOT re-armed by a restart — nothing was
        # in flight, and checkpoint state is not an instruction to spend tokens.
        assert waiting.complete_aside_calls == []
        assert waiting.goal_status == "active"
    finally:
        await other.dispose()


@pytest.mark.asyncio
async def test_a_stalled_record_is_not_rearmed_by_a_restart():
    session = GoalDouble()
    session.goal_state.arm("Ship it")
    session.goal_state.judge.state = "stalled"
    handle = _double_handle(session)
    try:
        handle.rearm_goal_judge()
        await asyncio.sleep(0.05)
        assert session.complete_aside_calls == []
    finally:
        await handle.dispose()


def _capture_notices(session: GoalDouble) -> list[Any]:
    """Install the runtime's emit seam and hand back what it collects.

    ``_emit_notice`` reads ``session._emit`` and drops to a log when it is
    absent, so a test that wants the user-visible half has to supply the seam —
    the same substitution ``test_serving``'s notice tests make.
    """
    seen: list[Any] = []

    async def _emit(event: Any) -> None:
        seen.append(event)

    session._emit = _emit
    return seen


async def _drain_notices(handle: ServingSessionHandle) -> None:
    """Let any queued notice task run, then drain its holder.

    ``_emit_notice`` is fire-and-forget by design (a notice must not delay the
    judge), so the work it did is only observable once these settle.
    """
    await asyncio.sleep(0.05)
    for task in list(handle._mcp_reload_tasks):
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_stall_reaches_attached_surfaces_once_and_names_the_bound():
    """Design round 1, D2: the state was announced nowhere.

    A goal whose judge has stopped auto-continuing looked exactly like one that
    is quietly waiting, which is the reading this receipt removes.

    Both arms stall on the FIRST judge call of the run, so no continuation turn
    is ever admitted — the assertion is about the announcement and not about the
    prompt queue — and each is reached through the runtime's OWN trigger
    (``rearm_goal_judge``, the boot path that drives ``_drive`` without the
    turn-end streak reset).
    """
    breaker = GoalDouble()
    breaker.goal_state.arm("Ship it")
    # A restored record that is one strike short of the breaker: the next
    # unreadable verdict is what fires it.
    breaker.goal_state.judge.state = "continuing"
    breaker.goal_state.judge.failures = GOAL_JUDGE_FAILURES - 1
    breaker.verdict = JUDGE_UNREADABLE
    breaker_seen = _capture_notices(breaker)
    handle = _double_handle(breaker)
    try:
        handle.rearm_goal_judge()
        await _drain_notices(handle)
        assert [event.text for event in breaker_seen] == [STALLED_BREAKER_NOTICE]
        assert breaker.goal_judge_state.reason == "judge could not decide"
        # ONCE PER ENTRY. The boot re-arm on a record that already reads
        # `stalled` publishes nothing at all (RULINGS R3), so it cannot announce
        # a second time; and a later frame that MOVES a field while the state
        # stays put is what the edge rule exists to keep quiet.
        handle.rearm_goal_judge()
        await _drain_notices(handle)
        assert len(breaker_seen) == 1
    finally:
        await handle.dispose()

    capped = GoalDouble()
    capped.goal_state.arm("Ship it")
    # The other bound: the run is already at the cap, so the next readable
    # CONTINUE is judged and then refused a continuation.
    capped.goal_state.judge.state = "continuing"
    capped.goal_state.judge.run = MAX_GOAL_CONTINUATIONS
    capped.verdict = JUDGE_CONTINUE
    cap_seen = _capture_notices(capped)
    handle = _double_handle(capped)
    try:
        handle.rearm_goal_judge()
        await _drain_notices(handle)
        assert [event.text for event in cap_seen] == [STALLED_CAP_NOTICE]
        assert capped.goal_judge_state.run == MAX_GOAL_CONTINUATIONS
        # `{"run": 0}` is exactly the diff `_drive`'s streak reset publishes at
        # the next turn end (`tests/unit/session/test_goal_judge.py` pins that
        # shape): the state does NOT move, so the receipt must not repeat.
        driver = handle._goal_judge_driver()
        assert driver is not None
        driver._publish(run=0)
        await _drain_notices(handle)
        assert [event.text for event in cap_seen] == [STALLED_CAP_NOTICE]
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_an_achieved_goal_reaches_surfaces_with_no_stall_receipt():
    """The negative control: settling passes through `done`, never `stalled`."""
    session = GoalDouble()
    session.goal_state.arm("Ship it")
    session.goal_state.judge.state = "continuing"
    session.verdict = JUDGE_ACHIEVED
    seen = _capture_notices(session)
    handle = _double_handle(session)
    try:
        handle.rearm_goal_judge()
        await _drain_notices(handle)
        assert session.goal_status == "done"
        assert [event for event in seen if isinstance(event, NoticeEvent)] == []
    finally:
        await handle.dispose()
