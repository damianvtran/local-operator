"""Session owns quiet acceptance, after ordinary host work—not child markers.

Real pipeline tests inject queued people at host settlement and new admissions
at emitted acceptance. Small seam tests pin mechanical compaction/latch guards.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    CustomMessage,
    Message,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
)
from local_operator.session.goal_judge import quiet_completion_accepted
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tools import eval as eval_tool
from tests.unit.session.test_session import ScriptedStream, make_session


def quiet_stream(*later):
    return ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="quiet", name="no_reply", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            *later,
        ]
    )


def peer_input():
    return CustomMessage(
        custom_type="peer_message",
        attribution="user",
        details={"text": "synthetic peer acknowledgement"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_real_pipeline_commits_only_one_terminal_quiet_end(tmp_path, wrapped):
    stream = quiet_stream()
    tools = None
    if wrapped:
        stream = ScriptedStream(
            [
                [
                    StreamToolCallDelta(
                        index=0,
                        id="wrapper",
                        name="eval",
                        argument_delta=json.dumps({"code": "tool('no_reply')"}),
                    ),
                    StreamEndEvent(stop_reason="toolUse"),
                ]
            ]
        )
        tools = [eval_tool.build_eval_tool()]
    session = make_session(tmp_path, stream, tools=tools)
    events = []
    session.subscribe(events.append)
    try:
        await session._prompt_messages([peer_input()])
        ends = [event for event in events if isinstance(event, AgentEndEvent)]
        assert len(stream.requests) == 1 and len(ends) == 1
        assert ends[0].quiet_requested and ends[0].quiet_accepted
        assert session._quiet_accepted_generation == ends[0].generation
        assert not ends[0].notify
        assert not session._continuation_queue
        assert not (await session.refresh_attention())["completion_token"]
    finally:
        await eval_tool.close_session_kernel(session.session_id)
        await session.dispose()


@pytest.mark.asyncio
async def test_person_arriving_during_host_settlement_is_consumed_once_before_acceptance(
    tmp_path,
    monkeypatch,
):
    stream = quiet_stream(
        [StreamTextDelta(delta="handled late person"), StreamEndEvent(stop_reason="stop")]
    )
    session = make_session(tmp_path, stream)
    injected = False
    host_refusals = []
    original_refusal = session._quiet_end_refusal
    events = []
    session.subscribe(events.append)

    async def inject_at_post_run_compaction():
        nonlocal injected
        if not injected:
            # Pin the ordinary/non-body side of the held-body exception: the
            # request is already provisional, but no bridge reservation ever
            # owned it. A newly queued person must still win at HOST recheck.
            assert session._held_end is not None and session._held_end.quiet_requested
            assert session._pending_body_turn is None
            assert session._quiet_requested_by_body is False
            injected = True
            session.steer("late-person-exactly-once")

    async def record_host_refusal(*, held_body_request=False):
        refusal = await original_refusal(held_body_request=held_body_request)
        if injected and session._held_end is not None and session._held_end.quiet_requested:
            host_refusals.append((held_body_request, refusal))
        return refusal

    monkeypatch.setattr(session, "_maybe_compact", inject_at_post_run_compaction)
    monkeypatch.setattr(session, "_quiet_end_refusal", record_host_refusal)
    try:
        await session._prompt_messages([peer_input()])
        ends = [event for event in events if isinstance(event, AgentEndEvent)]
        assert len(stream.requests) == 2 and len(ends) == 1
        assert not ends[0].quiet_accepted
        assert host_refusals == [(False, "A person asked this turn; answer them in one line.")]
        assert (
            sum(
                isinstance(row, Message) and "late-person-exactly-once" in row.text
                for row in stream.requests[1].messages
            )
            == 1
        )
        assert not session._has_awaiting_user()
    finally:
        await session.dispose()


# One constant feeds both the wait and its diagnostic so the message cannot drift.
SETTLE_TIMEOUT_S = 8


def timeout_state(session, events, steer_attempts, steer_returns):
    """Name which settle conjunct stayed true when a wait times out.

    A hosted run failed with a bare TimeoutError, leaving a product hang (the
    steer never produced the second turn) indistinguishable from a starved
    runner. Read-only: no behaviour of the session is touched. Every field is
    read defensively (``n/a`` on failure) so this can never replace the
    original timeout with an unrelated AttributeError.
    """

    def field(read):
        try:
            return read()
        except Exception:
            return "n/a"

    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    tail = ", ".join(
        f"{getattr(event, 'type', type(event).__name__)}"
        + (f"(quiet_accepted={event.quiet_accepted})" if isinstance(event, AgentEndEvent) else "")
        for event in events[-12:]
    )
    def describe_task(task):
        """Bare label plus the await chain a pending task is parked on.

        A measured healthy post-end tail is ~3 ms, so an 8 s timeout is a
        BLOCKED await, and bare ``Task-N:pending`` labels cannot name it. Walk
        the ``cr_await`` chain (bounded, read-only) to the innermost awaiting
        frame. Any failure degrades to the bare label, never a new exception.
        """
        label = f"{task.get_name()}:{'done' if task.done() else 'pending'}"
        try:
            chain = []
            node = task.get_coro()
            for _ in range(8):
                if node is None:
                    break
                frame = getattr(node, "cr_frame", None) or getattr(node, "gi_frame", None)
                name = getattr(node, "__qualname__", type(node).__name__)
                where = f"@{frame.f_code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}" if frame else ""
                chain.append(f"{name}{where}")
                node = getattr(node, "cr_await", None) or getattr(node, "gi_yieldfrom", None)
            if node is not None:
                chain.append(type(node).__name__)
            return f"{label}[{' > '.join(chain)}]" if chain else label
        except Exception:
            return label

    tasks = field(
        lambda: "; ".join(
            describe_task(task)
            for task in sorted(asyncio.all_tasks(), key=lambda task: task.get_name())[:10]
        )
    )
    return (
        f"cause unknown, not a fix: new owner never settled in {SETTLE_TIMEOUT_S}s; "
        f"agent_end_count={len(ends)} (needs 2), "
        f"is_streaming={field(lambda: session.is_streaming)}, "
        f"turn_lock_locked={field(lambda: session._turn_lock.locked())}, "
        f"steer attempts={steer_attempts[0]} returns={steer_returns[0]}, "
        f"steering_queue_size={field(lambda: session._steering_queue.qsize())}, "
        f"continuation_queue={field(lambda: len(session._continuation_queue))}, "
        f"last_events=[{tail}], tasks=[{tasks}]"
    )


@pytest.mark.asyncio
async def test_input_after_accepted_end_gets_a_new_pipeline_and_generation(tmp_path):
    stream = quiet_stream([StreamTextDelta(delta="new answer"), StreamEndEvent(stop_reason="stop")])
    session = make_session(tmp_path, stream)
    events = []
    steer_attempts = [0]
    steer_returns = [0]

    def observe(event):
        events.append(event)
        if isinstance(event, AgentEndEvent) and event.quiet_accepted:
            # The composer uses steer while this owner still holds its lock.
            # prompt() intentionally rejects that phase with TurnInFlight.
            steer_attempts[0] += 1
            session.steer("new-person-next-owner", message_id="qa-next-owner-person")
            # attempts=1 returns=0 means steer raised (the dispatcher swallows handler
            # exceptions); attempts=0 means this branch was never reached.
            steer_returns[0] += 1

    session.subscribe(observe)
    try:
        await session._prompt_messages([peer_input()])

        async def new_owner_settled():
            while (
                sum(isinstance(event, AgentEndEvent) for event in events) < 2
                or session.is_streaming
                or session._turn_lock.locked()
            ):
                await asyncio.sleep(0)

        try:
            await asyncio.wait_for(new_owner_settled(), timeout=SETTLE_TIMEOUT_S)
        except TimeoutError as exc:
            raise AssertionError(
                timeout_state(session, events, steer_attempts, steer_returns)
            ) from exc
        ends = [event for event in events if isinstance(event, AgentEndEvent)]
        starts = [event for event in events if getattr(event, "type", "") == "agent_start"]
        assert len(stream.requests) == 2 and len(ends) == len(starts) == 2
        assert (
            sum(
                isinstance(row, Message) and row.text == "new-person-next-owner"
                for row in session.history()
            )
            == 1
        )
        assert not session._has_awaiting_user()
        assert ends[0].quiet_accepted and not ends[1].quiet_accepted
        assert ends[0].generation != ends[1].generation
        assert events.index(ends[0]) < events.index(starts[1])
        assert (
            sum(
                isinstance(row, Message) and row.text == "new-person-next-owner"
                for row in stream.requests[1].messages
            )
            == 1
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,blocked", [("", True), ("active", True), ("done", False)])
async def test_only_outstanding_runtime_goal_blocks_quiet(tmp_path, status, blocked):
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        session._goal_state.text = "explicit runtime objective"
        session._goal_state.status = status
        assert bool(await session._quiet_end_refusal()) is blocked
        session._goal_state.text = ""
        assert await session._quiet_end_refusal() is None
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_delivered_notification_exception_never_overrides_a_person(tmp_path):
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        session._run_triggers = {"wake_prompt"}
        session._run_notify_requested = True
        session._run_notified = False
        assert await session._quiet_end_refusal()
        session._run_notified = True
        assert await session._quiet_end_refusal() is None
        session._run_triggers.add("user")
        refusal = await session._quiet_end_refusal()
        assert refusal is not None and "person" in refusal.lower()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_compaction_producer_does_not_revive_a_quiet_request(tmp_path):
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        session._held_end = AgentEndEvent(
            quiet_requested=True, messages=[Message(role="assistant", stop_reason="length")]
        )
        # No compaction API is provided: the quiet-specific producer gate must
        # return BEFORE consulting stale length/recovery-band bookkeeping.
        session._after_compaction_pass(
            cast(Any, SimpleNamespace()), cast(Any, SimpleNamespace(tokens_after=12))
        )
        assert not session._continuation_queue
        assert session._held_context_tokens == 12
    finally:
        session._held_end = None
        await session.dispose()


@pytest.mark.asyncio
async def test_committed_latch_prevents_internal_same_owner_inference_without_dropping_work(
    tmp_path,
):
    session = make_session(tmp_path, ScriptedStream([]))
    queued = Message.user("retained work")
    try:
        session._quiet_accepted_generation = 7
        session._continuation_queue.append(queued)
        await session._drain_continuation()
        assert session._continuation_queue == [queued]
        with pytest.raises(RuntimeError, match="new pipeline admission"):
            await session._run_turn([queued])
        assert session._continuation_queue == [queued]
    finally:
        session._quiet_accepted_generation = None
        session._continuation_queue.clear()
        await session.dispose()


@pytest.mark.asyncio
async def test_host_continuations_settle_before_commit(tmp_path, monkeypatch):
    session = make_session(tmp_path, ScriptedStream([]))
    first = AgentEndEvent(quiet_requested=True, generation=1)
    replacement = AgentEndEvent(quiet_requested=True, generation=1)
    calls = []

    async def drain():
        assert session._quiet_accepted_generation is None
        calls.append("required continuation")
        session._held_end = replacement

    monkeypatch.setattr(session, "_drain_continuation", drain)
    try:
        session._held_end = first
        await session._settle_quiet_request()
        assert calls == ["required continuation"]
        assert not first.quiet_accepted
        assert session._held_end.quiet_accepted
    finally:
        session._held_end = None
        await session.dispose()


def test_final_accepted_event_fences_both_runtime_model_subscribers():
    # A forged/late active goal at the subscriber cannot revive accepted work;
    # normal active goals are refused earlier by Session's permission door.
    owner = SimpleNamespace()
    event = AgentEndEvent(quiet_accepted=True)
    assert quiet_completion_accepted(event)
    ServingSessionHandle._maybe_judge_goal(cast(Any, owner), event)
    ServingSessionHandle._maybe_supplement(cast(Any, owner), event)
    assert not quiet_completion_accepted(AgentEndEvent(quiet_requested=True))


@pytest.mark.parametrize(
    "fields", [{"error": "failed"}, {"aborted": True}, {"cut_off_cause": "cut"}]
)
def test_error_abort_cutoff_take_precedence_over_acceptance(fields):
    from local_operator.session.session import Session

    event = AgentEndEvent(quiet_accepted=True, **fields)
    assert not Session._run_ended_quiet(cast(Any, None), event)
