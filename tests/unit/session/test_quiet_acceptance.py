"""Session owns quiet acceptance, after ordinary host work—not child markers.

Real pipeline tests inject queued people at host settlement and new admissions
at emitted acceptance. Small seam tests pin mechanical compaction/latch guards.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from local_operator.harness.types import (
    AgentEndEvent, CustomMessage, Message, StreamEndEvent, StreamTextDelta,
    StreamToolCallDelta,
)
from local_operator.session.goal_judge import quiet_completion_accepted
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tools import eval as eval_tool
from tests.unit.session.test_session import ScriptedStream, make_session


def quiet_stream(*later):
    return ScriptedStream([
        [StreamToolCallDelta(index=0, id="quiet", name="no_reply", argument_delta="{}"),
         StreamEndEvent(stop_reason="toolUse")], *later,
    ])


def peer_input():
    return CustomMessage(custom_type="peer_message", attribution="peer",
                         details={"text": "synthetic peer acknowledgement"})


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_real_pipeline_commits_only_one_terminal_quiet_end(tmp_path, wrapped):
    stream = quiet_stream()
    tools = None
    if wrapped:
        stream = ScriptedStream([[
            StreamToolCallDelta(index=0, id="wrapper", name="eval",
                                argument_delta=json.dumps({"code": "tool('no_reply')"})),
            StreamEndEvent(stop_reason="toolUse"),
        ]])
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
    tmp_path, monkeypatch,
):
    stream = quiet_stream([StreamTextDelta(delta="handled late person"), StreamEndEvent(stop_reason="stop")])
    session = make_session(tmp_path, stream)
    injected = False
    events = []
    session.subscribe(events.append)

    async def inject_at_post_run_compaction():
        nonlocal injected
        if not injected:
            injected = True
            session.steer("late-person-exactly-once")

    monkeypatch.setattr(session, "_maybe_compact", inject_at_post_run_compaction)
    try:
        await session._prompt_messages([peer_input()])
        ends = [event for event in events if isinstance(event, AgentEndEvent)]
        assert len(stream.requests) == 2 and len(ends) == 1
        assert not ends[0].quiet_accepted
        assert sum(isinstance(row, Message) and "late-person-exactly-once" in row.text
                   for row in stream.requests[1].messages) == 1
        assert not session._has_awaiting_user()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_input_after_accepted_end_gets_a_new_pipeline_and_generation(tmp_path):
    stream = quiet_stream([StreamTextDelta(delta="new answer"), StreamEndEvent(stop_reason="stop")])
    session = make_session(tmp_path, stream)
    events = []
    tasks = []

    def observe(event):
        events.append(event)
        if isinstance(event, AgentEndEvent) and event.quiet_accepted:
            # Real ingress queues behind the old pipeline lock; never call
            # _run_turn directly and pretend that means a different owner.
            tasks.append(asyncio.create_task(session.prompt("new-person-next-owner")))

    session.subscribe(observe)
    try:
        await session._prompt_messages([peer_input()])
        await asyncio.gather(*tasks)
        ends = [event for event in events if isinstance(event, AgentEndEvent)]
        starts = [event for event in events if getattr(event, "type", "") == "agent_start"]
        assert len(stream.requests) == 2 and len(ends) == len(starts) == 2
        assert ends[0].quiet_accepted and not ends[1].quiet_accepted
        assert ends[0].generation != ends[1].generation
        assert events.index(ends[0]) < events.index(starts[1])
        assert sum(isinstance(row, Message) and row.text == "new-person-next-owner"
                   for row in stream.requests[1].messages) == 1
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
        assert "person" in (await session._quiet_end_refusal()).lower()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_compaction_producer_does_not_revive_a_quiet_request(tmp_path):
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        session._held_end = AgentEndEvent(quiet_requested=True,
                                         messages=[Message(role="assistant", stop_reason="length")])
        # No compaction API is provided: the quiet-specific producer gate must
        # return BEFORE consulting stale length/recovery-band bookkeeping.
        session._after_compaction_pass(SimpleNamespace(), SimpleNamespace(tokens_after=12))
        assert not session._continuation_queue
        assert session._held_context_tokens == 12
    finally:
        session._held_end = None
        await session.dispose()


@pytest.mark.asyncio
async def test_committed_latch_prevents_internal_same_owner_inference_without_dropping_work(tmp_path):
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
    ServingSessionHandle._maybe_judge_goal(owner, event)
    ServingSessionHandle._maybe_supplement(owner, event)
    assert not quiet_completion_accepted(AgentEndEvent(quiet_requested=True))


@pytest.mark.parametrize("fields", [{"error": "failed"}, {"aborted": True}, {"cut_off_cause": "cut"}])
def test_error_abort_cutoff_take_precedence_over_acceptance(fields):
    from local_operator.session.session import Session
    event = AgentEndEvent(quiet_accepted=True, **fields)
    assert not Session._run_ended_quiet(None, event)
