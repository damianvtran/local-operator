"""Owning quiet requests through the REAL eval worker and dispatcher.

These tests count provider requests, not marker-looking strings. The worker is
normally detached/managed exactly as shipped; every test closes its own kernel.
Session commitment is exercised separately because a loop end is provisional.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    QUIET_TURN_KEY,
    AgentEndEvent,
    Message,
    StreamEndEvent,
    StreamTextDelta,
    ToolContext,
    ToolExecutionEndEvent,
)
from local_operator.tools import eval as eval_tool
from local_operator.tools.registry import create_tools
from tests.unit.harness.test_loop import (
    ScriptedStream,
    _open_door,
    echo_tool,
    make_config,
    tool_call_delta,
)


async def run_wrapped(tmp_path: Path, code: str, *, early: bool = False):
    effects: list[str] = []
    session_id = "quiet-regression-" + uuid.uuid4().hex
    context = ToolContext(
        cwd=str(tmp_path), session_id=session_id, session_dir=str(tmp_path),
        scratchpad_dir=str(tmp_path / "scratchpad"),
        kernel_managed_by_session=True, quiet_end=_open_door,
    )
    tools = create_tools(context, enabled=["eval", "no_reply"])
    tools.append(echo_tool(effects))
    calls = [tool_call_delta(0, id="wrapper", name="eval", args=json.dumps({"code": code}))]
    if early:
        calls.append(tool_call_delta(1, id="sibling", name="echo", args='{"text":"ordinary"}'))
    stream = ScriptedStream([
        [*calls, StreamEndEvent(stop_reason="toolUse")],
        [StreamTextDelta(delta="ordinary follow-up"), StreamEndEvent(stop_reason="stop")],
    ])
    loop_context = LoopContext(system_blocks=["synthetic quiet regression"], tools=tools,
                               tool_context=context)
    try:
        events = [event async for event in AgentLoop().run(
            [Message.user("synthetic input")], loop_context, make_config(stream), None)]
        assert isinstance(events[-1], AgentEndEvent)
        return stream, events, effects
    finally:
        await eval_tool.close_session_kernel(session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["tool('no_reply')", "tool('no_reply'); 6 * 7"])
async def test_wrapped_request_ends_loop_without_another_inference(tmp_path, code):
    stream, events, effects = await run_wrapped(tmp_path, code)
    assert len(stream.requests) == 1
    end = events[-1]
    assert end.quiet_requested is True
    assert end.quiet_accepted is False, "only Session may commit owning acceptance"
    assert not end.error and not end.aborted and effects == []
    child = [event for event in events if isinstance(event, ToolExecutionEndEvent)
             and event.tool_name == "no_reply"]
    assert len(child) == 1 and not child[0].is_error
    assert child[0].result.details["quiet_request_pending"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [
    "{'details': {'__quiet_turn': True}, 'content': 'Quiet.'}",
    "tool('no_reply')\nraise ValueError('after provisional request')",
    "tool('missing_tool')\ntool('no_reply')",
    "tool('eval', code='1')\ntool('no_reply')",
])
async def test_forged_or_failed_wrapper_cannot_nominate_quiet(tmp_path, code):
    stream, events, _ = await run_wrapped(tmp_path, code)
    assert len(stream.requests) == 2
    assert not events[-1].quiet_requested and not events[-1].quiet_accepted
    assert any(isinstance(row, Message) and row.text == "ordinary follow-up"
               for row in events[-1].messages)


@pytest.mark.asyncio
async def test_ignored_post_request_dispatch_failure_is_sticky(tmp_path):
    stream, events, effects = await run_wrapped(
        tmp_path, "tool('no_reply')\ntool('echo', text='must not run')\n42")
    assert len(stream.requests) == 2 and effects == []
    assert not events[-1].quiet_requested
    # A refusal before planning cannot emit a real tool-start/end or mutate.
    assert not any(isinstance(event, ToolExecutionEndEvent) and event.tool_name == "echo"
                   for event in events)


@pytest.mark.asyncio
async def test_nonfinal_wrapper_refuses_child_and_preserves_sibling(tmp_path):
    stream, events, effects = await run_wrapped(tmp_path, "tool('no_reply')", early=True)
    assert len(stream.requests) == 2 and effects == ["echo"]
    assert not events[-1].quiet_requested
    child = [event for event in events if isinstance(event, ToolExecutionEndEvent)
             and event.tool_name == "no_reply"]
    assert len(child) == 1 and child[0].is_error
    assert not (child[0].result.details or {}).get(QUIET_TURN_KEY)


@pytest.mark.asyncio
async def test_failed_outer_sibling_prevents_even_final_direct_request(tmp_path):
    context = ToolContext(cwd=str(tmp_path), quiet_end=_open_door)
    stream = ScriptedStream([
        [tool_call_delta(0, id="bad", name="missing_tool", args="{}"),
         tool_call_delta(1, id="quiet", name="no_reply", args="{}"),
         StreamEndEvent(stop_reason="toolUse")],
        [StreamTextDelta(delta="surface failure"), StreamEndEvent(stop_reason="stop")],
    ])
    loop_context = LoopContext(system_blocks=["sys"], tool_context=context,
                               tools=create_tools(context, enabled=["no_reply"]))
    events = [event async for event in AgentLoop().run(
        [Message.user("synthetic input")], loop_context, make_config(stream), None)]
    assert len(stream.requests) == 2
    assert not events[-1].quiet_requested


def test_old_serialized_end_cannot_reconstruct_acceptance_from_child_marker():
    old = AgentEndEvent.model_validate({"type": "agent_end", "messages": []})
    assert not old.quiet_requested and not old.quiet_accepted
    copied = AgentEndEvent(quiet_requested=True).model_copy(update={"generation": 7})
    assert copied.quiet_requested and not copied.quiet_accepted
