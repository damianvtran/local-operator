"""Exercise the real loop/executor boundary, not the detector's private state."""

from __future__ import annotations

import json

import pytest

from local_operator.harness.loop import (
    MAX_NO_PROGRESS_RECOVERIES,
    NO_PROGRESS_LIMIT,
    NO_PROGRESS_RECOVERY_PROMPT,
    NO_PROGRESS_WARNING,
    REPEATED_TOOL_ERROR_LIMIT,
    AgentLoop,
    LoopContext,
)
from local_operator.harness.rows import is_harness_chrome
from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    ImageContent,
    Message,
    MessageEndEvent,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolResult,
)
from tests.unit.harness.test_loop import ScriptedStream, make_config, tool_call_delta


async def run_calls(calls, *, results=None, errors=False, batch_size=1):
    """Distinct wire IDs and timings must not disguise identical observations."""
    executed = []

    async def execute(tool_call_id, args, signal, on_update, context):
        index = len(executed)
        executed.append((tool_call_id, args))
        content = results[index] if results is not None else [TextContent(text="unchanged")]
        return ToolResult(
            tool_call_id=tool_call_id,
            content=content,
            is_error=errors,
            duration_s=index / 1000,
        )

    tools = [
        AgentTool(name=name, parameters={"type": "object"}, execute=execute)
        for name in dict.fromkeys(name for name, _ in calls)
    ]
    turns = []
    for start in range(0, len(calls), batch_size):
        turns.append(
            [
                tool_call_delta(index, id=f"call-{start + index}", name=name, args=json.dumps(args))
                for index, (name, args) in enumerate(calls[start : start + batch_size])
            ]
            + [StreamEndEvent(stop_reason="tool_calls")]
        )
    turns.append([StreamTextDelta(delta="Finished."), StreamEndEvent(stop_reason="stop")])
    stream = ScriptedStream(turns)
    context = LoopContext(system_blocks=["Test repetition recovery"], tools=tools)
    events = [
        event
        async for event in AgentLoop().run(
            [Message.user("Exercise the tools")], context, make_config(stream)
        )
    ]
    nudges = [
        event.message
        for event in events
        if isinstance(event, MessageEndEvent)
        and isinstance(event.message, Message)
        and event.message.text == NO_PROGRESS_RECOVERY_PROMPT
    ]
    end = next(event for event in reversed(events) if isinstance(event, AgentEndEvent))
    return stream, executed, nudges, end


def cycle(period, repetitions, *, marker=0):
    actions = [
        ("wait", {"duration_ms": 120, "marker": marker}),
        ("key", {"keys": ["ESC"], "marker": marker}),
        ("read", {"path": "state", "marker": marker}),
    ][:period]
    return actions * repetitions


@pytest.mark.asyncio
@pytest.mark.parametrize("period", [1, 2, 3])
async def test_warning_boundary_and_single_nudge(period):
    # The below-boundary half alone would pass with a dead guard. The second
    # half is its positive control, so deleting the guard fails this test too.
    stream, executed, nudges, end = await run_calls(cycle(period, NO_PROGRESS_WARNING - 1))
    assert nudges == []
    assert end.error is None
    assert len(stream.requests) == len(executed) + 1

    stream, executed, nudges, end = await run_calls(cycle(period, NO_PROGRESS_WARNING + 2))
    assert len(nudges) == 1
    assert end.error is None
    assert len(stream.requests) == len(executed) + 1
    recovery_requests = [
        index
        for index, request in enumerate(stream.requests)
        if any(message.text == NO_PROGRESS_RECOVERY_PROMPT for message in request.messages)
    ]
    assert recovery_requests[0] == period * NO_PROGRESS_WARNING
    assert nudges[0] in end.messages
    assert is_harness_chrome(nudges[0].text)


@pytest.mark.asyncio
async def test_argument_key_order_is_not_progress():
    calls = [
        (
            ("wait", {"duration_ms": 120, "nested": {"a": 1, "b": 2}})
            if index % 2
            else ("wait", {"nested": {"b": 2, "a": 1}, "duration_ms": 120})
        )
        for index in range(NO_PROGRESS_WARNING)
    ]
    _, _, nudges, end = await run_calls(calls)
    assert len(nudges) == 1
    assert end.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("media", [False, True])
async def test_changing_results_are_progress_with_live_guard_control(media):
    count = NO_PROGRESS_LIMIT * 3
    changing = [
        (
            [ImageContent(data=f"image-{index}", mime_type="image/png")]
            if media
            else [TextContent(text=f"progress-{index}")]
        )
        for index in range(count)
    ]
    stream, _, nudges, end = await run_calls(cycle(1, count), results=changing)
    assert len(stream.requests) == count + 1
    assert nudges == []
    assert end.error is None
    # Appending an unchanged tail verifies the same detector was actually live,
    # rather than a disconnected guard satisfying the polling assertion.
    _, _, nudges, end = await run_calls(
        cycle(1, count + NO_PROGRESS_WARNING),
        results=changing + [[TextContent(text="settled")]] * NO_PROGRESS_WARNING,
    )
    assert len(nudges) == 1
    assert end.error is None


@pytest.mark.asyncio
async def test_recovery_budget_is_per_run_not_per_streak():
    calls = []
    for marker in range(MAX_NO_PROGRESS_RECOVERIES + 2):
        calls.extend(cycle(1, NO_PROGRESS_WARNING, marker=marker))
    stream, _, nudges, end = await run_calls(calls)
    assert len(nudges) == MAX_NO_PROGRESS_RECOVERIES
    assert end.error is None
    assert len(stream.requests) == len(calls) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("period,batch_size", [(1, 1), (2, 1), (3, 1), (2, 2), (3, 3)])
async def test_limit_ends_run_with_diagnosable_repetition_error(period, batch_size):
    stream, executed, nudges, end = await run_calls(
        cycle(period, NO_PROGRESS_LIMIT + 2), batch_size=batch_size
    )
    assert len(nudges) == 1
    assert end.error is not None
    assert "No progress: repeated tool action or short cycle" in end.error
    assert f"period {period}" in end.error
    assert f"{NO_PROGRESS_LIMIT} repetitions" in end.error
    assert len(executed) == period * NO_PROGRESS_LIMIT
    assert len(stream.requests) == len(executed) // batch_size


@pytest.mark.asyncio
async def test_existing_all_error_limit_wins_before_success_guard():
    _, executed, nudges, end = await run_calls(cycle(1, NO_PROGRESS_LIMIT), errors=True)
    assert len(executed) == REPEATED_TOOL_ERROR_LIMIT
    assert nudges == []
    assert end.error is not None
    assert "returned the same errors" in end.error
    # Positive control also makes this regression case fail if the new guard
    # is removed, without modifying the existing failure-guard tests.
    _, _, nudges, end = await run_calls(cycle(1, NO_PROGRESS_WARNING))
    assert len(nudges) == 1
    assert end.error is None
