"""Missing-required recovery exercises model requests and the dispatch boundary."""

from __future__ import annotations

import json

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    Message,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    ToolContext,
    ToolResult,
)
from tests.unit.harness.test_loop import ScriptedStream, make_config, tool_call_delta


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,field,value",
    [
        ("read", "path", "synthetic.txt"),
        ("sessions", "op", "list"),
        ("web_search", "query", "synthetic"),
        ("web_fetch", "url", "https://example.test"),
        ("notify", "title", "Synthetic reminder"),
    ],
)
@pytest.mark.parametrize("corrected", [False, True])
async def test_missing_required_corrects_once_or_stops(name, field, value, corrected):
    executed = []
    approvals = []

    async def execute(call_id, arguments, signal, on_update, context):
        executed.append(arguments)
        return ToolResult(tool_call_id=call_id, content=[])

    async def approve(*args, **kwargs):
        approvals.append((args, kwargs))
        return True

    tool = AgentTool(
        name=name,
        parameters={
            "type": "object",
            "properties": {field: {"type": "string"}},
            "required": [field],
        },
        execute=execute,
    )
    arguments = [{}, {field: value} if corrected else {}, {}, {}, {}, {}]
    turns: list[list[StreamEvent]] = [
        [
            tool_call_delta(0, id=f"synthetic-{index}", name=name, args=json.dumps(args)),
            StreamEndEvent(stop_reason="tool_calls"),
        ]
        for index, args in enumerate(arguments[:2] if corrected else arguments)
    ]
    turns.append([StreamTextDelta(delta="Finished."), StreamEndEvent(stop_reason="stop")])
    stream = ScriptedStream(turns)
    context = LoopContext(
        system_blocks=["Synthetic argument recovery"],
        tools=[tool],
        tool_context=ToolContext(request_approval=approve),
    )
    events = [
        event
        async for event in AgentLoop().run(
            [Message.user("Exercise synthetic tool")], context, make_config(stream)
        )
    ]
    end = next(event for event in reversed(events) if isinstance(event, AgentEndEvent))
    # The first refusal itself carries correction guidance, before another paid
    # generation. It must not fabricate a path, op, URL, query, or reminder.
    rejected = next(
        message
        for message in context.messages
        if isinstance(message, Message) and message.role == "tool"
    )
    assert f"missing required argument '{field}'" in rejected.text
    assert "each need their own populated parameters" in rejected.text
    recovery_request = stream.requests[1]
    assert any(
        "required tool arguments were omitted" in message.text
        for message in recovery_request.messages
    )
    if corrected:
        # A corrected call still crosses the ordinary approval boundary once;
        # the rejected predecessor must not have prompted the operator.
        assert len(approvals) == 1
        assert executed == [{field: value}]
        assert end.error is None
        assert len(stream.requests) == 3
    else:
        assert approvals == []
        assert executed == []
        assert len(stream.requests) == 2
        assert end.error is not None
        assert "2 unchanged tool batches" in end.error


@pytest.mark.asyncio
@pytest.mark.parametrize("child_arguments", [{}, {"path": {}}, {"path": []}])
async def test_nested_composition_invalid_child_never_executes_or_prompts(child_arguments):
    """The real eval bridge validates each child; no invented parallel wrapper.

    Nested object/list values fail the child's scalar contract. A free-form
    parameters object on an external wrapper is not a harness-owned schema.
    """
    executed = []
    approvals = []
    child_results = []

    async def read(call_id, args, signal, on_update, context):
        executed.append(args)
        return ToolResult(tool_call_id=call_id, content=[])

    async def compose(call_id, args, signal, on_update, context):
        child_results.append(await context.dispatch_tool("read", child_arguments))
        return ToolResult(tool_call_id=call_id, content=[])

    async def approve(*args, **kwargs):
        approvals.append(args)
        return True

    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="synthetic-eval", name="eval", args='{"code":"synthetic"}'),
                StreamEndEvent(stop_reason="tool_calls"),
            ],
            [StreamTextDelta(delta="Finished."), StreamEndEvent(stop_reason="stop")],
        ]
    )
    tools = [
        AgentTool(name="eval", parameters={"type": "object"}, execute=compose),
        AgentTool(
            name="read",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            execute=read,
        ),
    ]
    context = LoopContext(
        system_blocks=["Synthetic nested test"],
        tools=tools,
        tool_context=ToolContext(request_approval=approve),
    )
    await AgentLoop().run_to_end(
        [Message.user("Compose synthetic call")], context, make_config(stream), None
    )
    assert executed == []
    assert child_results[0]["is_error"] is True
    assert len(approvals) == 1  # Outer eval only: child validation precedes approval.
