"""Wire-to-dispatch proof: final Responses snapshots must not become empty calls."""

from __future__ import annotations

import json

import httpx
import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    Message,
    ModelSpec,
    ToolResult,
)
from local_operator.providers.clients import OpenAICompatClient
from tests.unit.harness.test_loop import make_config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,prefix",
    [
        ("added", False),
        ("arguments_done", False),
        ("item_done", False),
        ("completed", False),
        ("arguments_done", True),
        ("item_done", True),
        ("completed", True),
        ("done_only", False),
        ("completed_only", False),
        ("late_delta", False),
        ("delta", False),
        ("empty_object", False),
    ],
)
async def test_final_arguments_reach_real_dispatch_once(source, prefix):
    await exercise_lifecycle(source, prefix=prefix)


@pytest.mark.asyncio
async def test_conflicting_final_arguments_never_dispatch():
    await exercise_lifecycle("item_done", prefix=True, conflict=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["changed_name", "changed_id", "missing", "nonstring"])
async def test_conflicting_final_identity_or_shape_never_dispatch(source):
    await exercise_lifecycle(source, conflict=True)


async def exercise_lifecycle(source, *, prefix=False, conflict=False):
    arguments = "{}" if source == "empty_object" else '{"path":"synthetic.txt"}'
    item = {
        "type": "function_call",
        "id": "fc-synthetic",
        "call_id": "call-synthetic",
        "name": "read",
        "arguments": arguments,
    }
    payloads = [
        # A reasoning item makes output_index differ from normalized call index.
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "r"},
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**item, "arguments": arguments if source == "added" else ""},
        },
    ]
    if source in ("done_only", "completed_only"):
        payloads = []
    if source == "delta":
        payloads.append(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": 1,
                "delta": arguments,
            }
        )
    if source == "changed_name":
        item = {**item, "name": "write"}
    if source == "changed_id":
        item = {**item, "call_id": "other-call"}
    if source == "missing":
        item = {key: value for key, value in item.items() if key != "arguments"}
    if source == "nonstring":
        item = {**item, "arguments": {"path": "synthetic.txt"}}
    if prefix:
        payloads.append(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": 1,
                "delta": '{"path":"wrong' if conflict else arguments[:9],
            }
        )
    if source == "arguments_done":
        payloads.append(
            {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": 1,
                "arguments": arguments,
            }
        )
    if source not in ("completed", "completed_only"):
        # Duplicate done snapshots are normal and must never double arguments.
        payloads.extend(
            [
                {"type": "response.output_item.done", "output_index": 1, "item": item},
                {"type": "response.output_item.done", "output_index": 1, "item": item},
            ]
        )
    payloads.append(
        {
            "type": "response.completed",
            "response": {"id": "synthetic-response", "output": [item]},
        }
    )
    if source == "late_delta":
        payloads.append(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": 1,
                "delta": arguments,
            }
        )
    executed = []
    captured = []

    async def execute(call_id, args, signal, on_update, context):
        executed.append(args)
        return ToolResult(tool_call_id=call_id, content=[])

    tool = AgentTool(
        name="read",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [] if source == "empty_object" else ["path"],
        },
        execute=execute,
    )

    def handler(request):
        captured.append(json.loads(request.content))
        events = (
            payloads
            if len(captured) == 1
            else [
                {"type": "response.output_text.delta", "delta": "Finished."},
                {"type": "response.completed", "response": {"id": "finished", "output": []}},
            ]
        )
        return httpx.Response(
            200,
            content="".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatClient(
            "https://synthetic.invalid/v1", http_client=http, openai_api="responses"
        )

        def stream(request, signal):
            return client.stream(request, "synthetic-token")

        context = LoopContext(system_blocks=["Synthetic boundary test"], tools=[tool])
        config = make_config(
            stream,
            model=ModelSpec(provider="openai", model_id="gpt-4.1", supports_responses_api=True),
        )
        events = [
            event
            async for event in AgentLoop().run(
                [Message.user("Read synthetic file")], context, config
            )
        ]
    assert captured[0]["tools"][0]["parameters"] == tool.parameters
    end = next(event for event in reversed(events) if isinstance(event, AgentEndEvent))
    if conflict:
        assert executed == []
        assert end.error is not None
        assert "responses_tool_arguments_conflict" in end.error
    else:
        assert executed == [json.loads(arguments)]
        assert end.error is None
        calls = [
            call
            for message in context.messages
            if isinstance(message, Message)
            for call in message.tool_calls
        ]
        assert len(calls) == 1
        assert calls[0].raw_arguments == arguments
        assert calls[0].arguments == json.loads(arguments)
