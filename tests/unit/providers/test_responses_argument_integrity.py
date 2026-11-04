"""Wire-to-dispatch proof: final Responses snapshots must not become empty calls."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    Message,
    ModelSpec,
    ToolContext,
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
        ("added_then_deltas", False),
        ("added_prefix_then_deltas", False),
        ("added_only", False),
        ("metadata_tail", False),
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
    added_arguments = ""
    if source in ("added", "added_then_deltas", "added_only"):
        added_arguments = arguments
    elif source == "added_prefix_then_deltas":
        added_arguments = arguments[:9]
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
            "item": {**item, "arguments": added_arguments},
        },
    ]
    if source in ("done_only", "completed_only"):
        payloads = []
    if source in ("delta", "added_then_deltas", "added_prefix_then_deltas"):
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
    if source not in ("completed", "completed_only", "added_only"):
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
            "response": {
                "id": "synthetic-response",
                "output": [{"type": "reasoning", "id": "r", "encrypted_content": "opaque"}, item],
            },
        }
    )
    if source == "added_only":
        # No final argument/item source: legacy endpoints can omit the output
        # field entirely. A populated added snapshot still must reach dispatch.
        payloads[-1] = {"type": "response.completed", "response": {"id": "synthetic-response"}}
    if source == "metadata_tail":
        payloads.append(
            {
                "type": "response.subscription_usage",
                "subscription": {"window": {"used_percent": 7}},
            }
        )
    if source in ("late_delta", "metadata_tail"):
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
    observed_metadata = []

    class ObservingClient(OpenAICompatClient):
        def _observe_responses_event(self, payload):
            # Billing metadata is observational, not generated output. The real
            # Muse client uses this same hook for its required trailing receipt.
            if payload.get("type") == "response.subscription_usage":
                observed_metadata.append(payload)
            super()._observe_responses_event(payload)

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
        client = ObservingClient(
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
    if source == "metadata_tail":
        assert len(observed_metadata) == 1
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "done_replacement",
        "terminal_replacement",
        "explicit_empty",
        "reasoning_only",
        "partial_membership",
        "changed_identity",
        "invalid_output",
        "omitted_output",
        "reverse_event_order",
        "delta_control",
    ],
)
async def test_terminal_batch_authority_precedes_all_approvals_and_effects(case):
    """Two pending writes prove a contradiction retires the WHOLE batch.

    Executors only append to a list; no filesystem or network writes occur.
    Output indices are stable protocol positions, not arrival-order indices.
    """
    calls = [
        {
            "type": "function_call",
            "id": f"fc-{index}",
            "call_id": f"call-{index}",
            "name": "write",
            "arguments": json.dumps({"path": f"synthetic-{index}"}),
        }
        for index in range(2)
    ]
    reasoning = {"type": "reasoning", "id": "reasoning-synthetic", "encrypted_content": "opaque"}
    payloads = []
    slots = [2, 0] if case == "reverse_event_order" else [0, 2]
    for item, slot in zip(calls, slots):
        payloads.append(
            {
                "type": "response.output_item.added",
                "output_index": slot,
                "item": {**item, "arguments": "" if case == "delta_control" else item["arguments"]},
            }
        )
        if case == "delta_control":
            payloads.append(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": slot,
                    "delta": item["arguments"],
                }
            )
        payloads.append({"type": "response.output_item.done", "output_index": slot, "item": item})
    payloads.append({"type": "response.output_item.done", "output_index": 1, "item": reasoning})
    terminal: dict[str, Any] = {
        "id": "terminal-synthetic",
        "output": [calls[0], reasoning, calls[1]],
    }
    if case == "reverse_event_order":
        terminal["output"] = [calls[1], reasoning, calls[0]]
    elif case == "done_replacement":
        payloads.append({"type": "response.output_item.done", "output_index": 2, "item": reasoning})
    elif case == "terminal_replacement":
        terminal["output"] = [calls[0], reasoning, reasoning]
    elif case == "explicit_empty":
        terminal["output"] = []
    elif case == "reasoning_only":
        terminal["output"] = [reasoning]
    elif case == "partial_membership":
        terminal["output"] = [calls[0], reasoning]
    elif case == "changed_identity":
        terminal["output"] = [calls[0], reasoning, {**calls[1], "call_id": "different-call"}]
    elif case == "invalid_output":
        terminal["output"] = None
    elif case == "omitted_output":
        terminal.pop("output")
    payloads.append({"type": "response.completed", "response": terminal})
    executed = []
    approvals = []
    requests = []

    async def execute(call_id, args, signal, on_update, context):
        executed.append(args)
        return ToolResult(tool_call_id=call_id, content=[])

    async def approve(*args, **kwargs):
        approvals.append(args)
        return True

    def handler(request):
        requests.append(request.url.path)
        events = (
            payloads
            if len(requests) == 1
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

    tool = AgentTool(
        name="write",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        execute=execute,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatClient(
            "https://synthetic.invalid/v1", http_client=http, openai_api="responses"
        )

        def stream(request, signal):
            return client.stream(request, "synthetic-token")

        context = LoopContext(
            system_blocks=["Synthetic terminal-authority test"],
            tools=[tool],
            tool_context=ToolContext(request_approval=approve),
        )
        config = make_config(
            stream,
            model=ModelSpec(provider="openai", model_id="gpt-4.1", supports_responses_api=True),
        )
        events = [
            event
            async for event in AgentLoop().run([Message.user("Synthetic writes")], context, config)
        ]
    end = next(event for event in reversed(events) if isinstance(event, AgentEndEvent))
    if case in ("omitted_output", "reverse_event_order", "delta_control"):
        assert end.error is None
        assert executed == [{"path": "synthetic-0"}, {"path": "synthetic-1"}]
        assert len(approvals) == 2
        assert len(requests) == 2
    else:
        assert executed == []
        assert approvals == []
        assert len(requests) == 1
        assert end.error is not None and "responses_tool_arguments_conflict" in end.error
