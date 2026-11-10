"""Conflict diagnostics identify the check without exposing provider/user bytes."""

from __future__ import annotations

import json

import httpx
import pytest

from local_operator.harness.types import ChatRequest, ModelSpec
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.failover import ProviderError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,stage",
    [
        ("argument_prefix", "response.function_call_arguments.done"),
        ("slot_identity", "response.output_item.done"),
        ("added_argument_prefix", "response.output_item.done"),
        ("added_arguments_changed", "response.output_item.added"),
        ("done_arguments_type", "response.function_call_arguments.done"),
        ("delta_after_final", "response.function_call_arguments.delta"),
        ("duplicate_terminal_call", "response.completed"),
        ("legacy_argument_prefix", "response.completed"),
    ],
)
async def test_conflict_reason_is_safe_and_fail_closed(reason, stage, caplog):
    # Deliberately recognizable private-looking data proves diagnostics report
    # structure, not raw values, even when identities and arguments disagree.
    item = {
        "type": "function_call",
        "id": "private-item-id",
        "call_id": "private-call-id",
        "name": "private-tool-name",
        "arguments": '{"path":"private-fixture.txt"}',
    }
    added = {**item, "arguments": ""}
    events = [{"type": "response.output_item.added", "output_index": 0, "item": added}]

    def delta(text):
        return {
            "type": "response.function_call_arguments.delta",
            "item_id": item["id"],
            "output_index": 0,
            "delta": text,
        }

    def done(arguments):
        return {
            "type": "response.function_call_arguments.done",
            "item_id": item["id"],
            "output_index": 0,
            "arguments": arguments,
        }

    if reason == "argument_prefix":
        events.extend([delta('{"path":"other'), done(item["arguments"])])
    elif reason == "slot_identity":
        other = {**item, "id": "other-private-item", "call_id": "other-private-call"}
        events.extend(
            [
                {"type": "response.output_item.added", "output_index": 1, "item": other},
                {"type": stage, "output_index": 0, "item": other},
            ]
        )
    elif reason == "added_argument_prefix":
        added["arguments"] = '{"path":"other'
        events.append({"type": stage, "output_index": 0, "item": item})
    elif reason == "added_arguments_changed":
        added["arguments"] = '{"path":"other'
        events.append({"type": stage, "output_index": 0, "item": item})
    elif reason == "done_arguments_type":
        events.append(done({"path": "private-fixture.txt"}))
    elif reason == "delta_after_final":
        events.extend([done(item["arguments"]), delta(" ")])
    elif reason == "duplicate_terminal_call":
        events.append({"type": stage, "response": {"output": [item, item]}})
    elif reason == "legacy_argument_prefix":
        added["arguments"] = '{"path":"other'
        events.extend([delta(item["arguments"]), {"type": stage, "response": {}}])

    async def handler(request):
        return httpx.Response(
            200,
            content="".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatClient("https://synthetic.invalid/v1", http_client=http)
        request = ChatRequest(
            model=ModelSpec(provider="openai", model_id="synthetic", supports_responses_api=True)
        )
        with pytest.raises(ProviderError) as caught:
            _ = [
                event
                async for event in client._stream_responses(request, None, None, codex=False)
            ]
    error = caught.value
    assert error.message == f"responses_tool_arguments_conflict (reason={reason}, event={stage})"
    assert error.fail_closed is True
    assert error.retryable is False
    assert "private" not in str(error)
    assert "private" not in caplog.text
