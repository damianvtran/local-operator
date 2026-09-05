"""Wire-contract tests for durable native reasoning and visual tool results.

No live models: replay exact provider fixtures through the actual SSE parser,
persist the normalized Message through JSON, then inspect the next wire body.
"""

import json
from functools import partial
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
)
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.clients import (
    CODEX_RESPONSES_URL,
    GoogleClient,
    OpenAICompatClient,
)
from local_operator.providers.replay import credential_scope


def sse(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


async def message_from_stream(events):
    text = "".join(event.delta for event in events if isinstance(event, StreamTextDelta))
    calls = []
    for event in events:
        if isinstance(event, StreamToolCallDelta) and event.name:
            calls.append(ToolCall(id=event.id or "", name=event.name, arguments={"path": "a"}))
    terminal = next(event for event in reversed(events) if isinstance(event, StreamEndEvent))
    message = Message.assistant(
        text,
        tool_calls=calls,
        stop_reason=terminal.stop_reason,
        provider_payload=terminal.provider_payload,
    )
    # This is the transcript persistence/resume representation, not a retained
    # response object that accidentally keeps metadata only in memory.
    return Message.model_validate_json(message.model_dump_json())


@pytest.mark.asyncio
@pytest.mark.parametrize("codex", [False, True])
@pytest.mark.parametrize("terminal_output", [False, True])
async def test_responses_reasoning_survives_resume_order_and_route_switch(codex, terminal_output):
    reasoning = {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "opaque-fixture",
    }
    call = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "read",
        "arguments": '{"path": "a"}',
        "status": "completed",
    }
    output = [reasoning, call]
    events = [
        {"type": "response.output_item.added", "output_index": 0, "item": reasoning},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
        {"type": "response.output_item.added", "output_index": 1, "item": call},
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": call["arguments"],
        },
        {"type": "response.output_item.done", "output_index": 1, "item": call},
        {
            "type": "response.completed",
            "response": {"id": "resp_1", **({"output": output} if terminal_output else {})},
        },
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=sse(events)))
    ) as http:
        client = OpenAICompatClient(
            "https://api.openai.com/v1", http_client=http, openai_api="responses"
        )
        spec = ModelSpec(
            provider="openai",
            model_id="gpt-5",
            supports_responses_api=True,
            reasoning_efforts=("low", "high"),
            reasoning_effort="high",
        )
        request = ChatRequest(model=spec, messages=[Message.user("read it")])
        access = (
            OAuthAccess(access_token="fixture", credential_id=1, org_id="org", kind="oauth")
            if codex
            else None
        )
        result = [event async for event in client.stream(request, "fixture", access)]
        message = await message_from_stream(result)
        assert message.text == ""
        request.messages.extend(
            [
                message,
                Message(
                    role="tool",
                    tool_call_id="call_1",
                    tool_name="read",
                    content=[TextContent(text="result")],
                ),
            ]
        )
        scope = credential_scope("fixture", access)
        build = partial(
            client._build_codex_responses_body if codex else client._build_responses_body,
            scope=scope,
        )
        body = build(request)
        assert body["input"][1:3] == output
        assert body["input"][3]["type"] == "function_call_output"
        # A route/model switch must never forward another model's opaque state.
        changed = request.model_copy(
            update={"model": spec.model_copy(update={"model_id": "other"})}
        )
        assert not any(item.get("type") == "reasoning" for item in build(changed)["input"])
        other_endpoint = (
            "https://other.invalid/responses"
            if not codex
            else "https://api.openai.com/v1/responses"
        )
        assert not any(
            item.get("type") == "reasoning"
            for item in client._build_responses_body(request, endpoint=other_endpoint, scope=scope)[
                "input"
            ]
        )
        # Returning to the original route can safely replay retained state.
        assert build(request)["input"][1] == reasoning
        assert not any(
            item.get("type") == "reasoning"
            for item in build(request, scope="different-account")["input"]
        )
        message.tool_calls[0].arguments["path"] = "edited"
        assert not any(item.get("type") == "reasoning" for item in build(request)["input"])
        assert message.provider_payload is not None
        assert message.provider_payload["native_replay"]["endpoint"] == (
            CODEX_RESPONSES_URL if codex else "https://api.openai.com/v1/responses"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("native_id", [None, "fc_native_server_id"])
async def test_google_preserves_signature_parts_and_hides_thought_prose(native_id):
    parts: list[dict[str, Any]] = [
        {"text": "private summary", "thought": True},
        {"functionCall": {"name": "read", "args": {"path": "a"}}, "thoughtSignature": "sig-a"},
        {"text": "", "thoughtSignature": "sig-final"},
    ]
    if native_id is not None:
        parts[1]["functionCall"]["id"] = native_id
    events: list[dict[str, Any]] = [
        {"candidates": [{"content": {"parts": [part]}}]} for part in parts
    ]
    events.append({"candidates": [{"finishReason": "STOP"}]})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=sse(events)))
    ) as http:
        client = GoogleClient(http_client=http)
        spec = ModelSpec(provider="google", model_id="gemini-3-pro")
        request = ChatRequest(model=spec, messages=[Message.user("inspect")])
        message = await message_from_stream(
            [event async for event in client.stream(request, "fixture")]
        )
        assert not message.text
        request.messages.append(message)
        request.messages.append(
            Message(
                role="tool",
                tool_call_id=message.tool_calls[0].id,
                tool_name="read",
                content=[TextContent(text="done")],
            )
        )
        response = client._build_body(request, scope=credential_scope("fixture"))["contents"][2][
            "parts"
        ][0]["functionResponse"]
        if native_id is None:
            assert "id" not in response
        else:
            assert response["id"] == native_id
        assert (
            client._build_body(request, scope=credential_scope("fixture"))["contents"][1]["parts"]
            == parts
        )
        changed = request.model_copy(
            update={"model": spec.model_copy(update={"model_id": "gemini-2.5-pro"})}
        )
        assert "thoughtSignature" not in json.dumps(
            client._build_body(changed, scope=credential_scope("fixture"))
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini-3-pro", "gemini-2.5-pro"])
async def test_google_tool_images_reach_model_as_bytes(model):
    async with httpx.AsyncClient() as http:
        client = GoogleClient(http_client=http)
        message = Message(
            role="tool",
            tool_call_id="native_id",
            tool_name="screenshot",
            content=[TextContent(text="screenshot")],
        )
        message.content.append(ImageContent(data="aW1hZ2U=", mime_type="image/png"))
        body = client._build_body(
            ChatRequest(
                model=ModelSpec(provider="google", model_id=model),
                messages=[
                    Message.assistant(
                        "", tool_calls=[ToolCall(id="native_id", name="screenshot", arguments={})]
                    ),
                    message,
                ],
            )
        )
        parts = body["contents"][-1]["parts"]
        response = parts[0]["functionResponse"]
        image = response["parts"][0] if model.startswith("gemini-3") else parts[1]
        assert image == {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}
        assert response["id"] == "native_id"


@pytest.mark.asyncio
async def test_google_imported_parallel_calls_are_one_paired_result_turn():
    async with httpx.AsyncClient() as http:
        client = GoogleClient(http_client=http)
        calls = [ToolCall(id=f"call-{index}", name="read", arguments={}) for index in range(2)]
        messages = [Message.user("read both"), Message.assistant("", tool_calls=calls)]
        messages.extend(
            Message(
                role="tool",
                tool_call_id=call.id,
                tool_name=call.name,
                content=[TextContent(text="output")],
            )
            for call in calls
        )
        body = client._build_body(
            ChatRequest(
                model=ModelSpec(provider="google", model_id="gemini-3-pro"),
                messages=messages,
            )
        )
        assert len(body["contents"]) == 3
        assert len(body["contents"][-1]["parts"]) == 2
        assert [part["functionCall"]["id"] for part in body["contents"][1]["parts"]] == [
            part["functionResponse"]["id"] for part in body["contents"][2]["parts"]
        ]
        assert (
            body["contents"][1]["parts"][0]["thoughtSignature"]
            == "skip_thought_signature_validator"
        )
