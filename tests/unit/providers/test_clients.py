"""Wire client tests against httpx.MockTransport SSE fixtures. No network."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
)
from local_operator.providers.clients import (
    AnthropicClient,
    MockClient,
    OpenAICompatClient,
    _anthropic_stream_error,
    client_for_spec,
    raise_for_status,
)
from local_operator.providers.failover import ProviderError

pytestmark = pytest.mark.asyncio


def _sse(payloads: Sequence[dict[str, Any] | str]) -> bytes:
    """Render an SSE body (dicts as data JSON, strings verbatim)."""
    lines = []
    for payload in payloads:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        lines.append(f"data: {data}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _spec(provider: str = "openai", model_id: str = "gpt-4o") -> ModelSpec:
    return ModelSpec(provider=provider, model_id=model_id)


async def _collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


def _openai_sse_with_tool_call() -> bytes:
    return _sse(
        [
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": "Hello"}, "index": 0}]},
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_abc", "function": {"name": "get_weather"}}
                            ]
                        },
                        "index": 0,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]
                        },
                        "index": 0,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": ' "Paris"}'}}]
                        },
                        "index": 0,
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 12},
                },
            },
        ]
    )


async def test_openai_compat_text_tool_usage() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_openai_sse_with_tool_call(), headers={"content-type": "text/event-stream"}
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        "https://api.test.example/v1", http_client=httpx.AsyncClient(transport=transport)
    )
    request = ChatRequest(
        model=_spec(),
        system_blocks=["be brief"],
        messages=[Message.user("weather in paris?")],
    )
    events = await _collect(client.stream(request, "sk-test"))

    assert captured["url"] == "https://api.test.example/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["stream"] is True
    assert captured["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert captured["body"]["messages"][1]["content"] == "weather in paris?"

    texts = [e.delta for e in events if isinstance(e, StreamTextDelta)]
    assert texts == ["Hello"]

    tool_events = [e for e in events if isinstance(e, StreamToolCallDelta)]
    assert any(e.id == "call_abc" for e in tool_events)
    assert any(e.name == "get_weather" for e in tool_events)
    assembled = "".join(e.argument_delta for e in tool_events if e.argument_delta)
    assert json.loads(assembled) == {"city": "Paris"}

    usage_events = [e for e in events if isinstance(e, StreamUsageEvent)]
    assert usage_events and usage_events[0].usage.input_tokens == 40
    assert usage_events[0].usage.cache_read_tokens == 12  # prompt_tokens_details.cached_tokens

    end = events[-1]
    assert isinstance(end, StreamEndEvent)
    assert end.stop_reason == "toolUse"
    assert end.usage is not None and end.usage.output_tokens == 7


async def test_openai_compat_tool_history_roundtrip() -> None:
    """Assistant tool_calls and tool results serialize for replay.

    Tool results are rendered from content blocks: plain text stays a string,
    image-only results become multipart data-URL parts, and empty results are
    backfilled (HC-03: never re-flatten via ``message.text``).
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_sse([{"choices": [{"delta": {}, "finish_reason": "stop"}]}])
        )

    client = OpenAICompatClient(
        "https://api.test.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="call_1", name="echo", arguments={"x": 1})],
    )
    tool_result = Message(
        role="tool", tool_call_id="call_1", tool_name="echo", content=[TextContent(text="done")]
    )
    image_result = Message(
        role="tool",
        tool_call_id="call_2",
        tool_name="screenshot",
        content=[ImageContent(data="aW1n", mime_type="image/png")],
    )
    empty_result = Message(role="tool", tool_call_id="call_3", tool_name="noop", content=[])
    await _collect(
        client.stream(
            ChatRequest(
                model=_spec(), messages=[assistant, tool_result, image_result, empty_result]
            ),
            None,
        )
    )
    messages = captured["body"]["messages"]
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"] == {
        "name": "echo",
        "arguments": json.dumps({"x": 1}),
    }
    assert messages[1] == {"role": "tool", "tool_call_id": "call_1", "content": "done"}
    assert messages[2]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}}
    ]
    assert messages[3]["content"] == "[tool returned no output]"


@pytest.mark.parametrize(
    "status, auth_error, retryable",
    [
        (401, True, False),
        (403, True, False),
        (429, False, True),
        (500, False, True),
        (400, False, False),
    ],
)
async def test_openai_compat_error_mapping(status: int, auth_error: bool, retryable: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"message": "nope"}},
            headers={"retry-after": "2"} if status == 429 else {},
        )

    client = OpenAICompatClient(
        "https://api.test.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError) as excinfo:
        await _collect(client.stream(ChatRequest(model=_spec()), "k"))
    assert excinfo.value.status == status
    assert excinfo.value.auth_error is auth_error
    assert excinfo.value.retryable is retryable
    assert "nope" in str(excinfo.value)
    if status == 429:
        assert excinfo.value.retry_after_ms == 2000


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _anthropic_sse() -> bytes:
    events = [
        (
            "message_start",
            {"message": {"usage": {"input_tokens": 30, "cache_read_input_tokens": 5}}},
        ),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hi "}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "there"}}),
        ("content_block_stop", {"index": 0}),
        (
            "content_block_start",
            {"index": 1, "content_block": {"type": "tool_use", "id": "tu_1", "name": "bash"}},
        ),
        (
            "content_block_delta",
            {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"cmd"'}},
        ),
        (
            "content_block_delta",
            {"index": 1, "delta": {"type": "input_json_delta", "partial_json": ': "ls"}'}},
        ),
        ("content_block_stop", {"index": 1}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 11}}),
        ("message_stop", {}),
    ]
    body = ""
    for name, payload in events:
        body += f"event: {name}\ndata: {json.dumps({'type': name, **payload})}\n\n"
    return body.encode()


async def test_anthropic_stream_and_cache_headers() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_anthropic_sse(), headers={"content-type": "text/event-stream"}
        )

    client = AnthropicClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    request = ChatRequest(
        model=_spec("anthropic", "claude-3-5-sonnet-latest"),
        system_blocks=[
            "stable instructions",
            "tools inventory",
            "skills (volatile)",
            "env/date (volatile)",
        ],
        messages=[Message.user("hi")],
    )
    events = await _collect(client.stream(request, "sk-ant-test"))

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"

    # Cache breakpoints on the STABLE head only: the last two blocks
    # (skills + env/date) are volatile and stay breakpoint-free.
    system = captured["body"]["system"]
    assert len(system) == 4
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system[2]
    assert "cache_control" not in system[3]

    texts = [e.delta for e in events if isinstance(e, StreamTextDelta)]
    assert texts == ["Hi ", "there"]

    tool_events = [e for e in events if isinstance(e, StreamToolCallDelta)]
    assert tool_events[0].id == "tu_1" and tool_events[0].name == "bash"
    arguments = "".join(e.argument_delta for e in tool_events if e.argument_delta)
    assert json.loads(arguments) == {"cmd": "ls"}

    end = events[-1]
    assert isinstance(end, StreamEndEvent)
    assert end.stop_reason == "toolUse"
    assert end.usage is not None
    assert end.usage.input_tokens == 30
    assert end.usage.cache_read_tokens == 5
    assert end.usage.output_tokens == 11


async def test_anthropic_tool_result_content_blocks() -> None:
    """Tool results render from content blocks; empty results are backfilled
    (HC-03: Anthropic 400s on empty tool_result content)."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_anthropic_sse(), headers={"content-type": "text/event-stream"}
        )

    client = AnthropicClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    text_result = Message(
        role="tool", tool_call_id="c1", content=[TextContent(text="done")], is_error=False
    )
    image_result = Message(
        role="tool", tool_call_id="c2", content=[ImageContent(data="aW1n", mime_type="image/png")]
    )
    empty_result = Message(role="tool", tool_call_id="c3", content=[], is_error=True)
    assistant = Message(role="assistant", tool_calls=[ToolCall(id="c1", name="t", arguments={})])
    await _collect(
        client.stream(
            ChatRequest(
                model=_spec("anthropic", "claude"),
                messages=[assistant, text_result, image_result, empty_result],
            ),
            "sk",
        )
    )
    # All tool results land in one grouped user message after the assistant.
    messages = captured["body"]["messages"]
    grouped = messages[-1]
    assert grouped["role"] == "user"
    content = grouped["content"]
    by_id = {block["tool_use_id"]: block for block in content if block["type"] == "tool_result"}
    assert by_id["c1"]["content"] == [{"type": "text", "text": "done"}]
    assert by_id["c2"]["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1n"}}
    ]
    # Empty error result: backfilled, is_error preserved.
    assert by_id["c3"]["content"] == [{"type": "text", "text": "[tool returned no output]"}]
    assert by_id["c3"]["is_error"] is True


async def test_anthropic_error_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

    client = AnthropicClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError) as excinfo:
        await _collect(client.stream(ChatRequest(model=_spec("anthropic", "claude")), "bad"))
    assert excinfo.value.status == 401
    assert excinfo.value.auth_error is True


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


async def test_mock_client_text_is_deterministic() -> None:
    client = MockClient()
    first = await _collect(client.stream(ChatRequest(model=_spec("test", "mock")), None))
    second = await _collect(client.stream(ChatRequest(model=_spec("test", "mock")), None))
    assert first == second
    assert [e.delta for e in first if isinstance(e, StreamTextDelta)] == [
        "Hello",
        " from the mock provider!",
    ]
    assert first[-1].stop_reason == "stop"


async def test_mock_client_tool_branch() -> None:
    client = MockClient()
    request = ChatRequest(model=_spec("test", "mock"), messages=[Message.user("please [tool] now")])
    events = await _collect(client.stream(request, None))
    tool_events = [e for e in events if isinstance(e, StreamToolCallDelta)]
    assert tool_events[0].name == "echo"
    arguments = "".join(e.argument_delta for e in tool_events if e.argument_delta)
    assert json.loads(arguments) == {"text": "hi"}
    assert events[-1].stop_reason == "toolUse"


# ---------------------------------------------------------------------------
# OAuth inference routing (PR-01 blockers)
# ---------------------------------------------------------------------------


from local_operator.providers.auth_store import OAuthAccess  # noqa: E402
from local_operator.providers.clients import GoogleClient  # noqa: E402


async def test_anthropic_oauth_sends_bearer_and_beta_header() -> None:
    """Anthropic OAuth: Authorization: Bearer + anthropic-beta oauth header,
    NOT x-api-key (which 401s OAuth-issued tokens)."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200, content=_anthropic_sse(), headers={"content-type": "text/event-stream"}
        )

    client = AnthropicClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    access = OAuthAccess(
        access_token="oauth-token-1", credential_id=1, org_id="org-1", kind="oauth"
    )
    await _collect(
        client.stream(
            ChatRequest(model=_spec("anthropic", "claude"), messages=[Message.user("hi")]),
            "oauth-token-1",
            oauth_access=access,
        )
    )
    assert captured["headers"]["authorization"] == "Bearer oauth-token-1"
    assert captured["headers"]["anthropic-beta"] == "oauth-2025-04-20"
    assert "x-api-key" not in captured["headers"]


async def test_anthropic_api_key_still_uses_x_api_key() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200, content=_anthropic_sse(), headers={"content-type": "text/event-stream"}
        )

    client = AnthropicClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await _collect(
        client.stream(
            ChatRequest(model=_spec("anthropic", "claude"), messages=[Message.user("hi")]),
            "sk-ant-1",
        )
    )
    assert captured["headers"]["x-api-key"] == "sk-ant-1"
    assert "authorization" not in captured["headers"]
    assert "anthropic-beta" not in captured["headers"]


def _responses_sse() -> bytes:
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "m1"},
        },
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.output_text.delta", "delta": " ChatGPT"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 3},
                },
            },
        },
    ]
    return _sse(events)


def _public_responses_sse() -> bytes:
    return _sse(
        [
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_weather",
                    "name": "get_weather",
                },
            },
            {"type": "response.output_text.delta", "delta": "Checking"},
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 1,
                "delta": '{"city":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 1,
                "delta": '"Paris"}',
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_public_1",
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 9,
                        "input_tokens_details": {"cached_tokens": 48},
                    },
                },
            },
        ]
    )


async def test_openai_api_key_gpt5_uses_public_responses_end_to_end() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_public_responses_sse(),
            headers={"content-type": "text/event-stream"},
        )

    async def unused_execute(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("wire serialization must not execute tools")

    spec = ModelSpec(
        provider="openai",
        model_id="gpt-5.4",
        supports_responses_api=True,
        supports_prompt_cache=True,
        reasoning=True,
        reasoning_effort="high",
        reasoning_efforts=("low", "medium", "high"),
        supports_sampling_params=False,
    )
    client = OpenAICompatClient(
        base_url="https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    events = await _collect(
        client.stream(
            ChatRequest(
                model=spec,
                system_blocks=["Be concise."],
                messages=[Message.user("weather in Paris?")],
                tools=[
                    AgentTool(
                        name="get_weather",
                        description="Get weather",
                        parameters={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        execute=unused_execute,
                    )
                ],
                prompt_cache_key="session-123",
            ),
            "sk-public",
        )
    )

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["authorization"] == "Bearer sk-public"
    for private_header in ("chatgpt-account-id", "openai-beta", "originator"):
        assert private_header not in captured["headers"]
    body = captured["body"]
    assert body["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "weather in Paris?"}],
        }
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    assert body["reasoning"] == {"effort": "high"}
    assert body["prompt_cache_key"] == "session-123"
    assert body["prompt_cache_retention"] == "24h"

    assert [event.delta for event in events if isinstance(event, StreamTextDelta)] == ["Checking"]
    tool_events = [event for event in events if isinstance(event, StreamToolCallDelta)]
    assert tool_events[0].id == "call_weather" and tool_events[0].name == "get_weather"
    assert json.loads("".join(event.argument_delta for event in tool_events)) == {"city": "Paris"}
    usage = [event.usage for event in events if isinstance(event, StreamUsageEvent)][0]
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (80, 9, 48)
    assert events[-1].stop_reason == "toolUse"


@pytest.mark.parametrize("provider", ["openrouter", "deepseek"])
async def test_compat_providers_stay_on_chat_completions_for_responses_model(
    provider: str,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            content=_sse([{"choices": [{"delta": {}, "finish_reason": "stop"}]}]),
            headers={"content-type": "text/event-stream"},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    spec = ModelSpec(
        provider=provider,
        model_id="openai/gpt-5.4",
        supports_responses_api=True,
    )
    client = client_for_spec(spec, http_client=http, openai_api="responses")
    await _collect(client.stream(ChatRequest(model=spec, messages=[Message.user("hi")]), "key"))
    assert captured["url"].endswith("/chat/completions")
    await http.aclose()


async def test_openai_oauth_routes_to_codex_responses_with_required_headers() -> None:
    """ChatGPT OAuth uses the Codex endpoint rather than the public API."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=_responses_sse(), headers={"content-type": "text/event-stream"}
        )

    client = OpenAICompatClient(
        base_url="https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    access = OAuthAccess(
        access_token="chatgpt-token", credential_id=2, org_id="acct-42", kind="oauth"
    )
    events = await _collect(
        client.stream(
            ChatRequest(model=_spec("openai", "gpt-5"), messages=[Message.user("hi")]),
            "chatgpt-token",
            oauth_access=access,
        )
    )
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["headers"]["chatgpt-account-id"] == "acct-42"
    assert captured["headers"]["authorization"] == "Bearer chatgpt-token"
    assert captured["headers"]["openai-beta"] == "responses=experimental"
    assert captured["headers"]["originator"] == "local-operator"
    assert captured["body"]["store"] is False
    assert "input" in captured["body"] and "messages" not in captured["body"]
    assert "max_output_tokens" not in captured["body"]
    texts = [e.delta for e in events if isinstance(e, StreamTextDelta)]
    assert texts == ["Hello", " ChatGPT"]
    usage = events[-1].usage
    assert usage.input_tokens == 12 and usage.output_tokens == 4 and usage.cache_read_tokens == 3


async def test_openai_api_key_stays_on_completions() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content=_sse([{"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}]),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenAICompatClient(
        base_url="https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await _collect(
        client.stream(
            ChatRequest(model=_spec("openai", "gpt-4o"), messages=[Message.user("hi")]), "sk-test"
        )
    )
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert "chatgpt-account-id" not in captured["headers"]


async def test_google_system_instruction_and_tool_result_blocks() -> None:
    """PR-17: system blocks use systemInstruction; tool results render from
    content blocks with empty-backfill."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=(
                b'data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}, '
                b'"finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 3, '
                b'"candidatesTokenCount": 1}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = GoogleClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assistant = Message(role="assistant", tool_calls=[ToolCall(id="c1", name="t", arguments={})])
    result = Message(
        role="tool", tool_call_id="c1", tool_name="t", content=[TextContent(text="tool says hi")]
    )
    empty_result = Message(role="tool", tool_call_id="c2", tool_name="u", content=[])
    request = ChatRequest(
        model=_spec("google", "gemini-2.5-pro"),
        system_blocks=["instructions"],
        messages=[assistant, result, empty_result],
    )
    events = await _collect(client.stream(request, "goog-key"))
    body = captured["body"]
    assert body["systemInstruction"]["parts"] == [{"text": "instructions"}]
    # No system block folded into contents.
    assert all(c["parts"] != [{"text": "instructions"}] for c in body["contents"])
    responses = [
        p["functionResponse"]
        for c in body["contents"]
        for p in c["parts"]
        if "functionResponse" in p
    ]
    by_name = {r["name"]: r["response"]["content"] for r in responses}
    assert by_name["t"] == "tool says hi"
    assert by_name["u"] == "[tool returned no output]"
    assert events[-1].stop_reason == "stop"


def test_anthropic_cache_breakpoints_capped_at_four() -> None:
    """PR-18: many system blocks must not exceed the 4-marker cap; the
    stable head stays breakpointed, nothing beyond the cap."""
    blocks = [f"block-{i}" for i in range(9)]
    rendered = AnthropicClient._system_blocks(blocks)
    marked = [b for b in rendered if "cache_control" in b]
    assert len(marked) == 4  # capped at MAX_CACHE_BREAKPOINTS
    assert all(rendered[i]["cache_control"] == {"type": "ephemeral"} for i in range(4))
    assert all("cache_control" not in rendered[i] for i in range(4, 9))


def test_retry_after_http_date_parsed() -> None:
    """PR-22: Retry-After in HTTP-date form is converted to milliseconds."""
    import email.utils
    import time as _time

    from local_operator.providers.clients import _parse_retry_after

    future = _time.time() + 10
    header = email.utils.formatdate(future, usegmt=True)
    response = httpx.Response(429, headers={"retry-after": header})
    parsed = _parse_retry_after(response)
    assert parsed is not None and 8000 <= parsed <= 12000


# ---------------------------------------------------------------------------
# Error fidelity — what the provider actually said, and how fast it clears
# ---------------------------------------------------------------------------


class TestErrorMessageExtraction:
    """The reported frame was ``✕ HTTP 404:`` — a wrong status and an EMPTY
    message. This class covers the empty half: every body shape a provider
    packs its explanation into has to reach the frame, because that text is
    what says which limit was hit and when it resets.
    """

    def _error(self, response: httpx.Response) -> ProviderError:
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(response)
        return excinfo.value

    def test_openai_style_error_message(self) -> None:
        error = self._error(
            httpx.Response(429, json={"error": {"message": "You exceeded your quota."}})
        )
        assert error.message == "You exceeded your quota."

    def test_google_wraps_its_error_in_a_single_element_list(self) -> None:
        """``streamGenerateContent`` answers a pre-stream failure with
        ``[{"error": ...}]``. A mapping-only extractor read straight past the
        list to ``response.text``, which is how a real quota error arrived with
        its message replaced by raw JSON — or, once the body was empty, by
        nothing at all."""
        error = self._error(
            httpx.Response(
                429,
                json=[
                    {
                        "error": {
                            "code": 429,
                            "message": "Resource has been exhausted (e.g. check quota).",
                            "status": "RESOURCE_EXHAUSTED",
                        }
                    }
                ],
            )
        )
        assert error.message == "Resource has been exhausted (e.g. check quota)."
        assert error.kind == "quota"

    def test_blank_message_key_falls_through_instead_of_winning(self) -> None:
        """``error.get("message", error)`` treated a present-but-EMPTY key as an
        answer, so a provider that sent the field blank produced an error that
        printed nothing."""
        error = self._error(
            httpx.Response(400, json={"error": {"message": "", "detail": "model_not_supported"}})
        )
        assert error.message == "model_not_supported"

    def test_bare_error_string_and_detail_shapes(self) -> None:
        assert self._error(httpx.Response(500, json={"error": "server overloaded"})).message == (
            "server overloaded"
        )
        assert self._error(httpx.Response(422, json={"detail": "input too long"})).message == (
            "input too long"
        )

    def test_openrouter_upstream_text_is_recovered_from_metadata_raw(self) -> None:
        """OpenRouter's ``message`` is the useless half ("Provider returned
        error"); the ORIGIN provider's real text is JSON-encoded one level down
        in ``metadata.raw``. Both are kept because the raw part is the one that
        names the limit."""
        error = self._error(
            httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Provider returned error",
                        "code": 429,
                        "metadata": {
                            "provider_name": "Google",
                            "raw": json.dumps(
                                {"error": {"message": "Quota exceeded for quota metric 'Requests'"}}
                            ),
                        },
                    }
                },
            )
        )
        assert error.message == (
            "Provider returned error: Quota exceeded for quota metric 'Requests'"
        )
        assert error.kind == "quota"

    def test_empty_body_still_says_something(self) -> None:
        """The reported frame, reproduced: a gateway rejecting an unknown model
        answers 404 with no body at all. ``ProviderError`` refuses to be
        wordless, so the floor is the status's own meaning."""
        error = self._error(httpx.Response(404, content=b""))
        assert error.message == "Not Found — the provider sent no error message"
        assert str(error) == (
            "invalid request (HTTP 404): Not Found — the provider sent no error message"
        )

    def test_non_json_body_is_carried_through(self) -> None:
        error = self._error(httpx.Response(502, content=b"<html>bad gateway</html>"))
        assert error.message == "<html>bad gateway</html>"
        assert error.kind == "transient"


class TestRetryAfter:
    """ "try again in 40s" is the single most actionable fact in a rate-limit
    error, and providers disagree about where to put it."""

    def test_header_seconds(self) -> None:
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(
                httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "42"})
            )
        assert excinfo.value.retry_after_ms == 42_000
        assert "retry in 42s" in str(excinfo.value)

    def test_google_puts_the_delay_in_the_body_and_sends_no_header(self) -> None:
        """Gemini quota 429s carry ``error.details[].retryDelay`` ("41s") and NO
        ``Retry-After``, so the one number the user needs was being dropped."""
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(
                httpx.Response(
                    429,
                    json={
                        "error": {
                            "message": "Resource has been exhausted.",
                            "details": [
                                {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
                                {
                                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                    "retryDelay": "41.6s",
                                },
                            ],
                        }
                    },
                )
            )
        assert excinfo.value.retry_after_ms == 41_600
        assert "retry in 42s" in str(excinfo.value)

    def test_no_delay_anywhere_is_not_invented(self) -> None:
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(httpx.Response(429, json={"error": "slow down"}))
        assert excinfo.value.retry_after_ms is None
        assert "retry in" not in str(excinfo.value)


class TestStatusFlags:
    """``retryable`` and ``auth_error`` are what the failover driver acts on, so
    the mapping from status is pinned rather than inferred from behaviour."""

    @pytest.mark.parametrize(
        ("status", "retryable", "kind"),
        [
            (400, False, "request"),
            (401, False, "auth"),
            (403, False, "auth"),
            (404, False, "request"),
            (408, True, "timeout"),
            (429, True, "quota"),
            (500, True, "transient"),
            (503, True, "transient"),
            (504, True, "timeout"),
            (529, True, "transient"),
        ],
    )
    def test_status_maps_to_retryability_and_kind(
        self, status: int, retryable: bool, kind: str
    ) -> None:
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(httpx.Response(status, json={"error": "boom"}))
        assert excinfo.value.retryable is retryable
        assert excinfo.value.kind == kind


class TestAnthropicStreamErrorEvent:
    """A mid-stream ``error`` event arrives on an HTTP 200, so its status has to
    come from anthropic's ``type``. Blanket ``retryable=True`` re-sent requests
    the API had already refused."""

    def test_overloaded_is_transient(self) -> None:
        error = _anthropic_stream_error({"type": "overloaded_error", "message": "Overloaded"})
        assert (error.kind, error.retryable) == ("transient", True)
        assert error.message == "overloaded_error: Overloaded"

    def test_rate_limit_is_quota(self) -> None:
        error = _anthropic_stream_error(
            {"type": "rate_limit_error", "message": "Number of request tokens exceeded"}
        )
        assert (error.kind, error.retryable) == ("quota", True)

    def test_invalid_request_is_not_retried(self) -> None:
        error = _anthropic_stream_error(
            {"type": "invalid_request_error", "message": "max_tokens too large"}
        )
        assert (error.kind, error.retryable, error.status) == ("request", False, 400)

    def test_authentication_error_is_auth(self) -> None:
        error = _anthropic_stream_error({"type": "authentication_error", "message": "bad key"})
        assert (error.kind, error.auth_error) == ("auth", True)

    def test_an_event_with_no_text_still_names_itself(self) -> None:
        error = _anthropic_stream_error({"type": "api_error"})
        assert error.message == "api_error"
        assert str(error) == "transient provider error (HTTP 500): api_error"

    def test_an_empty_event_is_not_a_silent_error(self) -> None:
        error = _anthropic_stream_error({})
        assert error.message == "the provider failed without reporting a reason"
        assert error.retryable is True  # unknown type keeps the old assumption


class TestNoPythonReprReachesTheFrame:
    """``str(error)`` stood in the cascade's last slot and put a dict repr in the
    user's error frame — for the very shape the docstring claimed to fix."""

    def _error(self, response: httpx.Response) -> ProviderError:
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(response)
        return excinfo.value

    def test_a_blank_message_with_no_sibling_falls_to_the_wire_body(self) -> None:
        """The docstring's own example, which used to render
        `invalid request (HTTP 404): {'message': ''}` — a PYTHON repr of a parsed
        object. The fallback is now the body as the provider actually sent it:
        valid JSON, capped, and honest that the message field came through blank.
        A body that is genuinely empty still reaches ``_describe_bare_status`` —
        that is the owner's reported case, covered above."""
        error = self._error(httpx.Response(404, json={"error": {"message": ""}}))
        assert error.message == '{"error":{"message":""}}'
        assert "'message'" not in str(error), "a Python repr must never reach the frame"

    def test_a_non_string_message_falls_to_the_raw_body_not_a_repr(self) -> None:
        """This shape was strictly WORSE than before the cascade: `{'text': 'x'}`
        became `{'message': {'text': 'x'}}`. The real wire bytes beat both."""
        error = self._error(httpx.Response(400, json={"error": {"message": {"text": "nested"}}}))
        assert "'message':" not in error.message
        assert "nested" in error.message  # the body itself, JSON as sent

    def test_every_branch_is_length_capped(self) -> None:
        """The repr branch was the only uncapped one, so a 3 KB error object went
        into a one-line terminal notice whole."""
        from local_operator.providers.clients import MAX_ERROR_MESSAGE_CHARS

        for payload in (
            {"error": {"message": "x" * 4000}},
            {"error": {"code": "y" * 4000}},
            {"detail": "z" * 4000},
            {"error": {"blob": "q" * 4000}},
        ):
            error = self._error(httpx.Response(400, json=payload))
            assert len(error.message) <= MAX_ERROR_MESSAGE_CHARS, payload
        assert len(self._error(httpx.Response(502, content=b"w" * 4000)).message) <= (
            MAX_ERROR_MESSAGE_CHARS
        )


class TestAnAdvertisedWaitIsBounded:
    """``retry_after_ms`` is provider-supplied and reaches SQLite: a usage-limit
    failure feeds it to ``AuthStore.block_credential``, which floors the value
    and has no ceiling. One ``retryDelay: "99999999s"`` wrote a 27,777-hour block
    against a working credential and printed ``retry in 27777h46m``."""

    def _delay(self, response: httpx.Response) -> int | None:
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(response)
        return excinfo.value.retry_after_ms

    def test_an_absurd_body_delay_is_clamped(self) -> None:
        from local_operator.providers.clients import MAX_RETRY_AFTER_MS

        payload = {
            "error": {
                "message": "exhausted",
                "details": [{"@type": "RetryInfo", "retryDelay": "99999999s"}],
            }
        }
        assert self._delay(httpx.Response(429, json=payload)) == MAX_RETRY_AFTER_MS

    def test_an_absurd_header_delay_is_clamped(self) -> None:
        from local_operator.providers.clients import MAX_RETRY_AFTER_MS

        response = httpx.Response(429, json={"error": "slow"}, headers={"retry-after": "99999999"})
        assert self._delay(response) == MAX_RETRY_AFTER_MS

    def test_an_overflowing_header_does_not_escape_as_overflowerror(self) -> None:
        """``float("1e400")`` is ``inf`` and ``int(inf * 1000)`` raises. It escaped
        ``raise_for_status`` entirely, and in ``ApiEmbedder`` it escaped the
        degrade-gracefully handlers too."""
        response = httpx.Response(429, json={"error": "slow"}, headers={"retry-after": "1e400"})
        assert self._delay(response) is None

    def test_a_zero_header_does_not_erase_the_bodys_real_delay(self) -> None:
        """Zero is not an answer to "how long": the frame drops the wait entirely
        and ``_same_credential_retry_allowed`` reads it as a short throttle and
        grants an immediate same-key retry of a quota error."""
        response = httpx.Response(
            429,
            json={
                "error": {
                    "message": "exhausted",
                    "details": [{"@type": "RetryInfo", "retryDelay": "41s"}],
                }
            },
            headers={"retry-after": "0"},
        )
        assert self._delay(response) == 41_000


class TestEveryDocumentedAnthropicErrorTypeIsMapped:
    """Two were missing, and an unmapped type gets ``status=None`` so
    ``retryable = status is None`` is True — a billing failure was re-sent
    ``max_retries`` times and read as a transient blip."""

    def test_billing_and_conflict_are_not_retried(self) -> None:
        billing = _anthropic_stream_error(
            {"type": "billing_error", "message": "Your credit balance is too low"}
        )
        assert (billing.status, billing.kind, billing.retryable) == (402, "quota", False)
        conflict = _anthropic_stream_error(
            {"type": "conflict_error", "message": "concurrent write"}
        )
        assert (conflict.status, conflict.kind, conflict.retryable) == (409, "request", False)

    def test_the_documented_set_is_complete(self) -> None:
        """Anthropic documents exactly these; an omission is silently a retry."""
        from local_operator.providers.clients import _ANTHROPIC_ERROR_STATUS

        assert set(_ANTHROPIC_ERROR_STATUS) == {
            "invalid_request_error",
            "authentication_error",
            "billing_error",
            "permission_error",
            "not_found_error",
            "conflict_error",
            "request_too_large",
            "rate_limit_error",
            "api_error",
            "timeout_error",
            "overloaded_error",
        }


def test_unknown_provider_raises_value_error() -> None:
    """PR-22: client_for_spec rejects unknown providers instead of silently
    defaulting to the ollama endpoint."""
    from local_operator.providers.clients import client_for_spec

    with pytest.raises(ValueError, match="Unknown provider"):
        client_for_spec(ModelSpec(provider="not-a-provider", model_id="x"))


async def test_openai_compat_context_tokens_is_prompt_total() -> None:
    """OpenAI-style ``prompt_tokens`` already includes cached blocks, so the
    context the provider read equals prompt_tokens. The compaction trigger
    prefers this over its local estimate, so a missing value silently degrades
    the trigger to estimation only."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=_openai_sse_with_tool_call(), headers={"content-type": "text/event-stream"}
        )
    )
    client = OpenAICompatClient(
        "https://api.test.example/v1", http_client=httpx.AsyncClient(transport=transport)
    )
    events = await _collect(
        client.stream(
            ChatRequest(model=_spec(), system_blocks=["be brief"], messages=[Message.user("hi")]),
            "sk-test",
        )
    )
    usage = [e.usage for e in events if isinstance(e, StreamUsageEvent)][-1]
    assert usage.input_tokens == 40
    assert usage.cache_read_tokens == 12
    assert usage.context_tokens == 40


async def test_anthropic_context_tokens_sums_uncached_and_cached() -> None:
    """Anthropic reports ``input_tokens`` EXCLUDING cached blocks, so context is
    input + cache_read + cache_write. Normalizing here keeps the compaction
    trigger and the TUI status line provider-agnostic."""
    body = _sse(
        [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 500,
                        "cache_read_input_tokens": 8_000,
                        "cache_creation_input_tokens": 1_000,
                    }
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0, "delta": {"text": "ok"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ]
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    client = AnthropicClient(http_client=httpx.AsyncClient(transport=transport))
    events = await _collect(
        client.stream(
            ChatRequest(model=_spec(provider="anthropic"), messages=[Message.user("hi")]), "sk-ant"
        )
    )
    end = [e for e in events if isinstance(e, StreamEndEvent)][-1]
    assert end.usage is not None
    assert end.usage.input_tokens == 500
    assert end.usage.cache_read_tokens == 8_000
    assert end.usage.context_tokens == 9_500


def test_anthropic_message_breakpoints_cover_the_conversation():
    """System-only cache_control stops the warm prefix before the first
    message; the conversation must carry markers too (the §A >=90% target is
    unreachable without them)."""
    from local_operator.providers.clients import AnthropicClient

    request = ChatRequest(
        model=_spec(provider="anthropic"),
        system_blocks=["instructions", "inventory", "skills", "env"],
        messages=[
            Message.user("first"),
            Message.assistant("mid"),
            Message.user("second"),
        ],
    )
    body = AnthropicClient()._build_body(request)
    system_markers = [i for i, e in enumerate(body["system"]) if "cache_control" in e]
    messages = body["messages"]
    last_block = messages[-1]["content"][-1]
    assert "cache_control" in last_block
    prev_user_block = messages[0]["content"][-1]
    assert "cache_control" in prev_user_block
    total = len(system_markers) + 2
    assert total <= AnthropicClient.MAX_CACHE_BREAKPOINTS


def test_openai_compat_markers_gate_on_cache_support():
    from local_operator.providers.clients import OpenAICompatClient

    spec = _spec()
    spec.supports_prompt_cache = True
    cached = OpenAICompatClient("https://x")._build_body(
        ChatRequest(model=spec, messages=[Message.user("a"), Message.user("b")])
    )
    assert cached["messages"][-1]["content"][-1].get("cache_control") == {"type": "ephemeral"}

    spec_nocache = _spec()
    spec_nocache.supports_prompt_cache = False
    plain = OpenAICompatClient("https://x")._build_body(
        ChatRequest(model=spec_nocache, messages=[Message.user("a"), Message.user("b")])
    )
    assert "cache_control" not in str(plain["messages"])


def test_reasoning_effort_reaches_openai_and_anthropic_wires() -> None:
    # The ladder is declared on both specs: `_reasoning_effort` refuses a level
    # the model does not accept, so a spec that names one but never lists it is
    # sent no key at all.
    openai_spec = ModelSpec(
        provider="openai",
        model_id="gpt-5.3-codex",
        reasoning=True,
        reasoning_effort="high",
        reasoning_efforts=("low", "medium", "high", "xhigh"),
        supports_sampling_params=False,
    )
    request = ChatRequest(model=openai_spec, messages=[Message.user("hi")])
    openai = OpenAICompatClient("https://api.openai.com/v1")
    assert openai._build_body(request)["reasoning_effort"] == "high"
    assert openai._build_responses_body(request)["reasoning"] == {"effort": "high"}

    anthropic_spec = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        reasoning=True,
        reasoning_effort="max",
        reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
        supports_sampling_params=False,
    )
    anthropic = AnthropicClient()
    body = anthropic._build_body(ChatRequest(model=anthropic_spec, messages=[Message.user("hi")]))
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "max"}
    assert "effort-2025-11-24" in anthropic._headers("key", effort="max")["anthropic-beta"]


# ---------------------------------------------------------------------------
# Sampling parameters — omitted for models that reject them
# ---------------------------------------------------------------------------

#: (wire name, that wire's spelling of top_p). Every wire is exercised for
#: every case on purpose: the pair was written into four bodies independently,
#: and a fix that lands on one leaves the others 400ing — the anthropic wire,
#: the one the bug report was filed against, was the easiest to miss.
WIRES = [
    ("openai-completions", "top_p"),
    ("openai-responses", "top_p"),
    ("anthropic", "top_p"),
    ("google", "topP"),
]


def _bodies(request: ChatRequest) -> dict[str, dict[str, Any]]:
    """Every wire's request body for one request, keyed by wire name."""
    from local_operator.providers.clients import GoogleClient

    openai = OpenAICompatClient("https://x")
    return {
        "openai-completions": openai._build_body(request),
        "openai-responses": openai._build_responses_body(request),
        "anthropic": AnthropicClient()._build_body(request),
        "google": GoogleClient()._build_body(request)["generationConfig"],
    }


@pytest.mark.parametrize("wire,top_p_key", WIRES)
def test_sampling_params_omitted_when_model_rejects_them(wire: str, top_p_key: str) -> None:
    """``claude-opus-5`` answers HTTP 400 "``temperature`` is deprecated for
    this model." — and the same for ``top_p`` once temperature is gone, so both
    have to go. The keys must be ABSENT, not null: a provider that rejects the
    key rejects it with a null value just as hard."""
    spec = ModelSpec(provider="anthropic", model_id="claude-opus-5", supports_sampling_params=False)
    body = _bodies(ChatRequest(model=spec, messages=[Message.user("hi")]))[wire]
    assert "temperature" not in body
    assert top_p_key not in body
    serialised = json.dumps(body)
    assert '"temperature"' not in serialised and f'"{top_p_key}"' not in serialised


@pytest.mark.parametrize("wire,top_p_key", WIRES)
def test_sampling_params_still_sent_when_model_accepts_them(wire: str, top_p_key: str) -> None:
    """The regression guard for the fix itself: sampling is a real feature, and
    dropping it everywhere would be worse than the 400 it avoids."""
    spec = ModelSpec(provider="anthropic", model_id="claude-sonnet-4-5", temperature=0.7, top_p=0.4)
    body = _bodies(ChatRequest(model=spec, messages=[Message.user("hi")]))[wire]
    assert body["temperature"] == 0.7
    assert body[top_p_key] == 0.4


@pytest.mark.parametrize("wire,top_p_key", WIRES)
def test_explicit_request_sampling_loses_to_the_capability(wire: str, top_p_key: str) -> None:
    """A per-request override cannot resurrect a parameter the model rejects —
    honouring it would only move the 400 from the spec default to the caller."""
    spec = ModelSpec(provider="anthropic", model_id="claude-opus-5", supports_sampling_params=False)
    request = ChatRequest(model=spec, messages=[Message.user("hi")], temperature=0.9, top_p=0.1)
    body = _bodies(request)[wire]
    assert "temperature" not in body
    assert top_p_key not in body


@pytest.mark.parametrize("wire,top_p_key", WIRES)
def test_request_sampling_overrides_reach_the_wire_when_supported(wire: str, top_p_key: str):
    """...and the override path still works for a model that accepts them."""
    spec = ModelSpec(provider="openai", model_id="gpt-4o")
    request = ChatRequest(model=spec, messages=[Message.user("hi")], temperature=0.05, top_p=0.15)
    body = _bodies(request)[wire]
    assert body["temperature"] == 0.05
    assert body[top_p_key] == 0.15


def test_anthropic_oauth_prepends_the_claude_code_identity() -> None:
    """A subscription credential MUST carry the Claude Code identity block first.

    Anthropic gates OAuth credentials to Claude Code and refuses anything else with
    an opaque `HTTP 429 Error` — not a 401 — so the failure reads as rate limiting
    and sends the operator to look at their quota. Measured against the live
    endpoint with a valid subscription token and `claude-opus-5`: no system block
    and an ordinary system block both 429, this block first returns 200.
    """
    request = ChatRequest(
        model=_spec("anthropic", "claude-opus-5"),
        messages=[Message.user("hi")],
        system_blocks=["instructions", "tools", "env"],
    )
    access = OAuthAccess(access_token="oauth-token-1", credential_id=1, org_id=None, kind="oauth")
    client = AnthropicClient()

    oauth_body = client._build_body(request, oauth=client._is_oauth(access))
    texts = [block["text"] for block in oauth_body["system"]]
    assert texts[0] == AnthropicClient.CLAUDE_CODE_IDENTITY
    # The app's own blocks survive, in order, after it.
    assert texts[1:] == ["instructions", "tools", "env"]


def test_anthropic_api_key_does_not_get_the_claude_code_identity() -> None:
    """An API-key caller is not gated, and an identity line changes how the model
    answers — so a paying key user must not silently be told they are a CLI."""
    request = ChatRequest(
        model=_spec("anthropic", "claude-opus-5"),
        messages=[Message.user("hi")],
        system_blocks=["instructions", "tools", "env"],
    )
    client = AnthropicClient()

    assert client._is_oauth(None) is False
    key_body = client._build_body(request, oauth=client._is_oauth(None))
    texts = [block["text"] for block in key_body["system"]]
    assert AnthropicClient.CLAUDE_CODE_IDENTITY not in texts
    assert texts == ["instructions", "tools", "env"]


def test_anthropic_oauth_identity_keeps_the_cached_prefix_byte_stable() -> None:
    """The identity block is a constant, so two turns must render an identical
    system prefix — otherwise every turn would re-write the prompt cache instead
    of reading it (measured live: 3,911 cache-read vs 130 cache-write tokens)."""
    access = OAuthAccess(access_token="oauth-token-1", credential_id=1, org_id=None, kind="oauth")
    client = AnthropicClient()

    def system_for(user_text: str) -> list[dict[str, object]]:
        request = ChatRequest(
            model=_spec("anthropic", "claude-opus-5"),
            messages=[Message.user(user_text)],
            system_blocks=["instructions", "tools", "env"],
        )
        return client._build_body(request, oauth=client._is_oauth(access))["system"]

    first, second = system_for("turn one"), system_for("turn two different length")
    assert first == second, "the system prefix must not vary with the conversation"


# ---------------------------------------------------------------------------
# Reasoning effort — one setting, one key per family
# ---------------------------------------------------------------------------
#
# Specs come from `build_model_spec` rather than the bare `_spec` helper above,
# on purpose: the point of these is that a level chosen in the UI survives the
# whole path — derivation, spec, request, body — and a hand-built spec would
# test the last hop only. An effort the user selects that never reaches the
# provider is worse than no control at all, because the band then asserts a
# depth of thought that is not in force.

from local_operator.model.configure import build_model_spec  # noqa: E402


def _effort_spec(provider: str, model_id: str, level: str | None) -> ModelSpec:
    return build_model_spec(provider, model_id).model_copy(update={"reasoning_effort": level})


def test_anthropic_sends_effort_as_output_config() -> None:
    """Anthropic's own key. `output_config.effort` covers ALL response tokens —
    text, tool calls and thinking — where a `thinking` budget bounds only the
    thinking block and needs it enabled first."""
    request = ChatRequest(
        model=_effort_spec("anthropic", "claude-opus-5", "xhigh"),
        messages=[Message.user("hi")],
    )
    body = AnthropicClient()._build_body(request)
    assert body["output_config"] == {"effort": "xhigh"}


def test_anthropic_boots_carrying_its_documented_default() -> None:
    """No `/effort` typed: the level on the wire is the one the band shows from
    the first frame, which is only truthful because Anthropic documents `high`
    as identical to omitting the parameter."""
    request = ChatRequest(
        model=build_model_spec("anthropic", "claude-opus-5"), messages=[Message.user("hi")]
    )
    assert AnthropicClient()._build_body(request)["output_config"] == {"effort": "high"}


def test_openai_chat_completions_sends_effort_flat() -> None:
    request = ChatRequest(
        model=_effort_spec("openai", "gpt-5.4", "medium"), messages=[Message.user("hi")]
    )
    body = OpenAICompatClient("https://api.openai.com/v1")._build_body(request)
    assert body["reasoning_effort"] == "medium"


def test_openai_responses_nests_the_same_value() -> None:
    """The ChatGPT-OAuth route speaks Responses, where the same level lives
    under `reasoning.effort`. Same setting, two spellings — which is exactly
    why the spec carries the level and not the key."""
    request = ChatRequest(
        model=_effort_spec("openai", "gpt-5.4", "xhigh"), messages=[Message.user("hi")]
    )
    body = OpenAICompatClient("https://api.openai.com/v1")._build_responses_body(request)
    assert body["reasoning"] == {"effort": "xhigh"}


@pytest.mark.parametrize("model_id", ["gpt-4.1", "gpt-4o"])
def test_a_model_without_a_ladder_gets_no_key_at_all(model_id: str) -> None:
    """Omitted, never sent empty or null: a provider that rejects the key
    rejects it just as hard with a null value — the same rule
    `supports_sampling_params` follows for temperature."""
    spec = _effort_spec("openai", model_id, "high")  # a level nothing could honour
    request = ChatRequest(model=spec, messages=[Message.user("hi")])
    client = OpenAICompatClient("https://api.openai.com/v1")
    assert "reasoning_effort" not in client._build_body(request)
    assert "reasoning" not in client._build_responses_body(request)


def test_a_level_outside_the_models_ladder_is_dropped_rather_than_sent() -> None:
    """The spec is mutable at runtime — `/effort`, `shift+tab` and failover all
    write it — so the body builder re-checks rather than trusting. Dropping
    costs one turn's depth; sending costs the turn."""
    spec = build_model_spec("anthropic", "claude-opus-4-5-20251101").model_copy(
        update={"reasoning_effort": "xhigh"}  # only 4.7+ accepts xhigh
    )
    request = ChatRequest(model=spec, messages=[Message.user("hi")])
    assert "output_config" not in AnthropicClient()._build_body(request)


def test_google_sends_no_effort_key() -> None:
    """Deliberate: this client speaks `generateContent`, whose thinking control
    on the shipped models is a token budget rather than the named tiers the
    Interactions API exposes. No Gemini model is given a ladder for that reason,
    so nothing here can put an unmappable level on the wire."""
    spec = build_model_spec("google", "gemini-2.5-pro").model_copy(
        update={"reasoning_effort": "high"}
    )
    request = ChatRequest(model=spec, messages=[Message.user("hi")])
    body = GoogleClient()._build_body(request)
    # Scanned inside `generationConfig`, where this client writes every
    # generation setting: a top-level scan cannot fail, because no plausible
    # edit would put the key there.
    assert "thinkingConfig" not in body["generationConfig"]
    assert not any("effort" in key.lower() for key in body["generationConfig"])


def _google_sse_parallel_calls() -> bytes:
    """One Gemini response carrying TWO same-name functionCalls — the exact
    shape that used to mint identical ids (``fc_read`` twice)."""
    return _sse(
        [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "read", "args": {"path": "a.py"}}},
                                {"functionCall": {"name": "read", "args": {"path": "b.py"}}},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
        ]
    )


async def test_google_parallel_same_tool_calls_get_unique_ids() -> None:
    """Two same-tool functionCalls in one Gemini response must survive as two
    tool calls. The loop dedups tool calls by id (first-wins), so ids minted
    from the tool name alone silently dropped every call after the first —
    the model believed two reads ran when only one did. The per-response
    index is minted alongside the id because it is the stream slot the loop
    assembles argument deltas into; two calls sharing slot 0 would overwrite
    each other even with distinct ids."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_google_sse_parallel_calls(),
            headers={"content-type": "text/event-stream"},
        )

    client = GoogleClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    events = await _collect(
        client.stream(
            ChatRequest(model=_spec("google", "gemini-2.5-pro"), messages=[Message.user("hi")]),
            "g-key",
        )
    )
    calls = [e for e in events if isinstance(e, StreamToolCallDelta)]
    assert len(calls) == 2
    assert calls[0].id != calls[1].id
    assert {calls[0].index, calls[1].index} == {0, 1}
    assert all(call.name == "read" for call in calls)
