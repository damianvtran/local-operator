"""Wire client tests against httpx.MockTransport SSE fixtures. No network."""

from __future__ import annotations

import json
from collections.abc import Sequence
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
    StreamUsageEvent,
    TextContent,
    ToolCall,
)
from local_operator.providers.clients import (
    AnthropicClient,
    MockClient,
    OpenAICompatClient,
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


async def test_openai_oauth_routes_to_responses_with_account_header() -> None:
    """ChatGPT OAuth: /responses endpoint + chatgpt-account-id header."""
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
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["chatgpt-account-id"] == "acct-42"
    assert captured["headers"]["authorization"] == "Bearer chatgpt-token"
    assert "input" in captured["body"] and "messages" not in captured["body"]
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
