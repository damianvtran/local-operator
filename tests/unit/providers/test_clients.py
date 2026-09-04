"""Wire client tests against httpx.MockTransport SSE fixtures. No network."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

import httpx
import pytest

from local_operator.compaction.thresholds import (
    CompactionSettings,
    resolve_threshold_tokens,
)
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
    ToolContext,
    ToolResult,
)
from local_operator.providers.clients import (
    DEFAULT_ESTIMATE_SLOPE,
    MIN_OUTPUT_TOKENS,
    OUTPUT_CLAMP_SAFETY_MARGIN,
    AnthropicClient,
    GoogleClient,
    MockClient,
    OpenAICompatClient,
    _anthropic_stream_error,
    _effective_max_tokens,
    _estimate_slope,
    _estimated_prompt_tokens,
    _message_to_openai,
    _output_reserve_tokens,
    client_for_spec,
    raise_for_status,
)
from local_operator.providers.failover import ProviderError
from local_operator.tools.registry import create_tools

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
                    "completion_tokens_details": {"reasoning_tokens": 3},
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
    # completion_tokens_details.reasoning_tokens is the thinking slice of the
    # completion, a subset of output that analytics splits out.
    assert usage_events[0].usage.reasoning_tokens == 3

    end = events[-1]
    assert isinstance(end, StreamEndEvent)
    assert end.stop_reason == "toolUse"
    assert end.usage is not None and end.usage.output_tokens == 7


@pytest.mark.parametrize(
    ("provider_field", "cached"),
    [("cached_tokens", 12), ("prompt_cache_hit_tokens", 12)],
)
async def test_openai_compat_normalizes_provider_specific_cache_hits(
    provider_field: str, cached: int
) -> None:
    """Kimi and DeepSeek expose cache hits beside ``prompt_tokens``.

    Missing these aliases prices the cached prefix at the full input rate, which
    is the reported per-session cost inflation this regression reproduces.
    """
    body = _sse(
        [
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 7,
                    provider_field: cached,
                },
            }
        ]
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    client = OpenAICompatClient(
        "https://api.test.example/v1", http_client=httpx.AsyncClient(transport=transport)
    )
    events = await _collect(
        client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
    )

    usage = [event.usage for event in events if isinstance(event, StreamUsageEvent)][-1]
    assert usage.input_tokens == 40
    assert usage.cache_read_tokens == 12
    assert usage.context_tokens == 40


async def test_openai_compat_prefers_standard_cache_detail_over_alias() -> None:
    """A compatibility alias must not double-count a standard cache detail."""
    body = _sse(
        [
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 7,
                    "cached_tokens": 20,
                    "prompt_cache_hit_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        ]
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    client = OpenAICompatClient(
        "https://api.test.example/v1", http_client=httpx.AsyncClient(transport=transport)
    )
    events = await _collect(
        client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
    )

    usage = [event.usage for event in events if isinstance(event, StreamUsageEvent)][-1]
    assert usage.cache_read_tokens == 0


async def test_openai_compat_mid_stream_error_chunk_raises_named_error() -> None:
    """OpenRouter mid-stream failures arrive in-band on HTTP 200.

    The gateway commits 200 before the upstream dies, so the failure is a
    ``chat.completion.chunk`` with a top-level ``error`` object and
    ``finish_reason: "error"``. The parser must raise it NAMED (status,
    message, retryability) so the failover driver can journal, retry and
    rotate — the old behaviour dropped the object and the turn died as a
    wordless interruption.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                [
                    {"id": "gen-1", "choices": [{"delta": {"content": "Gate"}, "index": 0}]},
                    {
                        "id": "gen-1",
                        "object": "chat.completion.chunk",
                        "provider": "OpenAI",
                        "error": {
                            "code": 429,
                            "message": "Rate limit exceeded",
                            "metadata": {"error_type": "rate_limit_exceeded"},
                        },
                        "choices": [
                            {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
                        ],
                    },
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenAICompatClient(
        "https://api.test.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError) as excinfo:
        await _collect(
            client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
        )
    error = excinfo.value
    assert error.status == 429
    assert error.retryable is True
    assert error.kind == "quota"
    assert "Rate limit exceeded" in str(error)


async def test_openai_compat_mid_stream_error_chunk_unwraps_upstream_raw() -> None:
    """A 502-class chunk carries the upstream text in ``metadata.raw``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                [
                    {
                        "id": "gen-2",
                        "error": {
                            "code": 502,
                            "message": "Provider returned error",
                            "metadata": {
                                "raw": json.dumps({"error": {"message": "upstream overloaded"}})
                            },
                        },
                        "choices": [
                            {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
                        ],
                    },
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenAICompatClient(
        "https://api.test.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError) as excinfo:
        await _collect(
            client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
        )
    error = excinfo.value
    assert error.status == 502
    assert error.kind == "transient"
    assert "Provider returned error" in str(error)
    assert "upstream overloaded" in str(error)


async def test_openai_compat_mid_stream_error_chunk_bare_string() -> None:
    """Simpler compatible servers send the error as a bare string."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse([{"error": "upstream connection reset"}]),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenAICompatClient(
        "https://api.test.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError) as excinfo:
        await _collect(
            client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
        )
    assert "upstream connection reset" in str(excinfo.value)


async def test_openai_compat_error_finish_reason_without_error_object() -> None:
    """A bare ``finish_reason: "error"`` still names the failure.

    Without the top-level error object the old parser passed the raw reason
    through as an exotic stop reason and the loop recorded a wordless error
    turn — no incident, no diagnosis. The end event now carries the reason in
    ``error`` so the session journals it.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                [
                    {"id": "gen-3", "choices": [{"delta": {"content": "x"}, "index": 0}]},
                    {"id": "gen-3", "choices": [{"delta": {}, "finish_reason": "error"}]},
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenAICompatClient(
        "https://api.test.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    events = await _collect(
        client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
    )
    end = events[-1]
    assert isinstance(end, StreamEndEvent)
    assert end.stop_reason == "error"
    # The incident text must name the failure, not merely contain the word
    # "error" — a regression to a tautology like "error: error" would still
    # pass a substring check.
    assert end.error == "provider reported a mid-stream failure (finish_reason 'error')"


async def test_openai_compat_captures_provider_reported_cost() -> None:
    """OpenRouter's ``usage.cost`` is the provider's own bill and must reach
    ``Usage.usd_cost`` intact, so the TUI can prefer it over a token×rate
    reconstruction."""
    body = _sse(
        [
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": "boop"}, "index": 0}]},
            {
                "id": "chatcmpl-1",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 16,
                    "cost": 7.5e-6,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        ]
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    client = OpenAICompatClient(
        "https://api.test.example/v1", http_client=httpx.AsyncClient(transport=transport)
    )
    request = ChatRequest(
        model=_spec(provider="openrouter", model_id="deepseek/deepseek-v4-flash-0731"),
        messages=[Message.user("hi")],
    )
    events = await _collect(client.stream(request, "sk-test"))

    usage_events = [e for e in events if isinstance(e, StreamUsageEvent)]
    assert usage_events and usage_events[0].usage.usd_cost == pytest.approx(7.5e-6)


async def test_openai_compat_cost_is_none_when_provider_omits_it() -> None:
    """Providers that do not precompute billing leave ``usd_cost`` as ``None``,
    never a fabricated 0.0 — the distinction the whole estimate fallback depends on."""
    body = _sse(
        [
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": "boop"}, "index": 0}]},
            {
                "id": "chatcmpl-1",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        ]
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    client = OpenAICompatClient(
        "https://api.test.example/v1", http_client=httpx.AsyncClient(transport=transport)
    )
    request = ChatRequest(model=_spec(), messages=[Message.user("hi")])
    events = await _collect(client.stream(request, "sk-test"))

    usage_events = [e for e in events if isinstance(e, StreamUsageEvent)]
    assert usage_events and usage_events[0].usage.usd_cost is None


def test_usd_cost_coerces_and_rejects_malformed() -> None:
    """The cost field is a JSON number but must not abort a stream when a
    provider spells it differently; absent/negative/non-numeric all yield None."""
    from local_operator.providers.clients import _usd_cost

    assert _usd_cost({"cost": 7.5e-6}) == pytest.approx(7.5e-6)
    assert _usd_cost({"cost": "0.1"}) == pytest.approx(0.1)
    assert _usd_cost({"cost": 0.0}) == 0.0  # a real zero: billed as free
    assert _usd_cost({}) is None
    assert _usd_cost({"cost": None}) is None
    assert _usd_cost({"cost": -1}) is None
    assert _usd_cost({"cost": "not-a-number"}) is None
    assert _usd_cost(None) is None
    # Non-finite is wire-reachable: ``json.loads`` parses the non-standard
    # ``Infinity``/``NaN`` literals by default, so a provider can emit them. An
    # unfloored ``inf`` would pin every summed total at infinity forever
    # (``inf + x == inf``), so both the JSON literal and the numeric-string
    # spelling must fall through to ``None`` and let the estimate answer.
    assert _usd_cost(json.loads('{"cost": Infinity}')) is None
    assert _usd_cost(json.loads('{"cost": -Infinity}')) is None
    assert _usd_cost(json.loads('{"cost": NaN}')) is None
    assert _usd_cost({"cost": float("inf")}) is None
    assert _usd_cost({"cost": "inf"}) is None
    assert _usd_cost({"cost": "nan"}) is None


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
                        "output_tokens_details": {"reasoning_tokens": 4},
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

    async def unused_execute(*_args: Any, **_kwargs: Any) -> ToolResult:
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
    # output_tokens_details.reasoning_tokens is the thinking slice of output.
    assert usage.reasoning_tokens == 4
    assert events[-1].stop_reason == "toolUse"


def _enum_members(node: Any) -> list[Any]:
    """Collect enum members from exactly the provider-bound JSON payload.

    Walking the serialized body, rather than Pydantic models one by one, keeps
    this guard on the path gateways inspect after the complete tool inventory
    and the shared intent field have both been assembled.
    """

    members: list[Any] = []
    if isinstance(node, dict):
        enum = node.get("enum")
        if isinstance(enum, list):
            members.extend(enum)
        for value in node.values():
            members.extend(_enum_members(value))
    elif isinstance(node, list):
        for value in node:
            members.extend(_enum_members(value))
    return members


class _FakeScheduler:
    """Just enough surface for build_wake_tool's capability check."""

    @property
    def schedules(self) -> list[Any]:
        return []

    async def update(self, schedules: Any) -> None:
        pass


class _FakeJobs:
    """Just enough surface for the job-tracking tools' capability check."""

    def get(self, job_id: str, *, owner_id: str | None = None) -> Any:
        return None

    def list(self, *, owner_id: str | None = None) -> list[Any]:
        return []

    async def cancel(self, job_id: str, *, owner_id: str | None = None) -> bool:
        return False


class _FakeComms:
    """Just enough surface for build_hub_tool: the role test it branches on."""

    def is_child(self, job_id: str | None) -> bool:
        return False


async def _ask_user(questions: list[Any]) -> dict[str, list[str]] | None:
    return None


def _default_tools_with_agent() -> list[AgentTool]:
    """Build the genuinely COMPLETE inventory, not merely the ungated part.

    A context carrying only ``agent_registry`` builds 15 of the 23 tools: every
    ``createIf``-gated one drops out, and five of those (``task``, ``hub``,
    ``jobs``, ``wake``, ``team``) declare enums of their own. ``ask`` is gated
    too but has no enum; it is still in the floor so a silent drop of the ask
    hook still fails here. Auditing enum members over that reduced set would
    let an empty member ship in any of the five while a test named for the
    full bundle stayed green — the same class of gap that let the ``agent``
    effort sentinel reach a provider. Every capability is therefore attached
    so the audit sees the whole emitted surface.
    """

    tools = create_tools(
        ToolContext(
            cwd=".",
            agent_registry=object(),
            team_registry=object(),
            wake_scheduler=_FakeScheduler(),
            jobs=_FakeJobs(),
            subagent_comms=_FakeComms(),
            subagent_launcher=lambda label, prompt, **kwargs: "job-fake",
            has_ui=True,
            ask_user=_ask_user,
        )
    )
    # The point of this helper is coverage breadth, so assert the breadth rather
    # than trusting it: a future gate that silently drops a tool must fail here
    # instead of quietly narrowing every audit built on top of it. ``browser``
    # is excluded from the floor because its builder probes a real CMUX surface
    # that CI does not have.
    names = {tool.name for tool in tools}
    assert {"agent", "task", "hub", "jobs", "wake", "ask", "team"} <= names
    return tools


async def test_openrouter_gemini_full_tool_bundle_has_no_empty_enum_member() -> None:
    """Exercise the real OpenRouter client and complete registry-built bundle."""

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse([{"choices": [{"delta": {}, "finish_reason": "stop"}]}]),
            headers={"content-type": "text/event-stream"},
        )

    tools = _default_tools_with_agent()
    assert "agent" in {tool.name for tool in tools}
    spec = _spec(provider="openrouter", model_id="google/gemini-3.7-flash")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = client_for_spec(spec, http_client=http)

    await _collect(
        client.stream(
            ChatRequest(
                model=spec,
                messages=[Message.user("Use the tools if useful")],
                tools=tools,
            ),
            "test-key",
        )
    )
    await http.aclose()

    assert "" not in _enum_members(captured["body"]["tools"])
    agent = next(tool for tool in captured["body"]["tools"] if tool["function"]["name"] == "agent")
    effort = agent["function"]["parameters"]["properties"]["effort"]
    assert effort["anyOf"][0]["enum"] == ["lo", "med", "hi", "inherit"]


def test_direct_google_full_tool_bundle_has_no_empty_enum_member() -> None:
    tools = _default_tools_with_agent()
    body = GoogleClient()._build_body(
        ChatRequest(
            model=_spec(provider="google", model_id="gemini-3.7-flash"),
            messages=[Message.user("Use the tools if useful")],
            tools=tools,
        )
    )

    declarations = body["tools"][0]["function_declarations"]
    assert "" not in _enum_members(declarations)
    agent = next(declaration for declaration in declarations if declaration["name"] == "agent")
    effort = agent["parameters"]["properties"]["effort"]
    assert effort["anyOf"][0]["enum"] == ["lo", "med", "hi", "inherit"]


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [("openai", "gpt-5.4"), ("openrouter", "qwen/qwen3-coder")],
)
def test_openai_compatible_models_keep_agent_effort_semantics(provider: str, model_id: str) -> None:
    """Ordinary OpenAI/Qwen routes receive every tier plus the clear sentinel."""

    tools = _default_tools_with_agent()
    request = ChatRequest(
        model=_spec(provider=provider, model_id=model_id),
        messages=[Message.user("clear it")],
        tools=tools,
    )
    body = OpenAICompatClient("https://compat.example/v1")._build_body(request)
    agent = next(tool for tool in body["tools"] if tool["function"]["name"] == "agent")
    effort = agent["function"]["parameters"]["properties"]["effort"]

    assert effort["anyOf"][0]["enum"] == ["lo", "med", "hi", "inherit"]
    assert {branch["type"] for branch in effort["anyOf"]} == {"string", "null"}


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
            "Google returned error: Quota exceeded for quota metric 'Requests'"
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


class TestOpaqueAggregator400:
    """An aggregator relays an UPSTREAM failure as an HTTP 400 whose body names
    nothing. Session e13d092c093c recorded the shape — ``message`` exactly
    "Provider returned error", ``metadata.raw`` a bare "ERROR", provider_name
    "Stealth" — arriving intermittently on a request that live probes at ~750k
    tokens answered 200 seconds later, and again during the investigation.
    Classified ``request``, it aborted the turn before rotation or fallback
    could serve it; these tests pin both sides of the new line: the opaque
    sentinel is transient/retryable, every 400 with real diagnostics stays an
    answer."""

    def _error(self, response: httpx.Response) -> ProviderError:
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(response)
        return excinfo.value

    @staticmethod
    def _openrouter_body(raw: str, provider: str = "Stealth") -> dict[str, Any]:
        return {
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {"raw": raw, "provider_name": provider},
            }
        }

    def test_the_reported_sentinel_is_transient(self) -> None:
        """The exact wire body from the session: bare "ERROR" in ``raw``."""
        error = self._error(httpx.Response(400, json=self._openrouter_body("ERROR")))
        assert (error.kind, error.retryable) == ("transient", True)
        # The frame still shows what the wire said — no invented diagnostics —
        # but the generic "Provider" is replaced by the origin host the
        # aggregator named, so the reader knows WHO failed.
        assert error.message == "Stealth returned error: ERROR"

    def test_a_quoted_json_string_sentinel_is_transient_too(self) -> None:
        """``raw`` is defined as JSON-encoded, so the sentinel can arrive as
        ``\"\\\"ERROR\\\"\"`` and parse straight back to quotes."""
        error = self._error(httpx.Response(400, json=self._openrouter_body(json.dumps("ERROR"))))
        assert (error.kind, error.retryable) == ("transient", True)

    def test_actionable_upstream_diagnostics_stay_request(self) -> None:
        """A ``raw`` holding the origin provider's REAL body is an answer, not
        weather: the same outer message with real diagnostics keeps kind
        ``request`` so a genuinely broken request is not retried."""
        body = self._openrouter_body(
            json.dumps({"error": {"message": "context length exceeded"}}),
            provider="Google",
        )
        error = self._error(httpx.Response(400, json=body))
        assert (error.kind, error.retryable) == ("request", False)
        assert "context length exceeded" in error.message

    def test_real_text_in_raw_stays_request(self) -> None:
        """A non-JSON ``raw`` that still SAYS something is actionable."""
        error = self._error(
            httpx.Response(
                400,
                json=self._openrouter_body("This model's maximum context length is 16385 tokens"),
            )
        )
        assert (error.kind, error.retryable) == ("request", False)

    def test_direct_provider_400s_are_untouched(self) -> None:
        """Only the aggregator's exact outer message qualifies; ordinary 400s
        keep their classification whatever their metadata holds."""
        plain = self._error(httpx.Response(400, json={"error": {"message": "bad field"}}))
        assert (plain.kind, plain.retryable) == ("request", False)
        other_message = self._error(
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Provider error",
                        "code": 400,
                        "metadata": {"raw": "ERROR", "provider_name": "Stealth"},
                    }
                },
            )
        )
        assert (other_message.kind, other_message.retryable) == ("request", False)

    def test_nested_error_object_sentinel_is_transient(self) -> None:
        """``_openrouter_upstream_text`` unwraps a nested ``error`` object, so a
        structurally-richer body whose CONTENT is still the bare word carries no
        more information than the flat sentinel and matches too. Documented as a
        deliberate widening rather than left to chance."""
        body = self._openrouter_body(json.dumps({"error": {"message": "ERROR"}}))
        error = self._error(httpx.Response(400, json=body))
        assert (error.kind, error.retryable) == ("transient", True)

    def test_unbalanced_quote_runs_are_transient_too(self) -> None:
        """Only MATCHED quote pairs are peeled, so this does NOT reduce to the
        sentinel — and under the relay rule it no longer needs to. A malformed
        run of quotes inside the relay envelope still describes nothing the
        caller could fix, so it takes the same transient path as the sentinel.

        This inverts the previous expectation deliberately. The old predicate
        could only reach transient through an exact sentinel match, which made
        "unbalanced ⇒ request" the accidental default; the rule now keys on the
        envelope, and only :data:`_DETERMINISTIC_UPSTREAM_MARKERS` pulls a
        relayed body back to ``request``."""
        error = self._error(httpx.Response(400, json=self._openrouter_body("'''ERROR\"\"")))
        assert (error.kind, error.retryable) == ("transient", True)

    def test_opaque_5xx_shape_is_unchanged(self) -> None:
        """A 5xx carrying the same opaque body was already retryable by status
        and must stay that way."""
        body = self._openrouter_body("ERROR")
        body["error"]["code"] = 502
        error = self._error(httpx.Response(502, json=body))
        assert (error.kind, error.retryable) == ("transient", True)


class TestRelayedUpstream404:
    """A relayed upstream 404 whose text READS like an answer but is weather.

    Session 2be018a98088 (2026-09-04) died twice in 75 seconds on
    ``openrouter/meta/muse-spark-1.3`` with HTTP 404 and the upstream words
    "The requested model was not found." Six successful calls on the IDENTICAL
    model id landed between the two failures, and the model's single endpoint
    reported 99.98% 30-minute uptime, so the id cannot have been the cause: it
    was Meta transiently failing to resolve its own snapshot, relayed by
    OpenRouter under a 404.

    The old predicate missed it on two counts \u2014 it was gated on 400, and it
    demanded a bare sentinel where this body carried plausible prose \u2014 so the
    failure was classified ``request`` and the turn was aborted with no retry
    and no failover.

    The discrimination pinned here is STRUCTURAL, verified against the live API
    on 2026-09-04: OpenRouter answers a model id it does not know with a flat
    400 naming the slug, and a routing refusal with a flat 404 naming the
    preference. Neither carries ``metadata.raw``. Only a genuine relay does.
    """

    def _error(self, response: httpx.Response) -> ProviderError:
        with pytest.raises(ProviderError) as excinfo:
            raise_for_status(response)
        return excinfo.value

    @staticmethod
    def _relay(status: int, raw: str, provider: str = "Meta") -> dict[str, Any]:
        return {
            "error": {
                "message": "Provider returned error",
                "code": status,
                "metadata": {"raw": raw, "provider_name": provider, "is_byok": False},
            }
        }

    def test_the_recorded_404_is_transient_and_names_the_upstream(self) -> None:
        """The exact wire body from the session."""
        raw = json.dumps(
            {
                "error": {
                    "message": "The requested model was not found.",
                    "type": "invalid_request_error",
                }
            }
        )
        error = self._error(httpx.Response(404, json=self._relay(404, raw)))
        assert (error.kind, error.retryable) == ("transient", True)
        # The frame must name Meta rather than the ambiguous "Provider": the
        # user is looking at an aggregator, so "provider" alone does not say
        # whether the gateway or the model host failed.
        assert str(error) == (
            "transient provider error (HTTP 404): "
            "Meta returned error: The requested model was not found."
        )

    def test_flat_unknown_model_400_still_fails_fast(self) -> None:
        """OpenRouter's OWN refusal of a bad slug: flat body, no relay
        envelope, names the slug. This is the case a user really can fix, so it
        must keep failing immediately instead of burning the retry budget."""
        body = {"error": {"message": "meta/muse-spark-9.9 is not a valid model ID", "code": 400}}
        error = self._error(httpx.Response(400, json=body))
        assert (error.kind, error.retryable) == ("request", False)

    def test_flat_routing_404_still_fails_fast(self) -> None:
        """The aggregator's own routing refusal, also flat and also actionable
        (the caller's provider preference is wrong)."""
        body = {
            "error": {
                "message": (
                    "No allowed providers are available for the selected model. "
                    "Providers serving meta/muse-spark-1.3-20260902: meta, but your "
                    "request's provider.only preference permits only: groq."
                ),
                "code": 404,
                "metadata": {"available_providers": ["meta"], "requested_providers": ["groq"]},
            }
        }
        error = self._error(httpx.Response(404, json=body))
        assert (error.kind, error.retryable) == ("request", False)

    def test_a_relayed_complaint_naming_a_request_field_stays_request(self) -> None:
        """Review finding B1, pinned.

        The relay envelope proves PROVENANCE, not retryability. These bodies
        were captured from the LIVE API on 2026-09-04 by sending deliberately
        malformed requests: each arrives fully wrapped, with a
        ``provider_name``, and each is a defect in our own bytes that no retry
        can fix. An earlier draft keyed on the envelope alone and retried them
        12 times over ~35s apiece.

        What separates them from the incident is ``param``: the origin names
        the offending field of OUR request. The incident names none, because
        nothing about the request was the problem."""
        for inner in (
            {
                "message": "Invalid 'tools[0].function.name': does not match pattern.",
                "type": "invalid_request_error",
                "param": "tools[0].function.name",
                "code": "invalid_value",
            },
            {
                "message": "Invalid schema for response_format 'x'.",
                "type": "invalid_request_error",
                "param": "response_format",
                "code": None,
            },
            {
                "message": "Invalid value: 'wizard'.",
                "type": "invalid_request_error",
                "param": "messages[0].role",
            },
        ):
            error = self._error(
                httpx.Response(400, json=self._relay(400, json.dumps({"error": inner}), "OpenAI"))
            )
            assert (error.kind, error.retryable) == ("request", False), inner["param"]

    def test_type_alone_cannot_discriminate(self) -> None:
        """Why the predicate does not key on ``type``.

        The incident and the malformed-request bodies are BOTH
        ``invalid_request_error``. Only the presence of ``param`` tells them
        apart, so a future edit that reaches for ``type`` instead has this test
        to explain why that cannot work."""
        shared = {"message": "...", "type": "invalid_request_error"}
        weather = self._error(
            httpx.Response(404, json=self._relay(404, json.dumps({"error": dict(shared)})))
        )
        ours = self._error(
            httpx.Response(
                400, json=self._relay(400, json.dumps({"error": {**shared, "param": "top_p"}}))
            )
        )
        assert weather.kind == "transient"
        assert ours.kind == "request"

    def test_relayed_overflow_without_param_stays_request(self) -> None:
        """Review finding M1, pinned.

        A provider can report an overflow WITHOUT naming a param -- anthropic's
        "prompt is too long" and google's token-count wording both do. Those
        are deterministic in the same way (the request is too big and stays too
        big), so they are carved out by the harness's canonical
        ``CONTEXT_LENGTH_MARKERS`` rather than by a second list maintained
        here, which is what stops the two from drifting apart."""
        for text in (
            "prompt is too long: 250000 tokens > 200000 maximum",
            "The input token count exceeds the maximum context window",
            "This model's maximum context length is 16385 tokens",
        ):
            error = self._error(
                httpx.Response(400, json=self._relay(400, json.dumps({"error": {"message": text}})))
            )
            assert (error.kind, error.retryable) == ("request", False), text

    def test_the_shared_marker_list_is_the_harness_one(self) -> None:
        """The carve-out must stay wired to the canonical list, not a copy.

        Asserting identity rather than contents is deliberate: a test that
        re-listed the wordings would itself become the third copy this finding
        was about."""
        from local_operator.incidents import CONTEXT_LENGTH_MARKERS as canonical
        from local_operator.providers import clients

        assert clients.CONTEXT_LENGTH_MARKERS is canonical

    def test_a_quoted_overflow_is_still_recognised(self) -> None:
        """Quoting is the aggregator's formatting, not part of the signal: a
        quoted overflow must carve out exactly like a bare one, or a
        deterministic failure would be retried purely because it arrived
        wrapped in quotes."""
        raw = json.dumps("This model's maximum context length is 16385 tokens")
        error = self._error(httpx.Response(400, json=self._relay(400, raw)))
        assert (error.kind, error.retryable) == ("request", False)

    def test_relayed_401_still_reaches_credential_rotation(self) -> None:
        """401/403 are deliberately OUT of the relayed set: they describe the
        caller's credential and must keep reaching rotation rather than being
        retried as weather."""
        error = self._error(httpx.Response(401, json=self._relay(401, "ERROR")))
        assert error.kind == "auth"

    def test_an_absurdly_long_provider_name_is_not_rendered(self) -> None:
        """Review finding M3, pinned.

        ``provider_name`` is provider-controlled text sharing the 500-char
        frame budget with the upstream diagnostics. A 300-char attribution
        would push the part that says what actually went wrong off the end,
        making the frame strictly worse than the generic wording it replaced,
        so an implausible name is dropped instead of trusted."""
        body = {
            "error": {
                "message": "Provider returned error",
                "code": 404,
                "metadata": {
                    "raw": json.dumps({"error": {"message": "the real diagnostics"}}),
                    "provider_name": "X" * 300,
                },
            }
        }
        error = self._error(httpx.Response(404, json=body))
        assert "the real diagnostics" in error.message
        assert "XXX" not in error.message

    def test_a_provider_name_cannot_forge_a_second_line(self) -> None:
        """A name carrying a newline would render as a forged extra line in the
        frame, so whitespace is collapsed rather than passed through."""
        body = {
            "error": {
                "message": "Provider returned error",
                "code": 404,
                "metadata": {
                    "raw": json.dumps({"error": {"message": "real"}}),
                    "provider_name": "Meta\nAPPROVED BY: nobody",
                },
            }
        }
        error = self._error(httpx.Response(404, json=body))
        assert "\n" not in error.message

    def test_message_without_provider_name_keeps_its_wording(self) -> None:
        """Attribution is never invented: a relay body that omits
        ``provider_name`` keeps the aggregator's original wording."""
        body = {
            "error": {
                "message": "Provider returned error",
                "code": 404,
                "metadata": {"raw": json.dumps({"error": {"message": "upstream exploded"}})},
            }
        }
        error = self._error(httpx.Response(404, json=body))
        assert error.message == "Provider returned error: upstream exploded"
        assert (error.kind, error.retryable) == ("transient", True)


class TestOpaqueAggregator400InBand:
    """The aggregator's OTHER relay channel for the identical upstream failure.

    Once the gateway has committed HTTP 200 it cannot use the status line, so it
    delivers the same opaque body as an in-band error chunk (the shape
    ``test_openai_compat_mid_stream_error_chunk_unwraps_upstream_raw`` pins for a
    502). Classified from ``code`` alone that body was ``request``/not-retryable
    and killed the turn, so the fix had to reach both channels: which relay the
    aggregator picks is not something the caller can influence, and the two must
    agree."""

    @staticmethod
    def _chunk(raw: str, code: int = 400) -> dict[str, Any]:
        return {
            "id": "gen-1",
            "error": {
                "code": code,
                "message": "Provider returned error",
                "metadata": {"raw": raw, "provider_name": "Stealth"},
            },
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
        }

    async def _stream_error(self, chunk: dict[str, Any]) -> ProviderError:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse([chunk]),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.test.example/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(ProviderError) as excinfo:
            await _collect(
                client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
            )
        return excinfo.value

    async def test_in_band_sentinel_is_transient(self) -> None:
        """The reported body, delivered mid-stream instead of pre-stream."""
        error = await self._stream_error(self._chunk("ERROR"))
        assert error.status == 400
        assert (error.kind, error.retryable) == ("transient", True)
        assert "Stealth returned error" in str(error)

    async def test_in_band_actionable_diagnostics_stay_request(self) -> None:
        """Real upstream diagnostics in-band are still an answer, not weather."""
        chunk = self._chunk(json.dumps({"error": {"message": "context length exceeded"}}))
        error = await self._stream_error(chunk)
        assert (error.kind, error.retryable) == ("request", False)
        assert "context length exceeded" in str(error)

    async def test_in_band_other_400s_are_untouched(self) -> None:
        """A different outer message keeps the status-only classification."""
        chunk = self._chunk("ERROR")
        chunk["error"]["message"] = "Provider error"
        error = await self._stream_error(chunk)
        assert (error.kind, error.retryable) == ("request", False)


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


async def test_google_usage_includes_thinking_and_tool_use_tokens() -> None:
    """Gemini bills counters that sit outside candidates/prompt token counts."""
    body = _sse(
        [
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "cachedContentTokenCount": 80,
                    "candidatesTokenCount": 20,
                    "thoughtsTokenCount": 30,
                    "toolUsePromptTokenCount": 10,
                    "totalTokenCount": 160,
                },
            }
        ]
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    client = GoogleClient(http_client=httpx.AsyncClient(transport=transport))
    events = await _collect(
        client.stream(
            ChatRequest(model=_spec("google", "gemini-2.5-pro"), messages=[Message.user("hi")]),
            "g-key",
        )
    )

    usage = [event.usage for event in events if isinstance(event, StreamUsageEvent)][-1]
    assert usage.input_tokens == 110
    assert usage.context_tokens == 110
    assert usage.cache_read_tokens == 80
    assert usage.output_tokens == 50
    assert usage.reasoning_tokens == 30
    assert usage.total_tokens == 160


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


@pytest.mark.parametrize(
    "provider,model_id",
    [
        # Google: documented looping below 1.0, and "remove this parameter".
        ("google", "gemini-3.8-flash"),
        ("google", "gemini-3-flash"),
        # The aggregator routes to the SAME weights — what a provider-keyed
        # rule would have missed.
        ("openrouter", "google/gemini-3.8-flash"),
        ("radient", "google/gemini-3-pro"),
        # Anthropic 4.7+: a documented HTTP 400 on any value but 1.0. The app
        # sent 0.2, so this row was a live outage.
        ("anthropic", "claude-opus-4-7"),
        # OpenAI GPT-6: "Remove `temperature`, `top_p`, and `top_logprobs`."
        ("openai", "gpt-6-astra"),
        # Accepted-but-inert, pinned, per-series, or undocumented defaults.
        ("deepseek", "deepseek-v4-pro"),
        ("zai", "glm-5.3"),
        ("mistral", "mistral-large-latest"),
        ("xai", "grok-4"),
        # Never overrule a local publisher's Modelfile.
        ("ollama", "qwen3:32b"),
        # The fallback: an uncharacterised model gets silence, not 0.2.
        ("openrouter", "some-vendor/model-nobody-has-characterised"),
    ],
)
@pytest.mark.parametrize("wire,top_p_key", WIRES)
def test_families_that_omit_sampling_send_neither_key(
    wire: str, top_p_key: str, provider: str, model_id: str
) -> None:
    """The keys must be ABSENT from the real body, not null: a provider that
    ignores or rejects a value is not helped by receiving an explicit one, and
    omission is what lets the vendor's own default apply."""
    from local_operator.model.configure import build_model_spec

    spec = build_model_spec(provider, model_id)
    body = _bodies(ChatRequest(model=spec, messages=[Message.user("hi")]))[wire]
    assert "temperature" not in body
    assert top_p_key not in body
    serialised = json.dumps(body)
    assert '"temperature"' not in serialised and f'"{top_p_key}"' not in serialised


@pytest.mark.parametrize(
    "provider,model_id,temperature,top_p",
    [
        # Google's own getModel reports these as the 2.x defaults.
        ("google", "gemini-2.5-pro", 1.0, 0.95),
        # Qwen's per-model table: the pair DIVERGES, which is why the knobs are
        # resolved independently rather than as an all-or-nothing pair.
        ("alibaba", "qwen3-coder-plus", 0.7, 0.8),
    ],
)
@pytest.mark.parametrize("wire,top_p_key", WIRES)
def test_families_that_seed_send_the_vendors_documented_values(
    wire: str, top_p_key: str, provider: str, model_id: str, temperature: float, top_p: float
) -> None:
    """Where a vendor documents a value silence would not reproduce, the pair is
    still sent — at the vendor's number, never the app's invented 0.2/0.9."""
    from local_operator.model.configure import build_model_spec

    spec = build_model_spec(provider, model_id)
    body = _bodies(ChatRequest(model=spec, messages=[Message.user("hi")]))[wire]
    assert body["temperature"] == temperature
    assert body[top_p_key] == top_p


@pytest.mark.parametrize("wire,top_p_key", WIRES)
def test_an_explicit_request_value_still_reaches_the_wire(wire: str, top_p_key: str) -> None:
    """The escape hatch, asserted at the WIRE rather than on the spec: a user
    who deliberately asks for determinism must still get it on a family whose
    default policy is to stay silent."""
    from local_operator.model.configure import build_model_spec

    spec = build_model_spec("google", "gemini-3.8-flash")
    request = ChatRequest(model=spec, messages=[Message.user("hi")], temperature=0.0, top_p=0.5)
    body = _bodies(request)[wire]
    # 0.0 in particular: a falsy value that an `or`-style default would eat.
    assert body["temperature"] == 0.0
    assert body[top_p_key] == 0.5


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


def test_anthropic_omits_sampling_when_adaptive_thinking_is_on() -> None:
    """Thinking and the sampling pair are mutually exclusive on Anthropic's wire.

    ``claude-opus-4-8`` accepts ``temperature`` on its own — it is NOT in
    ``_NO_SAMPLING_PARAMS`` — but it carries an effort ladder, and the ladder is
    what switches ``thinking: {"type": "adaptive"}`` on. Anthropic answers that
    combination with HTTP 400 ``` `temperature` may only be set to 1 when
    thinking is enabled or in adaptive mode ``` (observed live 2026-08-21),
    which killed every turn on the 4.5–4.9 generation. Generation 5+ never hit
    it only because the sampling rule already dropped the pair.

    The guard is keyed on the SAME gate that writes ``thinking`` — not on a
    second model-name list — so any future model that gains an effort ladder is
    automatically safe whichever way the gate resolves.
    """
    spec = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-4-8",
        reasoning=True,
        reasoning_effort="high",
        reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
        temperature=0.2,
        top_p=0.9,
    )
    body = AnthropicClient()._build_body(ChatRequest(model=spec, messages=[Message.user("hi")]))
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    # ABSENT, not null: the 400 is about the key appearing at all.
    assert "temperature" not in body
    assert "top_p" not in body

    # And the flip side: clearing the effort level turns thinking off, which
    # makes the sampling pair legal again — it must come back, because losing
    # a real setting silently would be the worse bug.
    plain = spec.model_copy(update={"reasoning_effort": None})
    body = AnthropicClient()._build_body(ChatRequest(model=plain, messages=[Message.user("hi")]))
    assert "thinking" not in body
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


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


async def test_responses_failed_and_top_level_error_raise_provider_errors() -> None:
    for payload in (
        {
            "type": "response.failed",
            "response": {"error": {"code": "rate_limit_exceeded", "message": "quota gone"}},
        },
        {
            "type": "error",
            "error": {"code": "invalid_api_key", "message": "bad key"},
        },
    ):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=_sse([payload]), headers={"content-type": "text/event-stream"}
            )

        client = OpenAICompatClient(
            "https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            openai_api="responses",
        )
        with pytest.raises(ProviderError) as error:
            await _collect(
                client.stream(
                    ChatRequest(
                        model=_spec("openai", "gpt-5.4").model_copy(
                            update={"supports_responses_api": True}
                        ),
                        messages=[Message.user("hi")],
                    ),
                    "sk-test",
                )
            )
        assert error.value.message


async def test_responses_incomplete_max_output_maps_to_length() -> None:
    payloads = [
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "read"},
        },
        {
            "type": "response.incomplete",
            "response": {
                "id": "r1",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_sse(payloads), headers={"content-type": "text/event-stream"}
        )

    client = OpenAICompatClient(
        "https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        openai_api="responses",
    )
    events = await _collect(
        client.stream(
            ChatRequest(
                model=_spec("openai", "gpt-5.4").model_copy(
                    update={"supports_responses_api": True}
                ),
                messages=[Message.user("hi")],
            ),
            "sk-test",
        )
    )
    end = next(e for e in events if isinstance(e, StreamEndEvent))
    assert end.stop_reason == "length"  # loop will never execute the partial call


async def test_responses_stream_without_terminal_is_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse([{"type": "response.output_text.delta", "delta": "partial"}]),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenAICompatClient(
        "https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        openai_api="responses",
    )
    with pytest.raises(ProviderError, match="without a terminal event"):
        await _collect(
            client.stream(
                ChatRequest(
                    model=_spec("openai", "gpt-5.4").model_copy(
                        update={"supports_responses_api": True}
                    ),
                    messages=[Message.user("hi")],
                ),
                "sk-test",
            )
        )


def test_responses_tool_image_output_stays_native_image_content() -> None:
    from local_operator.providers.clients import _messages_to_openai_responses

    message = Message(
        role="tool",
        tool_call_id="c1",
        tool_name="read",
        content=[
            TextContent(text="screenshot"),
            ImageContent(data="aGVsbG8=", mime_type="image/png"),
        ],
    )
    output = _messages_to_openai_responses([message])[0]["output"]
    assert output == [
        {"type": "input_text", "text": "screenshot"},
        {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
    ]


def _truncated_call() -> ToolCall:
    """A tool call as the loop stores it when a turn dies mid-stream.

    The argument deltas are concatenated verbatim, so an aborted call leaves a
    JSON fragment in ``raw_arguments`` and an empty ``arguments`` (the loop
    could not parse it either). Copied from a real session transcript.
    """
    return ToolCall(id="t1", name="write", arguments={}, raw_arguments='{"path": "/tmp/x.py"')


def _assistant_with(calls: list[ToolCall]) -> Message:
    return Message(role="assistant", content=[TextContent(text="ok")], tool_calls=calls)


def test_anthropic_body_survives_truncated_tool_arguments() -> None:
    """A mid-stream abort must not brick every later turn in the session.

    The fragment is written to the transcript and replayed on EVERY subsequent
    request, so parsing it unguarded raised JSONDecodeError out of body
    construction. Failover could only read that as a transient provider fault,
    retried, and rebuilt the identical body — the session became unusable and
    blamed the provider for its own corrupt row.
    """
    request = ChatRequest(
        model=_spec(provider="anthropic"),
        messages=[
            Message.user("hi"),
            _assistant_with([_truncated_call()]),
            Message(
                role="tool",
                tool_call_id="t1",
                tool_name="write",
                content=[TextContent(text="aborted")],
                is_error=True,
            ),
        ],
    )
    body = AnthropicClient()._build_body(request)
    tool_use = [b for b in body["messages"][1]["content"] if b.get("type") == "tool_use"]
    assert tool_use[0]["input"] == {}


def test_openai_replays_valid_json_for_truncated_tool_arguments() -> None:
    """The OpenAI shapes send the argument STRING, so a fragment goes on the
    wire as invalid JSON and the provider rejects the request — the same dead
    session by a longer route."""
    from local_operator.providers.clients import _messages_to_openai_responses

    message = _assistant_with([_truncated_call()])
    arguments = _message_to_openai(message)["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {}

    calls = [
        i for i in _messages_to_openai_responses([message]) if i.get("type") == "function_call"
    ]
    assert json.loads(calls[0]["arguments"]) == {}


def test_wire_clients_replay_well_formed_raw_arguments_verbatim() -> None:
    """Byte fidelity is the reason raw_arguments exists: a model reading back
    its own call must see its own bytes, non-canonical spacing included. Only
    unparseable strings are salvaged."""
    from local_operator.providers.clients import _messages_to_openai_responses

    raw = '{"command":  "ls"}'
    call = ToolCall(id="t2", name="bash", arguments={"command": "ls"}, raw_arguments=raw)
    message = _assistant_with([call])

    assert _message_to_openai(message)["tool_calls"][0]["function"]["arguments"] == raw
    calls = [
        i for i in _messages_to_openai_responses([message]) if i.get("type") == "function_call"
    ]
    assert calls[0]["arguments"] == raw

    request = ChatRequest(model=_spec(provider="anthropic"), messages=[message])
    body = AnthropicClient()._build_body(request)
    tool_use = [b for b in body["messages"][0]["content"] if b.get("type") == "tool_use"]
    assert tool_use[0]["input"] == {"command": "ls"}


def test_empty_raw_arguments_never_reach_the_wire_as_an_empty_body() -> None:
    """An empty string is not valid JSON and must not be replayed as one.

    Guards the salvage against the failure it exists to prevent: a check
    written as ``json.loads(raw or "{}")`` validates the placeholder and then
    returns ``raw``, so an empty string passes and goes out as an empty body.
    The assembler normalizes empty to None today, but the field is typed
    ``str | None`` and transcripts are external input.
    """
    from local_operator.providers.clients import _messages_to_openai_responses

    call = ToolCall(id="t5", name="bash", arguments={"command": "ls"}, raw_arguments="")
    message = _assistant_with([call])

    arguments = _message_to_openai(message)["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"command": "ls"}

    calls = [
        i for i in _messages_to_openai_responses([message]) if i.get("type") == "function_call"
    ]
    assert json.loads(calls[0]["arguments"]) == {"command": "ls"}

    request = ChatRequest(model=_spec(provider="anthropic"), messages=[message])
    body = AnthropicClient()._build_body(request)
    tool_use = [b for b in body["messages"][0]["content"] if b.get("type") == "tool_use"]
    assert tool_use[0]["input"] == {"command": "ls"}


def test_non_object_raw_arguments_fall_back_to_parsed_arguments() -> None:
    """A bare string or list parses cleanly but is not a legal argument object;
    it is as unusable on the wire as a fragment."""
    call = ToolCall(id="t4", name="x", arguments={"a": 1}, raw_arguments='"hello"')
    arguments = _message_to_openai(_assistant_with([call]))["tool_calls"][0]["function"][
        "arguments"
    ]
    assert json.loads(arguments) == {"a": 1}


# ---------------------------------------------------------------------------
# Refusal surfacing: the provider said no, and the user must see its words
# ---------------------------------------------------------------------------


class TestRefusalsAreSurfacedNotSwallowed:
    """A provider refusal used to end the turn as a clean ``stop``: nothing on
    screen, no explanation, indistinguishable from a no-op. Every wire's
    refusal marker must map to ``stop_reason="refusal"`` with the provider's
    own message (or a line naming the marker, when the wire sends no prose)
    in ``StreamEndEvent.error`` — that message is what lets the user decide
    between rephrasing and switching models.
    """

    @staticmethod
    def _end(events: list[Any]) -> StreamEndEvent:
        end = events[-1]
        assert isinstance(end, StreamEndEvent)
        return end

    async def test_chat_completions_content_filter_is_a_refusal(self) -> None:
        """``finish_reason=content_filter`` (Azure/OpenRouter-style filters)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse([{"choices": [{"delta": {}, "finish_reason": "content_filter"}]}]),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.test.example/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        events = await _collect(
            client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "content_filter" in end.error

    async def test_chat_completions_refusal_delta_carries_the_providers_words(self) -> None:
        """OpenAI streams refusal prose in ``delta.refusal`` with
        ``finish_reason=stop`` — the prose slot, not the finish reason, is the
        signal, and its text must reach the frame verbatim."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"choices": [{"delta": {"refusal": "I can't help "}}]},
                        {
                            "choices": [
                                {"delta": {"refusal": "with that."}, "finish_reason": "stop"}
                            ]
                        },
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.test.example/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        events = await _collect(
            client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
        )
        # Refusal prose is not an answer: it must not have streamed as text.
        assert not [e for e in events if isinstance(e, StreamTextDelta)]
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "I can't help with that." in end.error

    async def test_responses_content_filter_is_a_refusal_not_a_provider_error(self) -> None:
        """``response.incomplete`` with ``reason=content_filter`` used to raise
        ProviderError, sending a content decline into transport-retry
        machinery. It is a refusal: terminal, visible, not retried."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {
                            "type": "response.incomplete",
                            "response": {
                                "id": "resp_1",
                                "incomplete_details": {"reason": "content_filter"},
                                "usage": {"input_tokens": 5, "output_tokens": 0},
                            },
                        }
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            openai_api="responses",
        )
        events = await _collect(
            client.stream(
                ChatRequest(
                    model=_spec("openai", "gpt-5.4").model_copy(
                        update={"supports_responses_api": True}
                    ),
                    messages=[Message.user("hi")],
                ),
                "sk-test",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "content_filter" in end.error

    async def test_responses_refusal_item_beats_the_completed_terminal(self) -> None:
        """A ``response.completed`` whose only output was a refusal item says
        "completed" on the wire and "no" in the content; the content wins."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"type": "response.refusal.delta", "delta": "I won't do that."},
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "resp_2",
                                "usage": {"input_tokens": 5, "output_tokens": 3},
                            },
                        },
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            openai_api="responses",
        )
        events = await _collect(
            client.stream(
                ChatRequest(
                    model=_spec("openai", "gpt-5.4").model_copy(
                        update={"supports_responses_api": True}
                    ),
                    messages=[Message.user("hi")],
                ),
                "sk-test",
            )
        )
        assert not [e for e in events if isinstance(e, StreamTextDelta)]
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "I won't do that." in end.error

    async def test_anthropic_refusal_stop_reason_is_mapped_and_named(self) -> None:
        """Anthropic's documented ``stop_reason: refusal`` passed through the
        terminal map unmapped, and downstream read the unknown value as a
        clean stop."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = _sse(
                [
                    {
                        "type": "message_start",
                        "message": {"usage": {"input_tokens": 7}},
                    },
                    {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {}},
                ]
            )
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        client = AnthropicClient(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec("anthropic", "claude-x"), messages=[Message.user("hi")]),
                "sk-ant",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "stop_reason=refusal" in end.error

    @pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "OTHER"])
    async def test_google_safety_finishes_are_refusals(self, reason: str) -> None:
        """Every non-STOP/MAX_TOKENS/TOOL_USE Gemini finishReason is a refusal,
        and the visible line names WHICH classifier fired."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse([{"candidates": [{"finishReason": reason}]}]),
                headers={"content-type": "text/event-stream"},
            )

        client = GoogleClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec("google", "gemini-2.5-pro"), messages=[Message.user("hi")]),
                "g-key",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and reason in end.error

    async def test_google_blocked_prompt_is_a_refusal(self) -> None:
        """A blocked prompt yields NO candidates, only promptFeedback — the
        exact shape that used to end as the silent ``stop`` default."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse([{"promptFeedback": {"blockReason": "SAFETY"}}]),
                headers={"content-type": "text/event-stream"},
            )

        client = GoogleClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec("google", "gemini-2.5-pro"), messages=[Message.user("hi")]),
                "g-key",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "blockReason=SAFETY" in end.error

    async def test_google_normal_finishes_are_untouched(self) -> None:
        """The refusal mapping must not reclassify the three normal finishes."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {
                            "candidates": [
                                {"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}
                            ]
                        }
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = GoogleClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec("google", "gemini-2.5-pro"), messages=[Message.user("hi")]),
                "g-key",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "stop"
        assert end.error is None

    async def test_mock_refuse_trigger_emits_the_full_shape(self) -> None:
        """``[refuse]`` exists so the whole path can be exercised against a
        real running app; it must produce the same event shape a real wire
        does."""
        client = MockClient()
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec("test", "m"), messages=[Message.user("[refuse] hi")]),
                None,
            )
        )
        end = events[-1]
        assert isinstance(end, StreamEndEvent)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "can't help" in end.error

    async def test_chat_completions_refusal_truncated_by_length_still_surfaces(self) -> None:
        """Refusal prose cut by the token cap ended as a bare ``length`` with
        the collected prose dropped — the swallowed-refusal shape again
        (review R1-3)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"choices": [{"delta": {"refusal": "I can't hel"}}]},
                        {"choices": [{"delta": {}, "finish_reason": "length"}]},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.test.example/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        events = await _collect(
            client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "I can't hel" in end.error

    async def test_responses_refusal_truncated_by_output_cap_still_surfaces(self) -> None:
        """Same gap on the Responses wire: ``response.incomplete`` with
        ``reason=max_output_tokens`` after refusal deltas (review R1-3)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"type": "response.refusal.delta", "delta": "I won't"},
                        {
                            "type": "response.incomplete",
                            "response": {
                                "id": "resp_3",
                                "incomplete_details": {"reason": "max_output_tokens"},
                                "usage": {"input_tokens": 5, "output_tokens": 2},
                            },
                        },
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            openai_api="responses",
        )
        events = await _collect(
            client.stream(
                ChatRequest(
                    model=_spec("openai", "gpt-5.4").model_copy(
                        update={"supports_responses_api": True}
                    ),
                    messages=[Message.user("hi")],
                ),
                "sk-test",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "I won't" in end.error

    async def test_responses_plain_length_is_still_length(self) -> None:
        """The R1-3 fix must not reclassify an ordinary truncation: no refusal
        deltas means ``max_output_tokens`` stays ``length``."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"type": "response.output_text.delta", "delta": "partial answ"},
                        {
                            "type": "response.incomplete",
                            "response": {
                                "id": "resp_4",
                                "incomplete_details": {"reason": "max_output_tokens"},
                                "usage": {"input_tokens": 5, "output_tokens": 2},
                            },
                        },
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            openai_api="responses",
        )
        events = await _collect(
            client.stream(
                ChatRequest(
                    model=_spec("openai", "gpt-5.4").model_copy(
                        update={"supports_responses_api": True}
                    ),
                    messages=[Message.user("hi")],
                ),
                "sk-test",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "length"
        assert end.error is None

    @pytest.mark.parametrize("reason", ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"])
    async def test_google_model_defects_are_errors_not_refusals(self, reason: str) -> None:
        """Gemini's tooling-defect finishes are not content refusals: calling
        them refusals steers the user to rephrase/switch when a plain retry
        usually works (review R1-2). They end as ``error`` naming the marker."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse([{"candidates": [{"finishReason": reason}]}]),
                headers={"content-type": "text/event-stream"},
            )

        client = GoogleClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec("google", "gemini-2.5-pro"), messages=[Message.user("hi")]),
                "g-key",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "error"
        assert end.error is not None and reason in end.error and "refused" not in end.error

    async def test_anthropic_refusal_after_partial_prose_says_cut_short(self) -> None:
        """Design review D1: 'sent no message' directly under a partially
        rendered answer asserts the opposite of what is on screen. When text
        streamed before the refusal terminal, the line says the reply was cut."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = _sse(
                [
                    {"type": "message_start", "message": {"usage": {"input_tokens": 7}}},
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Here is the beginning of"},
                    },
                    {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {}},
                ]
            )
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        client = AnthropicClient(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec("anthropic", "claude-x"), messages=[Message.user("hi")]),
                "sk-ant",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None
        assert "cut the reply short" in end.error
        assert "sent no message" not in end.error

    async def test_google_refusal_after_partial_prose_says_cut_short(self) -> None:
        """Same D1 state on the Gemini wire, where safety stops commonly cut a
        partial answer."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"candidates": [{"content": {"parts": [{"text": "Starting to answ"}]}}]},
                        {"candidates": [{"finishReason": "SAFETY"}]},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = GoogleClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        events = await _collect(
            client.stream(
                ChatRequest(model=_spec("google", "gemini-2.5-pro"), messages=[Message.user("hi")]),
                "g-key",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None and "cut the reply short" in end.error

    async def test_prose_slot_refusal_names_the_slot_not_just_the_finish(self) -> None:
        """Design review D2: '(finish_reason=stop)' beside the word 'refused'
        names a mechanism that did not fire. When the prose slot detected the
        refusal, the marker names the slot."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [{"choices": [{"delta": {"refusal": "No."}, "finish_reason": "stop"}]}]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.test.example/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        events = await _collect(
            client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
        )
        end = self._end(events)
        assert end.error is not None and "delta.refusal" in end.error

    async def test_chat_content_filter_after_prose_says_cut_short(self) -> None:
        """Review R3-1: a third-party filter (Azure-style) commonly terminates
        with ``content_filter`` AFTER answer chunks rendered and sends no
        refusal prose — 'sent no message' under that partial reply is the D1
        contradiction on this wire."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"choices": [{"delta": {"content": "The document says"}}]},
                        {"choices": [{"delta": {}, "finish_reason": "content_filter"}]},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.test.example/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        events = await _collect(
            client.stream(ChatRequest(model=_spec(), messages=[Message.user("hi")]), "sk-test")
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None
        assert "cut the reply short" in end.error
        assert "sent no message" not in end.error

    async def test_responses_content_filter_after_prose_says_cut_short(self) -> None:
        """Same R3-1 state on the Responses wire: output text streamed, then
        ``response.incomplete`` with ``reason=content_filter`` and no refusal
        deltas."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"type": "response.output_text.delta", "delta": "The document says"},
                        {
                            "type": "response.incomplete",
                            "response": {
                                "id": "resp_5",
                                "incomplete_details": {"reason": "content_filter"},
                                "usage": {"input_tokens": 5, "output_tokens": 3},
                            },
                        },
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAICompatClient(
            "https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            openai_api="responses",
        )
        events = await _collect(
            client.stream(
                ChatRequest(
                    model=_spec("openai", "gpt-5.4").model_copy(
                        update={"supports_responses_api": True}
                    ),
                    messages=[Message.user("hi")],
                ),
                "sk-test",
            )
        )
        end = self._end(events)
        assert end.stop_reason == "refusal"
        assert end.error is not None
        assert "cut the reply short" in end.error
        assert "sent no message" not in end.error


# Empty assistant turns (errored/aborted model turns) must not reach the wire
# ---------------------------------------------------------------------------


def _empty_assistant() -> Message:
    """The message an errored model turn persists: no text, no tool calls.

    `harness/loop.py` appends the assistant message to the context BEFORE the
    stream finishes; a stream that dies without a single token leaves exactly
    this shape in the transcript, and every later request replays it.
    """
    return Message(role="assistant", content=[], stop_reason="error")


def test_openai_chat_drops_empty_assistant_turns() -> None:
    """Moonshot/Kimi rejects the whole request over ONE empty assistant turn:
    HTTP 400 "the message at position N with role 'assistant' must not be
    empty" (observed live, 2026-08-19). The turn carries no information, so it
    is dropped at body-build time — which also repairs existing transcripts
    that already contain one."""
    request = ChatRequest(
        model=_spec(),
        messages=[Message.user("hi"), _empty_assistant(), Message.user("continue")],
    )
    body = OpenAICompatClient("https://x")._build_body(request)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "user"]


def test_openai_chat_drops_whitespace_only_assistant_turns() -> None:
    """Whitespace renders to content a strict provider may still reject, and
    a model cannot read anything from it — same drop as truly empty."""
    request = ChatRequest(
        model=_spec(),
        messages=[
            Message.user("hi"),
            Message(role="assistant", content=[TextContent(text="  \n")]),
        ],
    )
    body = OpenAICompatClient("https://x")._build_body(request)
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_openai_chat_keeps_assistant_turns_with_tool_calls_or_content() -> None:
    """The drop must be surgical: a tool-call-only assistant turn is NOT empty
    (the calls are the content, and a paired tool message references them), and
    normal text turns pass through untouched."""
    tool_turn = _assistant_with([ToolCall(id="t1", name="x", arguments={})])
    request = ChatRequest(
        model=_spec(),
        messages=[
            Message.user("hi"),
            tool_turn,
            Message(role="tool", tool_call_id="t1", content=[TextContent(text="done")]),
            Message.assistant("all set"),
        ],
    )
    body = OpenAICompatClient("https://x")._build_body(request)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_openai_responses_drops_empty_assistant_turns() -> None:
    """Same normalization on the Responses route: an errored turn's empty
    assistant message must not become an empty output_text item."""
    from local_operator.providers.clients import _messages_to_openai_responses

    items = _messages_to_openai_responses(
        [Message.user("hi"), _empty_assistant(), Message.user("continue")]
    )
    assert [i.get("role") for i in items] == ["user", "user"]


def test_anthropic_drops_empty_assistant_turns() -> None:
    """Anthropic 400s on an assistant message with an empty content array —
    the exact serialization of an errored model turn."""
    request = ChatRequest(
        model=_spec(provider="anthropic", model_id="claude-sonnet-4-5"),
        messages=[Message.user("hi"), _empty_assistant(), Message.user("continue")],
    )
    body = AnthropicClient()._build_body(request)
    assert [m["role"] for m in body["messages"]] == ["user", "user"]


def test_google_drops_whitespace_only_assistant_turns() -> None:
    """Google's builder already skipped no-part assistant turns; the shared
    predicate extends that to whitespace-only text so all three clients agree
    on what an empty assistant turn is."""
    request = ChatRequest(
        model=_spec(provider="google", model_id="gemini-2.0-flash"),
        messages=[
            Message.user("hi"),
            Message(role="assistant", content=[TextContent(text=" ")]),
            _empty_assistant(),
        ],
    )
    body = GoogleClient()._build_body(request)
    assert [c["role"] for c in body["contents"]] == ["user"]


def test_empty_assistant_with_image_content_is_kept() -> None:
    """An image block is content even with no text: dropping it would lose
    information. (No current model emits assistant images, but the predicate
    must not assume that.)"""
    from local_operator.providers.clients import _is_empty_assistant

    message = Message(
        role="assistant",
        content=[ImageContent(data="Zm9v", mime_type="image/png")],
    )
    assert not _is_empty_assistant(message)


def test_tool_call_turns_do_not_ship_whitespace_only_text() -> None:
    """F1 (PR #189 review round 1): the predicate treats whitespace as empty,
    so a tool-call turn must not ship a whitespace-only `content` either —
    the two paths have to agree about what counts as content."""
    from local_operator.providers.clients import _messages_to_openai_responses

    message = Message(
        role="assistant",
        content=[TextContent(text="  \n")],
        tool_calls=[ToolCall(id="t9", name="x", arguments={})],
    )
    chat = _message_to_openai(message)
    assert "content" not in chat
    assert chat["tool_calls"]

    items = _messages_to_openai_responses([message])
    assert [i.get("type") for i in items] == ["function_call"]


# ---------------------------------------------------------------------------
# GPT-5.6 prompt-caching (defects #1/#2 only): prompt_cache_key kept on the
# Codex ``store:false`` path, and encrypted-reasoning ``include`` requested when
# reasoning is on. Defect #3 (prompt_cache_breakpoint / prompt_cache_options /
# moving the stable prefix into developer messages) was REJECTED by the ChatGPT
# subscription Codex backend with HTTP 400 (OpenAI Codex bug #35300) and removed
# entirely; these tests are regression guards against it coming back.
# ---------------------------------------------------------------------------


def _gpt_5_6_spec(model_id: str = "gpt-5.6-sol") -> ModelSpec:
    """A reasoning GPT-5.6 Responses spec with prompt caching enabled."""
    return ModelSpec(
        provider="openai",
        model_id=model_id,
        supports_responses_api=True,
        supports_prompt_cache=True,
        reasoning=True,
        reasoning_effort="high",
        reasoning_efforts=("low", "medium", "high"),
        supports_sampling_params=False,
    )


def _has_breakpoint(items: list[dict[str, Any]]) -> bool:
    """Whether any content block in ``items`` carries a prompt_cache_breakpoint.

    Used purely as a regression guard: after defect #3 was removed, NO body we
    build may contain this field (the Codex backend 400s on it).
    """
    for item in items:
        for key in ("content", "output"):
            blocks = item.get(key)
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if isinstance(block, dict) and "prompt_cache_breakpoint" in block:
                    return True
    return False


def test_codex_5_6_body_keeps_cache_key_and_include() -> None:
    """Defects #1/#2 on the Codex ``store:false`` path: prompt_cache_key is kept
    (Codex parity), encrypted reasoning is requested, and the public-retention
    field is stripped. The stable prefix stays in top-level ``instructions`` and
    NONE of the rejected defect-#3 fields appear."""
    client = OpenAICompatClient("https://api.openai.com/v1")
    request = ChatRequest(
        model=_gpt_5_6_spec(),
        system_blocks=["Stable instructions.", "Tool inventory.", "Volatile env + date."],
        messages=[Message.user("hi")],
        prompt_cache_key="session-abc",
    )
    body = client._build_codex_responses_body(request)

    # #1: the key that was wrongly stripped is present; retention is not.
    assert body["prompt_cache_key"] == "session-abc"
    assert "prompt_cache_retention" not in body
    assert body["store"] is False
    # #2: reasoning is on, so encrypted reasoning content is requested.
    assert body["include"] == ["reasoning.encrypted_content"]
    # Stable prefix rides top-level ``instructions`` exactly as real Codex sends
    # it; nothing is moved into an injected developer message.
    assert body["instructions"] == "Stable instructions.\n\nTool inventory.\n\nVolatile env + date."
    assert not any(i.get("role") == "developer" for i in body["input"])
    # Defect #3 regression guard: the backend 400s on both of these.
    assert "prompt_cache_options" not in body
    assert not _has_breakpoint(body["input"])


def test_codex_5_6_body_no_include_when_reasoning_off() -> None:
    """Defect #2 is gated on reasoning: a 5.6 model with no effort ladder must
    not request encrypted reasoning content."""
    spec = _gpt_5_6_spec()
    spec = spec.model_copy(update={"reasoning_efforts": (), "reasoning_effort": None})
    client = OpenAICompatClient("https://api.openai.com/v1")
    body = client._build_codex_responses_body(
        ChatRequest(model=spec, system_blocks=["a", "b"], messages=[Message.user("hi")])
    )
    assert "include" not in body
    # #1 still holds regardless of reasoning: the codex path keeps the key.
    # Defect #3 fields never appear.
    assert "prompt_cache_options" not in body
    assert not _has_breakpoint(body["input"])
    assert body["instructions"] == "a\n\nb"


def test_public_5_6_body_keeps_original_shape_plus_include() -> None:
    """The public Responses path returns its ORIGINAL body plus only defect #2's
    ``include``: top-level instructions, prompt_cache_key, and the 24h retention
    are all present, and none of the rejected defect-#3 fields appear."""
    client = OpenAICompatClient("https://api.openai.com/v1")
    body = client._build_responses_body(
        ChatRequest(
            model=_gpt_5_6_spec(),
            system_blocks=["Stable.", "Volatile."],
            messages=[Message.user("hi")],
            prompt_cache_key="pub-1",
        )
    )
    assert body["prompt_cache_key"] == "pub-1"
    assert body["prompt_cache_retention"] == "24h"
    assert body["instructions"] == "Stable.\n\nVolatile."
    assert body["include"] == ["reasoning.encrypted_content"]
    assert "prompt_cache_options" not in body
    assert not _has_breakpoint(body["input"])


def test_public_pre_5_6_body_unchanged() -> None:
    """Regression guard: a non-reasoning model keeps top-level instructions,
    prompt_cache_retention, no ``include``, and no breakpoints."""
    spec = ModelSpec(
        provider="openai",
        model_id="gpt-5.4",
        supports_responses_api=True,
        supports_prompt_cache=True,
    )
    client = OpenAICompatClient("https://api.openai.com/v1")
    body = client._build_responses_body(
        ChatRequest(
            model=spec,
            system_blocks=["Stable.", "Volatile."],
            messages=[Message.user("hi")],
            prompt_cache_key="pub-2",
        )
    )
    assert body["instructions"] == "Stable.\n\nVolatile."
    assert body["prompt_cache_retention"] == "24h"
    assert "include" not in body
    assert "prompt_cache_options" not in body
    assert not _has_breakpoint(body["input"])


async def test_codex_5_6_stream_skips_reasoning_items() -> None:
    """Defect #2 stream side: encrypted ``reasoning`` output items and their
    deltas are dropped, never rendered as assistant text, and do not crash the
    parser."""
    events_sse = _sse(
        [
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "gAAAAAB-opaque-blob",
                },
            },
            {"type": "response.reasoning_summary_text.delta", "delta": "ignored thinking"},
            {"type": "response.output_text.delta", "delta": "Answer"},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_r",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 5},
                    },
                },
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=events_sse, headers={"content-type": "text/event-stream"}
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
            ChatRequest(model=_gpt_5_6_spec(), messages=[Message.user("hi")]),
            "chatgpt-token",
            oauth_access=access,
        )
    )
    texts = [e.delta for e in events if isinstance(e, StreamTextDelta)]
    assert texts == ["Answer"]  # the encrypted blob never became text
    assert events[-1].stop_reason == "stop"


# ---------------------------------------------------------------------------
# Aside cache-prefix regression: tools + tool_choice="none"
#
# `Session.complete_aside` (the /btw overlay and the /loop goal-mode judge)
# sends the SAME live tool schema a working turn sends so the aside rides the
# turn's cached prefix instead of re-processing the whole conversation. The
# tools block is the FRONT of every provider's cache prefix, so a future
# refactor that silently dropped `body["tools"]` for a `tool_choice="none"`
# request would reintroduce the exact prompt-cache regression these tests
# guard. Each of the three wires must emit the tools. The OpenAI and Gemini
# wires pin the choice to "none" (the aside reads the turn and calls nothing);
# Anthropic deliberately does NOT — see the block below.
# ---------------------------------------------------------------------------


async def _aside_execute(*_args: Any, **_kwargs: Any) -> ToolResult:
    """Never invoked: an aside sends tools but calls none."""
    raise AssertionError("aside tool must not be executed")


def _aside_tools() -> list[AgentTool]:
    return [
        AgentTool(
            name="bash",
            description="Run a shell command",
            parameters={"type": "object", "properties": {}},
            execute=_aside_execute,
        ),
        AgentTool(
            name="read",
            description="Read a file",
            parameters={"type": "object", "properties": {}},
            execute=_aside_execute,
        ),
    ]


def test_openai_compat_emits_tools_with_tool_choice_none() -> None:
    request = ChatRequest(
        model=_spec(provider="openai"),
        messages=[Message.user("why?")],
        tools=_aside_tools(),
        tool_choice="none",
    )
    body = OpenAICompatClient("https://x")._build_body(request)
    assert [t["function"]["name"] for t in body["tools"]] == ["bash", "read"]
    assert body["tool_choice"] == "none"


def test_openai_responses_emits_tools_with_tool_choice_none() -> None:
    request = ChatRequest(
        model=_spec(provider="openai"),
        messages=[Message.user("why?")],
        tools=_aside_tools(),
        tool_choice="none",
    )
    body = OpenAICompatClient("https://x")._build_responses_body(request)
    assert [t["name"] for t in body["tools"]] == ["bash", "read"]
    assert body["tool_choice"] == "none"


def test_anthropic_emits_tools_with_tool_choice_none_and_keeps_cache_control() -> None:
    request = ChatRequest(
        model=_spec(provider="anthropic"),
        system_blocks=["instructions", "env"],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("why?")],
        tools=_aside_tools(),
        tool_choice="none",
    )
    body = AnthropicClient()._build_body(request)
    assert [t["name"] for t in body["tools"]] == ["bash", "read"]
    # NOT {"type": "none"}: see test_anthropic_tool_choice_none_with_tools_rides_the_turns_auto.
    assert body["tool_choice"] == {"type": "auto"}
    # The message-tail cache_control placement is unchanged by restoring tools:
    # the tools block sits AHEAD of system+messages in the prefix, so the
    # existing breakpoint policy (last block of the final message, and the
    # prior user turn) must still hold.
    assert "cache_control" in body["messages"][-1]["content"][-1]
    assert "cache_control" in body["messages"][0]["content"][-1]


# ---------------------------------------------------------------------------
# Anthropic tool_choice: "none" with tools present goes out as the turn's
# "auto".
#
# Anthropic renders `tool_choice` into the MESSAGES level of its cache
# hierarchy (docs, "What invalidates the cache": Tool choice — tools ✓,
# system ✓, messages ✘). An aside/advisor sending `{"type": "none"}` against a
# working turn that sent `{"type": "auto"}` therefore kept only the
# tools+system head warm and re-wrote every message block — measured at a
# quarter of all daily cache-write tokens across the fleet. The fix is for the
# aside's body to be byte-identical to the turn's prefix, which means the same
# `tool_choice`; the "calls nothing" contract is enforced by the callers
# ignoring tool-call deltas (`Session.complete_aside`,
# `Session.advise_compaction`). The no-tools callers never reach the branch and
# must keep carrying no `tool_choice` key at all.
# ---------------------------------------------------------------------------


def _anthropic_request(
    tool_choice: Literal["auto", "none", "required"], *, tools: list[AgentTool]
) -> ChatRequest:
    return ChatRequest(
        model=_spec(provider="anthropic"),
        system_blocks=["instructions"],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("why?")],
        tools=tools,
        tool_choice=tool_choice,
    )


def test_anthropic_tool_choice_none_with_tools_rides_the_turns_auto() -> None:
    """The aside body must equal the turn body: same tools, same tool_choice."""
    client = AnthropicClient()
    turn = client._build_body(_anthropic_request("auto", tools=_aside_tools()))
    aside = client._build_body(_anthropic_request("none", tools=_aside_tools()))
    assert aside["tool_choice"] == {"type": "auto"}
    # Byte-identical prefix, not merely an equal tool_choice: any other key
    # that differed would break the messages cache just the same.
    assert aside == turn


def test_anthropic_tool_choice_mapping_with_tools() -> None:
    cases: list[tuple[Literal["auto", "none", "required"], dict[str, str]]] = [
        ("auto", {"type": "auto"}),
        ("none", {"type": "auto"}),
        ("required", {"type": "any"}),
    ]
    for choice, expected in cases:
        body = AnthropicClient()._build_body(_anthropic_request(choice, tools=_aside_tools()))
        assert body["tool_choice"] == expected, choice


def test_anthropic_tool_choice_none_without_tools_sends_no_tool_choice_key() -> None:
    """Naming, the compaction summary and the server operator send tools=[]
    with tool_choice="none"; they carry neither key, before and after."""
    body = AnthropicClient()._build_body(_anthropic_request("none", tools=[]))
    assert "tools" not in body
    assert "tool_choice" not in body


def test_other_wires_keep_a_literal_none_with_tools() -> None:
    """The mapping is Anthropic-only. OpenAI-compatible wires document no
    cache penalty for tool_choice, and Gemini's mode MUST stay NONE because
    its default with tools present is to allow calls."""
    openai_request = ChatRequest(
        model=_spec(provider="openai"),
        messages=[Message.user("why?")],
        tools=_aside_tools(),
        tool_choice="none",
    )
    assert OpenAICompatClient("https://x")._build_body(openai_request)["tool_choice"] == "none"
    assert (
        OpenAICompatClient("https://x")._build_responses_body(openai_request)["tool_choice"]
        == "none"
    )
    google_request = ChatRequest(
        model=_spec(provider="google", model_id="gemini-2.5-pro"),
        messages=[Message.user("why?")],
        tools=_aside_tools(),
        tool_choice="none",
    )
    body = GoogleClient()._build_body(google_request)
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"


def test_google_emits_tools_and_pins_function_calling_to_none() -> None:
    """Gemini has no tool_choice field; it takes the mode under
    toolConfig.functionCallingConfig. Before the aside sent tools this branch
    was never reached with tools present, so the mode defaulted to AUTO. Now
    that an aside sends the live tools with tool_choice="none", the mode MUST
    be pinned to NONE or the aside could newly call a tool."""
    request = ChatRequest(
        model=_spec(provider="google", model_id="gemini-2.5-pro"),
        messages=[Message.user("why?")],
        tools=_aside_tools(),
        tool_choice="none",
    )
    body = GoogleClient()._build_body(request)
    decls = body["tools"][0]["function_declarations"]
    assert [d["name"] for d in decls] == ["bash", "read"]
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"


def test_google_function_calling_mode_maps_auto_and_required() -> None:
    cases: list[tuple[Literal["auto", "none", "required"], str]] = [
        ("auto", "AUTO"),
        ("required", "ANY"),
        ("none", "NONE"),
    ]
    for choice, mode in cases:
        body = GoogleClient()._build_body(
            ChatRequest(
                model=_spec(provider="google", model_id="gemini-2.5-pro"),
                messages=[Message.user("hi")],
                tools=_aside_tools(),
                tool_choice=choice,
            )
        )
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == mode


# ---------------------------------------------------------------------------
# Anthropic 1-hour prompt-cache TTL for large contexts
# ---------------------------------------------------------------------------


def _every_cache_control(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Every ``cache_control`` marker in a body, system blocks and messages.

    The wire constraint is request-wide (a 1h marker may not follow a 5m one),
    so the assertions below inspect EVERY marker rather than sampling one.
    """
    markers = [e["cache_control"] for e in body.get("system", []) if "cache_control" in e]
    for message in body["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            markers.extend(b["cache_control"] for b in content if "cache_control" in b)
    return markers


def _large_context_request() -> ChatRequest:
    """A request whose TTL tests drive ``context_tokens_hint`` up to 900k.

    The window is stated explicitly (1M, Claude's real size) so these tests
    exercise the TTL policy on a session that plausibly holds that much, rather
    than inheriting the ``ModelSpec`` default of 128k.

    An earlier revision justified this by claiming a hint larger than the window
    "cannot occur in production". **That claim was false** and is corrected here
    rather than deleted, because it is the kind of premise someone later widens
    a refusal on: two real paths strand a hint from a bigger model —
    ``Session.set_model`` swaps the spec on a ``/model`` down-switch without
    clearing ``_context_tokens_hint``, and the failover clone keeps the primary's
    hint while moving to a smaller fallback spec. That case is covered directly
    by ``test_a_hint_larger_than_the_window_is_not_believed`` and by
    ``test_anthropic_ttl_hint_above_a_small_window_still_marks_1h`` below; the
    window here is large only so the TTL assertions stay about TTL.
    """
    return ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=1_000_000),
        system_blocks=["instructions", "inventory", "skills", "env"],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("second")],
    )


def test_anthropic_ttl_off_sends_no_ttl_keys() -> None:
    """Threshold 0 (and the constructor default) keeps the historical body:
    markers exist, none carries a ``ttl`` — even with a huge hint."""
    request = _large_context_request().model_copy(update={"context_tokens_hint": 900_000})
    for client in (AnthropicClient(), AnthropicClient(cache_ttl_1h_min_context_tokens=0)):
        body = client._build_body(request)
        markers = _every_cache_control(body)
        assert markers, "the breakpoint policy itself must be untouched"
        assert all(m == {"type": "ephemeral"} for m in markers)
        assert "ttl" not in json.dumps(body)


def test_anthropic_ttl_hint_above_threshold_marks_every_breakpoint_1h() -> None:
    """Above the threshold EVERY marker carries ``ttl: 1h`` and none is 5m —
    Anthropic rejects a 1h entry that follows a shorter one."""
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=150_000)
    request = _large_context_request().model_copy(update={"context_tokens_hint": 150_000})
    body = client._build_body(request, oauth=True)
    markers = _every_cache_control(body)
    assert len(markers) >= 3  # system head + last message + previous user turn
    assert all(m == {"type": "ephemeral", "ttl": "1h"} for m in markers)


def test_anthropic_ttl_hint_below_threshold_stays_5m() -> None:
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=150_000)
    request = _large_context_request().model_copy(update={"context_tokens_hint": 149_999})
    markers = _every_cache_control(client._build_body(request))
    assert markers and all(m == {"type": "ephemeral"} for m in markers)


def test_anthropic_ttl_hint_above_a_small_window_still_marks_1h() -> None:
    """The hint-larger-than-window case the original fixtures encoded, restored.

    A ``/model`` down-switch or a failover onto a smaller spec leaves the
    previous model's count on the request, so this pairing is real and the body
    must still build: the TTL policy reads the hint as a SIZE signal and is
    entitled to believe a large one, while the output clamp separately refuses to
    believe it as a prompt measurement. Raising these fixtures to a 1M window
    removed the only coverage of that combination and let the clamp's refusal
    ship without anyone deciding what should happen here (R10).
    """
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=150_000)
    request = _large_context_request().model_copy(
        update={
            "model": ModelSpec(
                provider="anthropic", model_id="claude-opus-4.1", context_window=128_000
            ),
            "context_tokens_hint": 900_000,
        }
    )

    body = client._build_body(request, oauth=True)

    markers = _every_cache_control(body)
    assert markers and all(m == {"type": "ephemeral", "ttl": "1h"} for m in markers)
    # And the clamp does not refuse it: the messages actually in hand are tiny.
    assert body["max_tokens"] > 0


def test_anthropic_ttl_without_hint_falls_back_to_byte_estimate() -> None:
    """No hint (first call, fork) → ``len(json)/4`` decides. A 2 KB body reads
    as ~500 tokens; the same body with a 4k-token threshold stays 5m and with
    a 100-token threshold goes 1h, so the estimate is what flipped it."""
    request = ChatRequest(
        model=_spec(provider="anthropic"),
        system_blocks=["x" * 1_000, "inventory", "skills", "env"],
        messages=[Message.user("a" * 1_000), Message.assistant("mid"), Message.user("second")],
    )
    assert request.context_tokens_hint is None
    small_threshold = AnthropicClient(cache_ttl_1h_min_context_tokens=100)
    assert all(
        m == {"type": "ephemeral", "ttl": "1h"}
        for m in _every_cache_control(small_threshold._build_body(request))
    )
    big_threshold = AnthropicClient(cache_ttl_1h_min_context_tokens=4_000)
    assert all(
        m == {"type": "ephemeral"} for m in _every_cache_control(big_threshold._build_body(request))
    )


def test_anthropic_ttl_estimate_excludes_base64_image_payloads() -> None:
    """The fallback byte estimate must not count base64 image data (F3).

    Anthropic bills an image by pixel area (~≤1.6k tokens), not by the bytes
    its base64 spelling takes in the serialized body. A naive ``len(json)/4``
    turns one ~1 MB screenshot into ~330k \"tokens\" and flips an image-bearing
    first/fork request to 1h on a prompt that is really tiny. The estimate
    swaps each image payload for a flat per-image allowance instead, wherever
    the image sits (message content or tool-result content)."""
    image = ImageContent(data="a" * 1_400_000, mime_type="image/png")  # ~1 MB decoded
    with_image = ChatRequest(
        model=_spec(provider="anthropic"),
        system_blocks=["instructions", "inventory", "skills", "env"],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("second")],
    ).model_copy(
        update={"messages": [Message(role="user", content=[TextContent(text="look"), image])]}
    )
    without_image = with_image.model_copy(update={"messages": [Message.user("look")]})
    two_images = with_image.model_copy(
        update={
            "messages": [Message(role="user", content=[TextContent(text="look"), image, image])]
        }
    )
    # Sanity: the payload is what dominates the serialized bytes — dropping it
    # must shrink the body by two orders of magnitude, or the test proves
    # nothing about the estimate.
    client = AnthropicClient()
    with_bytes = len(json.dumps(client._build_body(with_image), default=str))
    without_bytes = len(json.dumps(client._build_body(without_image), default=str))
    assert with_bytes > 100 * without_bytes

    # Threshold the naive estimate would blow past (1.4M base64 chars / 4 ≈
    # 350k \"tokens\") but the real prompt (~1.6k for the image + a tiny body)
    # cannot reach: the request stays 5m.
    stayed_5m = AnthropicClient(cache_ttl_1h_min_context_tokens=50_000)
    markers = _every_cache_control(stayed_5m._build_body(with_image))
    assert markers and all(m == {"type": "ephemeral"} for m in markers)
    # The allowance itself is counted, not just dropped: two images (2 x 1.6k
    # + body ≈ 3.3k) clear a 3k bar while the single-image body (≈1.7k) does
    # not — the estimate stays order-correct in image count.
    low_bar = AnthropicClient(cache_ttl_1h_min_context_tokens=3_300)
    assert all(
        m == {"type": "ephemeral"} for m in _every_cache_control(low_bar._build_body(with_image))
    )
    tipped = AnthropicClient(cache_ttl_1h_min_context_tokens=3_000)
    assert all(
        m == {"type": "ephemeral", "ttl": "1h"}
        for m in _every_cache_control(tipped._build_body(two_images))
    )


def test_anthropic_ttl_estimate_counts_tools_too() -> None:
    """Tools sit at position 0 of the cached prefix, so the byte estimate has
    to include them: a tool-heavy first request must not be undercounted."""
    tools = [
        AgentTool(
            name=f"tool{i}",
            description="d" * 400,
            parameters={"type": "object", "properties": {}},
            execute=_aside_execute,
        )
        for i in range(10)
    ]
    slim = ChatRequest(
        model=_spec(provider="anthropic"),
        system_blocks=["instructions", "inventory", "skills", "env"],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("second")],
    )
    heavy = slim.model_copy(update={"tools": tools})
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=800)
    assert all(m == {"type": "ephemeral"} for m in _every_cache_control(client._build_body(slim)))
    assert all(
        m == {"type": "ephemeral", "ttl": "1h"}
        for m in _every_cache_control(client._build_body(heavy))
    )


def test_anthropic_ttl_keeps_the_breakpoint_budget() -> None:
    """The TTL rides on the existing markers; it must not add any. Nine system
    blocks + two message targets still fit MAX_CACHE_BREAKPOINTS."""
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=1)
    request = ChatRequest(
        # Explicit 1M window for the same reason as ``_large_context_request``:
        # a 500k hint against the spec default's 128k describes a session that
        # cannot exist, and the output clamp refuses it.
        model=ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=1_000_000),
        system_blocks=[f"block-{i}" for i in range(9)],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("second")],
        context_tokens_hint=500_000,
    )
    markers = _every_cache_control(client._build_body(request, oauth=True))
    assert len(markers) == AnthropicClient.MAX_CACHE_BREAKPOINTS
    assert all(m["ttl"] == "1h" for m in markers)
    rendered = AnthropicClient._system_blocks([f"b{i}" for i in range(9)], ttl="1h")
    assert sum("cache_control" in b for b in rendered) == AnthropicClient.MAX_CACHE_BREAKPOINTS


def test_client_for_spec_passes_anthropic_ttl_threshold() -> None:
    client = client_for_spec(
        _spec(provider="anthropic", model_id="claude-opus-4-8"),
        anthropic_cache_ttl_1h_min_context_tokens=123_456,
    )
    assert isinstance(client, AnthropicClient)
    assert client._cache_ttl_1h_min_context_tokens == 123_456


def test_client_for_spec_openrouter_attribution_headers() -> None:
    """OpenRouter specs attach app attribution headers for rankings and discovery."""
    from local_operator.providers.clients import openrouter_attribution_headers

    expected = {
        "HTTP-Referer": "https://local-operator.com",
        "X-OpenRouter-Title": "Local Operator",
        "X-Title": "Local Operator",
        "X-OpenRouter-Categories": "cli-agent,personal-agent",
    }
    assert openrouter_attribution_headers() == expected

    client = client_for_spec(_spec(provider="openrouter", model_id="anthropic/claude-3.5-sonnet"))
    assert isinstance(client, OpenAICompatClient)
    assert client._extra_headers == expected


async def test_anthropic_usage_parses_cache_creation_ttl_split() -> None:
    """``usage.cache_creation`` splits the write count by TTL; both slices land
    on the Usage event and the sum still equals ``cache_write_tokens``."""
    body = _sse(
        [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 1_800,
                        "cache_creation_input_tokens": 248,
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 148,
                            "ephemeral_1h_input_tokens": 100,
                        },
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
    usage = [e for e in events if isinstance(e, StreamEndEvent)][-1].usage
    assert usage is not None
    assert usage.cache_write_tokens == 248
    assert usage.cache_write_5m_tokens == 148
    assert usage.cache_write_1h_tokens == 100
    assert usage.cache_write_5m_tokens + usage.cache_write_1h_tokens == usage.cache_write_tokens
    assert usage.context_tokens == 10 + 1_800 + 248


async def test_anthropic_usage_without_cache_creation_object_leaves_split_zero() -> None:
    """Older API versions omit the object; the split must read 0, not raise."""
    body = _sse(
        [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 5, "cache_creation_input_tokens": 40}},
            },
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
    usage = [e for e in events if isinstance(e, StreamEndEvent)][-1].usage
    assert usage is not None
    assert usage.cache_write_tokens == 40
    assert usage.cache_write_5m_tokens == 0
    assert usage.cache_write_1h_tokens == 0


# --- output-cap clamp -------------------------------------------------------
#
# Regression cover for the muse-spark 400: a listing that advertises 0.9 of the
# window as its completion cap reserves that much input capacity at admission,
# so the request overflows the window before a token is generated. See
# ``_effective_max_tokens``.

#: The wire key each client spells the output cap with. The Responses body and
#: the chat/completions body differ, and Google nests its own under
#: ``generationConfig`` (already unwrapped by ``_bodies``).
MAX_TOKEN_KEYS = [
    ("openai-completions", "max_tokens"),
    ("openai-responses", "max_output_tokens"),
    ("anthropic", "max_tokens"),
    ("google", "maxOutputTokens"),
]


def _muse_spark_spec() -> ModelSpec:
    """The real muse-spark shape: 1M window, 943718 advertised output cap."""
    return ModelSpec(
        provider="openrouter",
        model_id="meta/muse-spark-1.3",
        context_window=1_048_576,
        max_output_tokens=943_718,
    )


def _sonnet_spec() -> ModelSpec:
    """Claude Sonnet's real shape — the model round 1 silently truncated."""
    return ModelSpec(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        context_window=200_000,
        max_output_tokens=64_000,
    )


def _prose(tokens: int, *, non_ascii: bool = False) -> str:
    """About ``tokens`` tokens of ordinary prose (~4 chars/token).

    Realistic content matters here: the round-1 defect was invisible precisely
    because every test used a 2-token prompt or a 6x-window one, never the
    ordinary middle where real sessions spend their time.
    """
    chunk = "The quick brown fox jumps over the lazy dog. "
    if non_ascii:
        # One curly apostrophe per chunk. Under a byte-length bound this single
        # character multiplied the whole block's charge by 4; the output budget
        # must not depend on it.
        chunk = chunk.replace("dog.", "dog\u2019s.")
    return chunk * max(1, tokens * 4 // len(chunk))


@pytest.mark.parametrize("wire,key", MAX_TOKEN_KEYS)
def test_output_cap_clamped_so_prompt_plus_output_fits_window(wire: str, key: str) -> None:
    """The reported 400: 943718 of reserved output against a 1M window left only
    ~104k of prompt, and a real session died at ~113k of input. Every wire must
    now ask for an amount that fits BESIDE the prompt it is sending."""
    spec = _muse_spark_spec()
    # ~113k tokens of prompt — the size the real session failed at.
    request = ChatRequest(model=spec, messages=[Message.user("word " * 120_000)])

    body = _bodies(request)[wire]

    assert body[key] < spec.max_output_tokens
    # The whole point: admission counts prompt + max_tokens against the window.
    assert body[key] + len("word " * 120_000) // 4 < spec.context_window


@pytest.mark.parametrize("wire,key", MAX_TOKEN_KEYS)
@pytest.mark.parametrize("prompt_tokens", [2, 12_000, 30_000, 48_000])
def test_output_cap_unchanged_for_a_sanely_advertised_model(
    wire: str, key: str, prompt_tokens: int
) -> None:
    """The safeguard must not cost the models that work today anything — asserted
    across the range real sessions actually occupy, not just a 2-token prompt.

    This is the test that let the round-1 defect ship: it made this exact claim
    against ``Message.user("hi")``, the one input where a 4-18x over-estimate
    cannot bite. At 48,000 tokens (24% of Sonnet's window) the previous revision
    sent 512 instead of 64,000 and truncated a real answer mid-sentence.
    """
    spec = _sonnet_spec()
    request = ChatRequest(model=spec, messages=[Message.user(_prose(prompt_tokens))])

    assert _bodies(request)[wire][key] == 64_000


@pytest.mark.parametrize("wire,key", MAX_TOKEN_KEYS)
def test_output_budget_does_not_depend_on_non_ascii_characters(wire: str, key: str) -> None:
    """An identical prompt must not lose its output budget because it contains a
    curly apostrophe.

    A byte-length bound charges ``4 * len(text)`` for any block that is not
    ``str.isascii()``, so one ``\u2019`` — or an em dash, an emoji, an accented
    name, any non-English text — used to cut the same conversation's budget from
    64,000 to 512. The two asks must now agree.
    """
    spec = _sonnet_spec()
    ascii_body = _bodies(ChatRequest(model=spec, messages=[Message.user(_prose(30_000))]))
    unicode_body = _bodies(
        ChatRequest(model=spec, messages=[Message.user(_prose(30_000, non_ascii=True))])
    )

    assert ascii_body[wire][key] == unicode_body[wire][key] == 64_000


@pytest.mark.parametrize("wire,key", MAX_TOKEN_KEYS)
def test_explicit_small_max_tokens_is_still_honoured(wire: str, key: str) -> None:
    """``Session.ERRAND_MAX_TOKENS`` (1024, auto-naming) is a DELIBERATE ceiling.
    The clamp lowers an ask that does not fit; it must never raise one that
    does, or a title errand would start billing a full-window completion."""
    spec = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        context_window=1_000_000,
        max_output_tokens=128_000,
    )
    request = ChatRequest(model=spec, messages=[Message.user(_prose(30_000))], max_tokens=1024)

    assert _bodies(request)[wire][key] == 1024


@pytest.mark.parametrize("wire,key", MAX_TOKEN_KEYS)
def test_a_prompt_too_large_to_answer_is_refused_not_truncated(wire: str, key: str) -> None:
    """When the window cannot fund a usable reply the request is REFUSED.

    Sending a tiny cap anyway is the one outcome worse than the overflow this
    fixes: reasoning tokens are billed against the same budget (``grok-4.6``
    spent 689 of them thinking at a 512 cap and emitted no text at all), and
    ``harness/loop.py`` only retries a COMPLETELY silent truncation, so a partial
    answer is accepted with no notice. The user previously got a legible HTTP
    400 here and must still get something they can see.
    """
    spec = ModelSpec(
        provider="openrouter",
        model_id="tiny",
        context_window=10_000,
        max_output_tokens=8_000,
    )
    request = ChatRequest(model=spec, messages=[Message.user(_prose(20_000))])

    with pytest.raises(ProviderError) as excinfo:
        _bodies(request)[wire]

    # The message has to name the numbers that made it impossible, or it is just
    # another opaque failure.
    assert "too large" in str(excinfo.value)
    assert excinfo.value.kind == "request"


def test_the_clamp_still_lowers_an_overflowing_ask_before_refusing() -> None:
    """Between "fits untouched" and "cannot be answered" there is a real middle
    where the ask is reduced and the turn proceeds — the muse-spark case itself.
    Without this the refusal above could pass while the clamp did nothing."""
    spec = _muse_spark_spec()
    request = ChatRequest(model=spec, messages=[Message.user(_prose(400_000))])

    sent = _bodies(request)["openai-completions"]["max_tokens"]

    assert MIN_OUTPUT_TOKENS <= sent < spec.max_output_tokens


def test_system_blocks_and_tools_are_charged_against_the_window() -> None:
    """The reported 400 itemised **10,400 tokens of tool input** separately, so
    tools are a real term. A clamp that counted only ``messages`` would leave
    exactly that much of the overflow in place."""
    spec = _muse_spark_spec()
    tool = AgentTool(
        name="write",
        description="x" * 40_000,
        parameters={"type": "object", "properties": {"content": {"type": "string"}}},
        execute=lambda ctx, **kw: ToolResult(),  # type: ignore[arg-type,return-value]
    )
    messages = [Message.user("word " * 100_000)]

    bare = _bodies(ChatRequest(model=spec, messages=messages))["openai-completions"]
    with_extras = _bodies(
        ChatRequest(
            model=spec,
            messages=messages,
            system_blocks=["s" * 40_000],
            tools=[tool],
        )
    )["openai-completions"]

    assert with_extras["max_tokens"] < bare["max_tokens"]


def test_no_cap_anywhere_leaves_the_key_absent() -> None:
    """A spec with no cap must stay uncapped. Clamping a value nobody set would
    turn an absent key into a present one and put a ceiling on a model that
    currently has none."""
    spec = ModelSpec(
        provider="openrouter", model_id="x", context_window=100_000, max_output_tokens=0
    )
    request = ChatRequest(model=spec, messages=[Message.user("hi")])

    body = _bodies(request)
    assert "max_tokens" not in body["openai-completions"]
    assert "max_output_tokens" not in body["openai-responses"]
    assert "maxOutputTokens" not in body["google"]


# --- the clamp's safety properties, not just its arithmetic --------------------
#
# Round-2 review mutation-tested the previous set and found it ONE-SIDED: raising
# the slope failed 12 tests, but emptying the table entirely — which is the
# aggregator bug made global, and re-opens the HTTP 400 — left all 766 provider
# tests green. Downward mutations are the dangerous direction, so these pin the
# floor, the family keying, the hinted path (previously untested altogether) and
# the compaction interaction.


def _tool(name: str = "t", schema_chars: int = 1_500) -> AgentTool:
    """A tool whose schema is bulky enough to move a token estimate."""
    return AgentTool(
        name=name,
        description="d" * schema_chars,
        parameters={"type": "object"},
        execute=lambda ctx, **kw: ToolResult(),  # type: ignore[arg-type,return-value]
    )


def test_context_tokens_hint_is_used_whole_and_not_re_padded() -> None:
    """``Usage.context_tokens`` already covers system blocks and tool schemas, so
    adding a locally-estimated prefix on top double-counts it.

    Measured at ~21.8k phantom tokens with this repo's default tool set (R7).
    The same hint must therefore produce the same budget no matter how large the
    prefix is — the hint already contains it.
    """
    spec = _sonnet_spec()
    bare = ChatRequest(model=spec, messages=[Message.user("hi")], context_tokens_hint=100_000)
    padded = ChatRequest(
        model=spec,
        messages=[Message.user("hi")],
        system_blocks=["s" * 20_000],
        tools=[_tool(f"t{i}") for i in range(15)],
        context_tokens_hint=100_000,
    )

    assert _estimated_prompt_tokens(bare) == (100_000, 100_000)
    assert _estimated_prompt_tokens(padded) == _estimated_prompt_tokens(bare)


def test_the_refusal_never_pre_empts_compaction() -> None:
    """The invariant that keeps a recoverable session recoverable.

    The refusal must not fire below the compaction trigger, or the turn dies
    non-retryably on a session that would have compacted at the next boundary and
    continued. Asserted against the real ``resolve_threshold_tokens`` rather than
    a copied constant, and across tool counts, because it was a tool-scaled
    over-estimate that broke this (R8).
    """
    settings = CompactionSettings()
    # Starts at 8,000 deliberately. The previous range began at 64,000 — exactly
    # where the old constant reserve started holding — so the failing band was
    # never entered and the test passed while every 8k model was bricked.
    for window in (8_000, 8_192, 16_385, 32_768, 40_960, 64_000, 128_000, 200_000, 1_000_000):
        spec = ModelSpec(
            provider="anthropic",
            model_id="claude-sonnet-4-5",
            context_window=window,
            max_output_tokens=min(64_000, window // 2),
        )
        trigger = resolve_threshold_tokens(window, settings)
        for tool_count in (0, 10, 30):
            request = ChatRequest(
                model=spec,
                messages=[Message.user("hi")],
                tools=[_tool(f"t{i}") for i in range(tool_count)],
                # A session sitting exactly ON its compaction trigger is the last
                # moment compaction can still save it; it must not be refused.
                context_tokens_hint=trigger,
            )
            _effective_max_tokens(request)  # must not raise


def test_the_compaction_summarizer_is_never_refused() -> None:
    """``Session._one_shot_complete`` sends the transcript with
    ``context_tokens_hint=0``, so it takes the estimated branch.

    If that path can refuse, the one remedy the refusal's own message names
    cannot run and the session is wedged with no in-session recovery — strictly
    worse than the HTTP 400 this PR set out to fix. Measured before the fix: a
    200k Anthropic model past ~135k local tokens could not compact.
    """
    spec = _sonnet_spec()
    for local_tokens in (90_000, 135_000, 180_000):
        request = ChatRequest(
            model=spec,
            messages=[Message.user(_prose(local_tokens))],
            context_tokens_hint=0,
        )
        assert _effective_max_tokens(request) >= _output_reserve_tokens(spec.context_window)

    # Small windows too: the summarizer is refused there for the same reason a
    # turn is, and `/compact` is the remedy the refusal's own message names.
    for window in (8_192, 16_385, 32_768):
        small = ModelSpec(
            provider="openai",
            model_id="gpt-4",
            context_window=window,
            max_output_tokens=window // 2,
        )
        trigger = resolve_threshold_tokens(window, CompactionSettings())
        request = ChatRequest(
            model=small,
            messages=[Message.user(_prose(int(trigger * 0.9)))],
            context_tokens_hint=0,
        )
        assert _effective_max_tokens(request) > 0


def test_a_hint_larger_than_the_window_is_not_believed() -> None:
    """A stale hint must not refuse a session whose real context fits.

    Reachable two ways, both reproduced by reviewers: ``Session.set_model``
    swaps to a smaller-window model without clearing the hint (a ``/model``
    down-switch), and the failover clone keeps the primary's hint while moving
    to a smaller fallback spec. The hint cannot be describing THIS request, so
    the local estimate is the only honest figure (R10).
    """
    spec = ModelSpec(
        provider="anthropic",
        model_id="claude-opus-4.1",
        context_window=200_000,
        max_output_tokens=32_000,
    )
    request = ChatRequest(model=spec, messages=[Message.user("hi")], context_tokens_hint=600_000)

    assert _effective_max_tokens(request) == 32_000


@pytest.mark.parametrize(
    "provider,model_id",
    [
        ("anthropic", "claude-sonnet-4-5"),
        ("openrouter", "anthropic/claude-sonnet-4.5"),
        ("radient", "anthropic/claude-opus-4.1"),
    ],
)
def test_claude_gets_its_measured_slope_on_every_route(provider: str, model_id: str) -> None:
    """The ratio belongs to the TOKENIZER, so the route must not change it.

    Keying on ``ModelSpec.provider`` gave every aggregator-served Claude a
    different number than the same model served directly, re-opening the original
    400 on the exact provider this bug was reported against (R9). ``openrouter``
    and ``radient`` are real registry ids. The value is uniform today; what this
    pins is that the ROUTE cannot change it.
    """
    assert _estimate_slope(ModelSpec(provider=provider, model_id=model_id)) == (
        DEFAULT_ESTIMATE_SLOPE
    )


def test_an_unknown_family_fails_safe_rather_than_cheap() -> None:
    """The slope must stay above every ratio actually measured.

    Under-estimating re-opens the HTTP 400, and the refusal cannot catch it
    because that fires on OVER-estimation. Live measurement puts Claude at
    1.10-1.16 and the cross-family spread at 1.005-1.18, so the floor asserted
    here is what keeps a future tuning pass from drifting under the evidence.
    """
    slope = _estimate_slope(ModelSpec(provider="xai", model_id="grok-4.6"))

    assert slope == DEFAULT_ESTIMATE_SLOPE
    # The floor is the point: a default that drifts down re-opens the bug, and
    # every previous test stayed green while it did exactly that.
    assert slope >= 1.25


def test_the_ask_stays_admissible_for_an_expensive_tokenizer() -> None:
    """The end-to-end property the slope exists for, stated as admission.

    A Claude prompt whose real cost is ~1.9x the local estimate must still leave
    ``prompt + max_tokens`` inside the window. This is the assertion that fails
    when the slope table is emptied — the mutation the previous suite could not
    detect.
    """
    spec = _sonnet_spec()
    request = ChatRequest(model=spec, messages=[Message.user(_prose(90_000))])

    sent = _bodies(request)["anthropic"]["max_tokens"]

    # Measured, not the nominal size asked for: `_prose` targets a character
    # count and the tokenizer decides the rest, so pinning the assertion to the
    # request's own estimate is what keeps this about admission.
    _, measured = _estimated_prompt_tokens(request)
    real_prompt = measured * 1.2  # above every live-measured Claude ratio (1.10-1.16)
    assert real_prompt + sent <= spec.context_window


# --- small windows and the healthy-session ask --------------------------------
#
# Round-3 review and QA both found the same two defects, and both were invisible
# to the suite because its smallest window under test was 10,000 and used only to
# assert that a refusal SHOULD fire. Nothing asserted that an ordinary prompt on a
# small window ADMITS, and nothing checked what the ask becomes once the measured
# figure has proved a session healthy.


@pytest.mark.parametrize("window", [8_000, 8_192, 16_385, 32_768])
def test_a_small_prompt_on_a_small_window_still_admits(window: int) -> None:
    """An 8k model must serve ``"hi"``.

    ``MIN_OUTPUT_TOKENS + OUTPUT_CLAMP_SAFETY_MARGIN`` is 8192, so a constant
    reserve consumed the ENTIRE window of ``gpt-4`` and ``moonshot-v1-8k`` and
    refused every request to them — a one-token prompt included, with 8,185
    tokens of real headroom. 7 bundled registry rows and 12 live OpenRouter rows
    sit at or below 8,192.
    """
    spec = ModelSpec(
        provider="openai", model_id="gpt-4", context_window=window, max_output_tokens=window
    )
    request = ChatRequest(model=spec, messages=[Message.user("hi")])

    assert _effective_max_tokens(request) > 0


@pytest.mark.parametrize("window", [1_000, 8_192, 32_768, 40_960, 200_000, 2_000_000])
def test_the_reserve_stays_inside_the_compaction_headroom_at_every_window(window: int) -> None:
    """The ordering that keeps a recoverable session recoverable, as ALGEBRA.

    The refusal point and the compaction trigger must keep a fixed order at every
    window size. A constant reserve against a fractional trigger cannot: it held
    above ~41k and inverted below, which is the same wedge the round-2 review
    found, relocated one window-band lower. Expressing the reserve in the
    trigger's own shape makes the ordering true by construction — so this asserts
    it across four orders of magnitude rather than over a lucky range.
    """
    settings = CompactionSettings()

    refusal_point = window - _output_reserve_tokens(window, settings)

    assert refusal_point > resolve_threshold_tokens(window, settings)


def test_the_reserve_tracks_a_raised_compaction_threshold() -> None:
    """``threshold_percent`` is user-configurable, so a hardcoded fraction would
    silently re-invert for anyone who raises it. The reserve is derived from the
    caller's own settings for that reason."""
    aggressive = CompactionSettings(threshold_percent=0.95)

    window = 200_000
    refusal_point = window - _output_reserve_tokens(window, aggressive)

    assert refusal_point > resolve_threshold_tokens(window, aggressive)


@pytest.mark.parametrize("occupancy", [0.25, 0.41, 0.50])
def test_a_healthy_session_keeps_its_full_ask_without_a_hint(occupancy: float) -> None:
    """The hint-less path must not be quietly capped at the reserve.

    Flooring at ``MIN_OUTPUT_TOKENS`` once the measurement proved a session
    healthy was round-1's truncation defect by another route: at 50% occupancy a
    Sonnet first call asked for 4,096 with ~95k tokens of real headroom and
    returned an answer cut mid-word with ``finish_reason='length'``, where main
    completed it. It bit only the hint-less branch — first call, forks, errands,
    the compaction summarizer — which is why a hinted measurement did not show it.

    Capped at 50% deliberately: past roughly 55% a reduced ask is CORRECT, since
    the prompt plus a full 64k reply genuinely approaches the window. What this
    pins is the band where main answers in full and the branch must too.
    """
    spec = _sonnet_spec()
    request = ChatRequest(
        model=spec, messages=[Message.user(_prose(int(spec.context_window * occupancy)))]
    )

    assert _bodies(request)["anthropic"]["max_tokens"] == spec.max_output_tokens


@pytest.mark.parametrize("window", [8_192, 32_768, 200_000])
@pytest.mark.parametrize("explicit", [128, 1024])
def test_an_explicit_small_ask_is_never_raised(window: int, explicit: int) -> None:
    """The clamp only ever LOWERS, which is what keeps ``ERRAND_MAX_TOKENS`` small.

    A caller that names its own budget (auto-naming asks for 1024) has already
    said how much it needs. An earlier revision carried a dedicated escape hatch
    for this and it was dead code twice over; the rescue clause subsumes it, so
    what needs pinning is the GUARANTEE rather than the branch that used to
    implement it.
    """
    spec = ModelSpec(
        provider="anthropic",
        model_id="claude-x",
        context_window=window,
        max_output_tokens=64_000,
    )
    # Half-full: past the point where the scaled taper has started biting, but
    # still a session the measurement calls healthy.
    request = ChatRequest(
        model=spec,
        messages=[Message.user(_prose(window // 2))],
        max_tokens=explicit,
    )

    assert _effective_max_tokens(request) <= explicit


# --- the rescue bound, pinned so it cannot silently regress -------------------
#
# Round-4 review and QA both found that three of the mutations claimed for this
# clause killed ZERO tests: reverting the rescue to a bare `MIN_OUTPUT_TOKENS`
# floor passed the whole suite. The reason is that `_output_reserve_tokens`
# saturates at `MIN_OUTPUT_TOKENS` on every window >= 40,960, so the two forms are
# the same value everywhere the earlier tests looked. These pin the properties
# that actually distinguish a correct bound from a constant.


def test_the_ask_falls_monotonically_as_the_prompt_grows() -> None:
    """A bigger prompt must never be handed a bigger reply budget.

    This is what a constant grant breaks: flooring at the reserve made the ask
    flat at 4,096 across 80%, 88% and 95% occupancy alike, so the curve stopped
    carrying information about how full the session was. Monotonicity is the
    property that distinguishes a taper from a cliff, and no earlier test had it.
    """
    spec = _sonnet_spec()

    asks = []
    for occupancy in (0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85):
        request = ChatRequest(
            model=spec,
            messages=[Message.user(_prose(int(spec.context_window * occupancy)))],
        )
        asks.append(_effective_max_tokens(request))

    assert asks == sorted(asks, reverse=True), asks


def test_a_mostly_full_session_still_gets_a_proportionate_ask() -> None:
    """At 80% occupancy the ask must reflect the measured headroom, not a floor.

    QA measured this live: the constant grant returned ``finish_reason='length'``
    on an answer main completed, with tens of thousands of tokens of real
    headroom in hand — up to 264k on a 1M-window model. The bound has to stay
    well clear of the floor here or that truncation returns.
    """
    spec = _sonnet_spec()
    request = ChatRequest(
        model=spec,
        messages=[Message.user(_prose(int(spec.context_window * 0.8)))],
    )

    ask = _effective_max_tokens(request)

    assert ask > MIN_OUTPUT_TOKENS * 2, ask


def test_the_ask_stays_admissible_at_the_worst_measured_ratio() -> None:
    """The other side of the same bound: proportionate must still mean safe.

    Live measurement puts realistic agent content at 1.098-1.219 against the local
    estimate. The ask plus a prompt that bills at the top of that range must fit,
    or the taper has simply traded truncation for the HTTP 400 this clamp exists
    to prevent.
    """
    spec = _sonnet_spec()

    for occupancy in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85):
        request = ChatRequest(
            model=spec,
            messages=[Message.user(_prose(int(spec.context_window * occupancy)))],
        )
        _, measured = _estimated_prompt_tokens(request)
        ask = _effective_max_tokens(request)

        assert measured * 1.219 + ask <= spec.context_window, (occupancy, ask)


@pytest.mark.parametrize("percent", [0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95])
def test_the_ordering_holds_through_the_production_call(percent: float) -> None:
    """The refusal must sit above the trigger for a user's ACTUAL settings.

    The predecessor of this test passed `settings` to `_output_reserve_tokens` —
    an argument the production call site does not supply, since `ChatRequest`
    carries no compaction config. It therefore asserted the property on an input
    production cannot produce, and a raised `compaction.threshold_percent` (a
    first-class setting) re-opened the wedge underneath it (R19/Q12). This calls
    the helper the way production does: with no settings at all.
    """
    settings = CompactionSettings(threshold_percent=percent)

    for window in (8_192, 16_385, 32_768, 65_536, 200_000, 1_000_000):
        reserve = _output_reserve_tokens(window)  # production form
        margin = min(OUTPUT_CLAMP_SAFETY_MARGIN, reserve)
        trigger = resolve_threshold_tokens(window, settings)

        # Only meaningful where a pass could actually fund a reply. Past that the
        # trigger is so late that compaction reclaims less than a usable answer,
        # so there is no rescue for the refusal to pre-empt.
        if window - trigger < reserve:
            continue

        assert window - reserve - margin > trigger, (percent, window)


# --- the same bounds, exercised BELOW the reserve's saturation point ----------
#
# `_output_reserve_tokens` saturates at `MIN_OUTPUT_TOKENS` from a ~163,840-token
# window upward, so every assertion written against a 200k model compares two
# forms that are numerically identical there. That is why three separately
# claimed mutations killed nothing. These use small windows, where the forms
# genuinely differ.


@pytest.mark.parametrize("window", [16_385, 32_768, 65_536])
def test_the_rescue_is_proportional_to_the_window_not_a_constant(window: int) -> None:
    """Below saturation the rescue must scale, or it is a constant in disguise.

    A bare ``MIN_OUTPUT_TOKENS`` floor hands a 16k model the same 4,096 it hands a
    1M model — half that model's window as a reply reservation. The reserve is a
    fraction precisely so a small window gets a small one.
    """
    spec = ModelSpec(
        provider="anthropic",
        model_id="claude-x",
        context_window=window,
        max_output_tokens=window // 2,
    )
    # 0.92 is inside the band where the scaled taper has collapsed below the
    # reserve and the RESCUE, not the taper, decides the ask (measured at
    # 0.86-0.99 for every window here). Below it the taper is still positive and
    # the clause under test never runs, which is how earlier versions of this
    # assertion passed against a constant.
    request = ChatRequest(model=spec, messages=[Message.user(_prose(int(window * 0.92)))])

    assert _effective_max_tokens(request) < MIN_OUTPUT_TOKENS


@pytest.mark.parametrize("window", [16_385, 32_768, 65_536])
def test_the_rescue_never_spends_the_measured_headroom_outright(window: int) -> None:
    """The opposite bound, also invisible above saturation.

    Granting ``measured_available`` assumes the provider bills exactly the local
    estimate (ratio 1.0), which no measurement supports — it re-opens the overflow
    the clamp exists to prevent. The ask must stay admissible when the prompt
    bills at the top of the measured range.

    Asserted just BELOW the compaction trigger rather than at an arbitrary high
    occupancy. Past the trigger a pass fires and the session never presents such a
    prompt; there the prompt alone can exceed the window at 1.219 and no ask is
    admissible, so an assertion there would pin an unreachable state instead of
    the property.
    """
    spec = ModelSpec(
        provider="anthropic",
        model_id="claude-x",
        context_window=window,
        max_output_tokens=window // 2,
    )
    trigger = resolve_threshold_tokens(window, CompactionSettings())
    request = ChatRequest(model=spec, messages=[Message.user(_prose(int(trigger * 0.98)))])

    _, measured = _estimated_prompt_tokens(request)
    ask = _effective_max_tokens(request)

    assert measured * 1.219 + ask <= window, (window, measured, ask)


@pytest.mark.parametrize("window", [16_385, 32_768, 65_536])
def test_the_refusal_keeps_its_cushion_above_the_trigger(window: int) -> None:
    """The reserve and the margin must not sum to the whole post-trigger headroom.

    Both are subtracted at the refusal, so halving the headroom for each left
    ``window - 2*reserve == percent * window`` — the trigger exactly, with the
    only separation coming from integer truncation (1-2 tokens, review R20). A
    cushion that thin is consumed by any later change without a test noticing.
    """
    reserve = _output_reserve_tokens(window)
    margin = min(OUTPUT_CLAMP_SAFETY_MARGIN, reserve)
    trigger = resolve_threshold_tokens(window, CompactionSettings())

    cushion = (window - reserve - margin) - trigger

    assert cushion > window // 100, (window, cushion)
