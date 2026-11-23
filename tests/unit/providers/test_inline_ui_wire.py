"""Inline discovery must not move the durable wire prefix or mark unsupported output."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamStartEvent,
    StreamTextDelta,
    TextContent,
    ToolCall,
)
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.bedrock.converse import BedrockConverseClient
from local_operator.providers.clients import (
    AnthropicClient,
    GoogleClient,
    OpenAICompatClient,
)
from local_operator.providers.failover import stream_with_failover
from local_operator.session.inline_ui import INSTRUCTIONS
from tests.unit.providers.bedrock.test_converse_body import NOVA, TOOL
from tests.unit.providers.bedrock.test_converse_stream import KEY, events, wire
from tests.unit.providers.test_failover import FakeAuth
from tests.unit.providers.test_native_replay import sse

pytestmark = pytest.mark.asyncio

_REPLY = "Answer from the provider fixture."
_CHAT = (
    sse([{"id": "wire-chat", "choices": [{"delta": {"content": _REPLY}, "finish_reason": "stop"}]}])
    + b"data: [DONE]\n\n"
)
_ANTHROPIC = sse(
    [
        {"type": "message_start", "message": {"id": "wire-anthropic"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": _REPLY},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
)
_GOOGLE = sse(
    [{"candidates": [{"content": {"parts": [{"text": _REPLY}]}, "finishReason": "STOP"}]}]
)


def _tool_request(provider: str, model: str) -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(provider=provider, model_id=model, supports_prompt_cache=True),
        system_blocks=["stable instructions", "stable working context"],
        messages=[
            Message.user("Look up the data."),
            Message.assistant(
                "Working.", tool_calls=[ToolCall(id="call_1", name="lookup", arguments={})]
            ),
            Message(
                role="tool", tool_call_id="call_1", content=[TextContent(text="Lookup result.")]
            ),
        ],
        tools=[TOOL],
    )


@pytest.mark.parametrize("route", ["anthropic-key", "anthropic-oauth", "radient", "google"])
async def test_local_http_adapters_append_one_hint_beyond_original_wire_prefix(
    route: str,
) -> None:
    # Use a socket, not a stubbed builder: test the bytes POSTed and the receipt
    # emitted by the real SSE parser. No provider key or off-machine request.
    received: list[tuple[str, dict[str, str], dict[str, Any]]] = []
    anthropic = route.startswith("anthropic-")
    response = _ANTHROPIC if anthropic else _GOOGLE if route == "google" else _CHAT

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            start, *lines = (await reader.readuntil(b"\r\n\r\n")).decode().strip().split("\r\n")
            headers = {
                key.lower(): value.strip() for key, value in (line.split(":", 1) for line in lines)
            }
            body = json.loads(await reader.readexactly(int(headers["content-length"])))
            received.append((start.split()[1], headers, body))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
                + f"Content-Length: {len(response)}\r\n\r\n".encode()
                + response
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
        base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with httpx.AsyncClient(trust_env=False, timeout=5) as http:
            if anthropic:
                client = AnthropicClient(base, http_client=http)
                request = _tool_request("anthropic", "claude-sonnet-4-6")
                rows = "messages"
                suffix = {"role": "user", "content": [{"type": "text", "text": INSTRUCTIONS}]}
            elif route == "google":
                client = GoogleClient(base, http_client=http)
                request = _tool_request("google", "gemini-3.8-flash")
                rows = "contents"
                suffix = {"role": "user", "parts": [{"text": INSTRUCTIONS}]}
            else:
                client = OpenAICompatClient(base, http_client=http)
                request = _tool_request("radient", "anthropic/claude-sonnet-4-6")
                rows = "messages"
                suffix = {"role": "user", "content": INSTRUCTIONS}
            access = (
                OAuthAccess(kind="oauth", access_token="synthetic-token", credential_id=1)
                if route == "anthropic-oauth"
                else None
            )
            for hint in (None, INSTRUCTIONS):
                call = request.model_copy(update={"inline_ui_hint": hint})
                result = [event async for event in client.stream(call, "synthetic-key", access)]
                [start] = [event for event in result if isinstance(event, StreamStartEvent)]
                assert start.inline_ui_source_version == (1 if hint else None)
                assert "inline_ui_source_version" not in start.model_dump()
                assert (
                    "".join(event.delta for event in result if isinstance(event, StreamTextDelta))
                    == _REPLY
                )
                assert isinstance(result[-1], StreamEndEvent) and result[-1].stop_reason == "stop"

    (path_before, _, before), (path_after, headers, after) = received
    assert path_before == path_after
    assert after[rows].pop() == suffix
    assert after == before  # system, tools, cache points and tool-result shape are unchanged
    if anthropic or route == "radient":
        assert '"cache_control"' in json.dumps(before[rows])
    if route == "anthropic-oauth":
        assert headers["authorization"] == "Bearer synthetic-token"
        assert before["system"][0]["text"] == AnthropicClient.CLAUDE_CODE_IDENTITY
    print(f"{route}: {path_after} 200; exact original prefix, one suffix, accepted receipt")


@pytest.mark.parametrize("converse_first", [True, False])
async def test_failover_uses_the_serving_adapters_inline_ui_receipt(
    converse_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-bedrock-token")
    monkeypatch.delenv("LOP_BEDROCK_ENDPOINT_URL", raising=False)
    received: list[tuple[bool, dict[str, Any]]] = []

    def post(http_request: httpx.Request) -> httpx.Response:
        converse = http_request.url.path.endswith("/converse-stream")
        body = json.loads(http_request.content)
        received.append((converse, body))
        if converse == converse_first:
            if converse:
                return httpx.Response(503, json={"message": "fixture primary unavailable"})
            # An accepted but pre-content primary still permits fallback; its
            # receipt must not authorize the hint-free Converse answer.
            return httpx.Response(
                200,
                content=sse(
                    [
                        {"id": "accepted-primary", "choices": []},
                        {"error": {"code": 503, "message": "fixture primary unavailable"}},
                    ]
                ),
            )
        return httpx.Response(200, content=wire(events()) if converse else _CHAT)

    async with httpx.AsyncClient(transport=httpx.MockTransport(post)) as http:

        async def client_for(spec: ModelSpec) -> Any:
            if spec.provider == "bedrock":
                return BedrockConverseClient(http_client=http)
            return OpenAICompatClient("https://test.invalid/v1", http_client=http)

        bedrock, compat = f"bedrock/{NOVA}", "openai/gpt-4o"
        primary, fallback = (bedrock, compat) if converse_first else (compat, bedrock)
        provider, model = primary.split("/", 1)
        request = ChatRequest(
            model=ModelSpec(provider=provider, model_id=model),
            system_blocks=["stable"],
            messages=[Message.user("Give a brief summary.")],
            inline_ui_hint=INSTRUCTIONS,
        )
        auth = FakeAuth({"openai": ["synthetic-key"], "bedrock": [KEY]})
        settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": [fallback]}}}
        result = [
            event async for event in stream_with_failover(request, auth, settings, client_for)
        ]

    assert {converse for converse, _ in received} == {True, False}
    assert all((INSTRUCTIONS in json.dumps(body)) is not converse for converse, body in received)
    starts = [
        event.inline_ui_source_version for event in result if isinstance(event, StreamStartEvent)
    ]
    compatible_posts = sum(not converse for converse, _ in received)
    assert starts == [1] * compatible_posts + ([] if converse_first else [None])
    assert isinstance(result[-1], StreamEndEvent) and result[-1].stop_reason == "stop"
    print(f"{primary} → {fallback}: {len(starts)} start(s), final inline receipt {starts[-1]}")
