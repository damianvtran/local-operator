"""Inline discovery must not move the durable wire prefix or mark unsupported output."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    Message,
    MessageUpdateEvent,
    ModelSpec,
    ProviderTurnStartEvent,
    StreamEndEvent,
    StreamEvent,
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
from local_operator.session.runtime.inline_ui_egress import InlineUiEvents
from tests.unit.providers.bedrock.test_converse_body import NOVA, TOOL
from tests.unit.providers.bedrock.test_converse_stream import KEY, events, wire
from tests.unit.providers.test_failover import FakeAuth
from tests.unit.providers.test_native_replay import sse
from tests.unit.session.test_session import make_session

pytestmark = pytest.mark.asyncio

_REPLY = "Answer from the provider fixture."
_CHAT = (
    sse([{"id": "wire-chat", "choices": [{"delta": {"content": _REPLY}, "finish_reason": "stop"}]}])
    + b"data: [DONE]\n\n"
)
_ANTHROPIC_START = {"type": "message_start", "message": {"id": "wire-anthropic"}}
_ANTHROPIC_CONTENT = [
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


@pytest.mark.parametrize("provider", ["vllm", "openai", "radient"])
async def test_chat_completions_keeps_a_strict_template_user_tail(provider: str) -> None:
    request = ChatRequest(
        model=ModelSpec(provider=provider, model_id="mistralai/Mistral-7B-Instruct-v0.3"),
        system_blocks=["Stable system prompt."],
        messages=[Message.user("Give a brief summary.")],
    )
    async with httpx.AsyncClient() as http:
        client = OpenAICompatClient("https://fixture.invalid/v1", http_client=http)
        plain = client._build_body(request)
        hinted = client._build_body(request.model_copy(update={"inline_ui_hint": INSTRUCTIONS}))
    # Strict templates (e.g. Mistral Instruct) reject another user row after
    # the ordinary user turn, even if the client speaks Chat Completions.
    assert [message["role"] for message in hinted["messages"]] == ["system", "user"]
    assert hinted == plain


@asynccontextmanager
async def _local_sse(
    response_for: Callable[[str, dict[str, str], dict[str, Any]], bytes],
) -> AsyncIterator[str]:
    # Real POST and SSE without any provider auth or off-machine requests.
    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            start, *lines = (await reader.readuntil(b"\r\n\r\n")).decode().strip().split("\r\n")
            headers = {
                key.lower(): value.strip() for key, value in (line.split(":", 1) for line in lines)
            }
            body = json.loads(await reader.readexactly(int(headers["content-length"])))
            response = response_for(start.split()[1], headers, body)
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
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"


@pytest.mark.parametrize(
    "route", ["anthropic-key", "anthropic-oauth", "anthropic-no-start", "radient", "google"]
)
async def test_local_http_adapter_hint_receipt_matches_posted_wire(
    route: str,
) -> None:
    received: list[tuple[str, dict[str, str], dict[str, Any]]] = []
    anthropic = route.startswith("anthropic-")
    if anthropic:
        start_frames = [] if route == "anthropic-no-start" else [_ANTHROPIC_START]
        response = sse([*start_frames, *_ANTHROPIC_CONTENT])
    else:
        response = _GOOGLE if route == "google" else _CHAT

    def post(path: str, headers: dict[str, str], body: dict[str, Any]) -> bytes:
        received.append((path, headers, body))
        return response

    async with _local_sse(post) as base:
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
                suffix = None
            access = (
                OAuthAccess(kind="oauth", access_token="synthetic-token", credential_id=1)
                if route == "anthropic-oauth"
                else None
            )
            for hint in (None, INSTRUCTIONS):
                call = request.model_copy(update={"inline_ui_hint": hint})
                result = [event async for event in client.stream(call, "synthetic-key", access)]
                [start] = [event for event in result if isinstance(event, StreamStartEvent)]
                assert start.inline_ui_source_version == (1 if hint and suffix else None)
                assert isinstance(result[0], StreamStartEvent)
                if route == "anthropic-no-start":
                    assert start.response_id is None
                assert "inline_ui_source_version" not in start.model_dump()
                assert (
                    "".join(event.delta for event in result if isinstance(event, StreamTextDelta))
                    == _REPLY
                )
                assert isinstance(result[-1], StreamEndEvent) and result[-1].stop_reason == "stop"

    (path_before, _, before), (path_after, headers, after) = received
    assert path_before == path_after
    if suffix is not None:
        assert after[rows].pop() == suffix
    assert after == before  # system, tools, cache points and tool-result shape are unchanged
    if anthropic or route == "radient":
        assert '"cache_control"' in json.dumps(before[rows])
    if route == "anthropic-oauth":
        assert headers["authorization"] == "Bearer synthetic-token"
        assert before["system"][0]["text"] == AnthropicClient.CLAUDE_CODE_IDENTITY
    print(f"{route}: {path_after} 200; exact original prefix; hint={suffix is not None}")


async def test_gemini_diagnostics_distinguish_hint_movement_from_history_change() -> None:
    bodies: list[dict[str, Any]] = []

    def post(path: str, headers: dict[str, str], body: dict[str, Any]) -> bytes:
        assert path == "/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
        bodies.append(body)
        return sse(
            [
                {
                    "candidates": [
                        {"content": {"parts": [{"text": _REPLY}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 2_000 if len(bodies) == 1 else 3_000,
                        "cachedContentTokenCount": 0 if len(bodies) == 1 else 2_000,
                        "candidatesTokenCount": 10,
                    },
                }
            ]
        )

    prior = Message.user("First request.")
    grown = [prior, Message.assistant(_REPLY), Message.user("Second request.")]
    request = ChatRequest(
        model=ModelSpec(provider="google", model_id="gemini-3.8-flash"),
        messages=[prior],
        inline_ui_hint=INSTRUCTIONS,
        cache_diagnostics=True,
        cache_diagnostics_chain=prior.id,
    )
    outcomes = []
    async with _local_sse(post) as base, httpx.AsyncClient(trust_env=False, timeout=5) as http:
        for messages in ([prior], grown, [Message.user("Rewritten first request."), *grown[1:]]):
            client = GoogleClient(base, http_client=http)
            attempt = request.model_copy(update={"messages": messages})
            result = [event async for event in client.stream(attempt, "synthetic-key")]
            end = result[-1]
            assert isinstance(end, StreamEndEvent)
            outcomes.append((end.cache_miss_type, end.cache_missed_tokens))

    sent = [body["contents"] for body in bodies]
    assert all(rows[-1] == {"role": "user", "parts": [{"text": INSTRUCTIONS}]} for rows in sent)
    assert sent[0][:-1] == sent[1][:1] and sent[0] != sent[1][:2]
    assert outcomes == [("baseline", 0), ("unchanged", 0), ("messages_changed", 1000)]
    print(f"Gemini: unchanged posted catalog; three diagnostics={outcomes}")


_OPENER = 'Result: 42.\n\n```lop-ui\n{"type":"bar","title":"June'
_CLOSER = ' counts","data":[["June",42]]}\n```\n'
_WIDGET = {"type": "bar", "title": "June counts", "data": [["June", 42]]}
_CREATED = {"type": "response.created", "response": {"id": "resp-actual"}}
_COMPLETED = {"type": "response.completed", "response": {"id": "resp-actual", "output": []}}


def _responses_reply(lifecycle: str) -> bytes:
    first = {
        "type": "response.output_text.delta",
        "item_id": "msg-not-a-response",
        "delta": _OPENER,
    }
    second = {**first, "delta": _CLOSER}
    if lifecycle == "early":
        return sse([_CREATED, first, second, _COMPLETED])
    if lifecycle == "late":
        return sse([first, _CREATED, second, _COMPLETED])
    if lifecycle == "duplicate":
        return sse([_CREATED, _CREATED, first, second, _COMPLETED])
    assert lifecycle == "missing"
    return sse([first, second, _COMPLETED])


@pytest.mark.parametrize(
    ("lifecycle", "primary", "hinted"),
    [
        ("early", None, True),
        ("late", None, True),
        ("missing", None, True),
        ("missing", None, False),
        ("duplicate", None, True),
        ("missing", "anthropic", False),
        ("chat", "anthropic", False),
        ("late", "openrouter", True),
    ],
)
async def test_responses_acceptance_precedes_egress_even_without_created(
    tmp_path: Path,
    lifecycle: str,
    primary: Literal["anthropic", "openrouter"] | None,
    hinted: bool,
) -> None:
    posts: list[tuple[str, bool]] = []

    def post(path: str, headers: dict[str, str], body: dict[str, Any]) -> bytes:
        posts.append((path, INSTRUCTIONS in json.dumps(body)))
        if path.endswith("/messages"):
            return sse(
                [
                    {"type": "message_start", "message": {"id": "failed-native"}},
                    {"type": "error", "error": {"type": "overloaded_error", "message": "Fixture"}},
                ]
            )
        if path.endswith("/chat/completions"):
            if lifecycle == "chat":
                return (
                    sse(
                        [
                            {"id": "chat-actual", "choices": [{"delta": {"content": _OPENER}}]},
                            {
                                "id": "chat-actual",
                                "choices": [
                                    {"delta": {"content": _CLOSER}, "finish_reason": "stop"}
                                ],
                            },
                        ]
                    )
                    + b"data: [DONE]\n\n"
                )
            return sse(
                [
                    {"id": "failed-generic", "choices": []},
                    {"error": {"code": 503, "message": "Fixture"}},
                ]
            )
        assert path.endswith("/responses")
        return _responses_reply(lifecycle)

    starts: list[tuple[str | None, int | None]] = []
    with_start: list[str | None] = []
    start_counts_at_updates: list[int] = []
    updates: list[dict[str, Any]] = []
    markers: list[int | None] = []
    projector = InlineUiEvents(sidecars=True)

    async with _local_sse(post) as base, httpx.AsyncClient(trust_env=False, timeout=5) as http:
        native = OpenAICompatClient(base, openai_api="responses", http_client=http)
        chat = OpenAICompatClient(base, http_client=http)
        anthropic = AnthropicClient(base, http_client=http)

        class UnhintedResponses:
            def stream(self, request, api_key, oauth_access=None):
                # An attempt can decline discovery independently of the
                # primary/viewer. Drive the same no-created native parser.
                return native.stream(
                    request.model_copy(update={"inline_ui_hint": None}), api_key, oauth_access
                )

        async def client_for(spec: ModelSpec) -> Any:
            if spec.provider == "anthropic":
                return anthropic
            if spec.provider == "openrouter":
                return chat
            assert spec.provider == "openai" and spec.supports_responses_api
            return UnhintedResponses() if primary and not hinted else native

        auth = FakeAuth(
            {provider: ["synthetic-key"] for provider in ("openai", "openrouter", "anthropic")}
        )
        fallback = "openrouter/openai/gpt-4o" if lifecycle == "chat" else "openai/gpt-6-astra"
        settings = {"retry": {"maxRetries": 0, "fallbackChains": {"default": [fallback]}}}

        async def stream(request: ChatRequest, signal: AbortSignal) -> AsyncIterator[StreamEvent]:
            async for event in stream_with_failover(
                request, auth, settings, client_for, signal=signal
            ):
                if isinstance(event, StreamStartEvent):
                    starts.append((event.response_id, event.inline_ui_source_version))
                yield event

        def on_event(event: Any) -> None:
            if isinstance(event, ProviderTurnStartEvent):
                with_start.append(event.response_id)
            elif isinstance(event, MessageUpdateEvent):
                start_counts_at_updates.append(len(with_start))
                markers.append(event.message._inline_ui_source_version)
                updates.append(projector.dump(event))

        session = make_session(
            tmp_path,
            stream,
            model={
                "anthropic": ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6"),
                "openrouter": ModelSpec(provider="openrouter", model_id="openai/gpt-4o"),
                None: ModelSpec(provider="openai", model_id="gpt-4o", supports_responses_api=True),
            }[primary],
        )
        session.subscribe(on_event)
        try:
            await session.prompt("Show June counts.", inline_ui=bool(primary) or hinted)
            durable_markers = [
                message._inline_ui_source_version
                for message in session._context.messages
                if message.role == "assistant"
            ]
        finally:
            await session.dispose()

    # Inspect projection before asserting metadata: on the old code the
    # opener escaped unprojected, or inherited the failed attempt's marker.
    assert len(updates) >= 2
    if lifecycle != "chat":
        assert len(updates) == 2
    deltas = [update["delta"] for update in updates]
    widgets = [
        widget["payload"]
        for message in updates[1].get("inline_ui", {}).get("messages", {}).values()
        for widget in message["widgets"]
    ]
    assert "inline_ui" not in updates[0]
    if hinted:
        assert deltas == ["Result: 42.\n\n\n", ""]
        assert widgets == [_WIDGET]
    else:
        assert "".join(deltas) == _OPENER + _CLOSER
        assert all("inline_ui" not in update for update in updates)
    receipt = 1 if hinted else None
    assert markers == [receipt] * len(updates)
    assert durable_markers == [receipt]
    assert posts[-1] == ("/chat/completions" if lifecycle == "chat" else "/responses", hinted)
    assert len(posts) == len(starts)
    if primary == "anthropic":
        assert starts[:-1] and all(start == ("failed-native", 1) for start in starts[:-1])
    elif primary == "openrouter":
        assert starts[:-1] and all(start == ("failed-generic", None) for start in starts[:-1])
    actual_id = {"early": "resp-actual", "duplicate": "resp-actual", "chat": "chat-actual"}
    assert starts[-1] == (actual_id.get(lifecycle), receipt)
    assert with_start == [response_id for response_id, _ in starts]
    assert start_counts_at_updates == [len(starts)] * len(updates)
    print(
        f"{primary or 'direct'} / {lifecycle}: POSTs={posts}; starts={starts}; "
        f"egress={deltas!r}; widgets={widgets!r}"
    )


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
                        _CREATED,
                        {
                            "type": "error",
                            "code": "server_error",
                            "message": "fixture primary unavailable",
                        },
                    ]
                ),
            )
        return httpx.Response(
            200, content=wire(events()) if converse else _responses_reply("early")
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(post)) as http:

        async def client_for(spec: ModelSpec) -> Any:
            if spec.provider == "bedrock":
                return BedrockConverseClient(http_client=http)
            return OpenAICompatClient(
                "https://test.invalid/v1", openai_api="responses", http_client=http
            )

        bedrock, compat = f"bedrock/{NOVA}", "openai/gpt-6-astra"
        primary, fallback = (bedrock, compat) if converse_first else (compat, bedrock)
        provider, model = primary.split("/", 1)
        request = ChatRequest(
            model=ModelSpec(
                provider=provider, model_id=model, supports_responses_api=not converse_first
            ),
            system_blocks=["stable"],
            messages=[Message.user("Give a brief summary.")],
            inline_ui_hint=INSTRUCTIONS,
        )
        auth = FakeAuth({"openai": ["synthetic-key"], "bedrock": [KEY]})
        settings = {"retry": {"maxRetries": 0, "fallbackChains": {"default": [fallback]}}}
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
