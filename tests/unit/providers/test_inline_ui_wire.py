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
    AgentEndEvent,
    ChatRequest,
    Message,
    MessageEndEvent,
    MessageUpdateEvent,
    ModelSpec,
    StreamAttemptReceipt,
    StreamEndEvent,
    StreamEvent,
    StreamStartEvent,
    StreamTextDelta,
    TextContent,
    ToolCall,
)
from local_operator.headless_print import PrintRenderer
from local_operator.providers.account_affinity import (
    WireHintCapture,
    capture_wire_stream,
)
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.bedrock.converse import BedrockConverseClient
from local_operator.providers.clients import (
    AnthropicClient,
    GoogleClient,
    OpenAICompatClient,
)
from local_operator.providers.failover import stream_with_failover
from local_operator.server.utils.event_broker import EventBroker, job_channel
from local_operator.server.utils.sse_bridge import attach_broker
from local_operator.session.inline_ui import INSTRUCTIONS
from local_operator.session.runtime.inline_ui_egress import InlineUiEvents
from local_operator.session.transcript import Transcript
from local_operator.variables import VariableStore
from tests.unit.harness.test_one_turn_secret import _secret_tool, _tool_delta
from tests.unit.providers.bedrock.test_converse_body import NOVA, TOOL
from tests.unit.providers.bedrock.test_converse_stream import KEY, events, wire
from tests.unit.providers.test_failover import FakeAuth
from tests.unit.providers.test_native_replay import sse
from tests.unit.session.test_peer_cycle_display import exercise as exercise_peer_cycle
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
                capture = WireHintCapture()
                result = []
                async for event in capture_wire_stream(
                    client.stream(call, "synthetic-key", access), capture
                ):
                    assert capture.inline_ui_source_version == (1 if hint and suffix else None)
                    result.append(event)
                starts = [event for event in result if isinstance(event, StreamStartEvent)]
                if route == "anthropic-no-start":
                    assert not starts
                    assert isinstance(result[0], StreamTextDelta)
                else:
                    [start] = starts
                    assert result[0] is start
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
_NATIVE_IDS = {"messages": "wire-anthropic", "responses": "resp-actual"}
_CREATED = {"type": "response.created", "response": {"id": _NATIVE_IDS["responses"]}}


def _native_reply(route: str, lifecycle: str, *, native_id: str | None = None) -> bytes:
    if route == "messages":
        opener_type, envelope = "message_start", "message"
        block_start, template, *terminal = _ANTHROPIC_CONTENT
        prefix = [block_start]
        first = {**template, "delta": {"type": "text_delta", "text": _OPENER}}
        second = {**template, "delta": {"type": "text_delta", "text": _CLOSER}}
    else:
        opener_type, envelope = "response.created", "response"
        prefix = [{"type": "response.in_progress", "response": {"id": "not-created-id"}}]
        first = {
            "type": "response.output_text.delta",
            "item_id": "msg-not-a-response",
            "delta": _OPENER,
        }
        second = {**first, "delta": _CLOSER}
        terminal = [
            {"type": "response.completed", "response": {"id": "not-an-opener-id", "output": []}}
        ]

    opener = {"type": opener_type, envelope: {"id": native_id or _NATIVE_IDS[route]}}
    blank = {**opener, envelope: {"id": "  "}}
    conflict = {**opener, envelope: {"id": "conflicting-id"}}
    if lifecycle == "early":
        return sse([opener, *prefix, first, second, *terminal])
    if lifecycle == "duplicate":
        return sse([opener, opener, conflict, *prefix, first, second, *terminal])
    if lifecycle == "late":
        return sse([*prefix, first, blank, opener, opener, conflict, second, *terminal])
    if lifecycle == "tail":
        return sse([*prefix, first, second, *terminal, opener])
    assert lifecycle == "missing"
    return sse([*prefix, first, second, *terminal])


@pytest.mark.parametrize(
    ("route", "lifecycle", "primary", "hinted"),
    [
        *[("responses", row, None, True) for row in ("early", "late", "missing", "duplicate")],
        *[("messages", row, None, True) for row in ("early", "late", "missing", "duplicate")],
        ("responses", "tail", None, True),
        ("responses", "missing", None, False),
        ("responses", "missing", "anthropic", False),
        ("chat/completions", "early", "anthropic", False),
        ("responses", "late", "anthropic", True),
        ("messages", "late", "openai", True),
    ],
)
async def test_native_acceptance_and_private_receipts_preserve_egress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    route: str,
    lifecycle: str,
    primary: Literal["anthropic", "openai"] | None,
    hinted: bool,
) -> None:
    posts: list[tuple[str, bool]] = []

    def post(path: str, headers: dict[str, str], body: dict[str, Any]) -> bytes:
        posts.append((path, INSTRUCTIONS in json.dumps(body)))
        if path.endswith("/messages"):
            if route == "messages":
                return _native_reply(route, lifecycle)
            return sse(
                [
                    # An empty content block is not acceptance. The genuine
                    # late opener is safe to retry until output is forwarded.
                    _ANTHROPIC_CONTENT[0],
                    {"type": "message_start", "message": {"id": "failed-native"}},
                    {"type": "error", "error": {"type": "overloaded_error", "message": "Fixture"}},
                ]
            )
        if path.endswith("/chat/completions"):
            if route == "chat/completions":
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
        assert path.endswith("/responses")
        if route == "responses":
            return _native_reply(route, lifecycle)
        return sse(
            [
                {"type": "response.output_text.delta", "delta": ""},
                {"type": "response.created", "response": {"id": "failed-openai"}},
                {"type": "error", "code": "server_error", "message": "Fixture"},
            ]
        )

    wire_boundaries: list[tuple[str, str | int | None]] = []
    acceptance: list[Any] = []
    updates: list[dict[str, Any]] = []
    endings: list[dict[str, Any]] = []
    markers: list[int | None] = []
    projector = InlineUiEvents(sidecars=True)
    broker = EventBroker()

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
        fallback = {
            "chat/completions": "openrouter/openai/gpt-4o",
            "messages": "anthropic/claude-sonnet-4-6",
            "responses": "openai/gpt-6-astra",
        }[route]
        settings = {"retry": {"maxRetries": 0, "fallbackChains": {"default": [fallback]}}}

        async def stream(request: ChatRequest, signal: AbortSignal) -> AsyncIterator[StreamEvent]:
            async for event in stream_with_failover(
                request, auth, settings, client_for, signal=signal
            ):
                if isinstance(event, StreamAttemptReceipt):
                    wire_boundaries.append((event.type, event.inline_ui_source_version))
                elif isinstance(event, StreamStartEvent):
                    wire_boundaries.append((event.type, event.response_id))
                elif isinstance(event, StreamTextDelta):
                    wire_boundaries.append((event.type, None))
                yield event

        def on_event(event: Any) -> None:
            # Include any experimental provider or receipt frame so an unknown
            # public event cannot disappear from the CLI/quiet-cycle assertion.
            if event.type.startswith("provider_turn_") or event.type == "attempt_receipt":
                acceptance.append(event)
            elif isinstance(event, MessageUpdateEvent):
                assert isinstance(event.message, Message)
                markers.append(event.message._inline_ui_source_version)
                updates.append(projector.dump(event))
            elif isinstance(event, (MessageEndEvent, AgentEndEvent)):
                endings.append(projector.dump(event))

        session = make_session(
            tmp_path,
            stream,
            model={
                "anthropic": ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6"),
                "openai": ModelSpec(
                    provider="openai", model_id="gpt-4o", supports_responses_api=True
                ),
                None: (
                    ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6")
                    if route == "messages"
                    else ModelSpec(
                        provider="openai", model_id="gpt-4o", supports_responses_api=True
                    )
                ),
            }[primary],
        )
        session.subscribe(on_event)
        renderer = PrintRenderer(json_mode=True)
        detach_renderer = renderer.attach(session)
        attach_broker(session, broker, "inline-ui")
        try:
            await session.prompt("Show June counts.", inline_ui=bool(primary) or hinted)
            journal_directory = session.transcript.directory
        finally:
            detach_renderer()
            await session.dispose()

    journal = Transcript(journal_directory, read_only=True)
    [durable] = [row for row in journal.entries() if row.payload.get("role") == "assistant"]
    [rehydrated] = [
        msg
        for msg in journal.build_llm_history()
        if isinstance(msg, Message) and msg.id == durable.id
    ]
    # Inspect projection before asserting metadata: on the old code the
    # opener escaped unprojected, or inherited the failed attempt's marker.
    assert len(updates) >= 2
    if route != "chat/completions":
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
    assert durable.inline_ui_source_version == receipt
    assert "inline_ui_source_version" not in durable.payload
    assert "".join(block["text"] for block in durable.payload["content"]) == "".join(deltas)
    assert "".join(text for _, text in (durable.inline_ui_source_blocks or ())) == (
        _OPENER + _CLOSER if hinted else ""
    )
    [message_end] = [
        frame
        for frame in endings
        if frame["type"] == "message_end" and frame["message"]["role"] == "assistant"
    ]
    [agent_end] = [frame for frame in endings if frame["type"] == "agent_end"]
    [final_assistant] = [
        message for message in agent_end["messages"] if message["role"] == "assistant"
    ]
    assert message_end["message"] == final_assistant
    assert "".join(block["text"] for block in final_assistant["content"]) == "".join(deltas)
    durable_egress = InlineUiEvents(sidecars=True).dump(MessageEndEvent(message=rehydrated))
    if hinted:
        assert message_end["inline_ui"] == agent_end["inline_ui"] == updates[-1]["inline_ui"]
        assert durable_egress["inline_ui"] == message_end["inline_ui"]
    else:
        assert "inline_ui" not in message_end and "inline_ui" not in agent_end
        assert "inline_ui" not in durable_egress
    assert posts[-1] == ("/v1/messages" if route == "messages" else f"/{route}", hinted)
    # Use the real CLI writer and canonical SSE publisher, not a second
    # acceptance parser. Only IDs from on-wire native openers may reach them.
    cli = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    sse_frames = broker.retained(job_channel("inline-ui"))
    assert all("identity_only" not in e.data and e.name != "attempt_receipt" for e in sse_frames)
    assert all(frame["type"] != "attempt_receipt" for frame in cli)
    provider, model = fallback.split("/", 1)
    if route == "responses" and primary is None:
        model = "gpt-4o"
    if route == "chat/completions":
        # The original compatible Start uses the loop selector on fallback;
        # this is not a new native proof path and its attribution stays intact.
        provider, model = "anthropic", "claude-sonnet-4-6"
    known_id = "chat-actual" if route == "chat/completions" else _NATIVE_IDS[route]
    public: list[tuple[str, str | None, str | None, str | None]] = [
        ("delta", None, None, None)
    ] * len(updates)
    if lifecycle in {"early", "duplicate", "late"}:
        public.insert(int(lifecycle == "late"), ("start", known_id, provider, model))
    primary_start = {
        "anthropic": ("start", "failed-native", "anthropic", "claude-sonnet-4-6"),
        "openai": ("start", "failed-openai", "openai", "gpt-4o"),
    }.get(primary or "")
    if primary_start:
        public.insert(0, primary_start)
    assert [
        (kind, frame.get("response_id"), frame.get("provider"), frame.get("model_id"))
        for frame in cli
        if (kind := {"provider_turn_start": "start", "message_update": "delta"}.get(frame["type"]))
    ] == public
    assert [
        (
            kind,
            event.data.get("response_id"),
            event.data.get("provider"),
            event.data.get("model_id"),
        )
        for event in sse_frames
        if (kind := {"provider.start": "start", "message.delta": "delta"}.get(event.name))
    ] == public
    assert all(event.type == "provider_turn_start" for event in acceptance)
    # Feed the actual lifecycle frames into a pending peer's quiet-tool path.
    # A new unknown typed event would clear the candidate instead of sealing.
    assert len(exercise_peer_cycle(before_tools=acceptance).records) == 1

    raw: list[tuple[str, str | int | None]] = [("text_delta", None)] * len(updates)
    if lifecycle in {"early", "duplicate", "late"}:
        raw.insert(int(lifecycle == "late"), ("start", known_id))
    if hinted or primary:
        raw.insert(0, ("attempt_receipt", receipt))
    if primary_start:
        raw[:0] = [("attempt_receipt", 1), ("start", primary_start[1])]
    assert wire_boundaries == raw
    print(f"{primary or 'direct'} / {route} / {lifecycle}: {posts=}; {public=}; {deltas=}")


@pytest.mark.parametrize("route", ["responses", "messages"])
async def test_native_late_start_cannot_publish_a_one_turn_secret(
    tmp_path: Path, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The native tool is real; the unrelated process-wide ledger writer is not
    # part of this fixture's egress boundary and should not outlive it.
    monkeypatch.setattr("local_operator.analytics.record_tool_call", lambda *_: None)
    canary = "synthetic-one-turn-late-id-canary"
    provider = "openai" if route == "responses" else "anthropic"
    model = ModelSpec(
        provider=provider,
        model_id="gpt-4o" if route == "responses" else "claude-sonnet-4-6",
        supports_responses_api=route == "responses",
    )
    posts: list[dict[str, Any]] = []
    attempts = 0

    def post(_path: str, _headers: dict[str, str], body: dict[str, Any]) -> bytes:
        posts.append(body)
        return _native_reply(route, "late", native_id=canary)

    broker = EventBroker()
    scrubbed_ids: list[str | None] = []
    async with _local_sse(post) as base, httpx.AsyncClient(trust_env=False, timeout=5) as http:
        client = (
            OpenAICompatClient(base, openai_api="responses", http_client=http)
            if route == "responses"
            else AnthropicClient(base, http_client=http)
        )

        async def client_for(_spec: ModelSpec) -> Any:
            return client

        async def stream(request: ChatRequest, signal: AbortSignal) -> AsyncIterator[StreamEvent]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                for event in _tool_delta("secret-id-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert attempts == 2 and request.has_one_turn_secrets
            async for event in stream_with_failover(
                request, FakeAuth({provider: ["synthetic-key"]}), None, client_for, signal=signal
            ):
                if isinstance(event, StreamStartEvent):
                    scrubbed_ids.append(event.response_id)
                yield event

        session = make_session(
            tmp_path,
            stream,
            tools=[_secret_tool("native-secret", canary)],
            variables=VariableStore(),
            model=model,
        )
        attach_broker(session, broker, "secret-late-id")
        try:
            await session.prompt("Use the native tool, then answer.")
        finally:
            await session.dispose()

    assert len(posts) == 1 and json.dumps(posts[0]).count(canary) == 1
    assert scrubbed_ids == [None]
    public = [event.data for event in broker.retained(job_channel("secret-late-id"))]
    assert [frame["response_id"] for frame in public if frame["type"] == "provider.start"] == [None]
    assert canary not in json.dumps(public)
    print(f"{route}: armed POST reached loopback; late ID was withheld from the broker")


@pytest.mark.parametrize("converse_first", [True, False])
@pytest.mark.parametrize("replayable", [True, False])
async def test_failover_uses_the_serving_adapters_inline_ui_receipt(
    converse_first: bool, replayable: bool, monkeypatch: pytest.MonkeyPatch
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
            200, content=wire(events()) if converse else _native_reply("responses", "early")
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
            replayable=replayable,
        )
        auth = FakeAuth({"openai": ["synthetic-key"], "bedrock": [KEY]})
        settings = {"retry": {"maxRetries": 0, "fallbackChains": {"default": [fallback]}}}
        result = [
            event async for event in stream_with_failover(request, auth, settings, client_for)
        ]

    assert {converse for converse, _ in received} == {True, False}
    assert all((INSTRUCTIONS in json.dumps(body)) is not converse for converse, body in received)
    boundaries = [
        (
            (event.type, event.inline_ui_source_version)
            if isinstance(event, StreamAttemptReceipt)
            else (event.type, event.response_id)
        )
        for event in result
        if isinstance(event, (StreamAttemptReceipt, StreamStartEvent))
    ]
    compatible = [("attempt_receipt", 1), ("start", _NATIVE_IDS["responses"])]
    converse = [("attempt_receipt", None), ("start", None)]
    # A buffered attempt that never gets committed leaks neither its public
    # acceptance nor its private receipt into the serving attempt.
    expected = compatible if converse_first else converse
    if not converse_first and not replayable:
        expected = compatible + expected
    assert boundaries == expected
    assert isinstance(result[-1], StreamEndEvent) and result[-1].stop_reason == "stop"
    print(f"{primary} → {fallback}: {replayable=}; {boundaries=}")
