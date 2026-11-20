"""Google/Vertex future-generation replay and silent, local cache diagnosis.

The SSE transport is synthetic; the builders, stream parser, SessionStreamFn
recorder and SQLite store are real. No test needs a credential or live model.
"""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest

from local_operator.analytics.store import AnalyticsStore
from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamStartEvent,
    Usage,
)
from local_operator.model.configure import SessionStreamFn, create_stream_fn
from local_operator.providers import clients
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import GoogleClient, _is_gemini_3_or_later
from local_operator.providers.failover import ProviderError


@pytest.fixture(autouse=True)
def _isolated_prefixes():
    clients._GOOGLE_PREFIX_SNAPSHOTS.clear()
    yield
    clients._GOOGLE_PREFIX_SNAPSHOTS.clear()


@pytest.mark.parametrize(
    "model_id,expected",
    [
        (name, True)
        for name in (
            "gemini-3-flash",
            "gemini-3.8-flash",
            "gemini-4-neon",
            "gemini-5-pro",
            "gemini-10-ultra",
            "google/gemini-4-neon",
        )
    ]
    + [
        (name, False)
        for name in (
            "gemini-2.5-pro",
            "gemini-1.5-pro",
            "gemini-exp-1206",
            "gemma-3-27b",
            "xgemini-4",
        )
    ],
)
def test_generation_gate(model_id: str, expected: bool) -> None:
    assert _is_gemini_3_or_later(model_id) is expected


def _request(provider: str = "google", **updates: Any) -> ChatRequest:
    request = ChatRequest(
        model=ModelSpec(provider=provider, model_id="gemini-4-neon"),
        system_blocks=["stable"],
        messages=[Message.user("hi")],
        cache_diagnostics=True,
        cache_diagnostics_chain="synthetic-chain",
    )
    return request.model_copy(update=updates)


@pytest.mark.parametrize("model_id,new", [("gemini-4-neon", True), ("gemini-2.5-pro", False)])
@pytest.mark.asyncio
async def test_imported_calls_and_tool_images(model_id: str, new: bool) -> None:
    request = _request(
        model=ModelSpec(provider="google", model_id=model_id),
        messages=[
            Message.user("inspect"),
            Message.assistant("", tool_calls=[{"id": "probe", "name": "probe", "arguments": {}}]),
            Message(
                role="tool",
                tool_name="probe",
                tool_call_id="probe",
                content=[
                    ImageContent(mime_type="image/png", data="c3ludGhldGlj"),
                ],
            ),
        ],
    )
    async with httpx.AsyncClient() as http:
        client = GoogleClient(http_client=http)
        contents = client._build_body(request)["contents"]
    call = contents[1]["parts"][0]
    assert call.get("thoughtSignature") == ("skip_thought_signature_validator" if new else None)
    parts = contents[2]["parts"]
    image = {"inlineData": {"mimeType": "image/png", "data": "c3ludGhldGlj"}}
    if new:
        assert parts[0]["functionResponse"]["parts"] == [image]
        assert len(parts) == 1
    else:
        assert "parts" not in parts[0]["functionResponse"]
        assert parts[1] == image


def _sse(prompt: int = 30_000, cached: int = 0, *, finish: str = "STOP", text: str = "ok") -> bytes:
    payload = {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {
            "promptTokenCount": prompt,
            "cachedContentTokenCount": cached,
            "candidatesTokenCount": 1,
        },
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _end(
    http: httpx.AsyncClient, request: ChatRequest, key: str = "synthetic-key"
) -> StreamEndEvent:
    # Deliberately construct a new client EVERY turn, as the real failover
    # factory does. Instance-only state would make every diagnosis baseline.
    events = [event async for event in GoogleClient(http_client=http).stream(request, key)]
    return next(event for event in events if isinstance(event, StreamEndEvent))


@pytest.mark.parametrize("provider", ["google", "google-vertex"])
@pytest.mark.asyncio
async def test_multi_turn_prefix_diagnosis(provider: str) -> None:
    cached = 0
    prompt = 30_000
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(prompt, cached))

    request = _request(provider)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:

        async def check(kind: str, missed: int) -> None:
            before = copy.deepcopy(request)
            end = await _end(http, request)
            assert (end.cache_miss_type, end.cache_missed_tokens) == (kind, missed)
            assert request == before
            expected_body = GoogleClient(http_client=http)._build_body(request)
            assert bodies[-1] == expected_body, "diagnosis must not mutate wire content"
            assert "cache_miss_type" not in end.model_dump()
            assert "cache_missed_tokens" not in end.model_dump()

        await check("baseline", 0)
        cached = 30_000
        request = request.model_copy(update={"messages": [*request.messages, Message.user("next")]})
        await check("unchanged", 0)
        cached = 0
        request = request.model_copy(update={"system_blocks": ["changed"]})
        await check("system_changed", 30_000)
        tool = AgentTool(name="probe", description="probe", execute=MagicMock())
        request = request.model_copy(update={"tools": [tool]})
        await check("tools_changed", 30_000)
        request = request.model_copy(update={"tool_choice": "none"})
        await check("tool_config_changed", 30_000)
        request = request.model_copy(update={"messages": [Message.user("mutated")]})
        await check("messages_changed", 30_000)
        await check("evicted_or_rerouted", 30_000)
        cached = 20_000
        await check("partial_eviction", 10_000)
        cached = 27_952  # 2048-token tolerance, inclusive
        await check("unchanged", 0)
        cached = 27_951
        await check("partial_eviction", 2049)
        prompt = 1000
        cached = 0
        await check("evicted_or_rerouted", 30_000)
        await check("unchanged", 0)


@pytest.mark.asyncio
async def test_failed_empty_and_aborted_calls_preserve_the_snapshot() -> None:
    mode = "ok"

    def handler(request: httpx.Request) -> httpx.Response:
        if mode == "http-error":
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return httpx.Response(
            200,
            content=_sse(
                finish="MALFORMED_FUNCTION_CALL" if mode == "error-finish" else "STOP",
                text="" if mode == "empty" else "ok",
                prompt=0 if mode == "no-usage" else 30_000,
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        request = _request()
        assert (await _end(http, request)).cache_miss_type == "baseline"
        original = dict(clients._GOOGLE_PREFIX_SNAPSHOTS)
        changed = request.model_copy(update={"system_blocks": ["failed prompt"]})
        for mode in (
            "http-error",
            "error-finish",
            "empty",
            "no-usage",
            "aborted",
            "aborted-after-usage",
        ):
            if mode == "http-error":
                with pytest.raises(ProviderError, match="unavailable"):
                    await _end(http, changed)
            elif mode.startswith("aborted"):
                stream = GoogleClient(http_client=http).stream(changed, "synthetic-key")
                if mode == "aborted":
                    await anext(stream)
                else:
                    async for event in stream:
                        if event.type == "usage":
                            break
                await cast(AsyncGenerator[Any, None], stream).aclose()
            else:
                end = await _end(http, changed)
                assert end.cache_miss_type is None
            assert dict(clients._GOOGLE_PREFIX_SNAPSHOTS) == original
        mode = "ok"
        assert (await _end(http, request)).cache_miss_type == "evicted_or_rerouted"


@pytest.mark.asyncio
async def test_route_model_account_and_conversation_isolation() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=_sse()))
    ) as http:
        request = _request()
        assert (await _end(http, request)).cache_miss_type == "baseline"
        for switched in (
            request.model_copy(
                update={"model": ModelSpec(provider="google", model_id="gemini-5-pro")}
            ),
            _request("google-vertex"),
            request,
        ):
            assert (await _end(http, switched)).cache_miss_type == "baseline"
        assert (await _end(http, request, "other-key")).cache_miss_type == "baseline"
        assert (await _end(http, request, "synthetic-key")).cache_miss_type == "baseline"
        fork = request.model_copy(
            update={"cache_diagnostics_chain": "fork", "prompt_cache_key": "same"}
        )
        assert (await _end(http, fork)).cache_miss_type == "baseline"
        for ignored in (
            request.model_copy(update={"purpose": "aside"}),
            request.model_copy(update={"cache_diagnostics": False}),
            request.model_copy(update={"cache_diagnostics_chain": None}),
        ):
            assert (await _end(http, ignored)).cache_miss_type is None
        assert len(clients._GOOGLE_PREFIX_SNAPSHOTS) == 2
        for i in range(clients._GOOGLE_PREFIX_SNAPSHOTS_MAX + 1):
            await _end(http, request.model_copy(update={"cache_diagnostics_chain": f"chain-{i}"}))
        assert len(clients._GOOGLE_PREFIX_SNAPSHOTS) == clients._GOOGLE_PREFIX_SNAPSHOTS_MAX
        assert not any(key[0] == "chain-0" for key in clients._GOOGLE_PREFIX_SNAPSHOTS)


@pytest.mark.parametrize("provider", ["google", "google-vertex"])
@pytest.mark.asyncio
async def test_real_configure_records_sqlite_silently(
    tmp_path, monkeypatch, caplog, capsys, provider: str
) -> None:
    from local_operator import analytics

    ledger = AnalyticsStore(tmp_path / "analytics.db")
    recorded: list[Any] = []

    def record(snapshot):
        recorded.append(snapshot)
        assert ledger.record_batch([snapshot]) == 1

    monkeypatch.setattr(analytics, "record_call", record)
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential(provider, {"key": "synthetic-key", "source": "login"})
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=_sse())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stream = create_stream_fn(store, {}, session_id="synthetic-parent", http_client=http)
    request = _request(provider, cache_diagnostics=False, cache_diagnostics_chain=None)
    events: list[Any] = []
    try:
        with caplog.at_level(logging.INFO, logger="local_operator.model.configure"):
            for turn in (request, request.model_copy(update={"system_blocks": ["new system"]})):
                events.extend([event async for event in stream(turn, None)])
        # A fork shares cache-lineage routing but owns its own comparison chain.
        fork = stream.fork("synthetic-child")
        try:
            events.extend([event async for event in fork(request, None)])
        finally:
            await fork.close()
        events.extend(
            [event async for event in stream(request.model_copy(update={"purpose": "aside"}), None)]
        )
    finally:
        await stream.close()
        await http.aclose()
        store.close()
        ledger.close()
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        rows = conn.execute(
            "SELECT cache_miss_type, cache_missed_tokens FROM calls ORDER BY id"
        ).fetchall()
    assert rows == [("baseline", 0), ("system_changed", 30_000), ("baseline", 0), ("", 0)]
    assert len(bodies) == 4, "local telemetry adds no provider requests"
    lines = [r.getMessage() for r in caplog.records if "[prompt cache break]" in r.getMessage()]
    assert len(lines) == 1
    assert "type=system_changed" in lines[0] and "missed_tokens=30000" in lines[0]
    output = capsys.readouterr()
    assert output.out == "" and output.err == ""
    for event in events:
        if isinstance(event, StreamEndEvent):
            assert "cache_miss_type" not in event.model_dump()
            assert "cache_missed_tokens" not in event.model_dump()
        elif isinstance(event, StreamStartEvent):
            # Preserve Anthropic's existing start-frame serialization contract;
            # Google contributes no diagnosis there.
            assert event.cache_miss_type is None
            assert event.cache_missed_tokens == 0


@pytest.mark.parametrize("provider", ["google", "google-vertex"])
@pytest.mark.parametrize(
    "kind,missed,read,logged",
    [
        ("system_changed", 30_000, 0, True),
        ("evicted_or_rerouted", 30_000, 0, True),
        ("tools_changed", 19_999, 0, False),
        ("tools_changed", 30_000, 25_000, False),
        ("baseline", 30_000, 0, False),
        ("unchanged", 30_000, 0, False),
        ("pending", 30_000, 0, False),
        ("partial_eviction", 30_000, 0, False),
    ],
)
def test_google_log_gate(
    caplog, provider: str, kind: str, missed: int, read: int, logged: bool
) -> None:
    stream = SessionStreamFn(MagicMock(), {}, "synthetic-session")
    with caplog.at_level(logging.INFO, logger="local_operator.model.configure"):
        stream._log_cache_break(
            Usage(input_tokens=40_000, cache_read_tokens=read),
            kind,
            missed,
            provider=provider,
        )
    assert len([r for r in caplog.records if "[prompt cache break]" in r.getMessage()]) == int(
        logged
    )


@pytest.mark.asyncio
async def test_end_diagnosis_does_not_replace_anthropic_start(tmp_path, monkeypatch) -> None:
    from local_operator import analytics

    recorded: list[Any] = []
    monkeypatch.setattr(analytics, "record_call", recorded.append)
    store = AuthStore(tmp_path / "auth.db")
    stream = create_stream_fn(store, {}, session_id="synthetic-start-wins")

    async def events():
        yield StreamStartEvent(cache_miss_type="tools_changed", cache_missed_tokens=123)
        yield StreamEndEvent(
            stop_reason="stop",
            usage=Usage(input_tokens=100),
            cache_miss_type="system_changed",
            cache_missed_tokens=999,
        )

    try:
        _ = [event async for event in stream._record_stream(_request(), events())]
    finally:
        await stream.close()
        store.close()
    assert len(recorded) == 1
    assert (recorded[0].cache_miss_type, recorded[0].cache_missed_tokens) == ("tools_changed", 123)


@pytest.mark.parametrize("model_id", ["gemini-4-neon", "gemini-3-flash"])
def test_long_tool_argument_elision_keeps_a_signature(model_id: str) -> None:
    """The live gemini-4-neon 400 ("Function call is missing a thought_signature").

    A tool argument over ``DEFAULT_REPLAY_BOUND_CHARS`` (an 8,615-char ``task``
    prompt in the incident) is elided by ``bound_replay_payloads`` into a COPY.
    The copy no longer matches the native replay's visible fingerprint, so the
    Google wire falls back to the normalized path, which must still stamp the
    documented validator-skip marker on Gemini 3 AND later; before the
    generation gate it did so only for ``gemini-3*``, so every Gemini 4 turn
    after a long tool argument was refused.
    """
    from local_operator.harness.replay_bound import (
        DEFAULT_REPLAY_BOUND_CHARS,
        bound_replay_payloads,
    )
    from local_operator.harness.types import ToolCall
    from local_operator.providers.replay import credential_scope, native_payload

    spec = ModelSpec(provider="google", model_id=model_id)
    scope = credential_scope("synthetic-key", None)
    prompt = "p" * 8_615
    assert len(prompt) > DEFAULT_REPLAY_BOUND_CHARS
    args = {"agent": "coder", "prompt": prompt}
    native_part = {
        "functionCall": {"name": "task", "args": args},
        "thoughtSignature": "synthetic-real-signature",
    }
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="fc_0_task", name="task", arguments=args)],
        provider_payload=native_payload(
            spec,
            clients.GOOGLE_API_URL,
            "google-content",
            [native_part],
            "",
            [{"id": "fc_0_task", "name": "task", "args": args}],
            scope,
        ),
    )
    history = [
        Message.user("delegate"),
        assistant,
        Message(role="tool", tool_name="task", tool_call_id="fc_0_task", content=[]),
    ]
    client = GoogleClient()

    # Negative control: unbounded, the native part (real signature) replays.
    native = client._build_body(ChatRequest(model=spec, messages=history), scope=scope)
    assert native["contents"][1]["parts"] == [native_part]

    bounded = bound_replay_payloads(history)
    assert bounded[1] is not assistant, "the long argument must be elided into a copy"
    body = client._build_body(ChatRequest(model=spec, messages=bounded), scope=scope)
    call_part = body["contents"][1]["parts"][0]
    assert "functionCall" in call_part
    assert call_part.get("thoughtSignature") == "skip_thought_signature_validator"
