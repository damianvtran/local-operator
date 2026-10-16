"""Radient client-correlation headers on the chat wire.

Chat does not go through ``RadientClient`` -- the registry builds a generic
``OpenAICompatClient`` for ``radient``/``radient-key`` -- so the PR-3
correlation headers the speech/transcription paths already sent never
reached a chat completion and almost no inference usage row carried a
``client_session_id``. These tests pin the chat half: both headers on the
streaming chat/completions path and on the Responses variant, the session
value taken from the request's cache lineage, a fresh request id per call,
an ungrammatical session id dropped silently, and every other provider
left byte-identical.
"""

import json
import re

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.providers.clients import OpenAICompatClient

RADIENT_BASE = "https://api.radienthq.com/v1"
SESSION_ID = "000471114995"  # the real lop session-id shape: 12 hex chars
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")


def _request(key=SESSION_ID, provider="radient", model="radient/auto"):
    return ChatRequest(
        model=ModelSpec(provider=provider, model_id=model, supports_prompt_cache=True),
        messages=[Message.user("synthetic")],
        system_blocks=["stable"],
        prompt_cache_key=key,
    )


def test_radient_headers_carry_session_and_fresh_request_id():
    client = OpenAICompatClient(RADIENT_BASE)
    url = f"{RADIENT_BASE}/chat/completions"
    first = client._radient_correlation_headers(_request(), url)
    assert first["X-Radient-Client-Session-Id"] == SESSION_ID
    assert _HEX32.match(first["X-Radient-Client-Request-Id"])
    # Same conversation, next call: same session id, NEW request id -- each
    # HTTP attempt can produce its own usage row and needs its own join key.
    second = client._radient_correlation_headers(_request(), url)
    assert second["X-Radient-Client-Session-Id"] == SESSION_ID
    assert second["X-Radient-Client-Request-Id"] != first["X-Radient-Client-Request-Id"]


@pytest.mark.parametrize("provider", ["radient", "radient-key"])
def test_both_radient_credential_flavours_carry_headers(provider):
    client = OpenAICompatClient(RADIENT_BASE)
    headers = client._radient_correlation_headers(
        _request(provider=provider), f"{RADIENT_BASE}/chat/completions"
    )
    assert headers["X-Radient-Client-Session-Id"] == SESSION_ID


def test_custom_provider_pointed_at_radient_host_carries_headers():
    # A custom OpenAI-compatible provider whose actual request target is the
    # hub is served by the same server, so the host gates the headers too.
    client = OpenAICompatClient("https://gateway.internal/v1")
    headers = client._radient_correlation_headers(
        _request(provider="custom"), f"{RADIENT_BASE}/chat/completions"
    )
    assert headers["X-Radient-Client-Session-Id"] == SESSION_ID


@pytest.mark.parametrize(
    "provider,base",
    [
        ("openai", "https://api.openai.com/v1"),
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("deepseek", "https://api.deepseek.com/v1"),
        ("xai", "https://api.x.ai/v1"),
        ("custom", "https://gateway.internal/v1"),
    ],
)
def test_other_providers_are_unchanged(provider, base):
    client = OpenAICompatClient(base)
    assert (
        client._radient_correlation_headers(
            _request(provider=provider, model="some-model"), f"{base}/chat/completions"
        )
        == {}
    )


@pytest.mark.parametrize("key", ["has/slash", "x" * 65, "white space", None, "", "   "])
def test_ungrammatical_session_id_is_dropped_silently(key):
    # The grammar gate is the speech paths' own: an ungrammatical lineage
    # omits ONLY the session header (never raises, never fails the call) and
    # the fresh per-call request id still rides.
    client = OpenAICompatClient(RADIENT_BASE)
    headers = client._radient_correlation_headers(
        _request(key=key), f"{RADIENT_BASE}/chat/completions"
    )
    assert "X-Radient-Client-Session-Id" not in headers
    assert _HEX32.match(headers["X-Radient-Client-Request-Id"])


def _chat_sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _chat_wire(provider="radient", base=RADIENT_BASE, key=SESSION_ID):
    wire: list[httpx.Request] = []

    def respond(request):
        wire.append(request)
        return httpx.Response(
            200,
            content=_chat_sse(
                [
                    {"id": "chatcmpl-r", "choices": [{"delta": {"content": "ok"}}]},
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                    },
                ]
            ),
        )

    return wire, respond, provider, base, key


@pytest.mark.asyncio
async def test_chat_completions_stream_carries_both_headers():
    wire, respond, *_ = _chat_wire()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = OpenAICompatClient(RADIENT_BASE, http_client=http)
        async for _event in client.stream(_request(), "synthetic-key"):
            pass
        # A second call in the same conversation: session stable, request fresh.
        async for _event in client.stream(_request(), "synthetic-key"):
            pass
    assert len(wire) == 2
    request_ids = set()
    for sent in wire:
        assert str(sent.url) == f"{RADIENT_BASE}/chat/completions"
        assert sent.headers["X-Radient-Client-Session-Id"] == SESSION_ID
        assert _HEX32.match(sent.headers["X-Radient-Client-Request-Id"])
        request_ids.add(sent.headers["X-Radient-Client-Request-Id"])
    assert len(request_ids) == 2


@pytest.mark.asyncio
async def test_chat_completions_other_provider_carries_no_correlation_headers():
    wire, respond, *_ = _chat_wire(provider="openrouter", base="https://openrouter.ai/api/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = OpenAICompatClient("https://openrouter.ai/api/v1", http_client=http)
        request = _request(provider="openrouter", model="some/model")
        async for _event in client.stream(request, "synthetic-key"):
            pass
    assert "X-Radient-Client-Session-Id" not in wire[0].headers
    assert "X-Radient-Client-Request-Id" not in wire[0].headers


@pytest.mark.asyncio
async def test_chat_completions_ungrammatical_session_id_dropped_on_wire():
    wire, respond, *_ = _chat_wire()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = OpenAICompatClient(RADIENT_BASE, http_client=http)
        async for _event in client.stream(_request(key="bad/id"), "synthetic-key"):
            pass
    assert "X-Radient-Client-Session-Id" not in wire[0].headers
    assert _HEX32.match(wire[0].headers["X-Radient-Client-Request-Id"])


@pytest.mark.asyncio
async def test_responses_variant_carries_both_headers():
    """The Responses wire shape (a directly-constructed client is the only
    way Radient reaches it today) carries the same pair, so both variants
    of the OpenAI-compatible client are session-correlated."""
    wire: list[httpx.Request] = []

    def respond(request):
        wire.append(request)
        return httpx.Response(
            200,
            content=(
                'data: {"type":"response.completed","response":{"output":[],'
                '"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
            ),
        )

    request = _request()
    request = request.model_copy(
        update={"model": request.model.model_copy(update={"supports_responses_api": True})}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = OpenAICompatClient(RADIENT_BASE, http_client=http, openai_api="responses")
        async for _event in client.stream(request, "synthetic-key"):
            pass
    assert str(wire[0].url) == f"{RADIENT_BASE}/responses"
    assert wire[0].headers["X-Radient-Client-Session-Id"] == SESSION_ID
    assert _HEX32.match(wire[0].headers["X-Radient-Client-Request-Id"])
