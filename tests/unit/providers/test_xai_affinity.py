"""xAI cache affinity is transport-only, scoped, and stable across failover.

Mirrors ``test_openai_affinity.py``: the header is a routing key derived from
the cache lineage alone, present only where xAI actually serves the request,
and identical across the ``xai``/``xai-oauth`` credential flavours so failover
keeps a live conversation on its warm cache server.
"""

import json
import uuid
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    TextContent,
)
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.replay import credential_scope

XAI_BASE = "https://api.x.ai/v1"


def _request(key="lineage", provider="xai", model="grok-4.6"):
    return ChatRequest(
        model=ModelSpec(provider=provider, model_id=model, supports_prompt_cache=True),
        messages=[Message.user("synthetic")],
        system_blocks=["stable"],
        prompt_cache_key=key,
    )


def test_conv_id_preserves_retry_resume_and_fork_lineage():
    client = OpenAICompatClient(XAI_BASE)
    url = f"{XAI_BASE}/chat/completions"
    headers = client._grok_conv_headers(_request(), url)
    assert uuid.UUID(headers["x-grok-conv-id"]).version == 5
    assert client._grok_conv_headers(_request(), url) == headers
    resumed = OpenAICompatClient(XAI_BASE)
    for model in ("grok-4.5", "future-model"):
        assert resumed._grok_conv_headers(_request(model=model), url) == headers
    # A fork's inherited lineage routes together; another lineage must not
    # collapse into one application-wide cache group.
    assert resumed._grok_conv_headers(_request("other-lineage"), url) != headers


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_lineage_keeps_credential_agnostic_routing(key):
    # Isolated calls carry no key: no header, exactly as before this existed.
    client = OpenAICompatClient(XAI_BASE)
    assert client._grok_conv_headers(_request(key), f"{XAI_BASE}/chat/completions") == {}


@pytest.mark.parametrize(
    "base,provider,url",
    [
        # The credential flavour must not change the routing: xai and
        # xai-oauth share the xAI fleet, so failover between them keeps the
        # SAME conv id (both derive from prompt_cache_key alone).
        (XAI_BASE, "xai", f"{XAI_BASE}/chat/completions"),
        (XAI_BASE, "xai-oauth", f"{XAI_BASE}/chat/completions"),
        # A custom OpenAI-compatible provider pointed at api.x.ai is routed by
        # the same fleet, so the HOST gates the header too (omp precedent).
        ("https://api.x.ai/v1", "custom", "https://api.x.ai/v1/chat/completions"),
        ("https://gateway.internal/v1", "custom", "https://api.x.ai/v1/chat/completions"),
    ],
)
def test_header_present_where_xai_routes(base, provider, url):
    assert "x-grok-conv-id" in OpenAICompatClient(base)._grok_conv_headers(
        _request(provider=provider), url
    )


@pytest.mark.parametrize(
    "base,provider,url",
    [
        ("https://api.openai.com/v1", "openai", "https://api.openai.com/v1/chat/completions"),
        ("https://api.z.ai/api/coding/paas/v4", "zai", "https://api.z.ai/api/coding/paas/v4/x"),
        ("https://openrouter.ai/api/v1", "openrouter", "https://openrouter.ai/api/v1/x"),
        ("https://api.moonshot.cn/v1", "kimi", "https://api.moonshot.cn/v1/chat/completions"),
        # A provider gate match must not send grok headers to OpenAI's own
        # fleet: an openai-branded request to the codex responses URL never
        # touches the xAI routing layer.
        ("https://api.x.ai/v1", "openai", "https://chatgpt.com/backend-api/codex/responses"),
        ("https://gateway.internal/v1", "custom", "https://gateway.internal/v1/chat/completions"),
    ],
)
def test_header_never_leaks_to_other_providers_or_hosts(base, provider, url):
    assert OpenAICompatClient(base)._grok_conv_headers(_request(provider=provider), url) == {}


def test_failover_between_credential_flavours_keeps_conv_id():
    """The header derives from the lineage, not the credential, so an xai →
    xai-oauth failover (fresh client instance, different bearer) mid-session
    must not re-route the conversation to a cold server."""
    api_key_client = OpenAICompatClient(XAI_BASE)
    oauth_client = OpenAICompatClient(XAI_BASE)  # failover builds a new client
    url = f"{XAI_BASE}/chat/completions"
    first = api_key_client._grok_conv_headers(_request(), url)
    second = oauth_client._grok_conv_headers(_request(provider="xai-oauth"), url)
    assert first == second


def test_header_contains_neither_raw_lineage_nor_credentials():
    key = "~/private/session\r\nInjected: value ☃"
    client = OpenAICompatClient(XAI_BASE)
    headers = client._grok_conv_headers(_request(key), f"{XAI_BASE}/chat/completions")
    assert set(headers) == {"x-grok-conv-id"}
    value = headers["x-grok-conv-id"]
    assert len(value) == 36
    assert str(uuid.UUID(value)) == value
    assert key not in value


def _chat_sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_dispatch_adds_header_without_changing_body():
    """End to end on the chat path xai actually uses: header on the wire,
    and the chat body carries ``prompt_cache_key``.

    xAI lists ``prompt_cache_key`` as a first-class chat-completions field
    plumbed to ``x-grok-conv-id``, so the stamp that every cache-capable
    openai-compat client now sends is expected here too — the routing header
    is additive, not a substitute for the body field.
    """
    wire: list[httpx.Request] = []

    def respond(request):
        wire.append(request)
        return httpx.Response(
            200,
            content=_chat_sse(
                [
                    {"id": "chatcmpl-x", "choices": [{"delta": {"content": "ok"}}]},
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                    },
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = OpenAICompatClient(XAI_BASE, http_client=http)
        request = _request()
        # The stream computes the same scope from the credential; the routing
        # header is additive — the body still carries prompt_cache_key.
        expected_body = client._build_body(request, scope=credential_scope("synthetic-key"))
        for _ in range(2):  # retry/resume keeps the same conv id
            async for _event in client.stream(request, "synthetic-key"):
                pass
    assert len(wire) == 2
    ids = set()
    for sent in wire:
        assert str(sent.url) == f"{XAI_BASE}/chat/completions"
        assert json.loads(sent.content) == expected_body
        ids.add(sent.headers["x-grok-conv-id"])
    assert len(ids) == 1
    assert json.loads(wire[0].content)["prompt_cache_key"] == "lineage"


@pytest.mark.asyncio
async def test_responses_dispatch_also_carries_header():
    """The Responses path keeps its ``prompt_cache_key`` body field AND adds
    the routing header, so a direct-construction client (the only way xai
    reaches /responses today) stays affinity-routed on both wire shapes."""
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
        client = OpenAICompatClient(XAI_BASE, http_client=http, openai_api="responses")
        async for _event in client.stream(request, "synthetic-key"):
            pass
    assert str(wire[0].url) == f"{XAI_BASE}/responses"
    body = json.loads(wire[0].content)
    assert body["prompt_cache_key"] == "lineage"  # unchanged body behaviour
    conv_id = wire[0].headers["x-grok-conv-id"]
    # Same derivation as the chat path: one conversation, one routing key.
    assert conv_id == str(uuid.uuid5(uuid.NAMESPACE_URL, "local-operator:grok-cache:lineage"))


@pytest.mark.asyncio
async def test_grok_chat_reasoning_round_trip_survives_for_replay():
    """xAI's docs name omitted ``reasoning_content`` as the top cause of cache
    misses on reasoning models, so pin the persist→replay round-trip on the
    grok chat wire: captured deltas land in ``native_replay`` and are replayed
    on the next turn of the same endpoint+credential."""
    state: dict[str, list[dict[str, Any]]] = {"bodies": []}

    def respond(request):
        state["bodies"].append(json.loads(request.content))
        if len(state["bodies"]) == 1:
            deltas = [
                {"id": "chatcmpl-r", "choices": [{"delta": {"reasoning_content": "Private "}}]},
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "plan", "content": "GROK_OK"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ]
        else:
            deltas = [
                {"id": "chatcmpl-r2", "choices": [{"delta": {"content": "ok"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        usage = {"usage": {"prompt_tokens": 9, "completion_tokens": 2}}
        return httpx.Response(200, content=_chat_sse([*deltas, usage]))

    request = _request()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = OpenAICompatClient(XAI_BASE, http_client=http)
        events = [e async for e in client.stream(request, "synthetic-key")]
        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.provider_payload is not None
        items = end.provider_payload["native_replay"]["items"]
        assert items == [{"reasoning_content": "Private plan"}]
        history = Message(
            role="assistant",
            content=[TextContent(text="GROK_OK")],
            provider_payload=end.provider_payload,
        )
        second = request.model_copy(update={"messages": [*request.messages, history]})
        async for _event in client.stream(second, "synthetic-key"):
            pass
    assert state["bodies"][1]["messages"][-1]["reasoning_content"] == "Private plan"
