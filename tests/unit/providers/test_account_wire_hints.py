"""Lifetimes are derived at the real built-in wire builder, not guessed by auth."""

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.providers.account_affinity import clear_wire_hint, wire_hint
from local_operator.providers.clients import (
    AnthropicClient,
    OpenAICompatClient,
    _anthropic_cache_marker_ttl_ms,
)


def test_nested_tool_input_cannot_invent_a_cache_lifetime():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "input": {"cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    }
                ],
            }
        ]
    }
    assert _anthropic_cache_marker_ttl_ms(body) == 0
    body["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral"}
    assert _anthropic_cache_marker_ttl_ms(body) == 300_000


def request(provider="anthropic", model="claude-sonnet-5", **updates):
    return ChatRequest(
        model=ModelSpec(provider=provider, model_id=model, supports_prompt_cache=True),
        system_blocks=["Synthetic stable prefix", "Synthetic tail"],
        messages=[Message.user("Synthetic question")],
        prompt_cache_key="synthetic-cache",
        **updates,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hint,threshold,ttl",
    [
        (999, 1000, 300_000),
        (1000, 1000, 3_600_000),
        (1001, 1000, 3_600_000),
        (None, 1000, 300_000),
        (1001, 0, 300_000),
    ],
)
async def test_anthropic_threshold_matches_actual_request_markers(hint, threshold, ttl):
    clear_wire_hint()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200))
    ) as http:
        client = AnthropicClient(http_client=http, cache_ttl_1h_min_context_tokens=threshold)
        body = client._build_body(request(context_tokens_hint=hint))
    actual = wire_hint()
    assert actual.ttl_ms == ttl
    assert actual.ttl_source == "explicit-wire"

    def markers(value):
        if isinstance(value, dict):
            if "cache_control" in value:
                yield value["cache_control"]
            for child in value.values():
                yield from markers(child)
        elif isinstance(value, list):
            for child in value:
                yield from markers(child)

    actual_markers = list(markers(body))
    assert actual_markers
    assert {marker.get("ttl", "5m") for marker in actual_markers} == {
        "1h" if ttl == 3_600_000 else "5m"
    }


@pytest.mark.asyncio
async def test_unknown_compatible_endpoints_do_not_borrow_first_party_retention():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200))
    ) as http:
        clear_wire_hint()
        AnthropicClient(base_url="https://gateway.example.invalid", http_client=http)._build_body(
            request()
        )
        assert wire_hint() is None
        clear_wire_hint()
        OpenAICompatClient(
            "https://gateway.example.invalid/v1", http_client=http
        )._build_responses_body(request("openai", "gpt-5.6"))
        assert wire_hint() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,ttl,source",
    [
        ("gpt-5.6", 1_800_000, "documented-default"),
        ("gpt-6", 1_800_000, "documented-default"),
        ("gpt-5", 1_800_000, "bounded-heuristic"),
    ],
)
async def test_public_responses_documented_default_vs_legacy_heuristic(model, ttl, source):
    clear_wire_hint()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200))
    ) as http:
        OpenAICompatClient("https://api.openai.com/v1", http_client=http)._build_responses_body(
            request("openai", model)
        )
    assert wire_hint().ttl_ms == ttl
    assert wire_hint().ttl_source == source


@pytest.mark.asyncio
async def test_codex_and_legacy_chat_are_explicitly_short_heuristics():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200))
    ) as http:
        client = OpenAICompatClient("https://api.openai.com/v1", http_client=http)
        clear_wire_hint()
        body = client._build_codex_responses_body(request("openai-device", "gpt-5"))
        assert "prompt_cache_retention" not in body
        assert wire_hint().ttl_ms == 300_000
        assert wire_hint().ttl_source == "bounded-heuristic"
        clear_wire_hint()
        client._build_body(request("openai", "gpt-4o"))
        assert wire_hint().ttl_ms == 300_000
        assert wire_hint().endpoint_mode == "openai:chat"
