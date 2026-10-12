"""Cache policy and tool semantics must follow the actual endpoint contract.

These wire tests use the real adapter with a synthetic transport, not live keys.
The public cache policy is documentation-backed; OAuth optional-field behavior
was also independently reproduced against the real provider before this patch.
"""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from local_operator.harness.types import AgentTool, ChatRequest, Message, ModelSpec
from local_operator.providers.clients import OpenAICompatClient


async def _unused_execute(*_args, **_kwargs):
    raise AssertionError("The request contract test must not dispatch a tool")


def _request(model_id: str = "gpt-6.1-sol", provider: str = "openai") -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(
            provider=provider,
            model_id=model_id,
            supports_responses_api=True,
            supports_prompt_cache=True,
        ),
        system_blocks=["Stable synthetic instructions."],
        messages=[Message.user("Synthetic input")],
        prompt_cache_key="synthetic-contract",
    )


@pytest.mark.parametrize("codex", [False, True])
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["list", "spawn"]},
                "prompt": {"type": "string"},
            },
            "required": ["op"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"parameters": {"type": "object", "additionalProperties": True}},
            "required": ["parameters"],
        },
    ],
)
def test_responses_tools_preserve_optional_and_freeform_schema(parameters, codex) -> None:
    original = copy.deepcopy(parameters)
    request = _request()
    request.tools = [
        AgentTool(
            name="synthetic_tool",
            description="Synthetic tool; never executes.",
            parameters=parameters,
            execute=_unused_execute,
        )
    ]
    client = OpenAICompatClient("https://api.openai.com/v1")
    build = client._build_codex_responses_body if codex else client._build_responses_body
    body = build(request)
    assert body["tools"][0]["strict"] is False
    assert body["tools"][0]["parameters"] == original
    assert parameters == original
    # Chat Completions is non-strict by default and keeps its established shape.
    chat_tool = client._build_body(request)["tools"][0]["function"]
    assert "strict" not in chat_tool
    assert chat_tool["parameters"] == original


@pytest.mark.parametrize(
    ("model_id", "modern"),
    [
        ("gpt-5.6", True),
        ("gpt-5.6-sol", True),
        ("gpt-5.6-2026-09-01", True),
        ("gpt-5.10-sol", True),
        ("gpt-6-astra", True),
        ("gpt-6.1-sol", True),
        ("gpt-6.1-sol-2026-10-01", True),
        ("gpt-5", False),
        ("gpt-5.4", False),
        ("gpt-5.5-sol", False),
        ("gpt-4.1", False),
        ("gpt-7-sol", False),
        ("gpt-60-sol", False),
        ("gpt-5.6other", False),
        ("prefix-gpt-6.1-sol", False),
    ],
)
@pytest.mark.asyncio
async def test_public_wire_uses_documented_modern_cache_defaults(model_id, modern) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            text='data: {"type":"response.completed","response":{"output":[]}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatClient("https://api.openai.com/v1", http_client=http)
        _ = [event async for event in client.stream(_request(model_id), "synthetic-key")]
    body = captured["body"]
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert ("prompt_cache_retention" not in body) is modern
    if not modern:
        assert body["prompt_cache_retention"] == "24h"
    assert body["prompt_cache_key"] == "synthetic-contract"
    assert body["instructions"] == "Stable synthetic instructions."
    assert "prompt_cache_options" not in body


@pytest.mark.parametrize(
    ("base", "provider", "endpoint"),
    [
        ("https://custom.example/v1", "openai", None),
        ("https://api.openai.com/v1", "compatible", None),
        ("https://api.openai.com/v1", "openai", "https://custom.example/responses"),
        ("https://api.openai.com/v1", "openai", "https://chatgpt.com/backend-api/codex/responses"),
    ],
)
def test_custom_and_oauth_endpoint_policy_remains_legacy(base, provider, endpoint) -> None:
    body = OpenAICompatClient(base)._build_responses_body(
        _request(provider=provider), endpoint=endpoint
    )
    assert body["prompt_cache_retention"] == "24h"


@pytest.mark.parametrize("model_id", ["gpt-5.4", "gpt-5.6-sol", "gpt-6.1-sol", "gpt-7-sol"])
def test_codex_retention_removal_is_independent_of_public_family(model_id) -> None:
    body = OpenAICompatClient("https://api.openai.com/v1")._build_codex_responses_body(
        _request(model_id)
    )
    assert "prompt_cache_retention" not in body
    assert "prompt_cache_options" not in body
    assert body["prompt_cache_key"] == "synthetic-contract"
    assert body["store"] is False


def test_no_tools_and_no_cache_key_do_not_gain_new_fields() -> None:
    request = _request()
    request.prompt_cache_key = None
    body = OpenAICompatClient("https://api.openai.com/v1")._build_responses_body(request)
    assert "tools" not in body
    assert "prompt_cache_key" not in body
    assert "prompt_cache_retention" not in body
