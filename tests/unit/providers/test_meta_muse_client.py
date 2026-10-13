"""Exercise the shipped Muse factory, Responses dialect, tools and failure gate."""

from __future__ import annotations

import json
from typing import Literal

import httpx
import pytest

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    StreamEndEvent,
    StreamToolCallDelta,
    TextContent,
)
from local_operator.model.configure import build_model_spec
from local_operator.providers.clients import client_for_spec
from local_operator.providers.failover import ProviderError
from local_operator.providers.meta_muse_client import MetaMuseClient

KEY = "LLM|1234|fixture_subscription_key"


async def execute(*_args, **_kwargs):
    raise AssertionError("Wire fixture never executes tools")


def request(*, tools=(), choice: Literal["auto", "none", "required"] = "auto"):
    return ChatRequest(
        model=build_model_spec("meta-muse", "muse-wonder-1.0"),
        messages=[Message(role="user", content=[TextContent(text="Reply OK.")])],
        tools=[AgentTool(name=name, execute=execute) for name in tools],
        tool_choice=choice,
        max_tokens=16,
    )


def test_factory_and_responses_body_are_not_codex():
    req = request(tools=("a-b", "a_b"))
    client = client_for_spec(req.model)
    assert isinstance(client, MetaMuseClient)
    body = client._build_responses_body(req)
    assert body["max_output_tokens"] == 16
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert "tool_choice" not in body and "prompt_cache_retention" not in body
    names = [tool["name"] for tool in body["tools"]]
    assert len(set(names)) == 2
    assert "a_b" in names
    assert all("-" not in name for name in names)
    assert set(client._headers(KEY)) == {"Content-Type", "Accept", "x-api-version", "Authorization"}
    assert not client._codex_responses_mode(None)


def test_none_removes_tools_and_required_is_explicit_refusal():
    client = MetaMuseClient("https://api.meta.ai/v1", openai_api="responses")
    assert "tools" not in client._build_responses_body(request(tools=("tool",), choice="none"))
    with pytest.raises(ProviderError, match="automatic"):
        client._build_responses_body(request(tools=("tool",), choice="required"))


@pytest.mark.asyncio
async def test_actual_factory_posts_responses_and_restores_tool_names():
    # Reserve both the first hash candidate and its numeric retry: neither
    # requires a hash collision, just an ordinary registered function name.
    req = request(tools=("a-b", "a_b", "a_b_d44362d67d92", "a_b_d44362d67d92_1"))
    from local_operator.providers.meta_muse_client import tool_aliases

    aliases = tool_aliases(req)
    assert len(set(aliases.values())) == len(aliases)
    wire_name = aliases["a-b"]
    frames = [
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": wire_name, "call_id": "call-1"},
        },
        {"type": "response.function_call_arguments.delta", "call_id": "call-1", "delta": "{}"},
        {
            "type": "response.completed",
            "response": {
                "id": "fixture-response",
                "output": [
                    {
                        "type": "function_call",
                        "name": wire_name,
                        "call_id": "call-1",
                        "arguments": "{}",
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 1},
            },
        },
        {
            "type": "response.subscription_usage",
            "subscription": {
                "tier": "opaque-tier-is-not-a-plan-label",
                "window": {"used_percent": 7, "resets_at": 1788431188, "window_duration_mins": 300},
                "weekly": {"used_percent": 2, "resets_at": 1788739200},
                "api_key": KEY,
            },
        },
    ]

    def handle(wire):
        assert str(wire.url) == "https://api.meta.ai/v1/responses"
        assert wire.headers["authorization"] == f"Bearer {KEY}"
        assert wire.headers["x-api-version"] == "1.0.0"
        assert "openai-beta" not in wire.headers and "chatgpt-account-id" not in wire.headers
        assert json.loads(wire.content)["tools"][0]["name"] == wire_name
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-request-id": "fixture-request"},
            content="".join("data: " + json.dumps(frame) + "\n\n" for frame in frames),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = client_for_spec(req.model, http_client=http)
        events = [event async for event in client.stream(req, KEY)]
    names = [
        event.name for event in events if isinstance(event, StreamToolCallDelta) and event.name
    ]
    assert names == ["a-b"]
    end = next(event for event in events if isinstance(event, StreamEndEvent))
    assert end.provider_payload is not None
    assert wire_name not in json.dumps(end.provider_payload)
    assert end.provider_payload["meta_muse"]["subscription_usage"]["window"]["usedPercent"] == 7
    assert end.provider_payload["meta_muse"]["request_id"] == "fixture-request"
    assert (
        end.provider_payload["meta_muse"]["subscription_usage"]["window"]["resetsAtMs"]
        == 1788431188000
    )
    assert KEY not in json.dumps(end.provider_payload)
    assert "opaque-tier" not in json.dumps(end.provider_payload)


@pytest.mark.asyncio
async def test_reflected_key_error_is_redacted_and_cascade_eligible():
    """A dead sign-in is a normal auth failure, sanitised and cascade-eligible.

    The redaction is the point of this seam: an error body can echo the
    submitted key, and the provider's text must never ride out. Classification
    stays ``auth`` (never quota), and it no longer fails closed -- a dead key is
    exactly what the user's CONFIGURED fallback chains answer, while the
    billing-attribution and missing-meter refusals keep their own fail-closed
    flags untouched.
    """
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(401, json={"error": {"message": KEY}})
        )
    ) as http:
        client = client_for_spec(request().model, http_client=http)
        with pytest.raises(ProviderError) as caught:
            _ = [event async for event in client.stream(request(), KEY)]
    assert KEY not in str(caught.value)
    assert caught.value.kind == "auth"
    assert not caught.value.fail_closed
    assert caught.value.status == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("meter", [None, {"window": {"used_percent": "0"}}, {"api_key": KEY}])
async def test_missing_unknown_meter_never_borrows_old_receipt_or_succeeds(meter):
    frames = [
        {
            "type": "response.completed",
            "response": {"output": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
        }
    ]
    if meter is not None:
        frames.append({"type": "response.subscription_usage", "subscription": meter})
    with_body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=with_body))
    ) as http:
        client = client_for_spec(request().model, http_client=http)
        assert isinstance(client, MetaMuseClient)
        client.last_subscription_observation = {"window": {"usedPercent": 99}}
        with pytest.raises(
            ProviderError, match="subscription billing could not be verified"
        ) as caught:
            _ = [event async for event in client.stream(request(), KEY)]
    assert caught.value.fail_closed and not caught.value.retryable


def test_allowance_parser_retains_overage_not_a_fake_percent_clamp():
    from local_operator.providers.meta_muse_usage import subscription_observation

    result = subscription_observation(
        {
            "window": {"used_percent": 135.5, "resets_at": 1788431188},
            "weekly": {"used_percent": float("nan")},
            "token": KEY,
        }
    )
    assert result is not None
    assert result["window"]["usedPercent"] == 135.5
    assert result["window"]["resetsAtMs"] == 1788431188000
    assert "weekly" not in result and KEY not in json.dumps(result)
