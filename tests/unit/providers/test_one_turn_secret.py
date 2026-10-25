"""Provider-bound one-turn secret overlay tests; no network or credentials."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import SecretStr

from local_operator.harness.types import (
    ONE_TURN_SECRET_RECEIPT,
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    OneTurnSecretScopeError,
    OneTurnSecretSerializationError,
    TextContent,
    ToolCall,
    ToolResult,
    turn_output_budget,
)
from local_operator.providers.clients import (
    AnthropicClient,
    GoogleClient,
    OpenAICompatClient,
)
from local_operator.providers.failover import (
    _next_input_refusal_degrade,
    _request_with_degradations,
)


def _canary(label: str = "wire") -> str:
    return "v0rn-" + label + "-91b6"


def _request(canary: str) -> ChatRequest:
    result = ToolResult(
        tool_call_id="call-secret",
        tool_name="safe_tool",
        content=[TextContent(text=ONE_TURN_SECRET_RECEIPT)],
    )
    tool_row = Message.tool_result(result)
    tool_row.id = "tool-secret"
    assistant = Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id="call-secret",
                name="safe_tool",
                arguments={},
                raw_arguments="{}",
            )
        ],
    )
    assistant.id = "assistant-secret"
    user = Message.user("user-safe")
    user.id = "user-safe"

    async def execute(*_args: Any) -> ToolResult:
        raise AssertionError("wire tests never execute tools")

    return ChatRequest(
        model=ModelSpec(provider="test", model_id="m", max_output_tokens=256),
        max_tokens=128,
        system_blocks=["system-safe"],
        messages=[user, assistant, tool_row],
        tools=[
            AgentTool(
                name="safe_tool",
                parameters={"type": "object", "properties": {}},
                execute=execute,
            )
        ],
        prompt_cache_key="cache-safe",
    )


def _diff(
    public: Any,
    armed: Any,
    path: str = "$",
) -> list[tuple[str, Any, Any]]:
    if type(public) is not type(armed):
        return [(path, public, armed)]
    if isinstance(public, dict):
        if public.keys() != armed.keys():
            return [(path, public, armed)]
        return [
            difference
            for key in public
            for difference in _diff(public[key], armed[key], path + "." + key)
        ]
    if isinstance(public, list):
        if len(public) != len(armed):
            return [(path, public, armed)]
        return [
            difference
            for index, (left, right) in enumerate(zip(public, armed, strict=True))
            for difference in _diff(left, right, f"{path}[{index}]")
        ]
    return [] if public == armed else [(path, public, armed)]


_BUILDERS: dict[str, Callable[[ChatRequest], dict[str, Any]]] = {
    "openai-chat": lambda request: OpenAICompatClient("https://x")._build_body(request),
    "openai-responses": lambda request: OpenAICompatClient("https://x")._build_responses_body(
        request
    ),
    "anthropic": lambda request: AnthropicClient()._build_body(request),
    "google": lambda request: GoogleClient()._build_body(request),
}


@pytest.mark.parametrize("wire", sorted(_BUILDERS))
def test_each_provider_substitutes_exactly_one_tool_result_leaf(wire: str) -> None:
    canary = _canary(wire)
    request = _request(canary)
    build = _BUILDERS[wire]
    public_body = build(request)
    assert canary not in json.dumps(public_body)
    assert ONE_TURN_SECRET_RECEIPT in json.dumps(public_body)

    request.arm_one_turn_secrets({"call-secret": SecretStr(canary)})
    armed_body = build(request)
    encoded = json.dumps(armed_body)
    assert encoded.count(canary) == 1
    assert request.messages[-1].text == ONE_TURN_SECRET_RECEIPT
    differences = _diff(public_body, armed_body)
    assert len(differences) == 1
    _path, public_value, armed_value = differences[0]
    assert ONE_TURN_SECRET_RECEIPT in str(public_value)
    assert canary in str(armed_value)

    # Same-body transport/credential retries reuse the same request overlay.
    assert json.dumps(build(request), sort_keys=True) == json.dumps(armed_body, sort_keys=True)
    with pytest.raises(OneTurnSecretSerializationError):
        request.model_dump(mode="json")


def test_unarmed_wire_view_is_identical_and_has_no_private_keys() -> None:
    request = _request(_canary("ordinary"))
    assert request.wire_messages() is request.messages
    assert all(
        message is original
        for message, original in zip(request.wire_messages(), request.messages, strict=True)
    )
    dumped = request.model_dump(mode="json")
    assert "_one_turn_secrets" not in json.dumps(dumped)
    assert "_one_turn_secret" not in json.dumps(dumped)
    for build in _BUILDERS.values():
        assert canary_not_present(build, request)


def canary_not_present(
    build: Callable[[ChatRequest], dict[str, Any]], request: ChatRequest
) -> bool:
    body = json.dumps(build(request))
    return "v0rn-" not in body and "_one_turn" not in body


def _armed_request(canary: str) -> ChatRequest:
    request = _request(canary)
    request.arm_one_turn_secrets({"call-secret": SecretStr(canary)})
    return request


def test_with_model_allows_only_equal_model_and_effective_budget() -> None:
    request = _armed_request(_canary("with-model"))
    equal = request.model.model_copy()
    assert request.with_model(equal).has_one_turn_secrets

    for changed in (
        request.model.model_copy(update={"model_id": "other"}),
        request.model.model_copy(update={"provider": "other"}),
        request.model.model_copy(update={"fast_mode": True}),
    ):
        with pytest.raises(OneTurnSecretScopeError):
            request.with_model(changed)

    request._max_tokens_from_policy = True
    request.max_tokens = turn_output_budget(request.model)
    smaller = request.model.model_copy(update={"max_output_tokens": 1})
    with pytest.raises(OneTurnSecretScopeError):
        request.with_model(smaller)


def test_every_input_refusal_degradation_fails_before_a_client() -> None:
    request = _armed_request(_canary("degradation"))
    with pytest.raises(OneTurnSecretScopeError):
        _request_with_degradations(request, ["screenshots_removed"])
    with pytest.raises(OneTurnSecretScopeError):
        _request_with_degradations(request, ["history_dropped"])
    with pytest.raises(OneTurnSecretScopeError):
        _next_input_refusal_degrade(request, [])
    assert _request_with_degradations(request, []) is request


def test_armed_overlay_requires_exactly_one_matching_tool_row() -> None:
    request = _armed_request(_canary("missing"))
    request.messages[-1].tool_call_id = "other"
    with pytest.raises(OneTurnSecretScopeError):
        request.wire_messages()

    request = _armed_request(_canary("duplicate"))
    request.messages.append(request.messages[-1])
    with pytest.raises(OneTurnSecretScopeError):
        request.wire_messages()
