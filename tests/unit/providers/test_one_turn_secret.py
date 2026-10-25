"""Provider-bound one-turn secret overlay tests; no network or credentials."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
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
    ProviderTurnStartEvent,
    StreamEndEvent,
    StreamStartEvent,
    StreamTextDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    ToolResult,
    Usage,
    turn_output_budget,
)
from local_operator.providers.clients import (
    AnthropicClient,
    GoogleClient,
    OpenAICompatClient,
)
from local_operator.providers.failover import (
    ProviderError,
    _next_input_refusal_degrade,
    _request_with_degradations,
    stream_with_failover,
)
from local_operator.variables import VariableStore
from tests.unit.providers.test_failover import FakeAuth, _FnClient


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


def _live_request(canary: str, provider: str = "openai") -> ChatRequest:
    request = _request(canary)
    request.model = request.model.model_copy(update={"provider": provider})
    variables = VariableStore()
    token = variables.acquire_scoped_redaction(canary)
    request.arm_one_turn_secrets({"call-secret": SecretStr(canary)}, redact=variables.redact)
    request._test_redaction_token = token  # type: ignore[attr-defined]
    request._test_variables = variables  # type: ignore[attr-defined]
    return request


def _sse_bytes(payloads: list[dict[str, Any]]) -> bytes:
    return (
        "".join(f"data: {json.dumps(payload)}\n\n" for payload in payloads).encode()
        + b"data: [DONE]\n\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["openai-chat", "openai-responses", "anthropic"])
async def test_real_sse_response_id_is_withheld_before_broker(wire: str, tmp_path: Path) -> None:
    canary = _canary("response-" + wire)
    request = _live_request(canary, "anthropic" if wire == "anthropic" else "openai")
    bodies: list[dict[str, Any]] = []
    if wire == "openai-chat":
        payloads = [
            {"id": canary, "provider": canary, "choices": [{"delta": {}, "finish_reason": "stop"}]}
        ]
    elif wire == "openai-responses":
        payloads = [
            {"type": "response.created", "response": {"id": canary}},
            {"type": "response.completed", "response": {"output": []}},
        ]
    else:
        payloads = [
            {"type": "message_start", "message": {"id": canary, "usage": {}}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
        ]

    def handler(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(http_request.content))
        return httpx.Response(
            200, content=_sse_bytes(payloads), headers={"content-type": "text/event-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = (
            AnthropicClient(http_client=http)
            if wire == "anthropic"
            else OpenAICompatClient(
                "https://api.openai.com/v1",
                http_client=http,
                openai_api="responses" if wire == "openai-responses" else "chat",
            )
        )
        if wire == "openai-responses":
            request.model = request.model.model_copy(update={"supports_responses_api": True})

        async def client_for(_spec: ModelSpec) -> Any:
            return client

        events = [
            event
            async for event in stream_with_failover(
                request, FakeAuth({request.model.provider: ["synthetic-key"]}), None, client_for
            )
        ]
    assert json.dumps(bodies).count(canary) == 1
    start = next(event for event in events if isinstance(event, StreamStartEvent))
    assert start.response_id is None
    assert all(
        not isinstance(event, StreamEndEvent) or event.served_provider is None for event in events
    )

    from local_operator.jobs import JobStatus
    from local_operator.server.utils.event_broker import EventBroker, job_channel
    from local_operator.server.utils.sse_bridge import attach_broker
    from local_operator.server.utils.sse_publisher import publish_job_status
    from tests.unit.server.test_sse_bridge import _FakeSession

    broker = EventBroker()
    session = _FakeSession()
    attach_broker(session, broker, "secret-response")
    session.emit(
        ProviderTurnStartEvent(
            response_id=start.response_id,
            provider="test",
            model_id="m",
        )
    )
    publish_job_status(broker, "secret-response", JobStatus.COMPLETED)
    retained = [event.data for event in broker.retained(job_channel("secret-response"))]
    assert canary not in json.dumps(retained)


@pytest.mark.asyncio
async def test_encoded_metadata_and_errors_fail_closed_with_noop_sink() -> None:
    from local_operator.providers.failover import _one_turn_scrub_stream_event

    canary = 'v0rn-"encoded\\path\nmetadata-7d4c'
    request = _live_request(canary)
    request._one_turn_redact = lambda value: value
    encoded = json.dumps(canary)[1:-1]
    start_event = _one_turn_scrub_stream_event(request, StreamStartEvent(response_id=encoded))
    assert isinstance(start_event, StreamStartEvent)
    assert start_event.response_id is None
    end_event = _one_turn_scrub_stream_event(
        request,
        StreamEndEvent(
            stop_reason="error",
            served_provider=encoded,
            provider_payload={"native": canary},
            error="provider quoted " + canary,
            usage=Usage(provider=encoded, model_id=encoded),
        ),
    )
    assert isinstance(end_event, StreamEndEvent)
    assert end_event.served_provider is None
    assert end_event.provider_payload is None
    assert (
        end_event.usage is not None
        and end_event.usage.provider is None
        and end_event.usage.model_id is None
    )
    assert end_event.error is not None
    assert canary not in end_event.error and encoded not in end_event.error


async def _one_turn_metadata_driver(
    request: ChatRequest, stream: AsyncIterator[Any]
) -> AsyncIterator[Any]:
    # Use the production metadata boundary without requiring a credential route
    # in this observation-specific test.
    from local_operator.providers.failover import _one_turn_scrub_stream_event

    async for event in stream:
        yield _one_turn_scrub_stream_event(request, event)


@pytest.mark.asyncio
async def test_credential_rotation_and_flap_logs_and_file_are_canary_free(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = _canary("failover-log")
    log_path = tmp_path / "provider.log"
    handler = logging.FileHandler(log_path)
    logger = logging.getLogger("local_operator.providers.failover")
    logger.addHandler(handler)
    attempts: list[str | None] = []

    def rotate(request: ChatRequest, api_key: str | None, _oauth: Any = None) -> AsyncIterator[Any]:
        attempts.append(api_key)
        if api_key == "first":
            raise ProviderError(401, "provider quoted " + canary, auth_error=True)

        async def quota_failure() -> AsyncIterator[Any]:
            yield StreamUsageEvent(usage=Usage(input_tokens=1, output_tokens=0))
            raise ProviderError(429, "quota also quoted " + canary, retryable=True)

        return quota_failure()

    async def client_for(_spec: ModelSpec) -> Any:
        return _FnClient(rotate)

    request = _live_request(canary)
    try:
        with caplog.at_level("WARNING"):
            with pytest.raises(ProviderError) as failure:
                _ = [
                    event
                    async for event in stream_with_failover(
                        request,
                        FakeAuth({"openai": ["first", "second"]}),
                        {"retry": {"baseDelayMs": 1, "maxRetries": 0}},
                        client_for,
                    )
                ]
            assert failure.value.status == 429
        assert attempts == ["first", "second"]

        from local_operator.providers import failover as failover_module

        monkeypatch.setattr(failover_module, "MODEL_FLAP_RETRY_DELAY_MS", 1)
        monkeypatch.setattr(
            failover_module, "_SERVED_SELECTORS", {f"openai/{request.model.model_id}"}
        )
        flap_calls = 0

        def flap(_request: ChatRequest, _key: str | None, _oauth: Any = None) -> AsyncIterator[Any]:
            nonlocal flap_calls
            flap_calls += 1
            if flap_calls == 1:
                raise ProviderError(
                    400,
                    request.model.model_id + " is not a valid model ID; quoted " + canary,
                    retryable=False,
                )

            async def success() -> AsyncIterator[Any]:
                yield StreamTextDelta(delta="ok")
                yield StreamEndEvent(stop_reason="stop")

            return success()

        async def flap_client_for(_spec: ModelSpec) -> Any:
            return _FnClient(flap)

        with caplog.at_level("WARNING"):
            _ = [
                event
                async for event in stream_with_failover(
                    request,
                    FakeAuth({"openai": ["key"]}),
                    {"retry": {"baseDelayMs": 1}},
                    flap_client_for,
                )
            ]
        assert flap_calls == 2
    finally:
        logger.removeHandler(handler)
        handler.close()
    assert canary not in caplog.text
    assert canary not in log_path.read_text()
    assert "authentication failed (HTTP 401)" in caplog.text
    assert "superseded by a more diagnostic failure" in caplog.text


@pytest.mark.asyncio
async def test_real_openai_equal_metadata_resolution_dispatches() -> None:
    from local_operator.model.configure import build_model_spec, context_spec_for_access
    from local_operator.providers.auth_store import OAuthAccess

    canary = _canary("metadata-resolution")
    request = _request(canary)
    request.model = build_model_spec("openai", "gpt-4o")
    request.arm_one_turn_secrets({"call-secret": SecretStr(canary)}, redact=VariableStore().redact)
    access = OAuthAccess("synthetic-api-key", 1, kind="api_key")
    resolved = context_spec_for_access(request.model, access, {})
    changed = [
        key
        for key in request.model.model_dump()
        if getattr(request.model, key) != getattr(resolved, key)
    ]
    assert changed == ["context_metadata_resolved"]
    assert turn_output_budget(resolved) == turn_output_budget(request.model)
    request.max_tokens = turn_output_budget(request.model)
    request._max_tokens_from_policy = True
    calls = 0

    def stream(_request: ChatRequest, _key: str | None, _oauth: Any = None) -> AsyncIterator[Any]:
        nonlocal calls
        calls += 1

        async def generate() -> AsyncIterator[Any]:
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return generate()

    async def client_for(_spec: ModelSpec) -> Any:
        return _FnClient(stream)

    events = [
        event
        async for event in stream_with_failover(
            request, FakeAuth({"openai": ["synthetic-api-key"]}), None, client_for
        )
    ]
    assert calls == 1
    assert any(isinstance(event, StreamTextDelta) for event in events)


@pytest.mark.asyncio
async def test_session_stream_post_arming_clones_cannot_mutate_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers import failover as failover_module

    canary = _canary("session-clones")
    owner = SessionStreamFn(MagicMock(), {}, "secret-clone-session")
    seen: list[ChatRequest] = []

    async def fake_stream(request: ChatRequest, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        seen.append(request)
        yield StreamEndEvent(stop_reason="stop")

    async def no_preflight(_model: ModelSpec) -> None:
        return None

    monkeypatch.setattr(failover_module, "stream_with_failover", fake_stream)
    monkeypatch.setattr(owner, "preflight_usage", no_preflight)
    monkeypatch.setattr(owner, "_effort_for", lambda _model: "high")
    owner._cache_lineage_id = "cache-after-arming"
    owner._provider_affinity[request_model_id := "m"] = "host-after-arming"
    request = _live_request(canary)
    request.model = request.model.model_copy(
        update={"model_id": request_model_id, "reasoning_efforts": ("low", "high")}
    )
    request.arm_one_turn_secrets(request._one_turn_secrets, redact=request._one_turn_redact)
    with pytest.raises(OneTurnSecretScopeError):
        _ = [event async for event in owner(request, None)]
    assert seen == []


@pytest.mark.asyncio
async def test_served_host_never_reaches_observation_affinity_or_later_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers import failover as failover_module
    from local_operator.session.request_observation import RequestFinalObservation

    canary = _canary("served-host")
    owner = SessionStreamFn(MagicMock(), {}, "secret-host-session")
    seen_observations: list[RequestFinalObservation] = []
    owner.subscribe_requests(seen_observations.append)
    captured: list[ChatRequest] = []

    async def first_stream(request: ChatRequest, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        captured.append(request)
        yield StreamUsageEvent(
            usage=Usage(input_tokens=1, output_tokens=1, provider=canary, model_id=canary)
        )
        yield StreamEndEvent(
            stop_reason="stop",
            served_provider=canary,
            usage=Usage(input_tokens=1, output_tokens=1, provider=canary, model_id=canary),
            provider_payload={"native": canary},
        )

    async def first_stream_with_boundary(
        request: ChatRequest, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        async for event in first_stream(request, *args, **kwargs):
            yield failover_module._one_turn_scrub_stream_event(request, event)

    async def no_preflight(_model: ModelSpec) -> None:
        return None

    monkeypatch.setattr(owner, "preflight_usage", no_preflight)
    monkeypatch.setattr(failover_module, "stream_with_failover", first_stream_with_boundary)
    request = _live_request(canary, "openrouter")
    request.model = request.model.model_copy(update={"supports_prompt_cache": True})
    events = [event async for event in owner(request, None)]
    assert all(
        not isinstance(event, StreamEndEvent)
        or (
            event.served_provider is None
            and event.provider_payload is None
            and event.usage is not None
            and event.usage.provider is None
            and event.usage.model_id is None
        )
        for event in events
    )
    from dataclasses import asdict

    assert canary not in json.dumps(
        [asdict(row) for row in seen_observations],
        default=lambda value: value.model_dump(mode="json"),
    )
    assert canary not in json.dumps(owner._provider_affinity)

    async def second_stream(
        request: ChatRequest, *_args: Any, **_kwargs: Any
    ) -> AsyncIterator[Any]:
        captured.append(request)
        yield StreamEndEvent(stop_reason="stop")

    monkeypatch.setattr(failover_module, "stream_with_failover", second_stream)
    later = _request(canary)
    later.model = request.model
    _ = [event async for event in owner(later, None)]
    assert len(captured) == 2
    later_request = captured[-1]
    assert not later_request.has_one_turn_secrets
    assert canary not in json.dumps(later_request.model_dump(mode="json"))
    bodies = [build(later_request) for build in _BUILDERS.values()]
    assert all(canary not in json.dumps(body) for body in bodies)
