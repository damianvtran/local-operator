"""A shared cascade can include the selected model without inventing a hop."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.model.configure import build_model_spec, create_stream_fn
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import AnthropicClient, OpenAICompatClient
from local_operator.providers.failover import (
    FailoverRouteState,
    FallbackTarget,
    ProviderError,
    stream_with_failover,
)
from tests.unit.providers.test_failover import FakeAuth, ScriptedClient


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "high", "max"])
async def test_preflight_skips_explicit_primary_effort(tmp_path, effort) -> None:
    """Quota reserve must buy another route, not re-pin the route it just left."""
    store = AuthStore(tmp_path / "auth.db")
    selector = "anthropic/claude-opus-5"
    stream = create_stream_fn(
        store,
        {
            "retry": {
                "fallbackChains": {
                    "default": [
                        selector,
                        {"provider": "anthropic", "model": "claude-opus-5"},
                        {"provider": "anthropic", "model": "*", "effort": effort},
                        {"provider": "anthropic", "model": "claude-opus-5", "effort": effort},
                        {"provider": "anthropic", "model": "claude-opus-5", "effort": "medium"},
                        "openai/gpt-6-astra",
                    ]
                }
            }
        },
        session_id="synthetic-primary-cascade",
    )
    model = ModelSpec(provider="anthropic", model_id="claude-opus-5", reasoning_effort=effort)
    try:
        with (
            patch.object(stream, "_target_has_auth", AsyncMock(return_value=True)),
            patch.object(stream, "_provider_quota_availability", AsyncMock(return_value="usable")),
        ):
            target = await stream._first_available_fallback(model, reserve_percent=10)
        assert target == FallbackTarget(selector, "medium")
        assert stream._fallback_targets(model) == [
            FallbackTarget(selector, "medium"),
            FallbackTarget("openai/gpt-6-astra"),
        ]
    finally:
        await stream.close()
        store.close()


@pytest.mark.asyncio
async def test_different_effort_remains_a_pinned_stream_route() -> None:
    seen: list[str | None] = []

    async def client_for(spec):
        seen.append(spec.reasoning_effort)
        if spec.reasoning_effort == "high":
            return ScriptedClient(ProviderError(None, "route unavailable"))
        return ScriptedClient(
            [StreamTextDelta(delta="lower effort"), StreamEndEvent(stop_reason="stop")]
        )

    model = build_model_spec("anthropic", "claude-opus-5").model_copy(
        update={"reasoning_effort": "high"}
    )
    settings = {
        "retry": {
            "fallbackChains": {
                "default": [
                    "anthropic/claude-opus-5",
                    {"provider": "anthropic", "model": "claude-opus-5", "effort": "high"},
                    {"provider": "anthropic", "model": "claude-opus-5", "effort": "low"},
                    {"provider": "anthropic", "model": "claude-opus-5", "effort": "low"},
                ]
            }
        }
    }
    route = FailoverRouteState()
    for _ in range(2):
        events = [
            event
            async for event in stream_with_failover(
                ChatRequest(model=model),
                FakeAuth({"anthropic": ["synthetic"]}),
                settings,
                client_for,
                route_state=route,
            )
        ]
        assert any(
            isinstance(event, StreamTextDelta) and event.delta == "lower effort" for event in events
        )
    assert seen == ["high", "low", "low"]
    assert route.active == FallbackTarget("anthropic/claude-opus-5", "low")


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ["openai/gpt-6-astra", "anthropic/claude-opus-5"])
async def test_shared_cascade_over_real_http(primary) -> None:
    """Exercise both wire adapters and sibling rotation without paid provider calls.

    The server refuses every account on the primary. Re-listing that primary in
    either config syntax must not spend another request or obscure the next hop.
    Only synthetic credentials cross this loopback socket.
    """
    requests: list[tuple[str, str]] = []
    primary_provider, primary_model = primary.split("/", 1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            pass

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = payload["model"]
            account = self.headers.get("x-api-key") or self.headers.get("Authorization", "")
            requests.append((model, account))
            if model == primary_model:
                status = 403
                body = json.dumps({"error": {"message": "account denied"}}).encode()
            else:
                status = 200
                if model == "claude-opus-5":
                    events = [
                        ("message_start", {"message": {"usage": {"input_tokens": 1}}}),
                        (
                            "content_block_delta",
                            {"index": 0, "delta": {"type": "text_delta", "text": "served"}},
                        ),
                        (
                            "message_delta",
                            {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                        ),
                        ("message_stop", {}),
                    ]
                    body = "".join(
                        f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"
                        for kind, data in events
                    ).encode()
                else:
                    body = (
                        b'data: {"choices":[{"delta":{"content":"served"},'
                        b'"finish_reason":null}]}\n\ndata: [DONE]\n\n'
                    )
            self.send_response(status)
            self.send_header(
                "Content-Type", "text/event-stream" if status == 200 else "application/json"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    chain = [
        {"provider": "anthropic", "model": "claude-opus-5"},
        "anthropic/claude-opus-5",
        {"provider": primary_provider, "model": primary_model, "effort": "high"},
        {"provider": "zai", "model": "glm-5.3"},
        primary,
    ]
    keys = {
        "openai": ["synthetic-openai"],
        "anthropic": ["synthetic-anthropic"],
        "zai": ["synthetic-zai"],
    }
    keys[primary_provider] = ["synthetic-first", "synthetic-second"]
    auth = FakeAuth(keys)
    try:
        async with httpx.AsyncClient() as http:

            async def client_for(spec):
                if spec.provider == "anthropic":
                    return AnthropicClient(url, http_client=http)
                return OpenAICompatClient(url, http_client=http)

            model = build_model_spec(primary_provider, primary_model).model_copy(
                update={"reasoning_effort": "high"}
            )
            events = [
                event
                async for event in stream_with_failover(
                    ChatRequest(model=model),
                    auth,
                    {"retry": {"maxRetries": 0, "fallbackChains": {"default": chain}}},
                    client_for,
                )
            ]
        assert any(
            isinstance(event, StreamTextDelta) and event.delta == "served" for event in events
        )
        next_model = "claude-opus-5" if primary_provider == "openai" else "glm-5.3"
        assert [model for model, _ in requests] == [primary_model, primary_model, next_model]
        assert requests[0][1] != requests[1][1]
        assert auth.rotations == [
            (primary_provider, "synthetic-first"),
            (primary_provider, "synthetic-second"),
        ]
        print(
            f"{primary}: wire models={[model for model, _ in requests]}; "
            "accounts rotated=2; output=served"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
