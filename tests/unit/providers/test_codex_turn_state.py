"""Server affinity belongs to a logical turn, never a pooled HTTP adapter."""

import json

import httpx

from local_operator.harness.types import ChatRequest, ModelSpec, StreamEndEvent
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.turn_state import CodexTurnState


async def test_server_turn_state_survives_fresh_adapter_within_turn() -> None:
    captured: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        payload = {"type": "response.completed", "response": {"output": []}}
        return httpx.Response(
            200,
            content=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream", "x-codex-turn-state": "synthetic-route"},
        )

    request = ChatRequest(
        model=ModelSpec(provider="openai", model_id="synthetic", supports_responses_api=True)
    )
    request._codex_turn_state = CodexTurnState()
    access = OAuthAccess(
        "synthetic", credential_id=1, account_id="synthetic-account",
        org_id="synthetic-org", kind="oauth"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        for _ in range(2):
            adapter = OpenAICompatClient("https://api.openai.com/v1", http_client=http)
            events = [event async for event in adapter.stream(request, "synthetic", access)]
            assert any(isinstance(event, StreamEndEvent) for event in events)
    assert "x-codex-turn-state" not in captured[0].headers
    assert captured[1].headers.get("x-codex-turn-state") == "synthetic-route"


async def test_actual_adapter_owner_and_public_endpoint_isolation() -> None:
    captured: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        payload = {"type": "response.completed", "response": {"output": []}}
        return httpx.Response(
            200, content=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode(),
            headers={"x-codex-turn-state": f"server-{len(captured)}"},
        )

    holder = CodexTurnState()
    a = OAuthAccess("synthetic-a", credential_id=1, account_id="a", org_id="a", kind="oauth")
    b = OAuthAccess("synthetic-b", credential_id=2, account_id="b", org_id="b", kind="oauth")
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        for owner, attached, base, expected in [
            (a, holder, "https://api.openai.com/v1", None),
            (a, holder, "https://api.openai.com/v1", "server-1"),
            (a, holder, "https://api.openai.com/v1", "server-1"),
            (b, holder, "https://api.openai.com/v1", None),
            (a, holder, "https://api.openai.com/v1", None),
            (a, None, "https://api.openai.com/v1", None),
            (None, holder, "https://api.openai.com/v1", None),
            (a, holder, "https://api.openai.com/v1", None),
            (a, holder, "https://gateway.example/v1", None),
            (a, holder, "https://api.openai.com/v1", None),
            (a, CodexTurnState(), "https://api.openai.com/v1", None),
        ]:
            request = ChatRequest(model=ModelSpec(
                provider="openai", model_id="synthetic", supports_responses_api=True
            ))
            request._codex_turn_state = attached
            adapter = OpenAICompatClient(base, http_client=http)
            events = [event async for event in adapter.stream(request, "synthetic", owner)]
            assert any(isinstance(event, StreamEndEvent) for event in events)
            assert captured[-1].headers.get("x-codex-turn-state") == expected


async def test_real_failover_attempt_clears_token_for_ineligible_client() -> None:
    from local_operator.providers.failover import stream_with_failover

    # Keep this direct-process probe independent of pytest's module imports and
    # real AuthStore construction: no credentials, transports or keychain access.
    class FakeAuth:
        def __init__(self, keys):
            self.keys = keys

        async def get_api_key(self, provider, session_id=None, **kwargs):
            return self.keys[provider][0]

    class ScriptedClient:
        def __init__(self, events):
            self.events = events

        async def stream(self, request, api_key, oauth_access=None):
            for event in self.events:
                yield event

    class EligibleClient(ScriptedClient):
        def codex_turn_state_scope(self, request, api_key, oauth_access):
            assert api_key == "synthetic-key"
            return "openai", "synthetic-owner:model"

    holder = CodexTurnState()
    holder.observe_attempt("openai", "synthetic-owner:model")
    generation, _ = holder.bind_endpoint("oauth")
    holder.receive(generation, "must-not-resurrect")
    request = ChatRequest(model=ModelSpec(provider="openai", model_id="synthetic"))
    request._codex_turn_state = holder
    auth = FakeAuth({"openai": ["synthetic-key"]})
    eligible = EligibleClient([StreamEndEvent(stop_reason="stop")])
    ineligible = ScriptedClient([StreamEndEvent(stop_reason="stop")])
    for client, expected in [
        (eligible, "must-not-resurrect"), (ineligible, None), (eligible, None)
    ]:
        async def client_for(spec):
            return client

        events = [event async for event in stream_with_failover(request, auth, None, client_for)]
        assert any(isinstance(event, StreamEndEvent) for event in events)
        assert holder.bind_endpoint("oauth")[1] == expected
