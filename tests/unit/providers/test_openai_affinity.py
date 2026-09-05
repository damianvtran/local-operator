"""Codex affinity is transport-only, scoped, and stable across session replay."""

import json
import uuid

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.clients import CODEX_RESPONSES_URL, OpenAICompatClient


def _request(key="lineage", provider="openai", model="gpt-6-astra"):
    return ChatRequest(
        model=ModelSpec(provider=provider, model_id=model, supports_responses_api=True),
        messages=[Message.user("synthetic")],
        system_blocks=["stable"],
        prompt_cache_key=key,
    )


def test_affinity_preserves_retry_resume_model_switch_and_fork_lineage():
    client = OpenAICompatClient("https://api.openai.com/v1")
    auth = OAuthAccess("synthetic-access", 1, org_id="synthetic-account")
    request = _request()
    headers = client._codex_affinity_headers(request, auth, CODEX_RESPONSES_URL)
    assert headers["session-id"] == headers["thread-id"]
    assert uuid.UUID(headers["session-id"]).version == 5
    assert client._codex_affinity_headers(request, auth, CODEX_RESPONSES_URL) == headers
    resumed = OpenAICompatClient("https://api.openai.com/v1")
    assert (
        resumed._codex_affinity_headers(_request(model="gpt-5.6-sol"), auth, CODEX_RESPONSES_URL)
        == headers
    )
    assert (
        resumed._codex_affinity_headers(
            _request(provider="openai-device"), auth, CODEX_RESPONSES_URL
        )
        == headers
    )
    for model in ("gpt-4o", "future-model"):
        assert (
            resumed._codex_affinity_headers(_request(model=model), auth, CODEX_RESPONSES_URL)
            == headers
        )
    # A fork's inherited lineage must route together, but another lineage must
    # not collapse into a single application-wide cache group.
    assert (
        resumed._codex_affinity_headers(_request("other-lineage"), auth, CODEX_RESPONSES_URL)
        != headers
    )


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_lineage_does_not_invent_per_request_or_global_identity(key):
    client = OpenAICompatClient("https://api.openai.com/v1")
    auth = OAuthAccess("synthetic", 1, org_id="account")
    assert client._codex_affinity_headers(_request(key), auth, CODEX_RESPONSES_URL) == {}


@pytest.mark.parametrize(
    "base,provider,url,auth",
    [
        ("https://api.openai.com/v1", "openai", CODEX_RESPONSES_URL, None),
        (
            "https://api.openai.com/v1",
            "openai",
            CODEX_RESPONSES_URL,
            OAuthAccess("synthetic", 1, org_id="account", kind="api_key"),
        ),
        ("https://api.openai.com/v1", "openai", CODEX_RESPONSES_URL, OAuthAccess("synthetic", 1)),
        (
            "https://api.openai.com/v1",
            "openai",
            "https://api.openai.com/v1/responses",
            OAuthAccess("synthetic", 1, org_id="account"),
        ),
        (
            "https://example.invalid/v1",
            "openai",
            CODEX_RESPONSES_URL,
            OAuthAccess("synthetic", 1, org_id="account"),
        ),
        (
            "https://api.openai.com/v1",
            "custom",
            CODEX_RESPONSES_URL,
            OAuthAccess("synthetic", 1, org_id="account"),
        ),
    ],
)
def test_affinity_never_leaks_to_other_auth_routes_or_providers(base, provider, url, auth):
    assert (
        OpenAICompatClient(base)._codex_affinity_headers(_request(provider=provider), auth, url)
        == {}
    )


def test_header_contains_neither_raw_lineage_nor_credentials():
    key = "~/private/session\r\nInjected: value ☃"
    client = OpenAICompatClient("https://api.openai.com/v1")
    auth = OAuthAccess("synthetic-secret", 1, org_id="synthetic-account")
    headers = client._codex_affinity_headers(_request(key), auth, CODEX_RESPONSES_URL)
    assert set(headers) == {"session-id", "thread-id"}
    for value in headers.values():
        assert len(value) == 36
        assert str(uuid.UUID(value)) == value
        assert key not in value
        assert auth.access_token not in value
        assert auth.org_id is not None
        assert auth.org_id not in value


@pytest.mark.asyncio
async def test_actual_dispatch_adds_only_pair_without_changing_body_or_retry_identity():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            content='data: {"type":"response.completed","response":{"output":[],'
            '"usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = OpenAICompatClient("https://api.openai.com/v1", http_client=http)
        request = _request()
        auth = OAuthAccess("synthetic", 1, org_id="account")
        expected_body = client._build_codex_responses_body(request)
        for _ in range(2):
            async for _event in client.stream(request, None, auth):
                pass
    assert len(requests) == 2
    for wire in requests:
        assert str(wire.url) == CODEX_RESPONSES_URL
        assert wire.headers["session-id"] == wire.headers["thread-id"]
        assert json.loads(wire.content) == expected_body
        assert "x-client-request-id" not in wire.headers
        assert "x-codex-routing-hint" not in wire.headers
        assert "x-openai-internal-codex-responses-lite" not in wire.headers
    assert requests[0].headers["session-id"] == requests[1].headers["session-id"]
