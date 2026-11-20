"""T5/T6: discovery parsing + static fallback, per-Region cache key, key_check."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from local_operator.providers.bedrock import discovery as bdisc

KEY = "bedrock://env-bearer?region=eu-west-1"

PROFILES = {
    "inferenceProfileSummaries": [
        {
            "inferenceProfileId": "eu.anthropic.claude-sonnet-5-5",
            "inferenceProfileName": "EU Claude Sonnet 5.5",
            "status": "ACTIVE",
            "type": "SYSTEM_DEFINED",
        },
        {
            "inferenceProfileId": "global.anthropic.claude-opus-5-5",
            "inferenceProfileName": "Global Claude Opus 5.5",
            "status": "ACTIVE",
        },
        {"inferenceProfileId": "eu.anthropic.claude-old", "status": "LEGACY"},
        {"inferenceProfileId": "eu.amazon.nova-pro-v1:0", "status": "ACTIVE"},
    ]
}
FOUNDATION = {
    "modelSummaries": [
        {
            "modelId": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "modelName": "Claude Haiku 4.5",
            "responseStreamingSupported": True,
            "modelLifecycle": {"status": "ACTIVE"},
        },
        {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "modelLifecycle": {"status": "LEGACY"},
        },
        {"modelId": "meta.llama3-70b", "modelLifecycle": {"status": "ACTIVE"}},
    ]
}


@pytest.fixture(autouse=True)
def bearer_env(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "discovery-token")
    monkeypatch.delenv("LOP_BEDROCK_ENDPOINT_URL", raising=False)


class _Ctx:
    def __init__(self, handler, api_key=KEY):
        self.provider_id = "bedrock"
        self.base_url = "bedrock"
        self.api_key = api_key
        self.is_oauth = False
        self.account_id = None
        self.client = httpx.Client(transport=httpx.MockTransport(handler))
        self.timeout = 5.0


def test_discovery_keeps_active_claude_profiles_and_on_demand_models() -> None:
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/inference-profiles":
            return httpx.Response(200, json=PROFILES)
        return httpx.Response(200, json=FOUNDATION)

    rows = bdisc.fetch(_Ctx(handler))
    assert rows is not None
    assert [(r.id, r.name) for r in rows] == [
        ("eu.anthropic.claude-sonnet-5-5", "EU Claude Sonnet 5.5"),
        ("global.anthropic.claude-opus-5-5", "Global Claude Opus 5.5"),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku 4.5"),
    ]
    assert {r.url.host for r in seen} == {"bedrock.eu-west-1.amazonaws.com"}
    assert seen[0].url.params["typeEquals"] == "SYSTEM_DEFINED"
    assert seen[1].url.params["byProvider"] == "anthropic"
    assert seen[1].url.params["byInferenceType"] == "ON_DEMAND"


@pytest.mark.parametrize("status", [403, 500])
def test_listing_failure_keeps_the_static_list(status) -> None:
    rows = bdisc.fetch(_Ctx(lambda request: httpx.Response(status, json={"message": "no"})))
    assert rows is None


def test_no_credentials_means_no_listing(monkeypatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    assert bdisc.fetch(_Ctx(lambda r: httpx.Response(200, json=PROFILES))) is None


def test_cache_key_is_per_region_and_connection_and_hides_secrets() -> None:
    a = bdisc.listing_cache_key("bedrock://env-bearer?region=eu-west-1")
    b = bdisc.listing_cache_key("bedrock://env-bearer?region=us-east-1")
    c = bdisc.listing_cache_key("bedrock://profile?name=dev&region=eu-west-1")
    raw = bdisc.listing_cache_key("raw-bedrock-key-value")
    assert len({a, b, c, raw}) == 4
    assert "raw-bedrock-key-value" not in raw and "eu-west-1" not in a


def test_discovery_wire_has_a_transport_without_base_url() -> None:
    from local_operator.model import discovery

    assert "bedrock" not in discovery.NO_LISTING_PROVIDERS
    assert discovery._TRANSPORTS["bedrock"] is not None


# -- key_check (T6) -------------------------------------------------------------


def _check(handler, key=KEY):
    from local_operator.providers.key_check import check_api_key

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return asyncio.run(check_api_key("bedrock", key, timeout=5.0, client=client))


def test_check_200_is_valid() -> None:
    result = _check(lambda r: httpx.Response(200, json={"inferenceProfileSummaries": []}))
    assert result.valid is True and result.reason is None


def test_check_unrecognized_client_is_rejected() -> None:
    result = _check(
        lambda r: httpx.Response(
            403, headers={"x-amzn-ErrorType": "UnrecognizedClientException:"}, json={}
        )
    )
    assert result.valid is False and "eu-west-1" in (result.reason or "")


def test_check_access_denied_is_unverified_not_rejected() -> None:
    result = _check(
        lambda r: httpx.Response(
            403, headers={"x-amzn-ErrorType": "AccessDeniedException:"}, json={}
        )
    )
    assert result.valid is None and result.reason


def test_check_network_error_is_unverified() -> None:
    def boom(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    result = _check(boom)
    assert result.valid is None


def test_check_unresolvable_credentials_is_rejected(monkeypatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    result = _check(lambda r: httpx.Response(200, json={}))
    assert result.valid is False


def test_two_sealed_keys_in_one_region_never_share_a_listing() -> None:
    """R2: the scope carries a short ref fingerprint, so accounts never cross."""
    ref_a = "BEDROCK_SECRET_" + "a" * 32
    ref_b = "BEDROCK_SECRET_" + "b" * 32
    for method in ("api-key", "access-key"):
        a = bdisc.listing_cache_key(f"bedrock://{method}?region=us-east-1&ref={ref_a}")
        b = bdisc.listing_cache_key(f"bedrock://{method}?region=us-east-1&ref={ref_b}")
        assert a != b
        # Stable for one connection, and the scope never exposes the ref itself.
        assert a == bdisc.listing_cache_key(f"bedrock://{method}?region=us-east-1&ref={ref_a}")
        scope = bdisc.catalogue_scope(f"bedrock://{method}?region=us-east-1&ref={ref_a}")
        assert scope is not None and ref_a not in scope and "a" * 32 not in scope
