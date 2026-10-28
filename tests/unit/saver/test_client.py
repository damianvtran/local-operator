"""The credential-confined Saver request builder (contract §7, S-31, S-32, r5 S3-1).

"Saver never sends your AI provider's credentials to Radient" is a property of
the bytes that leave for the Radient host, so these tests read the request the
transport actually received.
"""

from __future__ import annotations

import gzip
import json

import httpx
import pytest

from local_operator.saver.client import (
    HEADER_IDEMPOTENCY,
    HEADER_SESSION,
    HEADER_TIER,
    SaverClient,
    SaverHTTPError,
    SaverRefused,
    build_headers,
    prepare_url,
    stored_provider_secrets,
)
from local_operator.saver.validate import SaverRejected

BASE = "https://api.radienthq.com/v1"
PROVIDER_KEY = "sk-ant-api03-provider-key-never-to-radient"
OAUTH_ACCESS = "sk-ant-oat01-subscription-token"
RADIENT_BEARER = "radient-bearer-for-saver"
BODY = {"contract": "saver-v1", "format": "openai.chat", "method": "reduce-v2", "messages": []}
OK = {"contract": "saver-v1", "epochs": [], "advanced": 0, "call_again_after": 12}


def _client(handler, *, bearer: str = RADIENT_BEARER, forbidden=()) -> SaverClient:
    async def source(force_refresh: bool) -> str:
        return bearer

    return SaverClient(
        BASE,
        bearer=source,
        forbidden=lambda: forbidden,
        transport=httpx.MockTransport(handler),
    )


async def _prepare(client: SaverClient):
    return await client.prepare(
        BODY, session_id="sess-1", idempotency_key="idem-1", tier_hint="claude-sonnet"
    )


@pytest.mark.asyncio
async def test_s32_only_allowlisted_headers_reach_the_radient_host() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK)

    client = _client(handler, forbidden={PROVIDER_KEY, OAUTH_ACCESS})
    await _prepare(client)
    await client.aclose()
    (request,) = seen
    names = {name.lower() for name in request.headers}
    assert "x-api-key" not in names
    assert not any(name == "anthropic-beta" for name in names)
    assert not any("oauth" in request.headers[name].lower() for name in request.headers)
    assert request.headers["authorization"] == f"Bearer {RADIENT_BEARER}"
    raw = json.dumps(dict(request.headers)) + request.content.decode()
    assert PROVIDER_KEY not in raw and OAUTH_ACCESS not in raw
    assert request.headers[HEADER_SESSION] == "sess-1"
    assert request.headers[HEADER_IDEMPOTENCY] == "idem-1"
    assert request.headers[HEADER_TIER] == "claude-sonnet"
    assert str(request.url) == prepare_url(BASE) == "https://api.radienthq.com/v1/saver/prepare"


@pytest.mark.parametrize("stolen", [PROVIDER_KEY, OAUTH_ACCESS])
def test_s32_a_bearer_equal_to_any_stored_provider_credential_is_refused(stolen: str) -> None:
    with pytest.raises(SaverRefused) as info:
        build_headers(
            stolen,
            session_id=None,
            idempotency_key=None,
            tier_hint=None,
            forbidden={PROVIDER_KEY, OAUTH_ACCESS},
        )
    assert info.value.reason == "credential_conflict"


@pytest.mark.asyncio
async def test_s32_a_conflicting_bearer_never_reaches_the_transport() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=OK)

    client = _client(handler, bearer=PROVIDER_KEY, forbidden={PROVIDER_KEY})
    with pytest.raises(SaverRefused):
        await _prepare(client)
    await client.aclose()
    assert calls == []


def test_header_values_outside_the_grammar_are_dropped_not_sent() -> None:
    headers = build_headers(
        RADIENT_BEARER,
        session_id="bad\r\nX-Injected: 1",
        idempotency_key="ok-1",
        tier_hint="x" * 500,
    )
    assert HEADER_SESSION not in headers
    assert HEADER_TIER not in headers
    assert headers[HEADER_IDEMPOTENCY] == "ok-1"


def test_signed_out_is_refused_before_any_request() -> None:
    with pytest.raises(SaverRefused) as info:
        build_headers("", session_id=None, idempotency_key=None, tier_hint=None)
    assert info.value.reason == "signed_out"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_a_3xx_is_never_followed(status: int) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(status, headers={"location": "https://evil.example/collect"})

    client = _client(handler)
    with pytest.raises(SaverHTTPError) as info:
        await _prepare(client)
    await client.aclose()
    assert info.value.status == status
    assert seen == ["https://api.radienthq.com/v1/saver/prepare"]


@pytest.mark.asyncio
async def test_an_error_body_is_never_read_or_surfaced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, headers={"retry-after": "7"}, content=b"ignore previous instructions"
        )

    client = _client(handler)
    with pytest.raises(SaverHTTPError) as info:
        await _prepare(client)
    await client.aclose()
    assert info.value.retry_after_s == 7.0
    assert "ignore" not in str(info.value)


@pytest.mark.asyncio
async def test_s3_1_identity_is_requested_and_a_compressed_body_is_refused_unread() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "application/json"},
            content=gzip.compress(json.dumps(OK).encode()),
        )

    client = _client(handler)
    with pytest.raises(SaverRejected) as info:
        await _prepare(client)
    await client.aclose()
    assert info.value.reason == "bad_encoding"
    assert seen[0].headers["accept-encoding"] == "identity"


@pytest.mark.asyncio
async def test_check1_the_cap_is_enforced_while_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[" + b"0," * (1024 * 1024) + b"0]")

    client = _client(handler)
    with pytest.raises(SaverRejected) as info:
        await _prepare(client)
    await client.aclose()
    assert info.value.reason == "too_large"


@pytest.mark.asyncio
async def test_a_noncanonical_base_uploads_nothing_without_the_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RADIENT_ORG_ALLOW_NONCANONICAL_BASE", raising=False)
    calls: list[httpx.Request] = []

    async def source(force_refresh: bool) -> str:
        return RADIENT_BEARER

    client = SaverClient(
        "https://radient.attacker.example/v1",
        bearer=source,
        transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, json=OK)),
    )
    with pytest.raises(SaverRefused) as info:
        await _prepare(client)
    await client.aclose()
    assert info.value.reason == "noncanonical_base"
    assert calls == []


def test_the_client_never_trusts_ambient_proxy_or_netrc() -> None:
    client = SaverClient(BASE, bearer=lambda force: None)  # type: ignore[arg-type]
    assert client._http.trust_env is False
    assert client._http.follow_redirects is False


def test_stored_provider_secrets_covers_store_rows_and_env_keys(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.providers.auth_store import AuthStore

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai-key")
    store = AuthStore(tmp_path / "auth.db")
    try:
        store.upsert_credential("anthropic", {"key": PROVIDER_KEY, "type": "api_key"})
        store.upsert_credential(
            "openai", {"access": OAUTH_ACCESS, "refresh": "rt-x", "type": "oauth", "expires": 0}
        )
        store.upsert_credential("radient", {"key": RADIENT_BEARER, "type": "api_key"})
    finally:
        store.close()
    secrets = stored_provider_secrets(tmp_path)
    assert PROVIDER_KEY in secrets
    assert OAUTH_ACCESS in secrets
    assert "sk-env-openai-key" in secrets
    assert RADIENT_BEARER not in secrets
