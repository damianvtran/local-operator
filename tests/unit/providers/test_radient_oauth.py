"""Unit tests for Radient OAuth PKCE flow and token refresh."""

from __future__ import annotations

import httpx
import pytest

from local_operator.providers.oauth.radient import (
    RadientOAuthFlow,
    refresh_radient_token,
)
from local_operator.providers.registry import get_provider_definition


def test_radient_provider_definition():
    radient = get_provider_definition("radient")
    assert radient is not None
    assert radient.id == "radient"
    assert radient.base_url == "https://api.radienthq.com/v1"
    assert radient.login is not None

    oauth_def = get_provider_definition("radient-oauth")
    assert oauth_def is not None
    assert oauth_def.id == "radient-oauth"
    assert oauth_def.store_credentials_as == "radient"
    assert oauth_def.callback_port == 54549
    assert oauth_def.login is not None
    assert oauth_def.refresh_token is not None


@pytest.mark.asyncio
async def test_radient_oauth_flow_generate_auth_url():
    flow = RadientOAuthFlow()
    url = await flow.generate_auth_url("test-state", "http://localhost:54549/callback")
    assert "https://console.radienthq.com/oauth/authorize" in url
    assert "client_id=lop" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A54549%2Fcallback" in url
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert "state=test-state" in url


@pytest.mark.asyncio
async def test_radient_oauth_exchange_code():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.radienthq.com/v1/auth/oauth/token"
        assert request.headers["content-type"] == "application/json"
        return httpx.Response(
            200,
            json={
                "access_token": "rad-jwt-access-token",
                "refresh_token": "rad-refresh-token-12345",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        flow = RadientOAuthFlow(http_client=client)
        # Generate auth url to set PKCE verifier
        await flow.generate_auth_url("state", "http://localhost:54549/callback")
        creds = await flow.exchange_token("code-123", "state", "http://localhost:54549/callback")
        assert creds["type"] == "oauth"
        assert creds["access"] == "rad-jwt-access-token"
        assert creds["refresh"] == "rad-refresh-token-12345"
        assert creds["access_token"] == "rad-jwt-access-token"
        assert creds["refresh_token"] == "rad-refresh-token-12345"
        assert creds["expires"] > 0
        assert creds["authorized_at"] > 0


@pytest.mark.asyncio
async def test_radient_refresh_token():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.radienthq.com/v1/auth/oauth/token"
        return httpx.Response(
            200,
            json={
                "access_token": "new-jwt-access-token",
                "refresh_token": "new-refresh-token-67890",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        initial = {
            "type": "oauth",
            "access": "old-token",
            "refresh": "old-refresh",
            "access_token": "old-token",
            "refresh_token": "old-refresh",
            "expires": 1000,
            "authorized_at": 500,
        }
        refreshed = await refresh_radient_token(initial, http_client=client)
        assert refreshed["type"] == "oauth"
        assert refreshed["access"] == "new-jwt-access-token"
        assert refreshed["refresh"] == "new-refresh-token-67890"
        assert refreshed["access_token"] == "new-jwt-access-token"
        assert refreshed["refresh_token"] == "new-refresh-token-67890"
        assert refreshed["expires"] > initial["expires"]
        assert refreshed["authorized_at"] == 500


@pytest.mark.asyncio
async def test_radient_auth_store_round_trip(tmp_path):
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(tmp_path / "auth.db")
    creds = {
        "type": "oauth",
        "access": "rad-jwt-access-token",
        "refresh": "rad-refresh-token",
        "access_token": "rad-jwt-access-token",
        "refresh_token": "rad-refresh-token",
        "expires": 2000000000000,
    }
    # Upsert under storage provider 'radient'
    stored = store.upsert_credential("radient", creds)
    assert stored.credential_type == "oauth"

    # Resolves through cascade via get_api_key and get_oauth_access
    key = await store.get_api_key("radient")
    assert key == "rad-jwt-access-token"

    oauth_access = await store.get_oauth_access("radient")
    assert oauth_access is not None
    assert oauth_access.access_token == "rad-jwt-access-token"
    assert oauth_access.kind == "oauth"
