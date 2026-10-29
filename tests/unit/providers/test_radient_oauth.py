"""Unit tests for Radient OAuth PKCE flow and token refresh."""

from __future__ import annotations

from typing import Any

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
    assert radient.refresh_token is not None
    assert radient.callback_port == 54549
    assert "radient-oauth" in radient.search_aliases

    key_def = get_provider_definition("radient-key")
    assert key_def is not None
    assert key_def.id == "radient-key"
    assert key_def.store_credentials_as == "radient"
    assert key_def.login is not None
    assert key_def.paste_prompt_required is True


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
        if request.method == "GET":
            assert str(request.url) == "https://api.radienthq.com/v1/me"
            assert request.headers["authorization"].startswith("Bearer ")
            return httpx.Response(
                200,
                json={
                    "result": {
                        "identity": {"account_id": "verified-account"},
                        "account": {"id": "verified-account"},
                    }
                },
            )
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
        assert creds["account_id"] == "verified-account"
        assert creds["api_endpoint"] == "https://api.radienthq.com/v1"


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


def _jwt(claims: dict[str, Any]) -> str:
    """An UNSIGNED JWT carrying ``claims`` — the decoder never verifies."""
    import base64
    import json

    def seg(value: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return f"{seg({'alg': 'HS256'})}.{seg(claims)}.sig"


@pytest.mark.asyncio
async def test_login_keeps_display_labels_but_profile_controls_account():
    """agent-server mints ``id_token`` with sub/email/name on the code grant;
    the stored credential now carries them (audit A5/A6) — the label the
    profile writer and the desktop's ``operator`` field read."""
    id_token = _jwt({"sub": "acct_123", "email": "jane@x.com", "name": "Jane Doe"})

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.headers["authorization"] == "Bearer a"
            return httpx.Response(
                200,
                json={
                    "result": {
                        "identity": {"account_id": "server-account"},
                        "account": {"id": "server-account"},
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "access_token": "a",
                "refresh_token": "r",
                "id_token": id_token,
                "expires_in": 3600,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        flow = RadientOAuthFlow(http_client=client)
        await flow.generate_auth_url("s", "http://localhost:54549/callback")
        creds = await flow.exchange_token("c", "s", "http://localhost:54549/callback")
    assert creds["email"] == "jane@x.com"
    assert creds["name"] == "Jane Doe"
    assert creds["account_id"] == "server-account"


def test_a_missing_or_malformed_id_token_changes_nothing():
    from local_operator.providers.oauth.radient import identity_from_token_response

    assert identity_from_token_response({}) == {}
    assert identity_from_token_response({"id_token": "not-a-jwt"}) == {}
    assert identity_from_token_response({"id_token": _jwt({"email": "  "})}) == {}


@pytest.mark.asyncio
async def test_the_refresh_grant_updates_the_identity_label():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "a2",
                "refresh_token": "r2",
                "id_token": _jwt({"sub": "acct_123", "email": "jane@x.com", "name": "Jane Q"}),
                "expires_in": 3600,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        refreshed = await refresh_radient_token(
            {"type": "oauth", "refresh": "r", "name": "Jane", "authorized_at": 1},
            http_client=client,
        )
    assert refreshed["name"] == "Jane Q"
    assert "account_id" not in refreshed  # Refresh labels cannot newly admit an account.


def test_a_pre_identity_radient_row_is_not_rebound_without_account_proof(tmp_path, monkeypatch):
    """A pre-identity row might belong to another account; preserve its pin.

    A newly identified login cannot prove ownership of an old constant-keyed
    row. Only matching account and nonempty stored hub permit adoption;
    the account-retention regressions exercise that separate upgrade path.
    """
    from local_operator.providers.auth_store import AuthStore

    monkeypatch.setattr(AuthStore, "_rearm_parked_tunnel_login", lambda *args: None)
    store = AuthStore(tmp_path / "auth.db")
    old = store.upsert_credential(
        "radient", {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}
    )
    assert old.identity_key == "oauth:radient"
    new = store.upsert_credential(
        "radient",
        {"type": "oauth", "access": "a2", "refresh": "r2", "expires": 2, "account_id": "acct_1"},
    )
    rows = store.list_credentials("radient")
    assert len(rows) == 2 and old.id != new.id
    assert rows[0].identity_key == "oauth:radient"
    assert new.data["account_id"] == "acct_1"
    # A SECOND account is still its own row.
    store.upsert_credential(
        "radient",
        {"type": "oauth", "access": "b", "refresh": "rb", "expires": 3, "account_id": "acct_2"},
    )
    assert len(store.list_credentials("radient")) == 3
    store.close()
