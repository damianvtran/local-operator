"""Unit tests for Google Vertex AI OAuth2 + PKCE and Service Account JWT flows."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from local_operator.providers.oauth import vertex
from local_operator.providers.oauth.callback_server import (
    InvalidGrantError,
    LoginCallbacks,
    LoginCancelledError,
    LoginError,
)


def _generate_rsa_key_pair() -> tuple[str, str]:
    """Generate an ephemeral 2048-bit RSA key pair (private PEM, public PEM)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    return private_pem, public_pem


@pytest.mark.asyncio
async def test_login_vertex_oauth_pkce_userinfo_and_vertex_project_autodiscovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browser login exchanges PKCE code, reads userinfo email, and prefers a *vertex* project."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)

    seen_requests: list[tuple[str, str, dict[str, Any]]] = []
    progress_messages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == vertex.TOKEN_URL:
            form = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
            seen_requests.append((request.method, url, form))
            return httpx.Response(
                200,
                json={
                    "access_token": "ya29.vertex-access-token",
                    "refresh_token": "1//vertex-refresh-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        if url == vertex.USERINFO_URL:
            seen_requests.append(
                (request.method, url, {"auth": request.headers.get("authorization")})
            )
            return httpx.Response(200, json={"email": "damian@example.com"})
        if url.startswith(vertex.PROJECTS_URL):
            seen_requests.append(
                (request.method, url, {"auth": request.headers.get("authorization")})
            )
            return httpx.Response(
                200,
                json={
                    "projects": [
                        {"projectId": "misc-analytics", "lifecycleState": "ACTIVE"},
                        {"projectId": "minerva-vertex", "lifecycleState": "ACTIVE"},
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {url}")

    auth_urls: list[str] = []

    async def on_auth_url(url: str, instructions: str | None = None) -> None:
        auth_urls.append(url)
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query)
        assert parsed.scheme == "https"
        assert parsed.netloc == "accounts.google.com"
        assert query["code_challenge_method"] == ["S256"]
        assert query["access_type"] == ["offline"]
        assert query["prompt"] == ["consent"]
        assert vertex.CLOUD_PLATFORM_SCOPE in query["scope"][0]

        # Simulate the browser redirect landing on the loopback callback server.
        redirect_uri = query["redirect_uri"][0]
        state = query["state"][0]
        async with httpx.AsyncClient() as loopback:
            resp = await loopback.get(
                redirect_uri, params={"code": "auth-code-123", "state": state}
            )
            assert resp.status_code == 200

    callbacks = LoginCallbacks(
        on_auth_url=on_auth_url,
        on_progress=progress_messages.append,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        creds = await vertex.login_vertex_oauth(
            callbacks,
            open_browser=lambda _url: None,
            http_client=http,
        )

    assert len(auth_urls) == 1
    assert creds["type"] == "oauth"
    assert creds["access"] == "ya29.vertex-access-token"
    assert creds["refresh"] == "1//vertex-refresh-token"
    assert creds["email"] == "damian@example.com"
    assert creds["account_id"] == "damian@example.com"
    assert creds["org_name"] == "minerva-vertex"
    assert creds["auth_mode"] == "oauth"
    assert creds["project_id"] == "minerva-vertex"
    assert creds["location"] == "global"
    assert isinstance(creds["expires"], int) and creds["expires"] > 0

    token_req = next(form for _m, url, form in seen_requests if url == vertex.TOKEN_URL)
    assert token_req["grant_type"] == "authorization_code"
    assert token_req["code"] == "auth-code-123"
    assert token_req["code_verifier"]
    assert any("minerva-vertex" in msg for msg in progress_messages)


@pytest.mark.asyncio
async def test_login_vertex_oauth_prefers_env_project_and_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION override heuristic project selection."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "prod-ml-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == vertex.TOKEN_URL:
            return httpx.Response(
                200,
                json={"access_token": "ya29.tok", "refresh_token": "1//ref", "expires_in": 3600},
            )
        if url == vertex.USERINFO_URL:
            return httpx.Response(200, json={"email": "ops@example.com"})
        if url.startswith(vertex.PROJECTS_URL):
            return httpx.Response(
                200,
                json={
                    "projects": [
                        {"projectId": "other-vertex", "lifecycleState": "ACTIVE"},
                        {"projectId": "prod-ml-project", "lifecycleState": "ACTIVE"},
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {url}")

    async def on_auth_url(url: str, instructions: str | None = None) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        async with httpx.AsyncClient() as loopback:
            await loopback.get(
                query["redirect_uri"][0],
                params={"code": "c-env", "state": query["state"][0]},
            )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        creds = await vertex.login_vertex_oauth(
            LoginCallbacks(on_auth_url=on_auth_url),
            open_browser=lambda _url: None,
            http_client=http,
        )

    assert creds["project_id"] == "prod-ml-project"
    assert creds["location"] == "us-central1"


@pytest.mark.asyncio
async def test_login_vertex_oauth_probes_service_usage_when_no_vertex_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When <=5 projects exist and none contains 'vertex', pick the one with aiplatform enabled."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == vertex.TOKEN_URL:
            return httpx.Response(
                200,
                json={"access_token": "ya29.tok", "refresh_token": "1//ref", "expires_in": 3600},
            )
        if url == vertex.USERINFO_URL:
            return httpx.Response(200, json={"email": "dev@example.com"})
        if url.startswith(vertex.PROJECTS_URL):
            return httpx.Response(
                200,
                json={
                    "projects": [
                        {"projectId": "alpha-web", "lifecycleState": "ACTIVE"},
                        {"projectId": "beta-ai", "lifecycleState": "ACTIVE"},
                    ]
                },
            )
        if "alpha-web/services/aiplatform.googleapis.com" in url:
            return httpx.Response(200, json={"state": "DISABLED"})
        if "beta-ai/services/aiplatform.googleapis.com" in url:
            return httpx.Response(200, json={"state": "ENABLED"})
        raise AssertionError(f"Unexpected URL: {url}")

    async def on_auth_url(url: str, instructions: str | None = None) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        async with httpx.AsyncClient() as loopback:
            await loopback.get(
                query["redirect_uri"][0],
                params={"code": "c-su", "state": query["state"][0]},
            )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        creds = await vertex.login_vertex_oauth(
            LoginCallbacks(on_auth_url=on_auth_url),
            open_browser=lambda _url: None,
            http_client=http,
        )

    assert creds["project_id"] == "beta-ai"


@pytest.mark.asyncio
async def test_login_vertex_oauth_falls_back_to_manual_project_prompt_when_none_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Cloud Resource Manager returns zero projects, prompt once via on_manual_code_input."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == vertex.TOKEN_URL:
            return httpx.Response(
                200,
                json={"access_token": "ya29.tok", "refresh_token": "1//ref", "expires_in": 3600},
            )
        if url == vertex.USERINFO_URL:
            return httpx.Response(200, json={"email": "dev@example.com"})
        if url.startswith(vertex.PROJECTS_URL):
            return httpx.Response(200, json={"projects": []})
        raise AssertionError(f"Unexpected URL: {url}")

    prompt_calls = 0

    async def on_auth_url(url: str, instructions: str | None = None) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        async with httpx.AsyncClient() as loopback:
            await loopback.get(
                query["redirect_uri"][0],
                params={"code": "c-empty", "state": query["state"][0]},
            )

    async def on_manual() -> str | None:
        nonlocal prompt_calls
        prompt_calls += 1
        # First call may race the loopback callback; delay it so the callback wins,
        # then the second call (post-exchange) supplies the project id.
        if prompt_calls == 1:
            import asyncio

            await asyncio.sleep(5)
            return None
        return "manual-gcp-project"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        creds = await vertex.login_vertex_oauth(
            LoginCallbacks(on_auth_url=on_auth_url, on_manual_code_input=on_manual),
            open_browser=lambda _url: None,
            http_client=http,
        )

    assert creds["project_id"] == "manual-gcp-project"


@pytest.mark.asyncio
async def test_refresh_vertex_token_oauth_preserves_metadata_and_refresh_token() -> None:
    """OAuth refresh preserves project_id, location, email, and refresh when omitted."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == vertex.TOKEN_URL
        form = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
        assert form["grant_type"] == "refresh_token"
        assert form["refresh_token"] == "1//original-refresh"
        return httpx.Response(
            200,
            json={"access_token": "ya29.refreshed-token", "expires_in": 3600},
        )

    stored = {
        "type": "oauth",
        "access": "ya29.old",
        "refresh": "1//original-refresh",
        "expires": 1,
        "email": "user@corp.com",
        "project_id": "corp-vertex",
        "location": "europe-west4",
        "auth_mode": "oauth",
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        refreshed = await vertex.refresh_vertex_token(stored, http_client=http)

    assert refreshed["access"] == "ya29.refreshed-token"
    assert refreshed["refresh"] == "1//original-refresh"
    assert refreshed["project_id"] == "corp-vertex"
    assert refreshed["location"] == "europe-west4"
    assert refreshed["email"] == "user@corp.com"
    assert refreshed["auth_mode"] == "oauth"


@pytest.mark.asyncio
async def test_refresh_vertex_token_invalid_grant_raises_terminal_error() -> None:
    """Google's 400 invalid_grant classifies as InvalidGrantError so auth_store tombstones it."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "Token has been expired or revoked.",
            },
        )

    stored = {"refresh": "1//revoked", "auth_mode": "oauth", "project_id": "proj"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(InvalidGrantError):
            await vertex.refresh_vertex_token(stored, http_client=http)


@pytest.mark.asyncio
async def test_service_account_login_from_json_and_jwt_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service account JSON mints an RS256-signed JWT assertion and refreshes identically."""
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)
    private_pem, public_pem = _generate_rsa_key_pair()
    sa_payload = {
        "type": "service_account",
        "project_id": "minerva-vertex",
        "private_key": private_pem,
        "client_email": "vertex-bot@minerva-vertex.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    minted_assertions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://oauth2.googleapis.com/token"
        form = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
        assert form["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
        assertion = form["assertion"]
        minted_assertions.append(assertion)
        claims = jwt.decode(
            assertion,
            public_pem,
            algorithms=["RS256"],
            audience="https://oauth2.googleapis.com/token",
        )
        assert claims["iss"] == sa_payload["client_email"]
        assert claims["sub"] == sa_payload["client_email"]
        assert claims["scope"] == vertex.CLOUD_PLATFORM_SCOPE
        assert claims["exp"] - claims["iat"] == vertex.JWT_LIFETIME_SECONDS
        return httpx.Response(
            200,
            json={
                "access_token": f"ya29.sa-token-{len(minted_assertions)}",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )

    callbacks = LoginCallbacks(on_manual_code_input=lambda: json.dumps(sa_payload))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        creds = await vertex.login_vertex_service_account(callbacks, http_client=http)
        assert creds["type"] == "oauth"
        assert creds["auth_mode"] == "service_account"
        assert creds["access"] == "ya29.sa-token-1"
        assert creds["email"] == sa_payload["client_email"]
        assert creds["project_id"] == "minerva-vertex"
        assert creds["location"] == "global"
        assert creds["private_key"] == private_pem

        refreshed = await vertex.refresh_vertex_token(creds, http_client=http)
        assert refreshed["access"] == "ya29.sa-token-2"
        assert refreshed["auth_mode"] == "service_account"
        assert refreshed["project_id"] == "minerva-vertex"
        assert refreshed["client_email"] == sa_payload["client_email"]
        assert refreshed["private_key"] == private_pem

    assert len(minted_assertions) == 2


@pytest.mark.asyncio
async def test_service_account_login_from_file_path(tmp_path: Path) -> None:
    """Passing a local .json path loads the service account key from disk."""
    private_pem, public_pem = _generate_rsa_key_pair()
    key_file = tmp_path / "vertex-sa.json"
    key_file.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "file-project",
                "private_key": private_pem,
                "client_email": "sa@file-project.iam.gserviceaccount.com",
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        form = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
        claims = jwt.decode(
            form["assertion"],
            public_pem,
            algorithms=["RS256"],
            audience=vertex.TOKEN_URL,
        )
        assert claims["iss"] == "sa@file-project.iam.gserviceaccount.com"
        return httpx.Response(200, json={"access_token": "ya29.from-file", "expires_in": 3600})

    callbacks = LoginCallbacks(on_manual_code_input=lambda: str(key_file))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        creds = await vertex.login_vertex_service_account(callbacks, http_client=http)

    assert creds["access"] == "ya29.from-file"
    assert creds["project_id"] == "file-project"
    assert creds["token_uri"] == vertex.TOKEN_URL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_input",
    [
        "not-json-at-all",
        json.dumps(
            {
                "type": "authorized_user",
                "client_email": "a@b",
                "private_key": "k",
                "project_id": "p",
            }
        ),
        json.dumps({"type": "service_account", "client_email": "a@b", "project_id": "p"}),
    ],
)
async def test_service_account_login_rejects_invalid_json(bad_input: str) -> None:
    callbacks = LoginCallbacks(on_manual_code_input=lambda: bad_input)
    with pytest.raises(LoginError):
        await vertex.login_vertex_service_account(callbacks)


@pytest.mark.asyncio
async def test_service_account_login_empty_or_none_cancels() -> None:
    with pytest.raises(LoginCancelledError):
        await vertex.login_vertex_service_account(
            LoginCallbacks(on_manual_code_input=lambda: None)
        )
    with pytest.raises(LoginCancelledError):
        await vertex.login_vertex_service_account(
            LoginCallbacks(on_manual_code_input=lambda: "   ")
        )
