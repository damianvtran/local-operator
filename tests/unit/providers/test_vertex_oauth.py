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
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
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
        await vertex.login_vertex_service_account(LoginCallbacks(on_manual_code_input=lambda: None))
    with pytest.raises(LoginCancelledError):
        await vertex.login_vertex_service_account(
            LoginCallbacks(on_manual_code_input=lambda: "   ")
        )


# --- Round 1 remediation regressions -------------------------------------


def _sa_payload(private_pem: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "service_account",
        "project_id": "minerva-vertex",
        "private_key": private_pem,
        "client_email": "sa@minerva-vertex.iam.gserviceaccount.com",
    }
    payload.update(extra)
    return payload


def _ok_token_handler(seen: list[str]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"access_token": "ya29.ok", "expires_in": 3600})

    return handler


@pytest.fixture
def _no_token_uri_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_VERTEX_ALLOW_CUSTOM_TOKEN_URI", raising=False)
    monkeypatch.delenv("GOOGLE_VERTEX_TOKEN_URI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token_uri_override")
async def test_service_account_login_rejects_non_google_token_uri_without_sending() -> None:
    """MAJOR 1: a forged token_uri must never receive the signed JWT assertion."""
    private_pem, _ = _generate_rsa_key_pair()
    seen: list[str] = []
    payload = _sa_payload(private_pem, token_uri="https://evil.example/token")
    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_token_handler(seen))) as http:
        with pytest.raises(LoginError, match="not a Google OAuth2 token endpoint"):
            await vertex.login_vertex_service_account(
                LoginCallbacks(on_manual_code_input=lambda: json.dumps(payload)),
                http_client=http,
            )
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token_uri_override")
async def test_service_account_refresh_rejects_stored_non_google_token_uri() -> None:
    """MAJOR 1: refresh re-validates a stored token_uri, not just login."""
    private_pem, _ = _generate_rsa_key_pair()
    seen: list[str] = []
    stored = {
        "type": "oauth",
        "auth_mode": "service_account",
        "client_email": "sa@p.iam.gserviceaccount.com",
        "private_key": private_pem,
        "token_uri": "https://evil.example/token",
        "project_id": "p-proj",
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_token_handler(seen))) as http:
        with pytest.raises(LoginError, match="not a Google OAuth2 token endpoint"):
            await vertex.refresh_vertex_token(stored, http_client=http)
    assert seen == []


@pytest.mark.usefixtures("_no_token_uri_override")
@pytest.mark.parametrize(
    "uri",
    ["https://oauth2.googleapis.com/token", "https://www.googleapis.com/oauth2/v4/token"],
)
def test_validate_token_uri_accepts_google_endpoints(uri: str) -> None:
    assert vertex._validate_token_uri(uri) == uri


@pytest.mark.usefixtures("_no_token_uri_override")
@pytest.mark.parametrize(
    "uri",
    [
        "http://oauth2.googleapis.com/token",
        "https://oauth2.googleapis.com.evil.example/token",
        "https://oauth2.googleapis.com:8443/token",
        "https://user@oauth2.googleapis.com/token",
        "https://oauth2.googleapis.com/other",
    ],
)
def test_validate_token_uri_rejects_lookalikes(uri: str) -> None:
    with pytest.raises(LoginError):
        vertex._validate_token_uri(uri)


@pytest.mark.usefixtures("_no_token_uri_override")
def test_validate_token_uri_custom_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = "https://private.example/token"
    with pytest.raises(LoginError):
        vertex._validate_token_uri(custom)
    monkeypatch.setenv("GOOGLE_VERTEX_TOKEN_URI", custom)
    assert vertex._validate_token_uri(custom) == custom
    with pytest.raises(LoginError):
        vertex._validate_token_uri("https://other.example/token")
    monkeypatch.delenv("GOOGLE_VERTEX_TOKEN_URI")
    monkeypatch.setenv("GOOGLE_VERTEX_ALLOW_CUSTOM_TOKEN_URI", "1")
    assert vertex._validate_token_uri("https://other.example/token")
    # Even opted in, plain http is never accepted.
    with pytest.raises(LoginError):
        vertex._validate_token_uri("http://other.example/token")


@pytest.mark.parametrize("bad", ["../other", "a/b", "proj#x", "proj?x=1", "", "-lead"])
def test_validate_project_id_rejects_url_breaking_values(bad: str) -> None:
    with pytest.raises(LoginError):
        vertex.validate_project_id(bad)


@pytest.mark.parametrize("bad", ["attacker.com/x#", "us-central1/../x", "US-CENTRAL1", ""])
def test_validate_location_rejects_url_breaking_values(bad: str) -> None:
    with pytest.raises(LoginError):
        vertex.validate_location(bad)


def test_validate_project_and_location_accept_real_values() -> None:
    assert vertex.validate_project_id("example.com:my-proj") == "example.com:my-proj"
    assert vertex.validate_location("global") == "global"
    assert vertex.validate_location("europe-west4") == "europe-west4"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token_uri_override")
async def test_service_account_login_rejects_bad_location_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAJOR 2: an injected location is refused before any token is minted."""
    private_pem, _ = _generate_rsa_key_pair()
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "attacker.com/x#")
    seen: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_token_handler(seen))) as http:
        with pytest.raises(LoginError, match="Invalid Google Cloud location"):
            await vertex.login_vertex_service_account(
                LoginCallbacks(on_manual_code_input=lambda: json.dumps(_sa_payload(private_pem))),
                http_client=http,
            )
    assert seen == []


def test_load_service_account_json_rejects_oversized_file(tmp_path: Path) -> None:
    """MAJOR 5: a >64 KiB 'key file' is refused before it is read."""
    big = tmp_path / "big.json"
    big.write_text("{" + " " * (vertex.MAX_SERVICE_ACCOUNT_JSON_BYTES + 10) + "}")
    with pytest.raises(LoginError, match="larger than 64 KiB"):
        vertex._load_service_account_json(str(big))


def test_load_service_account_json_rejects_non_json_suffix(tmp_path: Path) -> None:
    """MAJOR 5: arbitrary files (e.g. /etc/passwd) are not read as key files."""
    other = tmp_path / "passwd"
    other.write_text("root:x:0:0:root:/root:/bin/sh\n")
    with pytest.raises(LoginError, match="must end in .json") as excinfo:
        vertex._load_service_account_json(str(other))
    assert "root:x" not in str(excinfo.value)


def test_load_service_account_json_rejects_directory_named_json(tmp_path: Path) -> None:
    folder = tmp_path / "dir.json"
    folder.mkdir()
    with pytest.raises(LoginError, match="not a regular file"):
        vertex._load_service_account_json(str(folder))


def test_load_service_account_json_rejects_non_service_account_file(tmp_path: Path) -> None:
    other = tmp_path / "creds.json"
    other.write_text(json.dumps({"type": "authorized_user", "refresh_token": "x"}))
    with pytest.raises(LoginError, match="service_account"):
        vertex._load_service_account_json(str(other))


def test_load_service_account_json_accepts_path_and_pretty_json(tmp_path: Path) -> None:
    payload = _sa_payload("-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n")
    pretty = json.dumps(payload, indent=2)
    assert vertex._load_service_account_json(pretty)["project_id"] == "minerva-vertex"
    key_file = tmp_path / "sa.json"
    key_file.write_text(pretty)
    assert vertex._load_service_account_json(f'"{key_file}"')["client_email"].startswith("sa@")


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token_uri_override")
async def test_service_account_token_exchange_guards_bad_json_and_expiry() -> None:
    """MINOR 3: junk bodies and garbled expires_in become clean outcomes."""
    private_pem, _ = _generate_rsa_key_pair()
    bodies = iter(
        [
            httpx.Response(200, text="<html>oops</html>"),
            httpx.Response(200, json={"access_token": "ya29.x", "expires_in": "soon"}),
            httpx.Response(500, text="E" * 5000),
        ]
    )
    callbacks = LoginCallbacks(on_manual_code_input=lambda: json.dumps(_sa_payload(private_pem)))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: next(bodies))) as http:
        with pytest.raises(LoginError, match="non-JSON"):
            await vertex.login_vertex_service_account(callbacks, http_client=http)
        creds = await vertex.login_vertex_service_account(callbacks, http_client=http)
        assert creds["access"] == "ya29.x"
        with pytest.raises(LoginError) as excinfo:
            await vertex.login_vertex_service_account(callbacks, http_client=http)
    assert len(str(excinfo.value)) < 400


def test_default_client_secret_is_not_a_literal_in_source() -> None:
    """MINOR 3: the public installed-app secret is stored encoded, decoded at import."""
    source = Path(vertex.__file__).read_text(encoding="utf-8")
    assert vertex.DEFAULT_CLIENT_SECRET
    assert f'"{vertex.DEFAULT_CLIENT_SECRET}"' not in source


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_token_uri_override")
async def test_service_account_refresh_only_updates_token_fields() -> None:
    """NIT 1: refresh rewrites access/expires/token_uri and leaves the rest as stored."""
    private_pem, _ = _generate_rsa_key_pair()
    stored = {
        "type": "oauth",
        "auth_mode": "service_account",
        "client_email": "sa@p.iam.gserviceaccount.com",
        "private_key": private_pem,
        "project_id": "p-proj",
        "location": "us-central1",
        "access": "old",
        "expires": 1,
    }
    seen: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_token_handler(seen))) as http:
        refreshed = await vertex.refresh_vertex_token(stored, http_client=http)
    changed = {k for k in refreshed if refreshed.get(k) != stored.get(k)}
    assert changed == {"access", "expires", "token_uri"}
    assert seen == [vertex.TOKEN_URL]


@pytest.mark.asyncio
async def test_login_vertex_oauth_no_project_and_no_answer_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAJOR 4: zero projects and an empty project answer ends with guidance, not a
    credential that fails on the first request."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == vertex.TOKEN_URL:
            return httpx.Response(200, json={"access_token": "ya29.t", "expires_in": 3600})
        if url == vertex.USERINFO_URL:
            return httpx.Response(200, json={"email": "dev@example.com"})
        return httpx.Response(200, json={"projects": []})

    flow = vertex.VertexOAuthFlow(
        LoginCallbacks(on_manual_code_input=lambda: ""),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LoginError, match="GOOGLE_CLOUD_PROJECT"):
        await flow.exchange_token("code", "state", "http://127.0.0.1:8085/")


# --- CLI paste path (MAJOR 5 / QA Q1, MAJOR 4 CLI half) ---------------------


def _cli_prompt(provider_id: str) -> Any:
    from local_operator.providers.auth_cli import _callbacks_interactive
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(provider_id)
    assert definition is not None
    return _callbacks_interactive(definition)


@pytest.mark.parametrize("indent", [2, None])
def test_cli_reads_piped_multiline_service_account_json_whole(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], indent: int | None
) -> None:
    """``cat sa.json | lop login google-vertex-sa``: pretty-printed JSON used to be
    truncated to its first line ``{`` by a single readline."""
    import asyncio
    import io

    private_pem, _ = _generate_rsa_key_pair()
    text = json.dumps(_sa_payload(private_pem), indent=indent) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    callbacks = _cli_prompt("google-vertex-sa")
    pasted = asyncio.run(callbacks.on_manual_code_input())
    assert json.loads(pasted)["client_email"].startswith("sa@")
    err = capsys.readouterr().err
    assert "Paste credential or file path" in err
    assert "API key" not in err


def test_cli_reads_interactive_multiline_json_until_it_parses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A TTY paste of pretty-printed JSON is gathered line by line until complete."""
    import asyncio
    import io

    from local_operator.providers import auth_cli

    private_pem, _ = _generate_rsa_key_pair()
    text = json.dumps(_sa_payload(private_pem), indent=2) + "\nTRAILING-NOT-READ\n"
    stdin = io.StringIO(text)
    stdin.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr(auth_cli, "_echo_disabled", lambda: _null_context())
    callbacks = _cli_prompt("google-vertex-sa")
    pasted = asyncio.run(callbacks.on_manual_code_input())
    assert json.loads(pasted)["project_id"] == "minerva-vertex"
    assert stdin.readline() == "TRAILING-NOT-READ\n"
    out = capsys.readouterr().out
    assert "Paste credential or file path" in out and "API key" not in out


def _null_context() -> Any:
    import contextlib

    return contextlib.nullcontext()


def test_cli_follow_up_prompt_adopts_the_parked_reader(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MAJOR 4 (CLI): after the browser wins, the parked fallback reader answers
    the project-ID follow-up instead of a second thread racing it for stdin."""
    import asyncio
    import threading

    typed = threading.Event()
    reads: list[str] = []

    def fake_input(prompt: str = "") -> str:
        reads.append(prompt)
        typed.wait(5)
        return "my-manual-proj"

    monkeypatch.setattr("builtins.input", fake_input)
    callbacks = _cli_prompt("google-vertex")

    async def scenario() -> str | None:
        first = asyncio.ensure_future(callbacks.on_manual_code_input())
        await asyncio.sleep(0.05)
        first.cancel()  # the browser callback won the race
        await asyncio.gather(first, return_exceptions=True)
        second = asyncio.ensure_future(callbacks.on_manual_code_input())
        await asyncio.sleep(0.05)
        typed.set()
        return await asyncio.wait_for(second, 5)

    assert asyncio.run(scenario()) == "my-manual-proj"
    assert reads == ["Paste the code here (empty to cancel): "]
    assert "Enter value (empty to cancel): " in capsys.readouterr().out


@pytest.mark.usefixtures("_no_token_uri_override")
def test_cli_login_google_vertex_sa_end_to_end_with_piped_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The registry's own google-vertex-sa login driven by the CLI's callbacks."""
    import asyncio
    import io

    from local_operator.providers.registry import get_provider_definition

    private_pem, _ = _generate_rsa_key_pair()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_sa_payload(private_pem), indent=2)))
    definition = get_provider_definition("google-vertex-sa")
    assert definition is not None and definition.login is not None
    callbacks = _cli_prompt("google-vertex-sa")
    seen: list[str] = []

    async def run() -> Any:
        transport = httpx.MockTransport(_ok_token_handler(seen))
        real_client = httpx.AsyncClient

        def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr(vertex.httpx, "AsyncClient", patched)
        login = definition.login
        assert login is not None
        return await login(callbacks)

    creds = asyncio.run(run())
    assert creds["auth_mode"] == "service_account" and creds["access"] == "ya29.ok"
    assert seen == [vertex.TOKEN_URL]
    assert "API key" not in capsys.readouterr().err
