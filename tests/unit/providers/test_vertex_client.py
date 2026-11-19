"""Unit tests for Google Vertex AI client routing, key validation, and desktop auth census."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec, TextContent
from local_operator.providers import documents, key_check
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.clients import (
    VERTEX_API_URL,
    GoogleClient,
    ProviderError,
    client_for_spec,
)
from local_operator.providers.registry import get_provider_definition


def _vertex_request(model_id: str = "gemini-3.8-flash") -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(
            provider="google-vertex",
            model_id=model_id,
            context_window=1_000_000,
            max_output_tokens=8192,
        ),
        messages=[Message(role="user", content=[TextContent(text="hello")])],
    )


def _sse_body() -> bytes:
    payload = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "hi from vertex"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3, "totalTokenCount": 7},
    }
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


@pytest.mark.asyncio
async def test_vertex_client_api_key_mode_uses_publisher_endpoint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, content=_sse_body(), headers={"content-type": "text/event-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = client_for_spec(_vertex_request().model, http_client=http)
        assert isinstance(client, GoogleClient)
        events = [e async for e in client.stream(_vertex_request(), api_key="vx-api-key-123")]

    assert len(seen) == 1
    req = seen[0]
    expected_url = (
        f"{VERTEX_API_URL}/v1/publishers/google/models/gemini-3.8-flash"
        ":streamGenerateContent?alt=sse"
    )
    assert str(req.url) == expected_url
    assert req.headers["x-goog-api-key"] == "vx-api-key-123"
    assert "authorization" not in req.headers
    assert any(getattr(e, "delta", None) == "hi from vertex" for e in events)


@pytest.mark.asyncio
async def test_vertex_client_oauth_global_and_regional_endpoints() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, content=_sse_body(), headers={"content-type": "text/event-stream"}
        )

    global_access = OAuthAccess(
        access_token="ya29.global-bearer",
        credential_id=1,
        email="user@example.com",
        kind="oauth",
        raw={"project_id": "minerva-vertex", "location": "global", "auth_mode": "oauth"},
    )
    regional_access = OAuthAccess(
        access_token="ya29.regional-bearer",
        credential_id=2,
        email="sa@minerva-vertex.iam.gserviceaccount.com",
        kind="oauth",
        raw={
            "project_id": "minerva-vertex",
            "location": "us-central1",
            "auth_mode": "service_account",
        },
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GoogleClient(base_url=VERTEX_API_URL, vertex=True, http_client=http)
        _ = [
            e
            async for e in client.stream(
                _vertex_request(), api_key=None, oauth_access=global_access
            )
        ]
        _ = [
            e
            async for e in client.stream(
                _vertex_request("gemini-2.5-pro"), api_key=None, oauth_access=regional_access
            )
        ]

    assert len(seen) == 2
    expected_global = (
        "https://aiplatform.googleapis.com/v1/projects/minerva-vertex/locations/global"
        "/publishers/google/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
    )
    expected_regional = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/minerva-vertex"
        "/locations/us-central1/publishers/google/models/gemini-2.5-pro"
        ":streamGenerateContent?alt=sse"
    )
    assert str(seen[0].url) == expected_global
    assert seen[0].headers["authorization"] == "Bearer ya29.global-bearer"
    assert seen[0].headers["x-goog-user-project"] == "minerva-vertex"
    assert "x-goog-api-key" not in seen[0].headers

    assert str(seen[1].url) == expected_regional
    assert seen[1].headers["authorization"] == "Bearer ya29.regional-bearer"


@pytest.mark.asyncio
async def test_vertex_client_oauth_without_project_raises_clear_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)

    missing_proj = OAuthAccess(
        access_token="ya29.no-project",
        credential_id=3,
        kind="oauth",
        raw={"auth_mode": "oauth"},
    )
    client = GoogleClient(base_url=VERTEX_API_URL, vertex=True)
    try:
        with pytest.raises(ProviderError) as exc_info:
            async for _ in client.stream(
                _vertex_request(), api_key=None, oauth_access=missing_proj
            ):
                pass
        assert "GOOGLE_CLOUD_PROJECT" in str(exc_info.value)
        assert "/login google-vertex" in str(exc_info.value)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "status", "body", "expected_valid"),
    [
        ("google-vertex", 200, '{"totalTokens": 2}', True),
        ("google-vertex-key", 200, '{"totalTokens": 2}', True),
        (
            "google-vertex",
            401,
            '{"error": {"message": "API keys are not supported or invalid"}}',
            False,
        ),
        ("google-vertex-key", 400, '{"error": {"message": "API key not valid."}}', False),
        ("google-vertex", 500, '{"error": {"message": "internal error"}}', None),
    ],
)
async def test_vertex_key_check_posts_count_tokens(
    provider_id: str, status: int, body: str, expected_valid: bool | None
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        verdict = await key_check.check_api_key(provider_id, "vx-test-key", client=http)

    assert verdict.valid is expected_valid
    assert len(seen) == 1
    req = seen[0]
    assert req.method == "POST"
    expected_probe = (
        "https://aiplatform.googleapis.com/v1/publishers/google"
        "/models/gemini-3.8-flash:countTokens"
    )
    assert str(req.url) == expected_probe
    assert req.headers["x-goog-api-key"] == "vx-test-key"
    assert json.loads(req.content) == {"contents": [{"role": "user", "parts": [{"text": "ping"}]}]}


def test_vertex_registry_documents_and_model_tables() -> None:
    from local_operator.model.defaults import suggested_model_for
    from local_operator.model.prices import OPENROUTER_NAMESPACE, _PRICE_CATALOGUE_KEYS
    from local_operator.model.registry import get_model_info

    primary = get_provider_definition("google-vertex")
    key_flavour = get_provider_definition("google-vertex-key")
    sa_flavour = get_provider_definition("google-vertex-sa")
    assert primary is not None and key_flavour is not None and sa_flavour is not None
    assert primary.login_kind == "browser"
    assert primary.paste_code_flow is True
    assert key_flavour.login_kind == "api_key"
    assert key_flavour.store_credentials_as == "google-vertex"
    assert sa_flavour.store_credentials_as == "google-vertex"
    assert sa_flavour.paste_prompt_required is True
    assert sa_flavour.paste_is_api_key is False

    assert documents.DOCUMENT_FORMS["google-vertex"].wire == "google"
    assert documents.admission_wire("google-vertex") == "google-content"
    assert OPENROUTER_NAMESPACE["google-vertex"] == "google"
    assert "google" in _PRICE_CATALOGUE_KEYS["google-vertex"]
    suggested = suggested_model_for("google-vertex")
    assert suggested is not None and suggested.id == "gemini-3.8-flash"
    info = get_model_info("google-vertex", "gemini-3.8-flash")
    assert info is not None and info.id == "gemini-3.8-flash"


@pytest.mark.asyncio
async def test_desktop_auth_providers_lists_all_three_vertex_auth_methods(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /v1/auth/providers surfaces google-vertex with browser, api_key, and SA JSON methods."""
    from fastapi import FastAPI
    from httpx import ASGITransport

    from local_operator.config import ConfigManager
    from local_operator.providers.auth_store import AuthStore
    from local_operator.server.dependencies import get_config_manager
    from local_operator.server.desktop import require_desktop
    from local_operator.server.routes.auth import get_desktop_auth, router as auth_router
    from local_operator.server.utils.desktop_auth import DesktopAuth

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    config_manager = ConfigManager(cfg_dir)
    host = DesktopAuth(AuthStore(cfg_dir / "auth.db", config_dir=cfg_dir), cfg_dir, config_manager)

    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[require_desktop] = lambda: None
    app.dependency_overrides[get_config_manager] = lambda: config_manager
    app.dependency_overrides[get_desktop_auth] = lambda: host

    async def fake_check(_provider: str, _key: str, **_kwargs: Any) -> key_check.KeyCheck:
        return key_check.KeyCheck(True, None)

    monkeypatch.setattr(key_check, "check_api_key", fake_check)

    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/auth/providers")
            assert resp.status_code == 200
            rows = resp.json()["result"]["providers"]
            vertex_row = next(r for r in rows if r["id"] == "google-vertex")
            assert vertex_row["brand"] == "Google Vertex AI"
            methods = {m["id"]: m for m in vertex_row["auth_methods"]}
            assert set(methods) == {"google-vertex", "google-vertex-key", "google-vertex-sa"}
            assert methods["google-vertex"]["kind"] == "browser"
            assert methods["google-vertex"]["paste_fallback"] is True
            assert methods["google-vertex-key"]["kind"] == "api_key"
            assert methods["google-vertex-key"]["requires_secret_input"] is True
            assert methods["google-vertex-sa"]["requires_secret_input"] is True

            # POST /v1/auth/login for google-vertex-sa opens a paste prompt cleanly.
            started = await client.post("/v1/auth/login", json={"provider": "google-vertex-sa"})
            assert started.status_code == 200
            op = started.json()["result"]
            assert op["state"] == "input_required"
            assert op["input_required"] is True
            assert op["prompt_id"]
            await client.delete(f"/v1/auth/operations/{op['id']}")

            # PUT /v1/auth/providers/google-vertex-key/key stores under google-vertex.
            saved = await client.put(
                "/v1/auth/providers/google-vertex-key/key",
                json={"value": "vx-desktop-api-key"},
            )
            assert saved.status_code == 200
            stored = host.store.list_credentials("google-vertex")
            assert len(stored) == 1
            assert stored[0].data["key"] == "vx-desktop-api-key"
    finally:
        await host.close()
        host.store.close()
