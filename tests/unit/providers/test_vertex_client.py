"""Unit tests for Google Vertex AI client routing, key validation, and desktop auth census."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamModelEvent,
    TextContent,
)
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
@pytest.mark.parametrize("provider", ["google-vertex", "kimi"])
@pytest.mark.parametrize("credential_change", ["block", "override", "stale", "explicit", "none"])
async def test_default_follows_actual_wire_credential(
    tmp_path, monkeypatch, provider: str, credential_change: str
) -> None:
    from local_operator.model.configure import configure_model
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.failover import stream_with_failover

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    cloud_id, key_id = {
        "google-vertex": ("gemini-4-neon", "gemini-2.5-flash"),
        "kimi": ("k3.1-pro", "kimi-k3.1-pro"),
    }[provider]
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = (
            _sse_body()
            if provider == "google-vertex"
            else (
                b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    try:
        oauth_data: dict[str, Any] = {"type": "oauth", "access": "synthetic-oauth"}
        if provider == "google-vertex":
            oauth_data.update(project_id="synthetic-project", location="global")
        if credential_change == "stale":
            oauth_data["expires"] = 1
        oauth = store.upsert_credential(provider, oauth_data)
        store.upsert_credential(
            provider, {"type": "api_key", "key": "synthetic-key", "source": "login"}
        )
        explicit = credential_change == "explicit"
        spec = configure_model(provider, cloud_id if explicit else "", config_dir=tmp_path).spec
        assert spec.model_id == (key_id if credential_change == "stale" else cloud_id)
        assert spec.auth_route_default is not explicit
        # The route can change AFTER model construction. This must exercise
        # the real cascade and clients, not just compare two static suggestions.
        if credential_change in {"block", "explicit"}:
            store.block_credential(oauth.id, provider)
        elif credential_change == "override":
            store.set_runtime_api_key(provider, "synthetic-override")

        request = _vertex_request().with_model(spec)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            events = [
                event
                async for event in stream_with_failover(
                    request,
                    store,
                    None,
                    lambda target: client_for_spec(target, http_client=http),
                    session_id="synthetic-route-default",
                )
            ]
        assert len(seen) == 1
        on_cloud = credential_change == "none"
        expected = cloud_id if on_cloud or explicit else key_id
        wire = seen[0]
        if provider == "google-vertex":
            prefix = "/v1/projects/synthetic-project/locations/global" if on_cloud else "/v1"
            assert (
                wire.url.path
                == f"{prefix}/publishers/google/models/{expected}:streamGenerateContent"
            )
            if on_cloud:
                assert wire.headers["authorization"] == "Bearer synthetic-oauth"
            else:
                assert wire.headers["x-goog-api-key"] == (
                    "synthetic-override" if credential_change == "override" else "synthetic-key"
                )
        else:
            assert wire.url.host == ("api.kimi.com" if on_cloud else "api.moonshot.cn")
            assert json.loads(wire.content)["model"] == expected
        model_events = [
            event.model.model_id for event in events if isinstance(event, StreamModelEvent)
        ]
        assert model_events == ([expected] if credential_change in {"block", "override"} else [])
        terminal = events[-1]
        assert isinstance(terminal, StreamEndEvent) and terminal.stop_reason == "stop"
    finally:
        store.close()


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


@pytest.mark.parametrize("location", ["us", "eu"])
def test_vertex_neon_multi_region_uses_data_residency_endpoint(location: str) -> None:
    from local_operator.providers.oauth.vertex import validate_location

    client = GoogleClient(base_url=VERTEX_API_URL, vertex=True)
    access = OAuthAccess(
        access_token="synthetic",
        credential_id=1,
        email=None,
        kind="oauth",
        raw={"project_id": "example-project", "location": validate_location(location)},
    )
    url, _ = client._vertex_endpoint_and_headers("gemini-4-neon", None, access)
    assert url == (
        f"https://aiplatform.{location}.rep.googleapis.com/v1/projects/example-project"
        f"/locations/{location}/publishers/google/models/gemini-4-neon"
        ":streamGenerateContent?alt=sse"
    )


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
    from local_operator.model.prices import _PRICE_CATALOGUE_KEYS, OPENROUTER_NAMESPACE
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
    assert suggested is not None and suggested.id == "gemini-2.5-flash"
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
    from local_operator.model.defaults import local_route_suggestion
    from local_operator.providers.auth_store import AuthStore
    from local_operator.server.routes.auth import router as auth_router
    from local_operator.server.routes.models import router as models_router
    from local_operator.server.utils.desktop_auth import DesktopAuth

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    config_manager = ConfigManager(cfg_dir)
    host = DesktopAuth(AuthStore(cfg_dir / "auth.db", config_dir=cfg_dir), cfg_dir, config_manager)

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(models_router)
    app.state.config_manager = config_manager
    app.state.desktop_auth = host
    desktop_token = "synthetic-vertex-census-bearer"
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", desktop_token)

    async def fake_check(_provider: str, _key: str, **_kwargs: Any) -> key_check.KeyCheck:
        return key_check.KeyCheck(True, None)

    monkeypatch.setattr(key_check, "check_api_key", fake_check)

    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {desktop_token}"},
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
            assert {
                method_id: method["suggested_model"]["id"] for method_id, method in methods.items()
            } == {
                "google-vertex": "gemini-4-neon",
                "google-vertex-key": "gemini-2.5-flash",
                "google-vertex-sa": "gemini-4-neon",
            }

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
            assert config_manager.get_config_value("hosting") == "google-vertex"
            assert not config_manager.get_config_value("model_name")

            async def provider_suggestion(path: str, field: str) -> dict[str, str]:
                response = await client.get(path)
                assert response.status_code == 200
                row = next(
                    row
                    for row in response.json()["result"]["providers"]
                    if row["id"] == "google-vertex"
                )
                return row[field]

            express = {"id": "gemini-2.5-flash", "name": "Vertex Gemini 2.5 Flash"}
            neon = {"id": "gemini-4-neon", "name": "Vertex Gemini 4 Neon"}
            assert vertex_row["suggested_model"] == express
            assert await provider_suggestion("/v1/models/providers", "suggestedModel") == express

            async def no_refresh(*_args: Any, **_kwargs: Any) -> Any:
                pytest.fail("provider previews must not attempt OAuth refresh")

            monkeypatch.setattr(host.store, "_ensure_oauth_fresh", no_refresh)
            cloud = {"type": "oauth", "access": "cloud-test", "location": "eu"}
            cases: list[tuple[dict[str, Any], str, dict[str, str]]] = [
                ({}, "fresh", neon),
                ({"expires": 1}, "expired without refresh", express),
                ({"expires": 1, "refresh": "test-refresh"}, "expired browser grant", neon),
                ({"expires": 1, "refresh_token": "test-refresh"}, "alternate token key", neon),
                ({"expires": 1, "refresh": "test-refresh"}, "blocked", express),
                ({"expires": 1, "refresh": "test-refresh"}, "dead", express),
                ({"expires": 1, "refresh": "test-refresh"}, "moved", express),
                (
                    {
                        "expires": 1,
                        "auth_mode": "service_account",
                        "private_key": "test-key",
                        "client_email": "vertex@example.test",
                    },
                    "service account",
                    neon,
                ),
                (
                    {"expires": 1, "auth_mode": "service_account", "private_key": "test-key"},
                    "incomplete service account",
                    express,
                ),
            ]
            for fields, case, expected in cases:
                oauth = host.store.upsert_credential("google-vertex", {**cloud, **fields})
                if case == "blocked":
                    host.store.block_credential(oauth.id, "google-vertex")
                elif case == "dead":
                    assert host.store._mark_grant_dead(
                        oauth.id, rejected_refresh_token="test-refresh"
                    )
                elif case == "moved":
                    assert host.store.mark_family_moved(oauth.id)
                assert (
                    await provider_suggestion("/v1/auth/providers", "suggested_model") == expected
                ), case
                assert (
                    await provider_suggestion("/v1/models/providers", "suggestedModel") == expected
                ), case
                local = local_route_suggestion("google-vertex", cfg_dir)
                assert local is not None and local.id == expected["id"], case
                if case == "fresh":
                    async with httpx.AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as public:
                        for headers in (
                            {},
                            {"Authorization": "Bearer wrong"},
                            {
                                "Authorization": f"Bearer {desktop_token}",
                                "Origin": "https://denied.test",
                            },
                        ):
                            response = await public.get("/v1/models/providers", headers=headers)
                            assert response.status_code == 200
                            assert response.headers["cache-control"] == "private, no-store"
                            row = next(
                                row
                                for row in response.json()["result"]["providers"]
                                if row["id"] == "google-vertex"
                            )
                            assert row["suggestedModel"] == express
                        assert (await public.get("/v1/auth/providers")).status_code == 401
                host.store.delete_credential(oauth.id)
    finally:
        await host.close()
        host.store.close()


# -- Vertex real-time catalogue discovery, credential project_id, pricing & ranking --


def test_fetch_vertex_oauth_paginates_and_filters_non_chat_models() -> None:
    """``_fetch_vertex`` sends Bearer + quota project, paginates, and keeps chat Gemini models."""
    from local_operator.model.discovery import fetch_models

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "pageToken" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "publisherModels": [
                        {"name": "publishers/google/models/gemini-4-neon"},
                        {"name": "publishers/google/models/gemini-3.8-flash@default"},
                        {"name": "publishers/google/models/gemini-embedding-2"},
                        {"name": "publishers/google/models/gemini-2.5-flash-tts"},
                        {"name": "publishers/google/models/gemini-2.5-flash-image-preview"},
                        {"name": "publishers/google/models/gemini-2.5-flash-live-001"},
                        {"name": "publishers/google/models/gemini-nano-banana-pro"},
                        {"name": "publishers/google/models/imagen-4.0-generate-001"},
                        {"name": "publishers/google/models/veo-3.0"},
                    ],
                    "nextPageToken": "page-2",
                },
            )
        return httpx.Response(
            200,
            json={
                "publisherModels": [
                    {"name": "publishers/google/models/gemini-4-neon"},  # duplicate across pages
                    {"name": "publishers/google/models/gemini-2.5-pro"},
                    {"name": "publishers/google/models/gemini-2.5-flash-transcribe"},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        rows = fetch_models(
            "google-vertex",
            api_key="ya29.test-token",
            is_oauth=True,
            account_id="minerva-vertex",
            client=http,
        )

    assert rows is not None
    assert [row.id for row in rows] == ["gemini-4-neon", "gemini-3.8-flash", "gemini-2.5-pro"]
    assert len(seen) == 2
    assert seen[0].url.path == "/v1beta1/publishers/google/models"
    assert seen[0].url.params["pageSize"] == "100"
    assert seen[1].url.params["pageToken"] == "page-2"
    for req in seen:
        assert req.headers["authorization"] == "Bearer ya29.test-token"
        assert req.headers["x-goog-user-project"] == "minerva-vertex"
        assert "x-goog-api-key" not in req.headers


def test_fetch_vertex_api_key_401_falls_back_to_static_without_failure_banner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Express/restricted API keys that 401 on publisherModels degrade cleanly to static rows."""
    from local_operator.model.discovery import available_models
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import CatalogueEntry, ProviderController

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(401, json={"error": {"message": "API keys not supported"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        rows, status = available_models(
            "google-vertex",
            api_key="vx-express-key",
            is_oauth=False,
            cache_dir=tmp_path,
            client=http,
        )

    assert status == "static"
    assert len(rows) > 0
    assert any(row.id == "gemini-3.8-flash" for row in rows)
    assert len(seen) == 1
    assert seen[0].headers["x-goog-api-key"] == "vx-express-key"
    assert "authorization" not in seen[0].headers

    # Because google-vertex still contributes static rows, catalogue_failures
    # must NOT surface a spurious "Model listing unavailable" banner.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    store.upsert_credential("google-vertex", {"type": "api_key", "key": "vx-express-key"})
    controller = ProviderController(store, tmp_path)
    try:
        entries = [
            CatalogueEntry(
                provider="google-vertex",
                model_id="gemini-3.8-flash",
                label="Gemini 3.8 Flash",
                context_window=1_048_576,
                input_price=0.75,
                output_price=3.75,
                connected=True,
            )
        ]
        assert controller.catalogue_failures(entries, {"google-vertex": "static"}) == {}
    finally:
        controller.close()
        store.close()


@pytest.mark.asyncio
async def test_vertex_listing_and_catalogue_credentials_pass_project_id(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``listing_credential`` and ``_catalogue_credential`` pass ``project_id`` in slot 3."""
    import time

    from local_operator.model.configure import _catalogue_credential
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    store.upsert_credential(
        "google-vertex",
        {
            "type": "oauth",
            "access": "ya29.vertex-token",
            "refresh": "1//refresh-token",
            "expires": int(time.time() * 1000) + 3_600_000,
            "email": "damian@gominerva.com",
            "account_id": "damian@gominerva.com",
            "org_name": "minerva-vertex",
            "project_id": "minerva-vertex",
            "location": "global",
            "auth_mode": "oauth",
        },
    )
    controller = ProviderController(store, tmp_path)
    try:
        token, is_oauth, project_id = await controller.listing_credential(
            "google-vertex", read_only=True
        )
        assert token == "ya29.vertex-token"
        assert is_oauth is True
        assert project_id == "minerva-vertex"

        cat_token, cat_is_oauth, cat_project = _catalogue_credential("google-vertex", base=tmp_path)
        assert cat_token == "ya29.vertex-token"
        assert cat_is_oauth is True
        assert cat_project == "minerva-vertex"
    finally:
        controller.close()
        store.close()


def test_vertex_models_dev_projection_and_long_context_cost_for_usage() -> None:
    """models.dev ``google-vertex`` ``context_over_200k`` projects through to ``cost_for_usage``."""
    from local_operator.harness.types import Usage
    from local_operator.model.configure import (
        _cache_tokens_are_inside_input,
        cost_for_usage,
        info_from_discovered_model,
    )
    from local_operator.model.prices import PRICE_CATALOGUE_CAPTURE, price_row, project
    from local_operator.model.registry import ModelInfo

    assert PRICE_CATALOGUE_CAPTURE >= 4
    assert _cache_tokens_are_inside_input("google-vertex") is True

    body = {
        "google-vertex": {
            "id": "google-vertex",
            "name": "Vertex",
            "models": {
                "gemini-2.5-pro": {
                    "id": "gemini-2.5-pro",
                    "name": "Gemini 2.5 Pro",
                    "limit": {"context": 1_048_576, "output": 65_536},
                    "modalities": {"input": ["text", "image", "audio"], "output": ["text"]},
                    "cost": {
                        "input": 1.25,
                        "output": 10.0,
                        "cache_read": 0.125,
                        "context_over_200k": {
                            "input": 2.5,
                            "output": 15.0,
                            "cache_read": 0.25,
                        },
                    },
                },
                "gemini-4-neon": {
                    "id": "gemini-4-neon",
                    "name": "Gemini 4 Neon",
                    "limit": {"context": 1_048_576, "output": 65_536},
                    "modalities": {"input": ["text", "image", "audio"], "output": ["text"]},
                    "cost": {"input": 3.0, "output": 15.0, "cache_read": 0.15},
                },
            },
        }
    }
    projected = project(body, '"etag-v4"')
    assert projected["capture"] == PRICE_CATALOGUE_CAPTURE
    assert "google-vertex" in projected["providers"]

    row = price_row(
        "google-vertex",
        "gemini-2.5-pro",
        models_dev=projected["providers"],
        openrouter=[],
    )
    assert row is not None
    assert row.name == "Gemini 2.5 Pro"
    assert row.long_context_threshold == 200_000
    assert row.long_context_input_price == 2.5
    assert row.long_context_output_price == 15.0
    assert row.long_context_cache_read_price == 0.25

    info = info_from_discovered_model(
        "google-vertex",
        "gemini-2.5-pro",
        row,
        ModelInfo(id="gemini-2.5-pro", name="gemini-2.5-pro", description=""),
    )

    # <= 200k prompt: standard tier. Cached tokens are inside input_tokens, so
    # plain = 200_000 - 100_000 = 100_000 (never double-billed!).
    short_usage = Usage(
        input_tokens=200_000,
        output_tokens=10_000,
        cache_read_tokens=100_000,
        reasoning_tokens=4_000,
    )
    expected_short = (100_000 * 1.25 + 100_000 * 0.125 + 10_000 * 10.0) / 1_000_000
    assert cost_for_usage("google-vertex", info, short_usage) == pytest.approx(expected_short)

    # > 200k prompt (evaluated BEFORE subtracting cache_read_tokens): entire
    # request bills at >200k tier (plain 150k @ $2.50, cache_read 100k @ $0.25,
    # output 10k including thoughts @ $15.00).
    long_usage = Usage(
        input_tokens=250_000,
        output_tokens=10_000,
        cache_read_tokens=100_000,
        reasoning_tokens=4_000,
    )
    expected_long = (150_000 * 2.5 + 100_000 * 0.25 + 10_000 * 15.0) / 1_000_000
    assert cost_for_usage("google-vertex", info, long_usage) == pytest.approx(expected_long)


@pytest.mark.asyncio
async def test_vertex_thought_only_usage_chunk_marks_tokens_reported() -> None:
    """Vertex usage chunks with thoughtsTokenCount and no candidatesTokenCount stay reported."""
    payload = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"thought": True, "text": "thinking..."}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 6, "thoughtsTokenCount": 99, "totalTokenCount": 105},
    }
    body = f"data: {json.dumps(payload)}\n\n".encode("utf-8")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GoogleClient(base_url=VERTEX_API_URL, vertex=True, http_client=http)
        events = [e async for e in client.stream(_vertex_request(), api_key="vx-api-key")]

    usage_events = [u for e in events if (u := getattr(e, "usage", None)) is not None]
    assert usage_events
    usage = usage_events[-1]
    assert usage.input_tokens == 6
    assert usage.output_tokens == 99
    assert usage.reasoning_tokens == 99
    assert usage._token_counts_reported is True


def test_vertex_catalogue_search_ranks_vertex_and_gemini_4_neon() -> None:
    """Searching ``vertex`` and ``gemini-4-neon`` finds and ranks ``google-vertex/*`` rows."""
    from local_operator.model.ranking import ModelRow, rank_rows

    rows = [
        ModelRow(
            provider="google-vertex",
            model_id="gemini-4-neon",
            label="Gemini 4 Neon",
            listing_name="Gemini 4 Neon",
            context_window=1_048_576,
            input_price=3.0,
            output_price=15.0,
        ),
        ModelRow(
            provider="google-vertex",
            model_id="gemini-3.8-flash",
            label="Gemini 3.8 Flash",
            listing_name="Gemini 3.8 Flash",
            context_window=1_048_576,
            input_price=0.75,
            output_price=3.75,
        ),
        ModelRow(
            provider="anthropic",
            model_id="claude-opus-5",
            label="Claude Opus 5",
            listing_name="Claude Opus 5",
            context_window=1_000_000,
            input_price=5.0,
            output_price=25.0,
        ),
    ]

    vertex_matches = rank_rows(rows, "vertex")
    assert [r.selector for r in vertex_matches] == [
        "google-vertex/gemini-4-neon",
        "google-vertex/gemini-3.8-flash",
    ]
    neon_matches = rank_rows(rows, "gemini-4-neon")
    assert [r.selector for r in neon_matches] == ["google-vertex/gemini-4-neon"]
    flash_matches = rank_rows(rows, "gemini 3.8")
    assert [r.selector for r in flash_matches] == ["google-vertex/gemini-3.8-flash"]


# --- Round 1 remediation regressions -------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        {"project_id": "minerva-vertex", "location": "attacker.com/x#"},
        {"project_id": "../other", "location": "global"},
        {"project_id": "proj/../../x", "location": "us-central1"},
        {"project_id": "minerva-vertex", "location": "us-central1.evil.example"},
    ],
)
async def test_vertex_client_rejects_host_or_path_injection(raw: dict[str, str]) -> None:
    """MAJOR 2: stored project/location can never redirect the bearer token."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_sse_body())

    access = OAuthAccess(
        access_token="ya29.secret",
        credential_id=9,
        kind="oauth",
        raw={**raw, "auth_mode": "oauth"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GoogleClient(base_url=VERTEX_API_URL, vertex=True, http_client=http)
        with pytest.raises(ProviderError) as exc_info:
            async for _ in client.stream(_vertex_request(), api_key=None, oauth_access=access):
                pass
    assert exc_info.value.status == 400
    assert exc_info.value.retryable is False
    assert seen == []


@pytest.mark.asyncio
async def test_vertex_client_percent_encodes_model_id_in_both_auth_modes() -> None:
    """MAJOR 2: a model id cannot inject a query or path into the Vertex URL."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_sse_body())

    access = OAuthAccess(
        access_token="ya29.secret",
        credential_id=10,
        kind="oauth",
        raw={"project_id": "minerva-vertex", "location": "us-central1"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GoogleClient(base_url=VERTEX_API_URL, vertex=True, http_client=http)
        for key, oauth in ((None, access), ("AQ.key", None)):
            _ = [e async for e in client.stream(_vertex_request("m?x=1/../y"), key, oauth)]

    assert len(seen) == 2
    assert seen[0].url.host == "us-central1-aiplatform.googleapis.com"
    assert seen[1].url.host == "aiplatform.googleapis.com"
    for request in seen:
        assert "m%3Fx%3D1%2F..%2Fy:streamGenerateContent" in str(request.url)
        assert dict(request.url.params) == {"alt": "sse"}


def _anthropic_tiered_info() -> Any:
    from local_operator.model.registry import ModelInfo

    return ModelInfo(
        id="claude-tiered",
        name="Claude Tiered",
        description="",
        input_price=3.0,
        output_price=15.0,
        cache_reads_price=0.3,
        cache_writes_price=3.75,
        long_context_threshold=200_000,
        long_context_input_price=6.0,
        long_context_output_price=22.5,
        long_context_cache_reads_price=0.6,
        long_context_cache_writes_price=7.5,
    )


def test_anthropic_cached_prompt_crossing_200k_bills_long_context_tier() -> None:
    """MAJOR 3: Anthropic reports cache buckets disjoint from input_tokens, so the
    tier threshold counts input + read + written."""
    from local_operator.harness.types import Usage
    from local_operator.model.configure import cost_for_usage

    usage = Usage(input_tokens=10_000, output_tokens=1_000, cache_read_tokens=200_000)
    expected = (10_000 * 6.0 + 200_000 * 0.6 + 1_000 * 22.5) / 1_000_000
    assert cost_for_usage("anthropic", _anthropic_tiered_info(), usage) == pytest.approx(expected)


def test_anthropic_1h_cache_write_surcharge_uses_long_context_rates() -> None:
    """MAJOR 3: the 1h write surcharge is 2x the TIER input rate on a >200k prompt."""
    from local_operator.harness.types import Usage
    from local_operator.model.configure import cost_for_usage

    usage = Usage(
        input_tokens=10_000,
        output_tokens=0,
        cache_read_tokens=200_000,
        cache_write_tokens=10_000,
        cache_write_1h_tokens=10_000,
    )
    base = (10_000 * 6.0 + 200_000 * 0.6 + 10_000 * 7.5) / 1_000_000
    surcharge = 10_000 * (2 * 6.0 - 7.5) / 1_000_000
    assert cost_for_usage("anthropic", _anthropic_tiered_info(), usage) == pytest.approx(
        base + surcharge
    )


def test_vertex_cached_prompt_crossing_200k_bills_long_context_tier() -> None:
    """MAJOR 3: Google/Vertex count cache reads INSIDE input_tokens; the threshold
    uses raw input, and plain = input - read."""
    from local_operator.harness.types import Usage
    from local_operator.model.configure import cost_for_usage
    from local_operator.model.registry import ModelInfo

    info = ModelInfo(
        id="gemini-tiered",
        name="Gemini Tiered",
        description="",
        input_price=1.25,
        output_price=10.0,
        cache_reads_price=0.125,
        long_context_threshold=200_000,
        long_context_input_price=2.5,
        long_context_output_price=15.0,
        long_context_cache_reads_price=0.25,
    )
    usage = Usage(input_tokens=210_000, output_tokens=0, cache_read_tokens=150_000)
    expected = (60_000 * 2.5 + 150_000 * 0.25) / 1_000_000
    assert cost_for_usage("google-vertex", info, usage) == pytest.approx(expected)


@pytest.mark.parametrize("provider", ["google", "google-vertex"])
def test_static_gemini_25_pro_row_carries_long_context_tier(provider: str) -> None:
    """MINOR 2: a fully described static row never enters models.dev enrichment,
    so the >200k tier must be on the row itself."""
    from local_operator.harness.types import Usage
    from local_operator.model.configure import cost_for_usage
    from local_operator.model.registry import get_model_info

    info = get_model_info(provider, "gemini-2.5-pro-preview-05-06")
    assert info.long_context_threshold == 200_000
    usage = Usage(input_tokens=250_000, output_tokens=1_000)
    expected = (250_000 * 2.5 + 1_000 * 15.0) / 1_000_000
    assert cost_for_usage(provider, info, usage) == pytest.approx(expected)
