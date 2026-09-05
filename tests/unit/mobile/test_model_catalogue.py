"""A cold phone picker must list OAuth aggregators through real discovery.

Only the HTTP transport is replaced. AuthStore, expiry refresh, catalogue disk
I/O, provider parser, mobile authentication, and response projection are real.
The suite's fresh HOME also isolates the catalogue (config-dir alone does not).
"""

import contextlib
import json
import threading
import time

import httpx
import pytest
from starlette.testclient import TestClient

from local_operator.credentials import CredentialManager
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.model.catalogue import default_cache_dir
from local_operator.paths import config_dir
from local_operator.providers.auth_store import AuthStore


@pytest.mark.parametrize("expired", [False, True])
def test_cold_radient_oauth_models_refresh_parse_and_cache_off_loop(monkeypatch, expired):
    with contextlib.closing(AuthStore()) as store:
        credential = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "fixture-access",
                "refresh": "fixture-refresh",
                "expires": 0 if expired else int(time.time() * 1000) + 3_600_000,
            },
        )
    assert not default_cache_dir().exists()
    # The daemon may inherit other tooling's env. It must use this stored login
    # and never expose those unrelated providers or override it with an env key.
    monkeypatch.setenv("RADIENT_API_KEY", "ambient-radient-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-openrouter-key")
    requests = []
    refreshes = []
    loop_threads = set()

    def models(_transport, request):
        assert threading.get_ident() not in loop_threads
        assert str(request.url) == "https://api.radienthq.com/v1/models"
        expected = "fixture-refreshed" if expired else "fixture-access"
        assert request.headers["authorization"] == f"Bearer {expected}"
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "anthropic/mobile-fixture", "name": "Mobile fixture"},
                    {"id": "auto", "name": "Automatic"},
                ]
            },
        )

    async def refresh(_transport, request):
        assert threading.get_ident() not in loop_threads
        assert str(request.url) == "https://api.radienthq.com/v1/auth/oauth/token"
        assert json.loads(request.content)["refresh_token"] == "fixture-refresh"
        refreshes.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "fixture-refreshed",
                "refresh_token": "fixture-rotated",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refresh)
    relay = build_app(MobileDaemon(password="fixture-password", dial_registrants=False))

    async def app(scope, receive, send):
        loop_threads.add(threading.get_ident())
        await relay(scope, receive, send)

    with TestClient(app, follow_redirects=False) as client:
        assert client.get("/api/models").status_code == 401
        assert not requests
        client.post("/login", data={"password": "fixture-password"})
        first = client.get("/api/models")
        assert first.status_code == 200, first.json()
        assert first.json() == {
            "models": [
                {
                    "selector": "radient/anthropic/mobile-fixture",
                    "provider": "radient",
                    "model_id": "anthropic/mobile-fixture",
                    "name": "Mobile fixture",
                },
                {
                    "selector": "radient/auto",
                    "provider": "radient",
                    "model_id": "auto",
                    "name": "Automatic",
                },
            ]
        }
        assert client.get("/api/models").json() == first.json()
    assert len(requests) == 1  # Second picker open reads the real fresh cache.
    assert len(refreshes) == int(expired)
    assert "fixture-access" not in first.text
    assert "fixture-refreshed" not in first.text
    with contextlib.closing(AuthStore()) as store:
        row = store.get_credential(credential.id)
        assert row is not None
        assert row.disabled_cause is None
        if expired:
            assert row.data["refresh"] == "fixture-rotated"


@pytest.mark.parametrize("storage", ["login", "legacy"])
def test_openrouter_stored_key_lists_real_catalogue(monkeypatch, storage):
    if storage == "login":
        with contextlib.closing(AuthStore()) as store:
            store.upsert_credential("openrouter", {"type": "api_key", "key": "fixture-key"})
    else:
        CredentialManager(config_dir()).set_credential("OPENROUTER_API_KEY", "fixture-key")

    def models(_transport, request):
        assert str(request.url) == "https://openrouter.ai/api/v1/models"
        assert request.headers["authorization"] == "Bearer fixture-key"
        return httpx.Response(200, json={"data": [{"id": "vendor/model"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")
    assert response.status_code == 200
    assert [row["selector"] for row in response.json()["models"]] == ["openrouter/vendor/model"]


def test_failed_cold_catalogue_is_not_reported_as_empty_inventory(monkeypatch):
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("radient", {"type": "oauth", "access": "fixture-secret"})
    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        lambda *_: httpx.Response(503, text="upstream diagnostic with fixture-secret"),
    )
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")
    assert response.status_code == 502
    assert "retry or log in again" in response.json()["error"]
    assert "fixture-secret" not in response.text


def test_no_login_or_disabled_login_never_fetches_public_catalogue(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda *args: calls.append(args))
    with contextlib.closing(AuthStore()) as store:
        row = store.upsert_credential("radient", {"type": "oauth", "access": "disabled"})
        store.disable_credential(row.id, "test")
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        assert client.get("/api/models").json() == {"models": []}
    assert calls == []
