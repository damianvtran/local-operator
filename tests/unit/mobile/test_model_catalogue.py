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
        # The payload now carries what the picker ranks on, and the rows arrive
        # in the picker's own order rather than the registry's. ``name`` falls
        # back to the id for an aggregator because ``naming`` refuses a
        # reseller's listing name — the TUI paints the same row the same way.
        assert first.json() == {
            "models": [
                {
                    "selector": "radient/anthropic/mobile-fixture",
                    "provider": "radient",
                    "model_id": "anthropic/mobile-fixture",
                    "name": "anthropic/mobile-fixture",
                    "label": "radient/anthropic/mobile-fixture",
                    "connected": True,
                    "aggregated": True,
                    "routed": False,
                    "context_window": 0,
                    "input_price": -1.0,
                    "output_price": -1.0,
                },
                {
                    "selector": "radient/auto",
                    "provider": "radient",
                    "model_id": "auto",
                    "name": "auto",
                    "label": "radient/auto",
                    "connected": True,
                    "aggregated": True,
                    "routed": True,
                    "context_window": 0,
                    "input_price": -1.0,
                    "output_price": -1.0,
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


def test_provider_outside_the_stale_enumeration_is_listed_when_persisted(monkeypatch):
    """A login the OLD enumeration could not name must reach the phone's sheet.

    ``alibaba-token-plan`` is in ``providers.registry.PROVIDER_REGISTRY`` and not
    in ``model.registry.SupportedHostingProviders``, which is what this endpoint
    used to walk — so a fully logged-in owner saw none of its models on the phone
    while ``/model`` offered them on the desktop. It stands in here for the five
    ids in that gap (``openai-device``, ``radient-key``, ``xai-oauth``,
    ``zai-oauth`` and the two alibaba flavours).
    """
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("alibaba-token-plan", {"type": "api_key", "key": "fixture-key"})

    def models(_transport, request):
        assert request.headers["authorization"] == "Bearer fixture-key"
        return httpx.Response(200, json={"data": [{"id": "qwen3-max", "name": "Qwen3 Max"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")
    assert response.status_code == 200, response.json()
    rows = response.json()["models"]
    assert "alibaba-token-plan/qwen3-max" in {row["selector"] for row in rows}


def test_rows_arrive_ranked_direct_first_with_aggregators_last(monkeypatch):
    """The bug this endpoint was rebuilt for: the phone got REGISTRY order.

    Two providers hold the same model. The direct route must lead — it is the
    one the owner meant after logging in to xAI — and every aggregated row must
    sit behind every direct one, which is precisely what ``rank_rows``
    guarantees and what registry order destroyed by putting ~445 Radient rows
    ahead of the first direct provider.

    ``xai`` rather than ``anthropic`` as the direct route on purpose: it sorts
    AFTER ``openrouter`` alphabetically, so this cannot pass on the
    provider-name rung alone if the aggregator tier were lost.
    """
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("xai", {"type": "api_key", "key": "xai-key"})
        store.upsert_credential("openrouter", {"type": "api_key", "key": "openrouter-key"})

    def models(_transport, request):
        if request.url.host == "api.x.ai":
            return httpx.Response(200, json={"data": [{"id": "grok-4"}]})
        return httpx.Response(200, json={"data": [{"id": "x-ai/grok-4"}, {"id": "vendor/other"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        rows = client.get("/api/models").json()["models"]

    selectors = [row["selector"] for row in rows]
    # The same model by two routes: the DIRECT one leads.
    assert selectors.index("xai/grok-4") < selectors.index("openrouter/x-ai/grok-4")
    assert rows[0]["provider"] == "xai"
    assert rows[0]["aggregated"] is False
    aggregated = [index for index, row in enumerate(rows) if row["aggregated"]]
    direct = [index for index, row in enumerate(rows) if not row["aggregated"]]
    assert min(aggregated) > max(direct)


def test_ambient_env_key_never_joins_a_remote_picker(monkeypatch):
    """A service manager's environment is not the phone owner's consent.

    The daemon is typically launched by launchd/systemd, whose environment the
    phone's user never chose and cannot inspect. ``usable_providers`` counts an
    env key (correctly, for a local turn); ``persisted_providers`` does not, so
    an inherited ``ANTHROPIC_API_KEY`` must neither appear in the sheet nor
    cause a single request to Anthropic.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key")
    with contextlib.closing(AuthStore()) as store:
        store.upsert_credential("radient", {"type": "oauth", "access": "fixture-access"})
    hosts = []

    def models(_transport, request):
        hosts.append(request.url.host)
        return httpx.Response(200, json={"data": [{"id": "vendor/model"}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", models)
    with TestClient(build_app(MobileDaemon(password="pw", dial_registrants=False))) as client:
        client.post("/login", data={"password": "pw"})
        response = client.get("/api/models")

    assert response.status_code == 200, response.json()
    assert {row["provider"] for row in response.json()["models"]} == {"radient"}
    assert "api.anthropic.com" not in hosts
    assert "ambient-anthropic-key" not in response.text


def test_admitted_empty_set_fetches_nothing_at_all(monkeypatch):
    """``providers=set()`` is honoured literally, not read as "everything".

    An owner who has logged in to nothing is a real state with a real answer:
    no listing is authorized, so no provider may be contacted. A narrowing that
    fell back to the whole registry on an empty collection would turn the
    strictest case into the loosest one.
    """
    import asyncio

    from local_operator.credentials import CredentialManager
    from local_operator.providers.controller import ProviderController

    calls = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda *args: calls.append(args))
    with contextlib.closing(AuthStore()) as store:
        controller = ProviderController(store, CredentialManager(config_dir=config_dir()))
        entries, statuses = asyncio.run(controller.live_catalogue(providers=set()))
    assert entries == []
    assert statuses == {}
    assert calls == []
