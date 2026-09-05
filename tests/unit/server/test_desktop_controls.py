"""HTTP boundary + real settings/auth stores, without third-party credentials."""

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport, AsyncClient

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.providers import registry
from local_operator.server.app import desktop_validation_error, managed_desktop_boundary
from local_operator.server.routes import (
    auth,
    capabilities,
    config,
    credentials,
    settings,
)

TOKEN = "desktop-contract-test-token"
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(capabilities.router)
    app.include_router(auth.router)
    app.include_router(settings.router)
    app.include_router(config.router)
    app.include_router(credentials.router)
    app.middleware("http")(managed_desktop_boundary)
    app.exception_handler(RequestValidationError)(desktop_validation_error)
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.credential_manager = CredentialManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


async def wait_for_state(client: AsyncClient, operation_id: str, *states: str) -> dict[str, Any]:
    # The host publishes state on event-loop turns. Do not turn this into a
    # sleep calibrated to one developer's machine; the bound only catches hangs.
    for _ in range(1000):
        data = (await client.get(f"/v1/auth/operations/{operation_id}")).json()["result"]
        if data["state"] in states:
            return data
        await asyncio.sleep(0)
    pytest.fail(f"Login did not reach {states}")


async def test_desktop_token_origin_and_unconfigured_fail_closed(desktop, monkeypatch):
    client, _ = desktop
    assert (await client.get("/v1/settings", headers={"Authorization": ""})).status_code == 401
    assert (
        await client.get("/v1/settings", headers={"Origin": "https://evil.example"})
    ).status_code == 403
    assert (await client.get("/v1/settings", headers={"Origin": "null"})).status_code == 403
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "http://localhost:5187")
    assert (
        await client.get("/v1/settings", headers={"Origin": "http://localhost:5187"})
    ).status_code == 200
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN")
    assert (await client.get("/v1/settings")).status_code == 503
    public = (await client.get("/v1/capabilities")).json()["result"]
    assert public["desktop_available"] is False
    assert TOKEN not in str(public)


async def test_managed_legacy_controls_require_token_but_unmanaged_remains_compatible(
    desktop, monkeypatch
):
    client, _ = desktop
    for path in ("/v1/config", "/v1/credentials", "/v1/config/system-prompt"):
        assert (await client.get(path, headers={"Authorization": ""})).status_code == 401
        assert (await client.patch(path, json={}, headers={"Authorization": ""})).status_code == 401
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN")
    assert (await client.get("/v1/config", headers={"Authorization": ""})).status_code == 200
    assert (
        await client.patch(
            "/v1/credentials",
            json={"key": "EXAMPLE_KEY", "value": "example"},
            headers={"Authorization": ""},
        )
    ).status_code == 200


async def test_settings_census_typed_writes_reset_and_secret_exclusion(desktop):
    client, app = desktop
    manager = app.state.config_manager
    manager.set_config_value("private_secret", "never-serialize-me")
    data = (await client.get("/v1/settings")).json()["result"]
    assert {row["key"] for row in data["settings"]} == {s.key for s in settings_io.SETTINGS}
    assert "never-serialize-me" not in str(data)
    manager.set_config_value(
        "web_search", {"searxng_endpoint": "https://user:private-inline@example.org/?key=hidden"}
    )
    protected = await client.get("/v1/settings")
    assert protected.headers["cache-control"] == "no-store"
    assert "private-inline" not in protected.text
    assert "hidden" not in protected.text
    assert next(
        row
        for row in protected.json()["result"]["settings"]
        if row["key"] == "web_search.searxng_endpoint"
    )["redacted"]
    assert (
        await client.patch(
            "/v1/settings/web_search.searxng_endpoint",
            json={"value": "https://user:private@example.org"},
        )
    ).status_code == 422
    assert (
        await client.patch("/v1/settings/private_secret", json={"value": "changed"})
    ).status_code == 404
    key = "providers.anthropic.cache_ttl_1h_min_context_tokens"
    assert (await client.patch(f"/v1/settings/{key}", json={"value": 1.5})).status_code == 422
    assert (await client.patch(f"/v1/settings/{key}", json={"value": True})).status_code == 422
    response = await client.patch(f"/v1/settings/{key}", json={"value": 42})
    assert response.status_code == 200, response.text
    assert response.json()["result"]["value"] == 42
    fresh = ConfigManager(manager.config_dir)
    setting = settings_io.resolve_key(key)
    assert setting is not None
    assert settings_io.read_setting(fresh, setting) == 42
    assert (await client.post(f"/v1/settings/{key}/reset")).json()["result"]["is_default"]
    dotted = next(
        s for s in settings_io.SETTINGS if s.is_flat_dotted and s.kind is settings_io.Kind.BOOL
    )
    assert (
        await client.patch(f"/v1/settings/{dotted.key}", json={"value": not dotted.default})
    ).status_code == 200
    fresh.reload()
    assert fresh.config.values[dotted.key] is not dotted.default
    assert fresh.config.values["private_secret"] == "never-serialize-me"


async def test_settings_cascade_preserves_concurrent_siblings(desktop):
    client, app = desktop
    manager = app.state.config_manager
    settings_io.write_chains(manager, {"primary": ["openai/gpt-5"]})
    base = settings_io.read_chains(manager)
    other = ConfigManager(manager.config_dir)
    settings_io.write_chains(other, {**base, "other": ["anthropic/claude-sonnet-4"]}, base=base)
    response = await client.patch(
        "/v1/settings/retry.fallbackChains",
        json={
            "value": {"primary": ["openai/gpt-5", "openrouter/openai/gpt-5"]},
            "base": base,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["value"]["other"] == ["anthropic/claude-sonnet-4"]
    assert (
        await client.patch("/v1/settings/retry.fallbackChains", json={"value": {}})
    ).status_code == 422


async def test_unreadable_settings_are_not_replaced_by_a_get(desktop):
    client, app = desktop
    path = app.state.config_manager.config_file
    path.write_text("\tinvalid: yaml\n")
    before = path.read_bytes()
    assert (await client.get("/v1/settings")).status_code == 409
    assert path.read_bytes() == before


async def test_provider_census_alias_storage_and_redacted_keys(desktop):
    client, app = desktop
    rows = (await client.get("/v1/auth/providers")).json()["result"]["providers"]
    assert {row["id"] for row in rows} == {
        p.id
        for p in registry.PROVIDER_REGISTRY
        if registry.credential_provider_id(p.id) == p.id and p.wire != "mock"
    }
    assert {method["id"] for row in rows for method in row["auth_methods"]} == {
        p.id for p in registry.PROVIDER_REGISTRY if p.login is not None
    }
    assert next(row for row in rows if row["id"] == "radient")["login_kind"] == "browser"
    assert next(row for row in rows if row["id"] == "openrouter")["login_kind"] == "api_key"
    assert (
        next(
            method
            for row in rows
            for method in row["auth_methods"]
            if method["id"] == "alibaba-token-plan-oauth"
        )["kind"]
        == "device"
    )
    secret = "contract-secret-never-return"
    response = await client.put("/v1/auth/providers/xai-oauth/key", json={"value": secret})
    assert response.status_code == 200, response.text
    assert secret not in response.text
    stored = app.state.desktop_auth.store.list_credentials("xai")
    assert len(stored) == 1
    assert stored[0].data["key"] == secret
    assert secret not in (await client.get("/v1/auth/providers")).text
    bad = await client.put("/v1/auth/providers/xai/key", json={"value": {"secret": secret}})
    assert bad.status_code == 422
    assert secret not in bad.text


async def test_actual_registry_key_login_input_cancel_and_persistence(desktop):
    client, app = desktop
    started = await client.post("/v1/auth/login", json={"provider": "openrouter"})
    assert started.status_code == 200, started.text
    operation_id = started.json()["result"]["id"]
    awaiting = await wait_for_state(client, operation_id, "input_required")
    assert awaiting["input_required"]
    assert (await client.post("/v1/auth/login", json={"provider": "radient"})).status_code == 409
    secret = "registry-login-secret"
    response = await client.post(
        f"/v1/auth/operations/{operation_id}/input",
        json={"value": secret, "prompt_id": awaiting["prompt_id"]},
    )
    assert response.status_code == 200
    done = await wait_for_state(client, operation_id, "succeeded")
    assert secret not in str(done)
    assert done["auth_url"] is None
    assert app.state.desktop_auth.store.list_credentials("openrouter")[0].data["key"] == secret
    assert (
        await client.post(
            f"/v1/auth/operations/{operation_id}/input",
            json={"value": secret, "prompt_id": awaiting["prompt_id"]},
        )
    ).status_code == 409
    again = (await client.post("/v1/auth/login", json={"provider": "openrouter"})).json()["result"][
        "id"
    ]
    await wait_for_state(client, again, "input_required")
    assert (await client.delete(f"/v1/auth/operations/{again}")).json()["result"][
        "state"
    ] == "cancelled"
    assert (await client.delete("/v1/auth/providers/openrouter/credentials")).status_code == 200
    assert not app.state.desktop_auth.store.list_credentials("openrouter")


async def test_oauth_failure_is_redacted_and_browser_opener_is_per_flow(desktop, monkeypatch):
    client, _ = desktop
    observed = []

    async def login(callbacks, *, open_browser):
        observed.append(open_browser)
        await asyncio.sleep(0)
        raise RuntimeError("raw-provider-access-token-do-not-return")

    definition = registry.get_provider_definition("radient")
    assert definition is not None
    monkeypatch.setitem(registry._BY_ID, "radient", dataclasses.replace(definition, login=login))
    operation_id = (await client.post("/v1/auth/login", json={"provider": "radient"})).json()[
        "result"
    ]["id"]
    done = await wait_for_state(client, operation_id, "failed")
    assert observed
    assert "raw-provider-access-token" not in str(done)
