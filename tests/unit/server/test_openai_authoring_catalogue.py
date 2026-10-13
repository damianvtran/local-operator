"""The authoring HTTP list must share OpenAI's native discovery contract.

These calls use the shipped router and parser, not a parallel fixture catalogue.
Only account identity and HTTP transport are synthetic; no login, refresh token,
secret store, live provider, or installed desktop authorization is involved.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_operator.model import discovery
from local_operator.model.registry import openai_models
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.controller import PICKER_TTL_S, ProviderController
from local_operator.server.dependencies import (
    get_config_manager,
    get_env_config,
    get_provider_auth_store,
)
from local_operator.server.routes import models


@pytest.fixture
def native_catalogue(monkeypatch, tmp_path):
    # Both documents and config are isolated even when this file runs alone.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr("local_operator.providers.controller.resolve_env_key", lambda _: None)
    state = SimpleNamespace(
        body={"models": [{"slug": "gpt-6.1-sol", "display_name": "GPT-6.1 Sol"}]},
        status=200,
        requests=[],
        store=SimpleNamespace(
            get_oauth_access=AsyncMock(
                return_value=OAuthAccess("fixture", 1, account_id="account-one")
            ),
            get_api_key=AsyncMock(return_value=None),
        ),
    )

    def transport(request):
        # There is no fallback socket: every provider call must hit this seam.
        state.requests.append(request)
        assert request.url.host in {"chatgpt.com", "api.openai.com"}
        return httpx.Response(state.status, json=state.body)

    real_client = httpx.Client
    monkeypatch.setattr(
        discovery.httpx,
        "Client",
        lambda *args, **kwargs: real_client(
            transport=httpx.MockTransport(transport), timeout=kwargs.get("timeout", 2)
        ),
    )
    app = FastAPI()
    app.include_router(models.router)
    state.config = SimpleNamespace(
        config_dir=tmp_path,
        values={
            "hosting": "openai",
            "model_name": "gpt-6.1-sol",
            "effort": "high",
            "default": "unchanged",
            "failover": ["synthetic-fallback"],
        },
    )
    app.dependency_overrides[get_config_manager] = lambda: state.config
    app.dependency_overrides[get_env_config] = lambda: SimpleNamespace()
    app.dependency_overrides[get_provider_auth_store] = lambda: state.store
    state.client = TestClient(app)
    yield state
    state.client.close()


def listed(state):
    response = state.client.get("/v1/models?provider=openai")
    assert response.status_code == 200, response.text
    return response.json()["result"]["models"]


def test_native_subscription_model_is_not_a_static_or_custom_invention(native_catalogue):
    state = native_catalogue
    before = dict(state.config.values)
    rows = listed(state)
    assert state.config.values == before
    assert [row["id"] for row in rows] == ["gpt-6.1-sol"]
    row = rows[0]
    assert row["name"] == "GPT-6.1 Sol"
    assert row["provider"] == "openai"
    # The native row said nothing about pricing, windows or these capabilities.
    for field in ("supports_tools", "reasoning", "supports_images", "context_window"):
        assert row["info"][field] is None
    assert row["info"]["input_price"] == row["info"]["output_price"] == 0
    request = state.requests[0]
    assert request.url.host == "chatgpt.com"
    assert request.url.path == "/backend-api/codex/models"
    assert request.headers["ChatGPT-Account-ID"] == "account-one"
    state.store.get_oauth_access.assert_awaited_once_with("openai", read_only=True)
    state.store.get_api_key.assert_not_awaited()


def test_bundled_metadata_remains_the_fallback_not_a_new_price_claim(native_catalogue):
    state = native_catalogue
    model_id = "gpt-4o"
    state.body = {"models": [{"slug": model_id, "input_modalities": ["text"]}]}
    row = listed(state)[0]
    assert row["info"]["input_price"] == openai_models[model_id].input_price
    assert row["info"]["output_price"] == openai_models[model_id].output_price
    assert row["info"]["supports_images"] is False


def test_api_and_oauth_inventory_never_share_a_cache(native_catalogue):
    state = native_catalogue
    assert [row["id"] for row in listed(state)] == ["gpt-6.1-sol"]
    state.store.get_oauth_access.return_value = OAuthAccess("fixture", 2, kind="api_key")
    state.body = {"data": [{"id": "synthetic-api-model"}]}
    ids = {row["id"] for row in listed(state)}
    assert "synthetic-api-model" in ids
    assert "gpt-6.1-sol" not in ids
    assert state.requests[-1].url.host == "api.openai.com"
    assert state.requests[-1].url.path == "/v1/models"
    assert "ChatGPT-Account-ID" not in state.requests[-1].headers
    # Returning to the first account uses its own document, not the API rows.
    state.store.get_oauth_access.return_value = OAuthAccess(
        "fixture", 1, account_id="account-one"
    )
    assert [row["id"] for row in listed(state)] == ["gpt-6.1-sol"]
    assert len(state.requests) == 2


def test_oauth_account_scope_never_reuses_another_account(native_catalogue):
    state = native_catalogue
    listed(state)
    state.store.get_oauth_access.return_value = OAuthAccess(
        "fixture", 3, account_id="account-two"
    )
    state.body = {"models": [{"slug": "synthetic-other-account"}]}
    assert [row["id"] for row in listed(state)] == ["synthetic-other-account"]
    assert state.requests[-1].headers["ChatGPT-Account-ID"] == "account-two"


def test_refresh_uses_picker_cadence_and_keeps_stale_rows_on_error(native_catalogue, monkeypatch):
    state = native_catalogue
    listed(state)
    # Age discovery's clock, not disk timestamps; the provider changes after
    # the picker cadence, well inside discovery's longer background TTL.
    real_time = discovery.time.time
    monkeypatch.setattr(discovery.time, "time", lambda: real_time() + PICKER_TTL_S + 1)
    state.body = {"models": [{"slug": "synthetic-new-model"}]}
    assert [row["id"] for row in listed(state)] == ["synthetic-new-model"]
    assert len(state.requests) == 2
    monkeypatch.setattr(discovery.time, "time", lambda: real_time() + 2 * PICKER_TTL_S + 2)
    state.status = 503
    assert [row["id"] for row in listed(state)] == ["synthetic-new-model"]
    assert len(state.requests) == 3


@pytest.mark.parametrize("body,status", [({"models": []}, 200), ({"unexpected": []}, 200), ({}, 503)])
def test_empty_invalid_or_failed_native_listing_preserves_existing_fallback(
    native_catalogue, body, status
):
    state = native_catalogue
    state.body, state.status = body, status
    ids = {row["id"] for row in listed(state)}
    assert ids == set(openai_models)
    assert "synthetic-never-listed" not in ids


def test_missing_oauth_account_does_not_send_token_to_public_api(native_catalogue):
    state = native_catalogue
    state.store.get_oauth_access.return_value = OAuthAccess("fixture", 1)
    assert {row["id"] for row in listed(state)} == set(openai_models)
    assert not state.requests


def test_new_native_ids_need_no_shipped_catalogue_edit(native_catalogue):
    state = native_catalogue
    state.body = {"models": [{"slug": "synthetic-future-model"}]}
    assert [row["id"] for row in listed(state)] == ["synthetic-future-model"]
    assert "synthetic-never-listed" not in {row["id"] for row in listed(state)}


def test_credential_resolution_error_is_not_an_empty_success(native_catalogue):
    state = native_catalogue
    state.store.get_oauth_access.side_effect = RuntimeError("synthetic identity unavailable")
    response = state.client.get("/v1/models?provider=openai")
    assert response.status_code == 500
    assert not state.requests


@pytest.mark.asyncio
async def test_read_only_context_resolves_login_alias_without_losing_kind(native_catalogue):
    state = native_catalogue
    controller = ProviderController(state.store)
    assert await controller.listing_credential("openai-device", read_only=True) == (
        "fixture",
        True,
        "account-one",
    )
    state.store.get_oauth_access.assert_awaited_once_with("openai", read_only=True)


@pytest.mark.asyncio
async def test_api_key_fallback_stays_read_only(native_catalogue):
    state = native_catalogue
    state.store.get_oauth_access.return_value = None
    state.store.get_api_key.return_value = "fixture"
    assert await ProviderController(state.store).listing_credential(
        "openai", read_only=True
    ) == ("fixture", False, None)
    state.store.get_api_key.assert_awaited_once_with("openai", read_only=True)
