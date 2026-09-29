"""
Tests for the models endpoints.
"""

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from local_operator.clients.openrouter import (
    OpenRouterClient,
    OpenRouterListModelsResponse,
    OpenRouterModelData,
    OpenRouterModelPricing,
)
from local_operator.env import get_env_config
from local_operator.model import discovery
from local_operator.model.discovery import DiscoveredModel, merge_models
from local_operator.model.registry import anthropic_models, deepseek_models
from local_operator.server.app import app


@pytest.fixture
def client(monkeypatch):
    """Create a test client for the FastAPI app, with the state it depends on.

    A bare ``TestClient(app)`` does NOT run the lifespan handler — that only
    happens when the client is used as a context manager — so
    ``app.state.env_config`` is never set here. The ``/v1/models`` routes
    resolve it through ``Depends(get_env_config)``, which reads
    ``request.app.state.env_config`` directly, so these tests only passed when
    some EARLIER test in the same worker process had already populated that
    shared state (the async fixtures in ``conftest.py`` do, and ``app`` is a
    module-level singleton shared by every test in the process).

    That made the file order-dependent rather than broken: it passes in a run
    where a neighbour seeded the state first and fails with
    ``AttributeError: 'State' object has no attribute 'env_config'`` in one
    where it lands first on an xdist worker. Which happens is decided by how
    many tests the suite collects, so adding tests ANYWHERE could flip it —
    which is how it surfaced, on a branch whose only change here was 56 new TUI
    tests shifting the distribution.

    Setting the attribute the fixture's own tests need makes the file
    self-contained, which is the property it should have had: a test that
    depends on a sibling having run is not a test of anything it names.
    """
    # Restored afterwards so a lifespan-backed value from another test in the
    # same process is not clobbered — this fixture owns the value only for the
    # duration of its own test.
    # Local runtime integration has its own owned-HTTP tests. Enumeration
    # tests must not inspect whichever server a developer has running today.
    # Anthropic joins deepseek here because its branch now reads the same
    # discovery seam: answering it with the registry rows is what a real
    # answer looks like with no cache and no credential.
    monkeypatch.setattr(
        "local_operator.server.routes.models.available_models",
        lambda provider, **kw: (
            (
                merge_models(deepseek_models if provider == "deepseek" else anthropic_models, None)
                if provider in {"deepseek", "anthropic"}
                else []
            ),
            "static",
        ),
    )
    had_env_config = hasattr(app.state, "env_config")
    previous = getattr(app.state, "env_config", None)
    if not had_env_config or previous is None:
        app.state.env_config = get_env_config()
    try:
        yield TestClient(app)
    finally:
        if had_env_config:
            app.state.env_config = previous
        elif hasattr(app.state, "env_config"):
            delattr(app.state, "env_config")


def test_deepseek_http_catalogue_uses_authenticated_live_inventory(
    client, mock_credential_manager, monkeypatch
):
    """The older HTTP surface must not reintroduce retired picker entries."""
    seen = []

    def listing(provider, **kwargs):
        seen.append((provider, kwargs))
        return [
            DiscoveredModel(
                id="deepseek-flash",
                name="Live Flash",
                context_window=200_000,
                supports_images=False,
                supports_audio_input=True,
                supports_tools=False,
                reasoning=False,
            )
        ], "ok"

    monkeypatch.setattr("local_operator.server.routes.models.available_models", listing)
    response = client.get("/v1/models?provider=deepseek")
    assert response.status_code == 200
    rows = response.json()["result"]["models"]
    assert [row["id"] for row in rows] == ["deepseek-flash"]
    assert rows[0]["info"]["supports_images"] is False
    assert rows[0]["info"]["supports_audio_input"] is True
    assert rows[0]["info"]["supports_tools"] is False
    assert rows[0]["info"]["context_window"] == 200_000
    assert seen[0][0] == "deepseek"
    assert "api_key" in seen[0][1]


def test_list_providers_with_ollama_active(client):
    """Test the list_providers endpoint with Ollama server active."""
    # Mock the Ollama client to report that the server is healthy
    with patch("local_operator.clients.ollama.OllamaClient.is_healthy", return_value=True):
        response = client.get("/v1/models/providers")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["message"] == "Providers retrieved successfully"
        assert "result" in data
        assert "providers" in data["result"]
        providers = data["result"]["providers"]
        assert isinstance(providers, list)

        # Verify each provider has the expected fields
        for provider in providers:
            assert "id" in provider
            assert "name" in provider
            assert "description" in provider
            assert "url" in provider
            assert "requiredCredentials" in provider
            assert isinstance(provider["requiredCredentials"], list)

        # Verify expected providers are present
        provider_ids = [p["id"] for p in providers]
        expected_providers = [
            "openai",
            "anthropic",
            "google",
            "mistral",
            "ollama",  # Ollama should be included when server is active
            "openrouter",
            "deepseek",
            "kimi",
            "alibaba",
        ]
        for provider_id in expected_providers:
            assert provider_id in provider_ids

        # Verify some specific provider details
        openai = next(p for p in providers if p["id"] == "openai")
        assert openai["name"] == "OpenAI"
        assert openai["url"] == "https://platform.openai.com/"
        assert openai["requiredCredentials"] == ["OPENAI_API_KEY"]
        # The suggestion rides the same catalogue the onboarding step reads, from
        # the one backend table (`model.defaults`) -- never a renderer copy.
        assert openai["suggestedModel"] == {"id": "gpt-6-astra", "name": "GPT-6 Astra"}

        # Verify Ollama provider is present and has expected details
        ollama = next(p for p in providers if p["id"] == "ollama")
        assert ollama["name"] == "Ollama"
        assert ollama["requiredCredentials"] == []
        # A local runtime serves whatever the user pulled: no suggestion.
        assert ollama["suggestedModel"] is None


def test_list_providers_with_ollama_inactive(client):
    """Test the list_providers endpoint with Ollama server inactive."""
    # Mock the Ollama client to report that the server is not healthy
    with patch("local_operator.clients.ollama.OllamaClient.is_healthy", return_value=False):
        response = client.get("/v1/models/providers")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["message"] == "Providers retrieved successfully"
        assert "result" in data
        assert "providers" in data["result"]
        providers = data["result"]["providers"]
        assert isinstance(providers, list)

        # Verify each provider has the expected fields
        for provider in providers:
            assert "id" in provider
            assert "name" in provider
            assert "description" in provider
            assert "url" in provider
            assert "requiredCredentials" in provider
            assert isinstance(provider["requiredCredentials"], list)

        # Verify expected providers are present (except Ollama)
        provider_ids = [p["id"] for p in providers]
        expected_providers = [
            "openai",
            "anthropic",
            "google",
            "mistral",
            "openrouter",
            "deepseek",
            "kimi",
            "alibaba",
        ]
        for provider_id in expected_providers:
            assert provider_id in provider_ids

        # Stopped servers remain discoverable so the desktop can configure them.
        assert "ollama" in provider_ids


def test_list_models_with_ollama_active(client, mock_credential_manager, monkeypatch):
    """Test the list_models endpoint with Ollama server active."""
    rows = [DiscoveredModel(id="qwen-2.5:14b"), DiscoveredModel(id="phi-4:14b")]
    monkeypatch.setattr(
        "local_operator.server.routes.models.available_models",
        lambda provider, **kw: (rows if provider == "ollama" else [], "ok"),
    )

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Models retrieved successfully"
    assert "result" in data
    assert "models" in data["result"]
    models = data["result"]["models"]
    assert isinstance(models, list)

    # Find the Ollama models in the response
    ollama_models = [m for m in models if m.get("provider") == "ollama"]

    # Verify Ollama models are present
    assert len(ollama_models) == 2

    # Verify model details
    qwen = next((m for m in ollama_models if m.get("id") == "qwen-2.5:14b"), None)
    assert qwen is not None
    assert qwen["name"] == "qwen-2.5:14b"
    assert qwen["provider"] == "ollama"

    phi = next((m for m in ollama_models if m.get("id") == "phi-4:14b"), None)
    assert phi is not None
    assert phi["name"] == "phi-4:14b"
    assert phi["provider"] == "ollama"


def test_list_models_with_ollama_inactive(client, mock_credential_manager):
    """Test the list_models endpoint with Ollama server inactive."""
    # Mock the Ollama client to report that the server is not healthy
    with patch("local_operator.clients.ollama.OllamaClient.is_healthy", return_value=False):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["message"] == "Models retrieved successfully"
        assert "result" in data
        assert "models" in data["result"]
        models = data["result"]["models"]
        assert isinstance(models, list)

        # Verify no Ollama models are present
        ollama_models = [m for m in models if m.get("provider") == "ollama"]
        assert len(ollama_models) == 0


def test_list_models_no_provider(client, mock_credential_manager):
    """Test the list_models endpoint without a provider filter."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Models retrieved successfully"
    assert "result" in data
    assert "models" in data["result"]
    models = data["result"]["models"]
    assert isinstance(models, list)
    assert len(models) > 0
    # Check that we have models from different providers
    providers = set(model["provider"] for model in models)
    assert len(providers) > 1


def test_list_models_with_provider(client, mock_credential_manager):
    """Test the list_models endpoint with a provider filter."""
    response = client.get("/v1/models?provider=anthropic")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Models retrieved successfully"
    assert "result" in data
    assert "models" in data["result"]
    models = data["result"]["models"]
    assert isinstance(models, list)
    assert len(models) > 0
    # Check that all models are from the specified provider
    for model in models:
        assert model["provider"] == "anthropic"
    # The Anthropic rows on this surface come from the registry itself, so a
    # row added to `anthropic_models` must be visible here — this is the list
    # an offline or first-run picker reads.
    sonnet_55 = next((m for m in models if m["id"] == "claude-sonnet-5-5"), None)
    assert sonnet_55 is not None, sorted(m["id"] for m in models)
    assert sonnet_55["name"] == "Claude Sonnet 5.5"


def test_list_models_invalid_provider(client, mock_credential_manager):
    """Test the list_models endpoint with an invalid provider."""
    response = client.get("/v1/models?provider=invalid")
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Provider not found: invalid"


@patch.object(OpenRouterClient, "list_models")
def test_list_models_with_openrouter(mock_list_models, client, mock_credential_manager):
    """Test the list_models endpoint with OpenRouter models."""
    # Mock the OpenRouterClient.list_models method
    mock_pricing = OpenRouterModelPricing(prompt=0.001, completion=0.002)
    # Built through ``model_validate`` rather than the constructor so the
    # undeclared ``architecture`` rides in the entry's extras, exactly as the
    # live client parses a listing response.
    mock_model1 = OpenRouterModelData.model_validate(
        {
            "id": "model1",
            "name": "Model 1",
            "description": "Test model 1",
            "pricing": mock_pricing,
            "architecture": {"input_modalities": ["text", "audio"]},
        }
    )
    mock_model2 = OpenRouterModelData.model_validate(
        {
            "id": "model2",
            "name": "Model 2",
            "description": "Test model 2",
            "pricing": mock_pricing,
        }
    )
    mock_response = OpenRouterListModelsResponse(data=[mock_model1, mock_model2])
    mock_list_models.return_value = mock_response

    # The route resolves the key through the store-first reader; supply it as
    # that reader's return value rather than by patching the retired
    # credentials.env file, which is no longer on the resolution path.
    with patch(
        "local_operator.providers.registry.provider_env_key",
        return_value="fake_api_key",
    ):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["message"] == "Models retrieved successfully"
        assert "result" in data
        assert "models" in data["result"]
        models = data["result"]["models"]
        assert isinstance(models, list)

        # Find the OpenRouter models in the response
        openrouter_models = [m for m in models if m.get("provider") == "openrouter"]

        assert len(openrouter_models) == 2

        # Check for the mock models
        model1 = next((m for m in openrouter_models if m.get("id") == "model1"), None)
        assert model1 is not None
        assert model1["name"] == "Model 1"
        assert model1["provider"] == "openrouter"
        assert "info" in model1
        assert model1["info"]["description"] == "Test model 1"
        assert model1["info"]["input_price"] == 1000.0  # 0.001 * 1,000,000
        assert model1["info"]["output_price"] == 2000.0  # 0.002 * 1,000,000
        # The listing's own modality statement reaches the entry; a listing
        # that sent none leaves the field unstated (``None``, not a denial).
        assert model1["info"]["supports_audio_input"] is True
        model2 = next((m for m in openrouter_models if m.get("id") == "model2"), None)
        assert model2 is not None
        assert model2["info"]["supports_audio_input"] is None


@patch.object(OpenRouterClient, "list_models")
def test_list_models_no_api_key(mock_list_models, client, mock_credential_manager):
    """Test the list_models endpoint with no OpenRouter API key."""
    # No provider key resolves, so no OpenRouter client is built.
    with patch(
        "local_operator.providers.registry.provider_env_key",
        return_value=None,
    ):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["message"] == "Models retrieved successfully"

        # There should still be models from other providers
        assert "result" in data
        assert "models" in data["result"]
        models = data["result"]["models"]
        assert isinstance(models, list)
        assert len(models) > 0

        # There should be at least one OpenRouter model (the default one)
        openrouter_models = [m for m in models if m.get("provider") == "openrouter"]
        assert len(openrouter_models) == 0


def test_list_models_with_sort_and_direction(client, mock_credential_manager):
    """Test the list_models endpoint with sort and direction parameters."""
    # Test sorting by id in descending order (default)
    response = client.get("/v1/models?sort=id&direction=descending")
    assert response.status_code == 200
    data = response.json()
    models = data["result"]["models"]
    # Check that models are sorted by id in descending order
    for i in range(1, len(models)):
        assert models[i - 1]["id"] >= models[i]["id"]

    # Test sorting by id in ascending order
    response = client.get("/v1/models?sort=id&direction=ascending")
    assert response.status_code == 200
    data = response.json()
    models = data["result"]["models"]
    # Check that models are sorted by id in ascending order
    for i in range(1, len(models)):
        assert models[i - 1]["id"] <= models[i]["id"]

    # Test sorting by provider in descending order
    response = client.get("/v1/models?sort=provider&direction=descending")
    assert response.status_code == 200
    data = response.json()
    models = data["result"]["models"]
    # Check that models are sorted by provider in descending order
    for i in range(1, len(models)):
        assert models[i - 1]["provider"] >= models[i]["provider"]

    # Test sorting by provider in ascending order
    response = client.get("/v1/models?sort=provider&direction=ascending")
    assert response.status_code == 200
    data = response.json()
    models = data["result"]["models"]
    # Check that models are sorted by provider in ascending order
    for i in range(1, len(models)):
        assert models[i - 1]["provider"] <= models[i]["provider"]

    # Test sorting by name in descending order
    response = client.get("/v1/models?sort=name&direction=descending")
    assert response.status_code == 200
    data = response.json()
    models = data["result"]["models"]
    # Check that models are sorted by name in descending order
    # Note: None values are sorted first when direction is descending
    for i in range(1, len(models)):
        if models[i - 1]["name"] is None and models[i]["name"] is None:
            continue
        if models[i - 1]["name"] is None:
            assert False, "None values should be sorted first when direction is descending"
        if models[i]["name"] is None:
            continue
        assert models[i - 1]["name"] >= models[i]["name"]

    # Test sorting by name in ascending order
    response = client.get("/v1/models?sort=name&direction=ascending")
    assert response.status_code == 200
    data = response.json()
    models = data["result"]["models"]
    # Check that models are sorted by name in ascending order
    # Note: None values are sorted first when direction is ascending
    for i in range(1, len(models)):
        if models[i - 1]["name"] is None and models[i]["name"] is None:
            continue
        if models[i]["name"] is None:
            assert False, "None values should be sorted first when direction is ascending"
        if models[i - 1]["name"] is None:
            continue
        assert models[i - 1]["name"] <= models[i]["name"]


# -- Anthropic: the live listing over the registry ---------------------------
#
# `/v1/models?provider=anthropic` listed `anthropic_models` alone, so a model
# Anthropic's `GET /v1/models` already served stayed invisible on this surface
# until a release edited the registry by hand. It now reads that listing
# through the shared discovery cache, merged over the registry. These tests
# drive the REAL route -- no `available_models` patch, unlike the fixture
# above -- on an isolated HOME: the cache root derives from the home
# directory, not the config dir, so a run that sets only
# `LOCAL_OPERATOR_CONFIG_DIR` plants its document where the route never looks
# (AGENTS.md, "Isolating a run").

#: An id the shipped registry does not carry, in the shape Anthropic's listing
#: hands it over. Tests that rely on it assert it is still unshipped first, so
#: a future registry edit fails that guard rather than letting the test pass
#: for the registry's reason.
NOVEL_ANTHROPIC_ID = "claude-sonnet-6-0"

#: A bundled id the planted documents below deliberately do NOT list.
BUNDLED_ANTHROPIC_ID = "claude-sonnet-5-5"


def _plant_anthropic_listing(root: Path, entries: list[dict[str, Any]]) -> None:
    """Write the cached discovery document the route reads, fresh, under root.

    The capture-2 shape the reader expects, in the HOME-derived cache root
    (see the section comment). Aged "now" on purpose: a document old enough to
    trigger a refetch would measure the fetch stub instead of what the route
    serves from the cache.
    """
    cache_dir = root / ".local-operator" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "anthropic.listing.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "capture": discovery.listing_capture_version("anthropic"),
                    "models": entries,
                },
            }
        ),
        encoding="utf-8",
    )


class _Fetches:
    """Stands in for ``discovery.fetch_models``: records calls, serves the wire.

    Answered with a wire failure unless a test sets ``answer``: the planted
    documents here are fresh, so a recorded call means something in the route
    decided to fetch, and it must never reach the real network.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.answer: list[DiscoveredModel] | None = None

    def __call__(self, provider_id: str, **kwargs: object) -> list[DiscoveredModel] | None:
        self.calls.append(provider_id)
        return self.answer


@pytest.fixture
def discovery_client(tmp_path, monkeypatch):
    """The real route on an isolated HOME/config/cache, plus a fetch stub.

    The file's ``client`` fixture patches ``available_models``; these tests
    must NOT use it, because the seam under test is exactly what that patch
    replaces. Everything else follows its conventions (bare ``TestClient``
    with ``app.state`` seeded by hand, because the lifespan does not run).
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # An ambient credential would make the key the route resolves depend on
    # the developer's shell; arm exactly the one Anthropic API key it should
    # fetch with, on the environment rung of the credential cascade.
    for name in ("ANTHROPIC_OAUTH_TOKEN", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")

    fetched = _Fetches()
    monkeypatch.setattr(discovery, "fetch_models", fetched)

    had_env_config = hasattr(app.state, "env_config")
    previous_env_config = getattr(app.state, "env_config", None)
    if not had_env_config or previous_env_config is None:
        app.state.env_config = get_env_config()
    # The provider store is cached on app state and resolves its root through
    # `paths.config_dir()`; force it to build under THIS test's root so the
    # credential read and the cache the route reads share one isolated tree.
    previous_desktop_auth = getattr(app.state, "desktop_auth", None)
    app.state.desktop_auth = None
    try:
        yield TestClient(app), tmp_path, fetched
    finally:
        created = getattr(app.state, "desktop_auth", None)
        if created is not None:
            created.store.close()
        app.state.desktop_auth = previous_desktop_auth
        if had_env_config:
            app.state.env_config = previous_env_config
        elif hasattr(app.state, "env_config"):
            delattr(app.state, "env_config")


def test_anthropic_lists_a_model_only_the_live_listing_carries(
    discovery_client, mock_credential_manager
):
    """The pin this branch exists for: a model Anthropic released appears.

    The planted document is fresh, so the shared cache answers with no fetch;
    the id must reach the response over the registry alone, which has never
    heard of it.
    """
    client, root, fetched = discovery_client
    assert (
        NOVEL_ANTHROPIC_ID not in anthropic_models
    ), "the id this test treats as unshipped now ships"
    _plant_anthropic_listing(
        root,
        [{"id": NOVEL_ANTHROPIC_ID, "name": "Claude Sonnet 6", "context_window": 1_000_000}],
    )

    response = client.get("/v1/models?provider=anthropic")

    assert response.status_code == 200, response.text
    rows = response.json()["result"]["models"]
    by_id = {row["id"]: row for row in rows}
    assert NOVEL_ANTHROPIC_ID in by_id, sorted(by_id)
    assert by_id[NOVEL_ANTHROPIC_ID]["info"]["context_window"] == 1_000_000
    # The union half: the document never mentions this bundled row, so its
    # presence is the registry fallback and nothing else.
    assert BUNDLED_ANTHROPIC_ID in by_id
    # A fresh document must be SERVED, not fetched: this id is the test's own
    # planting, and a fetch here would mean the route went to the wire.
    assert fetched.calls == []


def test_a_bundled_anthropic_id_keeps_its_registry_prices_and_limits(
    discovery_client, mock_credential_manager
):
    """The merge is holes-only, per the catalogue design's field precedence.

    Anthropic's listing quotes no prices, so prices can only come from the
    registry, and the limits join them when the listing is silent. A merge
    that let the raw row win would zero the prices ("$—" in every picker
    that renders it) for rows this surface already priced correctly. The
    listed display name DOES win -- `_merge_name`'s existing rule, "that is
    how a renamed model gets its new label" -- which is also the proof the
    live row flowed at all.
    """
    client, root, _fetched = discovery_client
    _plant_anthropic_listing(
        root, [{"id": BUNDLED_ANTHROPIC_ID, "name": "Sonnet 5.5 (live label)"}]
    )

    response = client.get("/v1/models?provider=anthropic")

    assert response.status_code == 200, response.text
    rows = response.json()["result"]["models"]
    row = next(row for row in rows if row["id"] == BUNDLED_ANTHROPIC_ID)
    registry_row = anthropic_models[BUNDLED_ANTHROPIC_ID]
    info = row["info"]
    # Both directions, so a future edit to the fixture cannot silently weaken
    # the first half: the served numbers equal the registry's, and the
    # registry still carries the values this test was written against.
    assert info["input_price"] == registry_row.input_price == 2.0
    assert info["output_price"] == registry_row.output_price == 10.0
    assert info["cache_reads_price"] == registry_row.cache_reads_price == 0.20
    assert info["cache_writes_price"] == registry_row.cache_writes_price == 2.50
    assert info["context_window"] == registry_row.context_window == 1_000_000
    assert info["max_tokens"] == registry_row.max_tokens == 128_000
    assert info["supports_images"] is True
    assert info["name"] == "Sonnet 5.5 (live label)"


def test_anthropic_without_a_cached_document_still_lists_the_registry(
    discovery_client, mock_credential_manager
):
    """No cache: the route asks the endpoint, the attempt fails, the registry
    still answers. A flaky cache or an outage must never turn this surface
    into an empty model list.

    The fetch attempt is asserted on purpose -- it is what makes this the
    FALLBACK path rather than the unauthenticated one. Without it the test
    would pass even if the branch never consulted discovery at all.
    """
    client, _root, fetched = discovery_client

    response = client.get("/v1/models?provider=anthropic")

    assert response.status_code == 200, response.text
    rows = response.json()["result"]["models"]
    by_id = {row["id"]: row for row in rows}
    assert BUNDLED_ANTHROPIC_ID in by_id, sorted(by_id)
    assert by_id[BUNDLED_ANTHROPIC_ID]["info"]["input_price"] == 2.0
    assert fetched.calls == ["anthropic"], "a cold cache must consult the live endpoint"
