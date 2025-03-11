"""
Tests for the models endpoints.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from local_operator.clients.openrouter import (
    OpenRouterClient,
    OpenRouterListModelsResponse,
    OpenRouterModelData,
    OpenRouterModelPricing,
)
from local_operator.server.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_list_providers(client):
    """Test the list_providers endpoint."""
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
        "ollama",
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

    ollama = next(p for p in providers if p["id"] == "ollama")
    assert ollama["name"] == "Ollama"
    assert ollama["requiredCredentials"] == []


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
    mock_model1 = OpenRouterModelData(
        id="model1",
        name="Model 1",
        description="Test model 1",
        pricing=mock_pricing,
    )
    mock_model2 = OpenRouterModelData(
        id="model2",
        name="Model 2",
        description="Test model 2",
        pricing=mock_pricing,
    )
    mock_response = OpenRouterListModelsResponse(data=[mock_model1, mock_model2])
    mock_list_models.return_value = mock_response

    # Set up a mock credential manager to return a fake API key
    with patch(
        "local_operator.credentials.CredentialManager.get_credential",
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


@patch.object(OpenRouterClient, "list_models")
def test_list_models_no_api_key(mock_list_models, client, mock_credential_manager):
    """Test the list_models endpoint with no OpenRouter API key."""
    # Set up a mock credential manager to return None for the API key
    with patch(
        "local_operator.credentials.CredentialManager.get_credential",
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
