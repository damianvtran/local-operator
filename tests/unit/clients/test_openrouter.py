from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.clients.openrouter import (
    OpenRouterClient,
    OpenRouterListModelsResponse,
    OpenRouterModelData,
    OpenRouterModelPricing,
)


@pytest.fixture
def api_key() -> SecretStr:
    """Fixture for providing a test API key."""
    return SecretStr("test_api_key")


@pytest.fixture
def openrouter_client(api_key: SecretStr) -> OpenRouterClient:
    """Fixture for creating a OpenRouterClient instance.

    Args:
        api_key (SecretStr): API key for the client.

    Returns:
        OpenRouterClient: An instance of OpenRouterClient.
    """
    return OpenRouterClient(api_key=api_key)


@pytest.fixture
def mock_model_data() -> List[Dict[str, Any]]:
    """Fixture for providing mock model data."""
    return [
        {
            "id": "test_model_1",
            "name": "Test Model 1",
            "description": "A test model",
            "pricing": {"prompt": 0.001, "completion": 0.002},
        },
        {
            "id": "test_model_2",
            "name": "Test Model 2",
            "description": "Another test model",
            "pricing": {"prompt": 0.003, "completion": 0.004},
        },
    ]


@pytest.fixture
def mock_response(mock_model_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fixture for providing a mock JSON response from the OpenRouter API.

    Returns:
        Dict[str, Any]: Mock JSON data that simulates an OpenRouter API response.
    """
    return {"data": mock_model_data}


def test_list_models_success(
    openrouter_client: OpenRouterClient,
    mock_response: Dict[str, Any],
    mock_model_data: List[Dict[str, Any]],
) -> None:
    """Test successful API request to list models.

    Args:
        openrouter_client (OpenRouterClient): The OpenRouter API client fixture.
        mock_response (Dict[str, Any]): Mock JSON response.
        mock_model_data (List[Dict[str, Any]]): Mock model data.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_response

    with patch("requests.get", mock_requests_get):
        response = openrouter_client.list_models()

    assert isinstance(response, OpenRouterListModelsResponse)
    assert len(response.data) == len(mock_model_data)
    for i, model in enumerate(response.data):
        assert isinstance(model, OpenRouterModelData)
        assert model.id == mock_model_data[i]["id"]
        assert model.name == mock_model_data[i]["name"]
        assert model.description == mock_model_data[i]["description"]
        assert isinstance(model.pricing, OpenRouterModelPricing)
        assert model.pricing.prompt == mock_model_data[i]["pricing"]["prompt"]
        assert model.pricing.completion == mock_model_data[i]["pricing"]["completion"]


def test_list_models_api_error(openrouter_client: OpenRouterClient) -> None:
    """Test handling of API error response.

    Args:
        openrouter_client (OpenRouterClient): The OpenRouter API client fixture.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 400
    mock_requests_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_requests_get.return_value
    )

    with patch("requests.get", mock_requests_get):
        with pytest.raises(RuntimeError) as exc_info:
            openrouter_client.list_models()
        assert "Failed to fetch OpenRouter models due to a requests error" in str(exc_info.value)


def test_list_models_network_error(openrouter_client: OpenRouterClient) -> None:
    """Test handling of network error.

    Args:
        openrouter_client (OpenRouterClient): The OpenRouter API client fixture.
    """
    mock_requests_get = MagicMock(side_effect=requests.exceptions.RequestException("Network error"))

    with patch("requests.get", mock_requests_get):
        with pytest.raises(RuntimeError) as exc_info:
            openrouter_client.list_models()
        assert "Failed to fetch OpenRouter models due to a requests error" in str(exc_info.value)


def test_client_init_no_api_key() -> None:
    """Test client initialization with missing API key.

    Raises:
        RuntimeError: If no API key is provided.
    """
    with pytest.raises(RuntimeError) as exc_info:
        OpenRouterClient(SecretStr(""))
    assert "OpenRouter API key is required" in str(exc_info.value)
