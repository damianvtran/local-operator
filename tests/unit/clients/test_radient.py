from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.clients.radient import (
    RadientClient,
    RadientListModelsResponse,
    RadientModelData,
    RadientModelPricing,
)


@pytest.fixture
def api_key() -> SecretStr:
    """Fixture for providing a test API key."""
    return SecretStr("test_api_key")


@pytest.fixture
def base_url() -> str:
    """Fixture for providing a test base URL."""
    return "https://api.test.radient.com"


@pytest.fixture
def radient_client(api_key: SecretStr, base_url: str) -> RadientClient:
    """Fixture for creating a RadientClient instance.

    Args:
        api_key (SecretStr): API key for the client.
        base_url (str): Base URL for the Radient API.

    Returns:
        RadientClient: An instance of RadientClient.
    """
    return RadientClient(api_key=api_key, base_url=base_url)


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
    """Fixture for providing a mock JSON response from the Radient API.

    Returns:
        Dict[str, Any]: Mock JSON data that simulates a Radient API response.
    """
    return {"data": mock_model_data}


def test_list_models_success(
    radient_client: RadientClient,
    mock_response: Dict[str, Any],
    mock_model_data: List[Dict[str, Any]],
    base_url: str,
) -> None:
    """Test successful API request to list models.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_response (Dict[str, Any]): Mock JSON response.
        mock_model_data (List[Dict[str, Any]]): Mock model data.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_response

    with patch("requests.get", mock_requests_get):
        response = radient_client.list_models()

    # Verify the request was made with the correct parameters
    mock_requests_get.assert_called_once_with(
        f"{base_url}/models",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientListModelsResponse)
    assert len(response.data) == len(mock_model_data)
    for i, model in enumerate(response.data):
        assert isinstance(model, RadientModelData)
        assert model.id == mock_model_data[i]["id"]
        assert model.name == mock_model_data[i]["name"]
        assert model.description == mock_model_data[i]["description"]
        assert isinstance(model.pricing, RadientModelPricing)
        assert model.pricing.prompt == mock_model_data[i]["pricing"]["prompt"]
        assert model.pricing.completion == mock_model_data[i]["pricing"]["completion"]


def test_list_models_api_error(radient_client: RadientClient) -> None:
    """Test handling of API error response.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
    """
    mock_response = MagicMock()
    mock_response.content = b"Error message from API"

    mock_requests_get = MagicMock()
    mock_requests_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_response
    )

    with patch("requests.get", mock_requests_get):
        with pytest.raises(RuntimeError) as exc_info:
            radient_client.list_models()
        assert "Failed to fetch Radient models due to a requests error" in str(exc_info.value)
        assert "Error message from API" in str(exc_info.value)


def test_list_models_network_error(radient_client: RadientClient) -> None:
    """Test handling of network error.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
    """
    mock_response = MagicMock()
    mock_response.content = b"Network error"

    mock_requests_get = MagicMock(
        side_effect=requests.exceptions.RequestException("Network error", response=mock_response)
    )

    with patch("requests.get", mock_requests_get):
        with pytest.raises(RuntimeError) as exc_info:
            radient_client.list_models()
        assert "Failed to fetch Radient models due to a requests error" in str(exc_info.value)


def test_client_init_no_api_key(base_url: str) -> None:
    """Test client initialization with missing API key.

    Args:
        base_url (str): Base URL for the Radient API.

    Raises:
        RuntimeError: If no API key is provided.
    """
    with pytest.raises(RuntimeError) as exc_info:
        RadientClient(SecretStr(""), base_url)
    assert "Radient API key is required" in str(exc_info.value)
