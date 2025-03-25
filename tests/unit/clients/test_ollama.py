from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from local_operator.clients.ollama import OllamaClient, OllamaModelData


@pytest.fixture
def ollama_client() -> OllamaClient:
    """Fixture for creating an OllamaClient instance.

    Returns:
        OllamaClient: An instance of OllamaClient.
    """
    return OllamaClient()


@pytest.fixture
def mock_model_data() -> List[Dict[str, Any]]:
    """Fixture for providing mock model data."""
    return [
        {
            "name": "llama2",
            "modified_at": "2024-03-15T10:30:00Z",
            "size": 4000000000,
            "digest": "sha256:abc123",
            "details": {"parameter_size": "7B", "quantization_level": "Q4_0"},
        },
        {
            "name": "mistral",
            "modified_at": "2024-03-20T14:45:00Z",
            "size": 5000000000,
            "digest": "sha256:def456",
            "details": {"parameter_size": "7B", "quantization_level": "Q4_K_M"},
        },
    ]


@pytest.fixture
def mock_tags_response(mock_model_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fixture for providing a mock JSON response from the Ollama API tags endpoint.

    Returns:
        Dict[str, Any]: Mock JSON data that simulates an Ollama API response.
    """
    return {"models": mock_model_data}


def test_is_healthy_success(ollama_client: OllamaClient) -> None:
    """Test successful health check.

    Args:
        ollama_client (OllamaClient): The Ollama API client fixture.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200

    with patch("requests.get", mock_requests_get):
        result = ollama_client.is_healthy()

    assert result is True
    mock_requests_get.assert_called_once_with("http://localhost:11434/api/health", timeout=2)


def test_is_healthy_failure(ollama_client: OllamaClient) -> None:
    """Test failed health check due to non-200 status code.

    Args:
        ollama_client (OllamaClient): The Ollama API client fixture.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 500

    with patch("requests.get", mock_requests_get):
        result = ollama_client.is_healthy()

    assert result is False


def test_is_healthy_exception(ollama_client: OllamaClient) -> None:
    """Test failed health check due to request exception.

    Args:
        ollama_client (OllamaClient): The Ollama API client fixture.
    """
    mock_requests_get = MagicMock(
        side_effect=requests.exceptions.RequestException("Connection error")
    )

    with patch("requests.get", mock_requests_get):
        result = ollama_client.is_healthy()

    assert result is False


def test_list_models_success(
    ollama_client: OllamaClient,
    mock_tags_response: Dict[str, Any],
    mock_model_data: List[Dict[str, Any]],
) -> None:
    """Test successful API request to list models.

    Args:
        ollama_client (OllamaClient): The Ollama API client fixture.
        mock_tags_response (Dict[str, Any]): Mock JSON response.
        mock_model_data (List[Dict[str, Any]]): Mock model data.
    """
    # Mock the health check to return True
    mock_health_check = MagicMock(return_value=True)

    # Mock the API request to return the mock response
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_tags_response

    with patch.object(ollama_client, "is_healthy", mock_health_check):
        with patch("requests.get", mock_requests_get):
            response = ollama_client.list_models()

    assert len(response) == len(mock_model_data)
    for i, model in enumerate(response):
        assert isinstance(model, OllamaModelData)
        assert model.name == mock_model_data[i]["name"]
        assert model.modified_at == mock_model_data[i]["modified_at"]
        assert model.size == mock_model_data[i]["size"]
        assert model.digest == mock_model_data[i]["digest"]
        assert model.details == mock_model_data[i]["details"]


def test_list_models_unhealthy_server(ollama_client: OllamaClient) -> None:
    """Test list_models when server is not healthy.

    Args:
        ollama_client (OllamaClient): The Ollama API client fixture.
    """
    # Mock the health check to return False
    mock_health_check = MagicMock(return_value=False)

    with patch.object(ollama_client, "is_healthy", mock_health_check):
        with pytest.raises(RuntimeError) as exc_info:
            ollama_client.list_models()
        assert "Ollama server is not healthy" in str(exc_info.value)


def test_list_models_api_error(ollama_client: OllamaClient) -> None:
    """Test handling of API error response.

    Args:
        ollama_client (OllamaClient): The Ollama API client fixture.
    """
    # Mock the health check to return True
    mock_health_check = MagicMock(return_value=True)

    # Mock the API request to raise an HTTP error
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 400
    mock_requests_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_requests_get.return_value
    )

    with patch.object(ollama_client, "is_healthy", mock_health_check):
        with patch("requests.get", mock_requests_get):
            with pytest.raises(RuntimeError) as exc_info:
                ollama_client.list_models()
            assert "Failed to fetch Ollama models due to a requests error" in str(exc_info.value)


def test_list_models_network_error(ollama_client: OllamaClient) -> None:
    """Test handling of network error.

    Args:
        ollama_client (OllamaClient): The Ollama API client fixture.
    """
    # Mock the health check to return True
    mock_health_check = MagicMock(return_value=True)

    # Mock the API request to raise a network error
    mock_requests_get = MagicMock(side_effect=requests.exceptions.RequestException("Network error"))

    with patch.object(ollama_client, "is_healthy", mock_health_check):
        with patch("requests.get", mock_requests_get):
            with pytest.raises(RuntimeError) as exc_info:
                ollama_client.list_models()
            assert "Failed to fetch Ollama models due to a requests error" in str(exc_info.value)


def test_custom_base_url() -> None:
    """Test client initialization with custom base URL."""
    custom_url = "http://custom-ollama:11434"
    client = OllamaClient(base_url=custom_url)

    assert client.base_url == custom_url

    # Test that the custom URL is used in requests
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200

    with patch("requests.get", mock_requests_get):
        client.is_healthy()

    mock_requests_get.assert_called_once_with(f"{custom_url}/api/health", timeout=2)
