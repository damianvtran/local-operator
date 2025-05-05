from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.clients.radient import (
    RadientClient,
    RadientImage,
    RadientImageGenerationProvider,
    RadientImageGenerationProvidersResponse,
    RadientImageGenerationResponse,
    RadientListModelsResponse,
    RadientModelData,
    RadientModelPricing,
    RadientSearchProvider,
    RadientSearchProvidersResponse,
    RadientSearchResponse,
    RadientSearchResult,
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


@pytest.fixture
def mock_image_generation_response() -> Dict[str, Any]:
    """Fixture for providing a mock image generation response."""
    return {
        "request_id": "test-request-id",
        "status": "completed",
        "images": [
            {
                "url": "https://example.com/image1.jpg",
                "width": 1024,
                "height": 1024,
            }
        ],
    }


@pytest.fixture
def mock_image_generation_providers_response() -> Dict[str, Any]:
    """Fixture for providing a mock image generation providers response."""
    return {
        "providers": [
            {
                "id": "provider1",
                "name": "Provider 1",
                "description": "A test provider",
            },
            {
                "id": "provider2",
                "name": "Provider 2",
                "description": "Another test provider",
            },
        ]
    }


@pytest.fixture
def mock_search_response() -> Dict[str, Any]:
    """Fixture for providing a mock search response."""
    return {
        "query": "test query",
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/result1",
                "content": "This is test result 1",
                "raw_content": "Full content of test result 1",
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/result2",
                "content": "This is test result 2",
            },
        ],
    }


@pytest.fixture
def mock_search_providers_response() -> Dict[str, Any]:
    """Fixture for providing a mock search providers response."""
    return {
        "providers": [
            {
                "id": "search_provider1",
                "name": "Search Provider 1",
                "description": "A test search provider",
            },
            {
                "id": "search_provider2",
                "name": "Search Provider 2",
                "description": "Another test search provider",
            },
        ]
    }


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

    The client should not raise on init, but should raise when calling an API-key-required method.
    """
    client = RadientClient(api_key=None, base_url=base_url)
    # Should not raise on init
    assert isinstance(client, RadientClient)
    # Should raise when calling an API-key-required method
    with pytest.raises(RuntimeError) as exc_info:
        client.list_models()
    assert "Radient API key is required" in str(exc_info.value)


def test_upload_agent_to_marketplace_success(
    radient_client: RadientClient, base_url: str, tmp_path: Path
):
    """Test successful upload of a new agent to the marketplace."""
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"dummy zip content")
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": "new-agent-id"}
    with patch("requests.post", return_value=mock_response) as mock_post:
        agent_id = radient_client.upload_agent_to_marketplace(zip_path)
    assert agent_id == "new-agent-id"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == f"{base_url}/agents/upload"
    assert "files" in kwargs
    assert "headers" in kwargs
    assert kwargs["files"]["file"][0] == "agent.zip"


def test_overwrite_agent_in_marketplace_success(
    radient_client: RadientClient, base_url: str, tmp_path: Path
):
    """Test successful overwrite of an existing agent in the marketplace."""
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"dummy zip content")
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("requests.put", return_value=mock_response) as mock_put:
        radient_client.overwrite_agent_in_marketplace("existing-id", zip_path)
    mock_put.assert_called_once()
    args, kwargs = mock_put.call_args
    assert args[0] == f"{base_url}/agents/existing-id/upload"
    assert "files" in kwargs
    assert "headers" in kwargs
    assert kwargs["files"]["file"][0] == "agent.zip"


def test_download_agent_from_marketplace_success(
    radient_client: RadientClient, base_url: str, tmp_path: Path
):
    """Test successful download of an agent from the marketplace."""
    agent_id = "agent123"
    dest_path = tmp_path / "downloaded.zip"
    dummy_content = b"zip file content"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content = MagicMock(return_value=[dummy_content])
    with patch("requests.get", return_value=mock_response) as mock_get:
        radient_client.download_agent_from_marketplace(agent_id, dest_path)
    mock_get.assert_called_once_with(
        f"{base_url}/agents/{agent_id}/download",
        headers={
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
        stream=True,
    )
    assert dest_path.read_bytes() == dummy_content


def test_delete_agent_from_marketplace_success(radient_client: RadientClient, base_url: str):
    """Test successful deletion of an agent from the Radient Agent Hub (204)."""
    agent_id = "agent-to-delete"
    mock_response = MagicMock()
    mock_response.status_code = 204
    with patch("requests.delete", return_value=mock_response) as mock_delete:
        radient_client.delete_agent_from_marketplace(agent_id)
    mock_delete.assert_called_once_with(
        f"{base_url}/agents/{agent_id}",
        headers={
            "Authorization": "Bearer test_api_key",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
    )


def test_delete_agent_from_marketplace_error_response(radient_client: RadientClient, base_url: str):
    """Test error response (non-204) when deleting an agent from Radient."""
    agent_id = "bad-agent"
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.content = b"Agent not found"
    with patch("requests.delete", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc_info:
            radient_client.delete_agent_from_marketplace(agent_id)
        assert "Failed to delete agent from Radient Agent Hub" in str(exc_info.value)
        assert "Agent not found" in str(exc_info.value)


def test_delete_agent_from_marketplace_network_error(radient_client: RadientClient, base_url: str):
    """Test network error when deleting an agent from Radient."""
    agent_id = "network-error-agent"
    mock_response = MagicMock()
    mock_response.content = b"Network error"
    mock_delete = MagicMock(
        side_effect=requests.exceptions.RequestException("Network error", response=mock_response)
    )
    with patch("requests.delete", mock_delete):
        with pytest.raises(RuntimeError) as exc_info:
            radient_client.delete_agent_from_marketplace(agent_id)
        assert "Failed to delete agent from Radient Agent Hub" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)


# Image Generation Tests


def test_generate_image_success(
    radient_client: RadientClient,
    mock_image_generation_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test successful API request to generate an image.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_image_generation_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_post = MagicMock()
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = mock_image_generation_response

    with patch("requests.post", mock_requests_post):
        response = radient_client.generate_image(
            prompt="test prompt",
            num_images=1,
            image_size="square_hd",
            sync_mode=True,
        )

    # Verify the request was made with the correct parameters
    mock_requests_post.assert_called_once_with(
        f"{base_url}/tools/images/generate",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
        json={
            "prompt": "test prompt",
            "num_images": 1,
            "image_size": "square_hd",
            "sync_mode": True,
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientImageGenerationResponse)
    assert response.request_id == mock_image_generation_response["request_id"]
    assert response.status == mock_image_generation_response["status"]
    assert response.images is not None
    assert len(response.images) == len(mock_image_generation_response["images"])
    assert isinstance(response.images[0], RadientImage)
    assert response.images[0].url == mock_image_generation_response["images"][0]["url"]
    assert response.images[0].width == mock_image_generation_response["images"][0]["width"]
    assert response.images[0].height == mock_image_generation_response["images"][0]["height"]


def test_generate_image_with_provider(
    radient_client: RadientClient,
    mock_image_generation_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test image generation with provider specified.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_image_generation_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_post = MagicMock()
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = mock_image_generation_response

    with patch("requests.post", mock_requests_post):
        response = radient_client.generate_image(
            prompt="test prompt",
            provider="test_provider",
        )

    # Verify the request was made with the correct parameters
    mock_requests_post.assert_called_once_with(
        f"{base_url}/tools/images/generate",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
        json={
            "prompt": "test prompt",
            "num_images": 1,
            "image_size": "square_hd",
            "sync_mode": True,
            "provider": "test_provider",
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientImageGenerationResponse)


def test_generate_image_with_source_url(
    radient_client: RadientClient,
    mock_image_generation_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test image generation with source URL for image-to-image generation.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_image_generation_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_post = MagicMock()
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = mock_image_generation_response

    with patch("requests.post", mock_requests_post):
        response = radient_client.generate_image(
            prompt="test prompt",
            source_url="https://example.com/source.jpg",
            strength=0.7,
        )

    # Verify the request was made with the correct parameters
    mock_requests_post.assert_called_once_with(
        f"{base_url}/tools/images/generate",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
        json={
            "prompt": "test prompt",
            "num_images": 1,
            "image_size": "square_hd",
            "sync_mode": True,
            "source_url": "https://example.com/source.jpg",
            "strength": 0.7,
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientImageGenerationResponse)


def test_generate_image_api_error(radient_client: RadientClient) -> None:
    """Test handling of API error response for image generation.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
    """
    mock_response = MagicMock()
    mock_response.content = b"Error message from API"

    mock_requests_post = MagicMock()
    mock_requests_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_response
    )

    with patch("requests.post", mock_requests_post):
        with pytest.raises(RuntimeError) as exc_info:
            radient_client.generate_image(prompt="test prompt")
        assert "Failed to generate image" in str(exc_info.value)
        assert "Error message from API" in str(exc_info.value)


def test_get_image_generation_status_success(
    radient_client: RadientClient,
    mock_image_generation_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test successful API request to get image generation status.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_image_generation_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_image_generation_response

    with patch("requests.get", mock_requests_get):
        response = radient_client.get_image_generation_status(request_id="test-request-id")

    # Verify the request was made with the correct parameters
    mock_requests_get.assert_called_once_with(
        f"{base_url}/tools/images/status",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
        params={
            "request_id": "test-request-id",
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientImageGenerationResponse)
    assert response.request_id == mock_image_generation_response["request_id"]
    assert response.status == mock_image_generation_response["status"]


def test_get_image_generation_status_with_provider(
    radient_client: RadientClient,
    mock_image_generation_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test getting image generation status with provider specified.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_image_generation_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_image_generation_response

    with patch("requests.get", mock_requests_get):
        response = radient_client.get_image_generation_status(
            request_id="test-request-id",
            provider="test_provider",
        )

    # Verify the request was made with the correct parameters
    mock_requests_get.assert_called_once_with(
        f"{base_url}/tools/images/status",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
        params={
            "request_id": "test-request-id",
            "provider": "test_provider",
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientImageGenerationResponse)


def test_list_image_generation_providers_success(
    radient_client: RadientClient,
    mock_image_generation_providers_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test successful API request to list image generation providers.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_image_generation_providers_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_image_generation_providers_response

    with patch("requests.get", mock_requests_get):
        response = radient_client.list_image_generation_providers()

    # Verify the request was made with the correct parameters
    mock_requests_get.assert_called_once_with(
        f"{base_url}/tools/images/providers",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientImageGenerationProvidersResponse)
    assert len(response.providers) == len(mock_image_generation_providers_response["providers"])
    for i, provider in enumerate(response.providers):
        assert isinstance(provider, RadientImageGenerationProvider)
        assert provider.id == mock_image_generation_providers_response["providers"][i]["id"]
        assert provider.name == mock_image_generation_providers_response["providers"][i]["name"]
        assert (
            provider.description
            == mock_image_generation_providers_response["providers"][i]["description"]
        )


# Web Search Tests


def test_search_success(
    radient_client: RadientClient,
    mock_search_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test successful API request to search.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_search_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_search_response

    with patch("requests.get", mock_requests_get):
        response = radient_client.search(query="test query")

    # Verify the request was made with the correct parameters
    mock_requests_get.assert_called_once_with(
        f"{base_url}/tools/search",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
        params={
            "query": "test query",
            "max_results": 10,
            "include_raw": "false",
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientSearchResponse)
    assert response.query == mock_search_response["query"]
    assert len(response.results) == len(mock_search_response["results"])
    assert isinstance(response.results[0], RadientSearchResult)
    assert response.results[0].title == mock_search_response["results"][0]["title"]
    assert response.results[0].url == mock_search_response["results"][0]["url"]
    assert response.results[0].content == mock_search_response["results"][0]["content"]
    assert response.results[0].raw_content == mock_search_response["results"][0]["raw_content"]


def test_search_with_options(
    radient_client: RadientClient,
    mock_search_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test search with additional options.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_search_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_search_response

    with patch("requests.get", mock_requests_get):
        response = radient_client.search(
            query="test query",
            max_results=5,
            provider="test_provider",
            include_raw=True,
            search_depth="deep",
            domains=["example.com", "test.com"],
        )

    # Verify the request was made with the correct parameters
    mock_requests_get.assert_called_once_with(
        f"{base_url}/tools/search",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
        params={
            "query": "test query",
            "max_results": 5,
            "include_raw": "true",
            "provider": "test_provider",
            "search_depth": "deep",
            "domains": "example.com,test.com",
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientSearchResponse)


def test_search_api_error(radient_client: RadientClient) -> None:
    """Test handling of API error response for search.

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
            radient_client.search(query="test query")
        assert "Failed to execute search" in str(exc_info.value)
        assert "Error message from API" in str(exc_info.value)


def test_list_search_providers_success(
    radient_client: RadientClient,
    mock_search_providers_response: Dict[str, Any],
    base_url: str,
) -> None:
    """Test successful API request to list search providers.

    Args:
        radient_client (RadientClient): The Radient API client fixture.
        mock_search_providers_response (Dict[str, Any]): Mock JSON response.
        base_url (str): Base URL for the Radient API.
    """
    mock_requests_get = MagicMock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_search_providers_response

    with patch("requests.get", mock_requests_get):
        response = radient_client.list_search_providers()

    # Verify the request was made with the correct parameters
    mock_requests_get.assert_called_once_with(
        f"{base_url}/tools/search/providers",
        headers={
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
        },
    )

    # Verify the response was parsed correctly
    assert isinstance(response, RadientSearchProvidersResponse)
    assert len(response.providers) == len(mock_search_providers_response["providers"])
    for i, provider in enumerate(response.providers):
        assert isinstance(provider, RadientSearchProvider)
        assert provider.id == mock_search_providers_response["providers"][i]["id"]
        assert provider.name == mock_search_providers_response["providers"][i]["name"]
        assert provider.description == mock_search_providers_response["providers"][i]["description"]
