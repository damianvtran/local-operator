import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.agent_profiles import MAX_INSTRUCTIONS_CHARS
from local_operator.clients._http import (
    NO_RESPONSE_BODY,
    APIError,
    redact_secrets,
    response_body,
)
from local_operator.clients.radient import (
    INSTRUCTION_SET_DOCUMENT_TYPE,
    INSTRUCTION_SET_FIELDS,
    InstructionSetError,
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
    RadientTranscriptionResponseData,
    build_instruction_set_document,
    validate_document_overrides,
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


def test_upload_agent_to_marketplace_reads_the_id_from_the_live_envelope(
    radient_client: RadientClient, tmp_path: Path
):
    """The envelope the LIVE hub answers a created listing with (measured 2026-09-18).

    Returning ``next(iter(data.values()))`` handed the caller ``msg`` — the
    sentence "Agent imported successfully" — so ``push`` printed that as the new
    listing's id and the user was left holding a listing they could not name,
    find or delist.
    """
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"dummy zip content")
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "msg": "Agent imported successfully",
        "result": {"agent_id": "ae727e5c-a32e-48f0-9f8e-1f2a3b4c5d6e"},
    }
    with patch("requests.post", return_value=mock_response):
        agent_id = radient_client.upload_agent_to_marketplace(zip_path)
    assert agent_id == "ae727e5c-a32e-48f0-9f8e-1f2a3b4c5d6e"


def test_upload_agent_to_marketplace_refuses_a_payload_with_no_listing_id(
    radient_client: RadientClient, tmp_path: Path
):
    """A message is not an id, and the caller prints whatever this returns."""
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"dummy zip content")
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"msg": "Agent imported successfully"}
    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="no listing id"):
            radient_client.upload_agent_to_marketplace(zip_path)


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


def test_get_agent_joins_the_base_without_adding_a_version_segment(
    radient_client: RadientClient, base_url: str
):
    """``get_agent`` addresses the same root as its sibling methods.

    The product hands this client a base that already carries ``/v1``
    (``env_config.radient_api_base_url``), so any version segment written into
    this method's own path is a second one: the request went to
    ``/v1/v1/agents/{id}``, 404'd, and ``push --id`` fell through to creating a
    new listing instead of overwriting the named one.
    """
    agent_id = "agent-to-get"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": agent_id}
    with patch("requests.get", return_value=mock_response) as mock_get:
        assert radient_client.get_agent(agent_id) == {"id": agent_id}
    assert mock_get.call_args[0][0] == f"{base_url}/agents/{agent_id}"
    assert "/v1/v1" not in mock_get.call_args[0][0]


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


# Transcription: the provider/model passthrough contract.
#
# These tests pin what the daemon puts on the wire, so they assert on the
# SERIALISED multipart body — the bytes handed to the connection — rather than on
# the kwargs dict given to `requests.post`. The distinction is not pedantic: the
# dict carries `None`s that `requests` drops during preparation, so asserting on it
# would also constrain *how* the client builds the body, and a body-equivalent
# refactor would fail the test.


def _transcription_post() -> tuple[MagicMock, List[bytes]]:
    """Mock `requests.post`, recording the prepared body of every call.

    The body is prepared inside the call, not afterwards from the recorded kwargs:
    the client closes the audio handle as soon as `post` returns, and preparing a
    request reads the file.

    Returns:
        tuple[MagicMock, List[bytes]]: the patched `requests.post` and one
        serialised body per call it received.
    """
    bodies: List[bytes] = []

    def _post(url: str, **kwargs: Any) -> MagicMock:
        bodies.append(
            requests.Request("POST", url, **kwargs).prepare().body  # type: ignore[arg-type]
        )
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "result": {"text": "hello", "provider": "elevenlabs", "status": "completed"}
        }
        return response

    return MagicMock(side_effect=_post), bodies


def _multipart_fields(body: bytes) -> Dict[str, str]:
    """Map each field name in a serialised multipart body to its text value.

    The file part is reported as `<file>`; the point of these tests is which text
    fields travel, not the audio bytes.

    Args:
        body (bytes): The prepared request body (its boundary is the first token).

    Returns:
        Dict[str, str]: field name to value, for every part that has one.
    """
    boundary = body.split(b"\r\n", 1)[0]
    fields: Dict[str, str] = {}
    for part in body.split(boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, _, content = part.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]*)"', head)
        if name is None:
            continue
        fields[name.group(1).decode()] = (
            "<file>" if b"filename=" in head else content.rstrip(b"\r\n").decode()
        )
    return fields


def test_create_transcription_omits_unset_provider_and_model(
    radient_client: RadientClient, tmp_path: Path
) -> None:
    """Test that an unset provider and model are left out of the request body.

    This is the daemon's default path. The fields must be *absent* rather than
    sent empty: the server default governs, and a server defaulted to a
    non-OpenAI provider rejects an OpenAI model id, so sending one at all would
    lock the talk feature to OpenAI.
    """
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")
    mock_requests_post, bodies = _transcription_post()

    with patch("requests.post", mock_requests_post):
        response = radient_client.create_transcription(file_path=str(audio_file))

    assert isinstance(response, RadientTranscriptionResponseData)
    assert mock_requests_post.call_args[0][0].endswith("/tools/transcriptions")
    # The whole field set, so nothing else leaks either: no model, no provider,
    # no prompt, no language.
    assert _multipart_fields(bodies[0]) == {
        "file": "<file>",
        "response_format": "json",
        "temperature": "0.0",
    }


def test_create_transcription_drops_empty_provider_and_model(
    radient_client: RadientClient, tmp_path: Path
) -> None:
    """Test that empty-string provider and model are dropped, not sent empty.

    The client's guard is truthiness, and the route comment says so — an empty
    field (`-F "model="` from a caller) must not reach the wire as `model=`.
    """
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")
    mock_requests_post, bodies = _transcription_post()

    with patch("requests.post", mock_requests_post):
        radient_client.create_transcription(
            file_path=str(audio_file), model="", provider="", prompt=""
        )

    assert _multipart_fields(bodies[0]) == {
        "file": "<file>",
        "response_format": "json",
        "temperature": "0.0",
    }


def test_create_transcription_forwards_explicit_provider_and_model(
    radient_client: RadientClient, tmp_path: Path
) -> None:
    """Test that an explicitly requested provider and model are forwarded verbatim."""
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")
    mock_requests_post, bodies = _transcription_post()

    with patch("requests.post", mock_requests_post):
        radient_client.create_transcription(
            file_path=str(audio_file), model="scribe_v2", provider="elevenlabs"
        )

    fields = _multipart_fields(bodies[0])
    assert fields["model"] == "scribe_v2"
    assert fields["provider"] == "elevenlabs"


def test_create_transcription_forwards_provider_without_a_model(
    radient_client: RadientClient, tmp_path: Path
) -> None:
    """Test that `provider` alone travels, with no model invented alongside it.

    The daemon does not enforce a provider/model pairing rule; whatever the caller
    passes is forwarded and nothing else is added.
    """
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")
    mock_requests_post, bodies = _transcription_post()

    with patch("requests.post", mock_requests_post):
        radient_client.create_transcription(file_path=str(audio_file), provider="elevenlabs")

    fields = _multipart_fields(bodies[0])
    assert fields["provider"] == "elevenlabs"
    assert "model" not in fields


# --- Upstream failures on the transcription path -------------------------------
#
# WHY THESE USE A REAL `requests.Response` AND NOT A MagicMock: the callers of
# these helpers are responses, not mocks, and that difference is load-bearing --
# `requests.Response.__bool__` returns `response.ok`, so a 4xx/5xx is falsy while a
# MagicMock is always truthy. The mocked tests around this path stayed green
# through the bug that discarded every error body. `real_response` (tests/unit/
# conftest.py) builds the real thing.

# The refusal Radient actually returned on 2026-09-17: the provider's own words,
# in the body of a 500.
UPSTREAM_QUOTA_BODY = (
    b'{"error":"[internal] Transcription failed: OpenAI API error: OpenAI API error '
    b"(insufficient_quota): You have no credits remaining. Add credits to your plan to "
    b'continue."}'
)


def test_create_transcription_http_error_keeps_the_upstream_reason(
    radient_client: RadientClient,
    tmp_path: Path,
    real_response: Callable[[int, bytes], requests.Response],
) -> None:
    """A refused transcription carries the upstream status and the provider's words.

    The route classifies on those attributes, so losing either would take the
    failure's truthfulness with it: a status alone cannot tell a Radient edge
    refusal from a provider one, and the body is where the provider names itself.
    """
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")

    with patch("requests.post", MagicMock(return_value=real_response(500, UPSTREAM_QUOTA_BODY))):
        with pytest.raises(APIError) as exc_info:
            radient_client.create_transcription(file_path=str(audio_file))

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.body == UPSTREAM_QUOTA_BODY.decode()
    assert "500" in str(exc)
    assert "insufficient_quota" in str(exc)
    assert "You have no credits remaining" in str(exc)


def test_create_transcription_http_error_without_a_body_says_so(
    radient_client: RadientClient,
    tmp_path: Path,
    real_response: Callable[[int, bytes], requests.Response],
) -> None:
    """An empty 500 keeps the status and carries no body to quote.

    The status is what tells the two apart for a reader of the message: a 500
    that said nothing is not the same failure as a request that never arrived,
    and the latter reports no status at all.
    """
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")

    with patch("requests.post", MagicMock(return_value=real_response(500, b""))):
        with pytest.raises(APIError) as exc_info:
            radient_client.create_transcription(file_path=str(audio_file))

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.body is None
    assert "500" in str(exc)
    assert NO_RESPONSE_BODY in str(exc)


def test_create_transcription_network_failure_keeps_its_own_text(
    radient_client: RadientClient, tmp_path: Path
) -> None:
    """A request that never reached Radient carries the transport error alone."""
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")
    error = requests.exceptions.ConnectionError("Connection refused")

    with patch("requests.post", MagicMock(side_effect=error)):
        with pytest.raises(APIError) as exc_info:
            radient_client.create_transcription(file_path=str(audio_file))

    exc = exc_info.value
    assert exc.status_code is None
    assert exc.body is None
    assert "Connection refused" in str(exc)


def test_create_transcription_error_in_a_200_body_is_an_upstream_failure(
    radient_client: RadientClient,
    tmp_path: Path,
    real_response: Callable[[int, bytes], requests.Response],
) -> None:
    """Radient reports provider failures in a 200 body as well as in a 500.

    That branch used to be raised as a runtime error inside the try and then
    re-wrapped by the catch-all, so the message read "Failed to create
    transcription: Failed to create transcription: ..." and the shape of the
    failure was lost with it.
    """
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")

    with patch("requests.post", MagicMock(return_value=real_response(200, UPSTREAM_QUOTA_BODY))):
        with pytest.raises(APIError) as exc_info:
            radient_client.create_transcription(file_path=str(audio_file))

    exc = exc_info.value
    assert exc.status_code == 200
    assert exc.body == UPSTREAM_QUOTA_BODY.decode()
    assert str(exc).count("Failed to create transcription") == 1


def test_create_transcription_reports_a_non_json_body_as_upstream(
    radient_client: RadientClient,
    tmp_path: Path,
    real_response: Callable[[int, bytes], requests.Response],
) -> None:
    """A 200 that is not the documented shape is an upstream protocol failure."""
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")
    response = real_response(200, b"<html>gateway error</html>")

    with patch("requests.post", MagicMock(return_value=response)):
        with pytest.raises(APIError) as exc_info:
            radient_client.create_transcription(file_path=str(audio_file))

    exc = exc_info.value
    assert exc.status_code == 200
    assert "gateway error" in (exc.body or "")


def test_create_transcription_without_an_api_key_stays_internal(
    base_url: str, tmp_path: Path
) -> None:
    """The missing-key failure is ours, so it stays a plain RuntimeError.

    The route reports a plain RuntimeError as a 500 and a typed upstream error
    as a 502/402; anything typed here would blame Radient for our own
    misconfiguration.
    """
    audio_file = tmp_path / "sample.webm"
    audio_file.write_bytes(b"sample audio data")
    client = RadientClient(api_key=None, base_url=base_url)

    with pytest.raises(RuntimeError) as exc_info:
        client.create_transcription(file_path=str(audio_file))

    assert not isinstance(exc_info.value, APIError)
    assert "RADIENT_API_KEY is not configured" in str(exc_info.value)


# --- Instruction-set publication ----------------------------------------------
#
# The unit of publication is a bare JSON document (contract §1). The tests below
# pin three things that are easy to lose and expensive to discover late: the
# builder's closed field set, the client's transport shape, and the fact that a
# hub refusal arrives with its machine-readable code rather than folded into a
# sentence.


def test_build_instruction_set_document_is_the_closed_field_set() -> None:
    """A document carries the schema's fields and nothing else."""
    document = build_instruction_set_document(
        name="adverse-media-screener",
        description="Screens entities against adverse media.",
        instructions="You screen entities.",
        kind="role",
        version="1.0.0",
        when_to_use="Screening work.",
        tools=["read", "grep"],
        effort="hi",
        delegate=True,
        categories=["security"],
        tags=["osint"],
    )

    assert set(document) <= set(INSTRUCTION_SET_FIELDS)
    assert document == {
        "document_type": INSTRUCTION_SET_DOCUMENT_TYPE,
        "document_version": 1,
        "name": "adverse-media-screener",
        "description": "Screens entities against adverse media.",
        "instructions": "You screen entities.",
        "kind": "role",
        "when_to_use": "Screening work.",
        "tools": ["read", "grep"],
        "effort": "hi",
        "delegate": True,
        "version": "1.0.0",
        "categories": ["security"],
        "tags": ["osint"],
    }


def test_build_instruction_set_document_omits_absent_optionals() -> None:
    """An unspecified optional field is absent, not an empty string.

    "" is a value the hub stores and a client renders; absent is the absence the
    schema documents. `delegate` is the exception: a boolean states something
    either way, so it is always sent.
    """
    document = build_instruction_set_document(
        name="Coder",
        description="Writes code.",
        instructions="You write code.",
        kind="specialist",
        version="1.0.0",
    )

    assert "when_to_use" not in document
    assert "tools" not in document
    assert "effort" not in document
    assert "categories" not in document
    assert "tags" not in document
    assert document["delegate"] is False


@pytest.mark.parametrize(
    "overrides,field,rule",
    [
        ({"name": ""}, "name", "must not be empty"),
        ({"name": "a" * 129}, "name", "must be at most 128 characters"),
        # Whitespace that is also a control character is still refused, with the
        # hub's text; an ordinary space is not (see the deferral test below).
        ({"name": "two\twords"}, "name", "must not contain whitespace"),
        ({"name": "two\nwords"}, "name", "must not contain whitespace"),
        ({"name": "a/b"}, "name", 'must not contain "/", "\\" or ":"'),
        ({"name": "-leading"}, "name", 'must not begin or end with "-" or "."'),
        ({"name": "trailing."}, "name", 'must not begin or end with "-" or "."'),
        ({"name": "no\u202espam"}, "name", "must not contain Unicode bidirectional override"),
        ({"name": "bell\x07"}, "name", "must not contain control characters"),
        (
            {"name": "zero\u200bwidth"},
            "name",
            "must not contain invisible Unicode formatting characters",
        ),
        ({"description": "  "}, "description", "must not be empty"),
        ({"description": "d" * 2001}, "description", "must be at most 2000 characters"),
        ({"instructions": " "}, "instructions", "must not be empty"),
        ({"kind": "agent"}, "kind", 'must be "role" or "specialist"'),
        ({"when_to_use": "w" * 2001}, "when_to_use", "must be at most 2000 characters"),
        ({"tools": ["t" * 65]}, "tools", "must hold items of 1 to 64 characters"),
        ({"tools": [""]}, "tools", "must hold items of 1 to 64 characters"),
        ({"effort": "e" * 17}, "effort", "must be at most 16 characters"),
        ({"categories": ["not_a_category"]}, "categories", "must name categories"),
        ({"tags": ["t" * 65]}, "tags", "must hold items of 1 to 64 characters"),
        ({"version": " "}, "version", "must not be empty"),
    ],
)
def test_build_instruction_set_document_refuses_a_broken_field(
    overrides: Dict[str, Any], field: str, rule: str
) -> None:
    """Every rule the hub enforces is enforced here, with the hub's own words.

    The rule text is the value of `details.rule` on both sides, so a document
    refused locally and one refused by the hub read identically to a user.
    """
    fields: Dict[str, Any] = {
        "name": "Coder",
        "description": "Writes code.",
        "instructions": "You write code.",
        "kind": "role",
        "version": "1.0.0",
    }
    fields.update(overrides)

    with pytest.raises(InstructionSetError) as exc_info:
        build_instruction_set_document(**fields)

    assert exc_info.value.field == field
    assert rule in exc_info.value.rule
    assert exc_info.value.details == {"field": field, "rule": exc_info.value.rule}
    assert str(exc_info.value) == (
        f"The agent document is not valid: {field} {exc_info.value.rule}."
    )


def test_a_name_with_ordinary_spaces_is_sent_as_the_author_wrote_it() -> None:
    """Whitespace is the hub's decision, not this client's.

    agent-server refuses whitespace in a published name today and is relaxing
    exactly that (`dev-name-spaces`), because the live marketplace is already
    spelled with ordinary spaces. A client cannot mirror a rule that is moving:
    refusing an ordinary space here would refuse a name the hub is about to
    accept, so the spelling is sent as written and the hub normalises it — today
    it answers 422 with `details.field = "name"`, and this route carries that
    through unchanged.
    """
    document = build_instruction_set_document(
        name="Product Manager",
        description="Coordinates the roadmap.",
        instructions="You manage the product.",
        kind="role",
        version="1.0.0",
    )

    assert document["name"] == "Product Manager"


def test_a_name_is_trimmed_but_its_inner_spelling_is_kept() -> None:
    """The ends are the bound's business; the spelling in between is the hub's."""
    document = build_instruction_set_document(
        name="  Product  Manager  ",
        description="Coordinates the roadmap.",
        instructions="You manage the product.",
        kind="role",
        version="1.0.0",
    )

    assert document["name"] == "Product  Manager"


def test_build_instruction_set_document_counts_characters_not_bytes() -> None:
    """A non-Latin body is not refused at a fraction of its allowance."""
    document = build_instruction_set_document(
        name="コーダー",
        description="コーディングを行います。",
        instructions="指示" * 4000,
        kind="role",
        version="1.0.0",
    )

    assert len(document["instructions"]) == 8000


def test_build_instruction_set_document_refuses_an_over_long_body() -> None:
    """8000 is the profile's cap and the hub's cap: the same number, once."""
    with pytest.raises(InstructionSetError) as exc_info:
        build_instruction_set_document(
            name="Coder",
            description="Writes code.",
            instructions="x" * (MAX_INSTRUCTIONS_CHARS + 1),
            kind="role",
            version="1.0.0",
        )

    assert exc_info.value.field == "instructions"
    assert exc_info.value.rule == "must be at most 8000 characters"


def test_validate_document_overrides_refuses_an_unknown_field() -> None:
    """An undefined key is refused, not dropped.

    A pydantic model that dropped it would publish a document the caller did not
    describe, which is the failure mode the hub's own unknown-field rule exists
    to prevent.
    """
    assert validate_document_overrides({"description": "d"}) == {"description": "d"}

    with pytest.raises(InstructionSetError) as exc_info:
        validate_document_overrides({"description": "d", "conversation": []})

    assert exc_info.value.field == "conversation"
    assert exc_info.value.rule == "is not a recognised field"


def test_validate_document_overrides_refuses_the_schema_fields() -> None:
    """The client owns the schema it writes; a caller cannot declare one."""
    for key in ("document_type", "document_version"):
        with pytest.raises(InstructionSetError) as exc_info:
            validate_document_overrides({key: 1})
        assert exc_info.value.field == key


@pytest.mark.parametrize(
    "override,field,rule",
    [
        # The shapes that were COERCED before this check existed: the string
        # "false" is truthy, and list("osint") is five one-character tags, so both
        # published something the caller did not ask for, silently.
        ({"delegate": "false"}, "delegate", "must be true or false"),
        ({"delegate": 0}, "delegate", "must be true or false"),
        ({"delegate": None}, "delegate", "must be true or false"),
        ({"tags": "osint"}, "tags", "must be a list of strings"),
        ({"tools": "read"}, "tools", "must be a list of strings"),
        ({"tags": ["osint", 7]}, "tags", "must be a list of strings"),
        # And the shapes that escaped the builder as an AttributeError, which the
        # route reported as a 500 about this machine rather than a 422 about the
        # caller's request.
        ({"instructions": ["a"]}, "instructions", "must be a string"),
        ({"name": 42}, "name", "must be a string"),
        ({"when_to_use": None}, "when_to_use", "must be a string"),
    ],
)
def test_validate_document_overrides_refuses_a_value_of_the_wrong_shape(
    override: Dict[str, Any], field: str, rule: str
) -> None:
    """A known field carrying the wrong shape is refused in the hub's vocabulary.

    The rules are the client's, not the hub's, because the hub never sees these --
    it would refuse them at decode under `invalid_instruction_set`, which is what
    the route now answers. A client bound LOOSER than the server is the direction
    that publishes content the caller did not write.
    """
    with pytest.raises(InstructionSetError) as exc_info:
        validate_document_overrides(override)

    assert exc_info.value.field == field
    assert exc_info.value.rule == rule


def test_validate_document_overrides_accepts_the_shapes_that_are_right() -> None:
    """The control: the check refuses shapes, not values.

    ``delegate: False`` is the specific value that must survive it, since a check
    written as a truthiness test would refuse the one shape the wire uses for `no`.
    """
    overrides: Dict[str, Any] = {
        "delegate": False,
        "tags": ["osint"],
        "tools": ["read", "write"],
        "categories": ["software"],
        "instructions": "You help.",
        "name": "coder",
        "when_to_use": "Writing code.",
        "effort": "medium",
        "description": "Writes code.",
        "version": "1.0.0",
    }

    assert validate_document_overrides(overrides) == overrides
    # An empty list is a list: it publishes no tools, which is not a shape error.
    assert validate_document_overrides({"tags": []}) == {"tags": []}


def test_publish_agent_instruction_set_posts_the_document(
    radient_client: RadientClient, base_url: str
) -> None:
    """The document goes to the hub's publish endpoint as JSON, with the key."""
    document = build_instruction_set_document(
        name="Coder",
        description="Writes code.",
        instructions="You write code.",
        kind="role",
        version="1.0.0",
    )
    result = {"agent_id": "hub-1", "name": "Coder", "document_version": 1}
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"msg": "Agent published successfully", "result": result}

    with patch("requests.post", return_value=mock_response) as mock_post:
        published = radient_client.publish_agent_instruction_set(document)

    assert published == result
    args, kwargs = mock_post.call_args
    assert args[0] == f"{base_url}/agents/publish"
    assert kwargs["json"] == document
    assert kwargs["headers"]["Authorization"] == "Bearer test_api_key"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    # No client-side timeout: the hub reviews the submission with a model, and a
    # cutoff on a mutating request would leave it finishing a publication the
    # caller has walked away from.
    assert "timeout" not in kwargs


def test_republish_agent_instruction_set_puts_to_the_listing(
    radient_client: RadientClient, base_url: str
) -> None:
    """A republish addresses the HUB listing, not the local agent."""
    document = build_instruction_set_document(
        name="Coder",
        description="Writes code.",
        instructions="You write code.",
        kind="role",
        version="1.1.0",
    )
    result = {"agent_id": "hub-1", "name": "Coder", "version": "1.1.0"}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"msg": "Agent republished successfully", "result": result}

    with patch("requests.put", return_value=mock_response) as mock_put:
        republished = radient_client.republish_agent_instruction_set("hub-1", document)

    assert republished == result
    args, kwargs = mock_put.call_args
    assert args[0] == f"{base_url}/agents/hub-1/publish"
    assert kwargs["json"] == document


def test_check_agent_name_availability_needs_no_api_key(base_url: str) -> None:
    """The availability check is public, so it is callable before signing in."""
    client = RadientClient(api_key=None, base_url=base_url)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "msg": "Name availability checked",
        "result": {"name": "Coder", "name_key": "coder", "available": True},
    }

    with patch("requests.get", return_value=mock_response) as mock_get:
        availability = client.check_agent_name_availability("Coder")

    assert availability["available"] is True
    args, kwargs = mock_get.call_args
    assert args[0] == f"{base_url}/agent-name-availability"
    assert kwargs["params"] == {"name": "Coder"}
    assert "Authorization" not in kwargs["headers"]


@pytest.mark.parametrize(
    "status_code,code,details",
    [
        (409, "name_taken", {"existing_agent_id": "hub-9", "owned_by_caller": False}),
        (409, "name_reserved_builtin", {"builtin_name": "reviewer"}),
        (422, "moderation_rejected", {"categories": ["fraud_or_deception"]}),
        (422, "invalid_instruction_set", {"field": "kind", "rule": "must be"}),
        (413, "payload_too_large", {"limit_bytes": 65536}),
        (503, "moderation_unavailable", {"attempts": 2}),
        (403, "not_owner", {}),
        (404, "agent_not_found", {}),
    ],
)
def test_publish_raises_the_hub_code_and_details(
    radient_client: RadientClient, status_code: int, code: str, details: Dict[str, Any]
) -> None:
    """A refusal arrives as a code and its details, not as one prose string.

    This is the whole point of the structured error: the desktop app chooses a
    different next step for a taken name than for a moderation hold, and it cannot
    switch on a sentence.
    """
    document = {"document_type": INSTRUCTION_SET_DOCUMENT_TYPE, "document_version": 1}
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.content = json.dumps(
        {"error": "The hub refused this.", "code": code, "details": details}
    ).encode()
    http_error = requests.exceptions.HTTPError("refused", response=mock_response)

    with patch("requests.post", side_effect=http_error):
        with pytest.raises(APIError) as exc_info:
            radient_client.publish_agent_instruction_set(document)

    assert exc_info.value.code == code
    assert exc_info.value.status_code == status_code
    assert exc_info.value.details == details
    assert str(exc_info.value) == "The hub refused this."


def test_publish_error_does_not_carry_an_unrecognised_body_into_the_message(
    radient_client: RadientClient,
) -> None:
    """An error the hub did not describe is reported without its body.

    The body of a failure nobody designed is an HTML page or a stack trace from
    whatever answered instead, and interpolating it is how an intermediary's
    internals reach a desktop user and the log.
    """
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.content = b"<html><body>Traceback: SECRET_TOKEN=abc123</body></html>"

    with patch(
        "requests.post",
        side_effect=requests.exceptions.HTTPError("boom", response=mock_response),
    ):
        with pytest.raises(APIError) as exc_info:
            radient_client.publish_agent_instruction_set({"name": "Coder"})

    assert exc_info.value.code is None
    assert exc_info.value.status_code == 500
    assert exc_info.value.details == {}
    assert "SECRET_TOKEN" not in str(exc_info.value)
    assert "Traceback" not in str(exc_info.value)
    assert str(exc_info.value) == (
        "Could not publish the agent to the Radient Agent Hub (HTTP 500)"
    )


def test_publish_transport_failure_reports_no_status(radient_client: RadientClient) -> None:
    """A request that never got a response says so, rather than inventing one."""
    with patch(
        "requests.post",
        side_effect=requests.exceptions.ConnectionError("connection refused"),
    ):
        with pytest.raises(APIError) as exc_info:
            radient_client.publish_agent_instruction_set({"name": "Coder"})

    assert exc_info.value.status_code is None
    assert exc_info.value.code is None
    assert str(exc_info.value) == "Could not publish the agent to the Radient Agent Hub"


def test_publish_rejects_a_response_that_is_not_the_hub_envelope(
    radient_client: RadientClient,
) -> None:
    """A success without the hub's envelope is reported without its body."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"unexpected": "shape"}

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(APIError) as exc_info:
            radient_client.publish_agent_instruction_set({"name": "Coder"})

    assert "unrecognised response" in str(exc_info.value)
    assert exc_info.value.status_code == 200


def test_response_body_reports_a_4xx_body_it_used_to_call_absent() -> None:
    """The falsy bug: requests.Response.__bool__ is `ok`, so 4xx was "no body".

    Every error path is handed a 4xx/5xx response, which is precisely the case the
    old falsy test classified as bodyless.
    """
    response = MagicMock()
    response.status_code = 409
    response.content = b'{"error": "taken"}'
    exc = requests.exceptions.HTTPError("refused", response=response)

    assert response_body(exc) == '{"error": "taken"}'


def test_response_body_distinguishes_no_response_from_an_empty_one() -> None:
    """An absent response and an empty body both read as "none", not as ''."""
    assert response_body(requests.exceptions.ConnectionError("refused")) == NO_RESPONSE_BODY

    empty = MagicMock()
    empty.status_code = 500
    empty.content = b""
    assert response_body(requests.exceptions.HTTPError("boom", response=empty)) == NO_RESPONSE_BODY


def test_response_body_survives_a_body_that_is_not_utf8() -> None:
    """Reporting a failure must not raise from inside the error handler."""
    response = MagicMock()
    response.status_code = 502
    response.content = b"\xff\xfe not utf-8 \xff"

    assert "not utf-8" in response_body(requests.exceptions.HTTPError("boom", response=response))


# --- credentials in a surfaced upstream body ----------------------------------
#
# An upstream is free to reflect the request it received -- the Authorization
# header included -- into its error body, and the error body is exactly what a
# legacy failure message quotes. The e2e suite pins this against a fake upstream
# that does exactly that (tests/e2e/test_desktop_legacy_radient.py); these are the
# unit-level half, and they are what says which layer holds the property.


def test_an_upstream_body_that_reflects_the_key_is_redacted(
    radient_client: RadientClient, tmp_path: Path
) -> None:
    """A reflected credential never reaches the message a user or the log sees."""
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"dummy zip content")
    mock_response = MagicMock()
    mock_response.status_code = 400
    # The fake upstream's shape: the body quotes the credential back at us.
    mock_response.content = json.dumps({"error": "test_api_key"}).encode()

    with patch(
        "requests.post",
        side_effect=requests.exceptions.HTTPError("refused", response=mock_response),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            radient_client.upload_agent_to_marketplace(zip_path)

    message = str(exc_info.value)
    assert "test_api_key" not in message
    assert "[redacted]" in message


def test_publish_prose_that_reflects_the_key_is_redacted(
    radient_client: RadientClient,
) -> None:
    """The hub's own `error` prose is filtered too, because it is rendered."""
    mock_response = MagicMock()
    mock_response.status_code = 409
    mock_response.content = json.dumps(
        {
            "error": 'The name "test_api_key" is already published on the hub.',
            "code": "name_taken",
            "details": {"existing_agent_id": "hub-9", "owned_by_caller": False},
        }
    ).encode()

    with patch(
        "requests.post",
        side_effect=requests.exceptions.HTTPError("refused", response=mock_response),
    ):
        with pytest.raises(APIError) as exc_info:
            radient_client.publish_agent_instruction_set({"name": "Coder"})

    assert "test_api_key" not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)
    # The machine-readable half is the hub's, unchanged: the renderer switches on
    # these values and they are not prose.
    assert exc_info.value.code == "name_taken"
    assert exc_info.value.details == {"existing_agent_id": "hub-9", "owned_by_caller": False}


def test_redact_secrets_leaves_text_alone_without_a_secret() -> None:
    """A client with no credential configured changes nothing it surfaces."""
    assert redact_secrets("plain body", [None, ""]) == "plain body"
    assert redact_secrets("plain body", ["key"]) == "plain body"


# A provider failure reported inside a 200 body, in the shape Radient uses on
# this path too (`test_create_transcription_error_in_a_200_body_is_an_upstream_failure`
# pins the transcription sibling). The echoed credential is what makes it matter
# here: the speech call returns bytes, so an envelope like this used to be served
# to the daemon's own client as audio -- a payload path no `HTTPException`
# handler ever sees.
SPEECH_ERROR_IN_A_200_BODY = (
    b'{"error":"Incorrect API key provided: radient-key-with-no-published-shape-4a91"}'
)


def test_create_speech_returns_the_audio_body_unchanged(
    radient_client: RadientClient, real_response: Callable[[int, bytes], requests.Response]
) -> None:
    """Audio is not an envelope, and the error guard must not eat it.

    Both shapes a real upstream sends are covered: a text-ish fixture (what the
    end-to-end stub answers) and a body opening with an mp3 frame sync, which is
    what a real encoder emits and must never be parsed as JSON.
    """
    with patch("requests.post", MagicMock(return_value=real_response(200, b"fixture-audio"))):
        assert (
            radient_client.create_speech(input_text="hello", model="tts-1", voice="alloy")
            == b"fixture-audio"
        )

    frame_sync_audio = b"\xff\xfb\x90\x00" + bytes(range(64))
    with patch("requests.post", MagicMock(return_value=real_response(200, frame_sync_audio))):
        assert (
            radient_client.create_speech(input_text="hello", model="tts-1", voice="alloy")
            == frame_sync_audio
        )


def test_create_speech_treats_a_200_error_body_as_an_upstream_failure(
    radient_client: RadientClient, real_response: Callable[[int, bytes], requests.Response]
) -> None:
    """A 2xx that is an error envelope is an upstream failure, not audio.

    Radient reports provider failures in a 200 body, and this call returns its
    bytes to the route, which streams them with an audio media type. Typed as an
    upstream failure, the route reports it as one instead of serving the
    envelope -- and the credential an upstream echoed into it -- as audio.
    """
    with patch(
        "requests.post",
        MagicMock(return_value=real_response(200, SPEECH_ERROR_IN_A_200_BODY)),
    ):
        with pytest.raises(APIError) as exc_info:
            radient_client.create_speech(input_text="hello", model="tts-1", voice="alloy")

    exc = exc_info.value
    assert exc.status_code == 200
    assert exc.body == SPEECH_ERROR_IN_A_200_BODY.decode()
    # Exactly once: a second wrap would mean the catch-all swallowed the typed
    # error and the status and body the route reads were lost with it.
    assert str(exc).count("Failed to generate speech") == 1


def test_create_speech_treats_a_json_content_type_as_an_upstream_failure(
    radient_client: RadientClient, real_response: Callable[[int, bytes], requests.Response]
) -> None:
    """A JSON content type is not audio whatever the body does or does not parse as.

    A gateway can answer a 200 with a content type that names JSON and a body
    that is not valid JSON; the media type alone is enough to know the bytes are
    not the audio the caller asked for.
    """
    response = real_response(200, b"quota exceeded")
    response.headers["Content-Type"] = "application/json"

    with patch("requests.post", MagicMock(return_value=response)):
        with pytest.raises(APIError) as exc_info:
            radient_client.create_speech(input_text="hello", model="tts-1", voice="alloy")

    assert exc_info.value.status_code == 200
    assert exc_info.value.body == "quota exceeded"
