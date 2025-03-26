from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.clients.fal import (
    FalClient,
    FalImage,
    FalImageGenerationResponse,
    FalRequestStatus,
    ImageSize,
)


@pytest.fixture
def api_key() -> SecretStr:
    """Fixture for providing a test API key."""
    return SecretStr("test_api_key")


@pytest.fixture
def fal_client(api_key: SecretStr) -> FalClient:
    """Fixture for creating a FalClient instance.

    Args:
        api_key (SecretStr): API key for the client.

    Returns:
        FalClient: An instance of FalClient.
    """
    return FalClient(api_key=api_key)


@pytest.fixture
def mock_request_status() -> Dict[str, Any]:
    """Fixture for providing a mock request status response."""
    return {
        "request_id": "test-request-id",
        "status": "processing",
    }


@pytest.fixture
def mock_completed_request_status() -> Dict[str, Any]:
    """Fixture for providing a mock completed request status response."""
    return {
        "request_id": "test-request-id",
        "status": "completed",
    }


@pytest.fixture
def mock_image_generation_response() -> Dict[str, Any]:
    """Fixture for providing a mock image generation response."""
    return {
        "images": [
            {
                "url": "https://example.com/image.jpg",
                "width": 1024,
                "height": 768,
                "content_type": "image/jpeg",
            }
        ],
        "prompt": "test prompt",
        "seed": 42,
        "has_nsfw_concepts": [False],
    }


def test_client_init(api_key: SecretStr) -> None:
    """Test client initialization.

    Args:
        api_key (SecretStr): Test API key.
    """
    client = FalClient(api_key=api_key)
    assert client.api_key == api_key
    assert client.base_url == "https://queue.fal.run"
    assert client.model_path == "fal-ai/flux/dev"


def test_client_init_custom_base_url(api_key: SecretStr) -> None:
    """Test client initialization with custom base URL.

    Args:
        api_key (SecretStr): Test API key.
    """
    custom_url = "https://custom.fal.run"
    client = FalClient(api_key=api_key, base_url=custom_url)
    assert client.base_url == custom_url


def test_client_init_no_api_key() -> None:
    """Test client initialization with missing API key."""
    with pytest.raises(ValueError) as exc_info:
        FalClient(api_key=SecretStr(""))
    assert "FAL API key is required" in str(exc_info.value)


def test_get_headers(fal_client: FalClient) -> None:
    """Test the _get_headers method.

    Args:
        fal_client (FalClient): The FAL client fixture.
    """
    headers = fal_client._get_headers()
    assert headers["Authorization"] == "Key test_api_key"
    assert headers["Content-Type"] == "application/json"


def test_submit_request_success(fal_client: FalClient, mock_request_status: Dict[str, Any]) -> None:
    """Test successful request submission.

    Args:
        fal_client (FalClient): The FAL client fixture.
        mock_request_status (Dict[str, Any]): Mock request status response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_request_status

    with patch("requests.post", return_value=mock_response):
        result = fal_client._submit_request({"prompt": "test prompt"})

    assert isinstance(result, FalRequestStatus)
    assert result.request_id == "test-request-id"
    assert result.status == "processing"


def test_submit_request_error(fal_client: FalClient) -> None:
    """Test error handling in request submission.

    Args:
        fal_client (FalClient): The FAL client fixture.
    """
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.content = b"Bad Request"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_response
    )

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc_info:
            fal_client._submit_request({"prompt": "test prompt"})
        assert "Failed to submit FAL API request" in str(exc_info.value)


def test_get_request_status_success(
    fal_client: FalClient, mock_request_status: Dict[str, Any]
) -> None:
    """Test successful request status retrieval.

    Args:
        fal_client (FalClient): The FAL client fixture.
        mock_request_status (Dict[str, Any]): Mock request status response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_request_status

    with patch("requests.get", return_value=mock_response):
        result = fal_client._get_request_status("test-request-id")

    assert isinstance(result, FalRequestStatus)
    assert result.request_id == "test-request-id"
    assert result.status == "processing"


def test_get_request_status_error(fal_client: FalClient) -> None:
    """Test error handling in request status retrieval.

    Args:
        fal_client (FalClient): The FAL client fixture.
    """
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.content = b"Bad Request"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_response
    )

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc_info:
            fal_client._get_request_status("test-request-id")
        assert "Failed to get FAL API request status" in str(exc_info.value)


def test_get_request_result_success(
    fal_client: FalClient, mock_image_generation_response: Dict[str, Any]
) -> None:
    """Test successful request result retrieval.

    Args:
        fal_client (FalClient): The FAL client fixture.
        mock_image_generation_response (Dict[str, Any]): Mock image generation response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_image_generation_response

    with patch("requests.get", return_value=mock_response):
        result = fal_client._get_request_result("test-request-id")

    assert isinstance(result, FalImageGenerationResponse)
    assert len(result.images) == 1
    assert isinstance(result.images[0], FalImage)
    assert result.images[0].url == "https://example.com/image.jpg"
    assert result.images[0].width == 1024
    assert result.images[0].height == 768
    assert result.prompt == "test prompt"
    assert result.seed == 42
    assert result.has_nsfw_concepts == [False]


def test_get_request_result_error(fal_client: FalClient) -> None:
    """Test error handling in request result retrieval.

    Args:
        fal_client (FalClient): The FAL client fixture.
    """
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.content = b"Bad Request"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_response
    )

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc_info:
            fal_client._get_request_result("test-request-id")
        assert "Failed to get FAL API request result" in str(exc_info.value)


def test_generate_image_sync_mode_success(
    fal_client: FalClient, mock_image_generation_response: Dict[str, Any]
) -> None:
    """Test successful image generation in sync mode.

    Args:
        fal_client (FalClient): The FAL client fixture.
        mock_image_generation_response (Dict[str, Any]): Mock image generation response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_image_generation_response

    with patch("requests.post", return_value=mock_response):
        result = fal_client.generate_image(
            prompt="test prompt",
            image_size=ImageSize.LANDSCAPE_4_3,
            num_inference_steps=28,
            seed=42,
            guidance_scale=3.5,
            sync_mode=True,
            num_images=1,
            enable_safety_checker=True,
        )

    assert isinstance(result, FalImageGenerationResponse)
    assert len(result.images) == 1
    assert result.images[0].url == "https://example.com/image.jpg"
    assert result.prompt == "test prompt"
    assert result.seed == 42


def test_generate_image_sync_mode_error(fal_client: FalClient) -> None:
    """Test error handling in sync mode image generation.

    Args:
        fal_client (FalClient): The FAL client fixture.
    """
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.content = b"Bad Request"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_response
    )

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc_info:
            fal_client.generate_image(prompt="test prompt", sync_mode=True)
        assert "Failed to generate image" in str(exc_info.value)


def test_generate_image_async_mode_success(
    fal_client: FalClient,
    mock_request_status: Dict[str, Any],
) -> None:
    """Test successful image generation in async mode.

    Args:
        fal_client (FalClient): The FAL client fixture.
        mock_request_status (Dict[str, Any]): Mock request status response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_request_status

    with patch("requests.post", return_value=mock_response):
        result = fal_client.generate_image(
            prompt="test prompt",
            image_size=ImageSize.LANDSCAPE_4_3,
            num_inference_steps=28,
            seed=42,
            guidance_scale=3.5,
            sync_mode=False,
            num_images=1,
            enable_safety_checker=True,
        )

    assert isinstance(result, FalRequestStatus)
    assert result.request_id == "test-request-id"
    assert result.status == "processing"


def test_generate_image_async_mode_error(fal_client: FalClient) -> None:
    """Test error handling in async mode image generation.

    Args:
        fal_client (FalClient): The FAL client fixture.
    """
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.content = b"Bad Request"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Bad Request", response=mock_response
    )

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError) as exc_info:
            fal_client.generate_image(prompt="test prompt", sync_mode=False)
        assert "Failed to submit FAL API request" in str(exc_info.value)


def test_generate_image_wait_for_completion(
    fal_client: FalClient,
    mock_request_status: Dict[str, Any],
    mock_completed_request_status: Dict[str, Any],
    mock_image_generation_response: Dict[str, Any],
) -> None:
    """Test image generation with waiting for completion.

    Args:
        fal_client (FalClient): The FAL client fixture.
        mock_request_status (Dict[str, Any]): Mock request status response.
        mock_completed_request_status (Dict[str, Any]): Mock completed request status response.
        mock_image_generation_response (Dict[str, Any]): Mock image generation response.
    """
    # Mock the submit request response
    mock_submit_response = MagicMock()
    mock_submit_response.status_code = 200
    mock_submit_response.json.return_value = mock_request_status

    # Mock the get status response (first processing, then completed)
    mock_status_response1 = MagicMock()
    mock_status_response1.status_code = 200
    mock_status_response1.json.return_value = mock_request_status

    mock_status_response2 = MagicMock()
    mock_status_response2.status_code = 200
    mock_status_response2.json.return_value = mock_completed_request_status

    # Mock the get result response
    mock_result_response = MagicMock()
    mock_result_response.status_code = 200
    mock_result_response.json.return_value = mock_image_generation_response

    # Set up the mocks to be returned in sequence
    with patch("requests.post", return_value=mock_submit_response):
        with patch(
            "requests.get",
            side_effect=[mock_status_response1, mock_status_response2, mock_result_response],
        ):
            with patch("time.sleep", return_value=None):  # Mock sleep to speed up test
                result = fal_client.generate_image(
                    prompt="test prompt", sync_mode=True, max_wait_time=10, poll_interval=1
                )

    assert isinstance(result, FalImageGenerationResponse)
    assert len(result.images) == 1
    assert result.images[0].url == "https://example.com/image.jpg"
    assert result.prompt == "test prompt"
    assert result.seed == 42


def test_generate_image_timeout(
    fal_client: FalClient,
    mock_request_status: Dict[str, Any],
) -> None:
    """Test image generation timeout.

    Args:
        fal_client (FalClient): The FAL client fixture.
        mock_request_status (Dict[str, Any]): Mock request status response.
    """
    # Mock the submit request response
    mock_submit_response = MagicMock()
    mock_submit_response.status_code = 200
    mock_submit_response.json.return_value = mock_request_status

    # Mock the get status response (always processing)
    mock_status_response = MagicMock()
    mock_status_response.status_code = 200
    mock_status_response.json.return_value = mock_request_status

    # Set up the mocks
    with patch("requests.post", return_value=mock_submit_response):
        with patch("requests.get", return_value=mock_status_response):
            with patch("time.sleep", return_value=None):  # Mock sleep to speed up test
                with pytest.raises(RuntimeError) as exc_info:
                    fal_client.generate_image(
                        prompt="test prompt", sync_mode=True, max_wait_time=2, poll_interval=1
                    )
                assert "FAL API request timed out after 2 seconds" in str(exc_info.value)


def test_generate_image_failed_status(
    fal_client: FalClient,
    mock_request_status: Dict[str, Any],
) -> None:
    """Test image generation with failed status.

    Args:
        fal_client (FalClient): The FAL client fixture.
        mock_request_status (Dict[str, Any]): Mock request status response.
    """
    # Mock the submit request response
    mock_submit_response = MagicMock()
    mock_submit_response.status_code = 200
    mock_submit_response.json.return_value = mock_request_status

    # Mock the get status response (failed)
    mock_status_response = MagicMock()
    mock_status_response.status_code = 200
    mock_status_response.json.return_value = {"request_id": "test-request-id", "status": "failed"}

    # Set up the mocks
    with patch("requests.post", return_value=mock_submit_response):
        with patch("requests.get", return_value=mock_status_response):
            with patch("time.sleep", return_value=None):  # Mock sleep to speed up test
                with pytest.raises(RuntimeError) as exc_info:
                    fal_client.generate_image(
                        prompt="test prompt", sync_mode=True, max_wait_time=10, poll_interval=1
                    )
                assert "FAL API request failed" in str(exc_info.value)
