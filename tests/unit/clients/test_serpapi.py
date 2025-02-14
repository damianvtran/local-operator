from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from local_operator.clients.serpapi import (
    SerpApiClient,
    SerpApiResponse,
    SerpApiSearchInformation,
    SerpApiSearchMetadata,
    SerpApiSearchParameters,
)


@pytest.fixture
def api_key() -> str:
    """Fixture for providing a test API key."""
    return "test_api_key"


@pytest.fixture
def serp_client(api_key: str) -> SerpApiClient:
    """Fixture for creating a SerpApiClient instance.

    Args:
        api_key (str): API key for the client.

    Returns:
        SerpApiClient: An instance of SerpApiClient.
    """
    return SerpApiClient(api_key=api_key)


@pytest.fixture
def mock_response() -> Dict[str, Any]:
    """Fixture for providing a mock JSON response from the SERP API.

    Returns:
        Dict[str, Any]: Mock JSON data that simulates a SERP API response.
    """
    return {
        "search_metadata": {
            "id": "test_id",
            "status": "Success",
            "json_endpoint": "https://serpapi.com/searches/test.json",
            "created_at": "2023-01-01 00:00:00 UTC",
            "processed_at": "2023-01-01 00:00:01 UTC",
            "google_url": "https://www.google.com/search?q=test",
            "raw_html_file": "https://serpapi.com/searches/test.html",
            "total_time_taken": 1.23,
        },
        "search_parameters": {
            "engine": "google",
            "q": "test query",
            "location_requested": "New York",
            "location_used": "New York,New York,United States",
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "device": "desktop",
        },
        "search_information": {
            "organic_results_state": "Results for exact spelling",
            "query_displayed": "test query",
            "total_results": 1000000,
            "time_taken_displayed": 0.5,
        },
        "recipes_results": None,
        "shopping_results": None,
        "local_results": None,
        "organic_results": None,
        "related_searches": None,
        "pagination": None,
    }


def test_search_success(serp_client: SerpApiClient, mock_response: Dict[str, Any]) -> None:
    """Test successful API search request.

    Args:
        serp_client (SerpApiClient): The SERP API client fixture.
        mock_response (Dict[str, Any]): Mock JSON response.
    """
    mock_requests_get = Mock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_response

    with patch("requests.get", mock_requests_get):
        response = serp_client.search("test query")

    assert isinstance(response, SerpApiResponse)
    assert isinstance(response.search_metadata, SerpApiSearchMetadata)
    assert isinstance(response.search_parameters, SerpApiSearchParameters)
    assert isinstance(response.search_information, SerpApiSearchInformation)
    assert response.search_metadata.id == "test_id"
    assert response.search_parameters.q == "test query"
    assert response.recipes_results is None
    assert response.shopping_results is None
    assert response.local_results is None
    assert response.organic_results is None
    assert response.related_searches is None
    assert response.pagination is None


def test_search_api_error(serp_client: SerpApiClient) -> None:
    """Test handling of API error response.

    Args:
        serp_client (SerpApiClient): The SERP API client fixture.
    """
    mock_requests_get = Mock()
    mock_requests_get.return_value.status_code = 400

    with patch("requests.get", mock_requests_get):
        with pytest.raises(RuntimeError) as exc_info:
            serp_client.search("test query")
        assert "SERP API request failed with status 400" in str(exc_info.value)


def test_search_network_error(serp_client: SerpApiClient) -> None:
    """Test handling of network error.

    Args:
        serp_client (SerpApiClient): The SERP API client fixture.
    """
    mock_requests_get = Mock(side_effect=Exception("Network error"))

    with patch("requests.get", mock_requests_get):
        with pytest.raises(RuntimeError) as exc_info:
            serp_client.search("test query")
        assert "Failed to execute SERP API search: Network error" in str(exc_info.value)


def test_client_init_no_api_key() -> None:
    """Test client initialization with missing API key.

    Raises:
        RuntimeError: If no API key is provided.
    """
    with pytest.raises(RuntimeError) as exc_info:
        SerpApiClient("")
    assert "SERP API key must be provided" in str(exc_info.value)


def test_search_with_optional_params(
    serp_client: SerpApiClient, mock_response: Dict[str, Any]
) -> None:
    """Test search with all optional parameters.

    Args:
        serp_client (SerpApiClient): The SERP API client fixture.
        mock_response (Dict[str, Any]): Mock JSON response.
    """
    mock_requests_get = Mock()
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.json.return_value = mock_response

    with patch("requests.get", mock_requests_get):
        response = serp_client.search(
            query="test query",
            engine="google",
            num_results=30,
            location="New York",
            language="en",
            country="us",
            device="mobile",
        )

    assert isinstance(response, SerpApiResponse)
    # Verify that the URL contains all expected parameters
    called_url = mock_requests_get.call_args[0][0]
    assert "location=New+York" in called_url
    assert "hl=en" in called_url
    assert "gl=us" in called_url
    assert "device=mobile" in called_url
    assert "num=30" in called_url
