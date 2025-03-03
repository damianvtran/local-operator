from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

from local_operator.clients.tavily import TavilyClient, TavilyResponse, TavilyResult


@pytest.fixture
def api_key() -> SecretStr:
    """Fixture for providing a test API key."""
    return SecretStr("test_api_key")


@pytest.fixture
def tavily_client(api_key: SecretStr) -> TavilyClient:
    """Fixture for creating a TavilyClient instance.

    Args:
        api_key (SecretStr): API key for the client.

    Returns:
        TavilyClient: An instance of TavilyClient.
    """
    return TavilyClient(api_key=api_key)


@pytest.fixture
def mock_response() -> Dict[str, Any]:
    """Fixture for providing a mock JSON response from the Tavily API.

    Returns:
        Dict[str, Any]: Mock JSON data that simulates a Tavily API response.
    """
    return {
        "query": "What are the latest updates with agentic AI?",
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": [
            {
                "title": "The Current State and Future of Agentic AI - Sikich",
                "url": (
                    "https://www.sikich.com/insight/the-current-state-and-future-of-agentic-ai-"
                    "insights-and-innovations/"
                ),
                "content": "The Current State and Future of Agentic AI - Sikich Services",
                "score": 0.7336813,
                "raw_content": None,
            },
            {
                "title": "AutoGen v0.4: Reimagining the foundation of agentic AI",
                "url": "https://www.microsoft.com/en-us/research/articles/autogen-v0-4/",
                "content": "AutoGen v0.4: Reimagining the foundation of agentic AI",
                "score": 0.68748367,
                "raw_content": None,
            },
        ],
        "response_time": 2.17,
    }


def test_search_success(tavily_client: TavilyClient, mock_response: Dict[str, Any]) -> None:
    """Test successful API search request.

    Args:
        tavily_client (TavilyClient): The Tavily API client fixture.
        mock_response (Dict[str, Any]): Mock JSON response.
    """
    mock_requests_post = Mock()
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = mock_response

    with patch("requests.post", mock_requests_post):
        response = tavily_client.search("What are the latest updates with agentic AI?")

    assert isinstance(response, TavilyResponse)
    assert response.query == "What are the latest updates with agentic AI?"
    assert response.follow_up_questions is None
    assert response.answer is None
    assert len(response.results) == 2
    assert isinstance(response.results[0], TavilyResult)
    assert response.results[0].title == "The Current State and Future of Agentic AI - Sikich"
    assert response.results[0].score == 0.7336813
    assert response.response_time == 2.17


def test_search_api_error(tavily_client: TavilyClient) -> None:
    """Test handling of API error response.

    Args:
        tavily_client (TavilyClient): The Tavily API client fixture.
    """
    mock_requests_post = Mock()
    mock_requests_post.return_value.status_code = 400
    mock_requests_post.return_value.content = b"Bad Request"

    with patch("requests.post", mock_requests_post):
        with pytest.raises(RuntimeError) as exc_info:
            tavily_client.search("test query")
        assert "Tavily API request failed with status 400" in str(exc_info.value)


def test_search_network_error(tavily_client: TavilyClient) -> None:
    """Test handling of network error.

    Args:
        tavily_client (TavilyClient): The Tavily API client fixture.
    """
    mock_requests_post = Mock(side_effect=Exception("Network error"))

    with patch("requests.post", mock_requests_post):
        with pytest.raises(RuntimeError) as exc_info:
            tavily_client.search("test query")
        assert "Failed to execute Tavily API search: Network error" in str(exc_info.value)


def test_client_init_no_api_key() -> None:
    """Test client initialization with missing API key.

    Raises:
        RuntimeError: If no API key is provided.
    """
    with pytest.raises(RuntimeError) as exc_info:
        TavilyClient(SecretStr(""))
    assert "Tavily API key must be provided" in str(exc_info.value)


def test_search_with_optional_params(
    tavily_client: TavilyClient, mock_response: Dict[str, Any]
) -> None:
    """Test search with all optional parameters.

    Args:
        tavily_client (TavilyClient): The Tavily API client fixture.
        mock_response (Dict[str, Any]): Mock JSON response.
    """
    mock_requests_post = Mock()
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = mock_response

    with patch("requests.post", mock_requests_post):
        response = tavily_client.search(
            query="test query",
            search_depth="advanced",
            include_domains=["example.com"],
            exclude_domains=["exclude.com"],
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            max_results=5,
        )

    assert isinstance(response, TavilyResponse)
    # Verify that the payload contains all expected parameters
    called_args = mock_requests_post.call_args[1]["json"]
    assert called_args["query"] == "test query"
    assert called_args["search_depth"] == "advanced"
    assert called_args["include_domains"] == ["example.com"]
    assert called_args["exclude_domains"] == ["exclude.com"]
    assert called_args["include_answer"] is True
    assert called_args["include_raw_content"] is True
    assert called_args["include_images"] is True
    assert called_args["max_results"] == 5
