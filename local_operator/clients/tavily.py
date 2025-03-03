from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, SecretStr


class TavilyResult(BaseModel):
    """Individual search result from Tavily API.

    Attributes:
        title (str): Title of the search result
        url (str): URL of the search result
        content (str): Snippet or summary of the content
        score (float): Relevance score of the result
        raw_content (str | None): Full content of the result if requested
    """

    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable.

        Returns:
            Dict[str, Any]: A JSON serializable dictionary representation of the model.
        """
        return super().model_dump(*args, **kwargs)


class TavilyResponse(BaseModel):
    """Complete response from Tavily API search.

    Attributes:
        query (str): The original search query
        results (List[TavilyResult]): List of search results
        follow_up_questions (List[str] | None): Suggested follow-up questions if any
        answer (str | None): Generated answer if requested
        images (List[Dict[str, Any]] | None): Image results if any
        response_time (float | None): Time taken to process the request
    """

    query: str
    results: List[TavilyResult]
    follow_up_questions: Optional[List[str]] = None
    answer: Optional[str] = None
    images: Optional[List[Dict[str, Any]]] = None
    response_time: Optional[float] = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable.

        Returns:
            Dict[str, Any]: A JSON serializable dictionary representation of the model.
        """
        return super().model_dump(*args, **kwargs)


class TavilyClient:
    """Client for making requests to the Tavily API.

    Attributes:
        api_key (SecretStr): Tavily API key for authentication
        base_url (str): Base URL for the Tavily API
    """

    def __init__(self, api_key: SecretStr, base_url: str = "https://api.tavily.com"):
        """Initialize the Tavily API client.

        Args:
            api_key (SecretStr): Tavily API key for authentication
            base_url (str, optional): Base URL for the Tavily API.
            Defaults to "https://api.tavily.com".

        Raises:
            RuntimeError: If no API key is provided.
        """
        self.api_key = api_key
        self.base_url = base_url
        if not self.api_key:
            raise RuntimeError("Tavily API key must be provided")

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        max_results: int = 10,
    ) -> TavilyResponse:
        """Execute a search using the Tavily API.

        Makes an HTTP request to Tavily API with the provided parameters and
        returns a structured response.

        Args:
            query (str): The search query string
            search_depth (str, optional): Depth of search - "basic" or "advanced".
            Defaults to "basic".
            include_domains (List[str], optional): List of domains to include in search.
            Defaults to None.
            exclude_domains (List[str], optional): List of domains to exclude from search.
            Defaults to None.
            include_answer (bool, optional): Whether to include a generated answer.
            Defaults to False.
            include_raw_content (bool, optional): Whether to include full content of
            results. Defaults to False.
            include_images (bool, optional): Whether to include image results. Defaults to False.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            TavilyResponse: Structured search results from Tavily API

        Raises:
            RuntimeError: If the API request fails
        """
        # Build request payload
        payload = {
            "query": query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "max_results": max_results,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        url = f"{self.base_url}/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Tavily API request failed with status {response.status_code}, content:"
                    f" {response.content.decode()}"
                )
            data = response.json()
            return TavilyResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to execute Tavily API search due to a requests error: {str(e)}, Response"
                f" Body: {(
                    e.response.content.decode()
                    if hasattr(e, 'response') and e.response
                    else "No response body"
                )}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to execute Tavily API search: {str(e)}") from e
