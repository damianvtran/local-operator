"""
The Ollama client for Local Operator.

This module provides a client for interacting with the Ollama API to run
local language models.
"""

from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field


class OllamaModelData(BaseModel):
    """Data for an Ollama model.

    Attributes:
        name (str): Name of the model.
        modified_at (str): Timestamp when the model was last modified.
        size (int): Size of the model in bytes.
        digest (str): Unique digest of the model.
        details (Dict[str, Any]): Additional details about the model.
    """

    name: str
    modified_at: str
    size: int
    digest: str
    details: Optional[Dict[str, Any]] = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class OllamaGetTagsResponse(BaseModel):
    """Response from Ollama API's /api/tags endpoint.

    Attributes:
        models (List[Dict[str, Any]]): List of model data returned by the API.
    """

    models: List[Dict[str, Any]] = Field(default_factory=list)


class OllamaClient:
    """Client for interacting with the Ollama API.

    This client is used to check the health of the Ollama server and list available models.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        """Initializes the OllamaClient.

        Args:
            base_url (str): The base URL for the Ollama API.
        """
        self.base_url = base_url

    def is_healthy(self) -> bool:
        """Checks if the Ollama server is running and healthy.

        Returns:
            bool: True if the server is healthy, False otherwise.
        """
        try:
            # Based on testing, the root endpoint returns "Ollama is running" when healthy
            response = requests.get(self.base_url, timeout=2)
            return response.status_code == 200 and "Ollama is running" in response.text
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[OllamaModelData]:
        """Lists all available models on the Ollama server.

        Returns:
            List[OllamaModelData]: A list of available models.

        Raises:
            RuntimeError: If the API request fails or the Ollama server is not healthy.
        """
        if not self.is_healthy():
            raise RuntimeError("Ollama server is not healthy")

        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            tags_response = OllamaGetTagsResponse.model_validate(response.json())
            return [OllamaModelData.model_validate(model) for model in tags_response.models]
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            error_msg = f"Failed to fetch Ollama models due to a requests error: {str(e)}"
            error_msg += f", Response Body: {error_body}"
            raise RuntimeError(error_msg) from e
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Ollama models: {str(e)}") from e
