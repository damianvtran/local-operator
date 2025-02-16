from typing import Any, Dict, List

import requests
from pydantic import BaseModel, SecretStr


class OpenRouterModelPricing(BaseModel):
    """Pricing information for an OpenRouter model.

    Attributes:
        prompt (float): Cost per 1000 tokens for prompt processing.
        completion (float): Cost per 1000 tokens for completion generation.
    """

    prompt: float
    completion: float
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class OpenRouterModelData(BaseModel):
    """Data for an OpenRouter model.

    Attributes:
        id (str): Unique identifier for the model.
        name (str): Name of the model.
        description (str): Description of the model.
        pricing (OpenRouterModelPricing): Pricing information for the model.
    """

    id: str
    name: str
    description: str
    pricing: OpenRouterModelPricing
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class OpenRouterListModelsResponse(BaseModel):
    """Response from the OpenRouter list models API.

    Attributes:
        data (list[OpenRouterModelData]): List of OpenRouter models.
    """

    data: List[OpenRouterModelData]
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class OpenRouterClient:
    """Client for interacting with the OpenRouter API.

    This client is used to fetch model pricing information from OpenRouter.
    """

    def __init__(self, api_key: SecretStr, base_url: str = "https://openrouter.ai/api/v1") -> None:
        """Initializes the OpenRouterClient.

        Args:
            api_key (SecretStr | None): The OpenRouter API key. If None, it is assumed that
                the key is not needed for the specific operation (e.g., listing models).
            base_url (str): The base URL for the OpenRouter API.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.app_title = "Local Operator"
        self.http_referer = "https://local-operator.com"

        if not self.api_key:
            raise RuntimeError("OpenRouter API key is required")

    def list_models(self) -> OpenRouterListModelsResponse:
        """Lists all available models on OpenRouter along with their pricing.

        Returns:
            OpenRouterListModelsResponse: A list of available models and their pricing information.

        Raises:
            RuntimeError: If the API request fails.
        """
        url = f"{self.base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Title": self.app_title,
            "HTTP-Referer": self.http_referer,
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            return OpenRouterListModelsResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to fetch OpenRouter models due to a requests error: {str(e)}, Response"
                f" Body: {e.response.content.decode() if e.response else 'No response body'}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to fetch OpenRouter models: {str(e)}") from e
