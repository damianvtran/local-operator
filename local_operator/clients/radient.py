from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, SecretStr


# Image Generation Models
class RadientImage(BaseModel):
    """Image information returned by the Radient API.

    Attributes:
        url (str): URL of the generated image
        width (Optional[int]): Width of the image in pixels
        height (Optional[int]): Height of the image in pixels
    """

    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientImageGenerationResponse(BaseModel):
    """Response from the Radient API for image generation.

    Attributes:
        request_id (str): ID of the request
        status (str): Status of the request (e.g., "completed", "processing")
        images (Optional[List[RadientImage]]): List of generated images if available
    """

    request_id: str
    status: str
    images: Optional[List[RadientImage]] = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientImageGenerationProvider(BaseModel):
    """Information about an image generation provider.

    Attributes:
        id (str): Unique identifier for the provider
        name (str): Name of the provider
        description (str): Description of the provider
    """

    id: str
    name: str
    description: str
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientImageGenerationProvidersResponse(BaseModel):
    """Response from the Radient API for listing image generation providers.

    Attributes:
        providers (List[RadientImageGenerationProvider]): List of available providers
    """

    providers: List[RadientImageGenerationProvider]
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


# Web Search Models
class RadientSearchResult(BaseModel):
    """Individual search result from Radient API.

    Attributes:
        title (str): Title of the search result
        url (str): URL of the search result
        content (str): Snippet or summary of the content
        raw_content (Optional[str]): Full content of the result if requested
    """

    title: str
    url: str
    content: str
    raw_content: Optional[str] = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientSearchResponse(BaseModel):
    """Complete response from Radient API search.

    Attributes:
        query (str): The original search query
        results (List[RadientSearchResult]): List of search results
    """

    query: str
    results: List[RadientSearchResult]
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientSearchProvider(BaseModel):
    """Information about a web search provider.

    Attributes:
        id (str): Unique identifier for the provider
        name (str): Name of the provider
        description (str): Description of the provider
    """

    id: str
    name: str
    description: str
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientSearchProvidersResponse(BaseModel):
    """Response from the Radient API for listing web search providers.

    Attributes:
        providers (List[RadientSearchProvider]): List of available providers
    """

    providers: List[RadientSearchProvider]
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


# Model Pricing Models
class RadientModelPricing(BaseModel):
    """Pricing information for a Radient model.

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


class RadientModelData(BaseModel):
    """Data for a Radient model.

    Attributes:
        id (str): Unique identifier for the model.
        name (str): Name of the model.
        description (str): Description of the model.
        pricing (RadientModelPricing): Pricing information for the model.
    """

    id: str
    name: str
    description: str
    pricing: RadientModelPricing
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientListModelsResponse(BaseModel):
    """Response from the Radient list models API.

    Attributes:
        data (list[RadientModelData]): List of Radient models.
    """

    data: List[RadientModelData]
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientClient:
    """Client for interacting with the Radient API.

    This client is used to fetch model pricing information from Radient.
    """

    def __init__(self, api_key: SecretStr, base_url: str) -> None:
        """Initializes the RadientClient.

        Args:
            api_key (SecretStr | None): The Radient API key. If None, it is assumed that
                the key is not needed for the specific operation (e.g., listing models).
            base_url (str): The base URL for the Radient API.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.app_title = "Local Operator"
        self.http_referer = "https://local-operator.com"

        if not self.api_key:
            raise RuntimeError("Radient API key is required")

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for the Radient API request.

        Returns:
            Dict[str, str]: Headers for the API request
        """
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Title": self.app_title,
            "HTTP-Referer": self.http_referer,
        }

    def list_models(self) -> RadientListModelsResponse:
        """Lists all available models on Radient along with their pricing.

        Returns:
            RadientListModelsResponse: A list of available models and their pricing information.

        Raises:
            RuntimeError: If the API request fails.
        """
        url = f"{self.base_url}/models"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            return RadientListModelsResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to fetch Radient models due to a requests error: {str(e)}, Response"
                f" Body: {e.response.content.decode() if e.response else 'No response body'}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Radient models: {str(e)}") from e

    # Image Generation Methods

    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        image_size: str = "1024x1024",
        source_url: Optional[str] = None,
        strength: Optional[float] = None,
        sync_mode: bool = True,
        provider: Optional[str] = None,
    ) -> RadientImageGenerationResponse:
        """Generate an image using the Radient API.

        Args:
            prompt (str): The prompt to generate an image from
            num_images (int, optional): Number of images to generate. Defaults to 1.
            image_size (str, optional): Size of the generated image. Defaults to "1024x1024".
            source_url (Optional[str], optional): URL of the image to use as a base for
                image-to-image generation. Defaults to None.
            strength (Optional[float], optional): Strength parameter for image-to-image generation.
                Defaults to None.
            sync_mode (bool, optional): Whether to wait for the result. Defaults to True.
            provider (Optional[str], optional): The provider to use. Defaults to None.

        Returns:
            RadientImageGenerationResponse: The generated image information

        Raises:
            RuntimeError: If the API request fails
        """
        url = f"{self.base_url}/tools/images/generate"
        headers = self._get_headers()

        payload = {
            "prompt": prompt,
            "num_images": num_images,
            "image_size": image_size,
            "sync_mode": sync_mode,
        }

        if source_url:
            payload["source_url"] = source_url

        if strength is not None:
            payload["strength"] = strength

        if provider:
            payload["provider"] = provider

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return RadientImageGenerationResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to generate image: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {str(e)}") from e

    def get_image_generation_status(
        self, request_id: str, provider: Optional[str] = None
    ) -> RadientImageGenerationResponse:
        """Get the status of an image generation request.

        Args:
            request_id (str): ID of the request
            provider (Optional[str], optional): The provider to use. Defaults to None.

        Returns:
            RadientImageGenerationResponse: Status of the request

        Raises:
            RuntimeError: If the API request fails
        """
        url = f"{self.base_url}/tools/images/status"
        headers = self._get_headers()

        payload = {
            "request_id": request_id,
        }

        if provider:
            payload["provider"] = provider

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return RadientImageGenerationResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to get image generation status: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to get image generation status: {str(e)}") from e

    def list_image_generation_providers(self) -> RadientImageGenerationProvidersResponse:
        """List available image generation providers.

        Returns:
            RadientImageGenerationProvidersResponse: List of available providers

        Raises:
            RuntimeError: If the API request fails
        """
        url = f"{self.base_url}/tools/images/providers"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return RadientImageGenerationProvidersResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to list image generation providers: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to list image generation providers: {str(e)}") from e

    # Web Search Methods

    def search(
        self,
        query: str,
        max_results: int = 10,
        provider: Optional[str] = None,
        include_raw: bool = False,
        search_depth: Optional[str] = None,
        domains: Optional[List[str]] = None,
    ) -> RadientSearchResponse:
        """Execute a web search using the Radient API.

        Args:
            query (str): The search query string
            max_results (int, optional): Maximum number of results to return. Defaults to 10.
            provider (Optional[str], optional): The provider to use. Defaults to None.
            include_raw (bool, optional): Whether to include full content of results.
                Defaults to False.
            search_depth (Optional[str], optional): Depth of search. Defaults to None.
            domains (Optional[List[str]], optional): List of domains to include in search.
                Defaults to None.

        Returns:
            RadientSearchResponse: Structured search results from Radient API

        Raises:
            RuntimeError: If the API request fails
        """
        url = f"{self.base_url}/tools/search"
        headers = self._get_headers()

        payload = {
            "query": query,
            "max_results": max_results,
            "include_raw": include_raw,
        }

        if provider:
            payload["provider"] = provider

        if search_depth:
            payload["search_depth"] = search_depth

        if domains:
            payload["domains"] = domains

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return RadientSearchResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to execute search: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to execute search: {str(e)}") from e

    def list_search_providers(self) -> RadientSearchProvidersResponse:
        """List available web search providers.

        Returns:
            RadientSearchProvidersResponse: List of available providers

        Raises:
            RuntimeError: If the API request fails
        """
        url = f"{self.base_url}/tools/search/providers"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return RadientSearchProvidersResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to list search providers: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to list search providers: {str(e)}") from e
