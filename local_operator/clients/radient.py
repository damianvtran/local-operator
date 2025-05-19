import time
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, SecretStr


class ImageSize(str, Enum):
    """Image size options for the FAL API."""

    SQUARE_HD = "square_hd"
    SQUARE = "square"
    PORTRAIT_4_3 = "portrait_4_3"
    PORTRAIT_16_9 = "portrait_16_9"
    LANDSCAPE_4_3 = "landscape_4_3"
    LANDSCAPE_16_9 = "landscape_16_9"


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


# Email Sending Models
class RadientSendEmailRequest(BaseModel):
    """Request model for sending an email to self.

    Attributes:
        subject (str): The subject of the email.
        body (str): The body of the email (can be HTML or plain text).
    """

    subject: str
    body: str
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientSendEmailResponseData(BaseModel):
    """Data part of the response when sending an email to self.

    Attributes:
        message (str): Confirmation message.
        message_id (Optional[str]): Message ID from the email provider, if available.
    """

    message: str
    message_id: Optional[str] = None
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientSendEmailAPIResponse(BaseModel):
    """Overall API response structure for sending an email.

    Attributes:
        result (RadientSendEmailResponseData): The actual email sending result.
        error (Optional[str]): Error message if any.
        msg (Optional[str]): Additional message if any.
    """

    result: Optional[RadientSendEmailResponseData] = None
    error: Optional[str] = None
    msg: Optional[str] = None
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


# Token Refresh Models
class RadientTokenRefreshRequest(BaseModel):
    """Request model for refreshing an access token.

    Attributes:
        client_id (str): The client ID.
        grant_type (str): The grant type, typically "refresh_token".
        refresh_token (SecretStr): The refresh token.
    """

    client_id: str
    grant_type: str = "refresh_token"
    refresh_token: SecretStr
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        payload = super().model_dump(*args, **kwargs)
        payload["refresh_token"] = self.refresh_token.get_secret_value()
        return payload


class RadientTokenResponse(BaseModel):
    """Response model for token-related operations.

    Attributes:
        access_token (SecretStr): The new access token.
        expires_in (int): The lifetime in seconds of the access token.
        token_type (str): The token type, typically "Bearer".
        refresh_token (Optional[SecretStr]): The new refresh token, if issued.
        id_token (Optional[SecretStr]): The ID token (OpenID Connect).
        scope (Optional[str]): The scope of the access token.
    """

    access_token: SecretStr
    expires_in: int
    token_type: str
    refresh_token: Optional[SecretStr] = None
    id_token: Optional[SecretStr] = None
    scope: Optional[str] = None
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        payload = super().model_dump(*args, **kwargs)
        if self.access_token:
            payload["access_token"] = self.access_token.get_secret_value()
        if self.refresh_token:
            payload["refresh_token"] = self.refresh_token.get_secret_value()
        if self.id_token:
            payload["id_token"] = self.id_token.get_secret_value()
        return payload


class RadientTokenRefreshAPIResponse(BaseModel):
    """Overall API response structure for token refresh.

    Attributes:
        msg (str): Confirmation or status message.
        result (RadientTokenResponse): The actual token refresh result.
    """

    msg: str
    result: RadientTokenResponse
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientClient:
    """Client for interacting with the Radient API.

    This client is used to fetch model pricing information from Radient and
    interact with the Radient Agent Hub.
    """

    def __init__(self, api_key: Optional[SecretStr], base_url: str) -> None:
        """Initializes the RadientClient.

        Args:
            api_key (SecretStr | None): The Radient API key. If None, it is assumed that
                the key is not needed for the specific operation (e.g., listing
                models or downloading agents).
            base_url (str): The base URL for the Radient API.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.app_title = "Local Operator"
        self.http_referer = "https://local-operator.com"

    def _get_headers(
        self, content_type: Optional[str] = "application/json", require_api_key: bool = True
    ) -> Dict[str, str]:
        """Get the headers for the Radient API request.

        Args:
            content_type (Optional[str]): The Content-Type header value. If
            None, Content-Type is not set.
            require_api_key (bool): Whether to require the API key for this request.

        Returns:
            Dict[str, str]: Headers for the API request

        Raises:
            RuntimeError: If the API key is required but not set.
        """
        headers = {
            "X-Title": self.app_title,
            "HTTP-Referer": self.http_referer,
        }
        if require_api_key:
            if not self.api_key:
                raise RuntimeError("Radient API key is required for this operation")
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        if content_type is not None:
            headers["Content-Type"] = content_type
        return headers

    def upload_agent_to_marketplace(self, zip_path) -> str:
        """
        Upload a new agent to the Radient Agent Hub.

        Args:
            zip_path (Path): Path to the ZIP file containing agent data.

        Returns:
            str: The new agent ID returned by the marketplace.

        Raises:
            RuntimeError: If the API key is not set or the upload fails.
        """
        url = f"{self.base_url}/agents/upload"
        headers = self._get_headers(content_type=None, require_api_key=True)
        files = {"file": (zip_path.name, open(zip_path, "rb"), "application/zip")}

        try:
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            data = response.json()
            # The response is a dict with the new agent ID (e.g., {"id": "new-agent-id"})
            if not isinstance(data, dict) or not data:
                raise RuntimeError("Unexpected response from Radient agent upload")
            # Return the first value (agent ID)
            return next(iter(data.values()))
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to upload agent to Radient Agent Hub: {str(e)},"
                f"Response Body: {error_body}"
            ) from e
        finally:
            files["file"][1].close()

    def overwrite_agent_in_marketplace(self, agent_id: str, zip_path) -> None:
        """
        Overwrite an existing agent in the Radient Agent Hub.

        Args:
            agent_id (str): The agent ID to overwrite.
            zip_path (Path): Path to the ZIP file containing agent data.

        Raises:
            RuntimeError: If the API key is not set or the upload fails.
        """
        url = f"{self.base_url}/agents/{agent_id}/upload"
        headers = self._get_headers(content_type=None, require_api_key=True)
        files = {"file": (zip_path.name, open(zip_path, "rb"), "application/zip")}
        try:
            response = requests.put(url, headers=headers, files=files)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to overwrite agent in Radient Agent Hub: {str(e)},"
                f"Response Body: {error_body}"
            ) from e
        finally:
            files["file"][1].close()

    def download_agent_from_marketplace(self, agent_id: str, dest_path) -> None:
        """
        Download an agent from the Radient Agent Hub.

        Args:
            agent_id (str): The agent ID to download.
            dest_path (Path): Path to save the downloaded ZIP file.

        Raises:
            RuntimeError: If the download fails.
        """
        url = f"{self.base_url}/agents/{agent_id}/download"
        # Download does not require API key
        headers = self._get_headers(content_type=None, require_api_key=False)
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to download agent from Radient Agent Hub: {str(e)},"
                f"Response Body: {error_body}"
            ) from e

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent details from the Radient Agent Hub by ID.

        Args:
            agent_id (str): The agent ID to fetch.

        Returns:
            Optional[Dict[str, Any]]: The agent details as a dictionary if found,
                                      None if the agent is not found (404).

        Raises:
            RuntimeError: If the API request fails for reasons other than 404.
        """
        url = f"{self.base_url}/v1/agents/{agent_id}"
        # This is a public endpoint, no API key required
        headers = self._get_headers(content_type="application/json", require_api_key=False)
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None  # Agent not found
            # For other HTTP errors, raise a runtime error
            error_body = e.response.content.decode() if e.response.content else "No response body"
            raise RuntimeError(
                f"Failed to get agent {agent_id} from Radient Agent Hub: "
                f"HTTP {e.response.status_code}, Response Body: {error_body}"
            ) from e
        except requests.exceptions.RequestException as e:
            # For non-HTTP request errors (e.g., connection issues)
            raise RuntimeError(
                f"Failed to get agent {agent_id} from Radient Agent Hub due to a network error: "
                f"{str(e)}"
            ) from e

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
                f"Failed to fetch Radient models due to a requests error: {str(e)},"
                f"Response Body: {
                    e.response.content.decode() if e.response else 'No response body'
                }"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Radient models: {str(e)}") from e

    # Image Generation Methods

    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        image_size: str = "square_hd",
        source_url: Optional[str] = None,
        strength: Optional[float] = None,
        sync_mode: bool = True,  # This parameter is passed to the Radient API
        provider: Optional[str] = None,
        max_wait_time: int = 60,
        poll_interval: int = 2,
    ) -> RadientImageGenerationResponse:
        """Generate an image using the Radient API.

        Args:
            prompt (str): The prompt to generate an image from
            num_images (int, optional): Number of images to generate. Defaults to 1.
            image_size (str, optional): Size of the generated image. Defaults to
                "square_hd".
            source_url (Optional[str], optional): URL of the image to use as a base for
                image-to-image generation. Defaults to None.
            strength (Optional[float], optional): Strength parameter for image-to-image generation.
                Defaults to None.
            sync_mode (bool, optional): Whether to use sync_mode in the Radient API request.
                This affects how the Radient API handles the request but our function
                will always wait for the result. Defaults to True.
            provider (Optional[str], optional): The provider to use. Defaults to None.
            max_wait_time (int, optional): Maximum time to wait for image generation in seconds.
                Defaults to 60.
            poll_interval (int, optional): Time between status checks in seconds. Defaults to 2.

        Returns:
            RadientImageGenerationResponse: The generated image information with complete image data

        Raises:
            RuntimeError: If the API request fails or times out
        """
        url = f"{self.base_url}/tools/images/generate"
        headers = self._get_headers()

        # Use the sync_mode parameter as provided
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
            # Submit the initial request
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            result = RadientImageGenerationResponse.model_validate(data)

            # If the result already has images, return it immediately
            if result.images and len(result.images) > 0:
                return result

            # Otherwise, poll for the result
            request_id = result.request_id
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                # Get the current status
                status_response = self.get_image_generation_status(
                    request_id=request_id, provider=provider
                )

                # If the status is completed and we have images, return the result
                if status_response.status.upper() == "COMPLETED" and status_response.images:
                    return status_response

                # If the status is failed, raise an error
                if status_response.status.upper() == "FAILED":
                    raise RuntimeError(f"Radient API image generation failed: {request_id}")

                # Wait before polling again
                time.sleep(poll_interval)

            # If we get here, the request timed out
            raise RuntimeError(
                f"Radient API image generation timed out after {max_wait_time} seconds: "
                f"{request_id}"
            )

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

        params = {"request_id": request_id}

        if provider:
            params["provider"] = provider

        try:
            response = requests.get(url, headers=headers, params=params)
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

        # Build query parameters
        params = {
            "query": query,
            "max_results": max_results,
            "include_raw": str(include_raw).lower(),
        }

        if provider:
            params["provider"] = provider

        if search_depth:
            params["search_depth"] = search_depth

        if domains:
            params["domains"] = ",".join(domains)

        try:
            response = requests.get(url, headers=headers, params=params)
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

    def delete_agent_from_marketplace(self, agent_id: str) -> None:
        """
        Delete an agent from the Radient Agent Hub by ID.

        Args:
            agent_id (str): The agent ID to delete.

        Raises:
            RuntimeError: If the API key is not set or the delete fails.
        """
        url = f"{self.base_url}/agents/{agent_id}"
        headers = self._get_headers(content_type=None, require_api_key=True)
        try:
            response = requests.delete(url, headers=headers)
            if response.status_code == 204:
                return
            # If not 204, try to extract error details
            error_body = response.content.decode() if response.content else "No response body"
            raise RuntimeError(
                f"Failed to delete agent from Radient Agent Hub: HTTP {response.status_code}, "
                f"Response Body: {error_body}"
            )
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to delete agent from Radient Agent Hub: {str(e)}, "
                f"Response Body: {error_body}"
            ) from e

    def send_email_to_self(self, subject: str, body: str) -> RadientSendEmailResponseData:
        """Send an email to the authenticated user's email address.

        Args:
            subject (str): The subject of the email.
            body (str): The body of the email (can be HTML or plain text).

        Returns:
            RadientSendEmailResponseData: Response data containing confirmation message.

        Raises:
            RuntimeError: If the API key is not set or the request fails.
        """
        url = f"{self.base_url}/email/self/send"  # Corrected path based on OpenAPI
        headers = self._get_headers(require_api_key=True)
        payload = RadientSendEmailRequest(subject=subject, body=body).dict()

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            api_response_data = response.json()
            api_response = RadientSendEmailAPIResponse.model_validate(api_response_data)

            if api_response.error:
                raise RuntimeError(
                    f"Failed to send email: {api_response.error} - {api_response.msg}"
                )
            if not api_response.result:
                raise RuntimeError("Failed to send email: No result data in response.")
            return api_response.result
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to send email: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to send email: {str(e)}") from e

    # Token Refresh Methods
    def refresh_token(
        self, client_id: str, refresh_token: SecretStr, provider: Optional[str] = None
    ) -> RadientTokenResponse:
        """Refresh an access token using a refresh token.

        Args:
            client_id (str): The client ID.
            refresh_token (SecretStr): The refresh token.
            provider (Optional[str]): The provider ("google" or "microsoft").
                                      If None, uses the generic /auth/token endpoint.

        Returns:
            RadientTokenResponse: The new token information.

        Raises:
            RuntimeError: If the API request fails.
        """
        if provider and provider.lower() == "google":
            url = f"{self.base_url}/auth/google/refresh"
        elif provider and provider.lower() == "microsoft":
            url = f"{self.base_url}/auth/microsoft/refresh"
        else:
            url = f"{self.base_url}/auth/token"

        headers = self._get_headers(require_api_key=False)  # Refresh usually doesn't need API key
        payload = RadientTokenRefreshRequest(
            client_id=client_id, refresh_token=refresh_token
        ).dict()

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            api_response_data = response.json()
            api_response = RadientTokenRefreshAPIResponse.model_validate(api_response_data)

            # The actual token data is in api_response.result
            return api_response.result
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to refresh token: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to refresh token: {str(e)}") from e
