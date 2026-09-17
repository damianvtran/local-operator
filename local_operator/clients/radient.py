import json
import time
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import requests
from pydantic import BaseModel, SecretStr

from local_operator.agent_profiles import MAX_INSTRUCTIONS_CHARS
from local_operator.agents import MAX_AGENT_NAME_CHARS
from local_operator.clients._http import (
    APIError,
    api_error_from_exception,
    api_error_from_response,
    redact_secrets,
    response_body,
    scrubbed_response_body,
)


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


# Transcription Models
class RadientTranscriptionResponseData(BaseModel):
    """Data part of the response when creating a transcription.

    Attributes:
        text (str): The transcribed text from the audio.
        provider (str): The name of the provider that performed the transcription.
        status (str): The status of the transcription request.
        error (Optional[str]): An error message if the transcription failed.
        duration (Optional[float]): The duration of the transcribed audio in seconds.
    """

    text: str
    provider: str
    status: str
    error: Optional[str] = None
    duration: Optional[float] = None
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class RadientTranscriptionAPIResponse(BaseModel):
    """Overall API response structure for creating a transcription.

    Attributes:
        result (Optional[RadientTranscriptionResponseData]): The actual transcription result.
        error (Optional[str]): Error message if any.
        msg (Optional[str]): Additional message if any.
    """

    result: Optional[RadientTranscriptionResponseData] = None
    error: Optional[str] = None
    msg: Optional[str] = None
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


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


def _is_an_error_envelope(response: requests.Response) -> bool:
    """Is this successful response actually an error the upstream reported?

    Radient reports a provider failure in a 200 body as often as in an error
    status (the transcription path pins that behaviour), and this client's
    speech call hands its bytes straight back as audio. Audio never parses as
    JSON -- an mp3 opens with a frame sync or ``ID3``, a wav with ``RIFF``, an
    ogg stream with ``OggS`` -- so a JSON object or array here is an error
    envelope rather than a payload, whatever the content type claims.

    Args:
        response: A response whose status is already known to be 2xx.

    Returns:
        bool: True when the body is an error envelope, not audio.
    """
    if "json" in response.headers.get("Content-Type", "").lower():
        return True
    content = response.content
    # Cheap first: audio opens with a frame sync, `ID3`, `RIFF` or `OggS`, so a
    # body that does not open like JSON at all never needs decoding. Eight bytes
    # is the window because an envelope may be preceded by whitespace.
    if content[:8].lstrip()[:1] not in (b"{", b"["):
        return False
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return isinstance(payload, (dict, list))


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

    def _credential_values(self) -> List[str]:
        """This client's credential values, for removal from anything surfaced.

        One accessor so a second credential this client might hold is added in one
        place rather than at every message that quotes an upstream failure.
        """

        return [self.api_key.get_secret_value()] if self.api_key else []

    def _surfaceable_body(self, body: str) -> str:
        """An upstream error body with this client's credential removed.

        An upstream is free to reflect the request it received -- the Authorization
        header included -- into its error body, and an error body is exactly what a
        useful failure message quotes. Quoting it verbatim would put the operator's
        credential into a message the desktop app renders and the log keeps, so
        anything about to be surfaced goes through here first. The body is kept
        otherwise: it is the only thing that says WHY a legacy call failed.

        Args:
            body: The decoded body about to be interpolated into a message.

        Returns:
            The body with this client's API key replaced by a marker.
        """

        return redact_secrets(body, self._credential_values())

    def upload_agent_to_marketplace(self, zip_path: Path) -> str:
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
            error_body = self._surfaceable_body(response_body(e))
            raise RuntimeError(
                f"Failed to upload agent to Radient Agent Hub: {str(e)},"
                f"Response Body: {error_body}"
            ) from e
        finally:
            files["file"][1].close()

    def overwrite_agent_in_marketplace(self, agent_id: str, zip_path: Path) -> None:
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
            error_body = self._surfaceable_body(response_body(e))
            raise RuntimeError(
                f"Failed to overwrite agent in Radient Agent Hub: {str(e)},"
                f"Response Body: {error_body}"
            ) from e
        finally:
            files["file"][1].close()

    def download_agent_from_marketplace(self, agent_id: str, dest_path: Path) -> None:
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
            error_body = self._surfaceable_body(response_body(e))
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
            # raise_for_status() always attaches the failing response, but HTTPError can
            # also be raised without one, in which case there is nothing to inspect and
            # this is reported the same way as any other response-less request failure.
            error_response = e.response
            if error_response is None:
                raise RuntimeError(
                    f"Failed to get agent {agent_id} from Radient Agent Hub due to a "
                    f"network error: {str(e)}"
                ) from e
            if error_response.status_code == 404:
                return None  # Agent not found
            # For other HTTP errors, raise a runtime error
            error_body = self._surfaceable_body(scrubbed_response_body(error_response))
            raise RuntimeError(
                f"Failed to get agent {agent_id} from Radient Agent Hub: "
                f"HTTP {error_response.status_code}, Response Body: {error_body}"
            ) from e
        except requests.exceptions.RequestException as e:
            # For non-HTTP request errors (e.g., connection issues)
            raise RuntimeError(
                f"Failed to get agent {agent_id} from Radient Agent Hub due to a network error: "
                f"{str(e)}"
            ) from e

    def publish_agent_instruction_set(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        """Publish an agent to the Radient Agent Hub as an instruction-set document.

        The document is the version-1 JSON document of
        :func:`build_instruction_set_document` — a bare instruction set, not a zip.
        The legacy zip methods on this class stay as they are: they are how an
        agent published before this standard is updated and pulled, and an older
        desktop build still pushes through them.

        No client-side timeout, deliberately. The hub reviews the submission with a
        model before it accepts it (two attempts at 20s each is a legitimate
        duration), and a cutoff on a MUTATING request would leave the hub finishing
        a publication the caller has walked away from — the next attempt then sees
        its own name taken by the row that landed. A transport failure still
        surfaces as an :class:`APIError`; it is only the caller's own impatience
        that is not allowed to cancel the request.

        Args:
            document: The instruction-set document to publish.

        Returns:
            The hub's publication result (``agent_id``, ``name``, ``version``,
            ``document_version`` and the review that admitted it).

        Raises:
            APIError: When the hub refuses the publication. ``code`` and
                ``details`` carry the hub's machine-readable refusal (contract
                §2.4) — ``name_taken``, ``name_reserved_builtin``,
                ``moderation_rejected``, ``moderation_unavailable``,
                ``invalid_instruction_set``, ``payload_too_large`` — so the caller
                can render a different next step for each without reading prose.
            RuntimeError: When no API key is configured for this client.
        """
        url = f"{self.base_url}/agents/publish"
        headers = self._get_headers(content_type="application/json")
        try:
            response = requests.post(url, headers=headers, json=dict(document))
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise api_error_from_response(
                e.response,
                fallback_message="Could not publish the agent to the Radient Agent Hub",
                secrets=self._credential_values(),
            ) from e
        return self._publication_result(response, action="publish the agent")

    def republish_agent_instruction_set(
        self, agent_id: str, document: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Update an already-published agent with a new instruction-set document.

        Only the account that published the listing may: the hub answers
        ``403 not_owner`` for anyone else, with that code carrying the reason.

        Args:
            agent_id: The id of the HUB listing to update (not a local agent id;
                the local registry keeps no link to the listing a row was
                published as, so the caller names it).
            document: The instruction-set document to send.

        Returns:
            The hub's publication result for the updated listing.

        Raises:
            APIError: As :meth:`publish_agent_instruction_set`, plus
                ``not_owner`` (403) and ``agent_not_found`` (404).
            RuntimeError: When no API key is configured for this client.
        """
        url = f"{self.base_url}/agents/{agent_id}/publish"
        headers = self._get_headers(content_type="application/json")
        try:
            response = requests.put(url, headers=headers, json=dict(document))
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise api_error_from_response(
                e.response,
                fallback_message="Could not update the agent on the Radient Agent Hub",
                secrets=self._credential_values(),
            ) from e
        return self._publication_result(response, action="update the agent")

    def check_agent_name_availability(self, name: str) -> Dict[str, Any]:
        """Ask the hub whether a name can be published, before submitting one.

        Public on the hub — no API key — and advisory: it is answered from a point
        lookup, so a name reported available can still be taken by the time the
        publication lands. It exists so the desktop app can say "already
        published" while the user types rather than after they submit.

        Args:
            name: The name as the user typed it (the hub trims and normalises).

        Returns:
            ``{name, name_key, available, code?, details?}``. A name that cannot be
            published is a SUCCESSFUL answer: the question was "is this available",
            and ``available: false`` answers it, with ``code`` saying why
            (``name_taken`` / ``name_reserved_builtin``). The hub returns it as an
            error only when the name itself breaks the name rules.

        Raises:
            APIError: When the hub refuses the request itself (an illegal name, or a
                hub failure).
        """
        url = f"{self.base_url}/agent-name-availability"
        headers = self._get_headers(content_type="application/json", require_api_key=False)
        try:
            response = requests.get(url, headers=headers, params={"name": name})
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise api_error_from_response(
                e.response,
                fallback_message="Could not check the agent name on the Radient Agent Hub",
                secrets=self._credential_values(),
            ) from e
        return self._publication_result(response, action="check the agent name")

    def _publication_result(self, response: requests.Response, *, action: str) -> Dict[str, Any]:
        """Unwrap the hub's ``{msg, result}`` envelope around a publication response.

        The envelope is the hub's API-wide response shape, so a successful call that
        does not carry one means something other than the hub answered. That is
        reported as an :class:`APIError` WITHOUT the body: a response shape we do not
        recognise is exactly the case where the body may be someone else's HTML or
        an echo of the request, and neither belongs in a user-facing message.

        Args:
            response: The successful response.
            action: What the caller was trying to do, for the message.

        Returns:
            The ``result`` object.

        Raises:
            APIError: When the response is not the hub's envelope.
        """
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("result"), dict):
            raise APIError(
                "The Radient Agent Hub returned an unrecognised response while trying "
                f"to {action} (HTTP {response.status_code}).",
                status_code=response.status_code,
            )
        return body["result"]

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
                f"Response Body: {self._surfaceable_body(response_body(e))}"
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
            error_body = self._surfaceable_body(response_body(e))
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
            error_body = self._surfaceable_body(response_body(e))
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
            error_body = self._surfaceable_body(response_body(e))
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
            error_body = self._surfaceable_body(response_body(e))
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
            error_body = self._surfaceable_body(response_body(e))
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
            error_body = self._surfaceable_body(scrubbed_response_body(response))
            raise RuntimeError(
                f"Failed to delete agent from Radient Agent Hub: HTTP {response.status_code}, "
                f"Response Body: {error_body}"
            )
        except requests.exceptions.RequestException as e:
            error_body = self._surfaceable_body(response_body(e))
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
            error_body = self._surfaceable_body(response_body(e))
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
            error_body = self._surfaceable_body(response_body(e))
            raise RuntimeError(
                f"Failed to refresh token: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to refresh token: {str(e)}") from e

    # Transcription Methods
    def create_transcription(
        self,
        file_path: str,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = "json",
        temperature: Optional[float] = 0.0,
        language: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> RadientTranscriptionResponseData:
        """Create an audio transcription using the Radient API.

        Args:
            file_path (str): Path to the audio file to transcribe.
            model (Optional[str]): Transcription model id. Defaults to None, which
                                   sends no `model` field and leaves the choice to
                                   the Radient agent-server's configured default.
            prompt (Optional[str]): Optional text prompt to guide the model. Max 1000 chars.
            response_format (Optional[str]): Format of the response ('json', 'text', 'srt',
                                             'verbose_json', 'vtt'). Defaults to "json".
            temperature (Optional[float]): Sampling temperature (0-2). Defaults to 0.0.
            language (Optional[str]): Language of audio in ISO-639-1 format (e.g., "en").
            provider (Optional[str]): Transcription provider. Defaults to None, which
                                      sends no `provider` field and leaves the choice
                                      to the Radient agent-server's configured
                                      default. Most providers require a `model` id
                                      they actually serve, but nothing here enforces
                                      the pairing: whatever the caller passes is
                                      forwarded, and no model is invented for it.

        Returns:
            RadientTranscriptionResponseData: The transcription result.

        Raises:
            RuntimeError: If the API key is not set, or for an unforeseen internal
                fault. Deliberately a *plain* RuntimeError: the daemon's route
                reports it as a 500, so anything upstream-shaped must not use it.
            APIError: If Radient or the provider it called rejected the request.
                Subclasses RuntimeError, so existing handlers keep catching it,
                and carries the upstream status and body so the route can pick a
                status its own caller can act on.
            ValueError: If input parameters are invalid.
            FileNotFoundError: If the audio file does not exist.
        """
        if not self.api_key:
            raise RuntimeError("RADIENT_API_KEY is not configured. Cannot create transcription.")

        url = f"{self.base_url}/tools/transcriptions"
        # Headers for multipart/form-data will be set by requests library,
        # but we still need Authorization.
        headers = self._get_headers(content_type=None, require_api_key=True)

        form_data: Dict[str, Any] = {}
        if model:
            form_data["model"] = model
        if prompt:
            if len(prompt) > 1000:
                raise ValueError("Prompt cannot exceed 1000 characters.")
            form_data["prompt"] = prompt
        if response_format:
            form_data["response_format"] = response_format
        if temperature is not None:
            if not (0 <= temperature <= 2):
                raise ValueError("Temperature must be between 0 and 2.")
            form_data["temperature"] = str(temperature)  # Form data sends as string
        if language:
            form_data["language"] = language
        if provider:
            form_data["provider"] = provider

        # Bound before the try so the parse-failure handler inside can still
        # report the upstream status and body when requests.post() itself never
        # returned a response.
        response: Optional[requests.Response] = None
        try:
            with open(file_path, "rb") as audio_file:
                files = {"file": (file_path, audio_file)}
                response = requests.post(url, headers=headers, data=form_data, files=files)
            response.raise_for_status()
            try:
                api_response_data = response.json()
                api_response = RadientTranscriptionAPIResponse.model_validate(api_response_data)
            except ValueError as e:
                # A response that is not the documented shape. json() and
                # pydantic's model_validate both raise ValueError -- and requests'
                # own decode error is *also* an InvalidJSONError, i.e. a
                # RequestException. Caught here, while the response is still in
                # hand, so the body survives; the outer handler would report it
                # as a body-less transport failure.
                raise APIError(
                    f"Failed to create transcription: unexpected response from Radient ({e})",
                    status_code=response.status_code,
                    body=self._surfaceable_body(scrubbed_response_body(response)),
                ) from e

            if api_response.error:
                # Radient reports a provider failure in a 200 body as often as in
                # an error status, and this branch is where the provider's own
                # words arrive ("... (insufficient_quota): You have no credits
                # remaining"). That is an upstream failure, not a fault of ours,
                # so it is typed as one instead of being flattened into a 500.
                raise APIError(
                    f"Failed to create transcription: {api_response.error} - {api_response.msg}",
                    status_code=response.status_code,
                    body=self._surfaceable_body(scrubbed_response_body(response)),
                )
            if not api_response.result:
                raise APIError(
                    "Failed to create transcription: No result data in response.",
                    status_code=response.status_code,
                    body=self._surfaceable_body(scrubbed_response_body(response)),
                )
            return api_response.result
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        except requests.exceptions.RequestException as e:
            raise api_error_from_exception(
                e,
                prefix="Failed to create transcription",
                secrets=self._credential_values(),
            ) from e
        except APIError:
            # Raised above from a 2xx body carrying an error. Re-raise it
            # unchanged: the catch-all below would otherwise wrap it a second
            # time and bury the status and body the route classifies on.
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to create transcription: {str(e)}") from e

    # Speech Generation Methods
    def create_speech(
        self,
        input_text: str,
        model: str,
        voice: str,
        instructions: Optional[str] = None,
        response_format: Optional[str] = "mp3",
        speed: Optional[float] = 1.0,
        provider: Optional[str] = "openai",
    ) -> bytes:
        """Generate speech from text using the Radient API.

        Args:
            input_text (str): The text to convert to speech.
            instructions (Optional[str]): Additional prompt with instructions for the
            speech generation.
            model (str): The TTS model to use (e.g., "tts-1").
            voice (str): The voice to use (e.g., "alloy").
            response_format (Optional[str]): The audio format. Defaults to "mp3".
            speed (Optional[float]): The speech speed. Defaults to 1.0.
            provider (Optional[str]): The provider. Defaults to "openai".

        Returns:
            bytes: The binary audio data of the generated speech.

        Raises:
            RuntimeError: If the API request fails.
        """
        if not self.api_key:
            raise RuntimeError("RADIENT_API_KEY is not configured. Cannot create speech.")

        url = f"{self.base_url}/tools/speech"
        headers = self._get_headers(require_api_key=True)

        # Create the payload, excluding None values for optional fields
        payload_data = {
            "input": input_text,
            "instructions": instructions,
            "model": model,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "provider": provider,
        }
        payload = {k: v for k, v in payload_data.items() if v is not None}

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            if _is_an_error_envelope(response):
                # A provider failure reported in a 200 body, which is a shape
                # Radient uses. Without this the error body -- including any
                # credential the upstream echoed into it -- is returned as audio
                # bytes and served to this daemon's own client, where no
                # ``HTTPException`` handler ever sees it. Raised as an upstream
                # failure so the route reports it as one, through the same
                # scrubbed body every other surfaced failure goes through.
                raise APIError(
                    "Failed to generate speech: Radient returned an error body with a "
                    f"{response.status_code} status",
                    status_code=response.status_code,
                    body=self._surfaceable_body(scrubbed_response_body(response)),
                )
            return response.content
        except APIError:
            # Raised above from a 2xx body carrying an error. Re-raise it
            # unchanged: the catch-all below would otherwise wrap it a second
            # time and bury the status and body the route reads.
            raise
        except requests.exceptions.RequestException as e:
            error_body = self._surfaceable_body(response_body(e))
            raise RuntimeError(
                f"Failed to generate speech: {str(e)}, Response Body: {error_body}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e


# --- Instruction-set publication (the hub's document transport) ---------------
#
# The unit of publication is a single JSON document, `document_version: 1`,
# carrying an agent's instruction set. WHY A DOCUMENT AND NOT THE ZIP: the
# substrate is one text field, so a bundle carries nothing a body does not; it
# removes the untrusted-archive parsing surface from the new path entirely; and it
# is hashable as-is, which is what the hub's moderation content hash is taken
# over. Everything here mirrors agent-server's validator (contract §1) rather than
# re-deciding it: the refusal a user sees must not depend on which side refused
# it, so the rule text below is the same string the hub puts in
# `details.rule`, and the caps are the same numbers.

#: The document discriminator. An unrecognised value is refused rather than
#: parsed best-effort: a document that is not what the reader thinks it is must
#: never be half-understood.
INSTRUCTION_SET_DOCUMENT_TYPE = "radient.agent-instruction-set"

#: The schema version this client writes. The hub refuses `> 1` with
#: `details.supported_versions`, so an older hub answers honestly instead of
#: partially parsing a newer document.
INSTRUCTION_SET_DOCUMENT_VERSION = 1

#: Caps, mirroring agent-server's validator. Two of them are REUSED rather than
#: restated, because a second number is how the two sides drift apart:
#: `MAX_INSTRUCTIONS_CHARS` (the local profile cap and the hub's document cap are
#: the same bound) and `MAX_AGENT_NAME_CHARS` (the hub's `models.AgentNameMaxChars`,
#: which the local registry refuses to exceed when it invents a name of its own —
#: see `_collision_free_name`).
INSTRUCTION_SET_DESCRIPTION_MAX_CHARS = 2000
INSTRUCTION_SET_WHEN_TO_USE_MAX_CHARS = 2000
INSTRUCTION_SET_TOOLS_MAX_ITEMS = 64
INSTRUCTION_SET_TOOL_MAX_CHARS = 64
INSTRUCTION_SET_TAGS_MAX_ITEMS = 32
INSTRUCTION_SET_TAG_MAX_CHARS = 64
INSTRUCTION_SET_CATEGORIES_MAX_ITEMS = 8
INSTRUCTION_SET_EFFORT_MAX_CHARS = 16

#: The two kinds a document may declare. `kind` is explicit in a published
#: document because the receiving side cannot infer it the way local-operator
#: does locally (from a registry tag or a category).
INSTRUCTION_SET_KINDS = ("role", "specialist")

#: The hub's category enum. A category outside it is refused, not dropped: the
#: hub's category rail filters on these strings, so a free-text category would
#: produce a published row no filter can ever surface.
HUB_AGENT_CATEGORIES = (
    "investment",
    "accounting",
    "healthcare",
    "legal",
    "software",
    "security",
    "role_play",
    "personal_assistance",
    "education",
    "marketing",
    "sales",
    "research",
    "analysis",
    "management",
    "social_media",
    "other",
)

#: The content fields a caller may supply, in the schema's order. `document_type`
#: and `document_version` are deliberately absent: the CLIENT owns the schema it
#: writes, and letting a caller declare a version this code does not implement is
#: how a document gets sent that neither side understands.
INSTRUCTION_SET_CONTENT_FIELDS = (
    "name",
    "description",
    "instructions",
    "kind",
    "when_to_use",
    "tools",
    "effort",
    "delegate",
    "version",
    "categories",
    "tags",
)

#: Every key a version-1 document is allowed to carry. The publish body's key set
#: is asserted against this in the tests, because it is the property the whole
#: standard exists to restore: an agent is its instruction set, and nothing about
#: the machine it was authored on rides along.
INSTRUCTION_SET_FIELDS = (
    "document_type",
    "document_version",
    *INSTRUCTION_SET_CONTENT_FIELDS,
)

#: The SHAPE each overridable content field must have. Anything absent is text.
#:
#: WHY A SHAPE TABLE AND NOT JUST THE BUILDER'S OWN CHECKS: the builder coerces --
#: ``bool(delegate)``, ``list(tags)`` -- and a coercion publishes something the
#: caller did not ask for, silently. ``{"delegate": "false"}`` published ``true``
#: (the string is truthy), ``{"tags": "osint"}`` published five one-character
#: tags, and a list or an int where text belongs escaped the builder entirely as
#: an ``AttributeError`` -- a 500 blaming this machine for a request the hub would
#: have refused at decode with ``invalid_instruction_set``. So the shape is refused
#: HERE, with the rule text and the field the hub would use, before the builder
#: can coerce anything.
#:
#: ``bool`` rather than "accepts 0/1": the wire says true or false, Python's
#: ``bool`` is the only value whose meaning is not a guess, and ``isinstance(1,
#: bool)`` is False, so an int is refused rather than reinterpreted.
INSTRUCTION_SET_FIELD_SHAPES: Dict[str, str] = {
    "tools": "list",
    "categories": "list",
    "tags": "list",
    "delegate": "bool",
}

#: The rule text per shape, phrased as the hub phrases the rules it states itself
#: ("must be at most N characters", "must hold at most N items"), because
#: ``details.rule`` is what the desktop app's inline error quotes.
INSTRUCTION_SET_SHAPE_RULES: Dict[str, str] = {
    "str": "must be a string",
    "list": "must be a list of strings",
    "bool": "must be true or false",
}


def _matches_field_shape(value: Any, shape: str) -> bool:
    """Whether an override value has the shape its field requires.

    A ``str`` is deliberately NOT a list, however sequence-like it is: that is
    the whole production bug -- ``list("osint")`` is five tags -- and a check
    written as ``isinstance(value, (list, tuple))`` cannot make that mistake.
    """

    if shape == "list":
        return isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)
    if shape == "bool":
        return isinstance(value, bool)
    return isinstance(value, str)


#: Whitespace that is ALSO a control character: Go's `unicode.IsSpace ∩
#: unicode.IsControl`, which is the tab, the vertical tab, the form feed, the
#: line and carriage returns and NEL. Spelled out rather than written as
#: `character.isspace()`, because Python's `str.isspace()` additionally calls
#: U+001C..U+001F whitespace while Go reports those as control characters — and a
#: client that disagrees with the hub about WHICH rule a name broke is exactly
#: what this mirror exists to prevent.
_CONTROL_WHITESPACE = frozenset("\t\n\v\f\r\x85")


class InstructionSetError(ValueError):
    """A document refused before it was sent, in the hub's own vocabulary.

    ``field`` and ``rule`` are the same two values the hub returns as
    ``details`` for a refusal it makes, and the message is the same sentence the
    hub composes, so a caller renders a locally-refused document and a
    hub-refused one identically — one switch on one code, not two error paths.
    """

    def __init__(self, field: str, rule: str) -> None:
        super().__init__(f"The agent document is not valid: {field} {rule}.")
        self.field = field
        self.rule = rule

    @property
    def details(self) -> Dict[str, Any]:
        """The machine-readable half, shaped as the hub's ``details``."""

        return {"field": self.field, "rule": self.rule}


def _name_rule(name: str) -> Optional[str]:
    """The rule a published name breaks, or ``None`` when it is acceptable.

    Mirrors ``models.ValidateAgentName`` in agent-server: the same characters, the
    same order and the same rule text, because ``details.rule`` is what the
    desktop app's inline error quotes — a document refused here and one refused by
    the hub must read identically.

    THE ONE RULE THIS DELIBERATELY DOES NOT MIRROR IS THE WHITESPACE BAN.
    agent-server refuses any whitespace in a published name today, and is in the
    middle of relaxing exactly that (`dev-name-spaces`, `0a44f50`): the live
    marketplace is already spelled with ordinary spaces — 21 public rows, 13 of
    them in case-insensitive duplicate groups (Twitter, Product Manager, Codey,
    gitbot, Job automation for auto apply) — so the rule is being changed to
    "collapse every run of Unicode whitespace to one U+0020 and trim the ends".
    A client cannot mirror a rule that is moving: refusing an ordinary space here
    would refuse a name the hub is about to accept (a client bound stricter than
    the server is a bug report), and refusing it only after the change would mean
    the same release behaves differently against two hub versions. So whitespace
    is sent AS THE AUTHOR WROTE IT and the hub decides — today it answers 422
    ``invalid_instruction_set`` with ``details.field = "name"`` and the rule, which
    this route carries through unchanged, and after the relaxation it normalises
    and stores the name.

    Whitespace that is ALSO a control character (the tab, the vertical tab, the
    form feed, the line controls and NEL) stays refused, with the hub's own text
    for it: no legitimate name contains one, and the published-validator order
    reports those as whitespace rather than as control characters, so dropping the
    check would put a different ``details.rule`` on the same input than the hub's.

    Names are also file names on the machine that pulls them, and on a public
    marketplace a name whose rendered form differs from its bytes is a spoofing
    surface — that is what the bidi/control/format rules below are for, and why
    they are checked in the hub's order (bidi first, so its own rule text wins).
    """

    trimmed = name.strip()
    if not trimmed:
        return "must not be empty"
    if len(trimmed) > MAX_AGENT_NAME_CHARS:
        return f"must be at most {MAX_AGENT_NAME_CHARS} characters"
    if any(character in trimmed for character in ("/", "\\", ":")):
        return 'must not contain "/", "\\" or ":"'
    if any(
        "\u202a" <= character <= "\u202e" or "\u2066" <= character <= "\u2069"
        for character in trimmed
    ):
        return "must not contain Unicode bidirectional override characters"
    if any(character in _CONTROL_WHITESPACE for character in trimmed):
        return "must not contain whitespace"
    # `unicodedata.category == "Cc"` is Go's `unicode.IsControl` (the Cc table),
    # not `str.isprintable`: isprintable is False for every format character too,
    # which would report a name containing an invisible joiner as a control-char
    # violation and send its author looking for something that is not there.
    if any(unicodedata.category(character) == "Cc" for character in trimmed):
        return "must not contain control characters"
    # Format characters (Cf) render as nothing, so `reviewer` + U+200B and
    # `reviewer` draw identically while being two different keys: the shadowing an
    # exact local name lookup cannot see. The hub refuses them under this text
    # rather than folding them away, because folding would silently rewrite the
    # author's name.
    if any(unicodedata.category(character) == "Cf" for character in trimmed):
        return "must not contain invisible Unicode formatting characters"
    if trimmed[0] in "-." or trimmed[-1] in "-.":
        return 'must not begin or end with "-" or "."'
    return None


def _items_rule(values: Sequence[str], *, max_items: int, max_item_chars: int) -> Optional[str]:
    """The rule a list field breaks, or ``None``.

    One implementation for tools and tags: both are "at most N items of 1..M
    characters", and the hub reports the same two rules for both.
    """

    if len(values) > max_items:
        return f"must hold at most {max_items} items"
    for value in values:
        if not value.strip() or len(value) > max_item_chars:
            return f"must hold items of 1 to {max_item_chars} characters"
    return None


def build_instruction_set_document(
    *,
    name: str,
    description: str,
    instructions: str,
    kind: str,
    version: str,
    when_to_use: str = "",
    tools: Optional[Sequence[str]] = None,
    effort: str = "",
    delegate: bool = False,
    categories: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build the version-1 instruction-set document the hub publishes.

    The signature is the field set: every argument is a content field of the
    document, so there is no parameter through which anything else — a
    conversation, an execution history, a pickled context, a working directory, a
    model, a hosting provider, a security prompt — could reach the wire. That is
    the guarantee this standard exists to restore, and it is structural here
    rather than a filter applied afterwards.

    The check order matches the hub's validator, because the order decides which
    field a malformed document is refused for and a client that reports a
    different field than the server would is a bug report waiting to be filed.

    Args:
        name: The agent's published name (1..128 characters, after trim). The
            spelling is sent as the author wrote it: whitespace is the hub's to
            normalise (see :func:`_name_rule`).
        description: What the agent does (1..2000 characters).
        instructions: The instruction body (1..8000 characters). This is the
            publication; it is also the only field the hub's reviewer treats as
            behavioural evidence.
        kind: ``"role"`` or ``"specialist"``.
        version: The AUTHOR's version of their agent's content, free-form.
        when_to_use: Optional routing text (0..2000 characters).
        tools: Optional tool names (<=64 items of 1..64 characters).
        effort: Optional effort hint (<=16 characters).
        delegate: Whether the agent may delegate further. Sent explicitly even
            when false: it is a statement about the agent, not an absence.
        categories: Optional hub categories (<=8, from :data:`HUB_AGENT_CATEGORIES`).
        tags: Optional discovery tags (<=32 items of 1..64 characters).

    Returns:
        The document, carrying `document_type`/`document_version` and only the
        supplied optional fields.

    Raises:
        InstructionSetError: When a field breaks a rule of contract §1.4/§1.5.
            ``field``/``rule`` say which one, in the hub's own words.
    """

    rule = _name_rule(name)
    if rule:
        raise InstructionSetError("name", rule)
    if not description.strip():
        raise InstructionSetError("description", "must not be empty")
    if len(description) > INSTRUCTION_SET_DESCRIPTION_MAX_CHARS:
        raise InstructionSetError(
            "description", f"must be at most {INSTRUCTION_SET_DESCRIPTION_MAX_CHARS} characters"
        )
    if not instructions.strip():
        raise InstructionSetError("instructions", "must not be empty")
    if len(instructions) > MAX_INSTRUCTIONS_CHARS:
        raise InstructionSetError(
            "instructions", f"must be at most {MAX_INSTRUCTIONS_CHARS} characters"
        )
    if kind not in INSTRUCTION_SET_KINDS:
        raise InstructionSetError("kind", 'must be "role" or "specialist"')
    if len(when_to_use) > INSTRUCTION_SET_WHEN_TO_USE_MAX_CHARS:
        raise InstructionSetError(
            "when_to_use", f"must be at most {INSTRUCTION_SET_WHEN_TO_USE_MAX_CHARS} characters"
        )
    tool_list = list(tools or ())
    rule = _items_rule(
        tool_list,
        max_items=INSTRUCTION_SET_TOOLS_MAX_ITEMS,
        max_item_chars=INSTRUCTION_SET_TOOL_MAX_CHARS,
    )
    if rule:
        raise InstructionSetError("tools", rule)
    if len(effort) > INSTRUCTION_SET_EFFORT_MAX_CHARS:
        raise InstructionSetError(
            "effort", f"must be at most {INSTRUCTION_SET_EFFORT_MAX_CHARS} characters"
        )
    category_list = list(categories or ())
    if len(category_list) > INSTRUCTION_SET_CATEGORIES_MAX_ITEMS:
        raise InstructionSetError(
            "categories", f"must hold at most {INSTRUCTION_SET_CATEGORIES_MAX_ITEMS} items"
        )
    for category in category_list:
        if category not in HUB_AGENT_CATEGORIES:
            raise InstructionSetError("categories", "must name categories from the server enum")
    tag_list = list(tags or ())
    rule = _items_rule(
        tag_list,
        max_items=INSTRUCTION_SET_TAGS_MAX_ITEMS,
        max_item_chars=INSTRUCTION_SET_TAG_MAX_CHARS,
    )
    if rule:
        raise InstructionSetError("tags", rule)
    if not version.strip():
        raise InstructionSetError("version", "must not be empty")

    document: Dict[str, Any] = {
        "document_type": INSTRUCTION_SET_DOCUMENT_TYPE,
        "document_version": INSTRUCTION_SET_DOCUMENT_VERSION,
        "name": name.strip(),
        "description": description,
        "instructions": instructions,
        "kind": kind,
    }
    # Optional fields are OMITTED when empty rather than sent as "": an empty
    # string is a value the hub stores and a client renders, whereas an absent
    # field is the absence the schema documents. `delegate` is the exception —
    # a boolean states something either way — and it is always sent.
    if when_to_use:
        document["when_to_use"] = when_to_use
    if tool_list:
        document["tools"] = tool_list
    if effort:
        document["effort"] = effort
    document["delegate"] = bool(delegate)
    document["version"] = version
    if category_list:
        document["categories"] = category_list
    if tag_list:
        document["tags"] = tag_list
    return document


def validate_document_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the content fields a caller may override, refusing anything else.

    Used for the desktop app's edits: the UI holds the fields it is publishing, and
    this is what keeps "the fields it is publishing" and "the document's fields"
    the same set. An unknown key is refused rather than dropped, for the reason the
    hub refuses one — silent field-dropping is how a publisher believes it
    published something it did not — and ``document_type``/``document_version``
    are not overridable at all. A known field carrying a value of the wrong SHAPE
    is refused the same way, so a coercion the builder would have performed on the
    caller's behalf (``bool("false")`` is ``True``, ``list("osint")`` is five
    tags) cannot publish something the caller did not ask for, and a shape the
    builder cannot coerce is a 422 rather than an ``AttributeError`` reported as a
    failure of this machine.

    The FIRST unknown key in the caller's order is the one reported, matching the
    hub's streaming scan (a map's iteration order would make the reported field
    non-deterministic, and a non-deterministic error is untestable). The shape pass
    runs after it, in the same caller order, for the same reason.

    Args:
        overrides: The caller-supplied document fields.

    Returns:
        The same fields, validated as a known-key mapping of correctly shaped values.

    Raises:
        InstructionSetError: When a key is not part of a version-1 document, or a
            known key's value is not the shape that field takes.
    """

    for key in overrides:
        if key not in INSTRUCTION_SET_CONTENT_FIELDS:
            raise InstructionSetError(key, "is not a recognised field")
    for key, value in overrides.items():
        shape = INSTRUCTION_SET_FIELD_SHAPES.get(key, "str")
        if not _matches_field_shape(value, shape):
            raise InstructionSetError(key, INSTRUCTION_SET_SHAPE_RULES[shape])
    return dict(overrides)
