"""
Tests for the transcription endpoints of the FastAPI server.
"""

import os
import shutil
import tempfile
from typing import Any, Callable, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi import UploadFile

from local_operator.clients._http import (
    NO_RESPONSE_BODY,
    APIError,
    api_error_from_exception,
)
from local_operator.clients.radient import RadientTranscriptionResponseData
from local_operator.server.app import app
from local_operator.server.dependencies import get_radient_client

# Define a sample audio file content (can be any bytes)
SAMPLE_AUDIO_CONTENT = b"sample audio data"
SAMPLE_FILE_NAME = "test_audio.mp3"


@pytest.fixture
def temp_audio_file():
    """Creates a temporary audio file for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, SAMPLE_FILE_NAME)
    with open(temp_file_path, "wb") as f:
        f.write(SAMPLE_AUDIO_CONTENT)
    yield temp_file_path
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_create_transcription_success(test_app_client, temp_audio_file):
    """Test successful transcription creation.

    `model` and `provider` are deliberately NOT sent: the daemon must let
    Radient's configured default govern, so both must reach the client as None
    (the client then omits them from the upstream multipart body entirely).
    """
    mock_radient_client = MagicMock()
    mock_transcription_response = RadientTranscriptionResponseData(
        text="This is a test transcription.",
        provider="openai",
        status="completed",
    )
    mock_radient_client.create_transcription.return_value = mock_transcription_response
    mock_radient_client.api_key = "fake_api_key"

    def override_dependency():
        return mock_radient_client

    app.dependency_overrides[get_radient_client] = override_dependency

    try:
        with open(temp_audio_file, "rb") as f:
            files = {"file": (SAMPLE_FILE_NAME, f, "audio/mpeg")}
            data = {
                "response_format": "json",
                "temperature": 0.0,
            }
            response = await test_app_client.post("/v1/transcriptions", files=files, data=data)
    finally:
        del app.dependency_overrides[get_radient_client]

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["result"]["text"] == "This is a test transcription."
    mock_radient_client.create_transcription.assert_called_once()
    # Check that the temp file path was passed to create_transcription
    call_args = mock_radient_client.create_transcription.call_args[1]
    assert "file_path" in call_args
    assert call_args["model"] is None
    assert call_args["response_format"] == "json"
    assert call_args["temperature"] == 0.0
    assert call_args["provider"] is None


@pytest.mark.asyncio
async def test_create_transcription_no_api_key(test_app_client, temp_audio_file):
    """Test transcription creation when Radient API key is not configured."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = None  # Simulate no API key
    mock_radient_client.create_transcription = (
        MagicMock()
    )  # Ensure it's a mock for assert_not_called

    def override_dependency():
        return mock_radient_client

    app.dependency_overrides[get_radient_client] = override_dependency

    try:
        with open(temp_audio_file, "rb") as f:
            files = {"file": (SAMPLE_FILE_NAME, f, "audio/mpeg")}
            response = await test_app_client.post("/v1/transcriptions", files=files)
    finally:
        del app.dependency_overrides[get_radient_client]

    mock_radient_client.create_transcription.assert_not_called()
    assert response.status_code == 500
    assert "Radient API key is not configured" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_transcription_file_save_failure(test_app_client, tmp_path, monkeypatch):
    """Test transcription creation when saving the uploaded file fails.

    Also pins the leak fix: the temp directory used to be created *inside* the
    try, so a write failure raised straight past the ``finally: rmtree`` (which
    sits on a later try) and stranded a directory on every failed upload.
    ``tempfile.tempdir`` is redirected so the assertion sees only directories
    this test caused.
    """
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"

    # Mock shutil.copyfileobj to raise an exception
    # For this test, the dependency override is still useful to ensure the client is a mock,
    # even if its methods aren't directly called due to an earlier error.
    def override_dependency():
        return mock_radient_client

    app.dependency_overrides[get_radient_client] = override_dependency
    try:
        with patch("shutil.copyfileobj", side_effect=Exception("File save error")):
            # Create a dummy UploadFile object
            mock_upload_file = MagicMock(spec=UploadFile)
            mock_upload_file.filename = "audio.mp3"
            mock_upload_file.file = MagicMock()  # This would be a SpooledTemporaryFile or similar

            files = {"file": (mock_upload_file.filename, b"dummy content", "audio/mpeg")}
            response = await test_app_client.post("/v1/transcriptions", files=files)
    finally:
        del app.dependency_overrides[get_radient_client]

    assert response.status_code == 500
    assert "Failed to save uploaded audio file" in response.json()["detail"]
    # The failed upload must leave nothing behind.
    assert list(scratch.iterdir()) == []


@pytest.mark.asyncio
async def test_create_transcription_client_runtime_error(test_app_client, temp_audio_file):
    """Test transcription creation when Radient client raises a RuntimeError."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(
        side_effect=RuntimeError("Radient API error")
    )

    def override_dependency():
        return mock_radient_client

    app.dependency_overrides[get_radient_client] = override_dependency

    try:
        with open(temp_audio_file, "rb") as f:
            files = {"file": (SAMPLE_FILE_NAME, f, "audio/mpeg")}
            response = await test_app_client.post("/v1/transcriptions", files=files)
    finally:
        del app.dependency_overrides[get_radient_client]

    mock_radient_client.create_transcription.assert_called_once()
    assert response.status_code == 500
    assert "Radient API error" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_transcription_client_value_error(test_app_client, temp_audio_file):
    """Test transcription creation when Radient client raises a ValueError."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(
        side_effect=ValueError("Invalid parameter")
    )

    def override_dependency():
        return mock_radient_client

    app.dependency_overrides[get_radient_client] = override_dependency

    try:
        with open(temp_audio_file, "rb") as f:
            files = {"file": (SAMPLE_FILE_NAME, f, "audio/mpeg")}
            response = await test_app_client.post("/v1/transcriptions", files=files)
    finally:
        del app.dependency_overrides[get_radient_client]

    mock_radient_client.create_transcription.assert_called_once()
    assert response.status_code == 400
    assert "Invalid parameter" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_transcription_file_not_found_error(test_app_client, temp_audio_file):
    """Test transcription creation when the temporary file is not found after saving."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    # Simulate FileNotFoundError from the client's perspective
    mock_radient_client.create_transcription = MagicMock(
        side_effect=FileNotFoundError("Temporary audio file not found")
    )

    def override_dependency():
        return mock_radient_client

    app.dependency_overrides[get_radient_client] = override_dependency

    try:
        with open(temp_audio_file, "rb") as f:
            files = {"file": (SAMPLE_FILE_NAME, f, "audio/mpeg")}
            response = await test_app_client.post("/v1/transcriptions", files=files)
    finally:
        del app.dependency_overrides[get_radient_client]

    mock_radient_client.create_transcription.assert_called_once()
    assert response.status_code == 500
    assert "Temporary audio file not found after saving" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_transcription_unexpected_error(test_app_client, temp_audio_file):
    """Test transcription creation with an unexpected error during transcription."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(side_effect=Exception("Unexpected error"))

    def override_dependency():
        return mock_radient_client

    app.dependency_overrides[get_radient_client] = override_dependency

    try:
        with open(temp_audio_file, "rb") as f:
            files = {"file": (SAMPLE_FILE_NAME, f, "audio/mpeg")}
            response = await test_app_client.post("/v1/transcriptions", files=files)
    finally:
        del app.dependency_overrides[get_radient_client]

    mock_radient_client.create_transcription.assert_called_once()
    assert response.status_code == 500
    assert "An unexpected error occurred during transcription" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_transcription_with_all_optional_params(test_app_client, temp_audio_file):
    """Test successful transcription creation with all optional parameters.

    The counterpart to `test_create_transcription_success`: an explicitly
    requested model/provider pair is forwarded to the client verbatim, so a
    caller that does need to pin a backend still can.
    """
    mock_radient_client = MagicMock()
    mock_transcription_response = RadientTranscriptionResponseData(
        text="This is a detailed test transcription.",
        provider="custom_provider",
        status="completed",
    )
    mock_radient_client.create_transcription.return_value = mock_transcription_response
    mock_radient_client.api_key = "fake_api_key"

    def override_dependency():
        return mock_radient_client

    app.dependency_overrides[get_radient_client] = override_dependency

    try:
        with open(temp_audio_file, "rb") as f:
            files = {"file": (SAMPLE_FILE_NAME, f, "audio/mpeg")}
            data = {
                "model": "whisper-large-v2",
                "prompt": "Test prompt.",
                "response_format": "text",
                "temperature": 0.5,
                "language": "en",
                "provider": "custom_provider",
            }
            response = await test_app_client.post("/v1/transcriptions", files=files, data=data)
    finally:
        del app.dependency_overrides[get_radient_client]

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["result"]["text"] == "This is a detailed test transcription."
    mock_radient_client.create_transcription.assert_called_once()
    call_args = mock_radient_client.create_transcription.call_args[1]
    assert call_args["model"] == "whisper-large-v2"
    assert call_args["prompt"] == "Test prompt."
    assert call_args["response_format"] == "text"
    assert call_args["temperature"] == 0.5
    assert call_args["language"] == "en"
    assert call_args["provider"] == "custom_provider"


# --- Failure classification -------------------------------------------------
#
# The statuses below are a contract with clients that branch on them: 402 means
# "out of credit, a human can fix it", anything else that came from upstream
# means "the call failed on their side, retrying the same request will not help".
# They are pinned here because the route used to answer 500 for every one of
# them, which is what left the desktop with nothing to show but a generic
# "Error transcribing audio. Please try again."

# The refusal Radient actually returned on 2026-09-17, verbatim: the provider's
# own words in the body of a 500.
UPSTREAM_QUOTA_BODY = (
    b'{"error":"[internal] Transcription failed: OpenAI API error: OpenAI API error '
    b"(insufficient_quota): You have no credits remaining. Add credits to your plan to "
    b'continue."}'
)


def _upstream_failure(response: requests.Response) -> APIError:
    """Build the failure the Radient client raises, through the client's own helper.

    Driving :func:`api_error_from_exception` rather than hand-assembling an
    ``APIError`` is what keeps these rows honest: the route classifies the very
    object the client produces, so a change to the client's status or body
    carrying shows up here rather than passing as a route-only fixture.

    Args:
        response: A real ``requests.Response``, so ``raise_for_status()`` and the
            body extraction behave exactly as they do in production.

    Returns:
        APIError: What ``create_transcription`` raises for that response.
    """
    return api_error_from_exception(
        requests.exceptions.HTTPError("500 Server Error", response=response),
        prefix="Failed to create transcription",
    )


async def _post_transcription(
    test_app_client: Any,
    temp_audio_file: str,
    failing_client: MagicMock,
    data: Optional[Dict[str, str]] = None,
) -> Any:
    """POST an audio file through the route with ``failing_client`` injected."""

    def override_dependency():
        return failing_client

    app.dependency_overrides[get_radient_client] = override_dependency
    try:
        with open(temp_audio_file, "rb") as f:
            files = {"file": (SAMPLE_FILE_NAME, f, "audio/mpeg")}
            return await test_app_client.post(
                "/v1/transcriptions", files=files, data=data if data is not None else {}
            )
    finally:
        del app.dependency_overrides[get_radient_client]


@pytest.mark.asyncio
async def test_create_transcription_provider_quota_refusal_is_402(
    test_app_client: Any,
    temp_audio_file: str,
    real_response: Callable[[int, bytes], requests.Response],
):
    """A provider quota refusal names the provider and is not reported as a 500."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(
        side_effect=_upstream_failure(real_response(500, UPSTREAM_QUOTA_BODY))
    )

    response = await _post_transcription(
        test_app_client, temp_audio_file, mock_radient_client, {"provider": "openai"}
    )

    assert response.status_code == 402
    detail = response.json()["detail"]
    assert "the openai provider has run out of credits" in detail
    # The provider's own words survive to the client: without them the reader
    # cannot tell a quota refusal from an ordinary provider fault.
    assert "insufficient_quota" in detail
    assert "Upstream responded 500" in detail


@pytest.mark.asyncio
async def test_create_transcription_radient_credit_refusal_is_402(
    test_app_client: Any,
    temp_audio_file: str,
    real_response: Callable[[int, bytes], requests.Response],
):
    """Radient's own 402 keeps its status and names the account to top up."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(
        side_effect=_upstream_failure(real_response(402, b'{"error":"insufficient credits"}'))
    )

    response = await _post_transcription(
        test_app_client, temp_audio_file, mock_radient_client, {"provider": "openai"}
    )

    assert response.status_code == 402
    detail = response.json()["detail"]
    assert "your Radient credit balance is too low" in detail
    assert "Add credits to continue." in detail
    # Attributed to Radient rather than to the provider the request named.
    assert "openai provider has run out of credits" not in detail


@pytest.mark.asyncio
async def test_create_transcription_upstream_5xx_with_a_body_is_502(
    test_app_client: Any,
    temp_audio_file: str,
    real_response: Callable[[int, bytes], requests.Response],
):
    """An upstream 5xx carries its status and its body to the client."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(
        side_effect=_upstream_failure(
            real_response(500, b'{"error":"[internal] Transcription failed: worker crashed"}')
        )
    )

    response = await _post_transcription(test_app_client, temp_audio_file, mock_radient_client)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Transcription failed upstream." in detail
    assert "worker crashed" in detail
    assert "500" in detail


@pytest.mark.asyncio
async def test_create_transcription_upstream_5xx_without_a_body_is_502(
    test_app_client: Any,
    temp_audio_file: str,
    real_response: Callable[[int, bytes], requests.Response],
):
    """An upstream 5xx that sent no body says so, and quotes nothing.

    This is the shape the historical bug reported as "No response body" -- the
    one text that covered both "the server said nothing" and "we never reached
    the server", which is why the failure could not be triaged.
    """
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(
        side_effect=_upstream_failure(real_response(500, b""))
    )

    response = await _post_transcription(test_app_client, temp_audio_file, mock_radient_client)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "500" in detail
    assert "no body" in detail
    # The sentinel is the client's internal stand-in, not prose for a reader: it
    # must not reach the client's copy of the failure even though it is what the
    # same shape reads as inside the log message.
    assert NO_RESPONSE_BODY not in detail


@pytest.mark.asyncio
async def test_create_transcription_network_failure_is_502_with_its_own_text(
    test_app_client: Any, temp_audio_file: str
):
    """A request that never reached Radient is a gateway failure, not a 500."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(
        side_effect=api_error_from_exception(
            requests.exceptions.ConnectionError("Connection refused"),
            prefix="Failed to create transcription",
        )
    )

    response = await _post_transcription(test_app_client, temp_audio_file, mock_radient_client)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Connection refused" in detail


@pytest.mark.asyncio
async def test_create_transcription_provider_rejection_is_502_with_provider_words(
    test_app_client: Any,
    temp_audio_file: str,
    real_response: Callable[[int, bytes], requests.Response],
):
    """A provider auth/validation rejection is a 502 quoting the provider."""
    mock_radient_client = MagicMock()
    mock_radient_client.api_key = "fake_api_key"
    mock_radient_client.create_transcription = MagicMock(
        side_effect=_upstream_failure(
            real_response(401, b'{"error":"Incorrect API key provided: sk-..."}')
        )
    )

    response = await _post_transcription(
        test_app_client, temp_audio_file, mock_radient_client, {"provider": "openai"}
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "The openai provider rejected the transcription request." in detail
    assert "Incorrect API key provided" in detail
