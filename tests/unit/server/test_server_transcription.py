"""
Tests for the transcription endpoints of the FastAPI server.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

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
    """Test successful transcription creation."""
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
                "model": "gpt-4o-transcribe",
                "response_format": "json",
                "temperature": 0.0,
                "provider": "openai",
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
    assert call_args["model"] == "gpt-4o-transcribe"
    assert call_args["response_format"] == "json"
    assert call_args["temperature"] == 0.0
    assert call_args["provider"] == "openai"


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
async def test_create_transcription_file_save_failure(test_app_client):
    """Test transcription creation when saving the uploaded file fails."""
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
    """Test successful transcription creation with all optional parameters."""
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
