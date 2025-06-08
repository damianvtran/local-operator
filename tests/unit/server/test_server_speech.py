from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from local_operator.server.models.schemas import SpeechRequest
from local_operator.server.routes.speech import create_speech


@pytest.fixture
def mock_radient_client():
    """Fixture for a mocked Radient client."""
    return MagicMock()


@pytest.fixture
def speech_request_data():
    """Fixture for speech request data."""
    return {
        "input": "Hello, world!",
        "model": "tts-1",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
        "provider": "openai",
    }


@pytest.mark.asyncio
async def test_create_speech_success(speech_request_data, mock_radient_client):
    """Test successful speech creation."""
    mock_radient_client.create_speech.return_value = b"audio_data"
    speech_request = SpeechRequest(**speech_request_data)

    response = await create_speech(speech_request, mock_radient_client)

    assert response.status_code == 200
    assert response.body == b"audio_data"
    assert response.media_type == "audio/mp3"
    mock_radient_client.create_speech.assert_called_once_with(
        input_text="Hello, world!",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        speed=1.0,
        provider="openai",
    )


@pytest.mark.asyncio
async def test_create_speech_http_exception(speech_request_data, mock_radient_client):
    """Test speech creation when Radient client raises HTTPException."""
    mock_radient_client.create_speech.side_effect = HTTPException(
        status_code=404, detail="Model not found"
    )
    speech_request = SpeechRequest(**speech_request_data)

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 404
    assert "Model not found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_speech_generic_exception(speech_request_data, mock_radient_client):
    """Test speech creation when Radient client raises a generic exception."""
    mock_radient_client.create_speech.side_effect = Exception("Something went wrong")
    speech_request = SpeechRequest(**speech_request_data)

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 500
    assert "Failed to generate speech: Something went wrong" in exc_info.value.detail
