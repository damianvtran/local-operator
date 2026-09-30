from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from local_operator.agents import AgentData
from local_operator.clients._http import APIError
from local_operator.server.models.schemas import AgentSpeechRequest, SpeechRequest
from local_operator.server.routes.speech import create_agent_speech, create_speech


@pytest.fixture
def mock_radient_client():
    """Fixture for a mocked Radient client."""
    return MagicMock()


@pytest.fixture
def speech_request_data():
    """Fixture for speech request data."""
    return {
        "input": "Hello, world!",
        "instructions": "Please speak in a friendly and engaging tone.",
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
        instructions="Please speak in a friendly and engaging tone.",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        speed=1.0,
        provider="openai",
        language_code=None,
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


@pytest.mark.asyncio
async def test_create_speech_passes_language_code_through(mock_radient_client):
    """The language code is plumbing: forwarded exactly when the caller set it."""
    mock_radient_client.create_speech.return_value = b"audio_data"
    speech_request = _speech_request(input="Hola", language_code="es")

    response = await create_speech(speech_request, mock_radient_client)

    assert response.status_code == 200
    mock_radient_client.create_speech.assert_called_once_with(
        input_text="Hola",
        instructions=None,
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        speed=1.0,
        provider="openai",
        language_code="es",
    )


@pytest.mark.parametrize("missing", [None, SecretStr("")])
@pytest.mark.asyncio
async def test_create_speech_without_a_credential_answers_401(mock_radient_client, missing):
    """No resolved credential is the sign-in remedy, before any upstream work."""
    mock_radient_client.api_key = missing
    speech_request = _speech_request()

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Sign in to Radient to use speaking aloud"
    mock_radient_client.create_speech.assert_not_called()


@pytest.mark.parametrize(
    ("status", "sentence"),
    [
        (401, "Sign in to Radient to use speaking aloud"),
        (402, "Your Radient credit balance is too low for speech. Add credits to continue."),
        (429, "Speech is busy right now. Try again in a moment."),
        (503, "Speech is temporarily unavailable."),
    ],
)
@pytest.mark.asyncio
async def test_create_speech_passes_actionable_refusals_through(
    mock_radient_client, status, sentence
):
    """The four actionable statuses keep their status and get fixed copy.

    The upstream's own envelope must not reach the response: it is written for
    an operator, and the raw frame can carry request material.
    """
    mock_radient_client.create_speech.side_effect = APIError(
        "insufficient credits for this request",
        status_code=status,
        body='{"error": "insufficient credits for this request", "code": "billing"}',
    )
    speech_request = _speech_request()

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == status
    assert exc_info.value.detail == sentence


@pytest.mark.asyncio
async def test_create_speech_keeps_the_502_path_for_an_error_envelope(mock_radient_client):
    """A 200 body that is an error envelope keeps the pre-existing 502 diagnostic."""
    mock_radient_client.create_speech.side_effect = APIError(
        "Failed to generate speech: Radient returned an error body with a 200 status",
        status_code=200,
        body='{"error": "vendor quota exceeded"}',
    )
    speech_request = _speech_request()

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 502
    assert "Radient reported an error (HTTP 200)" in exc_info.value.detail
    # The envelope's designed `error` field is extracted; the raw frame is not quoted.
    assert exc_info.value.detail.endswith("vendor quota exceeded")
    assert '{"error"' not in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_speech_extracts_the_error_field_for_other_statuses(mock_radient_client):
    """A non-actionable status keeps 502 but carries only the designed `error` prose."""
    mock_radient_client.create_speech.side_effect = APIError(
        "Invalid model",
        status_code=400,
        body='{"error": "unknown model eleven_bogus_v1", "code": "invalid_request"}',
    )
    speech_request = _speech_request()

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 502
    assert "unknown model eleven_bogus_v1" in exc_info.value.detail
    assert '{"error"' not in exc_info.value.detail


def _speech_request(**overrides: Any) -> SpeechRequest:
    """Build a valid direct-route request.

    The fields travel through a typed dict because the type checker treats a
    pydantic model's defaulted fields as required keyword arguments.
    """
    fields: Dict[str, Any] = {"input": "Hello", "model": "tts-1", "voice": "alloy"}
    fields.update(overrides)
    return SpeechRequest(**fields)


def _agent_speech_request(**overrides: Any) -> AgentSpeechRequest:
    """Build a valid agent-route request (see ``_speech_request``)."""
    fields: Dict[str, Any] = {"input_text": "Hello"}
    fields.update(overrides)
    return AgentSpeechRequest(**fields)


def _agent() -> AgentData:
    fields: Dict[str, Any] = {
        "id": "test-agent",
        "name": "Aria",
        "created_date": datetime.now(),
        "version": "1.0.0",
        "description": "A friendly assistant",
        "hosting": "openai",
        "model": "gpt-4o",
    }
    return AgentData(**fields)


def _credentialed_client() -> MagicMock:
    client = MagicMock()
    client.api_key = SecretStr("test-key")
    client.create_speech.return_value = b"audio_data"
    return client


@pytest.mark.asyncio
async def test_create_agent_speech_uses_the_elevenlabs_contract():
    """The payload is provider=elevenlabs, the alias voice, no instructions/model."""
    radient_client = _credentialed_client()
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    with (
        patch("local_operator.server.routes.speech.configure_model", return_value=MagicMock()),
        patch(
            "local_operator.server.routes.speech.determine_voice",
            new_callable=AsyncMock,
            return_value="female",
        ),
    ):
        response = await create_agent_speech(
            "test-agent",
            _agent_speech_request(language_code="es"),
            radient_client,
            agent_registry,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    assert response.status_code == 200
    assert response.body == b"audio_data"
    # assert_called_once_with pins the ABSENCE of `instructions`/`model` too.
    radient_client.create_speech.assert_called_once_with(
        input_text="Hello",
        voice="female",
        response_format="mp3",
        speed=1.0,
        provider="elevenlabs",
        language_code="es",
    )


@pytest.mark.asyncio
async def test_create_agent_speech_404_for_an_unknown_agent():
    radient_client = _credentialed_client()
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await create_agent_speech(
            "missing",
            _agent_speech_request(),
            radient_client,
            agent_registry,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    assert exc_info.value.status_code == 404
    radient_client.create_speech.assert_not_called()


@pytest.mark.asyncio
async def test_create_agent_speech_requires_a_credential_before_any_work():
    """No credential answers 401 before configuration or any model call."""
    radient_client = MagicMock()
    radient_client.api_key = SecretStr("")
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    with (
        patch("local_operator.server.routes.speech.configure_model") as configure_call,
        patch("local_operator.server.routes.speech.determine_voice") as voice_call,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_agent_speech(
                "test-agent",
                _agent_speech_request(),
                radient_client,
                agent_registry,
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Sign in to Radient to use speaking aloud"
    configure_call.assert_not_called()
    voice_call.assert_not_called()
    radient_client.create_speech.assert_not_called()


@pytest.mark.asyncio
async def test_create_agent_speech_passes_refusals_through():
    """The agent route shares the direct route's refusal classification."""
    radient_client = _credentialed_client()
    radient_client.create_speech.side_effect = APIError(
        "insufficient credits for this request", status_code=402
    )
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    with (
        patch("local_operator.server.routes.speech.configure_model", return_value=MagicMock()),
        patch(
            "local_operator.server.routes.speech.determine_voice",
            new_callable=AsyncMock,
            return_value="male",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_agent_speech(
                "test-agent",
                _agent_speech_request(),
                radient_client,
                agent_registry,
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )

    assert exc_info.value.status_code == 402
    assert exc_info.value.detail == (
        "Your Radient credit balance is too low for speech. Add credits to continue."
    )
