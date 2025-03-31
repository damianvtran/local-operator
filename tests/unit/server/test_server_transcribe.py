import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch
import base64
import io
import os
import tempfile
import asyncio

from local_operator.server.app import app  # Import your FastAPI app instance
import numpy as np
import wave
import struct

# Create a test client
client = TestClient(app)

def generate_test_audio_data():
    # Generate a dummy audio file
    sample_rate = 16000  # Standard sample rate for Whisper
    duration = 1  # 1 second of audio
    frequency = 440  # A4 note
    num_samples = int(sample_rate * duration)
    audio = [int(32767.0*np.sin(2*np.pi*frequency*(x/float(sample_rate)))) for x in range(num_samples)]
    
    # Convert to bytes
    byte_io = io.BytesIO()
    wav_file = wave.open(byte_io, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    
    # Convert audio data to 16-bit integers and pack into bytes
    packed_audio = struct.pack("h" * num_samples, *audio)
    wav_file.writeframes(packed_audio)

    wav_file.close()
    
    audio_bytes = byte_io.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return audio_base64

@pytest.fixture
def mock_whisper():
    # Reset the _model to None to ensure get_model() will load a new model
    with patch("local_operator.server.routes.transcribe._model", None):
        # Mock the whisper.load_model function
        with patch("local_operator.server.routes.transcribe.whisper.load_model") as mock_load_model:
            mock_model = mock_load_model.return_value
            mock_model.transcribe.return_value = {"text": "test transcription", "segments": []}
            yield mock_model

def test_transcribe_endpoint(mock_whisper):
    # Create a test audio file
    test_audio_base64 = generate_test_audio_data()
    
    response = client.post(
        "/v1/transcribe",
        json={"data": test_audio_base64},
    )
    assert response.status_code == 200
    assert response.json()["status"] == 200
    assert response.json()["result"]["text"] == "test transcription"

def test_transcribe_endpoint_no_data():
    response = client.post(
        "/v1/transcribe",
        json={},
    )
    assert response.status_code == 400
    assert "Missing audio data" in response.json()["detail"]

def test_transcribe_endpoint_invalid_base64():
    response = client.post(
        "/v1/transcribe",
        json={"data": "invalid base64"},
    )
    assert response.status_code == 400  # Corrected: Should be 400 for invalid input
    assert "Invalid base64 audio data" in response.json()["detail"]

def test_transcribe_endpoint_empty_audio():
    # Send empty base64 data
    response = client.post(
        "/v1/transcribe",
        json={"data": ""},
    )
    assert response.status_code == 400
    # Empty string is treated as missing data by the route
    assert "Missing audio data" in response.json()["detail"]


def test_transcribe_endpoint_temp_file_error(mock_whisper):
    with patch("local_operator.server.routes.transcribe.open", side_effect=Exception("Failed to save")):
        test_audio_base64 = generate_test_audio_data()
        response = client.post(
            "/v1/transcribe",
            json={"data": test_audio_base64},
        )
        assert response.status_code == 500
        assert "Failed to save temporary audio file" in response.json()["detail"]

def test_transcribe_endpoint_transcription_error(mock_whisper):
    mock_whisper.transcribe.side_effect = Exception("Transcription failed")
    test_audio_base64 = generate_test_audio_data()
    response = client.post(
        "/v1/transcribe",
        json={"data": test_audio_base64},
    )
    assert response.status_code == 500
    assert "Transcription failed: Transcription failed" in response.json()["detail"]

def test_transcribe_endpoint_delete_temp_file_error(mock_whisper):
    with patch("local_operator.server.routes.transcribe.os.remove", side_effect=Exception("Failed to delete")):
        test_audio_base64 = generate_test_audio_data()
        response = client.post(
            "/v1/transcribe",
            json={"data": test_audio_base64},
        )
        assert response.status_code == 200
        assert response.json()["status"] == 200
        assert response.json()["result"]["text"] == "test transcription"
