"""Caps, mime/format helpers, and the sniff delegation."""

from __future__ import annotations

import pytest

from local_operator.media import AudioInfo
from local_operator.stt import audio


def test_caps_fit_inside_the_desktop_body_budget() -> None:
    """The raw cap must survive base64 inflation inside the 900 kB body cap.

    The bound is the desktop body limit (``desktop_sessions.py``: 900_000
    bytes), and base64 costs 4/3 of the raw bytes; the cap exists so a capture
    taken at it still fits with room for the rest of the payload.
    """
    assert audio.MAX_AUDIO_BYTES * 4 / 3 < 900_000
    assert audio.MAX_AUDIO_DURATION_S == 60


def test_mime_for_path_maps_the_allowlist_and_the_sniffer_extras() -> None:
    assert audio.mime_for_path("capture.webm") == "audio/webm"
    assert audio.mime_for_path("capture.MP3") == "audio/mpeg"
    assert audio.mime_for_path("capture.m4a") == "audio/mp4"
    assert audio.mime_for_path("capture.flac") == "audio/flac"
    # Unknown suffix: the documented wav fallback (the client legs capture wav
    # by default; every route in this tree names the temp file itself).
    assert audio.mime_for_path("capture.weird") == "audio/wav"


def test_filename_for_mime_never_guesses_a_container() -> None:
    assert audio.filename_for_mime("audio/wav") == "audio.wav"
    assert audio.filename_for_mime("audio/mpeg") == "audio.mp3"
    # An unknown mime gets .bin rather than a wrong container: the providers
    # key decoding off the filename, and a guess would decode as the wrong
    # format instead of refusing.
    assert audio.filename_for_mime("audio/x-adts") == "audio.bin"


def test_format_for_model_wire_is_wav_and_mp3_only() -> None:
    assert audio.format_for_model_wire("audio/wav") == "wav"
    assert audio.format_for_model_wire("audio/mpeg") == "mp3"
    # The model-audio wire's own restriction (OQ-3): everything else is None,
    # and the caller must report the rung unavailable rather than transcode.
    assert audio.format_for_model_wire("audio/webm") is None
    assert audio.format_for_model_wire("audio/ogg") is None
    assert audio.format_for_model_wire("audio/mp4") is None


def test_sniff_audio_delegates_to_media(monkeypatch) -> None:
    """The delegation is real: whatever media.sniff_audio answers, this returns."""
    import local_operator.media as media

    marker = object()

    def fake_sniff(data: bytes):
        assert data == b"bytes"
        return marker

    monkeypatch.setattr(media, "sniff_audio", fake_sniff)
    assert audio.sniff_audio(b"bytes") is marker


def test_sniff_audio_identifies_a_wav_header() -> None:
    """End to end through the real media implementation (the integration seam)."""
    info = audio.sniff_audio(b"RIFF" + b"\x00" * 4 + b"WAVEfmt ")
    assert isinstance(info, AudioInfo)
    assert info.mime_type == "audio/wav"


def test_sniff_audio_reports_unknown_rather_than_guessing() -> None:
    assert audio.sniff_audio(b"not audio at all") is None


@pytest.mark.parametrize(
    "mime",
    ["audio/wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg"],
)
def test_every_allowlisted_mime_has_an_upload_name(mime: str) -> None:
    filename = audio.filename_for_mime(mime)
    assert filename.startswith("audio.")
    assert audio.mime_for_path(filename) == mime
