"""The key-free ``local_tts_system`` rung (design V9/V10) and its opt-in."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.tts import VoicePath, cascade, local_say

ON_MAC_WITH_SAY = sys.platform == "darwin" and local_say.say_binary() is not None


class EmptyStore:
    """An auth store with no rows: every provider rung is unavailable."""

    async def has_persisted_credential(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    async def get_persisted_api_key(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_the_rung_is_absent_unless_a_caller_opts_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(local_say, "say_binary", lambda: "/usr/bin/say")
    store: Any = EmptyStore()
    plain = await cascade.resolve_voice_path(config_dir=tmp_path, store=store)
    assert [rung.path for rung in plain.rungs] == list(cascade.TTS_RUNG_PATHS)
    assert plain.path is VoicePath.NONE and plain.servable is False

    local = await cascade.resolve_voice_path(config_dir=tmp_path, store=store, include_local=True)
    assert local.rungs[-1].path is VoicePath.LOCAL_TTS_SYSTEM
    assert local.path is VoicePath.LOCAL_TTS_SYSTEM and local.servable is True


@pytest.mark.asyncio
async def test_without_say_the_opted_in_cascade_is_still_unservable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(local_say, "say_binary", lambda: None)
    store: Any = EmptyStore()
    resolution = await cascade.resolve_voice_path(
        config_dir=tmp_path, store=store, include_local=True
    )
    assert resolution.path is VoicePath.NONE
    assert resolution.rungs[-1].reason == local_say.UNAVAILABLE_REASON


@pytest.mark.asyncio
async def test_a_failing_say_falls_through_as_a_failed_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(local_say, "say_binary", lambda: "/usr/bin/say")

    async def broken(text: str, *, timeout_s: float) -> bytes:
        raise local_say.LocalSayError("say exited with status 1")

    monkeypatch.setattr(local_say, "synthesize", broken)
    store: Any = EmptyStore()
    with pytest.raises(cascade.TtsUnavailable) as raised:
        await cascade.synthesize_speech(
            "hello", config_dir=tmp_path, store=store, include_local=True
        )
    assert raised.value.failed_path is VoicePath.LOCAL_TTS_SYSTEM
    assert [attempt.outcome for attempt in raised.value.attempts] == ["failed"]


@pytest.mark.skipif(not ON_MAC_WITH_SAY, reason="needs macOS say")
@pytest.mark.asyncio
async def test_say_renders_a_real_wav(tmp_path: Path) -> None:
    """The real binary, not a double: the bytes are a RIFF/WAVE container."""
    store: Any = EmptyStore()
    outcome = await cascade.synthesize_speech(
        "-n this leading dash is text, not an option",
        config_dir=tmp_path,
        store=store,
        include_local=True,
    )
    assert outcome.path is VoicePath.LOCAL_TTS_SYSTEM
    assert outcome.provider == local_say.PROVIDER
    assert outcome.audio[:4] == b"RIFF" and outcome.audio[8:12] == b"WAVE"
    assert len(outcome.audio) > 1000
    assert ("X-Radient-Speech-Path", "local_tts_system") in outcome.speech_headers


@pytest.mark.skipif(not ON_MAC_WITH_SAY, reason="needs macOS say")
@pytest.mark.asyncio
async def test_a_timed_out_say_leaves_no_process_behind() -> None:
    """The bound kills the child's whole group; nothing outlives the request."""
    with pytest.raises(asyncio.TimeoutError):
        await local_say.synthesize("word " * 5000, timeout_s=0.05)
