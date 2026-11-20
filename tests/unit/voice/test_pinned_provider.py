"""A voice request can be PINNED to one provider (bring-your-own-key voice).

The speech cascades walk Radient first, then the user's own keys. A user who
chose "bring your own ElevenLabs key" must never be served by, or billed to,
another rung: not a Radient login that happens to be stored, not a Groq or
OpenAI key. Pinning narrows the walk to exactly one rung; if that rung has no
key the request is unavailable, and nothing else is tried.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.stt import AudioPath
from local_operator.stt import cascade as stt_cascade
from local_operator.stt.cascade import SttUnavailable
from local_operator.tts import VoicePath
from local_operator.tts import cascade as tts_cascade
from local_operator.tts.cascade import TtsUnavailable
from local_operator.tts.clients import TtsClientResult
from local_operator.tts.descriptor import VoiceDescriptor

CANONICAL = "https://api.radienthq.com/v1"


def _wav() -> bytes:
    data = b"\x00\x00" * 160
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-config"))
    for var in ("GROQ_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "RADIENT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path / "config")
    yield auth
    auth.close()


def _key(store: AuthStore, namespace: str) -> None:
    store.upsert_credential(
        namespace, {"type": "api_key", "source": "login", "key": f"{namespace}-test-key"}
    )


@pytest.fixture()
def radient_signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Radient login IS stored: the rung that would win an unpinned walk."""

    async def yes(_config_dir: Any, _base_url: Any, *, store: Any) -> bool:
        return True

    monkeypatch.setattr(tts_cascade, "has_persisted_radient_credential", yes)
    monkeypatch.setattr(stt_cascade, "has_persisted_radient_credential", yes)


class _Boom:
    """A seam that fails the test if the rung is ever called."""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("an unpinned rung was called")

    async def synthesize(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("an unpinned rung was called")


class _ElevenLabs:
    def __init__(self) -> None:
        self.calls = 0

    async def synthesize(self, text: str, **kwargs: Any) -> TtsClientResult:
        self.calls += 1
        return TtsClientResult(
            audio=b"ID3fake-mp3", model="eleven_flash_v2_5", provider="elevenlabs"
        )


# -- TTS -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_unpinned_walk_prefers_radient(
    store: AuthStore, tmp_path: Path, radient_signed_in: None
) -> None:
    """The baseline the pin exists to override."""
    _key(store, "elevenlabs")
    resolution = await tts_cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    assert resolution.path == VoicePath.PROVIDER_TTS_RADIENT


@pytest.mark.asyncio
async def test_tts_pinned_to_elevenlabs_skips_a_stored_radient_login(
    store: AuthStore, tmp_path: Path, radient_signed_in: None
) -> None:
    _key(store, "elevenlabs")
    eleven = _ElevenLabs()
    outcome = await tts_cascade.synthesize_speech(
        "Hello there.",
        config_dir=tmp_path / "config",
        base_url=CANONICAL,
        store=store,
        descriptor=VoiceDescriptor(gender="female"),
        radient_client=_Boom(),  # type: ignore[arg-type]
        elevenlabs_client=eleven,
        only=VoicePath.PROVIDER_TTS_ELEVENLABS,
    )
    assert outcome.path == VoicePath.PROVIDER_TTS_ELEVENLABS
    assert eleven.calls == 1
    assert [a.path for a in outcome.attempts] == [VoicePath.PROVIDER_TTS_ELEVENLABS]


@pytest.mark.asyncio
async def test_tts_pinned_resolution_names_the_pinned_rung_even_when_radient_wins_unpinned(
    store: AuthStore, tmp_path: Path, radient_signed_in: None
) -> None:
    _key(store, "elevenlabs")
    resolution = await tts_cascade.resolve_voice_path(
        config_dir=tmp_path / "config",
        base_url=CANONICAL,
        store=store,
        only=VoicePath.PROVIDER_TTS_ELEVENLABS,
    )
    assert resolution.path == VoicePath.PROVIDER_TTS_ELEVENLABS and resolution.servable


@pytest.mark.asyncio
async def test_tts_pinned_to_a_rung_with_no_key_is_unavailable_and_tries_nothing_else(
    store: AuthStore, tmp_path: Path, radient_signed_in: None
) -> None:
    _key(store, "groq")
    _key(store, "openai")  # other rungs ARE available; none may be used
    with pytest.raises(TtsUnavailable) as caught:
        await tts_cascade.synthesize_speech(
            "Hello there.",
            config_dir=tmp_path / "config",
            base_url=CANONICAL,
            store=store,
            descriptor=VoiceDescriptor(gender="female"),
            radient_client=_Boom(),  # type: ignore[arg-type]
            openai_client=_Boom(),
            groq_client=_Boom(),
            include_groq=True,
            only=VoicePath.PROVIDER_TTS_ELEVENLABS,
        )
    assert caught.value.error is None, "no rung was attempted, so there is no upstream error"
    assert "ElevenLabs" in str(caught.value.resolution.reason)
    resolution = await tts_cascade.resolve_voice_path(
        config_dir=tmp_path / "config",
        base_url=CANONICAL,
        store=store,
        include_groq=True,
        only=VoicePath.PROVIDER_TTS_ELEVENLABS,
    )
    assert resolution.servable is False and resolution.path == VoicePath.NONE


# -- STT -------------------------------------------------------------------------


class _ElevenLabsStt:
    calls = 0

    def __init__(self, api_key: str, **_: Any) -> None:
        assert api_key == "elevenlabs-test-key"

    async def transcribe(self, audio: bytes, **kwargs: Any) -> Any:
        type(self).calls += 1
        from local_operator.stt.clients import SttClientResult

        return SttClientResult(text="list the files", model="scribe_v2", provider="elevenlabs")


def _take(tmp_path: Path) -> Path:
    path = tmp_path / "take.wav"
    path.write_bytes(_wav())
    return path


@pytest.mark.asyncio
async def test_stt_pinned_to_elevenlabs_skips_a_stored_radient_login(
    store: AuthStore, tmp_path: Path, radient_signed_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _key(store, "elevenlabs")
    _ElevenLabsStt.calls = 0
    monkeypatch.setattr(stt_cascade, "ElevenLabsSttClient", _ElevenLabsStt)
    monkeypatch.setattr(stt_cascade, "_run_radient_rung", _Boom())
    outcome = await stt_cascade.transcribe_audio(
        _take(tmp_path),
        config_dir=tmp_path / "config",
        base_url=CANONICAL,
        store=store,
        only=AudioPath.PROVIDER_STT_ELEVENLABS,
    )
    assert outcome.path == AudioPath.PROVIDER_STT_ELEVENLABS
    assert outcome.text == "list the files"
    assert _ElevenLabsStt.calls == 1


@pytest.mark.asyncio
async def test_stt_pinned_to_a_rung_with_no_key_is_unavailable_and_tries_nothing_else(
    store: AuthStore, tmp_path: Path, radient_signed_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _key(store, "groq")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        seen.append(request)
        return httpx.Response(200, json={"text": "should never run"})

    monkeypatch.setattr(stt_cascade, "_run_radient_rung", _Boom())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(SttUnavailable) as caught:
            await stt_cascade.transcribe_audio(
                _take(tmp_path),
                config_dir=tmp_path / "config",
                base_url=CANONICAL,
                store=store,
                include_groq=True,
                groq_client=http,
                only=AudioPath.PROVIDER_STT_ELEVENLABS,
            )
    assert seen == [], "Groq was reached although the request was pinned to ElevenLabs"
    assert not caught.value.attempts, "no rung may be attempted when the pinned rung has no key"
    assert "ElevenLabs" in str(caught.value)
