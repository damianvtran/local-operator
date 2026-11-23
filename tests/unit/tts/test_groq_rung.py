"""The Groq TTS rung (session-voice-groq D1, B1): wire, guards and failure classes.

Fixtures are shaped from Groq's documented ``/audio/speech`` (Orpheus, wav,
200-character input limit). No real call is made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

from local_operator.clients._http import APIError
from local_operator.providers.auth_store import AuthStore
from local_operator.server.routes import speech as speech_routes
from local_operator.tts import VoicePath, cascade
from local_operator.tts import clients as tts_clients
from local_operator.tts.descriptor import VoiceDescriptor

CANONICAL = "https://api.radienthq.com/v1"
RIFF = b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio"


@pytest.fixture(autouse=True)
def _empty_radient(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_probe(_config_dir: Any, _base_url: Any, *, store: Any) -> bool:
        return False

    monkeypatch.setattr(cascade, "has_persisted_radient_credential", no_probe)


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-config"))
    for var in ("GROQ_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path / "config")
    yield auth
    auth.close()


def _store_groq(store: AuthStore) -> None:
    store.upsert_credential("groq", {"type": "api_key", "source": "login", "key": "gsk_test"})


async def _speak(
    store: AuthStore, tmp_path: Path, text: str, handler: Any, *, gender: str = "female", **kw: Any
) -> Any:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        groq = tts_clients.OpenAiTtsClient(
            "gsk_test", base_url=tts_clients.GROQ_TTS_BASE_URL, label="Groq", client=http
        )
        return await cascade.synthesize_speech(
            text,
            config_dir=tmp_path / "config",
            base_url=CANONICAL,
            store=store,
            descriptor=VoiceDescriptor(gender=gender),
            include_groq=True,
            groq_client=groq,
            **kw,
        )


@pytest.mark.asyncio
async def test_groq_call_has_the_exact_url_headers_and_body(
    store: AuthStore, tmp_path: Path
) -> None:
    _store_groq(store)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=RIFF, headers={"content-type": "audio/wav"})

    outcome = await _speak(store, tmp_path, "Hello there.", handler, response_format="mp3")

    assert outcome.path == VoicePath.PROVIDER_TTS_GROQ
    assert outcome.audio == RIFF
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.groq.com/openai/v1/audio/speech"
    assert request.headers["authorization"] == "Bearer gsk_test"
    assert request.headers["content-type"] == "application/json"
    # Exactly Groq's four fields: no instructions, no speed.
    assert json.loads(request.read()) == {
        "model": "canopylabs/orpheus-v1-english",
        "input": "Hello there.",
        "voice": "hannah",
        "response_format": "wav",
    }


@pytest.mark.asyncio
async def test_male_gender_maps_to_troy(store: AuthStore, tmp_path: Path) -> None:
    _store_groq(store)
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return httpx.Response(200, content=RIFF)

    outcome = await _speak(store, tmp_path, "Hi.", handler, gender="male")
    assert bodies[0]["voice"] == "troy"
    assert dict(outcome.speech_headers)["X-Radient-Speech-Voice"] == "troy"
    assert tts_clients.GROQ_TTS_VOICES == {"female": "hannah", "male": "troy"}


@pytest.mark.asyncio
async def test_brackets_are_escaped_to_parentheses(store: AuthStore, tmp_path: Path) -> None:
    _store_groq(store)
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return httpx.Response(200, content=RIFF)

    await _speak(store, tmp_path, "See [1] and [note] now.", handler)
    assert bodies[0]["input"] == "See (1) and (note) now."


@pytest.mark.asyncio
async def test_over_200_characters_is_refused_before_any_network_call(
    store: AuthStore, tmp_path: Path
) -> None:
    _store_groq(store)
    called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(str(request.url))
        return httpx.Response(200, content=RIFF)

    with pytest.raises(cascade.TtsUnavailable) as caught:
        await _speak(store, tmp_path, "a" * 201, handler)
    assert called == []
    assert isinstance(caught.value.error, APIError)
    assert "200" in str(caught.value.error)

    # The boundary itself is allowed.
    outcome = await _speak(store, tmp_path, "a" * 200, handler)
    assert outcome.path == VoicePath.PROVIDER_TTS_GROQ
    assert len(called) == 1


@pytest.mark.asyncio
async def test_429_carries_retry_after_and_remaining_requests(
    store: AuthStore, tmp_path: Path
) -> None:
    _store_groq(store)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "42", "x-ratelimit-remaining-requests": "0"},
            json={"error": {"message": "Rate limit reached for requests per day"}},
        )

    with pytest.raises(cascade.TtsUnavailable) as caught:
        await _speak(store, tmp_path, "Hi.", handler)
    error = caught.value.error
    assert isinstance(error, APIError) and error.status_code == 429
    assert error.details["retry_after_s"] == 42.0
    assert error.details["remaining_requests"] == 0
    assert caught.value.failed_path == VoicePath.PROVIDER_TTS_GROQ


@pytest.mark.parametrize(("status", "expected"), [(401, 401), (402, 402), (429, 429)])
def test_401_402_429_are_classified_as_groq(status: int, expected: int) -> None:
    exc = APIError("refused", status_code=status, body='{"error":{"message":"no"}}')
    http = speech_routes._speech_refusal(exc, rung=VoicePath.PROVIDER_TTS_GROQ)
    assert http.status_code == expected
    if status != 429:  # 429 keeps the generic sentence by design
        assert "Groq" in json.dumps(http.detail)
        assert "OpenAI" not in json.dumps(http.detail)


@pytest.mark.asyncio
async def test_an_exported_groq_key_does_not_light_the_rung(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_ambient")
    store.set_runtime_api_key("groq", "gsk_flag")
    resolution = await cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store, include_groq=True
    )
    groq = next(r for r in resolution.rungs if r.path == VoicePath.PROVIDER_TTS_GROQ)
    assert groq.available is False
    assert resolution.servable is False

    _store_groq(store)
    resolution = await cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store, include_groq=True
    )
    assert resolution.path == VoicePath.PROVIDER_TTS_GROQ


@pytest.mark.asyncio
async def test_default_never_reaches_groq(store: AuthStore, tmp_path: Path) -> None:
    _store_groq(store)
    resolution = await cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    assert VoicePath.PROVIDER_TTS_GROQ not in [r.path for r in resolution.rungs]
    assert resolution.servable is False

    called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(str(request.url))
        return httpx.Response(200, content=RIFF)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        groq = tts_clients.OpenAiTtsClient(
            "k", base_url=tts_clients.GROQ_TTS_BASE_URL, label="Groq", client=http
        )
        with pytest.raises(cascade.TtsUnavailable):
            await cascade.synthesize_speech(
                "Hi.",
                config_dir=tmp_path / "config",
                base_url=CANONICAL,
                store=store,
                descriptor=VoiceDescriptor(),
                groq_client=groq,
            )
    assert called == []


@pytest.mark.asyncio
async def test_label_names_the_vendor_in_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    from local_operator.tts.adapters import Params

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        groq = tts_clients.OpenAiTtsClient(
            "k", base_url="https://x.test/v1", label="Groq", client=http
        )
        with pytest.raises(APIError, match="Groq"):
            await groq.synthesize("hi", params=Params(voice_id="hannah"))
