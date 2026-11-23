"""The Groq STT rung (session-voice-groq D1, B1): wire, opt-in and failure classes.

Fixtures are shaped from Groq's documented API (OpenAI-compatible
``/audio/transcriptions``; ``retry-after`` on a 429 and
``x-ratelimit-remaining-requests`` for the daily count). No real call is made.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

from local_operator.clients._http import APIError
from local_operator.providers.auth_store import AuthStore
from local_operator.stt import AudioPath, cascade, clients
from local_operator.stt.errors import failure_status_and_detail

CANONICAL = "https://api.radienthq.com/v1"


def _wav() -> bytes:
    """A real, tiny PCM WAV so the executor's sniffer names it ``audio/wav``."""
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


def _store_groq(store: AuthStore, key: str = "gsk_test") -> None:
    store.upsert_credential("groq", {"type": "api_key", "source": "login", "key": key})


def _audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "take.wav"
    path.write_bytes(_wav())
    return path


async def _run(store: AuthStore, tmp_path: Path, handler: Any, **kwargs: Any) -> Any:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        return await cascade.transcribe_audio(
            _audio_file(tmp_path),
            config_dir=tmp_path / "config",
            base_url=CANONICAL,
            store=store,
            include_groq=True,
            groq_client=http,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_groq_call_has_the_exact_url_headers_and_body(
    store: AuthStore, tmp_path: Path
) -> None:
    _store_groq(store)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "hello there", "x_groq": {"id": "req_1"}})

    outcome = await _run(store, tmp_path, handler)

    assert outcome.path == AudioPath.PROVIDER_STT_GROQ
    assert outcome.text == "hello there"
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer gsk_test"
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.read()
    assert b'name="model"' in body and b"whisper-large-v3-turbo" in body
    assert b'name="response_format"' in body and b"json" in body
    # WAV passes through untouched, as a .wav part with the sniffed type.
    assert b'filename="audio.wav"' in body and b"audio/wav" in body
    assert _wav() in body


@pytest.mark.asyncio
async def test_a_pinned_model_never_falls_back() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.read().decode("latin-1"))
        return httpx.Response(
            404, json={"error": {"message": "model not found", "code": "model_not_found"}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient(
            "gsk_test", base_url=clients.GROQ_STT_BASE_URL, label="Groq", client=http
        )
        with pytest.raises(APIError) as caught:
            await client.transcribe(_wav(), mime="audio/wav", model=clients.GROQ_STT_MODEL)

    assert len(calls) == 1, "a refusal of the pinned model must not try another id"
    assert "whisper-1" not in calls[0]
    assert caught.value.status_code == 404


@pytest.mark.asyncio
async def test_label_names_the_vendor_in_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": 1})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient(
            "k", base_url="https://x.test/v1", label="Groq", client=http
        )
        with pytest.raises(APIError, match="Groq returned a response without transcript"):
            await client.transcribe(b"x", mime="audio/wav", model="m")


@pytest.mark.asyncio
async def test_default_label_is_still_openai() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": 1})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient("k", client=http)
        with pytest.raises(APIError, match="OpenAI returned a response without transcript"):
            await client.transcribe(b"x", mime="audio/wav", model="m")


@pytest.mark.asyncio
async def test_429_carries_retry_after_and_remaining_requests(
    store: AuthStore, tmp_path: Path
) -> None:
    _store_groq(store)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "7", "x-ratelimit-remaining-requests": "0"},
            json={"error": {"message": "Rate limit reached", "type": "requests"}},
        )

    with pytest.raises(cascade.SttUnavailable) as caught:
        await _run(store, tmp_path, handler)

    error = caught.value.error
    assert error is not None and error.status_code == 429
    assert error.details["retry_after_s"] == 7.0
    assert error.details["remaining_requests"] == 0
    assert caught.value.status_code == 429
    assert "Groq" in (caught.value.detail or "")


@pytest.mark.asyncio
async def test_429_without_headers_omits_the_numbers(store: AuthStore, tmp_path: Path) -> None:
    _store_groq(store)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(cascade.SttUnavailable) as caught:
        await _run(store, tmp_path, handler)

    assert caught.value.error is not None
    assert "retry_after_s" not in caught.value.error.details
    assert "remaining_requests" not in caught.value.error.details


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, 502), (402, 402), (429, 429)],
)
def test_401_402_429_are_classified_as_groq(status: int, expected: int) -> None:
    exc = APIError("refused", status_code=status, body='{"error":{"message":"no"}}')
    code, detail = failure_status_and_detail(exc, "upstream", rung=AudioPath.PROVIDER_STT_GROQ)
    assert code == expected
    assert "Groq" in detail
    assert "OpenAI" not in detail and "Radient" not in detail


@pytest.mark.asyncio
async def test_an_exported_groq_key_does_not_light_the_rung(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_ambient")
    store.set_runtime_api_key("groq", "gsk_flag")
    resolution = await cascade.resolve_audio_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store, include_groq=True
    )
    groq = next(r for r in resolution.rungs if r.path == AudioPath.PROVIDER_STT_GROQ)
    assert groq.available is False

    _store_groq(store)
    resolution = await cascade.resolve_audio_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store, include_groq=True
    )
    assert resolution.path == AudioPath.PROVIDER_STT_GROQ


@pytest.mark.asyncio
async def test_groq_is_the_last_provider_rung(store: AuthStore, tmp_path: Path) -> None:
    resolution = await cascade.resolve_audio_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store, include_groq=True
    )
    paths = [r.path for r in resolution.rungs]
    assert (
        paths.index(AudioPath.PROVIDER_STT_GROQ) == paths.index(AudioPath.PROVIDER_STT_OPENAI) + 1
    )


@pytest.mark.asyncio
async def test_default_never_reaches_groq(store: AuthStore, tmp_path: Path) -> None:
    _store_groq(store)
    called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(str(request.url))
        return httpx.Response(200, json={"text": "x"})

    resolution = await cascade.resolve_audio_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    assert AudioPath.PROVIDER_STT_GROQ not in [r.path for r in resolution.rungs]
    assert resolution.path == AudioPath.NONE

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(cascade.SttUnavailable):
            await cascade.transcribe_audio(
                _audio_file(tmp_path),
                config_dir=tmp_path / "config",
                base_url=CANONICAL,
                store=store,
                groq_client=http,
            )
    assert called == []


@pytest.mark.asyncio
async def test_the_rung_names_itself_when_it_times_out_label() -> None:
    assert cascade._rung_label(AudioPath.PROVIDER_STT_GROQ) == "Groq"
