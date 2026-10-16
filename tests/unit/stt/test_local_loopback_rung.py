"""R8: the opt-in ``local_stt_loopback`` rung, against a real loopback server.

The fake is a uvicorn server bound to 127.0.0.1 (the in-repo pattern of
``tests/unit/test_tunnel_sockets.py``) serving the OpenAI-compatible
``POST /v1/audio/transcriptions`` mlx_audio exposes (``mlx_audio/server.py``:
multipart ``file``, form ``model``, form ``response_format=json`` ->
``{"text": ...}``). It is NOT mlx_audio: the point is the wire, and the real
httpx client crosses a real socket to reach it.
"""

from __future__ import annotations

import asyncio
import io
import socket
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from local_operator.stt import AudioPath, cascade
from local_operator.stt.cascade import SttUnavailable


def _wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class _Loopback:
    def __init__(self, port: int, requests: list[dict[str, Any]], state: dict[str, Any]) -> None:
        self.port = port
        self.requests = requests
        self.state = state

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _app(requests: list[dict[str, Any]], state: dict[str, Any]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(...),
        response_format: str = Form("ndjson"),
    ) -> JSONResponse:
        requests.append(
            {
                "path": request.url.path,
                "authorization": request.headers.get("authorization"),
                "model": model,
                "response_format": response_format,
                "audio_head": (await file.read())[:4],
            }
        )
        status, payload = state["reply"]
        return JSONResponse(payload, status_code=status)

    return app


@asynccontextmanager
async def _serve(app: FastAPI) -> AsyncIterator[int]:
    ready = asyncio.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            ready.set()

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.setblocking(False)
    running = ReadyServer(
        uvicorn.Config(app, log_level="error", access_log=False, timeout_graceful_shutdown=1)
    )
    task = asyncio.create_task(running.serve(sockets=[listener]))
    try:
        await asyncio.wait_for(ready.wait(), timeout=10)
        yield listener.getsockname()[1]
    finally:
        running.should_exit = True
        await task
        listener.close()


@pytest_asyncio.fixture
async def loopback() -> AsyncIterator[_Loopback]:
    requests: list[dict[str, Any]] = []
    state: dict[str, Any] = {"reply": (200, {"text": " list files "})}
    async with _serve(_app(requests, state)) as port:
        yield _Loopback(port, requests, state)


class _NoKeys:
    """An AuthStore with nothing stored: every credentialed rung is unavailable."""

    async def has_persisted_credential(self, provider: str, session_id: Optional[str] = None):
        return False

    async def get_persisted_api_key(self, provider: str, session_id=None, *, kinds=None):
        return None

    async def get_api_key(self, provider: str, session_id=None, **_: Any):
        return None


@pytest.fixture(autouse=True)
def _no_radient(monkeypatch: pytest.MonkeyPatch) -> None:
    async def probe(_config_dir, _base_url, *, store):
        return False

    monkeypatch.setattr(cascade, "has_persisted_radient_credential", probe)


async def _resolve(local_stt_url: Optional[str]):
    return await cascade.resolve_audio_path(
        config_dir=Path("/nonexistent-config"),
        base_url="https://radient.invalid/v1",
        store=_NoKeys(),  # type: ignore[arg-type]
        local_stt_url=local_stt_url,
    )


@pytest.mark.asyncio
async def test_without_the_opt_in_the_rung_does_not_exist() -> None:
    """``/v1/stt/paths`` and every other caller pass nothing: unchanged."""
    resolution = await _resolve(None)
    assert AudioPath.LOCAL_STT_LOOPBACK not in [rung.path for rung in resolution.rungs]
    assert resolution.path == AudioPath.NONE


@pytest.mark.asyncio
async def test_an_empty_or_remote_url_is_listed_but_unavailable() -> None:
    empty = await _resolve("")
    (rung,) = [r for r in empty.rungs if r.path == AudioPath.LOCAL_STT_LOOPBACK]
    assert rung.available is False and "No local transcription server" in rung.reason
    remote = await _resolve("http://10.0.0.5:8000")
    (rung,) = [r for r in remote.rungs if r.path == AudioPath.LOCAL_STT_LOOPBACK]
    assert rung.available is False and "on this machine" in rung.reason
    assert remote.path == AudioPath.NONE


@pytest.mark.asyncio
async def test_the_rung_sits_after_the_credentialed_rungs(loopback: _Loopback) -> None:
    resolution = await _resolve(loopback.url)
    paths = [rung.path for rung in resolution.rungs]
    assert paths.index(AudioPath.LOCAL_STT_LOOPBACK) > paths.index(AudioPath.PROVIDER_STT_OPENAI)
    assert paths.index(AudioPath.LOCAL_STT_LOOPBACK) < paths.index(AudioPath.MODEL_AUDIO_SIDECAR)
    assert resolution.path == AudioPath.LOCAL_STT_LOOPBACK


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "/", "/v1"])
async def test_the_executor_transcribes_through_the_loopback_server(
    loopback: _Loopback, tmp_path: Path, suffix: str
) -> None:
    audio = tmp_path / "take.wav"
    audio.write_bytes(_wav())
    outcome = await cascade.transcribe_audio(
        audio,
        config_dir=tmp_path,
        base_url="https://radient.invalid/v1",
        store=_NoKeys(),  # type: ignore[arg-type]
        local_stt_url=loopback.url + suffix,
    )
    assert outcome.path == AudioPath.LOCAL_STT_LOOPBACK
    assert outcome.text == " list files "
    assert outcome.cost is None and outcome.provider == ""
    (seen,) = loopback.requests
    assert seen["path"] == "/v1/audio/transcriptions"
    assert seen["model"] == cascade.LOCAL_STT_MODEL
    assert seen["response_format"] == "json"
    assert seen["audio_head"] == b"RIFF"
    # No stored key ever travels to a local process: the placeholder does.
    assert seen["authorization"] == f"Bearer {cascade.LOCAL_STT_PLACEHOLDER_KEY}"


@pytest.mark.asyncio
async def test_a_refusing_local_server_is_named_as_the_local_server(
    loopback: _Loopback, tmp_path: Path
) -> None:
    # A 4xx is where the shared mapper names the rung (a 5xx reads "failed
    # upstream" for every rung alike); 404 is the misconfigured-URL case.
    loopback.state["reply"] = (404, {"error": {"message": "Not Found"}})
    audio = tmp_path / "take.wav"
    audio.write_bytes(_wav())
    with pytest.raises(SttUnavailable) as raised:
        await cascade.transcribe_audio(
            audio,
            config_dir=tmp_path,
            store=_NoKeys(),  # type: ignore[arg-type]
            local_stt_url=loopback.url,
        )
    (attempt,) = raised.value.attempts
    assert attempt.path == AudioPath.LOCAL_STT_LOOPBACK and attempt.outcome == "failed"
    detail = raised.value.detail or ""
    assert "OpenAI" not in detail
    assert "the local STT server transcription endpoint was not found" in detail


@pytest.mark.asyncio
async def test_a_down_local_server_is_a_transport_failure(tmp_path: Path) -> None:
    audio = tmp_path / "take.wav"
    audio.write_bytes(_wav())
    with pytest.raises(SttUnavailable) as raised:
        await cascade.transcribe_audio(
            audio,
            config_dir=tmp_path,
            store=_NoKeys(),  # type: ignore[arg-type]
            local_stt_url="http://127.0.0.1:9",
        )
    assert raised.value.status_code == 502
    assert "The local transcription server request failed" in (raised.value.detail or "")
