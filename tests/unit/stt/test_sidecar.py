"""Sidecar: statuses, bounds, the durable record, and cancellation on dispose."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.clients._http import APIError
from local_operator.harness.types import ModelSpec
from local_operator.session.session import Session
from local_operator.session.transcript import ENTRY_CUSTOM, Transcript, replay_entries
from local_operator.stt import AudioPath, sidecar

_WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 16


class Block:
    """The little of ``AudioContent`` the sidecar reads (phase 1 passes the real block)."""

    def __init__(self, data: bytes = _WAV_BYTES, mime: str = "audio/wav"):
        self.data = base64.b64encode(data).decode("ascii")
        self.mime_type = mime


class Store:
    def __init__(self, keys=None):
        self.keys = dict(keys or {})

    async def get_api_key(self, provider, session_id=None, *, read_only=False, **kwargs):
        return self.keys.get(provider)


class Host:
    """A minimal ``SidecarHost``: records spawns instead of opening a task group."""

    def __init__(self, transcript: Transcript, model=None):
        self._transcript = transcript
        self._model = (
            model
            if model is not None
            else SimpleNamespace(provider="openrouter", model_id="openrouter/some-audio-model")
        )
        self.spawned: list[asyncio.Task[Any]] = []

    @property
    def transcript(self) -> Transcript:
        return self._transcript

    @property
    def model(self):
        return self._model

    def _spawn_background(self, coro):
        task = asyncio.ensure_future(coro)
        self.spawned.append(task)
        return task


def _record(transcript: Transcript) -> dict[str, Any]:
    record = transcript.latest_custom(sidecar.STT_TRANSCRIPT_CUSTOM_TYPE)
    assert record is not None
    return record


@pytest.mark.asyncio
async def test_fork_never_raises_when_the_host_refuses_to_spawn(tmp_path, monkeypatch) -> None:
    """Starting is best-effort: a torn-down host must not fail the message."""
    transcript = Transcript(tmp_path / "sess")
    host = Host(transcript)

    def refuse(coro):
        raise RuntimeError("no running loop")

    host._spawn_background = refuse  # type: ignore[method-assign]

    class FakeCoro:
        closed = False

        def close(self):
            self.closed = True

    fake = FakeCoro()

    def build(*args, **kwargs):
        return fake

    monkeypatch.setattr(sidecar, "_run_audio_sidecar", build)
    sidecar.fork_audio_sidecar(
        host, message_id="m9", audio=Block(), config_dir=tmp_path, store=Store()
    )
    assert fake.closed is True
    assert transcript.latest_custom(sidecar.STT_TRANSCRIPT_CUSTOM_TYPE) is None


@pytest.mark.asyncio
async def test_ok_writes_the_exact_record_and_nothing_else(tmp_path, monkeypatch, caplog) -> None:
    transcript = Transcript(tmp_path / "sess")
    host = Host(transcript)

    async def fake(session, audio, *, store, timeout_s, client=None):
        assert isinstance(audio, Block)  # the block travels through untouched
        return "hello there"

    monkeypatch.setattr(sidecar, "_model_transcript", fake)

    with caplog.at_level(logging.WARNING, logger="local_operator.stt.sidecar"):
        await sidecar._run_audio_sidecar(
            host, message_id="msg-1", audio=Block(), config_dir=tmp_path, store=Store()
        )

    # THE HAPPY PATH IS SILENT (agent review round 1, m1): the WARNING the
    # failed and timed-out paths owe the operator must not fire for an ok run.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    record = _record(transcript)
    assert set(record) == {"message_id", "status", "path", "text", "error", "at"}
    assert record["message_id"] == "msg-1"
    assert record["status"] == "ok"
    assert record["path"] == AudioPath.MODEL_AUDIO_SIDECAR.value
    assert record["text"] == "hello there"
    assert record["error"] is None
    assert isinstance(record["at"], float)

    entry = transcript.latest_custom_entry(sidecar.STT_TRANSCRIPT_CUSTOM_TYPE)
    assert entry is not None and entry.type == ENTRY_CUSTOM
    # SILENCE: the record never replays into the model's context (nor even the
    # audit stream) — it is bookkeeping, not a message.
    assert replay_entries(transcript.entries(), None, mode="audit") == []
    assert replay_entries(transcript.entries(), None) == []


@pytest.mark.asyncio
async def test_failed_records_the_error(tmp_path, monkeypatch, caplog) -> None:
    transcript = Transcript(tmp_path / "sess")
    host = Host(transcript)

    async def fake(session, audio, *, store, timeout_s, client=None):
        raise RuntimeError("the wire said no")

    monkeypatch.setattr(sidecar, "_model_transcript", fake)
    with caplog.at_level(logging.WARNING, logger="local_operator.stt.sidecar"):
        await sidecar._run_audio_sidecar(
            host, message_id="m2", audio=Block(), config_dir=tmp_path, store=Store()
        )

    record = _record(transcript)
    assert record["status"] == "failed"
    assert record["error"] == "the wire said no"
    assert record["path"] is None and record["text"] is None
    # THE OPERATOR-FACING HALF (agent review round 1, m1 / QA Q2): a failure
    # is written to the transcript AND logged at warning, because v1 renders
    # the record nowhere and a sidecar that whispers only into a JSONL file
    # is invisible in the one place failures are watched.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("failed" in r.getMessage() and "m2" in r.getMessage() for r in warnings), [
        r.getMessage() for r in warnings
    ]


@pytest.mark.asyncio
async def test_unavailable_records_the_reason(tmp_path, monkeypatch, caplog) -> None:
    transcript = Transcript(tmp_path / "sess")
    host = Host(transcript)

    async def fake(session, audio, *, store, timeout_s, client=None):
        raise sidecar.ModelTranscriptUnavailable("the anthropic wire cannot take audio")

    monkeypatch.setattr(sidecar, "_model_transcript", fake)
    with caplog.at_level(logging.WARNING, logger="local_operator.stt.sidecar"):
        await sidecar._run_audio_sidecar(
            host, message_id="m3", audio=Block(), config_dir=tmp_path, store=Store()
        )

    record = _record(transcript)
    assert record["status"] == "unavailable"
    assert record["error"] == "the anthropic wire cannot take audio"
    # DELIBERATELY SILENT (agent review round 1, m1 disposition): unavailable
    # is the deterministic honest answer on a machine with no route for the
    # bounded model call — e.g. no provider key on a model-audio door send —
    # not a fault; warning on it would train the operator to ignore the line
    # the failed/timeout paths actually need read.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_still_recorded(tmp_path, monkeypatch, caplog) -> None:
    transcript = Transcript(tmp_path / "sess")
    host = Host(transcript)
    monkeypatch.setattr(sidecar, "SIDECAR_TIMEOUT_S", 0.05)

    started = asyncio.Event()

    async def fake(session, audio, *, store, timeout_s, client=None):
        started.set()
        await asyncio.sleep(5)
        return "never"

    monkeypatch.setattr(sidecar, "_model_transcript", fake)
    with caplog.at_level(logging.WARNING, logger="local_operator.stt.sidecar"):
        await sidecar._run_audio_sidecar(
            host, message_id="m4", audio=Block(), config_dir=tmp_path, store=Store()
        )

    assert started.is_set()
    record = _record(transcript)
    assert record["status"] == "timeout"
    assert "did not return a transcript within" in record["error"]
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("timeout" in r.getMessage() and "m4" in r.getMessage() for r in warnings), [
        r.getMessage() for r in warnings
    ]


@pytest.mark.asyncio
async def test_an_empty_message_never_loses_the_clue(tmp_path, monkeypatch) -> None:
    transcript = Transcript(tmp_path / "sess")
    host = Host(transcript)

    async def fake(session, audio, *, store, timeout_s, client=None):
        raise ValueError("")

    monkeypatch.setattr(sidecar, "_model_transcript", fake)
    await sidecar._run_audio_sidecar(
        host, message_id="m5", audio=Block(), config_dir=tmp_path, store=Store()
    )
    record = _record(transcript)
    assert record["status"] == "failed"
    assert record["error"] == "ValueError"


@pytest.mark.asyncio
async def test_fork_registers_with_the_session_spawn(tmp_path, monkeypatch) -> None:
    transcript = Transcript(tmp_path / "sess")
    host = Host(transcript)

    async def fake(session, audio, *, store, timeout_s, client=None):
        return "spawned text"

    monkeypatch.setattr(sidecar, "_model_transcript", fake)
    sidecar.fork_audio_sidecar(
        host, message_id="m6", audio=Block(), config_dir=tmp_path, store=Store()
    )

    assert len(host.spawned) == 1
    await host.spawned[0]
    assert _record(transcript)["status"] == "ok"


@pytest.mark.asyncio
async def test_a_record_that_cannot_be_written_is_swallowed(tmp_path, monkeypatch, caplog) -> None:
    transcript = Transcript(tmp_path / "sess")
    host = Host(transcript)

    async def fake(session, audio, *, store, timeout_s, client=None):
        return "text"

    async def exploding_append(custom_type, details, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sidecar, "_model_transcript", fake)
    monkeypatch.setattr(transcript, "append_custom", exploding_append)
    # Must not raise (the sidecar never fails the turn it rode in on).
    await sidecar._run_audio_sidecar(
        host, message_id="m7", audio=Block(), config_dir=tmp_path, store=Store()
    )


@pytest.mark.asyncio
async def test_real_session_dispose_cancels_the_sidecar(tmp_path, monkeypatch) -> None:
    """The fork rides the session's own tracking, so dispose cancels it."""
    cancelled = asyncio.Event()

    async def fake(session, audio, *, store, timeout_s, client=None):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    monkeypatch.setattr(sidecar, "_model_transcript", fake)
    session = _real_session(tmp_path)
    try:
        sidecar.fork_audio_sidecar(
            session, message_id="m8", audio=Block(), config_dir=tmp_path, store=Store()
        )
        await asyncio.sleep(0.05)  # let the task start
        assert session._background_tasks, "the fork must be tracked on the session"
        await session.dispose()
    finally:
        await session.dispose()

    assert cancelled.is_set()
    # Cancellation is not an outcome: no record is written for a disposed run.
    assert session.transcript.latest_custom(sidecar.STT_TRANSCRIPT_CUSTOM_TYPE) is None


def _real_session(tmp_path) -> Session:
    def stream_fn(request, signal):
        async def gen():
            if False:  # pragma: no cover - never yields; makes gen an async iterator
                yield None

        return gen()

    return Session(
        model=ModelSpec(provider="test", model_id="m", context_window=100_000),
        stream_fn=stream_fn,
        tools=[],
        transcript=Transcript(tmp_path / "real-sess"),
        system_blocks_provider=lambda: [],
    )


# -- _model_transcript: the one place the call shape lives -------------------


@pytest.mark.asyncio
async def test_no_model_means_unavailable(tmp_path) -> None:
    host = Host(Transcript(tmp_path / "sess"), model=SimpleNamespace())
    with pytest.raises(sidecar.ModelTranscriptUnavailable) as caught:
        await sidecar._model_transcript(
            host, Block(), store=Store({"openrouter": "k"}), timeout_s=1.0
        )
    assert "no selected model" in str(caught.value)


@pytest.mark.asyncio
async def test_unknown_provider_means_unavailable(tmp_path) -> None:
    host = Host(
        Transcript(tmp_path / "sess"),
        model=SimpleNamespace(provider="nope", model_id="x"),
    )
    with pytest.raises(sidecar.ModelTranscriptUnavailable):
        await sidecar._model_transcript(host, Block(), store=Store({"nope": "k"}), timeout_s=1.0)


@pytest.mark.asyncio
async def test_a_non_chat_wire_means_unavailable(tmp_path) -> None:
    host = Host(
        Transcript(tmp_path / "sess"),
        model=SimpleNamespace(provider="anthropic", model_id="claude-x"),
    )
    with pytest.raises(sidecar.ModelTranscriptUnavailable) as caught:
        await sidecar._model_transcript(
            host, Block(), store=Store({"anthropic": "k"}), timeout_s=1.0
        )
    assert "wire" in str(caught.value)


@pytest.mark.asyncio
async def test_a_wire_refused_format_means_unavailable(tmp_path) -> None:
    host = Host(Transcript(tmp_path / "sess"))
    with pytest.raises(sidecar.ModelTranscriptUnavailable) as caught:
        await sidecar._model_transcript(
            host, Block(b"OggSdata", "audio/ogg"), store=Store({"openrouter": "k"}), timeout_s=1.0
        )
    assert "wav or mp3" in str(caught.value)


@pytest.mark.asyncio
async def test_no_stored_credential_means_unavailable(tmp_path) -> None:
    host = Host(Transcript(tmp_path / "sess"))
    with pytest.raises(sidecar.ModelTranscriptUnavailable) as caught:
        await sidecar._model_transcript(host, Block(), store=Store(), timeout_s=1.0)
    assert "credential" in str(caught.value)


@pytest.mark.asyncio
async def test_the_model_call_shape(tmp_path) -> None:
    """The POC-proven shape: chat completions, input_audio part, instruction."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["payload"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": " hi there "}}]})

    host = Host(Transcript(tmp_path / "sess"))
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        text = await sidecar._model_transcript(
            host,
            Block(),
            store=Store({"openrouter": "or-secret"}),
            timeout_s=5.0,
            client=http,
        )

    assert text == "hi there"
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["authorization"] == "Bearer or-secret"
    payload = json.loads(seen["payload"])
    assert payload["model"] == "openrouter/some-audio-model"
    content = payload["messages"][0]["content"]
    assert content[0]["text"] == sidecar.TRANSCRIBE_INSTRUCTION
    assert content[1]["input_audio"]["format"] == "wav"
    assert content[1]["input_audio"]["data"] == base64.b64encode(_WAV_BYTES).decode()


@pytest.mark.asyncio
async def test_the_radient_host_comes_from_its_resolver(tmp_path, monkeypatch) -> None:
    from local_operator import env

    monkeypatch.setattr(
        env, "resolve_radient_api_base_url", lambda: "https://staging.radient.example/v1"
    )
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    host = Host(
        Transcript(tmp_path / "sess"),
        model=SimpleNamespace(provider="radient", model_id="model-x"),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await sidecar._model_transcript(
            host,
            Block(),
            store=Store({"radient": "r-key"}),
            timeout_s=5.0,
            client=http,
        )
    assert seen["url"] == "https://staging.radient.example/v1/chat/completions"


@pytest.mark.asyncio
async def test_a_refused_model_call_is_an_apierror(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "model not permitted"})

    host = Host(Transcript(tmp_path / "sess"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(APIError) as caught:
            await sidecar._model_transcript(
                host, Block(), store=Store({"openrouter": "k"}), timeout_s=5.0, client=http
            )
    assert caught.value.status_code == 403


@pytest.mark.asyncio
async def test_an_empty_completion_is_refused(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    host = Host(Transcript(tmp_path / "sess"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(APIError) as caught:
            await sidecar._model_transcript(
                host, Block(), store=Store({"openrouter": "k"}), timeout_s=5.0, client=http
            )
    assert "no transcript text" in str(caught.value)


@pytest.mark.asyncio
async def test_the_attempt_bound_is_enforced(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})

    host = Host(Transcript(tmp_path / "sess"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(asyncio.TimeoutError) as caught:
            await sidecar._model_transcript(
                host, Block(), store=Store({"openrouter": "k"}), timeout_s=0.05, client=http
            )
    assert "did not return a transcript within" in str(caught.value)
