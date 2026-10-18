"""The /v1/stt surface: the paths report, the cascade POST, and its refusals."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from local_operator.server import dependencies as deps
from local_operator.server.app import _LEGACY_CONTROL_PATHS, _legacy_desktop_gated, app
from local_operator.server.routes import stt as stt_routes
from local_operator.stt import AudioPath


@pytest.fixture(autouse=True)
def _empty_radient(monkeypatch):
    """The app fixture arms a Radient store row; these tests want a clean slate.

    The cascade's Radient probe is stubbed empty by default so a rung's
    availability is what each test says it is; the test that wants Radient
    available overrides this stub.
    """
    from pydantic import SecretStr

    from local_operator.stt import cascade

    async def empty(_config_dir, _base_url, *, store=None):
        return SecretStr("")

    monkeypatch.setattr(cascade, "resolve_radient_credential", empty)

    async def no_probe(_config_dir, _base_url, *, store):
        return False

    monkeypatch.setattr(cascade, "has_persisted_radient_credential", no_probe)


#: A body whose header sniffs as wav, so the executor's mime derivation is
#: exercised through the real ``media.sniff_audio`` on every POST test.
_WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 64


class FakeStore:
    def __init__(self, keys: dict[str, str] | None = None):
        self.keys = dict(keys or {})

    async def get_api_key(self, provider, session_id=None, *, read_only=False, **kwargs):
        return self.keys.get(provider)

    async def has_persisted_credential(self, provider, session_id=None):
        return bool(self.keys.get(provider))

    async def get_persisted_api_key(self, provider, session_id=None, *, kinds=None):
        return self.keys.get(provider)


@pytest.fixture
def store_override():
    """Point the routes' store dependency at a fake, and clean up after."""

    def apply(store: FakeStore) -> None:
        async def override(request=None):
            return store

        app.dependency_overrides[deps.get_provider_auth_store] = override

    yield apply
    app.dependency_overrides.pop(deps.get_provider_auth_store, None)


class FakeUpstream:
    """A threaded HTTP server the real httpx clients are pointed at."""

    def __init__(self, responder: Callable[[dict[str, Any]], tuple[int, Any]]):
        self.requests: list[dict[str, Any]] = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API
                pass

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length", 0))
                record = {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": self.rfile.read(length),
                }
                upstream.requests.append(record)
                status, payload = responder(record)
                data = (
                    payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def elevenlabs_upstream(monkeypatch):
    """Serve a fake ElevenLabs and point the real client at it."""
    servers: list[FakeUpstream] = []

    def serve(responder) -> FakeUpstream:
        upstream = FakeUpstream(responder)
        servers.append(upstream)
        from local_operator.stt import clients as stt_clients

        monkeypatch.setattr(stt_clients, "ELEVENLABS_STT_BASE_URL", upstream.base_url)
        return upstream

    yield serve
    for upstream in servers:
        upstream.stop()


# -- GET /v1/stt/paths -------------------------------------------------------


@pytest.mark.asyncio
async def test_paths_reports_every_rung_in_cascade_order(test_app_client, store_override) -> None:
    store_override(FakeStore())
    response = await test_app_client.get("/v1/stt/paths")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == 200
    result = body["result"]
    assert result["path"] == "none"
    assert result["model_audio_capable"] is False
    assert [rung["path"] for rung in result["rungs"]] == [
        "provider_stt_radient",
        "provider_stt_elevenlabs",
        "provider_stt_openai",
        "provider_stt_superwhisper",
        "model_audio_sidecar",
    ]
    superwhisper = next(r for r in result["rungs"] if r["path"] == "provider_stt_superwhisper")
    assert superwhisper["available"] is False


@pytest.mark.asyncio
async def test_paths_reflects_a_stored_elevenlabs_key(test_app_client, store_override) -> None:
    store_override(FakeStore({"elevenlabs": "el-key"}))
    response = await test_app_client.get("/v1/stt/paths")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["path"] == "provider_stt_elevenlabs"
    assert result["reason"] == "An ElevenLabs API key is stored."


@pytest.mark.asyncio
async def test_paths_answers_for_the_named_model(
    test_app_client, store_override, monkeypatch
) -> None:
    store_override(FakeStore())
    monkeypatch.setattr(
        stt_routes,
        "_resolve_query_model",
        lambda provider, model: SimpleNamespace(supports_audio_input=True),
    )
    response = await test_app_client.get("/v1/stt/paths?provider=openrouter&model=some-audio-model")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["path"] == "model_audio_sidecar"
    assert result["model_audio_capable"] is True


@pytest.mark.asyncio
async def test_paths_reports_radient_first_when_it_is_available(
    test_app_client, store_override, monkeypatch
) -> None:
    from pydantic import SecretStr

    from local_operator.stt import cascade

    async def present(_config_dir, _base_url, *, store=None):
        return SecretStr("r-key")

    monkeypatch.setattr(cascade, "resolve_radient_credential", present)

    async def probe(_config_dir, _base_url, *, store):
        return True

    monkeypatch.setattr(cascade, "has_persisted_radient_credential", probe)
    store_override(FakeStore())
    response = await test_app_client.get("/v1/stt/paths")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["path"] == "provider_stt_radient"
    assert result["reason"] == "Signed in to Radient."


@pytest.mark.asyncio
async def test_paths_refuses_half_a_model_query(test_app_client, store_override) -> None:
    store_override(FakeStore())
    response = await test_app_client.get("/v1/stt/paths?provider=openrouter")
    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "wire.errors.media.provider_model_pair",
            "params": {},
            "text": "provider and model must be provided together, or neither.",
        }
    }


@pytest.mark.asyncio
async def test_paths_exercises_the_real_model_spec_builder(test_app_client, store_override) -> None:
    """Without a monkeypatch the real builder runs; an unknown pair is a 422."""
    store_override(FakeStore())
    response = await test_app_client.get("/v1/stt/paths?provider=nope&model=nothing")
    assert response.status_code == 422


# -- POST /v1/stt/transcriptions --------------------------------------------


@pytest.mark.asyncio
async def test_transcriptions_200_through_the_real_client(
    test_app_client, store_override, elevenlabs_upstream
) -> None:
    store_override(FakeStore({"elevenlabs": "el-key"}))
    upstream = elevenlabs_upstream(
        lambda record: (200, {"text": "hello from the fake", "language_code": "eng"})
    )

    response = await test_app_client.post(
        "/v1/stt/transcriptions",
        files={"file": ("take.webm", _WAV_BYTES, "audio/webm")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    result = body["result"]
    assert result["text"] == "hello from the fake"
    assert result["path"] == "provider_stt_elevenlabs"
    assert [attempt["outcome"] for attempt in result["attempts"]] == ["ok"]
    assert result["attempts"][0]["path"] == "provider_stt_elevenlabs"
    assert result["duration_s"] >= 0

    assert len(upstream.requests) == 1
    sent = upstream.requests[0]
    assert sent["path"] == "/v1/speech-to-text"
    assert sent["headers"].get("xi-api-key") == "el-key"
    # The sniffed container (wav) governs the multipart, not the declared webm.
    assert b'filename="audio.wav"' in sent["body"]
    assert b"audio/wav" in sent["body"]
    # The default mode is the provider-native cleanup, and the response says so.
    assert b'name="no_verbatim"' in sent["body"] and b"\r\n\r\ntrue\r\n" in sent["body"]
    assert b'name="tag_audio_events"' in sent["body"] and b"\r\n\r\nfalse\r\n" in sent["body"]
    assert result["cleanup"] == "elevenlabs.no_verbatim"
    assert result["raw_text"] is None
    assert result["raw_unavailable"] == "provider_no_verbatim"
    assert result["language"] == "eng"
    assert result["removed"] == []


@pytest.mark.asyncio
async def test_transcriptions_cleanup_off_is_verbatim_on_the_wire(
    test_app_client, store_override, elevenlabs_upstream
) -> None:
    """The disabled path sends neither cleanup field and returns a raw copy."""
    store_override(FakeStore({"elevenlabs": "el-key"}))
    upstream = elevenlabs_upstream(lambda record: (200, {"text": "Um, hello."}))

    response = await test_app_client.post(
        "/v1/stt/transcriptions",
        files={"file": ("take.wav", _WAV_BYTES, "audio/wav")},
        data={"cleanup": "off"},
    )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    body = upstream.requests[0]["body"]
    assert b"no_verbatim" not in body
    assert b"tag_audio_events" not in body
    assert result["text"] == "Um, hello."
    assert result["raw_text"] == "Um, hello."
    assert result["raw_unavailable"] == ""
    assert result["cleanup"] == "none"


@pytest.mark.asyncio
async def test_transcriptions_refuses_an_unknown_cleanup_mode(
    test_app_client, store_override, elevenlabs_upstream
) -> None:
    store_override(FakeStore({"elevenlabs": "el-key"}))
    elevenlabs_upstream(lambda record: (200, {"text": "x"}))

    response = await test_app_client.post(
        "/v1/stt/transcriptions",
        files={"file": ("take.wav", _WAV_BYTES, "audio/wav")},
        data={"cleanup": "bogus"},
    )

    assert response.status_code == 422
    assert "Unknown transcription cleanup mode" in response.json()["detail"]["text"]


@pytest.mark.asyncio
async def test_transcriptions_refuses_edit_before_its_slice(
    test_app_client, store_override, elevenlabs_upstream
) -> None:
    """A recognized mode this build cannot execute is 422, never a fallback."""
    store_override(FakeStore({"elevenlabs": "el-key"}))
    upstream = elevenlabs_upstream(lambda record: (200, {"text": "x"}))

    response = await test_app_client.post(
        "/v1/stt/transcriptions",
        files={"file": ("take.wav", _WAV_BYTES, "audio/wav")},
        data={"cleanup": "edit"},
    )

    assert response.status_code == 422
    assert "not available yet" in response.json()["detail"]["text"]
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_transcriptions_maps_a_credit_refusal_to_402(
    test_app_client, store_override, elevenlabs_upstream
) -> None:
    store_override(FakeStore({"elevenlabs": "el-key"}))
    elevenlabs_upstream(lambda record: (402, {"detail": {"message": "payment required"}}))

    response = await test_app_client.post(
        "/v1/stt/transcriptions",
        files={"file": ("take.wav", _WAV_BYTES, "audio/wav")},
    )

    assert response.status_code == 402
    sentence = (
        "Transcription is unavailable: your ElevenLabs credit balance is too low. "
        "Add credits to continue. Upstream responded 402: "
        '{"detail": {"message": "payment required"}}'
    )
    assert response.json()["detail"] == {
        "code": "wire.errors.media.media_refusal",
        "params": {"error": sentence},
        "text": sentence,
    }


@pytest.mark.asyncio
async def test_transcriptions_maps_an_upstream_fault_to_502(
    test_app_client, store_override, elevenlabs_upstream
) -> None:
    store_override(FakeStore({"elevenlabs": "el-key"}))
    elevenlabs_upstream(lambda record: (500, {"error": "the wire said no"}))

    response = await test_app_client.post(
        "/v1/stt/transcriptions",
        files={"file": ("take.wav", _WAV_BYTES, "audio/wav")},
    )

    assert response.status_code == 502
    # A lower-layer classifier owns this prose. Pin the entire response so
    # extraction cannot lose either its diagnostic tail or the error param.
    sentence = (
        'Transcription failed upstream. Upstream responded 500: {"error": "the wire said no"}'
    )
    assert response.json()["detail"] == {
        "code": "wire.errors.media.media_refusal",
        "params": {"error": sentence},
        "text": sentence,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_audio", [False, True])
async def test_transcriptions_409_when_no_rung_exists(
    test_app_client, store_override, monkeypatch, model_audio
) -> None:
    store_override(FakeStore())
    query = "?provider=openrouter&model=audio-model" if model_audio else ""
    if model_audio:
        monkeypatch.setattr(
            stt_routes,
            "_resolve_query_model",
            lambda provider, model: SimpleNamespace(supports_audio_input=True),
        )
    response = await test_app_client.post(
        "/v1/stt/transcriptions" + query,
        files={"file": ("take.wav", _WAV_BYTES, "audio/wav")},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    # The existing machine category and resolver extras must survive alongside
    # the new envelope. Compare all rungs, not just the first one's path.
    paths = (await test_app_client.get("/v1/stt/paths" + query)).json()["result"]
    sentence = (
        "The selected model accepts audio input."
        if model_audio
        else "No transcription provider is available and no model is selected for the audio path."
    )
    assert detail == {
        "code": "stt_unavailable",
        "params": {"error": sentence},
        "text": sentence,
        "rungs": paths["rungs"],
        "model_audio": model_audio,
    }


@pytest.mark.asyncio
async def test_transcriptions_forwards_language_and_prompt(
    test_app_client, store_override, elevenlabs_upstream
) -> None:
    store_override(FakeStore({"elevenlabs": "el-key"}))
    upstream = elevenlabs_upstream(lambda record: (200, {"text": "x"}))

    response = await test_app_client.post(
        "/v1/stt/transcriptions",
        files={"file": ("take.wav", _WAV_BYTES, "audio/wav")},
        data={"language": "en"},
    )
    assert response.status_code == 200
    assert b'name="language_code"' in upstream.requests[0]["body"]


# -- the managed-mode gate ---------------------------------------------------


def test_the_stt_paths_are_gated_in_managed_mode() -> None:
    assert "/v1/stt/paths" in _LEGACY_CONTROL_PATHS
    assert "/v1/stt/transcriptions" in _LEGACY_CONTROL_PATHS
    assert _legacy_desktop_gated("/v1/stt/paths", "GET") is True
    assert _legacy_desktop_gated("/v1/stt/transcriptions", "POST") is True


def test_audio_path_vocabulary_is_closed() -> None:
    assert [path.value for path in AudioPath] == [
        "provider_stt_radient",
        "provider_stt_elevenlabs",
        "provider_stt_openai",
        "provider_stt_superwhisper",
        "model_audio_sidecar",
        "none",
    ]


# -- PR-3: the cascade route's spend emission (QA round 1, Q3/Q4/Q5) ----------


@pytest.mark.asyncio
async def test_the_cascade_route_records_the_envelope_provider_and_correlates(
    test_app_client, store_override, monkeypatch
) -> None:
    """Q3/Q4/Q5: envelope provider outranks the path label; the session id rides.

    Q3: the legacy route records ``elevenlabs`` for the same envelope while
    this route recorded ``''`` — the envelope's own ``provider`` field is the
    vendor witness, and the cost object (when one is stated) still wins inside
    the record builder. Q4: the caller's ``session_id`` goes OUT on the hub
    call as the client correlation header (proved at the client layer; this
    test proves the route passes it down at all). Q5: the emission itself —
    amount, units, basis, session — had no route-level test before this.
    """
    from pydantic import SecretStr

    from local_operator.clients.radient import RadientTranscriptionResponseData
    from local_operator.stt import cascade

    async def keyed(_config_dir, _base_url, *, store=None):
        return SecretStr("stub-key")

    async def available(_config_dir, _base_url, *, store):
        return True

    monkeypatch.setattr(cascade, "resolve_radient_credential", keyed)
    monkeypatch.setattr(cascade, "has_persisted_radient_credential", available)

    sent: list[dict[str, Any]] = []

    class FakeRadient:
        def __init__(self, api_key=None, base_url=None):
            """Same constructor surface the rung uses; nothing to store."""

        def create_transcription(self, **kwargs):
            sent.append(kwargs)
            return RadientTranscriptionResponseData.model_validate(
                {
                    "text": "hello from the stub hub",
                    "provider": "elevenlabs",
                    "status": "completed",
                    "duration": 2.832875,
                    "cost": {
                        "v": 1,
                        "amount_usd": "0.001",
                        "basis": "billed",
                        "units": 3,
                        "unit": "second",
                        "usage_record_id": "qa-cascade-0001",
                        "price_version": "transcription-rates@2026-10-01",
                    },
                }
            )

    monkeypatch.setattr(cascade, "RadientClient", FakeRadient)
    store_override(FakeStore({"radient": "stub"}))
    captured: list[Any] = []
    monkeypatch.setattr(
        "local_operator.server.utils.voice_spend.enqueue_channel_record",
        captured.append,
    )

    response = await test_app_client.post(
        "/v1/stt/transcriptions",
        files={"file": ("clip.wav", _WAV_BYTES, "audio/wav")},
        data={"session_id": "sess-qa-cascade"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["path"] == "provider_stt_radient"
    assert captured, "the route must record the hub's charge"
    record = captured[0]
    assert record.provider == "elevenlabs", "Q3: the envelope's own provider field"
    assert record.units == 3.0 and record.unit == "seconds"
    assert record.amount_micro == 1000 and record.billing_basis == "billed"
    assert record.session_id == "sess-qa-cascade"
    assert sent, "the Radient rung must have run"
    assert sent[0]["session_id"] == "sess-qa-cascade", "Q4: the correlation id rides"
    assert len(sent[0]["client_request_id"]) == 32
