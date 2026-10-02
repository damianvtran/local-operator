"""The TTS surface: the paths report, the managed-mode gate, and the BYO legs.

The BYO half is exercised through the REAL clients against a transport fake, so
what is asserted is the request the vendor would actually receive — status,
headers and JSON body — rather than a mock's idea of it. That is the only
evidence that the vendored map's params reach a wire in the right shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.server import dependencies as deps
from local_operator.server.app import _LEGACY_CONTROL_PATHS, _legacy_desktop_gated, app
from local_operator.tts import VoicePath, cascade
from local_operator.tts import clients as tts_clients
from local_operator.tts.adapters import Legacy
from local_operator.tts.descriptor import VoiceDescriptor

CANONICAL = "https://api.radienthq.com/v1"


@pytest.fixture(autouse=True)
def _empty_radient(monkeypatch):
    """The app fixture arms a Radient store row; these tests want a clean slate."""

    async def no_probe(_config_dir, _base_url, *, store):
        return False

    monkeypatch.setattr(cascade, "has_persisted_radient_credential", no_probe)


class FakeStore:
    """The AuthStore seam: persisted rows only, which is what the probes read."""

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
    def apply(store: FakeStore) -> None:
        async def override(request=None):
            return store

        app.dependency_overrides[deps.get_provider_auth_store] = override

    yield apply
    app.dependency_overrides.pop(deps.get_provider_auth_store, None)


# -- GET /v1/tts/paths -------------------------------------------------------


@pytest.mark.asyncio
async def test_paths_reports_every_rung_in_cascade_order(test_app_client, store_override) -> None:
    store_override(FakeStore())
    response = await test_app_client.get("/v1/tts/paths")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == 200
    result = body["result"]
    assert result["path"] == "none"
    assert result["servable"] is False
    assert [rung["path"] for rung in result["rungs"]] == [
        "provider_tts_radient",
        "provider_tts_elevenlabs",
        "provider_tts_openai",
    ]


@pytest.mark.asyncio
async def test_paths_reflects_a_stored_elevenlabs_key(test_app_client, store_override) -> None:
    store_override(FakeStore({"elevenlabs": "el-key"}))
    response = await test_app_client.get("/v1/tts/paths")
    result = response.json()["result"]
    assert result["path"] == "provider_tts_elevenlabs"
    assert result["reason"] == "An ElevenLabs API key is stored."
    assert result["servable"] is True


@pytest.mark.asyncio
async def test_paths_reflects_a_stored_openai_key(test_app_client, store_override) -> None:
    store_override(FakeStore({"openai-key": "oa-key"}))
    result = (await test_app_client.get("/v1/tts/paths")).json()["result"]
    assert result["path"] == "provider_tts_openai"


def test_the_tts_paths_are_gated_in_managed_mode() -> None:
    assert "/v1/tts/paths" in _LEGACY_CONTROL_PATHS
    assert _legacy_desktop_gated("/v1/tts/paths", "GET") is True


def test_voice_path_vocabulary_is_closed() -> None:
    assert [path.value for path in VoicePath] == [
        "provider_tts_radient",
        "provider_tts_elevenlabs",
        "provider_tts_openai",
        "none",
    ]


# -- the BYO legs, over a real transport -------------------------------------


class Recorder:
    """One vendor endpoint over ``httpx.MockTransport``, recording the request."""

    def __init__(self, status: int = 200, body: bytes = b"AUDIO") -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = status
        self.body = body

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query": dict(request.url.params),
                "headers": dict(request.headers),
                "json": json.loads(request.content) if request.content else None,
            }
        )
        return httpx.Response(
            self.status, content=self.body, headers={"content-type": "audio/mpeg"}
        )


def _descriptor() -> VoiceDescriptor:
    """The follower case's descriptor: male, calm, with the caller's own text."""
    return VoiceDescriptor(
        gender="male",
        tone="calm",
        expressiveness="medium",
        pace=1.0,
        language="auto",
        accent="",
        instructions="Speak the text in the text's own language.",
    )


async def _run(store: FakeStore, **kwargs) -> Any:
    """Resolve, then synthesize over the caller's injected clients."""
    resolution = await cascade.resolve_voice_path(
        config_dir=Path("/nonexistent-config"), base_url=CANONICAL, store=cast(AuthStore, store)
    )
    return await cascade.synthesize_speech(
        "Hello there.",
        config_dir=Path("/nonexistent-config"),
        base_url=CANONICAL,
        store=cast(AuthStore, store),
        descriptor=_descriptor(),
        legacy=Legacy(),
        resolution=resolution,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_the_openai_leg_speaks_the_descriptor_it_was_given() -> None:
    """The follower case: Radient unavailable, the user's own OpenAI key serves.

    A male, calm descriptor with the caller's own instructions must reach the
    vendor as ``onyx`` plus the calm SENTENCE appended to those instructions —
    that is the emulation the map promises, and the reason the two repositories
    share one vector file.
    """
    recorder = Recorder()
    store = FakeStore({"openai-key": "sk-user"})
    transport = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler))
    from local_operator.tts.clients import OpenAiTtsClient

    outcome = await _run(store, openai_client=OpenAiTtsClient("sk-user", client=transport))
    await transport.aclose()

    assert outcome.path is VoicePath.PROVIDER_TTS_OPENAI
    assert outcome.provider == "openai"
    assert outcome.audio == b"AUDIO"
    sent = recorder.requests[0]
    assert sent["path"] == "/v1/audio/speech"
    assert sent["headers"]["authorization"] == "Bearer sk-user"
    assert sent["json"]["voice"] == "onyx"
    assert sent["json"]["model"] == tts_clients.OPENAI_TTS_MODEL
    assert sent["json"]["input"] == "Hello there."
    assert sent["json"]["instructions"] == (
        "Speak the text in the text's own language. Speak in a calm, unhurried tone."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [503, 402])
async def test_the_hub_leg_falls_forward_to_the_users_own_key(status, monkeypatch) -> None:
    """A hub refusal is a step, not the end -- including a 402.

    A 402 from the hub means the user's CREDITS are out, not that the platform
    is broken, so spending their own stored key is the right economics. The 503
    variant is the same walk for a temporarily unavailable provider.
    """
    from pydantic import SecretStr

    from local_operator.clients._http import APIError

    async def credential(_config_dir, _base_url, *, store=None):
        return SecretStr("radient-key")

    async def probe(_config_dir, _base_url, *, store):
        return True

    monkeypatch.setattr(cascade, "resolve_radient_credential", credential)
    monkeypatch.setattr(cascade, "has_persisted_radient_credential", probe)

    hub = MagicMock()
    hub.create_speech_response.side_effect = APIError("hub refused", status_code=status)

    recorder = Recorder(body=b"FALLBACK")
    store = FakeStore({"radient": "radient-key", "openai-key": "sk-user"})
    oa = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler))
    from local_operator.tts.clients import OpenAiTtsClient

    try:
        outcome = await _run(
            store, radient_client=hub, openai_client=OpenAiTtsClient("sk-user", client=oa)
        )
    finally:
        await oa.aclose()

    assert outcome.path is VoicePath.PROVIDER_TTS_OPENAI
    assert outcome.provider == "openai"
    assert [attempt.outcome for attempt in outcome.attempts] == ["failed", "ok"]
    # The daemon mapped this leg, so it EMITS the voicing headers itself — same
    # names and same grammar a hub-served call relays.
    headers = dict(outcome.speech_headers)
    assert headers["X-Radient-Speech-Path"] == "provider_tts_openai"
    assert headers["X-Radient-Speech-Map"] == "1.0"
    assert "gender" in headers["X-Radient-Speech-Applied"]
    assert "tone:emulated=instructions" in headers["X-Radient-Speech-Degraded"]


@pytest.mark.asyncio
async def test_the_elevenlabs_leg_ignores_instructions_and_maps_the_settings() -> None:
    """The same descriptor on the primary leg: the pool voice, no instructions field."""
    recorder = Recorder()
    store = FakeStore({"elevenlabs": "el-user"})
    transport = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler))
    from local_operator.tts.clients import ElevenLabsTtsClient

    outcome = await _run(store, elevenlabs_client=ElevenLabsTtsClient("el-user", client=transport))
    await transport.aclose()

    assert outcome.path is VoicePath.PROVIDER_TTS_ELEVENLABS
    sent = recorder.requests[0]
    # The voice id is a PATH segment and the format a query value.
    assert sent["path"] == "/v1/text-to-speech/8dEUmyPMdDdK91vboYih"
    assert sent["query"] == {"output_format": tts_clients.OUTPUT_FORMAT}
    assert sent["headers"]["xi-api-key"] == "el-user"
    assert sent["json"]["voice_settings"]["stability"] == 0.5
    assert sent["json"]["voice_settings"]["similarity_boost"] == 0.75
    # ElevenLabs has no instructions field, and this integration never folds
    # any hint into the spoken text.
    assert "instructions" not in sent["json"]


@pytest.mark.asyncio
async def test_the_openai_leg_does_not_claim_the_descriptor_without_a_key() -> None:
    """A rung with no persisted key is never attempted (it is not advertised)."""
    store = FakeStore({"elevenlabs": "el-user"})
    resolution = await cascade.resolve_voice_path(
        config_dir=Path("/nonexistent-config"), base_url=CANONICAL, store=cast(AuthStore, store)
    )
    assert resolution.path is VoicePath.PROVIDER_TTS_ELEVENLABS
    assert not any(
        rung.path is VoicePath.PROVIDER_TTS_OPENAI and rung.available for rung in resolution.rungs
    )


@pytest.mark.asyncio
async def test_a_vendor_refusal_falls_forward_to_the_next_rung() -> None:
    """A failed leg is a step, not the end: the walk reaches the one that works."""
    failing = Recorder(status=503)
    working = Recorder(body=b"FALLBACK")
    store = FakeStore({"elevenlabs": "el-user", "openai-key": "sk-user"})

    el = httpx.AsyncClient(transport=httpx.MockTransport(failing.handler))
    oa = httpx.AsyncClient(transport=httpx.MockTransport(working.handler))
    from local_operator.tts.clients import ElevenLabsTtsClient, OpenAiTtsClient

    try:
        outcome = await _run(
            store,
            elevenlabs_client=ElevenLabsTtsClient("el-user", client=el),
            openai_client=OpenAiTtsClient("sk-user", client=oa),
        )
    finally:
        await el.aclose()
        await oa.aclose()

    assert outcome.path is VoicePath.PROVIDER_TTS_OPENAI
    assert outcome.audio == b"FALLBACK"
    assert [attempt.outcome for attempt in outcome.attempts] == ["failed", "ok"]
    assert failing.requests and working.requests


@pytest.mark.asyncio
async def test_the_path_header_names_the_daemon_rung_not_the_hubs_leg(monkeypatch) -> None:
    """M1: Path is the RUNG; the hub's own Provider header is the leg.

    Filling Path from the hub's Provider made "the hub used the platform's
    ElevenLabs" and "my own ElevenLabs key ran" the same string. The two facts
    now travel separately, and Path stays inside the closed ``VoicePath``
    vocabulary even when the hub names something unexpected (O2).
    """
    from pydantic import SecretStr

    async def credential(_config_dir, _base_url, *, store=None):
        return SecretStr("radient-key")

    async def probe(_config_dir, _base_url, *, store):
        return True

    monkeypatch.setattr(cascade, "resolve_radient_credential", credential)
    monkeypatch.setattr(cascade, "has_persisted_radient_credential", probe)

    hub = MagicMock()
    hub.create_speech_response.return_value = (
        b"AUDIO",
        {
            **{name: "v" for name in cascade.HUB_SPEECH_HEADERS},
            "X-Radient-Speech-Provider": "elevenlabs",
            # A header the hub does NOT send today, inside the family namespace:
            # the relay is a fixed set, so it must not cross (security S-3).
            "X-Radient-Speech-Tenant": "tenant-42",
            # ...and neither does anything outside the family.
            "X-Internal-Trace": "secret",
            "Set-Cookie": "session=abc",
        },
    )
    store = FakeStore({"radient": "radient-key"})
    outcome = await _run(store, radient_client=hub)

    headers = dict(outcome.speech_headers)
    assert headers["X-Radient-Speech-Path"] == "provider_tts_radient"
    assert headers["X-Radient-Speech-Provider"] == "elevenlabs"
    assert "X-Radient-Speech-Tenant" not in headers
    assert "X-Internal-Trace" not in headers
    assert "Set-Cookie" not in headers
    # Only the named hub headers plus the daemon's own Path.
    assert len(headers) == len(cascade.HUB_SPEECH_HEADERS) + 1


@pytest.mark.asyncio
async def test_a_failed_rung_travels_with_the_refusal(monkeypatch) -> None:
    """The route's vocabulary selector: WHICH rung refused is part of the report."""
    from pydantic import SecretStr

    from local_operator.clients._http import APIError

    async def credential(_config_dir, _base_url, *, store=None):
        return SecretStr("radient-key")

    async def probe(_config_dir, _base_url, *, store):
        return True

    monkeypatch.setattr(cascade, "resolve_radient_credential", credential)
    monkeypatch.setattr(cascade, "has_persisted_radient_credential", probe)

    # A BYO-only failure reports the BYO rung...
    failing = Recorder(status=401)
    el = httpx.AsyncClient(transport=httpx.MockTransport(failing.handler))
    from local_operator.tts.clients import ElevenLabsTtsClient

    try:
        with pytest.raises(cascade.TtsUnavailable) as exc_info:
            await _run(
                FakeStore({"elevenlabs": "el-user"}),
                elevenlabs_client=ElevenLabsTtsClient("el-user", client=el),
            )
    finally:
        await el.aclose()
    assert exc_info.value.failed_path is VoicePath.PROVIDER_TTS_ELEVENLABS

    # ...and a hub failure reports the hub rung.
    hub = MagicMock()
    hub.create_speech_response.side_effect = APIError("no credit", status_code=402)
    with pytest.raises(cascade.TtsUnavailable) as hub_exc:
        await _run(FakeStore({"radient": "radient-key"}), radient_client=hub)
    assert hub_exc.value.failed_path is VoicePath.PROVIDER_TTS_RADIENT


@pytest.mark.asyncio
async def test_no_rung_at_all_raises_without_attempting_anything() -> None:
    """The refusal the route turns into the sign-in sentence, and no vendor call."""
    store = FakeStore()
    with pytest.raises(cascade.TtsUnavailable) as exc_info:
        await _run(store)
    assert exc_info.value.attempts == ()
    assert exc_info.value.error is None


def test_the_cascade_is_not_a_second_source_of_the_rung_order() -> None:
    """The order lives in one constant, and the resolver reads it."""
    assert [path.value for path in cascade.TTS_RUNG_PATHS] == [
        "provider_tts_radient",
        "provider_tts_elevenlabs",
        "provider_tts_openai",
    ]
