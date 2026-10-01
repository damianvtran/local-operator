"""Executor fall-forward, budgets, and the all-failed report."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, cast

import pytest
from pydantic import SecretStr

from local_operator.clients._http import APIError
from local_operator.clients.radient import RadientTranscriptionResponseData
from local_operator.providers.auth_store import AuthStore
from local_operator.stt import AudioPath, SttAttempt, cascade

#: A file whose header verifies as wav AND whose suffix says wav, so the
#: sniff-first mime derivation and the suffix fallback agree by default.
_WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 32


class FakeStore:
    def __init__(self, keys: Optional[dict[str, str]] = None):
        self.keys = dict(keys or {})
        self.raises = False
        self.probe_calls: list[tuple[str, Optional[str]]] = []

    async def get_api_key(self, provider, session_id=None, *, read_only=False, **kwargs):
        return self.keys.get(provider)

    async def has_persisted_credential(self, provider, session_id=None):
        """The probe seam: a fake's keys ARE its persisted rows."""
        self.probe_calls.append((provider, session_id))
        if self.raises:
            raise RuntimeError("store unavailable")
        return bool(self.keys.get(provider))


class FakeRadientClient:
    """Drop-in for ``RadientClient`` as the rung runner uses it (sync client)."""

    def __init__(self, behavior: dict[str, Any], api_key: Any = None, base_url: Any = None):
        self.behavior = behavior
        self.api_key = str(api_key) if api_key is not None else None
        self.base_url = base_url
        self.calls: list[dict[str, Any]] = []

    def create_transcription(self, **kwargs):
        self.calls.append(kwargs)
        error = self.behavior.get("error")
        if error is not None:
            raise error
        return self.behavior.get("result") or RadientTranscriptionResponseData(
            text=self.behavior.get("text", "radient text"),
            provider="elevenlabs",
            status="completed",
        )


class FakeWireClient:
    """The shared ``transcribe`` contract of the ElevenLabs/OpenAI clients."""

    def __init__(self, kind: str, behavior: dict[str, Any], api_key: Any = None):
        self.kind = kind
        self.behavior = behavior
        self.api_key = str(api_key) if api_key is not None else None
        self.calls: list[dict[str, Any]] = []

    async def transcribe(
        self, audio, *, mime, model=None, language=None, prompt=None, timeout_s=60.0
    ):
        self.calls.append(
            {
                "audio": audio,
                "mime": mime,
                "model": model,
                "language": language,
                "prompt": prompt,
                "timeout_s": timeout_s,
            }
        )
        delay = self.behavior.get("delay", 0.0)
        if delay:
            await asyncio.sleep(delay)
        error = self.behavior.get("error")
        if error is not None:
            raise error
        return type(
            "Result",
            (),
            {
                "text": self.behavior.get("text", f"{self.kind} text"),
                "model": model or "some-model",
                "provider": self.kind,
            },
        )()


class Rig:
    def __init__(self, tmp_path, monkeypatch):
        self.audio_path = tmp_path / "take.wav"
        self.audio_path.write_bytes(_WAV_BYTES)
        self.behaviors: dict[str, dict[str, Any]] = {
            "radient": {},
            "elevenlabs": {},
            "openai": {},
        }
        self.clients: list[Any] = []
        self.radient_credential = {"value": SecretStr("radient-key")}

        async def fake_resolver(_config_dir, _base_url, *, store=None):
            return self.radient_credential["value"]

        monkeypatch.setattr(cascade, "resolve_radient_credential", fake_resolver)

        async def fake_probe(_config_dir, _base_url, *, store):
            return bool(self.radient_credential["value"].get_secret_value())

        monkeypatch.setattr(cascade, "has_persisted_radient_credential", fake_probe)

        rig = self

        def radient_factory(api_key=None, base_url=None):
            client = FakeRadientClient(rig.behaviors["radient"], api_key, base_url)
            rig.clients.append(client)
            return client

        def wire_factory(kind: str):
            def make(api_key=None, **kwargs):
                client = FakeWireClient(kind, rig.behaviors[kind], api_key)
                rig.clients.append(client)
                return client

            return make

        monkeypatch.setattr(cascade, "RadientClient", radient_factory)
        monkeypatch.setattr(cascade, "ElevenLabsSttClient", wire_factory("elevenlabs"))
        monkeypatch.setattr(cascade, "OpenAiSttClient", wire_factory("openai"))

    def client(self, kind: str):
        return next(client for client in self.clients if getattr(client, "kind", "radient") == kind)

    def run(self, store_keys=None, **kwargs):
        return cascade.transcribe_audio(
            self.audio_path,
            config_dir=self.audio_path.parent,
            base_url="https://api.radienthq.com/v1",
            store=cast("AuthStore", FakeStore(store_keys)),
            **kwargs,
        )


@pytest.fixture
def rig(tmp_path, monkeypatch) -> Rig:
    return Rig(tmp_path, monkeypatch)


@pytest.mark.asyncio
async def test_radient_success_stops_the_walk(rig) -> None:
    outcome = await rig.run()  # only the Radient credential is present
    assert outcome.path == AudioPath.PROVIDER_STT_RADIENT
    assert outcome.text == "radient text"
    assert [attempt.outcome for attempt in outcome.attempts] == ["ok"]
    assert rig.clients == [rig.client("radient")]  # no BYO client was built
    kwargs = rig.client("radient").calls[0]
    assert kwargs["file_path"] == str(rig.audio_path)
    assert kwargs["model"] is None and kwargs["provider"] is None


@pytest.mark.asyncio
async def test_radient_402_falls_forward_to_elevenlabs(rig) -> None:
    rig.behaviors["radient"]["error"] = APIError("up", status_code=402, body="out of credits")
    outcome = await rig.run({"elevenlabs": "el-key"})
    assert outcome.text == "elevenlabs text"
    assert outcome.path == AudioPath.PROVIDER_STT_ELEVENLABS
    assert [attempt.path for attempt in outcome.attempts] == [
        AudioPath.PROVIDER_STT_RADIENT,
        AudioPath.PROVIDER_STT_ELEVENLABS,
    ]
    assert [attempt.outcome for attempt in outcome.attempts] == ["failed", "ok"]
    assert "credit balance" in outcome.attempts[0].detail
    wire = rig.client("elevenlabs")
    assert wire.calls[0]["audio"] == _WAV_BYTES
    assert wire.calls[0]["mime"] == "audio/wav"


@pytest.mark.asyncio
async def test_all_rungs_fail_prefers_the_payment_refusal(rig) -> None:
    rig.behaviors["radient"]["error"] = APIError("up", status_code=402, body="out of credits")
    rig.behaviors["elevenlabs"]["error"] = APIError("up", status_code=401)
    rig.behaviors["openai"]["error"] = APIError("up", status_code=500)

    with pytest.raises(cascade.SttUnavailable) as caught:
        await rig.run({"elevenlabs": "el-key", "openai": "oai-key"})

    exc = caught.value
    assert [attempt.outcome for attempt in exc.attempts] == ["failed", "failed", "failed"]
    assert exc.error is rig.behaviors["radient"]["error"]
    assert exc.status_code == 402
    assert "credit balance" in (exc.detail or "")


@pytest.mark.asyncio
async def test_all_rungs_fail_reports_the_last_failure_without_a_payment_refusal(rig) -> None:
    rig.behaviors["radient"]["error"] = APIError("up", status_code=500)
    rig.behaviors["elevenlabs"]["error"] = APIError("up", status_code=401)

    with pytest.raises(cascade.SttUnavailable) as caught:
        await rig.run({"elevenlabs": "el-key"})

    exc = caught.value
    assert exc.error is rig.behaviors["elevenlabs"]["error"]
    assert exc.status_code == 502
    assert "ElevenLabs" in (exc.detail or "")


@pytest.mark.asyncio
async def test_no_rungs_available_raises_with_no_attempts(rig) -> None:
    rig.radient_credential["value"] = SecretStr("")
    with pytest.raises(cascade.SttUnavailable) as caught:
        await rig.run()

    exc = caught.value
    assert exc.attempts == ()
    assert exc.resolution.path == AudioPath.NONE
    assert exc.status_code is None and exc.error is None
    assert "No transcription provider is available" in str(exc)


@pytest.mark.asyncio
async def test_an_attempt_timeout_falls_forward(rig, monkeypatch) -> None:
    monkeypatch.setattr(cascade, "STT_ATTEMPT_TIMEOUT_S", 0.05)
    rig.radient_credential["value"] = SecretStr("")
    rig.behaviors["elevenlabs"]["delay"] = 5.0
    rig.behaviors["openai"]["text"] = "openai beats the stall"

    outcome = await rig.run({"elevenlabs": "el-key", "openai": "oai-key"})
    assert outcome.path == AudioPath.PROVIDER_STT_OPENAI
    assert outcome.text == "openai beats the stall"
    assert outcome.attempts[0].outcome == "failed"
    assert "did not respond within" in outcome.attempts[0].detail


@pytest.mark.asyncio
async def test_a_timeout_of_the_only_rung_classifies_as_a_transport_failure(
    rig, monkeypatch
) -> None:
    monkeypatch.setattr(cascade, "STT_ATTEMPT_TIMEOUT_S", 0.05)
    rig.radient_credential["value"] = SecretStr("")
    rig.behaviors["elevenlabs"]["delay"] = 5.0

    with pytest.raises(cascade.SttUnavailable) as caught:
        await rig.run({"elevenlabs": "el-key"})

    exc = caught.value
    assert exc.status_code == 502
    assert exc.error is not None and exc.error.status_code is None
    assert "did not respond within" in (exc.detail or "")


@pytest.mark.asyncio
async def test_the_overall_budget_skips_what_it_cannot_fund(rig, monkeypatch) -> None:
    monkeypatch.setattr(cascade, "STT_ATTEMPT_TIMEOUT_S", 5.0)
    monkeypatch.setattr(cascade, "STT_OVERALL_TIMEOUT_S", 0.05)
    rig.radient_credential["value"] = SecretStr("")
    rig.behaviors["elevenlabs"]["delay"] = 5.0

    with pytest.raises(cascade.SttUnavailable) as caught:
        await rig.run({"elevenlabs": "el-key", "openai": "oai-key"})

    attempts = caught.value.attempts
    assert [attempt.outcome for attempt in attempts] == ["failed", "skipped"]
    assert attempts[1].path == AudioPath.PROVIDER_STT_OPENAI
    assert "budget was spent" in attempts[1].detail


@pytest.mark.asyncio
async def test_language_and_prompt_are_forwarded_to_the_rungs(rig) -> None:
    outcome = await rig.run(language="en", prompt="names please")
    assert outcome.path == AudioPath.PROVIDER_STT_RADIENT
    kwargs = rig.client("radient").calls[0]
    assert kwargs["language"] == "en"
    assert kwargs["prompt"] == "names please"
    assert kwargs["response_format"] == "json"
    assert kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_language_and_prompt_reach_the_byo_clients(rig) -> None:
    rig.behaviors["radient"]["error"] = APIError("up", status_code=500)
    await rig.run({"elevenlabs": "el-key"}, language="en", prompt="names please")
    call = rig.client("elevenlabs").calls[0]
    assert call["language"] == "en"
    assert call["prompt"] == "names please"


@pytest.mark.asyncio
async def test_the_sniffed_container_beats_the_suffix(rig) -> None:
    """Content over extension: the header is the stronger evidence."""
    mangled = rig.audio_path.with_suffix(".mp3")
    mangled.write_bytes(_WAV_BYTES)  # wav bytes behind an mp3 suffix
    rig.radient_credential["value"] = SecretStr("")

    outcome = await cascade.transcribe_audio(
        mangled,
        config_dir=rig.audio_path.parent,
        base_url="https://api.radienthq.com/v1",
        store=cast("AuthStore", FakeStore({"elevenlabs": "el-key"})),
    )
    assert outcome.path == AudioPath.PROVIDER_STT_ELEVENLABS
    assert rig.client("elevenlabs").calls[0]["mime"] == "audio/wav"


@pytest.mark.asyncio
async def test_an_unrecognised_header_falls_back_to_the_suffix(rig) -> None:
    mangled = rig.audio_path.with_suffix(".mp3")
    mangled.write_bytes(b"\x00\x01\x02\x03 not a container")
    rig.radient_credential["value"] = SecretStr("")

    outcome = await cascade.transcribe_audio(
        mangled,
        config_dir=rig.audio_path.parent,
        base_url="https://api.radienthq.com/v1",
        store=cast("AuthStore", FakeStore({"elevenlabs": "el-key"})),
    )
    assert outcome.path == AudioPath.PROVIDER_STT_ELEVENLABS
    assert rig.client("elevenlabs").calls[0]["mime"] == "audio/mpeg"


# ---------------------------------------------------------------------------
# The token executor (``transcribe_backend`` — the mobile bridge's dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_token_executor_runs_only_the_named_rung(rig) -> None:
    """The surface named ONE token; exactly that rung runs (no walk, no
    fall-forward). The other BYO key is stored and still stays cold, which is
    the whole point of a token-targeted executor — a call can never land on a
    path the phone did not advertise."""
    store = FakeStore({"elevenlabs": "el-key", "openai": "oa-key"})

    outcome = await cascade.transcribe_backend(
        "provider_stt_elevenlabs",
        _WAV_BYTES,
        "audio/wav",
        config_dir=rig.audio_path.parent,
        store=cast("AuthStore", store),
        language="en",
        prompt="hint",
    )

    assert outcome.text == "elevenlabs text"
    assert outcome.path == AudioPath.PROVIDER_STT_ELEVENLABS
    assert outcome.attempts == (SttAttempt(path=AudioPath.PROVIDER_STT_ELEVENLABS, outcome="ok"),)
    call = rig.client("elevenlabs").calls[0]
    assert call["audio"] == _WAV_BYTES
    assert call["mime"] == "audio/wav"
    assert call["language"] == "en"
    assert call["prompt"] == "hint"
    assert not any(getattr(client, "kind", "") == "openai" for client in rig.clients)


@pytest.mark.asyncio
async def test_the_token_executor_refuses_a_rung_without_a_credential(rig) -> None:
    """Availability is re-checked through the resolver the surface read: a
    missing key is :class:`SttUnavailable` (the bridge's 503 class) with the
    rung's own sentence and the report attached — not a provider 401."""
    with pytest.raises(cascade.SttUnavailable) as caught:
        await cascade.transcribe_backend(
            "provider_stt_elevenlabs",
            _WAV_BYTES,
            "audio/wav",
            config_dir=rig.audio_path.parent,
            store=cast("AuthStore", FakeStore({})),
        )
    assert "No ElevenLabs API key is stored" in str(caught.value)
    assert caught.value.resolution is not None
    assert rig.clients == [], "the refusal must not spend a provider call"


@pytest.mark.asyncio
async def test_the_token_executor_refuses_a_token_it_cannot_run(rig) -> None:
    """A token outside the byte-runners (the Radient leg belongs to the
    bridge's own adapter) or outside the vocabulary at all is refused with
    the report attached — never silently run as something else."""
    for backend in ("provider_stt_radient", "provider_stt_deepgram"):
        with pytest.raises(cascade.SttUnavailable) as caught:
            await cascade.transcribe_backend(
                backend,
                _WAV_BYTES,
                "audio/wav",
                config_dir=rig.audio_path.parent,
                store=cast("AuthStore", FakeStore({"elevenlabs": "el-key"})),
            )
        assert "cannot run" in str(caught.value)
        assert caught.value.resolution is not None


@pytest.mark.asyncio
async def test_the_token_executor_bounds_a_hung_rung_as_a_transport_failure(
    rig, monkeypatch
) -> None:
    """The walk's timeout convention: no upstream status was reached, so the
    failure reads as transport (the phone classifier's 502), never a 401."""
    monkeypatch.setattr(cascade, "STT_ATTEMPT_TIMEOUT_S", 0.05)
    rig.behaviors["openai"] = {"delay": 5}

    with pytest.raises(APIError) as caught:
        await cascade.transcribe_backend(
            "provider_stt_openai",
            _WAV_BYTES,
            "audio/wav",
            config_dir=rig.audio_path.parent,
            store=cast("AuthStore", FakeStore({"openai": "oa-key"})),
        )
    assert caught.value.status_code is None
    assert "did not respond within" in str(caught.value)
