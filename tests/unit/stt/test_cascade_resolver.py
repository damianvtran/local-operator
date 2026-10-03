"""Resolver permutations: every rung alone, precedence, and the reserved rung."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest
from pydantic import SecretStr

from local_operator.providers.auth_store import AuthStore
from local_operator.stt import AudioPath, cascade


class FakeStore:
    """The AuthStore seam: read-only key probes over an in-memory mapping."""

    def __init__(self, keys: Optional[dict[str, str]] = None, *, raises: bool = False):
        self.keys = dict(keys or {})
        self.raises = raises
        self.calls: list[tuple[str, Optional[str], bool]] = []
        #: The AVAILABILITY probe's calls. The resolver must ask this and never
        #: ``get_api_key`` (the call-time cascade), so ``calls`` stays empty.
        self.probe_calls: list[tuple[str, Optional[str]]] = []

    async def get_api_key(self, provider, session_id=None, *, read_only=False, **kwargs):
        self.calls.append((provider, session_id, read_only))
        if self.raises:
            raise RuntimeError("store unavailable")
        return self.keys.get(provider)

    async def has_persisted_credential(self, provider, session_id=None):
        """The probe seam: a fake's keys ARE its persisted rows."""
        self.probe_calls.append((provider, session_id))
        if self.raises:
            raise RuntimeError("store unavailable")
        return bool(self.keys.get(provider))

    async def get_persisted_api_key(self, provider, session_id=None, *, kinds=None):
        """Rung 3's probe AND call-time seam: the key, from persisted rows only."""
        self.probe_calls.append((provider, session_id))
        if self.raises:
            raise RuntimeError("store unavailable")
        return self.keys.get(provider)


def _no_radient(monkeypatch, *, present: bool = False):
    async def fake(_config_dir, _base_url, *, store=None):
        return SecretStr("radient-key" if present else "")

    monkeypatch.setattr(cascade, "resolve_radient_credential", fake)

    async def probe(_config_dir, _base_url, *, store):
        return present

    monkeypatch.setattr(cascade, "has_persisted_radient_credential", probe)


def _model(capable: bool | None):
    if capable is None:
        return SimpleNamespace()
    return SimpleNamespace(supports_audio_input=capable)


async def _resolve(store, model=None, session_id=None):
    return await cascade.resolve_audio_path(
        config_dir=Path("/nonexistent-config"),
        base_url="https://api.radienthq.com/v1",
        model=model,
        session_id=session_id,
        store=store,
    )


RUNG_ORDER = [
    AudioPath.PROVIDER_STT_RADIENT,
    AudioPath.PROVIDER_STT_ELEVENLABS,
    AudioPath.PROVIDER_STT_OPENAI,
    AudioPath.PROVIDER_STT_SUPERWHISPER,
    AudioPath.MODEL_AUDIO_SIDECAR,
]


def _rung(resolution, path):
    return next(rung for rung in resolution.rungs if rung.path == path)


@pytest.mark.asyncio
async def test_radient_alone_wins(monkeypatch) -> None:
    _no_radient(monkeypatch, present=True)
    resolution = await _resolve(FakeStore())
    assert resolution.path == AudioPath.PROVIDER_STT_RADIENT
    assert resolution.reason == "Signed in to Radient."


@pytest.mark.asyncio
async def test_elevenlabs_alone(monkeypatch) -> None:
    _no_radient(monkeypatch)
    resolution = await _resolve(FakeStore({"elevenlabs": "el-key"}))
    assert resolution.path == AudioPath.PROVIDER_STT_ELEVENLABS
    assert _rung(resolution, AudioPath.PROVIDER_STT_ELEVENLABS).available is True


@pytest.mark.asyncio
async def test_openai_alone(monkeypatch) -> None:
    _no_radient(monkeypatch)
    resolution = await _resolve(FakeStore({"openai-key": "oai-key"}))
    assert resolution.path == AudioPath.PROVIDER_STT_OPENAI


@pytest.mark.asyncio
async def test_radient_wins_over_every_later_rung(monkeypatch) -> None:
    _no_radient(monkeypatch, present=True)
    resolution = await _resolve(
        FakeStore({"elevenlabs": "el-key", "openai-key": "oai-key"}),
        model=_model(True),
    )
    assert resolution.path == AudioPath.PROVIDER_STT_RADIENT


@pytest.mark.asyncio
async def test_elevenlabs_wins_over_openai_and_model_audio(monkeypatch) -> None:
    _no_radient(monkeypatch)
    resolution = await _resolve(
        FakeStore({"elevenlabs": "el-key", "openai-key": "oai-key"}), model=_model(True)
    )
    assert resolution.path == AudioPath.PROVIDER_STT_ELEVENLABS


@pytest.mark.asyncio
async def test_none_with_audio_model_falls_to_model_audio_sidecar(monkeypatch) -> None:
    _no_radient(monkeypatch)
    resolution = await _resolve(FakeStore(), model=_model(True))
    assert resolution.path == AudioPath.MODEL_AUDIO_SIDECAR
    assert resolution.model_audio_capable is True
    assert _rung(resolution, AudioPath.MODEL_AUDIO_SIDECAR).available is True


@pytest.mark.asyncio
async def test_none_without_audio_model_reports_none_with_reason(monkeypatch) -> None:
    _no_radient(monkeypatch)
    resolution = await _resolve(FakeStore(), model=_model(False))
    assert resolution.path == AudioPath.NONE
    assert resolution.model_audio_capable is False
    assert "does not accept audio" in resolution.reason


@pytest.mark.asyncio
async def test_none_without_any_model_names_that(monkeypatch) -> None:
    _no_radient(monkeypatch)
    resolution = await _resolve(FakeStore(), model=None)
    assert resolution.path == AudioPath.NONE
    assert "no model is selected" in resolution.reason


@pytest.mark.asyncio
async def test_a_model_without_the_attribute_reads_as_not_audio_capable(monkeypatch) -> None:
    """Phase-0 defensive read: a spec that predates the field is not capable."""
    _no_radient(monkeypatch)
    resolution = await _resolve(FakeStore(), model=SimpleNamespace())
    assert resolution.model_audio_capable is False
    assert resolution.path == AudioPath.NONE


@pytest.mark.asyncio
async def test_the_rung_order_is_the_frozen_cascade_order() -> None:
    """Order is a contract: it is what makes "first available" deterministic."""
    resolution = await _resolve(FakeStore())
    assert [rung.path for rung in resolution.rungs] == RUNG_ORDER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keys",
    [{}, {"elevenlabs": "k"}, {"openai-key": "k"}, {"elevenlabs": "k", "openai-key": "k"}],
)
@pytest.mark.parametrize("radient_present", [False, True])
@pytest.mark.parametrize("model_capable", [None, False, True])
async def test_superwhisper_is_never_returned_and_always_unavailable(
    monkeypatch, keys, radient_present, model_capable
) -> None:
    """The reserved rung is report-only: present, unavailable, never a path."""
    _no_radient(monkeypatch, present=radient_present)
    resolution = await _resolve(FakeStore(keys), model=_model(model_capable))
    assert resolution.path != AudioPath.PROVIDER_STT_SUPERWHISPER
    rung = _rung(resolution, AudioPath.PROVIDER_STT_SUPERWHISPER)
    assert rung.available is False
    assert rung.reason == cascade.SUPERWHISPER_REASON


@pytest.mark.asyncio
async def test_probes_are_read_only_and_carry_the_session(monkeypatch) -> None:
    """A probe must not rotate account stickiness (design §3b)."""
    _no_radient(monkeypatch)
    store = FakeStore()
    await _resolve(store, session_id="sess-1")
    # The probe (``has_persisted_credential``) fixes ``read_only`` itself, so the
    # resolver's contract is "ask the probe, carry the session, never call the
    # call-time cascade": ``calls`` is empty.
    assert store.probe_calls == [("elevenlabs", "sess-1"), ("openai-key", "sess-1")]
    assert store.calls == []


@pytest.mark.asyncio
async def test_a_failing_probe_marks_its_rung_unavailable_not_the_resolve(monkeypatch) -> None:
    _no_radient(monkeypatch)
    resolution = await _resolve(FakeStore(raises=True))
    assert resolution.path == AudioPath.NONE
    assert _rung(resolution, AudioPath.PROVIDER_STT_ELEVENLABS).available is False


@pytest.mark.asyncio
async def test_real_auth_store_seam_resolves_a_stored_key(monkeypatch, tmp_path) -> None:
    """The strongest form: a real AuthStore row is what the probe reads."""
    _no_radient(monkeypatch)
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        store.upsert_credential(
            "elevenlabs", {"type": "api_key", "source": "login", "key": "el-live-key"}
        )
        resolution = await _resolve(store)
    finally:
        store.close()
    assert resolution.path == AudioPath.PROVIDER_STT_ELEVENLABS
    assert _rung(resolution, AudioPath.PROVIDER_STT_ELEVENLABS).reason == (
        "An ElevenLabs API key is stored."
    )


@pytest.mark.asyncio
async def test_owned_store_is_created_and_closed_when_none_is_injected(
    monkeypatch, tmp_path
) -> None:
    """Called without a store, the resolver builds one on the config dir."""
    calls: list[dict[str, Any]] = []
    real_close = AuthStore.close

    def recording_close(self):
        calls.append({"closed": True})
        return real_close(self)

    monkeypatch.setattr(AuthStore, "close", recording_close)

    async def fake(_config_dir, _base_url, *, store=None):
        assert isinstance(store, AuthStore)
        return SecretStr("")

    monkeypatch.setattr(cascade, "resolve_radient_credential", fake)
    resolution = await cascade.resolve_audio_path(
        config_dir=tmp_path, base_url="https://api.radienthq.com/v1"
    )
    assert resolution.path == AudioPath.NONE
    assert calls == [{"closed": True}]


@pytest.mark.asyncio
async def test_an_injected_store_is_not_closed(monkeypatch, tmp_path) -> None:
    store = FakeStore()
    closed: list[bool] = []
    store.close = lambda: closed.append(True)  # type: ignore[attr-defined]

    async def fake(_config_dir, _base_url, *, store=None):
        return SecretStr("")

    monkeypatch.setattr(cascade, "resolve_radient_credential", fake)
    await cascade.resolve_audio_path(
        config_dir=tmp_path,
        base_url="https://api.radienthq.com/v1",
        store=cast("AuthStore", store),
    )
    assert closed == []
