"""The TTS resolver advertises a rung from PERSISTED rows only.

The same contract the S0 slice pinned for STT, restated for TTS so the two
cascades cannot diverge: runtime/config overrides and environment variables do
NOT light a rung, stored rows (OAuth included) DO. Exercised through a real
:class:`AuthStore` on an isolated root, because the whole failure mode is a
resolver that asked the wrong method and got an ambient value back.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.tts import VoicePath, cascade

CANONICAL = "https://api.radienthq.com/v1"

RUNG_ORDER = [
    VoicePath.PROVIDER_TTS_RADIENT,
    VoicePath.PROVIDER_TTS_ELEVENLABS,
    VoicePath.PROVIDER_TTS_OPENAI,
]


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
    # A root with no provider-class rows, and no ambient provider vars: the
    # point of every test below is what the probe does WITHOUT them.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-config"))
    for var in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "RADIENT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path / "config")
    yield auth
    auth.close()


def _oauth_payload() -> dict[str, Any]:
    return {
        "type": "oauth",
        "refresh": "r1",
        "access": "radient-access-token",
        "expires": int(time.time() * 1000) + 3_600_000,
    }


def _api_key_row() -> dict[str, Any]:
    return {"type": "api_key", "source": "login", "key": "stored-key"}


async def _available(store: AuthStore, tmp_path: Path) -> dict[VoicePath, bool]:
    resolution = await cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    return {rung.path: rung.available for rung in resolution.rungs}


def test_the_rung_order_is_written_once() -> None:
    """Radient → ElevenLabs → OpenAI, from the one constant the resolver reads."""
    assert list(cascade.TTS_RUNG_PATHS) == RUNG_ORDER


def test_the_openai_namespace_is_the_speech_only_one() -> None:
    """A key stored under ``openai`` would be the chat OAuth row, not an API key."""
    assert cascade.OPENAI_TTS_NAMESPACE == "openai-key"
    assert cascade.OPENAI_TTS_KINDS == frozenset({"api_key"})


@pytest.mark.asyncio
async def test_ambient_state_lights_nothing_and_stored_rows_light_each_rung(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both directions, one test, on a real store."""
    monkeypatch.setenv("OPENAI_API_KEY", "ambient")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "ambient")
    monkeypatch.setenv("RADIENT_API_KEY", "ambient")
    store.set_runtime_api_key("openai", "flag-key")
    store.set_runtime_api_key("openai-key", "flag-key")
    store.set_runtime_api_key("elevenlabs", "flag-key")

    before = await _available(store, tmp_path)
    assert before == dict.fromkeys(RUNG_ORDER, False)

    store.upsert_credential("elevenlabs", _api_key_row())
    store.upsert_credential("openai-key", _api_key_row())
    store.upsert_credential("radient", _oauth_payload())
    after = await _available(store, tmp_path)
    assert after == dict.fromkeys(RUNG_ORDER, True)


@pytest.mark.asyncio
async def test_nothing_stored_reads_none_and_says_why(store: AuthStore, tmp_path: Path) -> None:
    resolution = await cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    assert resolution.path is VoicePath.NONE
    assert resolution.servable is False
    assert "Radient" in resolution.reason and "ElevenLabs" in resolution.reason


@pytest.mark.asyncio
async def test_a_chatgpt_oauth_row_does_not_light_the_openai_rung(
    store: AuthStore, tmp_path: Path
) -> None:
    """An OAuth row is not an API key: the audio endpoint would 401 on it."""
    store.upsert_credential("openai", _oauth_payload())
    assert (await _available(store, tmp_path))[VoicePath.PROVIDER_TTS_OPENAI] is False
    # ...and the call-time key is absent too, so the ChatGPT token is never sent.
    assert await cascade._openai_tts_key(store, None) is None


@pytest.mark.asyncio
async def test_the_legacy_provider_class_store_row_still_lights_the_openai_rung(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lop credential update OPENAI_API_KEY`` keeps working; the env does not."""
    from local_operator.providers.registry import store_provider_key

    monkeypatch.setenv("OPENAI_API_KEY", "ambient")
    assert (await _available(store, tmp_path))[VoicePath.PROVIDER_TTS_OPENAI] is False
    store_provider_key("OPENAI_API_KEY", "legacy-stored", base=tmp_path / "config")
    assert (await _available(store, tmp_path))[VoicePath.PROVIDER_TTS_OPENAI] is True
    assert await cascade._openai_tts_key(store, None) == "legacy-stored"


@pytest.mark.asyncio
async def test_a_probe_that_raises_reads_unavailable_never_raises(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed: a broken probe is "not available", not an exception."""

    async def boom(*_args, **_kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(cascade, "has_persisted_radient_credential", boom)
    monkeypatch.setattr(store, "has_persisted_credential", boom)
    monkeypatch.setattr(store, "get_persisted_api_key", boom)
    resolution = await cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    assert resolution.path is VoicePath.NONE
    assert resolution.servable is False
    assert all(rung.available is False for rung in resolution.rungs)


@pytest.mark.asyncio
async def test_resolving_twice_is_stable(store: AuthStore, tmp_path: Path) -> None:
    """The resolver holds no cross-call state: the same store answers the same way."""
    store.upsert_credential("elevenlabs", _api_key_row())
    first = await cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    second = await cascade.resolve_voice_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    assert first == second
