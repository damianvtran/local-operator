"""The STT resolver advertises a rung from PERSISTED rows only (voicing S0).

Cascade-lane sign-off, 2026-10-01, condition 4: pin BOTH directions through a
real ``AuthStore`` -- runtime/config override and env do NOT light the rungs,
stored rows (OAuth included) DO. Kept apart from the rest of the S0 tests so the
STT half reverts, and is reviewed, as one piece.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.providers.auth_store import AuthStore

CANONICAL = "https://api.radienthq.com/v1"


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
        "access": "chatgpt-access-token",
        "expires": int(time.time() * 1000) + 3_600_000,
    }


# ---------------------------------------------------------------------------
# The STT rungs are on the probe (both directions, real store)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_rungs_ignore_ambient_state_and_light_on_stored_rows(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.stt import AudioPath, cascade

    monkeypatch.setenv("OPENAI_API_KEY", "ambient")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "ambient")
    monkeypatch.setenv("RADIENT_API_KEY", "ambient")
    store.set_runtime_api_key("openai", "flag-key")
    store.set_runtime_api_key("openai-key", "flag-key")

    async def resolve() -> Any:
        return await cascade.resolve_audio_path(
            config_dir=tmp_path / "config", base_url=CANONICAL, store=store
        )

    def available(resolution: Any) -> dict[AudioPath, bool]:
        return {rung.path: rung.available for rung in resolution.rungs}

    before = available(await resolve())
    assert before[AudioPath.PROVIDER_STT_RADIENT] is False
    assert before[AudioPath.PROVIDER_STT_ELEVENLABS] is False
    assert before[AudioPath.PROVIDER_STT_OPENAI] is False

    store.upsert_credential("elevenlabs", {"type": "api_key", "source": "login", "key": "k"})
    store.upsert_credential("openai-key", {"type": "api_key", "source": "login", "key": "sk"})
    store.upsert_credential("radient", _oauth_payload())
    after = available(await resolve())
    assert after[AudioPath.PROVIDER_STT_RADIENT] is True, "rung 1 keeps OAuth"
    assert after[AudioPath.PROVIDER_STT_ELEVENLABS] is True
    assert after[AudioPath.PROVIDER_STT_OPENAI] is True, "the openai-key login lights rung 3"


# ---------------------------------------------------------------------------
# Rung 3's credential CLASS (review round 1, MAJOR / S-1): an OpenAI API key,
# never a ChatGPT OAuth grant. This REFINES the cascade lane's condition 3 for
# rung 3 only -- its call-time key moves with its probe, because a ChatGPT token
# is valid for the full cascade yet is not accepted at /v1/audio/*. The earlier
# version of this file pinned the opposite (an OAuth row lighting rung 3), i.e.
# it pinned the false positive as intended.
# ---------------------------------------------------------------------------


async def _rung3(store: AuthStore, tmp_path: Path) -> bool:
    from local_operator.stt import AudioPath, cascade

    resolution = await cascade.resolve_audio_path(
        config_dir=tmp_path / "config", base_url=CANONICAL, store=store
    )
    return {rung.path: rung.available for rung in resolution.rungs}[AudioPath.PROVIDER_STT_OPENAI]


@pytest.mark.asyncio
async def test_a_chatgpt_oauth_login_does_not_light_rung_3_or_feed_it(
    store: AuthStore, tmp_path: Path
) -> None:
    from local_operator.stt import cascade

    store.upsert_credential("openai", _oauth_payload())
    assert await _rung3(store, tmp_path) is False
    # ...and the call-time key is absent too, so the ChatGPT token is never sent.
    assert await cascade._openai_stt_key(store, None) is None


@pytest.mark.asyncio
async def test_an_openai_key_login_lights_rung_3_and_is_the_key_it_sends(
    store: AuthStore, tmp_path: Path
) -> None:
    from local_operator.stt import cascade

    store.upsert_credential("openai-key", {"type": "api_key", "source": "login", "key": "sk-real"})
    store.upsert_credential("openai", _oauth_payload())  # beside it, must be ignored
    assert await _rung3(store, tmp_path) is True
    assert await cascade._openai_stt_key(store, None) == "sk-real"


@pytest.mark.asyncio
async def test_the_legacy_provider_class_store_row_still_lights_rung_3(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lop credential update OPENAI_API_KEY`` keeps working; the env does not."""
    from local_operator.providers.registry import store_provider_key
    from local_operator.stt import cascade

    monkeypatch.setenv("OPENAI_API_KEY", "ambient")
    assert await _rung3(store, tmp_path) is False
    store_provider_key("OPENAI_API_KEY", "legacy-stored", base=tmp_path / "config")
    assert await _rung3(store, tmp_path) is True
    assert await cascade._openai_stt_key(store, None) == "legacy-stored"


@pytest.mark.asyncio
async def test_the_executor_rung_refuses_when_only_chatgpt_oauth_is_stored(
    store: AuthStore,
) -> None:
    """The call-time half: no key means the rung's own refusal, never a 401."""
    from local_operator.clients._http import APIError
    from local_operator.stt import cascade

    store.upsert_credential("openai", _oauth_payload())
    with pytest.raises(APIError, match="No OpenAI API key is stored"):
        await cascade._run_openai_rung(
            b"x",
            mime="audio/wav",
            store=store,
            session_id=None,
            language=None,
            prompt=None,
            timeout_s=1.0,
        )
