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
    store.upsert_credential("openai", _oauth_payload())
    store.upsert_credential("radient", _oauth_payload())
    after = available(await resolve())
    assert after[AudioPath.PROVIDER_STT_RADIENT] is True
    assert after[AudioPath.PROVIDER_STT_ELEVENLABS] is True
    assert after[AudioPath.PROVIDER_STT_OPENAI] is True
