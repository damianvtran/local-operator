"""``AuthStore.has_persisted_credential``, both directions.

The defect they close was invisible to a green suite: an availability probe
that read the full 7-tier cascade advertised a rung from an exported env var or
a ``--api-key`` flag the user never signed in to. Every test here pins what
does NOT count as well as what does, on a real ``AuthStore`` over a temp db.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.radient_credentials import (
    has_persisted_radient_credential,
)
from local_operator.providers.registry import store_provider_key

CANONICAL = "https://api.radienthq.com/v1"
LEGACY = "https://gateway.example.test/v1"


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
# AuthStore.has_persisted_credential
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stored_login_key_counts(store: AuthStore) -> None:
    store.upsert_credential("elevenlabs", {"type": "api_key", "source": "login", "key": "k"})
    assert await store.has_persisted_credential("elevenlabs") is True


@pytest.mark.asyncio
async def test_a_stored_non_login_key_counts(store: AuthStore) -> None:
    """Tier 6 (e.g. a broker migration) is a stored row too."""
    store.upsert_credential("elevenlabs", {"type": "api_key", "key": "k"})
    assert await store.has_persisted_credential("elevenlabs") is True


@pytest.mark.asyncio
async def test_a_stored_oauth_row_counts(store: AuthStore) -> None:
    store.upsert_credential("openai", _oauth_payload())
    assert await store.has_persisted_credential("openai") is True


@pytest.mark.asyncio
async def test_the_provider_class_store_row_counts(store: AuthStore, tmp_path: Path) -> None:
    """``lop credential`` writes an encrypted STORE row, which is persisted."""
    store_provider_key("ELEVENLABS_API_KEY", "stored", base=tmp_path / "config")
    assert await store.has_persisted_credential("elevenlabs") is True


@pytest.mark.asyncio
async def test_the_process_environment_never_counts(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "ambient")
    # The call-time cascade DOES resolve the export (that is correct there)...
    assert await store.get_api_key("openai", read_only=True) == "ambient"
    # ...and the availability probe does not.
    assert await store.has_persisted_credential("openai") is False
    assert await store.has_persisted_credential("elevenlabs") is False


@pytest.mark.asyncio
async def test_runtime_and_config_overrides_never_count(store: AuthStore) -> None:
    store.set_runtime_api_key("elevenlabs", "flag-key")
    store.set_config_api_key("openai", "config-key")
    assert await store.get_api_key("elevenlabs", read_only=True) == "flag-key"
    assert await store.has_persisted_credential("elevenlabs") is False
    assert await store.has_persisted_credential("openai") is False


@pytest.mark.asyncio
async def test_the_fallback_resolver_never_counts(store: AuthStore) -> None:
    store.set_fallback_resolver("elevenlabs", lambda _provider: "fallback-key")
    assert await store.get_api_key("elevenlabs", read_only=True) == "fallback-key"
    assert await store.has_persisted_credential("elevenlabs") is False


@pytest.mark.asyncio
async def test_a_chatgpt_oauth_row_does_not_light_the_openai_key_namespace(
    store: AuthStore,
) -> None:
    """The ChatGPT login and the speech key are different accounts.

    The probe asks about a STORAGE id, so a ChatGPT OAuth row under ``openai``
    never answers for ``openai-key``.
    """
    store.upsert_credential("openai", _oauth_payload())
    assert await store.has_persisted_credential("openai-key") is False
    store.upsert_credential("openai-key", {"type": "api_key", "source": "login", "key": "k"})
    assert await store.has_persisted_credential("openai-key") is True


@pytest.mark.asyncio
async def test_a_logged_out_row_does_not_count(store: AuthStore) -> None:
    store.upsert_credential("elevenlabs", {"type": "api_key", "source": "login", "key": "k"})
    store.delete_credentials_for_provider("elevenlabs")
    assert await store.has_persisted_credential("elevenlabs") is False


@pytest.mark.asyncio
async def test_the_probe_decides_nothing_about_routing(store: AuthStore) -> None:
    """read_only: no stickiness is written for the session."""
    store.upsert_credential("openai", _oauth_payload())
    assert await store.has_persisted_credential("openai", "sess-1") is True
    assert ("openai", "sess-1") not in store._sticky


@pytest.mark.asyncio
async def test_the_probe_fails_closed_and_never_raises(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.upsert_credential("elevenlabs", {"type": "api_key", "source": "login", "key": "k"})

    async def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(store, "_resolve", boom)
    assert await store.has_persisted_credential("elevenlabs") is False


@pytest.mark.asyncio
async def test_the_executor_path_keeps_the_full_cascade(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Condition 3: only the PROBE narrowed. ``get_api_key`` still reads env."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "ambient")
    assert await store.get_api_key("elevenlabs", read_only=True) == "ambient"
    store.set_runtime_api_key("elevenlabs", "flag-key")
    assert await store.get_api_key("elevenlabs", read_only=True) == "flag-key"


# ---------------------------------------------------------------------------
# The Radient half (rung 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_radient_canonical_counts_a_stored_row_and_not_the_env(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RADIENT_API_KEY", "ambient")
    assert (
        await has_persisted_radient_credential(tmp_path / "config", CANONICAL, store=store) is False
    )
    store.upsert_credential("radient", _oauth_payload())
    assert (
        await has_persisted_radient_credential(tmp_path / "config", CANONICAL, store=store) is True
    )


@pytest.mark.asyncio
async def test_radient_legacy_gateway_means_a_store_row_never_env(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-canonical branch must still say "a STORE row", never the env."""
    config = tmp_path / "config"
    monkeypatch.setenv("RADIENT_API_KEY", "ambient")
    # A central login is for the canonical host only; it must not answer here.
    store.upsert_credential("radient", _oauth_payload())
    assert await has_persisted_radient_credential(config, LEGACY, store=store) is False
    store_provider_key("RADIENT_API_KEY", "stored", base=config)
    assert await has_persisted_radient_credential(config, LEGACY, store=store) is True


@pytest.mark.asyncio
async def test_radient_probe_never_raises(tmp_path: Path) -> None:
    class Broken:
        async def has_persisted_credential(self, *_args: Any, **_kwargs: Any) -> bool:
            raise RuntimeError("store gone")

    broken: Any = Broken()
    assert await has_persisted_radient_credential(tmp_path, CANONICAL, store=broken) is False


# ---------------------------------------------------------------------------
# Row KINDS and the namespace-derived store-row leg (review round 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kinds_api_key_skips_an_oauth_row(store: AuthStore) -> None:
    """``kinds={"api_key"}``: a ChatGPT-style OAuth row alone is not a login."""
    store.upsert_credential("openai", _oauth_payload())
    assert await store.has_persisted_credential("openai") is True
    assert await store.has_persisted_credential("openai", kinds={"api_key"}) is False
    assert await store.get_persisted_api_key("openai", kinds={"api_key"}) is None


@pytest.mark.asyncio
async def test_kinds_api_key_still_counts_both_api_key_tiers(store: AuthStore) -> None:
    store.upsert_credential("elevenlabs", {"type": "api_key", "source": "login", "key": "k1"})
    assert await store.has_persisted_credential("elevenlabs", kinds={"api_key"}) is True
    store.delete_credentials_for_provider("elevenlabs")
    store.upsert_credential("elevenlabs", {"type": "api_key", "key": "k2"})
    assert await store.get_persisted_api_key("elevenlabs", kinds={"api_key"}) == "k2"


@pytest.mark.asyncio
async def test_the_store_row_leg_is_derived_from_the_namespace_not_env_keys(
    store: AuthStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``openai-key`` declares no ``env_keys``; its legacy store row still counts.

    The leg used to read ``env_key_names`` and was therefore EMPTY for exactly the
    row S2's TTS probe asks about. ``legacy_store_keys`` names the store row; the
    process environment is still never read.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "ambient")
    assert await store.has_persisted_credential("openai-key") is False
    store_provider_key("OPENAI_API_KEY", "stored", base=tmp_path / "config")
    assert await store.has_persisted_credential("openai-key") is True
    assert await store.get_persisted_api_key("openai-key") == "stored"
    # An OAuth-only ask never reads the store row (it is an api_key).
    assert await store.has_persisted_credential("openai-key", kinds={"oauth"}) is False


@pytest.mark.asyncio
async def test_the_store_row_leg_works_for_a_callable_env_keys_row(
    store: AuthStore, tmp_path: Path
) -> None:
    """``anthropic`` has callable ``env_keys`` (no plain name): documented, pinned.

    ``env_key_names`` is empty for it, so the leg cannot name a store row and
    answers from credential rows alone -- the honest behaviour, written down so a
    later probe of ``anthropic`` does not assume otherwise.
    """
    store_provider_key("ANTHROPIC_API_KEY", "stored", base=tmp_path / "config")
    assert await store.has_persisted_credential("anthropic") is False
