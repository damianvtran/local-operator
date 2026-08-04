"""AuthStore tests: cascade order, legacy credentials.env tier, refresh,
rotation, blocking. No network: fakes for the refresh capability."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.credentials import CredentialManager
from local_operator.providers.auth_store import AuthStore, AuthStoreError

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AuthStore:
    # Hermeticity: the env-tier legacy loader reads ~/.local-operator/
    # credentials.env; point it at an empty dir so real user keys never leak in.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-config"))
    # No real provider env vars may leak into cascade assertions.
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_OAUTH_TOKEN",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "KIMI_API_KEY",
        "XAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    auth = AuthStore(db_path=tmp_path / "auth.db")
    yield auth
    auth.close()


def _oauth(refresh: str = "r1", access: str = "access-1", expires: int | None = None) -> dict[str, Any]:
    import time

    return {
        "refresh": refresh,
        "access": access,
        "expires": expires if expires is not None else int(time.time() * 1000) + 3600_000,
    }


async def test_cascade_runtime_override_wins(store: AuthStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    store.set_config_api_key("openai", "config-key")
    store.upsert_credential("openai", _oauth())
    store.set_runtime_api_key("openai", "runtime-key")
    assert await store.get_api_key("openai") == "runtime-key"
    store.set_runtime_api_key("openai", None)
    assert await store.get_api_key("openai") == "config-key"


async def test_cascade_config_beats_oauth(store: AuthStore) -> None:
    store.set_config_api_key("openai", "config-key")
    store.upsert_credential("openai", _oauth())
    assert await store.get_api_key("openai") == "config-key"


async def test_cascade_oauth_beats_login_key_and_env(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    store.upsert_credential("openai", {"key": "pasted-key", "source": "login", "type": "api_key"})
    store.upsert_credential("openai", _oauth())
    assert await store.get_api_key("openai") == "access-1"


async def test_cascade_login_key_beats_env(store: AuthStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    store.upsert_credential("openai", {"key": "pasted-key", "source": "login", "type": "api_key"})
    assert await store.get_api_key("openai") == "pasted-key"


async def test_cascade_env_beats_plain_stored_key(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    store.upsert_credential("openai", {"key": "stored-key", "type": "api_key"})
    assert await store.get_api_key("openai") == "env-key"


async def test_cascade_stored_key_when_no_env(store: AuthStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store.upsert_credential("openai", {"key": "stored-key", "type": "api_key"})
    assert await store.get_api_key("openai") == "stored-key"


async def test_cascade_legacy_credentials_env_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env tier reads legacy credentials.env via CredentialManager."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manager = CredentialManager(tmp_path / "config")
    manager.set_credential("OPENAI_API_KEY", "legacy-file-key", write=True)
    auth = AuthStore(db_path=tmp_path / "auth.db", credential_manager=manager)
    try:
        assert await auth.get_api_key("openai") == "legacy-file-key"
        # Stored non-login keys rank AFTER the legacy file tier.
        auth.upsert_credential("openai", {"key": "stored-key", "type": "api_key"})
        assert await auth.get_api_key("openai") == "legacy-file-key"
    finally:
        auth.close()


async def test_cascade_fallback_resolver(store: AuthStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store.set_fallback_resolver("openai", lambda provider: "fallback-key")
    assert await store.get_api_key("openai") == "fallback-key"
    store.set_fallback_resolver("openai", None)
    assert await store.get_api_key("openai") is None


async def test_allows_missing_api_key_returns_none(store: AuthStore) -> None:
    assert await store.get_api_key("ollama") is None
    assert await store.get_api_key("test") is None


async def test_oauth_auto_refresh_with_skew(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expired-by-skew credentials are refreshed exactly once (single-flight)."""
    import time

    expired = _oauth(expires=int(time.time() * 1000) + 1000)  # inside 60s skew
    expired["org_id"] = "org-1"  # identity so re-upserts hit the SAME row
    store.upsert_credential("openai", expired)
    calls: list[dict[str, Any]] = []

    async def fake_refresh(creds: dict[str, Any]) -> dict[str, Any]:
        calls.append(creds)
        return {"access": "access-2", "expires": int(time.time() * 1000) + 3600_000}

    monkeypatch.setattr(store, "_refresh_fn", lambda provider: fake_refresh)
    key = await store.get_api_key("openai")
    assert key == "access-2"
    assert len(calls) == 1

    # Concurrent resolution still refreshes only once (per-credential lock).
    re_expire = _oauth(expires=int(time.time() * 1000) + 1000)
    re_expire["org_id"] = "org-1"
    store.upsert_credential("openai", re_expire)
    calls.clear()
    keys = await asyncio.gather(store.get_api_key("openai"), store.get_api_key("openai"))
    assert keys == ["access-2", "access-2"]
    assert len(calls) == 1

    # Refreshed data is persisted; the stored row now carries access-2.
    rows = store.list_credentials("openai")
    assert rows[-1].data["access"] == "access-2"


async def test_force_refresh_failure_blocks_and_raises(store: AuthStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.upsert_credential("openai", _oauth())

    async def bad_refresh(creds: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("idp down")

    monkeypatch.setattr(store, "_refresh_fn", lambda provider: bad_refresh)
    with pytest.raises(AuthStoreError):
        await store.get_api_key("openai", force_refresh=True)


async def test_session_stickiness(store: AuthStore) -> None:
    store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    store.upsert_credential("openai", {"key": "k2", "type": "api_key"})
    first = await store.get_api_key("openai", session_id="session-a")
    # Sticky: the same session keeps hitting the same credential row.
    for _ in range(5):
        assert await store.get_api_key("openai", session_id="session-a") == first
    # Without a session id, round-robin alternates.
    assert {await store.get_api_key("openai") for _ in range(4)} == {"k1", "k2"}


async def test_blocking_backoff(store: AuthStore) -> None:
    row = store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    assert not store.is_blocked(row.id, "openai")
    store.block_credential(row.id, "openai", block_ms=60_000)
    assert store.is_blocked(row.id, "openai")
    assert await store.get_api_key("openai") is None  # only credential blocked
    store.clear_blocks(row.id)
    assert await store.get_api_key("openai") == "k1"


async def test_rotate_sibling_blocks_failing_and_reports_remaining(store: AuthStore) -> None:
    store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    row2 = store.upsert_credential("openai", {"key": "k2", "type": "api_key"})

    from local_operator.providers.failover import ProviderError

    error = ProviderError(401, "invalid token", auth_error=True)
    assert store.rotate_sibling("openai", None, error, api_key="k1") is True
    # k1 is now blocked; resolution lands on k2.
    assert await store.get_api_key("openai") == "k2"
    # No sibling remains once k2 fails too.
    assert store.rotate_sibling("openai", None, error, api_key="k2") is False
    _ = row2


async def test_rotate_sibling_usage_limit_preserves_sticky(store: AuthStore) -> None:
    store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    store.upsert_credential("openai", {"key": "k2", "type": "api_key"})
    await store.get_api_key("openai", session_id="s1")  # establishes sticky

    from local_operator.providers.failover import ProviderError

    usage_limit = ProviderError(429, "rate limit reached", retryable=True)
    assert store.rotate_sibling("openai", "s1", usage_limit, api_key="k1") is True
    # Sticky for s1 was preserved through the usage-limit block.
    sticky = store._sticky.get(("openai", "s1"))
    assert sticky is not None


async def test_invalidated_token_soft_deletes_row(store: AuthStore) -> None:
    store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    from local_operator.providers.failover import ProviderError

    invalidated = ProviderError(401, "invalid api key", auth_error=True)
    store.rotate_sibling("openai", None, invalidated, api_key="k1")
    rows = store.list_credentials("openai")
    assert rows == []
    all_rows = store.list_credentials("openai", include_disabled=True)
    assert all_rows[0].disabled_cause == "invalidated-token"


async def test_upsert_identity_dedupes_oauth_rows(store: AuthStore) -> None:
    creds = _oauth()
    creds["org_id"] = "org-1"
    first = store.upsert_credential("openai", creds)
    relogin = dict(creds)
    relogin["access"] = "access-new"
    second = store.upsert_credential("openai", relogin)
    assert first.id == second.id  # same org → same row, updated in place
    assert second.data["access"] == "access-new"
    # A different org gets its own row.
    other = dict(creds)
    other["org_id"] = "org-2"
    third = store.upsert_credential("openai", other)
    assert third.id != first.id
