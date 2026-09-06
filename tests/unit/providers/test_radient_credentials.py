"""Legacy callers share canonical precedence without another refresh store."""

import asyncio
import time

import pytest

from local_operator.credentials import CredentialManager
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.radient_credentials import (
    resolve_radient_credential,
    resolve_radient_credential_sync,
)

URL = "https://api.radienthq.com/v1"


@pytest.mark.asyncio
async def test_canonical_precedence_and_explicit_gateway_fallback(tmp_path, monkeypatch):
    manager = CredentialManager(tmp_path)
    manager.set_credential("RADIENT_API_KEY", "legacy-fixture", write=False)
    monkeypatch.setenv("RADIENT_API_KEY", "environment-fixture")
    store = AuthStore(tmp_path / "auth.db", credential_manager=manager)
    try:
        key = store.upsert_credential(
            "radient", {"type": "api_key", "source": "login", "key": "login-fixture"}
        )
        oauth = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "oauth-fixture",
                "refresh": "refresh-fixture",
                "expires": int(time.time() * 1000) + 3600000,
            },
        )
        assert (
            await resolve_radient_credential(manager, URL, store=store)
        ).get_secret_value() == "oauth-fixture"
        # A configured foreign gateway never receives that central bearer.
        assert (
            await resolve_radient_credential(manager, "https://gateway.example/v1", store=store)
        ).get_secret_value() == "legacy-fixture"
        store.delete_credential(oauth.id)
        assert (
            await resolve_radient_credential(manager, URL, store=store)
        ).get_secret_value() == "login-fixture"
        store.delete_credential(key.id)
        assert (
            await resolve_radient_credential(manager, URL, store=store)
        ).get_secret_value() == "environment-fixture"
        monkeypatch.delenv("RADIENT_API_KEY")
        assert (
            await resolve_radient_credential(manager, URL, store=store)
        ).get_secret_value() == "legacy-fixture"
    finally:
        store.close()


def test_cli_sync_reader_uses_same_store_and_preserves_legacy_key(tmp_path, monkeypatch):
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    manager = CredentialManager(tmp_path)
    manager.set_credential("RADIENT_API_KEY", "legacy-fixture", write=False)
    store = AuthStore(tmp_path / "auth.db")
    row = store.upsert_credential(
        "radient", {"type": "api_key", "source": "login", "key": "central-fixture"}
    )
    store.close()
    assert (
        resolve_radient_credential_sync(manager, "https://api.radienthq.com").get_secret_value()
        == "central-fixture"
    )
    assert manager.get_credential("RADIENT_API_KEY").get_secret_value() == "legacy-fixture"
    store = AuthStore(tmp_path / "auth.db")
    store.delete_credential(row.id)
    store.close()
    assert resolve_radient_credential_sync(manager, URL).get_secret_value() == "legacy-fixture"


@pytest.mark.asyncio
async def test_parallel_legacy_readers_share_one_refresh_lock(tmp_path, monkeypatch):
    from local_operator.providers.oauth import radient

    manager = CredentialManager(tmp_path)
    store = AuthStore(tmp_path / "auth.db", credential_manager=manager)
    row = store.upsert_credential(
        "radient",
        {"type": "oauth", "access": "expired-fixture", "refresh": "refresh-fixture", "expires": 1},
    )
    count = 0

    async def refresh(credentials):
        nonlocal count
        count += 1
        await asyncio.sleep(0)
        return {
            **credentials,
            "access": "fresh-fixture",
            "expires": int(time.time() * 1000) + 3600000,
        }

    monkeypatch.setattr(radient, "refresh_radient_token", refresh)
    try:
        results = await asyncio.gather(
            *(resolve_radient_credential(manager, URL, store=store) for _ in range(3))
        )
        assert count == 1
        assert all(value.get_secret_value() == "fresh-fixture" for value in results)
        refreshed = store.get_credential(row.id)
        assert refreshed is not None
        assert refreshed.data["access"] == "fresh-fixture"
    finally:
        store.close()
