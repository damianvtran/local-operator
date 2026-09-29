"""Legacy callers share canonical precedence without another refresh store."""

import asyncio
import time
from typing import Any

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.radient_credentials import (
    resolve_radient_credential,
    resolve_radient_credential_sync,
    resolve_radient_oauth_access,
    resolve_radient_oauth_access_sync,
)

URL = "https://api.radienthq.com/v1"


@pytest.mark.asyncio
async def test_canonical_precedence_and_explicit_gateway_fallback(tmp_path, monkeypatch):
    from local_operator.providers.registry import store_provider_key

    # ``readonly`` + a provider-class store row (PR2a): the plaintext file leg is
    # gone, so a static key lives in the store or in the environment.
    manager = tmp_path
    store_provider_key("RADIENT_API_KEY", "legacy-fixture", base=tmp_path)
    monkeypatch.setenv("RADIENT_API_KEY", "environment-fixture")
    store = AuthStore(tmp_path / "auth.db", config_dir=manager)
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
        # Store-first: with both login rows gone the provider store row still
        # outranks the exported variable (the order every reader shares).
        assert (
            await resolve_radient_credential(manager, URL, store=store)
        ).get_secret_value() == "legacy-fixture"
        from local_operator.providers.registry import remove_provider_key

        remove_provider_key("RADIENT_API_KEY", base=tmp_path)
        assert (
            await resolve_radient_credential(manager, URL, store=store)
        ).get_secret_value() == "environment-fixture"
        monkeypatch.delenv("RADIENT_API_KEY")
        # Nothing stored and nothing exported: the leg reports no credential.
        assert (
            await resolve_radient_credential(manager, URL, store=store)
        ).get_secret_value() == ""
    finally:
        store.close()


def test_cli_sync_reader_uses_same_store_and_preserves_legacy_key(tmp_path, monkeypatch):
    from local_operator.providers.registry import store_provider_key

    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    manager = tmp_path
    store_provider_key("RADIENT_API_KEY", "legacy-fixture", base=tmp_path)
    store = AuthStore(tmp_path / "auth.db")
    row = store.upsert_credential(
        "radient", {"type": "api_key", "source": "login", "key": "central-fixture"}
    )
    store.close()
    assert (
        resolve_radient_credential_sync(manager, "https://api.radienthq.com").get_secret_value()
        == "central-fixture"
    )
    from local_operator.providers.registry import provider_secret_value

    assert provider_secret_value("RADIENT_API_KEY", base=tmp_path) == "legacy-fixture"
    store = AuthStore(tmp_path / "auth.db")
    store.delete_credential(row.id)
    store.close()
    assert resolve_radient_credential_sync(manager, URL).get_secret_value() == "legacy-fixture"


@pytest.mark.asyncio
async def test_parallel_legacy_readers_share_one_refresh_lock(tmp_path, monkeypatch):
    from local_operator.providers.oauth import radient

    manager = tmp_path
    store = AuthStore(tmp_path / "auth.db", config_dir=manager)
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


@pytest.mark.asyncio
async def test_an_injected_store_is_used_without_the_mesh_predicate(tmp_path, monkeypatch) -> None:
    """P4: an injected store is used VERBATIM — the mesh predicate is never consulted.

    ``build_auth_store`` replaced the resolvers' own construction; the ``store=`` seam
    exists for tests and embedded callers (the server passes its own) and must stay a
    plain pass-through, or an injected root would be quietly swapped for the ambient
    one. Red the moment ``build_auth_store`` is consulted on this path.
    """

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the resolver consulted build_auth_store for an injected store")

    monkeypatch.setattr("local_operator.network.credentials.build_auth_store", _forbidden)
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "oauth-fixture",
                "refresh": "refresh-fixture",
                "expires": int(time.time() * 1000) + 3600000,
            },
        )
        access = await resolve_radient_oauth_access(tmp_path, URL, store=store)
        assert access is not None
        assert access.access_token == "oauth-fixture"
        value = await resolve_radient_credential(tmp_path, URL, store=store)
        assert value.get_secret_value() == "oauth-fixture"
    finally:
        store.close()


# --- Organization (person-scoped) access: design §8.3 -------------------------


@pytest.mark.asyncio
async def test_oauth_access_resolves_only_an_oauth_row(tmp_path) -> None:
    """Org calls take the PERSON's credential: an OAuth row, never a pasted key.

    An API key proves an application tenant, not a person's membership (§2.2),
    so a ``radient-key`` login must resolve to None -- the caller answers with
    the re-login remedy instead of acting under a credential that cannot
    represent a person.
    """
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        store.upsert_credential(
            "radient", {"type": "api_key", "source": "login", "key": "pasted-fixture"}
        )
        assert await resolve_radient_oauth_access(tmp_path, URL, store=store) is None

        oauth = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "oauth-fixture",
                "refresh": "refresh-fixture",
                "expires": int(time.time() * 1000) + 3600000,
            },
        )
        access = await resolve_radient_oauth_access(tmp_path, URL, store=store)
        assert access is not None
        assert access.kind == "oauth"
        assert access.access_token == "oauth-fixture"

        store.delete_credential(oauth.id)
        assert await resolve_radient_oauth_access(tmp_path, URL, store=store) is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_oauth_access_resolves_only_for_a_permitted_destination(
    tmp_path, monkeypatch
) -> None:
    """The person's org bearer travels only where the public guard allows.

    Mirror of ``test_canonical_precedence_and_explicit_gateway_fallback``'s
    gateway leg (security round 1, S-1): org calls carry the CENTRAL OAuth
    token, so a configured foreign gateway resolves to None -- refused before
    any credential is attached -- and only the explicit
    ``RADIENT_ORG_ALLOW_NONCANONICAL_BASE`` opt-in lets a local/staging hub
    receive it. A truthy-string check, so ``0`` stays refused.
    """
    from local_operator.providers.radient_credentials import ORG_ALLOW_NONCANONICAL_ENV

    monkeypatch.delenv(ORG_ALLOW_NONCANONICAL_ENV, raising=False)
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "oauth-fixture",
                "refresh": "refresh-fixture",
                "expires": int(time.time() * 1000) + 3600000,
            },
        )
        # Canonical: allowed, as before. Foreign gateway, default: refused.
        access = await resolve_radient_oauth_access(tmp_path, URL, store=store)
        assert access is not None and access.access_token == "oauth-fixture"
        assert (
            await resolve_radient_oauth_access(tmp_path, "https://gateway.example/v1", store=store)
            is None
        )
        assert (
            await resolve_radient_oauth_access(tmp_path, "http://127.0.0.1:28571/v1", store=store)
            is None
        )
        # Explicit opt-in, truthy only: the local hub is then allowed.
        monkeypatch.setenv(ORG_ALLOW_NONCANONICAL_ENV, "0")
        assert (
            await resolve_radient_oauth_access(tmp_path, "http://127.0.0.1:28571/v1", store=store)
            is None
        )
        monkeypatch.setenv(ORG_ALLOW_NONCANONICAL_ENV, "1")
        access = await resolve_radient_oauth_access(
            tmp_path, "http://127.0.0.1:28571/v1", store=store
        )
        assert access is not None and access.access_token == "oauth-fixture"
    finally:
        store.close()


def test_oauth_access_sync_bridge_uses_the_store(tmp_path) -> None:
    """The CLI-only bridge resolves from the same central store."""
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "oauth-fixture",
                "refresh": "refresh-fixture",
                "expires": int(time.time() * 1000) + 3600000,
            },
        )
    finally:
        store.close()

    access = resolve_radient_oauth_access_sync(tmp_path, URL)
    assert access is not None
    assert access.access_token == "oauth-fixture"


def test_oauth_access_sync_bridge_is_empty_with_nothing_stored(tmp_path) -> None:
    assert resolve_radient_oauth_access_sync(tmp_path, URL) is None
