"""Real AuthStore row retention prerequisite, not Bridge onboarding evidence.

These tests must run only in the granted isolated validation slot. No caller may
infer an account binding from whichever credential the model cascade selects.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.providers.auth_store import (
    AuthStore,
    AuthStoreError,
    StoredCredential,
)
from local_operator.team_bridge.client import TeamBridgeClient


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
    # Keep genuine SQL/post-write bookkeeping, but never wake a real tunnel
    # service as a side effect of these synthetic interactive login receipts.
    monkeypatch.setattr(AuthStore, "_rearm_parked_tunnel_login", lambda *args, **kwargs: None)
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)
    yield auth
    auth.close()


def _stored(store: AuthStore, row_id: int) -> StoredCredential:
    row = store.get_credential(row_id)
    assert row is not None
    return row


def _login(
    account: str, *, org: str = "shared", endpoint: str = "https://hub.example"
) -> dict[str, Any]:
    return {
        "type": "oauth",
        "access": "test-only-access",
        "refresh": "test-only-refresh",
        "expires": 9999999999999,
        "account_id": account,
        "org_id": org,
        "api_endpoint": endpoint,
    }


def test_accounts_in_same_org_keep_distinct_rows(store: AuthStore) -> None:
    first = store.upsert_credential("radient", _login("account-a"))
    second = store.upsert_credential("radient", _login("account-b"))
    assert first.id != second.id
    assert {row.data["account_id"] for row in store.list_credentials("radient")} == {
        "account-a",
        "account-b",
    }
    assert _stored(store, first.id).data["account_id"] == "account-a"


def test_same_binding_relogin_reuses_row_without_touching_other_account(store: AuthStore) -> None:
    first = store.upsert_credential("radient", _login("account-a"))
    other = store.upsert_credential("radient", _login("account-b"))
    again = store.upsert_credential("radient", {**_login("account-a"), "access": "new-test-access"})
    assert again.id == first.id
    assert _stored(store, other.id).data["account_id"] == "account-b"


def test_default_org_change_reuses_account_but_another_hub_never_aliases(store: AuthStore) -> None:
    first = store.upsert_credential("radient", _login("account-a"))
    changed_org = store.upsert_credential("radient", _login("account-a", org="later-org"))
    other_hub = store.upsert_credential(
        "radient", _login("account-a", endpoint="https://other.example")
    )
    assert changed_org.id == first.id
    assert _stored(store, first.id).data["org_id"] == "later-org"
    assert other_hub.id != first.id
    assert len(store.list_credentials("radient")) == 2


def test_upgrade_adopts_only_matching_proven_legacy_binding(store: AuthStore) -> None:
    first = store.upsert_credential("radient", _login("account-a"))
    # Simulate the old org-only dedupe key without rewriting identity metadata.
    store._conn.execute(
        "UPDATE auth_credentials SET identity_key = ? WHERE id = ?", ("shared", first.id)
    )
    store._conn.commit()
    other = store.upsert_credential("radient", _login("account-b"))
    again = store.upsert_credential("radient", _login("account-a"))
    assert again.id == first.id
    assert other.id != first.id


def test_unknown_legacy_account_is_never_rebound_to_a_new_login(store: AuthStore) -> None:
    unknown = store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "old-test-access",
            "refresh": "old-test-refresh",
        },
    )
    known = store.upsert_credential("radient", _login("account-a", org=""))
    assert known.id != unknown.id
    assert not _stored(store, unknown.id).data.get("account_id")


def test_ambiguous_legacy_rows_fail_without_overwriting_either(store: AuthStore) -> None:
    first = store.upsert_credential("radient", _login("account-a"))
    second = store.upsert_credential("radient", _login("account-b"))
    # Old versions could already leave duplicate identities under different
    # keys. Guessing which id to keep would invalidate someone's captured pin.
    store._conn.execute(
        "UPDATE auth_credentials SET data = (SELECT data FROM auth_credentials WHERE id = ?) "
        "WHERE id = ?",
        (first.id, second.id),
    )
    store._conn.commit()
    before = store.list_credentials("radient", include_disabled=True)
    with pytest.raises(AuthStoreError, match="Ambiguous"):
        store.upsert_credential("radient", _login("account-a"))
    assert store.list_credentials("radient", include_disabled=True) == before


def test_unattributed_login_cannot_overwrite_proven_legacy_org_row(store: AuthStore) -> None:
    known = store.upsert_credential("radient", _login("account-a"))
    store._conn.execute(
        "UPDATE auth_credentials SET identity_key = ? WHERE id = ?", ("shared", known.id)
    )
    store._conn.commit()
    without_account = _login("account-a")
    without_account.pop("account_id")
    with pytest.raises(AuthStoreError, match="Unproven"):
        store.upsert_credential("radient", without_account)
    assert _stored(store, known.id).data["account_id"] == "account-a"


@pytest.mark.asyncio
async def test_pinned_access_uses_retained_row_not_current_default(store: AuthStore) -> None:
    first = store.upsert_credential("radient", _login("account-a"))
    store.upsert_credential("radient", {**_login("account-b"), "access": "b-test-access"})
    captured = await store.ensure_oauth_fresh_or_raise(first.id)
    assert captured is not None
    assert captured["api_endpoint"] == "https://hub.example"
    assert captured["account_id"] == "account-a"
    assert captured["access"] == "test-only-access"


@pytest.mark.parametrize("incoming_hub", [None, "https://hub.example"])
def test_legacy_missing_hub_is_not_adopted_from_current_default(
    store: AuthStore,
    incoming_hub: str | None,
) -> None:
    missing_hub = _login("account-a")
    missing_hub.pop("api_endpoint")
    old = store.upsert_credential("radient", missing_hub)
    store._conn.execute(
        "UPDATE auth_credentials SET identity_key = ? WHERE id = ?", ("account-a", old.id)
    )
    store._conn.commit()
    incoming = _login("account-a")
    if incoming_hub is None:
        incoming.pop("api_endpoint")
    else:
        incoming["api_endpoint"] = incoming_hub
    new = store.upsert_credential("radient", incoming)
    assert new.id != old.id
    assert "api_endpoint" not in _stored(store, old.id).data
    # Neither an unknown legacy hub nor a mutable default org may shadow the
    # explicitly selected new credential on subsequent login.
    again = store.upsert_credential("radient", {**incoming, "org_id": "new-default"})
    assert again.id == new.id


@pytest.mark.asyncio
async def test_unknown_legacy_does_not_shadow_explicit_client_pin(store: AuthStore) -> None:
    store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "legacy-test-only",
            "refresh": "legacy-test-only",
        },
    )
    new = store.upsert_credential(
        "radient", _login("account-a", endpoint=DEFAULT_RADIENT_API_BASE_URL)
    )
    client = TeamBridgeClient(
        store=store, base_url=DEFAULT_RADIENT_API_BASE_URL, credential_id=new.id
    )
    access = await client.current_access()
    assert access.credential_id == new.id
    assert access.account_id == "account-a"
    assert access.api_endpoint == DEFAULT_RADIENT_API_BASE_URL


@pytest.mark.asyncio
async def test_default_post_login_access_is_not_shadowed_by_unknown_legacy(
    store: AuthStore,
) -> None:
    legacy = store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "legacy-test-only",
            "refresh": "legacy-test-only",
            "expires": 9999999999999,
        },
    )
    new = store.upsert_credential(
        "radient", _login("account-a", endpoint=DEFAULT_RADIENT_API_BASE_URL)
    )
    default = await store.get_oauth_access("radient")
    assert default is not None and default.credential_id == new.id
    client = TeamBridgeClient(store=store, base_url=DEFAULT_RADIENT_API_BASE_URL)
    assert (await client.current_access()).credential_id == new.id
    preserved = _stored(store, legacy.id)
    assert not preserved.data.get("account_id")
    assert preserved.identity_key == legacy.identity_key
    assert preserved.data == legacy.data
    assert preserved.disabled_cause == "identity_unattributed"
    assert len(store.list_credentials("radient", include_disabled=True)) == 2


def test_account_namespace_and_row_ids_survive_reopen(store: AuthStore) -> None:
    first = store.upsert_credential("radient", _login("account-a"))
    second = store.upsert_credential("radient", _login("account-b"))
    path = store.db_path
    store.close()
    reopened = AuthStore(db_path=path, config_dir=path.parent)
    try:
        assert _stored(reopened, first.id).identity_key == first.identity_key
        assert _stored(reopened, second.id).data["account_id"] == "account-b"
        again = reopened.upsert_credential("radient", _login("account-a", org="new-default"))
        assert again.id == first.id
        assert _stored(reopened, second.id).data["account_id"] == "account-b"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_exact_row_refresh_recovers_presend_failure_without_account_or_hub_swap(
    store: AuthStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = store.upsert_credential("radient", {**_login("account-a"), "expires": 0})
    other = store.upsert_credential("radient", _login("account-b"))
    calls = []

    async def refresh(data: dict[str, Any]) -> dict[str, Any]:
        calls.append((data["account_id"], data["api_endpoint"]))
        if len(calls) == 1:
            raise httpx.ConnectError("synthetic pre-send failure")
        return {
            "access": "refreshed-test-only",
            "refresh": "rotated-test-only",
            "expires": 9999999999999,
        }

    monkeypatch.setattr(store, "_refresh_fn", lambda provider: refresh)
    with pytest.raises(AuthStoreError):
        await store.ensure_oauth_fresh_or_raise(first.id)
    captured = await store.ensure_oauth_fresh_or_raise(first.id)
    assert captured is not None
    assert calls == [("account-a", "https://hub.example")] * 2
    assert captured["account_id"] == "account-a"
    assert captured["api_endpoint"] == "https://hub.example"
    assert _stored(store, other.id).data["account_id"] == "account-b"


def test_other_provider_dedupe_is_unchanged(store: AuthStore) -> None:
    first = store.upsert_credential("anthropic", _login("account-a"))
    second = store.upsert_credential("anthropic", _login("account-b"))
    assert first.id == second.id
