"""AuthStore tests: cascade order, legacy credentials.env tier, refresh,
rotation, blocking. No network: fakes for the refresh capability."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.credentials import CredentialManager
from local_operator.providers.auth_store import AuthStore, AuthStoreError

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
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


def _oauth(
    refresh: str = "r1", access: str = "access-1", expires: int | None = None
) -> dict[str, Any]:
    import time

    return {
        "refresh": refresh,
        "access": access,
        "expires": expires if expires is not None else int(time.time() * 1000) + 3600_000,
    }


async def test_cascade_runtime_override_wins(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
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


async def test_cascade_login_key_beats_env(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    store.upsert_credential("openai", {"key": "pasted-key", "source": "login", "type": "api_key"})
    assert await store.get_api_key("openai") == "pasted-key"


async def test_cascade_env_beats_plain_stored_key(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    store.upsert_credential("openai", {"key": "stored-key", "type": "api_key"})
    assert await store.get_api_key("openai") == "env-key"


async def test_cascade_stored_key_when_no_env(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
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


async def test_force_refresh_failure_blocks_and_raises(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
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


async def test_a_blocked_row_returns_to_service_when_its_window_passes(
    store: AuthStore, monkeypatch: Any
) -> None:
    """Expiry by TIME, not by an explicit clear. Every other test unblocks via
    ``clear_blocks``, which leaves the actual recovery path — the comparison
    against ``blocked_until_ms`` — unpinned; a regression that wrote
    far-future blocks (the reserve-blocking incident) would have looked
    exactly like this test never existing."""
    clock = {"now": 1_000_000}
    monkeypatch.setattr(AuthStore, "_now_ms", staticmethod(lambda: clock["now"]))
    row = store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    store.block_credential(row.id, "openai", block_ms=60_000)
    assert await store.get_api_key("openai") is None
    clock["now"] += 60_001
    assert not store.is_blocked(row.id, "openai")
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
    """Only TRUE invalidation signals soft-delete (PR-03): an explicit
    revocation marker, never a generic expired/unauthorized 401."""
    store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    from local_operator.providers.failover import ProviderError

    revoked = ProviderError(401, "Your OAuth token has been revoked", auth_error=True)
    store.rotate_sibling("openai", None, revoked, api_key="k1")
    rows = store.list_credentials("openai")
    assert rows == []
    all_rows = store.list_credentials("openai", include_disabled=True)
    assert all_rows[0].disabled_cause == "invalidated-token"


async def test_expired_token_401_is_not_invalidated(store: AuthStore) -> None:
    """An ordinary expired-token 401 must NOT soft-delete — the row stays
    enabled so the a/b/c refresh step (b) can recover it (PR-03)."""
    store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    from local_operator.providers.failover import ProviderError

    expired = ProviderError(401, "invalid_request_error: token expired", auth_error=True)
    store.rotate_sibling("openai", None, expired, api_key="k1")
    # Blocked (backoff) but NOT soft-deleted.
    all_rows = store.list_credentials("openai", include_disabled=True)
    assert all_rows[0].disabled_cause is None
    store.clear_blocks(all_rows[0].id)
    assert await store.get_api_key("openai") == "k1"


async def test_refresh_recovers_expired_token_row_stays_enabled(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: an expired-token 401 goes through refresh; the credential
    row survives enabled and serves the refreshed bearer (PR-03)."""
    import time

    expired = _oauth(expires=int(time.time() * 1000) - 60_000)
    expired["org_id"] = "org-1"
    store.upsert_credential("openai", expired)

    async def fake_refresh(creds: dict[str, Any]) -> dict[str, Any]:
        return {"access": "access-fresh", "expires": int(time.time() * 1000) + 3600_000}

    monkeypatch.setattr(store, "_refresh_fn", lambda provider: fake_refresh)
    key = await store.get_api_key("openai", force_refresh=True)
    assert key == "access-fresh"
    rows = store.list_credentials("openai")
    assert len(rows) == 1 and rows[0].disabled_cause is None


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


async def test_refresh_never_rewrites_org_fields(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-12: a refresh fn that TRIES to clobber org_id/org_name/authorized_at
    cannot — the stored values are restored over the merge."""
    import time

    expired = _oauth(expires=int(time.time() * 1000) + 1000)
    expired.update(org_id="org-original", org_name="Original Org", authorized_at=12345)
    store.upsert_credential("openai", expired)

    async def clobbering_refresh(creds: dict[str, Any]) -> dict[str, Any]:
        return {
            "access": "access-2",
            "expires": int(time.time() * 1000) + 3600_000,
            "org_id": "org-HIJACKED",
            "org_name": "Hijacked Org",
            "authorized_at": 999999,
        }

    monkeypatch.setattr(store, "_refresh_fn", lambda provider: clobbering_refresh)
    key = await store.get_api_key("openai")
    assert key == "access-2"
    row = store.list_credentials("openai")[0]
    assert row.data["org_id"] == "org-original"
    assert row.data["org_name"] == "Original Org"
    assert row.data["authorized_at"] == 12345


async def test_sticky_cleared_when_leaving_oauth_tier(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-16: once resolution leaves the OAuth tier, the session sticky no
    longer attributes the OAuth account — cleared before the env tier runs,
    whatever later tier wins."""
    store.upsert_credential("openai", _oauth())
    await store.get_api_key("openai", session_id="s1")
    oauth_row_id = store._sticky.get(("openai", "s1"))
    assert oauth_row_id is not None

    # Block the only OAuth credential so resolution falls through.
    store.block_credential(oauth_row_id, "openai", block_ms=60_000)
    store.upsert_credential("openai", {"key": "stored", "type": "api_key"})
    key = await store.get_api_key("openai", session_id="s1")
    assert key == "stored"
    # Sticky must NOT still attribute the blocked OAuth row.
    assert store._sticky.get(("openai", "s1")) != oauth_row_id

    # Env tier winning also leaves sticky cleared (the original side effect).
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert await store.get_api_key("openai", session_id="s1") == "env-key"
    assert ("openai", "s1") not in store._sticky


async def test_force_refresh_without_oauth_falls_through(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-15: force_refresh=True with NO oauth credential must reach tiers
    4-7 instead of raising."""
    store.upsert_credential("openai", {"key": "pasted", "source": "login", "type": "api_key"})
    assert await store.get_api_key("openai", force_refresh=True) == "pasted"


async def test_get_oauth_access_oauth_vs_api_key(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-01: get_oauth_access returns the identity-carrying record."""
    creds = _oauth()
    creds.update(org_id="org-1", account_id="acct-1", email="user@example.com")
    store.upsert_credential("openai", creds)
    access = await store.get_oauth_access("openai")
    assert access is not None
    assert access.kind == "oauth"
    assert access.access_token == "access-1"
    assert access.org_id == "org-1" and access.account_id == "acct-1"
    assert access.email == "user@example.com"
    assert access.credential_id > 0

    # Overrides short-circuit to None (gateway-targeted keys carry no identity).
    store.set_runtime_api_key("openai", "cli-key")
    assert await store.get_oauth_access("openai") is None
    store.set_runtime_api_key("openai", None)

    # api_key tier → kind api_key.
    store.upsert_credential("mistral", {"key": "pasted", "source": "login", "type": "api_key"})
    mistral_access = await store.get_oauth_access("mistral")
    assert mistral_access is not None and mistral_access.kind == "api_key"
    assert mistral_access.access_token == "pasted"


def test_db_and_sidecars_created_0600(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PR-11: the DB file AND its WAL sidecars are 0600."""
    import os
    import stat

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-config"))
    db_path = tmp_path / "auth.db"
    auth = AuthStore(db_path=db_path)
    try:
        auth.upsert_credential("openai", {"key": "k1", "type": "api_key"})
        mode = stat.S_IMODE(os.stat(db_path).st_mode)
        assert mode == 0o600
        for sidecar in (db_path.with_name("auth.db-wal"), db_path.with_name("auth.db-shm")):
            if sidecar.exists():
                assert stat.S_IMODE(os.stat(sidecar).st_mode) == 0o600, sidecar.name
    finally:
        auth.close()


class TestListOauthAccesses:
    """Enumeration for REPORTING, which is not the same question as routing.

    The cascade answers "which account does the next request run as" and can
    only ever name one. Quota is per account, so a usage screen needs all of
    them — and needs the ones the cascade has deliberately taken out of service
    most of all.
    """

    @staticmethod
    def _account(email: str, account_id: str) -> dict[str, Any]:
        return {**_oauth(access=f"access-{account_id}"), "email": email, "account_id": account_id}

    async def test_every_account_is_returned_in_id_order(self, store: AuthStore) -> None:
        store.upsert_credential("anthropic", self._account("a@example.com", "acct-a"))
        store.upsert_credential("anthropic", self._account("b@example.com", "acct-b"))
        accesses = await store.list_oauth_accesses("anthropic")
        assert [a.email for a in accesses] == ["a@example.com", "b@example.com"]
        assert [a.access_token for a in accesses] == ["access-acct-a", "access-acct-b"]

    async def test_order_does_not_depend_on_prior_requests(self, store: AuthStore) -> None:
        """The cascade round-robins with no session id; enumeration must not.

        Two `get_api_key` calls advance that rotation, and if enumeration
        shared it the account list would reshuffle under the reader between
        one refresh and the next.
        """
        store.upsert_credential("anthropic", self._account("a@example.com", "acct-a"))
        store.upsert_credential("anthropic", self._account("b@example.com", "acct-b"))
        first = [a.email for a in await store.list_oauth_accesses("anthropic")]
        await store.get_api_key("anthropic")
        await store.get_api_key("anthropic")
        second = [a.email for a in await store.list_oauth_accesses("anthropic")]
        assert first == second == ["a@example.com", "b@example.com"]

    async def test_a_blocked_account_is_still_reported(self, store: AuthStore) -> None:
        """The whole point: an account is usually blocked because it ran out.

        Filtering blocked rows is right for routing and exactly wrong here —
        it guarantees the exhausted account is the one missing from the screen
        that exists to show exhaustion.
        """
        first = store.upsert_credential("anthropic", self._account("a@example.com", "acct-a"))
        store.upsert_credential("anthropic", self._account("b@example.com", "acct-b"))
        store.block_credential(first.id, "anthropic", block_ms=3600_000)

        assert store.is_blocked(first.id, "anthropic") is True
        # Routing has rotated away from it...
        assert await store.get_api_key("anthropic") == "access-acct-b"
        # ...and reporting still sees it.
        assert [a.email for a in await store.list_oauth_accesses("anthropic")] == [
            "a@example.com",
            "b@example.com",
        ]

    async def test_enumeration_does_not_pin_a_sessions_account(self, store: AuthStore) -> None:
        """Reading quota must not repoint which credential a session transacts on."""
        store.upsert_credential("anthropic", self._account("a@example.com", "acct-a"))
        store.upsert_credential("anthropic", self._account("b@example.com", "acct-b"))
        before = dict(store._sticky)
        await store.list_oauth_accesses("anthropic")
        assert dict(store._sticky) == before

    async def test_a_logged_out_account_is_not_reported(self, store: AuthStore) -> None:
        """Blocked is temporary and worth showing; signed out is not ours to show."""
        store.upsert_credential("anthropic", self._account("a@example.com", "acct-a"))
        store.upsert_credential("anthropic", self._account("b@example.com", "acct-b"))
        store.delete_credentials_for_provider("anthropic")
        assert await store.list_oauth_accesses("anthropic") == []

    async def test_an_unrefreshable_account_is_omitted_without_blocking_it(
        self, store: AuthStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read may drop a row from its own output; it may not retire it."""
        row = store.upsert_credential("anthropic", self._account("a@example.com", "acct-a"))
        store.upsert_credential("anthropic", self._account("b@example.com", "acct-b"))

        async def explode(self_, credential_row, *, force=False):  # noqa: ANN001
            if credential_row.id == row.id:
                raise AuthStoreError("refresh failed")
            return dict(credential_row.data)

        monkeypatch.setattr(AuthStore, "_ensure_oauth_fresh", explode)
        accesses = await store.list_oauth_accesses("anthropic")
        assert [a.email for a in accesses] == ["b@example.com"]
        assert store.is_blocked(row.id, "anthropic") is False


class TestProviderOutageDoesNotBlockHealthyAccounts:
    """A 529 is the provider failing, not the credential.

    Blocking on a provider-side fault walks the whole pool into the blocked
    state during an outage -- every account unusable for a minute because the
    provider had a bad second -- which is how a session with four Anthropic
    accounts ran out of credentials to try.
    """

    @staticmethod
    def _store(tmp_path: Any, count: int = 3) -> tuple[Any, list[Any]]:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        # Distinct emails: the real pool is several ACCOUNTS, and rows without
        # a distinct identity dedupe onto one row by design.
        rows = [
            store.upsert_credential(
                "anthropic",
                {
                    "type": "oauth",
                    "access": f"tok-{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"damian+{i}@example.com",
                },
            )
            for i in range(count)
        ]
        return store, rows

    def test_a_529_deprioritizes_rather_than_blocks(self, tmp_path: Any) -> None:
        from local_operator.providers.failover import ProviderError

        store, rows = self._store(tmp_path)
        overloaded = ProviderError(529, "overloaded_error: Overloaded", retryable=True)

        assert store.rotate_sibling("anthropic", "s1", overloaded, api_key="tok-0") is True
        # Still usable: the account did nothing wrong.
        assert store.is_blocked(rows[0].id, "anthropic") is False
        # But it is no longer first choice, so the next attempt moves on.
        order = store._selection_order(store.list_credentials("anthropic"), "anthropic", None)
        assert order[-1].id == rows[0].id

    def test_a_quota_error_still_blocks_the_account_that_ran_out(self, tmp_path: Any) -> None:
        """The distinction is the point: a spent window IS about the credential."""
        from local_operator.providers.failover import ProviderError

        store, rows = self._store(tmp_path)
        quota = ProviderError(429, "rate_limit_error: usage limit reached", retryable=True)

        store.rotate_sibling("anthropic", "s1", quota, api_key="tok-0")
        assert store.is_blocked(rows[0].id, "anthropic") is True

    def test_an_outage_across_the_whole_pool_leaves_every_account_usable(
        self, tmp_path: Any
    ) -> None:
        """The regression, stated directly: after a 529 on each account in turn,
        the pool must not be empty."""
        from local_operator.providers.failover import ProviderError

        store, rows = self._store(tmp_path)
        overloaded = ProviderError(529, "overloaded_error: Overloaded", retryable=True)

        for row in rows:
            store.rotate_sibling("anthropic", "s1", overloaded, api_key=row.data["access"])

        assert [store.is_blocked(row.id, "anthropic") for row in rows] == [False] * len(rows)
        # And selection still offers all of them once the pool has been walked.
        order = store._selection_order(store.list_credentials("anthropic"), "anthropic", None)
        assert len(order) == len(rows)


class TestAggregatorKeysNeverSatisfyANamedProvider:
    """An aggregator credential must not silently answer for a direct provider.

    A session diagnosed its own `No API key configured for provider 'openai'`
    as "auth is via RADIENT_API_KEY, so the openai fallback has no key". The
    first half was right and the second was a guess: the resolution cascade is
    strictly per-provider, and this pins that so a future convenience shortcut
    ("fall back to whatever gateway key we have") cannot be added by accident.
    Spending an aggregator's credit for a provider the user named, and routing
    through two hops instead of one, are both silent when they happen.
    """

    async def test_a_radient_key_answers_only_for_radient(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator.providers.auth_store import AuthStore

        monkeypatch.setenv("RADIENT_API_KEY", "radient-secret")
        store = AuthStore(db_path=tmp_path / "auth.db")

        assert await store.get_api_key("radient") == "radient-secret"
        for named in ("openai", "anthropic", "kimi", "zai"):
            assert await store.get_api_key(named) is None, named

    async def test_an_aggregator_is_reached_only_when_named(self) -> None:
        """Aggregators are opt-in by SELECTION, not by credential availability:
        they resell the same models, so nothing may route to one implicitly."""
        from local_operator.providers.registry import AGGREGATOR_PROVIDERS

        # The set exists and is exactly the resellers -- a direct provider
        # appearing here would make it reachable as an implicit substitute.
        assert AGGREGATOR_PROVIDERS == {"openrouter", "radient"}


class TestDeprioritizationSurvivesTheOrderingItIsAppliedTo:
    """Demotion must not be undone by the ordering that follows it.

    Round 1 review (R1) caught this shipping green: the mark was applied BEFORE
    the sticky/hash/round-robin step, and both of those rotate the list
    (``rows[i:] + rows[:i]``), so a row moved to the back was rotated forward
    again. The tests then in place only asserted ``session_id=None`` on a
    four-row pool -- the one shape whose rotation happened to be a no-op.
    """

    @staticmethod
    def _store(tmp_path: Any, count: int) -> tuple[Any, list[Any]]:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        rows = [
            store.upsert_credential(
                "anthropic",
                {
                    "type": "oauth",
                    "access": f"k{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"a{i}@example.com",
                },
            )
            for i in range(count)
        ]
        return store, rows

    @pytest.mark.parametrize("pool_size", [2, 3, 4, 5])
    @pytest.mark.parametrize("session_id", [None, "s1", "s2", "s3", "session-abc"])
    def test_a_demoted_row_is_last_for_every_pool_size_and_session(
        self, tmp_path: Any, pool_size: int, session_id: str | None
    ) -> None:
        """Swept, because the bug hid in the arithmetic of ONE size/session pair."""
        store, rows = self._store(tmp_path, pool_size)
        store.deprioritize_credential("anthropic", rows[0].id)

        order = store._selection_order(store.list_credentials("anthropic"), "anthropic", session_id)

        assert order[-1].id == rows[0].id, [r.data["access"] for r in order]
        assert len(order) == pool_size  # still selectable, never dropped

    def test_a_demoted_row_outranked_by_nothing_is_still_offered(self, tmp_path: Any) -> None:
        """A single-credential pool must still return that credential."""
        store, rows = self._store(tmp_path, 1)
        store.deprioritize_credential("anthropic", rows[0].id)

        order = store._selection_order(store.list_credentials("anthropic"), "anthropic", "s1")
        assert [r.id for r in order] == [rows[0].id]

    def test_clearing_one_tiers_marks_leaves_another_tiers_alone(self, tmp_path: Any) -> None:
        """R3: the cascade calls `_selection_order` once per credential TIER with
        a different subset, so a one-row tier finding its only row demoted must
        not wipe marks belonging to rows it never saw."""
        store, rows = self._store(tmp_path, 3)
        store.deprioritize_credential("anthropic", rows[0].id)
        store.deprioritize_credential("anthropic", rows[1].id)

        # A "tier" containing only the first row: every row in it is demoted, so
        # its marks are treated as stale and cleared.
        store._selection_order([rows[0]], "anthropic", "s1")

        assert rows[0].id not in store._deprioritized.get("anthropic", set())
        assert rows[1].id in store._deprioritized.get("anthropic", set())


class TestDemotionLifecycle:
    """A demotion mark must end -- on success, and never from a read."""

    @staticmethod
    def _store(tmp_path: Any) -> tuple[Any, list[Any]]:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        rows = [
            store.upsert_credential(
                "anthropic",
                {
                    "type": "oauth",
                    "access": f"k{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"a{i}@example.com",
                },
            )
            for i in range(2)
        ]
        return store, rows

    def test_an_isolated_read_never_clears_a_mark(self, tmp_path: Any) -> None:
        """`read_only` is the isolated request's contract: it resolves without
        making any routing decision, and clearing a demotion is one. A
        decorative call running beside a turn must not move that turn's pool."""
        store, rows = self._store(tmp_path)
        for row in rows:
            store.deprioritize_credential("anthropic", row.id)

        # Every row demoted is the branch that clears; under read_only it must not.
        store._selection_order(rows, "anthropic", "s1", read_only=True)
        assert set(store._deprioritized.get("anthropic", {})) == {rows[0].id, rows[1].id}

        store._selection_order(rows, "anthropic", "s1")
        assert not store._deprioritized.get("anthropic")

    async def test_a_credential_that_serves_a_request_regains_its_priority(
        self, tmp_path: Any
    ) -> None:
        """Otherwise the mark outlives the outage for the life of the process,
        leaving a healthy account permanently last."""
        from local_operator.harness.types import ChatRequest, ModelSpec
        from local_operator.providers.failover import stream_with_failover

        store, rows = self._store(tmp_path)
        store.deprioritize_credential("anthropic", rows[0].id)

        class _Ok:
            async def stream(self, request: Any, key: Any, oauth_access: Any = None) -> Any:
                return
                yield

        async def client_for(spec: Any) -> Any:
            return _Ok()

        request = ChatRequest(
            model=ModelSpec(provider="anthropic", model_id="claude-opus-5"), messages=[]
        )
        async for _ in stream_with_failover(
            request, store, {"retry": {"enabled": True}}, client_for, session_id="s1"
        ):
            pass

        # The row that served the request is no longer demoted.
        assert rows[1].id not in store._deprioritized.get("anthropic", set())


class TestADemotionEndsOnItsOwn:
    """R18: clearing on success cannot be the only exit.

    A demoted credential sorts LAST, so it is not selected, so it never earns
    the success that would clear it. Without an expiry a single 529 left an
    account at the back of the pool for the life of the process -- the same
    "healthy account out of rotation" outcome the demotion exists to prevent.
    """

    def test_a_mark_expires_after_its_ttl(self, tmp_path: Any) -> None:
        from local_operator.providers import auth_store as auth_store_mod
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        rows = [
            store.upsert_credential(
                "anthropic",
                {
                    "type": "oauth",
                    "access": f"k{i}",
                    "refresh": "r",
                    "expires": None,
                    "email": f"a{i}@example.com",
                },
            )
            for i in range(2)
        ]
        store.deprioritize_credential("anthropic", rows[0].id)
        assert store._active_demotions("anthropic") == {rows[0].id}

        # Travel past the TTL rather than sleeping through it.
        real_now = store._now_ms()
        store._now_ms = staticmethod(  # type: ignore[method-assign]
            lambda: real_now + auth_store_mod.DEPRIORITIZE_TTL_MS + 1
        )

        assert store._active_demotions("anthropic") == set()
        order = store._selection_order(store.list_credentials("anthropic"), "anthropic", "s1")
        assert [r.id for r in order] == [r.id for r in store.list_credentials("anthropic")]


class TestARefreshlessOAuthCredentialIsReadable:
    """R21: a credential that is OAuth-issued but never expires.

    `upsert_credential` guessed the type from "has both refresh and access",
    which cannot see a credential with no refresh token because it does not
    expire -- Z.AI's coding-plan sign-in mints exactly that. Such a row landed
    as `api_key` with its secret under `data["access"]`, where NOTHING can read
    it: tiers 4 and 6 read `data["key"]`, tier 3 only walks `oauth` rows. The
    login reported success and every later request failed with no credential,
    behind this PR's new "temporarily unavailable" message.
    """

    async def test_an_explicit_type_survives_the_structural_guess(self, tmp_path: Any) -> None:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        row = store.upsert_credential(
            "zai",
            {
                "type": "oauth",
                "access": "key-id.sec-ret",
                "expires": None,  # minted key: never expires, so no refresh exists
                "email": "damian@example.com",
                "account_id": "42",
            },
        )

        assert row.credential_type == "oauth"
        assert await store.get_api_key("zai") == "key-id.sec-ret"
        access = await store.get_oauth_access("zai")
        assert access is not None and access.kind == "oauth"
        assert access.email == "damian@example.com"

    async def test_an_undeclared_credential_still_gets_the_old_guess(self, tmp_path: Any) -> None:
        """The structural fallback is unchanged for every existing caller."""
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        oauth = store.upsert_credential("openai", {"access": "a", "refresh": "r", "expires": 1})
        pasted = store.upsert_credential("openai", {"key": "sk-1", "source": "login"})

        assert oauth.credential_type == "oauth"
        assert pasted.credential_type == "api_key"

    async def test_signing_in_repeatedly_keeps_one_row(self, tmp_path: Any) -> None:
        """Z1: a refreshless OAuth credential must dedupe like every other one.

        `_identity_key_for`'s per-provider constant asked "is this OAuth?" by
        testing for a refresh token -- the same blind spot the type derivation
        had. A credential whose token never expires carries none, so it fell
        through to `None` and each re-login left another row. Five sign-ins
        meant five rows, and `/usage` rendered one account five times.

        Z.AI's token response makes the `user` block optional, so the identity
        fields cannot be relied on to cover this.
        """
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        # No email/account_id: the shape Z.AI returns without a `user` block.
        payload = {"type": "oauth", "access": "key-id.sec-ret", "expires": None}
        for attempt in range(5):
            store.upsert_credential("zai", dict(payload, authorized_at=attempt))

        rows = store.list_credentials("zai")
        assert len(rows) == 1, [r.id for r in rows]
        assert await store.get_api_key("zai") == "key-id.sec-ret"

    async def test_a_pasted_key_still_gets_its_own_row(self, tmp_path: Any) -> None:
        """The control: deduping OAuth must not start deduping API keys.

        Each pasted key is a distinct credential and has always earned its own
        row -- that is what the `source == "login"` guard at the top of
        `_identity_key_for` is for. Without this, the fix above would read as
        correct while silently collapsing a user's key pool to one row.
        """
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        store.upsert_credential("zai", {"key": "sk-1", "source": "login", "type": "api_key"})
        store.upsert_credential("zai", {"key": "sk-2", "source": "login", "type": "api_key"})

        assert len(store.list_credentials("zai")) == 2


class TestADemotedRowDoesNotHoldItsTier:
    """R34: demotion must move a turn on, even across cascade TIERS.

    The cascade is a sequence of tiers consulted whole -- an OAuth row, or an
    api_key row with `source="login"`, wins before a later tier is looked at.
    Sorting a demoted row last therefore did nothing when it was ALONE in its
    tier: it kept winning the cascade, while `rotate_sibling` kept reporting a
    sibling existed, so the driver believed rotation was progressing while the
    same failing bearer came back every time. A healthy credential one tier down
    never received a single request -- and this is exactly the shape `zai-oauth`
    creates beside a pasted key.
    """

    @staticmethod
    def _store(tmp_path: Any) -> Any:
        from local_operator.providers.auth_store import AuthStore

        return AuthStore(db_path=tmp_path / "auth.db")

    async def test_a_sibling_in_another_tier_is_reachable(self, tmp_path: Any) -> None:
        store = self._store(tmp_path)
        login_row = store.upsert_credential(
            "anthropic", {"type": "api_key", "key": "login-key", "source": "login"}
        )
        store.upsert_credential("anthropic", {"type": "api_key", "key": "migrated-key"})

        # The tier-4 row fails on a PROVIDER fault, so it is demoted, not blocked.
        store.deprioritize_credential("anthropic", login_row.id)

        # The cascade must now reach the tier-6 row rather than re-serving the
        # demoted one from the tier above.
        assert await store.get_api_key("anthropic") == "migrated-key"

    async def test_the_last_credential_is_still_served_when_all_are_demoted(
        self, tmp_path: Any
    ) -> None:
        """Dropping is conditional: with nothing else reachable the marks
        describe an outage rather than an account, and must not empty the pool."""
        store = self._store(tmp_path)
        row = store.upsert_credential(
            "anthropic", {"type": "api_key", "key": "only-key", "source": "login"}
        )
        store.deprioritize_credential("anthropic", row.id)

        assert await store.get_api_key("anthropic") == "only-key"


class TestADemotionSurvivesAMixedPool:
    """R36/R37: the two ways a demotion stopped moving a turn on.

    Both were edges of the same idea -- "is anything else reachable?" answered
    by looking at the credential TABLE, which is the question this PR has
    already been wrong about five times on the failover side.
    """

    async def test_a_row_with_no_same_type_sibling_stays_demoted(self, tmp_path: Any) -> None:
        """R36: `rotate_sibling`'s sibling list is filtered by credential_type,
        so an OAuth row beside a pasted key has no same-type sibling. Clearing
        the mark there erased it in the very call that set it, and tier 3
        re-served the identical failing row forever -- the exact shape a Z.AI
        sign-in creates beside an API key."""
        from local_operator.providers.auth_store import AuthStore
        from local_operator.providers.failover import ProviderError

        store = AuthStore(db_path=tmp_path / "auth.db")
        oauth_row = store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "oauth-down", "refresh": "r", "expires": None},
        )
        store.upsert_credential(
            "anthropic", {"type": "api_key", "key": "apikey-healthy", "source": "login"}
        )

        store.rotate_sibling(
            "anthropic",
            "s1",
            ProviderError(500, "permanent", retryable=True),
            api_key="oauth-down",
        )

        assert oauth_row.id in store._active_demotions("anthropic")
        # And the cascade now reaches the healthy credential in the next tier.
        assert await store.get_api_key("anthropic") == "apikey-healthy"

    async def test_a_demoted_lone_row_still_yields_to_the_env_var(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R37: the cascade also resolves from the env var (tier 5) and the
        fallback resolver (tier 7), which are not rows -- so a guard that
        counted rows to decide whether to yield made an exported key
        unreachable where it used to be the fallback."""
        from local_operator.providers.auth_store import AuthStore

        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-fallback-key")
        store = AuthStore(db_path=tmp_path / "auth.db")
        row = store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "oauth-down", "refresh": "r", "expires": None},
        )

        assert await store.get_api_key("anthropic") == "oauth-down"
        store.deprioritize_credential("anthropic", row.id)
        assert await store.get_api_key("anthropic") == "env-fallback-key"

    async def test_an_isolated_resolve_is_not_handed_only_the_destructive_half(
        self, tmp_path: Any
    ) -> None:
        """A ``read_only`` resolve must serve a demoted lone row.

        Demotion has a destructive half (``_usable_key_rows`` drops the row from
        its tier) and a restorative half (the second pass that resolves again
        once demotions are the only thing left standing in the way). The drop
        carries no ``read_only`` gate, so gating only the restorative half gave
        the isolated caller the destructive one alone: a lone credential with a
        single provider-side fault resolved to ``None`` for it while the normal
        path still returned the key. That reports "no credential configured" for
        a credential that is merely deprioritised -- the misdiagnosis this whole
        change set exists to remove -- and it silently stopped session titling
        for the mark's full TTL.
        """
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        row = store.upsert_credential(
            "anthropic", {"type": "api_key", "key": "only-key", "source": "login"}
        )
        store.deprioritize_credential("anthropic", row.id)

        assert await store.get_api_key("anthropic", read_only=True) == "only-key"

    async def test_an_isolated_resolve_serves_a_demoted_row_without_clearing_it(
        self, tmp_path: Any
    ) -> None:
        """...and it gets there WITHOUT taking the routing decision.

        Clearing the marks is what the user's own turn does when it finds
        nowhere left to route; an isolated request running beside that turn must
        not be able to move it. So the second pass suppresses the clear under
        ``read_only`` and re-resolves ignoring the marks instead: same answer,
        no state touched. A demoted row that still has a healthy sibling must
        also keep losing to it, or "ignore the marks" would become "resurrect
        the failing account".
        """
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        row = store.upsert_credential(
            "anthropic", {"type": "api_key", "key": "only-key", "source": "login"}
        )
        store.deprioritize_credential("anthropic", row.id)

        assert await store.get_api_key("anthropic", read_only=True) == "only-key"
        assert store._active_demotions("anthropic") == {row.id}

        # The normal path still owns the decision, and still clears.
        assert await store.get_api_key("anthropic") == "only-key"
        assert store._active_demotions("anthropic") == set()

        # A healthy sibling still outranks a demoted row under read_only.
        sibling_store = AuthStore(db_path=tmp_path / "sibling.db")
        bad = sibling_store.upsert_credential(
            "anthropic", {"type": "api_key", "key": "bad-key", "source": "login"}
        )
        sibling_store.upsert_credential(
            "anthropic", {"type": "api_key", "key": "good-key", "source": "login"}
        )
        sibling_store.deprioritize_credential("anthropic", bad.id)

        assert await sibling_store.get_api_key("anthropic", read_only=True) == "good-key"

    async def test_the_second_pass_leaves_other_providers_marks_alone(self, tmp_path: Any) -> None:
        """The second pass is provider-keyed.

        It runs at the end of a cascade for ONE provider, so it may only speak
        for that provider's marks: a concurrent turn on another provider is
        mid-rotation and its demotions are load-bearing.
        """
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")
        anthropic_row = store.upsert_credential(
            "anthropic", {"type": "api_key", "key": "anthropic-key", "source": "login"}
        )
        openai_row = store.upsert_credential(
            "openai", {"type": "api_key", "key": "openai-key", "source": "login"}
        )
        store.deprioritize_credential("anthropic", anthropic_row.id)
        store.deprioritize_credential("openai", openai_row.id)

        assert await store.get_api_key("anthropic") == "anthropic-key"
        assert store._active_demotions("openai") == {openai_row.id}

    async def test_a_provider_with_no_credential_still_resolves_to_none(
        self, tmp_path: Any
    ) -> None:
        """The second pass must not invent a credential.

        It only fires when marks exist, so an unconfigured provider (and a
        blocked one, which the marks do not describe) still comes back ``None``
        rather than being papered over by the retry.
        """
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(db_path=tmp_path / "auth.db")

        assert await store.get_api_key("anthropic") is None
        assert await store.get_api_key("anthropic", read_only=True) is None
