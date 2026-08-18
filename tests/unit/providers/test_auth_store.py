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
