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


async def test_block_credential_caps_at_max(store: AuthStore, monkeypatch: Any) -> None:
    """A multi-day block request is clamped to MAX_CREDENTIAL_BLOCK_MS.

    A genuinely full-week-depleted account keys its block to the weekly
    reset (days out); the cap guarantees no reading -- a usage reset estimate
    or a hostile Retry-After -- strands an account past the point where
    re-probing is cheaper than waiting. This is the single choke point the cap
    lives at, so it protects every caller (preflight AND rotate_sibling)."""
    from local_operator.providers.auth_store import MAX_CREDENTIAL_BLOCK_MS

    clock = {"now": 1_000_000}
    monkeypatch.setattr(AuthStore, "_now_ms", staticmethod(lambda: clock["now"]))
    row = store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    # A seven-day block request -- the shape a raw weekly usage reset writes.
    store.block_credential(row.id, "openai", block_ms=7 * 86_400_000)
    stored = store._conn.execute(
        "SELECT blocked_until_ms FROM auth_credential_blocks WHERE credential_id = ?",
        (row.id,),
    ).fetchone()
    assert stored is not None
    # Capped: the stored horizon is at most one hour out, not seven days.
    assert stored[0] - clock["now"] == MAX_CREDENTIAL_BLOCK_MS


async def test_block_credential_floor_still_applies(store: AuthStore, monkeypatch: Any) -> None:
    """A 0/negative block still floors at 1000 ms (the cap did not remove it)."""
    clock = {"now": 1_000_000}
    monkeypatch.setattr(AuthStore, "_now_ms", staticmethod(lambda: clock["now"]))
    row = store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    store.block_credential(row.id, "openai", block_ms=0)
    stored = store._conn.execute(
        "SELECT blocked_until_ms FROM auth_credential_blocks WHERE credential_id = ?",
        (row.id,),
    ).fetchone()
    assert stored is not None
    assert stored[0] - clock["now"] == 1000


async def test_rotate_sibling_retry_after_is_capped(store: AuthStore, monkeypatch: Any) -> None:
    """The reactive rotation path inherits the cap for free.

    ``rotate_sibling`` writes ``block_ms=max(block_ms, retry_after or 0)``, so
    a provider that answers a quota 429 with ``Retry-After: 604800`` (a week)
    would, uncapped, strand the account for a week. The cap lives in
    ``block_credential`` -- the single choke point -- so this site needs no
    inline clamp and still cannot write a multi-day block."""
    from local_operator.providers.auth_store import MAX_CREDENTIAL_BLOCK_MS
    from local_operator.providers.failover import ProviderError

    clock = {"now": 1_000_000}
    monkeypatch.setattr(AuthStore, "_now_ms", staticmethod(lambda: clock["now"]))
    row = store.upsert_credential("openai", {"key": "k1", "type": "api_key"})
    store.upsert_credential("openai", {"key": "k2", "type": "api_key"})
    # A hostile week-long Retry-After on a usage-limit 429.
    hostile = ProviderError(429, "rate limit reached", retryable=True, retry_after_ms=604_800_000)
    store.rotate_sibling("openai", None, hostile, api_key="k1")
    stored = store._conn.execute(
        "SELECT blocked_until_ms FROM auth_credential_blocks WHERE credential_id = ?",
        (row.id,),
    ).fetchone()
    assert stored is not None
    assert stored[0] - clock["now"] == MAX_CREDENTIAL_BLOCK_MS


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


async def test_rotate_sibling_finds_a_token_plan_row_by_its_wire_key(
    store: AuthStore,
) -> None:
    """The failing bearer is the EXTRACTOR's output, not ``data["access"]``.

    A QwenCloud row holds the management token in ``access`` and the ``sk-sp-…``
    inference key in ``api_key``; the cascade authenticates with the latter, so
    the key failover reports on failure is the latter. Matching the raw field
    found no row and the failing credential was never blocked, demoted or
    unstuck — the failover layer could not rotate away from it."""
    row = store.upsert_credential(
        "alibaba-token-plan",
        {"type": "oauth", "access": "mgmt-token", "api_key": "sk-sp-wire"},
    )
    from local_operator.providers.failover import ProviderError

    error = ProviderError(401, "invalid api key", auth_error=True)
    # The management token is NOT the wire key: reporting it rotates nothing —
    # no row matches, so the lone row is still its own untried sibling (True)
    # and nothing is blocked.
    assert store.rotate_sibling("alibaba-token-plan", None, error, api_key="mgmt-token") is True
    assert store.is_blocked(row.id, "alibaba-token-plan") is False
    # Asked under the flavour id, with the WIRE key the request actually
    # carried: the row is found and blocked, and no sibling remains (False).
    assert (
        store.rotate_sibling("alibaba-token-plan-oauth", None, error, api_key="sk-sp-wire") is False
    )
    assert store.is_blocked(row.id, "alibaba-token-plan-oauth") is True


async def test_rotate_sibling_survives_a_malformed_oauth_row(store: AuthStore) -> None:
    """A hand-written row with neither ``access`` nor ``api_key`` must not turn
    the failure path into a crash: the extractor raises KeyError on it, and
    ``_row_matches_key`` has to swallow that and report "no match" so failover
    still rotates. The healthy sibling must still be found and served."""
    bad = store.upsert_credential("alibaba-token-plan", {"type": "oauth"})
    failing = store.upsert_credential("alibaba-token-plan", {"key": "k-good", "type": "api_key"})
    sibling = store.upsert_credential("alibaba-token-plan", {"key": "k-other", "type": "api_key"})
    from local_operator.providers.failover import ProviderError

    error = ProviderError(401, "invalid api key", auth_error=True)
    # The bad row is walked FIRST and must read as "not the failing key", not
    # raise; the real failing row is blocked and the sibling reported.
    assert store.rotate_sibling("alibaba-token-plan", None, error, api_key="k-good") is True
    assert store.is_blocked(failing.id, "alibaba-token-plan") is True
    assert store.is_blocked(bad.id, "alibaba-token-plan") is False
    assert await store.get_api_key("alibaba-token-plan") == "k-other"
    _ = sibling


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

    async def test_a_refresh_failed_account_is_still_named_for_usage(
        self, store: AuthStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reporting must still list the login even when no bearer can be minted.

        ``list_oauth_accesses`` omits the row (and must not block it). The
        sibling enumerator is what ``/usage`` uses to keep the identity on
        the panel as last-known / unavailable.
        """
        row = store.upsert_credential("anthropic", self._account("a@example.com", "acct-a"))
        store.upsert_credential("anthropic", self._account("b@example.com", "acct-b"))

        async def explode(self_, credential_row, *, force=False):  # noqa: ANN001
            if credential_row.id == row.id:
                raise AuthStoreError("refresh failed")
            return dict(credential_row.data)

        monkeypatch.setattr(AuthStore, "_ensure_oauth_fresh", explode)
        named = store.list_oauth_identities("anthropic")
        assert [a.email for a in named] == ["a@example.com", "b@example.com"]
        assert all(a.access_token == "" for a in named)
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


class TestLoginFlavourAliases:
    """A login flavour and its base provider are ONE credential.

    ``xai-oauth``, ``openai-device`` and ``alibaba-token-plan-oauth`` write their
    row under another provider's name (``store_credentials_as``), and the store's
    SQL is exact. Before this, every lookup for the flavour id matched no row and
    the cascade reported "No API key configured for provider 'xai-oauth'" at the
    end of a successful OAuth login -- the one failure an OAuth login exists to
    prevent, and one that told the user to go get an API key they should not need.
    """

    async def test_oauth_login_flavour_resolves_the_row_its_login_wrote(
        self, store: AuthStore
    ) -> None:
        """The reported bug: log in with `/login xai-oauth`, then stream on it."""
        store.upsert_credential("xai", _oauth(access="xai-access"))
        assert await store.get_api_key("xai-oauth") == "xai-access"
        access = await store.get_oauth_access("xai-oauth")
        assert access is not None and access.kind == "oauth"
        assert access.access_token == "xai-access"

    async def test_every_registry_alias_resolves_not_just_xai(self, store: AuthStore) -> None:
        """Fixed for the property, not for the one provider that was reported."""
        for flavour, storage in (
            ("xai-oauth", "xai"),
            ("openai-device", "openai"),
            ("alibaba-token-plan-oauth", "alibaba-token-plan"),
            # The flavour the bug was actually reported for.
            ("zai-oauth", "zai"),
        ):
            store.upsert_credential(storage, _oauth(access=f"{storage}-access"))
            assert await store.get_api_key(flavour) == f"{storage}-access"

    async def test_the_alias_and_its_base_share_one_backoff(self, store: AuthStore) -> None:
        """One credential ⇒ one block: a 429 earned as `xai` is not evaded by
        asking again as `xai-oauth`, which would spend a rate-limited account
        twice and re-trip the limit."""
        row = store.upsert_credential("xai", _oauth(access="xai-access"))
        store.block_credential(row.id, "xai")
        assert store.is_blocked(row.id, "xai-oauth") is True

    async def test_the_alias_and_its_base_share_one_sticky_account(self, store: AuthStore) -> None:
        """Stickiness pins an ACCOUNT, so it cannot depend on the spelling used.

        Two DISTINCT identities: rows sharing one identity key collapse into a
        single row on upsert, and with only one row to choose from every
        selection path returns the same token whether stickiness was honoured or
        not -- a test that cannot fail. ``session-1`` also hashes to index 0, so
        the unsticky fallback would return the FIRST row, making the assertion
        below sensitive to the pin rather than to luck.
        """
        store.upsert_credential(
            "xai", {**_oauth(refresh="r1", access="a1"), "account_id": "acct-a"}
        )
        second = store.upsert_credential(
            "xai", {**_oauth(refresh="r2", access="a2"), "account_id": "acct-b"}
        )
        store._set_sticky("xai-oauth", "session-1", second.id)
        assert await store.get_api_key("xai", "session-1") == "a2"

    async def test_the_base_providers_env_key_authenticates_the_flavour(
        self, store: AuthStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is no XAI_OAUTH_API_KEY: same wire, same endpoint, same key."""
        monkeypatch.setenv("XAI_API_KEY", "env-xai")
        assert await store.get_api_key("xai-oauth") == "env-xai"

    async def test_token_plan_flavour_gets_the_inference_key_not_the_mgmt_token(
        self, store: AuthStore
    ) -> None:
        """QwenCloud's row holds two tokens and only the STORAGE definition knows
        which one the inference endpoint accepts; resolving the flavour by its own
        definition would authenticate with the token that endpoint rejects."""
        store.upsert_credential(
            "alibaba-token-plan",
            {**_oauth(access="mgmt-token"), "api_key": "sk-sp-inference"},
        )
        assert await store.get_api_key("alibaba-token-plan-oauth") == "sk-sp-inference"

    async def test_logout_still_removes_the_row_through_either_name(self, store: AuthStore) -> None:
        """list_credentials now aliases, so the logout path must not double-count
        or miss the row it is asked to delete."""
        store.upsert_credential("xai", _oauth())
        assert store.delete_credentials_for_provider("xai-oauth") == 1
        assert await store.get_api_key("xai-oauth") is None

    async def test_a_row_written_under_the_flavour_id_is_still_findable(
        self, store: AuthStore
    ) -> None:
        """Writes alias too, so no caller can strand a credential.

        The reads above resolve to the storage id; if a write did not, a
        credential saved under ``xai-oauth`` would live at an id no lookup
        visits -- invisible to the cascade, to `/logout` and to the usage view
        at once.
        """
        row = store.upsert_credential("xai-oauth", _oauth(access="flavour-write"))
        assert row.provider == "xai"
        assert await store.get_api_key("xai") == "flavour-write"
        assert await store.get_api_key("xai-oauth") == "flavour-write"


class TestADemotionNeverMovesASessionOffItsStickyAccount:
    """A demotion is a preference for NEW picks, never an eviction.

    The provider's prompt cache is per account. Before this, a reserve
    demotion written by the quota preflight — process-wide and shared by every
    session in the process — dropped the demoted row from its tier, so a
    session sticky to it fell to the hash pick and was re-pinned to a sibling
    whose cache had never seen its conversation: the whole prefix rewritten at
    cache-write price, and with several accounts low, again at the next
    boundary once the 120s mark expired. Measured on a five-account host at
    ~38% of all Anthropic cache writes over 30 hours. The reactive 429 path
    already kept the sticky (``rotate_sibling``: "sticky preserved"); this
    class pins the same rule for the preference marks.
    """

    @staticmethod
    def _pool(tmp_path: Any) -> tuple[AuthStore, Any, Any]:
        store = AuthStore(db_path=tmp_path / "auth.db")
        a = store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "tok-a", "refresh": "r", "email": "a@example.com"},
        )
        b = store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "tok-b", "refresh": "r", "email": "b@example.com"},
        )
        return store, a, b

    async def test_sticky_survives_its_own_demotion_when_a_sibling_is_available(
        self, tmp_path: Any
    ) -> None:
        """(a) The regression: sticky on A, A demoted, B healthy → still A."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)

        store.deprioritize_credential("anthropic", a.id)

        access = await store.get_oauth_access("anthropic", "s1")
        assert access is not None and access.credential_id == a.id
        # The mark is untouched: it still steers OTHER picks (below).
        assert store._active_demotions("anthropic") == {a.id}

    async def test_a_session_without_a_sticky_still_skips_the_demoted_row(
        self, tmp_path: Any
    ) -> None:
        """(b) Unchanged: a fresh session's hash pick never lands on a demoted row."""
        store, a, b = self._pool(tmp_path)
        store.deprioritize_credential("anthropic", a.id)

        # Every session id, whatever its hash, avoids A while B is usable.
        for session in ("s1", "s2", "s3", "session-abc", "another"):
            access = await store.get_oauth_access("anthropic", session)
            assert access is not None and access.credential_id == b.id, session

    async def test_a_blocked_sticky_still_moves(self, tmp_path: Any) -> None:
        """(c) Unchanged: a BLOCK is a verdict that the account cannot serve, and
        the block filter runs ahead of the sticky exemption."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)

        store.block_credential(a.id, "anthropic", block_ms=60_000)

        access = await store.get_oauth_access("anthropic", "s1")
        assert access is not None and access.credential_id == b.id

    async def test_a_blocked_and_demoted_sticky_still_moves(self, tmp_path: Any) -> None:
        """Both marks at once (the depleted-then-demoted shape) still moves."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)
        store.deprioritize_credential("anthropic", a.id)
        store.block_credential(a.id, "anthropic", block_ms=60_000)

        access = await store.get_oauth_access("anthropic", "s1")
        assert access is not None and access.credential_id == b.id

    async def test_the_usage_limit_path_still_preserves_the_sticky(self, tmp_path: Any) -> None:
        """(d) The reactive promise this change extends is itself unchanged."""
        from local_operator.providers.failover import ProviderError

        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)

        usage_limit = ProviderError(429, "rate limit reached", retryable=True)
        assert store.rotate_sibling("anthropic", "s1", usage_limit, api_key="tok-a") is True

        assert store.session_credential_id("anthropic", "s1") == a.id
        # Blocked for the backoff, so the request in hand goes to B; the pin
        # brings the session back to A once the block lapses.
        assert store.is_blocked(a.id, "anthropic")

    async def test_the_exemption_is_per_session(self, tmp_path: Any) -> None:
        """Another session's warm account is not this session's: a demotion
        written by (or for) one session still steers every OTHER session."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "warm", a.id)
        store.deprioritize_credential("anthropic", a.id)

        warm = await store.get_oauth_access("anthropic", "warm")
        cold = await store.get_oauth_access("anthropic", "cold")
        assert warm is not None and warm.credential_id == a.id
        assert cold is not None and cold.credential_id == b.id

    async def test_releasing_the_pin_lets_the_demotion_take_effect(self, tmp_path: Any) -> None:
        """The escape hatch the preflight uses for a FRESH pick: a row the walk
        only just pinned holds nothing cached, so release + demote moves on."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)
        store.deprioritize_credential("anthropic", a.id)
        store.release_session_credential("anthropic", "s1")

        access = await store.get_oauth_access("anthropic", "s1")
        assert access is not None and access.credential_id == b.id
        # ...and the session is now pinned to the account that served.
        assert store.session_credential_id("anthropic", "s1") == b.id

    async def test_a_read_only_resolve_also_stays_on_the_demoted_sticky(
        self, tmp_path: Any
    ) -> None:
        """An isolated request beside the turn lands on the SAME credential the
        turn is transacting on — the point of ``read_only`` reading stickiness."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)
        store.deprioritize_credential("anthropic", a.id)

        access = await store.get_oauth_access("anthropic", "s1", read_only=True)
        assert access is not None and access.credential_id == a.id
        assert store._active_demotions("anthropic") == {a.id}

    def test_the_all_demoted_stale_marks_rule_ignores_the_exemption(self, tmp_path: Any) -> None:
        """ "Every row in the tier is demoted" is judged on the raw marks, so the
        stale-marks clear in ``_selection_order`` fires exactly as before."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)
        store.deprioritize_credential("anthropic", a.id)
        store.deprioritize_credential("anthropic", b.id)

        order = store._selection_order(store.list_credentials("anthropic"), "anthropic", "s1")
        assert [r.id for r in order] == [a.id, b.id]
        assert store._active_demotions("anthropic") == set()

    async def test_an_all_demoted_tier_clears_both_marks_through_resolve(
        self, tmp_path: Any
    ) -> None:
        """Review F3: the same rule seen from ``_resolve``, where the tier
        filter runs first. Sticky on A, A and B both demoted (an outage, not a
        verdict about either account). The exemption used to hand the cascade
        ``[A]`` alone, ``_selection_order`` read that one-row tier as "all
        demoted" and cleared only A's mark — B stayed demoted, and every other
        session's fresh pick then skewed onto A for the rest of the TTL. The
        all-demoted judgement runs on the RAW rows so both marks clear
        together, exactly as they did before the exemption existed, and the
        sticky session still lands on A."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)
        store.deprioritize_credential("anthropic", a.id)
        store.deprioritize_credential("anthropic", b.id)

        access = await store.get_oauth_access("anthropic", "s1")
        assert access is not None and access.credential_id == a.id
        assert store._active_demotions("anthropic") == set()

        # With the marks gone, a non-sticky session spreads by hash again
        # rather than being funnelled onto the sticky's account.
        picks = set()
        for session in ("s2", "s3", "s4", "s5", "s6", "s7"):
            other = await store.get_oauth_access("anthropic", session, read_only=True)
            assert other is not None
            picks.add(other.credential_id)
        assert picks == {a.id, b.id}

    def test_the_sticky_sorts_first_even_while_demoted(self, tmp_path: Any) -> None:
        """The ordering half agrees with the tier-filter half: a demoted sticky
        is not sorted last either, or the two would disagree on the second
        (``ignore_demotions``) pass."""
        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)
        store.deprioritize_credential("anthropic", a.id)

        order = store._selection_order(store.list_credentials("anthropic"), "anthropic", "s1")
        assert [r.id for r in order] == [a.id, b.id]
        # A session with no sticky still sees A last.
        order = store._selection_order(store.list_credentials("anthropic"), "anthropic", "s2")
        assert order[-1].id == a.id

    async def test_a_server_fault_on_the_sticky_still_moves_the_session(
        self, tmp_path: Any
    ) -> None:
        """The exemption must not undo 529 rotation: ``rotate_sibling`` clears
        the session's sticky for a server fault before the mark is consulted,
        so the demoted row has no pin to hide behind and the next attempt
        moves to the sibling as it always did."""
        from local_operator.providers.failover import ProviderError

        store, a, b = self._pool(tmp_path)
        store.pin_session_credential("anthropic", "s1", a.id)

        overloaded = ProviderError(529, "overloaded_error: Overloaded", retryable=True)
        assert store.rotate_sibling("anthropic", "s1", overloaded, api_key="tok-a") is True

        access = await store.get_oauth_access("anthropic", "s1")
        assert access is not None and access.credential_id == b.id


class TestUsageAwareFirstPick:
    """A session's FIRST account pick prefers cached headroom.

    Before this, the pick was ``crc32(session_id) % n`` alone: uniform over
    sessions and blind to how full each account already was. Field data over
    one day showed three of five Anthropic accounts at 65-99% of their 5-hour
    window while two sat at 6% and 29%, and 33 "All 5 OAuth credentials
    unusable" incidents. These tests pin the ranking (``_usage_ranked_order``)
    against a temp usage cache carrying synthetic reports.
    """

    @pytest.fixture(autouse=True)
    def _close_stores(self) -> Iterator[None]:
        """Release every store ``_store`` opened once the test is over.

        Each store holds two sqlite connections (auth + usage cache, the
        latter with WAL sidecars), and a helper that never closed them leaked
        ~3 fds per test. Under macOS's default ``ulimit -n 256`` that made an
        xdist worker running this file flaky with ``Errno 24`` (review round
        2, F6). ``AuthStore.close`` closes the cache it was handed too.
        """
        self._opened: list[AuthStore] = []
        yield
        for store in self._opened:
            store.close()

    def _store(self, tmp_path: Any, count: int = 5) -> "tuple[AuthStore, list[Any], Any]":
        from local_operator.providers.usage_cache import UsageCacheStore

        cache = UsageCacheStore(tmp_path / "usage_cache.db")
        store = AuthStore(db_path=tmp_path / "auth.db", usage_cache=cache)
        self._opened.append(store)
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
        return store, rows, cache

    @staticmethod
    def _cache_report(
        cache: Any,
        row: Any,
        *,
        five_hour: float,
        # Zero by default so the 5-hour figure IS the binding shared window;
        # the ranking takes the worst shared window, and a 7-day figure that
        # happened to bind would silently change which account a case is about.
        seven_day: float = 0.0,
        fable: float | None = None,
        age_ms: int = 0,
        resets_in_ms: int | None = None,
        via: str = "preflight",
    ) -> None:
        """Write one account's report the way the preflight writes it.

        Same key derivation (``account_preflight_key`` over the row's email)
        and same limit shape (shared 5h/7d rows, an optional fable-scoped
        weekly) as ``fetch_anthropic_oauth`` produces, so the test exercises
        the exact read path a live cache would.

        ``via="warmer"`` writes the SAME numbers the way ``/usage`` and the
        TUI's background warm do instead: one per-provider-set row keyed by
        ``provider_cache_key`` whose reports carry ``identity``. That is the
        only shape on disk when ``retry.usageAwareFallback`` is off (the
        shipped default), so the pick must read it too. The row is merged
        with any sibling already there, as the warmer's union write would.
        """
        import time

        from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
        from local_operator.providers.usage_cache import (
            account_preflight_key,
            fingerprint_accounts,
            provider_cache_key,
        )

        limits = [
            UsageLimit(
                id="anthropic:5h",
                label="5 hour",
                amount=UsageAmount(used_fraction=five_hour, unit="percent"),
                shared=True,
            ),
            UsageLimit(
                id="anthropic:7d",
                label="7 day",
                amount=UsageAmount(used_fraction=seven_day, unit="percent"),
                shared=True,
            ),
        ]
        if fable is not None:
            limits.append(
                UsageLimit(
                    id="anthropic:7d:fable",
                    label="7 day (Fable)",
                    amount=UsageAmount(used_fraction=fable, unit="percent"),
                    tier="fable",
                )
            )
        now = int(time.time() * 1000)
        if resets_in_ms is not None:
            for limit in limits:
                limit.resets_at_ms = now + resets_in_ms
        report = UsageReport(provider="anthropic", fetched_at=now - age_ms, limits=limits)
        if via == "warmer":
            report.identity = row.data["email"]
            key = provider_cache_key("anthropic", fingerprint_accounts(["the-whole-set"]))
            siblings = [
                r
                for r in (cache.get(key, include_expired=True) or [])
                if r.identity != report.identity
            ]
            cache.set(key, "anthropic", [*siblings, report], expires_at_ms=now + 60_000)
            return
        key = account_preflight_key("anthropic", row.data["email"])
        cache.set(key, "anthropic", [report], expires_at_ms=now + 60_000)

    @staticmethod
    def _order(store: AuthStore, session_id: str, model_id: str = "claude-opus-5") -> list[int]:
        rows = store.list_credentials("anthropic")
        return [
            r.id for r in store._selection_order(rows, "anthropic", session_id, model_id=model_id)
        ]

    @staticmethod
    def _hash_order(store: AuthStore, session_id: str) -> list[int]:
        rows = store.list_credentials("anthropic")
        return [r.id for r in AuthStore._hash_order(rows, session_id)]

    def test_least_loaded_wins_when_reports_differ_beyond_tolerance(self, tmp_path: Any) -> None:
        """(a) The observed skew: three near-full, two mostly empty. Every
        session must start on one of the two, whatever its hash says."""
        store, rows, cache = self._store(tmp_path)
        used = [0.65, 0.92, 0.99, 0.06, 0.29]
        for row, fraction in zip(rows, used):
            self._cache_report(cache, row, five_hour=fraction)

        winners = {self._order(store, f"session-{i}")[0] for i in range(40)}

        # 0.06 used is best (0.94 remaining); 0.29 (0.71) trails it by more
        # than the tolerance, so the bucket is the single emptiest account.
        assert winners == {rows[3].id}
        # The full order is best-first behind the bucket.
        assert self._order(store, "session-0") == [
            rows[3].id,
            rows[4].id,
            rows[0].id,
            rows[1].id,
            rows[2].id,
        ]

    def test_within_tolerance_ties_spread_by_the_session_hash(self, tmp_path: Any) -> None:
        """(b) Herding guard: accounts within the tolerance are one bucket and
        the per-session hash rotates it, so concurrent starts fan out."""
        store, rows, cache = self._store(tmp_path, 4)
        # Two within 10 points of each other, two far behind.
        for row, fraction in zip(rows, [0.10, 0.15, 0.80, 0.90]):
            self._cache_report(cache, row, five_hour=fraction)

        firsts = {self._order(store, f"session-{i}")[0] for i in range(40)}
        assert firsts == {rows[0].id, rows[1].id}
        for i in range(40):
            order = self._order(store, f"session-{i}")
            # Deterministic per session, bucket first, then the rest best-first.
            assert order == self._order(store, f"session-{i}")
            assert set(order[:2]) == {rows[0].id, rows[1].id}
            assert order[2:] == [rows[2].id, rows[3].id]

    def test_the_warmer_row_ranks_when_no_preflight_row_exists(self, tmp_path: Any) -> None:
        """(a, stock config) ``retry.usageAwareFallback`` ships OFF, so on a
        default install the preflight never writes a ``:pf:`` row and the
        only usage on disk is the per-provider payload ``/usage`` and the
        background warmer keep fresh. Reading only the preflight row made the
        pick a silent no-op there (review round 1, F1); the same skew as (a)
        written in the warmer's shape must rank identically."""
        store, rows, cache = self._store(tmp_path)
        used = [0.65, 0.92, 0.99, 0.06, 0.29]
        for row, fraction in zip(rows, used):
            self._cache_report(cache, row, five_hour=fraction, via="warmer")

        winners = {self._order(store, f"session-{i}")[0] for i in range(40)}
        assert winners == {rows[3].id}
        assert self._order(store, "session-0") == [
            rows[3].id,
            rows[4].id,
            rows[0].id,
            rows[1].id,
            rows[2].id,
        ]

    def test_the_newer_of_preflight_and_warmer_reports_wins(self, tmp_path: Any) -> None:
        """Both sources can name the same account; whichever was fetched
        last is the truth. Neither namespace is privileged -- a preflight row
        from an hour ago must not outrank a warm row from a minute ago, and
        vice versa."""
        store, rows, cache = self._store(tmp_path, 2)
        # rows[0]: preflight says nearly empty (stale), warmer says nearly full (fresh).
        self._cache_report(cache, rows[0], five_hour=0.05, age_ms=10 * 60_000)
        self._cache_report(cache, rows[0], five_hour=0.95, via="warmer")
        # rows[1]: warmer says nearly full (stale), preflight says nearly empty (fresh).
        self._cache_report(cache, rows[1], five_hour=0.95, age_ms=10 * 60_000, via="warmer")
        self._cache_report(cache, rows[1], five_hour=0.05)
        for i in range(20):
            assert self._order(store, f"session-{i}") == [rows[1].id, rows[0].id]

    def test_a_fresh_limit_less_stub_does_not_mask_a_real_report(self, tmp_path: Any) -> None:
        """The warmer's first failure for a never-seen account is a stub
        stamped ``now`` with an empty ``limits`` list. "Newest wins" must
        skip it: ranking by recency alone let that stub outvote a slightly
        older real preflight report and left a 95%-used account looking
        neutral (review round 2, F7)."""
        import time

        from local_operator.providers.usage import UsageReport
        from local_operator.providers.usage_cache import (
            account_backoff_ms,
            fingerprint_accounts,
            provider_cache_key,
        )

        store, rows, cache = self._store(tmp_path, 2)
        self._cache_report(cache, rows[0], five_hour=0.05, age_ms=2 * 60_000)
        # rows[1]: a REAL preflight report, 95% used, two minutes old...
        self._cache_report(cache, rows[1], five_hour=0.95, age_ms=2 * 60_000)
        # ...and the warmer's failure stub for the SAME account, fresher.
        now = int(time.time() * 1000)
        stub = UsageReport(
            provider="anthropic",
            fetched_at=now,
            identity=rows[1].data["email"],
            consecutive_failures=1,
            next_probe_at_ms=now + account_backoff_ms(1),
        )
        key = provider_cache_key("anthropic", fingerprint_accounts(["the-whole-set"]))
        cache.set(key, "anthropic", [stub], expires_at_ms=now + 60_000)

        # The stub is skipped, so rows[1]'s real 5% remaining still ranks it
        # last; masked, it would look neutral and share first place with
        # rows[0] under the hash rotation.
        for i in range(20):
            assert self._order(store, f"session-{i}") == [rows[0].id, rows[1].id]

    def test_a_preflight_row_is_never_mistaken_for_a_warmer_row(self, tmp_path: Any) -> None:
        """The warmer scan walks every row under the provider; the ``:pf:``
        rows live under the same prefix and must be skipped there, or a
        preflight report (which carries no ``identity``) could never match
        and a future one that did would be counted twice."""
        from local_operator.providers.usage import UsageReport
        from local_operator.providers.usage_cache import account_preflight_key

        store, rows, cache = self._store(tmp_path, 2)
        # A preflight row for rows[0] that ALSO carries an identity naming
        # rows[1]: if the scan read it, rows[1] would inherit rows[0]'s numbers.
        self._cache_report(cache, rows[0], five_hour=0.95)
        key = account_preflight_key("anthropic", rows[0].data["email"])
        poisoned = cache.get(key, include_expired=True)
        assert poisoned
        poisoned[0].identity = rows[1].data["email"]
        cache.set(key, "anthropic", poisoned, expires_at_ms=poisoned[0].fetched_at + 60_000)
        assert isinstance(poisoned[0], UsageReport)

        now_ms = store._now_ms()
        assert (
            store._cached_remaining_fraction(rows[1], "anthropic", "claude-opus-5", now_ms) is None
        )
        seen = store._cached_remaining_fraction(rows[0], "anthropic", "claude-opus-5", now_ms)
        assert seen is not None and abs(seen - 0.05) < 1e-9

    def test_all_equal_is_exactly_the_hash_order(self, tmp_path: Any) -> None:
        """(b, degenerate) A uniform pool must reproduce the pre-existing order
        byte for byte -- the ranking is a no-op when it has nothing to say."""
        store, rows, cache = self._store(tmp_path)
        for row in rows:
            self._cache_report(cache, row, five_hour=0.5)
        for i in range(20):
            assert self._order(store, f"session-{i}") == self._hash_order(store, f"session-{i}")

    def test_stale_or_missing_reports_fall_back_to_hash_order(self, tmp_path: Any) -> None:
        """(c) No evidence is not evidence: an empty cache, and a cache whose
        rows are past the max age, both leave the hash order untouched."""
        from local_operator.providers import auth_store as auth_store_mod

        store, rows, cache = self._store(tmp_path)
        for i in range(20):
            assert self._order(store, f"session-{i}") == self._hash_order(store, f"session-{i}")

        for row, fraction in zip(rows, [0.65, 0.92, 0.99, 0.06, 0.29]):
            self._cache_report(
                cache,
                row,
                five_hour=fraction,
                age_ms=auth_store_mod.USAGE_PICK_MAX_REPORT_AGE_MS + 1_000,
            )
        for i in range(20):
            assert self._order(store, f"session-{i}") == self._hash_order(store, f"session-{i}")

    def test_an_old_report_whose_windows_have_not_reset_is_still_trusted(
        self, tmp_path: Any
    ) -> None:
        """(c, refined) Usage inside a window only rises until the window
        resets, so an old report with every reset still ahead is a lower
        bound and must keep ranking. Observed live: a 99%-full account with
        a two-hour-old report ranked neutral and drew most new sessions."""
        from local_operator.providers import auth_store as auth_store_mod

        store, rows, cache = self._store(tmp_path, 3)
        old = auth_store_mod.USAGE_PICK_MAX_REPORT_AGE_MS * 4
        self._cache_report(cache, rows[0], five_hour=0.10, age_ms=old, resets_in_ms=60 * 60_000)
        self._cache_report(cache, rows[1], five_hour=0.99, age_ms=old, resets_in_ms=60 * 60_000)
        # rows[2]: as old, as full, but its window reset since -- unknown.
        self._cache_report(cache, rows[2], five_hour=0.99, age_ms=old, resets_in_ms=-60_000)

        firsts = {self._order(store, f"session-{i}")[0] for i in range(40)}
        assert firsts == {rows[0].id, rows[2].id}
        for i in range(40):
            assert self._order(store, f"session-{i}")[-1] == rows[1].id

    def test_an_unknown_account_is_neutral_not_penalised(self, tmp_path: Any) -> None:
        """(c, partial) One account with no report joins the bucket rather than
        sorting last: a cold row must not be starved by its own missing data."""
        store, rows, cache = self._store(tmp_path, 3)
        self._cache_report(cache, rows[0], five_hour=0.10)
        self._cache_report(cache, rows[1], five_hour=0.95)
        # rows[2] has no report.
        firsts = {self._order(store, f"session-{i}")[0] for i in range(40)}
        assert firsts == {rows[0].id, rows[2].id}
        for i in range(40):
            assert self._order(store, f"session-{i}")[-1] == rows[1].id

    async def test_sticky_is_respected_regardless_of_usage(self, tmp_path: Any) -> None:
        """(d) Once a session is on an account it stays there even when the
        cache later says a sibling has more headroom -- the provider's prompt
        cache is per account, and moving would rewrite the whole prefix."""
        store, rows, cache = self._store(tmp_path, 3)
        for row in rows:
            self._cache_report(cache, row, five_hour=0.5)
        first = await store.get_oauth_access("anthropic", "s-sticky", model_id="claude-opus-5")
        assert first is not None
        chosen = first.credential_id
        # Now make the chosen account look nearly spent and a sibling empty.
        for row in rows:
            self._cache_report(cache, row, five_hour=0.99 if row.id == chosen else 0.01)
        again = await store.get_oauth_access("anthropic", "s-sticky", model_id="claude-opus-5")
        assert again is not None and again.credential_id == chosen
        # A NEW session, with no sticky, does move away from it.
        fresh = await store.get_oauth_access("anthropic", "s-new", model_id="claude-opus-5")
        assert fresh is not None and fresh.credential_id != chosen

    def test_a_tier_cap_does_not_penalise_a_model_outside_that_tier(self, tmp_path: Any) -> None:
        """(e) A spent Fable weekly must not push an account down for an Opus
        request; for a Fable request it must."""
        store, rows, cache = self._store(tmp_path, 2)
        # Both accounts identical on the shared windows; only rows[0] has a
        # nearly spent fable cap.
        self._cache_report(cache, rows[0], five_hour=0.30, fable=0.98)
        self._cache_report(cache, rows[1], five_hour=0.30)

        opus = {self._order(store, f"session-{i}", "claude-opus-5")[0] for i in range(20)}
        assert opus == {rows[0].id, rows[1].id}, "opus must see a tie and spread"
        fable = {self._order(store, f"session-{i}", "claude-fable-5-1")[0] for i in range(20)}
        assert fable == {rows[1].id}, "fable must avoid the account whose fable cap is spent"

    def test_opt_out_restores_the_hash_order(self, tmp_path: Any) -> None:
        """``retry.usageAwareAccountPick: false`` reaches the store through
        ``configure_usage_aware_pick`` and the ranking steps aside entirely."""
        store, rows, cache = self._store(tmp_path)
        for row, fraction in zip(rows, [0.65, 0.92, 0.99, 0.06, 0.29]):
            self._cache_report(cache, row, five_hour=fraction)
        store.configure_usage_aware_pick(False)
        for i in range(20):
            assert self._order(store, f"session-{i}") == self._hash_order(store, f"session-{i}")
        store.configure_usage_aware_pick(True)
        assert self._order(store, "session-0")[0] == rows[3].id

    def test_a_broken_cache_fails_open_to_the_hash_order(self, tmp_path: Any) -> None:
        """Fail open: a cache read that raises must never change the resolve."""
        store, rows, cache = self._store(tmp_path)

        def boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("cache exploded")

        cache.get = boom  # type: ignore[method-assign]
        for i in range(20):
            assert self._order(store, f"session-{i}") == self._hash_order(store, f"session-{i}")

    def test_the_session_stream_forwards_the_opt_out(self, tmp_path: Any) -> None:
        """The setting is parsed in exactly one place -- ``SessionStreamFn`` --
        and must reach the store; a default settings mapping leaves it on."""
        from local_operator.model.configure import SessionStreamFn

        store, _rows, _cache = self._store(tmp_path, 2)
        stream = SessionStreamFn(store, {"retry": {"usageAwareAccountPick": False}}, "s1")
        assert store._usage_aware_pick is False
        stream2 = SessionStreamFn(store, {}, "s2")
        assert store._usage_aware_pick is True
        # Close the streams' http clients without awaiting: no loop is running.
        del stream, stream2
