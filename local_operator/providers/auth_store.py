"""SQLite credential store + the API-key resolution cascade.

Credentials live in ``~/.local-operator/auth.db`` (0600, WAL, busy
timeout); no OS keychain is used.

The resolution cascade (``get_api_key``) is the no-auth-mode-switch design
First match wins:

1. runtime override (CLI ``--api-key``)
2. config override (``models.yml``/gateway pointer)
3. OAuth credential (auto-refresh + stickiness/round-robin)
4. API key persisted by ``login`` (``source="login"``)
5. env var — including the legacy ``credentials.env`` file read through
   ``local_operator.credentials.CredentialManager`` when importable
6. stored API key without ``source="login"``
7. fallback resolver (custom providers)

Side effect preserved: session stickiness for a provider is cleared as soon
as resolution LEAVES the OAuth tier (before the env tier), so identity
lookups stop attributing OAuth accounts.

Single-process limitation (PR-24): refresh single-flight is a per-process
``asyncio.Lock``. Two local-operator processes racing a rotating OAuth
refresh token can invalidate each other's new token; a guard with a
cross-process ``auth_credential_refresh_leases`` table, which is deliberately
deferred here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import sqlite3
import time
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from local_operator.paths import config_dir
from local_operator.providers.registry import (
    ProviderDefinition,
    RefreshFn,
    get_provider_definition,
)

if TYPE_CHECKING:  # the legacy reader stays an optional, import-guarded tier
    from local_operator.credentials import CredentialManager

logger = logging.getLogger("local_operator.providers.auth_store")

OAUTH_REFRESH_SKEW_MS = 60_000  # pre-emptive refresh trigger
DEFAULT_BLOCK_MS = 60_000  # rate-limit / 401 backoff

_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_credentials (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  credential_type TEXT NOT NULL,
  data TEXT NOT NULL,
  disabled_cause TEXT DEFAULT NULL,
  identity_key TEXT DEFAULT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auth_provider ON auth_credentials(provider);
CREATE INDEX IF NOT EXISTS idx_auth_provider_identity
  ON auth_credentials(provider, identity_key) WHERE identity_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS auth_credential_blocks (
  credential_id INTEGER NOT NULL,
  provider_key TEXT NOT NULL,
  block_scope TEXT NOT NULL DEFAULT '',
  blocked_until_ms INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (credential_id, provider_key, block_scope)
);
CREATE INDEX IF NOT EXISTS idx_auth_credential_blocks_expires
  ON auth_credential_blocks(blocked_until_ms);
"""


class AuthStoreError(Exception):
    """Credential resolution/refresh failure (never a bare sqlite error)."""


@dataclasses.dataclass
class StoredCredential:
    """One row of ``auth_credentials`` with its parsed payload."""

    id: int
    provider: str
    credential_type: str  # 'api_key' | 'oauth'
    data: dict[str, Any]
    disabled_cause: str | None = None
    identity_key: str | None = None
    created_at: int = 0
    updated_at: int = 0


@dataclasses.dataclass(frozen=True)
class OAuthAccess:
    """The identity-carrying credential record handed to wire clients.

    ``get_oauth_access()``: everything a provider-specific request
    shaper needs beyond the bare bearer — which account/org pays, and
    whether the token is OAuth (needs provider-specific auth headers/routes)
    or a plain API key.
    """

    access_token: str
    credential_id: int
    account_id: str | None = None
    email: str | None = None
    org_id: str | None = None
    api_endpoint: str | None = None
    kind: str = "oauth"  # 'oauth' | 'api_key'


def default_db_path() -> Path:
    return config_dir() / "auth.db"


def _identity_key_for(provider: str, credential: dict[str, Any]) -> str | None:
    """Dedupe key so one account holds one row (org scope ⇒ separate rows).

    API keys and CLI-stored credentials never dedupe (each key is its own
    row). OAuth payloads dedupe on the account identity when the IdP returns
    one; when it does not (Kimi returns none, xAI/Anthropic only with an
    id_token), a deterministic per-provider constant keeps re-login on ONE
    row — otherwise two logins leave two rows and the older one carries a
    dead rotated refresh token that the cascade keeps selecting.
    """
    if credential.get("type") == "api_key" or credential.get("source") == "login":
        return None
    for field in ("org_id", "account_id", "email", "project_id"):
        value = credential.get(field)
        if value:
            return str(value)
    if credential.get("refresh") and credential.get("access"):
        return f"oauth:{provider}"
    return None


class AuthStore:
    """Credential persistence + the 7-step cascade.

    ``credential_manager`` (legacy ``CredentialManager``) feeds the env tier
    from ``credentials.env``. ``config_overrides`` seeds the config-override
    tier. All DB access is local and synchronous; async methods only exist
    where refresh/network happens.

    .. note::
        Single-process (PR-24): the refresh lock is a per-process
        ``asyncio.Lock``; cross-process refresh leases are not implemented.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        credential_manager: "CredentialManager | None" = None,
        config_overrides: dict[str, str] | None = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else default_db_path()
        self._credential_manager = credential_manager
        self._config_overrides = dict(config_overrides or {})
        self._runtime_overrides: dict[str, str] = {}
        self._fallback_resolvers: dict[str, Callable[[str], str | None]] = {}
        self._sticky: dict[tuple[str, str], int] = {}
        self._round_robin: dict[str, int] = {}
        self._refresh_locks: dict[int, asyncio.Lock] = {}
        self._conn = self._connect()

    # -- connection ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file 0600 BEFORE sqlite opens it: the plain
        # connect-then-chmod leaves a window where secrets sit 0644 (PR-11).
        if not self._db_path.exists():
            fd = os.open(self._db_path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_SCHEMA)
        conn.commit()
        for path in (
            self._db_path,
            self._db_path.with_suffix(self._db_path.suffix + "-wal"),
            self._db_path.with_suffix(self._db_path.suffix + "-shm"),
        ):
            # WAL sidecars hold the same plaintext; keep them 0600 too.
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        return conn

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # -- overrides -----------------------------------------------------------

    def set_runtime_api_key(self, provider: str, api_key: str | None) -> None:
        """Tier 1: CLI ``--api-key``. ``None`` clears."""
        if api_key:
            self._runtime_overrides[provider] = api_key
        else:
            self._runtime_overrides.pop(provider, None)

    def set_config_api_key(self, provider: str, api_key: str | None) -> None:
        """Tier 2: models.yml pointer. Beats OAuth because the user aimed the
        provider at a custom base URL/gateway."""
        if api_key:
            self._config_overrides[provider] = api_key
        else:
            self._config_overrides.pop(provider, None)

    def set_fallback_resolver(
        self, provider: str, resolver: Callable[[str], str | None] | None
    ) -> None:
        """Tier 7: custom-provider hook."""
        if resolver is None:
            self._fallback_resolvers.pop(provider, None)
        else:
            self._fallback_resolvers[provider] = resolver

    # -- credential CRUD -------------------------------------------------------

    @staticmethod
    def _row_to_credential(row: tuple[Any, ...]) -> StoredCredential:
        return StoredCredential(
            id=row[0],
            provider=row[1],
            credential_type=row[2],
            data=json.loads(row[3]),
            disabled_cause=row[4],
            identity_key=row[5],
            created_at=row[6],
            updated_at=row[7],
        )

    def list_credentials(
        self, provider: str | None = None, include_disabled: bool = False
    ) -> list[StoredCredential]:
        """Enabled credentials (all providers or one), oldest first."""
        if provider is not None:
            rows = self._conn.execute(
                "SELECT id, provider, credential_type, data, disabled_cause, identity_key,"
                " created_at, updated_at FROM auth_credentials WHERE provider = ? ORDER BY id",
                (provider,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, provider, credential_type, data, disabled_cause, identity_key,"
                " created_at, updated_at FROM auth_credentials ORDER BY id"
            ).fetchall()
        creds = [self._row_to_credential(r) for r in rows]
        if include_disabled:
            return creds
        return [c for c in creds if c.disabled_cause is None]

    def upsert_credential(self, provider: str, credential: dict[str, Any]) -> StoredCredential:
        """Insert, or update the row for the same identity (org scope ⇒ rows).

        Revives soft-deleted rows for the same identity (re-login).
        ``store_credentials_as`` aliasing happens at the caller (login path).
        """
        credential_type = (
            "oauth" if credential.get("refresh") and credential.get("access") else "api_key"
        )
        identity = _identity_key_for(provider, credential)
        payload = dict(credential)
        payload["type"] = credential_type
        now = self._now_ms()
        data_json = json.dumps(payload)

        if identity is not None:
            row = self._conn.execute(
                "SELECT id FROM auth_credentials WHERE provider = ? "
                "AND identity_key = ? ORDER BY id",
                (provider, identity),
            ).fetchone()
            if row is not None:
                self._conn.execute(
                    "UPDATE auth_credentials SET credential_type = ?, data = ?, "
                    "disabled_cause = NULL,"
                    " updated_at = ? WHERE id = ?",
                    (credential_type, data_json, now, row[0]),
                )
                self._conn.commit()
                return self._reread_after_write(row[0])

        cursor = self._conn.execute(
            "INSERT INTO auth_credentials "
            "(provider, credential_type, data, identity_key, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (provider, credential_type, data_json, identity, now, now),
        )
        self._conn.commit()
        return self._reread_after_write(cursor.lastrowid)

    def _reread_after_write(self, credential_id: int | None) -> StoredCredential:
        """Re-read a row this connection just wrote.

        The write and the read share one connection, so a miss is impossible;
        surfacing it as an error beats leaking ``None`` out of a non-optional
        return and failing somewhere further away.
        """
        stored = self.get_credential(credential_id) if credential_id is not None else None
        if stored is None:
            raise AuthStoreError("Credential row could not be read back after write")
        return stored

    def get_credential(self, credential_id: int) -> StoredCredential | None:
        row = self._conn.execute(
            "SELECT id, provider, credential_type, data, disabled_cause, identity_key,"
            " created_at, updated_at FROM auth_credentials WHERE id = ?",
            (credential_id,),
        ).fetchone()
        return self._row_to_credential(row) if row else None

    def disable_credential(self, credential_id: int, cause: str) -> None:
        """Soft-delete tombstone (keeps history, blocks selection)."""
        self._conn.execute(
            "UPDATE auth_credentials SET disabled_cause = ?, updated_at = ? WHERE id = ?",
            (cause, self._now_ms(), credential_id),
        )
        self._conn.commit()

    def delete_credential(self, credential_id: int) -> None:
        """Remove a row entirely (logout)."""
        self._conn.execute(
            "DELETE FROM auth_credential_blocks WHERE credential_id = ?", (credential_id,)
        )
        self._conn.execute("DELETE FROM auth_credentials WHERE id = ?", (credential_id,))
        self._conn.commit()

    def delete_credentials_for_provider(
        self, provider: str, disabled_cause: str = "logged-out"
    ) -> int:
        """Logout: wipe every credential stored under ``provider``. Returns count."""
        rows = self.list_credentials(provider)
        for row in rows:
            self.delete_credential(row.id)
        return len(rows)

    # -- blocking --------------------------------------------------------------

    def block_credential(
        self,
        credential_id: int,
        provider: str,
        block_scope: str = "",
        block_ms: int = DEFAULT_BLOCK_MS,
    ) -> None:
        """Record a backoff keyed by ``provider:type``."""
        credential = self.get_credential(credential_id)
        provider_key = f"{provider}:{credential.credential_type if credential else 'api_key'}"
        until = self._now_ms() + max(1000, int(block_ms))
        self._conn.execute(
            "INSERT INTO auth_credential_blocks (credential_id, provider_key, block_scope,"
            " blocked_until_ms, updated_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(credential_id, provider_key, block_scope)"
            " DO UPDATE SET blocked_until_ms = excluded.blocked_until_ms, "
            "updated_at = excluded.updated_at",
            (credential_id, provider_key, block_scope, until, self._now_ms()),
        )
        self._conn.commit()

    def is_blocked(self, credential_id: int, provider: str) -> bool:
        credential = self.get_credential(credential_id)
        provider_key = f"{provider}:{credential.credential_type if credential else 'api_key'}"
        row = self._conn.execute(
            "SELECT blocked_until_ms FROM auth_credential_blocks"
            " WHERE credential_id = ? AND provider_key = ? AND block_scope = ''",
            (credential_id, provider_key),
        ).fetchone()
        return bool(row and row[0] > self._now_ms())

    def clear_blocks(self, credential_id: int) -> None:
        self._conn.execute(
            "DELETE FROM auth_credential_blocks WHERE credential_id = ?", (credential_id,)
        )
        self._conn.commit()

    # -- OAuth refresh -----------------------------------------------------------

    def _refresh_fn(self, provider: str) -> RefreshFn | None:
        definition = get_provider_definition(provider)
        if definition is not None and definition.refresh_token is not None:
            return definition.refresh_token
        # Rows stored under an alias (xai-oauth ⇒ xai): find the origin def.
        from local_operator.providers.registry import PROVIDER_REGISTRY

        for other in PROVIDER_REGISTRY:
            if other.store_credentials_as == provider and other.refresh_token is not None:
                return other.refresh_token
        return None

    def _refresh_lock_for(self, credential_id: int) -> asyncio.Lock:
        lock = self._refresh_locks.get(credential_id)
        if lock is None:
            lock = asyncio.Lock()
            self._refresh_locks[credential_id] = lock
        return lock

    @staticmethod
    def _needs_refresh(creds: dict[str, Any], *, force: bool = False) -> bool:
        if force:
            return True
        access = creds.get("access")
        if not access:
            return True
        expires = creds.get("expires")
        if expires is None:  # static token; never expires
            return False
        return int(expires) <= AuthStore._now_ms() + OAUTH_REFRESH_SKEW_MS

    async def _ensure_oauth_fresh(
        self, row: StoredCredential, *, force: bool = False
    ) -> dict[str, Any]:
        """Return usable OAuth data for ``row``, refreshing single-flight.

        Raises :class:`AuthStoreError` when a refresh is required and fails;
        callers treat that row as unusable and rotate.

        Org fields are restored from the stored row AFTER the merge (PR-12):
        a refresh function that (mistakenly) returns org_id/org_name/
        authorized_at can never rewrite them — identity is fixed at login.
        """
        creds = dict(row.data)
        if not self._needs_refresh(creds, force=force):
            return creds
        refresh = self._refresh_fn(row.provider)
        if refresh is None:
            raise AuthStoreError(f"No refresh capability for provider '{row.provider}'")

        async with self._refresh_lock_for(row.id):
            # Re-read inside the lock: another coroutine may have refreshed.
            current = self.get_credential(row.id)
            if current is None or current.disabled_cause is not None:
                raise AuthStoreError(f"Credential {row.id} disappeared during refresh")
            fresh = dict(current.data)
            if not self._needs_refresh(fresh, force=force):
                return fresh
            try:
                refreshed = await refresh(fresh)
            except AuthStoreError:
                raise
            except Exception as exc:
                raise AuthStoreError(f"OAuth refresh failed for '{row.provider}': {exc}") from exc
            merged = dict(fresh)
            merged.update(refreshed)
            # Restore identity fields from the stored credential — NEVER
            # rewritten by refresh, whatever the refresh fn returns.
            for field in ("org_id", "org_name", "authorized_at"):
                if field in fresh:
                    merged[field] = fresh[field]
                else:
                    merged.pop(field, None)
            # Cross-process guard: the server's job processes each build their
            # own AuthStore, so the per-process refresh lock does not cover
            # them. Two processes racing a rotating refresh token both POST
            # the same token; the IdP rotates and the loser's new token is
            # dead. If the stored refresh token changed under us (another
            # process won), skip our write — overwriting would clobber the
            # winner's live token with our dead one and soft-delete the row.
            now_row = self.get_credential(row.id)
            if now_row is not None and dict(now_row.data).get("refresh") != fresh.get("refresh"):
                logger.warning(
                    "refresh race on %s: another process refreshed first; " "keeping its token",
                    row.id,
                )
                return dict(now_row.data)
            self._conn.execute(
                "UPDATE auth_credentials SET data = ?, updated_at = ? WHERE id = ?",
                (json.dumps(merged), self._now_ms(), row.id),
            )
            self._conn.commit()
            return merged

    # -- selection: stickiness + round-robin -------------------------------------

    def _selection_order(
        self, rows: list[StoredCredential], provider: str, session_id: str | None
    ) -> list[StoredCredential]:
        if not rows:
            return []
        if session_id:
            sticky_id = self._sticky.get((provider, session_id))
            sticky = next((r for r in rows if r.id == sticky_id), None)
            if sticky is not None:
                rest = [r for r in rows if r.id != sticky.id]
                return [sticky, *rest]
            index = zlib.crc32(session_id.encode("utf-8")) % len(rows)
            return rows[index:] + rows[:index]
        # No session: round-robin across calls.
        provider_key = f"{provider}:any"
        start = self._round_robin.get(provider_key, 0) % len(rows)
        self._round_robin[provider_key] = start + 1
        return rows[start:] + rows[:start]

    def _set_sticky(self, provider: str, session_id: str | None, credential_id: int | None) -> None:
        if not session_id:
            return
        if credential_id is None:
            self._sticky.pop((provider, session_id), None)
        else:
            self._sticky[(provider, session_id)] = credential_id

    def _usable_key_rows(
        self, provider: str, credential_type: str, source: str | None
    ) -> list[StoredCredential]:
        rows = [
            r
            for r in self.list_credentials(provider)
            if r.credential_type == credential_type and not self.is_blocked(r.id, provider)
        ]
        if source is None:
            return rows
        return [r for r in rows if r.data.get("source") == source]

    # -- the cascade ---------------------------------------------------------

    async def get_api_key(
        self, provider: str, session_id: str | None = None, *, force_refresh: bool = False
    ) -> str | None:
        """Resolve the API key for ``provider`` via the 7-step cascade."""
        key, _row = await self._resolve(provider, session_id, force_refresh=force_refresh)
        return key

    async def get_oauth_access(
        self, provider: str, session_id: str | None = None, *, force_refresh: bool = False
    ) -> OAuthAccess | None:
        """The identity-carrying record for wire clients.

        Returns :class:`OAuthAccess` for whichever credential the cascade
        picks — ``kind == "oauth"`` with account/org identity when an OAuth
        row wins, ``kind == "api_key"`` otherwise. Runtime/config overrides
        deliberately short-circuit to ``None`` (they aim at gateways where
        stored identity does not apply).
        """
        if self._runtime_overrides.get(provider) or self._config_overrides.get(provider):
            return None
        key, row = await self._resolve(provider, session_id, force_refresh=force_refresh)
        if key is None:
            return None
        if row is not None and row.credential_type == "oauth":
            data = row.data
            return OAuthAccess(
                access_token=key,
                credential_id=row.id,
                account_id=data.get("account_id"),
                email=data.get("email"),
                org_id=data.get("org_id"),
                api_endpoint=data.get("api_endpoint"),
                kind="oauth",
            )
        return OAuthAccess(
            access_token=key, credential_id=row.id if row is not None else 0, kind="api_key"
        )

    async def _resolve(
        self, provider: str, session_id: str | None, *, force_refresh: bool = False
    ) -> tuple[str | None, StoredCredential | None]:
        """The 7-step cascade; returns ``(key, winning row or None)``."""
        definition: ProviderDefinition | None = get_provider_definition(provider)

        # 1. Runtime override
        runtime = self._runtime_overrides.get(provider)
        if runtime:
            return runtime, None

        # 2. Config override
        config = self._config_overrides.get(provider)
        if config:
            return config, None

        # 3. OAuth credential
        oauth_rows = self._usable_key_rows(provider, "oauth", source=None)
        for row in self._selection_order(oauth_rows, provider, session_id):
            try:
                creds = await self._ensure_oauth_fresh(row, force=force_refresh)
            except AuthStoreError:
                self.block_credential(row.id, provider)  # try a sibling
                continue
            key_fn = definition.get_api_key if definition else None
            key = key_fn(creds) if key_fn else creds.get("access")
            if key:
                self._set_sticky(provider, session_id, row.id)
                refreshed = self.get_credential(row.id)
                return key, refreshed or row
        if oauth_rows and force_refresh:
            # Every sibling failed its refresh — surface the failure so the
            # failover layer can block/back off instead of silently looping.
            raise AuthStoreError(f"All OAuth credentials for '{provider}' failed to refresh")
        # PR-15: with NO oauth rows, force_refresh falls through to tiers 4-7.

        # 4. API key persisted by interactive login
        login_rows = self._usable_key_rows(provider, "api_key", source="login")
        for row in self._selection_order(login_rows, provider, session_id):
            key = row.data.get("key")
            if key:
                self._set_sticky(provider, session_id, row.id)
                return key, row

        # Leaving the OAuth tier: clear session stickiness so identity
        # attribution stops for non-OAuth requests (PR-16; cleared before
        # step 5, regardless of which later tier ends up winning).
        self._set_sticky(provider, session_id, None)

        # 5. Env var tier (process env, then legacy credentials.env).
        env_key = self._env_api_key(provider)
        if env_key:
            return env_key, None

        # 6. Stored api_key without source="login" (e.g. broker migration)
        stored_rows = [
            row
            for row in self._usable_key_rows(provider, "api_key", source=None)
            if row.data.get("source") != "login"
        ]
        for row in self._selection_order(stored_rows, provider, session_id):
            key = row.data.get("key")
            if key:
                self._set_sticky(provider, session_id, row.id)
                return key, row
        # 7. Fallback resolver
        resolver = self._fallback_resolvers.get(provider)
        if resolver is not None:
            return resolver(provider), None

        return None, None

    def _env_api_key(self, provider: str) -> str | None:
        definition = get_provider_definition(provider)
        if definition is not None and definition.env_keys is not None:
            if callable(definition.env_keys):
                value = definition.env_keys()
                if value:
                    return value
            else:
                value = os.environ.get(definition.env_keys)
                if value:
                    return value
        # Legacy credentials.env tier via CredentialManager (lazy import so a
        # missing legacy module degrades to env-only).
        if self._credential_manager is None:
            self._credential_manager = _load_legacy_credential_manager()
        manager = self._credential_manager
        if manager is not None and definition is not None and isinstance(definition.env_keys, str):
            try:
                secret = manager.get_credentials().get(definition.env_keys)
            except Exception:
                secret = None
            if secret is not None:
                value = secret.get_secret_value()
                if value:
                    return value
        return None

    # -- failover support --------------------------------------------------------

    def rotate_sibling(
        self,
        provider: str,
        session_id: str | None,
        error: BaseException,
        api_key: str | None = None,
        block_ms: int = DEFAULT_BLOCK_MS,
    ) -> bool:
        """a/b/c tier-1 step (c): drop the failing credential, keep a sibling.

        Usage-limit errors only get a temporary block (sticky preserved — the
        sibling rotation happens outside the backoff window). Invalidated
        tokens are soft-deleted. Returns whether another enabled credential
        of the same type remains.
        """
        from local_operator.providers.failover import (
            is_invalidated_credential_error,
            is_usage_limit_error,
            retry_after_ms_from_error,
        )

        rows = self.list_credentials(provider)
        if api_key is not None:
            failing = next((r for r in rows if self._row_matches_key(r, api_key)), None)
        elif session_id:
            failing = next(
                (r for r in rows if r.id == self._sticky.get((provider, session_id))), None
            )
        else:
            failing = None

        usage_limited = is_usage_limit_error(error)
        if failing is not None:
            retry_after = retry_after_ms_from_error(error)
            self.block_credential(failing.id, provider, block_ms=max(block_ms, retry_after or 0))
            if usage_limited:
                # Sticky preserved: same account stays first after backoff.
                pass
            else:
                self._set_sticky(provider, session_id, None)
                if is_invalidated_credential_error(error):
                    self.disable_credential(failing.id, cause="invalidated-token")

        credential_type = failing.credential_type if failing else None
        siblings = [
            r
            for r in self.list_credentials(provider)
            if r.id != (failing.id if failing else None)
            and not self.is_blocked(r.id, provider)
            and (credential_type is None or r.credential_type == credential_type)
        ]
        return len(siblings) > 0

    @staticmethod
    def _row_matches_key(row: StoredCredential, api_key: str) -> bool:
        if row.credential_type == "api_key":
            return row.data.get("key") == api_key
        return row.data.get("access") == api_key

    def credential_id_for_key(self, provider: str, api_key: str) -> int | None:
        """Reverse lookup used by failover to block the exact bearer."""
        for row in self.list_credentials(provider):
            if self._row_matches_key(row, api_key):
                return row.id
        return None


def _load_legacy_credential_manager() -> "CredentialManager | None":
    """Best-effort legacy ``credentials.env`` reader (import-guarded)."""
    try:
        from local_operator.credentials import CredentialManager

        base = config_dir()
        if (base / "credentials.env").exists():
            return CredentialManager(base)
    except Exception:
        pass
    return None
