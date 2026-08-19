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
from collections.abc import Iterable
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

#: How long a provider-fault demotion keeps a credential at the back of the pool
#: (see ``AuthStore.deprioritize_credential``). Deliberately short: the mark says
#: "this account was failing a moment ago", which stops being useful information
#: quickly, and it cannot expire by being USED -- a demoted row sorts last, so it
#: is not selected, so it never earns the success that would clear it. Two
#: minutes outlives a burst of 529s without outliving the outage that caused it.
DEPRIORITIZE_TTL_MS = 120_000

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
    #: The full stored credential dict, when this row is OAuth. The wire's
    #: bearer is ``access_token`` (already mapped through the provider's
    #: ``get_api_key``), but some providers split duties across two tokens —
    #: the QwenCloud Token Plan infers on a pasted ``sk-sp-…`` key while quota
    #: needs the OAuth ``access`` — and a usage fetcher must see the raw row
    #: to spend the right one. None for plain API-key rows.
    raw: dict[str, Any] | None = None


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
        # Credentials that just failed on a PROVIDER-side fault, which is not
        # their fault and so must not block them (see ``rotate_sibling``). They
        # are merely sorted last, so an attempt moves to a sibling while the
        # deprioritised row stays available as a last resort.
        #
        # Deliberately unpersisted: the condition it describes lasts seconds,
        # and a mark surviving a restart would misroute a session for a fault
        # that had long cleared.
        #
        # Deliberately keyed by PROVIDER rather than by session, and so shared
        # by every session in the process — like ``_round_robin`` above. That is
        # the correct scope for what it records: "this account was failing at
        # this provider a moment ago" is a fact about the provider and the
        # account, not about who observed it, so a sibling session benefits from
        # the discovery instead of paying to repeat it. Nothing here can make a
        # credential unusable, so the worst a stale mark can do is reorder a
        # pool whose members are all equally valid.
        self._deprioritized: dict[str, dict[int, int]] = {}
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
        # An EXPLICIT type wins; the structural guess is the fallback for the
        # callers (and stored rows) that never declared one.
        #
        # The guess reads "has both refresh and access", which cannot see a
        # credential that is OAuth-issued but has no refresh token because it
        # never expires -- Z.AI's coding-plan sign-in mints exactly that. Such a
        # row landed as `api_key` with its secret under `data["access"]`, where
        # nothing can read it: tiers 4 and 6 read `data["key"]`, and tier 3 only
        # walks `oauth`-typed rows. The login reported success and every request
        # afterwards failed with no credential at all.
        declared = credential.get("type")
        credential_type = (
            declared
            if declared in ("oauth", "api_key")
            else ("oauth" if credential.get("refresh") and credential.get("access") else "api_key")
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

    def deprioritize_credential(self, provider: str, credential_id: int) -> None:
        """Sort ``credential_id`` last for ``provider`` without blocking it.

        The half of ``rotate_sibling`` that applies when the PROVIDER failed
        rather than the credential: a 529 storm must move the next attempt onto
        another account, but blocking would strand a healthy account (and,
        repeated across the pool, strand every one of them).

        The mark EXPIRES on its own after :data:`DEPRIORITIZE_TTL_MS`. Clearing
        it on a successful request is not sufficient by itself, and the reason is
        circular: a demoted credential sorts last, so it is not selected, so it
        never gets the success that would clear it. Without a TTL a single 529
        left an account bottom-of-pool for the life of the process -- the same
        "healthy account effectively out of rotation" outcome this whole change
        exists to prevent, arrived at the slow way.
        """
        marks = self._deprioritized.setdefault(provider, {})
        marks[credential_id] = self._now_ms() + DEPRIORITIZE_TTL_MS

    def clear_deprioritized(
        self, provider: str, credential_id: int | Iterable[int] | None = None
    ) -> None:
        """Restore full priority for a credential, several, or all of them.

        Takes an explicit id set rather than always clearing the provider,
        because ``_selection_order`` runs once per cascade TIER with a different
        subset of rows: a one-row tier finding its only row demoted must not
        drop the marks belonging to rows in another tier that it never saw.
        """
        marks = self._deprioritized.get(provider)
        if marks is None:
            return
        if credential_id is None:
            self._deprioritized.pop(provider, None)
            return
        ids = {credential_id} if isinstance(credential_id, int) else {int(i) for i in credential_id}
        for one in ids:
            marks.pop(one, None)
        if not marks:
            self._deprioritized.pop(provider, None)

    def _active_demotions(self, provider: str) -> set[int]:
        """Ids still demoted, dropping any whose TTL has passed.

        Expiry is evaluated on READ rather than by a timer: the marks are only
        consulted here, so a lazy sweep is both sufficient and free of a
        background task that would have to be owned and cancelled.
        """
        marks = self._deprioritized.get(provider)
        if not marks:
            return set()
        now = self._now_ms()
        for cid in [cid for cid, until in marks.items() if until <= now]:
            marks.pop(cid, None)
        if not marks:
            self._deprioritized.pop(provider, None)
            return set()
        return set(marks)

    def _selection_order(
        self,
        rows: list[StoredCredential],
        provider: str,
        session_id: str | None,
        *,
        read_only: bool = False,
    ) -> list[StoredCredential]:
        if not rows:
            return []
        ordered = self._base_selection_order(rows, provider, session_id)
        # Demotion is applied LAST, to the finished order.
        #
        # It used to run first, which did not work: both orderings below rotate
        # the list (`rows[i:] + rows[:i]`), so a row moved to the back was
        # rotated straight back towards the front. With three credentials and a
        # session whose hash landed on index 1, the account that had just failed
        # was tried SECOND -- the pool was never fully walked, which is the bug
        # the demotion exists to fix. Applying it here cannot be undone by a
        # later step, and because the partition is stable the relative order the
        # sticky/hash/round-robin choice produced is otherwise preserved.
        demoted = self._active_demotions(provider)
        if not demoted:
            return ordered
        preferred = [r for r in ordered if r.id not in demoted]
        # Every row demoted means the pool has been walked once, so the marks
        # describe an outage rather than any one account: they are stale and the
        # rows are equally good again. Only THIS tier's rows are cleared -- the
        # cascade calls this once per credential tier with a different subset,
        # so popping the whole provider would let a one-row tier wipe the marks
        # belonging to rows it never saw.
        #
        # NOTE this branch is not reachable from :meth:`_resolve`'s cascade:
        # ``_usable_key_rows`` drops demoted rows before this method is called,
        # so an all-demoted tier arrives here as an EMPTY list and returns at the
        # guard above. It is retained for direct callers that order a row set
        # themselves, and it is NOT the cascade's safety net -- that is the
        # ``ignore_demotions`` second pass in :meth:`_resolve`. Reasoning about
        # the cascade's all-demoted behaviour from here gives the wrong answer.
        if not preferred:
            # Not under ``read_only``: clearing the marks is a routing DECISION,
            # and an isolated request running beside a user's turn must not be
            # able to move that turn's account. It still gets the same order --
            # all rows demoted means no reordering either way -- so the only
            # difference is that it decides nothing, which is the contract.
            if not read_only:
                self.clear_deprioritized(provider, [r.id for r in ordered])
            return ordered
        return preferred + [r for r in ordered if r.id in demoted]

    def _base_selection_order(
        self, rows: list[StoredCredential], provider: str, session_id: str | None
    ) -> list[StoredCredential]:
        """Stickiness, then a per-session hash, then round-robin."""
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
        self,
        provider: str,
        credential_type: str,
        source: str | None,
        *,
        ignore_demotions: bool = False,
    ) -> list[StoredCredential]:
        rows = [
            r
            for r in self.list_credentials(provider)
            if r.credential_type == credential_type and not self.is_blocked(r.id, provider)
        ]
        if source is not None:
            rows = [r for r in rows if r.data.get("source") == source]
        # Drop demoted rows from the TIER, not merely sort them last, when some
        # other credential is still reachable.
        #
        # Ordering alone is not enough here, because the cascade is a sequence
        # of tiers and a tier is consulted whole: an OAuth row, or an api_key
        # row with `source="login"`, wins its tier before a row in a later tier
        # is ever looked at. So a demoted row that is ALONE in its tier kept
        # winning the cascade -- and `rotate_sibling` kept reporting that a
        # sibling existed, which told the driver rotation was progressing while
        # the same failing bearer came back every time. A healthy credential one
        # tier down never received a single request.
        #
        # Dropping is safe WITHOUT a "is anything else reachable?" guard, and
        # deliberately has none. Such a guard could only count database ROWS,
        # while the cascade also resolves from the env var (tier 5) and the
        # fallback resolver (tier 7), which are not rows: a demoted lone stored
        # row would then never yield, and an exported ANTHROPIC_API_KEY beside a
        # signed-in account became unreachable where it used to be the fallback.
        #
        # The safety net for the resulting empty tier is the second pass at the
        # end of :meth:`_resolve`: if demotions are the ONLY reason the whole
        # cascade came back empty, it resolves once more with
        # ``ignore_demotions``, so a demoted lone row is still served rather
        # than reported as no credential at all. ``_selection_order``'s
        # all-demoted branch cannot be that net, because this filter runs first
        # and hands it an empty list.
        demoted = set() if ignore_demotions else self._active_demotions(provider)
        if demoted:
            remaining = [r for r in rows if r.id not in demoted]
            if remaining:
                return remaining
            # Every row in THIS tier is demoted: yield the tier so the cascade
            # moves on to whatever comes next -- another tier, the env var, or
            # the resolver.
            return []
        return rows

    # -- the cascade ---------------------------------------------------------

    async def get_api_key(
        self,
        provider: str,
        session_id: str | None = None,
        *,
        force_refresh: bool = False,
        read_only: bool = False,
    ) -> str | None:
        """Resolve the API key for ``provider`` via the 7-step cascade.

        ``read_only`` makes the resolve decide nothing about routing — see
        :meth:`_resolve`.
        """
        key, _row = await self._resolve(
            provider, session_id, force_refresh=force_refresh, read_only=read_only
        )
        return key

    async def get_oauth_access(
        self,
        provider: str,
        session_id: str | None = None,
        *,
        force_refresh: bool = False,
        read_only: bool = False,
    ) -> OAuthAccess | None:
        """The identity-carrying record for wire clients.

        Returns :class:`OAuthAccess` for whichever credential the cascade
        picks — ``kind == "oauth"`` with account/org identity when an OAuth
        row wins, ``kind == "api_key"`` otherwise. Runtime/config overrides
        deliberately short-circuit to ``None`` (they aim at gateways where
        stored identity does not apply).

        ``read_only`` resolves without blocking a credential or moving session
        stickiness, for a decorative call running beside a live turn — see
        :meth:`_resolve` and
        :attr:`~local_operator.harness.types.ChatRequest.isolated`.
        """
        if self._runtime_overrides.get(provider) or self._config_overrides.get(provider):
            return None
        key, row = await self._resolve(
            provider, session_id, force_refresh=force_refresh, read_only=read_only
        )
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
                raw=data,
            )
        return OAuthAccess(
            access_token=key, credential_id=row.id if row is not None else 0, kind="api_key"
        )

    async def list_oauth_accesses(self, provider: str) -> list[OAuthAccess]:
        """EVERY logged-in OAuth account for ``provider``, each one refreshed.

        :meth:`get_oauth_access` answers "which account will the next request
        run as", and that is the right question for the wire. It is the wrong
        question for a usage report: quota is per account, so a user with two
        accounts on one provider has two answers and the cascade can only ever
        return one of them. Worse, with no ``session_id`` the cascade's
        selection order ROUND-ROBINS, so the single account that got reported
        was not even stable between refreshes.

        Four differences from the cascade, all deliberate, and all the same
        principle: routing decisions must not become reporting decisions.

        - **Blocked credentials are INCLUDED.** ``_usable_key_rows`` drops rows
          under a backoff, which is right for "where do I send this request"
          and exactly wrong here — the commonest reason a credential is blocked
          is that it ran out of quota, so the account a user most needs to see
          on a usage screen was the one guaranteed to be missing from it. Its
          exhausted window IS the explanation for the block.
        - **Stable order, by row id.** Enumeration must not depend on which
          request happened last, or a list of accounts reshuffles itself while
          the user is reading it.
        - **No stickiness.** ``_set_sticky`` pins which credential a SESSION
          transacts on. Reading a quota must not repoint the session's account
          as a side effect.
        - **No blocking on refresh failure.** A routing resolve blocks a row that
          fails to refresh so it can rotate to a sibling for the request in
          hand. Here the row is simply omitted: taking a credential out of
          service is a routing decision, and a read is not entitled to make it.
          The last two are the same principle ``_resolve``'s ``read_only`` mode
          applies to a decorative REQUEST, which needs a bearer but is likewise
          not entitled to route the session.

        Logged-out rows are still excluded — ``list_credentials`` filters on
        ``disabled_cause``, and an account the user signed out of is genuinely
        not theirs to report on.

        Overrides short-circuit for the same reason they do in
        :meth:`get_oauth_access` — they aim at a gateway, where stored identity
        does not apply.
        """
        if self._runtime_overrides.get(provider) or self._config_overrides.get(provider):
            return []
        definition: ProviderDefinition | None = get_provider_definition(provider)
        key_fn = definition.get_api_key if definition else None
        rows = [r for r in self.list_credentials(provider) if r.credential_type == "oauth"]
        accesses: list[OAuthAccess] = []
        for row in sorted(rows, key=lambda r: r.id):
            try:
                creds = await self._ensure_oauth_fresh(row)
            except AuthStoreError:
                logger.debug(
                    "usage: credential %s for %s failed to refresh; omitting",
                    row.id,
                    provider,
                    exc_info=True,
                )
                continue
            key = key_fn(creds) if key_fn else creds.get("access")
            if not key:
                continue
            accesses.append(
                OAuthAccess(
                    access_token=key,
                    credential_id=row.id,
                    account_id=creds.get("account_id"),
                    email=creds.get("email"),
                    org_id=creds.get("org_id"),
                    api_endpoint=creds.get("api_endpoint"),
                    kind="oauth",
                    raw=creds,
                )
            )
        return accesses

    async def _resolve(
        self,
        provider: str,
        session_id: str | None,
        *,
        force_refresh: bool = False,
        read_only: bool = False,
        ignore_demotions: bool = False,
    ) -> tuple[str | None, StoredCredential | None]:
        """The 7-step cascade; returns ``(key, winning row or None)``.

        ``ignore_demotions`` runs the cascade as if no credential were demoted.
        It is set only by this method's own second pass (see the tail), where
        demotions have been found to be the sole reason the cascade came back
        empty. Because the second pass sets it, the tail's branch cannot re-arm
        and the recursion terminates at depth two.

        ``read_only`` resolves WITHOUT making any routing decision: no
        credential blocked when its refresh fails, no session stickiness
        written and none cleared. It exists for a request that runs beside a
        user's turn and must not be able to move that turn's account — see
        :attr:`~local_operator.harness.types.ChatRequest.isolated`. The cascade
        still READS stickiness, so a read-only resolve lands on the same
        credential the turn is transacting on, which is the point. A successful
        OAuth refresh still persists the rotated token: that is the same
        account's own bookkeeping, not a decision about where requests go, and
        dropping it would throw away a single-use refresh token.
        """
        definition: ProviderDefinition | None = get_provider_definition(provider)

        def pin(credential_id: int | None) -> None:
            """Write (or, with ``None``, clear) session stickiness — unless this
            resolve is read-only, in which case it is not ours to move."""
            if not read_only:
                self._set_sticky(provider, session_id, credential_id)

        # 1. Runtime override
        runtime = self._runtime_overrides.get(provider)
        if runtime:
            return runtime, None

        # 2. Config override
        config = self._config_overrides.get(provider)
        if config:
            return config, None

        # 3. OAuth credential
        oauth_rows = self._usable_key_rows(
            provider, "oauth", source=None, ignore_demotions=ignore_demotions
        )
        for row in self._selection_order(oauth_rows, provider, session_id, read_only=read_only):
            try:
                creds = await self._ensure_oauth_fresh(row, force=force_refresh)
            except AuthStoreError:
                if not read_only:
                    self.block_credential(row.id, provider)  # try a sibling
                continue
            key_fn = definition.get_api_key if definition else None
            key = key_fn(creds) if key_fn else creds.get("access")
            if key:
                pin(row.id)
                refreshed = self.get_credential(row.id)
                return key, refreshed or row
        if oauth_rows and force_refresh:
            # Every sibling failed its refresh — surface the failure so the
            # failover layer can block/back off instead of silently looping.
            raise AuthStoreError(f"All OAuth credentials for '{provider}' failed to refresh")
        # PR-15: with NO oauth rows, force_refresh falls through to tiers 4-7.

        # 4. API key persisted by interactive login
        login_rows = self._usable_key_rows(
            provider, "api_key", source="login", ignore_demotions=ignore_demotions
        )
        for row in self._selection_order(login_rows, provider, session_id, read_only=read_only):
            key = row.data.get("key")
            if key:
                pin(row.id)
                return key, row

        # Leaving the OAuth tier: clear session stickiness so identity
        # attribution stops for non-OAuth requests (PR-16; cleared before
        # step 5, regardless of which later tier ends up winning).
        pin(None)

        # 5. Env var tier (process env, then legacy credentials.env).
        env_key = self._env_api_key(provider)
        if env_key:
            return env_key, None

        # 6. Stored api_key without source="login" (e.g. broker migration)
        stored_rows = [
            row
            for row in self._usable_key_rows(
                provider, "api_key", source=None, ignore_demotions=ignore_demotions
            )
            if row.data.get("source") != "login"
        ]
        for row in self._selection_order(stored_rows, provider, session_id, read_only=read_only):
            key = row.data.get("key")
            if key:
                pin(row.id)
                return key, row
        # 7. Fallback resolver
        resolver = self._fallback_resolvers.get(provider)
        if resolver is not None:
            return resolver(provider), None

        # Nothing in the whole cascade -- but demotions are a ROUTING
        # preference, never a statement that a credential is unusable. If they
        # are the only reason this came back empty, they have outlived their
        # purpose (there is nowhere else to route to), so clear them and resolve
        # once more. Without this a sole demoted credential resolved to None and
        # the caller was told no credential was configured, which is exactly the
        # misdiagnosis this change set out to remove.
        #
        # A ``read_only`` resolve takes this pass too. It must: dropping a
        # demoted row from its tier is the destructive half of demotion, and a
        # resolve that is forbidden from deciding anything about routing cannot
        # be handed that half alone -- it would report "no credential" for a
        # credential that is merely deprioritised, which is the misdiagnosis at
        # issue. What it does NOT do is clear the marks: that is the routing
        # decision, and it stays reserved for the caller who owns the turn.
        # ``ignore_demotions`` gives the same answer without touching state.
        if not ignore_demotions and self._active_demotions(provider):
            if not read_only:
                self.clear_deprioritized(provider)
            return await self._resolve(
                provider,
                session_id,
                force_refresh=force_refresh,
                read_only=read_only,
                ignore_demotions=True,
            )

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
            is_server_side_failure,
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
        # A provider-wide fault (5xx/529 overload, timeout) is not evidence
        # against the CREDENTIAL. Blocking it would take a healthy account out
        # of the pool for a minute because the provider had a bad second, and
        # under a sustained outage that walks the whole pool into the blocked
        # state until the session has nothing left to try -- while every one of
        # those accounts would have served the very next request. So the row is
        # left usable and only the sticky pointer moves, which is enough to send
        # THIS attempt to a sibling.
        server_side = is_server_side_failure(error)
        if failing is not None:
            if server_side:
                self.deprioritize_credential(provider, failing.id)
            else:
                retry_after = retry_after_ms_from_error(error)
                self.block_credential(
                    failing.id, provider, block_ms=max(block_ms, retry_after or 0)
                )
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
        if siblings:
            return True
        # No untried sibling of the SAME TYPE remains -- which is not the same
        # as "nothing else is reachable". The cascade has other tiers: another
        # credential type, the env var, the fallback resolver. Clearing the
        # demotion here erased the mark in the very call that set it whenever
        # the failing row had no same-type sibling (an OAuth account beside a
        # pasted key -- precisely the shape a Z.AI sign-in creates), so tier 3
        # re-served the identical failing row and the healthy credential one
        # tier down was never asked.
        #
        # So the mark STANDS. It is not permanent: it expires on its TTL, it is
        # cleared when the credential next serves a request, and
        # `_selection_order` drops the whole set as stale once every row it sees
        # is demoted. Any of those returns this credential to service; none of
        # them requires pretending here that the fault never happened.
        return False

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
