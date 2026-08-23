"""Shared on-disk cache for provider usage reports.

``/usage`` used to cross the network once per logged-in provider, every time,
sequentially — four logged-in providers meant four serial round trips before
the panel had anything to show. The quota data itself moves slowly (rolling
windows that reset hours apart), so the answer is a cache with a TTL measured
in minutes, not seconds.

The cache lives in ``~/.local-operator/usage_cache.db`` (0600, WAL, busy
timeout — the same discipline as ``auth.db``) rather than in process memory
because a user typically runs SEVERAL lop sessions at once (one per cmux
workspace). An in-process cache would make every session pay the full fetch
cost; a shared one means the first session to refresh populates the answer for
all of them, and a lease table ensures that when several sessions notice an
expired entry at the same moment, only ONE of them actually hits the provider.
The others serve the last good value, which is exactly as fresh — it is the
same row.

Semantics (modelled on omp's ``AuthStorage`` usage cache, which solved the
same problem for the same providers):

- **Fresh entry** (``expires_at`` in the future): served as-is, no network.
- **Expired entry**: one process wins the lease and fetches; everyone else
  serves the stale row rather than joining a synchronized fan-out. Anthropic
  and OpenAI rate-limit their usage endpoints per source IP regardless of
  account, so N sessions refreshing in lockstep is how a quota screen earns a
  429 of its own.
- **Fetch failure**: the winner writes the last good value back with a short
  cool-down instead of dropping the provider from the report. A transient
  network blip must not make an account's quota unreadable.
- **TTL jitter**: each entry's lifetime is spread ±25% around the base TTL so
  several accounts on one provider do not all expire into the same refresh
  window (the same per-IP burst, one cycle later).

Cache keys are ``provider:<account fingerprint>``: the fingerprint summarises
WHICH accounts the provider was fetched for (the identity keys of the stored
credential rows, plus any env key), and is computed synchronously from the
credential store — no OAuth refresh, no network. Folding the account set into
the key makes login/logout self-invalidating, and keying on the account rather
than the access token matters because OAuth tokens rotate on every refresh:
a key that named the token would expire exactly as often as the token rotates,
which is to say, constantly. Usage is per account, not per token.

Everything here degrades silently: a cache that cannot be read or written
falls back to the uncached behaviour (fetch live, show live). The cache is an
accelerator, never a dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport

logger = logging.getLogger("local_operator.providers.usage_cache")

#: Base time-to-live for a successful report. Five minutes matches omp: long
#: enough that back-to-back ``/usage`` calls and every concurrent session read
#: the same row, short enough that a rolling window's approach to its cap is
#: still visible while it matters. Jittered ±25% per entry (see module doc).
USAGE_REPORT_TTL_MS = 5 * 60_000

#: Cool-down written after a fetch failure, over the LAST GOOD value. The
#: failure is remembered briefly so a dead endpoint is not re-probed on every
#: keystroke, while the numbers themselves stay readable.
USAGE_FAILURE_BACKOFF_MS = 10_000

#: How long a last-good row survives past its expiry, for stale serving and
#: failure fallback. One day: longer than any rolling window the fetchers
#: report, so a row old enough to be interesting is also old enough to be
#: wrong, and the disk stays bounded.
USAGE_LAST_GOOD_RETENTION_MS = 24 * 60 * 60_000

#: Cross-process fetch lease lifetime. A refresh for these endpoints takes at
#: most ~10 s (the fetchers' own timeout); a lease comfortably longer than
#: that covers a slow response without stranding the row when a process dies
#: mid-fetch (it expires, and the next session fetches).
USAGE_FETCH_LEASE_MS = 30_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_reports (
  key TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  payload TEXT NOT NULL,
  fetched_at_ms INTEGER NOT NULL,
  expires_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_reports_provider ON usage_reports(provider);

CREATE TABLE IF NOT EXISTS usage_fetch_leases (
  key TEXT PRIMARY KEY,
  holder TEXT NOT NULL,
  expires_at_ms INTEGER NOT NULL
);
"""


def default_cache_path() -> Path:
    return config_dir() / "usage_cache.db"


# ---------------------------------------------------------------------------
# Wire format: UsageReport <-> JSON-safe dict
# ---------------------------------------------------------------------------


def report_to_dict(report: UsageReport) -> dict[str, Any]:
    """Serialize a report to a JSON-safe dict.

    The report shape is plain dataclasses of scalars, so ``asdict`` is the
    whole job; the explicit function exists so the cache's wire format has ONE
    named definition to version if the shape ever grows.
    """
    return asdict(report)


def report_from_dict(data: Any) -> UsageReport | None:
    """Rebuild a report from a cached dict, or None when it no longer parses.

    Defensive on purpose: the cache outlives the process that wrote it, and a
    schema change between versions must read as a cache MISS — never an
    exception on the ``/usage`` path.
    """
    if not isinstance(data, dict):
        return None
    try:
        limits = [
            UsageLimit(
                id=str(limit["id"]),
                label=str(limit["label"]),
                amount=UsageAmount(**limit["amount"]),
                window=str(limit.get("window", "")),
                status=limit.get("status"),
                resets_at=limit.get("resets_at"),
                resets_at_ms=limit.get("resets_at_ms"),
                tier=str(limit.get("tier", "")),
                shared=bool(limit.get("shared", False)),
            )
            for limit in data.get("limits", [])
        ]
        return UsageReport(
            provider=str(data["provider"]),
            fetched_at=int(data.get("fetched_at", 0)),
            limits=limits,
            notes=data.get("notes"),
            identity=data.get("identity"),
        )
    except Exception:  # noqa: BLE001 — a bad row is a miss, not a failure
        logger.debug("usage cache: unparseable report row dropped", exc_info=True)
        return None


def provider_cache_key(provider: str, account_fingerprint: str) -> str:
    """The cache key for one provider's full usage report set.

    ``account_fingerprint`` summarises WHICH accounts the row was fetched for
    (the sorted identity keys of the stored credentials, or a hash of the API
    key on the key route). Folding the account set into the key is what makes
    login/logout self-invalidating: the moment the set of logged-in accounts
    changes, the fingerprint changes, the old row stops matching, and the next
    read fetches fresh instead of rendering an account the user no longer owns
    (or hiding one they just added).
    """
    return f"{provider}:{account_fingerprint}"


def fingerprint_accounts(identities: list[str]) -> str:
    """A stable fingerprint for a set of account identities.

    Sorted so enumeration order (row id, round-robin) never changes the key —
    the same accounts must hit the same row no matter which session asks.
    """
    if not identities:
        return "none"
    joined = "\n".join(sorted(identities))
    return hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()[:16]


def fingerprint_secret(secret: str) -> str:
    """Fingerprint for the API-key route: a hash of the key, never the key.

    The key bytes must not sit in the row's primary key (or the WAL) any
    longer than they must; a digest identifies the account just as well for
    cache-hit purposes.
    """
    return "key:" + hashlib.sha256(secret.encode("utf-8", "replace")).hexdigest()[:16]


def account_preflight_key(storage_id: str, account_identity: str) -> str:
    """Cache key for ONE account's preflight usage probe.

    A namespace disjoint from the warmer's per-provider-set keys: the ``:pf:``
    segment appears in no ``provider_cache_key`` output, so a preflight row can
    never collide with a warmer row even when a provider has exactly one account
    (where ``fingerprint_accounts([id])`` would otherwise match). Kept separate on
    purpose — see the module note in the spec: the two subsystems cache different
    shapes (one account vs the full set) with different freshness policies, and the
    raw preflight fetch does not backfill ``UsageReport.identity``, so the warmer's
    set-row cannot be reliably sliced per account anyway.

    ``account_identity`` is a NON-SECRET stable identifier (email, account id, or a
    ``fingerprint_secret`` digest for the API-key route — never a raw key or a
    rotating OAuth token). It is hashed again here so no identity string sits in
    the row's primary key, matching the warmer's discipline.
    """
    return f"{storage_id}:pf:{fingerprint_accounts([account_identity])}"


class UsageCacheStore:
    """SQLite-backed shared usage cache; every method is exception-safe."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else default_cache_path()
        #: Identity of this process in lease rows, so a release only frees a
        #: lease THIS process took.
        self._holder = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._conn: sqlite3.Connection | None = None

    # -- connection ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection | None:
        """Open (once) and return the connection, or None when unavailable.

        A cache that cannot open is a permanent miss, not an error: the caller
        fetches live. The alternative — raising — would make a read-only home
        directory or a locked file take ``/usage`` down entirely.
        """
        if self._conn is not None:
            return self._conn
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            # Create the file 0600 BEFORE sqlite opens it (same rule as
            # auth.db): quota rows carry account identity, and the
            # connect-then-chmod pattern leaves a readable window.
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
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
            self._conn = conn
            return conn
        except Exception:  # noqa: BLE001 — cache unavailable = cache miss
            logger.debug("usage cache: cannot open %s", self._db_path, exc_info=True)
            return None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # -- reads ---------------------------------------------------------------
    def get(self, key: str, *, include_expired: bool = False) -> list[UsageReport] | None:
        """The cached report list for ``key``, or None when absent/expired/bad.

        With ``include_expired`` this is the stale-serving half of the design:
        a row past its TTL is still the freshest answer on hand while a refresh
        is in flight elsewhere or has failed.
        """
        conn = self._connect()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT payload, expires_at_ms FROM usage_reports WHERE key = ?", (key,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            logger.debug("usage cache: read failed", exc_info=True)
            return None
        if row is None:
            return None
        payload, expires_at_ms = row
        if not include_expired and int(expires_at_ms) <= self._now_ms():
            return None
        try:
            data = json.loads(payload)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(data, list):
            return None
        reports = [report_from_dict(item) for item in data]
        reports = [report for report in reports if report is not None]
        return reports

    def expiry_ms(self, key: str) -> int | None:
        """When the cached row for ``key`` expires, or None when absent."""
        conn = self._connect()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT expires_at_ms FROM usage_reports WHERE key = ?", (key,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            return None
        return int(row[0]) if row is not None else None

    def fetched_at_ms(self, key: str) -> int | None:
        """When the cached row for ``key`` was fetched, or None when absent.

        Unlike :meth:`get`, this answers for EMPTY rows too (a provider that
        legitimately reports no quota): the warmer asks "how old is the answer",
        and "no quota, checked a minute ago" IS an answer.
        """
        conn = self._connect()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT fetched_at_ms FROM usage_reports WHERE key = ?", (key,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            return None
        return int(row[0]) if row is not None else None

    # -- writes ----------------------------------------------------------------
    def set(
        self,
        key: str,
        provider: str,
        reports: list[UsageReport],
        *,
        expires_at_ms: int,
    ) -> None:
        """Write one provider's report list, then opportunistically prune."""
        conn = self._connect()
        if conn is None:
            return
        try:
            payload = json.dumps([report_to_dict(report) for report in reports])
            now = self._now_ms()
            fetched_at = max((report.fetched_at for report in reports), default=now)
            conn.execute(
                """
                INSERT INTO usage_reports (
                  key, provider, payload, fetched_at_ms, expires_at_ms, updated_at_ms
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  provider = excluded.provider,
                  payload = excluded.payload,
                  fetched_at_ms = excluded.fetched_at_ms,
                  expires_at_ms = excluded.expires_at_ms,
                  updated_at_ms = excluded.updated_at_ms
                """,
                (key, provider, payload, fetched_at, expires_at_ms, now),
            )
            conn.commit()
            self._cleanup(conn, now)
        except Exception:  # noqa: BLE001 — a failed write is a future miss
            logger.debug("usage cache: write failed", exc_info=True)

    def write_failure(self, key: str, provider: str) -> list[UsageReport] | None:
        """Record a failed fetch; return the last good reports if any exist.

        The cool-down row keeps the failure warm briefly (no re-probe per
        keystroke) while the stale values stay servable — the caller renders
        the returned reports exactly as if they had been fetched.
        """
        last_good = self.get(key, include_expired=True)
        if last_good is not None:
            self.set(
                key,
                provider,
                last_good,
                expires_at_ms=self._now_ms() + USAGE_FAILURE_BACKOFF_MS,
            )
        return last_good

    def _cleanup(self, conn: sqlite3.Connection, now_ms: int) -> None:
        """Drop rows whose retention has passed. Cheap: one indexed scan.

        ``with conn:`` is load-bearing, not style. Python's sqlite3 opens an
        implicit transaction on the first write and leaves it OPEN until
        something commits — so bare DELETEs here held the WAL write lock from
        one refresh to the next, and every OTHER session's cache call blocked
        for the full 5 s busy timeout and then failed: leases collapsed to
        "everyone fetches" and writes were silently lost, each stall freezing
        that session's whole TUI (these calls are synchronous inside async
        workers). The context manager commits on exit, releasing the lock the
        moment the pruning is done.
        """
        try:
            with conn:
                conn.execute(
                    "DELETE FROM usage_reports WHERE expires_at_ms < ?",
                    (now_ms - USAGE_LAST_GOOD_RETENTION_MS,),
                )
                conn.execute("DELETE FROM usage_fetch_leases WHERE expires_at_ms < ?", (now_ms,))
        except Exception:  # noqa: BLE001
            pass

    # -- leases ----------------------------------------------------------------
    def try_lease(self, key: str, ttl_ms: int = USAGE_FETCH_LEASE_MS) -> bool:
        """Take the fetch lease for ``key`` if it is free (or expired).

        True means THIS process is the one that should fetch. False means a
        peer session already owns the refresh; the caller serves its stale row
        instead of joining the fan-out. The lease expires on its own, so a
        process that dies mid-fetch cannot strand the row forever.
        """
        conn = self._connect()
        if conn is None:
            # No shared coordination available: every process fetches. That is
            # the pre-cache behaviour, and correctness does not depend on the
            # lease — only efficiency does.
            return True
        now = self._now_ms()
        expires = now + ttl_ms
        try:
            # ONE atomic statement, judged by rowcount. A SELECT-then-INSERT
            # pair is not atomic here even inside `with conn:` — Python's
            # sqlite3 defers BEGIN until the first write, so two processes can
            # both read "no live lease" and both take it, precisely at the
            # synchronized-expiry moment the lease guards. The conditional
            # upsert moves the check into the same write: the UPDATE arm only
            # fires when the standing lease has expired, so exactly one
            # process's statement reports a changed row.
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO usage_fetch_leases (key, holder, expires_at_ms)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                      holder = excluded.holder,
                      expires_at_ms = excluded.expires_at_ms
                    WHERE usage_fetch_leases.expires_at_ms <= ?
                    """,
                    (key, self._holder, expires, now),
                )
            return cursor.rowcount > 0
        except Exception:  # noqa: BLE001 — coordination failure = go fetch
            logger.debug("usage cache: lease failed", exc_info=True)
            return True

    def release_lease(self, key: str) -> None:
        """Free the lease if this process holds it (a fetch finished).

        Released rather than left to expire so the NEXT session to notice the
        stale row a moment later can refresh immediately instead of waiting out
        the lease TTL. Only the holder releases: freeing a peer's lease would
        re-open the very fan-out the lease prevents.
        """
        conn = self._connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute(
                    "DELETE FROM usage_fetch_leases WHERE key = ? AND holder = ?",
                    (key, self._holder),
                )
        except Exception:  # noqa: BLE001
            logger.debug("usage cache: lease release failed", exc_info=True)


async def leased_account_usage(
    store: "UsageCacheStore | None",
    key: str,
    provider: str,
    fetch: Callable[[], Awaitable["UsageReport | None"]],
) -> "UsageReport | None":
    """One account's usage for a ROUTING decision: fetch live, but collapse a
    concurrent peer's duplicate fetch.

    Unlike the display read-through (:meth:`ProviderController._refresh_provider_usage`)
    this NEVER serves a fresh cached value on the fast path — a routing probe must be
    free to notice recovery/depletion on its own next boundary. The cache does exactly
    one job here: when a PEER process already holds the fetch lease for this account,
    this process serves the peer's last-good value instead of crossing the network for
    the identical answer (Anthropic/OpenAI rate-limit the usage endpoint per source IP,
    so a synchronized fan-out is how a routing check earns its own 429).

    Degrades to the pre-cache behaviour whenever coordination is unavailable: no store,
    or a lease it could not take with nothing on hand, means fetch live. ``fetch``
    returning ``None`` (transport/HTTP failure OR no quota) is passed through unchanged,
    so the caller's fail-open path is preserved exactly.
    """
    if store is None:
        return await fetch()

    stale = store.get(key, include_expired=True)
    stale_report = stale[0] if stale else None

    # The lease is contended only when a peer is mid-fetch. A free lease (the common
    # case, and ALWAYS the case in a single-process test) means: this process fetches.
    if not store.try_lease(key):
        # A peer owns the in-flight fetch; its result lands in this same row for the
        # next boundary. Serve what we have rather than doubling the network hit. With
        # nothing on hand (cold start) we cannot serve nothing — fetch live, matching
        # the controller's "the lease only protects a stale value" rule.
        return stale_report if stale_report is not None else await fetch()

    try:
        report = await fetch()
        if report is not None:
            store.set(
                key,
                provider,
                [report],
                expires_at_ms=store._now_ms() + USAGE_REPORT_TTL_MS,
            )
            return report
        # None = failure or genuinely-empty. Keep the last-good value servable to a
        # concurrent lease-loser under the short failure cool-down; return None so the
        # ROUTING caller fails open exactly as it does today.
        store.write_failure(key, provider)
        return None
    finally:
        store.release_lease(key)
