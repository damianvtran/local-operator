"""SQLite-backed, parallel-safe store for token-consumption analytics.

Shape of the data. Every provider call across every ``lop`` session appends
one row to ``calls``. That table is the raw ledger; it is bounded by a rolling
retention window (old rows are pruned) so a machine that runs for months does
not accumulate an unbounded database. A per-call row is small — a dozen
integers and two short strings — so even a busy week is a few megabytes.

Why one shared database and not per-session files. Several sessions run at
once (one per cmux workspace), and the whole point of the feature is a
*universal* view. WAL mode plus a busy timeout makes concurrent writes from
different processes atomic and serialised by SQLite itself — the same
discipline ``providers/usage_cache.py`` and ``auth.db`` already rely on — so
"parallel safe" is a property of the engine, not something this module has to
reinvent with file locks.

Why aggregation is a query, not a running counter. Keeping live totals would
mean a read-modify-write on every call and a lock contended by every session.
Instead each call is an append (no contention beyond the WAL) and the
``/analytics`` screen runs a ``GROUP BY`` when it opens. On a bounded ledger
that scan is milliseconds, and it is paid only when someone actually looks.

Everything degrades silently: a store that cannot open is a no-op recorder and
an empty report, never an exception on a session's path.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    CallSnapshot,
    UsageAggregate,
    apportion_components,
)
from local_operator.paths import config_dir

logger = logging.getLogger("local_operator.analytics.store")

#: Default retention: keep 90 days of per-call rows. Long enough to see trends
#: ("where did usage go this month"), short enough to bound the file. Pruning
#: runs opportunistically on write, not on a timer.
DEFAULT_RETENTION_DAYS = 90

#: Bounded retry for a write that loses the lock race past ``busy_timeout``.
#: Runs on the background writer thread, so waiting a moment is free to the
#: session and buys accuracy under many-parallel-session contention.
_WRITE_RETRIES = 4
_WRITE_RETRY_BACKOFF_S = 0.05

#: One component column per COMPONENT_KEYS entry, holding the ESTIMATED token
#: attribution for that call. Storing the apportioned tokens (not just chars)
#: means the aggregate query is a plain SUM with no per-row arithmetic, and the
#: estimate a report shows is exactly the one recorded — reproducible after the
#: fact. Adding a component is a migration: bump the schema and backfill 0.
_COMPONENT_COLUMNS = ",\n  ".join(f"c_{key} INTEGER NOT NULL DEFAULT 0" for key in COMPONENT_KEYS)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  model_id TEXT NOT NULL,
  ok INTEGER NOT NULL DEFAULT 1,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens INTEGER NOT NULL DEFAULT 0,
  context_tokens INTEGER NOT NULL DEFAULT 0,
  {_COMPONENT_COLUMNS}
);
CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls(ts_ms);
CREATE INDEX IF NOT EXISTS idx_calls_session ON calls(session_id);
CREATE INDEX IF NOT EXISTS idx_calls_provider ON calls(provider);

-- Human-readable session names, so the per-session table can show a title
-- rather than a 12-hex id. Upserted opportunistically; absence just means the
-- report falls back to the id, never an error.
CREATE TABLE IF NOT EXISTS session_names (
  session_id TEXT PRIMARY KEY,
  name TEXT NOT NULL DEFAULT '',
  updated_at_ms INTEGER NOT NULL
);
"""

_CALL_COLUMNS = (
    "ts_ms",
    "session_id",
    "provider",
    "model_id",
    "ok",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "context_tokens",
    *(f"c_{key}" for key in COMPONENT_KEYS),
)

_INSERT_SQL = (
    f"INSERT INTO calls ({', '.join(_CALL_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _CALL_COLUMNS)})"
)


def default_db_path() -> Path:
    """The shared analytics database, next to the other per-user stores."""
    return config_dir() / "analytics.db"


def _row_values(snapshot: CallSnapshot) -> tuple[Any, ...]:
    """A snapshot as the positional tuple ``_INSERT_SQL`` expects.

    The estimated component split is computed HERE, in the writer, against the
    authoritative ``context_tokens`` — never on the event loop. A call the
    provider gave no context total for stores 0s for every component, which
    reads as "unknown" rather than a fabricated breakdown.
    """
    components = apportion_components(snapshot.component_chars, snapshot.context_tokens)
    return (
        snapshot.ts_ms,
        snapshot.session_id,
        snapshot.provider,
        snapshot.model_id,
        1 if snapshot.ok else 0,
        snapshot.input_tokens,
        snapshot.output_tokens,
        snapshot.cache_read_tokens,
        snapshot.cache_write_tokens,
        snapshot.reasoning_tokens,
        snapshot.context_tokens,
        *(components[key] for key in COMPONENT_KEYS),
    )


class AnalyticsStore:
    """Append-only ledger of provider calls; every method is exception-safe."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else default_db_path()
        self._retention_ms = max(1, int(retention_days)) * 24 * 60 * 60 * 1000
        #: One connection PER THREAD. SQLite connections are thread-bound, and
        #: this store is touched from two threads by design: the recorder's
        #: writer thread appends rows, and the event loop's thread reads the
        #: aggregate when ``/analytics`` opens. WAL lets those coexist across
        #: separate connections to the same file, so each thread gets its own
        #: rather than sharing one (which raises ``ProgrammingError``) or
        #: serialising every access behind a lock.
        self._local = threading.local()
        #: Set once opening fails, so a broken store stops retrying every call
        #: (a read-only home directory should cost one log line, not one per
        #: provider round trip for the life of the process). Shared across
        #: threads: if the file cannot be opened at all, no thread should keep
        #: trying.
        self._broken = False
        #: Guards the one-time schema creation so two threads opening their
        #: first connections at once do not race on ``executescript``.
        self._init_lock = threading.Lock()
        self._initialized = False

    # -- connection ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection | None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if self._broken:
            return None
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            # 0600 BEFORE sqlite opens it (same rule as auth.db / usage_cache):
            # per-call rows carry session ids and model identifiers, and the
            # connect-then-chmod pattern leaves a world-readable window.
            if not self._db_path.exists():
                fd = os.open(self._db_path, os.O_CREAT | os.O_WRONLY, 0o600)
                os.close(fd)
            conn = sqlite3.connect(str(self._db_path), timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            # Schema creation is idempotent (IF NOT EXISTS) but should run once,
            # under a lock, so two threads opening their first connections
            # simultaneously do not both executescript into the same file.
            with self._init_lock:
                conn.executescript(_SCHEMA)
                conn.commit()
                if not self._initialized:
                    for path in (
                        self._db_path,
                        self._db_path.with_suffix(self._db_path.suffix + "-wal"),
                        self._db_path.with_suffix(self._db_path.suffix + "-shm"),
                    ):
                        try:
                            os.chmod(path, 0o600)
                        except OSError:
                            pass
                    self._initialized = True
            self._local.conn = conn
            return conn
        except Exception:  # noqa: BLE001 — store unavailable = analytics off
            logger.debug("analytics: cannot open %s", self._db_path, exc_info=True)
            self._broken = True
            return None

    def close(self) -> None:
        """Close THIS thread's connection.

        Per-thread by design (see ``_connect``): a thread closes only its own
        handle. The writer thread's connection is closed when the recorder
        shuts down and calls this from that thread; a reader's connection is
        left to be reclaimed when its thread ends. SQLite forbids closing a
        connection from another thread, so a cross-thread close would raise —
        which is why this only touches the calling thread's handle.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._local.conn = None

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # -- writes --------------------------------------------------------------
    def record_batch(self, snapshots: Sequence[CallSnapshot]) -> int:
        """Insert a batch of calls in one transaction. Returns rows written.

        Batched because the writer thread drains the queue in bursts: N
        concurrent calls become one transaction and one fsync, not N.

        Retried on ``SQLITE_BUSY``. This runs on the recorder's BACKGROUND
        thread, never on a session's event loop, so a few hundred milliseconds
        spent waiting out another process's write lock costs a session nothing
        — and it is what makes the ledger accurate under the load this feature
        is built for: several parallel ``lop`` sessions ending turns at once,
        all writing to one file. ``busy_timeout`` already blocks inside SQLite
        for up to 5s per attempt; the bounded retry on top covers the rare case
        where a lock hand-off still surfaces as BUSY. Only a genuinely wedged
        database (every attempt exhausted) drops the batch — best-effort to the
        end, but accuracy first while the write stays cheap, exactly as asked.
        """
        if not snapshots:
            return 0
        conn = self._connect()
        if conn is None:
            return 0
        rows = [_row_values(s) for s in snapshots]
        for attempt in range(_WRITE_RETRIES):
            try:
                conn.executemany(_INSERT_SQL, rows)
                conn.commit()
                return len(snapshots)
            except sqlite3.OperationalError as exc:
                # "database is locked" / "database is busy": another writer holds
                # the lock past our busy_timeout. Roll back and retry with a
                # short backoff rather than dropping rows a slightly longer wait
                # would have saved.
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
                if "lock" not in str(exc).lower() and "busy" not in str(exc).lower():
                    logger.debug("analytics: batch insert failed", exc_info=True)
                    return 0
                if attempt == _WRITE_RETRIES - 1:
                    logger.debug("analytics: batch dropped after %d busy retries", _WRITE_RETRIES)
                    return 0
                time.sleep(_WRITE_RETRY_BACKOFF_S * (attempt + 1))
            except Exception:  # noqa: BLE001 — a lost batch must not kill the writer
                logger.debug("analytics: batch insert failed", exc_info=True)
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
                return 0
        return 0

    def upsert_session_name(self, session_id: str, name: str) -> None:
        """Record (or update) a session's human name for the per-session table."""
        if not session_id:
            return
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO session_names (session_id, name, updated_at_ms) "
                "VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "name=excluded.name, updated_at_ms=excluded.updated_at_ms",
                (session_id, name or "", self._now_ms()),
            )
            conn.commit()
        except Exception:  # noqa: BLE001
            logger.debug("analytics: session name upsert failed", exc_info=True)

    def prune(self, *, now_ms: int | None = None) -> int:
        """Delete rows older than the retention window. Returns rows removed."""
        conn = self._connect()
        if conn is None:
            return 0
        cutoff = (now_ms if now_ms is not None else self._now_ms()) - self._retention_ms
        try:
            cur = conn.execute("DELETE FROM calls WHERE ts_ms < ?", (cutoff,))
            conn.commit()
            return cur.rowcount or 0
        except Exception:  # noqa: BLE001
            logger.debug("analytics: prune failed", exc_info=True)
            return 0

    def _read_connection(self) -> sqlite3.Connection | None:
        """A FRESH, short-lived connection for a read, or None when unavailable.

        Reads deliberately do not reuse the cached per-thread connection the
        writes use. This store is written from a background thread and read
        from the event-loop thread, and in WAL a long-lived reader connection
        can hold a snapshot that predates the writer's latest commit — the
        reader would then show stale (or empty) totals until it happened to
        start a new read transaction. A fresh connection per ``aggregate`` call
        always sees the newest committed state, and the read is infrequent (a
        report opening, not a hot path), so the connect cost is irrelevant.

        The file already exists by read time in every real path (a read only
        matters once something has been written), but ``_connect`` is called
        first so a first-ever read still creates the schema rather than raising
        on a missing table.
        """
        if self._connect() is None:
            return None
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            return conn
        except Exception:  # noqa: BLE001 — a read that cannot open is empty
            logger.debug("analytics: cannot open read connection", exc_info=True)
            return None

    # -- reads ---------------------------------------------------------------
    def _session_names(self, conn: sqlite3.Connection) -> dict[str, str]:
        try:
            rows = conn.execute("SELECT session_id, name FROM session_names").fetchall()
        except Exception:  # noqa: BLE001
            return {}
        return {str(sid): str(name) for sid, name in rows if name}

    def aggregate(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        session_id: str | None = None,
    ) -> UsageAggregate:
        """Sum the ledger into one :class:`UsageAggregate`.

        Optionally scoped to a time window and/or a single session. The result
        carries flat totals, a per-provider breakdown, and a per-session
        breakdown (each a one-level :class:`UsageAggregate`) so the report can
        render every table it needs from a single call. An unopenable or empty
        store returns a zeroed aggregate, which the screen renders as "no data
        yet" rather than an error.
        """
        conn = self._read_connection()
        if conn is None:
            return UsageAggregate()

        where: list[str] = []
        params: list[Any] = []
        if since_ms is not None:
            where.append("ts_ms >= ?")
            params.append(int(since_ms))
        if until_ms is not None:
            where.append("ts_ms < ?")
            params.append(int(until_ms))
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        clause = (" WHERE " + " AND ".join(where)) if where else ""

        component_sum = ", ".join(f"SUM(c_{key})" for key in COMPONENT_KEYS)
        base_cols = (
            "COUNT(*), SUM(ok), SUM(input_tokens), SUM(output_tokens), "
            "SUM(cache_read_tokens), SUM(cache_write_tokens), "
            "SUM(reasoning_tokens), SUM(context_tokens)"
        )
        try:
            top = conn.execute(
                f"SELECT {base_cols}, {component_sum} FROM calls{clause}", params
            ).fetchone()
            per_provider = conn.execute(
                f"SELECT provider, {base_cols}, {component_sum} FROM calls{clause} "
                "GROUP BY provider",
                params,
            ).fetchall()
            per_session = conn.execute(
                f"SELECT session_id, {base_cols}, {component_sum} FROM calls{clause} "
                "GROUP BY session_id",
                params,
            ).fetchall()
            names = self._session_names(conn)
        except Exception:  # noqa: BLE001 — a report must never raise
            logger.debug("analytics: aggregate query failed", exc_info=True)
            return UsageAggregate()
        finally:
            # The read connection is short-lived (see ``_read_connection``);
            # close it so a report open does not leak a handle per call.
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

        result = _aggregate_from_row(top)
        result.by_provider = {
            str(row[0]): _aggregate_from_row(row[1:]) for row in per_provider if row[0]
        }
        for row in per_session:
            sid = str(row[0])
            agg = _aggregate_from_row(row[1:])
            # Stash the human name (when known) on the id key's aggregate via a
            # side map the caller reads; kept on the object would widen the
            # dataclass for one table, so the report reads names from here.
            result.by_session[sid] = agg
        # Attach names as an attribute the report layer reads without widening
        # the dataclass contract used elsewhere.
        result_session_names: dict[str, str] = {
            sid: names.get(sid, "") for sid in result.by_session
        }
        setattr(result, "session_names", result_session_names)
        return result


def _aggregate_from_row(row: Iterable[Any] | None) -> UsageAggregate:
    """Build a UsageAggregate from a SUM row (base columns then components)."""
    if row is None:
        return UsageAggregate()
    values = list(row)
    if not values or values[0] in (None, 0) and all(v in (None, 0) for v in values):
        # COUNT(*) is values[0]; an all-NULL/zero row is an empty scope.
        return UsageAggregate()

    def _n(idx: int) -> int:
        try:
            return int(values[idx] or 0)
        except (TypeError, ValueError, IndexError):
            return 0

    agg = UsageAggregate(
        calls=_n(0),
        ok_calls=_n(1),
        input_tokens=_n(2),
        output_tokens=_n(3),
        cache_read_tokens=_n(4),
        cache_write_tokens=_n(5),
        reasoning_tokens=_n(6),
        context_tokens=_n(7),
    )
    agg.components = {key: _n(8 + i) for i, key in enumerate(COMPONENT_KEYS)}
    return agg
