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

Failures never interrupt a session: a store that cannot open is a no-op
recorder. Aggregate reads retain their empty fallback; the current-session
diagnostic additionally distinguishes an unavailable ledger from zero usage.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    CallSnapshot,
    SessionReport,
    SessionRequest,
    TimingSummary,
    UsageAggregate,
    UsagePeriod,
    apportion_components,
    price_snapshot,
)
from local_operator.paths import config_dir

logger = logging.getLogger("local_operator.analytics.store")

#: Default retention: keep 90 days of per-call rows. Long enough to see trends
#: ("where did usage go this month"), short enough to bound the file. Pruning
#: runs opportunistically on write, not on a timer.
DEFAULT_RETENTION_DAYS = 90

#: Retention for the calendar ROLLUP tables, independent of the raw ledger's
#: 90-day window. The rollups exist precisely so history survives the ledger's
#: prune: a daily bar can look back a full year and a monthly bar much further,
#: without keeping a year of per-call rows on disk. Daily is capped at the most
#: recent 365 DISTINCT days (many more physical rows, since each day holds one
#: row per model used); monthly is effectively unbounded with a 120-month
#: (10-year) safety cap so a decade-old machine cannot grow it without limit.
DAILY_ROLLUP_RETENTION_DAYS = 365
MONTHLY_ROLLUP_RETENTION_MONTHS = 120

#: Bounded retry for a write that loses the lock race past ``busy_timeout``.
#: Runs on the background writer thread, so waiting a moment is free to the
#: session and buys accuracy under many-parallel-session contention.
_WRITE_RETRIES = 4
_WRITE_RETRY_BACKOFF_S = 0.05

#: Bounded retry for the DELETE->WAL journal-mode transition, which is NOT
#: covered by ``busy_timeout`` and so needs its own loop (see ``_set_wal``).
#: Same shape and budget as the write retry above: a few short backoffs on the
#: background thread, then give up and run in whatever mode the file is in.
_WAL_RETRIES = 6
_WAL_RETRY_BACKOFF_S = 0.05

#: Precedence of a name written to ``session_names``, mirroring the rules
#: ``session/naming.py`` documents for the live ``ConversationName`` holder.
#: Higher wins; equal replaces (a re-title must be able to replace the title it
#: supersedes). The gate lives in the SQL of ``upsert_session_name`` rather than
#: in each caller, because the callers run on three different threads and in two
#: different processes — several ``lop`` sessions share this file — so a
#: read-then-write check in Python would be a race by construction.
#:
#: ``PROVISIONAL`` is the opener-derived stand-in the TUI already paints on the
#: status band the instant a message is submitted. It is deliberately BELOW a
#: real title: it quotes the question rather than answering it, and it exists so
#: that a session whose naming call never lands still reads as something a human
#: recognises instead of a bare 12-hex id.
#:
#: ``BACKFILL`` sits at the same level as ``PROVISIONAL`` and not higher,
#: despite often recovering a genuine journalled title: the sweep cannot tell
#: from disk whether what it found was user-set, so ranking it above a live
#: title would let a startup sweep overwrite a rename that had not yet been
#: journalled. Filling an empty slot is all it is for.
SESSION_NAME_RANK_PROVISIONAL = 10
SESSION_NAME_RANK_BACKFILL = 10
SESSION_NAME_RANK_TITLE = 20


def _is_lock_error(exc: BaseException) -> bool:
    """Whether an OperationalError is contention (retryable) or a real fault.

    SQLite reports both SQLITE_BUSY and SQLITE_LOCKED through
    ``OperationalError`` with only the message to tell them apart from a
    genuine fault such as a corrupt file or a read-only directory. That
    distinction decides whether a failure may be retried or must disable the
    store, so it lives in ONE predicate used by both the connect path and the
    write path rather than being sniffed for separately in each.
    """
    text = str(exc).lower()
    return "lock" in text or "busy" in text


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
  -- Dollar cost of the call in MICRO-USD (USD × 1e6) and whether it was
  -- priceable. Integer so the aggregate SUM is exact; see CallSnapshot.
  cost_micro INTEGER NOT NULL DEFAULT 0,
  cost_known INTEGER NOT NULL DEFAULT 0,
  {_COMPONENT_COLUMNS},
  -- The slice of cache_write_tokens written with the 1-hour TTL (Anthropic
  -- ``cache_creation.ephemeral_1h_input_tokens``); the 5m slice is the
  -- remainder. Priced at 2x base rather than 1.25x, so the two must be
  -- separable to judge whether the large-context 1h TTL pays for itself.
  -- SCOPE: a RAW-LEDGER diagnostic, deliberately NOT in the usage_daily /
  -- usage_monthly rollups or the report projection — the rollup tables have
  -- no ALTER migration path (CREATE TABLE IF NOT EXISTS cannot add a column
  -- to an existing one, unlike calls' _MIGRATION_COLUMNS), so threading it
  -- there is a schema change of its own (follow-up tracked on the feature
  -- PR). Answers the trade question within the 90-day ledger window
  -- (DEFAULT_RETENTION_DAYS); beyond that, price the split when it ships.
  cache_write_1h_tokens INTEGER NOT NULL DEFAULT 0,
  request_id TEXT NOT NULL DEFAULT '',
  parent_session_id TEXT NOT NULL DEFAULT '',
  purpose TEXT NOT NULL DEFAULT 'unknown',
  duration_ms REAL NOT NULL DEFAULT -1,
  ttft_ms REAL NOT NULL DEFAULT -1,
  preparation_ms REAL NOT NULL DEFAULT -1,
  outcome TEXT NOT NULL DEFAULT 'unknown',
  usage_reported INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls(ts_ms);
CREATE INDEX IF NOT EXISTS idx_calls_session ON calls(session_id);
CREATE INDEX IF NOT EXISTS idx_calls_provider ON calls(provider);

-- Human-readable session names, so the per-session table can show a title
-- rather than a 12-hex id. Upserted opportunistically; absence just means the
-- report falls back to the id, never an error.
--
-- ``rank`` carries the PRECEDENCE of the name in the row, mirroring the rules
-- ``session/naming.py`` documents for the live holder. Several sources now
-- mirror a label here and they do not arrive in quality order: the instant
-- opener-derived stand-in is written at submit, seconds before the model's
-- real title, and a startup backfill can reconstruct either from disk long
-- after both. Without a rank the last writer would win and a session that HAS
-- a real title could be relabelled with a quote of its own opening question.
-- The upsert is therefore rank-gated (see ``upsert_session_name``): a name may
-- only be replaced by one of equal or higher rank. See ``SESSION_NAME_RANK_*``.
CREATE TABLE IF NOT EXISTS session_names (
  session_id TEXT PRIMARY KEY,
  name TEXT NOT NULL DEFAULT '',
  updated_at_ms INTEGER NOT NULL,
  rank INTEGER NOT NULL DEFAULT 20
);

-- Calendar ROLLUP tables. These are NOT a second source of truth: every row is
-- maintained by the same ``record_batch`` write that appends to ``calls`` (in
-- the SAME transaction), so a call is counted exactly once and there is no
-- separate recording hook to double-count against. They exist because the raw
-- ledger is pruned at 90 days while the operator wants a daily view back a year
-- and a monthly view further still, and because a per-(day, model) /
-- (month, model) grain answers "which model did my spend go to over time" — a
-- question a flat GROUP BY over a pruned ledger cannot, once the rows are gone.
--
-- ``day`` is the LOCAL calendar date (YYYY-MM-DD) the call's ts_ms falls on and
-- ``month`` the local YYYY-MM; local rather than UTC because this is a
-- single-machine tool and "today's spend" means the user's wall-clock day (a
-- turn spanning midnight records under its end day, which is acceptable and
-- documented). Both are TEXT so they sort lexically and read correctly in a
-- range query. ``cost_micro`` accumulates micro-USD (exact SUM); ``cost_known``
-- counts the priced calls so a bucket that used an unpriceable model renders as
-- a lower bound rather than a confident understatement. The composite PK is the
-- ON CONFLICT target the accumulate upsert needs and indexes every range scan.
--
-- FORWARD-FILL, not backfill (review C1): on upgrade these tables are created
-- empty and populated only by calls recorded from that point on. The up-to-90
-- days of pre-existing ``calls`` history is deliberately NOT rolled up. Two
-- reasons: (1) re-bucketing stored ``ts_ms`` would need a strftime that exactly
-- reproduces the LOCAL bucketing ``_local_day_month`` does, and a UTC/local
-- mismatch there would silently misattribute a day's spend — the one thing this
-- store must never do; (2) the ledger prune bounds any backfill to 90 days
-- anyway. So the historical view starts near-empty on the release that ships it
-- and fills in over the following days/weeks. A user-visible, intentional
-- trade; see the design doc's "forward-fill" note.
CREATE TABLE IF NOT EXISTS usage_daily (
  day TEXT NOT NULL,
  model TEXT NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens INTEGER NOT NULL DEFAULT 0,
  context_tokens INTEGER NOT NULL DEFAULT 0,
  cost_micro INTEGER NOT NULL DEFAULT 0,
  cost_known INTEGER NOT NULL DEFAULT 0,
  calls INTEGER NOT NULL DEFAULT 0,
  updated_at_ms INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, model)
);

CREATE TABLE IF NOT EXISTS usage_monthly (
  month TEXT NOT NULL,
  model TEXT NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens INTEGER NOT NULL DEFAULT 0,
  context_tokens INTEGER NOT NULL DEFAULT 0,
  cost_micro INTEGER NOT NULL DEFAULT 0,
  cost_known INTEGER NOT NULL DEFAULT 0,
  calls INTEGER NOT NULL DEFAULT 0,
  updated_at_ms INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (month, model)
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
    "cost_micro",
    "cost_known",
    *(f"c_{key}" for key in COMPONENT_KEYS),
    "cache_write_1h_tokens",
    "request_id",
    "parent_session_id",
    "purpose",
    "duration_ms",
    "ttft_ms",
    "preparation_ms",
    "outcome",
    "usage_reported",
)

#: Columns added AFTER the first shipped schema. A database created by an older
#: release is missing these, and ``CREATE TABLE IF NOT EXISTS`` will not add a
#: column to an existing table — so ``_connect`` runs an idempotent
#: ``ALTER TABLE ADD COLUMN`` for each on open. ``(name, definition)``; the
#: definition carries the default so old rows read as 0 rather than NULL.
#:
#: This is the SINGLE registry of optional columns (review C2, Option A): the
#: first release had cost absent, and now ``c_images`` (the 9th component) is
#: absent on any DB written before it. Rather than a bespoke ``_NO_COST`` insert
#: variant per absent column — which multiplies combinatorially with the next
#: component added — ``_migrate`` records which of THESE are actually present and
#: the insert/aggregate paths are driven by that set. A column that is in this
#: tuple but missing from the table is simply dropped from the insert and read
#: as 0 in the aggregate, giving every optional column the same "analytics must
#: never break a turn" guarantee the cost columns already had.
_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cost_micro", "INTEGER NOT NULL DEFAULT 0"),
    ("cost_known", "INTEGER NOT NULL DEFAULT 0"),
    # Old rows keep their image tokens baked into the conversation/tool_results
    # estimates they were recorded with (forward-fill, same philosophy as the
    # rollup tables). After this ALTER they read ``c_images=0`` rather than
    # being re-apportioned — we cannot honestly unbake a historical estimate.
    ("c_images", "INTEGER NOT NULL DEFAULT 0"),
    # Rows recorded before the Anthropic 1h TTL shipped were all 5m writes, so
    # 0 here is the truth for them, not a placeholder.
    ("cache_write_1h_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("request_id", "TEXT NOT NULL DEFAULT ''"),
    ("parent_session_id", "TEXT NOT NULL DEFAULT ''"),
    ("purpose", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("duration_ms", "REAL NOT NULL DEFAULT -1"),
    ("ttft_ms", "REAL NOT NULL DEFAULT -1"),
    ("preparation_ms", "REAL NOT NULL DEFAULT -1"),
    ("outcome", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("usage_reported", "INTEGER NOT NULL DEFAULT 1"),
)

#: The names in ``_MIGRATION_COLUMNS`` as a set, for the "is this column optional
#: (i.e. possibly absent on an old DB)?" test. A column NOT in here — the base
#: token columns and the original eight ``c_*`` components — is present on every
#: DB that has ever existed and is never dropped from a query.
_OPTIONAL_COLUMN_NAMES: frozenset[str] = frozenset(name for name, _ in _MIGRATION_COLUMNS)

#: The all-columns insert, used when the DB has every optional column (a fresh DB
#: always does). ``_migrate`` narrows this to the present columns per DB.
_INSERT_SQL = (
    f"INSERT INTO calls ({', '.join(_CALL_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _CALL_COLUMNS)})"
)

#: The measure columns a rollup row accumulates. Every one is summed on
#: conflict, so an upsert is a pure ``x = x + excluded.x`` accumulate and N
#: processes incrementing the same (day, model) never lose an update — the
#: multi-``lop`` reality this store is built for. ``calls`` and ``cost_known``
#: are counts (1 per row here); the token/cost fields carry the call's amounts.
_ROLLUP_MEASURE_COLUMNS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "context_tokens",
    "cost_micro",
    "cost_known",
    "calls",
)


def _rollup_upsert_sql(table: str, key: str) -> str:
    """The accumulate-upsert for one rollup table, keyed on ``(key, model)``.

    ``INSERT ... ON CONFLICT DO UPDATE SET x = x + excluded.x`` so concurrent
    writers merge losslessly without application locking (WAL + busy_timeout
    serialise the physical write; the accumulate makes the logical result
    order-independent). ``updated_at_ms`` takes the newest writer's clock so a
    reader can tell a live bucket from a stale one. Built from
    ``_ROLLUP_MEASURE_COLUMNS`` so the daily and monthly statements cannot
    drift apart.
    """
    cols = (key, "model", *_ROLLUP_MEASURE_COLUMNS, "updated_at_ms")
    placeholders = ", ".join("?" for _ in cols)
    accumulate = ", ".join(f"{c} = {c} + excluded.{c}" for c in _ROLLUP_MEASURE_COLUMNS)
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({key}, model) DO UPDATE SET {accumulate}, "
        "updated_at_ms = excluded.updated_at_ms"
    )


_DAILY_UPSERT_SQL = _rollup_upsert_sql("usage_daily", "day")
_MONTHLY_UPSERT_SQL = _rollup_upsert_sql("usage_monthly", "month")

#: The rollup measure columns selected/summed by the read API, in the order
#: :class:`UsagePeriod` consumes them. One list so the SELECT projection and the
#: dataclass construction cannot fall out of step.
_ROLLUP_READ_COLUMNS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "context_tokens",
    "cost_micro",
    "cost_known",
    "calls",
)


def default_db_path() -> Path:
    """The shared analytics database, next to the other per-user stores."""
    return config_dir() / "analytics.db"


def _row_values(snapshot: CallSnapshot, cost_micro: int, cost_known: bool) -> tuple[Any, ...]:
    """A snapshot as the positional tuple ``_INSERT_SQL`` expects.

    The estimated component split is computed HERE, in the writer, against the
    authoritative ``context_tokens`` — never on the event loop. A call the
    provider gave no context total for stores 0s for every component, which
    reads as "unknown" rather than a fabricated breakdown.

    ``cost_micro``/``cost_known`` are passed in already priced (once per
    snapshot in ``record_batch``) rather than priced here, so the same figure
    feeds this ledger row AND the rollup rows without calling the potentially
    cold ``resolve_model_info`` twice for one call.
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
        int(cost_micro),
        1 if cost_known else 0,
        *(components[key] for key in COMPONENT_KEYS),
        snapshot.cache_write_1h_tokens,
        snapshot.request_id,
        snapshot.parent_session_id,
        snapshot.purpose,
        snapshot.duration_ms,
        snapshot.ttft_ms,
        snapshot.preparation_ms,
        snapshot.outcome,
        int(snapshot.usage_reported),
    )


def _rollup_model_key(snapshot: CallSnapshot) -> str:
    """The (day/month, model) dimension for a snapshot's rollup rows.

    The FINEST model identity the snapshot carries — ``provider/model_id`` —
    because cost depends entirely on it, a session can switch models mid-life,
    and subagents routinely run on a different model from the parent, so
    collapsing to the provider would throw away exactly the per-model
    attribution the time-series view exists to show. Falls back to the bare
    provider when a model id is absent (never expected on a real call, but a
    rollup key must not be empty), matching the ``calls`` ledger which stores
    both fields separately.
    """
    provider = (snapshot.provider or "").strip()
    model_id = (snapshot.model_id or "").strip()
    if provider and model_id:
        return f"{provider}/{model_id}"
    return model_id or provider


def _local_day_month(ts_ms: int) -> tuple[str, str]:
    """``(local YYYY-MM-DD, local YYYY-MM)`` for an epoch-ms timestamp.

    LOCAL time, not UTC: a single-machine tool's "today" is the user's
    wall-clock day (see the schema comment). ``datetime.fromtimestamp`` with no
    tz argument converts using the system local zone, which is the same clock
    ``ts_ms`` was stamped from.
    """
    moment = datetime.fromtimestamp(ts_ms / 1000.0)
    return moment.strftime("%Y-%m-%d"), moment.strftime("%Y-%m")


def _rollup_row_values(
    snapshot: CallSnapshot, bucket: str, cost_micro: int, cost_known: bool
) -> tuple[Any, ...]:
    """A snapshot as the positional tuple a rollup upsert expects.

    ``bucket`` is the day or month string. The tuple order matches
    ``_rollup_upsert_sql``'s column list (bucket, model, then the measures,
    then ``updated_at_ms``). Uses the SAME ``cost_micro``/``cost_known`` the
    ledger row got so the rollup and the raw ledger can never disagree on a
    call's cost; ``cost_known`` is 1/0 so its SUM is the count of priceable
    calls in the bucket.
    """
    return (
        bucket,
        _rollup_model_key(snapshot),
        snapshot.input_tokens,
        snapshot.output_tokens,
        snapshot.cache_read_tokens,
        snapshot.cache_write_tokens,
        snapshot.reasoning_tokens,
        snapshot.context_tokens,
        int(cost_micro),
        1 if cost_known else 0,
        1,
        int(snapshot.ts_ms),
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
        #: Whether the cost columns exist on THIS database, scoped EXPLICITLY to
        #: ``cost_micro``/``cost_known`` (not "all optional columns present"):
        #: once ``c_images`` joined ``_MIGRATION_COLUMNS`` a blanket check would
        #: conflate "cost present" with "images present" and mislabel a
        #: cost-capable DB. The report reads this to choose ``$—`` vs a real sum.
        self._has_cost = True
        #: Which OPTIONAL columns (``_MIGRATION_COLUMNS``) actually exist on this
        #: DB. A fresh DB has all of them (the ``CREATE TABLE`` includes them); an
        #: old one gets them from ``_migrate``. If a migration ALTER genuinely
        #: fails (a locked/corrupt DB), the absent column is dropped from every
        #: insert and read as 0 in the aggregate rather than being referenced and
        #: failing EVERY write — the generalised C2 "never break a turn" path.
        #: Defaults to all-present; ``_migrate`` narrows it to the truth per DB.
        self._present_optional: frozenset[str] = _OPTIONAL_COLUMN_NAMES
        #: The insert column list + SQL for THIS DB, derived from
        #: ``_present_optional`` in ``_migrate``. ``_insert_indices`` selects the
        #: matching values out of ``_row_values``' full (``_CALL_COLUMNS``-order)
        #: tuple so a missing column is dropped from both the SQL and the row.
        self._insert_indices: tuple[int, ...] = tuple(range(len(_CALL_COLUMNS)))
        self._insert_sql: str = _INSERT_SQL

    # -- connection ----------------------------------------------------------
    @staticmethod
    def _set_wal(conn: sqlite3.Connection) -> None:
        """Switch the journal to WAL, retrying the contended DELETE->WAL step.

        ``busy_timeout`` does NOT cover this statement. Changing the journal
        mode needs an exclusive lock on the database, and SQLite fails that
        acquisition with SQLITE_BUSY immediately instead of invoking the busy
        handler, so the 5s timeout set just above buys nothing here. Measured on
        a fresh database opened simultaneously by 16 processes: 25/320 opens
        raised ``database is locked`` at this statement, and setting
        ``busy_timeout`` first only brought that to 15/320 — reordering alone is
        not a fix. With this bounded retry the same probe reports 0/320.

        Only the FIRST process to reach a fresh file pays anything: once the
        file is in WAL the pragma is a no-op that cannot fail (0/320 failures
        against an already-WAL database), so this loop costs established
        installations nothing.

        A database that stays un-WAL after every attempt is still usable —
        rollback-journal mode serialises writers rather than losing them — so
        this returns quietly rather than raising and disabling the store.
        """
        for attempt in range(_WAL_RETRIES):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as exc:
                if not _is_lock_error(exc) or attempt == _WAL_RETRIES - 1:
                    logger.debug("analytics: could not enable WAL", exc_info=True)
                    return
                time.sleep(_WAL_RETRY_BACKOFF_S * (attempt + 1))

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
            # busy_timeout FIRST: it arms SQLite's busy handler for everything
            # that follows, including the schema script below. It is set before
            # the journal-mode switch rather than after it because the switch is
            # the single most lock-contended statement here (see ``_set_wal``).
            conn.execute("PRAGMA busy_timeout=5000")
            self._set_wal(conn)
            conn.execute("PRAGMA synchronous=NORMAL")
            # Schema creation is idempotent (IF NOT EXISTS) but should run once,
            # under a lock, so two threads opening their first connections
            # simultaneously do not both executescript into the same file.
            with self._init_lock:
                conn.executescript(_SCHEMA)
                self._migrate(conn)
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
        except sqlite3.OperationalError as exc:
            # A LOCK failure here is TRANSIENT — another process is opening the
            # same fresh database this instant — so it must NOT latch _broken.
            # Latching it silently zeroed a whole process's analytics for its
            # entire lifetime on a momentary race (#391: one of four parallel
            # writers contributing exactly zero rows). Leaving _broken clear
            # means the next write simply opens again and succeeds.
            if _is_lock_error(exc):
                logger.debug("analytics: %s busy while opening", self._db_path, exc_info=True)
                return None
            logger.debug("analytics: cannot open %s", self._db_path, exc_info=True)
            self._broken = True
            return None
        except Exception:  # noqa: BLE001 — store unavailable = analytics off
            # Anything that is not a lock (a read-only home, a corrupt file, a
            # bad path) IS permanent, and latching stops one log line per
            # provider round trip for the life of the process.
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

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns that a database from an older release is missing.

        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a
        ledger written by the token-only release has no ``cost_*`` columns. Add
        each via ``ALTER TABLE ADD COLUMN`` (idempotent: skip the ones already
        present). Old rows take the column default — cost 0, unknown — so a
        pre-cost call reads as "unpriced", never as a confident $0. Called under
        ``_init_lock`` on open; a failure here degrades to the pre-migration
        shape rather than raising, because analytics is never a hard dependency.

        Records which OPTIONAL columns are ACTUALLY present afterward
        (``self._present_optional``) and rebuilds the per-DB insert plan from it:
        if an ALTER failed, that column is dropped from the insert AND read as 0
        in the aggregate, so a missing column cannot fail every write and blank
        the screen (review C2, generalised to every optional column via Option A).
        ``self._has_cost`` is scoped EXPLICITLY to the two cost columns so adding
        ``c_images`` to ``_MIGRATION_COLUMNS`` does not conflate "cost present"
        with "images present".
        """
        try:
            existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(calls)").fetchall()}
        except Exception:  # noqa: BLE001 — an unreadable schema is a no-op migration
            self._has_cost = False
            self._present_optional = frozenset()
            self._rebuild_insert_plan()
            return
        for name, definition in _MIGRATION_COLUMNS:
            if name in existing:
                continue
            try:
                conn.execute(f"ALTER TABLE calls ADD COLUMN {name} {definition}")
                existing.add(name)
            except Exception:  # noqa: BLE001 — a failed add leaves the older shape
                logger.debug("analytics: could not add column %s", name, exc_info=True)
        # ``session_names.rank`` reaches an existing ledger the same way: the
        # CREATE TABLE in _SCHEMA never alters a table that already exists, so a
        # database from any earlier release has the name table without it. The
        # DEFAULT is TITLE, which is the truth for every row written before this
        # column existed — the only writer then was ``set_conversation_name``,
        # i.e. a real generated or user-set title. Defaulting to PROVISIONAL
        # instead would let the new backfill sweep overwrite genuine titles.
        try:
            name_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(session_names)").fetchall()
            }
            if name_columns and "rank" not in name_columns:
                conn.execute(
                    "ALTER TABLE session_names ADD COLUMN rank INTEGER NOT NULL "
                    f"DEFAULT {SESSION_NAME_RANK_TITLE}"
                )
        except Exception:  # noqa: BLE001 — an un-migratable name table is not fatal
            logger.debug("analytics: could not add session_names.rank", exc_info=True)
        # Scope cost to the cost columns only (NOT "all optional present"): with
        # c_images now in _MIGRATION_COLUMNS an all-present check would flip cost
        # off on a cost-capable DB that merely lacks images.
        self._has_cost = all(n in existing for n in ("cost_micro", "cost_known"))
        self._present_optional = frozenset(
            name for name, _ in _MIGRATION_COLUMNS if name in existing
        )
        self._rebuild_insert_plan()

    def _rebuild_insert_plan(self) -> None:
        """Recompute the insert SQL + value selector from ``_present_optional``.

        The insert columns are ``_CALL_COLUMNS`` minus any optional column absent
        on this DB, in the SAME order; ``_insert_indices`` picks the matching
        values out of ``_row_values``' full tuple so the row and the column list
        stay aligned. One selector drives the general degraded path — no bespoke
        ``_NO_COST``/``_NO_IMAGES`` variant per column, which is the whole point
        of Option A.
        """

        # A column is KEPT when it is not optional (always present) or it is an
        # optional column that this DB actually has.
        def _keep(col: str) -> bool:
            return col not in _OPTIONAL_COLUMN_NAMES or col in self._present_optional

        self._insert_indices = tuple(i for i, c in enumerate(_CALL_COLUMNS) if _keep(c))
        columns = [_CALL_COLUMNS[i] for i in self._insert_indices]
        self._insert_sql = (
            f"INSERT INTO calls ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})"
        )

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
        # Open is retried independently of the insert retry below: a lock on
        # the DELETE->WAL transition used to return None here (and, before
        # #391, latch ``_broken``), which dropped the WHOLE batch — one
        # process contributing zero rows. A genuinely broken store still
        # returns 0 on the first attempt (``_broken`` is sticky for those).
        conn: sqlite3.Connection | None = None
        for attempt in range(_WRITE_RETRIES):
            conn = self._connect()
            if conn is not None or self._broken:
                break
            time.sleep(_WRITE_RETRY_BACKOFF_S * (attempt + 1))
        if conn is None:
            return 0
        # Price each snapshot ONCE here (writer thread), then feed that figure
        # to both the ledger row and the two rollup rows. Pricing on the event
        # loop is forbidden (a cold ``resolve_model_info`` blocks for seconds,
        # review C1); doing it once rather than per-row also keeps a batch cheap.
        priced = [price_snapshot(s) for s in snapshots]
        rows = [_row_values(s, cm, ck) for s, (cm, ck) in zip(snapshots, priced)]
        # Rollup rows for the SAME calls, keyed by the LOCAL day/month of each
        # call's ts_ms. Written in the same transaction as the ledger insert
        # (below) so a call lands in the ledger and both rollups together or not
        # at all — a turn is never half-recorded. This is why there is no
        # double-count: the rollups are fed by the ledger's ONE write path, not
        # by a separate app-level hook that could also observe the same spend.
        daily_rows: list[tuple[Any, ...]] = []
        monthly_rows: list[tuple[Any, ...]] = []
        for snap, (cm, ck) in zip(snapshots, priced):
            day, month = _local_day_month(int(snap.ts_ms))
            daily_rows.append(_rollup_row_values(snap, day, cm, ck))
            monthly_rows.append(_rollup_row_values(snap, month, cm, ck))
        # Option A: the insert SQL and the value selector were computed once in
        # ``_migrate`` from the columns this DB actually has. Select exactly the
        # present columns' values out of each full row tuple, so an absent
        # optional column (failed cost or images migration) is dropped from both
        # the SQL and the row rather than referenced and failing every write.
        insert_sql = self._insert_sql
        if len(self._insert_indices) != len(_CALL_COLUMNS):
            rows = [tuple(row[i] for i in self._insert_indices) for row in rows]
        for attempt in range(_WRITE_RETRIES):
            try:
                conn.executemany(insert_sql, rows)
                # The rollups accumulate in the same transaction. A failure to
                # write them must not lose the ledger row, but SQLite gives us
                # atomicity for free here: both executemany calls commit
                # together, so either all three tables advance or the whole
                # attempt rolls back and retries. The rollup tables always carry
                # the cost columns (they are created with them and never shed
                # them the way the ledger's C2 path does), so no cost-less
                # variant is needed.
                conn.executemany(_DAILY_UPSERT_SQL, daily_rows)
                conn.executemany(_MONTHLY_UPSERT_SQL, monthly_rows)
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
                if not _is_lock_error(exc):
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

    def upsert_session_name(
        self, session_id: str, name: str, *, rank: int = SESSION_NAME_RANK_TITLE
    ) -> None:
        """Record (or update) a session's human name for the per-session table.

        RANK-GATED, which is the whole reason this is not a plain upsert. The
        ledger is now mirrored from several sources that do not arrive in
        quality order — the provisional stand-in lands at submit, the model's
        title a second or two later, a resume restores whichever was journalled,
        and a startup backfill reconstructs one from disk at any time. Letting
        the last writer win would have a provisional excerpt displace a real
        title, which is precisely the precedence ``session/naming.py`` protects
        on the live holder. The ``WHERE excluded.rank >= session_names.rank``
        clause makes the same rule true of the ledger: a same-or-better source
        may correct the row, a weaker one may only fill an empty slot.

        Equal rank still overwrites, deliberately: a re-title is the same rank
        as the title it replaces and MUST be able to replace it.
        """
        if not session_id:
            return
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO session_names (session_id, name, updated_at_ms, rank) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "name=excluded.name, updated_at_ms=excluded.updated_at_ms, "
                "rank=excluded.rank WHERE excluded.rank >= session_names.rank",
                (session_id, name or "", self._now_ms(), int(rank)),
            )
            conn.commit()
        except Exception:  # noqa: BLE001
            logger.debug("analytics: session name upsert failed", exc_info=True)

    def session_names_present(self) -> set[str]:
        """Every session id that already carries a ledger name.

        Read by the startup backfill so it can skip the sessions that need no
        work without opening a transcript for each — the sweep walks the whole
        session store and a per-directory read would be the expensive half.
        """
        conn = self._read_connection()
        if conn is None:
            return set()
        try:
            rows = conn.execute("SELECT session_id FROM session_names WHERE name <> ''").fetchall()
            return {str(row[0]) for row in rows}
        except Exception:  # noqa: BLE001 — a failed read means "backfill nothing"
            logger.debug("analytics: session name read failed", exc_info=True)
            return set()
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def session_names_map(self) -> dict[str, str]:
        """Every known ledger name, keyed by session id.

        Read by the backfill so a delegated session can be labelled with its
        PARENT's title without one query per row.
        """
        conn = self._read_connection()
        if conn is None:
            return {}
        try:
            rows = conn.execute("SELECT session_id, name FROM session_names WHERE name <> ''")
            return {str(sid): str(name) for sid, name in rows.fetchall()}
        except Exception:  # noqa: BLE001
            logger.debug("analytics: session name map read failed", exc_info=True)
            return {}
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def session_parents(self) -> dict[str, str]:
        """child session id -> parent session id, from the recorded call rows.

        The self-parent edge is EXCLUDED. 224 rows on the operator's real ledger
        carry ``parent_session_id == session_id`` (all of them the degenerate
        empty-id case), which is a genuine cycle edge in the data; a consumer
        that walked this map without the guard would loop. Filtering it here
        means every caller inherits the guard rather than having to remember it.
        """
        conn = self._read_connection()
        if conn is None:
            return {}
        try:
            rows = conn.execute(
                "SELECT DISTINCT session_id, parent_session_id FROM calls "
                "WHERE parent_session_id <> '' AND session_id <> '' "
                "AND session_id <> parent_session_id"
            ).fetchall()
            return {str(child): str(parent) for child, parent in rows}
        except Exception:  # noqa: BLE001
            logger.debug("analytics: session parent read failed", exc_info=True)
            return {}
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def sessions_missing_names(self) -> set[str]:
        """Session ids that HAVE ledger rows but no name — the backfill's worklist.

        Scoped to ids the ledger actually knows about so the sweep never mints a
        name for a session that cost nothing, and so its work is bounded by the
        ledger rather than by the session store.
        """
        conn = self._read_connection()
        if conn is None:
            return set()
        try:
            rows = conn.execute(
                "SELECT DISTINCT c.session_id FROM calls c "
                "LEFT JOIN session_names n ON n.session_id = c.session_id "
                "WHERE c.session_id <> '' AND (n.name IS NULL OR n.name = '')"
            ).fetchall()
            return {str(row[0]) for row in rows}
        except Exception:  # noqa: BLE001 — a failed read means "backfill nothing"
            logger.debug("analytics: missing-name read failed", exc_info=True)
            return set()
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def prune(self, *, now_ms: int | None = None) -> int:
        """Delete rows past their retention window. Returns raw-ledger rows removed.

        Three independent windows are enforced in one call, none affecting the
        others:

        - The raw ``calls`` ledger keeps ``retention_days`` (default 90) by
          ``ts_ms`` — unchanged; its row count is the return value, preserving
          the original contract.
        - ``usage_daily`` keeps the most recent 365 DISTINCT ``day`` values
          (not 365 rows — each day holds one row per model), so the daily bar
          can look back a year regardless of how many models ran. The subquery
          finds the 365th-newest distinct day and deletes everything older.
        - ``usage_monthly`` keeps the most recent 120 DISTINCT months — a
          10-year safety cap on an effectively-unbounded table (12 rows/year ×
          models is negligible), so the monthly arc survives far beyond the
          daily window without growing without limit.

        The rollup prunes are keyed on the stored ``day``/``month`` STRINGS, not
        on ``now_ms``: they keep the newest N buckets that exist rather than a
        window relative to the wall clock, so a machine idle for a week does not
        silently drop a still-recent bucket. ``now_ms`` is still injected for
        the ledger cutoff (and for tests). Each statement is guarded on its own
        so a missing rollup table (a very old DB mid-migration) degrades to
        pruning what it can rather than raising.
        """
        conn = self._connect()
        if conn is None:
            return 0
        cutoff = (now_ms if now_ms is not None else self._now_ms()) - self._retention_ms
        removed = 0
        try:
            cur = conn.execute("DELETE FROM calls WHERE ts_ms < ?", (cutoff,))
            removed = cur.rowcount or 0
            conn.commit()
        except Exception:  # noqa: BLE001
            logger.debug("analytics: prune failed", exc_info=True)
        # Rollup prunes are best-effort and independent of the ledger prune
        # above: a failure here must not undo the ledger delete or raise.
        try:
            conn.execute(
                "DELETE FROM usage_daily WHERE day < ("
                "  SELECT MIN(day) FROM ("
                "    SELECT DISTINCT day FROM usage_daily ORDER BY day DESC LIMIT ?"
                "  )"
                ")",
                (DAILY_ROLLUP_RETENTION_DAYS,),
            )
            conn.execute(
                "DELETE FROM usage_monthly WHERE month < ("
                "  SELECT MIN(month) FROM ("
                "    SELECT DISTINCT month FROM usage_monthly ORDER BY month DESC LIMIT ?"
                "  )"
                ")",
                (MONTHLY_ROLLUP_RETENTION_MONTHS,),
            )
            conn.commit()
        except Exception:  # noqa: BLE001 — a rollup prune failure is non-fatal
            logger.debug("analytics: rollup prune failed", exc_info=True)
        return removed

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

        # Substitute a constant 0 for any optional component column this DB lacks
        # (a failed/old-DB migration, review C2 generalised): an absent ``c_*``
        # reads as 0 rather than failing the whole query on a missing column. The
        # base ``c_*`` columns (the original eight) are not optional and always
        # summed. Positions stay a contract with ``_aggregate_from_row``.
        def _component_expr(key: str) -> str:
            col = f"c_{key}"
            present = col not in _OPTIONAL_COLUMN_NAMES or col in self._present_optional
            return f"SUM({col})" if present else "0"

        component_sum = ", ".join(_component_expr(key) for key in COMPONENT_KEYS)
        # Order is a contract with ``_aggregate_from_row``, which indexes this
        # tuple positionally: cost_micro and cost_known_calls come after the
        # token sums and before the component sums. When the cost columns are
        # absent (a failed migration, review C2), substitute constant 0 sums so
        # the positions still line up and the report shows $— instead of the
        # query failing on a missing column.
        cost_cols = "SUM(cost_micro), SUM(cost_known)" if self._has_cost else "0, 0"
        base_cols = (
            "COUNT(*), SUM(ok), SUM(input_tokens), SUM(output_tokens), "
            "SUM(cache_read_tokens), SUM(cache_write_tokens), "
            f"SUM(reasoning_tokens), SUM(context_tokens), {cost_cols}"
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

    def session_report(self, session_id: str, *, recent_limit: int = 12) -> SessionReport:
        """Read one exact session ID without creating, migrating or pricing data.

        A single explicit read transaction pins all queries to the same WAL
        snapshot, even if the recorder commits between totals and recent rows.
        Inspect columns on THIS connection rather than writer migration flags:
        diagnostics must also work against a read-only, older ledger. Missing
        optional fields remain unknown, not invented successes or zero timings.
        """
        conn: sqlite3.Connection | None = None
        try:
            if not self._db_path.exists():
                return SessionReport(session_id=session_id)
            conn = sqlite3.connect(
                self._db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5
            )
            conn.execute("BEGIN")
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(calls)")}
            if not {"session_id", "provider", "model_id", "ts_ms", "id"} <= columns:
                return SessionReport(session_id=session_id, available=False)

            def col(name: str, default: str = "0") -> str:
                # Names are code-owned constants, never user input. Only the ID
                # is caller supplied, and every query binds it as a parameter.
                return name if name in columns else default

            sums = ["COUNT(*)", f"SUM({col('ok')})"]
            sums += [
                f"SUM({col(name)})"
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "reasoning_tokens",
                    "context_tokens",
                    "cost_micro",
                    "cost_known",
                    *(f"c_{key}" for key in COMPONENT_KEYS),
                )
            ]
            measures = ", ".join(sums)
            scope = " FROM calls WHERE session_id = ?"
            params = (session_id,)
            aggregate = _aggregate_from_row(
                conn.execute(f"SELECT {measures}" + scope, params).fetchone()
            )
            by_model = {
                (str(row[0]), str(row[1])): _aggregate_from_row(row[2:])
                for row in conn.execute(
                    f"SELECT provider, model_id, {measures}"
                    + scope
                    + " GROUP BY provider, model_id",
                    params,
                )
            }
            purpose = col("purpose", "'unknown'")
            outcome = col("outcome", "'unknown'")
            # Consumption per purpose, on the SAME ``measures`` contract as
            # ``by_model`` — one extra GROUP BY on a column that already exists
            # beside the token and cost sums, so no schema change and no second
            # aggregation vocabulary. ``col`` folds an older ledger without the
            # column into a single ``unknown`` row, which is honest: we know the
            # tokens, we do not know what they were spent on.
            by_purpose = {
                str(row[0]): _aggregate_from_row(row[1:])
                for row in conn.execute(
                    f"SELECT {purpose}, {measures}" + scope + " GROUP BY 1", params
                )
            }
            groups = {
                (str(row[0]), str(row[1])): int(row[2])
                for row in conn.execute(
                    f"SELECT {purpose}, {outcome}, COUNT(*)" + scope + " GROUP BY 1, 2", params
                )
            }
            usage = col("usage_reported", "NULL")
            missing, unknown, first, last = conn.execute(
                f"SELECT SUM({usage} = 0), SUM({usage} IS NULL), MIN(ts_ms), MAX(ts_ms)" + scope,
                params,
            ).fetchone()
            timings: dict[str, TimingSummary] = {}
            for name in ("duration_ms", "ttft_ms", "preparation_ms"):
                expression = col(name, "NULL")
                row = conn.execute(
                    f"SELECT COUNT({expression}), AVG({expression}), "
                    f"MIN({expression}), MAX({expression})" + scope + f" AND {expression} >= 0",
                    params,
                ).fetchone()
                timings[name] = TimingSummary(int(row[0]), row[1], row[2], row[3])
            fields = [
                col("request_id", "''"),
                "ts_ms",
                "provider",
                "model_id",
                purpose,
                outcome,
                usage,
                col("context_tokens"),
                col("output_tokens"),
                *(f"NULLIF({col(name, '-1')}, -1)" for name in timings),
            ]
            recent = tuple(
                SessionRequest(
                    request_id=row[0],
                    ts_ms=row[1],
                    provider=row[2],
                    model_id=row[3],
                    purpose=row[4],
                    outcome=row[5],
                    usage_reported=None if row[6] is None else bool(row[6]),
                    context_tokens=row[7],
                    output_tokens=row[8],
                    duration_ms=row[9],
                    ttft_ms=row[10],
                    preparation_ms=row[11],
                )
                for row in conn.execute(
                    "SELECT " + ", ".join(fields) + scope + " ORDER BY ts_ms DESC, id DESC LIMIT ?",
                    (*params, max(0, min(int(recent_limit), 50))),
                )
            )
            return SessionReport(
                session_id=session_id,
                aggregate=aggregate,
                by_model=by_model,
                by_purpose=by_purpose,
                by_purpose_outcome=groups,
                missing_usage_calls=int(missing or 0),
                unknown_usage_calls=int(unknown or 0),
                timings=timings,
                recent=recent,
                first_ts_ms=first,
                last_ts_ms=last,
            )
        except Exception:  # noqa: BLE001 — diagnostics must not interrupt a turn
            logger.debug("analytics: session report unavailable", exc_info=True)
            return SessionReport(session_id=session_id, available=False)
        finally:
            if conn is not None:
                conn.close()

    # -- rollup reads (calendar time series) ---------------------------------
    def _series(self, table: str, key: str, buckets: int, *, by_model: bool) -> list[UsagePeriod]:
        """The most recent ``buckets`` calendar buckets from a rollup table.

        Shared by :meth:`daily_series` and :meth:`monthly_series` — the only
        difference is the table and its key column. Two shapes:

        - ``by_model=False``: one :class:`UsagePeriod` per bucket, SUMMED across
          models in SQL (``GROUP BY key``), ``model=""``. This is the primary
          series the bar chart draws.
        - ``by_model=True``: one row per ``(bucket, model)``, so the view can
          break a period down by which model spent it.

        Returned oldest-LAST (``key`` ascending) so the caller can render newest
        at the bottom to match the transcript's reading order, or reverse it
        cheaply. The window is "the newest N DISTINCT buckets that exist", found
        with a subquery, not a wall-clock cutoff — a gap of idle days does not
        cost a bar. Never raises: a degraded or empty store returns ``[]``.
        """
        conn = self._read_connection()
        if conn is None:
            return []
        limit = max(1, int(buckets))
        measures = ", ".join(f"SUM({c})" for c in _ROLLUP_READ_COLUMNS)
        # The N newest distinct buckets, oldest-first for rendering. An inner
        # DESC LIMIT picks the window; the outer ASC orders it for the reader.
        window = f"SELECT DISTINCT {key} AS b FROM {table} ORDER BY {key} DESC LIMIT {limit}"
        try:
            if by_model:
                sql = (
                    f"SELECT {key}, model, {measures} FROM {table} "
                    f"WHERE {key} IN ({window}) "
                    f"GROUP BY {key}, model ORDER BY {key} ASC, model ASC"
                )
                rows = conn.execute(sql).fetchall()
                return [_period_from_row(str(r[0]), str(r[1]), r[2:]) for r in rows]
            sql = (
                f"SELECT {key}, {measures} FROM {table} "
                f"WHERE {key} IN ({window}) "
                f"GROUP BY {key} ORDER BY {key} ASC"
            )
            rows = conn.execute(sql).fetchall()
            return [_period_from_row(str(r[0]), "", r[1:]) for r in rows]
        except Exception:  # noqa: BLE001 — a report read must never raise
            logger.debug("analytics: %s series query failed", table, exc_info=True)
            return []
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def daily_series(self, days: int = 30, *, by_model: bool = False) -> list[UsagePeriod]:
        """The most recent ``days`` distinct days of usage, oldest-first."""
        return self._series("usage_daily", "day", days, by_model=by_model)

    def monthly_series(self, months: int = 12, *, by_model: bool = False) -> list[UsagePeriod]:
        """The most recent ``months`` distinct months of usage, oldest-first."""
        return self._series("usage_monthly", "month", months, by_model=by_model)

    def series_totals(self, *, daily_days: int = 30) -> UsagePeriod:
        """Grand totals over the most recent ``daily_days`` daily buckets.

        A single summed :class:`UsagePeriod` (``period=""``, ``model=""``) over
        the same window the daily chart draws, so the header figure and the bars
        describe the same span. Reads the daily rollup rather than the raw
        ledger so it survives the ledger's 90-day prune, and sums the same
        ``by_model=False`` series the chart uses so the two cannot disagree.
        """
        rows = self.daily_series(daily_days, by_model=False)
        if not rows:
            return UsagePeriod(period="", model="")
        return UsagePeriod(
            period="",
            model="",
            input_tokens=sum(r.input_tokens for r in rows),
            output_tokens=sum(r.output_tokens for r in rows),
            cache_read_tokens=sum(r.cache_read_tokens for r in rows),
            cache_write_tokens=sum(r.cache_write_tokens for r in rows),
            reasoning_tokens=sum(r.reasoning_tokens for r in rows),
            context_tokens=sum(r.context_tokens for r in rows),
            cost_micro=sum(r.cost_micro for r in rows),
            cost_known_calls=sum(r.cost_known_calls for r in rows),
            calls=sum(r.calls for r in rows),
        )


def _period_from_row(period: str, model: str, measures: Iterable[Any]) -> UsagePeriod:
    """Build a :class:`UsagePeriod` from a SUM row's measure columns.

    ``measures`` is the projection of ``_ROLLUP_READ_COLUMNS`` in order; a NULL
    (an empty SUM) reads as 0 so an all-empty bucket is a zeroed period rather
    than a crash.
    """
    values = list(measures)

    def _n(idx: int) -> int:
        try:
            return int(values[idx] or 0)
        except (TypeError, ValueError, IndexError):
            return 0

    return UsagePeriod(
        period=period,
        model=model,
        input_tokens=_n(0),
        output_tokens=_n(1),
        cache_read_tokens=_n(2),
        cache_write_tokens=_n(3),
        reasoning_tokens=_n(4),
        context_tokens=_n(5),
        cost_micro=_n(6),
        cost_known_calls=_n(7),
        calls=_n(8),
    )


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
        cost_micro=_n(8),
        cost_known_calls=_n(9),
    )
    # Components follow the two cost sums (see ``base_cols``).
    agg.components = {key: _n(10 + i) for i, key in enumerate(COMPONENT_KEYS)}
    return agg
