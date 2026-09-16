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
``/analytics`` screen reads a GROUP BY when it opens. That GROUP BY is over the
maintained ``session_daily`` rollup rather than the raw ledger, because the
ledger stopped being bounded in practice: on the operator's 342.8 MB, 1 155 845
call ledger the panel's 30-day window costs 4.8 s wall / 2 765 ms CPU on the raw
ledger and 6.4 s for the first read of a fresh copy, against the 30 s client
timeout it already hits — while the rollup answers the same window in 334 ms
wall / 110 ms CPU (``bench/analytics-rollup-before.json`` and ``-after.json``,
reproduced by ``scripts/bench_panel_latency.py``). The raw-ledger query is still
there, unchanged, behind a fail-closed gate that answers whenever the rollup
cannot prove the same numbers (``aggregate()``'s docstring has the account).

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    EXCLUDED_FAULTS,
    ORIGIN_MODEL,
    CallSnapshot,
    SessionReport,
    SessionRequest,
    TimingSummary,
    ToolCallStats,
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
  rank INTEGER NOT NULL DEFAULT {SESSION_NAME_RANK_TITLE}
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

-- One row per TOOL CALL the harness dispatched or rejected. Separate from
-- ``calls`` rather than a pair of counters on it, because the provider row is
-- written in ``_record_stream``'s ``finally`` BEFORE the tools run and the
-- ledger is append-only — there is no row to increment and no request id
-- reaching the harness to increment it by. A table also gets the per-tool-name
-- breakdown a counter never could. Measured cost: 96 bytes/row, ~18 MB
-- steady-state at the observed tool-call rate.
CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  tool_name TEXT NOT NULL DEFAULT '',
  -- 'model' = the model emitted a tool_use block; 'nested' = eval's
  -- dispatch_tool bridge. Separable because a nested call is the model's CODE
  -- calling a tool, not the model emitting a call, and conflating them would
  -- let one scripted retry loop dominate the accuracy figure.
  origin TEXT NOT NULL DEFAULT 'model',
  -- '' = the call ran and returned without error. Model faults:
  -- unknown_tool | invalid_arguments | duplicate_id. Not the model's fault:
  -- execution | denied | aborted | skipped | gate_failed. The value is set at
  -- the SOURCE via ToolResult.details['__fault'] where the reason is known,
  -- never text-matched out of a result string afterwards.
  fault TEXT NOT NULL DEFAULT '',
  duration_ms REAL NOT NULL DEFAULT -1
);

-- Maintained DAY-GRAIN, PER-SESSION rollup: what ``aggregate()`` reads instead
-- of scanning the ledger. Same mechanism as ``usage_daily`` one comment above
-- (accumulate-upsert in the ledger's own transaction) at the one grain that
-- serves all three of ``aggregate()``'s outputs:
--
--   headline      SUM(...) over a day range, no GROUP BY
--   by_provider   GROUP BY provider
--   by_session    GROUP BY session_id, with MAX(parent_session_id)
--
-- WHY THIS EXISTS. ``aggregate()`` used to be three full scans of ``calls`` with
-- a non-covering index range scan, so every one of the ledger's 1.16 M rows cost
-- a random table lookup: measured 4.8 s wall / 2 765 ms CPU for the desktop
-- panel's 30-day window on the operator's 342.8 MB ledger, and 6.4 s for the
-- FIRST touch of a fresh copy, which is what a cold start feels like. The rollup
-- answers the same window in 334 ms wall / 110 ms CPU, reads 1.1 MB instead of
-- the ledger's hundreds, and its cost stops tracking ledger growth.
-- See ``bench/analytics-rollup-before.json`` / ``-after.json``.
--
-- ``day`` is the LOCAL calendar date (YYYY-MM-DD), from the same
-- ``_local_day_month`` the write path already uses for ``usage_daily``, so there
-- is ONE spelling of "which day is this". It is TEXT so a day range sorts
-- lexically. ``model_id`` is deliberately NOT in the key: nothing in
-- ``aggregate()`` groups by model (the per-model view is ``usage_daily``'s job),
-- and a key dimension no read consumes cannot be removed later without a table
-- rebuild.
--
-- ``parent_session_id`` holds ``_PARENT_EDGE_SQL`` FOR THIS BUCKET — the shared
-- parent rule, never a second spelling of it. NULL means "no edge in this
-- bucket"; the read combines buckets with the NULL-ignoring aggregate MAX, and
-- the upsert combines them with the NULL-safe COALESCE form, so the whole-window
-- edge is the same string as the ledger's. (SQLite's multi-argument ``MAX()``
-- returns NULL if ANY argument is NULL, unlike the aggregate — that trap is why
-- the combine is spelled out; see ``_SESSION_DAILY_UPSERT_SQL``.)
--
-- ``max_ts_ms`` is the newest call ts in the bucket and is what the read's
-- in-sync gate compares against the ledger's ``MAX(ts_ms)``.
--
-- NO FORWARD-FILL HERE, unlike ``usage_daily``: the rollup starts empty and is
-- filled by the backfill sweep (``analytics/backfill.py``), because a rollup
-- missing days the ledger still holds would serve a SMALLER number than the
-- audit trail. Until the sweep covers a day, reads of windows touching it fall
-- back to the ledger (see ``_session_daily_window``) — slower, never wrong.
--
-- Appended at the very END of this script on purpose: ``executescript`` runs it
-- as ONE unit, so a statement that raises silently drops every statement AFTER
-- it. Worst case here is losing this table, and nothing that already shipped
-- (see ``_OPTIONAL_INDEXES`` for the full account of that failure mode).
CREATE TABLE IF NOT EXISTS session_daily (
  day               TEXT    NOT NULL,
  session_id        TEXT    NOT NULL,
  provider          TEXT    NOT NULL,
  parent_session_id TEXT,
  ok                INTEGER NOT NULL DEFAULT 0,
  input_tokens      INTEGER NOT NULL DEFAULT 0,
  output_tokens     INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens  INTEGER NOT NULL DEFAULT 0,
  context_tokens    INTEGER NOT NULL DEFAULT 0,
  cost_micro        INTEGER NOT NULL DEFAULT 0,
  cost_known        INTEGER NOT NULL DEFAULT 0,
  calls             INTEGER NOT NULL DEFAULT 0,
  {_COMPONENT_COLUMNS},
  max_ts_ms         INTEGER NOT NULL DEFAULT 0,
  updated_at_ms     INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, session_id, provider)
);
CREATE INDEX IF NOT EXISTS idx_session_daily_session ON session_daily(session_id);
CREATE INDEX IF NOT EXISTS idx_session_daily_provider ON session_daily(provider);

-- One row per key for state the rollup cannot derive from itself. Three keys:
--
-- ``covered_from_day``   every day at or after this is COMPLETE in
--                        ``session_daily``. Absent means "no day is proven
--                        complete yet", which is the honest state of a ledger
--                        whose rollup has not been swept (an upgrade). Only the
--                        backfill lowers it, and only downwards through days it
--                        derived contiguously; the writer never sets it, because
--                        a writer's first batch contributes only the calls it
--                        saw, not the ones the earlier binary already wrote.
-- ``ledger_whole_from_day``   every day at or after this is WHOLE in ``calls`` —
--                        i.e. the retention prune has not cut inside it. Absent
--                        means no prune has ever removed a row, so the ledger's
--                        oldest day is whole by construction. The prune raises
--                        it (never lowers): a window that includes the day the
--                        prune cut is served from the ledger, not the rollup.
-- ``zone``              the local zone name the buckets were labelled in. A
--                        window whose zone differs would misattribute rows near
--                        each day boundary, so the gate refuses and the ledger
--                        path (today's behaviour, never a wrong number) runs.
CREATE TABLE IF NOT EXISTS session_daily_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
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

#: Indexes over OPTIONAL columns, created after ``_migrate`` rather than in
#: ``_SCHEMA``. They cannot live in the schema script: ``executescript`` runs it
#: as one unit BEFORE the ALTERs, so on a ledger written by a release that
#: predates the column, ``CREATE INDEX ... ON calls(parent_session_id)`` raises
#: "no such column" and aborts the REST of the script — losing the rollup tables
#: and latching ``_broken`` for the process. Verified against a pre-column DB.
#:
#: ``idx_calls_parent`` is load-bearing for the subagent rollup: without it a
#: recursive subtree walk is a full scan (measured 154 ms on a 475k-row ledger
#: vs 1.03 ms with it), which is the difference between a viable ``/session``
#: tree total and an unviable one. It is NOT free: building it over an existing
#: 475k-row ledger costs a ONE-TIME ~677 ms stall on the first open after
#: upgrade and ~4.9 MB of file growth (82.2 -> 87.1 MB). That stall lands on
#: the recorder's background thread in the normal path, and buys every later
#: report a two-orders-of-magnitude faster walk, so it is paid once and
#: deliberately.
_OPTIONAL_INDEXES: tuple[tuple[str, str], ...] = (
    (
        "parent_session_id",
        "CREATE INDEX IF NOT EXISTS idx_calls_parent ON calls(parent_session_id)",
    ),
)

#: Indexes on tables OTHER than ``calls``, created next to ``_OPTIONAL_INDEXES``
#: and for the same reason: keeping them out of ``_SCHEMA`` means a failure here
#: costs the index alone, where a raising statement inside ``executescript``
#: aborts the REST of that script — losing the rollup tables and latching
#: ``_broken`` for the process. They are unconditional (the table is created in
#: ``_SCHEMA`` immediately above), so unlike ``_OPTIONAL_INDEXES`` there is no
#: column to gate on. ``session`` serves the per-session report rollup;
#: ``ts_ms`` serves ``prune``'s retention delete.
_TABLE_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(ts_ms)",
)

#: THE parent-edge rule, as one SQL expression, used by every surface that
#: derives session parentage. It exists as a single constant because the two
#: surfaces previously encoded DIFFERENT rules and could therefore disagree
#: about the same session — which is the exact defect this whole rollup was
#: written to eliminate, so letting it back in through two spellings of
#: "who is the parent" would be self-defeating (review F1).
#:
#: Read it inside-out. ``NULLIF(parent_session_id, '')`` discards the "no
#: parent" sentinel (the column is ``NOT NULL DEFAULT ''``); the outer
#: ``NULLIF(..., session_id)`` discards a SELF edge, of which the ledger holds
#: 224 real rows from a degenerate empty id. Both must be discarded BEFORE the
#: ``MAX``, not after it: a plain ``MAX(parent_session_id)`` over a session that
#: has both a real parent and a self row returns whichever id sorts larger, so a
#: real edge is lost whenever the session's own id happens to sort above its
#: parent's. That is a lexical coin-flip, not a rule.
#:
#: ``MAX`` still picks one parent when a session genuinely carries rows under
#: two different parents. That is a deliberate, DOCUMENTED tie-break rather than
#: an oversight: the per-session table must keep summing to the headline total
#: printed above it, and a child credited to two parents is counted twice. The
#: ledger has zero such sessions today (verified), and both surfaces now resolve
#: the tie identically, which is the property that matters — they agree.
#:
#: Measured on the 475k-row ledger: this expression costs 201 ms against 191 ms
#: for the bare ``MAX`` in ``aggregate``'s existing GROUP BY (~10 ms, on a
#: worker thread), and yields the identical 467 edges on today's data.
_PARENT_EDGE_SQL = "MAX(NULLIF(NULLIF(parent_session_id, ''), session_id))"

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

#: The measure columns ``session_daily`` accumulates. ``ok`` and ``calls`` are
#: counts (1 per row), the token/cost fields carry the call's amounts, and the
#: nine ``c_*`` sums are the estimated component split. Kept as one tuple so the
#: upsert, the re-derive and the read projection cannot drift apart.
_SESSION_DAILY_MEASURE_COLUMNS: tuple[str, ...] = (
    "ok",
    *_ROLLUP_MEASURE_COLUMNS,
    *(f"c_{key}" for key in COMPONENT_KEYS),
)

_SESSION_DAILY_INSERT_COLUMNS: tuple[str, ...] = (
    "day",
    "session_id",
    "provider",
    "parent_session_id",
    *_SESSION_DAILY_MEASURE_COLUMNS,
    "max_ts_ms",
    "updated_at_ms",
)

#: The accumulate-upsert for ``session_daily``, in the ledger's own transaction.
#:
#: Two columns are NOT plain accumulates, and both are load-bearing:
#:
#: ``parent_session_id`` — SQLite's multi-argument ``MAX()`` returns NULL when
#: ANY argument is NULL, unlike the aggregate, which ignores NULLs. So a plain
#: ``MAX(old, excluded)`` DROPS a real edge the moment either side is NULL: the
#: sequence ``MAX(NULL, 'aa')`` then ``MAX('aa', NULL)`` leaves NULL. That would
#: re-create the F1 defect ``_PARENT_EDGE_SQL`` exists to prevent (two surfaces
#: disagreeing about who a session's parent is). The COALESCE form below is the
#: NULL-safe combine: both NULL -> NULL, one set -> that one, both set -> the
#: lexical max, which is exactly what the aggregate ``MAX`` over all rows gives.
#:
#: ``max_ts_ms`` — a scalar MAX over two NOT NULL integers, which is safe and is
#: what the read's in-sync gate compares against ``MAX(calls.ts_ms)``.
_SESSION_DAILY_UPSERT_SQL = (
    f"INSERT INTO session_daily ({', '.join(_SESSION_DAILY_INSERT_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _SESSION_DAILY_INSERT_COLUMNS)}) "
    "ON CONFLICT(day, session_id, provider) DO UPDATE SET "
    + ", ".join(f"{col} = {col} + excluded.{col}" for col in _SESSION_DAILY_MEASURE_COLUMNS)
    + ", max_ts_ms = MAX(max_ts_ms, excluded.max_ts_ms)"
    + ", parent_session_id = MAX("
    "COALESCE(session_daily.parent_session_id, excluded.parent_session_id), "
    "COALESCE(excluded.parent_session_id, session_daily.parent_session_id))"
    + ", updated_at_ms = excluded.updated_at_ms"
)

#: Meta keys for ``session_daily_meta``. Named constants because the writer, the
#: backfill, the prune and the read gate all have to agree on the spelling.
_SESSION_DAILY_META_COVERED = "covered_from_day"
_SESSION_DAILY_META_LEDGER_WHOLE = "ledger_whole_from_day"
_SESSION_DAILY_META_ZONE = "zone"

#: Monotone (downwards) coverage write, used by the backfill after each committed
#: day. ``MIN`` is what makes a pass resumable and a re-run harmless: a pass that
#: re-derives an already-covered day cannot raise the watermark, and a pass that
#: derives a day one older than the frontier lowers it by exactly one day.
_SESSION_DAILY_COVERAGE_SQL = (
    "INSERT INTO session_daily_meta(key, value) VALUES(?, ?) "
    "ON CONFLICT(key) DO UPDATE SET value = MIN(value, excluded.value)"
)

#: First-writer-wins meta write, used for the zone (and by the prune, which only
#: ever RAISES the ledger-whole watermark, via ``_SESSION_DAILY_RAISE_SQL``).
_SESSION_DAILY_INSERT_META_SQL = (
    "INSERT INTO session_daily_meta(key, value) VALUES(?, ?) " "ON CONFLICT(key) DO NOTHING"
)

#: Monotone (upwards) watermark write: the prune uses it for
#: ``ledger_whole_from_day``, which may only ever move forward as the ledger's
#: bottom moves forward.
_SESSION_DAILY_RAISE_SQL = (
    "INSERT INTO session_daily_meta(key, value) VALUES(?, ?) "
    "ON CONFLICT(key) DO UPDATE SET value = MAX(value, excluded.value)"
)


def _session_daily_rederive_sql() -> str:
    """The whole-day rebuild of ``session_daily`` from the ledger.

    One statement per day, in the backfill's ``BEGIN IMMEDIATE`` transaction:
    the day's buckets are dropped and recomputed from ``calls``. Buckets, not
    rows: the SELECT groups at the table's own grain, and the parent edge comes
    from ``_PARENT_EDGE_SQL`` — the shared rule — so a re-derived day and an
    accumulated day are the same value, not merely a similar one.

    Placeholders, in order: the day sting, ``updated_at_ms``, and the day's
    ``[start_ms, end_ms)`` bounds.
    """
    measures = ", ".join(
        # ``calls`` is a COUNT on this side: the ledger has one ROW per call where
        # the bucket has a accumulated count column, and the ledger table is
        # itself called ``calls``, so a bare ``SUM(calls)`` would be a no-such-
        # column error rather than a mistake anyone could read.
        "COUNT(*)" if col == "calls" else f"SUM({col})"
        for col in _SESSION_DAILY_MEASURE_COLUMNS
    )
    columns = ", ".join(_SESSION_DAILY_INSERT_COLUMNS)
    return (
        f"INSERT INTO session_daily ({columns}) "
        f"SELECT ?, session_id, provider, {_PARENT_EDGE_SQL}, {measures}, "
        "MAX(ts_ms), ? FROM calls WHERE ts_ms >= ? AND ts_ms < ? "
        "GROUP BY session_id, provider"
    )


_SESSION_DAILY_REDERIVE_SQL = _session_daily_rederive_sql()

#: The measure sums ``session_daily``'s READ projects. Positionally identical to
#: ``_aggregate_from_row``'s contract: ``calls`` first (it takes COUNT(*)'s
#: place), then ok, the six token sums, the two cost sums, then the components.
_SESSION_DAILY_READ_SUMS = (
    "SUM(calls)",
    "SUM(ok)",
    "SUM(input_tokens)",
    "SUM(output_tokens)",
    "SUM(cache_read_tokens)",
    "SUM(cache_write_tokens)",
    "SUM(reasoning_tokens)",
    "SUM(context_tokens)",
    "SUM(cost_micro)",
    "SUM(cost_known)",
    *(f"SUM(c_{key})" for key in COMPONENT_KEYS),
)
_SESSION_DAILY_READ_COLUMNS_SQL = ", ".join(_SESSION_DAILY_READ_SUMS)

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


def _component_split(snapshot: CallSnapshot) -> dict[str, int]:
    """The ESTIMATED component split for one call, computed once per snapshot.

    Computed in the writer, against the authoritative ``context_tokens`` — never
    on the event loop. Split out of ``_row_values`` so the ledger row and the
    ``session_daily`` buckets are fed the SAME apportionment rather than each
    recomputing it (and, more importantly, so they cannot disagree).
    """
    return apportion_components(snapshot.component_chars, snapshot.context_tokens)


def _row_values(
    snapshot: CallSnapshot,
    cost_micro: int,
    cost_known: bool,
    components: dict[str, int],
) -> tuple[Any, ...]:
    """A snapshot as the positional tuple ``_INSERT_SQL`` expects.

    ``components`` is passed in already apportioned (see ``_component_split``)
    so the ledger row and the rollup buckets share one split. A call the
    provider gave no context total for stores 0s for every component, which
    reads as "unknown" rather than a fabricated breakdown.

    ``cost_micro``/``cost_known`` are passed in already priced (once per
    snapshot in ``record_batch``) rather than priced here, so the same figure
    feeds this ledger row AND the rollup rows without calling the potentially
    cold ``resolve_model_info`` twice for one call.
    """
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


def _local_zone_key() -> str:
    """A STABLE name for the zone this process buckets days in.

    The rollup's ``zone`` meta value exists so a read can refuse when the
    machine's zone has changed since the buckets were labelled (travel, or a
    laptop whose offset rule differs) — see ``AnalyticsStore._session_daily_window``.
    The comparison is only useful if the name does not change on its own, so
    this deliberately does NOT use ``tzname()``: that returns the DST-specific
    ABBREVIATION (``EDT`` in July, ``EST`` in January), and a zone that flips
    abbreviation twice a year would refuse the fast path for half of it.

    Order: ``TZ`` when the user set it (it is what ``localtime`` resolves
    through), then the IANA key behind ``/etc/localtime`` (``America/Toronto``
    from ``/var/db/timezone/zoneinfo/America/Toronto`` on macOS,
    ``/usr/share/zoneinfo/Europe/London`` on Linux), then the abbreviation as a
    last resort. The fallback is the DST-sensitive one and that is accepted:
    a name that changes costs the fast path (slow), never a wrong number.
    """
    env_zone = os.environ.get("TZ", "").strip()
    if env_zone:
        return env_zone
    try:
        target = os.path.realpath("/etc/localtime")
        marker = "zoneinfo/"
        index = target.find(marker)
        if index >= 0:
            name = target[index + len(marker) :]
            if name:
                return name
    except OSError:
        pass
    try:
        return datetime.now().astimezone().tzname() or ""
    except Exception:  # noqa: BLE001 — an unresolvable zone is an empty key
        return ""


def _local_day_bounds_ms(day: str) -> tuple[int, int]:
    """``[start_ms, end_ms)`` of a local ``YYYY-MM-DD`` day, DST-correct.

    The inverse of :func:`_local_day_month`, and it has to be computed per day
    rather than from a fixed day length: a DST transition makes a local day 23
    or 25 hours, so the ledger rows belonging to one bucket are not a
    ``86_400_000``-wide slice of ``ts_ms``. A naive local ``datetime``'s
    ``.timestamp()`` resolves through the same zone rules ``fromtimestamp``
    uses, so the two directions agree on which day an instant is in.
    """
    start = datetime.strptime(day, "%Y-%m-%d")
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _day_shift(day: str, days: int) -> str:
    """``day`` moved by ``days`` local calendar days (negative moves back).

    Calendar arithmetic on the date, not the timestamp, so it never lands at
    23:00 of the wrong day across a DST transition.
    """
    return (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def _is_day_boundary(ts_ms: int) -> bool:
    """Whether ``ts_ms`` is the first instant of a LOCAL day.

    Written as "is this instant the first in its day" rather than "is the clock
    at 0000h", which makes it zone- and DST-agnostic: true for an ordinary
    local midnight, for a DST-shifted midnight, and for the first valid instant
    of a day whose 0000h does not exist — false for everything else. The day
    grain depends on exactly this property, so the read gate is built on it
    instead of on a string comparison that would silently accept 00:00:01.
    """
    return _local_day_month(int(ts_ms))[0] != _local_day_month(int(ts_ms) - 1)[0]


def _parent_edge_for(session_id: str, parent_session_id: str) -> str | None:
    """``_PARENT_EDGE_SQL`` for ONE row, as a Python value (``None`` = no edge).

    The writer accumulates a bucket in Python before the upsert, so it needs
    the rule's per-row half here; the SQL half is applied verbatim by the
    re-derive (``_SESSION_DAILY_REDERIVE_SQL``) so the two entry points cannot
    disagree about which edges count.
    """
    if not parent_session_id or parent_session_id == session_id:
        return None
    return parent_session_id


def _combine_parent_edges(left: str | None, right: str | None) -> str | None:
    """The NULL-safe combine two partial maxima merge with (SQL's MAX(a, b) is not).

    Mirrors the COALESCE form in ``_SESSION_DAILY_UPSERT_SQL`` exactly: both
    empty -> empty, one set -> that one, both set -> the lexical max. The
    property test in ``tests/unit/analytics/test_session_daily_rollup.py`` pins
    that this matches the aggregate ``MAX`` over the union of the two row sets.
    """
    if left is None:
        return right
    if right is None:
        return left
    return left if left >= right else right


def _session_daily_rows(
    snapshots: Sequence[CallSnapshot],
    priced: Sequence[tuple[int, bool]],
    splits: Sequence[dict[str, int]],
    now_ms: int,
) -> list[tuple[Any, ...]]:
    """The ``session_daily`` upsert rows for one batch, one per bucket.

    Accumulated in Python rather than with ``INSERT ... SELECT ... GROUP BY``
    because the write path already holds every value: a batch is a handful of
    snapshots, and grouping here keeps the upsert's measured cost (~tens of
    microseconds over the same transaction's commit) instead of adding a second
    read of the ledger to the highest-volume write path in the repo.

    Buckets are keyed by ``(local day, session_id, provider)``. The parent edge
    is combined as the batch accumulates so a session carrying two distinct
    parents inside one batch still resolves to the same string the aggregate
    ``MAX`` over those rows would give.
    """
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for snap, (cost_micro, cost_known), components in zip(snapshots, priced, splits):
        day, _ = _local_day_month(int(snap.ts_ms))
        key = (day, snap.session_id, snap.provider)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "measures": [0] * len(_SESSION_DAILY_MEASURE_COLUMNS),
                "max_ts_ms": 0,
                "parent": None,
            }
            buckets[key] = bucket
        measures = bucket["measures"]
        measures[0] += 1 if snap.ok else 0
        measures[1] += snap.input_tokens
        measures[2] += snap.output_tokens
        measures[3] += snap.cache_read_tokens
        measures[4] += snap.cache_write_tokens
        measures[5] += snap.reasoning_tokens
        measures[6] += snap.context_tokens
        measures[7] += int(cost_micro)
        measures[8] += 1 if cost_known else 0
        measures[9] += 1
        for offset, component in enumerate(COMPONENT_KEYS):
            measures[10 + offset] += components[component]
        bucket["max_ts_ms"] = max(bucket["max_ts_ms"], int(snap.ts_ms))
        bucket["parent"] = _combine_parent_edges(
            bucket["parent"], _parent_edge_for(snap.session_id, snap.parent_session_id)
        )
    return [
        (
            day,
            session_id,
            provider,
            bucket["parent"],
            *bucket["measures"],
            bucket["max_ts_ms"],
            now_ms,
        )
        for (day, session_id, provider), bucket in buckets.items()
    ]


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
        #: Whether ``session_daily`` exists on THIS database, for the same
        #: never-break-a-turn reason the optional columns get: a ledger whose
        #: schema script aborted before reaching the rollup's CREATE TABLE (see
        #: ``_OPTIONAL_INDEXES`` for how that happens) must keep recording calls
        #: rather than failing every batch on a missing table. Absent means the
        #: write path omits the rollup upsert and every read takes the ledger
        #: path, which is today's behaviour exactly.
        self._has_session_daily = True
        #: Which path the LAST ``aggregate()`` on this instance took: ``"rollup"``
        #: or ``"ledger"`` (``""`` before the first call). A diagnostic seam and
        #: a test seam in one: the gate's refusals are only trustworthy if a
        #: test can see WHICH path ran without timing anything (AGENTS.md
        #: §Timing — assert the structure, not a latency). The refusal itself is
        #: also logged at ``debug`` with its reason, so a permanent fallback on a
        #: real machine is diagnosable rather than invisible.
        self._last_aggregate_source = ""
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
            self._has_session_daily = False
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
        # Whether the per-session day rollup table is actually there. It is
        # created in ``_SCHEMA`` (it has no migrated column, so it needs none of
        # ``_OPTIONAL_INDEXES``' machinery), but ``executescript`` aborts the
        # rest of that script on the first raising statement — so a ledger that
        # lost the tail of the script, or one whose file went read-only mid-open,
        # is detected here and degrades to "no rollup": the ledger keeps
        # recording and every read takes the ledger path, rather than every
        # batch failing on a missing table.
        try:
            self._has_session_daily = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_daily'"
                ).fetchone()
                is not None
            )
        except Exception:  # noqa: BLE001 — no rollup table means no fast path
            logger.debug("analytics: could not inspect for session_daily", exc_info=True)
            self._has_session_daily = False
        self._rebuild_insert_plan()
        self._create_optional_indexes(conn, existing)

    @staticmethod
    def _create_optional_indexes(conn: sqlite3.Connection, existing: set[str]) -> None:
        """Index the optional columns that this DB actually has.

        Runs AFTER the ALTERs so the column is guaranteed present; see
        ``_OPTIONAL_INDEXES`` for why this cannot be part of ``_SCHEMA``. Each
        statement is guarded on its own so one failure (a read-only file, a
        disk-full during the ~677 ms build) costs its index and nothing else —
        analytics degrades to the slower scan rather than failing to open.
        """
        for column, statement in _OPTIONAL_INDEXES:
            if column not in existing:
                continue
            try:
                conn.execute(statement)
            except Exception:  # noqa: BLE001 — a missing index is slow, not broken
                logger.debug("analytics: could not create index on %s", column, exc_info=True)
        for statement in _TABLE_INDEXES:
            try:
                conn.execute(statement)
            except Exception:  # noqa: BLE001 — a missing index is slow, not broken
                logger.debug("analytics: could not create a table index", exc_info=True)

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
        # The component split is apportioned ONCE per call here and handed to
        # both the ledger row and the ``session_daily`` buckets, so the two can
        # never disagree about a call's estimate.
        splits = [_component_split(s) for s in snapshots]
        rows = [
            _row_values(s, cm, ck, components)
            for s, (cm, ck), components in zip(snapshots, priced, splits)
        ]
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
        # The per-session day rollup the panel actually reads. Same transaction,
        # same write path, same accumulate discipline as the two calendar
        # rollups above — which is what makes it impossible to double-count a
        # call or to record a turn in the ledger without it.
        session_rows = (
            _session_daily_rows(snapshots, priced, splits, self._now_ms())
            if self._has_session_daily
            else []
        )
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
                if session_rows:
                    # Guarded as a GROUP, not per statement: like the two
                    # calendar rollups this is one transaction, so a failed
                    # upsert rolls the whole attempt back and retries rather
                    # than committing a ledger row the rollup never saw.
                    conn.executemany(_SESSION_DAILY_UPSERT_SQL, session_rows)
                    conn.execute(
                        _SESSION_DAILY_INSERT_META_SQL,
                        (_SESSION_DAILY_META_ZONE, _local_zone_key()),
                    )
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

    def record_tool_calls(self, rows: Sequence[tuple[int, str, str, str, str, float]]) -> int:
        """Insert tool-call samples in one transaction. Returns rows written.

        ``rows`` are ``(ts_ms, session_id, tool_name, origin, fault, duration_ms)``
        — a plain tuple rather than a dataclass because the producer is the
        harness, which deliberately has no analytics import, so the shape has to
        survive a ``LoopConfig`` callback signature.

        Batched, retried on ``SQLITE_BUSY`` and best-effort exactly like
        :meth:`record_batch`, and on the SAME writer thread and connection.
        Never raises: a lost tool sample is a slightly-wrong accuracy figure,
        and a raise here would be a broken turn.

        Deliberately its OWN transaction rather than joined to the ledger write:
        tool calls are produced during a turn while the ledger row is written at
        the end of the provider stream, so the two never arrive in the same
        batch and pairing them would only delay one of them.
        """
        if not rows:
            return 0
        conn: sqlite3.Connection | None = None
        for attempt in range(_WRITE_RETRIES):
            conn = self._connect()
            if conn is not None or self._broken:
                break
            time.sleep(_WRITE_RETRY_BACKOFF_S * (attempt + 1))
        if conn is None:
            return 0
        sql = (
            "INSERT INTO tool_calls "
            "(ts_ms, session_id, tool_name, origin, fault, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        for attempt in range(_WRITE_RETRIES):
            try:
                conn.executemany(sql, rows)
                conn.commit()
                return len(rows)
            except sqlite3.OperationalError as exc:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
                if not _is_lock_error(exc):
                    logger.debug("analytics: tool-call insert failed", exc_info=True)
                    return 0
                if attempt == _WRITE_RETRIES - 1:
                    logger.debug("analytics: tool-call batch dropped after busy retries")
                    return 0
                time.sleep(_WRITE_RETRY_BACKOFF_S * (attempt + 1))
            except Exception:  # noqa: BLE001 — a lost batch must not kill the writer
                logger.debug("analytics: tool-call insert failed", exc_info=True)
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
        may correct the row, a weaker one may only fill an empty slot — and an
        EMPTY incumbent name counts as an empty slot at any rank (see the
        ``session_names.name = ''`` arm of the gate below).

        Equal rank still overwrites, deliberately: a re-title is the same rank
        as the title it replaces and MUST be able to replace it.

        An empty ``name`` is rejected outright rather than written. Storing one
        creates a row that is simultaneously PRESENT (so it pins a rank) and
        MISSING (``sessions_missing_names`` selects on ``name = ''``), which the
        startup backfill would re-derive and re-attempt on every launch forever
        while the rank gate rejected it — unbounded work that can never
        converge. The recorder already guards this; the store is the shared
        surface, so the guard belongs here too.
        """
        if not session_id or not name:
            return
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO session_names (session_id, name, updated_at_ms, rank) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "name=excluded.name, updated_at_ms=excluded.updated_at_ms, "
                "rank=excluded.rank WHERE excluded.rank >= session_names.rank "
                "OR session_names.name = ''",
                (session_id, name, self._now_ms(), int(rank)),
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
        - ``tool_calls`` keeps the SAME ``retention_days`` window by ``ts_ms``.
          It is a raw per-event ledger like ``calls`` and grows with tool-call
          volume rather than request volume, so it needs the bound more than
          ``calls`` does. Its rows are NOT added to the return value, which
          contractually counts raw-ledger requests.
        - ``usage_daily`` keeps the most recent 365 DISTINCT ``day`` values
          (not 365 rows — each day holds one row per model), so the daily bar
          can look back a year regardless of how many models ran. The subquery
          finds the 365th-newest distinct day and deletes everything older.
        - ``session_daily`` keeps the SAME 365 distinct days, for the same
          "newest N buckets that exist" reason. It is not read by
          ``daily_series``; it is what ``aggregate()`` reads, and it deliberately
          outlives the ledger's 90 days so the day-grain history does not
          evaporate with the per-call rows. Serving a window the ledger has
          already dropped is a separate decision, and the gate makes it a
          refusal rather than an over-count.
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
        watermark = ""
        try:
            cur = conn.execute("DELETE FROM calls WHERE ts_ms < ?", (cutoff,))
            removed = cur.rowcount or 0
            if removed:
                # WHERE THE LEDGER'S BOTTOM MOVED TO, computed and written in
                # the SAME transaction as the delete. Deleting by ``ts_ms`` cuts
                # INSIDE the oldest surviving local day: the ledger then holds
                # only that day's remainder while the rollup holds the day whole,
                # so a window reaching it would over-count. Recording the first
                # day the ledger is whole again is what lets the read gate refuse
                # those windows instead of serving them; doing it here rather
                # than after the commit is what keeps the ledger from ever being
                # seen pruned with no record of where its bottom went.
                try:
                    row = conn.execute("SELECT MIN(ts_ms) FROM calls").fetchone()
                    oldest_raw = None if row is None else row[0]
                    if oldest_raw is not None:
                        oldest_ms = int(oldest_raw)
                        watermark = _local_day_month(oldest_ms)[0]
                        if not _is_day_boundary(oldest_ms):
                            watermark = _day_shift(watermark, 1)
                        conn.execute(
                            _SESSION_DAILY_RAISE_SQL,
                            (_SESSION_DAILY_META_LEDGER_WHOLE, watermark),
                        )
                except (
                    Exception
                ):  # noqa: BLE001 — an unrecorded bottom is a refusal, not a wrong read
                    logger.debug(
                        "analytics: could not record the pruned ledger bottom", exc_info=True
                    )
                    watermark = ""
                else:
                    logger.debug(
                        "analytics: ledger pruned below %s, so the rollup will not serve below it",
                        watermark,
                    )
            conn.commit()
        except Exception:  # noqa: BLE001
            logger.debug("analytics: prune failed", exc_info=True)
        # Guarded on its own, like the rollups: a DB whose ``tool_calls`` table
        # predates this feature (or failed to create) must still prune what it
        # can rather than losing the ledger delete above.
        try:
            conn.execute("DELETE FROM tool_calls WHERE ts_ms < ?", (cutoff,))
            conn.commit()
        except Exception:  # noqa: BLE001 — a tool-call prune failure is non-fatal
            logger.debug("analytics: tool-call prune failed", exc_info=True)
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
            # The per-session day rollup keeps the SAME reach as the daily
            # calendar rollup (365 distinct days), and it is the same subquery
            # shape, so one constant governs both. It deliberately SURVIVES the
            # ledger's 90-day prune: its rows are day-grain aggregates, not
            # per-call ones, and the whole point of a rollup is to outlive the
            # rows it summarises. The read gate is what keeps that from moving a
            # number — a window the ledger can no longer answer is refused and
            # answered from the ledger, so `aggregate()` never reports history
            # the audit trail has dropped (see ``_session_daily_window``).
            conn.execute(
                "DELETE FROM session_daily WHERE day < ("
                "  SELECT MIN(day) FROM ("
                "    SELECT DISTINCT day FROM session_daily ORDER BY day DESC LIMIT ?"
                "  )"
                ")",
                (DAILY_ROLLUP_RETENTION_DAYS,),
            )
            conn.commit()
        except Exception:  # noqa: BLE001 — a rollup prune failure is non-fatal
            logger.debug("analytics: rollup prune failed", exc_info=True)
        return removed

    def session_daily_state(self) -> dict[str, str]:
        """The rollup's meta map, or ``{}`` when it cannot be read.

        The backfill reads it to find its frontier; a store that has no rollup
        table yet (or an unreadable one) reads as empty, which the sweep treats
        as "start from the top" rather than as an error.
        """
        conn = self._read_connection()
        if conn is None:
            return {}
        try:
            return {
                str(row[0]): str(row[1])
                for row in conn.execute("SELECT key, value FROM session_daily_meta")
            }
        except Exception:  # noqa: BLE001 — no meta is "nothing covered"
            return {}
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def ledger_day_span(self) -> tuple[str, str] | None:
        """``(oldest local day, newest local day)`` the ledger holds, or ``None``.

        Both bounds come from indexed aggregates (``MIN``/``MAX`` on
        ``idx_calls_ts``), so this is cheap enough for the read gate AND for the
        backfill's per-pass worklist. The earliest day is the FLOOR of the
        sweep: deriving a day the ledger no longer holds would delete history
        the rollup exists to keep, so the sweep never walks below it.
        """
        conn = self._read_connection()
        if conn is None:
            return None
        try:
            row = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM calls").fetchone()
            if row is None or row[0] is None or row[1] is None:
                return None
            return _local_day_month(int(row[0]))[0], _local_day_month(int(row[1]))[0]
        except Exception:  # noqa: BLE001 — an unreadable ledger has no worklist
            logger.debug("analytics: could not read the ledger day span", exc_info=True)
            return None
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def session_daily_worklist(self, *, max_days: int) -> list[str]:
        """The local days one backfill pass should re-derive, newest first.

        Two jobs, one mechanism (they are the same ``DELETE day + INSERT from
        ledger`` sweep):

        - **the recent window** — the newest three local days, every pass. This
          is what heals a hole left by a ``lop`` still running the pre-rollup
          binary, and the mid-day boundary an upgrade leaves in today's bucket
          (the writer only ever adds the calls it saw, so today is the one day
          that can start out half-recorded).
        - **the historical hole** — every day below the frontier down to the
          ledger's oldest day. Newest-first is what lets the watermark advance
          per committed day, so a read that only needs the last few days becomes
          fast after a few transactions instead of after the whole sweep.

        Bounded by the LEDGER, not by a constant: the walk stops at the oldest
        day the ledger still holds, because going below it would delete rollup
        history that survived the prune (the retention rule is that the rollup
        outlives the ledger, not that the ledger bounds it). ``max_days`` only
        caps how much of that bounded work ONE pass takes on; whatever is left
        is picked up next launch, since every pass re-derives its worklist.
        """
        span = self.ledger_day_span()
        if span is None:
            return []
        oldest_day, newest_day = span
        today = _local_day_month(self._now_ms())[0]
        top = max(today, newest_day)
        covered = self.session_daily_state().get(_SESSION_DAILY_META_COVERED, "")
        days: list[str] = [top, _day_shift(top, -1), _day_shift(top, -2)]
        # ``covered`` present => every day at or after it is complete, so the
        # sweep resumes one day below it; absent => nothing is proven, so it
        # starts just under the recent window.
        day = _day_shift(covered, -1) if covered else _day_shift(top, -3)
        while day >= oldest_day and len(days) < max_days:
            days.append(day)
            day = _day_shift(day, -1)
        seen: set[str] = set()
        ordered: list[str] = []
        for day in sorted((d for d in days if oldest_day <= d <= top), reverse=True):
            if day not in seen:
                seen.add(day)
                ordered.append(day)
        return ordered[:max_days] if max_days > 0 else []

    def rederive_session_daily_day(self, day: str) -> int | None:
        """Rebuild one local day's buckets from the ledger. ``None`` = not committed.

        Returns the number of buckets written (0 is a legitimate answer: a day
        with no calls is COMPLETE and empty, not missing), or ``None`` when the
        transaction did not commit — which is what stops the coverage watermark
        from advancing past a day that is not actually there.

        ``BEGIN IMMEDIATE``, not a bare transaction: without it the SELECT runs
        outside the write transaction and a call committed between the DELETE
        and the INSERT is silently dropped from the rollup until the next sweep.
        Holding the write lock also means no writer can interleave, and under WAL
        no reader can observe the uncommitted DELETE — so a report never sees a
        half-rebuilt day (it sees the old whole day or the new whole day).

        Runs on the store-maintenance thread (``asyncio.to_thread``), through the
        store's own per-thread connection, which is the same discipline the
        session-name backfill already follows: it is never the FIRST connection
        to a fresh file (the pass runs after the recorder exists) and it inherits
        ``busy_timeout`` and the bounded write retry.
        """
        if not self._has_session_daily:
            return None
        conn = self._connect()
        if conn is None:
            return None
        start_ms, end_ms = _local_day_bounds_ms(day)
        try:
            if conn.in_transaction:
                conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            # The bucketing zone is recorded by the FIRST write to this table and
            # never rewritten, and a sweep REFUSES to run under a different zone
            # than the one already recorded. Both halves are load-bearing: the
            # zone is what makes a changed zone a refusal rather than a wrong
            # number at every day boundary, and without the refusal a re-derive
            # under a new zone would leave the table holding buckets labelled by
            # TWO different rules while the meta still named the old one — the
            # one state in which the gate could be talked into serving a mixed
            # table. Re-bucketing after a move is a separate, deliberate sweep.
            recorded = conn.execute(
                "SELECT value FROM session_daily_meta WHERE key = ?",
                (_SESSION_DAILY_META_ZONE,),
            ).fetchone()
            current_zone = _local_zone_key()
            if recorded is not None and str(recorded[0]) != current_zone:
                conn.rollback()
                logger.debug(
                    "analytics: not re-deriving %s: buckets are in zone %s, process is in %s",
                    day,
                    recorded[0],
                    current_zone,
                )
                return None
            conn.execute(_SESSION_DAILY_INSERT_META_SQL, (_SESSION_DAILY_META_ZONE, current_zone))
            conn.execute("DELETE FROM session_daily WHERE day = ?", (day,))
            cursor = conn.execute(
                _SESSION_DAILY_REDERIVE_SQL, (day, self._now_ms(), start_ms, end_ms)
            )
            written = int(cursor.rowcount or 0)
            conn.execute(_SESSION_DAILY_COVERAGE_SQL, (_SESSION_DAILY_META_COVERED, day))
            conn.commit()
            return written
        except Exception:  # noqa: BLE001 — a failed chunk leaves a consistent table
            logger.debug("analytics: could not re-derive %s", day, exc_info=True)
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            return None

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

        TWO PATHS, ONE ANSWER. The same three grouped reads exist twice: over
        ``session_daily`` (the maintained day-grain rollup, cheap, and what the
        panel opens with) and over ``calls`` (the raw ledger, the original shape
        and now the FALLBACK). Which one runs is decided by
        :meth:`_session_daily_window`, which returns a refusal reason instead of
        a window whenever any precondition does not hold. FAIL CLOSED is the
        whole point: a wrong fast-path number is far worse than a slow one, so
        every precondition is checked and the ledger path — today's behaviour,
        bit for bit — runs otherwise.

        The two paths assemble their rows through :meth:`_assemble_aggregate`,
        so `dataclasses.asdict` equality between them is structural rather than
        a thing two code blocks are separately trusted to maintain.
        """
        conn = self._read_connection()
        if conn is None:
            return UsageAggregate()
        try:
            window, refusal = self._session_daily_window(conn, since_ms, until_ms)
            if window is not None:
                try:
                    self._last_aggregate_source = "rollup"
                    return self._session_daily_aggregate(conn, window, session_id)
                except Exception:  # noqa: BLE001 — a fast path that fails is a slow path
                    logger.debug(
                        "analytics: session_daily read failed, using the ledger", exc_info=True
                    )
                    refusal = "rollup-read-failed"
            self._last_aggregate_source = "ledger"
            # Named, not silent: a refusal that never expires is a feature that
            # silently stopped working, and the only way to see that on a real
            # machine is to say which precondition failed.
            logger.debug("analytics: aggregate on the ledger (%s)", refusal)
            return self._ledger_aggregate(
                conn, since_ms=since_ms, until_ms=until_ms, session_id=session_id
            )
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def last_aggregate_source(self) -> str:
        """WHICH path the last :meth:`aggregate` took: ``rollup``, ``ledger`` or ``""``.

        The gate is only trustworthy if its refusals are observable, and a
        timing assertion is the wrong instrument for that (it is a bet on
        machine load — AGENTS.md §Timing). This is the structural fact instead:
        a test asserts the path, never a duration.
        """
        return self._last_aggregate_source

    def _ledger_aggregate(
        self,
        conn: sqlite3.Connection,
        *,
        since_ms: int | None,
        until_ms: int | None,
        session_id: str | None,
    ) -> UsageAggregate:
        """The original read: three grouped scans of the raw ``calls`` ledger.

        Unchanged apart from taking its connection from the caller and sharing
        the row-set assembly with the rollup path, so the fallback really is
        today's behaviour rather than a re-derivation of it.
        """
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
                "GROUP BY provider ORDER BY provider",
                params,
            ).fetchall()
            # The edge rides the GROUP BY the query already computes rather than
            # costing a second pass. ``_PARENT_EDGE_SQL`` is THE rule, shared
            # verbatim with the ``/session`` subtree walk
            # (``_canonical_parents``) so the two surfaces cannot drift into
            # disagreeing about who a session's parent is (review F1). Measured
            # +6 ms on 475k rows, against +58 ms for a global recursive CTE and
            # 610 ms (vs 290) for widening the GROUP BY to two columns, which
            # forces a temp B-tree for identical output.
            parent_col = _PARENT_EDGE_SQL if self._has_parent_column() else "''"
            per_session = conn.execute(
                f"SELECT session_id, {parent_col}, {base_cols}, {component_sum} FROM calls{clause} "
                "GROUP BY session_id ORDER BY session_id",
                params,
            ).fetchall()
            names = self._session_names(conn)
        except Exception:  # noqa: BLE001 — a report must never raise
            logger.debug("analytics: aggregate query failed", exc_info=True)
            return UsageAggregate()

        return self._assemble_aggregate(top, per_provider, per_session, names)

    def _session_daily_window(
        self, conn: sqlite3.Connection, since_ms: int | None, until_ms: int | None
    ) -> tuple[tuple[str, str | None] | None, str]:
        """THE GATE. Returns ``((day_lo, day_hi), "")`` to serve the rollup.

        Any other outcome is ``(None, reason)`` and the caller runs the ledger
        query unchanged. There is deliberately no third option: this answers
        "may the rollup be trusted for this window", never "is the rollup close
        enough". A window it cannot prove is a window it refuses, because the
        cost of a wrong number on this screen is not comparable to the cost of
        being slow (the operator's rule: wrong beats slow).

        The five preconditions, and why each one can fail on a real machine:

        ``no-rollup-table``  the schema script aborted before the CREATE TABLE,
            or this is a ledger old enough that it never ran it.
        ``empty-ledger``     nothing to serve; the ledger path is trivially fast.
        ``no-parent-column``  a ledger predating ``parent_session_id``. The
            rollup's buckets are derived from the SAME snapshots, so their
            edges would be visible where the ledger's ``_PARENT_EDGE_SQL``
            substitutes ``''`` — the two surfaces would disagree about the tree.
        ``not-day-aligned``  ``aggregate()`` accepts arbitrary millisecond
            bounds; the rollup is day-grain. A bound that is not the first
            instant of a local day cannot be expressed as a day range, so the
            ledger answers it. This is the assumption the whole design rests on
            (both real callers are day-aligned or unbounded) and therefore the
            first thing the gate refuses.
        ``zone-changed``     the buckets were labelled in a different local zone
            (travel, a laptop moved across a zone, a re-run under ``TZ=``).
            Historic buckets are then labelled by the old rule while the read
            derives local midnights by the new one, which misattributes rows
            near every day boundary by up to the offset delta.
        ``no-coverage``      the backfill has not swept far enough down yet: days
            at or after ``covered_from_day`` are complete, anything older is a
            hole. Also the state of every existing ledger on the upgrade launch,
            which is why the panel's first read after upgrading is still the
            ledger's.
        ``ledger-bottom-partial``  the retention prune cut INSIDE the ledger's
            oldest surviving day. The rollup holds that day whole while the
            ledger holds only its remainder, so a window including it would
            over-count. The prune records where the cut landed
            (``ledger_whole_from_day``); this refuses windows reaching below it.
        ``tail-unsynced``    the ledger's newest row is newer than the rollup's.
            The two advance in one transaction, so this means something wrote
            ``calls`` WITHOUT maintaining the rollup — a ``lop`` still running
            the pre-rollup binary. Cheap to check (both sides are an indexed
            MAX) and it is the difference between "the rollup is current" and
            "the rollup is current as of whenever it was last written".

        The window is returned as a half-open ``day`` string range for
        ``day >= lo AND day < hi``, built from ``_local_day_month`` — the same
        function the writer buckets with, so there is one spelling of "which day
        is this". ``day_lo`` is additionally clamped up to the ledger's oldest
        day: post-prune the rollup can hold days the ledger has dropped, and
        serving them from an unbounded window would return rows the ledger path
        cannot.
        """
        if not self._has_session_daily:
            return None, "no-rollup-table"
        if since_ms is not None and not _is_day_boundary(int(since_ms)):
            return None, "not-day-aligned"
        if until_ms is not None and not _is_day_boundary(int(until_ms)):
            return None, "not-day-aligned"
        if not self._has_parent_column():
            return None, "no-parent-column"
        try:
            span = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM calls").fetchone()
            oldest_raw = None if span is None else span[0]
            newest_raw = None if span is None else span[1]
            if oldest_raw is None or newest_raw is None:
                return None, "empty-ledger"
            rollup_span = conn.execute(
                "SELECT MIN(day), MAX(max_ts_ms) FROM session_daily"
            ).fetchone()
            meta = {
                str(row[0]): str(row[1])
                for row in conn.execute("SELECT key, value FROM session_daily_meta")
            }
        except Exception as exc:  # noqa: BLE001 — an unreadable rollup has no fast path
            logger.debug("analytics: session_daily gate could not read state", exc_info=True)
            return None, f"unreadable: {type(exc).__name__}"
        if rollup_span is None:
            return None, "no-coverage"
        if meta.get(_SESSION_DAILY_META_ZONE, "") != _local_zone_key():
            return None, "zone-changed"
        covered = meta.get(_SESSION_DAILY_META_COVERED, "")
        if not covered:
            return None, "no-coverage"
        # The newest row on each side is the in-sync check. The ledger side is an
        # indexed MAX (``idx_calls_ts``); the rollup side scans its own 9 060
        # buckets (1.1 MB, ~1-2 ms) because the PRIMARY KEY is the day/session/
        # provider grain rather than ``max_ts_ms`` — an index for it would cost
        # every write to save a low single-digit millisecond on a read that is
        # already two orders of magnitude inside its budget.
        if int(rollup_span[1] or 0) != int(newest_raw):
            return None, "tail-unsynced"
        ledger_day = _local_day_month(int(oldest_raw))[0]
        day_lo = (
            ledger_day if since_ms is None else max(_local_day_month(int(since_ms))[0], ledger_day)
        )
        day_hi = None if until_ms is None else _local_day_month(int(until_ms))[0]
        if day_hi is not None and day_hi <= day_lo:
            # An empty half-open range is not an error, but it is also not worth
            # a rollup query: the ledger answers it exactly and instantly.
            return None, "empty-window"
        if covered > day_lo:
            return None, "no-coverage"
        whole_from = meta.get(_SESSION_DAILY_META_LEDGER_WHOLE)
        if whole_from and day_lo < whole_from:
            return None, "ledger-bottom-partial"
        return (day_lo, day_hi), ""

    def _session_daily_aggregate(
        self,
        conn: sqlite3.Connection,
        window: tuple[str, str | None],
        session_id: str | None,
    ) -> UsageAggregate:
        """The three grouped reads, over ``session_daily`` instead of ``calls``.

        Deliberately the same shape as ``_ledger_aggregate``: one flat sum, one
        ``GROUP BY provider``, one ``GROUP BY session_id`` — the same columns in
        the same order, so ``_aggregate_from_row`` reads either without knowing.
        The only difference beyond the table is the edge: ``MAX(parent_session_id)``
        over the buckets, which is the same string as ``_PARENT_EDGE_SQL`` over
        the rows (NULL-ignoring MAX over per-bucket NULL-ignoring MAXs), and the
        per-session row set is identical because the grain carries every session
        the ledger holds — including the empty id the real ledger has 224 rows
        for, which ``by_session`` keys today.

        ``ORDER BY`` on both groupings, and the same in ``_ledger_aggregate``:
        the route serialises these maps in dict order, so "the payload is
        byte-identical between the two paths" needs the row order pinned rather
        than left to whichever plan SQLite picks. It costs nothing — both
        groupings are already sorted by a B-tree to compute the GROUP BY.
        """
        day_lo, day_hi = window
        where = ["day >= ?"]
        params: list[Any] = [day_lo]
        if day_hi is not None:
            where.append("day < ?")
            params.append(day_hi)
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        clause = " WHERE " + " AND ".join(where)
        sums = _SESSION_DAILY_READ_COLUMNS_SQL
        top = conn.execute(f"SELECT {sums} FROM session_daily{clause}", params).fetchone()
        per_provider = conn.execute(
            f"SELECT provider, {sums} FROM session_daily{clause} "
            "GROUP BY provider ORDER BY provider",
            params,
        ).fetchall()
        per_session = conn.execute(
            f"SELECT session_id, MAX(parent_session_id), {sums} FROM session_daily{clause} "
            "GROUP BY session_id ORDER BY session_id",
            params,
        ).fetchall()
        names = self._session_names(conn)
        return self._assemble_aggregate(top, per_provider, per_session, names)

    def _assemble_aggregate(
        self,
        top: Iterable[Any] | None,
        per_provider: Sequence[Any],
        per_session: Sequence[Any],
        names: dict[str, str],
    ) -> UsageAggregate:
        """Build the :class:`UsageAggregate` from the three grouped row sets.

        SHARED BY BOTH PATHS ON PURPOSE. The equivalence the gate promises is
        "``dataclasses.asdict`` is identical", and the cheapest way to keep that
        true is for exactly one block of code to turn rows into the result: the
        two paths then differ only in which table the rows came from and in
        whether the parent edge column is ``_PARENT_EDGE_SQL`` or a
        ``MAX(parent_session_id)`` over buckets of it. Two assemblers would be
        two places to drift.

        Row shapes, both satisfying this function: ``top`` is the flat sum row;
        ``per_provider`` rows are ``(provider, <sums>)``; ``per_session`` rows
        are ``(session_id, parent_edge, <sums>)``.
        """
        result = _aggregate_from_row(top)
        result.by_provider = {
            str(row[0]): _aggregate_from_row(row[1:]) for row in per_provider if row[0]
        }
        parents: dict[str, str] = {}
        for row in per_session:
            sid = str(row[0])
            parent = str(row[1] or "")
            agg = _aggregate_from_row(row[2:])
            # Stash the human name (when known) on the id key's aggregate via a
            # side map the caller reads; kept on the object would widen the
            # dataclass for one table, so the report reads names from here.
            result.by_session[sid] = agg
            # ``_PARENT_EDGE_SQL`` already discarded the empty and self edges in
            # SQL, before the MAX. The ``parent != sid`` test is kept as a cheap
            # belt-and-braces for the ``''`` fallback branch above (an old ledger
            # with no parent column), NOT as the self-edge rule — doing it here
            # rather than in SQL is precisely what lost a real edge to a lexical
            # tie-break before review F1.
            if parent and parent != sid:
                parents[sid] = parent
        # Attach names as an attribute the report layer reads without widening
        # the dataclass contract used elsewhere.
        result_session_names: dict[str, str] = {
            sid: names.get(sid, "") for sid in result.by_session
        }
        setattr(result, "session_names", result_session_names)
        # The parent edges the /analytics table re-partitions itself with. Same
        # side-map convention as ``session_names`` and for the same reason: one
        # table's structure is not worth widening a dataclass three other
        # consumers (including the desktop HTTP route) also read.
        #
        # DESKTOP ROUTE, DELIBERATELY FLAT (review F4): because this is a side
        # attribute, ``dataclasses.asdict`` drops it, so ``/v1/desktop/analytics``
        # keeps serving OWN per-session figures while the TUI shows tree totals.
        # That is the intended split, not an oversight — the HTTP route is a raw
        # per-session data feed whose consumers do their own grouping, and the
        # rollup is a presentation choice made by the screen that can also draw
        # the indented children explaining it. Rolling up in the payload would
        # give clients a column that no longer sums to the total they are also
        # served, with nothing on the wire to say why. A client wanting the tree
        # should be given the edges explicitly (a new, versioned field), not a
        # silently re-scoped existing one.
        #
        # WINDOW RULE: ``since_ms``/``until_ms`` filter CALLS, then the rollup
        # runs over whatever survived. A child's calls can fall outside a window
        # containing its parent's; any other rule makes the per-session column
        # stop summing to the headline total, which is the invariant this whole
        # re-partition exists to protect.
        setattr(result, "session_parents", parents)
        return result

    def _has_parent_column(self) -> bool:
        """Whether this DB carries ``parent_session_id`` (absent on old ledgers).

        Reads the migration's own record rather than re-inspecting the schema:
        an absent column must degrade to "no edges, every session is a root",
        which renders exactly today's flat table.
        """
        return "parent_session_id" in self._present_optional

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
                # ``NULL``, NOT the usual ``"0"`` default. ``col``'s zero
                # fallback would report every request on a column-less ledger as
                # FAILED, which is precisely the false alarm this field was added
                # to remove. ``NULL`` yields ``ok=None`` = unknown, and the
                # renderer draws unknown clean.
                col("ok", "NULL"),
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
                    ok=None if row[12] is None else bool(row[12]),
                )
                for row in conn.execute(
                    "SELECT " + ", ".join(fields) + scope + " ORDER BY ts_ms DESC, id DESC LIMIT ?",
                    (*params, max(0, min(int(recent_limit), 50))),
                )
            )
            descendants, descendant_ids = self._descendant_usage(
                conn, session_id, columns, measures
            )
            tool_calls = self._tool_call_stats(conn, session_id)
            return SessionReport(
                session_id=session_id,
                aggregate=aggregate,
                descendants_aggregate=descendants,
                descendant_ids=descendant_ids,
                by_model=by_model,
                by_purpose=by_purpose,
                by_purpose_outcome=groups,
                missing_usage_calls=int(missing or 0),
                unknown_usage_calls=int(unknown or 0),
                timings=timings,
                recent=recent,
                first_ts_ms=first,
                last_ts_ms=last,
                tool_calls=tool_calls,
            )
        except Exception:  # noqa: BLE001 — diagnostics must not interrupt a turn
            logger.debug("analytics: session report unavailable", exc_info=True)
            return SessionReport(session_id=session_id, available=False)
        finally:
            if conn is not None:
                conn.close()

    #: How deep a session tree is walked before the walk stops. The ledger's
    #: tree is one level today (33 parents, 467 children, zero sessions that are
    #: both), but depth is a property of USAGE, not of schema: a subagent that
    #: launches its own subagent is stamped with the middle session as parent by
    #: ``SessionStreamFn.fork``, and ``AsyncJobManager`` already carries the
    #: ``child_jobs``/``descendant_usage`` machinery for it. So the walk is
    #: recursive, and this is the backstop that keeps a malformed or cyclic
    #: ledger from turning a report into a hang. 32 is far past any real nesting.
    #:
    #: ANCHOR (review F2): this cap counts levels from the QUERIED session, while
    #: ``model.MAX_SESSION_TREE_DEPTH`` counts them from the forest ROOT. The two
    #: therefore truncate different chains on a tree deeper than the cap, and a
    #: mid-tree row can legitimately read differently on the two screens there.
    #: This is not reconcilable without one of the surfaces walking a tree it has
    #: no reason to build (``/session`` does not know its own root; ``/analytics``
    #: does not know which session you are asking about), and it is unreachable
    #: on real data — the ledger's maximum observed depth is 1, and zero sessions
    #: are both a parent and a child. Recorded rather than fixed, so the next
    #: reader does not mistake it for a rollup bug.
    _MAX_TREE_DEPTH = 32

    def _canonical_parents(
        self, conn: sqlite3.Connection, session_ids: Sequence[str]
    ) -> dict[str, str]:
        """The canonical parent of each given session, by the ONE shared rule.

        Resolves ``_PARENT_EDGE_SQL`` — the same expression ``aggregate`` puts in
        its per-session GROUP BY — for a bounded set of ids, so the ``/session``
        subtree walk and the ``/analytics`` table answer "who is this session's
        parent" identically. Before review F1 the walk filtered per ROW ("any row
        claims this edge") while the aggregate took a lexical ``MAX``, and the
        two disagreed on any session carrying more than one distinct parent
        value.

        Chunked at 500 ids: SQLite's default host-parameter limit is 999, and a
        subtree level can be arbitrarily wide (the widest real fan-out is 46).
        """
        parents: dict[str, str] = {}
        ids = [sid for sid in session_ids if sid]
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT session_id, {_PARENT_EDGE_SQL} FROM calls "
                f"WHERE session_id IN ({placeholders}) GROUP BY session_id",
                chunk,
            ).fetchall()
            for sid, parent in rows:
                if parent:
                    parents[str(sid)] = str(parent)
        return parents

    @staticmethod
    def _tool_call_stats(conn: sqlite3.Connection, session_id: str) -> ToolCallStats | None:
        """Tool-call outcomes for one session, or ``None`` meaning UNKNOWN.

        ``None`` is returned when the ``tool_calls`` table does not exist (a
        ledger written before this feature) **or** when it holds no rows for this
        session. The second case is the important one: every session recorded
        before this shipped has requests but no tool rows, and rendering that as
        ``0 calls / 0% invalid`` would be a fabricated measurement on the exact
        screen whose standing invariant is that an absent measurement is never
        drawn as a measured zero.

        The cost of that rule is a session which genuinely made zero tool calls,
        which is INDISTINGUISHABLE from an unrecorded one in a bare ``COUNT(*)``
        and therefore also reads ``unknown``. That is the safe error in both
        directions — a wrong ``unknown`` withholds a fact, a wrong ``0%`` states
        one — and it self-corrects the moment the session makes a tool call.

        **``origin`` is in the GROUP BY because the read side must honour the
        partition the schema records.** ``ToolCallStats``' counts are
        model-origin only; nested (``eval``-bridge) rows are tallied apart and
        never reach a rate. Dropping ``origin`` from this projection is not a
        cosmetic simplification — it silently re-pools the two populations and
        lets one scripted loop set the headline accuracy figure. See the origin
        partition on ``ToolCallStats`` and the schema comment on ``tool_calls``.
        """
        try:
            rows = list(
                conn.execute(
                    "SELECT origin, fault, tool_name, COUNT(*) FROM tool_calls "
                    "WHERE session_id = ? GROUP BY origin, fault, tool_name",
                    (session_id,),
                )
            )
        except sqlite3.Error:
            # No such table: an older ledger. Unknown, not zero, and not an
            # error — diagnostics must open against any ledger version.
            logger.debug("analytics: tool_calls unavailable", exc_info=True)
            return None
        if not rows:
            return None
        total = 0
        ok = 0
        faults: dict[str, int] = {}
        faults_by_tool: dict[str, int] = {}
        nested_total = 0
        nested_ok = 0
        nested_excluded = 0
        for origin, fault, tool_name, count in rows:
            count = int(count)
            if str(origin) != ORIGIN_MODEL:
                # Anything that is not model-emitted is nested by definition, and
                # an UNRECOGNISED origin lands here too rather than in the rates:
                # if a future writer adds a third origin, the safe default is to
                # keep it out of the benchmarking number until someone decides
                # where it belongs.
                nested_total += count
                if not fault:
                    nested_ok += count
                elif fault in EXCLUDED_FAULTS:
                    nested_excluded += count
                continue
            total += count
            if not fault:
                ok += count
                continue
            faults[str(fault)] = faults.get(str(fault), 0) + count
            name = str(tool_name)
            faults_by_tool[name] = faults_by_tool.get(name, 0) + count
        return ToolCallStats(
            total=total,
            ok=ok,
            faults=faults,
            faults_by_tool=faults_by_tool,
            nested_total=nested_total,
            nested_ok=nested_ok,
            nested_excluded=nested_excluded,
        )

    def _descendant_usage(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        columns: set[str],
        measures: str,
    ) -> tuple[UsageAggregate | None, tuple[str, ...]]:
        """Sum every DESCENDANT session's usage, own scope excluded.

        Returns ``(None, ())`` when the walk cannot run — an older ledger with
        no ``parent_session_id`` column — which the report renders as "unknown"
        rather than as "$0.00 of subagent spend". Runs on the caller's pinned
        read transaction so the subtree and the own scope describe the same WAL
        snapshot.

        A level-by-level descent on the caller's pinned transaction, NOT the
        recursive CTE this originally used. The CTE could only apply the shared
        edge rule as a correlated scalar subquery re-evaluated per candidate row,
        which measured 620 ms on the widest real fan-out (46 children) against
        6.7 ms for this descent — and expressing the rule as an ``edges``
        CTE instead costs a full 196 ms scan of ``calls``. Each level is two
        indexed statements on the SAME pinned transaction, so the single-snapshot
        property the CTE was chosen for is preserved. Measured end to end on the
        475k-row ledger: 0.87 ms for the reported session, 6.7 ms worst case.
        Without ``idx_calls_parent`` each level is a full scan — see
        ``_OPTIONAL_INDEXES``.

        Three guards, because the ledger contains a real cycle edge (224 rows
        carry ``parent_session_id == session_id``, all from a degenerate empty
        id):

        1. A candidate is kept only when its CANONICAL parent (the one shared
           rule, ``_canonical_parents``) is the node currently being expanded.
           This is what makes the walk agree with ``/analytics`` by construction
           rather than by coincidence: a session whose rows merely mention this
           node, but whose canonical parent is some other session, is not a child
           here — and the aggregate would not have drawn that edge
           either (review F1). It also subsumes the old self-edge predicate,
           since the rule discards a self parent in SQL, before the MAX.
        2. ``visited`` means a cycle that re-reaches an already-counted session
           stops there, so no session contributes its calls twice.
        3. ``depth < _MAX_TREE_DEPTH`` bounds the walk whatever the data does.
           See that constant for why its anchor differs from the forest's.

        Note on retention: ``prune`` deletes by ``ts_ms``, so a root whose
        children aged out first reports a smaller subtree than it really spent.
        Parent and child age out together in practice; a shrinking figure here
        is retention, not a rollup bug.
        """
        if "parent_session_id" not in columns or not session_id:
            return None, ()
        try:
            found: list[str] = []
            visited: set[str] = {session_id}
            frontier: set[str] = {session_id}
            depth = 0
            while frontier and depth < self._MAX_TREE_DEPTH:
                placeholders = ", ".join("?" for _ in frontier)
                # Every session carrying a row that POINTS AT this level: an
                # index seek on idx_calls_parent. Which of those mentions is
                # actually an edge is decided by the canonical rule below.
                candidates = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT DISTINCT session_id FROM calls "
                        f"WHERE parent_session_id IN ({placeholders})",
                        sorted(frontier),
                    )
                    if row[0]
                } - visited
                if not candidates:
                    break
                edges = self._canonical_parents(conn, sorted(candidates))
                nxt = {sid for sid in candidates if edges.get(sid) in frontier}
                if not nxt:
                    break
                visited |= nxt
                found.extend(sorted(nxt))
                frontier = nxt
                depth += 1
        except Exception:  # noqa: BLE001 — a failed walk is unknown, not zero
            logger.debug("analytics: descendant walk failed", exc_info=True)
            return None, ()
        ids = tuple(found)
        if not ids:
            return UsageAggregate(), ()
        # Bind the ids rather than interpolating: they are ledger-sourced, and a
        # bounded IN list keeps this one statement on the same snapshot.
        placeholders = ", ".join("?" for _ in ids)
        try:
            row = conn.execute(
                f"SELECT {measures} FROM calls WHERE session_id IN ({placeholders})", ids
            ).fetchone()
        except Exception:  # noqa: BLE001
            logger.debug("analytics: descendant aggregate failed", exc_info=True)
            return None, ()
        return _aggregate_from_row(row), ids

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
