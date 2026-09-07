"""The SQLite ledger: recording, aggregation scoping, retention, and the
estimated component split that is computed at write time.

The store is the accuracy contract: what a report shows is what was recorded,
so these tests write known snapshots and read the aggregate back, including the
per-provider and per-session breakdowns the ``/analytics`` screen renders.
"""

from __future__ import annotations

import time

from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.store import AnalyticsStore


def _snap(
    *,
    session_id="s1",
    provider="anthropic",
    model_id="claude",
    input_tokens=100,
    output_tokens=40,
    cache_read=800,
    cache_write=20,
    reasoning=10,
    context=920,
    chars=None,
    ok=True,
    ts_ms=None,
    cost_micro=1000,
    cost_known=True,
):
    return CallSnapshot(
        ts_ms=ts_ms if ts_ms is not None else int(time.time() * 1000),
        session_id=session_id,
        provider=provider,
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        context_tokens=context,
        component_chars=chars or {"conversation": 500, "system_prompt": 420},
        ok=ok,
        cost_micro=cost_micro,
        cost_known=cost_known,
        # Pre-priced so the store records these exact figures rather than
        # re-pricing the fixture's fake model (which has no price table entry).
        priced=True,
    )


def test_record_and_aggregate_roundtrip(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    assert store.record_batch([_snap(), _snap()]) == 2
    agg = store.aggregate()
    assert agg.calls == 2
    assert agg.ok_calls == 2
    assert agg.input_tokens == 200
    assert agg.output_tokens == 80
    assert agg.cache_read_tokens == 1600
    assert agg.reasoning_tokens == 20
    assert agg.context_tokens == 1840
    # Component split was apportioned against context at write time and sums
    # back to the context total.
    assert sum(agg.components.values()) == agg.context_tokens
    # Images is a first-class component in the roundtrip: image chars apportion
    # to it and read back non-zero.
    store2 = AnalyticsStore(tmp_path / "img.db")
    assert (
        store2.record_batch([_snap(context=1000, chars={"conversation": 500, "images": 500})]) == 1
    )
    assert store2.aggregate().components["images"] > 0
    store2.close()
    store.close()


def test_component_split_is_stored_estimate(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    # 900 chars conversation, 100 chars system prompt, 1000 context tokens ->
    # 900/100 split of the authoritative total.
    store.record_batch([_snap(context=1000, chars={"conversation": 900, "system_prompt": 100})])
    agg = store.aggregate()
    assert agg.components["conversation"] == 900
    assert agg.components["system_prompt"] == 100
    store.close()


def test_zero_context_stores_zero_components(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(context=0, chars={"conversation": 500})])
    agg = store.aggregate()
    assert agg.calls == 1
    assert sum(agg.components.values()) == 0
    store.close()


def test_by_provider_and_by_session(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch(
        [
            _snap(session_id="s1", provider="anthropic", input_tokens=100),
            _snap(session_id="s2", provider="openai", input_tokens=50),
            _snap(session_id="s2", provider="anthropic", input_tokens=25),
        ]
    )
    agg = store.aggregate()
    assert set(agg.by_provider) == {"anthropic", "openai"}
    assert agg.by_provider["anthropic"].input_tokens == 125
    assert agg.by_provider["openai"].input_tokens == 50
    assert set(agg.by_session) == {"s1", "s2"}
    assert agg.by_session["s2"].calls == 2
    store.close()


def test_cost_roundtrip_and_grouping(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch(
        [
            _snap(session_id="s1", provider="anthropic", cost_micro=6_000_000, cost_known=True),
            _snap(session_id="s1", provider="openai", cost_micro=1_000_000, cost_known=True),
            # A local model with no price: recorded, but cost_known=False.
            _snap(session_id="s2", provider="ollama", cost_micro=0, cost_known=False),
        ]
    )
    agg = store.aggregate()
    assert agg.cost_micro == 7_000_000
    assert agg.cost_usd == 7.0
    assert agg.cost_known_calls == 2
    assert agg.cost_is_partial is True  # the ollama call had no price
    assert agg.by_provider["anthropic"].cost_usd == 6.0
    assert agg.by_provider["openai"].cost_usd == 1.0
    assert agg.by_provider["ollama"].cost_is_known is False
    store.close()


def test_migration_adds_cost_columns_to_old_db(tmp_path):
    # A database written by the token-only release has no cost_* columns.
    # Opening it with the current store must ALTER them in and read old rows as
    # cost 0 / unknown, never raise on a missing column.
    import sqlite3

    from local_operator.analytics.store import _COMPONENT_COLUMNS

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(f"""
        CREATE TABLE calls (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, session_id TEXT,
          provider TEXT, model_id TEXT, ok INTEGER, input_tokens INTEGER,
          output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
          reasoning_tokens INTEGER, context_tokens INTEGER, {_COMPONENT_COLUMNS}
        );
        CREATE TABLE session_names (
          session_id TEXT PRIMARY KEY, name TEXT, updated_at_ms INTEGER
        );
        """)
    conn.execute(
        "INSERT INTO calls (ts_ms, session_id, provider, model_id, ok, input_tokens, "
        "output_tokens, context_tokens) VALUES (1, 's', 'anthropic', 'm', 1, 100, 20, 120)"
    )
    conn.commit()
    conn.close()

    store = AnalyticsStore(db)
    agg = store.aggregate()
    assert agg.calls == 1
    assert agg.cost_micro == 0
    assert agg.cost_known_calls == 0
    assert agg.cost_is_known is False  # old row reads as unpriced, not $0
    # A new priced call can be recorded into the migrated DB.
    assert store.record_batch([_snap(cost_micro=500, cost_known=True)]) == 1
    assert store.aggregate().cost_micro == 500
    store.close()


#: The eight component columns a pre-``images`` DB had, so a migration test can
#: build a genuinely old schema (``_COMPONENT_COLUMNS`` now includes c_images and
#: would not reproduce the missing-column case). Order matches the original
#: COMPONENT_KEYS before ``images`` was appended.
_PRE_IMAGES_COMPONENT_COLUMNS = ",\n  ".join(
    f"c_{key} INTEGER NOT NULL DEFAULT 0"
    for key in (
        "system_prompt",
        "custom_instructions",
        "tool_inventory",
        "tool_schemas",
        "environment",
        "knowledge",
        "conversation",
        "tool_results",
    )
)


def test_migration_adds_images_column_to_old_db(tmp_path):
    # A DB written before the images component existed has c_* columns for the
    # original eight only. Opening it must ALTER c_images in, read old rows as
    # images 0, and record new calls WITH an images estimate — never raise on a
    # missing column.
    import sqlite3

    db = tmp_path / "preimages.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(f"""
        CREATE TABLE calls (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, session_id TEXT,
          provider TEXT, model_id TEXT, ok INTEGER, input_tokens INTEGER,
          output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
          reasoning_tokens INTEGER, context_tokens INTEGER,
          cost_micro INTEGER NOT NULL DEFAULT 0, cost_known INTEGER NOT NULL DEFAULT 0,
          {_PRE_IMAGES_COMPONENT_COLUMNS}
        );
        CREATE TABLE session_names (
          session_id TEXT PRIMARY KEY, name TEXT, updated_at_ms INTEGER
        );
        """)
    # An old row: its image tokens (if any) stayed baked into the text-bucket
    # estimates it was recorded with; it reads images 0 after migration.
    conn.execute(
        "INSERT INTO calls (ts_ms, session_id, provider, model_id, ok, input_tokens, "
        "output_tokens, context_tokens, c_conversation) "
        "VALUES (1, 's', 'anthropic', 'm', 1, 100, 20, 120, 120)"
    )
    conn.commit()
    conn.close()

    store = AnalyticsStore(db)
    agg = store.aggregate()
    assert agg.calls == 1
    assert agg.components["images"] == 0  # old row forward-fills to images 0
    # A new call carrying image chars records a non-zero images estimate into the
    # migrated column.
    assert (
        store.record_batch([_snap(context=1000, chars={"conversation": 500, "images": 500})]) == 1
    )
    assert store.aggregate().components["images"] > 0
    store.close()


def test_recording_degrades_when_component_column_absent(tmp_path):
    # Option A: a missing OPTIONAL component column (here c_images) must degrade
    # the same way a missing cost column does — the column is dropped from the
    # insert and read as 0 in the aggregate, never failing the write. Simulated
    # by pretending the images migration failed while the rest is present.
    import sqlite3

    db = tmp_path / "noimages.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(f"""
        CREATE TABLE calls (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, session_id TEXT,
          provider TEXT, model_id TEXT, ok INTEGER, input_tokens INTEGER,
          output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
          reasoning_tokens INTEGER, context_tokens INTEGER,
          cost_micro INTEGER NOT NULL DEFAULT 0, cost_known INTEGER NOT NULL DEFAULT 0,
          {_PRE_IMAGES_COMPONENT_COLUMNS}
        );
        CREATE TABLE session_names (
          session_id TEXT PRIMARY KEY, name TEXT, updated_at_ms INTEGER
        );
        """)
    conn.commit()
    conn.close()

    store = AnalyticsStore(db)
    store._connect()  # runs the migration, which would normally add c_images
    # Drop the column the migration just added so the insert actually hits a
    # table that lacks it. Flipping ``_present_optional`` alone left the
    # column in place, so a broken insert plan that still named ``c_images``
    # would still return record_batch == 1.
    conn = store._connect()
    assert conn is not None
    conn.execute("ALTER TABLE calls DROP COLUMN c_images")
    conn.commit()
    from local_operator.analytics.store import _OPTIONAL_COLUMN_NAMES

    store._present_optional = _OPTIONAL_COLUMN_NAMES - {"c_images"}
    store._rebuild_insert_plan()
    # A call with image chars still records — images just degrade to 0.
    assert (
        store.record_batch([_snap(context=1000, chars={"conversation": 500, "images": 500})]) == 1
    )
    agg = store.aggregate()
    assert agg.calls == 1
    assert agg.context_tokens == 1000  # token analytics survive
    assert agg.components["images"] == 0  # images degraded to 0, not a crash
    # Cost is unaffected: only the images column was dropped.
    assert agg.cost_is_known is True
    store.close()


def test_recording_degrades_when_cost_columns_absent(tmp_path):
    # C2: if the cost columns cannot be added (simulated by forcing _has_cost
    # False against a table that lacks them), recording must NOT fail every
    # write — it drops cost and keeps token analytics; the report shows $—.
    import sqlite3

    from local_operator.analytics.store import _COMPONENT_COLUMNS

    db = tmp_path / "nocost.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(f"""
        CREATE TABLE calls (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, session_id TEXT,
          provider TEXT, model_id TEXT, ok INTEGER, input_tokens INTEGER,
          output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
          reasoning_tokens INTEGER, context_tokens INTEGER, {_COMPONENT_COLUMNS}
        );
        CREATE TABLE session_names (
          session_id TEXT PRIMARY KEY, name TEXT, updated_at_ms INTEGER
        );
        """)
    conn.commit()
    conn.close()

    store = AnalyticsStore(db)
    store._connect()  # runs the migration, which will add the columns
    # Drop the cost columns the migration just added so the insert actually
    # hits a table that lacks them. Flipping ``_has_cost`` / ``_present_optional``
    # alone left the columns in place, so a broken insert plan that still
    # named them would still return record_batch == 1.
    conn = store._connect()
    assert conn is not None
    conn.execute("ALTER TABLE calls DROP COLUMN cost_micro")
    conn.execute("ALTER TABLE calls DROP COLUMN cost_known")
    conn.commit()
    from local_operator.analytics.store import _OPTIONAL_COLUMN_NAMES

    store._has_cost = False
    store._present_optional = _OPTIONAL_COLUMN_NAMES - {"cost_micro", "cost_known"}
    store._rebuild_insert_plan()
    assert store.record_batch([_snap(input_tokens=100, cost_micro=999, cost_known=True)]) == 1
    agg = store.aggregate()
    assert agg.calls == 1
    assert agg.input_tokens == 100  # token analytics survive
    assert agg.cost_micro == 0  # cost degraded to $— rather than failing
    assert agg.cost_is_known is False
    store.close()


def test_failed_calls_counted_separately(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(ok=True), _snap(ok=False), _snap(ok=False)])
    agg = store.aggregate()
    assert agg.calls == 3
    assert agg.ok_calls == 1
    store.close()


def test_time_window_scoping(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    now = int(time.time() * 1000)
    old = now - 10 * 24 * 60 * 60 * 1000
    store.record_batch([_snap(ts_ms=old), _snap(ts_ms=now)])
    recent = store.aggregate(since_ms=now - 1000)
    assert recent.calls == 1
    all_time = store.aggregate()
    assert all_time.calls == 2
    store.close()


def test_session_scoping(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(session_id="s1"), _snap(session_id="s2")])
    only = store.aggregate(session_id="s1")
    assert only.calls == 1
    store.close()


def test_prune_removes_old_rows(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db", retention_days=7)
    now = int(time.time() * 1000)
    old = now - 30 * 24 * 60 * 60 * 1000
    store.record_batch([_snap(ts_ms=old), _snap(ts_ms=now)])
    removed = store.prune(now_ms=now)
    assert removed == 1
    assert store.aggregate().calls == 1
    store.close()


def test_session_names_surface_in_aggregate(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(session_id="abc")])
    store.upsert_session_name("abc", "my conversation")
    agg = store.aggregate()
    names = getattr(agg, "session_names", {})
    assert names.get("abc") == "my conversation"
    store.close()


def test_empty_store_returns_zeroed_aggregate(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    agg = store.aggregate()
    assert agg.calls == 0
    assert agg.total_tokens == 0
    assert agg.by_provider == {}
    assert agg.by_session == {}
    store.close()


def test_broken_store_degrades_to_noop(tmp_path):
    # A path that cannot be a database (a directory) makes the store a no-op
    # rather than raising — analytics is an accelerator, never a dependency.
    bad = tmp_path / "dir.db"
    bad.mkdir()
    store = AnalyticsStore(bad)
    assert store.record_batch([_snap()]) == 0
    assert store.aggregate().calls == 0
    store.close()


# ---------------------------------------------------------------------------
# Calendar rollups (usage_daily / usage_monthly) — the historical time series.
#
# These share the ONE ``record_batch`` write path with the raw ledger (same
# transaction), so their whole contract is: the same call that lands in
# ``calls`` accumulates into the day and month buckets keyed by its LOCAL
# calendar date, losslessly under concurrent writers, pruned on its own horizon.
# ---------------------------------------------------------------------------

from datetime import datetime  # noqa: E402 — grouped with the rollup tests it serves


def _ts(year, month, day, hour=12):
    """Epoch-ms for a LOCAL wall-clock moment (the bucketing is local, §2.6)."""
    return int(datetime(year, month, day, hour, 0, 0).timestamp() * 1000)


def test_rollup_upsert_accumulates_same_bucket(tmp_path):
    # Two calls on the same local day + model land in ONE daily row, summed —
    # the ON CONFLICT accumulate, not two rows.
    store = AnalyticsStore(tmp_path / "a.db")
    day = _ts(2026, 8, 21)
    store.record_batch([_snap(ts_ms=day, input_tokens=100), _snap(ts_ms=day, input_tokens=50)])
    series = store.daily_series(30)
    assert len(series) == 1
    row = series[0]
    assert row.period == "2026-08-21"
    assert row.calls == 2
    assert row.input_tokens == 150
    # Cost accumulates too (both priced fixtures at 1000 micro-USD).
    assert row.cost_micro == 2000
    assert row.cost_known_calls == 2
    assert not row.cost_is_floor
    store.close()


def test_rollup_buckets_by_local_day_and_month(tmp_path):
    # Calls on three distinct days across a month boundary produce three daily
    # rows and two monthly rows, each keyed by the local calendar bucket.
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch(
        [
            _snap(ts_ms=_ts(2026, 7, 31)),
            _snap(ts_ms=_ts(2026, 8, 1)),
            _snap(ts_ms=_ts(2026, 8, 1, hour=23)),
        ]
    )
    daily = store.daily_series(30)
    assert [r.period for r in daily] == ["2026-07-31", "2026-08-01"]
    # Aug 1 folded both of its calls together.
    assert daily[1].calls == 2
    monthly = store.monthly_series(12)
    assert [r.period for r in monthly] == ["2026-07", "2026-08"]
    assert monthly[0].calls == 1
    assert monthly[1].calls == 2
    store.close()


def test_rollup_year_boundary(tmp_path):
    # Dec 31 and Jan 1 are different days, months, AND years — each its own
    # bucket, sorted lexically so the year rollover reads in order.
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(ts_ms=_ts(2025, 12, 31)), _snap(ts_ms=_ts(2026, 1, 1))])
    assert [r.period for r in store.daily_series(30)] == ["2025-12-31", "2026-01-01"]
    assert [r.period for r in store.monthly_series(12)] == ["2025-12", "2026-01"]
    store.close()


def test_daily_series_by_model_splits_and_aggregates(tmp_path):
    # The same day with two models: by_model=True gives a row per model;
    # by_model=False sums them into one bucket row keyed model="".
    store = AnalyticsStore(tmp_path / "a.db")
    day = _ts(2026, 8, 21)
    store.record_batch(
        [
            _snap(ts_ms=day, provider="anthropic", model_id="claude", input_tokens=100),
            _snap(ts_ms=day, provider="anthropic", model_id="haiku", input_tokens=40),
        ]
    )
    per_model = store.daily_series(30, by_model=True)
    assert {r.model for r in per_model} == {"anthropic/claude", "anthropic/haiku"}
    aggregated = store.daily_series(30, by_model=False)
    assert len(aggregated) == 1
    assert aggregated[0].model == ""
    assert aggregated[0].input_tokens == 140
    store.close()


def test_daily_series_window_keeps_newest_n_days(tmp_path):
    # daily_series(days=N) returns the N NEWEST distinct days, oldest-first,
    # regardless of a gap of idle days between them.
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(ts_ms=_ts(2026, 8, d)) for d in (1, 5, 10, 20, 25)])
    window = store.daily_series(3)
    assert [r.period for r in window] == ["2026-08-10", "2026-08-20", "2026-08-25"]
    store.close()


def test_unpriced_call_makes_bucket_a_floor(tmp_path):
    # A day mixing a priced and an unpriced call: cost accumulates only the
    # priced one, and cost_known_calls < calls flags the bucket as a lower
    # bound (the ≥ floor the chart renders).
    store = AnalyticsStore(tmp_path / "a.db")
    day = _ts(2026, 8, 21)
    store.record_batch(
        [
            _snap(ts_ms=day, cost_micro=1500, cost_known=True),
            _snap(ts_ms=day, cost_micro=0, cost_known=False),
        ]
    )
    row = store.daily_series(30)[0]
    assert row.calls == 2
    assert row.cost_micro == 1500
    assert row.cost_known_calls == 1
    assert row.cost_is_floor
    store.close()


def test_rollup_daily_prune_keeps_365_distinct_days(tmp_path):
    # 400 distinct days recorded; prune keeps exactly the newest 365 in the
    # daily rollup. The raw-ledger prune is independent and unchanged.
    store = AnalyticsStore(tmp_path / "a.db")
    day0 = datetime(2025, 1, 1, 12, 0, 0)
    snaps = [_snap(ts_ms=int((day0.replace() + _days(i)).timestamp() * 1000)) for i in range(400)]
    store.record_batch(snaps)
    # Before: all 400 days present.
    assert len(store.daily_series(1000)) == 400
    store.prune(now_ms=int((day0 + _days(399)).timestamp() * 1000))
    kept = store.daily_series(1000)
    assert len(kept) == 365
    # The newest day survives; the 366th-oldest is gone.
    assert kept[-1].period == (day0 + _days(399)).strftime("%Y-%m-%d")
    assert kept[0].period == (day0 + _days(35)).strftime("%Y-%m-%d")
    store.close()


def test_rollup_monthly_prune_caps_at_120_months(tmp_path):
    # 130 distinct months; the monthly cap keeps the newest 120.
    store = AnalyticsStore(tmp_path / "a.db")
    snaps = []
    for i in range(130):
        year = 2016 + i // 12
        month = i % 12 + 1
        snaps.append(_snap(ts_ms=_ts(year, month, 15)))
    store.record_batch(snaps)
    assert len(store.monthly_series(1000)) == 130
    store.prune(now_ms=_ts(2026, 12, 31))
    assert len(store.monthly_series(1000)) == 120
    store.close()


def test_rollup_concurrent_stores_accumulate_losslessly(tmp_path):
    # Two stores on the SAME file (the multi-lop reality), interleaved
    # record_batch into the same day/model bucket, must sum losslessly — this
    # is the WAL + ON CONFLICT accumulate the whole design rests on.
    path = tmp_path / "a.db"
    a = AnalyticsStore(path)
    b = AnalyticsStore(path)
    day = _ts(2026, 8, 21)
    for _ in range(10):
        a.record_batch([_snap(ts_ms=day, input_tokens=10)])
        b.record_batch([_snap(ts_ms=day, input_tokens=10)])
    # A fresh reader sees the merged total: 20 calls, 200 input tokens, one row.
    reader = AnalyticsStore(path)
    series = reader.daily_series(30)
    assert len(series) == 1
    assert series[0].calls == 20
    assert series[0].input_tokens == 200
    a.close()
    b.close()
    reader.close()


def test_rollup_reads_empty_on_broken_store(tmp_path):
    # A degraded store returns empty series and zeroed totals, never raises.
    bad = tmp_path / "dir.db"
    bad.mkdir()
    store = AnalyticsStore(bad)
    assert store.daily_series(30) == []
    assert store.monthly_series(12) == []
    totals = store.series_totals()
    assert totals.calls == 0
    assert totals.total_tokens == 0
    store.close()


def test_series_totals_sums_the_daily_window(tmp_path):
    # series_totals sums the same by-model-false daily window the chart draws,
    # so the header figure and the bars describe one span.
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch(
        [
            _snap(ts_ms=_ts(2026, 8, 20), input_tokens=100, cost_micro=1000, cost_known=True),
            _snap(ts_ms=_ts(2026, 8, 21), input_tokens=50, cost_micro=0, cost_known=False),
        ]
    )
    totals = store.series_totals()
    assert totals.calls == 2
    assert totals.input_tokens == 150
    assert totals.cost_micro == 1000
    assert totals.cost_is_floor  # one call was unpriced
    store.close()


def _days(n):
    """A ``timedelta`` of ``n`` days, kept local to the retention tests."""
    from datetime import timedelta

    return timedelta(days=n)


# ---------------------------------------------------------------------------
# Opening the database under cross-process contention (#391).
#
# The failure these guard is DATA LOSS, not an exception: a momentary lock on
# the very first open latched ``_broken``, and every write for the rest of that
# process then returned 0 silently. With four parallel writers that surfaces as
# exactly one of them contributing zero rows.
#
# Both tests inject the failure rather than racing for it, so they are
# deterministic; the real contended measurement is in the PR's evidence.
# ---------------------------------------------------------------------------


def test_a_locked_first_open_does_not_disable_the_store(tmp_path, monkeypatch):
    """A BUSY at open is transient: the next write must still land.

    This is the #391 defect itself. ``_connect`` caught every exception and set
    ``_broken``, which is correct for a read-only directory and catastrophic for
    a lock another process releases microseconds later.
    """
    import sqlite3

    from local_operator.analytics import store as store_mod

    store = AnalyticsStore(tmp_path / "a.db")
    real_connect = sqlite3.connect
    calls = {"n": 0}

    def flaky_connect(*args, **kwargs):
        # Fail ONLY the first open, the way a real lock race does.
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(store_mod.sqlite3, "connect", flaky_connect)

    # The open is retried, so a single lock does not drop the batch — and
    # ``_broken`` stays clear so a later write would also succeed.
    assert store.record_batch([_snap()]) == 1
    assert not store._broken

    monkeypatch.undo()
    assert store.record_batch([_snap(), _snap()]) == 2
    assert store.aggregate().calls == 3
    store.close()


def test_a_real_open_fault_still_disables_the_store(tmp_path, monkeypatch):
    """The complement: a NON-lock fault must still latch ``_broken``.

    Without this, the #391 fix would turn one log line into one per provider
    round trip for the life of the process against a genuinely unusable path.
    """
    import sqlite3

    from local_operator.analytics import store as store_mod

    store = AnalyticsStore(tmp_path / "a.db")

    def broken_connect(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(store_mod.sqlite3, "connect", broken_connect)
    assert store.record_batch([_snap()]) == 0
    assert store._broken
    store.close()


def test_wal_is_enabled_despite_a_busy_journal_transition(tmp_path, monkeypatch):
    """The DELETE->WAL switch is retried, because busy_timeout does not cover it.

    Changing the journal mode needs an exclusive lock, and SQLite fails that
    acquisition with SQLITE_BUSY immediately instead of calling the busy
    handler — so the 5s ``busy_timeout`` set beside it buys nothing. Measured on
    a fresh database opened by 16 processes at once: 25/320 opens raised here.
    """
    import sqlite3

    from local_operator.analytics import store as store_mod

    store = AnalyticsStore(tmp_path / "a.db")
    real_connect = sqlite3.connect
    busied = {"n": 0}

    class _FlakyConn:
        """Wraps a real connection because ``sqlite3.Connection`` is immutable
        (its ``execute`` cannot be monkeypatched), so the injection has to
        happen at the ``connect`` boundary instead."""

        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            # Fail the journal switch twice, which the bounded retry must absorb.
            if "journal_mode=WAL" in sql and busied["n"] < 2:
                busied["n"] += 1
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(
        store_mod.sqlite3, "connect", lambda *a, **kw: _FlakyConn(real_connect(*a, **kw))
    )
    assert store.record_batch([_snap()]) == 1
    monkeypatch.undo()
    store.close()

    # The REAL property first: the database ended up in WAL despite the busy
    # transition. Asserting the injection count before this would let the test
    # die on its own proxy rather than on the behaviour it guards.
    conn = sqlite3.connect(str(tmp_path / "a.db"))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()
    assert busied["n"] == 2  # and the retry was genuinely exercised, not skipped


def test_cache_write_1h_tokens_is_recorded_and_migrated(tmp_path):
    """The 1h slice of a cache write lands in its own ledger column, and a
    database from before the column existed gains it on open (old rows read
    0, which is the truth: every pre-1h write was a 5m write)."""
    import sqlite3

    db = tmp_path / "pre1h.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(f"""
        CREATE TABLE calls (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, session_id TEXT,
          provider TEXT, model_id TEXT, ok INTEGER, input_tokens INTEGER,
          output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
          reasoning_tokens INTEGER, context_tokens INTEGER,
          cost_micro INTEGER NOT NULL DEFAULT 0, cost_known INTEGER NOT NULL DEFAULT 0,
          {_PRE_IMAGES_COMPONENT_COLUMNS}, c_images INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE session_names (
          session_id TEXT PRIMARY KEY, name TEXT, updated_at_ms INTEGER
        );
        """)
    conn.execute(
        "INSERT INTO calls (ts_ms, session_id, provider, model_id, ok, input_tokens, "
        "output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, "
        "context_tokens) VALUES (1, 'old', 'anthropic', 'claude', 1, 1, 1, 0, 500, 0, 501)"
    )
    conn.commit()
    conn.close()

    store = AnalyticsStore(db)
    import dataclasses

    snap = dataclasses.replace(_snap(cache_write=300), cache_write_1h_tokens=120)
    assert store.record_batch([snap]) == 1
    conn = store._connect()
    assert conn is not None
    rows = conn.execute(
        "SELECT session_id, cache_write_tokens, cache_write_1h_tokens FROM calls ORDER BY id"
    ).fetchall()
    assert rows == [("old", 500, 0), ("s1", 300, 120)]
    store.close()


def _child(parent, session_id, **kw):
    """A snapshot stamped as a subagent call of ``parent``."""
    import dataclasses

    return dataclasses.replace(_snap(session_id=session_id, **kw), parent_session_id=parent)


def test_session_report_rolls_up_descendant_spend(tmp_path):
    """``/session``'s headline question is "what did this cost me", and a
    session that spent through subagents did not spend only its own rows."""
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch(
        [
            _snap(session_id="root", cost_micro=1_000_000),
            _child("root", "kid1", cost_micro=2_000_000),
            _child("root", "kid2", cost_micro=3_000_000),
            # A grandchild: the ledger is one level deep today, but fork()
            # stamps the MIDDLE session as parent the moment a subagent spawns
            # its own, so the walk must be recursive rather than one level.
            _child("kid1", "grandkid", cost_micro=4_000_000),
            _snap(session_id="stranger", cost_micro=9_000_000),
        ]
    )
    report = store.session_report("root")
    assert report.aggregate.cost_micro == 1_000_000
    assert report.descendants_aggregate is not None
    assert report.descendants_aggregate.cost_micro == 9_000_000
    assert report.subtree_aggregate.cost_micro == 10_000_000
    assert set(report.descendant_ids) == {"kid1", "kid2", "grandkid"}
    # The unrelated session is not swept in by the recursion.
    assert "stranger" not in report.descendant_ids
    # And the own scope stays own: the diagnostic sections read this.
    assert report.aggregate.calls == 1
    store.close()


def test_session_report_without_children_reports_empty_not_unknown(tmp_path):
    """A childless session must be distinguishable from an unwalkable ledger:
    the first shows no subagent split, the second must not claim $0.00."""
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(session_id="solo")])
    report = store.session_report("solo")
    assert report.descendants_aggregate is not None
    assert report.descendants_aggregate.calls == 0
    assert report.has_descendants is False
    assert report.subtree_aggregate.cost_micro == report.aggregate.cost_micro
    store.close()


def _both_surfaces(store, session_id):
    """``(/analytics row total, /session subtree total)`` for one session.

    The agreement property this whole change exists to establish, expressed once
    so a test can assert it instead of restating the plumbing. ``/analytics``
    goes through ``aggregate`` + ``build_session_forest``; ``/session`` goes
    through ``session_report``. Review F1/F7: asserting only one side is what let
    the two encode different edge rules.
    """
    from local_operator.analytics.model import build_session_forest

    aggregate = store.aggregate()
    forest = build_session_forest(
        aggregate.by_session, getattr(aggregate, "session_parents", {}) or {}
    )
    rows = {node.session_id: node.total.cost_micro for node in forest}
    report = store.session_report(session_id)
    return rows.get(session_id), report.subtree_aggregate.cost_micro


def test_both_surfaces_share_one_parent_edge_rule(tmp_path):
    """Review F1: /analytics and /session must not encode different edge rules.

    Two shapes the reviewer reproduced as divergences. Both are unreachable on
    today's ledger (zero sessions carry more than one distinct parent), which is
    exactly why they need a test: the property is asserted by a code comment, so
    only a test defends it against the next change.
    """
    import sqlite3

    # SHAPE 1 — a real parent PLUS a degenerate self row. A lexical MAX over the
    # raw column returns 'xxx' (it sorts above 'aaa'), the self edge is then
    # discarded, and the REAL edge is lost with it: /analytics said $1000 where
    # /session said $2000. The shared rule discards the self edge BEFORE the MAX.
    store = AnalyticsStore(tmp_path / "self.db")
    store.record_batch([_snap(session_id="aaa"), _child("aaa", "xxx")])
    conn = store._connect()
    assert conn is not None
    conn.execute(
        "INSERT INTO calls (ts_ms, session_id, provider, model_id, ok, input_tokens, "
        "output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, "
        "context_tokens, cost_micro, cost_known, parent_session_id) "
        "VALUES (1, 'xxx', 'anthropic', 'claude', 1, 1, 1, 0, 0, 0, 1, 500, 1, 'xxx')"
    )
    conn.commit()
    assert getattr(store.aggregate(), "session_parents", {}) == {"xxx": "aaa"}
    analytics, session = _both_surfaces(store, "aaa")
    assert analytics == session
    assert store.session_report("aaa").descendant_ids == ("xxx",)
    store.close()

    # SHAPE 2 — a child with rows under TWO real parents. MAX still picks one,
    # which is the documented tie-break (crediting both would double-count and
    # break the column-sums-to-total invariant); the requirement is that BOTH
    # surfaces pick the SAME one. Previously /session credited both parents and
    # /analytics only the lexically larger, so 'aaa' read 1000 vs 2000.
    store2 = AnalyticsStore(tmp_path / "two.db")
    store2.record_batch([_snap(session_id="aaa"), _snap(session_id="bbb"), _child("aaa", "kid")])
    conn2 = store2._connect()
    assert conn2 is not None
    conn2.execute(
        "INSERT INTO calls (ts_ms, session_id, provider, model_id, ok, input_tokens, "
        "output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, "
        "context_tokens, cost_micro, cost_known, parent_session_id) "
        "VALUES (1, 'kid', 'anthropic', 'claude', 1, 1, 1, 0, 0, 0, 1, 500, 1, 'bbb')"
    )
    conn2.commit()
    for sid in ("aaa", "bbb"):
        analytics, session = _both_surfaces(store2, sid)
        assert analytics == session, sid
    # And the child is credited to exactly one parent, so nothing is counted twice.
    parents = getattr(store2.aggregate(), "session_parents", {})
    assert list(parents) == ["kid"]
    assert isinstance(sqlite3.connect(str(tmp_path / "two.db")), sqlite3.Connection)
    store2.close()


def test_descendant_walk_survives_a_self_parent_cycle(tmp_path):
    """224 rows in the operator's real ledger carry parent == session. A walk
    that follows that edge either loops or double-counts; this one does
    neither."""
    import sqlite3

    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(session_id="root"), _child("root", "kid")])
    conn = store._connect()
    assert conn is not None
    # Stamp the child as its own parent too — the degenerate shape the real
    # ledger contains — alongside the legitimate edge.
    conn.execute("UPDATE calls SET parent_session_id = 'kid' WHERE session_id = 'kid'")
    conn.commit()
    report = store.session_report("root")
    assert report.descendant_ids == ()  # the self-loop replaced the real edge
    # The point is that it TERMINATED and stayed honest, not what it found.
    assert report.subtree_aggregate.cost_micro == report.aggregate.cost_micro

    # And with both edges present the child is counted exactly once.
    conn.execute(
        "INSERT INTO calls (ts_ms, session_id, provider, model_id, ok, input_tokens, "
        "output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, "
        "context_tokens, cost_micro, cost_known, parent_session_id) "
        "VALUES (1, 'kid', 'anthropic', 'claude', 1, 1, 1, 0, 0, 0, 1, 500, 1, 'root')"
    )
    conn.commit()
    report = store.session_report("root")
    assert report.descendant_ids == ("kid",)
    assert report.descendants_aggregate is not None
    # Both of kid's rows, once each.
    assert report.descendants_aggregate.calls == 2
    assert isinstance(sqlite3.connect(str(tmp_path / "a.db")), sqlite3.Connection)
    store.close()


def test_parent_index_is_created_and_survives_an_old_ledger(tmp_path):
    """The index is load-bearing for the rollup (154 ms -> 1 ms), and it must
    reach a database written before ``parent_session_id`` existed WITHOUT
    aborting the schema script that also creates the rollup tables."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    # A pre-parent_session_id ledger: putting the index in _SCHEMA rather than
    # after _migrate makes executescript raise "no such column" here and lose
    # every statement after it.
    conn.executescript(f"""
        CREATE TABLE calls (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, session_id TEXT,
          provider TEXT, model_id TEXT, ok INTEGER, input_tokens INTEGER,
          output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
          reasoning_tokens INTEGER, context_tokens INTEGER,
          {_PRE_IMAGES_COMPONENT_COLUMNS}
        );
        CREATE TABLE session_names (
          session_id TEXT PRIMARY KEY, name TEXT, updated_at_ms INTEGER
        );
        """)
    conn.execute(
        "INSERT INTO calls (ts_ms, session_id, provider, model_id, ok, input_tokens, "
        "output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, "
        "context_tokens) VALUES (1, 'old', 'anthropic', 'claude', 1, 1, 1, 0, 0, 0, 1)"
    )
    conn.commit()
    conn.close()

    store = AnalyticsStore(db)
    assert store.record_batch([_snap(session_id="new")]) == 1
    live = store._connect()
    assert live is not None
    names = {row[0] for row in live.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_calls_parent" in names
    # The rest of the schema script survived: the rollup tables exist.
    tables = {row[0] for row in live.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"usage_daily", "usage_monthly"} <= tables
    store.close()


def test_aggregate_exposes_parent_edges_for_the_table_rollup(tmp_path):
    """``/analytics`` re-partitions the per-session table with these edges."""
    store = AnalyticsStore(tmp_path / "a.db")
    store.record_batch([_snap(session_id="root"), _child("root", "kid"), _snap(session_id="solo")])
    agg = store.aggregate()
    parents = getattr(agg, "session_parents", {})
    assert parents == {"kid": "root"}
    # Every session still gets its OWN row in by_session: the rollup is the
    # renderer's job, so the aggregate keeps summing to the headline total.
    assert set(agg.by_session) == {"root", "kid", "solo"}
    assert sum(a.cost_micro for a in agg.by_session.values()) == agg.cost_micro
    store.close()
