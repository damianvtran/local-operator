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
    # Force the degraded path: pretend the migration failed.
    store._has_cost = False
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
