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
