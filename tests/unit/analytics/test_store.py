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
