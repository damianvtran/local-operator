"""Exact-ID reports use one read-only snapshot, never transcript inheritance."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

from local_operator.analytics.store import AnalyticsStore
from tests.unit.analytics.test_store import _snap


def test_exact_session_scope_models_purposes_and_costs(tmp_path):
    store = AnalyticsStore(tmp_path / "ledger.db")
    first = replace(
        _snap(session_id="parent", model_id="one"),
        request_id="r1",
        purpose="turn",
        outcome="ok",
        duration_ms=200,
        ttft_ms=50,
        preparation_ms=10,
    )
    assert (
        store.record_batch(
            [
                first,
                replace(first, request_id="r2", model_id="two", cost_micro=0, cost_known=True),
                replace(
                    first,
                    request_id="r3",
                    provider="other",
                    model_id="one",
                    cost_micro=0,
                    cost_known=False,
                    usage_reported=False,
                    outcome="error",
                    ok=False,
                ),
                replace(first, session_id="child", parent_session_id="parent"),
                replace(first, session_id="fork"),
            ]
        )
        == 5
    )
    report = store.session_report("parent")
    assert report.available and report.aggregate.calls == 3
    assert report.aggregate.total_tokens == 3 * (920 + 40)
    assert report.aggregate.generation_tokens == 90
    assert report.aggregate.cost_micro == 1000
    assert report.aggregate.cost_is_partial
    assert set(report.by_model) == {("anthropic", "one"), ("anthropic", "two"), ("other", "one")}
    assert report.by_model["anthropic", "two"].cost_is_known
    assert report.by_model["anthropic", "two"].cost_micro == 0
    assert not report.by_model["other", "one"].cost_is_known
    assert report.by_purpose_outcome == {("turn", "ok"): 2, ("turn", "error"): 1}
    # by_purpose carries CONSUMPTION, which the count-only cross-tab above
    # cannot: "what did compaction cost me" is not answerable from a count.
    # Same `measures` contract as by_model, so the two must agree on the total.
    assert set(report.by_purpose) == {"turn"}
    assert report.by_purpose["turn"].calls == 3
    assert report.by_purpose["turn"].total_tokens == report.aggregate.total_tokens
    assert report.by_purpose["turn"].cost_micro == report.aggregate.cost_micro
    assert report.by_purpose["turn"].cost_is_partial
    assert report.missing_usage_calls == 1
    assert report.unknown_usage_calls == 0
    assert report.timings["duration_ms"].samples == 3
    assert report.timings["duration_ms"].mean_ms == 200
    assert len(report.recent) == 3
    assert store.session_report("parent").aggregate.calls == 3  # resume retains the ID
    assert store.session_report("fork-new").aggregate.calls == 0  # copied history isn't ledger data
    store.close()


def test_recent_bounded_order_and_unknown_timing(tmp_path):
    store = AnalyticsStore(tmp_path / "ledger.db")
    assert (
        store.record_batch(
            [
                replace(_snap(ts_ms=100 + n), request_id=f"r{n}", duration_ms=n if n else -1)
                for n in range(60)
            ]
        )
        == 60
    )
    report = store.session_report("s1", recent_limit=500)
    assert len(report.recent) == 50
    assert [r.request_id for r in report.recent[:2]] == ["r59", "r58"]
    assert report.timings["duration_ms"].samples == 59
    assert report.timings["duration_ms"].min_ms == 1
    assert report.timings["ttft_ms"].samples == 0
    assert report.timings["ttft_ms"].mean_ms is None
    assert report.first_ts_ms == 100 and report.last_ts_ms == 159
    assert not store.session_report("s1", recent_limit=-1).recent
    store.close()


def test_missing_database_is_empty_without_creating_it(tmp_path):
    path = tmp_path / "missing" / "ledger.db"
    report = AnalyticsStore(path).session_report("s1")
    assert report.available and report.aggregate.calls == 0
    assert not path.parent.exists()


def test_corrupt_database_is_unavailable_not_empty(tmp_path):
    path = tmp_path / "ledger.db"
    path.write_text("not sqlite")
    report = AnalyticsStore(path).session_report("s1")
    assert not report.available
    assert path.read_text() == "not sqlite"


def test_legacy_columns_remain_unknown_and_schema_unchanged(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE calls(id INTEGER PRIMARY KEY, ts_ms INTEGER, session_id TEXT, "
            "provider TEXT, model_id TEXT, context_tokens INTEGER, output_tokens INTEGER)"
        )
        conn.execute("INSERT INTO calls VALUES(1, 100, 's1', 'p', 'm', 100, 20)")
    before = path.read_bytes()
    report = AnalyticsStore(path).session_report("s1")
    assert report.available and report.aggregate.calls == 1
    assert report.aggregate.total_tokens == 120
    assert not report.aggregate.cost_is_known
    assert report.unknown_usage_calls == 1 and report.missing_usage_calls == 0
    assert report.by_purpose_outcome == {("unknown", "unknown"): 1}
    # No `purpose` column on this ledger: `col()` folds every row into one
    # honest `unknown` bucket rather than failing the query or inventing labels.
    assert set(report.by_purpose) == {"unknown"}
    assert report.by_purpose["unknown"].total_tokens == 120
    assert report.recent[0].duration_ms is None
    assert report.recent[0].usage_reported is None
    assert path.read_bytes() == before


def test_id_is_bound_not_interpolated(tmp_path):
    store = AnalyticsStore(tmp_path / "ledger.db")
    store.record_batch([_snap()])
    assert store.session_report("' OR 1=1 --").aggregate.calls == 0
    assert store.session_report("s1").aggregate.calls == 1
    store.close()


def test_all_sections_share_snapshot_during_concurrent_commit(tmp_path, monkeypatch):
    store = AnalyticsStore(tmp_path / "ledger.db")
    store.record_batch([replace(_snap(), request_id="before")])
    original = sqlite3.connect
    added = False

    class Connection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal added
            result = super().execute(sql, parameters)
            if sql.startswith("SELECT COUNT(*)") and not added:
                added = True
                # A separate WAL writer commits AFTER the read transaction's
                # snapshot is established, before the group/recent queries.
                other = AnalyticsStore(tmp_path / "ledger.db")
                assert other.record_batch([replace(_snap(), request_id="after")]) == 1
                other.close()
            return result

    def connect(*args, **kwargs):
        if kwargs.get("uri"):
            kwargs["factory"] = Connection
        return original(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    report = store.session_report("s1")
    assert added and report.aggregate.calls == 1
    assert sum(g.calls for g in report.by_model.values()) == 1
    assert [r.request_id for r in report.recent] == ["before"]
    assert store.session_report("s1").aggregate.calls == 2
    store.close()


def test_by_purpose_splits_consumption_across_real_purposes(tmp_path):
    """One GROUP BY on an existing column; the parts must sum to the whole."""
    store = AnalyticsStore(tmp_path / "ledger.db")
    base = replace(_snap(), request_id="r0", outcome="ok")
    store.record_batch(
        [
            replace(base, request_id="r1", purpose="turn"),
            replace(base, request_id="r2", purpose="turn"),
            replace(base, request_id="r3", purpose="compaction", cost_micro=250),
            replace(
                base, request_id="r4", purpose="naming", cost_micro=90, ok=False, outcome="error"
            ),
        ]
    )
    report = store.session_report("s1")
    assert set(report.by_purpose) == {"turn", "compaction", "naming"}
    assert report.by_purpose["turn"].calls == 2
    # The partition is exact: every purpose's tokens and cost sum to the total,
    # which is what lets the by-purpose bars carry share-of-session percentages.
    assert sum(a.calls for a in report.by_purpose.values()) == report.aggregate.calls
    assert sum(a.total_tokens for a in report.by_purpose.values()) == report.aggregate.total_tokens
    assert sum(a.cost_micro for a in report.by_purpose.values()) == report.aggregate.cost_micro
    # The failure annotation still comes from the outcome cross-tab, not here.
    assert report.by_purpose_outcome[("naming", "error")] == 1
    assert report.by_purpose["naming"].calls == 1
    store.close()
