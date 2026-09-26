"""The decode-rate measures: capture shape, migration, aggregation, per-model read.

The metric is a RATE, so these tests are about the two things a rate can get
wrong and a total cannot: which calls are in the numerator and denominator (they
must be the SAME calls), and whether "no sample" is distinguishable from a
measured zero. Both are pinned here rather than left to the screens, because a
screen cannot recover a fact the store did not keep.

The recording predicate itself lives in ``SessionStreamFn._record_usage`` and is
covered by ``test_stream_recording.py``; this file covers everything downstream of
a snapshot: the columns, the rollup, the migration, the two aggregate paths, the
per-model read and the row-to-result positional contract.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from local_operator.analytics.model import CallSnapshot, UsageAggregate
from local_operator.analytics.store import AnalyticsStore

#: The pre-decode ``session_daily`` shape, copied from the schema that shipped
#: before these measures. Written out rather than derived so a test that adds a
#: column to the real schema cannot silently stop testing the OLD shape — the
#: whole point is to open a database no current code created.
_OLD_SESSION_DAILY = """
CREATE TABLE session_daily (
  day TEXT NOT NULL,
  session_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  parent_session_id TEXT,
  ok INTEGER NOT NULL DEFAULT 0,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens INTEGER NOT NULL DEFAULT 0,
  context_tokens INTEGER NOT NULL DEFAULT 0,
  cost_micro INTEGER NOT NULL DEFAULT 0,
  cost_known INTEGER NOT NULL DEFAULT 0,
  calls INTEGER NOT NULL DEFAULT 0,
  c_system_prompt INTEGER NOT NULL DEFAULT 0,
  c_custom_instructions INTEGER NOT NULL DEFAULT 0,
  c_tool_inventory INTEGER NOT NULL DEFAULT 0,
  c_tool_schemas INTEGER NOT NULL DEFAULT 0,
  c_environment INTEGER NOT NULL DEFAULT 0,
  c_knowledge INTEGER NOT NULL DEFAULT 0,
  c_conversation INTEGER NOT NULL DEFAULT 0,
  c_tool_results INTEGER NOT NULL DEFAULT 0,
  c_images INTEGER NOT NULL DEFAULT 0,
  max_ts_ms INTEGER NOT NULL DEFAULT 0,
  updated_at_ms INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, session_id, provider)
);
CREATE TABLE session_daily_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL DEFAULT ''
);
"""


def _snap(
    *,
    session_id: str = "s1",
    provider: str = "anthropic",
    model_id: str = "claude",
    output_tokens: int = 40,
    decode_us: int = 0,
    decode_tokens: int = 0,
    decode_calls: int = 0,
    duration_ms: float = 5000.0,
    ts_ms: int | None = None,
) -> CallSnapshot:
    """A snapshot with the decode measures set explicitly.

    The three are passed INDEPENDENTLY of ``output_tokens`` on purpose: the
    recording path's eligibility predicate is what keeps them in step in
    production, and a test that derives them from ``output_tokens`` here would
    hide a reader that assumed they always agree.
    """
    return CallSnapshot(
        ts_ms=ts_ms if ts_ms is not None else int(time.time() * 1000),
        session_id=session_id,
        provider=provider,
        model_id=model_id,
        input_tokens=100,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_tokens=100,
        component_chars={"conversation": 100},
        ok=True,
        cost_micro=1000,
        cost_known=True,
        priced=True,
        duration_ms=duration_ms,
        decode_us=decode_us,
        decode_tokens=decode_tokens,
        decode_calls=decode_calls,
    )


def _store(tmp_path) -> AnalyticsStore:
    return AnalyticsStore(tmp_path / "analytics.db")


# ---------------------------------------------------------------------------
# Aggregation arithmetic
# ---------------------------------------------------------------------------


def test_decode_tps_is_none_with_no_coverage_and_never_zero() -> None:
    """Unknown is ``None``: a zero here would claim every call decoded instantly."""
    assert UsageAggregate().decode_tps is None
    assert UsageAggregate().decode_coverage is None
    # Calls that exist but carry no window: still unknown, not zero.
    covered = UsageAggregate(calls=10, output_tokens=500)
    assert covered.decode_tps is None
    assert covered.decode_coverage == 0.0


def test_decode_tps_is_sum_weighted_not_a_mean_of_rates() -> None:
    """The headline must be SUM/SUM, or a short call outvotes a long one.

    One 20-token call decoding in 20 ms is 1000 tok/s; one 20 000-token call
    decoding in 20 s is 1000 tok/s. Deliberately chosen so SUM/SUM and the mean
    agree — then a third call breaks the tie: 10 tokens in 100 ms (100 tok/s).
    The mean is ~700 tok/s; SUM/SUM is the real thing.
    """
    agg = UsageAggregate(
        calls=3,
        decode_us=20_000 + 20_000_000 + 100_000,
        decode_tokens=20 + 20_000 + 10,
        decode_calls=3,
    )
    weighted = (20 + 20_000 + 10) / ((20_000 + 20_000_000 + 100_000) / 1e6)
    mean_of_rates = (1000 + 1000 + 100) / 3
    assert agg.decode_tps == pytest.approx(weighted)
    assert agg.decode_tps != pytest.approx(mean_of_rates)


def test_decode_coverage_states_the_share_of_calls_the_rate_speaks_for() -> None:
    agg = UsageAggregate(calls=8, decode_calls=2, decode_us=1_000_000, decode_tokens=100)
    assert agg.decode_coverage == pytest.approx(0.25)
    assert agg.decode_tps == pytest.approx(100.0)


def test_a_genuinely_slow_rate_is_returned_rather_than_flattened_to_none() -> None:
    """Under 1 tok/s is a measurement, not an absence — the inverse trap."""
    agg = UsageAggregate(calls=1, decode_calls=1, decode_us=2_500_000, decode_tokens=1)
    assert agg.decode_tps == pytest.approx(0.4)


def test_sum_aggregates_adds_the_counters_so_a_mixed_tree_comes_out_right() -> None:
    covered = UsageAggregate(calls=1, decode_calls=1, decode_us=1_000_000, decode_tokens=100)
    uncovered = UsageAggregate(calls=9)
    from local_operator.analytics.model import sum_aggregates

    total = sum_aggregates([covered, uncovered])
    assert (total.calls, total.decode_calls) == (10, 1)
    assert total.decode_tps == pytest.approx(100.0)
    assert total.decode_coverage == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Ledger, rollup, and the two-paths-one-answer property
# ---------------------------------------------------------------------------


def test_the_measures_survive_the_ledger_and_the_rollup_identically(tmp_path) -> None:
    """``aggregate()`` must answer the same on either path.

    The rollup path is what the panel normally reads; the ledger path is the
    fail-closed fallback. A metric that only one of them carries is a number that
    changes when the store decides to be slower, which is the defect this whole
    two-path arrangement exists to prevent.
    """
    store = _store(tmp_path)
    now = int(time.time() * 1000)
    store.record_batch(
        [
            _snap(decode_us=1_000_000, decode_tokens=100, decode_calls=1, ts_ms=now),
            _snap(
                session_id="s2",
                model_id="other",
                decode_us=4_000_000,
                decode_tokens=200,
                decode_calls=1,
                ts_ms=now,
            ),
            # A call with output tokens and NO window: counted in ``calls`` and
            # in no rate. This is the population the eligibility predicate
            # excludes, and the one a naive SUM(output_tokens) would smuggle in.
            _snap(session_id="s3", output_tokens=900, ts_ms=now),
        ]
    )
    # Whole-window read with no bounds takes the rollup, which is the panel's path.
    rolled = store.aggregate()
    store.close()

    ledger = AnalyticsStore(tmp_path / "analytics.db")
    ledger._has_session_daily = False  # force the fallback on the same file
    fell_back = ledger.aggregate()
    ledger.close()

    assert rolled.decode_us == fell_back.decode_us == 5_000_000
    assert rolled.decode_tokens == fell_back.decode_tokens == 300
    assert rolled.decode_calls == fell_back.decode_calls == 2
    assert rolled.decode_tps == pytest.approx(fell_back.decode_tps)
    # The 900-token call is in neither numerator nor denominator.
    assert rolled.decode_tps == pytest.approx(60.0)
    assert rolled.decode_coverage == pytest.approx(2 / 3)


def test_session_report_by_model_inherits_the_measures(tmp_path) -> None:
    """One indexed scan feeds the per-model table, so it must carry the rate."""
    store = _store(tmp_path)
    now = int(time.time() * 1000)
    store.record_batch(
        [
            _snap(decode_us=2_000_000, decode_tokens=400, decode_calls=1, ts_ms=now),
            _snap(
                provider="openai",
                model_id="gpt",
                decode_us=1_000_000,
                decode_tokens=50,
                decode_calls=1,
                ts_ms=now + 1,
            ),
        ]
    )
    report = store.session_report("s1")
    store.close()
    assert set(report.by_model) == {("anthropic", "claude"), ("openai", "gpt")}
    assert report.by_model[("anthropic", "claude")].decode_tps == pytest.approx(200.0)
    assert report.by_model[("openai", "gpt")].decode_tps == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# The rollup migration, and the guard that has to grow with it
# ---------------------------------------------------------------------------


def _pre_column_db(path) -> None:
    """A ledger with the OLD ``session_daily``: no decode columns anywhere."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_OLD_SESSION_DAILY)
    conn.commit()
    conn.close()


def test_a_pre_column_rollup_gains_the_measures_on_open(tmp_path) -> None:
    path = tmp_path / "analytics.db"
    _pre_column_db(path)
    # A fresh store migrates on connect, exactly as the recorder's writer does.
    store = AnalyticsStore(path)
    store._connect()
    columns = {
        row[1] for row in sqlite3.connect(str(path)).execute("PRAGMA table_info(session_daily)")
    }
    assert {"decode_us", "decode_tokens", "decode_calls"} <= columns
    # And the guard accepts the shape it just produced.
    assert store._has_session_daily is True
    store.close()


def test_a_failed_rollup_alter_reads_as_no_rollup_and_the_ledger_still_records(
    tmp_path, monkeypatch
) -> None:
    """The fail-closed path: slow reads, never a dropped ledger batch.

    A rollup upsert that names a column the table lacks raises inside
    ``record_batch``'s single transaction, and the whole batch — ledger row
    included — is rolled back and DROPPED. So when the ALTER cannot run, the
    guard must switch the rollup write path OFF.

    The ALTER is made to fail through the migration REGISTRY rather than by
    monkeypatching ``sqlite3.Connection`` (which is an immutable type): a
    definition carrying a second ``ADD COLUMN`` clause is not valid ``ALTER
    TABLE`` syntax, so the statement raises exactly as a locked or read-only
    database would.
    """
    from local_operator.analytics import store as store_module

    path = tmp_path / "analytics.db"
    _pre_column_db(path)
    monkeypatch.setattr(
        store_module,
        "_SESSION_DAILY_MIGRATION_COLUMNS",
        (("decode_us", "INTEGER NOT NULL DEFAULT 0, ADD COLUMN zzz INTEGER"),),
    )
    store = AnalyticsStore(path)
    store._connect()

    assert store._has_session_daily is False
    # The ledger still records, which is the property the guard buys.
    assert store.record_batch([_snap()]) == 1
    rows = sqlite3.connect(str(path)).execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    assert rows == 1
    store.close()


def test_the_guard_rejects_a_column_the_upsert_cannot_fill(tmp_path) -> None:
    """The case review R15 left open now fails CLOSED.

    A superset check alone cannot see a column this code does not insert into:
    one that is ``NOT NULL`` with no default passes the superset test and then
    makes the upsert raise. The check now reads ``notnull``/``dflt_value`` too.
    """
    path = tmp_path / "analytics.db"
    store = AnalyticsStore(path)
    store._connect()
    store.close()
    conn = sqlite3.connect(str(path))
    # ``session_daily_meta`` is created by ``_SCHEMA``; add the offending column
    # to a fully-migrated table so ONLY this one property differs.
    conn.execute("ALTER TABLE session_daily ADD COLUMN future_measure INTEGER NOT NULL")
    conn.commit()
    conn.close()

    reopened = AnalyticsStore(path)
    reopened._connect()
    assert reopened._has_session_daily is False
    reopened.close()


# ---------------------------------------------------------------------------
# The per-model read
# ---------------------------------------------------------------------------


def test_model_rates_groups_by_model_and_keeps_the_two_rates_apart(tmp_path) -> None:
    store = _store(tmp_path)
    now = int(time.time() * 1000)
    store.record_batch(
        [
            # A measured decode window AND a wall duration.
            _snap(
                output_tokens=100,
                decode_us=1_000_000,
                decode_tokens=100,
                decode_calls=1,
                duration_ms=2500.0,
                ts_ms=now,
            ),
            # Output tokens, no window, but a real duration: appears in ``calls``
            # and in the wall rate only.
            _snap(output_tokens=300, duration_ms=3000.0, ts_ms=now + 1),
            _snap(
                provider="openai", model_id="gpt", output_tokens=50, duration_ms=0.0, ts_ms=now + 2
            ),
        ]
    )
    rows = {f"{r.provider}/{r.model_id}": r for r in store.model_rates()}
    store.close()

    anthropic = rows["anthropic/claude"]
    assert anthropic.calls == 2
    assert anthropic.output_tokens == 100 + 300
    assert anthropic.decode_tps == pytest.approx(100.0)
    assert anthropic.decode_calls == 1
    # SUM(duration) = 5.5 s over 400 output tokens = 72.7 tok/s, and the 300-token
    # call is IN this rate even though it is out of the decode one.
    assert anthropic.wall_tps == pytest.approx(400 / 5.5)
    assert anthropic.wall_calls == 2

    openai = rows["openai/gpt"]
    # No duration sample: unknown, never zero, even though the call exists.
    assert openai.wall_tps is None
    assert openai.decode_tps is None
    assert openai.calls == 1


def test_model_rates_orders_by_output_tokens_and_honours_the_limit(tmp_path) -> None:
    store = _store(tmp_path)
    now = int(time.time() * 1000)
    store.record_batch(
        [
            _snap(model_id="small", output_tokens=10, ts_ms=now),
            _snap(model_id="big", output_tokens=9000, ts_ms=now + 1),
            _snap(model_id="mid", output_tokens=500, ts_ms=now + 2),
        ]
    )
    rows = store.model_rates()
    all_rows = store.model_rates(limit=2)
    store.close()
    assert [r.model_id for r in rows] == ["big", "mid", "small"]
    assert [r.model_id for r in all_rows] == ["big", "mid"]


def test_model_rates_scopes_to_a_session_and_a_window(tmp_path) -> None:
    store = _store(tmp_path)
    now = int(time.time() * 1000)
    store.record_batch(
        [
            _snap(session_id="s1", model_id="mine", output_tokens=100, ts_ms=now - 10_000),
            _snap(session_id="s2", model_id="theirs", output_tokens=100, ts_ms=now - 10_000),
            _snap(session_id="s1", model_id="mine", output_tokens=100, ts_ms=now),
        ]
    )
    scoped = store.model_rates(session_id="s1")
    windowed = store.model_rates(since_ms=now - 1_000)
    store.close()
    assert [(r.model_id, r.calls) for r in scoped] == [("mine", 2)]
    assert [(r.model_id, r.calls) for r in windowed] == [("mine", 1)]


def test_model_rates_on_a_pre_column_ledger_answers_unknown_rather_than_failing(
    tmp_path,
) -> None:
    """An un-migrated copy must answer, not raise.

    ``decode_*`` and ``duration_ms`` are all post-first-release columns, and the
    read substitutes a constant 0 for whichever the ledger lacks — the same
    treatment ``_ledger_aggregate`` gives the cost columns. The rates then read
    as UNKNOWN, which is what a ledger with no such samples actually knows.
    """
    path = tmp_path / "analytics.db"
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE calls (
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
          context_tokens INTEGER NOT NULL DEFAULT 0
        );
        """)
    conn.execute(
        "INSERT INTO calls (ts_ms, session_id, provider, model_id, output_tokens) "
        "VALUES (?, 's1', 'anthropic', 'claude', 700)",
        (int(time.time() * 1000),),
    )
    conn.commit()
    conn.close()

    store = AnalyticsStore(path)
    rows = store.model_rates()
    aggregate = store.aggregate()
    store.close()
    assert len(rows) == 1
    assert rows[0].calls == 1 and rows[0].output_tokens == 700
    assert rows[0].decode_tps is None and rows[0].wall_tps is None
    assert aggregate.decode_tps is None
