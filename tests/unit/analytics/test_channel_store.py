"""The channel half of the analytics ledger: upsert, deltas, view, retention.

The load-bearing claim (design §3.4, risk 5) is that the day/month rollups stay
EQUAL to the ledger after a quote→settled upgrade: the delta path is the only
new correctness-sensitive SQL here, so these tests compare the rollup against a
fresh GROUP BY over ``channel_calls`` rather than against a hand-written
expectation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.store import AnalyticsStore

DAY_MS = 1_760_000_000_000


def channel_row(
    record_id: str = "image:req1",
    *,
    rev: int = 0,
    ts_ms: int = DAY_MS,
    channel: str = "image",
    provider: str = "radient",
    model: str = "gpt-image-2",
    units: float = 1.0,
    unit: str = "images",
    amount_micro: int | None = 53000,
    basis: str = "estimated",
    status: str = "ok",
    session_id: str = "s1",
) -> tuple[Any, ...]:
    """One store insert tuple: ``_CHANNEL_COLUMNS`` minus ``updated_at_ms``."""
    return (
        record_id,
        rev,
        ts_ms,
        session_id,
        "",
        channel,
        provider,
        model,
        units,
        unit,
        amount_micro,
        basis,
        "server_reported",
        "quote-v1",
        status,
        record_id.split(":", 1)[-1],
    )


def rollup_matches_ledger(conn: sqlite3.Connection, table: str, key: str) -> bool:
    """Every rollup bucket equals the GROUP BY of the raw ledger."""
    from local_operator.analytics.store import _local_day_month

    rollup = {
        row[:5]: (row[5], row[6], row[7], row[8])
        for row in conn.execute(
            f"SELECT {key}, channel, provider, model, billing_basis, units, "
            f"amount_micro, calls, known_calls FROM {table}"
        )
    }
    ledger: dict[tuple[Any, ...], list[Any]] = {}
    for row in conn.execute(
        "SELECT ts_ms, channel, provider, model_id, billing_basis, units, amount_micro "
        "FROM channel_calls"
    ):
        ts_ms, channel, provider, model_id, basis, units, amount = row
        bucket = _local_day_month(int(ts_ms))[0 if key == "day" else 1]
        entry = ledger.setdefault((bucket, channel, provider, model_id, basis), [0.0, 0, 0, 0])
        entry[0] += float(units or 0.0)
        entry[1] += int(amount or 0)
        entry[2] += 1
        entry[3] += 0 if amount is None else 1
    return rollup == {k: tuple(v) for k, v in ledger.items()}


def test_insert_replay_and_rev_upgrade_move_the_rollups_exactly(tmp_path: Path) -> None:
    path = tmp_path / "analytics.db"
    store = AnalyticsStore(db_path=path)
    assert store.record_channel_batch([channel_row()]) == 1
    assert store.record_channel_batch([channel_row()]) == 1, "a replay is processed, not written"
    assert (
        store.record_channel_batch([channel_row(rev=0, amount_micro=999)]) == 1
    ), "a stale rev is processed and ignored"

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT rev, amount_micro FROM channel_calls").fetchall() == [(0, 53000)]
    assert rollup_matches_ledger(conn, "channel_daily", "day")
    assert rollup_matches_ledger(conn, "channel_monthly", "month")

    # Quote -> settled: the amount AND the basis move; the call count does not.
    assert store.record_channel_batch([channel_row(rev=1, amount_micro=61000, basis="billed")]) == 1
    assert conn.execute(
        "SELECT rev, amount_micro, billing_basis FROM channel_calls"
    ).fetchall() == [(1, 61000, "billed")]
    daily = conn.execute(
        "SELECT billing_basis, units, amount_micro, calls, known_calls FROM channel_daily"
    ).fetchall()
    assert daily == [("billed", 1.0, 61000, 1, 1)], daily
    assert rollup_matches_ledger(conn, "channel_daily", "day")
    assert rollup_matches_ledger(conn, "channel_monthly", "month")


def test_unknown_amounts_are_null_and_known_calls_track_them(tmp_path: Path) -> None:
    path = tmp_path / "analytics.db"
    store = AnalyticsStore(db_path=path)
    assert (
        store.record_channel_batch(
            [
                channel_row(
                    record_id="tts:r1", channel="tts", amount_micro=None, basis="not_tracked"
                ),
                channel_row(record_id="tts:r2", channel="tts", amount_micro=0, basis="billed"),
            ]
        )
        == 2
    )
    conn = sqlite3.connect(path)
    amounts = dict(conn.execute("SELECT record_id, amount_micro FROM channel_calls").fetchall())
    assert amounts == {"tts:r1": None, "tts:r2": 0}, "NULL is not 0, and 0 is not NULL"
    rows = {
        row[0]: tuple(row[1:])
        for row in conn.execute(
            "SELECT billing_basis, units, amount_micro, calls, known_calls "
            "FROM channel_daily WHERE channel = 'tts'"
        )
    }
    assert rows == {"not_tracked": (1.0, 0, 1, 0), "billed": (1.0, 0, 1, 1)}, rows


def test_spend_all_view_unions_inference_and_channels(tmp_path: Path) -> None:
    path = tmp_path / "analytics.db"
    store = AnalyticsStore(db_path=path)
    assert store.record_channel_batch([channel_row()]) == 1
    snapshot = CallSnapshot(
        ts_ms=DAY_MS,
        session_id="s1",
        provider="deepseek",
        model_id="deepseek-chat",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_tokens=10,
        priced=True,
        cost_micro=2000,
        cost_known=True,
    )
    assert store.record_batch([snapshot]) == 1
    store.close()

    reopened = AnalyticsStore(db_path=path)
    conn = sqlite3.connect(path)
    rows = dict(
        (row[0], row)
        for row in conn.execute(
            "SELECT channel, session_id, amount_micro, status FROM spend_all ORDER BY channel"
        )
    )
    assert set(rows) == {"image", "inference"}
    assert rows["inference"][2] == 2000, "a known call's cost is the amount"
    assert rows["image"][2] == 53000
    reopened.close()


def test_reopening_an_old_database_migrates_and_recreates_the_view(tmp_path: Path) -> None:
    path = tmp_path / "analytics.db"
    store = AnalyticsStore(db_path=path)
    assert (
        store.record_channel_batch([channel_row()]) == 1
    ), "a write is what opens (and creates the schema); the ctor does not"
    store.close()
    # Simulate a later release's column drift by dropping the view; the open
    # must recreate it (a VIEW has no ALTER path — design §3.4).
    conn = sqlite3.connect(path)
    conn.execute("DROP VIEW spend_all")
    conn.commit()
    conn.close()
    reopened = AnalyticsStore(db_path=path)
    assert reopened.record_channel_batch([channel_row()]) == 1
    conn = sqlite3.connect(path)
    assert (
        conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'spend_all'").fetchone()[0]
        == 1
    )
    reopened.close()


def test_prune_drops_old_channel_rows_and_keeps_the_rollups(tmp_path: Path) -> None:
    path = tmp_path / "analytics.db"
    store = AnalyticsStore(db_path=path)
    old = DAY_MS - 200 * 24 * 60 * 60 * 1000
    assert store.record_channel_batch([channel_row(record_id="old:1", ts_ms=old)]) == 1
    assert store.record_channel_batch([channel_row()]) == 1
    store.prune(now_ms=DAY_MS + 24 * 60 * 60 * 1000)
    conn = sqlite3.connect(path)
    assert [row[0] for row in conn.execute("SELECT record_id FROM channel_calls")] == ["image:req1"]
    assert (
        conn.execute("SELECT COUNT(*) FROM channel_daily").fetchone()[0] >= 1
    ), "the rollup survives the raw ledger's window"


def test_calls_basis_columns_migrate_and_the_view_reads_them(tmp_path: Path) -> None:
    """PR-3: the two columns arrive by ALTER, and ``spend_all`` reads them.

    Old rows read the schema DEFAULTs — ``not_tracked``/``none`` — which is the
    truth about them: no basis was ever recorded, and a historic row is never
    retro-labelled. New rows carry the labels decided at record time, so a
    surface filtering by basis sees inference and channel money in one
    vocabulary.
    """
    path = tmp_path / "analytics.db"
    store = AnalyticsStore(db_path=path)
    snapshot = CallSnapshot(
        ts_ms=DAY_MS,
        session_id="s1",
        provider="openrouter",
        model_id="m",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_tokens=10,
        priced=True,
        cost_micro=2000,
        cost_known=True,
        billing_basis="billed",
        cost_source="provider_reported",
    )
    assert store.record_batch([snapshot]) == 1
    assert store.record_channel_batch([channel_row()]) == 1
    store.close()

    conn = sqlite3.connect(path)
    rows = {
        row[0]: (row[1], row[2])
        for row in conn.execute("SELECT channel, billing_basis, cost_source FROM spend_all")
    }
    assert rows["inference"] == ("billed", "provider_reported")
    assert rows["image"] == ("estimated", "server_reported")

    # Simulate a PRE-PR-3 database: the view must go first, because SQLite
    # refuses to drop a column a view references (the same rule the cost_micro
    # migration test documents).
    conn.execute("DROP VIEW spend_all")
    conn.execute("ALTER TABLE calls DROP COLUMN billing_basis")
    conn.execute("ALTER TABLE calls DROP COLUMN cost_source")
    conn.commit()
    conn.close()

    reopened = AnalyticsStore(db_path=path)
    # ``_connect`` is what opens the file and runs the ALTER pass: the store
    # opens lazily, so asserting straight after construction would test
    # nothing (the pattern the first_reasoning_ms migration test set).
    conn = reopened._connect()
    assert conn is not None
    # The ALTER path put them back; the old row reads the defaults.
    assert conn.execute("SELECT billing_basis, cost_source FROM calls").fetchall() == [
        ("not_tracked", "none")
    ]
    assert (
        conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'spend_all'").fetchone()[0]
        == 1
    ), "the view is recreated after the migration"
    assert conn.execute(
        "SELECT billing_basis, cost_source FROM spend_all WHERE channel = 'inference'"
    ).fetchone() == ("not_tracked", "none"), "an old row is not relabelled by the view"
    reopened.close()


# -- PR-2: the channel READS (breakdown + calendar series) -------------------


def test_channel_breakdown_groups_and_honours_window_and_session(tmp_path: Path) -> None:
    """The windowed breakdown: grouped per (channel, provider, basis, source).

    The stated/unstated pair is the load-bearing half: an amount-unknown call
    must leave ``amount_micro`` as the KNOWN sum and be visible through
    ``known_calls`` — a group of zeros with ``known_calls=0`` reads as "could
    not be sized", never as "free".
    """
    store = AnalyticsStore(db_path=tmp_path / "analytics.db")
    assert (
        store.record_channel_batch(
            [
                channel_row("image:r1", ts_ms=DAY_MS, amount_micro=53_000, session_id="s1"),
                channel_row("image:r2", ts_ms=DAY_MS + 1, amount_micro=None, session_id="s1"),
                channel_row(
                    "tts:r1",
                    ts_ms=DAY_MS + 2,
                    channel="tts",
                    provider="elevenlabs",
                    basis="billed",
                    amount_micro=1_200,
                    session_id="s2",
                ),
            ]
        )
        == 3
    )

    rows = store.channel_breakdown(session_id="s1")
    assert [(row.channel, row.calls, row.known_calls, row.amount_micro) for row in rows] == [
        ("image", 2, 1, 53_000)
    ], "the unknown amount is not a zero and does not vanish"

    every = store.channel_breakdown()
    assert {row.channel for row in every} == {"image", "tts"}
    tts = next(row for row in every if row.channel == "tts")
    assert (tts.provider, tts.billing_basis, tts.cost_source) == (
        "elevenlabs",
        "billed",
        "server_reported",
    )

    # The window is half-open on ``ts_ms``, the same bounds ``aggregate`` uses.
    windowed = store.channel_breakdown(since_ms=DAY_MS + 2)
    assert {row.channel for row in windowed} == {"tts"}
    store.close()


def test_channel_series_sum_providers_and_split_them_on_request(tmp_path: Path) -> None:
    """The rollup series: one (period, channel, basis) row, or per provider.

    Two providers under one channel SUM into the default series and SPLIT when
    ``by_provider=True`` — the two shapes the payload offers a client that
    charts spend over time.
    """
    store = AnalyticsStore(db_path=tmp_path / "analytics.db")
    assert (
        store.record_channel_batch(
            [
                channel_row(
                    "search:r1",
                    ts_ms=DAY_MS,
                    channel="search",
                    provider="brave",
                    amount_micro=250,
                ),
                channel_row(
                    "search:r2",
                    ts_ms=DAY_MS + 1,
                    channel="search",
                    provider="exa",
                    amount_micro=150,
                ),
            ]
        )
        == 2
    )

    daily = store.channel_daily_series(7)
    assert len(daily) == 1
    row = daily[0]
    assert (row.channel, row.provider, row.billing_basis) == ("search", "", "estimated")
    assert (row.amount_micro, row.calls, row.known_calls) == (400, 2, 2)

    split = store.channel_daily_series(7, by_provider=True)
    assert {(row.provider, row.amount_micro) for row in split} == {
        ("brave", 250),
        ("exa", 150),
    }

    monthly = store.channel_monthly_series(12)
    assert len(monthly) == 1 and monthly[0].amount_micro == 400
    store.close()


def test_channel_reads_answer_empty_without_the_channel_tables(tmp_path: Path) -> None:
    """A ledger without the channel tables reads as empty, never as an error.

    ``_has_channel_tables`` is False on a pre-feature or read-only database;
    the reads are additive to payloads that already have an answer, so all
    three return ``[]`` rather than raising into a report.
    """
    store = AnalyticsStore(db_path=tmp_path / "analytics.db")
    assert store.record_channel_batch([channel_row()]) == 1
    store._has_channel_tables = False  # the state the flag holds on an old db
    assert store.channel_breakdown() == []
    assert store.channel_daily_series() == []
    assert store.channel_monthly_series() == []
    store.close()
