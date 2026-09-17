"""``session_report`` reads fewer times and returns exactly the same report.

The ``/session`` panel's op is ``sessions.report`` ->
``AnalyticsStore.session_report``, and on the operator's busiest real session
(27,974 calls) it took ~1.7 s. The merge that fixes that is arithmetic, not
semantics: three GROUP BYs over the same rows become ONE at the finest grain the
three can be re-derived from, and the three timing aggregates plus the
missing/unknown/span figures fold into the totals scan.

So the property to prove is not "the panel still works" but "every field is the
same value, key for key" — including the ones that only differ on an edge: a
session whose rows span two providers, a purpose appearing under two outcomes, a
call with no timing sample, an older ledger missing the optional columns
entirely.

``legacy_session_report`` is the PRE-CHANGE implementation, kept here verbatim as
the oracle. It re-implements only the statements this change replaced;
``_descendant_usage`` and ``_tool_call_stats`` are untouched by it and are called
on the store, so the comparison isolates the rewrite. The same oracle is what
``scripts/bench_panel_latency.py`` checks the REAL busiest sessions against on a
copy of the operator's ledger, so equivalence is asserted on synthetic edges here
and on production data there.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, replace
from typing import Any

import pytest

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    SessionReport,
    SessionRequest,
    TimingSummary,
)
from local_operator.analytics.store import AnalyticsStore, _aggregate_from_row
from tests.unit.analytics.test_store import _snap

CHILD = "aaaaaaaaaaaa"
GRANDCHILD = "bbbbbbbbbbbb"
#: A session whose rows name TWO different parents (the ``_PARENT_EDGE_SQL`` MAX
#: tie-break), and a self-parenting row beside it.
TWO_PARENTS = "cccccccccccc"
ROOT = "dddddddddddd"
SELF_PARENT = "eeeeeeeeeeee"

_FIELDS = (
    "request_id",
    "ts_ms",
    "provider",
    "model_id",
    "purpose",
    "outcome",
    "usage_reported",
    "context_tokens",
    "output_tokens",
    "duration_ms",
    "ttft_ms",
    "preparation_ms",
    "ok",
)


def legacy_session_report(
    store: AnalyticsStore, conn: sqlite3.Connection, session_id: str, *, recent_limit: int = 12
) -> SessionReport:
    """The pre-change ``session_report`` statements, as the equivalence oracle.

    Copied from the version this commit replaces, including its ``col()``
    fallbacks (an absent optional column reads as a constant), the ``-1``
    "no sample" filter on each timing column, and the exact field order of the
    recent-rows projection. Deliberately not tidied: its value is being the old
    behaviour, not a nicer statement of it.
    """
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(calls)")}
    if not {"session_id", "provider", "model_id", "ts_ms", "id"} <= columns:
        return SessionReport(session_id=session_id, available=False)

    def col(name: str, default: str = "0") -> str:
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
    aggregate = _aggregate_from_row(conn.execute(f"SELECT {measures}" + scope, params).fetchone())
    by_model = {
        (str(row[0]), str(row[1])): _aggregate_from_row(row[2:])
        for row in conn.execute(
            f"SELECT provider, model_id, {measures}" + scope + " GROUP BY provider, model_id",
            params,
        )
    }
    purpose = col("purpose", "'unknown'")
    outcome = col("outcome", "'unknown'")
    by_purpose = {
        str(row[0]): _aggregate_from_row(row[1:])
        for row in conn.execute(f"SELECT {purpose}, {measures}" + scope + " GROUP BY 1", params)
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
            f"SELECT COUNT({expression}), AVG({expression}), MIN({expression}), "
            f"MAX({expression})" + scope + f" AND {expression} >= 0",
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
    descendants, descendant_ids = store._descendant_usage(conn, session_id, columns, measures)
    tool_calls = store._tool_call_stats(conn, session_id)
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


def _rows_for_one_session(count: int = 240) -> list[Any]:
    """One session's rows spanning every dimension the report groups by.

    Deliberately uneven: three providers, two models each, several
    purpose/outcome pairs (one purpose under TWO outcomes, so the pair grouping
    cannot be reconstructed from either single grouping), a fifth of the calls
    with timing sentinels, a tenth with ``usage_reported`` false, and a couple of
    failures. Uneven counts are what catch a merge that silently averages or
    deduplicates instead of summing.
    """
    base = int(time.time() * 1000) - 6 * 60 * 60 * 1000
    providers = (("anthropic", "claude"), ("openai", "gpt"), ("google", "gemini"))
    purposes = ("turn", "tool", "compaction", "summary")
    outcomes = ("stop", "length")
    rows = []
    for index in range(count):
        provider, model = providers[index % 3]
        purpose = purposes[(index // 3) % len(purposes)]
        # ``turn`` appears under BOTH outcomes; every other purpose only under
        # the one it is paired with here.
        outcome = outcomes[index % 2] if purpose == "turn" else outcomes[0]
        rows.append(
            _snap(
                session_id=CHILD,
                ts_ms=base + index * 1000,
                provider=provider,
                model_id=model,
                input_tokens=100 + index,
                output_tokens=40 + (index % 7),
                ok=index % 23 != 0,
                cost_micro=1000 + index,
                cost_known=index % 5 != 0,
            )
        )
        rows[-1] = replace(
            rows[-1],
            purpose=purpose,
            outcome=outcome,
            usage_reported=index % 10 != 0,
            duration_ms=-1.0 if index % 5 == 0 else float(100 + index),
            ttft_ms=-1.0 if index % 4 == 0 else float(20 + index),
            preparation_ms=-1.0 if index % 7 == 0 else float(5 + index),
        )
    return rows


def _build_fixture(tmp_path) -> AnalyticsStore:
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch(_rows_for_one_session())
    store.record_batch(
        [
            # Two levels of descendants, so the walk has a frontier to descend.
            _snap(session_id=GRANDCHILD, ts_ms=int(time.time() * 1000), provider="openai"),
            _snap(session_id=ROOT, ts_ms=int(time.time() * 1000) - 5000),
        ]
    )
    # Parent edges: CHILD under ROOT, GRANDCHILD under CHILD, a session with two
    # distinct parents (MAX tie-break) and a self-parented row.
    store.record_batch(
        [
            replace(
                _snap(session_id=TWO_PARENTS, ts_ms=int(time.time() * 1000) - 4000),
                parent_session_id=ROOT,
            ),
            replace(
                _snap(session_id=TWO_PARENTS, ts_ms=int(time.time() * 1000) - 3999),
                parent_session_id=SELF_PARENT,
            ),
            replace(
                _snap(session_id=SELF_PARENT, ts_ms=int(time.time() * 1000) - 3000),
                parent_session_id=SELF_PARENT,
            ),
        ]
    )
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        conn.execute("UPDATE calls SET parent_session_id = ? WHERE session_id = ?", (ROOT, CHILD))
        conn.execute(
            "UPDATE calls SET parent_session_id = ? WHERE session_id = ?", (CHILD, GRANDCHILD)
        )
    return store


def _compare(store: AnalyticsStore, path, session_id: str, *, recent_limit: int = 12) -> None:
    fresh = store.session_report(session_id, recent_limit=recent_limit)
    with sqlite3.connect(path) as conn:
        legacy = legacy_session_report(store, conn, session_id, recent_limit=recent_limit)
    assert asdict(fresh) == asdict(legacy)


@pytest.mark.parametrize("session_id", [CHILD, ROOT, GRANDCHILD, TWO_PARENTS, SELF_PARENT, "nope"])
@pytest.mark.parametrize("recent_limit", [1, 12, 50, 0])
def test_the_merged_read_matches_the_frozen_statements(tmp_path, session_id, recent_limit):
    """Field-for-field equality, including the ``recent_limit`` clamp.

    ``0`` and ``50`` are the clamp's ends and ``1`` is the boundary the panel
    actually passes, so a merge that moved the clamp would fail here.
    """
    store = _build_fixture(tmp_path)
    _compare(store, tmp_path / "analytics.db", session_id, recent_limit=recent_limit)


def test_a_session_with_children_matches_including_the_subtree(tmp_path):
    """The descendant walk is part of the comparison, not beside it.

    ``by_model`` and the breakdowns are own-scope by design, so the subtree total
    has to agree separately — and it is the field whose parent edges come from
    the shared rule rather than from the merged statements.
    """
    store = _build_fixture(tmp_path)
    fresh = store.session_report(ROOT)
    assert fresh.descendant_ids, "the fixture must give the root a subtree to walk"
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        legacy = legacy_session_report(store, conn, ROOT)
    assert asdict(fresh) == asdict(legacy)


def test_an_older_ledger_without_the_optional_columns_matches(tmp_path):
    """The ``col()`` fallbacks, on a ledger that has none of the optional columns.

    Every merged statement uses a constant in place of an absent column, and the
    timing merge in particular turns a per-column ``WHERE x >= 0`` into a
    conditional aggregate. If that rewrite got the absent-column case wrong it
    would fabricate 0 ms timings or, worse, count a row under ``unknown`` that
    the old code counted under ``missing``.
    """
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE calls ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL,"
            " session_id TEXT NOT NULL, provider TEXT NOT NULL, model_id TEXT NOT NULL,"
            " ok INTEGER NOT NULL DEFAULT 1, input_tokens INTEGER NOT NULL DEFAULT 0,"
            " output_tokens INTEGER NOT NULL DEFAULT 0,"
            " cache_read_tokens INTEGER NOT NULL DEFAULT 0,"
            " cache_write_tokens INTEGER NOT NULL DEFAULT 0,"
            " reasoning_tokens INTEGER NOT NULL DEFAULT 0,"
            " context_tokens INTEGER NOT NULL DEFAULT 0)"
        )
        for index in range(20):
            conn.execute(
                "INSERT INTO calls (ts_ms, session_id, provider, model_id, ok, input_tokens) "
                "VALUES (?, ?, 'anthropic', 'claude', ?, ?)",
                (1_700_000_000_000 + index, CHILD, index % 4 != 0, 10 + index),
            )
        conn.commit()
    store = AnalyticsStore(path)
    fresh = store.session_report(CHILD)
    with sqlite3.connect(path) as conn:
        legacy = legacy_session_report(store, conn, CHILD)
    assert asdict(fresh) == asdict(legacy)
    assert fresh.available is True
    # The old behaviour, stated where a reader will see it: an absent column is
    # UNKNOWN, never a confident zero.
    assert fresh.timings["duration_ms"] == TimingSummary(0, None, None, None)
    assert fresh.missing_usage_calls == 0
    # Every call reads as UNKNOWN when the column is absent (``NULL IS NULL``
    # is true for every row), which is the pre-change behaviour and is kept:
    # an unrecorded flag is not a claim that usage was reported.
    assert fresh.unknown_usage_calls == 20


def test_the_merged_read_is_the_same_on_a_copy_of_the_real_ledger(tmp_path):
    """The busiest real session, when the operator's ledger is available.

    Skipped, not failed, on a machine without it: ``scripts/bench_panel_latency.py``
    runs this same oracle against the real busiest sessions and records the
    result in ``bench/analytics-rollup-*.json``, so production data is covered
    even where this test cannot see it.
    """
    import shutil
    from pathlib import Path

    live = Path.home() / ".local-operator" / "analytics.db"
    if not live.exists():
        pytest.skip("no operator ledger on this machine")
    copy = tmp_path / "analytics.db"
    shutil.copy2(live, copy)
    wal = live.with_name(live.name + "-wal")
    if wal.exists():
        shutil.copy2(wal, copy.with_name(copy.name + "-wal"))
    store = AnalyticsStore(copy)
    with sqlite3.connect(copy) as conn:
        busiest = [
            str(row[0])
            for row in conn.execute(
                "SELECT session_id, COUNT(*) AS n FROM calls WHERE session_id <> '' "
                "GROUP BY session_id ORDER BY n DESC LIMIT 3"
            )
        ]
    assert busiest, "the copied ledger has no session rows"
    for session_id in busiest:
        _compare(store, copy, session_id)
