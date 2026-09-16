"""The maintained day-grain rollup behind ``aggregate()``: equivalence, the gate,
the write path, the sweep and retention.

The whole change rests on ONE claim — for any window the gate accepts, the
rollup answers with exactly the ``UsageAggregate`` the raw ledger would — and
everything here exists to falsify that claim from a different direction:

- the fixture is the architect's list of shapes that break a naive rollup
  (multi-provider days, self and empty parents, a session with two distinct
  parents, an empty session id, a midnight-straddling turn, unpriced calls) and
  the assertion is ``dataclasses.asdict`` equality plus both side maps;
- the property test drives the parent rule through the real upsert, because the
  ``MAX(a, NULL)`` trap loses an edge only for row ORDERINGS a hand-written
  fixture will not have;
- the gate tests assert which path RAN (a spy and ``last_aggregate_source``),
  never how long it took — a latency bound here would be a bet on machine load,
  and the numbers belong in ``bench/`` (AGENTS.md §Timing).
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any

import pytest

from local_operator.analytics import store as store_module
from local_operator.analytics.backfill import backfill_analytics_session_daily
from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.store import (
    _PARENT_EDGE_SQL,
    AnalyticsStore,
    _combine_parent_edges,
    _day_shift,
    _local_day_bounds_ms,
    _local_day_month,
    _parent_edge_for,
)
from tests.unit.analytics.test_store import _snap

PARENT = "aaaaaaaaaaaa"
CHILD = "bbbbbbbbbbbb"
TWO_PARENTS = "cccccccccccc"
UNPRICED = "dddddddddddd"
NAMED = "eeeeeeeeeeee"
SELF = "ffffffffffff"
EMPTY = ""


def _at(day: str, hour: int, minute: int = 0) -> int:
    """Epoch-ms for a local wall-clock time on a local ``YYYY-MM-DD`` day."""
    moment = datetime.strptime(day, "%Y-%m-%d").replace(hour=hour, minute=minute)
    return int(moment.timestamp() * 1000)


def _today() -> str:
    return _local_day_month(int(time.time() * 1000))[0]


def _parents(agg) -> dict[str, str]:
    """The parent-edge side map, which ``asdict`` drops.

    Read through ``getattr`` because it is attached with ``setattr`` rather than
    declared on the dataclass (three other consumers read the same object, and
    widening it for one table's structure is not worth it — the store documents
    the split).
    """
    return dict(getattr(agg, "session_parents"))


def _call(
    *,
    session_id: str,
    ts_ms: int,
    parent: str = "",
    provider: str = "anthropic",
    model_id: str = "claude",
    ok: bool = True,
    known: bool = True,
    cost: int = 1000,
    tokens: int = 100,
):
    return replace(
        _snap(
            session_id=session_id,
            ts_ms=ts_ms,
            provider=provider,
            model_id=model_id,
            ok=ok,
            cost_micro=cost,
            cost_known=known,
            input_tokens=tokens,
        ),
        parent_session_id=parent,
    )


def _fixture_calls() -> list[CallSnapshot]:
    """Every shape the rollup has to survive, spread over three local days.

    Returned as one list so a test can write it in ONE batch or split it into
    several, which is the difference between exercising the upsert's accumulate
    path and its insert path.
    """
    today = _today()
    d0, d1, d2 = _day_shift(today, -2), _day_shift(today, -1), today
    midnight_d1 = _local_day_bounds_ms(d1)[0]
    return [
        # --- d0: a parent, its child, and the empty session id the real ledger
        # keys in ``by_session`` (224 rows today).
        _call(session_id=PARENT, ts_ms=_at(d0, 9)),
        _call(session_id=CHILD, ts_ms=_at(d0, 10), parent=PARENT, provider="openai"),
        _call(session_id=TWO_PARENTS, ts_ms=_at(d0, 11), parent=PARENT),
        _call(session_id=EMPTY, ts_ms=_at(d0, 12), tokens=7),
        # The turn that straddles local midnight: its last call is the last
        # millisecond of d0 and its next is the first of d1, so the two must land
        # in different buckets and the same window must see both.
        _call(session_id=PARENT, ts_ms=midnight_d1 - 1),
        # --- d1: a day with TWO providers for one session, a session whose two
        # rows name DIFFERENT parents (the documented MAX tie-break), a self
        # edge, an empty id again, and the named session.
        _call(session_id=PARENT, ts_ms=_at(d1, 8)),
        _call(session_id=PARENT, ts_ms=_at(d1, 9), provider="openai", model_id="gpt"),
        _call(session_id=TWO_PARENTS, ts_ms=_at(d1, 10), parent=NAMED),
        _call(session_id=EMPTY, ts_ms=_at(d1, 12), tokens=3),
        _call(session_id=SELF, ts_ms=_at(d1, 15), parent=SELF),
        _call(session_id=NAMED, ts_ms=_at(d1, 18), cost=2500),
        # --- d2 (today): a second provider for the child, an unpriced call, and
        # a FAILED call (``ok`` is summed per bucket, so it has to survive too).
        _call(session_id=PARENT, ts_ms=_at(d2, 1)),
        _call(session_id=NAMED, ts_ms=_at(d2, 2), cost=2500),
        _call(session_id=UNPRICED, ts_ms=_at(d2, 3), known=False, cost=0),
        _call(session_id=UNPRICED, ts_ms=_at(d2, 3, 30), known=False, cost=0, ok=False),
        _call(session_id=CHILD, ts_ms=_at(d2, 4), parent=PARENT, provider="openai"),
    ]


def _seeded_store(tmp_path, *, retention_days: int = 90, batches: int = 1) -> AnalyticsStore:
    """A store holding the fixture, with a session name, and NO sweep run yet."""
    store = AnalyticsStore(tmp_path / "analytics.db", retention_days=retention_days)
    calls = _fixture_calls()
    step = max(1, len(calls) // max(1, batches))
    for start in range(0, len(calls), step):
        store.record_batch(calls[start : start + step])
    store.upsert_session_name(NAMED, "The named one")
    return store


def _windows() -> list[tuple[int | None, int | None, str | None]]:
    """The windows both paths must agree on: None, 1/7/30-day, and scoped."""
    today = _today()
    start_1 = _local_day_bounds_ms(today)[0]
    start_7 = _local_day_bounds_ms(_day_shift(today, -6))[0]
    start_30 = _local_day_bounds_ms(_day_shift(today, -29))[0]
    end = _local_day_bounds_ms(_day_shift(today, 1))[0]
    return [
        (None, None, None),
        (start_1, end, None),
        (start_7, end, None),
        (start_30, end, None),
        (None, None, PARENT),
        (start_7, end, UNPRICED),
    ]


def _result_key(agg) -> dict[str, Any]:
    """Everything the two paths must agree on, flattened for comparison.

    ``asdict`` covers the dataclass tree (headline, per-provider, per-session);
    the two side maps are set as ATTRIBUTES and are dropped by ``asdict``, so
    they have to be compared deliberately — a gate that got the tree right and
    the names wrong would still look identical through the dataclass.
    """
    return {
        "asdict": asdict(agg),
        "by_session_keys": sorted(agg.by_session),
        "session_names": dict(getattr(agg, "session_names")),
        "session_parents": _parents(agg),
    }


def test_the_rollup_matches_the_ledger_for_every_real_window(tmp_path):
    """The central claim, over the fixture and every window shape."""
    store = _seeded_store(tmp_path, batches=3)
    before = {}
    for since, until, session in _windows():
        agg = store.aggregate(since_ms=since, until_ms=until, session_id=session)
        # Precondition of the test itself: the fast path is not merely unused,
        # it is REFUSED here (no day has been swept yet), so `before` really is
        # the raw-ledger answer rather than a rollup answer that happens to
        # match another rollup answer.
        assert store.last_aggregate_source == "ledger"
        before[(since, until, session)] = _result_key(agg)
    assert before[(None, None, None)]["asdict"]["calls"] == len(_fixture_calls())

    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1

    for since, until, session in _windows():
        agg = store.aggregate(since_ms=since, until_ms=until, session_id=session)
        assert store.last_aggregate_source == "rollup", (since, until, session)
        assert _result_key(agg) == before[(since, until, session)], (since, until, session)


def test_the_rollup_keeps_the_session_set_the_tui_forest_needs(tmp_path):
    """The forest drops an edge whose parent is absent, so the SET must match.

    ``build_session_forest`` re-parents a session whose parent is missing from
    the window and the table's column widths are maxima over every row, so a
    rollup that lost one session would change the geometry of the frame, not
    just a number.
    """
    store = _seeded_store(tmp_path)
    ledger_agg = store.aggregate()
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    rollup_agg = store.aggregate()
    assert set(rollup_agg.by_session) == set(ledger_agg.by_session)
    # The empty session id is NOT filtered out by either path — it is a key in
    # ``by_session`` today (the real ledger has 224 rows under it), and the
    # rollup's PRIMARY KEY carries it as the empty string it is.
    assert EMPTY in rollup_agg.by_session
    assert getattr(rollup_agg, "session_parents") == getattr(ledger_agg, "session_parents")


def test_two_distinct_parents_resolve_to_the_same_edge_on_both_paths(tmp_path):
    """The documented MAX tie-break, checked from both sides of the gate."""
    store = _seeded_store(tmp_path)
    # PARENT sorts below NAMED, and the two rows name them on different days, so
    # the window-level answer must be the lexical max across BUCKETS as well as
    # across rows. Read on each side of the gate rather than assuming which side
    # ran: the ledger answer must not move either.
    assert _parents(store.aggregate())[TWO_PARENTS] == NAMED
    assert store.last_aggregate_source == "ledger"
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    assert _parents(store.aggregate())[TWO_PARENTS] == NAMED
    assert store.last_aggregate_source == "rollup"


def test_a_null_combine_would_lose_a_real_parent_edge(tmp_path):
    """The trap this test exists for: SQLite's scalar ``MAX`` is NOT NULL-tolerant.

    ``MAX('aa', NULL)`` is NULL while the aggregate ``MAX`` ignores NULLs, so an
    upsert written as ``parent_session_id = MAX(parent_session_id, excluded...)``
    DROPS an edge whenever either side of the pair is NULL — and it loses it in
    whichever order the batches happen to land, which is exactly the shape a
    single hand-written fixture cannot catch. Both directions are asserted here,
    plus a live check that the trap still exists so this test fails loudly if a
    future SQLite changes the semantics it documents.
    """
    conn = sqlite3.connect(":memory:")
    assert conn.execute("SELECT MAX(?, NULL)", ("aa",)).fetchone()[0] is None
    assert conn.execute("SELECT MAX(NULL, ?)", ("aa",)).fetchone()[0] is None
    aggregated = conn.execute(
        "SELECT MAX(x) FROM (SELECT ? AS x UNION ALL SELECT NULL)", ("aa",)
    ).fetchone()[0]
    assert aggregated == "aa"

    for first_has_edge in (True, False):
        path = tmp_path / f"{first_has_edge}.db"
        store = AnalyticsStore(path)
        day = _today()
        edge = _call(session_id=CHILD, ts_ms=_at(day, 9), parent=PARENT)
        # Same bucket (same local day, session, provider), different parent: one
        # batch carries the edge and the other carries the ``''`` sentinel, so
        # the upsert's combine sees a NULL on one side whichever order they run.
        plain = _call(session_id=CHILD, ts_ms=_at(day, 10))
        store.record_batch([edge, plain] if first_has_edge else [plain, edge])
        store.close()
        with sqlite3.connect(path) as inspect:
            stored = inspect.execute(
                "SELECT MAX(parent_session_id) FROM session_daily WHERE session_id = ?", (CHILD,)
            ).fetchone()[0]
        assert stored == PARENT


def test_parent_rule_combines_the_same_way_the_ledger_aggregates(tmp_path):
    """Property test: ``combine(leaf(A), leaf(B)) == leaf(A ∪ B)``.

    Driven through the REAL upsert (two batches into one bucket) and scored
    against ``_PARENT_EDGE_SQL`` computed by SQL over the same rows, so it tests
    the mechanism and not a Python re-statement of it.
    """
    rng = random.Random(20260916)
    parents = ["", TWO_PARENTS, PARENT, NAMED, SELF]
    for trial in range(60):
        path = tmp_path / f"prop{trial}.db"
        store = AnalyticsStore(path)
        rows: list[tuple[str, str]] = []
        batch_a = []
        batch_b = []
        for index in range(rng.randint(1, 6)):
            session_id = rng.choice([PARENT, CHILD, rng.choice(parents) or PARENT])
            parent = rng.choice(parents)
            if parent == session_id:
                parent = session_id  # the self edge
            ts = _at(_today(), 9) + index * 1000
            rows.append((session_id, parent))
            call = _call(session_id=session_id, ts_ms=ts, parent=parent)
            (batch_a if index % 2 == 0 else batch_b).append(call)
        store.record_batch(batch_a)
        store.record_batch(batch_b)
        store.close()
        # Every row must actually be in the ledger: ``record_batch`` is
        # best-effort and returns 0 on a failed attempt, so a fixture value the
        # schema rejects (a NULL parent in a NOT NULL column) would silently
        # empty the property test instead of failing it.
        with sqlite3.connect(path) as conn:
            written = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        assert written == len(rows), (trial, written, rows)

        with sqlite3.connect(path) as conn:
            ledger = {
                str(sid): parent
                for sid, parent in conn.execute(
                    f"SELECT session_id, {_PARENT_EDGE_SQL} FROM calls GROUP BY session_id"
                )
            }
            rollup = {
                str(sid): parent
                for sid, parent in conn.execute(
                    "SELECT session_id, MAX(parent_session_id) FROM session_daily "
                    "GROUP BY session_id"
                )
            }
        # The two surfaces must agree per session, including on the sessions
        # whose only edge was dropped by a NULL combine.
        assert rollup == ledger, (trial, rows)
        # And the Python half of the same rule (the writer accumulates in
        # Python before the upsert) must fold the same way.
        for sid in {sid for sid, _ in rows}:
            folded = None
            for row_sid, parent in rows:
                if row_sid == sid:
                    folded = _combine_parent_edges(folded, _parent_edge_for(row_sid, parent))
            assert rollup.get(sid) == folded, (trial, sid, rows)


@pytest.mark.parametrize(
    "reason, perturb, window",
    [
        ("not-day-aligned", None, "unaligned"),
        ("no-coverage", "partial-sweep", "seven-day"),
        ("zone-changed", "zone", "today"),
        ("tail-unsynced", "stale-writer", "today"),
        ("no-rollup-table", "drop-table", "today"),
        ("empty-ledger", "no-rows", "all-time"),
        ("ledger-bottom-partial", "prune", "all-time"),
        ("no-parent-column", "drop-parent-column", "today"),
    ],
)
def test_the_gate_refuses_and_the_ledger_answers(tmp_path, monkeypatch, reason, perturb, window):
    """Every refusal, with both the reason AND the path asserted.

    A gate tested only on its happy path is a gate that silently stops gating,
    so each precondition gets a case that trips it, and each case asserts the
    LEDGER path ran — through a spy on the method, not through ``last_...``
    alone, because the attribute is written by the same code that decides.
    """
    store = _seeded_store(tmp_path, retention_days=2 if perturb == "prune" else 90)
    if perturb != "partial-sweep":
        assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    if perturb == "partial-sweep":
        # One day derived (today): coverage is today, so a 7-day window has a
        # hole below it and must be refused.
        assert backfill_analytics_session_daily(tmp_path, store=store, max_days=1) == 1

    since, until, session = {
        "today": (_local_day_bounds_ms(_today())[0], None, None),
        "seven-day": (_local_day_bounds_ms(_day_shift(_today(), -6))[0], None, None),
        "all-time": (None, None, None),
        "unaligned": (_local_day_bounds_ms(_today())[0] + 5000, None, None),
    }[window]

    if perturb == "zone":
        monkeypatch.setattr(store_module, "_local_zone_key", lambda: "Elsewhere/Nowhere")
    elif perturb == "stale-writer":
        # A ``lop`` on the pre-rollup binary: a row in the ledger the rollup
        # never saw, newer than everything the rollup holds.
        with sqlite3.connect(tmp_path / "analytics.db") as conn:
            conn.execute(
                "INSERT INTO calls (ts_ms, session_id, provider, model_id) VALUES (?, ?, ?, ?)",
                (int(time.time() * 1000) + 60_000, PARENT, "anthropic", "claude"),
            )
    elif perturb == "drop-table":
        with sqlite3.connect(tmp_path / "analytics.db") as conn:
            conn.execute("DROP TABLE session_daily")
        store = AnalyticsStore(tmp_path / "analytics.db")
    elif perturb == "no-rows":
        for path in tmp_path.iterdir():
            path.unlink()
        store = AnalyticsStore(tmp_path / "analytics.db")
    elif perturb == "prune":
        store.prune(now_ms=int(time.time() * 1000))
        store = AnalyticsStore(tmp_path / "analytics.db")
    elif perturb == "drop-parent-column":
        monkeypatch.setattr(store, "_has_parent_column", lambda: False)

    calls: list[str] = []
    real = store._ledger_aggregate

    def spy(conn, **kwargs):
        calls.append("ledger")
        return real(conn, **kwargs)

    monkeypatch.setattr(store, "_ledger_aggregate", spy)
    store.aggregate(since_ms=since, until_ms=until, session_id=session)
    assert calls == ["ledger"], f"{reason}: the ledger path did not run"
    assert store.last_aggregate_source == "ledger"


def test_the_gate_serves_the_panels_own_window(tmp_path, monkeypatch):
    """The positive case, asserted the same structural way.

    The desktop panel asks for ``sinceMs = local midnight, -(days-1)`` and
    ``untilMs = local midnight, +1``, which is exactly the window this builds.
    """
    store = _seeded_store(tmp_path)
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    since = _local_day_bounds_ms(_day_shift(_today(), -29))[0]
    until = _local_day_bounds_ms(_day_shift(_today(), 1))[0]
    calls: list[str] = []

    def spy(conn, **kwargs):
        calls.append("ledger")
        return store._ledger_aggregate(conn, **kwargs)

    monkeypatch.setattr(store, "_ledger_aggregate", spy)
    agg = store.aggregate(since_ms=since, until_ms=until)
    assert calls == []
    assert store.last_aggregate_source == "rollup"
    assert agg.calls == len(_fixture_calls())


def test_the_sweep_is_bounded_by_the_ledger_and_never_deletes_older_history(tmp_path):
    """The worklist's floor is the ledger's oldest day, not a constant.

    Going below it would delete rollup history that survived the prune, which is
    the one thing a rollup exists to keep. The floor is asserted directly, and
    then the surviving rows are asserted after a prune + sweep.
    """
    store = _seeded_store(tmp_path, retention_days=2)
    store.prune(now_ms=int(time.time() * 1000))
    store = AnalyticsStore(tmp_path / "analytics.db", retention_days=2)
    span = store.ledger_day_span()
    assert span is not None
    worklist = store.session_daily_worklist(max_days=90)
    assert worklist == sorted(worklist, reverse=True)
    assert all(day >= span[0] for day in worklist)


def test_a_pruned_bottom_is_recorded_and_the_gate_refuses_windows_reaching_it(tmp_path):
    """The prune cuts INSIDE the ledger's oldest day; the rollup holds it whole.

    Awaiting that difference would be an over-count, so the prune records where
    the ledger became whole again and the gate refuses anything below it — while
    the rollup's rows for those older days are left alone.
    """
    store = _seeded_store(tmp_path, retention_days=2)
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    store.prune(now_ms=int(time.time() * 1000))
    store = AnalyticsStore(tmp_path / "analytics.db", retention_days=2)
    whole_from = store.session_daily_state().get("ledger_whole_from_day")
    assert whole_from, "the prune did not record where the ledger became whole"

    span = store.ledger_day_span()
    assert span is not None and span[0] < whole_from
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        surviving = {str(row[0]) for row in conn.execute("SELECT DISTINCT day FROM session_daily")}
    assert span[0] in surviving, "the sweep deleted rollup history the prune had already dropped"

    # A window reaching the cut day is answered by the ledger; one starting at or
    # after it is served by the rollup.
    store.aggregate(since_ms=_local_day_bounds_ms(span[0])[0], until_ms=None)
    assert store.last_aggregate_source == "ledger"
    store.aggregate(since_ms=_local_day_bounds_ms(whole_from)[0], until_ms=None)
    assert store.last_aggregate_source == "rollup"


def test_the_sweep_stops_at_the_first_day_it_cannot_commit(tmp_path, monkeypatch):
    """Interruption leaves a monotone frontier rather than a claimed hole."""
    store = _seeded_store(tmp_path)
    days = store.session_daily_worklist(max_days=90)
    assert len(days) >= 3
    failed = days[1]
    real = store.rederive_session_daily_day

    def flaky(day):
        if day == failed:
            return None
        return real(day)

    monkeypatch.setattr(store, "rederive_session_daily_day", flaky)
    assert backfill_analytics_session_daily(tmp_path, store=store) == 1
    # The watermark stopped one day ABOVE the failure, so the hole is still a
    # refusal rather than a silently-served gap.
    assert store.session_daily_state()["covered_from_day"] == days[0]
    store.aggregate()
    assert store.last_aggregate_source == "ledger"


def test_the_sweep_records_the_bucketing_zone_and_refuses_another(tmp_path, monkeypatch):
    """The zone is what makes a changed zone a refusal instead of a wrong number.

    Two halves, both required: a sweep RECORDS the zone it labelled buckets in
    (a ledger being swept for the first time has no zone row yet, and without
    writing it the gate would have nothing to compare and would refuse the fast
    path forever), and it REFUSES to re-derive under a different one — otherwise
    the table would end up holding buckets labelled by two rules while the meta
    named one.
    """
    store = _seeded_store(tmp_path)
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        conn.execute("DELETE FROM session_daily_meta WHERE key = 'zone'")
    store = AnalyticsStore(tmp_path / "analytics.db")
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    recorded = store.session_daily_state()["zone"]
    assert recorded
    assert store.aggregate() is not None
    assert store.last_aggregate_source == "rollup"

    monkeypatch.setattr(store_module, "_local_zone_key", lambda: "Elsewhere/Nowhere")
    assert store.rederive_session_daily_day(_today()) is None
    assert store.session_daily_state()["zone"] == recorded
    store.aggregate()
    assert store.last_aggregate_source == "ledger"


def test_two_writers_on_one_bucket_lose_nothing(tmp_path):
    """Several ``lop`` processes write one file; the accumulate must be lossless.

    Each thread here owns its own ``AnalyticsStore``, which is what a second
    PROCESS looks like to SQLite: two connections, WAL, ``busy_timeout``, one
    transaction each. A read-modify-write would lose increments under this and
    the ledger and the rollup would disagree; the accumulate-upsert cannot.

    The file is created — and its schema written — before any thread starts,
    deliberately. Two threads racing to open a FRESH sqlite file is the one
    ordering this store forbids (it leaves a writer unable to see its own
    commits), and it is not what this test is about: in production the recorder
    exists before any maintenance write, and every process here opens a file
    that is already there.
    """
    path = tmp_path / "analytics.db"
    AnalyticsStore(path).close()
    day = _today()
    batches = 4
    size = 5
    errors: list[BaseException] = []

    def worker(offset: int) -> None:
        store = AnalyticsStore(path)
        try:
            for index in range(batches):
                store.record_batch(
                    [
                        _call(
                            session_id=PARENT, ts_ms=_at(day, 9) + offset * 60_000 + index * 100 + k
                        )
                        for k in range(size)
                    ]
                )
        except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
            errors.append(exc)
        finally:
            store.close()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []

    expected = 4 * batches * size
    store = AnalyticsStore(path)
    assert store.aggregate().calls == expected
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == expected
        assert conn.execute("SELECT SUM(calls) FROM session_daily").fetchone()[0] == expected
        # One bucket, not four: the accumulate merged every writer's rows.
        assert conn.execute("SELECT COUNT(*) FROM session_daily").fetchone()[0] == 1


def test_recording_survives_a_missing_rollup_table(tmp_path):
    """A ledger that lost the table keeps recording, and reads use the ledger."""
    _seeded_store(tmp_path)
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        conn.execute("DROP TABLE session_daily")
    reopened = AnalyticsStore(tmp_path / "analytics.db")
    assert reopened.record_batch([_call(session_id=PARENT, ts_ms=_at(_today(), 9))]) == 1
    agg = reopened.aggregate()
    assert agg.calls == len(_fixture_calls()) + 1
    assert reopened.last_aggregate_source == "ledger"


def test_the_write_path_adds_buckets_without_disturbing_the_ledger_or_the_charts(tmp_path):
    """One batch advances the ledger, both calendar rollups and the new rollup.

    The three rollups are written in ONE transaction, so a call is never
    half-recorded: the assertion is that after a batch the ledger, ``usage_daily``
    and ``session_daily`` all moved, and that the daily series (which the charts
    read) is unaffected by the new table.
    """
    store = AnalyticsStore(tmp_path / "analytics.db")
    calls = _fixture_calls()
    store.record_batch(calls)
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        ledger = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        daily = conn.execute("SELECT SUM(calls) FROM usage_daily").fetchone()[0]
        session_daily = conn.execute("SELECT SUM(calls) FROM session_daily").fetchone()[0]
    assert ledger == daily == session_daily == len(calls)
    series = store.daily_series(30)
    assert sum(row.calls for row in series) == len(calls)


def test_an_unrelated_write_error_rolls_the_rollup_back_with_the_ledger(tmp_path, monkeypatch):
    """A batch is atomic across all four tables, in both directions.

    The flip side of "one transaction": if the rollup write fails, the attempt
    rolls back and retries rather than committing a ledger row the rollup never
    saw — because a ledger row without its bucket would be invisible to a
    fast-path read until the next sweep.
    """
    store = AnalyticsStore(tmp_path / "analytics.db")
    monkeypatch.setattr(
        store_module,
        "_SESSION_DAILY_UPSERT_SQL",
        "INSERT INTO session_daily (nope) VALUES (?)",
    )
    assert store.record_batch([_call(session_id=PARENT, ts_ms=_at(_today(), 9))]) == 0
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0


def test_the_route_payload_serialises_identically_from_both_paths(tmp_path):
    """The desktop route builds ``asdict(aggregate)`` plus the daily series.

    Reproduced here byte-for-byte through ``json.dumps``, which is what the
    panel receives: the rollup's ``ORDER BY`` on both groupings is what makes
    the dict order — and therefore the bytes — the same on both paths.
    """
    store = _seeded_store(tmp_path)

    def payload() -> str:
        aggregate = store.aggregate()
        return json.dumps(
            {
                "aggregate": asdict(aggregate),
                "daily": [asdict(row) for row in store.daily_series(30)],
                "daily_scope": "all_sessions",
            }
        )

    before = payload()
    assert store.last_aggregate_source == "ledger"
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    after = payload()
    assert store.last_aggregate_source == "rollup"
    assert before == after
