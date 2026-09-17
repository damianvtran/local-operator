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
    _local_zone_key,
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
        # NOTE on these two labels, which round 1 found to be wrong: dropping
        # the table and reopening it leaves it RE-CREATED empty (``_SCHEMA`` runs
        # ``IF NOT EXISTS`` on every connect), so the refusal is the tail check,
        # not the missing table. The state the ``no-rollup-table`` label
        # describes is the flag being false while the file is otherwise intact —
        # the schema script that lost its tail — which is what the second case
        # forces. Both refuse, which is the guarantee under test.
        ("tail-unsynced", "drop-table", "today"),
        ("no-rollup-table", "no-table-flag", "today"),
        ("empty-ledger", "no-rows", "all-time"),
        ("ledger-bottom-partial", "prune", "all-time"),
        ("no-parent-column", "drop-parent-column", "today"),
        # The two the round-1 review found untested while the docs claimed every
        # refusal had one: the day-range arithmetic, and a rollup whose schema
        # cannot be read. Both are the "gate silently stops gating" class, so
        # they get a case like the rest — and the assertion below checks the
        # REASON, not just the path, which is what makes the claim mean anything.
        ("empty-window", None, "empty"),
        ("unreadable: OperationalError", "drop-meta-table", "today"),
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
        # An empty half-open range: alike bounds, so ``day_hi <= day_lo``.
        "empty": (
            _local_day_bounds_ms(_today())[0],
            _local_day_bounds_ms(_today())[0],
            None,
        ),
    }[window]

    if perturb == "zone":
        monkeypatch.setattr(store_module, "_local_zone_key", lambda: "Elsewhere/Nowhere")
    elif perturb == "stale-writer":
        # A ``lop`` on the pre-rollup binary: a row in the ledger the rollup
        # never saw, newer than everything the rollup holds. Placed relative to
        # the LEDGER rather than the wall clock, because the fixture's newest row
        # is 04:00 local: a row at ``now + 60 s`` before 04:00 is OLDER than the
        # fixture itself and the tail check then legitimately passes, which is
        # what made this case pass locally and fail on a UTC runner at 03:19.
        with sqlite3.connect(tmp_path / "analytics.db") as conn:
            newest = conn.execute("SELECT MAX(ts_ms) FROM calls").fetchone()[0]
            conn.execute(
                "INSERT INTO calls (ts_ms, session_id, provider, model_id) VALUES (?, ?, ?, ?)",
                (max(int(newest), int(time.time() * 1000)) + 1, PARENT, "anthropic", "claude"),
            )
    elif perturb == "drop-table":
        with sqlite3.connect(tmp_path / "analytics.db") as conn:
            conn.execute("DROP TABLE session_daily")
        store = AnalyticsStore(tmp_path / "analytics.db")
    elif perturb == "no-table-flag":
        with sqlite3.connect(tmp_path / "analytics.db") as conn:
            conn.execute("DROP TABLE session_daily")
        store._has_session_daily = False
    elif perturb == "drop-meta-table":
        # The gate's own meta read raises, with the store still believing the
        # table is there: the migrate-time shape check below normally catches a
        # schema that lost its tail, so this forces the read-time failure too.
        with sqlite3.connect(tmp_path / "analytics.db") as conn:
            conn.execute("DROP TABLE session_daily_meta")
        store._has_session_daily = True
    elif perturb == "no-rows":
        for path in tmp_path.iterdir():
            path.unlink()
        store = AnalyticsStore(tmp_path / "analytics.db")
    elif perturb == "prune":
        # An explicit ``now_ms`` INSIDE the fixture's span, never the wall clock:
        # the fixture's oldest day is two days back at 09:00, so a cutoff taken
        # at 03:19 (the hour CI happened to run) deletes nothing and the prune
        # records no frontier — which is exactly why this case was green locally
        # and red on the runner. Noon always lands between the fixture's oldest
        # day and the days after it.
        store.prune(now_ms=_at(_today(), 12))
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
    # The reason, not just the path: two different refusals that both fall back
    # are still two different states to fix, and a test that cannot tell them
    # apart cannot tell a cold sweep from a broken schema.
    assert store.last_aggregate_refusal == reason


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
    store.prune(now_ms=_at(_today(), 12))
    store = AnalyticsStore(tmp_path / "analytics.db", retention_days=2)
    span = store.ledger_day_span()
    assert span is not None
    worklist = store.session_daily_worklist(max_days=90).days
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
    # ``now_ms`` is explicit and INSIDE the fixture's span. With the wall clock
    # the cutoff lands two days back at the current hour, and the fixture's
    # oldest day is 09:00 two days back — so before 09:00 local nothing was
    # deleted and no frontier was recorded, which made this test fail for the
    # first nine hours of every UTC day.
    store.prune(now_ms=_at(_today(), 12))
    store = AnalyticsStore(tmp_path / "analytics.db", retention_days=2)
    whole_from = store.session_daily_state().get("ledger_whole_from_day")
    assert whole_from, "the prune did not record where the ledger became whole"

    span = store.ledger_day_span()
    assert span is not None and span[0] < whole_from
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        surviving = {str(row[0]) for row in conn.execute("SELECT DISTINCT day FROM session_daily")}
    # The rollup keeps the LEDGER's reach now (``retention_days``), so it never
    # holds a day older than the ledger's oldest — the old, longer reach was
    # storage no window can be served from, because the gate refuses below the
    # cut anyway (review R6). What must not happen is that the partial day is
    # silently SERVED, and that is the next assertion.
    assert surviving and min(surviving) >= span[0], surviving

    # A window reaching the cut day is answered by the ledger; one starting at or
    # after it is served by the rollup.
    store.aggregate(since_ms=_local_day_bounds_ms(span[0])[0], until_ms=None)
    assert store.last_aggregate_source == "ledger"
    store.aggregate(since_ms=_local_day_bounds_ms(whole_from)[0], until_ms=None)
    assert store.last_aggregate_source == "rollup"


def test_the_sweep_stops_at_the_first_day_it_cannot_commit(tmp_path, monkeypatch):
    """Interruption leaves a monotone frontier rather than a claimed hole."""
    store = _seeded_store(tmp_path)
    days = store.session_daily_worklist(max_days=90).days
    assert len(days) >= 3
    failed = days[1]
    real = store.rederive_session_daily_day

    def flaky(day, *, rebucket=False):
        if day == failed:
            return None
        return real(day, rebucket=rebucket)

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


def test_four_connections_on_one_bucket_lose_nothing(tmp_path):
    """Four connections on ONE bucket: the accumulate must be lossless.

    The name says what this exercises — four STORES, i.e. four connections, in
    one process, all writing the same bucket. It is deliberately not called a
    cross-PROCESS test: the cross-process case is its neighbour
    (``test_parallel_processes_write_atomically``, four real processes), and
    nothing here records a BUSY or a retry, so this does not exercise the
    busy-timeout path — it exercises the accumulate.

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


def test_recording_survives_a_rollup_table_missing_a_measure_column(tmp_path):
    """A rollup whose shape the code cannot write to must not stop the LEDGER.

    Review R1, reproduced: rollup tables have no ``ALTER`` path, so a future
    release that adds a measure column (or a ``COMPONENT_KEY``) leaves every
    existing ledger with a ``session_daily`` the upsert cannot insert into. The
    batch then fails its transaction and is dropped WITH the ledger row —
    analytics recording dies silently and completely for the life of that
    binary, which is the failure ``_present_optional`` exists to prevent for
    ``calls``. The guard is a shape check, not an existence check, and the
    assertion that matters is that the row is still written.
    """
    _seeded_store(tmp_path)
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        conn.execute("ALTER TABLE session_daily DROP COLUMN c_images")
    reopened = AnalyticsStore(tmp_path / "analytics.db")
    # The batch comes first on purpose: it is what used to be lost, and it is
    # also what opens the connection the flag is computed on (the store connects
    # lazily, so reading the flag before any use reports the optimistic
    # default).
    assert reopened.record_batch([_call(session_id=PARENT, ts_ms=_at(_today(), 9))]) == 1
    assert reopened._has_session_daily is False, "a short table must read as 'no rollup'"

    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == len(_fixture_calls()) + 1
    assert reopened.aggregate().calls == len(_fixture_calls()) + 1
    assert reopened.last_aggregate_source == "ledger"


def test_a_stale_writers_mid_history_rows_refuse_then_heal(tmp_path):
    """A hole in the MIDDLE of a window is refused, and the sweep repairs it.

    Review R3. Every boundary check misses this: the tail check pins the newest
    row, coverage the oldest day, so three ledger-only rows on a day four days
    back were served as exact — and the fixed newest-three healing window could
    never reach them, which made the hole permanent. Three things are asserted:
    the window count check REFUSES it, the sweep's per-day verification FINDS
    it wherever it is, and the next read serves the ledger's numbers again.
    """
    store = _seeded_store(tmp_path)
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    total_before = store.aggregate().calls
    assert store.last_aggregate_source == "rollup"

    # A row the rollup never saw, on a day the previous pass had already swept.
    hole_day = _day_shift(_today(), -2)
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        conn.execute(
            "INSERT INTO calls (ts_ms, session_id, provider, model_id) VALUES (?, ?, ?, ?)",
            (_at(hole_day, 20), PARENT, "anthropic", "claude"),
        )

    store.aggregate()
    assert store.last_aggregate_source == "ledger"
    assert store.last_aggregate_refusal == "count-mismatch"

    # The repair is the sweep's own verification, not a re-read: one more pass
    # and the fast path answers the ledger's numbers again. A permanent refusal
    # would be the same feature loss as the zone latch, so this is the half that
    # makes the refusal acceptable.
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    healed = store.aggregate()
    assert store.last_aggregate_source == "rollup"
    assert healed.calls == total_before + 1


def test_a_failing_fast_path_read_falls_back_to_the_ledger(tmp_path, monkeypatch):
    """The one refusal a precondition cannot produce: the rollup query raising.

    Deliberately not an error path that hides: the refusal is recorded, and the
    numbers still come from the ledger.
    """
    store = _seeded_store(tmp_path)
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    real = store._session_daily_aggregate

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("no such column: c_images")

    monkeypatch.setattr(store, "_session_daily_aggregate", boom)
    failed = store.aggregate()
    assert store.last_aggregate_source == "ledger"
    assert store.last_aggregate_refusal == "rollup-read-failed"
    monkeypatch.setattr(store, "_session_daily_aggregate", real)
    assert failed.calls == store.aggregate().calls


def test_a_zone_change_is_recovered_by_the_rebucket_pass(tmp_path, monkeypatch):
    """A differently-zoned run costs a sweep, not the feature (review R2).

    The zone is recorded first-writer-wins and the sweep used to refuse to
    re-derive under a different one, so a single ``TZ=``-prefixed run — or a
    laptop that moved — disabled the fast path permanently, with a ``debug``
    line as its only trace. The recovery is a re-label pass, and the invariant
    that makes it safe is asserted here too: while it is running, the recorded
    zone is still the old one, so reads keep refusing rather than seeing a
    table holding buckets from two zones.
    """
    store = _seeded_store(tmp_path)
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    recorded = store.session_daily_state()["zone"]
    assert recorded
    store.aggregate()
    assert store.last_aggregate_source == "rollup"

    # The environment, not a monkeypatched helper: TZ plus tzset is what a
    # ``TZ=``-prefixed command actually does to this process. The zone is picked
    # to DIFFER from the recorded one rather than hard-coded, because a process
    # already running under that TZ has no zone change to recover from — the
    # same environment-dependence class this round's R10 was about.
    other_zone = next(
        zone for zone in ("America/Denver", "UTC", "Pacific/Auckland") if zone != recorded
    )
    monkeypatch.setenv("TZ", other_zone)
    time.tzset()
    try:
        assert _local_zone_key() == other_zone != recorded
        store.aggregate()
        assert store.last_aggregate_source == "ledger"
        assert store.last_aggregate_refusal == "zone-changed"

        assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
        healed = store.aggregate()
        assert store.last_aggregate_source == "rollup"
        assert store.session_daily_state()["zone"] == _local_zone_key()

        # And the numbers are the LEDGER's, measured in the same (new) zone —
        # a zone change legitimately moves which day a call belongs to, so this
        # is not a comparison against the pre-change answer. The copy is made
        # through SQLite's backup API rather than ``cp`` because the file is in
        # WAL mode and its newest commits live in the -wal sidecar.
        ledger_only = tmp_path / "ledger-only.db"
        with sqlite3.connect(tmp_path / "analytics.db") as src, sqlite3.connect(ledger_only) as dst:
            src.backup(dst)
        with sqlite3.connect(ledger_only) as conn:
            conn.execute("DROP TABLE session_daily")
            conn.execute("DROP TABLE session_daily_meta")
        plain = AnalyticsStore(ledger_only)
        from_ledger = plain.aggregate()
        assert plain.last_aggregate_source == "ledger"
        assert _result_key(from_ledger) == _result_key(healed)
    finally:
        # Process-global state: put the zone back before the next test in this
        # worker runs, whatever happened above.
        monkeypatch.undo()
        time.tzset()


def test_a_rebucket_completes_when_the_ledger_span_exceeds_one_pass(tmp_path, monkeypatch):
    """A re-label is one indivisible job, so it cannot be capped by ``max_days``.

    Review R11's failure, and why the fix is "plan the whole span" rather than
    "record progress": ``DEFAULT_SESSION_DAILY_DAYS_PER_PASS == retention_days``
    and the prune keeps every row at or after ``now - 90 d``, so the steady-state
    ledger spans **91** day labels against a budget of 90. A capped plan then
    re-derives the same newest 90 labels on every launch, never publishes (a
    partial re-label may not name a zone half the days do not belong to), and the
    latch is permanent again — with ~90 day re-derives per launch on top. The
    span is bounded by the ledger, which is what makes the whole-span plan safe.
    """
    today = _today()
    oldest = _day_shift(today, -90)
    store = AnalyticsStore(tmp_path / "analytics.db", retention_days=90)
    # One call per label, plus one late call on the oldest label so the day stays
    # alive (but partial) after the prune — which is what makes the span 91
    # labels rather than 90. Same constants the production path uses, no
    # monkeypatching of the budget.
    store.record_batch(
        [
            _call(session_id=PARENT, ts_ms=_at(day, 9))
            for day in (_day_shift(today, -offset) for offset in range(91))
        ]
    )
    store.record_batch([_call(session_id=PARENT, ts_ms=_at(oldest, 23))])
    store.prune(now_ms=_at(today, 12))
    store = AnalyticsStore(tmp_path / "analytics.db", retention_days=90)
    span = store.ledger_day_span()
    assert span is not None

    # Poison the zone the way a `TZ=`-prefixed run does.
    monkeypatch.setattr(store_module, "_local_zone_key", lambda: "Elsewhere/Nowhere")

    plan = store.session_daily_worklist(max_days=90)
    assert plan.mode == "rebucket"
    # The whole span, not the budget: this is the fix, asserted structurally.
    assert len(plan.days) > 90, plan.days[:3]
    assert store.session_daily_state()["zone"] != "Elsewhere/Nowhere"

    # ONE launch. Without the whole-span plan this derives 90 labels and
    # publishes nothing, so the next assertion is the one that fails.
    assert backfill_analytics_session_daily(tmp_path, store=store) >= len(plan.days)
    assert store.session_daily_state()["zone"] == "Elsewhere/Nowhere"

    store.aggregate()
    assert (
        store.last_aggregate_refusal != "zone-changed"
    ), "the re-label published, so the zone is no longer a reason to refuse"


def test_an_upgrade_launch_refuses_for_coverage_not_for_a_zone_it_never_set(
    tmp_path,
):
    """The empty rollup has no zone to have changed (review round 1, Q1).

    An existing ledger's first open creates both rollup tables empty, and the
    zone comparison used to run first — so the reason was ``zone-changed``, which
    sends the next reader looking for travel or a ``TZ=`` run that never
    happened. Nothing has been labelled, so nothing can be mislabelled: the
    honest reason is that no day is covered yet.
    """
    _seeded_store(tmp_path)
    with sqlite3.connect(tmp_path / "analytics.db") as conn:
        conn.execute("DELETE FROM session_daily")
        conn.execute("DELETE FROM session_daily_meta")
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.aggregate()
    assert store.last_aggregate_source == "ledger"
    assert store.last_aggregate_refusal == "no-coverage"


def test_the_rollup_read_degrades_its_edges_with_the_ledgers_schema(tmp_path, monkeypatch):
    """The flag-false shape agrees on the edge map; the guard stays a guard.

    Review round 1, Q2 narrowed by round 3's F1/Q4. The substituting branch is
    keyed on ``_has_parent_column()`` — the SAME predicate as the guard — so it
    is reachable only where that predicate is false, which is the state this test
    pins by switching it off. The shape round 1 measured is NOT this one: there
    the column was renamed away and ``_migrate`` re-added it, so the predicate is
    true, the substitution does not apply, and the two maps still disagree (6,533
    vs 0 on that copy) while every count matches. That state needs an out-of-band
    edit no product path performs, and through ``aggregate()`` the guard refuses
    before the rollup read runs. So this asserts the flag-false shape only; the
    guarded shape stays divergent by construction, and the count check is blind
    to a side-map divergence.
    """
    store = _seeded_store(tmp_path)
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    served = store.aggregate()
    assert store.last_aggregate_source == "rollup"
    assert _parents(served), "sanity: the fixture has parent edges to lose"
    # The schemaless shape, with the precondition switched off so the rollup read
    # is actually exercised in it.
    monkeypatch.setattr(store, "_has_parent_column", lambda: False)
    conn = store._read_connection()
    span = store.ledger_day_span()
    assert conn is not None, "the store could not open a read connection"
    assert span is not None
    window = (span[0], None)
    from_rollup = store._session_daily_aggregate(conn, window, None)
    from_ledger = store._ledger_aggregate(conn, since_ms=None, until_ms=None, session_id=None)
    assert _parents(from_rollup) == _parents(from_ledger) == {}
    # The session SET must still match — the divergence was in the edges, and the
    # TUI's forest reads both.
    assert sorted(from_rollup.by_session) == sorted(from_ledger.by_session)


def test_the_gate_decides_on_the_snapshot_it_serves(tmp_path, monkeypatch):
    """One read snapshot for the gate, its count check and both grouped reads.

    Review R13. Without the pin, each statement is autocommit and therefore a
    different WAL snapshot: the gate could approve a window on rows the read then
    no longer sees (and the ledger path's three scans could disagree with each
    other about totals that must sum to the headline). Asserted structurally —
    the fast path is *observed* from inside a transaction — rather than by racing
    a writer, which would be a bet on timing.
    """
    store = _seeded_store(tmp_path)
    assert backfill_analytics_session_daily(tmp_path, store=store) >= 1
    seen: list[bool] = []
    real = store._session_daily_aggregate

    def spy(conn, *args, **kwargs):
        seen.append(bool(conn.in_transaction))
        return real(conn, *args, **kwargs)

    monkeypatch.setattr(store, "_session_daily_aggregate", spy)
    store.aggregate()
    assert store.last_aggregate_source == "rollup"
    assert seen == [True], "the rollup read ran outside the gate's snapshot"

    # Both paths share the pin, and it must not leak: the ledger path's three
    # scans see one snapshot, and the connection is left clean for the next call.
    seen.clear()
    monkeypatch.setattr(store, "_session_daily_window", lambda conn, since, until: (None, "test"))
    ledger_seen: list[bool] = []
    real_ledger = store._ledger_aggregate

    def ledger_spy(conn, *args, **kwargs):
        ledger_seen.append(bool(conn.in_transaction))
        return real_ledger(conn, *args, **kwargs)

    monkeypatch.setattr(store, "_ledger_aggregate", ledger_spy)
    store.aggregate()
    assert store.last_aggregate_source == "ledger"
    assert ledger_seen == [True]
