"""Panel read latency: the real ``aggregate()`` call the desktop route makes.

The operator's complaint is that ``/analytics`` takes seconds to open on a
production-sized ledger. This script measures the read path the panel actually
takes — ``AnalyticsStore.aggregate(since_ms=…, until_ms=…)`` on a COPY of the
live ledger, exactly as ``/v1/desktop/analytics`` calls it — plus the write path
it adds to (``record_batch``) and the payload the route then serialises.

Run against the ledger the app owns, which is copied first and never written::

    PYTHONPATH=. .venv/bin/python scripts/bench_panel_latency.py \
        --ledger ~/.local-operator/analytics.db --label after --json out.json

It is written to run UNCHANGED on both sides of the change, because the honest
"before" is the parent tree's own code rather than a flag that skips the new
table: on a tree without the rollup the rollup arms simply do not exist, and the
JSON records ``path: ""`` where the after tree records ``rollup``/``ledger``.

What the numbers mean, and what they do not:

- **CPU is the portable term.** ``time.thread_time`` is per-thread and excludes
  time asleep or waiting on the GIL, so it does not inflate on a host carrying
  other worktrees. This host runs at load 100-250 on 14 cores (sibling agents),
  so the WALL column is inflated and is reported next to CPU rather than instead
  of it. Never quote a wall figure from here as the panel's latency on an idle
  machine.
- **"Cold" is a fresh copy, not an evicted cache.** Evicting the unified buffer
  cache on macOS needs ``sudo purge``, which is not available non-interactively.
  A freshly copied file is the closest honest stand-in, and the byte counts in
  the JSON (table and index sizes) are the portable part of the claim.
- **Latency lives here, never in a test.** Tests assert WHICH path ran; this
  script is where a duration is a measurement rather than a bet on machine load
  (AGENTS.md §Timing).

Usage::

    python scripts/bench_panel_latency.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Clear every multiplexer identifier before any application import: a headless
# run must not be able to touch the operator's real cmux workspaces through an
# inherited id, and these variables are read by the product rather than only by
# a terminal.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import local_operator  # noqa: E402
from local_operator.analytics.model import CallSnapshot  # noqa: E402
from local_operator.analytics.store import AnalyticsStore  # noqa: E402

#: The panel's own window, in days: ``analytics-model.ts`` sends
#: ``sinceMs = localMidnight(now, -(days-1))`` and ``untilMs =
#: localMidnight(now, +1)`` for the default view. Both are day-aligned, which is
#: exactly what the rollup's gate needs, so this is the arm that matters.
PANEL_DAYS = 30


def _ms(value: float) -> float:
    return round(value * 1000.0, 2)


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50": _ms(statistics.median(ordered)),
        "p90": _ms(ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]),
        "min": _ms(ordered[0]),
        "max": _ms(ordered[-1]),
        "n": len(ordered),
    }


def _timed(call, *, samples: int) -> tuple[dict[str, float], dict[str, float], Any]:
    """Run ``call`` ``samples`` times, returning (wall, cpu, last result)."""
    wall: list[float] = []
    cpu: list[float] = []
    result: Any = None
    for _ in range(samples):
        wall_start = time.perf_counter()
        cpu_start = time.thread_time()
        result = call()
        cpu.append(time.thread_time() - cpu_start)
        wall.append(time.perf_counter() - wall_start)
    return _stats(wall), _stats(cpu), result


def _copy_ledger(source: Path, target: Path) -> None:
    """Copy the ledger and its WAL, so the copy is the LIVE state.

    Copying only the ``.db`` file would silently drop whatever is still in the
    write-ahead log — on a machine with running sessions that is the newest
    turns, i.e. exactly the rows this change is about. SQLite recovers the
    copied WAL on first open, which is why the ``-shm`` file is deliberately NOT
    copied: it is a rebuildable index of that WAL, and a stale one copied from a
    file still being written is the one way to make the copy inconsistent.
    """
    shutil.copy2(source, target)
    wal = source.with_name(source.name + "-wal")
    if wal.exists():
        shutil.copy2(wal, target.with_name(target.name + "-wal"))
    stale = target.with_name(target.name + "-shm")
    if stale.exists():
        stale.unlink()


def _ledger_shape(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as conn:
        try:
            # dbstat is a compile-time option; it is present in the Python
            # sqlite3 shipped here, but a build without it must cost the byte
            # table rather than the whole run.
            tables = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC"
                )
            }
        except sqlite3.Error:
            tables = {}
        calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM calls").fetchone()[0]
        span = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM calls").fetchone()
        has_rollup = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_daily'"
            ).fetchone()
            is not None
        )
        rollup_rows = 0
        if has_rollup:
            rollup_rows = conn.execute("SELECT COUNT(*) FROM session_daily").fetchone()[0]
    return {
        "file_bytes": path.stat().st_size,
        "calls": calls,
        "sessions": sessions,
        "span_ms": [span[0], span[1]],
        "table_bytes": tables,
        "has_session_daily": has_rollup,
        "session_daily_rows": rollup_rows,
    }


def _local_midnight_ms(day: str) -> int:
    """Epoch-ms of the first instant of a local ``YYYY-MM-DD`` day.

    Spelled here rather than imported from ``store._local_day_bounds_ms``
    because this script has to run UNCHANGED on both sides of the change and
    that helper only exists on the after side. It is the same arithmetic, and it
    is what the panel's own ``localMidnight`` mirrors: a UTC-aligned midnight
    would quietly measure the ledger fallback instead of the fast path on any
    machine east or west of UTC, which is exactly the mistake this script exists
    to make visible.
    """
    return int(datetime.strptime(day, "%Y-%m-%d").timestamp() * 1000)


def _local_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d")


def _shift(day: str, days: int) -> str:
    return (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def _windows(path: Path) -> dict[str, tuple[int | None, int | None]]:
    """The three windows the two real callers ask for, in ms.

    ``all_time`` is the TUI (``store.aggregate()`` with no bounds), and the two
    day-aligned windows are the desktop panel. The newest row rather than the
    wall clock anchors them, so a ledger copied weeks ago still exercises real
    windows.
    """
    with sqlite3.connect(path) as conn:
        newest = conn.execute("SELECT MAX(ts_ms) FROM calls").fetchone()[0]
    today = _local_day(int(newest if newest is not None else time.time() * 1000))
    return {
        "all_time": (None, None),
        f"panel_{PANEL_DAYS}d": (
            _local_midnight_ms(_shift(today, -(PANEL_DAYS - 1))),
            _local_midnight_ms(_shift(today, 1)),
        ),
        "last_7d": (
            _local_midnight_ms(_shift(today, -6)),
            _local_midnight_ms(_shift(today, 1)),
        ),
    }


def _read_arms(ledger: Path, *, samples: int) -> dict[str, Any]:
    store = AnalyticsStore(ledger)
    out: dict[str, Any] = {}
    for name, (since, until) in _windows(ledger).items():

        def call(since=since, until=until) -> Any:
            return store.aggregate(since_ms=since, until_ms=until)

        wall, cpu, agg = _timed(call, samples=samples)
        out[name] = {
            "wall_ms": wall,
            "cpu_ms": cpu,
            "path": getattr(store, "last_aggregate_source", ""),
            "calls": agg.calls,
            "by_session": len(agg.by_session),
        }
    store.close()
    return out


def _session_report_arms(ledger: Path, *, samples: int, sessions: int = 3) -> dict[str, Any]:
    """``session_report`` for the BUSIEST real sessions — the ``/session`` panel.

    The operator-visible complaint surface for this arm is ``sessions.report``
    (``/v1/desktop/sessions/{id}/report``), whose op is
    ``AnalyticsStore.session_report``. The cost is entirely this read: the app
    adds tens of milliseconds around it, so timing the store call on the real
    largest sessions is timing the panel.

    The sessions are chosen by ROW COUNT, from the ledger itself, so the arm
    measures the worst case that exists rather than a convenient one.
    """
    with sqlite3.connect(ledger) as conn:
        busiest = [
            (str(row[0]), int(row[1]))
            for row in conn.execute(
                "SELECT session_id, COUNT(*) AS n FROM calls WHERE session_id <> '' "
                "GROUP BY session_id ORDER BY n DESC LIMIT ?",
                (sessions,),
            )
        ]
    store = AnalyticsStore(ledger)
    out: dict[str, Any] = {}
    for session_id, calls in busiest:
        # The FIRST call is recorded on its own: on a ledger copy its pages are
        # not in cache yet, and that is the state a cold morning's panel opens
        # in. The warm samples that follow are the steady state, and the two are
        # reported separately rather than averaged into one number.
        start = time.perf_counter()
        first = store.session_report(session_id)
        first_wall = time.perf_counter() - start
        wall, cpu, report = _timed(
            lambda sid=session_id: store.session_report(sid), samples=samples
        )
        out[session_id] = {
            "calls": calls,
            "first_wall_ms": _ms(first_wall),
            "wall_ms": wall,
            "cpu_ms": cpu,
            "by_model": len(first.by_model),
            "descendants": len(first.descendant_ids),
            "recent": len(report.recent),
        }
    store.close()
    return out


def _session_report_equivalence(ledger: Path, sessions: list[str]) -> dict[str, Any]:
    """Compare the merged ``session_report`` against the FROZEN old statements.

    The same oracle the unit test uses (``legacy_session_report``), run here on
    the operator's OWN ledger — the unit suite cannot see it, because it isolates
    ``HOME``. So the equivalence claim is asserted on synthetic edges in the test
    and on production data in this JSON, which is what makes it evidence rather
    than an argument.
    """
    try:
        from tests.unit.analytics.test_session_report_equivalence import (
            legacy_session_report,
        )
    except ImportError:
        # The parent tree has no oracle to compare against: the frozen statements
        # ARE the code there. Recorded rather than silent, so a reader of the
        # before JSON can see the check was not applicable instead of absent.
        return {"oracle": "unavailable on this tree"}

    store = AnalyticsStore(ledger)
    out: dict[str, Any] = {}
    try:
        for session_id in sessions:
            fresh = store.session_report(session_id)
            with sqlite3.connect(ledger) as conn:
                legacy = legacy_session_report(store, conn, session_id)
            fresh_fields = asdict(fresh)
            legacy_fields = asdict(legacy)
            differing = sorted(
                key
                for key in set(fresh_fields) | set(legacy_fields)
                if fresh_fields.get(key) != legacy_fields.get(key)
            )
            out[session_id] = {
                "equal": not differing,
                "differing_fields": differing,
                "fields": len(fresh_fields),
            }
    finally:
        store.close()
    return out


def _first_touch(ledger: Path) -> dict[str, Any]:
    """The FIRST ``aggregate()`` on a file whose pages are not in cache yet.

    This is the number that decides how the panel feels on a cold morning, and
    it is the one a warm A/B cannot see: a fresh copy faults in hundreds of MB
    of table and index, which is why bytes-touched is the term the design
    argues from.
    """
    wall_start = time.perf_counter()
    cpu_start = time.thread_time()
    store = AnalyticsStore(ledger)
    since, until = _windows(ledger)["all_time"]
    agg = store.aggregate(since_ms=since, until_ms=until)
    wall = time.perf_counter() - wall_start
    cpu = time.thread_time() - cpu_start
    try:
        store.close()
    except Exception:  # noqa: BLE001 — a benchmark must not fail on teardown
        pass
    return {
        "wall_ms": _ms(wall),
        "cpu_ms": _ms(cpu),
        "path": getattr(store, "last_aggregate_source", ""),
        "calls": agg.calls,
    }


def _prep_rollup(ledger: Path) -> dict[str, Any] | None:
    """Re-derive the day rollup on the copy, the way the first launch does.

    Absent on a tree that has no rollup at all, which is how one script serves
    both sides of the change.
    """
    store = AnalyticsStore(ledger)
    if not hasattr(store, "session_daily_worklist"):
        store.close()
        return None
    days = store.session_daily_worklist(max_days=3650)
    per_day: list[float] = []
    wall_start = time.perf_counter()
    cpu_start = time.thread_time()
    for day in days:
        day_start = time.perf_counter()
        store.rederive_session_daily_day(day)
        per_day.append(time.perf_counter() - day_start)
    wall = time.perf_counter() - wall_start
    cpu = time.thread_time() - cpu_start
    store.close()
    reader = AnalyticsStore(ledger)
    state = dict(reader.session_daily_state())
    reader.close()
    shape = _ledger_shape(ledger)
    return {
        "days": len(days),
        "wall_ms": _ms(wall),
        "cpu_ms": _ms(cpu),
        "per_day_wall_ms": _stats(per_day) if per_day else {},
        "ledger": {
            "file_bytes": shape["file_bytes"],
            "session_daily_rows": shape["session_daily_rows"],
        },
        "meta": state,
    }


def _route_payload(ledger: Path, *, samples: int) -> dict[str, Any]:
    """``asdict`` + ``json.dumps`` of the route's response, in bytes and ms.

    Part of the read path proper: the panel cannot render until this has been
    serialised, and ``by_session`` is the overwhelming majority of it.
    """
    store = AnalyticsStore(ledger)
    since, until = _windows(ledger)[f"panel_{PANEL_DAYS}d"]
    aggregate = store.aggregate(since_ms=since, until_ms=until)
    daily = store.daily_series(PANEL_DAYS)

    def call() -> str:
        return json.dumps(
            {
                "aggregate": asdict(aggregate),
                "daily": [asdict(row) for row in daily],
                "daily_scope": "all_sessions",
            }
        )

    wall, cpu, text = _timed(call, samples=samples)
    path = getattr(store, "last_aggregate_source", "")
    store.close()
    return {
        "wall_ms": wall,
        "cpu_ms": cpu,
        "bytes": len(text.encode()),
        "path": path,
        "by_session_entries": len(aggregate.by_session),
    }


def _snapshots(count: int, *, session_ids: list[str], base_ts: int) -> list[CallSnapshot]:
    """Realistic call snapshots for the write arm, in a plausible mix.

    A batch is what the recorder's queue actually drains — a handful of calls
    from parallel sessions — so the sizes tested here are 1, 5 and 20 rather than
    a synthetic thousand.
    """
    out: list[CallSnapshot] = []
    for index in range(count):
        out.append(
            CallSnapshot(
                ts_ms=base_ts + index,
                session_id=session_ids[index % len(session_ids)],
                provider="anthropic" if index % 2 else "openai",
                model_id="claude" if index % 2 else "gpt",
                input_tokens=4_000,
                output_tokens=600,
                cache_read_tokens=2_400,
                cache_write_tokens=120,
                reasoning_tokens=80,
                context_tokens=7_200,
                component_chars={"conversation": 9_000, "system_prompt": 5_000},
                ok=True,
                cost_micro=12_345,
                cost_known=True,
                priced=True,
                parent_session_id=session_ids[0] if index % 5 == 0 else "",
            )
        )
    return out


def _write_arms(ledger: Path, *, samples: int) -> dict[str, Any]:
    """``record_batch`` cost, with and without the new rollup upsert.

    On the after tree the two arms run INTERLEAVED inside one interpreter, so
    host load drifts across both rather than favouring whichever ran second: the
    control arm sets ``_has_session_daily = False``, which is precisely the
    transaction the parent tree runs — the same code minus the third upsert — and
    the difference is therefore the write amplification and nothing else.

    On the parent tree there is no second arm to run, so it reports the plain
    ``record_batch`` cost, which is the absolute figure the after arm's control
    is checked against.
    """
    store = AnalyticsStore(ledger)
    has_rollup = hasattr(store, "_has_session_daily")
    with sqlite3.connect(ledger) as conn:
        session_ids = [
            str(row[0])
            for row in conn.execute(
                "SELECT session_id FROM calls GROUP BY session_id ORDER BY COUNT(*) DESC LIMIT 8"
            )
        ]
    if not session_ids:
        session_ids = ["benchmark0001"]
    base_ts = int(time.time() * 1000)
    out: dict[str, Any] = {"with_rollup_arm": has_rollup}
    for size in (1, 5, 20):
        with_rollup: list[float] = []
        without: list[float] = []
        for index in range(samples):
            if has_rollup:
                store._has_session_daily = True
                batch = _snapshots(size, session_ids=session_ids, base_ts=base_ts + index * 1000)
                start = time.thread_time()
                store.record_batch(batch)
                with_rollup.append(time.thread_time() - start)
                store._has_session_daily = False
            batch = _snapshots(size, session_ids=session_ids, base_ts=base_ts + index * 1000 + 500)
            start = time.thread_time()
            store.record_batch(batch)
            without.append(time.thread_time() - start)
        if has_rollup:
            store._has_session_daily = True
        entry: dict[str, Any] = {"ledger_plus_calendar_cpu_ms": _stats(without)}
        if has_rollup:
            entry["with_rollup_cpu_ms"] = _stats(with_rollup)
            entry["delta_p50_ms"] = round(_stats(with_rollup)["p50"] - _stats(without)["p50"], 3)
            entry["delta_p99_ms"] = round(_stats(with_rollup)["max"] - _stats(without)["max"], 3)
        out[f"batch_{size}"] = entry
    store.close()
    return out


def _host() -> dict[str, Any]:
    try:
        load = list(os.getloadavg())
    except OSError:  # pragma: no cover — not available on every platform
        load = []
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "loadavg": [round(value, 2) for value in load],
        "python": sys.version.split()[0],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path.home() / ".local-operator" / "analytics.db",
        help="source ledger; it is COPIED and never opened for writing",
    )
    parser.add_argument("--label", default="after", help="name this run in the JSON")
    parser.add_argument("--json", type=Path, default=None, help="write the results here")
    parser.add_argument("--samples", type=int, default=5, help="read samples per arm")
    parser.add_argument("--write-samples", type=int, default=25, help="write samples per size")
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="directory for the copy (default: a fresh temp dir)",
    )
    parser.add_argument("--skip-writes", action="store_true", help="reads only")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the working copy (default: removed, 341 MB is a lot of disk)",
    )
    args = parser.parse_args(argv)

    source = args.ledger.expanduser()
    if not source.exists():
        parser.error(f"no ledger at {source}")
    work = args.work or Path(tempfile.mkdtemp(prefix="bench-panel-"))
    work.mkdir(parents=True, exist_ok=True)
    copy = work / "analytics.db"
    # The copy is a SNAPSHOT of the live state at this instant; every arm below
    # runs against it or against a second copy of it, and the source is never
    # opened by SQLite at all.
    _copy_ledger(source, copy)

    result: dict[str, Any] = {
        "label": args.label,
        "tree": str(Path(__file__).resolve().parents[1]),
        # The module path is the only proof of WHICH tree produced these numbers:
        # a worktree with a symlinked venv, or a PYTHONPATH left over from an
        # earlier command, both run code that is not the checkout you are
        # standing in — and the failure is silent (AGENTS.md §Environment).
        "module": local_operator.__file__,
        "source_ledger": str(source),
        "work_copy": str(copy),
        "host": _host(),
        "samples": args.samples,
        "ledger": _ledger_shape(copy),
    }

    print(
        f"[{args.label}] ledger: {result['ledger']['file_bytes'] / 1e6:.1f} MB, "
        f"{result['ledger']['calls']:,} calls, {result['ledger']['sessions']:,} sessions"
    )
    print(
        f"[{args.label}] host: load {result['host']['loadavg']} "
        f"on {result['host']['cpu_count']} cores"
    )

    # --- the raw ledger: what the panel does today, before any rollup exists.
    result["first_touch_ledger"] = _first_touch(copy)
    result["read_ledger"] = _read_arms(copy, samples=args.samples)

    # --- the rollup, on the same copy: sweep it, then re-measure.
    prep = _prep_rollup(copy)
    if prep is not None:
        result["prep_rollup"] = prep
        result["ledger"]["session_daily_rows"] = prep["ledger"]["session_daily_rows"]
        result["read_rollup"] = _read_arms(copy, samples=args.samples)
        # A SECOND copy, so "first touch" means the same thing for both sides:
        # the first aggregate on a file whose pages are not in the cache.
        cold = work / "analytics-cold.db"
        _copy_ledger(copy, cold)
        result["first_touch_rollup"] = _first_touch(cold)

    result["route_payload"] = _route_payload(copy, samples=args.samples)
    result["session_report"] = _session_report_arms(copy, samples=args.samples)
    result["session_report_equivalence"] = _session_report_equivalence(
        copy, list(result["session_report"])
    )
    if not args.skip_writes:
        result["write"] = _write_arms(copy, samples=args.write_samples)

    def _row(arm: str, stats: dict[str, Any]) -> str:
        return (
            f"  {arm:<22} wall p50 {stats['wall_ms']['p50']:>9.1f} ms  "
            f"p90 {stats['wall_ms']['p90']:>9.1f}  |  cpu p50 {stats['cpu_ms']['p50']:>8.2f} ms  "
            f"p90 {stats['cpu_ms']['p90']:>8.2f}  path={stats.get('path', '')}"
        )

    for group in ("read_ledger", "read_rollup"):
        if group in result:
            print(f"[{args.label}] {group}:")
            for name, stats in result[group].items():
                print(_row(name, stats))
    for key in ("first_touch_ledger", "first_touch_rollup"):
        if key in result:
            stats = result[key]
            print(
                f"[{args.label}] {key}: wall {stats['wall_ms']:.1f} ms, "
                f"cpu {stats['cpu_ms']:.2f} ms, path={stats['path']}"
            )
    if prep is not None:
        print(
            f"[{args.label}] prep_rollup: {prep['days']} days in {prep['wall_ms']:.0f} ms wall, "
            f"{prep['cpu_ms']:.0f} ms cpu ({prep['per_day_wall_ms'].get('p50', 0):.0f} ms/day p50)"
        )
    if result.get("write"):
        print(f"[{args.label}] write:")
        for name, stats in result["write"].items():
            if not isinstance(stats, dict) or "ledger_plus_calendar_cpu_ms" not in stats:
                continue
            line = (
                f"  {name:<22} ledger+calendar cpu p50 "
                f"{stats['ledger_plus_calendar_cpu_ms']['p50']:>7.3f} ms"
            )
            if "with_rollup_cpu_ms" in stats:
                line += (
                    f"  (+session_daily {stats['with_rollup_cpu_ms']['p50']:.3f} ms, "
                    f"delta p50 {stats['delta_p50_ms']:+.3f} / p99 {stats['delta_p99_ms']:+.3f} ms)"
                )
            print(line)
    payload = result["route_payload"]
    print(
        f"[{args.label}] route payload: {payload['bytes'] / 1e6:.2f} MB, "
        f"{payload['by_session_entries']} sessions, cpu p50 {payload['cpu_ms']['p50']:.1f} ms"
    )
    if result.get("session_report"):
        print(f"[{args.label}] session_report (busiest real sessions):")
        for session_id, stats in result["session_report"].items():
            calls = f"{stats['calls']:>7,} calls"
            first = f"first {stats['first_wall_ms']:>8.1f} ms"
            warm = f"warm wall p50 {stats['wall_ms']['p50']:>8.1f} ms"
            cpu = f"cpu p50 {stats['cpu_ms']['p50']:>7.2f} ms"
            print(f"  {session_id} {calls}  {first}  {warm}  {cpu}")
    for session_id, check in result.get("session_report_equivalence", {}).items():
        if not isinstance(check, dict) or "equal" not in check:
            continue  # the parent tree has no oracle to run
        state = "same" if check["equal"] else f"DIFFERS {check['differing_fields']}"
        print(f"[{args.label}] session_report equivalence {session_id}: {state}")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[{args.label}] wrote {args.json}")
    if not args.keep and args.work is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
