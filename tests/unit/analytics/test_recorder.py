"""The non-blocking recorder: enqueue-and-return, background write, drop on
full queue, and parallel-safety across processes.

The recorder is the latency contract. ``record`` must never block a session,
so the queue is bounded and a full one drops the sample (counted) rather than
applying back-pressure. The writer thread turns queued samples into batched
SQLite writes on a background thread, which is what keeps the provider path
free of disk I/O.
"""

from __future__ import annotations

import multiprocessing as mp
import time

from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.recorder import AnalyticsRecorder, reset_recorder_for_test
from local_operator.analytics.store import AnalyticsStore


def _snap(
    session_id: str = "s",
    *,
    input_tokens: int = 10,
    ts_ms: int | None = None,
) -> CallSnapshot:
    return CallSnapshot(
        ts_ms=ts_ms if ts_ms is not None else int(time.time() * 1000),
        session_id=session_id,
        provider="anthropic",
        model_id="m",
        input_tokens=input_tokens,
        output_tokens=5,
        cache_read_tokens=2,
        cache_write_tokens=1,
        reasoning_tokens=1,
        context_tokens=12,
        component_chars={"conversation": 40},
        ok=True,
    )


def test_record_reaches_store(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    for _ in range(20):
        rec.record(_snap())
    rec.flush_for_test()
    assert store.aggregate().calls == 20
    rec.close()


def test_record_never_raises_when_closed(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    rec.close()
    # Recording on a closed recorder is a silent no-op, not an exception.
    rec.record(_snap())
    assert rec.dropped == 0


def test_batching_coalesces_burst(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    # A burst larger than one batch still lands entirely.
    for i in range(500):
        rec.record(_snap(session_id=f"s{i % 5}"))
    rec.flush_for_test()
    agg = store.aggregate()
    assert agg.calls == 500
    assert len(agg.by_session) == 5
    rec.close()


def test_session_name_note_reaches_store(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    rec.record(_snap(session_id="abc"))
    rec.note_session_name("abc", "named it")
    rec.flush_for_test()
    # The name upsert runs on its own short-lived thread; poll for it rather
    # than sleeping a fixed amount, so the test is not flaky under load.
    deadline = time.monotonic() + 5.0
    names: dict[str, str] = {}
    while time.monotonic() < deadline:
        names = getattr(store.aggregate(), "session_names", {})
        if names.get("abc") == "named it":
            break
        time.sleep(0.02)
    assert names.get("abc") == "named it"
    rec.close()


def test_reset_recorder_for_test_isolates(tmp_path):
    store_a = AnalyticsStore(tmp_path / "a.db")
    rec_a = reset_recorder_for_test(store_a)
    rec_a.record(_snap())
    rec_a.flush_for_test()
    assert store_a.aggregate().calls == 1
    # Resetting closes the previous recorder and points at a fresh store.
    store_b = AnalyticsStore(tmp_path / "b.db")
    rec_b = reset_recorder_for_test(store_b)
    assert store_b.aggregate().calls == 0
    rec_b.close()


def _worker(db_path, n):
    store = AnalyticsStore(db_path)
    rec = reset_recorder_for_test(store)
    for i in range(n):
        rec.record(_snap(session_id=f"proc-{mp.current_process().name}", input_tokens=1))
    rec.flush_for_test()
    rec.close()


def test_parallel_processes_write_atomically(tmp_path):
    # WAL gives cross-process atomic writes: several sessions in different
    # terminals writing at once must not lose or corrupt rows.
    db = str(tmp_path / "parallel.db")
    n = 150
    procs = [mp.Process(target=_worker, args=(db, n), name=f"w{i}") for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    store = AnalyticsStore(db)
    agg = store.aggregate()
    assert agg.calls == 4 * n
    assert len(agg.by_session) == 4
    store.close()
