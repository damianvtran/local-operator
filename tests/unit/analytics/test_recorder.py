"""The non-blocking recorder: enqueue-and-return, background write, drop on
full queue, and parallel-safety across processes.

The recorder is the latency contract. ``record`` must never block a session,
so the queue is bounded and a full one drops the sample (counted) rather than
applying back-pressure. The writer thread turns queued samples into batched
SQLite writes on a background thread, which is what keeps the provider path
free of disk I/O.

The barrier those writes are observed through is ``flush_for_test``: a plain
``record`` is asynchronous by design, so every test here that reads the store
is really asserting something about that barrier as well. A group of tests
below exists only to pin the barrier itself — deliberately slow stores widen the
dequeued-but-unwritten window into a fact rather than a race, and the rest pin
the ways it is allowed to fail (a deadline, a write that RAISED, and a write the
store RETURNED as dropped, which is how the shipped store loses one).
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from local_operator.analytics import store as store_module
from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.recorder import AnalyticsRecorder, reset_recorder_for_test
from local_operator.analytics.store import SESSION_NAME_RANK_TITLE, AnalyticsStore


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
    # Read straight after the barrier, with no poll loop in between: the barrier
    # now covers a name task, so the loop this test used to carry — which only
    # narrowed the race — is gone. See the two slow-store tests below for why it
    # could not have been trusted without that. ``session_names`` is a SIDE
    # attribute the store attaches to the aggregate, hence the ``getattr``.
    assert getattr(store.aggregate(), "session_names", {}).get("abc") == "named it"
    rec.close()


#: How long the deliberately slow stores below hold the writer inside one
#: write. Long enough that a barrier which polls the queue and then sleeps a
#: flat 50 ms provably returns first even on an idle machine, short enough to
#: stay cheap in CI.
_HOLD_S = 0.5


class _SlowNameStore(AnalyticsStore):
    """A store that parks the writer INSIDE the session-name upsert.

    The window #1250 records — item dequeued, write not yet done — is invisible
    at SQLite speed, so the witness has to widen it on purpose rather than hope
    a loaded machine widens it. ``entered`` lets a test wait until the writer is
    committed to the upsert, which is exactly the state the old barrier's
    ``queue.empty()`` check read as "drained".
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.entered = threading.Event()

    def upsert_session_name(
        self, session_id: str, name: str, *, rank: int = SESSION_NAME_RANK_TITLE
    ) -> bool:
        self.entered.set()
        time.sleep(_HOLD_S)
        return super().upsert_session_name(session_id, name, rank=rank)


class _SlowToolCallStore(AnalyticsStore):
    """The same widening for a tool-call row, which never moved a counter at all."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.entered = threading.Event()

    def record_tool_calls(self, rows: Sequence[tuple[int, str, str, str, str, float]]) -> int:
        self.entered.set()
        time.sleep(_HOLD_S)
        return super().record_tool_calls(rows)


class _BlockedBatchStore(AnalyticsStore):
    """A store that will not finish a batch until the test releases it.

    For the expiry path: the writer has to be genuinely stuck for the deadline
    to be exercised at all, and the release has to exist so the test does not
    leave a thread blocked inside SQLite behind it.
    """

    #: A ceiling the test never intends to reach — it releases first. The wait
    #: is only here so a broken test cannot hang the suite instead of failing.
    _CEILING_S = 10.0

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.entered = threading.Event()
        self.release = threading.Event()

    def record_batch(self, snapshots: Sequence[CallSnapshot]) -> int:
        self.entered.set()
        self.release.wait(self._CEILING_S)
        return super().record_batch(snapshots)


def test_flush_for_test_waits_for_a_name_upsert_already_dequeued(tmp_path):
    """#1250 item 2's witness: a name the writer has DEQUEUED is still waited on.

    This is the shape that went red intermittently: ``create_provider_model_client``
    names its episode through ``note_session_name`` and the caller reads
    ``session_names`` immediately after ``flush_for_test()``. A name task moves
    no snapshot counter, and the queue is empty the instant the writer takes it
    — so with the old body the read raced the upsert, and the assertion failed
    under load with no hint that nothing had been waited for (run 35241086560,
    ``assert None is not None``).
    """
    store = _SlowNameStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    try:
        rec.note_session_name("slow-session", "held inside the writer")
        assert store.entered.wait(5.0), "the writer never reached the name upsert"
        rec.flush_for_test()
        conn = store._connect()
        assert conn is not None
        row = conn.execute(
            "SELECT name FROM session_names WHERE session_id = ?", ("slow-session",)
        ).fetchone()
        assert row is not None and row[0] == "held inside the writer"
    finally:
        rec.close()


def test_flush_for_test_waits_for_a_tool_call_row_already_dequeued(tmp_path):
    """The same hole for tool calls, which the old barrier never covered at all.

    A ``_ToolCallTask`` carries no snapshot either, so it moved neither of the
    counters the old body waited on: only the flat 50 ms settle ever stood
    between this read and the write.
    """
    store = _SlowToolCallStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    try:
        rec.record_tool_call("tool-session", "bash", "model", "")
        assert store.entered.wait(5.0), "the writer never reached the tool-call write"
        rec.flush_for_test()
        conn = store._connect()
        assert conn is not None
        row = conn.execute(
            "SELECT tool_name FROM tool_calls WHERE session_id = ?", ("tool-session",)
        ).fetchone()
        assert row is not None and row[0] == "bash"
    finally:
        rec.close()


class _FailingWriteStore(AnalyticsStore):
    """A store whose writes always raise, for the swallowed-failure path.

    The writer must survive this (fail-soft is the deliberate policy), so the
    only thing left to check is that the barrier does not report success.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.attempts = 0

    def upsert_session_name(
        self, session_id: str, name: str, *, rank: int = SESSION_NAME_RANK_TITLE
    ) -> bool:
        self.attempts += 1
        raise sqlite3.OperationalError("disk I/O error")


def test_flush_for_test_reports_a_write_the_writer_swallowed(tmp_path):
    """A settled item is not a written row, and the barrier has to say so.

    ``_flush`` swallows a failed store call on purpose — a bad sample must
    never kill the writer — so an item whose write raised settles exactly like
    one that succeeded and the completion count cannot tell them apart. Without
    this, the barrier would return clean and the caller would read a row that
    is not there: the same bare ``None`` the rest of this file exists to remove.
    """
    store = _FailingWriteStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    try:
        rec.note_session_name("never-made-it", "never lands")
        with pytest.raises(RuntimeError, match="session name"):
            rec.flush_for_test()
        # Reported ONCE: a failure is surfaced by the next barrier, not by every
        # barrier for the rest of the process, so a test that reads its store
        # after a deliberate failure is not fighting the barrier forever.
        rec.flush_for_test()
        # And a report counts what is NEW since the previous one, not a running
        # total: the second loss is one loss, not two.
        rec.note_session_name("never-made-it-either", "also never lands")
        with pytest.raises(RuntimeError, match=r"1 analytics write\(s\).*session name ×1"):
            rec.flush_for_test()
    finally:
        rec.close()
    assert store.attempts == 2, "the writer retried a name the store refused"


class _DroppingStore(AnalyticsStore):
    """A store that reports a DROPPED write the way the shipped one does.

    The real ``AnalyticsStore`` never raises on a lost lock: ``record_batch``
    and ``record_tool_calls`` give up after their retries and return 0 rows,
    and ``upsert_session_name`` returns False. Holding a real lock long enough
    to exhaust the real retry budget costs ~5 s per call at a shortened budget
    and ~21 s at the shipped one, so the cells for the two kinds the locked
    real-store test below does not cover use this double; that test pins the
    same contract against the shipped store itself.
    """

    def record_tool_calls(self, rows: Sequence[tuple[int, str, str, str, str, float]]) -> int:
        return 0

    def upsert_session_name(
        self, session_id: str, name: str, *, rank: int = SESSION_NAME_RANK_TITLE
    ) -> bool:
        return False


def test_flush_for_test_reports_writes_the_store_returned_as_dropped(tmp_path):
    """A store reporting a drop by RETURNING must not read as success.

    The shipped store's failure mode is silence rather than an exception, so a
    barrier that only watched for raises would report clean on the shape that
    actually happens in production — and the caller's next read would get the
    bare ``None`` this PR exists to remove.
    """
    store = _DroppingStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    try:
        rec.record_tool_call("tool-session", "bash", "model", "")
        rec.note_session_name("name-session", "never lands")
        with pytest.raises(RuntimeError, match=r"session name ×1.*tool call ×1"):
            rec.flush_for_test()
    finally:
        rec.close()


def test_flush_for_test_reports_a_batch_the_real_store_dropped(tmp_path, monkeypatch):
    """The same loss against the SHIPPED store, with no mock in the way.

    ``AnalyticsStore.record_batch`` retries a lost lock (``busy_timeout`` is 5 s
    per attempt) and then returns 0 rows — silently, with no exception — so with
    a second connection holding the write lock the row never lands while
    ``flush_for_test`` used to return cleanly (review round 2 measured 21.5 s to
    exhaustion and ``aggregate().calls == 0``). The retry budget is shortened to
    ONE attempt here: the drop being pinned is the same drop at any budget, and
    what this cell asserts is that the store RETURNED 0 instead of raising.
    Four attempts would add ~16 s to this file for no extra coverage. The
    writer's connection is warmed first, so the lock is taken against the INSERT
    rather than against the schema-creation path, which is a different branch.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    holder: sqlite3.Connection | None = None
    try:
        rec.record(_snap(session_id="before"))
        rec.flush_for_test(timeout=30)
        assert store.aggregate().calls == 1

        monkeypatch.setattr(store_module, "_WRITE_RETRIES", 1)
        holder = sqlite3.connect(str(tmp_path / "a.db"))
        holder.execute("BEGIN IMMEDIATE")  # hold the write lock

        rec.record(_snap(session_id="dropped"))
        with pytest.raises(RuntimeError, match="ledger batch"):
            rec.flush_for_test(timeout=60)
        # The row really is gone: this is a LOST write, not a slow one, so a
        # barrier that reported success would be reporting it wrongly.
        assert store.aggregate().calls == 1
    finally:
        if holder is not None:
            holder.rollback()
            holder.close()
        rec.close()


def test_flush_for_test_raises_at_its_deadline_instead_of_returning(tmp_path):
    """Expiry is a failure, not a return: a barrier without its guarantee IS the bug.

    The old body let the deadline pass and returned as though the queue were
    drained, so a caller got a silent half-drain. Raising is what makes the
    difference between "the store is wrong" and "nothing was ever waited for".
    """
    store = _BlockedBatchStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    try:
        rec.record(_snap())
        assert store.entered.wait(5.0), "the writer never reached the batch write"
        with pytest.raises(TimeoutError, match="1 unsettled item"):
            rec.flush_for_test(timeout=0.2)
        # Not a wedge: once the writer is free the barrier completes and the
        # row lands, which is what makes the deadline a bound rather than a
        # verdict on the store.
        store.release.set()
        rec.flush_for_test()
        assert store.aggregate().calls == 1
    finally:
        store.release.set()
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


def test_tool_calls_ride_the_same_queue_and_writer_thread(tmp_path):
    """One writer, never two.

    ``AGENTS.md`` is explicit that a second thread writing to the store is
    forbidden: two threads opening their first connection to a fresh database
    race in a way that leaves the writer unable to see its own commits. Tool
    calls therefore share the queue, the thread and the connection that call
    samples and name upserts already use \u2014 asserted here by counting the
    recorder's threads, not by trusting the implementation.
    """
    import threading

    store = AnalyticsStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    before = {t.name for t in threading.enumerate()}
    rec.record(_snap(session_id="s1"))
    for i in range(10):
        rec.record_tool_call("s1", "read", "model", "" if i % 2 else "execution", 12.0)
    rec.note_session_name("s1", "named")
    rec.flush_for_test()
    writers = {t.name for t in threading.enumerate()} - before
    assert writers == {"lo-analytics-writer"}, f"expected ONE writer thread, got {writers}"

    stats = store.session_report("s1").tool_calls
    assert stats is not None
    assert stats.total == 10 and stats.ok == 5
    assert stats.faults == {"execution": 5}
    rec.close()


def test_record_tool_call_never_raises(tmp_path):
    """It runs on the event loop inside a live turn; it may not throw or block."""
    store = AnalyticsStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    rec.close()
    # Closed recorder, empty session id, absurd values: all silent no-ops.
    rec.record_tool_call("s1", "read", "model", "", 1.0)
    rec.record_tool_call("", "read", "model", "", 1.0)


def test_record_tool_call_is_put_nowait_only(tmp_path):
    """A full queue DROPS the sample rather than blocking the turn.

    Same contract as ``record``: accuracy matters, but never at the cost of a
    stalled session. Measured here by filling the queue and timing the call \u2014
    a blocking implementation would sit on the queue's put instead.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = AnalyticsRecorder(store=store)
    # Fill the queue without letting the writer drain it.
    rec._closed = False
    while not rec._queue.full():
        try:
            rec._queue.put_nowait(_snap())
        except Exception:  # noqa: BLE001
            break
    started = time.monotonic()
    rec.record_tool_call("s1", "read", "model", "", 1.0)
    assert time.monotonic() - started < 0.5, "record_tool_call blocked on a full queue"
    rec.close()
