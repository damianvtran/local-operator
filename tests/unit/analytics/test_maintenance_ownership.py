"""One owner for host-wide analytics maintenance, and a bounded WAL.

Two defects this file pins, both measured on the operator's live host:

* **18-22 processes each ran their own hourly retention sweep.**
  ``AnalyticsRecorder._last_prune`` is a per-PROCESS attribute, so every process
  holding ``analytics.db`` swept the same ledger on its own timer. A steady-state
  prune is cheap (0.7 ms of CPU) and an idempotent one is not wrong, but a sweep
  that finds a backlog is not cheap (892.8 ms of CPU for a 400k-row delete) and
  each one wrote a WAL as large as the database.
* **The WAL was 528,204,632 bytes and did not move** across samples 20 s apart —
  128,956 pages, 129x SQLite's 1000-page auto-checkpoint threshold — because
  nothing in the tree ever truncated it.

The tests below are structural wherever the property IS a structure (which lock
file was created, whether ``prune`` ran, on which thread, in which order), and
they wait on events rather than on the clock. The one timing bound is called out
where it appears, with the measurement that gives it its margin.

THE ELECTION, in one paragraph, because every test here is about one part of it:
the recorder whose hourly timer fires takes ``<root>/run/analytics-maintenance.lock``
with ``LOCK_EX | LOCK_NB`` (never waiting), reads the wall-clock stamp in that
file, and sweeps only if the stamp is missing or older than an hour; it stamps
the file after the sweep returns. The lock makes the read-modify-write atomic;
the stamp is what makes ownership survive the winner's exit, so a process whose
timer fires ten minutes later does not sweep the same hour again.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from local_operator.analytics import recorder as recorder_module
from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.recorder import AnalyticsRecorder, maintenance_lock_path
from local_operator.analytics.store import _WAL_SIZE_LIMIT_BYTES, AnalyticsStore

#: How long a test will wait for work the code under test has ALREADY been told
#: to publish before calling it wedged. Not an assertion about speed: every
#: assertion in this file is about what happened, and this only keeps a
#: regression that blocks forever from hanging the suite instead of failing it.
#:
#: 20 s was measured generous on a dev box and is NOT generous on a loaded CI
#: runner, where this file failed with "the writer thread never reached the
#: sweep". The wait is on the right thing — an event the sweep sets — but what
#: sets it is a writer thread reaching its IDLE tick, and that tick comes only
#: after the ``_FLUSH_INTERVAL_S`` flush completes. On a runner sharing cores
#: with four sibling shards, 20 s stops being the catastrophe margin AGENTS.md
#: asks a backstop to be and becomes a second, quieter speed assertion. Raised
#: to 120 s for that reason, and deliberately WITHOUT touching anything these
#: tests assert: no ceiling on the suite, no retry loop, no sleep.
_BACKSTOP_S = 240.0

#: Ceiling on a store that parks the writer on purpose, so a broken test cannot
#: leave a thread blocked inside the sweep forever.
_CEILING_S = 10.0


def _snap(session_id: str = "s") -> CallSnapshot:
    return CallSnapshot(
        ts_ms=int(time.time() * 1000),
        session_id=session_id,
        provider="anthropic",
        model_id="m",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=2,
        cache_write_tokens=1,
        reasoning_tokens=1,
        context_tokens=12,
        component_chars={"conversation": 40},
        ok=True,
    )


class _CountingStore:
    """A duck-typed store that COUNTS the sweep instead of performing it.

    Deliberately NOT an ``AnalyticsStore`` and deliberately without a
    ``db_path``: the recorder only calls ``prune``/``bound_wal``/``close``
    during a sweep, and a store that cannot name its database is the arm of the
    root resolution that falls back to ``config_dir()`` — which is what the
    isolation tests are about.

    The sweep's two calls are the whole observable surface of "did maintenance
    happen", so counting them is a structural answer — no clock, no database, no
    rows. It also records the thread each call arrived on, which is how the
    off-the-event-loop property is asserted (AGENTS.md: prefer thread identity to
    a timing bound).

    ``entered``/``release`` let a test park the sweep INSIDE ``prune`` so the
    election lock is held for a fact rather than for a hope.
    """

    def __init__(
        self,
        *,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.threads: list[str] = []
        self._entered = entered
        self._release = release

    def _note(self, name: str) -> None:
        self.calls.append(name)
        self.threads.append(threading.current_thread().name)

    def prune(self) -> int:
        self._note("prune")
        if self._entered is not None:
            self._entered.set()
        if self._release is not None:
            self._release.wait(_CEILING_S)
        return 0

    def bound_wal(self) -> bool:
        self._note("bound_wal")
        return True

    def record_batch(self, snapshots) -> int:
        return len(snapshots)

    def close(self) -> None:
        pass


class _Rows:
    """Stand-in for a ``sqlite3.Cursor`` carrying one canned row."""

    def __init__(self, row) -> None:
        self._row = row

    def fetchone(self):
        return self._row


class _RecordingConn:
    """A fake connection that records the SQL and answers the checkpoints.

    Used to pin the SHAPE of ``bound_wal`` — which statements it issues, and in
    which order — because that shape is what makes the difference between a
    checkpoint that can wait five seconds and one that cannot wait at all.
    """

    def __init__(self, passive_row, truncate_row=(0, 0, 0)) -> None:
        self.sql: list[str] = []
        self.passive_row = passive_row
        self.truncate_row = truncate_row
        self.truncate_calls = 0

    def execute(self, sql: str) -> _Rows:
        self.sql.append(sql)
        if sql.startswith("PRAGMA wal_checkpoint(PASSIVE)"):
            return _Rows(self.passive_row)
        if sql.startswith("PRAGMA wal_checkpoint(TRUNCATE)"):
            self.truncate_calls += 1
            return _Rows(self.truncate_row)
        return _Rows(None)


def _fake(**kwargs) -> tuple[AnalyticsStore, "_CountingStore"]:
    """A counting store, plus a handle typed as the store the recorder takes.

    The recorder's parameter is annotated ``AnalyticsStore | None`` while its
    sweep only needs the two calls, so one cast here is honest about the
    duck-typing and keeps every construction site free of a pyright ignore.
    """
    counter = _CountingStore(**kwargs)
    return cast(AnalyticsStore, counter), counter


# ---------------------------------------------------------------------------
# One owner per hour, whatever process it is
# ---------------------------------------------------------------------------


def _hour_turns(*recorders: AnalyticsRecorder) -> None:
    """Model an hour passing for the per-process timers.

    The gate is ``now - self._last_prune >= 3600``, and setting the stamp to
    ``0.0`` is exactly what an hour of monotonic time does to it — on any clock
    that has been up for an hour, which is every clock this runs on. Driving the
    gate this way keeps the test off the process-wide ``time`` module, which
    several other threads in a pytest worker are using at the same moment.
    """
    for rec in recorders:
        rec._last_prune = 0.0


def _age_claim(root: Path, seconds: float) -> None:
    """Move the last-sweep stamp back, the way an hour of wall clock would.

    The file IS the cross-process interface, so a test edits it directly rather
    than reaching inside the recorder: a reader cannot tell how a stamp got old,
    and neither should a test pretend to.
    """
    path = maintenance_lock_path(root)
    stamp = float(path.read_text().strip())
    path.write_text(f"{int(stamp - seconds)}\n")


def test_two_recorders_over_one_root_sweep_exactly_once_an_hour(tmp_path):
    """The headline: N recorders on one root produce ONE sweep per hour.

    Before this, every recorder swept on its own timer — the structural count
    was N sweeps per hour at N recorders, which is the measurement this lane
    exists for. Two hours are simulated so the test also shows that ownership is
    per HOUR rather than per process: the second hour is won by the recorder
    that lost the first.
    """
    root = tmp_path / "iso"
    first_handle, first = _fake()
    second_handle, second = _fake()
    a = AnalyticsRecorder(store=first_handle, maintenance_root=root)
    b = AnalyticsRecorder(store=second_handle, maintenance_root=root)

    # Hour 1: whoever ticks first sweeps, and the other one skips entirely.
    _hour_turns(a, b)
    assert a._run_owned_maintenance() is True
    assert b._run_owned_maintenance() is False
    assert first.calls == ["prune", "bound_wal"]
    assert second.calls == [], "the loser reached the store"

    # Hour 2: the stamp is an hour old, so the hour is open again — and B wins
    # it this time, which is what makes the owner "whichever recorder ticks
    # first" rather than "whichever recorder started first".
    _age_claim(root, recorder_module._PRUNE_INTERVAL_S + 60)
    _hour_turns(a, b)
    assert b._run_owned_maintenance() is True
    assert a._run_owned_maintenance() is False
    assert second.calls == ["prune", "bound_wal"]
    assert first.calls == ["prune", "bound_wal"]

    # Two hours, four attempts, two sweeps — and the stamp names the winner's
    # hour, so a fresh process reads it as "done" without any in-memory state.
    stamp = float(maintenance_lock_path(root).read_text().strip())
    assert abs(stamp - time.time()) < 60.0


def test_a_loser_never_prunes_and_never_checkpoints(tmp_path):
    """The checkpoint is as exclusive as the sweep: a loser does neither.

    Asserted separately from the count above because these are the two calls
    that write to the shared ledger: a loser that pruned nothing but still
    checkpointed would be doing I/O on a file another process owns this hour.
    """
    root = tmp_path / "iso"
    winner_handle, winner = _fake()
    loser_handle, loser = _fake()
    a = AnalyticsRecorder(store=winner_handle, maintenance_root=root)
    b = AnalyticsRecorder(store=loser_handle, maintenance_root=root)
    _hour_turns(a, b)
    a._run_owned_maintenance()
    b._run_owned_maintenance()
    assert "prune" in winner.calls and "bound_wal" in winner.calls
    assert loser.calls == []


def test_the_sweep_runs_on_the_writer_thread_and_never_on_the_caller(tmp_path):
    """The maintenance runs on ``lo-analytics-writer``, driven by the real loop.

    Structural, in the sense AGENTS.md asks for: thread identity is a fact about
    where the code ran, so it cannot flake, and it fails the moment somebody
    moves the sweep onto the event loop or onto a recording caller's thread.
    """
    entered = threading.Event()
    handle, store = _fake(entered=entered)
    rec = AnalyticsRecorder(store=handle, maintenance_root=tmp_path / "iso")
    try:
        # THE WRITER PATH IS THE SUBJECT, so it cannot be driven directly: an
        # earlier revision called `_run_owned_maintenance()` from the test thread
        # and the store recorded `MainThread` — which is the *caller* case this
        # cell exists to forbid, not evidence about the writer.
        #
        # What it waits on is therefore the writer's idle tick (`queue.Empty`
        # after `_FLUSH_INTERVAL_S`), which is real production behaviour but is a
        # CADENCE rather than a contract. That is why `_BACKSTOP_S` is minutes and
        # not seconds: on a CI runner sharing cores with four sibling shards a
        # 0.5 s tick can take a long time to arrive, and the cell reports "the
        # writer thread never reached the sweep" rather than a wrong thread
        # identity when it does not. The assertion it makes once the tick lands is
        # the one that cannot flake.
        _hour_turns(rec)
        rec.record(_snap())
        assert entered.wait(_BACKSTOP_S), "the writer thread never reached the sweep"
        rec.flush_for_test()
        assert store.threads and set(store.threads) == {"lo-analytics-writer"}, store.threads
    finally:
        rec.close()


def test_a_second_recorder_does_not_wait_for_a_sweep_already_in_flight(tmp_path):
    """A sweep holding the lock is a REFUSAL for a peer, never a queue to join.

    ``prune`` is parked on an event, so the lock is provably held while the
    second recorder runs its election on this thread. The assertion is that the
    election returned and the second store was never touched — if the election
    had waited for the first sweep, this test would hang until the backstop.
    """
    root = tmp_path / "iso"
    entered, release = threading.Event(), threading.Event()
    first_handle, first = _fake(entered=entered, release=release)
    a = AnalyticsRecorder(store=first_handle, maintenance_root=root)
    second_handle, second = _fake()
    b = AnalyticsRecorder(store=second_handle, maintenance_root=root)

    returned, outcome = threading.Event(), []

    def winner() -> None:
        # The holder: it takes the lock and parks in `prune` until released.
        a._run_owned_maintenance()

    def loser() -> None:
        # The result is carried out to the test thread rather than asserted in
        # here: an exception raised on a worker thread would be swallowed by the
        # ``finally`` and the test could pass on a wrong answer.
        try:
            outcome.append(b._run_owned_maintenance())
        finally:
            returned.set()

    try:
        # The holder stands ON the writer path, so the lock is held by the real
        # mechanism rather than by a test thread imitating it. Same cadence
        # caveat as the sibling cell above: the wait is on the writer's idle tick,
        # with a minutes-long backstop for a loaded runner.
        _hour_turns(a, b)
        rec_starter = threading.Thread(target=lambda: a.record(_snap()), daemon=True)
        rec_starter.start()
        assert entered.wait(_BACKSTOP_S), "the first sweep never started"
        thread = threading.Thread(target=loser, daemon=True)
        thread.start()
        assert returned.wait(_BACKSTOP_S), "the loser waited on the lock instead of refusing it"
        assert outcome == [False], "the loser believed it owned the sweep"
        assert second.calls == []
        assert first.calls == ["prune"], "the winner was still inside its prune"
    finally:
        release.set()
        a.close()
        b.close()


def test_a_wedged_holder_costs_the_loser_nothing_and_it_performs_no_sweep(tmp_path):
    """A holder that never releases must not block the caller, and is not swept by.

    ``flock`` contends between separate OPEN FILE DESCRIPTIONS, so a holder in
    this very test is a faithful stand-in for a wedged peer process. The holder
    stays wedged for the whole assertion, which is the point: the loser must
    return on its own, and the only thing that makes that true is ``LOCK_NB``
    with no poll loop (a blocking ``flock`` here is the #401 freeze).
    """
    import fcntl

    root = tmp_path / "iso"
    path = maintenance_lock_path(root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    holder = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A peer's stamp, which a wedged loser must not disturb.
    os.write(holder, b"1700000000\n")

    handle, store = _fake()
    rec = AnalyticsRecorder(store=handle, maintenance_root=root)
    returned, outcome = threading.Event(), []

    def loser() -> None:
        try:
            outcome.append(rec._run_owned_maintenance())
        finally:
            returned.set()

    try:
        thread = threading.Thread(target=loser, daemon=True)
        thread.start()
        assert returned.wait(_BACKSTOP_S), "the election waited on a wedged holder"
        assert outcome == [False], "the loser believed it owned the sweep"
        assert store.calls == []
        # The holder is STILL holding it: nothing in the loser released or broke
        # the peer's lock, and the peer's stamp is byte-for-byte unchanged.
        probe = os.open(path, os.O_RDWR)
        try:
            with pytest.raises(OSError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)
        assert path.read_text() == "1700000000\n"
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)


def test_a_claim_that_cannot_be_read_sweeps_rather_than_skipping(tmp_path):
    """An unreadable stamp fails toward maintenance, not away from it.

    A truncated write from a process that died mid-stamp must not be able to
    stop the host's retention for an hour, so anything unparseable reads as "no
    claim" and the sweep runs. The reverse choice would be a silent way to
    disable the prune.
    """
    root = tmp_path / "iso"
    path = maintenance_lock_path(root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text("")  # an empty file, which is also a valid flock target

    handle, store = _fake()
    rec = AnalyticsRecorder(store=handle, maintenance_root=root)
    assert rec._run_owned_maintenance() is True
    assert store.calls == ["prune", "bound_wal"]
    assert float(path.read_text().strip()) == pytest.approx(time.time(), abs=60.0)


# ---------------------------------------------------------------------------
# The root the election is held in
# ---------------------------------------------------------------------------


def test_an_isolated_root_elects_under_itself_and_cannot_see_another_roots_lock(
    tmp_path, monkeypatch
):
    """The isolation rule, and the browser-bridge precedent behind it.

    An isolated run must elect within its own root: the supervisor-label bug in
    AGENTS.md ("Isolating a run") is what happens when a per-root thing keys on
    the wrong root — a redirected ``HOME`` registered the GLOBAL launchd label
    and evicted the operator's live daemon. So this test asserts both halves:
    the lock is created UNDER the redirected root, and a recorder on another
    root does not see the first root's lock at all (it steps past a held lock
    belonging to a different root and sweeps its own).
    """
    import fcntl

    root_a = tmp_path / "iso-a"
    root_b = tmp_path / "iso-b"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root_a))

    # A stub store has no database path, so the election resolves the config
    # root — the resolution this test is about.
    a_handle, a_store = _fake()
    a = AnalyticsRecorder(store=a_handle)
    assert a.maintenance_lock == root_a / "run" / "analytics-maintenance.lock"
    assert a.maintenance_lock.is_relative_to(root_a)
    assert a._run_owned_maintenance() is True
    assert a_store.calls == ["prune", "bound_wal"]
    assert a.maintenance_lock.exists()
    assert not (root_b / "run").exists(), "root B was touched by a root A sweep"

    # Wedge root A's lock, then elect on root B. A different root's lock is not
    # this root's lock, so B must sweep and stamp its own file.
    holder = os.open(a.maintenance_lock, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        stamp_a = a.maintenance_lock.read_text()
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root_b))
        b_handle, b_store = _fake()
        b = AnalyticsRecorder(store=b_handle)
        assert b.maintenance_lock == root_b / "run" / "analytics-maintenance.lock"
        assert b._run_owned_maintenance() is True
        assert b_store.calls == ["prune", "bound_wal"]
        assert b.maintenance_lock.exists()
        assert a.maintenance_lock.read_text() == stamp_a, "root A's stamp was rewritten"
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)


def test_the_election_follows_the_redirected_home_when_no_config_override_is_set(
    tmp_path, monkeypatch
):
    """With no override the root is still the RESOLVED one, never ``Path.home()``.

    ``config_dir()`` is the only root resolver in this path: it honours the
    override and otherwise derives the dot-directory from the home directory, so
    a run that redirects ``HOME`` alone (the suite's own isolation shape, and the
    one AGENTS.md insists on) elects inside the redirected home.
    """
    iso_home = tmp_path / "iso-home"
    monkeypatch.setenv("HOME", str(iso_home))
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)

    handle, store = _fake()
    rec = AnalyticsRecorder(store=handle)
    assert rec.maintenance_lock == iso_home / ".local-operator" / "run" / (
        "analytics-maintenance.lock"
    )
    assert rec._run_owned_maintenance() is True
    assert store.calls == ["prune", "bound_wal"]
    assert rec.maintenance_lock.exists()


def test_a_store_that_is_not_the_default_elects_beside_its_own_database(tmp_path, monkeypatch):
    """An injected store's root follows the STORE, not the ambient config root.

    This is what keeps the suite (and any caller that passes an explicit
    ``db_path``) from writing a lock file into the operator's real config root:
    the store that would be swept is the store whose root is elected on, even
    when the environment points somewhere else entirely.
    """
    ambient = tmp_path / "operator-root"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(ambient))
    store = AnalyticsStore(tmp_path / "somewhere-else" / "analytics.db")
    rec = AnalyticsRecorder(store=store)
    assert rec.maintenance_lock == tmp_path / "somewhere-else" / "run" / (
        "analytics-maintenance.lock"
    )
    assert not (ambient / "run").exists()


# ---------------------------------------------------------------------------
# The WAL bound
# ---------------------------------------------------------------------------


def _grow_wal(store: AnalyticsStore, rows: int = 400) -> Path:
    """Write real frames through the store's own connection."""
    conn = store._connect()
    assert conn is not None
    for i in range(rows):
        conn.execute(
            "INSERT INTO calls (ts_ms, session_id, provider, model_id) VALUES (?,?,?,?)",
            (int(time.time() * 1000), f"s{i}", "anthropic", "m"),
        )
    conn.commit()
    return store.db_path.with_suffix(store.db_path.suffix + "-wal")


def test_bound_wal_reclaims_the_file_and_sets_the_retained_bound(tmp_path):
    """A completed sweep gives the WAL's bytes back and leaves a bound behind."""
    store = AnalyticsStore(tmp_path / "a.db")
    wal = _grow_wal(store)
    assert wal.stat().st_size > 0, "the fixture did not actually grow a WAL"

    conn = store._connect()
    assert conn is not None
    conn.execute("PRAGMA journal_size_limit=-1")  # the SQLite default, per connection
    assert store.bound_wal() is True
    assert wal.stat().st_size == 0, "the checkpoint did not give the space back"
    assert conn.execute("PRAGMA journal_size_limit").fetchone()[0] == _WAL_SIZE_LIMIT_BYTES


def test_every_connection_carries_the_bound_not_only_the_electing_one(tmp_path):
    """The pragma is per-connection, so it is set where connections are made.

    Measured (``.perf/bench/lane2/wal_limit_scope.txt``): a limit set on one
    connection does NOT truncate a WAL restarted by another, and every process
    on the host restarts this WAL at some point. Setting it only at the sweep
    would leave every other process's restarts unbounded.
    """
    db = tmp_path / "a.db"
    AnalyticsStore(db)._connect()  # a first process's connection
    second = AnalyticsStore(db)._connect()
    assert second is not None
    assert second.execute("PRAGMA journal_size_limit").fetchone()[0] == _WAL_SIZE_LIMIT_BYTES


def test_bound_wal_never_attempts_a_truncate_that_would_have_to_wait(tmp_path):
    """THE HAZARD GUARD: a half-done passive checkpoint skips the truncate.

    Measured (``.perf/bench/lane2/wal_blocked_checkpoint.txt``): a
    ``wal_checkpoint(TRUNCATE)`` behind a reader pinning an old snapshot waited
    **5.183 s** — the connection's whole ``busy_timeout`` — and then gave up with
    ``(busy=1, log=3011, checkpointed=2)``. On the analytics writer thread that
    is a five-second stall per attempt, so the truncate must not be attempted
    when the passive pass reports a partial result.

    Asserted structurally, on a fake connection: a partial passive row means
    ``truncate_calls == 0``, so there is no wait to make. Note that a partial
    result reports ``busy == 0`` — ``log == checkpointed`` is the completeness
    test, and a gate written on the busy flag alone would let this through.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    conn = _RecordingConn(passive_row=(0, 3011, 2))  # backed up 2 of 3011 frames
    store._connect = lambda: conn  # type: ignore[method-assign]
    assert store.bound_wal() is False
    assert conn.truncate_calls == 0, "a truncate that had to wait was attempted anyway"
    assert not any(sql.startswith("PRAGMA busy_timeout=0") for sql in conn.sql)


def test_bound_wal_truncates_only_with_the_busy_handler_off(tmp_path):
    """A complete passive pass is what unlocks the truncate, and it cannot wait.

    The exact statement sequence is the guard: the truncate is issued between
    ``busy_timeout=0`` and the restore, so a reader arriving in that window can
    only REFUSE the truncate (measured: 0.000 s and ``(1, 3011, 2)``) instead of
    costing the writer its five seconds.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    conn = _RecordingConn(passive_row=(0, 3011, 3011))
    store._connect = lambda: conn  # type: ignore[method-assign]
    assert store.bound_wal() is True
    assert conn.sql == [
        f"PRAGMA journal_size_limit={_WAL_SIZE_LIMIT_BYTES}",
        "PRAGMA wal_checkpoint(PASSIVE)",
        "PRAGMA busy_timeout=0",
        "PRAGMA wal_checkpoint(TRUNCATE)",
        "PRAGMA busy_timeout=5000",
    ]


def test_bound_wal_leaves_a_pinned_wal_alone_and_still_reclaims_it_later(tmp_path):
    """Real SQLite: a pinned snapshot costs a refusal now and nothing later.

    The reader takes its snapshot BEFORE the frames are written, which is the
    only way to pin a backfill (a reader that arrives afterwards pins nothing and
    the checkpoint simply succeeds). The WAL is byte-for-byte unchanged by the
    refused sweep, and the sweep after the reader goes away reclaims it — so the
    maintenance is deferred, never lost.

    THE ONE TIMING BOUND, and why it is wall time: the regression it guards is a
    wait inside the busy handler, which is sleeping, not computing — CPU time
    would not see it at all. The healthy path does no I/O here (every frame is
    pinned, so the passive pass backfills nothing) and returns in microseconds;
    the regressed shape is the measured 5.18 s, so 2.0 s sits between them with
    room for a loaded runner. The structural guard for the same hazard is
    ``test_bound_wal_never_attempts_a_truncate_that_would_have_to_wait``.
    """
    db = tmp_path / "a.db"
    store = AnalyticsStore(db)
    store._connect()  # create the schema before the pinning reader opens it

    reader = sqlite3.connect(db)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM calls")  # the pinned snapshot
    try:
        wal = _grow_wal(store)
        pinned_bytes = wal.stat().st_size
        assert pinned_bytes > 0

        returned = threading.Event()
        outcome: list[bool] = []

        def sweep() -> None:
            try:
                outcome.append(store.bound_wal())
            finally:
                returned.set()

        thread = threading.Thread(target=sweep, daemon=True)
        thread.start()
        started = time.perf_counter()
        assert returned.wait(_BACKSTOP_S), "bound_wal never returned behind a pinned reader"
        elapsed = time.perf_counter() - started
        assert outcome == [False], "a pinned WAL reported itself as reclaimed"
        assert wal.stat().st_size == pinned_bytes, "the refused sweep changed the file"
        assert elapsed < 2.0, f"the sweep waited {elapsed:.2f}s behind a pinned reader"
    finally:
        reader.rollback()
        reader.close()

    assert store.bound_wal() is True
    assert wal.stat().st_size == 0
