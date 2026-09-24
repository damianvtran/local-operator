"""A contended ``attention.db`` may cost a completion, never the caller.

THE INCIDENT THIS PINS (2026-09-20, the operator's own log,
``~/Library/Logs/local-operator/mobile.log``): the mobile daemon's write path
opened the store with ``sqlite3.connect(self.path, timeout=2.0)`` -- a two
second busy timeout, no ``PRAGMA busy_timeout``, and one attempt -- while ~25
concurrent ``lop`` sessions, the daemon itself, the tunnel connector and the
browser bridge all publish completions into the one file. The 2 s expired first,
``publish`` raised a bare ``sqlite3.OperationalError`` straight out of the
caller, and the log ends::

    File ".../local_operator/session/attention.py", line 1297, in publish
        with closing(self._connect()) as conn, conn:
    File ".../local_operator/session/attention.py", line 909, in _connect
        with conn:
    sqlite3.OperationalError: database is locked
    ERROR:    ASGI callable returned without completing response.

The two halves of the fix are tested separately below: the store now waits in
the house range (``_BUSY_TIMEOUT_MS``, the value the sibling stores document) and
retries a contended write on a bounded budget, and a caller that treats a store
failure as fatal now degrades, observably.

THE READ CLASS GETS THE SAME TREATMENT, and had tests of its own added later
because it did not: 36 RECORDS of the incident's log are the DAEMON'S SCAN, a
read, against 12 records on the publish path, over 60 `database is locked` text
occurrences -- records and occurrences are different units, because a publish
record carries the phrase twice. A wider window left those
36 exactly as they were -- a read that raised at 2 s raised at 5 s instead. So
the reads are retried too (:meth:`AttentionStore._retry_read`), they get their
own typed verdict (:class:`AttentionReadDeferred`), and the retry set is asserted
to BE the classifier's set, because the version that shipped listed two of its
five members and let the other three escape as the bare error this file exists to
remove.

WHY THE FIRST TEST SPENDS ~2 SECONDS. It is the one test whose subject IS a
duration: the shipped 2 s window is what has to be shown losing a race the new
budget wins. Nothing is polled and no wall-clock assertion is made -- the lock is
released on the EVENT of the old shape's refusal, so the test is a sequence, not
a bet on machine load (see AGENTS.md, "Timing, flakes").
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

import local_operator.session.attention as attention
import local_operator.session.session as session_module
from local_operator.session.attention import (
    ATTENTION_CUSTOM_TYPE,
    AttentionReadDeferred,
    AttentionStore,
    AttentionWriteDeferred,
)
from local_operator.session.store_failures import (
    BUSY_ERRONAMES,
    STORE_BUSY,
    store_failure,
)

#: The shipped connection shape, quoted from ``_connect`` as it stood before the
#: fix (``sqlite3.connect(self.path, timeout=2.0)``). Written out rather than
#: derived so the "old" arm of the comparison cannot silently become the new one.
_SHIPPED_TIMEOUT_S = 2.0


class _HeldWriteLock:
    """A second connection holding SQLite's write lock on ``path``.

    This is what a sibling session's ``publish`` looks like from the outside: a
    ``BEGIN IMMEDIATE`` transaction that has not committed. The lock is released
    explicitly, so a test controls the window instead of racing a timer.
    """

    def __init__(self, path: Path) -> None:
        # `check_same_thread=False` only so a test can release from another
        # thread; the lock is identical either way.
        self._conn = sqlite3.connect(path, timeout=0.0, check_same_thread=False)
        self._conn.execute("BEGIN IMMEDIATE")
        self._conn.execute(
            "INSERT INTO completions(conversation,token,anchor,kind) VALUES(?,?,?,?)",
            ("session/a", str(uuid.uuid4()), "held-by-a-sibling-writer", "complete"),
        )

    def release(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


class _HeldExclusiveLock:
    """A sibling holding SQLite's EXCLUSIVE lock -- the one that blocks a READER.

    The write-path fixture above uses ``BEGIN IMMEDIATE``, which conflicts with a
    sibling WRITER but leaves readers free on a rollback-journal store: a reader
    that is not itself blocked is exactly how the read class went untested while
    the write class was covered. ``BEGIN EXCLUSIVE`` is the shape QA's daemon rig
    held for its read cells, and it is what the operator's log shows the daemon's
    own scan losing to.
    """

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path, timeout=0.0, check_same_thread=False)
        self._conn.execute("BEGIN EXCLUSIVE")
        self._conn.execute(
            "INSERT INTO completions(conversation,token,anchor,kind) VALUES(?,?,?,?)",
            ("session/b", str(uuid.uuid4()), "held-exclusively", "complete"),
        )

    def release(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


def _seed(path: Path) -> AttentionStore:
    """A store with its schema materialised, as any live machine's would be."""
    store = AttentionStore(path)
    store.publish("session/a", str(uuid.uuid4()), "anchor-first", "complete")
    return store


def _shrink_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A millisecond-scale retry budget, so the bounded case is not a wait."""
    monkeypatch.setattr(attention, "_BUSY_TIMEOUT_MS", 50)
    monkeypatch.setattr(attention, "_CONTENTION_BACKOFF_S", 0)


def _shrink_the_ladder(monkeypatch: pytest.MonkeyPatch, *delays: float) -> None:
    """Millisecond rungs, so the bounded case is a wait on the EVENT, not a clock.

    Patched on the SESSION MODULE, which is where the rung reads it: the delays are
    module-level precisely so a test can change how long the ladder waits without
    changing how many rungs it has (`_run_attention_republish` receives its delay
    as an argument, so the shipped tuple is what names the rung).
    """
    monkeypatch.setattr(session_module, "ATTENTION_REPUBLISH_DELAYS_S", delays)


def _completion_rows(path: Path, conversation: str) -> int:
    """How many completions the store holds for one conversation."""
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM completions WHERE conversation=?", (conversation,)
        ).fetchone()[0]


def _delivery_row(path: Path, conversation: str) -> tuple[object, ...] | None:
    """The conversation's delivery row, so "untouched" can be asserted exactly."""
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(
            "SELECT conversation, delivered, delivered_at, backend FROM deliveries "
            "WHERE conversation=?",
            (conversation,),
        ).fetchone()


async def _wait_for_store(
    path: Path, conversation: str, token: str, *, seconds: float = 30.0
) -> dict[str, Any]:
    """Wait for THAT token to be the store's completion, or fail saying where it stood."""
    store = AttentionStore(path)
    deadline = time.monotonic() + seconds
    state = store.state(conversation)
    while state["completion_token"] != token:
        if time.monotonic() >= deadline:
            raise AssertionError(f"{token} never reached the store; it holds {state}")
        await asyncio.sleep(0.02)
        state = store.state(conversation)
    return state


async def _wait_until(predicate, *, seconds: float = 30.0, what: str = "condition") -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.02)


def _contention_error(errorname: str) -> sqlite3.OperationalError:
    """A refusal shaped the way SQLite reports one: text, plus the certified code."""
    error = sqlite3.OperationalError("database is locked")
    error.sqlite_errorcode = sqlite3.SQLITE_BUSY
    error.sqlite_errorname = errorname
    return error


def test_a_held_write_lock_used_to_cost_the_completion_and_no_longer_does(
    tmp_path: Path,
) -> None:
    """The shipped 2 s window loses this race; the aligned one wins it.

    Both arms meet the SAME held lock, started within milliseconds of each
    other, and the release is triggered by the old shape's own refusal -- the
    event, not a sleep. The old arm is the connection shape ``_connect`` shipped:
    a 2 s driver timeout with no PRAGMA and a single attempt.
    """
    path = tmp_path / "attention.db"
    store = _seed(path)
    holder = _HeldWriteLock(path)
    token = str(uuid.uuid4())
    old_failures: list[BaseException] = []

    def old_shape() -> None:
        try:
            with closing(sqlite3.connect(path, timeout=_SHIPPED_TIMEOUT_S)) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO completions(conversation,token,anchor,kind) VALUES(?,?,?,?)",
                    ("session/a", token, "anchor-after", "complete"),
                )
        except BaseException as exc:  # noqa: BLE001 — recording the failure IS the assertion
            old_failures.append(exc)

    def new_shape() -> None:
        store.publish("session/a", token, "anchor-after", "complete")

    old = threading.Thread(target=old_shape)
    new = threading.Thread(target=new_shape)
    old.start()
    # Started while the lock is still held, so the store's first attempt meets it.
    new.start()
    try:
        old.join(timeout=30)
        assert not old.is_alive()
        # The old shape cannot outlast the holder: a BARE OperationalError, the
        # exact type the daemon's request handler died on.
        assert [type(error) for error in old_failures] == [sqlite3.OperationalError], old_failures
        holder.release()  # the event, not a clock
        new.join(timeout=30)
        assert not new.is_alive()
    finally:
        holder.release()
        holder.close()
    # The same race, through the fixed store: the completion is durable.
    assert AttentionStore(path).state("session/a")["completion_token"] == token


def test_a_lock_that_never_clears_is_a_classified_deferral_not_a_bare_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What escapes is named, bounded, and still classifies as contention.

    Three claims, and each is a separate way the original defect hurt:
    the type is specific (a caller can defer), the code is SQLite's own (the
    surface ladders keep answering "busy, retry" rather than "check the
    machine"), and the attempts are bounded (a caller always gets an answer).
    """
    _shrink_the_budget(monkeypatch)
    path = tmp_path / "attention.db"
    store = _seed(path)
    seeded = store.state("session/a")["completion_token"]
    attempts: list[int] = []
    original = AttentionStore._connect

    def counting(inner_self: AttentionStore) -> sqlite3.Connection:
        attempts.append(1)
        return original(inner_self)

    monkeypatch.setattr(AttentionStore, "_connect", counting)
    holder = _HeldWriteLock(path)
    try:
        with pytest.raises(AttentionWriteDeferred) as raised:
            store.publish("session/a", str(uuid.uuid4()), "anchor-after", "complete")
    finally:
        holder.release()
        holder.close()

    error = raised.value
    assert type(error).__name__ == "AttentionWriteDeferred"
    # Still an `sqlite3.OperationalError`, so every `except sqlite3.Error` ladder
    # in the tree keeps catching it -- which is why the type subclasses it.
    assert isinstance(error, sqlite3.OperationalError)
    assert error.sqlite_errorname == "SQLITE_BUSY"
    # And the shared classifier still reads it as contention: 503 "busy", never
    # 500 "the store is broken".
    classified = store_failure(error, tmp_path)
    assert classified is not None, "a SQLite error must classify as a store failure"
    assert classified.code == STORE_BUSY
    assert len(attempts) == attention._CONTENTION_ATTEMPTS, "the retry must be bounded"
    # Nothing half-written: a failed attempt leaves no row, and the previous
    # completion still stands.
    assert store.state("session/a")["completion_token"] == seeded


def test_a_non_contention_failure_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk is not a race, so it is reported on the first attempt.

    The retry exists for a lock; re-running an unopenable store or a corrupt
    schema would only delay the sentence the surface has for it.
    """
    _shrink_the_budget(monkeypatch)
    store = AttentionStore(tmp_path / "attention.db")
    attempts: list[int] = []

    def refuse() -> dict[str, object]:
        attempts.append(1)
        error = sqlite3.OperationalError("disk I/O error")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        error.sqlite_errorname = "SQLITE_IOERR"
        raise error

    with pytest.raises(sqlite3.OperationalError) as raised:
        store._retry_write(refuse)
    assert type(raised.value) is sqlite3.OperationalError, "not a deferral"
    assert len(attempts) == 1


def test_both_connection_helpers_carry_the_house_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write path AND the read-only path wait in the same range.

    Reads contend for the same lock in this store's default rollback journal:
    the operator's log shows the daemon's own scan losing exactly this race
    (`AttentionStore().revision` raising out of `_uninitialized`, 36 times), so
    the window is asserted on both helpers -- and against a patched value, which
    also pins that the policy is read per connection rather than frozen.
    """
    monkeypatch.setattr(attention, "_BUSY_TIMEOUT_MS", 1234)
    path = tmp_path / "attention.db"
    store = _seed(path)
    for connect in (store._connect, store._connect_read_only):
        conn = connect()
        try:
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
        finally:
            conn.close()


@pytest.mark.asyncio
async def test_the_turn_outcome_publish_degrades_instead_of_killing_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The outage path: a locked store during a turn's outcome publish.

    ``Session._publish_attention_outcome`` runs in the turn's ``finally`` on the
    runtime an ASGI request handler drives, so a raise there is the log's last
    line -- "ASGI callable returned without completing response" -- and it skips
    the rest of the teardown. The turn's own answer must survive a store it could
    not write to, the delay must be visible, and the durable journal must still
    hold the outcome -- which the REPUBLISH LADDER armed by this arm now retries
    against the live store in this process (see the section at the foot of this
    file), because the next boot's import was the only remedy and a finished
    session never boots again.
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        before = store.revision()
        holder = _HeldWriteLock(path)
        try:
            session._attention_outcome = AgentEndEvent(messages=[], error="Fixture failure")
            with caplog.at_level(logging.WARNING, logger="local_operator.session.session"):
                # NO RAISE: this is the whole fix, asserted at the caller.
                await session._publish_attention_outcome()
        finally:
            holder.release()
            holder.close()
        assert "deferred" in caplog.text, caplog.text
        # The write genuinely did not land -- the caller survived a real store
        # failure rather than a simulated one.
        assert store.revision() == before
        assert session._attention_republish_due, "a deferral must arm the ladder"
        # And the completion is not lost: the journal marker precedes the publish,
        # which is what the ladder and the next boot both republish from.
        saved = session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
        assert saved is not None and saved["conversation_id"] == identity
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# The retry set and the classifier's set are one fact, not two lists.
#
# `attention.py` retried on two hand-written names while `store_failures.py`
# classified five, so `SQLITE_BUSY_SNAPSHOT`, `SQLITE_BUSY_RECOVERY` and
# `SQLITE_LOCKED_SHAREDCACHE` escaped `publish` as a bare `OperationalError`
# while the very same ladder answered 503 for them. Parametrizing over the
# CLASSIFIER's set is what makes a re-narrowing fail a NAMED case instead of
# going unnoticed -- the drift is unrepresentable now (the set is imported), and
# these two tests are what would notice if somebody un-imported it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("errorname", sorted(BUSY_ERRONAMES))
def test_every_verdict_the_classifier_calls_contention_is_retried(
    errorname: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each contended verdict gets a second attempt and a typed deferral."""
    _shrink_the_budget(monkeypatch)
    assert (
        attention._CONTENTION_ERRONAMES == BUSY_ERRONAMES
    ), "the retry set must BE the classifier's set, not a narrower copy of it"
    assert attention._is_contention(_contention_error(errorname))
    store = AttentionStore(tmp_path / "attention.db")
    attempts: list[int] = []

    def refuse() -> None:
        attempts.append(1)
        raise _contention_error(errorname)

    with pytest.raises(AttentionWriteDeferred) as raised:
        store._retry_write(refuse)
    # SQLite's own verdict rides through, so the ladder keeps naming the
    # condition rather than the type.
    assert raised.value.sqlite_errorname == errorname
    assert len(attempts) == attention._CONTENTION_ATTEMPTS


@pytest.mark.parametrize("errorname", sorted(BUSY_ERRONAMES))
def test_every_contended_verdict_is_503_busy_at_the_ladder_and_never_500(
    errorname: str, tmp_path: Path
) -> None:
    """Retrying must not move a verdict's answer: contention is 503, never 500."""
    classified = store_failure(_contention_error(errorname), tmp_path)
    assert classified is not None, "a SQLite error must classify as a store failure"
    assert (classified.status, classified.code) == (503, STORE_BUSY)


def test_a_directly_constructed_deferral_already_carries_the_busy_code(
    tmp_path: Path,
) -> None:
    """The code belongs to the TYPE, not to the helper that happens to build it.

    ``AttentionWriteDeferred`` used to take only a message, with
    ``sqlite_errorcode``/``sqlite_errorname`` attached afterwards by ``_deferred``:
    a direct construction -- a caller, a rig, a future test -- classified as 500
    ``store_unavailable`` and told the operator to check the machine, which is
    the exact downgrade the type exists to prevent.
    """
    for deferred_type in (AttentionWriteDeferred, AttentionReadDeferred):
        error = deferred_type("attention store stayed busy through 2 attempts: database is locked")
        assert error.sqlite_errorname == "SQLITE_BUSY"
        assert error.sqlite_errorcode == sqlite3.SQLITE_BUSY
        classified = store_failure(error, tmp_path)
        assert classified is not None, "a SQLite error must classify as a store failure"
        assert (classified.status, classified.code) == (503, STORE_BUSY)


def test_a_third_attempt_is_a_retry_not_an_index_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attempt count and the wait between attempts are independent.

    The backoff used to be a per-attempt tuple, so the variant the constants
    discuss -- three attempts -- raised ``IndexError`` on the third instead of
    retrying. A scalar cannot be indexed out of range; this pins that raising the
    budget actually buys the attempts it says it does.
    """
    _shrink_the_budget(monkeypatch)
    monkeypatch.setattr(attention, "_CONTENTION_ATTEMPTS", 3)
    store = AttentionStore(tmp_path / "attention.db")
    attempts: list[int] = []

    def refuse() -> None:
        attempts.append(1)
        raise _contention_error("SQLITE_BUSY")

    with pytest.raises(AttentionWriteDeferred):
        store._retry_write(refuse)
    assert len(attempts) == 3


# ---------------------------------------------------------------------------
# The read class: 36 scan RECORDS (60 `database is locked` occurrences) in the
# incident's log.
# ---------------------------------------------------------------------------


def test_a_read_rides_out_a_lock_that_outlasts_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked read is RETRIED, not merely given a wider window.

    The operator's log is mostly this shape: the daemon's own scan
    (``revision -> _uninitialized``) meeting a sibling writer's lock. Pre-fix it
    raised a bare ``OperationalError``; with the wider window alone it still
    raised, five seconds later instead of two. The release is triggered by the
    read's own second connection -- the sequence, not a clock.
    """
    monkeypatch.setattr(attention, "_BUSY_TIMEOUT_MS", 60)
    monkeypatch.setattr(attention, "_CONTENTION_BACKOFF_S", 0)
    path = tmp_path / "attention.db"
    store = _seed(path)
    holder = _HeldExclusiveLock(path)
    original = AttentionStore._connect_read_only
    connects: list[int] = []

    def release_for_the_second_attempt(inner_self: AttentionStore) -> sqlite3.Connection:
        connects.append(1)
        if len(connects) == 2:
            # The event, not a sleep: attempt 1 has already paid its whole
            # window and failed, which is what makes this "outlasts the window".
            holder.release()
        return original(inner_self)

    monkeypatch.setattr(AttentionStore, "_connect_read_only", release_for_the_second_attempt)
    started = time.monotonic()
    try:
        revision = store.revision()
    finally:
        holder.release()
        holder.close()
    elapsed = time.monotonic() - started

    assert len(connects) == 2, "the read must have been attempted twice"
    assert elapsed >= 0.06, "the first attempt must have paid its whole window"
    assert revision[0] >= 1, "the retried read returns the real frame"


def test_a_read_that_never_gets_the_lock_is_a_classified_deferral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What escapes a blocked read is named and still classifies as contention."""
    _shrink_the_budget(monkeypatch)
    path = tmp_path / "attention.db"
    store = _seed(path)
    original = AttentionStore._connect_read_only
    attempts: list[int] = []

    def counting(inner_self: AttentionStore) -> sqlite3.Connection:
        attempts.append(1)
        return original(inner_self)

    monkeypatch.setattr(AttentionStore, "_connect_read_only", counting)
    holder = _HeldExclusiveLock(path)
    try:
        with pytest.raises(AttentionReadDeferred) as raised:
            store.revision()
    finally:
        holder.release()
        holder.close()

    error = raised.value
    assert isinstance(error, sqlite3.OperationalError)
    assert error.sqlite_errorname == "SQLITE_BUSY"
    assert len(attempts) == attention._CONTENTION_ATTEMPTS, "the retry must be bounded"
    # 503 "busy, retry" at every ladder that already classifies store failures --
    # a blocked read is never reported as a broken store.
    classified = store_failure(error, tmp_path)
    assert classified is not None, "a SQLite error must classify as a store failure"
    assert (classified.status, classified.code) == (503, STORE_BUSY)


def test_a_non_contention_read_failure_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt or unreadable store is not a race, so it is reported at once."""
    _shrink_the_budget(monkeypatch)
    store = AttentionStore(tmp_path / "attention.db")
    attempts: list[int] = []

    def refuse() -> dict[str, object]:
        attempts.append(1)
        error = sqlite3.OperationalError("disk I/O error")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        error.sqlite_errorname = "SQLITE_IOERR"
        raise error

    with pytest.raises(sqlite3.OperationalError) as raised:
        store._retry_read(refuse)
    assert type(raised.value) is sqlite3.OperationalError, "not a deferral"
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_the_refresh_read_degrades_instead_of_killing_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The read arm of the outage path, asserted at the caller.

    ``Session.refresh_attention`` runs on request paths (the runtime's refresh op,
    the mobile handle, the desktop poll), so a raise out of its state read is the
    same defect the write arm's deferral exists to remove -- one lock met an
    ASGI request handler and the request never completed. The previous state
    stands and the next tick re-reads it; the delay is logged, never silent.
    """
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        session._attention = store.state(identity)
        previous = dict(session._attention)
        holder = _HeldExclusiveLock(path)
        try:
            with caplog.at_level(logging.WARNING, logger="local_operator.session.session"):
                state = await session.refresh_attention()
        finally:
            holder.release()
            holder.close()
        assert state == previous, "the previous state stands rather than raising"
        assert "keeping the previous state" in caplog.text, caplog.text
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# The republish LADDER: a deferred completion is retried in THIS process.
#
# The store's own budget (~11 s at the shipped constants) is the bound that keeps
# a turn's `finally` prompt, and it stays exactly as it is. What was missing is
# what happens AFTER it: the only remedy used to be the next boot's
# `bootstrap_transcript`, and a session that finishes a turn and then sits idle --
# which is what a finished session IS -- never boots again. The operator saw the
# consequence rather than the cause (2026-09-23): completed sessions stopped
# raising OS notifications and stopped drawing the sidebar's "completed, unread"
# checkmark, because the store row that both of those read was never written.
#
# These tests drive the real paths: a real `Session`, a real held `BEGIN
# IMMEDIATE` from a sibling connection, the real publish, and the real rung.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_deferred_completion_is_republished_in_process_by_the_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The incident, repaired where the operator was waiting.

    A busy store costs the completion for ~11 s, never for the rest of the run: the
    journal marker is appended BEFORE the publish, so the ladder has something
    durable to republish, and it republishes the SAME token -- the store ends up
    holding the completion the turn meant to publish, with the anchor the turn
    chose, and neither watermark moving: `receipts` is untouched, and the delivery
    already claimed for the EARLIER completion is exactly as it was.

    THE ROW THIS INSERTS IS NEW, and that is the point of the ladder: a deferred
    completion never reached the store, so the republish is its FIRST insert and
    takes a fresh `sequence` -- which is why `_attention_publish_lock` serialises
    this read-then-insert against the session's own turn-end publication. What is
    genuinely idempotent is a republish of a token the store ALREADY holds
    (`INSERT OR IGNORE`): no second row, no second `sequence`, no duplicate
    notification (`claim_delivery` dedupes on that sequence). The old wording here
    claimed "no new `sequence`" of the deferred case too, which is false -- and
    believing it is what made the ordering hazard invisible (review round 1,
    MINOR-1).
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    # Rungs a second apart: the release below happens milliseconds after the
    # deferral arms this, so the ladder still has rungs left to spend on the store
    # that has just freed up.
    _shrink_the_ladder(monkeypatch, 1.0, 1.0, 1.0, 1.0)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        seeded = store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        assert store.claim_delivery(identity, seeded["completion_token"], "phone")
        delivered_before = _delivery_row(path, identity)
        rows_before = _completion_rows(path, identity)
        assert delivered_before is not None

        holder = _HeldWriteLock(path)
        try:
            session._attention_outcome = AgentEndEvent(messages=[], error="Fixture failure")
            with caplog.at_level(logging.WARNING, logger="local_operator.session.session"):
                await session._publish_attention_outcome()
            assert "deferred" in caplog.text, caplog.text
            assert session._attention_republish_due, "a deferral must arm the ladder"
            journal = session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
            assert journal is not None
        finally:
            holder.release()
            holder.close()

        # The lock is gone and NOTHING else happens -- no boot, no new turn, no
        # viewer: the ladder alone has to publish this.
        state = await _wait_for_store(path, identity, journal["token"])
        assert state["anchor_id"] == journal["anchor"] == f"completion-{journal['token']}"
        assert state["kind"] == "error"
        assert state["unseen"] is True, "a republished completion is still unread"
        assert _completion_rows(path, identity) == rows_before + 1, "one row, one token"
        assert _delivery_row(path, identity) == delivered_before, "deliveries untouched"
        await _wait_until(
            lambda: not session._attention_republish_due,
            what="the ladder to stand down once the store has it",
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_ladder_that_never_gets_the_store_gives_up_bounded_and_says_so_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Bounded rungs, one line when they run out, and no cost after that.

    The lock never clears here, which is the case the ladder must LOSE gracefully:
    it makes exactly one attempt per rung (`len(delays)` of them, and no more),
    says ONCE that only a boot will publish the completion, and then clears the
    latch -- an armed latch would make every viewer tick attempt the publish again,
    which is up to a full store budget per tick. The tick at the end is the
    assertion: while the store is still contended, it costs no attempt at all.
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    _shrink_the_ladder(monkeypatch, 0.0, 0.0, 0.0)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        seeded = store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        rows_before = _completion_rows(path, identity)
        attempts: list[str] = []
        real_publish = AttentionStore.publish

        def counting_publish(store_self, conversation, token, anchor, kind, **kwargs):
            attempts.append(token)
            return real_publish(store_self, conversation, token, anchor, kind, **kwargs)

        holder = _HeldWriteLock(path)
        try:
            # Consume the boot-restore FIRST, while there is nothing journalled to
            # restore: that costs no publish and leaves `_attention_restored` set,
            # so every attempt counted below belongs to the ladder alone.
            await session.refresh_attention()
            assert session._attention_restored and not session._attention_republish_due
            monkeypatch.setattr(AttentionStore, "publish", counting_publish)
            session._attention_outcome = AgentEndEvent(messages=[], error="Fixture failure")
            with caplog.at_level(logging.WARNING, logger="local_operator.session.session"):
                await session._publish_attention_outcome()
                assert len(attempts) == 1, "the turn's own publish is the only one yet"
                await _wait_until(
                    lambda: not session._attention_republish_due,
                    what="the ladder to exhaust itself against a held lock",
                )
            assert len(attempts) == 4, f"one bounded attempt per rung, got {attempts}"
            exhausted = [
                record
                for record in caplog.records
                if "only the next boot's import" in record.getMessage()
            ]
            assert len(exhausted) == 1, caplog.text
            # Losing the ladder is not losing the completion: the store is
            # unchanged, so nothing was half-written, and the journal still holds
            # the marker the next boot imports.
            assert store.state(identity)["completion_token"] == seeded["completion_token"]
            assert _completion_rows(path, identity) == rows_before
            assert session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE) is not None
            # The tick the exhausted ladder must not make expensive.
            await session.refresh_attention()
            assert len(attempts) == 4, f"an exhausted ladder must not re-arm: {attempts}"
        finally:
            holder.release()
            holder.close()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_newer_outcome_cancels_a_pending_republish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ladder must never revive a completion a later turn has replaced.

    The outcome is cleared at the top of ``_publish_attention_outcome``, before the
    newest journal append, so a rung that wakes after a newer turn published finds
    a clear latch and stops. Driving the rung DIRECTLY is the point: it is the same
    call the ladder makes, so the assertion is about what a rung does with a
    superseded deferral rather than about a timer happening not to fire.
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    _shrink_the_ladder(monkeypatch, 5.0, 5.0)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")

        holder = _HeldWriteLock(path)
        try:
            session._attention_outcome = AgentEndEvent(messages=[], error="First failure")
            await session._publish_attention_outcome()
            stale = session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
            assert stale is not None and session._attention_republish_due
        finally:
            holder.release()
            holder.close()

        # The next turn ends before any rung fires: its outcome is the newest fact
        # there is, and it publishes on its own.
        session._attention_outcome = AgentEndEvent(messages=[], error="Second failure")
        await session._publish_attention_outcome()
        newest = session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
        assert newest is not None and newest["token"] != stale["token"]
        assert not session._attention_republish_due, "a newer outcome cancels the pending republish"

        await session._run_attention_republish(0, 0.0)
        assert store.state(identity)["completion_token"] == newest["token"]
        assert _completion_rows(path, identity) == 2, "the superseded turn was never published"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_dispose_does_not_wait_for_a_parked_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown cancels the ladder rather than waiting out its rungs.

    The rungs are parked 30 s apart here and `dispose` is given a 5 s bound, so a
    regression is a `TimeoutError` rather than a teardown that hangs for a minute.
    The ladder runs through the same tracked-task machinery as every other
    background spawn for exactly this reason: `dispose` cancels it, so a session
    that quits with a deferral outstanding does not hold the process open.
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    _shrink_the_ladder(monkeypatch, 30.0, 30.0)
    session = make_session(tmp_path, ScriptedStream([]))
    path = config_dir() / "attention.db"
    # Materialise the store before a sibling takes its lock, and give it a row: the
    # ladder is the only thing that will publish the next one.
    AttentionStore(path).publish("session/sess", str(uuid.uuid4()), "anchor-earlier", "complete")
    holder = _HeldWriteLock(path)
    try:
        session._attention_outcome = AgentEndEvent(messages=[], error="Fixture failure")
        await session._publish_attention_outcome()
        assert session._attention_republish_due
        parked = session._attention_republish_task
        assert parked is not None
    finally:
        holder.release()
        holder.close()

    await asyncio.wait_for(session.dispose(), timeout=5.0)
    assert parked.done() and parked.cancelled(), "dispose must cancel the parked rung"


# ---------------------------------------------------------------------------
# The republish ladder's own failure modes, from review round 1.
#
# Three of these are about the same thing: a retry path that is supposed to be
# BOUNDED has to be bounded in every arm, not only the happy one. The fourth is
# the reason the first one matters -- a fresh `sequence` is exactly how an older
# completion can outrank a newer one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rung_cannot_outrank_a_newer_turn_that_publishes_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred OLDER completion must never take the newest `sequence`.

    `completions.sequence` is AUTOINCREMENT, i.e. INSERT order, and `state_many`
    reports `MAX(sequence)` as the conversation's newest completion -- so a rung
    that reads the journal, then loses the store to a NEWER same-session turn inside
    its `await`, would insert the older completion on top and the sidebar mark, the
    Active ordering and the newest-row pointer would all move to the older result.
    Once a client acknowledged that highest sequence, the newer completion would
    never be reported unread again: the exact symptom class this ladder exists to
    remove, re-entered through the window the ladder opens.

    The interleaving is injected at the one await boundary (`AttentionStore.publish`
    for the deferred token parks until this test says go), because only a
    same-process newer token can invert this conversation's order.
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    _shrink_the_ladder(monkeypatch, 30.0)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        # Consume the boot restore first, so the tick below reaches the LADDER.
        await session.refresh_attention()

        holder = _HeldWriteLock(path)
        try:
            session._attention_outcome = AgentEndEvent(messages=[], error="First failure")
            await session._publish_attention_outcome()
            older = (session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE) or {})["token"]
            assert session._attention_republish_due
        finally:
            holder.release()
            holder.close()

        real_publish = AttentionStore.publish
        entered = threading.Event()
        release_older = threading.Event()
        seen: list[str] = []

        def parked_publish(store_self, conversation, token, anchor, kind, **kwargs):
            # `publish` runs in a WORKER THREAD (`asyncio.to_thread`), so the park is
            # a thread-side wait: the event loop stays free to run the newer turn,
            # which is the interleaving this test is about.
            seen.append(token)
            if token == older:
                entered.set()
                release_older.wait(timeout=30)
            return real_publish(store_self, conversation, token, anchor, kind, **kwargs)

        monkeypatch.setattr(AttentionStore, "publish", parked_publish)
        # Drive the republish as a TASK, not through the tick: the tick fires the
        # ladder now and performed the publish before the fix (MAJOR-3), and this
        # test is about the republish path itself, which both trees share.
        republish = asyncio.create_task(session._republish_journalled_outcome())
        await _wait_until(entered.is_set, seconds=10, what="the republish to reach its insert")

        # The newer turn completes HERE, between that journal read and its insert --
        # the window MAJOR-1 is about.
        session._attention_outcome = AgentEndEvent(messages=[], error="Second failure")
        newer_publish = asyncio.create_task(session._publish_attention_outcome())
        # Give it a generous chance to land first. THE LOCK MAKES THAT IMPOSSIBLE
        # while the republish is inside its critical section, which is the point of
        # the fix -- so this is a bounded wait, never an assumption about who wins a
        # race: without the lock the newer turn lands here, and the release below
        # then stamps the OLDER completion on top of it.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            landed = [token for token in seen if token != older]
            if landed and store.state(identity)["completion_token"] == landed[0]:
                break
            await asyncio.sleep(0.02)
        release_older.set()
        await asyncio.wait_for(republish, timeout=30)
        await asyncio.wait_for(newer_publish, timeout=30)

        newer = (session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE) or {})["token"]
        assert newer != older
        with closing(sqlite3.connect(path)) as conn:
            sequences = dict(
                conn.execute(
                    "SELECT token, sequence FROM completions WHERE conversation=?",
                    (identity,),
                ).fetchall()
            )
        assert sequences[newer] > sequences[older], "the newer completion must outrank the older"
        state = store.state(identity)
        assert state["completion_token"] == newer, "the newest completion is the newer turn"
        assert state["unseen"] is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_non_contention_store_failure_disarms_the_latch_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken store is not a busy one: one attempt, one traceback, then stop.

    The helper returns `True` for a non-contention failure -- "nothing more is owed
    in this process" -- so the latch has to go with it. Left armed, every caller of
    `refresh_attention` (the runtime's 1 Hz loop, the TUI's 1 Hz poll, the mobile
    handle) re-read the journal, re-attempted the write and logged a traceback EVERY
    SECOND for the life of the session, which is the per-tick cost this ladder
    exists to bound (review round 1, MAJOR-2).
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    _shrink_the_ladder(monkeypatch, 0.0, 0.0, 0.0)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        await session.refresh_attention()

        holder = _HeldWriteLock(path)
        try:
            session._attention_outcome = AgentEndEvent(messages=[], error="Fixture failure")
            await session._publish_attention_outcome()
            assert session._attention_republish_due
        finally:
            holder.release()
            holder.close()

        attempts: list[str] = []

        def broken_publish(store_self, conversation, token, anchor, kind, **kwargs):
            attempts.append(token)
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(AttentionStore, "publish", broken_publish)
        with caplog.at_level(logging.WARNING, logger="local_operator.session.session"):
            for _ in range(3):
                await session.refresh_attention()
                await asyncio.sleep(0.02)
            assert len(attempts) == 1, f"a broken store must be tried once, got {attempts}"
            assert not session._attention_republish_due, "the latch must go with the attempt"
            tracebacks = [
                record
                for record in caplog.records
                if "could not republish the journalled outcome" in record.getMessage()
            ]
            assert len(tracebacks) == 1, "one traceback, not one per tick"
            # And the ticks after it cost nothing at all.
            for _ in range(3):
                await session.refresh_attention()
                await asyncio.sleep(0.02)
        assert len(attempts) == 1, f"a disarmed latch must not retry per tick: {attempts}"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_armed_latch_costs_a_tick_no_store_write_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tick FIRES the parked rung; it never performs its own publish.

    Publishing from the tick meant one full store budget (the desktop bridge
    measures ~10.8 s worst case) per tick, per client-attached session, for the
    whole contention window -- up to ~86 attempts at 1 Hz against the ladder's four
    rungs, on the store this module calls the most contended on the machine. The
    bound asserted here is the ladder's own rung count, which is the whole point of
    having a ladder (review round 1, MAJOR-3).
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    delays = (30.0, 30.0, 30.0, 30.0)
    _shrink_the_ladder(monkeypatch, *delays)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        await session.refresh_attention()

        attempts: list[str] = []
        real_publish = AttentionStore.publish

        def counting_publish(store_self, conversation, token, anchor, kind, **kwargs):
            attempts.append(token)
            return real_publish(store_self, conversation, token, anchor, kind, **kwargs)

        holder = _HeldWriteLock(path)  # held for the whole test: every rung defers
        try:
            session._attention_outcome = AgentEndEvent(messages=[], error="Fixture failure")
            await session._publish_attention_outcome()
            assert session._attention_republish_due
            monkeypatch.setattr(AttentionStore, "publish", counting_publish)
            for _ in range(8):
                await session.refresh_attention()
                await asyncio.sleep(0.05)
            await _wait_until(
                lambda: not session._attention_republish_due,
                what="the ladder to spend its rungs and stand down",
            )
            assert 0 < len(attempts) <= len(delays), (
                f"8 ticks must cost the ladder's {len(delays)} rungs, not one write "
                f"each: {attempts}"
            )
            spent = len(attempts)
            for _ in range(3):
                await session.refresh_attention()
                await asyncio.sleep(0.02)
            assert len(attempts) == spent, "a disarmed latch costs a tick nothing"
        finally:
            holder.release()
            holder.close()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_fresh_deferral_inside_the_last_rung_restarts_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A deferral that arrives under a spent ladder gets a budget of its own.

    `_schedule_attention_republish` returns early while a rung is in flight, so a
    deferral landing during the FINAL rung used to inherit that rung's spent budget
    and then be cleared along with the latch the exhausted arm takes -- the
    completion (and the log line blaming "4 republish attempts") then waited for a
    boot that an idle session never has, which is the defect this PR fixes, under a
    condition a fleet-wide storm can produce (review round 1, MINOR-2).

    The newer marker is journalled from inside the rung's own publish, which is
    exactly where a newer turn's outcome lands.
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    _shrink_the_ladder(monkeypatch, 30.0)  # one rung: the parked rung IS the last one
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        await session.refresh_attention()

        holder = _HeldWriteLock(path)
        try:
            session._attention_outcome = AgentEndEvent(messages=[], error="First failure")
            await session._publish_attention_outcome()
            older = (session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE) or {})["token"]
            assert session._attention_republish_due
        finally:
            holder.release()
            holder.close()

        newest_marker = {
            "conversation_id": identity,
            "token": str(uuid.uuid4()),
            "anchor": "anchor-from-the-newer-turn",
            "kind": "complete",
            "cause": "",
            "reason": "",
        }
        real_publish = AttentionStore.publish
        entered = threading.Event()
        release_thread = threading.Event()
        injected = {"done": False}

        def injecting_publish(store_self, conversation, token, anchor, kind, **kwargs):
            # Thread-side again: the rung's insert runs in a worker thread, so this
            # parks the WRITE while the test's loop appends the newer turn's marker
            # underneath it -- exactly where a newer outcome lands.
            if token == older and not injected["done"]:
                injected["done"] = True
                entered.set()
                release_thread.wait(timeout=30)
            return real_publish(store_self, conversation, token, anchor, kind, **kwargs)

        holder = _HeldWriteLock(path)  # held for the WHOLE test: every attempt defers
        try:
            first_task = session._attention_republish_task
            monkeypatch.setattr(AttentionStore, "publish", injecting_publish)
            with caplog.at_level(logging.WARNING, logger="local_operator.session.session"):
                await session.refresh_attention()  # fires the parked last rung
                await _wait_until(entered.is_set, seconds=10, what="the rung to reach its insert")
                # The newer turn journals while this rung is trying to insert.
                await session._transcript.append_custom(ATTENTION_CUSTOM_TYPE, newest_marker)
                release_thread.set()
                # The rung spends its last attempt and gives up -- and the handle it
                # replaces is the proof that the budget was RESTARTED, not cleared.
                await _wait_until(
                    lambda: session._attention_republish_task is not first_task,
                    seconds=10,
                    what="the restarted ladder to replace the spent rung",
                )
            assert session._attention_republish_due, "the newer deferral keeps its own budget"
            assert not any(
                "only the next boot's import" in record.getMessage() for record in caplog.records
            ), "a restart must not blame the ladder that just gave up"
            assert session._attention_republish_task is not None
            assert not session._attention_republish_task.done()
            # With the store free again, the restarted ladder publishes the marker
            # the journal holds now -- the newer turn's, not the one it gave up on.
            holder.release()
            await session.refresh_attention()
            await _wait_for_store(path, identity, newest_marker["token"])
        finally:
            release_thread.set()
            holder.release()
            holder.close()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_ladder_handle_is_dropped_once_it_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tracked handle does not outlive the ladder (review round 1, NIT-1).

    A rung spawns its successor and the successor becomes the ladder, so the clear
    is guarded on identity rather than blind -- but when the LAST rung is gone the
    session must not keep it (and its coroutine frame) for the rest of its life.
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.paths import config_dir
    from tests.unit.session.test_session import ScriptedStream, make_session

    _shrink_the_budget(monkeypatch)
    _shrink_the_ladder(monkeypatch, 0.0, 0.0)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        path = config_dir() / "attention.db"
        store = AttentionStore(path)
        identity = "session/sess"
        store.publish(identity, str(uuid.uuid4()), "anchor-earlier", "complete")
        await session.refresh_attention()

        holder = _HeldWriteLock(path)
        try:
            session._attention_outcome = AgentEndEvent(messages=[], error="Fixture failure")
            await session._publish_attention_outcome()
            assert session._attention_republish_task is not None
        finally:
            holder.release()
            holder.close()

        token = (session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE) or {})["token"]
        await _wait_for_store(path, identity, token)
        await _wait_until(
            lambda: session._attention_republish_task is None,
            seconds=5.0,
            what="the finished ladder's handle to be dropped",
        )
    finally:
        await session.dispose()
