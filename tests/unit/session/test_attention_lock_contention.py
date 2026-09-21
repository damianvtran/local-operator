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

WHY THE FIRST TEST SPENDS ~2 SECONDS. It is the one test whose subject IS a
duration: the shipped 2 s window is what has to be shown losing a race the new
budget wins. Nothing is polled and no wall-clock assertion is made -- the lock is
released on the EVENT of the old shape's refusal, so the test is a sequence, not
a bet on machine load (see AGENTS.md, "Timing, flakes").
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from contextlib import closing
from pathlib import Path

import pytest

import local_operator.session.attention as attention
from local_operator.session.attention import (
    ATTENTION_CUSTOM_TYPE,
    AttentionStore,
    AttentionWriteDeferred,
)
from local_operator.session.store_failures import STORE_BUSY, store_failure

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


def _seed(path: Path) -> AttentionStore:
    """A store with its schema materialised, as any live machine's would be."""
    store = AttentionStore(path)
    store.publish("session/a", str(uuid.uuid4()), "anchor-first", "complete")
    return store


def _shrink_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A millisecond-scale retry budget, so the bounded case is not a wait."""
    monkeypatch.setattr(attention, "_BUSY_TIMEOUT_MS", 50)
    monkeypatch.setattr(attention, "_WRITE_BACKOFF_S", (0, 0))


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
    assert len(attempts) == attention._WRITE_ATTEMPTS, "the retry must be bounded"
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
    hold the outcome for the next boot's import.
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
        # And the completion is not lost: the journal marker precedes the publish
        # exactly so the next boot re-imports what the store refused.
        saved = session._transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
        assert saved is not None and saved["conversation_id"] == identity
    finally:
        await session.dispose()
