"""Review-boundary regressions for the machine-wide desktop feed.

WHICH OF THESE REPRODUCE, AND WHICH IS ONLY A PIN (read this before trusting a
red/green run). Six of the seven FAIL against the pinned head `a8bed4a67` with
``_db_fingerprint`` present but NOT wired — the state this remediation started
from — and are therefore genuine reproductions of reviewer findings R1-R5.
Measured, not asserted: with the pinned ``_tick`` and the inert helper restored
the run is ``6 failed, 1 passed``, and the six are
``test_a_commit_that_moves_only_the_wal_is_still_a_doorbell_delta`` (R1),
``test_catalogue_timer_runs_while_attention_database_is_quiet`` (R2),
``test_late_subscriber_uses_durable_not_envelope_sequence`` and
``test_pre_subscription_unpolled_publication_is_not_replayed`` (R3),
``test_healed_outcome_emits_corrected_attention_without_new_banner`` (R4) and
``test_transient_state_read_failure_does_not_consume_publication`` (R5).

Against the pinned commit as it stands the module does not even COLLECT:
``_db_fingerprint`` does not exist there, so the import at the top of this file
fails. That is the strongest form of the R1 statement and the weakest form of
evidence — restore the helper before measuring anything.

The seventh, ``test_a_closing_writer_checkpoints_the_retained_wal_into_the_main_file``,
PASSES on the inert head: the reviewer's R1 report (a poll landing between a
writer's ``_connect`` touch and its commit caching the new main-file fingerprint
against the old database contents) did NOT reproduce through THAT test, because
the writer it uses is ``publish``, which CLOSES its connection — and closing the
last connection checkpoints the WAL back into the main file, so the pinned
single-file doorbell saw the change after all. It is kept as a PIN on the
boundary, but nobody should read its green as "R1 is fixed".

The genuine interleaving, with the WAL retained as a live serve process retains
it, is ``test_a_commit_that_moves_only_the_wal_is_still_a_doorbell_delta``
below: it reproduces the defect against the pinned doorbell and proves the cure
in the same test.

The WAL/journal awareness R1 asks for is genuinely absent from the pinned head
(``_tick`` fingerprints ``attention.db`` alone). ``_db_fingerprint`` — the
sidecar-aware doorbell in the module under test — plus the bounded authoritative
revision read in ``_tick`` are what close it.

Seeded through committed SQLite writes and the real ``AttentionStore``, not by
inventing feed frames: a hand-built frame would let these tests pass against a
producer that never emits one.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator.server.utils import desktop_feed as feed_module
from local_operator.server.utils.desktop_feed import (
    DesktopFeed,
    _db_fingerprint,
    _fingerprint,
)
from local_operator.session.attention import AttentionStore, provisional_anchor
from tests.notification_opt_in import notification_path_opt_in


@pytest.fixture(autouse=True)
def notification_path_on() -> Iterator[None]:
    """This suite's subject IS the ``notification`` frame, so it opts in.

    Both process-wide gates suppress it otherwise: the kill switch armed by
    ``tests/conftest.py`` (no frame is composed at all) and, for the legs whose
    sessions carry a journal, the test-hosting rule. These tests fabricate
    session ids with no journal, so only the first applies here — the shared
    helper clears both, because a module that later seeds a real selection must
    not have to remember a second variable. See ``tests/notification_opt_in``
    for why this is a module-level fixture and what the escape can and cannot
    do.
    """
    with notification_path_opt_in():
        yield


def publish(
    root: Path, sid: str, token: str | None = None, anchor: str = "done", kind: str = "complete"
) -> str:
    (root / "sessions" / sid).mkdir(parents=True, exist_ok=True)
    token = token or str(uuid.uuid4())
    AttentionStore(root / "attention.db").publish(f"session/{sid}", token, anchor, kind)
    return token


def tick(feed: DesktopFeed) -> None:
    asyncio.run(feed._tick())


def drain(sub):
    frames = []
    while not sub.queue.empty():
        frames.append(sub.queue.get_nowait())
    sub.queued_sizes.clear()
    sub.queued_bytes = 0
    return frames


def test_late_subscriber_uses_durable_not_envelope_sequence(tmp_path):
    feed = DesktopFeed(tmp_path)
    feed.subscribe()
    feed._take_baseline()
    for index in range(2):
        publish(tmp_path, f"{index:012x}")
        tick(feed)
    late = feed.subscribe()
    publish(tmp_path, "cccccccccccc")
    tick(feed)
    assert any(f["type"] == "notification" for f in drain(late))


def test_pre_subscription_unpolled_publication_is_not_replayed(tmp_path):
    feed = DesktopFeed(tmp_path)
    feed.subscribe()
    feed._take_baseline()
    for index in range(8):
        publish(tmp_path, f"{index:012x}")
    late = feed.subscribe()
    tick(feed)
    assert not any(f["type"] == "notification" for f in drain(late))


def test_catalogue_timer_runs_while_attention_database_is_quiet(tmp_path):
    publish(tmp_path, "aaaaaaaaaaaa")
    feed = DesktopFeed(tmp_path)
    sub = feed.subscribe()
    feed._take_baseline()
    tick(feed)
    drain(sub)
    (tmp_path / "sessions" / "bbbbbbbbbbbb").mkdir()
    feed._catalogue_probed_at = 0
    tick(feed)
    assert any(f["type"] == "catalogue" for f in drain(sub))


def test_healed_outcome_emits_corrected_attention_without_new_banner(tmp_path):
    feed = DesktopFeed(tmp_path)
    sub = feed.subscribe()
    feed._take_baseline()
    token = str(uuid.uuid4())
    publish(tmp_path, "aaaaaaaaaaaa", token, provisional_anchor(token), "interrupted")
    tick(feed)
    drain(sub)
    publish(tmp_path, "aaaaaaaaaaaa", token, "real-result", "complete")
    tick(feed)
    frames = drain(sub)
    assert any(f["type"] == "attention" and f["payload"]["kind"] == "complete" for f in frames)
    assert not any(f["type"] == "notification" for f in frames)


def test_transient_state_read_failure_does_not_consume_publication(tmp_path, monkeypatch):
    feed = DesktopFeed(tmp_path)
    sub = feed.subscribe()
    feed._take_baseline()
    publish(tmp_path, "aaaaaaaaaaaa")
    original = feed.store.state_many

    def fail(_identities):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(feed.store, "state_many", fail)
    with pytest.raises(sqlite3.OperationalError):
        tick(feed)
    monkeypatch.setattr(feed.store, "state_many", original)
    tick(feed)
    assert any(f["type"] == "notification" for f in drain(sub))


def test_a_closing_writer_checkpoints_the_retained_wal_into_the_main_file(tmp_path, monkeypatch):
    """A PIN on the boundary, NOT an R1 reproduction (see the module docstring).

    Named for what it actually pins (review round 2, N4). The old name —
    ``test_commit_after_connect_touch_invalidates_retained_wal`` — read as an R1
    repro, which is the reading the module docstring spends a paragraph warning
    against, and a later edit could have lost the honest labelling while the
    name kept asserting the opposite.
    """
    publish(tmp_path, "aaaaaaaaaaaa")
    held = sqlite3.connect(tmp_path / "attention.db")
    held.execute("PRAGMA journal_mode=WAL")
    feed = DesktopFeed(tmp_path)
    sub = feed.subscribe()
    feed._take_baseline()
    tick(feed)
    drain(sub)
    connected = threading.Event()
    release = threading.Event()
    original = AttentionStore._connect

    def paused(store):
        conn = original(store)
        connected.set()
        assert release.wait(5)
        return conn

    monkeypatch.setattr(AttentionStore, "_connect", paused)
    writer = threading.Thread(target=publish, args=(tmp_path, "bbbbbbbbbbbb"))
    writer.start()
    try:
        assert connected.wait(5)
        tick(feed)
        drain(sub)
        release.set()
        writer.join(5)
        assert not writer.is_alive()
        tick(feed)
        assert any(f["type"] == "notification" for f in drain(sub))
    finally:
        release.set()
        writer.join(5)
        held.close()


def test_a_commit_that_moves_only_the_wal_is_still_a_doorbell_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1, derived rather than inherited: the doorbell cannot be the main file.

    The finding is an ORDERING one, so the test drives the ordering explicitly:
    the writer's ``_connect`` touch happens, a poll lands in the window after it,
    and only THEN does the transaction commit. Two facts are asserted about that
    commit, and together they ARE the finding:

    * the main file's stat tuple is UNCHANGED — a doorbell reading it alone
      learns nothing, and asserting this is what makes the blindness a fact in
      the test rather than a claim in a comment;
    * a notification still reaches the subscriber, which is only possible
      because the fingerprint now covers the sidecar that DID move.

    Phase A runs the same interleaving with the PINNED doorbell restored and
    asserts the completion is lost, reproducing the defect on the production
    ``_tick`` path; phase B is the fixed doorbell and asserts the frame arrives.
    Both phases are needed — without A this would only be a test that a frame
    appears, which says nothing about the bug it exists for. The writer's
    connection stays OPEN until after the assertion, because closing the last
    connection checkpoints ``-wal`` into the main file and would quietly repair
    the very blindness under test.
    """

    def run(doorbell, root: Path) -> tuple[bool, bool, bool]:
        root.mkdir(parents=True, exist_ok=True)
        publish(root, "dddddddddddd")
        monkeypatch.setattr(feed_module, "_db_fingerprint", doorbell)

        # THE RETAINED WAL, which is the precondition the whole finding rests on
        # and the thing ``publish`` cannot supply. Two settings, both load-bearing:
        # WAL mode puts the commit in the sidecar instead of rewriting the main
        # file, and ``wal_autocheckpoint=0`` keeps it there. This connection stays
        # open for the entire phase, because SQLite checkpoints on the last
        # close — which is precisely how the boundary pin above passes on both
        # heads, and why reproducing R1 needs a writer that outlives its commit.
        keeper = sqlite3.connect(root / "attention.db", timeout=5.0)
        keeper.execute("PRAGMA journal_mode=WAL")
        keeper.execute("PRAGMA wal_autocheckpoint=0")
        keeper.execute("SELECT COUNT(*) FROM completions").fetchone()

        feed = DesktopFeed(root)
        feed._take_baseline()
        subscription = feed.subscribe()
        tick(feed)
        drain(subscription)

        # THE TOUCH: a real ``_connect``, so the main file moves exactly as
        # production moves it, and then a poll lands and caches what it saw.
        writer = AttentionStore(root / "attention.db")
        conn = writer._connect()
        main_after_touch = _fingerprint(root / "attention.db")
        sidecars_before = _db_fingerprint(root / "attention.db")
        tick(feed)
        drain(subscription)

        # THE COMMIT, after the poll: this is the window R1 describes. Plain
        # sqlite rather than ``store.publish`` because the point is the
        # file-level effect of a commit, which is what the doorbell sees.
        conn.execute(
            "INSERT INTO completions(conversation,token,anchor,kind) VALUES(?,?,?,?)",
            ("session/eeeeeeeeeeee", str(uuid.uuid4()), "done", "complete"),
        )
        conn.commit()

        main_unmoved = _fingerprint(root / "attention.db") == main_after_touch
        sidecar_moved = _db_fingerprint(root / "attention.db") != sidecars_before

        tick(feed)
        delivered = any(f["type"] == "notification" for f in drain(subscription))
        # Closed only now, so the close-checkpoint cannot flatter the result: the
        # write had to be seen while it was still only in ``-wal``.
        keeper.close()
        conn.close()
        return delivered, main_unmoved, sidecar_moved

    def pinned_doorbell(path: Path):
        """The pinned head's doorbell: ``_tick`` fingerprinted the main file."""
        return (_fingerprint(path),)

    delivered_old, main_unmoved, sidecar_moved = run(pinned_doorbell, tmp_path / "pinned")
    assert main_unmoved, "the commit moved the main file; the window did not reproduce"
    assert sidecar_moved, "the commit moved no sidecar; there was nothing to notice"
    assert not delivered_old, "the pinned doorbell saw a WAL-only commit; R1 not reproduced"

    delivered, _, _ = run(_db_fingerprint, tmp_path / "fixed")
    assert delivered, "a WAL-only commit still does not reach the subscriber"
