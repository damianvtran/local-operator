"""The background indexer: incremental scan, coalescing, backfill, roster, locks.

The OS event stream is deliberately NOT under test — FSEvents is not reproducible
in CI. What is tested is the contract the watcher feeds: a dirty set, a journal
signature, a roster identity, and the queue that turns a burst into one scan.
"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path
from typing import Any

from local_operator.library import indexer, service
from local_operator.library.metadata import MetadataStore
from local_operator.library.store import LibraryStore
from local_operator.session.attachments import cache_media

from ._fixtures import (
    OTHER,
    SESSION,
    config_root,
    image_rows,
    png,
    session,
    user_row,
    write_rows,
)

CHILD = "0123456789ab"
SECOND = "222222222222"


def names(root: Path, **kwargs: Any) -> set[str]:
    frame = service.list_assets(root, service.Query(**kwargs))
    return {row["name"] for row in frame["rows"]}


def written(root: Path, name: str) -> Path:
    path = root.parent / name
    path.write_text(name)
    return path


def test_incremental_update_after_a_journal_append(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    first = written(root, "first.txt")
    directory = session(root, SESSION, write_rows("w1", 1.0, first))
    indexer.rebuild_now(root)
    assert names(root) == {"first.txt"}
    revision = LibraryStore(root).revision()

    second = written(root, "second.txt")
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        for row in write_rows("w2", 2.0, second):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    index = indexer.LibraryIndexer(root)
    assert index.scan_session(SESSION) is True
    assert names(root) == {"first.txt", "second.txt"}
    assert LibraryStore(root).revision() > revision

    # A second pass over an unchanged journal is free and changes nothing.
    assert index.scan_session(SESSION) is False
    assert LibraryStore(root).revision() > revision


def test_a_burst_coalesces_into_one_batch(tmp_path, monkeypatch) -> None:
    """The dirty SET, not the event stream: 40 events for one session = one scan."""
    root = config_root(monkeypatch, tmp_path)
    index = indexer.LibraryIndexer(root)
    monkeypatch.setattr(indexer, "SETTLE_S", 0.02)
    monkeypatch.setattr(indexer, "MAX_COALESCE_S", 0.05)
    for number in range(40):
        index._mark_dirty(f"session{number}")
    batch, wake = index.coalesce("first")
    assert wake is False
    assert len(batch) == 41
    assert index._queue.empty(), "the batch must consume the whole burst"

    scanned: list[set[str]] = []
    monkeypatch.setattr(
        index, "scan_session", lambda session_id: scanned.append({session_id}) or True
    )
    index.process_dirty(batch)
    assert len(scanned) == 41


def test_backfill_is_resumable_and_reports_partial_progress(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    for number, session_id in enumerate((SESSION, OTHER, SECOND)):
        path = written(root, f"file{number}.txt")
        session(root, session_id, write_rows(f"w{number}", float(number + 1), path))

    first = indexer.LibraryIndexer(root)
    assert first.rebuild() == 3
    state = indexer.index_state(root)
    assert state["backfill_done"] == "1"
    assert (state["total"], state["scanned"]) == (3, 3)
    assert names(root) == {"file0.txt", "file1.txt", "file2.txt"}

    # A restart re-scans nothing that still matches its checkpoint.
    store = LibraryStore(root)
    assert indexer.LibraryIndexer(root).rebuild() == 0

    # Losing ONE checkpoint costs exactly one session, not the whole catalog.
    with store.write() as connection:
        connection.execute("DELETE FROM idx_sessions WHERE session_id = ?", (OTHER,))
        store.set_state(connection, backfill_done="0")
    assert indexer.LibraryIndexer(root).rebuild() == 1
    assert names(root) == {"file0.txt", "file1.txt", "file2.txt"}


def test_a_roster_only_change_indexes_the_new_child(tmp_path, monkeypatch) -> None:
    """A child appears by roster edit alone; the parent journal never changes."""
    root = config_root(monkeypatch, tmp_path)
    parent_file = written(root, "parent.txt")
    parent = session(root, SESSION, write_rows("p1", 1.0, parent_file))
    child_dir = session(root, CHILD, write_rows("c1", 2.0, written(root, "child.txt")))
    (child_dir / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    (parent / "subagent-roster.v1.json").write_text(json.dumps(_roster([child_dir])))
    indexer.rebuild_now(root)
    assert names(root) == {"parent.txt", "child.txt"}

    extra = session(root, SECOND, write_rows("c2", 3.0, written(root, "grandchild.txt")))
    (extra / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    before = indexer.index_state(root)
    (parent / "subagent-roster.v1.json").write_text(json.dumps(_roster([child_dir, extra])))

    # The transcript is untouched: only the roster moved.
    assert indexer.LibraryIndexer(root).scan_session(SESSION) is True
    assert names(root) == {"parent.txt", "child.txt", "grandchild.txt"}
    assert indexer.index_state(root)["revision"] != before["revision"]


def test_removing_a_session_drops_its_assets(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    keep = written(root, "keep.txt")
    gone = written(root, "gone.txt")
    session(root, SESSION, write_rows("w1", 1.0, keep))
    directory = session(root, OTHER, write_rows("w2", 2.0, gone))
    indexer.rebuild_now(root)
    assert names(root) == {"keep.txt", "gone.txt"}

    import shutil

    shutil.rmtree(directory)
    indexer.LibraryIndexer(root).reconcile()

    assert names(root) == {"keep.txt"}
    assert LibraryStore(root).session(OTHER) is None
    assert LibraryStore(root).counts("all")["category"] == {"document": 1}


def test_only_one_process_writes_one_config(tmp_path, monkeypatch) -> None:
    """``flock`` is the cross-process rule; the singleton is the in-process one."""
    root = config_root(monkeypatch, tmp_path)
    lock_path = root / "cache" / "library" / "indexer.lock"
    first = indexer._ProcessLock(lock_path)
    second = indexer._ProcessLock(lock_path)
    assert first.acquire() is True
    try:
        assert second.acquire() is False, "a second writer must be refused, not queued"
    finally:
        first.release()
    assert second.acquire() is True
    second.release()

    # In-process: one object per resolved config, and it holds the writer lock.
    a = indexer.ensure_started(root)
    b = indexer.ensure_started(root)
    assert a is b
    assert indexer.writer_available(root) is True
    a.stop()
    assert indexer.writer_available(root) is False


def test_thumbnails_are_keyed_by_content_and_width_and_invalidate(tmp_path, monkeypatch) -> None:
    """Pillow is mocked: this pins the KEY and the invalidation rule, not JPEG."""
    root = config_root(monkeypatch, tmp_path)
    image = root.parent / "photo.png"
    image.write_bytes(b"png-bytes")
    session(root, SESSION, write_rows("w1", 1.0, image))
    indexer.rebuild_now(root)
    asset_id = service.list_assets(root, service.Query())["rows"][0]["id"]
    store = LibraryStore(root)

    saved: list[tuple[str, str, int]] = []

    class FakeImage:
        width = 1200
        height = 800
        mode = "RGB"

        def __enter__(self) -> "FakeImage":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def load(self) -> None:
            return None

        def resize(self, size: tuple[int, int], _resample: Any) -> "FakeImage":
            saved.append(("resize", f"{size[0]}x{size[1]}", 0))
            return self

        def convert(self, mode: str) -> "FakeImage":
            return self

        def save(self, target: Any, fmt: str, quality: int) -> None:
            saved.append((fmt, str(target), quality))
            Path(str(target)).write_bytes(b"webp")

    fake_image = types.SimpleNamespace(
        open=lambda path: FakeImage(), Resampling=types.SimpleNamespace(LANCZOS=1)
    )
    # The module's own seam, not ``sys.modules["PIL"]``: swapping the real
    # package leaked into later tests and made their generation "unavailable".
    monkeypatch.setattr("local_operator.library.store._image_module", lambda: fake_image)

    made = store.ensure_thumbnail(asset_id, width=480, quality=82)
    assert made is not None
    assert made.endswith("-480.webp"), "the key is the content hash plus the width"
    assert saved[0] == ("resize", "480x320", 0)
    assert saved[1][0] == "WEBP" and saved[1][2] == 82
    assert store.thumb_is_live(asset_id, 480) == Path(made)
    assert store.thumb_is_live(asset_id, 240) is None, "a different width is a different file"

    # A source that moved invalidates the cached row, not just the file.
    image.write_bytes(b"png-bytes-v2")
    assert store.thumb_is_live(asset_id, 480) is None
    saved.clear()
    store.ensure_thumbnail(asset_id, width=480, quality=82)
    assert saved, "a stale thumbnail must be regenerated"

    # SVG/GIF are served original: no rasterisation, no cache entry.
    svg = root.parent / "icon.svg"
    svg.write_text("<svg/>")
    session(root, OTHER, write_rows("w2", 2.0, svg))
    indexer.rebuild_now(root)
    svg_id = next(
        row["id"]
        for row in service.list_assets(root, service.Query())["rows"]
        if row["name"] == "icon.svg"
    )
    assert store.ensure_thumbnail(svg_id, width=480, quality=82) is None


def _roster(children: list[Path]) -> dict[str, Any]:
    return {
        "version": 1,
        "generation": 1,
        "jobs": [],
        "accounting": {},
        "records": [
            {
                "job_id": f"j{number}",
                "label": "scout",
                "agent_role": "scout",
                "depth": 1,
                "session_dir": str(child),
            }
            for number, child in enumerate(children)
        ],
    }


def test_a_session_dir_with_no_journal_does_not_hold_the_backfill_open(
    tmp_path, monkeypatch
) -> None:
    """The denominator counts scannable sessions, not directories.

    The operator's real config contains ``sessions/000000000000`` — an empty
    directory. Counting it would leave ``backfill_done`` at 0 forever and the
    grid on "scanning" over a finished index.
    """
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "real.txt")))
    session(root, OTHER)  # a directory, no transcript.jsonl

    indexer.rebuild_now(root)
    state = indexer.index_state(root)
    assert state["state"] == "ready" and state["backfill_done"] == "1"
    assert (state["total"], state["scanned"]) == (1, 1)
    assert names(root) == {"real.txt"}


def test_a_reader_only_process_writes_nothing(tmp_path, monkeypatch) -> None:
    """Second process: reads the database, never generates thumbnails."""
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "photo.png")))
    indexer.rebuild_now(root)

    # Hand the writer role away, exactly as if a daemon in another process held
    # it: this process must then stay a reader.
    indexer._release_writer(root)
    blocker = indexer._ProcessLock(root / "cache" / "library" / "indexer.lock")
    assert blocker.acquire() is True
    try:
        index = indexer.LibraryIndexer(root)
        index.start()
        try:
            assert indexer.writer_available(root) is False
            assert index._thumb_thread is None, "a reader must not run the writer thread"
            index.enqueue_thumbnail("la_0123456789abcdef")
            assert index._thumb_queue.empty()
        finally:
            index.stop()
    finally:
        blocker.release()


def test_a_deleted_session_directory_is_still_a_session(tmp_path, monkeypatch) -> None:
    """The removal event arrives after the directory is gone; it must queue.

    Reproduced on the round-1 head: ``existing dir -> id`` but
    ``deleted dir -> None``, so a deleted conversation's assets stayed listed
    until the 6 h reconcile.
    """
    import shutil

    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "gone.txt")))
    handler = indexer._JournalHandler(root / "sessions", lambda _sid: None)

    assert handler.session_id(str(root / "sessions" / SESSION)) == SESSION
    assert handler.session_id(str(root / "sessions" / SESSION / "transcript.jsonl")) == SESSION
    shutil.rmtree(root / "sessions" / SESSION)
    assert handler.session_id(str(root / "sessions" / SESSION)) == SESSION, "deleted directory"
    # A stray FILE in ``sessions/`` is still not a session.
    (root / "sessions" / "notes.txt").write_text("x")
    assert handler.session_id(str(root / "sessions" / "notes.txt")) is None


def test_a_deleted_session_leaves_the_library_immediately(tmp_path, monkeypatch) -> None:
    """End to end: the event runs ``_forget``, not a 6 h wait."""
    import shutil

    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "keep.txt")))
    session(root, OTHER, write_rows("w2", 2.0, written(root, "gone.txt")))
    indexer.rebuild_now(root)
    # No lazy worker: a concurrent reconcile that enumerated before the delete
    # would write the old denominator back after this test's decrement.
    indexer.reset_for_tests()

    index = indexer.LibraryIndexer(root)
    index.reconcile()
    assert LibraryStore(root).counts("all")["category"] == {"document": 2}
    handler = indexer._JournalHandler(root / "sessions", index._mark_dirty)
    shutil.rmtree(root / "sessions" / OTHER)
    handler.on_any_event(types.SimpleNamespace(src_path=str(root / "sessions" / OTHER)))

    first = index._queue.get(timeout=1.0)
    index.process_dirty({first})
    assert names(root) == {"keep.txt"}
    store = LibraryStore(root)
    assert store.session(OTHER) is None
    # Q-2: the deleted session also left the denominator, on the spot — the old
    # check ran after ``forget_session`` deleted the row it looked for, so this
    # only ever passed because a lazy worker reconciled behind the test.
    state = indexer.index_state(root)
    assert state["total"] == 1
    assert store.counts("all")["category"] == {"document": 1}


def test_a_reader_only_process_takes_over_when_the_writer_stops(tmp_path, monkeypatch) -> None:
    """Q-1: retry on a short bound, not the 6 h idle cadence."""
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "keep.txt")))
    indexer.rebuild_now(root)
    monkeypatch.setattr(indexer, "WRITER_RETRY_S", 0.05)

    indexer._release_writer(root)
    blocker = indexer._ProcessLock(root / "cache" / "library" / "indexer.lock")
    assert blocker.acquire() is True

    reader = indexer.LibraryIndexer(root)
    reader.start()
    try:
        assert indexer.writer_available(root) is False
        assert reader._thumb_thread is None
        # Touch the journal FIRST: an unchanged one is skipped by signature, so
        # the assertion below would pass without proving anything. A changed
        # journal is work the reader-only process must refuse to write.
        journal = root / "sessions" / SESSION / "transcript.jsonl"
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(user_row("u2", 9.0, "one more turn"), separators=(",", ":")) + "\n"
            )
        # Nothing may be written while another process owns the index: a scan
        # attempt must leave the revision and the database file untouched.
        with LibraryStore(root).read() as connection:
            revision_before = connection.execute(
                "SELECT value FROM idx_state WHERE key = 'revision'"
            ).fetchone()[0]
        mtime_before = (root / "library" / "library.sqlite3").stat().st_mtime_ns
        reader.scan_session(SESSION)
        with LibraryStore(root).read() as connection:
            revision_after = connection.execute(
                "SELECT value FROM idx_state WHERE key = 'revision'"
            ).fetchone()[0]
        assert revision_after == revision_before
        assert (root / "library" / "library.sqlite3").stat().st_mtime_ns == mtime_before

        blocker.release()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not indexer.writer_available(root):
            time.sleep(0.02)
        assert indexer.writer_available(root) is True, "the reader must retry the lock"
        # Ownership now starts the thumbnail worker it could not run before.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and reader._thumb_thread is None:
            time.sleep(0.02)
        assert reader._thumb_thread is not None
        assert revision_before is not None
    finally:
        reader.stop()


def test_the_thumbnail_queue_is_deduplicated(tmp_path, monkeypatch) -> None:
    """A grid asks for the same tile on every poll; the queue must not grow."""
    root = config_root(monkeypatch, tmp_path)
    image = root.parent / "photo.png"
    image.write_bytes(png())
    session(root, SESSION, write_rows("w1", 1.0, image))
    indexer.rebuild_now(root)
    index = indexer.ensure_started(root)
    try:
        index.enqueue_thumbnail("la_0123456789abcdef", 480)
        index.enqueue_thumbnail("la_0123456789abcdef", 480)
        index.enqueue_thumbnail("la_0123456789abcdef", 240)
        assert index._thumb_queue.qsize() == 2, "one entry per (asset, width)"
        assert ("la_0123456789abcdef", 480) in index._thumb_pending
    finally:
        index.stop()


def test_the_thumbnail_sweep_puts_kept_images_first(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    scratch = written(root, "scratch.png")
    scratch.write_bytes(png())
    kept = written(root, "kept.png")
    kept.write_bytes(png())
    # A generated artifact is 'final'; a written path asset is not.
    block = cache_media(png(), "image/png", kind="image", name="kept.png")
    assert block and block.attachment
    session(
        root,
        SESSION,
        [
            *write_rows("w1", 1.0, scratch),
            *image_rows("g1", 2.0, block.attachment),
        ],
    )
    indexer.rebuild_now(root)
    candidates = LibraryStore(root).thumbnail_candidates(width=480)
    assert len(candidates) == 2
    assert (
        candidates[0][0]
        == service.list_assets(root, service.Query(origin_groups=frozenset({"generated"})))["rows"][
            0
        ]["id"]
    )


def test_an_unknown_session_id_does_not_move_the_revision(tmp_path, monkeypatch) -> None:
    """A top-level file under ``sessions/`` maps to an id the index never knew.

    Deleting one must not bump the listing revision: that would reset a
    paginating client's cursor for a cleanup-log rotation.
    """
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "keep.txt")))
    indexer.rebuild_now(root)
    indexer.reset_for_tests()
    index = indexer.LibraryIndexer(root)
    index.reconcile()
    before = indexer.index_state(root)["revision"]

    index._forget("last_cleanup.json")

    after = indexer.index_state(root)
    assert after["revision"] == before
    assert after["total"] == 1


def test_recovery_wakes_a_parked_worker(tmp_path, monkeypatch) -> None:
    """A rebuilt index must refill now, not on the next 6 h tick.

    The worker parks on ``queue.get`` with the idle cadence; recovery resets
    ``backfill_done`` from the request thread, so without an explicit wake the
    page would answer "building" with no rows until that timeout expired.
    """
    from local_operator.library.store import IndexUnavailable, LibraryStore

    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "keep.txt")))
    indexer.rebuild_now(root)
    indexer.reset_for_tests()
    indexer.ensure_started(root)

    original = LibraryStore.query_page
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        # Twice: the list retries once before it commits to recovery.
        if calls["n"] < 2:
            calls["n"] += 1
            raise IndexUnavailable("database disk image is malformed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LibraryStore, "query_page", flaky)
    assert service.list_assets(root, service.Query())["state"] == "building"

    # Recovery already happened inside that call; the WAKE must make the worker
    # reconcile rather than wait out its timer.
    deadline = time.monotonic() + 5.0
    rows: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        rows = service.list_assets(root, service.Query())["rows"]
        if rows:
            break
        time.sleep(0.05)
    assert rows, "the worker must rebuild immediately after recovery"


def test_a_wake_mid_batch_is_not_a_session_id(tmp_path, monkeypatch) -> None:
    """QA Q-2: ``coalesce`` absorbed the sentinel and the reconcile was lost."""
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "keep.txt")))
    indexer.rebuild_now(root)
    index = indexer.LibraryIndexer(root)

    # A wake that arrives while a batch is settling still flags the reconcile
    # and never appears in the dirty set.
    index._queue.put(indexer._WAKE)
    dirty, wake = index.coalesce(SESSION)
    assert wake is True
    assert dirty == {SESSION}

    dirty, wake = index.coalesce(indexer._WAKE)
    assert (dirty, wake) == (set(), True)
    assert indexer._WAKE not in dirty


def test_a_wake_during_a_batch_reaches_the_denominator(tmp_path, monkeypatch) -> None:
    """QA Q-R4-1: a recovery that swaps the file mid-batch must still finish.

    Recovery drops every derived table AND the checkpoints, so the batch that
    the pre-recovery reconcile queued is rescanned from scratch. The build was
    never marked done: every finish check during the scans saw a 0 denominator,
    and the wake reconcile that restored it did not call the finish check.
    """
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "keep.txt")))
    session(root, OTHER, write_rows("w2", 2.0, written(root, "other.txt")))
    indexer.rebuild_now(root)

    index = indexer.LibraryIndexer(root)
    store = LibraryStore(root)
    # Recovery's reset, in full: derived rows, checkpoints and counters.
    with store.write() as connection:
        store.set_state(
            connection,
            **{"sessions_total": "0", "backfill_done": "0", "sessions_scanned": "0"},
        )
        connection.execute("DELETE FROM idx_occurrences")
        connection.execute("DELETE FROM idx_assets")
        connection.execute("DELETE FROM idx_sessions")

    index.reconcile()
    # Nothing has been scanned yet: the catalog must NOT be called done.
    assert index._queue.qsize() == 2
    index._maybe_finish_backfill()
    assert store.state()["backfill_done"] == "0"

    first = index._queue.get(timeout=1)
    batch, _wake = index.coalesce(first)
    index._batch_active = True
    index.process_dirty(batch)
    index._batch_active = False
    index._queue.put(indexer._WAKE)
    batch, wake = index.coalesce(index._queue.get(timeout=1))
    assert wake is True
    index.reconcile()

    state = indexer.index_state(root)
    assert state["total"] == 2 and state["scanned"] == 2
    assert state["backfill_done"] == "1"
    assert state["state"] == "ready"
    assert len(service.list_assets(root, service.Query())["rows"]) == 2


def test_backfill_never_finishes_on_a_reset_denominator(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1.0, written(root, "keep.txt")))
    indexer.rebuild_now(root)
    store = LibraryStore(root)
    index = indexer.LibraryIndexer(root)

    with store.write() as connection:
        store.set_state(connection, **{"sessions_total": "0", "backfill_done": "0"})
    index._maybe_finish_backfill()
    assert store.state()["backfill_done"] == "0"

    # An EMPTY sessions directory may finish on 0: a fresh install is done.
    (tmp_path / "empty").mkdir()
    empty = config_root(monkeypatch, tmp_path / "empty")
    LibraryStore(empty).ensure_schema()
    indexer.LibraryIndexer(empty)._maybe_finish_backfill()
    assert LibraryStore(empty).state()["backfill_done"] == "1"


def test_a_fresh_database_is_wal_after_first_use(tmp_path, monkeypatch) -> None:
    """WAL is set at bootstrap; the connection pragma must not be per-connect."""
    import sqlite3

    root = config_root(monkeypatch, tmp_path)
    LibraryStore(root).ensure_schema()
    with sqlite3.connect(root / "library" / "library.sqlite3") as raw:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    MetadataStore(root).ensure_schema()
    with sqlite3.connect(root / "library" / "library.sqlite3") as raw:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_a_sessions_dir_with_only_housekeeping_files_is_ready(tmp_path, monkeypatch) -> None:
    """R4-2: a non-session file beside zero journals is a finished empty index."""
    root = config_root(monkeypatch, tmp_path)
    sessions = root / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / "last_cleanup.json").write_text("{}")

    index = indexer.LibraryIndexer(root)
    index.reconcile()

    state = indexer.index_state(root)
    assert state["backfill_done"] == "1", "nothing is scannable, so nothing is owed"
    assert state["total"] == 0 and state["state"] == "ready"
    # The idle cadence too: 60 s forever was the visible symptom.
    assert index._idle_timeout() == indexer.RECONCILE_IDLE_S
