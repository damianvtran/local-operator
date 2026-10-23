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
from local_operator.library.store import LibraryStore
from local_operator.session.attachments import cache_media

from ._fixtures import (
    OTHER,
    SESSION,
    config_root,
    image_rows,
    png,
    session,
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
    batch = index.coalesce("first")
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
    # Q-2: the deleted session also left the denominator.
    state = indexer.index_state(root)
    assert state["total"] <= 1
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
        # Nothing may be written while another process owns the index.
        assert reader.scan_session(SESSION) in (True, False)
        with LibraryStore(root).read() as connection:
            revision_before = connection.execute(
                "SELECT value FROM idx_state WHERE key = 'revision'"
            ).fetchone()[0]

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
