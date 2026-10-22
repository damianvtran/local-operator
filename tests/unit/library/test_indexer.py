"""The background indexer: incremental scan, coalescing, backfill, roster, locks.

The OS event stream is deliberately NOT under test — FSEvents is not reproducible
in CI. What is tested is the contract the watcher feeds: a dirty set, a journal
signature, a roster identity, and the queue that turns a burst into one scan.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

from local_operator.library import indexer, service
from local_operator.library.store import LibraryStore

from ._fixtures import (
    OTHER,
    SESSION,
    config_root,
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

    fake_pil = types.ModuleType("PIL")
    fake_image = types.SimpleNamespace(
        open=lambda path: FakeImage(), Resampling=types.SimpleNamespace(LANCZOS=1)
    )
    fake_pil.Image = fake_image  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

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
