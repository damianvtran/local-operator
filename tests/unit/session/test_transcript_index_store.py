"""The incremental index store: snapshot + checksummed delta + small head.

WHY THESE TESTS EXIST. ``write_index`` used to serialise the WHOLE derived
document on every refresh; on a real 470 MB journal that was 62.9 MB per
one-row append. The store publishes only the re-derived suffix as a delta frame
and commits by replacing a small head. These tests pin what that must not cost:
every reader still sees exactly the document a whole write would have produced
(paging/tail order, search docs, rows/ranges), a torn delta is never read, a
damaged committed frame sends the reader to a rescan rather than a wrong answer,
a concurrent reader always sees one whole publication, and an existing v8
document migrates without a forced rebuild.

The store only engages above ``_STORE_MIN_BYTES``; the fixture lowers it so a
small synthetic journal exercises the real path.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import transcript_index as ti

SID = "aabbccddee02"


@pytest.fixture(autouse=True)
def _store(monkeypatch):
    ti._reset_for_tests()
    monkeypatch.setattr(ti, "_STORE_MIN_BYTES", 0)
    yield
    ti._reset_for_tests()


def _journal(root: Path) -> Path:
    return root / "sessions" / SID / "transcript.jsonl"


def _rows(start: int, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in range(start, start + count):
        rows.append(
            {
                "id": f"u{turn:05d}",
                "ts": float(turn),
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"text": f"question {turn} needle{turn}"}],
                },
            }
        )
        rows.append(
            {
                "id": f"a{turn:05d}",
                "ts": turn + 0.5,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": f"answer {turn}"}],
                },
            }
        )
    return rows


def _append(root: Path, rows: list[dict[str, Any]]) -> None:
    path = _journal(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _comparable(index: ti.TranscriptIndex) -> dict[str, Any]:
    """Everything a reader consumes, minus the per-scan lineage token."""
    document = ti._document(index)
    document.pop("journal_generation")
    document["journal_ranges"] = {k: list(v) for k, v in index.journal_ranges.items()}
    return json.loads(json.dumps(document))


def _full_scan(root: Path) -> ti.TranscriptIndex:
    return ti._scan_full(_journal(root), None)


def _store_of(root: Path) -> dict[str, Any] | None:
    head = json.loads(ti.index_path(root, SID).read_text())
    return head.get("store")


def _cold(root: Path) -> ti.TranscriptIndex | None:
    """A reader in ANOTHER process: no parse memo, only the disk."""
    with ti._PARSED_LOCK:
        ti._PARSED.clear()
    return ti.read_index(root, SID)


def test_appends_publish_frames_and_a_cold_reader_sees_the_full_scan(tmp_path):
    _append(tmp_path, _rows(0, 40))
    assert ti.refresh_index(tmp_path, SID) is not None
    store = _store_of(tmp_path)
    assert store is not None and store["frames"] == 0, "the first build is a snapshot"
    head_bytes = []
    for step in range(5):
        _append(tmp_path, _rows(40 + step, 1))
        assert ti.refresh_index(tmp_path, SID) is not None
        head_bytes.append(ti.index_path(tmp_path, SID).stat().st_size)
    store = _store_of(tmp_path)
    assert store is not None and store["frames"] == 5, store
    # The commit point stays a few hundred bytes however long the history is.
    assert max(head_bytes) < 2048, head_bytes
    cold = _cold(tmp_path)
    assert cold is not None
    assert _comparable(cold) == _comparable(_full_scan(tmp_path))


def test_tail_paging_order_and_search_docs_survive_the_store(tmp_path):
    _append(tmp_path, _rows(0, 30))
    ti.refresh_index(tmp_path, SID)
    for step in range(3):
        _append(tmp_path, _rows(30 + 2 * step, 2))
        ti.refresh_index(tmp_path, SID)
    cold = _cold(tmp_path)
    assert cold is not None
    # Ordering: the rail and the tail anchor read checkpoints/messages by seq.
    seqs = [m.seq for m in cold.messages]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    assert [c.seq for c in cold.checkpoints] == sorted(c.seq for c in cold.checkpoints)
    # Paging from the tail: the last page's ids map to the journal ordinals the
    # full scan assigns, and their byte spans are the same.
    full = _full_scan(tmp_path)
    tail_ids = [m.id for m in cold.messages[-10:]]
    assert [cold.journal_rows[i] for i in tail_ids] == [full.journal_rows[i] for i in tail_ids]
    assert [cold.journal_ranges[i] for i in tail_ids] == [full.journal_ranges[i] for i in tail_ids]
    # Search reads MessageDoc text: a needle from a frame-only row is found.
    assert any("needle35" in m.text for m in cold.messages)
    assert cold.scan == full.scan


def test_a_torn_delta_tail_is_ignored_and_the_next_append_repairs_it(tmp_path):
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    _append(tmp_path, _rows(20, 1))
    ti.refresh_index(tmp_path, SID)
    store = _store_of(tmp_path)
    assert store is not None and store["frames"] == 1
    delta = ti._store_paths(ti._index_dir(tmp_path), SID, store)[1]
    # A writer died mid-append: bytes past the committed length, never headed.
    with delta.open("ab") as handle:
        handle.write(b'99999 deadbeef\n{"keep":')
    before = _cold(tmp_path)
    assert before is not None and before.coverage["last_id"] == "a00020"
    _append(tmp_path, _rows(21, 1))
    assert ti.refresh_index(tmp_path, SID) is not None
    store = _store_of(tmp_path)
    assert store is not None and store["frames"] == 2
    assert delta.stat().st_size == store["length"], "the torn tail was not truncated"
    cold = _cold(tmp_path)
    assert cold is not None
    assert _comparable(cold) == _comparable(_full_scan(tmp_path))


def test_a_crash_before_the_head_commit_serves_the_previous_publication(tmp_path, monkeypatch):
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    committed = ti.index_path(tmp_path, SID).read_bytes()
    _append(tmp_path, _rows(20, 1))
    real_replace = Path.replace

    def crash(self, target):
        if Path(target) == ti.index_path(tmp_path, SID):
            raise OSError("simulated crash before the head replace")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", crash)
    ti.refresh_index(tmp_path, SID)  # frame durable, head never replaced
    monkeypatch.setattr(Path, "replace", real_replace)
    assert ti.index_path(tmp_path, SID).read_bytes() == committed
    old = _cold(tmp_path)
    assert old is not None and old.coverage["last_id"] == "a00019"
    # The next refresh re-derives the same suffix over the orphaned frame bytes.
    rebuilt = ti.refresh_index(tmp_path, SID)
    assert rebuilt is not None
    cold = _cold(tmp_path)
    assert cold is not None
    assert _comparable(cold) == _comparable(_full_scan(tmp_path))


def test_a_damaged_committed_frame_is_unusable_and_the_refresh_rebuilds(tmp_path):
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    _append(tmp_path, _rows(20, 1))
    ti.refresh_index(tmp_path, SID)
    store = _store_of(tmp_path)
    assert store is not None
    delta = ti._store_paths(ti._index_dir(tmp_path), SID, store)[1]
    data = bytearray(delta.read_bytes())
    data[-5] ^= 0x01  # flip a bit INSIDE the committed frame body
    delta.write_bytes(bytes(data))
    assert _cold(tmp_path) is None, "a frame failing its CRC must never be served"
    rebuilt = ti.refresh_index(tmp_path, SID)
    assert rebuilt is not None
    assert _comparable(rebuilt) == _comparable(_full_scan(tmp_path))
    assert _cold(tmp_path) is not None


def test_a_missing_snapshot_is_unusable_not_a_crash(tmp_path):
    _append(tmp_path, _rows(0, 10))
    ti.refresh_index(tmp_path, SID)
    store = _store_of(tmp_path)
    assert store is not None
    ti._store_paths(ti._index_dir(tmp_path), SID, store)[0].unlink()
    assert _cold(tmp_path) is None
    assert ti.refresh_index(tmp_path, SID) is not None
    assert _cold(tmp_path) is not None


def test_a_concurrent_reader_always_sees_one_whole_publication(tmp_path):
    _append(tmp_path, _rows(0, 30))
    ti.refresh_index(tmp_path, SID)
    stop = threading.Event()
    seen: list[tuple[int, int]] = []
    errors: list[BaseException] = []

    def reader():
        while not stop.is_set():
            try:
                raw = ti._read_raw(tmp_path, SID)  # never the memo: always the disk
                index = ti._index_from_raw(raw)
                if index is None:
                    errors.append(AssertionError("a reader saw an unusable publication"))
                    return
                # A whole publication: the scan's row total matches its facts and
                # every message's ordinal is a known row.
                assert index.scan.rows == len(index.source_projection_facts)
                assert all(index.journal_rows.get(m.id) == m.seq for m in index.messages)
                seen.append((index.scan.rows, len(index.messages)))
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)
                return

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for step in range(25):
            _append(tmp_path, _rows(30 + step, 1))
            ti.refresh_index(tmp_path, SID)
    finally:
        stop.set()
        thread.join(10)
    assert not errors, errors[:1]
    assert seen, "the reader never completed a read"


def test_compaction_folds_the_delta_into_a_new_snapshot_and_retires_the_old(tmp_path, monkeypatch):
    monkeypatch.setattr(ti, "_DELTA_COMPACT_FRAMES", 3)
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    first = _store_of(tmp_path)
    assert first is not None
    for step in range(4):
        _append(tmp_path, _rows(20 + step, 1))
        ti.refresh_index(tmp_path, SID)
    store = _store_of(tmp_path)
    assert store is not None and store["snapshot"] != first["snapshot"]
    assert store["frames"] <= 1
    names = sorted(p.name for p in ti._index_dir(tmp_path).iterdir())
    assert not any(first["snapshot"] in name for name in names), names
    assert len([n for n in names if ".snap." in n]) == 1, names
    assert len([n for n in names if ".delta." in n]) <= 1, names
    cold = _cold(tmp_path)
    assert cold is not None
    assert _comparable(cold) == _comparable(_full_scan(tmp_path))


def test_a_classic_v8_document_migrates_by_link_without_a_rebuild(tmp_path, monkeypatch):
    """Migration: an existing whole document reads as before, and the first
    append links it as the snapshot instead of re-serialising it."""
    _append(tmp_path, _rows(0, 20))
    monkeypatch.setattr(ti, "_STORE_MIN_BYTES", 1 << 40)  # write the classic layout
    ti.refresh_index(tmp_path, SID)
    path = ti.index_path(tmp_path, SID)
    classic = json.loads(path.read_text())
    assert "store" not in classic and classic["version"] == ti.TRANSCRIPT_INDEX_VERSION
    old_inode = path.stat().st_ino
    monkeypatch.setattr(ti, "_STORE_MIN_BYTES", 0)
    ti._reset_for_tests()  # a fresh process after the upgrade
    assert ti.read_index(tmp_path, SID) is not None, "the v8 document must still read"
    scans: list[int] = []
    real_full = ti._scan_full
    monkeypatch.setattr(ti, "_scan_full", lambda *a: scans.append(1) or real_full(*a))
    _append(tmp_path, _rows(20, 1))
    assert ti.refresh_index(tmp_path, SID) is not None
    assert scans == [], "migration forced a full rescan"
    store = _store_of(tmp_path)
    assert store is not None and store["frames"] == 1
    snapshot = ti._store_paths(ti._index_dir(tmp_path), SID, store)[0]
    assert snapshot.stat().st_ino == old_inode, "the snapshot is the linked v8 file"
    cold = _cold(tmp_path)
    assert cold is not None
    assert _comparable(cold) == _comparable(_full_scan(tmp_path))


def test_a_naming_patch_rewrites_only_the_head(tmp_path):
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    _append(tmp_path, _rows(20, 1))
    ti.refresh_index(tmp_path, SID)
    before = _store_of(tmp_path)
    assert ti.patch_naming(tmp_path, SID, {"u00003": {"name": "Named", "summary": "s"}})
    after = _store_of(tmp_path)
    assert after == before, "a naming-only write appended or compacted"
    cold = _cold(tmp_path)
    assert cold is not None and cold.naming["items"]["u00003"]["name"] == "Named"


def test_the_receipt_names_the_head_and_moves_on_every_publication(tmp_path):
    """``_publication_identity`` (patch_naming's compare-and-merge receipt) is the
    HEAD's identity, so it moves on a frame-only publication exactly as it moved
    on a whole rewrite."""
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    first = ti._read_raw(tmp_path, SID)
    assert first is not None
    _append(tmp_path, _rows(20, 1))
    ti.refresh_index(tmp_path, SID)
    second = ti._read_raw(tmp_path, SID)
    assert second is not None
    st = ti.index_path(tmp_path, SID).stat()
    assert second["_publication_identity"] == (st.st_ino, st.st_size, st.st_mtime_ns)
    assert first["_publication_identity"] != second["_publication_identity"]
    served = ti.read_index(tmp_path, SID)
    assert served is not None
    assert served.publication_identity == second["_publication_identity"]


def test_the_resume_head_prices_hydration_from_the_store(tmp_path):
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    _append(tmp_path, _rows(20, 1))
    ti.refresh_index(tmp_path, SID)
    store = _store_of(tmp_path)
    assert store is not None
    head = ti._read_resume_head(tmp_path, SID)
    assert head is not None
    head_bytes = ti.index_path(tmp_path, SID).stat().st_size
    assert head[3] == head_bytes + store["snapshot_bytes"] + store["length"]


def test_sweep_missing_reclaims_store_files_of_deleted_sessions_only(tmp_path):
    _append(tmp_path, _rows(0, 10))
    ti.refresh_index(tmp_path, SID)
    directory = ti._index_dir(tmp_path)
    orphan = directory / f"deadbeef0001.snap.{'0' * 32}.snapshot"
    orphan.write_text("{}")
    live = sorted(p.name for p in directory.iterdir() if p.name != orphan.name)
    ti._sweep_missing(tmp_path)
    assert not orphan.exists()
    assert sorted(p.name for p in directory.iterdir()) == live


def test_a_shrunk_journal_rebuilds_into_a_fresh_snapshot(tmp_path):
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    first = _store_of(tmp_path)
    path = _journal(tmp_path)
    lines = path.read_text().splitlines(keepends=True)
    temp = path.with_suffix(".rewrite")
    temp.write_text("".join(lines[:10]))
    os.replace(temp, path)  # compact_file's shape: a NEW inode
    rebuilt = ti.refresh_index(tmp_path, SID)
    assert rebuilt is not None
    store = _store_of(tmp_path)
    assert store is not None and first is not None and store["snapshot"] != first["snapshot"]
    cold = _cold(tmp_path)
    assert cold is not None
    assert _comparable(cold) == _comparable(_full_scan(tmp_path))


def _origin_main_sweep(config_dir: Path) -> None:
    """VERBATIM predicate of ``_sweep_missing`` as shipped before the store
    (origin/main 756d853a4bf, transcript_index.py): every older build that can
    still be running beside this one sweeps the shared cache directory so."""
    entries = list((config_dir / "cache" / "transcript_index").iterdir())
    sessions = config_dir / "sessions"
    for entry in entries:
        if entry.suffix != ".json":
            continue
        try:
            if not (sessions / entry.stem).is_dir():
                entry.unlink()
        except OSError:
            continue


def test_an_old_builds_sweep_leaves_store_files_alone(tmp_path):
    """Review R2 / QA Q1: a pre-store build's full scan must not strip snapshots."""
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    _append(tmp_path, _rows(20, 1))
    ti.refresh_index(tmp_path, SID)
    store = _store_of(tmp_path)
    assert store is not None and store["frames"] == 1
    snapshot, delta = ti._store_paths(ti._index_dir(tmp_path), SID, store)
    _origin_main_sweep(tmp_path)
    assert snapshot.exists() and delta.exists(), "an old build's sweep deleted the store"
    cold = _cold(tmp_path)
    assert cold is not None
    assert _comparable(cold) == _comparable(_full_scan(tmp_path))


def test_a_read_only_probe_does_not_retain_a_parse(tmp_path):
    """Review R1: probes/reads hit the memo but never fill it (memory)."""
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    with ti._PARSED_LOCK:
        ti._PARSED.clear()
    assert ti.probe_index(tmp_path, SID).state == "ready"
    assert ti.read_index(tmp_path, SID) is not None
    assert ti.patch_naming(tmp_path, SID, {"u00001": {"name": "n", "summary": "s"}})
    with ti._PARSED_LOCK:
        # patch_naming PUBLISHES, so its write is remembered; the reads are not.
        assert len(ti._PARSED) <= 1
        ti._PARSED.clear()
    assert ti.probe_index(tmp_path, SID).state == "ready"
    assert ti.read_index(tmp_path, SID) is not None
    with ti._PARSED_LOCK:
        assert not ti._PARSED, "a read-only probe retained a whole parsed index"


def test_a_missing_delta_with_committed_bytes_resnapshots_instead_of_padding(tmp_path):
    """Review R5: never append after zero padding to a vanished delta."""
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    _append(tmp_path, _rows(20, 1))
    ti.refresh_index(tmp_path, SID)
    store = _store_of(tmp_path)
    assert store is not None and store["length"] > 0
    ti._store_paths(ti._index_dir(tmp_path), SID, store)[1].unlink()
    _append(tmp_path, _rows(21, 1))
    assert ti.refresh_index(tmp_path, SID) is not None
    fresh = _store_of(tmp_path)
    assert fresh is not None and fresh["snapshot"] != store["snapshot"] and fresh["frames"] == 0
    cold = _cold(tmp_path)
    assert cold is not None
    assert _comparable(cold) == _comparable(_full_scan(tmp_path))


def test_an_orphaned_staged_snapshot_is_reclaimed(tmp_path):
    """Review R4: a dead writer's staged snapshot does not leak forever."""
    _append(tmp_path, _rows(0, 20))
    ti.refresh_index(tmp_path, SID)
    directory = ti._index_dir(tmp_path)
    dead = directory / f"{SID}.snap.{'a' * 32}.snapshot.999999999.tmp"
    dead.write_text("partial")
    other = directory / f"deadbeef0002.snap.{'b' * 32}.snapshot.999999999.tmp"
    other.write_text("partial")
    # Force a moved publication (rebuild) so _retire runs for this session.
    path = _journal(tmp_path)
    temp = path.with_suffix(".rewrite")
    temp.write_text(path.read_text())
    os.replace(temp, path)
    assert ti.refresh_index(tmp_path, SID) is not None
    assert not dead.exists(), "a dead pid's staged snapshot survived"
    assert other.exists(), "another session's staged file was touched"


def test_frame_exactness_rejects_a_kept_id_that_moved_below_the_cut():
    """Q3's O(suffix) check must still refuse what the O(history) one refused."""

    def fact(id_: str, seq: int) -> dict[str, Any]:
        return {"id": id_, "seq": seq, "source_class": "plain_message", "flags": []}

    def index(facts: list[dict[str, Any]]) -> ti.TranscriptIndex:
        rows = {f["id"]: f["seq"] for f in facts}
        return ti.TranscriptIndex(
            checkpoints=[],
            messages=[],
            sig={"size": 100, "mtime": 1.0, "inode": 7, "last_id": facts[-1]["id"]},
            coverage={},
            naming={},
            scan=ti.ScanState(rows=len(facts), offset=100, window_offset=0, window_rows=0),
            journal_rows=rows,
            journal_ranges={k: (v, v + 1) for k, v in rows.items()},
            source_projection_facts=facts,
        )

    base = index([fact("a", 0), fact("b", 1), fact("x", 2)])
    grown = index([fact("a", 0), fact("b", 1), fact("x", 2), fact("c", 3)])
    frame = ti._delta_frame(base, grown)
    assert ti._frame_is_exact(base, grown, frame)
    # A suffix rewrite that keeps the ordinals below the cut is exact too, and
    # applying it reproduces the rewritten map.
    rewritten = index([fact("a", 0), fact("b", 1), fact("c", 2)])
    frame = ti._delta_frame(base, rewritten)
    assert frame["keep"]["rows"] == 2 and ti._frame_is_exact(base, rewritten, frame)
    applied = json.loads(json.dumps(ti._document(base)))
    ti._apply_frame(applied, json.loads(json.dumps(frame)))
    assert applied["journal_rows"] == rewritten.journal_rows
    # The shape the count exists for: ``a`` recurs past the cut in ``base`` (its
    # LAST ordinal is 2), and in ``index`` only its ordinal-0 row survives.
    # Applying the frame would drop ``a`` entirely, so exactness must refuse.
    dup = index([fact("a", 0), fact("b", 1), fact("a", 2)])
    survivor = index([fact("a", 0), fact("b", 1), fact("c", 2)])
    frame = ti._delta_frame(dup, survivor)
    assert frame["keep"]["rows"] == 2
    assert not ti._frame_is_exact(dup, survivor, frame)
