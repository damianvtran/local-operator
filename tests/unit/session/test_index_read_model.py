"""Immutable projection tests: canonical JSON commits, SQLite only accelerates reads.

Small synthetic journals keep corruption/publication tests independent of load;
large frame/manifest parity is covered at the consumer seams separately.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.session import transcript_index as ti

SID = "abc123abc123"


@pytest.fixture
def publication(tmp_path):
    path = tmp_path / "sessions" / SID / "transcript.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "id": "u1",
            "ts": 1,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": "hello"}]},
        },
        {
            "id": "a1",
            "ts": 2,
            "type": "message",
            "payload": {"kind": "message", "role": "assistant", "content": [{"text": "answer"}]},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    yield tmp_path, index
    ti._reset_for_tests()


def test_writer_commits_read_model_after_existing_resume_fields(publication):
    root, index = publication
    raw = json.loads(ti.index_path(root, SID).read_text())
    assert list(raw)[:5] == ["version", "sig", "scan", "journal_generation", "read_model"]
    assert raw["read_model"]["schema"] == 2
    token = raw["read_model"]["publication_id"]
    assert len(token) == 32 and all(c in "0123456789abcdef" for c in token)
    directory = ti.index_path(root, SID).parent / "read-models" / SID
    assert (directory / f"{token}.sqlite").is_file()
    # The mutable naming section lives beside the immutable companion, so a
    # naming-only write never rewrites the multi-megabyte body.
    assert (directory / f"{token}.naming.json").is_file()
    head = ti._read_resume_head(root, SID)
    assert head is not None
    assert head[:3] == (index.sig, index.scan, index.journal_generation)
    # document_bytes comes from the head's own descriptor, not a second stat.
    assert head[3] == ti.index_path(root, SID).stat().st_size


def test_manifest_projection_matches_complete_canonical_output(publication):
    from local_operator.session import index_read_model as rm

    root, index = publication
    actual = rm.checkpoint_manifest(root, SID)
    assert actual.state == "ready"
    assert actual.value == ti._manifest_state(
        SID, "ready", index, ti.index_path(root, SID).stat().st_mtime
    )


@pytest.mark.parametrize("ids,eof", [(["u1", "a1"], True), (["a1"], False), (["u1"], False)])
def test_frame_projection_matches_existing_derivation(publication, ids, eof):
    from local_operator.session import index_read_model as rm
    from local_operator.session import open_frame as frame

    root, index = publication
    source = (index.sig["inode"], index.sig["size"], index.sig["mtime"])
    result = rm.frame_facts(
        root, SID, collected_ids=ids, reaches_eof=eof, order_ids=["a1", "absent"], source=source
    )
    assert result.state == "ready", result.reason
    bounds = frame.page_seq_bounds(index, [{"id": id_} for id_ in ids], reaches_eof=eof)
    expected = frame.publish_runs(index, first_seq=bounds[0], last_seq=bounds[1]) if bounds else []
    assert result.value.runs == expected
    # ``ordinals`` is what the REQUEST asked about; the OPENERS of the served page
    # are added by ``apply_order`` against the final (capped) answer.
    assert result.value.ordinals == {
        id_: index.journal_rows[id_] for id_ in ("a1",) if id_ in index.journal_rows
    }
    # A served row's own coordinate rides the row; the opener map is what
    # ``apply_order`` adds for ids the answer actually serves.
    assert set(result.value.openers) <= set(ids) | {"a1"}
    assert result.value.missing_ids == ["absent"]
    assert result.value.generation == index.journal_generation
    # rows carries every proven identity (page + requested), which is what the
    # resident join's own map is keyed by.
    assert result.value.rows == {
        id_: index.journal_rows[id_] for id_ in (*ids, "a1") if id_ in index.journal_rows
    }


@pytest.mark.parametrize("mutation", ["append", "truncate", "rewrite", "replace"])
def test_changed_journal_rejects_prior_publication(publication, mutation):
    import os

    from local_operator.session import index_read_model as rm

    root, _ = publication
    path = root / "sessions" / SID / "transcript.jsonl"
    before = path.stat()
    content = path.read_bytes()
    if mutation == "append":
        path.write_bytes(content + b"\n")
    elif mutation == "truncate":
        path.write_bytes(content[:20])
    elif mutation == "rewrite":
        path.write_bytes(content.replace(b"hello", b"other"))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    else:
        replacement = path.with_suffix(".new")
        replacement.write_bytes(content)
        replacement.replace(path)
    assert rm.checkpoint_manifest(root, SID).state == "stale"


def test_missing_checkpoint_is_not_complete_ready(publication):
    import sqlite3

    from local_operator.session import index_read_model as rm

    root, _ = publication
    pub = rm.header(root, SID).value
    with sqlite3.connect(pub.path) as db:
        db.execute("DELETE FROM checkpoints WHERE position=0")
    assert rm.checkpoint_manifest(root, SID).state == "corrupt"


def test_failed_canonical_commit_keeps_previous_companion(publication, monkeypatch):
    from local_operator.session import index_read_model as rm

    root, index = publication
    before = rm.header(root, SID).value
    real_replace = Path.replace

    def fail_commit(path, destination):
        if destination == ti.index_path(root, SID):
            raise OSError("injected canonical commit failure")
        return real_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_commit)
    ti.write_index(root, SID, index)
    assert rm.header(root, SID).value == before
    assert rm.checkpoint_manifest(root, SID).state == "ready"
    assert len(list(before.path.parent.glob("*.sqlite"))) == 2


def test_publication_flip_retries_one_coherent_snapshot(publication, monkeypatch):
    from local_operator.session import index_read_model as rm

    root, index = publication
    original = rm._meta
    calls = []

    def flip(db, pub):
        meta = original(db, pub)
        calls.append(pub.identity)
        if len(calls) == 1:
            ti.write_index(root, SID, index)
        return meta

    monkeypatch.setattr(rm, "_meta", flip)
    result = rm.checkpoint_manifest(root, SID)
    assert result.state == "ready" and len(calls) == 2 and calls[0] != calls[1]
    # built_at is the publication the READ resolved, not a later stat: the flip
    # above rewrote the document while the first attempt was in flight.
    assert result.value["index"]["built_at"] == rm.header(root, SID).value.built_at


def test_naming_publication_and_cooldown_are_not_frozen(publication, monkeypatch):
    from local_operator.session import checkpoint_naming as naming
    from local_operator.session import index_read_model as rm

    root, _ = publication
    before = rm.header(root, SID).value
    monkeypatch.setattr(naming.time, "time", lambda: 1000.0)
    assert ti.patch_naming(root, SID, {"u1": {"failed_ts": 1000.0}})
    failed = rm.header(root, SID).value
    # Q3/m4: a naming patch must NOT mint a new companion body.
    assert before.path == failed.path
    first = rm.checkpoint_manifest(root, SID).value
    assert first["checkpoints"][-1]["naming"]["state"] == "unavailable"
    monkeypatch.setattr(
        naming.time, "time", lambda: 1000.0 + naming.NAMING_UNAVAILABLE_COOLDOWN_S + 1
    )
    later = rm.checkpoint_manifest(root, SID).value
    assert later["checkpoints"][-1]["naming"]["state"] == "pending"
    assert rm.header(root, SID).value.path == failed.path
    assert ti.patch_naming(root, SID, {"u1": {"name": "Done", "summary": "Result"}})
    named = rm.checkpoint_manifest(root, SID).value
    assert named["checkpoints"][-1]["naming"] == {
        "state": "ready",
        "name": "Done",
        "summary": "Result",
    }


def test_corrupt_companion_never_becomes_an_empty_ready_manifest(publication):
    from local_operator.session import index_read_model as rm

    root, _ = publication
    pub = rm.header(root, SID).value
    pub.path.write_bytes(b"not sqlite")
    assert rm.checkpoint_manifest(root, SID).state == "corrupt"


def test_duplicate_checkpoint_ids_keep_manifest_rows_and_last_lookup(publication):
    from dataclasses import replace

    from local_operator.session import index_read_model as rm
    from local_operator.session import open_frame

    root, index = publication
    checkpoints = index.checkpoints + [replace(index.checkpoints[0], seq=1)]
    duplicate = replace(index, checkpoints=checkpoints)
    ti.write_index(root, SID, duplicate)
    manifest = rm.checkpoint_manifest(root, SID)
    assert manifest.state == "ready"
    assert len(manifest.value["checkpoints"]) == len(checkpoints)
    source = (index.sig["inode"], index.sig["size"], index.sig["mtime"])
    facts = rm.frame_facts(
        root, SID, collected_ids=["u1"], reaches_eof=False, order_ids=[], source=source
    )
    assert facts.state == "ready", facts.reason
    bounds = open_frame.page_seq_bounds(duplicate, [{"id": "u1"}], reaches_eof=False)
    assert facts.value.runs == open_frame.publish_runs(
        duplicate, first_seq=bounds[0], last_seq=bounds[1]
    )


def test_restored_mtime_is_explicit_existing_witness_limit(publication):
    import os

    from local_operator.session import index_read_model as rm

    root, _ = publication
    path = root / "sessions" / SID / "transcript.jsonl"
    before = path.stat()
    path.write_bytes(path.read_bytes().replace(b"hello", b"other"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    # This cache preserves the existing inode/size/mtime protocol. A stronger
    # witness needs a scan/page/cache migration, not a fresh guessed ctime here.
    assert rm.checkpoint_manifest(root, SID).state == "ready"


@pytest.mark.asyncio
async def test_exact_parsed_hit_migrates_without_second_hydration(publication, monkeypatch):
    from local_operator.session import index_read_model as rm

    root, _ = publication
    path = ti.index_path(root, SID)
    raw = json.loads(path.read_text())
    marker = raw.pop("read_model")
    path.write_text(json.dumps(raw, separators=(",", ":")))
    calls = []
    real_read = ti._read_raw

    def counted(*args, **kwargs):
        calls.append(True)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(ti, "_read_raw", counted)
    first = await ti.checkpoints_view(root, SID)
    assert first["index"]["state"] == "ready"
    assert calls == [True]
    task = next(iter(rm._MIGRATIONS.values()))
    await task
    second = await ti.checkpoints_view(root, SID)
    assert second["checkpoints"] == first["checkpoints"]
    assert calls == [True]
    assert (
        marker["publication_id"]
        != rm.header(root, SID).value.header["read_model"]["publication_id"]
    )


@pytest.mark.asyncio
async def test_migration_cannot_overwrite_a_newer_naming_snapshot(publication, monkeypatch):
    from local_operator.session import index_read_model as rm

    root, _ = publication
    path = ti.index_path(root, SID)
    raw = json.loads(path.read_text())
    raw.pop("read_model")
    path.write_text(json.dumps(raw, separators=(",", ":")))
    original = await asyncio.to_thread(ti.refresh_index, root, SID)
    ti.patch_naming(root, SID, {"u1": {"name": "new", "summary": "new"}})
    ti.write_index(root, SID, original, expected_publication=original.publication_identity)
    result = rm.checkpoint_manifest(root, SID).value
    assert result["checkpoints"][-1]["naming"]["name"] == "new"


def test_scan_merges_a_racing_name_into_both_publications(publication, monkeypatch):
    from local_operator.session import index_read_model as rm

    root, _ = publication
    raw_path = root / "sessions" / SID / "transcript.jsonl"
    original = raw_path.read_bytes()
    # A replace (new inode) forces the full-scan branch this test intercepts.
    replacement = raw_path.with_suffix(".rebuilt")
    replacement.write_bytes(original + original)
    replacement.replace(raw_path)
    real_scan = ti._scan_full

    def racing_scan(path, raw):
        index = real_scan(path, raw)
        ti.patch_naming(root, SID, {"u1": {"name": "Raced", "summary": "kept"}})
        return index

    monkeypatch.setattr(ti, "_scan_full", racing_scan)
    assert ti.refresh_index(root, SID) is not None
    answer = rm.checkpoint_manifest(root, SID)
    assert answer.state == "ready"
    assert answer.value["checkpoints"][-1]["naming"]["name"] == "Raced"
    assert rm.header(root, SID).value.header["sig"]["size"] == len(original) * 2


def test_publication_tokens_are_not_paths(publication):
    from local_operator.session import index_read_model as rm

    root, _ = publication
    path = ti.index_path(root, SID)
    raw = json.loads(path.read_text())
    raw["read_model"]["publication_id"] = "../../outside"
    path.write_text(json.dumps(raw, separators=(",", ":")))
    assert rm.checkpoint_manifest(root, SID).state == "unsupported"


# ---------------------------------------------------------------------------
# Review round 1 (PR #3111): platform lock, failure memory, retention, polls,
# and one discriminating test per safety check.
# ---------------------------------------------------------------------------


def _rows(count: int) -> list[dict]:
    rows = []
    for turn in range(1, count + 1):
        rows.append(
            {
                "id": f"u{turn}",
                "ts": float(turn),
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": [{"text": f"q{turn}"}]},
            }
        )
        rows.append(
            {
                "id": f"a{turn}",
                "ts": turn + 0.5,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": f"a{turn}"}],
                },
            }
        )
    return rows


def _journal(root: Path, rows: list[dict]) -> Path:
    path = root / "sessions" / SID / "transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_publication_survives_a_platform_without_fcntl(publication, monkeypatch):
    """R1/Q2: Windows has no ``fcntl`` and this runs inside refresh/patch_naming."""
    import builtins

    root, _ = publication
    real_import = builtins.__import__

    def no_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("fcntl is POSIX-only")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fcntl)
    assert ti.patch_naming(root, SID, {"u1": {"name": "Portable", "summary": "yes"}})
    raw = json.loads(ti.index_path(root, SID).read_text())
    assert raw["naming"]["items"]["u1"]["name"] == "Portable"
    view = await ti.checkpoints_view(root, SID)
    assert view["index"]["state"] == "ready"
    assert view["checkpoints"][-1]["naming"]["name"] == "Portable"


def test_a_failed_companion_is_not_retried_for_the_same_publication(publication, monkeypatch):
    """R3: a doomed build must not rewrite the canonical JSON on every open."""
    from local_operator.session import index_read_model as rm

    root, _ = publication
    path = ti.index_path(root, SID)
    raw = json.loads(path.read_text())
    raw.pop("read_model")
    path.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")
    parsed = ti.read_index(root, SID)
    assert parsed is not None and parsed.publication_identity is not None
    attempts = []

    def cannot_publish(*args, **kwargs):
        attempts.append(1)
        raise OSError("cache is not writable")

    monkeypatch.setattr(rm, "publish", cannot_publish)
    ti.write_index(root, SID, parsed)
    ti.write_index(root, SID, parsed)
    assert len(attempts) == 1, "the same publication was rebuilt after it failed"
    # The canonical document itself is still rewritten (that is the fallback),
    # so the failure must not have cost the index its correctness.
    assert json.loads(path.read_text())["read_model"] is None


def test_summary_and_frame_facts_reject_a_foreign_source(publication):
    """R4: the witness check must discriminate, not merely exist."""
    from local_operator.session import index_read_model as rm

    root, _ = publication
    foreign = (1, 2, 3.0)
    assert rm.summary(root, SID, foreign).state != "ready"
    assert rm.summary(root, SID, None).state != "ready"
    facts = rm.frame_facts(
        root,
        SID,
        collected_ids=["u1"],
        reaches_eof=False,
        order_ids=[],
        source=foreign,
    )
    assert facts.state != "ready"


def test_companion_metadata_mismatch_is_never_ready(publication):
    """R4: meta is compared against the canonical head, not trusted."""
    import sqlite3

    from local_operator.session import index_read_model as rm

    root, _ = publication
    pub = rm.header(root, SID).value
    with sqlite3.connect(pub.path) as db:
        (payload,) = db.execute("SELECT payload FROM meta").fetchone()
        meta = json.loads(payload)
        meta["journal_generation"] = "a-different-scan"
        db.execute("DELETE FROM meta")
        db.execute("INSERT INTO meta VALUES (?)", (json.dumps(meta),))
    assert rm.checkpoint_manifest(root, SID).state == "corrupt"


def test_the_following_run_boundary_is_materialized(publication):
    """R4: without the next-run query the window silently grows into run N+1."""
    from local_operator.session import index_read_model as rm
    from local_operator.session import open_frame as frame

    root, _ = publication
    _journal(root, _rows(4))
    index = ti.refresh_index(root, SID)
    assert index is not None and len(index.runs) >= 3
    source = (index.sig["inode"], index.sig["size"], index.sig["mtime"])
    page = ["u3", "a3"]  # mid conversation, NOT the tail
    facts = rm.frame_facts(
        root, SID, collected_ids=page, reaches_eof=False, order_ids=[], source=source
    )
    assert facts.state == "ready", facts.reason
    bounds = frame.page_seq_bounds(index, [{"id": id_} for id_ in page], reaches_eof=False)
    assert facts.value.runs == frame.publish_runs(index, first_seq=bounds[0], last_seq=bounds[1])


def test_the_cross_process_lock_is_actually_taken(publication, monkeypatch):
    """R4: an in-process lock alone is not the cross-process guarantee."""
    import fcntl

    root, index = publication
    taken = []
    real_flock = fcntl.flock

    def record(fd, operation):
        taken.append(operation)
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", record)
    ti.write_index(root, SID, index)
    assert taken == [fcntl.LOCK_EX]


def test_naming_patches_reuse_one_companion(publication):
    """Q3/m4: a naming patch must not mint another multi-MB body."""
    from local_operator.session import index_read_model as rm

    root, _ = publication
    directory = ti.index_path(root, SID).parent / "read-models" / SID
    for turn in range(1, 9):
        assert ti.patch_naming(root, SID, {"u1": {"name": f"n{turn}", "summary": "s"}})
    assert len(list(directory.glob("*.sqlite"))) == 1
    answer = rm.checkpoint_manifest(root, SID)
    assert answer.state == "ready"
    assert answer.value["checkpoints"][-1]["naming"]["name"] == "n8"


def test_collection_keeps_the_referenced_publication_and_a_small_window(publication):
    """Q3/n1: retention is bounded and never drops what the head names."""
    from local_operator.session import index_read_model as rm

    root, _ = publication
    directory = ti.index_path(root, SID).parent / "read-models" / SID
    for turn in range(2, 8):
        _journal(root, _rows(turn))
        assert ti.refresh_index(root, SID) is not None
    kept = sorted(path.name for path in directory.glob("*.sqlite"))
    assert len(kept) <= rm._RETAIN_PREVIOUS + 1
    referenced = rm.header(root, SID).value.path
    assert referenced.name in kept
    assert rm.checkpoint_manifest(root, SID).state == "ready"


def test_read_models_of_deleted_sessions_are_reclaimed(publication):
    """Q3: a deleted session's companions do not accumulate forever."""
    import os as _os

    root, index = publication
    directory = ti.index_path(root, SID).parent / "read-models"
    orphan = directory / "deleted-session"
    orphan.mkdir(parents=True)
    stale = orphan / ("0" * 32 + ".sqlite")
    stale.write_bytes(b"x")
    old = 1_600_000_000
    _os.utime(stale, (old, old))
    _os.utime(orphan, (old, old))
    ti.write_index(root, SID, index)
    assert not orphan.exists()


def test_repeated_manifest_polls_do_not_reopen_the_companion(publication, monkeypatch):
    """Q4: a companion-served session must reach the resident poll steady state."""
    from local_operator.session import index_read_model as rm

    root, _ = publication
    first = rm.checkpoint_manifest(root, SID)
    assert first.state == "ready"
    connects = []
    real_connect = rm.sqlite3.connect

    def counting(*args, **kwargs):
        connects.append(1)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(rm.sqlite3, "connect", counting)
    for _ in range(3):
        assert rm.checkpoint_manifest(root, SID).value == first.value
    assert connects == [], "a cached poll still opened the companion"
    # AND the memo must not freeze naming: the sidecar's identity is in the key.
    assert ti.patch_naming(root, SID, {"u1": {"name": "Fresh", "summary": "s"}})
    fresh = rm.checkpoint_manifest(root, SID)
    assert fresh.value["checkpoints"][-1]["naming"]["name"] == "Fresh"


def test_the_companion_is_built_outside_the_publish_lock(publication, monkeypatch):
    """Q5: a slow companion build must not serialize other writers of the session."""
    import threading

    from local_operator.session import index_read_model as rm

    root, index = publication
    inside = threading.Event()
    release = threading.Event()
    real_publish = rm.publish

    def slow_publish(*args, **kwargs):
        inside.set()
        assert release.wait(10), "the test never released the companion build"
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(rm, "publish", slow_publish)
    writer = threading.Thread(
        target=lambda: ti.write_index(root, SID, index),
        daemon=True,
    )
    writer.start()
    try:
        assert inside.wait(10), "the companion build never started"
        patcher = threading.Thread(
            target=lambda: ti.patch_naming(root, SID, {"u1": {"name": "During", "summary": "s"}}),
            daemon=True,
        )
        patcher.start()
        patcher.join(5)
        assert not patcher.is_alive(), "a naming patch waited for the companion build"
    finally:
        release.set()
        writer.join(10)
    on_disk = json.loads(ti.index_path(root, SID).read_text())
    assert on_disk["naming"]["items"]["u1"]["name"] == "During"


def test_the_paint_cost_does_not_stat_the_cache(publication, monkeypatch):
    """m2: the paint-hold decision must not add a stat to the event loop."""
    root, _ = publication
    journal = (root / "sessions" / SID / "transcript.jsonl").stat()
    calls = []
    real_stat = ti._stat_or_none

    def counting(path):
        calls.append(path)
        return real_stat(path)

    monkeypatch.setattr(ti, "_stat_or_none", counting)
    has_previous, _rederive, hydration = ti._cached_refresh_cost(root, SID, journal)
    assert has_previous and hydration > 0
    assert calls == [], "the paint decision stat'ed the cache on the loop"
