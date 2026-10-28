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
    assert raw["read_model"]["schema"] == 1
    token = raw["read_model"]["publication_id"]
    assert len(token) == 32 and all(c in "0123456789abcdef" for c in token)
    companion = ti.index_path(root, SID).parent / "read-models" / SID / f"{token}.sqlite"
    assert companion.is_file()
    assert ti._read_resume_head(root, SID) == (index.sig, index.scan, index.journal_generation)


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
    assert result.value.ordinals == {"a1": index.journal_rows["a1"], "u1": index.journal_rows["u1"]}
    assert result.value.missing_ids == ["absent"]
    assert result.value.generation == index.journal_generation
    assert result.value.rows == {id_: index.journal_rows[id_] for id_ in ids}


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
    assert result.value["index"]["built_at"] == ti.index_path(root, SID).stat().st_mtime


def test_naming_publication_and_cooldown_are_not_frozen(publication, monkeypatch):
    from local_operator.session import checkpoint_naming as naming
    from local_operator.session import index_read_model as rm

    root, _ = publication
    before = rm.header(root, SID).value
    monkeypatch.setattr(naming.time, "time", lambda: 1000.0)
    assert ti.patch_naming(root, SID, {"u1": {"failed_ts": 1000.0}})
    failed = rm.header(root, SID).value
    assert before.path != failed.path
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
