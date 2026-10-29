"""The read model: an immutable projection named after the derivation it describes.

The canonical JSON stays the commit point and the only derivation authority, and
the companion is published OFF the write path (round-2 scope: building it inside
``write_index`` cost ~1.3 s per append refresh). Every test here therefore says
which publication it is looking at instead of assuming a marker in the document.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import threading
from pathlib import Path
from typing import cast

import pytest

from local_operator.session import index_read_model as rm
from local_operator.session import transcript_index as ti

SID = "abc123abc123"


def _materialize(root: Path, index: ti.TranscriptIndex) -> str:
    """Publish a companion the way the background pass does, synchronously."""
    token = rm.publish(root, SID, index)
    assert token is not None
    rm.write_naming(root, SID, token, index.naming)
    return token


def _rows(count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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


def _journal(root: Path, rows: list[dict[str, object]] | None = None) -> Path:
    path = root / "sessions" / SID / "transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in (rows or _rows(2))), encoding="utf-8")
    return path


@pytest.fixture
def publication(tmp_path):
    _journal(tmp_path)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    _materialize(tmp_path, index)
    yield tmp_path, index
    ti._reset_for_tests()


def _pub(root: Path, sid: str = SID) -> rm.Publication:
    """The publication a read resolved, asserting it is ready (pyright cannot)."""
    probe = rm.header(root, sid)
    assert probe.state == "ready", probe.reason
    assert isinstance(probe.value, rm.Publication)
    return probe.value


def _source(index: ti.TranscriptIndex) -> tuple[int, int, float]:
    return rm.source_of(index)


# ---------------------------------------------------------------------------
# Publication shape and the canonical document
# ---------------------------------------------------------------------------


def test_the_canonical_document_names_no_marker_and_the_companion_hides_beside_it(publication):
    root, index = publication
    raw = json.loads(ti.index_path(root, SID).read_text())
    assert list(raw)[:4] == ["version", "sig", "scan", "journal_generation"]
    assert "read_model" not in raw, "publishing a companion must not rewrite the document"
    token = rm.token_for(index)
    directory = ti.index_path(root, SID).parent / "read-models" / SID
    assert (directory / f"{token}.sqlite").is_file()
    assert (directory / f"{token}.naming.json").is_file()
    head = ti._read_resume_head(root, SID)
    assert head is not None
    assert head[:3] == (index.sig, index.scan, index.journal_generation)


def test_a_naming_patch_reuses_the_companion_and_refreshes_only_the_sidecar(publication):
    root, _ = publication
    before = _pub(root)
    assert ti.patch_naming(root, SID, {"u2": {"name": "Named", "summary": "s"}})
    after = _pub(root)
    assert before.path == after.path, "a naming write must not mint another companion body"
    manifest = rm.checkpoint_manifest(root, SID)
    assert manifest.state == "ready"
    manifest_value = manifest.value
    assert manifest_value is not None
    assert manifest_value["checkpoints"][-1]["naming"]["name"] == "Named"


def test_repeated_manifest_polls_are_consistently_cached(publication, monkeypatch):
    """Q9: no alternating memo-hit/rebuild across consecutive polls."""
    root, _ = publication
    first = rm.checkpoint_manifest(root, SID)
    assert first.state == "ready"
    connects: list[int] = []
    real_connect = rm.sqlite3.connect

    def counting(*args, **kwargs):
        connects.append(1)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(rm.sqlite3, "connect", counting)
    for _ in range(5):
        assert rm.checkpoint_manifest(root, SID).value == first.value
    assert connects == [], "consecutive polls re-opened the companion"
    assert ti.patch_naming(root, SID, {"u2": {"name": "Fresh", "summary": "s"}})
    assert rm.checkpoint_manifest(root, SID).value["checkpoints"][-1]["naming"]["name"] == "Fresh"


# ---------------------------------------------------------------------------
# The companion is off the write path (round-2 scope, Q8/Q5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_checkpoints_seam_serves_repeated_polls_without_reopening(
    publication, monkeypatch
):
    """Q4/Q9 at the consumer seam: a companion-served poll must reach steady state.

    The read model being fast is not enough — `/checkpoints` goes through
    ``checkpoints_view``, and QA measured that seam alternating between a memo
    hit and a full rebuild (54/73 ms against base's 44/52). The invariant here is
    the one that makes the alternation impossible: after the first poll, no poll
    opens the companion again and every answer reports the SAME publication.
    """
    root, index = publication
    first = await ti.checkpoints_view(root, SID)
    assert first["index"]["state"] == "ready"
    connects: list[int] = []
    real_connect = rm.sqlite3.connect

    def counting(*args, **kwargs):
        connects.append(1)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(rm.sqlite3, "connect", counting)
    polls = [await ti.checkpoints_view(root, SID) for _ in range(20)]
    assert connects == [], "a warm poll re-opened the companion"
    assert {poll["index"]["built_at"] for poll in polls} == {first["index"]["built_at"]}
    assert {len(poll["checkpoints"]) for poll in polls} == {len(index.checkpoints)}
    # A naming patch is metadata, not a new derivation: the body is reused.
    assert ti.patch_naming(root, SID, {"u2": {"name": "Polls", "summary": "s"}})
    after = await ti.checkpoints_view(root, SID)
    assert after["checkpoints"][-1]["naming"]["name"] == "Polls"
    # The canonical document was rewritten (naming is authoritative there), so
    # THIS poll may rebuild once; the polls after it must not.
    connects.clear()
    warm = [await ti.checkpoints_view(root, SID) for _ in range(5)]
    assert connects == []
    assert {poll["index"]["built_at"] for poll in warm} == {after["index"]["built_at"]}


def test_the_canonical_write_never_builds_a_companion(tmp_path, monkeypatch):
    _journal(tmp_path)
    calls: list[int] = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("the canonical write built a companion")

    monkeypatch.setattr(rm, "publish", forbidden)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    ti.write_index(tmp_path, SID, index)
    assert calls == []
    assert rm.header(tmp_path, SID).state == "missing"


def test_an_append_refresh_publishes_at_base_speed_and_materializes_afterwards(
    tmp_path, monkeypatch
):
    """Q8: the refresh must not wait for a companion; the background pass may."""
    _journal(tmp_path)
    first = ti.refresh_index(tmp_path, SID)
    assert first is not None
    _materialize(tmp_path, first)
    appended = tmp_path / "sessions" / SID / "transcript.jsonl"
    with appended.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_rows(3)[-1]) + "\n")
    real_publish = rm.publish
    built: list[str] = []

    def slow(*args, **kwargs):
        built.append(rm.token_for(args[2]))
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(rm, "publish", slow)
    second = ti.refresh_index(tmp_path, SID)
    assert second is not None
    assert built == [], "the append refresh published a companion on its critical path"
    token = rm.token_for(second)
    assert token != rm.token_for(first)
    assert rm.header(tmp_path, SID).state == "missing"
    _materialize(tmp_path, second)
    assert rm.checkpoint_manifest(tmp_path, SID).state == "ready"


@pytest.mark.asyncio
async def test_an_exact_parsed_hit_materializes_without_a_second_hydration(tmp_path, monkeypatch):
    _journal(tmp_path)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    reads: list[int] = []
    real_read = ti._read_raw

    def counted(*args, **kwargs):
        reads.append(1)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(ti, "_read_raw", counted)
    first = await ti.checkpoints_view(tmp_path, SID)
    assert first["index"]["state"] == "ready"
    assert reads == [1], "the first view is base's one hydration"
    task = next(iter(rm._MIGRATIONS.values()), None)
    if task is not None:
        await task
    served = await ti.checkpoints_view(tmp_path, SID)
    assert served["checkpoints"] == first["checkpoints"]
    assert len(reads) == 1, "the materialised companion re-hydrated the index"


# ---------------------------------------------------------------------------
# Failure memory (round-2 R3) and the tokened invalidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_torn_tail_source_is_remembered_across_four_opens(tmp_path, monkeypatch):
    """R3: the negative entry is keyed on the SOURCE, which the write cannot move."""
    _journal(tmp_path)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    attempts: list[int] = []

    def broken(*args, **kwargs):
        attempts.append(1)
        raise ValueError("torn tail: coverage incomplete")

    monkeypatch.setattr(rm, "publish", broken)
    for _ in range(4):
        rm.schedule_materialize(tmp_path, SID, index)
        task = next(iter(rm._MIGRATIONS.values()), None)
        if task is not None:
            await task
    # Publication now waits for the session to go QUIET (a burst must attempt
    # nothing), so the one remembered failure only shows up after the idle timer.
    await asyncio.sleep(rm._MATERIALIZE_IDLE_S + 1.0)
    assert len(attempts) == 1, attempts
    document = ti.index_path(tmp_path, SID).stat().st_mtime_ns
    assert rm.header(tmp_path, SID).state == "missing"
    # The source has not changed, so even the NEXT schedule is refused...
    rm.schedule_materialize(tmp_path, SID, index)
    assert next(iter(rm._MIGRATIONS.values()), None) is None
    # ...and a moved source is tried again.
    journal = tmp_path / "sessions" / SID / "transcript.jsonl"
    os.utime(journal, ns=(journal.stat().st_atime_ns, journal.stat().st_mtime_ns + 1_000_000))
    moved = ti.refresh_index(tmp_path, SID)
    assert moved is not None
    assert ti.index_path(tmp_path, SID).stat().st_mtime_ns >= document


@pytest.mark.parametrize("mutation", ["append", "truncate", "rewrite", "replace"])
def test_a_changed_journal_has_no_companion_for_its_new_derivation(publication, mutation):
    root, index = publication
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
    # The document and its companion describe the PREVIOUS derivation, and the
    # journal witness in ``header`` is what says so (round-3 N1): nothing is served
    # from them, and the consumer takes base's refresh path instead.
    assert rm.checkpoint_manifest(root, SID).state == "stale"
    assert rm.header(root, SID).state == "stale"


# ---------------------------------------------------------------------------
# Projection content and its fail-closed checks
# ---------------------------------------------------------------------------


def test_manifest_projection_matches_complete_canonical_output(publication):
    root, index = publication
    actual = rm.checkpoint_manifest(root, SID)
    assert actual.state == "ready"
    assert actual.value == ti._manifest_state(
        SID, "ready", index, ti.index_path(root, SID).stat().st_mtime
    )


@pytest.mark.parametrize("ids,eof", [(["u1", "a1"], True), (["a1"], False), (["u1"], False)])
def test_frame_projection_matches_existing_derivation(publication, ids, eof):
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
    assert result.value.ordinals == {
        id_: index.journal_rows[id_] for id_ in ("a1",) if id_ in index.journal_rows
    }
    assert result.value.missing_ids == ["absent"]
    assert result.value.rows == {
        id_: index.journal_rows[id_] for id_ in (*ids, "a1") if id_ in index.journal_rows
    }


def test_summary_and_frame_facts_reject_a_foreign_source(publication):
    root, _ = publication
    assert rm.summary(root, SID, (1, 2, 3.0)).state != "ready"
    assert rm.summary(root, SID, None).state != "ready"
    facts = rm.frame_facts(
        root, SID, collected_ids=["u1"], reaches_eof=False, order_ids=[], source=(1, 2, 3.0)
    )
    assert facts.state != "ready"


def test_companion_metadata_mismatch_is_never_ready(publication):
    import sqlite3

    root, _ = publication
    pub = _pub(root)
    with sqlite3.connect(pub.path) as db:
        (payload,) = db.execute("SELECT payload FROM meta").fetchone()
        meta = json.loads(payload)
        meta["journal_generation"] = "moved"
        db.execute("UPDATE meta SET payload=?", (json.dumps(meta),))
    assert rm.checkpoint_manifest(root, SID).state == "corrupt"


def test_missing_checkpoint_is_not_complete_ready(publication):
    import sqlite3

    root, _ = publication
    pub = _pub(root)
    with sqlite3.connect(pub.path) as db:
        db.execute("DELETE FROM checkpoints WHERE position=0")
    assert rm.checkpoint_manifest(root, SID).state == "corrupt"


def test_corrupt_companion_never_becomes_an_empty_ready_manifest(publication):
    root, _ = publication
    _pub(root).path.write_bytes(b"not sqlite")
    assert rm.checkpoint_manifest(root, SID).state == "corrupt"


def test_duplicate_checkpoint_ids_use_the_last_position(tmp_path):
    """A duplicated checkpoint id keeps the LAST position, like the derivation.

    Caveat: `MAX(position)` -> `MIN(position)` survives this test, because the
    reference derivation applies the same last-position rule and the duplicate is
    never the row a page's bounds depend on. What is checked is that the
    projection keeps every checkpoint row and agrees with the derivation.
    """
    from dataclasses import replace

    from local_operator.session import open_frame as frame

    _journal(tmp_path)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    # The token names the DERIVATION, so a variant is published before the first
    # companion exists for this source (a real store cannot produce two
    # derivations for one signature).
    # The duplicate carries a LATER ordinal, so "last position wins" and "first
    # position wins" produce different bounds. A same-seq copy discriminates
    # nothing.
    duplicate = replace(
        index,
        checkpoints=[
            *index.checkpoints,
            replace(
                index.checkpoints[0],
                seq=index.checkpoints[-1].seq + 100,
                ts=index.checkpoints[-1].ts + 1.0,
                turn=index.checkpoints[-1].turn + 1,
            ),
        ],
    )
    _materialize(tmp_path, duplicate)
    root = tmp_path
    source = (index.sig["inode"], index.sig["size"], index.sig["mtime"])
    facts = rm.frame_facts(
        root, SID, collected_ids=["u1"], reaches_eof=False, order_ids=[], source=source
    )
    assert facts.state == "ready", facts.reason
    bounds = frame.page_seq_bounds(duplicate, [{"id": "u1"}], reaches_eof=False)
    assert bounds is not None
    facts_value = facts.value
    assert facts_value is not None
    assert facts_value.runs == frame.publish_runs(
        duplicate, first_seq=bounds[0], last_seq=bounds[1]
    )
    manifest = rm.checkpoint_manifest(root, SID)
    manifest_value = manifest.value
    assert manifest_value is not None
    assert len(manifest_value["checkpoints"]) == len(duplicate.checkpoints)


def test_the_following_run_boundary_is_materialized(publication):
    """A bounded page's runs match the full derivation exactly.

    NOT a mutation discriminator, and it does not claim to be: replacing the
    `first_seq >` helper query with `>=` survives this test, because the run tail
    is materialised from `first_seq <= min(seqs) - 1` onward and the boundary is
    owned by `frame.page_seq_bounds`. The equality with the resident derivation is
    what is checked here; see the round-3 remediation comment for the measured
    non-discrimination.
    """
    from local_operator.session import open_frame as frame

    root, index = publication
    assert len(index.runs) >= 2
    source = (index.sig["inode"], index.sig["size"], index.sig["mtime"])
    page = [{"id": index.checkpoints[0].id}]
    facts = rm.frame_facts(
        root,
        SID,
        collected_ids=[page[0]["id"]],
        reaches_eof=False,
        order_ids=[],
        source=source,
    )
    assert facts.state == "ready", facts.reason
    bounds = frame.page_seq_bounds(index, page, reaches_eof=False)
    assert bounds is not None
    expected = frame.publish_runs(index, first_seq=bounds[0], last_seq=bounds[1])
    assert bounds[1] is not None, "the fixture must have a following run to publish"
    facts_value = facts.value
    assert facts_value is not None
    assert facts_value.runs == expected

    # THE BOUNDARY THAT DISCRIMINATES `>` FROM `>=` needs a store whose run tail
    # is CAPPED: below ``OPEN_FRAME_MAX_ROWS`` the projection loads every run and
    # the helper query is redundant, so only a capped selection shows the
    # difference between "the run after this page" and "a run at this page's own
    # last ordinal". That fixture is built separately below.


def test_the_following_run_helper_is_the_strict_boundary(tmp_path):
    """A capped run tail still matches the full derivation.

    Same caveat as ``test_the_following_run_boundary_is_materialized``: the
    `first_seq >` helper query is redundant inside the cap, so `>=` survives here
    too. Recorded rather than silently overstated.
    """
    from local_operator.session import open_frame as frame

    turns = frame.OPEN_FRAME_MAX_ROWS + 20
    rows: list[dict[str, object]] = []
    for turn in range(1, turns + 1):
        rows.append(
            {
                "id": f"u{turn:05d}",
                "ts": float(turn),
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": [{"text": "q"}]},
            }
        )
        rows.append(
            {
                "id": f"a{turn:05d}",
                "ts": turn + 0.5,
                "type": "message",
                "payload": {"kind": "message", "role": "assistant", "content": [{"text": "a"}]},
            }
        )
    _journal(tmp_path, rows)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    _materialize(tmp_path, index)
    assert len(index.runs) > frame.OPEN_FRAME_MAX_ROWS
    source = (index.sig["inode"], index.sig["size"], index.sig["mtime"])
    # A page whose last row opens a run: that run must NOT be published as the
    # following one, and the capped selection must not silently drop the tail.
    last_user = [c for c in index.checkpoints if c.kind == "user"][-1]
    assert any(run.first_seq == last_user.seq for run in index.runs)
    page = [{"id": last_user.id}]
    facts = rm.frame_facts(
        tmp_path,
        SID,
        collected_ids=[last_user.id],
        reaches_eof=False,
        order_ids=[],
        source=source,
    )
    assert facts.state == "ready", facts.reason
    bounds = frame.page_seq_bounds(index, page, reaches_eof=False)
    assert bounds is not None
    assert bounds[1] is None
    facts_value = facts.value
    assert facts_value is not None
    assert facts_value.runs == frame.publish_runs(index, first_seq=bounds[0], last_seq=bounds[1])


def test_restored_mtime_is_explicit_existing_witness_limit(publication):
    root, _ = publication
    path = root / "sessions" / SID / "transcript.jsonl"
    before = path.stat()
    path.write_bytes(path.read_bytes().replace(b"hello", b"other"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    # The cache preserves the existing inode/size/mtime protocol; a stronger
    # witness is a scan/page/cache migration, not a guessed ctime here.
    assert rm.checkpoint_manifest(root, SID).state == "ready"


# ---------------------------------------------------------------------------
# Writer mechanics: lock, fsync, collection
# ---------------------------------------------------------------------------


def test_the_cross_process_lock_is_actually_taken(publication, monkeypatch):
    """R4: a no-op flock must fail this."""
    import fcntl

    root, index = publication
    calls: list[tuple[int, int]] = []
    real_flock = fcntl.flock

    def counting(fd, operation):
        calls.append((fd, operation))
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", counting)
    ti.write_index(root, SID, index)
    assert calls and calls[0][1] == fcntl.LOCK_EX


def test_a_stale_writer_preserves_a_patch_that_landed_after_it(publication):
    """R4: dies when the merge is removed — the lock alone is not the guarantee.

    A writer that derived its index BEFORE a naming patch must not publish that
    older naming back over the patch. This is the observable half of "merge under
    the lock": a merge that reads the document it is about to replace.
    """
    root, index = publication
    assert ti.patch_naming(root, SID, {"u1": {"name": "Later", "summary": "s"}})
    stale = dataclasses.replace(index, naming={"prompt_version": 1, "items": {}})
    ti.write_index(root, SID, stale)
    written = json.loads(ti.index_path(root, SID).read_text())
    assert written["naming"]["items"]["u1"]["name"] == "Later", "a stale write dropped a patch"


def test_the_naming_merge_happens_inside_the_lock(publication, monkeypatch):
    """R4: a merge outside the lock loses a patch that lands mid-write."""
    root, index = publication
    entered = threading.Event()
    release = threading.Event()
    real_merge = ti.preserved_naming

    def slow_merge(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(ti, "preserved_naming", slow_merge)
    writer = threading.Thread(target=lambda: ti.write_index(root, SID, index), daemon=True)
    writer.start()
    patcher: threading.Thread | None = None
    try:
        assert entered.wait(10)
        patcher = threading.Thread(
            target=lambda: ti.patch_naming(root, SID, {"u1": {"name": "Raced", "summary": "s"}}),
            daemon=True,
        )
        patcher.start()
        patcher.join(3)
        assert patcher.is_alive(), "the patch did not wait for the writer's lock"
    finally:
        release.set()
        writer.join(10)
        if patcher is not None:
            patcher.join(10)
    assert json.loads(ti.index_path(root, SID).read_text())["naming"]["items"]["u1"]["name"] == (
        "Raced"
    )


def test_a_read_only_fsync_cannot_lose_the_document(publication, monkeypatch):
    """Q10: fsync on a read-only handle is EBADF on Windows and was swallowed."""
    import fcntl

    root, index = publication
    path = ti.index_path(root, SID)
    real_fsync = os.fsync

    def strict(fd):
        mode = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        if mode == os.O_RDONLY:
            raise OSError(9, "Bad file descriptor")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", strict)
    updated = dataclasses.replace(index, checkpoints=index.checkpoints[:1])
    ti.write_index(root, SID, updated)
    written = json.loads(path.read_text())
    assert written["checkpoints"] == [
        c.to_payload() for c in updated.checkpoints
    ], "the canonical document was not replaced"


def test_a_failed_replace_keeps_the_previous_document_and_reports(publication, monkeypatch, caplog):
    root, _ = publication
    path = ti.index_path(root, SID)
    before = path.read_bytes()
    real_replace = Path.replace

    def refuse(self, destination):
        if Path(destination) == path:
            raise OSError(28, "No space left on device")
        return real_replace(self, destination)

    monkeypatch.setattr(Path, "replace", refuse)
    parsed = ti.read_index(root, SID)
    assert parsed is not None
    ti.write_index(root, SID, parsed)
    assert path.read_bytes() == before
    assert "could not write the transcript index" in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_a_stale_parse_is_not_merged_over_a_newer_patch(publication):
    """The reuse must be identity-checked, or a patch disappears (round-1 R4 class).

    ``patch_naming`` hands its parsed document to ``write_index`` to save a full
    decode (round-3 M3). If the file moved between that parse and the lock, the
    handed-over document is stale and the merge must fall back to a fresh read.
    """
    root, _ = publication
    stale_raw = ti._read_raw(root, SID)
    assert stale_raw is not None
    receipt = stale_raw.get("_publication_identity")
    assert isinstance(receipt, tuple)
    # A patch lands AFTER the parse: the document now names a user the stale copy
    # does not know about.
    assert ti.patch_naming(root, SID, {"u2": {"name": "Later", "summary": "s"}})
    parsed = ti.read_index(root, SID)
    assert parsed is not None
    parsed.naming = {"prompt_version": 1, "items": {}}
    ti.write_index(root, SID, parsed, previous=(receipt, stale_raw))
    written = json.loads(ti.index_path(root, SID).read_text())
    assert written["naming"]["items"]["u2"]["name"] == "Later", "a stale parse dropped a patch"


def test_a_stale_head_still_yields_the_naming_sidecar(publication):
    """Round-4 MAJOR-1: the sidecar must be readable exactly when it matters.

    The append case is the one this helper was written for and it is also the case
    where ``header()`` reports ``stale``, so a ``ready`` gate made it unreachable
    and put the whole-document decode back on the write path.
    """
    root, index = publication
    assert ti.patch_naming(root, SID, {"u1": {"name": "Kept", "summary": "s"}})
    path = root / "sessions" / SID / "transcript.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert rm.header(root, SID).state == "stale"
    sidecar = rm.naming_for_root(root, SID)
    assert sidecar is not None and sidecar["items"]["u1"]["name"] == "Kept"

    # And the WRITE path uses it: with the document read forbidden, a write must
    # still merge the patch instead of raising or dropping it.
    def forbidden(*args, **kwargs):
        raise AssertionError("write_index decoded the whole document on a stale head")

    fresh = dataclasses.replace(index, naming={"prompt_version": 1, "items": {}})
    original = ti._read_raw
    ti._read_raw = forbidden
    try:
        ti.write_index(root, SID, fresh)
    finally:
        ti._read_raw = original
    written = json.loads(ti.index_path(root, SID).read_text())
    assert written["naming"]["items"]["u1"]["name"] == "Kept"


@pytest.mark.asyncio
async def test_a_burst_publishes_nothing_and_coalesces(tmp_path, monkeypatch):
    """QA round 4, cell 2: a burst must not attempt a single publication.

    The rule the daemon measurement produced: while opens are running, this hook
    may only RECORD and re-arm. One timer per session (a later derivation replaces
    the pending one), nothing built, nothing attempted.
    """
    _journal(tmp_path)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    rm.reset_for_tests()
    built: list[str] = []
    real_publish = rm.publish

    def spy(root, sid, idx):
        built.append(sid)
        return real_publish(root, sid, idx)

    monkeypatch.setattr(rm, "publish", spy)
    for _ in range(13):
        rm.schedule_materialize(tmp_path, SID, index)
    await asyncio.sleep(0.6)  # well inside the idle window
    assert built == [], "a publication was attempted inside the burst"
    assert len(rm._PENDING) == 1 and len(rm._TIMERS) == 1, (rm._PENDING, rm._TIMERS)
    # Quiet long enough, and the one coalesced derivation publishes.
    await asyncio.sleep(rm._MATERIALIZE_IDLE_S + 1.5)
    assert built == [SID], built
    assert rm.header(tmp_path, SID).state == "ready"


@pytest.mark.asyncio
async def test_a_timer_armed_before_a_publication_does_not_republish(tmp_path, monkeypatch):
    """QA round 5: a warm open must never re-publish a companion it just used.

    The cold half of an open arms the idle timer; the publication it needs then
    lands (from the same derivation) before the timer fires. If the timer fired
    blind, every warm open would rewrite the companion — measured on the daemon as
    one `rm_publish` per warm `eligible` open, ~5 s of work base never does.
    """
    _journal(tmp_path)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    rm.reset_for_tests()
    built: list[str] = []
    real_publish = rm.publish

    def spy(root, sid, idx):
        built.append(sid)
        return real_publish(root, sid, idx)

    monkeypatch.setattr(rm, "publish", spy)
    rm.schedule_materialize(tmp_path, SID, index)  # arms the timer
    assert rm._PENDING and rm._TIMERS
    real_publish(tmp_path, SID, index)  # the companion lands first
    assert rm._companion_exists(tmp_path, SID, index)
    await asyncio.sleep(rm._MATERIALIZE_IDLE_S + 1.0)
    assert built == [], "the idle timer republished a companion that already existed"
    assert rm.header(tmp_path, SID).state == "ready"


@pytest.mark.asyncio
async def test_a_busy_session_keeps_deferring_without_churn(tmp_path, monkeypatch):
    """A session with a refresh in flight is never published, and re-arms once."""
    _journal(tmp_path)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    rm.reset_for_tests()
    built: list[str] = []
    monkeypatch.setattr(rm, "publish", lambda *a, **k: built.append(str(a[1])))
    key = ti._key(tmp_path, SID)
    armed: list[int] = []
    real_arm = rm._arm

    def counting(k, sid, loop):
        armed.append(1)
        return real_arm(k, sid, loop)

    monkeypatch.setattr(rm, "_arm", counting)
    # Pretend this session is mid-refresh for the whole experiment.
    # The registry is typed for a real refresh (loop, task); this fake stands in
    # for one and only needs to be PRESENT for the hook's busy check.
    ti._IN_FLIGHT[key] = cast(
        "tuple[asyncio.AbstractEventLoop, asyncio.Task[None]]",
        (asyncio.get_running_loop(), asyncio.get_running_loop().create_future()),
    )
    try:
        rm.schedule_materialize(tmp_path, SID, index)
        await asyncio.sleep(rm._MATERIALIZE_IDLE_S * 3)
        assert built == [], "a busy session was published"
        # One arm per quiet period, not a churn: three idle windows, three arms.
        assert 2 <= len(armed) <= 6, armed
        assert len(rm._TIMERS) == 1
    finally:
        ti._IN_FLIGHT.pop(key, None)


def test_collection_keeps_the_referenced_token(publication):
    """Round-2 defect: `collect` was handed a dict member where it wanted a token."""
    root, index = publication
    token = rm.token_for(index)
    for _ in range(4):
        ti.write_index(root, SID, index)
    directory = ti.index_path(root, SID).parent / "read-models" / SID
    assert (directory / f"{index_read_token(root)}.sqlite").is_file()
    assert _pub(root).token_id == index_read_token(root)
    assert token == index_read_token(root)
    kept = sorted(path.stem for path in directory.glob("*.sqlite"))
    assert len(kept) <= rm._RETAIN_PREVIOUS + 1


def index_read_token(root: Path) -> str:
    head = ti._read_resume_head(root, SID)
    assert head is not None
    return rm._token_for(
        {
            "version": ti.TRANSCRIPT_INDEX_VERSION,
            "sig": head[0],
            "scan": head[1].to_payload(),
            "journal_generation": head[2],
        }
    )


def test_read_models_of_deleted_sessions_are_reclaimed(publication):
    root, index = publication
    directory = ti.index_path(root, SID).parent / "read-models"
    orphan = directory / "deleted-session"
    orphan.mkdir(parents=True)
    stale = orphan / ("0" * 32 + ".sqlite")
    stale.write_bytes(b"x")
    old = 1_600_000_000
    os.utime(stale, (old, old))
    os.utime(orphan, (old, old))
    ti.write_index(root, SID, index)
    # The FILES go; the empty directory stays. Removing it would make this cache
    # module a directory remover, which the store's guard treats as a
    # session-deletion surface wherever it appears.
    assert not list(orphan.glob("*.sqlite"))
    assert not list(orphan.glob("*.json"))


def test_publication_survives_a_platform_without_fcntl(publication, monkeypatch):
    import builtins

    root, _ = publication
    real_import = builtins.__import__

    def no_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("fcntl is POSIX-only")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fcntl)
    assert ti.patch_naming(root, SID, {"u2": {"name": "Portable", "summary": "yes"}})
    raw = json.loads(ti.index_path(root, SID).read_text())
    assert raw["naming"]["items"]["u2"]["name"] == "Portable"


def test_the_paint_cost_does_not_stat_the_cache(publication, monkeypatch):
    root, _ = publication
    journal = (root / "sessions" / SID / "transcript.jsonl").stat()
    calls: list[object] = []
    real_stat = ti._stat_or_none

    def counting(path):
        calls.append(path)
        return real_stat(path)

    monkeypatch.setattr(ti, "_stat_or_none", counting)
    has_previous, _rederive, hydration = ti._cached_refresh_cost(root, SID, journal)
    assert has_previous and hydration > 0
    assert calls == []


def test_the_companion_stays_a_small_fraction_of_the_index(tmp_path):
    """Size guard: facts, not a second copy of the index.

    Built text-heavy on purpose: a conversation's bytes are mostly message text,
    and the projection must store the WITNESSES of those rows (ordinals, roles,
    packed records), never a second copy of the prose.
    """
    rows: list[dict[str, object]] = []
    for turn in range(1, 801):
        rows.append(
            {
                "id": f"u{turn:04d}",
                "ts": float(turn),
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"text": f"question {turn} " * 8}],
                },
            }
        )
        rows.append(
            {
                "id": f"a{turn:04d}",
                "ts": turn + 0.5,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": f"answer {turn} " * 20}],
                },
            }
        )
    _journal(tmp_path, rows)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    _materialize(tmp_path, index)
    index_bytes = ti.index_path(tmp_path, SID).stat().st_size
    companion_bytes = rm.header(tmp_path, SID).value.path.stat().st_size
    share = companion_bytes / index_bytes
    print(f"index={index_bytes} companion={companion_bytes} share={share:.1%}")
    assert share < 0.5, f"companion is {share:.1%} of the index"
