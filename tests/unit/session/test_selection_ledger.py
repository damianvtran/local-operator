"""The expendable ledger is source evidence, never a cached registry verdict."""

import pytest

from local_operator.session import model_selection as selection
from local_operator.session.tail_anchor import read_anchor
from local_operator.session.transcript import Transcript


@pytest.mark.asyncio
async def test_head_selection_ledger_equals_full_scan_without_backward_walk(tmp_path, monkeypatch):
    """A cold process must not parse the megabytes after a birth selection."""
    transcript = Transcript(tmp_path)
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/default"})
    # Imported custom rows exercise the shared durable writer without generating
    # thousands of individual fsyncs. The selection remains near BOF.
    from local_operator.session.transcript import TranscriptEntry

    rows = [
        TranscriptEntry(
            f"pad-{i}", 1, "custom", {"custom_type": "padding", "details": {"blob": "x" * 8192}}
        )
        for i in range(512)
    ]
    await transcript.append_imported(rows)
    expected = selection.selection_from_payloads(selection._forward_payloads(tmp_path))
    assert expected is not None
    anchor = read_anchor(tmp_path)
    assert anchor is not None and anchor.selection_payloads is not None
    monkeypatch.setattr(
        selection, "_settled_selection", lambda *_: pytest.fail("whole-journal backward walk")
    )
    assert selection._read_selection(tmp_path) == expected
    transcript.close()


@pytest.mark.asyncio
async def test_ledger_legacy_provenance_and_registry_changes(tmp_path, monkeypatch):
    transcript = Transcript(tmp_path)
    await transcript.append_custom(
        "selected_model", {"selector": "test/default", "boot": "test/birth"}
    )
    await transcript.append_custom(
        "frontend_state_checkpoint_v1",
        {
            "state": {
                "selected_model": {
                    "provider": "test",
                    "model_id": "default",
                    "reasoning_effort": "high",
                }
            }
        },
    )

    def oracle():
        return selection.selection_from_payloads(selection._forward_payloads(tmp_path))

    assert selection._read_selection(tmp_path) == oracle()
    stored = oracle()
    assert stored is not None and stored.boot_selector == "test/birth"
    import local_operator.providers.registry as registry

    original = registry.get_provider_definition
    monkeypatch.setattr(
        registry,
        "get_provider_definition",
        lambda provider: None if provider == "test" else original(provider),
    )
    assert selection._read_selection(tmp_path) == oracle() is None
    transcript.close()


@pytest.mark.asyncio
async def test_ledger_failure_does_not_fail_commit_and_truncate_falls_back(tmp_path, monkeypatch):
    from local_operator.session import tail_anchor

    transcript = Transcript(tmp_path)
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/first"})
    before = transcript.path.read_bytes()
    monkeypatch.setattr(
        tail_anchor, "write_anchor", lambda *_: (_ for _ in ()).throw(OSError("disk full"))
    )
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/second"})
    stored = selection._read_selection(tmp_path)
    assert stored is not None and stored.model_id == "second"
    transcript.path.write_bytes(before)
    stored = selection._read_selection(tmp_path)
    assert stored is not None and stored.model_id == "first"
    transcript.close()


@pytest.mark.asyncio
async def test_ledger_concurrent_commits_compaction_and_missing_sidecar(tmp_path):
    import asyncio

    from local_operator.session.tail_anchor import anchor_path

    transcript = Transcript(tmp_path)
    await asyncio.gather(
        *(
            transcript.append_custom(
                "selected_model", {"version": 2, "selector": f"test/model-{i}"}
            )
            for i in range(12)
        )
    )
    await transcript.append_custom("subagent_roster", {"pad": "x" * 16384})
    await transcript.append_custom("subagent_roster", {"pad": "new"})

    def oracle():
        return selection.selection_from_payloads(selection._forward_payloads(tmp_path))

    assert selection._read_selection(tmp_path) == oracle()
    old_inode = transcript.path.stat().st_ino
    assert await transcript.compact_file(min_reclaim_bytes=1) > 0
    assert transcript.path.stat().st_ino != old_inode
    assert selection._read_selection(tmp_path) == oracle()
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/final"})
    assert selection._read_selection(tmp_path) == oracle()
    anchor_path(tmp_path).unlink()
    assert selection._read_selection(tmp_path) == oracle()
    transcript.close()


@pytest.mark.asyncio
async def test_failed_journal_fsync_cannot_publish_selection(tmp_path, monkeypatch):
    import os

    transcript = Transcript(tmp_path)
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/first"})
    before = transcript.path.read_bytes()
    old_anchor = read_anchor(tmp_path)
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda *_: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        await transcript.append_custom(
            "selected_model", {"version": 2, "selector": "test/rejected"}
        )
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert transcript.path.read_bytes() == before
    assert read_anchor(tmp_path) == old_anchor
    stored = selection._read_selection(tmp_path)
    assert stored is not None and stored.model_id == "first"
    transcript.close()


@pytest.mark.asyncio
async def test_no_selection_and_torn_tail_are_exact(tmp_path):
    from local_operator.session.tail_anchor import update_selection_after_commit

    transcript = Transcript(tmp_path)
    await transcript.append_custom("other", {"pad": "hello"})
    assert selection._read_selection(tmp_path) is None
    with transcript.path.open("ab") as handle:
        handle.write(b'{"id":"torn","type":"custom","payload":{"custom_type":"selected_model",')
    update_selection_after_commit(tmp_path)
    assert (
        selection._read_selection(tmp_path)
        == selection.selection_from_payloads(selection._forward_payloads(tmp_path))
        is None
    )
    transcript.close()


@pytest.mark.asyncio
async def test_selection_only_record_never_grants_a_replay_cut(tmp_path):
    from local_operator.session.tail_anchor import anchor_path, proves_absent
    from local_operator.session.transcript import read_replay_suffix

    transcript = Transcript(tmp_path)
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/default"})
    anchor = read_anchor(tmp_path)
    assert anchor is not None and not anchor.checkpoint_absent
    assert not proves_absent(tmp_path)
    with_ledger = read_replay_suffix(tmp_path, checkpoint_types="frontend_state_checkpoint_v1")
    anchor_path(tmp_path).unlink()
    without_ledger = read_replay_suffix(tmp_path, checkpoint_types="frontend_state_checkpoint_v1")
    assert with_ledger == without_ledger
    transcript.close()


@pytest.mark.asyncio
async def test_real_prewarm_preserves_selection_hit(tmp_path, monkeypatch):
    """R1: the absence-proof writer must not erase independent source evidence."""
    from local_operator.session.index_prewarm import write_tail_anchor

    directory = tmp_path / "sessions" / "prewarm-ledger"
    transcript = Transcript(directory)
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/first"})
    await transcript.append_custom("padding", {"blob": "x" * (4 << 20)})
    expected = selection.selection_from_payloads(selection._forward_payloads(directory))
    assert write_tail_anchor(tmp_path, directory.name)
    anchor = read_anchor(directory)
    assert anchor is not None and anchor.checkpoint_absent
    monkeypatch.setattr(selection, "_settled_selection", lambda *_: pytest.fail("full settle scan"))
    monkeypatch.setattr(selection, "_forward_payloads", lambda *_: pytest.fail("full forward scan"))
    assert selection._read_selection(directory) == expected
    transcript.close()


@pytest.mark.asyncio
async def test_ordinary_same_size_rewrite_invalidates_selection_and_writer_prefix(tmp_path):
    """R2: a stable terminal row is not proof an earlier selector is unchanged."""
    transcript = Transcript(tmp_path)
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/first"})
    await transcript.append_custom("padding", {"blob": "unchanged terminal row"})
    stored = selection.read_model_selection(tmp_path)
    assert stored is not None and stored.model_id == "first"
    before = transcript.path.stat()
    data = transcript.path.read_bytes()
    rewritten = data.replace(b"test/first", b"test/other")
    assert len(data) == len(rewritten) and rewritten != data
    with transcript.path.open("r+b") as handle:
        handle.write(rewritten)
    after = transcript.path.stat()
    assert (after.st_ino, after.st_size) == (before.st_ino, before.st_size)
    assert (after.st_mtime_ns, after.st_ctime_ns) != (before.st_mtime_ns, before.st_ctime_ns)
    expected = selection.selection_from_payloads(selection._forward_payloads(tmp_path))
    assert selection.read_model_selection(tmp_path) == expected
    assert expected is not None and expected.model_id == "other"
    # The next ordinary commit must bootstrap from changed source, not bless the
    # stale persistent prefix merely because that source has now grown.
    await transcript.append_custom("padding", {"blob": "after edit"})
    assert selection._read_selection(tmp_path) == expected
    transcript.close()


@pytest.mark.asyncio
async def test_generation_proof_keeps_normal_writer_incremental(tmp_path, monkeypatch):
    from local_operator.session import transcript as transcript_module

    transcript = Transcript(tmp_path)
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/first"})
    anchor = read_anchor(tmp_path)
    assert anchor is not None
    prefix = anchor.size
    starts = []
    original = transcript_module._iter_complete_lines_forward

    def tracked(handle, start, **kwargs):
        starts.append(start)
        yield from original(handle, start, **kwargs)

    monkeypatch.setattr(transcript_module, "_iter_complete_lines_forward", tracked)
    await transcript.append_custom("padding", {"blob": "ordinary append"})
    assert starts == [prefix], "a normal commit must not reparse its whole prefix"
    source = transcript.path.stat()
    anchor = read_anchor(tmp_path)
    assert anchor is not None
    assert anchor.selection_mtime_ns == source.st_mtime_ns
    assert anchor.selection_ctime_ns == source.st_ctime_ns
    stored = selection._read_selection(tmp_path)
    assert stored is not None and stored.model_id == "first"
    transcript.close()


@pytest.mark.asyncio
async def test_missing_generation_falls_back_and_recovered_state_survives_prewarm(tmp_path):
    import json

    from local_operator.session.index_prewarm import write_tail_anchor
    from local_operator.session.tail_anchor import anchor_path, read_selection_payloads

    directory = tmp_path / "sessions" / "recovered"
    transcript = Transcript(directory)
    await transcript.append_custom("selected_model", {"version": 2, "selector": "test/first"})
    await transcript.append_custom(
        "selected_model", {"version": 2, "selector": "unknown-provider/no"}
    )
    expected = selection.selection_from_payloads(selection._forward_payloads(directory))
    assert expected is not None and expected.recovered
    assert write_tail_anchor(tmp_path, directory.name)
    assert selection._read_selection(directory) == expected
    path = anchor_path(directory)
    raw = json.loads(path.read_text())
    raw.pop("selection_ctime_ns")
    path.write_text(json.dumps(raw))
    assert read_selection_payloads(directory) is None
    assert selection._read_selection(directory) == expected
    transcript.close()


@pytest.mark.asyncio
async def test_archived_session_answers_selection_without_a_ledger(tmp_path):
    """A compressed journal has no plain inode, so the ledger must stand aside.

    This is the behaviour the archive-awareness guard's allow-list row claims: the
    plain stat fails, the reader answers None, and the model-selection fallback
    pays today's exact scan through the codec resolver.
    """
    from local_operator.harness.types import Message
    from local_operator.session import compression
    from local_operator.session.tail_anchor import read_selection_payloads

    transcript = Transcript(tmp_path)
    await transcript.append_custom(
        "selected_model", {"version": 2, "selector": "test/second-choice"}
    )
    await transcript.append_message(Message.user("hello"))
    transcript.close()

    outcome = compression.compress_session(tmp_path)
    assert outcome.status == "compressed", outcome
    assert not (tmp_path / "transcript.jsonl").is_file()

    assert read_selection_payloads(tmp_path) is None
    assert not (tmp_path / "tail-anchor.json").exists()
    assert not (tmp_path / "tail-anchor.v1.json").exists()
    selected = selection.read_model_selection(tmp_path)
    assert selected is not None and selected.model_id == "second-choice"
