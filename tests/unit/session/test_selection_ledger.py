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
    assert oracle().boot_selector == "test/birth"
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
    assert selection._read_selection(tmp_path).model_id == "second"
    transcript.path.write_bytes(before)
    assert selection._read_selection(tmp_path).model_id == "first"
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
    assert selection._read_selection(tmp_path).model_id == "first"
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
