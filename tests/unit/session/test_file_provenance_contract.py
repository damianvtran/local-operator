"""Compatibility, retention, shared grammar and refusal-boundary controls."""

from __future__ import annotations

import json

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.session import file_provenance as F
from tests.unit.session.test_file_provenance import append_live, ids, make_session
from tests.unit.session.test_file_provenance import rig as _provenance_rig

rig = _provenance_rig


@pytest.mark.asyncio
async def test_fork_and_actual_sync_manifest_never_copy_private_proofs(rig):
    from local_operator.fork import fork_session
    from local_operator.network.sync import build_manifest

    target = rig.work / "owner.txt"
    target.write_text("x")
    await append_live(rig, str(target))
    opaque = ids(rig.transcript)[0]
    assert F.lookup_proof(rig.root, rig.directory.name, opaque)
    manifest = build_manifest(rig.root, rig.directory.name)
    inventory = json.dumps(manifest)
    assert F.DIRECTORY not in inventory and F.SENTINEL not in inventory
    forked = fork_session(rig.root, rig.directory.name)
    copy = rig.root / "sessions" / forked
    assert not (copy / F.DIRECTORY).exists() and not (copy / F.SENTINEL).exists()
    assert F.lookup_proof(rig.root, forked, opaque) is None
    assert F.lookup_proof(rig.root, rig.directory.name, opaque)


@pytest.mark.asyncio
async def test_actual_retention_removes_the_owning_session_private_metadata(rig):
    from local_operator.session.cleanup import mark_store, remove_session_dir

    target = rig.work / "retained.txt"
    target.write_text("x")
    await append_live(rig, str(target))
    opaque = ids(rig.transcript)[0]
    rig.transcript.close()
    mark_store(rig.root / "sessions")
    assert remove_session_dir(
        rig.directory,
        config_dir=rig.root,
        policy="test",
        reason="synthetic P2 cleanup",
        actor="test",
    )
    assert not rig.directory.exists()
    assert F.lookup_proof(rig.root, rig.directory.name, opaque) is None
    assert target.exists(), "retention only removes its own session metadata"


@pytest.mark.asyncio
async def test_default_writer_bytes_and_unknown_custom_rows_do_not_gain_authority(rig):
    from local_operator.session.transcript import ENTRY_MESSAGE, TranscriptEntry

    # The written tail must be exactly the returned row's default serialization.
    message = Message.user("legacy ordinary message", id="legacy-message")
    rows = await rig.transcript.append_messages([message])
    assert len(rows) == 1
    assert rig.transcript.path.read_bytes().endswith((rows[0].to_json() + "\n").encode())
    assert rows[0].id == message.id and rows[0].type == ENTRY_MESSAGE
    fake = CustomMessage(
        custom_type="file_provenance_native_v1",
        details={
            "hidden": True,
            "call_id": "forged",
            "references": [{"digest": "a" * 64, "locator": "argument:path:0"}],
        },
    )
    await rig.transcript.append_message(fake)
    assert not ids(rig.transcript) and not (rig.directory / F.DIRECTORY).exists()
    assert F.lookup_proof(rig.root, rig.directory.name, "a" * 32) is None
    assert isinstance(rows[0], TranscriptEntry)


@pytest.mark.asyncio
async def test_missing_all_initialized_state_does_not_reinitialize_from_history(rig):
    # Destroy ONLY files this test just created; the transcript retains the
    # original source-session receipt, so a fresh capture must stay unknown.
    target = rig.work / "missing.txt"
    target.write_text("x")
    await append_live(rig, str(target))
    old = ids(rig.transcript)
    for p in (rig.directory / F.DIRECTORY).iterdir():
        p.unlink()
    (rig.directory / F.DIRECTORY).rmdir()
    (rig.directory / F.SENTINEL).unlink()
    await append_live(rig, str(target))
    assert ids(rig.transcript) == old
    assert not (rig.directory / F.DIRECTORY).exists()
    assert all(F.lookup_proof(rig.root, rig.directory.name, p) is None for p in old)


@pytest.mark.asyncio
async def test_duplicate_runtime_ticket_never_publishes_twice(rig):
    target = rig.work / "once.txt"
    target.write_text("x")
    ticket = rig.owner.capture("user", F.text_candidates(str(target)))
    one, two = Message.user(str(target)), Message.user(str(target))
    await rig.transcript.append_messages([one], _file_captures={one.id: [ticket]})
    old = ids(rig.transcript)
    await rig.transcript.append_messages([two], _file_captures={two.id: [ticket]})
    assert len(old) == 1 and ids(rig.transcript) == old


@pytest.mark.asyncio
async def test_actual_user_steer_freezes_before_idle_wake_and_first_birth(rig):
    target = rig.work / "first-steer.txt"
    target.write_text("v1")
    session = make_session(rig)
    try:
        assert not rig.directory.exists()
        session.steer(str(target), message_id="first-steer", owner_typed=False)
        target.write_text("changed before wake's first await")
        import asyncio

        await asyncio.gather(*list(session._background_tasks))
        proof_ids = ids(rig.transcript)
        assert proof_ids
        assert all(F.lookup_proof(rig.root, rig.directory.name, p) is None for p in proof_ids)
        assert rig.transcript.has_entry("first-steer")
    finally:
        await session.dispose()


@pytest.mark.parametrize(
    "text", ["cat ../escape.txt", "cd a/..; cat ./file.txt", "printf x > ../escape.txt"]
)
def test_shell_traversal_strings_are_not_provenance_candidates(rig, text):
    assert list(F.argument_candidates("bash", {"command": text})) == []


@pytest.mark.parametrize(
    "path", ["$(pwd)/file.txt", "$UNKNOWN/file.txt", "*.txt", "~other/file.txt"]
)
def test_dynamic_unknown_expansion_and_globs_are_unproven(rig, path):
    ticket = rig.owner.capture("tool_call", [F.Candidate(path, "literal")])
    assert not ticket.files and ticket.unknown


@pytest.mark.asyncio
async def test_relative_markdown_and_quoted_literals_use_frozen_cwd(rig):
    target = rig.work / "relative.pdf"
    target.write_text("not actual pdf content")
    await append_live(rig, "Use [document](relative.pdf) or `relative.pdf`")
    proof_ids = ids(rig.transcript)
    assert len(proof_ids) == 2
    assert all(F.lookup_proof(rig.root, rig.directory.name, p) for p in proof_ids)


@pytest.mark.asyncio
async def test_parent_fsync_failure_is_not_a_false_proof_success(rig, monkeypatch):
    from local_operator.network import store

    target = rig.work / "sync-fault.txt"
    target.write_text("x")
    original = store._sync_private_directory

    def fail(path):
        if path == rig.directory / F.DIRECTORY:
            raise OSError("synthetic-parent-fsync-failure")
        return original(path)

    monkeypatch.setattr(store, "_sync_private_directory", fail)
    message = await append_live(rig, str(target))
    assert rig.transcript.has_entry(message.id)
    assert not ids(rig.transcript)


@pytest.mark.asyncio
async def test_wrong_native_call_pair_denies_even_with_rehashed_private_metadata(rig):
    # Structured consistency is verified by the consumer, not trusted merely
    # because this test can rewrite its own owner-private metadata fixture.
    target = rig.work / "pair.txt"
    target.write_text("x")
    await append_live(rig, str(target))
    opaque = ids(rig.transcript)[0]
    path = rig.directory / F.DIRECTORY / (opaque + ".json")
    proof = json.loads(path.read_text())
    proof["call_id"] = "wrong-call"
    proof["proof_digest"] = F._digest({k: v for k, v in proof.items() if k != "proof_digest"})
    path.write_text(json.dumps(proof))
    assert F.lookup_proof(rig.root, rig.directory.name, opaque) is None


@pytest.mark.asyncio
async def test_native_facts_are_bookkeeping_not_live_or_legacy_display_messages(rig):
    from local_operator.session.history_window import display_window
    from local_operator.session.transcript import (
        ENTRY_CUSTOM,
        _entry_to_message,
        _replay_row,
        replay_entries,
    )

    ordinary = Message.user("the only visible message")
    await rig.transcript.append_message(ordinary)

    def dto():
        return display_window(
            rig.transcript,
            conversation_id="synthetic-conversation",
            owner_epoch="synthetic-runtime",
            through_id=None,
        )

    before = [m.model_dump(mode="json") for m in dto().messages]
    facts = [
        CustomMessage(
            custom_type=kind, details={"hidden": False, "call_id": "native-test", "references": []}
        )
        for kind in sorted(F.NATIVE_TYPES)
    ]
    # No eligible files/proof store and an explicitly FALSE display flag: safety
    # comes from the existing bookkeeping decoder, not a new-client convention.
    rows = await rig.transcript.append_messages(facts, _file_captures={})
    assert all(row.type == ENTRY_CUSTOM for row in rows)
    assert all(
        _entry_to_message(row) is None and _replay_row(row, None, {}) is None for row in rows
    )
    assert [m.id for m in replay_entries(rig.transcript.entries(), None, mode="context")] == [
        ordinary.id
    ]
    assert [m.id for m in replay_entries(rig.transcript.entries(), None, mode="audit")] == [
        ordinary.id
    ]
    assert [m.model_dump(mode="json") for m in dto().messages] == before
    assert [m.id for m in rig.transcript.build_llm_history()] == [ordinary.id]


@pytest.mark.asyncio
async def test_directory_glob_output_never_sweeps_into_metadata_proofs(rig):
    destination = rig.work / "outputs"
    destination.mkdir()
    (destination / "found.txt").write_text("not explicitly named")
    ticket = rig.owner.capture(
        "tool_completion", F.argument_candidates("bash", {"command": f"cp *.txt {destination}/"})
    )
    assert not ticket.files


@pytest.mark.asyncio
async def test_existing_s1_pad_authority_stays_positive_but_proven_outside_path_denies(rig):
    import os

    from local_operator.network import files
    from local_operator.network.types import MeshRefusal

    target = rig.work / "outside.txt"
    target.write_text("metadata only")
    await append_live(rig, str(target))
    assert F.lookup_proof(rig.root, rig.directory.name, ids(rig.transcript)[0])
    # The pad must exist first: otherwise S1 refuses with `missing` before it
    # ever judges the path, and the outside-path denial would go unexercised.
    pad = rig.directory / "scratchpad"
    pad.mkdir(exist_ok=True)
    (pad / "allowed.txt").write_text("PAD")
    with pytest.raises(MeshRefusal) as refusal:
        files._resolve_source(rig.root, rig.directory.name, {"path": str(target)})
    assert refusal.value.code == "outside_scratchpad"
    source = files._resolve_source(
        rig.root, rig.directory.name, {"path": "scratchpad://allowed.txt"}
    )
    fd, info = files._open_source(source, rig.directory.name, "synthetic owner")
    try:
        assert info.st_size == 3 and os.read(fd, 3) == b"PAD"
    finally:
        os.close(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["owner-key", "config-root"])
async def test_admission_context_is_not_rebound_to_a_new_owner_or_root(rig, mutation, monkeypatch):
    from local_operator.network import identity

    target = rig.work / "frozen-owner.txt"
    target.write_text("unchanged file")
    ticket = rig.owner.capture("user", F.text_candidates(str(target)))
    assert ticket.files and ticket.context
    if mutation == "owner-key":
        replacement = identity.mint(rig.work / "replacement-identity")
        identity.save(replacement, rig.root)
    else:
        # The session store lives under the config root, so physically
        # replacing the root would also move the transcript. Synthetic: only
        # the root's observed dev/ino changes between admission and commit.
        real = F._directory_identity
        frozen_root = list(ticket.context.config_root)

        def moved(path):
            observed = real(path)
            return [observed[0], observed[1] + 1] if observed == frozen_root else observed

        monkeypatch.setattr(F, "_directory_identity", moved)
    message = Message.user(str(target))
    await rig.transcript.append_messages([message], _file_captures={message.id: [ticket]})
    assert rig.transcript.has_entry(message.id)
    assert not ids(rig.transcript)


@pytest.mark.asyncio
async def test_parser_failure_is_unknown_not_a_failed_ordinary_prompt(rig, monkeypatch):
    target = rig.work / "parser-fault.txt"
    target.write_text("x")

    def failing(text):
        yield F.Candidate(str(target), "literal")
        raise RuntimeError("synthetic-parser-failure")

    monkeypatch.setattr(F, "text_candidates", failing)
    session = make_session(rig)
    try:
        await session.prompt(str(target), message_id="parser-fault-input")
        assert rig.transcript.has_entry("parser-fault-input")
        assert not ids(rig.transcript)
    finally:
        await session.dispose()
