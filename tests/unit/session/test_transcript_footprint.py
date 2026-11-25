"""Footprint behaviour of the transcript: slim rows, the prune journal, and
the file compaction that folds it in.

Every test here defends an invariant the size optimisation is allowed to
break in exactly zero ways: what comes back out of ``build_llm_history`` must
still be the conversation, and ``first_kept_entry_id`` must still resolve.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import threading
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    CustomMessage,
    Message,
    TextContent,
    ToolCall,
    Usage,
)
from local_operator.session.transcript import (
    ENTRY_PRUNE,
    Transcript,
    encode_message_payload,
)


def _row(transcript: Transcript, index: int) -> dict[str, Any]:
    return json.loads(transcript.path.read_text().splitlines()[index])


@pytest.mark.asyncio
async def test_slim_row_omits_defaults_and_duplicate_id(tmp_path):
    """A plain user message costs its role and its text, nothing else."""
    transcript = Transcript(tmp_path / "sess")
    message = Message.user("hello")
    await transcript.append_message(message)

    payload = _row(transcript, 0)["payload"]
    assert payload["role"] == "user"
    assert payload["content"][0]["text"] == "hello"
    # These are all pydantic defaults; writing them is pure overhead.
    for absent in ("tool_calls", "tool_call_id", "tool_name", "is_error", "usage"):
        assert absent not in payload
    # The entry id already carries the message id.
    assert "id" not in payload
    assert _row(transcript, 0)["id"] == message.id


@pytest.mark.asyncio
async def test_slim_row_still_replays_identically(tmp_path):
    """Omitting defaults must be invisible to replay, not merely small."""
    transcript = Transcript(tmp_path / "sess")
    original = [
        Message.user("write the file"),
        Message.assistant(
            "",
            tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py"})],
            stop_reason="toolUse",
            usage=Usage(input_tokens=11, output_tokens=3),
        ),
        Message(
            role="tool",
            content=[TextContent(text="Created a.py")],
            tool_call_id="c1",
            tool_name="write",
        ),
    ]
    for message in original:
        await transcript.append_message(message)

    replayed = Transcript(tmp_path / "sess").build_llm_history()
    assert [m.model_dump() for m in replayed] == [m.model_dump() for m in original]


@pytest.mark.asyncio
async def test_redundant_raw_arguments_dropped_but_odd_ones_kept(tmp_path):
    """The escaped duplicate goes; a string that does not round-trip stays."""
    transcript = Transcript(tmp_path / "sess")
    redundant = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    redundant.raw_arguments = '{"command": "ls"}'
    divergent = ToolCall(id="c2", name="bash", arguments={"command": "ls"})
    divergent.raw_arguments = '{"command": "rm -rf /"}'
    await transcript.append_message(Message.assistant("", tool_calls=[redundant, divergent]))

    calls = _row(transcript, 0)["payload"]["tool_calls"]
    assert "raw_arguments" not in calls[0]
    assert calls[1]["raw_arguments"] == '{"command": "rm -rf /"}'

    # Replay recovers the arguments for both; only byte-level fidelity of the
    # redundant one is given up, and wire clients regenerate that with
    # json.dumps.
    replayed = Transcript(tmp_path / "sess").build_llm_history()
    first = replayed[0]
    assert isinstance(first, Message)
    assert [c.arguments for c in first.tool_calls] == [
        {"command": "ls"},
        {"command": "ls"},
    ]
    assert first.tool_calls[1].raw_arguments == '{"command": "rm -rf /"}'


@pytest.mark.asyncio
async def test_legacy_fat_rows_still_load(tmp_path):
    """Rows written by the pre-slim encoder must keep replaying."""
    directory = tmp_path / "sess"
    directory.mkdir()
    message = Message.user("hello")
    legacy = {
        "id": message.id,
        "ts": 1.0,
        "type": "message",
        "payload": {"kind": "message", **message.model_dump()},
    }
    (directory / "transcript.jsonl").write_text(json.dumps(legacy) + "\n")

    replayed = Transcript(directory).build_llm_history()
    assert len(replayed) == 1
    first = replayed[0]
    assert isinstance(first, Message)
    assert first.text == "hello"
    assert first.id == message.id


@pytest.mark.asyncio
async def test_custom_entry_keeps_its_entry_id(tmp_path):
    """A rendered custom entry is a legal ``first_kept_entry_id`` target, so
    its id must survive the round trip that no longer stores it in-payload."""
    transcript = Transcript(tmp_path / "sess")
    marker = CustomMessage(custom_type="skill_prompt", details={"name": "deploy"})
    await transcript.append_message(marker)

    replayed = Transcript(tmp_path / "sess").build_llm_history()
    first = replayed[0]
    assert isinstance(first, CustomMessage)
    assert first.id == marker.id
    assert first.details == {"name": "deploy"}


@pytest.mark.asyncio
async def test_prune_journal_applies_on_replay(tmp_path):
    """The whole point: a resumed session sees the blanked result, not the
    12 KB output the live session already threw away."""
    transcript = Transcript(tmp_path / "sess")
    big = Message(
        role="tool",
        content=[TextContent(text="x" * 12000)],
        tool_call_id="c1",
        tool_name="bash",
    )
    await transcript.append_message(big)
    await transcript.append_prune(big.id, "[Superseded by a newer read of this file]")

    replayed = Transcript(tmp_path / "sess").build_llm_history()
    assert len(replayed) == 1
    first = replayed[0]
    assert isinstance(first, Message)
    assert first.text == "[Superseded by a newer read of this file]"
    # Flagged the way the live pruning pass flags it, so the next pass skips
    # it instead of re-blanking and re-journalling it every turn.
    assert (first.provider_payload or {}).get("pruned") is True


@pytest.mark.asyncio
async def test_compact_file_folds_journal_and_shrinks_disk(tmp_path):
    transcript = Transcript(tmp_path / "sess")
    keep = Message.user("keep me")
    await transcript.append_message(keep)
    big = Message(
        role="tool",
        content=[TextContent(text="y" * 400_000)],
        tool_call_id="c1",
        tool_name="bash",
    )
    await transcript.append_message(big)
    await transcript.append_prune(big.id, "[pruned]")

    before = transcript.path.stat().st_size
    expected = transcript.reclaimable_bytes()
    reclaimed = await transcript.compact_file()

    assert reclaimed == expected > 0
    assert transcript.path.stat().st_size == before - reclaimed
    # Journal folded away, message rows intact and in order.
    types = [e.type for e in transcript.entries()]
    assert ENTRY_PRUNE not in types
    assert len(types) == 2

    replayed = Transcript(tmp_path / "sess").build_llm_history()
    assert [m.text for m in replayed if isinstance(m, Message)] == ["keep me", "[pruned]"]
    assert [m.id for m in replayed] == [keep.id, big.id]


@pytest.mark.asyncio
async def test_compact_file_below_threshold_is_a_no_op(tmp_path):
    """A prune pass runs most turns; rewriting a large file for a few hundred
    bytes would cost more I/O than it reclaims."""
    transcript = Transcript(tmp_path / "sess")
    small = Message(role="tool", content=[TextContent(text="z" * 200)], tool_call_id="c1")
    await transcript.append_message(small)
    await transcript.append_prune(small.id, "[pruned]")

    before = transcript.path.read_bytes()
    assert await transcript.compact_file() == 0
    assert transcript.path.read_bytes() == before
    # Still correct on replay — folding is an optimisation, not the mechanism.
    first = Transcript(tmp_path / "sess").build_llm_history()[0]
    assert isinstance(first, Message)
    assert first.text == "[pruned]"


@pytest.mark.asyncio
async def test_compaction_boundary_survives_folding(tmp_path):
    """``first_kept_entry_id`` must still resolve after the file is rewritten
    — the documented fallback is 'replay everything', so a broken reference
    is a silent doubling of the prompt rather than a crash."""
    transcript = Transcript(tmp_path / "sess")
    dropped = Message.user("ancient history")
    await transcript.append_message(dropped)
    pruned = Message(
        role="tool",
        content=[TextContent(text="w" * 400_000)],
        tool_call_id="c1",
        tool_name="bash",
    )
    await transcript.append_message(pruned)
    kept = Message.user("recent")
    await transcript.append_message(kept)
    await transcript.append_compaction("summary so far", kept.id, tokens_before=999)
    await transcript.append_prune(pruned.id, "[pruned]")

    assert await transcript.compact_file() > 0

    replayed = Transcript(tmp_path / "sess").build_llm_history()
    marker = replayed[0]
    assert isinstance(marker, CustomMessage)
    assert marker.custom_type == "compaction_summary"
    assert marker.details["summary"] == "summary so far"
    # Exactly the kept window: the cut point resolved, so nothing before it
    # came back.
    assert [m.text for m in replayed[1:] if isinstance(m, Message)] == ["recent"]


@pytest.mark.asyncio
async def test_compact_file_heals_legacy_roster_bloat(tmp_path):
    """A pre-v0.40.0 transcript with a long run of superseded ``subagent_roster``
    custom entries sheds all but the newest on compaction, reclaiming the bytes,
    while messages, replay, and ``latest_custom`` stay intact.

    This is the heal for the real 125 MB session that re-appended a full roster
    snapshot on every roster move. The old bloat never journals a prune, so the
    fold must run on the superseded-custom signal alone (no pending prune)."""
    transcript = Transcript(tmp_path / "sess")
    keep_msg = Message.user("keep me")
    await transcript.append_message(keep_msg)
    # A long run of superseded roster snapshots, each carrying a big record tail
    # (the pre-cap shape). Only the last one is live; the rest are dead weight.
    for generation in range(50):
        await transcript.append_custom(
            "subagent_roster",
            {"generation": generation, "jobs": [], "records": [{"blob": "X" * 2_000}]},
        )
    # An unrelated newest-wins custom that is NOT collapsible must be untouched.
    await transcript.append_custom("todo_snapshot", {"items": ["a"]})

    before = transcript.path.stat().st_size
    n_roster_before = sum(
        1
        for e in transcript.entries()
        if e.type == "custom" and e.payload.get("custom_type") == "subagent_roster"
    )
    assert n_roster_before == 50

    # No pending prune: the heal fires on the superseded-custom signal alone.
    expected = transcript.reclaimable_bytes()
    reclaimed = await transcript.compact_file(min_reclaim_bytes=1)
    assert reclaimed == expected > 0
    assert transcript.path.stat().st_size == before - reclaimed

    reopened = Transcript(tmp_path / "sess")
    roster_entries = [
        e
        for e in reopened.entries()
        if e.type == "custom" and e.payload.get("custom_type") == "subagent_roster"
    ]
    # Exactly one roster entry survives, and it is the NEWEST (generation 49).
    assert len(roster_entries) == 1
    assert roster_entries[0].payload["details"]["generation"] == 49
    # latest_custom is unchanged by the collapse.
    latest_roster = reopened.latest_custom("subagent_roster")
    assert latest_roster is not None and latest_roster["generation"] == 49
    # The non-collapsible custom and the message are byte-preserved.
    assert reopened.latest_custom("todo_snapshot") == {"items": ["a"]}
    replayed = reopened.build_llm_history()
    assert [m.text for m in replayed if isinstance(m, Message)] == ["keep me"]


@pytest.mark.asyncio
async def test_compact_file_keeps_a_single_roster_entry(tmp_path):
    """One roster entry is already minimal: nothing to collapse, no rewrite."""
    transcript = Transcript(tmp_path / "sess")
    await transcript.append_message(Message.user("hi"))
    await transcript.append_custom("subagent_roster", {"generation": 0, "records": []})
    before = transcript.path.read_bytes()
    assert await transcript.compact_file(min_reclaim_bytes=1) == 0
    assert transcript.path.read_bytes() == before


def test_encode_rejects_nothing_it_cannot_rebuild():
    """Belt and braces on the encoder itself: every field it drops must be
    reconstructible by pydantic from the model default."""
    message = Message(
        role="tool",
        content=[TextContent(text="out")],
        tool_call_id="c1",
        tool_name="grep",
        is_error=True,
    )
    payload = encode_message_payload(message)
    payload["id"] = message.id
    assert Message.model_validate(payload).model_dump() == message.model_dump()


# -- the cold-open load path: what compaction can shed, and how the file is read


def test_the_collapsible_allowlist_names_the_frontend_checkpoint_type():
    """The type is written as a LITERAL; this is what stops it drifting.

    ``transcript`` must stay a leaf module (``read_replay_suffix`` says why), and
    the constant's owner — ``session/frontend_state.py`` — reaches the TUI, so the
    allowlist cannot import it. A rename there would otherwise turn the fold off
    silently, which is the failure this test exists for.
    """
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
    from local_operator.session.transcript import _COLLAPSIBLE_CUSTOM_TYPES

    assert FRONTEND_CHECKPOINT_CUSTOM_TYPE in _COLLAPSIBLE_CUSTOM_TYPES


@pytest.mark.asyncio
async def test_compaction_drops_superseded_frontend_checkpoints(tmp_path):
    """Every turn end appends the FULL frontend state; only the newest is read.

    Measured on the operator's store: 1,100 rows / 379.8 MB across 216 sessions,
    93.1% of it superseded, and on the largest journal those rows alone are 75% of
    the whole-file parse cost. Every reader takes the newest entry of the type, so
    the older copies are dead bytes — and the fold must keep the answer a reader
    gets before it identical.
    """
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    transcript = Transcript(tmp_path / "sess")
    await transcript.append_message(Message.user("hello"))
    for index in range(3):
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"cwd": f"/work/{index}"}}
        )
    before = transcript.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert before == {"state": {"cwd": "/work/2"}}

    assert await transcript.compact_file(min_reclaim_bytes=0) > 0

    rows = [
        entry
        for entry in transcript.entries()
        if entry.payload.get("custom_type") == FRONTEND_CHECKPOINT_CUSTOM_TYPE
    ]
    assert len(rows) == 1, "the fold kept more than the newest checkpoint"
    # ... and what a reader is told did not change: newest-wins is the contract
    # every caller depends on (cold open, desktop locate, the picker pane).
    assert transcript.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE) == before
    assert transcript.entries()[0].payload["content"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_compacting_checkpoints_leaves_the_message_history_alone(tmp_path):
    """The fold is a byte operation on ONE type, not a rewrite of the transcript."""
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    transcript = Transcript(tmp_path / "sess")
    for index in range(3):
        await transcript.append_message(Message.user(f"turn {index}"))
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"cwd": f"/work/{index}"}}
        )
    replayed = [m.text for m in transcript.build_llm_history() if isinstance(m, Message)]
    await transcript.compact_file(min_reclaim_bytes=0)
    reopened = Transcript(tmp_path / "sess")
    assert [m.text for m in reopened.build_llm_history() if isinstance(m, Message)] == replayed
    # Only the NEWEST checkpoint survives, and the surviving rows keep their
    # relative order — a fold that reordered rows would change what a reader
    # paging by ``before_id`` reconstructs.
    assert [entry.type for entry in reopened.entries()] == [
        "message",
        "message",
        "message",
        "custom",
    ]
    assert reopened.entries()[-1].payload["details"] == {"state": {"cwd": "/work/2"}}


@pytest.mark.asyncio
async def test_construction_streams_the_journal_instead_of_materialising_it(tmp_path, monkeypatch):
    """3.38x the file, gone — asserted as a MECHANISM and as a peak.

    ``Transcript.__init__`` used to build the whole journal twice before parsing
    the first row (``read_text()``'s single decoded string, then ``splitlines()``'s
    list of it), measured at 885 MB of traced peak on a 262 MB journal. The
    mechanism assertion is the half that cannot rot: ``read_text`` is made to
    raise, so a revert to the eager form fails here rather than merely being
    slower. The peak bound is the other half, with the measured separation stated
    so it can be re-derived: on an 8.3 MB synthetic journal of 2 000 rows the
    eager form peaks at 2.24x the file and the streamed form at 1.24x.
    """
    directory = tmp_path / "sess"
    directory.mkdir(parents=True)
    path = directory / "transcript.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(2_000):
            handle.write(
                json.dumps(
                    {
                        "id": f"{index:032x}",
                        "ts": 1.0 + index,
                        "type": "message",
                        "payload": {
                            "kind": "message",
                            "role": "user",
                            "content": [{"text": f"row {index} " + "x" * 4_000}],
                        },
                    }
                )
                + "\n"
            )
    size = path.stat().st_size

    real_read_text = Path.read_text

    def refuse(self: Path, *args: Any, **kwargs: Any) -> str:
        # Only the JOURNAL is a tripwire: ``created_at.json`` and the other
        # sidecars are small and are still read this way.
        if self == path:
            raise AssertionError("Transcript.__init__ materialised the journal as one string")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse)
    gc.collect()
    tracemalloc.start()
    try:
        transcript = Transcript(directory)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(transcript.entries()) == 2_000
    assert peak < 1.8 * size, f"peak {peak / size:.2f}x the file (eager form was 2.24x)"


@pytest.mark.asyncio
async def test_construction_keeps_order_tolerance_and_a_torn_tail(tmp_path):
    """The streaming read must not change WHICH rows load, or in what order.

    A blank line, a malformed row and a final row with no trailing newline, in
    one journal: the first two are dropped individually, and the last one is
    still a row because the handle yields the unterminated tail.
    """
    directory = tmp_path / "sess"
    directory.mkdir(parents=True)
    path = directory / "transcript.jsonl"
    good = json.dumps(
        {
            "id": "a" * 32,
            "ts": 1.0,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": "first"}]},
        }
    )
    tail = json.dumps(
        {
            "id": "b" * 32,
            "ts": 2.0,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": "torn tail"}]},
        }
    )
    path.write_text(f"{good}\n\n{{not json\n{tail}", encoding="utf-8")
    transcript = Transcript(directory)
    assert [entry.id for entry in transcript.entries()] == ["a" * 32, "b" * 32]
    assert transcript.entries()[-1].payload["content"][0]["text"] == "torn tail"


# -- superseded collapsible rows are never materialised on open (incident
#    2026-11-24, session 96cf57e7acb7: a 974 MB journal that never pruned, so
#    ``compact_file`` never ran, cost 2,392 MiB of footprint per cold open)


async def _journal_with_superseded_checkpoints(
    directory: Path, *, copies: int, size: int
) -> tuple[list[str], dict[str, Any] | None]:
    """A closed journal: messages interleaved with ``copies`` checkpoints of ``size`` bytes."""
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    transcript = Transcript(directory)
    for index in range(copies):
        await transcript.append_message(Message.user(f"turn {index}"))
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"n": index, "pad": "x" * size}}
        )
    newest = transcript.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    texts = [m.text for m in transcript.build_llm_history() if isinstance(m, Message)]
    transcript.close()
    return texts, newest


def _checkpoint_rows(transcript: Transcript) -> list[Any]:
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    return [
        entry
        for entry in transcript.entries()
        if entry.payload.get("custom_type") == FRONTEND_CHECKPOINT_CUSTOM_TYPE
    ]


@pytest.mark.asyncio
async def test_owning_open_folds_superseded_rows_off_disk_without_loading_them(
    tmp_path, monkeypatch
):
    """An owning writer folds the dead rows on open; survivors stay byte-identical.

    Mechanism, not just outcome: ``TranscriptEntry.from_json`` is counted, so a
    revert that parses the superseded rows (the incident's 2.45x-the-file
    residency) fails here even though the end state would look the same.
    """
    from local_operator.session import transcript as transcript_module
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    directory = tmp_path / "sess"
    texts, newest = await _journal_with_superseded_checkpoints(directory, copies=4, size=120 * 1024)
    path = directory / "transcript.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    survivors = [line for line in lines if '"pad"' not in line or '"n":3' in line]
    assert len(lines) == 8 and len(survivors) == 5

    parsed: list[str] = []
    real = transcript_module.TranscriptEntry.from_json

    def counting(line: str):
        parsed.append(line)
        return real(line)

    monkeypatch.setattr(transcript_module.TranscriptEntry, "from_json", staticmethod(counting))
    reopened = Transcript(directory)

    assert not any('"pad"' in line and '"n":3' not in line for line in parsed)
    assert path.read_text() == "".join(survivors), "fold must copy survivors verbatim"
    assert not path.with_suffix(".jsonl.fold").exists()
    assert len(_checkpoint_rows(reopened)) == 1
    assert reopened.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE) == newest
    assert [m.text for m in reopened.build_llm_history() if isinstance(m, Message)] == texts
    assert reopened.reclaimable_bytes() == 0
    reopened.close()


@pytest.mark.asyncio
async def test_unowned_open_skips_in_memory_and_compact_file_reclaims_exactly(tmp_path):
    """Below the rewrite threshold the rows stay on disk, are skipped, and are priced.

    ``reclaimable_bytes`` is compared for EQUALITY with what ``compact_file``
    frees, so bytes the open skipped must be in both — and must make
    ``compact_file`` run even with no prune and no in-memory superseded row.
    """
    directory = tmp_path / "sess"
    texts, newest = await _journal_with_superseded_checkpoints(directory, copies=3, size=64)
    path = directory / "transcript.jsonl"
    size_before = path.stat().st_size

    reopened = Transcript(directory)
    assert path.stat().st_size == size_before, "below the threshold the open must not rewrite"
    assert len(_checkpoint_rows(reopened)) == 1
    assert [m.text for m in reopened.build_llm_history() if isinstance(m, Message)] == texts
    expected = reopened.reclaimable_bytes()
    assert expected > 0
    assert await reopened.compact_file(min_reclaim_bytes=0) == expected
    assert path.stat().st_size == size_before - expected
    assert reopened.reclaimable_bytes() == 0
    reopened.close()


@pytest.mark.asyncio
async def test_read_only_open_skips_superseded_rows_and_never_writes(tmp_path):
    directory = tmp_path / "sess"
    texts, newest = await _journal_with_superseded_checkpoints(directory, copies=4, size=120 * 1024)
    path = directory / "transcript.jsonl"
    before = path.read_bytes()
    viewer = Transcript(directory, read_only=True)
    assert path.read_bytes() == before
    assert len(_checkpoint_rows(viewer)) == 1
    assert [m.text for m in viewer.build_llm_history() if isinstance(m, Message)] == texts


@pytest.mark.asyncio
async def test_failed_fold_on_open_leaves_the_journal_intact(tmp_path, monkeypatch):
    """ENOSPC mid-fold (the incident's disk was at 0.65 GiB) must cost nothing.

    The original journal is untouched, the partial temp file is removed, and
    the open still loads bounded by skipping the rows in memory.
    """
    import errno

    from local_operator.session import transcript as transcript_module

    directory = tmp_path / "sess"
    texts, _ = await _journal_with_superseded_checkpoints(directory, copies=4, size=120 * 1024)
    path = directory / "transcript.jsonl"
    before = path.read_bytes()

    def full_disk(_fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(transcript_module.os, "fsync", full_disk)
    reopened = Transcript(directory)
    assert path.read_bytes() == before
    assert not path.with_suffix(".jsonl.fold").exists()
    assert len(_checkpoint_rows(reopened)) == 1
    assert [m.text for m in reopened.build_llm_history() if isinstance(m, Message)] == texts
    assert reopened.reclaimable_bytes() > 0
    reopened.close()


def test_superseded_scan_is_sound_against_lookalike_rows(tmp_path):
    """Only a real custom row of a collapsible type can be dropped.

    A message whose TEXT spells the checkpoint prefix is escaped in JSON and
    never matches; a checkpoint row in a shape the prefix does not recognise
    (payload key order) is a miss — kept, never dropped. The scan is allowed
    to be incomplete, never unsound.
    """
    from local_operator.session.transcript import _superseded_collapsible_lines

    lookalike_text = '{"custom_type":"frontend_state_checkpoint_v1"}'
    rows = [
        {
            "id": "a",
            "ts": 1.0,
            "type": "custom",
            "payload": {"custom_type": "frontend_state_checkpoint_v1", "details": 1},
        },
        {
            "id": "b",
            "ts": 2.0,
            "type": "message",
            "payload": {"role": "user", "content": [{"type": "text", "text": lookalike_text}]},
        },
        {
            "id": "c",
            "ts": 3.0,
            "type": "custom",
            "payload": {"details": 2, "custom_type": "frontend_state_checkpoint_v1"},
        },
        {
            "id": "d",
            "ts": 4.0,
            "type": "custom",
            "payload": {"custom_type": "hub_communication", "details": 3},
        },
        {
            "id": "e",
            "ts": 5.0,
            "type": "custom",
            "payload": {"custom_type": "hub_communication", "details": 4},
        },
        {
            "id": "f",
            "ts": 6.0,
            "type": "custom",
            "payload": {"custom_type": "frontend_state_checkpoint_v1", "details": 5},
        },
    ]
    path = tmp_path / "transcript.jsonl"
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    with path.open(encoding="utf-8", newline="") as handle:
        skip, dead = _superseded_collapsible_lines(handle)
    # Only row "a" is superseded-and-recognised: "c" is a miss (kept), "b" is a
    # message, "d"/"e" accumulate (not collapsible), "f" is the newest.
    assert skip == frozenset({0})
    assert dead == len(path.read_text().splitlines(keepends=True)[0].encode())


@pytest.mark.asyncio
async def test_fold_on_open_keeps_duplicate_id_rows_where_and_as_they_were(tmp_path):
    """Duplicate ids: the fold matches the loader exactly — every occurrence, in place.

    Real journals carry them (a stopped 428 MB session held four duplicate-id
    ``peer_message`` rows with different ``ts``). The loader keeps EVERY
    occurrence, in file order, and the id index simply holds the id. The fold
    on open must not pick a winner, move or re-stamp one: it drops only
    superseded collapsible rows, by POSITION (newest line of the type wins, as
    ``_superseded_custom_indices`` decides by index), and copies every other line
    verbatim — so a duplicate-id pair reads back byte-identical and in its slot,
    including a duplicate id on a collapsible row, where the later LINE wins.
    """
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    pad = "x" * (100 * 1024)
    rows = [
        {
            "id": "peer-1",
            "ts": 1.0,
            "type": "custom",
            "payload": {"custom_type": "peer_message", "details": {"v": "first"}},
        },
        {
            "id": "ck",
            "ts": 2.0,
            "type": "custom",
            "payload": {
                "custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                "details": {"n": 0, "p": pad},
            },
        },
        {
            "id": "peer-1",
            "ts": 3.0,
            "type": "custom",
            "payload": {"custom_type": "peer_message", "details": {"v": "second"}},
        },
        {
            "id": "ck",
            "ts": 4.0,
            "type": "custom",
            "payload": {
                "custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                "details": {"n": 1, "p": pad},
            },
        },
        {
            "id": "ck2",
            "ts": 5.0,
            "type": "custom",
            "payload": {
                "custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                "details": {"n": 2, "p": pad},
            },
        },
        {
            "id": "ck",
            "ts": 6.0,
            "type": "custom",
            "payload": {"custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE, "details": {"n": 3}},
        },
    ]
    directory = tmp_path / "sess"
    directory.mkdir()
    path = directory / "transcript.jsonl"
    lines = [json.dumps(row, separators=(",", ":")) + "\n" for row in rows]
    path.write_text("".join(lines))

    reopened = Transcript(directory)
    # Superseded checkpoints (lines 1, 3, 4) are gone; both peer-1 rows survive
    # byte-identical and in their original relative order.
    assert path.read_text() == lines[0] + lines[2] + lines[5]
    assert [(e.id, e.ts) for e in reopened.entries()] == [
        ("peer-1", 1.0),
        ("peer-1", 3.0),
        ("ck", 6.0),
    ]
    assert reopened.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE) == {"n": 3}
    reopened.close()


# ---------------------------------------------------------------------------
# PR #4513 round-1 remediation: each test below fails on 1b9211a3 (the
# reviewed head) and pins one ledger item.
# ---------------------------------------------------------------------------


def _journal_lines(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


@pytest.mark.asyncio
async def test_torn_newest_checkpoint_never_supersedes_the_last_valid_copy(tmp_path):
    """R1: a prefix-matching row that does not parse is not "the newest" of its type.

    A crash tears the newest checkpoint mid-row and the next append runs onto it,
    so the line still OPENS with the checkpoint prefix. Counting it as the newest
    made the fold delete the last valid checkpoint, which ``latest_custom`` then
    could not return — on disk or in memory.
    """
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    directory = tmp_path / "sess"
    texts, newest = await _journal_with_superseded_checkpoints(directory, copies=4, size=120 * 1024)
    path = directory / "transcript.jsonl"
    lines = _journal_lines(path)
    newest_line = lines[-1]
    assert '"n":3' in newest_line
    torn = newest_line[: len(newest_line) // 2]  # crash mid-row, no newline
    survivor = Message.user("after the crash")
    from local_operator.session.transcript import TranscriptEntry

    after = TranscriptEntry(
        id="after-crash",
        ts=99.0,
        type="message",
        payload=encode_message_payload(survivor),
    ).to_json()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(torn + after + "\n")

    reopened = Transcript(directory)
    assert reopened.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE) == newest
    assert any(
        '"n":3' in line and line == newest_line for line in _journal_lines(path)
    ), "the last VALID checkpoint must survive on disk"
    reopened.close()


@pytest.mark.asyncio
async def test_journal_replaced_between_scan_and_load_never_misapplies_skips(tmp_path, monkeypatch):
    """R2: the skip list indexes the handle it was computed from, never a successor.

    The scan and the load used to open the path separately, so a journal replaced
    in between had journal A's line numbers applied to journal B: a viewer
    silently lost B's rows at those positions. A read-only open is used so no
    fold muddies the picture.
    """
    from local_operator.session import transcript as transcript_module

    directory = tmp_path / "sess"
    texts, _ = await _journal_with_superseded_checkpoints(directory, copies=4, size=64)
    path = directory / "transcript.jsonl"

    other = tmp_path / "other"
    other_transcript = Transcript(other)
    for index in range(8):
        await other_transcript.append_message(Message.user(f"other {index}"))
    other_transcript.close()

    real = transcript_module._superseded_collapsible_lines

    def scan_then_replace(source):
        result = real(source)
        os.replace(other / "transcript.jsonl", path)
        return result

    monkeypatch.setattr(transcript_module, "_superseded_collapsible_lines", scan_then_replace)
    viewer = Transcript(directory, read_only=True)
    loaded = [m.text for m in viewer.build_llm_history() if isinstance(m, Message)]
    # Either whole journal is acceptable; a splice of B minus A's skips is not.
    assert loaded in (texts, [f"other {index}" for index in range(8)])


@pytest.mark.asyncio
async def test_second_in_process_open_never_folds_under_a_live_appender(tmp_path):
    """Q1: a refcounted share of the writer lock is not ownership of the file.

    The parent's peek of a running in-process subagent opens a second
    ``Transcript`` on the child's directory; the per-process refcount made it
    "hold" the lock, and it folded the journal under the child's appends
    (QA measured 2 and 7 of 20 rows lost). The second open must skip in memory
    and leave the file — same inode — alone, and no appended row may go missing.
    """
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    directory = tmp_path / "sess"
    writer = Transcript(directory)
    pad = "x" * (120 * 1024)
    for index in range(4):
        await writer.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"n": index, "p": pad}}
        )
    path = directory / "transcript.jsonl"
    inode = path.stat().st_ino

    appended = 200
    stop_error: list[BaseException] = []

    def append_loop() -> None:
        async def run() -> None:
            for index in range(appended):
                await writer.append_message(Message.user(f"live {index}"))

        try:
            asyncio.run(run())
        except BaseException as exc:  # pragma: no cover - surfaced below
            stop_error.append(exc)

    thread = threading.Thread(target=append_loop)
    thread.start()
    peeks = 0
    while thread.is_alive() or peeks == 0:
        peek = Transcript(directory)
        assert len(_checkpoint_rows(peek)) == 1
        peek.close()
        peeks += 1
    thread.join()
    assert not stop_error
    assert path.stat().st_ino == inode, "a shared hold must never rewrite the journal"
    on_disk = [line for line in _journal_lines(path) if '"live ' in line]
    assert len(on_disk) == appended
    writer.close()


@pytest.mark.asyncio
async def test_fold_on_open_leaves_compact_file_temp_alone(tmp_path):
    """Q1 (temp name): the fold must not share ``compact_file``'s temp path.

    Both used ``transcript.jsonl.compact``; a fold would truncate and rename a
    file a compaction was writing.
    """
    directory = tmp_path / "sess"
    await _journal_with_superseded_checkpoints(directory, copies=4, size=120 * 1024)
    path = directory / "transcript.jsonl"
    sentinel = path.with_suffix(".jsonl.compact")
    sentinel.write_text("compaction in flight\n")
    size_before = path.stat().st_size

    reopened = Transcript(directory)
    assert path.stat().st_size < size_before, "the fold must still run"
    assert sentinel.read_text() == "compaction in flight\n"
    assert sorted(p.name for p in directory.iterdir() if p.name.startswith("transcript.jsonl")) == [
        "transcript.jsonl",
        "transcript.jsonl.compact",
    ], "the fold's own temp file must be gone"
    reopened.close()


@pytest.mark.asyncio
async def test_fold_on_open_keeps_the_journals_timestamps(tmp_path):
    """R3: an idle boot's fold must not make the session look recently active."""
    directory = tmp_path / "sess"
    await _journal_with_superseded_checkpoints(directory, copies=4, size=120 * 1024)
    path = directory / "transcript.jsonl"
    old = 1_600_000_000_000_000_000
    os.utime(path, ns=(old, old + 7))
    size_before = path.stat().st_size

    reopened = Transcript(directory)
    stat = path.stat()
    assert stat.st_size < size_before, "the fold must still run"
    assert stat.st_mtime_ns == old + 7
    reopened.close()


@pytest.mark.asyncio
async def test_fold_on_open_fsyncs_the_directory_after_the_rename(tmp_path, monkeypatch):
    """R4: the rename is durable only once the directory entry is synced."""
    from local_operator.session import compression

    directory = tmp_path / "sess"
    await _journal_with_superseded_checkpoints(directory, copies=4, size=120 * 1024)
    synced: list[Path] = []
    monkeypatch.setattr(compression, "_fsync_dir", lambda p: synced.append(Path(p)))
    reopened = Transcript(directory)
    assert directory in synced
    reopened.close()


@pytest.mark.asyncio
async def test_fold_on_open_preserves_crlf_bytes(tmp_path):
    """Bot minor: "verbatim" means bytes — a CRLF journal must not be normalised."""
    directory = tmp_path / "sess"
    await _journal_with_superseded_checkpoints(directory, copies=4, size=120 * 1024)
    path = directory / "transcript.jsonl"
    crlf = [line.rstrip("\n") + "\r\n" for line in _journal_lines(path)]
    path.write_bytes("".join(crlf).encode())
    survivors = [line for line in crlf if '"pad"' not in line or '"n":3' in line]

    reopened = Transcript(directory)
    assert path.read_bytes() == "".join(survivors).encode()
    assert len(_checkpoint_rows(reopened)) == 1
    reopened.close()


# -- todo_snapshot / wake_schedules: full snapshots, newest-wins (incident 2026-11-24)
#
# Session 1375449bf925's 820 MB journal carried 4,425 ``todo_snapshot`` rows
# (403 MB) and 3,028 ``wake_schedules`` rows (27 MB), every one but the newest
# superseded, and all of them were parsed on every cold open. These tests pin
# the two types into the fold and prove each reader's answer is unchanged.

_TODO = "todo_snapshot"
_WAKES = "wake_schedules"


def test_the_collapsible_allowlist_names_the_todo_and_wake_snapshot_types():
    """Both literals are restated inline (leaf module); a rename upstream fails here."""
    from local_operator.harness.wake import WAKE_SCHEDULES_CUSTOM_TYPE
    from local_operator.session.session import TODO_SNAPSHOT_CUSTOM_TYPE
    from local_operator.session.transcript import _COLLAPSIBLE_CUSTOM_TYPES

    assert TODO_SNAPSHOT_CUSTOM_TYPE == _TODO
    assert WAKE_SCHEDULES_CUSTOM_TYPE == _WAKES
    assert {TODO_SNAPSHOT_CUSTOM_TYPE, WAKE_SCHEDULES_CUSTOM_TYPE} <= _COLLAPSIBLE_CUSTOM_TYPES


def _todo_details(index: int, pad: int) -> dict[str, Any]:
    return {
        "items": [
            {"phase": f"p{index}", "items": [{"text": f"task {index}", "done": False}]},
            {"phase": "pad", "items": [{"text": "x" * pad, "done": True}]},
        ]
    }


def _wake_details(index: int, pad: int) -> dict[str, Any]:
    return {"schedules": [{"id": f"w{index}", "message": "y" * pad, "due_ms": 1000 + index}]}


async def _journal_with_snapshots(directory: Path, *, copies: int, pad: int) -> list[str]:
    """Messages interleaved with ``copies`` todo and wake snapshots; returns texts."""
    transcript = Transcript(directory)
    for index in range(copies):
        await transcript.append_message(Message.user(f"turn {index}"))
        await transcript.append_custom(_TODO, _todo_details(index, pad))
        await transcript.append_custom(_WAKES, _wake_details(index, pad))
    texts = [m.text for m in transcript.build_llm_history() if isinstance(m, Message)]
    transcript.close()
    return texts


def _rows_of(transcript: Transcript, custom_type: str) -> list[Any]:
    return [e for e in transcript.entries() if e.payload.get("custom_type") == custom_type]


def _reader_answers(directory: Path, transcript: Transcript) -> dict[str, Any]:
    """Every newest-wins entry point the audited readers use, for both types."""
    from local_operator.network.carry import latest_custom_details
    from local_operator.session.transcript import (
        read_latest_custom,
        read_latest_custom_entry,
    )

    answers: dict[str, Any] = {}
    for custom_type in (_TODO, _WAKES):
        entry = transcript.latest_custom_entry(custom_type)
        file_entry = read_latest_custom_entry(directory, custom_type)
        answers[custom_type] = (
            transcript.latest_custom(custom_type),
            (entry.id, entry.ts, entry.payload) if entry else None,
            read_latest_custom(directory, custom_type),
            (file_entry.id, file_entry.ts, file_entry.payload) if file_entry else None,
            latest_custom_details(directory, custom_type),
        )
    return answers


@pytest.mark.asyncio
async def test_owning_open_folds_todo_and_wake_snapshots_to_the_newest(tmp_path):
    """N snapshots of each type fold to ONE each on open; every reader answers the same."""
    directory = tmp_path / "sess"
    copies = 6
    texts = await _journal_with_snapshots(directory, copies=copies, pad=40 * 1024)
    path = directory / "transcript.jsonl"
    size_before = path.stat().st_size

    # Negative control: before the open folds, the journal really holds N of each.
    raw = _journal_lines(path)
    assert sum(f'"custom_type":"{_TODO}"' in line for line in raw) == copies
    assert sum(f'"custom_type":"{_WAKES}"' in line for line in raw) == copies
    before = _reader_answers(directory, Transcript(directory, read_only=True))

    reopened = Transcript(directory)
    assert path.stat().st_size < size_before, "an owning open above threshold must fold"
    assert len(_rows_of(reopened, _TODO)) == 1
    assert len(_rows_of(reopened, _WAKES)) == 1
    assert reopened.latest_custom(_TODO) == _todo_details(copies - 1, 40 * 1024)
    assert reopened.latest_custom(_WAKES) == _wake_details(copies - 1, 40 * 1024)
    assert _reader_answers(directory, reopened) == before
    assert [m.text for m in reopened.build_llm_history() if isinstance(m, Message)] == texts
    assert reopened.reclaimable_bytes() == 0
    reopened.close()


@pytest.mark.asyncio
async def test_compact_file_folds_todo_and_wake_snapshots_to_the_newest(tmp_path):
    """Below the open-time threshold ``compact_file`` drops them, priced exactly."""
    directory = tmp_path / "sess"
    copies = 5
    texts = await _journal_with_snapshots(directory, copies=copies, pad=16)
    path = directory / "transcript.jsonl"
    size_before = path.stat().st_size

    transcript = Transcript(directory)
    assert path.stat().st_size == size_before, "below the threshold the open must not rewrite"
    before = _reader_answers(directory, transcript)
    expected = transcript.reclaimable_bytes()
    assert expected > 0
    assert await transcript.compact_file(min_reclaim_bytes=0) == expected
    assert path.stat().st_size == size_before - expected
    assert len(_rows_of(transcript, _TODO)) == 1
    assert len(_rows_of(transcript, _WAKES)) == 1
    assert _reader_answers(directory, transcript) == before
    transcript.close()

    reopened = Transcript(directory)
    assert [e.payload.get("custom_type") for e in reopened.entries() if e.type == "custom"] == [
        _TODO,
        _WAKES,
    ]
    assert _reader_answers(directory, reopened) == before
    assert [m.text for m in reopened.build_llm_history() if isinstance(m, Message)] == texts
    reopened.close()


@pytest.mark.asyncio
async def test_todos_and_wakes_restore_identically_after_the_fold(tmp_path):
    """The Session loaders' inputs survive the fold: todo restore and wake re-arm.

    ``Session._load_todo_snapshot`` feeds ``latest_custom(todo_snapshot)["items"]``
    to ``restore_todos``; the wake restore re-arms from ``latest_custom_entry``
    and gates on that entry's ``ts`` against the fork instant — so the entry's
    ts must be the newest row's, unchanged.
    """
    from local_operator.harness.wake_types import WakeSchedule
    from local_operator.tools.builtin import TODO_STORE, restore_todos, todo_snapshot

    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    newest_schedule = WakeSchedule(id="w-new", message="check CI", next_due_at=1_900_000_000_000)
    for index in range(4):
        await transcript.append_custom(
            _TODO, {"items": [{"name": "Work", "items": [{"text": f"t{index}", "done": False}]}]}
        )
    # The older wake row holds a schedule the newest no longer contains: a fold
    # that kept the wrong row would resurrect a cancelled wake.
    cancelled = WakeSchedule(id="w-cancelled", message="old", next_due_at=1_800_000_000_000)
    await transcript.append_custom(_WAKES, {"schedules": [cancelled.model_dump()]})
    await transcript.append_custom(_WAKES, {"schedules": [newest_schedule.model_dump()]})
    entry_before = transcript.latest_custom_entry(_WAKES)
    todo_before = transcript.latest_custom(_TODO)
    assert await transcript.compact_file(min_reclaim_bytes=0) > 0
    transcript.close()

    reopened = Transcript(directory)
    entry_after = reopened.latest_custom_entry(_WAKES)
    # Both rows must survive the fold, so narrow before reading their fields.
    assert entry_before is not None
    assert entry_after is not None
    assert (entry_after.id, entry_after.ts) == (entry_before.id, entry_before.ts)
    schedules = entry_after.payload["details"]["schedules"]
    assert [WakeSchedule.model_validate(row) for row in schedules] == [newest_schedule]
    assert len(_rows_of(reopened, _WAKES)) == 1
    assert len(_rows_of(reopened, _TODO)) == 1

    assert reopened.latest_custom(_TODO) == todo_before
    todo_after = reopened.latest_custom(_TODO)
    assert todo_after is not None
    probe = "fold-restore-probe"
    TODO_STORE.pop(probe, None)
    try:
        restore_todos(probe, list(todo_after["items"]))
        restored = todo_snapshot(probe)
    finally:
        TODO_STORE.pop(probe, None)
    assert [item["text"] for phase in restored for item in phase["items"]] == ["t3"]
    reopened.close()


@pytest.mark.asyncio
async def test_torn_newest_todo_and_wake_rows_never_supersede_the_last_valid_copy(tmp_path):
    """R1 holds for the new members: a torn newest row keeps the last VALID one."""
    directory = tmp_path / "sess"
    copies = 6
    await _journal_with_snapshots(directory, copies=copies, pad=40 * 1024)
    path = directory / "transcript.jsonl"
    lines = _journal_lines(path)
    valid_todo, valid_wake = lines[-2], lines[-1]
    assert f'"custom_type":"{_TODO}"' in valid_todo and f'"custom_type":"{_WAKES}"' in valid_wake
    from local_operator.session.transcript import TranscriptEntry

    def torn(custom_type: str, details: dict[str, Any]) -> str:
        row = TranscriptEntry(
            id=f"torn-{custom_type}",
            ts=98.0,
            type="custom",
            payload={"custom_type": custom_type, "details": details},
        ).to_json()
        return row[: len(row) // 2]  # crash mid-row, no newline

    after = TranscriptEntry(
        id="after-crash",
        ts=99.0,
        type="message",
        payload=encode_message_payload(Message.user("after the crash")),
    ).to_json()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(torn(_TODO, _todo_details(99, 40 * 1024)) + after + "\n")
        handle.write(
            torn(_WAKES, _wake_details(99, 40 * 1024))
            + after.replace("after-crash", "after-crash-2")
            + "\n"
        )

    reopened = Transcript(directory)
    assert reopened.latest_custom(_TODO) == _todo_details(copies - 1, 40 * 1024)
    assert reopened.latest_custom(_WAKES) == _wake_details(copies - 1, 40 * 1024)
    on_disk = _journal_lines(path)
    assert valid_todo in on_disk and valid_wake in on_disk, "the last VALID rows must survive"
    reopened.close()
