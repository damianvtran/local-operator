"""Footprint behaviour of the transcript: slim rows, the prune journal, and
the file compaction that folds it in.

Every test here defends an invariant the size optimisation is allowed to
break in exactly zero ways: what comes back out of ``build_llm_history`` must
still be the conversation, and ``first_kept_entry_id`` must still resolve.
"""

from __future__ import annotations

import gc
import json
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
