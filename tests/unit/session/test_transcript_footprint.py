"""Footprint behaviour of the transcript: slim rows, the prune journal, and
the file compaction that folds it in.

Every test here defends an invariant the size optimisation is allowed to
break in exactly zero ways: what comes back out of ``build_llm_history`` must
still be the conversation, and ``first_kept_entry_id`` must still resolve.
"""

from __future__ import annotations

import json
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
