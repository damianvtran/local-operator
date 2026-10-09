"""A sidecar that tells a cold reader where the newest checkpoint row is.

WHY THIS FILE EXISTS. ``read_replay_suffix`` stops at the newest compaction once
every requested custom type has been met. On a journal with NO
``frontend_state_checkpoint_v1`` row at all, the type is never met, so the walk
runs to BOF to prove the absence: 570-650 ms of CPU on a 35 MB checkpointless
journal and 1.0-1.9 s on a 118 MB one, once per fresh process (the replay cache
makes the repeats free and cannot make the first read cheap).

``session.tail_anchor`` records that fact — the offset of the newest checkpoint
row, or the proof that there is none — from the pre-warm job, because the reader
itself is not allowed to write. The tests below prove the two halves that make it
safe: the shortcut returns EXACTLY what the full walk returned, and every way the
record could go stale is refused.
"""

from __future__ import annotations

import json
import os

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.tail_anchor import (
    ANCHOR_FILENAME,
    build_anchor,
    read_anchor,
    validate_anchor,
    write_anchor,
)
from local_operator.session.transcript import (
    Transcript,
    read_replay_suffix,
    replay_entries,
)


async def journal_without_a_checkpoint(directory, *, messages: int = 400) -> None:
    """A real journal: a compaction near the tail, no checkpoint row.

    Written through ``Transcript``'s own appenders rather than by hand, so the
    bytes on disk are the bytes a runtime writes — which is what the anchor's
    needle scan assumes.

    The default size is deliberate: the reader walks 1 MiB chunks, so a fixture
    that fits in ONE chunk reaches BOF on its first iteration and cannot tell a
    bounded walk from an unbounded one. 400 messages of 8 KiB are ~3.2 MB, and the
    compaction keeps the ten newest rows — the shape a real tail-anchored journal
    has, where the boundary is inside the first chunk and everything below it is
    what the checkpoint requirement used to force the walk through.
    """
    transcript = Transcript(directory, defer_materialise=False)
    for index in range(messages):
        await transcript.append_message(
            Message(
                role="user" if index % 2 == 0 else "assistant",
                content=[TextContent(text=f"turn {index} " + "x" * 8192)],
            )
        )
    entries = transcript.entries()
    await transcript.append_compaction(
        summary="earlier turns",
        first_kept_entry_id=entries[-min(10, len(entries))].id,
        tokens_before=1000,
    )
    for index in range(messages):
        await transcript.append_message(
            Message(role="user", content=[TextContent(text=f"after {index} " + "y" * 4096)])
        )


@pytest.mark.asyncio
async def test_the_shortcut_returns_exactly_what_the_full_walk_returned(tmp_path):
    """The differential that makes the hint safe: same answer, fewer bytes."""
    await journal_without_a_checkpoint(tmp_path)

    before = read_replay_suffix(tmp_path, checkpoint_types=FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert before.bytes_read == (tmp_path / "transcript.jsonl").stat().st_size, (
        "without the sidecar the reader must walk the whole journal — this is the "
        "cost the anchor exists to remove"
    )

    anchor = build_anchor(tmp_path)
    assert anchor is not None and anchor.checkpoint_offset is None
    assert write_anchor(tmp_path, anchor) is True

    after = read_replay_suffix(tmp_path, checkpoint_types=FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    # The REPLAY of both reads must be identical: the walks stop at different
    # offsets, so the raw row lists differ by the older rows ``replay_entries``
    # discards below the compaction — what a caller ever sees is this.
    assert replay_entries(after.entries, None) == replay_entries(before.entries, None)
    assert after.checkpoint == before.checkpoint is None
    assert after.checkpoint_order == before.checkpoint_order
    assert after.through_present == before.through_present
    assert (
        after.bytes_read < before.bytes_read // 2
    ), "the walk must stop at the journal's own compaction boundary instead of BOF"


@pytest.mark.asyncio
async def test_a_newer_checkpoint_row_is_found_and_the_hint_dropped(tmp_path):
    """The proof is about the journal as it was; an append invalidates it.

    The validation scans the bytes appended since the record for the checkpoint
    type's own spelling, so the reader goes back to walking and FINDS the row
    rather than trusting a record that predates it.
    """
    await journal_without_a_checkpoint(tmp_path)
    first = build_anchor(tmp_path)
    assert first is not None
    write_anchor(tmp_path, first)

    transcript = Transcript(tmp_path, defer_materialise=False)
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"session_id": "sess", "sequence": 1}}
    )

    assert validate_anchor(tmp_path, read_anchor(tmp_path)) is None
    suffix = read_replay_suffix(tmp_path, checkpoint_types=FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert suffix.checkpoint == {"state": {"session_id": "sess", "sequence": 1}}


@pytest.mark.asyncio
async def test_a_replaced_or_truncated_journal_is_never_described_by_an_old_record(tmp_path):
    """Inode and size are the two ways the record can name bytes that moved."""
    await journal_without_a_checkpoint(tmp_path, messages=4)
    anchor = build_anchor(tmp_path)
    assert anchor is not None
    assert write_anchor(tmp_path, anchor) is True

    # A replaced file (the shape ``compact_file`` leaves): the record names the
    # inode it scanned, and a forged record proves the check is what refuses it
    # rather than the filesystem having handed out a different number.
    path = tmp_path / "transcript.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    forged_inode = type(anchor)(
        inode=anchor.inode + 1,
        size=anchor.size,
        checkpoint_offset=anchor.checkpoint_offset,
        row_id=anchor.row_id,
    )
    assert validate_anchor(tmp_path, forged_inode) is None

    # A shrink: the record's size is above the file now.
    fresh = build_anchor(tmp_path)
    assert fresh is not None
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    assert validate_anchor(tmp_path, fresh) is None


@pytest.mark.asyncio
async def test_a_checkpoint_row_at_the_head_of_a_journal_is_recorded_by_offset(tmp_path):
    """The recorded-offset half, and the row it names is re-read before use."""
    transcript = Transcript(tmp_path, defer_materialise=False)
    await transcript.append_message(Message(role="user", content=[TextContent(text="hi")]))
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"session_id": "sess", "sequence": 7}}
    )
    await transcript.append_message(Message(role="user", content=[TextContent(text="and more")]))

    anchor = build_anchor(tmp_path)
    assert anchor is not None and anchor.checkpoint_offset is not None
    assert validate_anchor(tmp_path, anchor) is not None

    suffix = read_replay_suffix(tmp_path, checkpoint_types=FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert suffix.checkpoint == {"state": {"session_id": "sess", "sequence": 7}}

    # A record whose offset no longer parses as the row it names is refused: the
    # journal is otherwise untouched, so only the row read can catch this.
    forged = type(anchor)(
        inode=anchor.inode,
        size=anchor.size,
        checkpoint_offset=0,
        row_id=anchor.row_id,
    )
    assert validate_anchor(tmp_path, forged) is None


def test_a_missing_journal_or_a_malformed_sidecar_reads_as_no_hint(tmp_path):
    """Everything the reader cannot prove it falls back from, quietly."""
    assert read_anchor(tmp_path) is None
    assert build_anchor(tmp_path) is None
    (tmp_path / ANCHOR_FILENAME).write_text("{not json")
    assert read_anchor(tmp_path) is None
    (tmp_path / ANCHOR_FILENAME).write_text(
        json.dumps({"version": 99, "inode": 1, "size": 1, "checkpoint_offset": None})
    )
    assert read_anchor(tmp_path) is None


@pytest.mark.asyncio
async def test_the_prewarm_writes_the_anchor_once_per_journal_version(tmp_path):
    """The writer half: the pre-warm queue is what records these journals.

    Called directly rather than through ``warm_index_cache`` because the guards
    that decide WHICH journals are warmed belong to that module's own tests; what
    this pins is that the write happens, describes the journal, and does not
    repeat while the record still holds.
    """
    from local_operator.session.index_prewarm import write_tail_anchor

    root = tmp_path / "config"
    session_id = "sess-anchor"
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True)
    await journal_without_a_checkpoint(directory, messages=4)

    assert write_tail_anchor(root, session_id) is True
    written = read_anchor(directory)
    assert written is not None and written.checkpoint_offset is None

    # Idempotent: the second pass costs a stat, not a scan.
    assert write_tail_anchor(root, session_id) is False
