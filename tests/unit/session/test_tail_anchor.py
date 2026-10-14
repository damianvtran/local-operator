"""A sidecar that proves a journal has no checkpoint row, so the read is bounded.

WHY THIS FILE EXISTS. ``read_replay_suffix`` stops at the newest compaction once
every requested custom type has been met. On a journal with NO
``frontend_state_checkpoint_v1`` row at all, the type is never met, so the walk
runs to BOF to prove the absence: 570-650 ms of CPU on a 35 MB checkpointless
journal and 1.0-1.9 s on a 118 MB one, once per fresh process (the replay cache
makes the repeats free and cannot make the first read cheap).

``session.tail_anchor`` records that fact — the ABSENCE, for one journal version —
from the pre-warm job, because the reader itself is not allowed to write. The tests
below prove the two halves that make it safe: the shortcut returns EXACTLY what the
full walk returned, and every way the record could go stale is refused.
"""

from __future__ import annotations

import json
import os

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.tail_anchor import (
    ANCHOR_FILENAME,
    ANCHOR_VERSION,
    TailAnchor,
    build_anchor,
    proves_absent,
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
    assert anchor is not None, "a checkpointless journal must get a record"
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
    assert (
        validate_anchor(
            tmp_path,
            TailAnchor(inode=anchor.inode + 1, size=anchor.size, tail_digest=anchor.tail_digest),
        )
        is None
    )

    # A shrink: the record's size is above the file now.
    fresh = build_anchor(tmp_path)
    assert fresh is not None
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    assert validate_anchor(tmp_path, fresh) is None


@pytest.mark.asyncio
def test_a_missing_journal_or_a_malformed_sidecar_reads_as_no_hint(tmp_path):
    """Everything the reader cannot prove it falls back from, quietly."""
    assert read_anchor(tmp_path) is None
    assert build_anchor(tmp_path) is None
    (tmp_path / ANCHOR_FILENAME).write_text("{not json")
    assert read_anchor(tmp_path) is None
    (tmp_path / ANCHOR_FILENAME).write_text(
        json.dumps({"version": 99, "inode": 1, "size": 1, "tail_digest": "0" * 16})
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
    assert written is not None

    # Idempotent: the second pass costs a stat, not a scan.
    assert write_tail_anchor(root, session_id) is False


@pytest.mark.asyncio
async def test_a_torn_trailing_row_is_not_recorded_as_absent(tmp_path):
    """F1: the record must name the END OF THE LAST COMPLETE ROW.

    A journal is appended to while it is scanned. If the record named the file's
    size, a half-written checkpoint row below that offset would sit inside the
    "nothing happened here" region — and when the writer finished the row, the
    validator's suffix scan (everything from the recorded size on) would read only
    the row's TAIL, miss the marker that was already there, and tell the reader a
    checkpoint row does not exist. Recording the end of the last complete row
    instead pushes the torn bytes into the validated suffix.
    """
    await journal_without_a_checkpoint(tmp_path, messages=4)
    path = tmp_path / "transcript.jsonl"
    complete = path.stat().st_size

    torn = (
        b'{"id":"t1","ts":1,"type":"custom","payload":{'
        b'"custom_type":"frontend_state_checkpoint_v1"'
    )
    with path.open("ab") as handle:
        handle.write(torn)

    anchor = build_anchor(tmp_path)
    assert anchor is not None
    assert anchor.size == complete, (
        "the record named bytes that are only part of a row: a row completing below "
        "it could then hide its marker from the suffix scan"
    )
    write_anchor(tmp_path, anchor)

    # The writer finishes the row. Its marker is entirely BELOW the old file size,
    # so only a record that named the last COMPLETE row catches it.
    with path.open("ab") as handle:
        handle.write(b',"details":{"state":{"session_id":"s","sequence":1}}}}\n')

    assert (
        validate_anchor(tmp_path, read_anchor(tmp_path)) is None
    ), "a completed checkpoint row was accepted as absent"
    suffix = read_replay_suffix(tmp_path, checkpoint_types=FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert suffix.checkpoint is not None, "the reader must find the row it was told was absent"


@pytest.mark.asyncio
async def test_a_marker_straddling_a_scan_window_is_found(tmp_path):
    """F2: the suffix scan's windows overlap by a needle's length.

    The scan is the only thing between a stale record and a reader that skips a
    real checkpoint row. Fixed, non-overlapping windows cannot see a marker split
    across a boundary — the fixture below places the needle exactly three bytes
    across one — so the windows overlap and the scan starts a needle's length below
    the recorded size.
    """
    await journal_without_a_checkpoint(tmp_path, messages=4)
    path = tmp_path / "transcript.jsonl"
    anchor = build_anchor(tmp_path)
    assert anchor is not None
    write_anchor(tmp_path, anchor)

    needle = b'"custom_type":"frontend_state_checkpoint_v1"'
    prefix = b'{"id":"straddle","ts":1,"type":"custom","payload":{'
    # A detail blob sized so a 1 MiB window boundary falls INSIDE the needle — the
    # split two adjacent windows cannot cover between them. The geometry is
    # ASSERTED rather than described (review round 2, N2): the split point depends
    # on the prefix's length, so only a measurement says where it lands, and the
    # earlier comment's "three bytes" was arithmetic about the wrong offset.
    straddle = 3
    tail_len = (1 << 20) + straddle - len(needle)
    row = prefix + needle + b',"details":{"state":{"pad":"' + b"p" * tail_len + b'"}}}\n'
    before = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(row)
    needle_start = before + len(prefix)
    needle_end = needle_start + len(needle)
    boundary = path.stat().st_size - (1 << 20)
    assert needle_start < boundary < needle_end, (
        f"the fixture no longer straddles a window boundary: needle "
        f"[{needle_start}, {needle_end}) vs boundary {boundary}"
    )

    assert (
        validate_anchor(tmp_path, read_anchor(tmp_path)) is None
    ), "a marker split across the scan's window boundary was reported absent"


@pytest.mark.asyncio
async def test_a_pre_remediation_record_is_ignored_rather_than_reinterpreted(tmp_path):
    """M1: a record from this branch's earlier revision must not become a proof.

    That revision also recorded an offset for journals that DO hold the checkpoint
    row, and ``read_anchor`` ignores fields it does not know — so such a record would
    read as the proof of ABSENCE version 2 records: for a journal that has the row,
    the exact opposite of what the file says.

    MEASURED, two independent refusals hold it back, and this test pins both so
    neither can be dropped silently: the digest this version requires (the old
    record has none — with ``ANCHOR_VERSION`` reverted to 1 the record is STILL
    refused, for that reason) and the version field, which moved with it.
    """
    await journal_without_a_checkpoint(tmp_path, messages=4)
    path = tmp_path / "transcript.jsonl"
    (tmp_path / ANCHOR_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "custom_type": "frontend_state_checkpoint_v1",
                "inode": path.stat().st_ino,
                "size": path.stat().st_size,
                "checkpoint_offset": 12,
                "row_id": "t1",
            }
        )
    )

    assert ANCHOR_VERSION == 2, "the version moved; a v1 record must not be reinterpreted"
    assert read_anchor(tmp_path) is None
    assert proves_absent(tmp_path) is False


@pytest.mark.asyncio
async def test_a_boundary_that_moved_under_the_record_is_refused(tmp_path):
    """M2: a truncate-and-regrow of the same inode must not keep the proof alive.

    The write path's rollback removes bytes with ``os.truncate``, and the file can
    then be appended past the recorded size again — after which the not-shrunk guard
    sees a long file and the suffix scan sees only bytes the record never claimed.
    The digest of the row ending at the recorded size is what closes it.
    """
    await journal_without_a_checkpoint(tmp_path, messages=4)
    path = tmp_path / "transcript.jsonl"
    anchor = build_anchor(tmp_path)
    assert anchor is not None
    write_anchor(tmp_path, anchor)

    os.truncate(path, anchor.size // 2)
    # Regrow PAST the recorded size, which is what makes the not-shrunk guard blind:
    # the padding is sized from the record rather than fixed, so the fixture keeps
    # its shape whatever the journal's length is.
    padding = max(4096, anchor.size - path.stat().st_size + 4096)
    with path.open("ab") as handle:
        handle.write(b'{"id":"new","ts":1,"type":"message","payload":{"kind":"message"}}\n')
        handle.write(
            b'{"id":"pad","ts":1,"type":"message","payload":{"pad":"' + b"q" * padding + b'"}}\n'
        )
    assert path.stat().st_size > anchor.size, "the fixture must regrow past the record"

    assert (
        validate_anchor(tmp_path, read_anchor(tmp_path)) is None
    ), "a record whose boundary moved under it was still treated as proof"
    assert proves_absent(tmp_path) is False
