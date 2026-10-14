"""A bounded, explicitly non-canonical journal preview for terminal navigation.

READ BACKWARD BY ROWS, never by a fixed byte window, and that is the whole shape
of this reader. It used to take the last ``PREVIEW_BYTES`` (256 KiB) of the file
and refuse the whole preview whenever that window cut a row, missed a
compaction's ``first_kept_entry_id``, or ended on an append in progress — which
on the operator's real store is not an edge case but the common one: checkpoint
rows reach 0.9 MB and compaction summaries average 568 KB, so a 256 KiB tail
sits *inside* a single row on exactly the sessions the sidebar most needs to
paint. Three of the 40 largest real journals painted a blank pane there (audit
F9). The walk now uses the shared backward line iterator
(:func:`~local_operator.session.transcript._iter_complete_lines_backward`), the
same byte handling the display page and the replay suffix use, so an oversized
row costs one pass and is then simply not a row this reader displays.

WHAT BOUNDS THE WALK, and why it is two conditions rather than one. The walk
stops once it holds ``DISPLAY_HISTORY_MESSAGES`` display rows AND the suffix it
holds is replayable — meaning the newest ``compaction`` row's
``first_kept_entry_id`` is in hand, or there is no such row. The second
condition is :func:`read_replay_suffix`'s own stop rule and it is not
decoration: :func:`~local_operator.session.transcript.replay_entries` cannot
resolve a boundary it was not handed, and its documented answer for that input
is to log an error and replay everything it does have. Stopping inside an
unresolved boundary would therefore pay a spurious error log on every sidebar
open and leave the preview's notion of "what the model still sees" resting on a
fallback. With the boundary in hand the replay applies the journal's own
compaction semantics — including dropping rows before ``first_kept`` — through
the one shared implementation, so this reader cannot drift from the model's.

WHY A SUFFIX IS ENOUGH, the property the old byte window was reaching for:

- **A prune can never be missed.** Prunes are appended AFTER their targets
  (``Transcript.append_prune``), so every prune affecting a row this reader
  holds was appended after that row and is therefore inside the walk. Bytes
  below the window can only retract rows below the window. This is the argument
  :func:`read_replay_suffix` makes for itself, and it is why this preview is
  non-canonical but never stale.
- **An unresolved boundary is walked past, not crossed.** While the newest
  compaction's ``first_kept_entry_id`` has not been met, the walk continues (up
  to ``PREVIEW_SCAN_BYTES``), so the replay never has to guess. Reaching the
  file's start with the id still absent is the honestly-unsatisfiable case — a
  compaction naming a row no longer on disk — and there the shared replay's own
  fallback (replay everything, log the error) is what any other reader would do
  with the same journal, so the preview neither invents nor hides anything.
- **A torn final line is one row that does not parse.** A live append caught
  mid-write is the normal state of a file a running session writes; the walker
  hands the fragment over and ``TranscriptEntry.from_json`` drops it exactly as
  it drops any malformed row. Refusing the whole preview for it — the previous
  behaviour — is the blank-pane failure one append wide. The dropped fragment is
  NOT durable: ``_write_entries`` writes whole lines under fsync and truncates
  on failure, so nothing it might have said is missing from the state on disk.
  It is reported through ``partial``, because a preview that is a few rows
  shorter than the journal is an excerpt and should say so.
- **A non-empty journal does not come back empty.** The walk steps over
  bookkeeping (checkpoints, spend records, compactions, prunes) and keeps going
  until it has display rows, reaches the file's start, or hits the scan
  ceiling; hitting the ceiling returns what it found with ``partial`` set,
  rather than an empty list for a conversation that demonstrably has content.

WHAT ``partial`` MEANS, stated once because three conditions set it. It is "this
is an excerpt", never "this failed": true when the walk stopped before the
file's start (the display budget, the scan ceiling), when the journal's newest
row was a torn append, or when the replay produced more rows than the ceiling
shows. A journal holding no display rows at all — one whose every row is
bookkeeping — answers ``[]`` with ``partial`` False, which is the honest "nothing
to preview" the caller renders as an empty conversation rather than as an error.

Externalized attachments ARE hydrated, through the store that owned the journal
(``store_for_transcript_dir``) rather than the reader's environment, because a
preview that paints "image unavailable" for a picture the session actually has
is the same false receipt this reader exists to avoid. The cost is bounded and
paid off the loop: each image reference costs one content read, one tiny
sidecar read, and a full-digest re-hash (the store validates the file against
its name) — all proportional to that image, with no decoding here: the widget
decodes. What stays true from before: this reader never starts an owner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.session.attachments import store_for_transcript_dir
from local_operator.session.history_window import DISPLAY_HISTORY_MESSAGES
from local_operator.session.transcript import (
    ENTRY_COMPACTION,
    ENTRY_MESSAGE,
    TranscriptEntry,
    _iter_complete_lines_backward,
    replay_entries,
)

#: Ceiling on how far back from EOF one preview may walk. The walk is bounded by
#: the ROW budget it needs, not by this figure; the figure exists because the
#: budget alone is unbounded work on a file a writer can grow while we read, and
#: "the newest rows are 2 GB of checkpoints" is a shape a corrupt journal can
#: reach. Sized from the measured store rather than guessed: on the 40 largest
#: real journals the suffix a replay needs was 1.0-17.8 MB (median 2.1 MB,
#: measured as ``read_replay_suffix().bytes_read`` over those files), so 32 MiB
#: clears the worst real case with room to spare and still bounds the walk.
PREVIEW_SCAN_BYTES = 32 * 1024 * 1024

#: Rows the walk may parse, whichever comes first with the byte ceiling.
#:
#: The byte ceiling alone is not a bound on the WORK: a journal whose tail is a
#: long run of bookkeeping rows costs one JSON decode per row, and review round 1
#: (F6) measured 37-106 ms on a 132.5 MB journal against 1.2-2.1 ms before this
#: reader existed. 40,000 rows is far above the newest 120 display rows of every
#: journal measured here (the 40 largest real journals reach them within a few
#: hundred rows) and far below the tens of millions a 132 MB journal can hold, so
#: it binds only on the pathological shape — and it binds HONESTLY: the preview
#: comes back as the excerpt it is, with ``partial`` set, rather than paying an
#: unbounded decode loop on the sidebar's first-paint path.
PREVIEW_SCAN_ROWS = 40_000


@dataclass(frozen=True)
class SavedPreview:
    messages: list[Any]
    partial: bool
    cwd: str = ""


def read_saved_preview(directory: Path) -> SavedPreview:
    path = directory / "transcript.jsonl"
    # A missing journal is not evidence of an empty conversation. The caller
    # may separately prove an unmaterialised live owner exists; this reader
    # never invents that authority from an absent path.
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end_of_file = handle.tell()
        if end_of_file == 0:
            # An existing but empty journal is a valid empty conversation.
            return SavedPreview([], False)
        # Whether the newest row is still being written. Read from the byte
        # test the writer's own contract implies (whole lines, then fsync), not
        # by parsing: a row that ends in a newline was written in full.
        torn_tail = not _ends_with_newline(handle, end_of_file)
        rows: list[TranscriptEntry] = []  # newest-first, as walked
        seen_ids: set[str] = set()
        display_rows = 0
        # The newest compaction seen walking backward — the one the replay
        # treats as the boundary (see ``replay_entries``).
        first_kept: str | None = None
        boundary_seen = False
        at_start = False
        bytes_walked = 0
        rows_parsed = 0
        for chunk_start, lines in _iter_complete_lines_backward(handle, end_of_file):
            # Priced from the chunk last consumed, exactly as
            # ``read_replay_suffix`` prices its own read: what matters is the
            # bytes this call actually looked at.
            bytes_walked = end_of_file - chunk_start
            for raw in lines:
                if not raw.strip():
                    continue
                rows_parsed += 1
                entry = TranscriptEntry.from_json(raw.decode("utf-8", errors="replace"))
                if entry is None:
                    # Malformed rows are dropped individually, as every other
                    # reader does — including the torn fragment of a live
                    # append, which must not blank a pane.
                    continue
                rows.append(entry)
                seen_ids.add(entry.id)
                if entry.type == ENTRY_MESSAGE:
                    display_rows += 1
                elif entry.type == ENTRY_COMPACTION and not boundary_seen:
                    # First hit walking backward is the NEWEST marker: the one
                    # ``replay_entries`` cuts at.
                    boundary_seen = True
                    first_kept = entry.payload.get("first_kept_entry_id") or None
            at_start = chunk_start == 0
            replayable = not boundary_seen or first_kept is None or first_kept in seen_ids
            if at_start or (display_rows >= DISPLAY_HISTORY_MESSAGES and replayable):
                break
            if bytes_walked >= PREVIEW_SCAN_BYTES or rows_parsed >= PREVIEW_SCAN_ROWS:
                break
    if not rows:
        # A journal whose every row is unparseable, or a scan that hit the
        # ceiling on bookkeeping alone. Both are honest empties, and
        # ``partial`` distinguishes them from a conversation with no content.
        return SavedPreview([], torn_tail or not at_start)
    # Oldest-first for the replay: ``replay_entries`` and its prune map both
    # depend on append order, and the walk hands them newest-first.
    rows.reverse()
    cwd = next(
        (str(entry.payload.get("cwd", "")) for entry in rows if entry.type == "session"),
        "",
    )
    messages = replay_entries(rows, store_for_transcript_dir(directory))
    # The row ceiling bounds replay bookkeeping and Textual's preparation of
    # many tiny messages. Apply prunes/compaction BEFORE slicing, never
    # truncate away their instructions.
    return SavedPreview(
        messages[-DISPLAY_HISTORY_MESSAGES:],
        torn_tail or not at_start or len(messages) > DISPLAY_HISTORY_MESSAGES,
        cwd,
    )


def _ends_with_newline(handle: Any, end_of_file: int) -> bool:
    """Whether the journal's last byte terminates a row.

    One byte read, off the same handle the walk uses, so the answer is about the
    file this call read rather than a second stat of one a writer may have
    grown since. ``b""`` (an unreadable last byte) counts as terminated: the
    conservative direction is to claim nothing was cut, and the walk itself will
    drop the row if it does not parse.
    """
    if end_of_file <= 0:
        return True
    handle.seek(end_of_file - 1)
    return handle.read(1) in (b"", b"\n")
