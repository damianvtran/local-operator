"""A bounded, explicitly non-canonical journal preview for terminal navigation.

Only complete suffix rows are replayed. Prunes occur after their targets, so
all prunes affecting these rows are in this same suffix. An incomplete final
row, malformed row, or unresolved compaction cut makes the preview unavailable:
we cannot prove that omitted bytes did not retract something we would display.

Externalized attachments ARE hydrated, through the store that owned the journal
(``store_for_transcript_dir``) rather than the reader's environment, because a
preview that paints "image unavailable" for a picture the session actually has
is the same false receipt this reader exists to avoid. The cost is bounded and
paid off the loop: rows are still capped at ``PREVIEW_BYTES`` of journal, and
each image reference is one file read whose size is that image (no decoding
here — the widget decodes). What stays true from before: this reader never
starts an owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.session.attachments import store_for_transcript_dir
from local_operator.session.history_window import DISPLAY_HISTORY_MESSAGES
from local_operator.session.transcript import TranscriptEntry, replay_entries

PREVIEW_BYTES = 256 * 1024


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
        handle.seek(0, 2)
        size = handle.tell()
        start = max(0, size - PREVIEW_BYTES)
        handle.seek(start)
        data = handle.read(PREVIEW_BYTES)
    partial = start > 0
    if partial:
        _, separator, data = data.partition(b"\n")
        if not separator:
            return SavedPreview([], True)
    if data and not data.endswith(b"\n"):
        return SavedPreview([], True)
    entries = []
    try:
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            entry = TranscriptEntry.from_json(line)
            if entry is None:
                return SavedPreview([], True)
            entries.append(entry)
    except UnicodeError:
        return SavedPreview([], True)
    ids = {entry.id for entry in entries}
    for entry in reversed(entries):
        if entry.type == "compaction":
            kept = entry.payload.get("first_kept_entry_id")
            if kept is not None and kept not in ids:
                return SavedPreview([], True)
            break
    cwd = next(
        (str(entry.payload.get("cwd", "")) for entry in entries if entry.type == "session"),
        "",
    )
    messages = replay_entries(entries, store_for_transcript_dir(directory))
    # The byte ceiling bounds disk/parse work; a separate row ceiling bounds
    # replay bookkeeping and Textual's preparation of many tiny messages. Apply
    # prunes/compaction BEFORE slicing, never truncate away their instructions.
    return SavedPreview(
        messages[-DISPLAY_HISTORY_MESSAGES:],
        partial or len(messages) > DISPLAY_HISTORY_MESSAGES,
        cwd,
    )
