"""A sidecar recording where a journal's newest frontend checkpoint row is.

THE COST THIS REMOVES. A journal with no ``frontend_state_checkpoint_v1`` row
anywhere puts ``read_replay_suffix`` back at BOF: the reader must reach the newest
row of every type it was asked for, and when the type is ABSENT that means walking
the whole file to prove it — measured 570-650 ms of CPU on a 35 MB checkpointless
journal and 1.0-1.9 s on a 118 MB one, on the cold open of every fresh process.
The replay cache (#2108) makes the repeats free; nothing makes the FIRST read of
such a journal cheap, because "there is no such row" is a whole-file fact.

So the fact is written down once, by a job that has time: this sidecar records, for
one journal version, either the byte offset of its newest checkpoint row or the
PROOF that it has none. A reader that needs that type then stops at the journal's
own compaction boundary instead of the file's start, which is the same answer the
full walk produces — the type is not there, and the reader now knows it.

WHY A SIDECAR AND NOT A CHECKPOINT ROW. Writing a level checkpoint row at teardown
(as #2108 does for the populations that HAVE a runtime) cannot reach these
journals: the population is sessions whose runtimes are long gone. The reader is
not allowed to write (the cold reader's contract), so the writer is a background
job at core start — ``index_prewarm``'s queue, the K most recently modified
journals, off the hot path, behind the same disk and load guards.

IT IS A HINT, NEVER AN AUTHORITY. Every use is validated against the journal the
reader has open: same inode, the journal has not SHRUNK below the recorded size,
the delta appended since has been scanned for the checkpoint's own bytes, and the
recorded row still parses with the recorded id. Anything unproven falls back to
the walk. A sidecar that cannot be trusted is a sidecar that costs nothing.

WHY THE BYTES ARE EXACT FOR THIS FORMAT. The scan uses
``transcript.find_row_for_custom_type``'s needle, which is the spelling every
writer of this format emits (``to_json`` has used compact separators since the
format was introduced, 5cf2814a4f) — the same assumption the cursor locator
already makes for row ids, and the reason a proof of absence from this scan is a
proof rather than a guess.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    find_row_for_custom_type,
)

logger = logging.getLogger(__name__)

#: Sidecar name, beside the journal it describes.
ANCHOR_FILENAME = "tail-anchor.v1.json"

#: Sidecar schema version. A file from another version is ignored (the reader's
#: validation would have to guess at fields a future writer owns).
ANCHOR_VERSION = 1

#: The custom type this sidecar answers for. One type, deliberately: it is the one
#: the cold open requires and cannot bound without, and a second type would need
#: its own proof rather than a second field in this record.
ANCHOR_CUSTOM_TYPE = "frontend_state_checkpoint_v1"


@dataclass(frozen=True)
class TailAnchor:
    """Where a journal's newest checkpoint row is — or that it has none.

    ``checkpoint_offset`` is ``None`` for the proven-absent case, which is the one
    that matters most: it is the fact a reader cannot derive cheaply. ``row_id``
    accompanies an offset so the reader can verify it is looking at the row this
    record was written for.
    """

    inode: int
    size: int
    checkpoint_offset: int | None
    row_id: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "version": ANCHOR_VERSION,
            "custom_type": ANCHOR_CUSTOM_TYPE,
            "inode": self.inode,
            "size": self.size,
            "checkpoint_offset": self.checkpoint_offset,
            "row_id": self.row_id,
        }


def anchor_path(directory: str | Path) -> Path:
    return Path(directory) / ANCHOR_FILENAME


def read_anchor(directory: str | Path) -> TailAnchor | None:
    """The sidecar for ``directory``, or ``None`` — never raises, never creates."""
    try:
        with anchor_path(directory).open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("version") != ANCHOR_VERSION:
        return None
    if raw.get("custom_type") != ANCHOR_CUSTOM_TYPE:
        return None
    offset = raw.get("checkpoint_offset")
    row_id = raw.get("row_id")
    if offset is not None and not isinstance(offset, int):
        return None
    if offset is None and row_id is not None:
        return None
    inode = raw.get("inode")
    size = raw.get("size")
    if not isinstance(inode, int) or not isinstance(size, int):
        return None
    return TailAnchor(
        inode=inode,
        size=size,
        checkpoint_offset=offset,
        row_id=row_id if isinstance(row_id, str) else None,
    )


def write_anchor(directory: str | Path, anchor: TailAnchor) -> bool:
    """Persist ``anchor`` atomically; ``False`` when it could not be written.

    Best-effort by contract: the sidecar is an optimisation, and a failed write
    costs one open the cost it removes — never an answer.
    """
    target = anchor_path(directory)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(anchor.as_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return True
    except OSError:
        logger.debug("tail anchor not written for %s", directory, exc_info=True)
        try:
            temporary.unlink()
        except OSError:
            pass
        return False


def build_anchor(directory: str | Path) -> TailAnchor | None:
    """Scan ``directory``'s journal once and return the anchor it proves.

    One backward byte scan (``find_row_for_custom_type``: measured 66 ms for a
    118 MB journal, decoding only the candidate row) answers both halves: the
    newest checkpoint row's offset and id when there is one, and the proof of
    absence when there is not. ``None`` means the journal could not be measured at
    all — an absent file, an unreadable directory — and a caller must not turn that
    into "there is no checkpoint".
    """
    path = Path(directory) / TRANSCRIPT_FILENAME
    try:
        stat = os.stat(path)
    except OSError:
        return None
    located = find_row_for_custom_type(directory, ANCHOR_CUSTOM_TYPE)
    if located is None:
        return TailAnchor(inode=stat.st_ino, size=stat.st_size, checkpoint_offset=None, row_id=None)
    offset, entry = located
    return TailAnchor(
        inode=stat.st_ino, size=stat.st_size, checkpoint_offset=offset, row_id=entry.id
    )


def _needle(handle: "BinaryIO", start: int, end: int) -> bool:
    """Whether the checkpoint type's bytes appear in ``[start, end)``."""
    needle = ('"custom_type":"%s"' % ANCHOR_CUSTOM_TYPE).encode("utf-8")
    position = end
    while position > start:
        chunk_start = max(start, position - (1 << 20))
        handle.seek(chunk_start)
        window = handle.read(position - chunk_start)
        if needle in window:
            return True
        position = chunk_start
    return False


def validate_anchor(directory: str | Path, anchor: TailAnchor | None) -> TailAnchor | None:
    """``anchor`` when it still describes the journal on disk, else ``None``.

    THE READER'S GATE, and the whole reason a hint is safe. It refuses when the
    file was replaced (inode), when it SHRANK below the recorded size (a rollback
    or a ``compact_file`` that reused the path), when the bytes appended since the
    record contain the checkpoint type's own spelling (a newer row exists and the
    record does not know about it), or when a recorded row no longer parses with
    the recorded id. Only the proven-absent half additionally needs the delta scan:
    a recorded OFFSET stays valid while the file only grows, because it is still
    the offset of a row of that type — a newer one above it does not move it, and
    the caller's own walk covers everything from the file's end down.
    """
    if anchor is None:
        return None
    path = Path(directory) / TRANSCRIPT_FILENAME
    try:
        stat = os.stat(path)
    except OSError:
        return None
    if stat.st_ino != anchor.inode or stat.st_size < anchor.size:
        return None
    if anchor.checkpoint_offset is None:
        if stat.st_size == anchor.size:
            return anchor
        try:
            with path.open("rb") as handle:
                if _needle(handle, anchor.size, stat.st_size):
                    return None
        except OSError:
            return None
        return anchor
    # A recorded offset: re-read that row and check it is the row this record
    # names. Cheap (one small read), and the only way an offset can be wrong.
    try:
        with path.open("rb") as handle:
            handle.seek(anchor.checkpoint_offset)
            raw = handle.readline()
    except OSError:
        return None
    from local_operator.session.transcript import TranscriptEntry

    entry = TranscriptEntry.from_json(raw.decode("utf-8", errors="replace"))
    if entry is None or entry.id != anchor.row_id:
        return None
    return anchor


def proves_absent(directory: str | Path) -> bool:
    """Whether a valid sidecar PROVES this journal has no checkpoint row.

    The one question the reader asks, and the only one this module answers for it.
    """
    anchor = validate_anchor(directory, read_anchor(directory))
    return anchor is not None and anchor.checkpoint_offset is None


__all__ = [
    "ANCHOR_CUSTOM_TYPE",
    "ANCHOR_FILENAME",
    "ANCHOR_VERSION",
    "TailAnchor",
    "anchor_path",
    "build_anchor",
    "proves_absent",
    "read_anchor",
    "validate_anchor",
    "write_anchor",
]
