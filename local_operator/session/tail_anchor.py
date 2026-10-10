"""A sidecar recording that a journal carries NO frontend checkpoint row.

THE COST THIS REMOVES. A journal with no ``frontend_state_checkpoint_v1`` row
anywhere puts ``read_replay_suffix`` back at BOF: the reader must reach the newest
row of every type it was asked for, and when the type is ABSENT that means walking
the whole file to prove it — measured 94 ms of CPU on a 27.2 MB checkpointless
journal, 570-650 ms on the 35 MB shape the audit used and 1.0-1.9 s on a 118 MB
one, on the cold open of every fresh process. The replay cache makes the repeats
free; nothing makes the FIRST read of such a journal cheap, because "there is no
such row" is a whole-file fact.

So the fact is written down once, by a job that has time: this sidecar states that,
as of a journal version, the checkpoint row is absent. A reader that needs the type
then stops at the journal's own compaction boundary instead of the file's start,
which is the same answer the full walk produces.

WHY A SIDECAR AND NOT A CHECKPOINT ROW. Writing a level checkpoint row at teardown
cannot reach these journals: the population is sessions whose runtimes are long
gone. The cold reader is not allowed to write (its contract), so the writer is a
background job at core start — ``index_prewarm``'s queue, the most recently
modified journals, off the hot path, behind the same disk and load guards.

WHY THE RECORDED SIZE IS THE END OF THE LAST COMPLETE ROW. A journal is appended
to while it is read. If the record named the file's size, a half-written checkpoint
row below that size would be inside the "nothing has happened here" region: the
scan that justifies the record would have seen a torn line, and the validator's
suffix scan would never look at the bytes the row completes into (review round 1,
F1 — the shape that made a torn append able to hide a checkpoint row). Recording
the end of the last COMPLETE row instead pushes every incomplete byte into the
validated suffix, where the needle scan sees it.

WHY THE SCAN'S WINDOWS OVERLAP. The point of the suffix scan is that the type
cannot have appeared in the bytes appended since the record. A needle split across
two fixed 1 MiB windows is invisible to both of them, so the windows overlap by one
needle's length, and the scan starts a needle's length below the recorded size
(review round 1, F2) — a marker written across the boundary is found, never
assumed absent.

IT IS A HINT, NEVER AN AUTHORITY. Every use is validated against the journal the
reader has open: same inode, a journal that has not shrunk below the recorded size,
the same last complete row ending at that size (a digest of it), and a scan of
everything from the recorded size on that proves the type's spelling absent.
Anything unproven falls back to the walking reader. The scan's assumption is that
the format writes compact JSON separators, which
``transcript.find_row_for_custom_type`` already assumes for its cursor and which
every writer of this format has emitted since the format was introduced.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    find_row_for_custom_type,
)

logger = logging.getLogger(__name__)

#: Sidecar name, beside the journal it describes.
ANCHOR_FILENAME = "tail-anchor.v1.json"

#: Sidecar schema version. A file from another version is ignored.
#:
#: 2 since review round 2 (M1). Version 1 also carried a recorded offset for
#: journals that DO hold the checkpoint row; this reader ignores fields it does not
#: know, so such a record — which says "a row exists at N" — would be read as the
#: proof of ABSENCE that version 2 records — the exact opposite of what it says.
#: No released store can hold one (that revision never merged), but a development
#: store that ran it keeps the file forever, because the selector sees a matching
#: size and never revisits the journal, so the version moves and those records are
#: ignored rather than reinterpreted.
ANCHOR_VERSION = 2

#: The custom type this sidecar answers for. One type, deliberately: it is the one
#: the cold open requires and cannot bound without.
ANCHOR_CUSTOM_TYPE = "frontend_state_checkpoint_v1"

#: Window size for the suffix scan. 1 MiB matches the reader's own read chunk,
#: which is what makes this cost the same per byte as the walk it replaces.
_WINDOW_BYTES = 1 << 20


def _needle() -> bytes:
    """The byte spelling of a checkpoint row's type, as the format writes it.

    Compact separators (no space after the colon) — see the module docstring on
    why that assumption is the format's own.
    """
    return ('"custom_type":"%s"' % ANCHOR_CUSTOM_TYPE).encode("utf-8")


@dataclass(frozen=True)
class TailAnchor:
    """A journal version at which the checkpoint row is known to be absent.

    ``size`` is the END OF THE LAST COMPLETE ROW at the moment of the scan, not the
    file's size: everything after it (a torn append, and anything written since) is
    re-examined by :func:`validate_anchor` rather than assumed.
    """

    inode: int
    size: int
    #: sha256 (first 16 hex) of the last complete row — its newline included.
    #:
    #: ``size`` alone cannot show that the bytes BELOW it are the ones the scan saw
    #: (review round 2, M2): the write path's rollback removes bytes with
    #: ``os.truncate``, and a later append can regrow the same inode past the
    #: recorded size, after which the not-shrunk guard sees a long file and the
    #: suffix scan sees only bytes the record never claimed. The digest pins the row
    #: that ends at ``size``, so a boundary that moved under the record is refused.
    tail_digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "version": ANCHOR_VERSION,
            "custom_type": ANCHOR_CUSTOM_TYPE,
            "inode": self.inode,
            "size": self.size,
            "tail_digest": self.tail_digest,
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
    inode = raw.get("inode")
    size = raw.get("size")
    digest = raw.get("tail_digest")
    if not isinstance(inode, int) or not isinstance(size, int):
        return None
    if not isinstance(digest, str):
        return None
    return TailAnchor(inode=inode, size=size, tail_digest=digest)


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


def _complete_row_end(path: Path, size: int) -> int:
    """The offset just past the last row that ends in a newline.

    ``0`` when no complete row exists in the last window (a journal holding one
    enormous unterminated line): the conservative answer, because a record at ``0``
    puts the whole file into the validated suffix, which costs a scan and risks
    nothing.
    """
    window = min(size, _WINDOW_BYTES)
    try:
        with path.open("rb") as handle:
            handle.seek(size - window)
            tail = handle.read(window)
    except OSError:
        return 0
    cut = tail.rfind(b"\n")
    return 0 if cut < 0 else size - window + cut + 1


def _last_complete_row(path: Path, end: int) -> bytes:
    """The bytes of the row that ends at ``end``, its newline included.

    Reads BACKWARD from ``end`` to the previous newline, in windows that start at
    2 MiB and grow, because a real checkpoint row is 0.65-0.83 MB — a fixed small
    window would return a partial row and its digest would then depend on the
    window size rather than on the bytes.
    """
    window = min(end, 2 << 20)
    while True:
        try:
            with path.open("rb") as handle:
                handle.seek(end - window)
                tail = handle.read(window)
        except OSError:
            return b""
        cut = tail.rfind(b"\n", 0, len(tail) - 1)
        if cut >= 0 or window >= end:
            return tail[cut + 1 :]
        window = min(end, window * 4)


def build_anchor(directory: str | Path) -> TailAnchor | None:
    """The anchor for this journal, or ``None`` when none is needed.

    ``None`` means "no sidecar belongs here": the journal carries the checkpoint row
    already (the walking reader finds it near the tail, so there is nothing to
    prove), or it cannot be read at all. The expensive case — the type is absent,
    and proving that costs a whole-file scan — is the one that gets a record, and it
    is paid here rather than on a reader's first paint.
    """
    path = Path(directory) / TRANSCRIPT_FILENAME
    try:
        stat = os.stat(path)
    except OSError:
        return None
    if find_row_for_custom_type(directory, ANCHOR_CUSTOM_TYPE) is not None:
        return None
    complete = _complete_row_end(path, stat.st_size)
    if complete == 0:
        # Nothing complete to anchor on: a record here would claim a region the
        # scan cannot describe, so no record is written and the walking reader
        # serves this journal (the ``0`` return of ``_complete_row_end`` has no
        # other caller — review round 2, N4).
        return None
    row = _last_complete_row(path, complete)
    if not row:
        return None
    return TailAnchor(
        inode=stat.st_ino,
        size=complete,
        tail_digest=hashlib.sha256(row).hexdigest()[:16],
    )


def _needle_absent(path: Path, start: int, end: int) -> bool:
    """Whether the checkpoint spelling is absent from ``[start, end)``.

    The windows OVERLAP by one needle length (review round 1, F2): a marker written
    across a window boundary is otherwise invisible to both windows that contain
    halves of it, and this scan is the only thing standing between a stale record
    and a reader that skips a real checkpoint row.
    """
    needle = _needle()
    overlap = len(needle) - 1
    position = end
    # ONE handle for the whole scan (review round 2, N3): the loop re-reads a
    # needle's worth of overlap per window, and a large appended delta would
    # otherwise pay an open() per MiB on the reader's cold path.
    try:
        handle = path.open("rb")
    except OSError:
        return False
    with handle:
        while position > start:
            window_start = max(start, position - _WINDOW_BYTES)
            try:
                handle.seek(window_start)
                window = handle.read(position - window_start + overlap)
            except OSError:
                return False
            if needle in window:
                return False
            position = window_start
    return True


def validate_anchor(directory: str | Path, anchor: TailAnchor | None) -> TailAnchor | None:
    """``anchor`` when it still describes the journal on disk, else ``None``.

    THE READER'S GATE, and the whole reason a hint is safe. It refuses when the file
    was replaced (inode changed), when it SHRANK below the recorded size (a rollback
    or a ``compact_file`` that reused the path), or when the bytes from the recorded
    size on contain the checkpoint type's own spelling (a row appeared, or a torn
    row completed into one).
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
    # The boundary pin (M2): the row ending at the recorded size must be the row the
    # scan saw. A truncate-and-regrow of the same inode leaves the size guard blind
    # and can hide a checkpoint row below the record.
    if anchor.size:
        row = _last_complete_row(path, anchor.size)
        if not row or hashlib.sha256(row).hexdigest()[:16] != anchor.tail_digest:
            return None
    if stat.st_size == anchor.size:
        # Nothing was appended since the scan; the record describes the file.
        return anchor
    needle = _needle()
    # Start a needle length below the recorded size: a marker written across that
    # offset has its first bytes in the scanned window and the rest after it.
    start = max(0, anchor.size - (len(needle) - 1))
    if not _needle_absent(path, start, stat.st_size):
        return None
    return anchor


def proves_absent(directory: str | Path) -> bool:
    """Whether a valid sidecar proves this journal has no checkpoint row.

    The one question the reader asks, and the only one this module answers.
    """
    return validate_anchor(directory, read_anchor(directory)) is not None


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
