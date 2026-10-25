"""Validated checkpoint-absence and model-selection evidence for a journal.

The optional selection ledger shares this module's inode, complete-row boundary,
digest, atomic publication and suffix-needle validation. It stores only the input
to the existing selection fold, never a provider verdict. A writer can publish a
selection-only record (checkpoint_absent=False); that is NOT permission to shorten
replay. Version 2 absence-only records remain readable, and prewarm still owns
building an absence proof. Missing evidence always falls back to the exact scan.

A sidecar originally recorded that a journal carries NO frontend checkpoint row.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    find_row_for_custom_type,
)

logger = logging.getLogger(__name__)

#: Sidecar name, beside the journal it describes.
ANCHOR_FILENAME = "tail-anchor.v1.json"

#: Sidecar schema version. A file from another version is ignored.
#:
#: Version 3 adds selection evidence and an explicit checkpoint-absence flag.
#: Version 2 remains readable as an absence-only proof; v1 is still refused.
#: 2 since review round 2 (M1). Version 1 also carried a recorded offset for
#: journals that DO hold the checkpoint row; this reader ignores fields it does not
#: know, so such a record — which says "a row exists at N" — would be read as the
#: proof of ABSENCE that version 2 records — the exact opposite of what it says.
#: No released store can hold one (that revision never merged), but a development
#: store that ran it keeps the file forever, because the selector sees a matching
#: size and never revisits the journal, so the version moves and those records are
#: ignored rather than reinterpreted.
ANCHOR_VERSION = 3

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
    # Source evidence, not a resolved provider verdict: registry changes must
    # still be evaluated by the selection fold on every fresh process.
    selection_payloads: tuple[dict[str, Any], ...] | None = None
    checkpoint_absent: bool = True
    # The selection prefix can change without moving the terminal row. Ordinary
    # in-place edits change these stamps; the absence-only legacy proof never
    # claimed to validate such edits. Missing stamps mean untrusted selection.
    selection_mtime_ns: int | None = None
    selection_ctime_ns: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "version": ANCHOR_VERSION,
            "custom_type": ANCHOR_CUSTOM_TYPE,
            "inode": self.inode,
            "size": self.size,
            "tail_digest": self.tail_digest,
            "selection_payloads": self.selection_payloads,
            "checkpoint_absent": self.checkpoint_absent,
            "selection_mtime_ns": self.selection_mtime_ns,
            "selection_ctime_ns": self.selection_ctime_ns,
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
    if not isinstance(raw, dict) or raw.get("version") not in (2, ANCHOR_VERSION):
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
    payloads = raw.get("selection_payloads")
    absent = (
        raw.get("checkpoint_absent", True) if raw["version"] == 2 else raw.get("checkpoint_absent")
    )
    if not isinstance(absent, bool):
        return None
    if payloads is not None and (
        not isinstance(payloads, list) or not all(isinstance(item, dict) for item in payloads)
    ):
        return None
    mtime = raw.get("selection_mtime_ns")
    ctime = raw.get("selection_ctime_ns")
    return TailAnchor(
        inode,
        size,
        digest,
        None if payloads is None else tuple(payloads),
        absent,
        mtime if type(mtime) is int else None,
        ctime if type(ctime) is int else None,
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


def drop_anchor(directory: str | Path) -> None:
    """Remove this session's anchor sidecar, if any. Best-effort by contract.

    An anchor describes a PLAIN journal (an inode, a size, the digest of the row
    ending there). The moment that journal is stored compressed the record can
    never validate again, so keeping it only misleads the prewarm selector into
    calling the journal warm. Called by the compression pass's after-care, by
    :func:`build_anchor` when it meets an archive, and by the prewarm selector.
    """
    try:
        (Path(directory) / ANCHOR_FILENAME).unlink()
    except OSError:
        pass


def build_anchor(directory: str | Path) -> TailAnchor | None:
    """The anchor for this journal, or ``None`` when none is needed.

    ``None`` means "no sidecar belongs here": the journal carries the checkpoint row
    already (the walking reader finds it near the tail, so there is nothing to
    prove), or it cannot be read at all. The expensive case — the type is absent,
    and proving that costs a whole-file scan — is the one that gets a record, and it
    is paid here rather than on a reader's first paint.
    """
    from local_operator.session import compression

    if compression.archive_path(Path(directory)) is not None:
        # Nothing to anchor: the reader validates a plain inode and offsets, and
        # an archive has neither. Drop any stale record so the prewarm selector
        # does not read it as "this journal is warm".
        drop_anchor(directory)
        return None
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


def _needle_absent(path: Path, start: int, end: int, needle: bytes | None = None) -> bool:
    """Whether the checkpoint spelling is absent from ``[start, end)``.

    The windows OVERLAP by one needle length (review round 1, F2): a marker written
    across a window boundary is otherwise invisible to both windows that contain
    halves of it, and this scan is the only thing standing between a stale record
    and a reader that skips a real checkpoint row.
    """
    needle = _needle() if needle is None else needle
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


def validate_anchor(
    directory: str | Path,
    anchor: TailAnchor | None,
    *,
    needles: tuple[bytes, ...] | None = None,
) -> TailAnchor | None:
    """``anchor`` when it still describes the journal on disk, else ``None``.

    THE READER'S GATE, and the whole reason a hint is safe. It refuses when the file
    was replaced (inode changed), when it SHRANK below the recorded size (a rollback
    or a ``compact_file`` that reused the path), or when the bytes from the recorded
    size on contain the checkpoint type's own spelling (a row appeared, or a torn
    row completed into one).
    """
    if anchor is None or (needles is None and not anchor.checkpoint_absent):
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
    # Start a needle length below the recorded size: a marker written across that
    # offset has its first bytes in the scanned window and the rest after it.
    for needle in (_needle(),) if needles is None else needles:
        start = max(0, anchor.size - (len(needle) - 1))
        if not _needle_absent(path, start, stat.st_size, needle):
            return None
    return anchor


def proves_absent(directory: str | Path) -> bool:
    """Whether a valid sidecar proves this journal has no checkpoint row.

    The one question the reader asks, and the only one this module answers.
    """
    anchor = validate_anchor(directory, read_anchor(directory))
    return anchor is not None and anchor.checkpoint_absent


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


_SELECTION_NEEDLES = (b'"custom_type":"selected_model"', _needle())


def read_selection_payloads(directory: str | Path) -> tuple[dict[str, Any], ...] | None:
    """Source ledger validated with the existing inode/boundary/suffix gate."""
    anchor = _validated_selection_anchor(directory, read_anchor(directory))
    return None if anchor is None else anchor.selection_payloads


def _validated_selection_anchor(
    directory: str | Path, anchor: TailAnchor | None
) -> TailAnchor | None:
    anchor = validate_anchor(directory, anchor, needles=_SELECTION_NEEDLES)
    try:
        source = (Path(directory) / TRANSCRIPT_FILENAME).stat()
    except OSError:
        return None
    # Check generation AFTER the bounded boundary/suffix reads, so an ordinary
    # mutation during those reads also refuses this evidence.
    return anchor if _selection_generation_matches(anchor, source) else None


def _selection_generation_matches(anchor: TailAnchor | None, source: os.stat_result) -> bool:
    """Bounded proof for ordinary edits, not adversarial restored timestamps.

    Both stamps are required: bookkeeping restores mtime, while ctime still
    changes. An unobserved generation change is not assumed append-only. Readers
    fall back; writers may reuse a prefix only against their PRE-append stat.
    """
    return (
        anchor is not None
        and anchor.selection_payloads is not None
        and anchor.inode == source.st_ino
        and anchor.selection_mtime_ns == source.st_mtime_ns
        and anchor.selection_ctime_ns == source.st_ctime_ns
    )


def preserve_selection_evidence(directory: str | Path, anchor: TailAnchor) -> TailAnchor:
    """Keep compatible source evidence when prewarm publishes an absence proof.

    Neither artifact grants the other's privilege: the new absence proof comes
    from build_anchor's real scan; selection is copied only from a currently
    validated record naming that SAME complete-row boundary. A racing append or
    incompatible proof loses the optimization, never changes an answer.
    """
    # Validate ONE captured record. Re-reading its payload separately could mix
    # a racing writer's new payload with the first record's old boundary/stamps.
    saved = _validated_selection_anchor(directory, read_anchor(directory))
    if saved is None:
        return anchor
    if (saved.inode, saved.size, saved.tail_digest) != (
        anchor.inode,
        anchor.size,
        anchor.tail_digest,
    ):
        return anchor
    return replace(
        anchor,
        selection_payloads=saved.selection_payloads,
        selection_mtime_ns=saved.selection_mtime_ns,
        selection_ctime_ns=saved.selection_ctime_ns,
    )


def update_selection_after_commit(
    directory: str | Path,
    *,
    previous_source: os.stat_result | None = None,
    appended_end: int | None = None,
    tail_row: bytes | None = None,
) -> None:
    """Persist expendable selection evidence only after the journal's fsync.

    ``previous_source`` / ``appended_end`` / ``tail_row`` are the DURABLE WRITER's
    own facts: the file as it was immediately before this append, the size it
    reached, and the last complete row's bytes. Supplying them keeps the refresh
    proportional to the bytes just written. Without them the refresh must
    re-derive the boundary from disk, which is a whole-journal read on the first
    commit that meets an unledgered journal — paid once, never per message.

    A crash before replace leaves an older prefix; the suffix needle scan refuses
    it when selection evidence changed. Truncation and compaction use the existing
    size/inode/digest gates. This never changes replay's checkpoint-absence proof:
    L3b is deliberately deferred. Failure must never reject a durable append.
    """
    try:
        _update_selection_after_commit(Path(directory), previous_source, appended_end, tail_row)
    except Exception:  # noqa: BLE001 — derived data cannot reject durable work
        logger.debug("selection ledger not written for %s", directory, exc_info=True)


def _update_selection_after_commit(
    directory: Path,
    previous_source: os.stat_result | None,
    appended_end: int | None,
    tail_row: bytes | None,
) -> None:
    from local_operator.session.transcript import _iter_complete_lines_forward

    path = directory / TRANSCRIPT_FILENAME
    # Writer-only: consume ALL new rows rather than claiming their needles absent.
    # No read-path writer and no migration of a journal nobody is writing.
    saved = read_anchor(directory)
    # THE STEADY PATH. Only the durable writer can attest which source generation
    # preceded its append, so its pre-append stat is what lets this reuse the
    # stored prefix without re-reading it. Comparing a stored record to POST-append
    # stamps instead would either force a whole-journal read every turn or bless an
    # in-place edit; the stamps therefore never validate the prefix on their own.
    tail_payload = tail_row or b""
    incremental = (
        saved is not None
        and saved.selection_payloads is not None
        and previous_source is not None
        and appended_end is not None
        and tail_row
        and appended_end >= previous_source.st_size
        and _selection_generation_matches(saved, previous_source)
    )
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        # ONE scope for both branches: ``complete`` is the boundary the record will
        # name, and the reader's validator pins it with the tail digest.
        if incremental:
            assert saved is not None and previous_source is not None and appended_end is not None
            start = previous_source.st_size
            complete = appended_end
            payloads = list(saved.selection_payloads or ())
            # Carrying the absence proof forward is sound only because the validated
            # generation proves nothing outside this append changed, so the appended
            # rows themselves are the whole question. A checkpoint row in them costs
            # the proof, never the answer.
            absent = bool(saved.checkpoint_absent)
            digest = hashlib.sha256(tail_payload).hexdigest()[:16]
        else:
            start = 0
            complete = _complete_row_end(path, stat.st_size)
            payloads = []
            row_bytes = _last_complete_row(path, complete)
            if not row_bytes:
                return
            digest = hashlib.sha256(row_bytes).hexdigest()[:16]
            # Bootstrap only: the absence proof is about the WHOLE file, so it is
            # re-derived by the existing validator rather than assumed.
            proof = validate_anchor(directory, saved)
            absent = proof is not None and proof.checkpoint_absent
        for raw in _iter_complete_lines_forward(handle, start, end_of_file=complete):
            if b'"custom_type"' not in raw:
                continue
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(row, dict) or row.get("type") != "custom":
                continue
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            details = payload.get("details")
            if not isinstance(details, dict):
                continue
            kind = payload.get("custom_type")
            if kind == "selected_model":
                reduced = {
                    key: details[key]
                    for key in ("version", "selector", "effort", "boot", "fast")
                    if key in details
                }
            elif kind == ANCHOR_CUSTOM_TYPE:
                absent = False
                state = details.get("state")
                model = state.get("selected_model") if isinstance(state, dict) else None
                if not isinstance(model, dict):
                    continue
                reduced = {
                    "state": {
                        "selected_model": {
                            key: model[key]
                            for key in ("provider", "model_id", "reasoning_effort")
                            if key in model
                        }
                    }
                }
            else:
                continue
            payloads.append({"custom_type": kind, "details": reduced})
    write_anchor(
        directory,
        TailAnchor(
            stat.st_ino,
            complete,
            digest,
            tuple(payloads),
            absent,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ),
    )
