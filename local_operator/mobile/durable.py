"""Incremental durable fold for the mobile daemon's read paths.

The daemon's durable routes — the SSE seed for a session with no live process
(:func:`.daemon._durable_projection`) and every history page the phone scrolls
up into (:func:`.daemon._history_page`) — used to construct a fresh
:class:`~local_operator.session.transcript.Transcript` per request: read the
WHOLE JSONL file, replay it, and fold it. On the operator's store that
measured 90-900 ms per session open and 50-150 ms per history page, O(full
transcript) every time, on transcripts up to 52 MB.

This module is the opt-in cache layer the daemon uses INSTEAD, built beside
the ``Transcript`` class rather than inside it: ``Transcript`` is the
LLM-facing store with append/compaction semantics of its own and must keep
its exact behaviour.

Design, in three facts:

- **The file is append-only per inode.** Appends write one whole JSON line,
  flush, and fsync; ``compact_file`` is the only rewrite and it replaces the
  file atomically (new inode). So a cached byte offset into a same-inode,
  non-shrunk file is a durable cursor: everything before it is unchanged, and
  the new content is exactly the bytes after it. A shrunk file or a new inode
  invalidates the cursor and forces a full rebuild.
- **Replay state stays small.** The cache retains the REPLAYED history
  (bounded by the compaction window) and the folded render rows — never the
  raw parsed entries of a 50 MB file, which would cost 3-5x the file size in
  Python objects. The rare tail events that need more are handled in place:
  a prune entry blanks its target message in the cached history; a compaction
  entry rebuilds the history from the cached window (its ``first_kept_entry_id``
  resolves there because the live session compacts exactly this window). Only
  when that resolution fails — the same edge ``build_llm_history`` logs an
  error for — does the cache fall back to a full re-read. Correctness over
  speed, and the slow path is the one the old code ran on EVERY request.
- **Fold cost is O(history), not O(file).** ``fold_messages_to_entries`` over
  the replayed history measures ~1 ms where the file parse measured hundreds,
  so rebuilding the render rows wholesale on every observed change is cheap;
  the incrementality that matters is the disk read (appended kilobytes, not
  the whole file).

Threading: the daemon calls from ``asyncio.to_thread`` workers, so the cache
is thread-safe — one lock guards the LRU dict, one lock per cached session
serializes its rebuilds (two concurrent opens of the same session pay one
fold, not two).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Container

from local_operator.harness.types import AgentMessage, Message, TextContent
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import (
    CUSTOM_KIND_CUSTOM,
    ENTRY_COMPACTION,
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    ENTRY_PRUNE,
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
    _compaction_marker,
    _entry_to_message,
    _journal_injection_ids,
    find_row_for_custom_type,
    read_latest_custom_entry,
    read_replay_suffix,
    read_transcript_page,
)

logger = logging.getLogger(__name__)

#: The subagent roster's custom-entry type, restated from
#: ``session.session.SUBAGENT_ROSTER_CUSTOM_TYPE`` (the same restatement
#: discipline the tracked-type set documents below: importing the session
#: module here would drag its whole import graph into every daemon boot).
ROSTER_CUSTOM_TYPE = "subagent_roster"

#: Custom-entry types the durable projection reads newest-wins (the store's
#: ``latest_custom`` contract). Tracked in the fold cache so a cached session
#: never re-parses its transcript to answer them. Importing the roster
#: constant pulls in the session module, so the literals are restated here and
#: asserted against the source of truth in the unit tests.
_TRACKED_CUSTOM_TYPES = frozenset({ROSTER_CUSTOM_TYPE, "todo_snapshot"})

#: The roster's own store, restated for the same reason as the type above:
#: ``session.session.SUBAGENT_ROSTER_SIDECAR`` is the name writers use, and a
#: unit test pins the two spellings together. Read (never written) here so the
#: fold can seed a roster whose transcript row sits far above its replay window
#: — 0 of the 40 largest real journals carry that row inside the window, while
#: 30 of them have this file.
ROSTER_SIDECAR_FILENAME = "subagent-roster.v1.json"

#: The sidecar's own schema version, mirrored from
#: ``session.session._SUBAGENT_ROSTER_VERSION``. A sidecar written by another
#: version is not read: the fold would have to guess at a payload shape the
#: writer owns, and the transcript fallback below is a correct answer for it.
_ROSTER_SIDECAR_VERSION = 1

#: Ceiling above which the fold will NOT go back into the journal for the roster
#: when the window and the sidecar both came up empty.
#:
#: WHY A CEILING AT ALL, AND WHY IT IS NO LONGER 32 MiB. QA round 1 (Q1) measured
#: the previous 32 MiB line LOSING data: a 35.5 MB journal with no sidecar and its
#: only ``subagent_roster`` row above the replay window served 0 subagent rows on
#: this branch and 4 on main. The line existed because the lookup was
#: ``read_latest_custom_entry``, whose cost is the ANSWER'S DISTANCE FROM EOF — a
#: parse of every row it steps over, measured at 457 ms on a 118 MB journal with
#: neither a sidecar nor a roster row.
#:
#: That lookup now finds its row by BYTES (``transcript.find_row_for_custom_type``,
#: the same needle the anchor lane uses): it scans chunks backward for the string
#: every row of the type carries and JSON-decodes only the candidate, measured at
#: 66 ms for the whole 118 MB journal — 7x cheaper, and no longer proportional to
#: how many rows it steps over. So the ceiling is now only a sanity bound against
#: a pathological file, set far above every journal that exists here (the largest
#: real one is 129 MB) rather than below a population that needs its roster.
_TRANSCRIPT_LOOKUP_MAX_BYTES = 256 * 1024 * 1024

#: Bound on cached durable folds, in sessions. One entry holds the replayed
#: history plus the UNCAPPED render rows (the history endpoint serves the full
#: conversation, not a tail).
#:
#: MEASURED, not estimated: ``tracemalloc`` against the operator's largest real
#: transcript (55.8 MB file, 441 replayed messages, 358 render rows) retains
#: **6.9 MB per entry** and takes ~1.2 s to fold. An earlier revision of this
#: comment called 16 entries "a bounded ~tens-of-MB", which was ~4x optimistic:
#: 16 x 6.9 MB is ~110 MB resident for a daemon that runs all day, and that is
#: a different decision from the one the number implied.
#:
#: 8 is the corrected bound, ~55 MB worst case. The phone opens a handful of
#: sessions in a sitting, so the hit rate barely moves; eviction costs only a
#: re-fold on the next open of an evicted session, and the incremental tail
#: read means a re-fold is paid once, not per request. Worst-case entries are
#: also the rare ones — a typical session folds to well under a megabyte.
MAX_DURABLE_FOLD_CACHE = 8


@dataclass
class _FileFingerprint:
    """Identity of one transcript file at one read position.

    The inode distinguishes ``compact_file``'s atomic replacement from an
    append (a rewrite that happens to keep or grow the size is still a
    different file); the size says how much has been consumed. mtime rides
    along for diagnostics but is not load-bearing — APFS timestamps are
    nanosecond-precise, yet a same-inode same-size file cannot have changed
    regardless of what its mtime claims.
    """

    inode: int
    size: int
    mtime: float


@dataclass
class DurableFoldState:
    """One session's cached durable fold.

    ``history`` is the replay (``Transcript.build_llm_history`` semantics),
    ``render`` the folded phone rows (``fold_messages_to_entries`` over that
    history, UNCAPPED). ``prunes`` mirrors ``Transcript.pending_prunes``:
    later entries win, applied at replay and re-applied after a compaction
    rebuild so a cached window stays byte-identical to a fresh replay.
    """

    directory: Path
    history: list[AgentMessage] = field(default_factory=list)
    render: list[Any] = field(default_factory=list)
    #: The newest replayed compaction's ``first_kept_entry_id``: the journal entry
    #: ``history`` begins at, and therefore the boundary the scroll-back archive
    #: (:func:`journal_rows_older_than`) pages below. ``None`` when the replay
    #: reaches the journal's first row — there is then nothing behind it to page.
    #: Kept here rather than re-derived by scanning the file: the fold has already
    #: located this row, and a second scan would be a second answer to one question.
    keep_start_id: str | None = None
    #: Whether the entries this fold read were the journal FROM ITS FIRST ROW.
    #: Two answers ride on this one fact, and both are wrong in the same way when
    #: it is assumed rather than known:
    #:
    #: - :func:`_replay` decides ``keep_start_id`` from it. The boundary's index
    #:   being 0 means "the journal starts here" only if the read started at the
    #:   journal's start; on a SUFFIX fold (lane T3's bounded read) an index of 0
    #:   just means the retained window happens to open the chunk, and answering
    #:   ``None`` there disables the archive for a conversation that still has
    #:   everything below it on disk.
    #: - :func:`journal_rows_older_than` reads the prune map from it. A
    #:   whole-file fold's map covers every row; a suffix fold's covers only the
    #:   window, and a page served with the wrong map shows tool output the live
    #:   fold had blanked.
    #:
    #: Set by :meth:`DurableFoldCache._rebuild`, from the fact the READ reports:
    #: :attr:`ReplaySuffix.reached_bof`, true only when the backward walk consumed
    #: the chunk holding the journal's first row. It used to be the caller's to
    #: declare — when the only reader read the whole file, the answer was always
    #: True and a caller could honestly say so. A bounded window is where the
    #: question first has a real answer, so it is derived where the read happens
    #: instead of being asked for (review round 9's fix, round 10's MINOR-10-1).
    #: Every reader of this field degrades to the honest answer.
    scan_from_bof: bool = True
    prunes: dict[str, str] = field(default_factory=dict)
    #: Transcript entries consumed so far. Not read by the fold itself — it is
    #: the cheap invariant that says the incremental cursor and the file agree,
    #: which the cache tests assert against a full re-parse. After the windowed
    #: cold fold it counts the rows the fold CONSUMED (the replayed window, then
    #: whatever the tail read appended), never the journal's total.
    entry_count: int = 0
    #: Bytes of journal the cold fold last READ, for the caller's own evidence
    #: and for the tests that have to prove the read was bounded (the walk's
    #: granularity is a chunk, so a row count cannot show it). Not a contract:
    #: nothing in the fold consults it. Mirrors ``ReplaySuffix.bytes_read``.
    window_bytes: int = 0
    #: Ids of message entries journalled from a harness aside, accumulated as
    #: entries stream in. The INCREMENTAL fold has no journal in hand, so
    #: without this it could not resolve preserved-turn provenance and would
    #: disagree with the full rebuild on a legacy marker. An earlier revision
    #: asserted in a comment that a legacy marker "reaches this path only
    #: through the full rebuild" — true in practice, enforced by nothing, and
    #: forced onto the incremental path the fold leaked an injection. Carrying
    #: the ids makes the two paths agree by construction instead.
    injection_ids: set[str] = field(default_factory=set)
    #: Newest-wins custom-entry details by type, mirroring the store's
    #: ``latest_custom``. The durable projection reads the subagent roster and
    #: a child's todo snapshot from here instead of re-parsing transcripts.
    latest_customs: dict[str, dict[str, Any]] = field(default_factory=dict)
    offset: int = 0
    fingerprint: _FileFingerprint | None = None
    #: Serializes rebuilds of THIS session; concurrent opens of one session pay
    #: one fold, not one per caller.
    lock: threading.Lock = field(default_factory=threading.Lock)


class DurableFoldCache:
    """Bounded LRU of :class:`DurableFoldState`, keyed by session directory."""

    def __init__(self, max_entries: int = MAX_DURABLE_FOLD_CACHE) -> None:
        self._max_entries = max_entries
        self._states: OrderedDict[str, DurableFoldState] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, directory: Path) -> DurableFoldState:
        """The fold state for ``directory``, LRU-touched and created if new."""
        key = str(directory)
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                self._states.move_to_end(key)
                return state
            state = DurableFoldState(directory=directory)
            self._states[key] = state
            while len(self._states) > self._max_entries:
                self._states.popitem(last=False)
            return state

    def invalidate(self, directory: Path) -> None:
        with self._lock:
            self._states.pop(str(directory), None)

    def invalidate_all(self) -> None:
        """Drop every cached fold — the invalidation a flag change needs.

        ``load`` re-derives ``render`` only when the transcript GROWS (or the
        file's inode moves); the display flags are not part of that
        fingerprint, so a cross-process ``display.*`` write must drop the
        states outright or the phone's scroll-back pages keep serving the
        pre-flip rows until an unrelated append (design review round 1 on
        #1746, D2). The next open of each session pays one full fold — the
        accepted cost of a config change, which is rare.
        """
        with self._lock:
            self._states.clear()

    def load(self, directory: Path) -> DurableFoldState:
        """Fold state brought current with the file on disk.

        The common path reads only the bytes appended since the last load;
        rotation (``compact_file``) or a failed incremental repair falls back
        to a rebuild, which reads the REPLAYED WINDOW rather than the whole file
        (:meth:`_rebuild`) — and reports whether that window reached the
        journal's first row, which the archive reads as
        :attr:`DurableFoldState.scan_from_bof`."""
        state = self.get(directory)
        with state.lock:
            path = directory / TRANSCRIPT_FILENAME
            try:
                stat = os.stat(path)
            except OSError:
                # The file vanished (retention sweep, manual cleanup): the
                # session has no history to serve. Reset so a re-created file
                # is not read against a stale cursor.
                self.invalidate(directory)
                raise FileNotFoundError(path)
            fingerprint = _FileFingerprint(
                inode=stat.st_ino, size=stat.st_size, mtime=stat.st_mtime
            )
            previous = state.fingerprint
            if previous is not None and fingerprint.inode == previous.inode:
                if fingerprint.size == previous.size:
                    return state  # nothing appended; the cache IS the file
                if fingerprint.size > previous.size:
                    try:
                        if self._apply_tail(state, path, fingerprint):
                            return state
                    except Exception:  # noqa: BLE001 — a bad tail must not 500 the route
                        logger.exception(
                            "durable fold: incremental read failed for %s; rebuilding", directory
                        )
            # New file, shrunk file, or failed increment: full rebuild.
            self._rebuild(state, path, fingerprint)
            return state

    # -- internals -----------------------------------------------------------

    def _apply_tail(
        self, state: DurableFoldState, path: Path, fingerprint: _FileFingerprint
    ) -> bool:
        """Consume the bytes appended since the last load. False = rebuild.

        Returns False (rather than raising) whenever the tail cannot be folded
        in place — the caller's full rebuild is always correct, so an
        incremental shortcut never gets to trade correctness for speed.
        """
        with path.open("rb") as handle:
            handle.seek(state.offset)
            data = handle.read(fingerprint.size - state.offset)
        if not data.endswith(b"\n"):
            # A writer may be mid-line when we read (appends are whole lines +
            # fsync, but the read can race the write). Consume only complete
            # lines; the fragment is picked up on the next load.
            cut = data.rfind(b"\n")
            if cut < 0:
                return True  # nothing complete yet; keep the cursor
            data = data[: cut + 1]
        consumed = len(data)
        new_entries: list[TranscriptEntry] = []
        for line in data.decode("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            entry = TranscriptEntry.from_json(line)
            if entry is not None:
                new_entries.append(entry)
        if not new_entries:
            state.offset += consumed
            state.fingerprint = fingerprint
            return True

        # Replay-affecting tail events are folded in place below; anything the
        # in-place rules cannot express defers to the caller's full rebuild.
        rebuild_render = False
        for entry in new_entries:
            if entry.type == ENTRY_MESSAGE:
                if entry.payload.get("kind") == CUSTOM_KIND_CUSTOM and entry.payload.get(
                    "custom_type"
                ):
                    # Same predicate as ``_journal_injection_ids``, applied
                    # incrementally so a later marker on this path sheds exactly
                    # what the full rebuild would.
                    state.injection_ids.add(entry.id)
                message = _entry_to_message(entry, _attachments())
                if message is None:
                    continue
                notice = state.prunes.get(entry.id)
                if notice is not None and isinstance(message, Message):
                    _apply_prune_to(message, notice)
                state.history.append(message)
                rebuild_render = True
            elif entry.type == ENTRY_PRUNE:
                target = entry.payload.get("target")
                if not target:
                    continue
                notice = str(entry.payload.get("notice", ""))
                state.prunes[str(target)] = notice
                # Blank the cached message the live session already blanked, so
                # the render rebuilt below matches a fresh replay byte for byte.
                for message in state.history:
                    if message.id == str(target) and isinstance(message, Message):
                        _apply_prune_to(message, notice)
                        break
                rebuild_render = True
            elif entry.type == ENTRY_COMPACTION:
                if not _rebuild_history_after_compaction(state, entry):
                    return False
                rebuild_render = True
            elif entry.type == ENTRY_CUSTOM:
                # Replay ignores custom entries; the durable projection reads a
                # few newest-wins snapshots (roster, todo) that the store
                # exposes via ``latest_custom``. Track them here so a cached
                # session never re-parses for them.
                custom_type = entry.payload.get("custom_type")
                if custom_type in _TRACKED_CUSTOM_TYPES:
                    state.latest_customs[str(custom_type)] = dict(entry.payload.get("details", {}))
        if rebuild_render:
            state.render = _fold(state.history)
        state.entry_count += len(new_entries)
        state.offset += consumed
        state.fingerprint = fingerprint
        return True

    def _rebuild(self, state: DurableFoldState, path: Path, fingerprint: _FileFingerprint) -> None:
        """Fold from the journal's REPLAYED WINDOW, not from the whole file.

        This used to be the pre-cache behaviour run verbatim: read every row of
        the journal, parse it, and replay it. On the operator's store that is
        1229 ms for a 121 MB conversation and grows with the file, for a replay
        that only ever reads the rows after the latest compaction — the same
        argument (and the same reader) the desktop's cold attach already uses
        (:func:`read_replay_suffix`, 11-130 ms on those files).

        WHY THE SHARED READER RATHER THAN A SECOND SUFFIX WALK: the stop rule
        here has to be the desktop's, byte for byte — the newest compaction's
        ``first_kept_entry_id`` must be in hand before the replay may cut, and a
        journal with no compaction has no boundary to stop at, so the honest
        answer there is the whole file (which is what this reader returns). Two
        implementations of that rule is exactly how the phone's notion of "what
        the model still sees" drifts from the desktop's.

        THE ROWS THE WINDOW DOES NOT CONTAIN are the other half of the change,
        and they are why this is not merely "parse less": it is the derived
        state (the subagent roster, the todo snapshot) whose rows are written
        once and then live far above the cut — the roster's legacy row is inside
        the window on 0 of the 40 largest real journals, its newest occurrence
        sat 32-670 ms of backward scan away. The rule for seeding it is the one
        the cold facet already applies in
        ``AttachedSession._restore_cold_subagents``: the SIDECAR is the roster's
        own store and the fresher of the two, so it wins when it is there; only
        a session that has none pays a backward lookup for the transcript row.
        Anything the window already carried costs nothing at all.

        TWO FACTS THE WINDOW HAS TO REPORT, both of them lane T2's, and they are
        the two answers ``at_bof`` below stands for: where the archive's cut is
        (``_replay``'s ``keep_start_id``) and whether the prune map can be
        trusted as the file's own (``scan_from_bof``). This reader used to be
        handed the fact by its caller, which read the whole file and therefore
        always said ``True``; a bounded window is where the question first has a
        real answer, so it is derived from the read itself rather than asked for.
        """
        # Sorted, so the tuple a reader sees is deterministic rather than
        # whatever order a frozenset happened to iterate in.
        suffix = read_replay_suffix(
            state.directory, opportunistic_types=tuple(sorted(_TRACKED_CUSTOM_TYPES))
        )
        entries = list(suffix.entries)
        state.window_bytes = suffix.bytes_read
        # THE READER'S OWN FACT, CARRIED RATHER THAN INFERRED. Two answers turn on
        # it: this fold's boundary (``_replay``'s ``keep_start_id``, the archive's
        # cut) and whether the prune map is the FILE's own (``scan_from_bof``,
        # which ``_journal_page`` reads to decide whether the archive must rebuild
        # it from the journal before serving a page — a prune marker sits ABOVE
        # the row it blanks, so a windowed fold that served its own map would hand
        # the phone output the live fold had hidden).
        #
        # ``reached_bof`` is true only when the backward walk consumed the chunk
        # at offset 0, i.e. the file's own first row was in hand; a journal with no
        # compaction to stop at walks all the way, a bounded window does not.
        #
        # NOT A BYTE-SPAN COMPARISON, and this comment used to make the argument
        # for exactly the wrong one. ``suffix.bytes_read >= fingerprint.size``
        # compares a span the READER measured at its own EOF against a size
        # ``load`` stat'ed earlier, so a file that GREW in between inflates the
        # left side: for growth G the test becomes ``chunk_start <= G``, which a
        # window that stopped short satisfies as soon as it stopped within G bytes
        # of the start. That answers True while the walk never reached the file's
        # beginning — the non-conservative direction, and the one that matters
        # here, because True tells the archive to trust a map that is incomplete
        # by construction (review round 9, MAJOR-9-1).
        at_bof = suffix.reached_bof
        # ``opportunistic_types`` never gates the scan (a type that may
        # legitimately be absent must not, see ``read_replay_suffix``), so this
        # is the free half: snapshots the window already passed.
        latest_customs: dict[str, dict[str, Any]] = {
            name: dict(details)
            for name, details in suffix.checkpoints.items()
            if name in _TRACKED_CUSTOM_TYPES
        }
        roster = _read_roster_sidecar(state.directory)
        if roster is not None:
            latest_customs[ROSTER_CUSTOM_TYPE] = roster
        elif fingerprint.size <= _TRANSCRIPT_LOOKUP_MAX_BYTES:
            # Only the ROSTER goes back into the journal. It is the one tracked
            # type with a consumer (``daemon._durable_projection`` reads it to
            # rebuild the child rows), and a session with no sidecar is a legacy
            # one whose roster row sits near the head of the journal — written
            # once, before the sidecar existed — so nothing nearer than a
            # backward scan can answer it. That scan is now a BYTE scan (see
            # ``_TRANSCRIPT_LOOKUP_MAX_BYTES``), which is why the ceiling is a
            # sanity bound rather than a population filter: QA round 1 (Q1) caught
            # the 32 MiB line losing a legacy roster on a 35.5 MB journal, and a
            # served roster is worth 66 ms of one-off scanning on the two journals
            # out of this store's forty that have no sidecar at all.
            #
            # THE TODO SNAPSHOT DELIBERATELY DOES NOT, and that is a trade rather
            # than an oversight. It IS harvested from the window whenever the
            # window passes it (17 of the 40 largest real journals); beyond that
            # nothing reads the entry: the daemon's durable projection reads the
            # roster only, and a child's todos come from ``CustomSnapshotCache``.
            # The scan costs O(distance) in the common case — measured 340 ms on
            # a 5.9 MB fixture where the row does not exist at all, i.e. work paid
            # on every cold open to fill a field no caller reads. A future
            # consumer of it needs a bounded reader of its own; this comment is
            # the hand-off.
            entry = read_latest_custom_entry(state.directory, ROSTER_CUSTOM_TYPE)
            if entry is not None:
                # Newest-wins, exactly as ``Transcript.latest_custom`` answers it
                # (same predicate, same projection) — one backward scan, not a
                # second full parse.
                latest_customs[ROSTER_CUSTOM_TYPE] = dict(entry.payload.get("details", {}))
        else:
            logger.debug(
                "durable fold: %s is too large to scan for a roster row",
                state.directory,
            )

        state.injection_ids = _journal_injection_ids(entries)
        state.history, state.keep_start_id = _replay(entries, at_bof=at_bof)
        state.scan_from_bof = at_bof
        state.render = _fold(state.history)
        state.prunes = {
            str(entry.payload.get("target")): str(entry.payload.get("notice", ""))
            for entry in entries
            if entry.type == ENTRY_PRUNE and entry.payload.get("target")
        }
        state.latest_customs = latest_customs
        # The rows this fold actually consumed. It is still the cursor/file
        # agreement invariant (the incremental path adds to it), but it now
        # counts the WINDOW rather than the file — asserting equality with the
        # file's line count would be asserting the whole-file read back.
        state.entry_count = len(entries)
        state.offset = fingerprint.size
        state.fingerprint = fingerprint


def _replay(
    entries: list[TranscriptEntry], *, at_bof: bool = True
) -> tuple[list[AgentMessage], str | None]:
    """``Transcript.build_llm_history`` semantics over parsed entries.

    Kept beside the cache rather than reused THROUGH a ``Transcript`` instance
    because constructing one mkdirs, takes an asyncio lock, and retains the raw
    entries — everything this module exists to avoid. The semantics are the
    contract: latest compaction wins, preserved user turns re-injected
    verbatim, prunes applied last.

    Returns the replayed history and the id of the journal entry it begins at
    (``None`` when that is the journal's own first row). The second value is the
    scroll-back archive's boundary and has to come from HERE: ``start`` is where
    the newest compaction's window opens, and any second derivation of it in the
    archive reader would be a second answer to one question.
    """
    compaction_index: int | None = None
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].type == ENTRY_COMPACTION:
            compaction_index = i
            break

    start = 0
    keep_start_id: str | None = None
    prefix: list[AgentMessage] = []
    if compaction_index is not None:
        compaction = entries[compaction_index]
        prefix = _compaction_prefix(compaction, _journal_injection_ids(entries))
        first_kept_id = compaction.payload.get("first_kept_entry_id")
        if first_kept_id is None:
            start = compaction_index + 1
        else:
            for i in range(len(entries)):
                if entries[i].id == first_kept_id:
                    start = i
                    break
            else:
                # Mirror ``build_llm_history``: replaying too much is
                # recoverable at the next compaction; silent amnesia is not.
                logger.error(
                    "durable fold: first_kept_entry_id %s not found; replaying full history",
                    first_kept_id,
                )
                start = 0
        # The boundary is the row the kept window OPENS at. A compaction with no
        # rows left below it opens at the compaction row itself, so the archive
        # still knows which line everything older than the window sits behind.
        boundary = start if start < len(entries) else compaction_index
        if boundary > 0 or not at_bof:
            keep_start_id = entries[boundary].id

    prunes = {
        str(entry.payload.get("target")): str(entry.payload.get("notice", ""))
        for entry in entries
        if entry.type == ENTRY_PRUNE and entry.payload.get("target")
    }
    out: list[AgentMessage] = list(prefix)
    for entry in entries[start:]:
        if entry.type != ENTRY_MESSAGE:
            continue
        message = _entry_to_message(entry, _attachments())
        if message is None:
            continue
        notice = prunes.get(entry.id)
        if notice is not None and isinstance(message, Message):
            _apply_prune_to(message, notice)
        out.append(message)
    return out, keep_start_id


def _compaction_prefix(
    compaction: TranscriptEntry,
    injection_ids: Container[str] | None = None,
) -> list[AgentMessage]:
    """The marker summary plus preserved user turns, exactly as
    ``build_llm_history`` injects them (see its docstring for why the turns
    ride the payload verbatim).

    ``injection_ids`` carries the same journal-resolved provenance the
    transcript read path uses. It is a parameter rather than something derived
    here because the fold's incremental rebuild has only the new entry in hand,
    not the journal — see the note at its call site for why passing ``None``
    there is correct rather than a gap.

    The marker itself comes from the transcript module's own
    ``_compaction_marker`` rather than being built here: that helper stamps
    ``id=entry.id``, and a locally built copy that let pydantic mint a uuid made
    the phone disagree with every other surface about which row it was holding.
    It was not cosmetic — the web client pages history with the id of its OLDEST
    row, so a marker id that changed on every fold made the cursor unresolvable
    and stopped scroll-back dead at the compaction for any session whose render
    was short enough to start with the marker.
    """
    prefix: list[AgentMessage] = [_compaction_marker(compaction)]
    preserved_turns = compaction.payload.get("preserved_user_turns") or ()
    if preserved_turns:
        from local_operator.compaction.cutpoint import (
            preserved_turn_payload,
            replay_preserved_turns,
        )

        # THE shared helper, not a copy of it: the mobile fold and a resumed
        # session must agree message for message, and three hand-kept copies of
        # this logic is where a later fix lands in two places out of three.
        preserved_turns = replay_preserved_turns(compaction.payload, injection_ids)

        for turn in preserved_turns:
            message = Message.user(str(turn.get("text", "")))
            turn_id = turn.get("id")
            if isinstance(turn_id, str) and turn_id:
                message.id = turn_id
            message.provider_payload = preserved_turn_payload(turn)
            prefix.append(message)
    return prefix


def _rebuild_history_after_compaction(state: DurableFoldState, entry: TranscriptEntry) -> bool:
    """Fold a tail compaction into the cached history without re-reading.

    The live session compacts exactly the window this cache replays, so the
    new marker's ``first_kept_entry_id`` resolves inside ``state.history``;
    when it does not (a dropped malformed line, a converter-minted id) the
    caller falls back to a full rebuild — the same direction
    ``build_llm_history`` chooses when it logs this edge.
    """
    # Provenance comes from ``state.injection_ids``, accumulated as entries
    # streamed in, because this path has no journal to scan. The previous
    # revision passed nothing and justified it by asserting that a legacy
    # marker could only arrive via the full rebuild. That was an unverified
    # structural claim of exactly the kind this PR exists to remove: nothing
    # enforced it, and driven onto this path directly the fold leaked an
    # injection and disagreed with the full rebuild. Now both paths resolve
    # from the same predicate, so they agree by construction.
    prefix = _compaction_prefix(entry, state.injection_ids)
    first_kept_id = entry.payload.get("first_kept_entry_id")
    kept: list[AgentMessage] = []
    if first_kept_id is not None:
        index = next((i for i, m in enumerate(state.history) if m.id == first_kept_id), None)
        if index is None:
            return False
        kept = state.history[index:]
        # The scroll-back archive boundary moves with the newest window, and it
        # moves HERE too: a compaction that arrives on the tail path (a live
        # session compacting while the phone is attached) leaves the cached
        # history beginning at the new marker, so anything that still read the
        # old boundary would page rows the render already holds. ``index > 0``
        # is the same condition ``_replay`` uses — a window that opens at the
        # history's own first row dropped nothing, so the boundary it had is
        # still the right one — and it needs the same qualification that reading
        # carries: "the history's own first row" means the JOURNAL's first row
        # only while the fold began there (``state.scan_from_bof``), which a
        # bounded suffix read does not.
        if index > 0 or not state.scan_from_bof:
            state.keep_start_id = str(first_kept_id)
    # Prunes journalled before this marker already shaped the kept window;
    # re-apply the map so a kept message blanked by an older prune stays
    # blanked (idempotent — same notice, same result).
    for message in kept:
        notice = state.prunes.get(message.id)
        if notice is not None and isinstance(message, Message):
            _apply_prune_to(message, notice)
    state.history = [*prefix, *kept]
    return True


def _apply_prune_to(message: Message, notice: str) -> None:
    """``transcript._apply_prune`` without the module-private import: blank
    the message the way the live pruning pass did, so a cached replay is
    indistinguishable from a fresh one."""
    message.content = [TextContent(text=notice)]
    message.provider_payload = {**(message.provider_payload or {}), "pruned": True}


def _fold(history: list[AgentMessage]) -> list[Any]:
    from local_operator.mobile.projection import fold_messages_to_entries

    return fold_messages_to_entries(history)


#: Journal entries read per backward step of the archive walk. The reader's own
#: ceiling (``api_session_history`` clamps ``limit`` to 200) so the archive never
#: asks the file for more than the route could have served.
_ARCHIVE_READ_LIMIT = 200

#: Rows read NEWER than each page's anchor, used only to pair the page's newest
#: calls with the results that answer them (:func:`_pairing_messages`). A tool
#: result is written immediately after its call, so one message's fan-out covers
#: every realistic split; the cost is a bounded forward walk the reader already
#: performs for an anchored page.
_ARCHIVE_PAIRING_MARGIN = 64


def _row_group(row_id: str, known: Container[str]) -> str:
    """The message a rendered row belongs to, resolved against the page at hand.

    A message paints its own row first and then one row per tool call, so the
    rows of one message are ``<message id>`` plus ``<message id>:<call id>`` —
    which makes the group a PREFIX relationship, not a stripped key: two sibling
    calls (``m:call-2`` and ``m:call-3``) share a base but neither is a prefix of
    the other, and an entry id may itself contain a colon
    (``subagent-launch:<job>``), so neither "strip everything after the first
    colon" nor "strip the last segment" is right on its own.

    ``known`` IS THE ROWS BEING CUT *AND* THE JOURNAL ENTRY IDS READ ON THE WAY
    (on a multi-read page, every entry id seen so far — a superset of the page's
    own, which can only move a cut further back), and the entry ids are
    load-bearing rather than belt-and-braces: an assistant message with
    tool calls and NO TEXT paints no row of its own, only ``m:call-0``,
    ``m:call-1``, …, so ``m`` never appears among the row ids and every sibling
    would answer with its own id as its group. The cut could then land between
    siblings, and the cursor it handed over (``m:call-2``) resolves — one prefix
    down — to ``m``, which serves only rows OLDER than the message: the first
    siblings were never asked for again. Measured on a real journal, 25-572 such
    messages in each of the 12 largest, and 2-3 rows lost per walk on both a
    synthetic case and a real clone (review round 2, R2-1). A row whose message
    is in neither set keeps its OWN id as its group, and that stays the safe
    direction: the cut can only fall a row lower.
    """
    head = row_id
    while ":" in head:
        head = head.rsplit(":", 1)[0]
        if head in known:
            return head
    return row_id


def _cursor_candidates(row_id: str) -> Iterator[str]:
    """The ids to try, longest first, for a rendered row id used as a cursor.

    A tool row's id is ``<message id>:<call id>``, and EITHER half can carry a
    colon of its own: ``harness/subagent.py`` mints ``subagent-launch:<job>``
    entries (13 of the 60 largest journals here name one as their newest
    ``first_kept_entry_id``), and real providers mint call ids shaped
    ``call_00_x7:06c08d4847``. So the id AS SERVED is tried first, then each
    prefix cut at a colon from the right, and the first one the reader can
    locate wins.

    Narrowing exactly ONCE was tried and is what this replaces: by the first
    colon it broke ``subagent-launch:<job>``, and by the last it broke
    ``m:call_00_X:hex`` (the cursor missed as given, the single narrowing named
    nothing, and the route answered ``([], False)`` — every older row gone,
    review round 2, R2-2). Each attempt costs the reader's own cursor lookup and
    nothing else, and the shapes that need a second attempt are rare (7 rows in
    one of the 150 largest journals).
    """
    yield row_id
    head = row_id
    while ":" in head:
        head = head.rsplit(":", 1)[0]
        yield head


#: The needle a prune entry's line carries, exactly as the journal writes it:
#: the writer emits compact JSON (``{"id":…,"type":"prune",…}``), so a byte scan
#: finds the candidate lines without parsing a single other row.
_PRUNE_LINE_NEEDLE = b'"type":"prune"'
#: Prune scans kept, keyed by the file's identity (path, inode, size — the same
#: reasoning as ``_CustomSnapshotEntry``: an append-only file cannot change
#: without changing its size, and ``compact_file`` replaces the inode).
_PRUNE_CACHE_MAX = 8
_PRUNE_CACHE: OrderedDict[tuple[str, int, int], dict[str, str]] = OrderedDict()
_PRUNE_CACHE_LOCK = threading.Lock()


def _journal_prunes(directory: Path, *, fallback: Mapping[str, str]) -> dict[str, str]:
    """Every prune the journal carries, for a reader that folded only a window.

    A prune marker is written ABOVE the row it blanks, so a reader that folds a
    suffix — or pages backward — cannot see the prunes that apply to the rows it
    serves. Serving them unblended is a redaction failure: the phone paints the
    tool output the live fold had already hidden (measured on the merged tree, a
    pruned result over 1 MiB above the replay window came back verbatim). The
    TUI's audit window sidesteps this by reading the resident entries
    (``collect_prunes``); this reader is disk-based and stateless, so it makes
    one byte pass looking for the prune needle and parses only the lines that
    carry it, cached against the file's identity ``(path, inode, size)`` — so the
    pass happens once per file STATE, and an appending session gets a new state
    per append: on a live session this is per page, not once per session as an
    earlier version of this sentence implied.

    THE INCREMENTAL SHAPE, named here as the follow-up rather than done here:
    prune rows are append-only, so a cache keyed on ``(inode, size_scanned)``
    could scan only the new span and merge it into the map it already holds,
    which would make the pass once per session again without changing what the
    map means. Until then the full pass is what guarantees the redaction, and it
    is cheaper than the parse it replaced (a C-level needle scan, not a decode
    per row).

    ``fallback`` is the map the caller already holds — the fold's own, which is
    complete whenever the fold read the whole journal. It is what an unreadable
    file answers with: the walk's own read is about to fail on the same file, so
    refusing to page would trade a real answer for no answer.
    """
    path = directory / TRANSCRIPT_FILENAME
    try:
        stat = os.stat(path)
    except OSError:
        return dict(fallback)
    key = (str(path), stat.st_ino, stat.st_size)
    with _PRUNE_CACHE_LOCK:
        cached = _PRUNE_CACHE.get(key)
        if cached is not None:
            _PRUNE_CACHE.move_to_end(key)
            return dict(cached)
    prunes: dict[str, str] = {}
    try:
        with path.open("rb") as handle:
            for raw in handle:
                if _PRUNE_LINE_NEEDLE not in raw:
                    continue
                entry = TranscriptEntry.from_json(raw.decode("utf-8", errors="replace"))
                if entry is None or entry.type != ENTRY_PRUNE:
                    continue
                target = entry.payload.get("target")
                if target:
                    prunes[str(target)] = str(entry.payload.get("notice", ""))
    except OSError:
        return dict(fallback)
    with _PRUNE_CACHE_LOCK:
        _PRUNE_CACHE[key] = prunes
        _PRUNE_CACHE.move_to_end(key)
        while len(_PRUNE_CACHE) > _PRUNE_CACHE_MAX:
            _PRUNE_CACHE.popitem(last=False)
    return dict(prunes)


def _page_messages(
    entries: Sequence[TranscriptEntry], prunes: Mapping[str, str]
) -> list[AgentMessage]:
    """Rehydrate one journal range into the messages the fold walks.

    The archive's own reader, shared by the page and by the pairing rows
    (:func:`_pairing_messages`): one conversion, so a page and its pairing
    cannot disagree about what a journal row is.
    """
    messages: list[AgentMessage] = []
    for entry in entries:
        if entry.type == ENTRY_COMPACTION:
            # A compaction row a reader scrolls past: the desktop paints it
            # where it sits in the journal, and dropping it here would make the
            # archived stretch look like a conversation that simply continued
            # across a summary that never happened. Same helper the live prefix
            # uses, so both markers are one object shape.
            messages.append(_compaction_marker(entry))
            continue
        if entry.type != ENTRY_MESSAGE:
            continue
        message = _entry_to_message(entry, _attachments())
        if message is None:
            continue
        notice = prunes.get(entry.id)
        if notice is not None and isinstance(message, Message):
            _apply_prune_to(message, notice)
        messages.append(message)
    return messages


def _pairing_messages(
    newer: Sequence[TranscriptEntry],
    page: Sequence[AgentMessage],
    prunes: Mapping[str, str],
) -> list[AgentMessage]:
    """The newer rows that answer the page's calls, for folding but not serving.

    WHY THIS EXISTS. ``fold_messages_to_entries`` pairs a call with its result by
    looking the call up in what it has ALREADY walked, so a range folded on its
    own leaves the calls at its newest edge unpaired — and an unpaired call
    paints ``interrupted``, which is a lie about a call that returned. Measured
    by diffing paged rows against a whole-journal fold on the fixtures: 2 tool
    rows per walk came out ``done -> interrupted``.

    The answers are appended to the page's fold INPUT and are never rows of their
    own: a result's row is the call's row, so settling one adds nothing to the
    page's row set, and the rule (tool results only, matching a call in the page)
    is what keeps a newer assistant row from importing its own calls as new rows.
    """
    answered = {
        str(message.tool_call_id or "")
        for message in page
        if isinstance(message, Message) and message.role == "tool"
    }
    wanted: set[str] = set()
    for message in page:
        if not isinstance(message, Message):
            continue
        for call in message.tool_calls or []:
            call_id = str(call.id or "")
            if call_id and call_id not in answered:
                wanted.add(call_id)
    if not wanted:
        return []
    return [
        message
        for message in _page_messages(newer, prunes)
        if isinstance(message, Message)
        and message.role == "tool"
        and str(message.tool_call_id or "") in wanted
    ]


def journal_rows_older_than(
    directory: Path,
    prunes: Mapping[str, str],
    *,
    before_id: str,
    limit: int,
    prunes_complete: bool = True,
) -> tuple[list[Any], bool] | None:
    """Phone rows for the journal entries OLDER than ``before_id``.

    WHY THIS EXISTS, and it is a reach bug rather than a speed one. The durable
    fold starts its replay at the newest compaction's ``first_kept_entry_id``, so
    everything a compaction dropped is absent from ``state.render`` — and
    ``_history_page`` pages the RENDER. On the S6 fixture that made 640 of the
    journal's 3,348 rows reachable from the phone: ``/history`` anchored at the
    oldest unfolded row answered the one row above it and then
    ``has_more: false``, so a phone reader could not scroll back past the last
    compaction at all. The desktop has always reached those rows — its history
    route pages the JOURNAL through :func:`read_transcript_page`, which is
    exactly what this function calls — so the two surfaces disagreed about what
    the same conversation contains.

    BOUNDED BY THE PAGE, NOT THE FILE, and that is the point of reusing the
    journal reader rather than re-folding a segment of the file: the reader
    byte-locates the cursor and walks one chunk plus the page (see its
    docstring), so a scroll-back costs the page it returns. Folding is still
    :func:`fold_messages_to_entries`, the ONE phone fold — the rows behind a
    compaction have the same shapes as the rows in front of it, and a parallel
    renderer here is how the two would drift.

    ``prunes`` is the fold state's own map (the ``[pruned]`` blanks the replay
    applies), and ``prunes_complete`` says whether that map covered the WHOLE
    journal. When it did not — a suffix fold (lane T3's bounded read) sees only
    its window — the map is rebuilt from the journal itself before anything is
    served, because a prune marker sits ABOVE the row it blanks: a page served
    without it hands the phone output the live fold had hidden, which is a
    redaction failure rather than a cosmetic one.

    EACH PAGE IS FOLDED WITH THE NEWER ROWS THAT ANSWER ITS CALLS
    (:func:`_pairing_messages`), because the fold pairs a call with its result by
    looking the call up in what it has already walked: folded alone, the calls at
    a page's newest edge render ``interrupted`` however long ago they returned.

    ``before_id`` is a rendered ROW id, not necessarily a bare journal entry id:
    the web client pages with the id it was served, and a tool row's id carries
    its call. :func:`_cursor_candidates` yields the id as given first and then
    each prefix cut at a colon from the right, so an id that carries a colon in
    EITHER half — a ``subagent-launch:<job>`` entry, a
    ``call_00_x7:06c08d4847`` call id — is located by whichever attempt names a
    real row, and a miss is a miss only once every attempt has failed.

    Returns ``None`` when ``before_id`` names no journal row — the cursor a
    client holds can outlive the file it came from (a compaction that landed
    mid-scroll, a replaced transcript) — so the caller keeps its own
    end-of-history answer instead of serving a duplicate tail. A ``before_id``
    the reader cannot locate is reported as the tail with ``reconciled=True``,
    which is what makes that distinguishable from a genuine miss.
    """
    rows: list[Any] = []
    has_more = False
    if not prunes_complete:
        prunes = _journal_prunes(directory, fallback=prunes)
    # Journal entry ids read on the way, for the page cut's row GROUPING: a
    # message with calls and no text paints no row of its own, so its own id is
    # only ever visible here (see :func:`_row_group`).
    entry_ids: set[str] = set()
    candidates = _cursor_candidates(before_id)
    candidate = next(candidates)
    # Whether the narrowed-cursor recovery below has run: it guards a second
    # splice on a multi-read page (every later cursor is a page row, and a page
    # never starts mid-group, so the first splice is the only one it can undo).
    recovered = False
    while len(rows) < limit:
        try:
            # AN ANCHORED read rather than a plain ``before_id`` one: the page
            # needs the newer rows that answer its newest calls, and the anchor
            # is where both halves meet. ``before`` is the reader's own ceiling
            # for a page, ``after`` the pairing margin — one message's fan-out,
            # so a call and its result are inside it on any real journal.
            page = read_transcript_page(
                directory,
                around_id=candidate,
                before=_ARCHIVE_READ_LIMIT,
                after=_ARCHIVE_PAIRING_MARGIN,
                limit=_ARCHIVE_READ_LIMIT + _ARCHIVE_PAIRING_MARGIN + 1,
            )
        except FileNotFoundError:
            return None
        if page.reconciled:
            candidate = next(candidates, None)
            if candidate is None:
                return None
            continue
        anchor = next((i for i, entry in enumerate(page.entries) if entry.id == candidate), None)
        if anchor is None:
            # Defensive only: an anchored page that did not reconcile always
            # contains its anchor. Falling through to the next candidate rather
            # than giving up keeps a future change to that contract from
            # re-breaking the colon-id fix above.
            candidate = next(candidates, None)
            if candidate is None:
                return None
            continue
        older = list(page.entries[:anchor])
        newer = list(page.entries[anchor:])
        entry_ids.update(entry.id for entry in page.entries)
        messages = _page_messages(older, prunes)
        if messages:
            messages += _pairing_messages(newer, messages, prunes)
            rows = _fold(messages) + rows
        if not recovered and candidate != before_id:
            # THE CALLER'S ID WAS ONE OF A MESSAGE'S CALL ROWS, so the page it is
            # asking for is not simply "everything older than that message".
            #
            # WHY THIS IS NEEDED AT ALL. The mobile projection caps its
            # transcript and makes room for the pinned opener by dropping the
            # OLDEST row of the tail (``projection.py:_cap_tail``), so a message
            # with tool calls can be split across the cap: the client keeps
            # ``m:call_0…`` and loses ``m`` itself. Its oldest row is then a call
            # row, that call row is its first cursor, and the entry that cursor
            # resolves to is the message — whose rows strictly older than the
            # ENTRY are the rows before ``m``. The rows ABOVE the cursor row
            # inside ``m``'s own group (``m`` itself, and any call row before it)
            # are in no other page: the walk has already started below them.
            # Measured by review round 3 on a real journal: one row stranded at
            # page 20 and at page 120, and 5 of the 14 largest journals here have
            # the shape (1-2 rows).
            #
            # ONLY when the id was narrowed to reach this entry: had the caller
            # named the entry itself, its own row is the caller's to hold, and
            # serving it back would put a repeat in every page.
            anchor_messages = _page_messages([page.entries[anchor]], prunes)
            anchor_messages += _pairing_messages(newer, anchor_messages, prunes)
            above = _fold(anchor_messages)
            stop = next((i for i, row in enumerate(above) if str(row.id) == before_id), None)
            # ONLY when the caller's id really is one of that message's rows. If
            # it names nothing in the group — a cursor from a transcript that was
            # replaced under the client, an id minted by an older build — there is
            # no row above it to recover, and the honest answer stays the one
            # below: the rows strictly older than the message, and end-of-history
            # when there are none (which is what keeps a stale cursor from
            # re-serving the client's own window).
            if stop is not None:
                # Older than the caller's row, newer than this page's rows.
                rows = rows + above[:stop]
                recovered = True
        has_more = page.has_more
        if not older or not has_more:
            break
        candidate = older[0].id
    if not rows:
        return [], False
    if len(rows) > limit:
        # Keep the rows NEXT to the cursor: the older ones are what the caller's
        # next page is for, and the walk above may overshoot by a fold's fan-out
        # (one message can paint several rows).
        cut = len(rows) - limit
        # EXCEPT that the cut has to fall on a row GROUP boundary. The cursor a
        # caller holds names a message, and a message's rows are ordered own-row
        # first then one per tool call — so a page that STARTS mid-group makes
        # the next page start strictly below the message, and the rows above the
        # cut are never asked for again. Measured: on the S6 fixture 2 of 1,813
        # folded rows were unreachable exactly this way; on this module's own
        # fixture, a message with more rows than a page stranded its own row and
        # its first call. So the cut MOVES BACK to the group's first row and the
        # page is that much longer — bounded by one message's fan-out, and the
        # reader pays rows it was going to be served in the next page anyway.
        known = {str(row.id) for row in rows} | entry_ids
        key = _row_group(str(rows[cut].id), known)
        while cut > 0 and _row_group(str(rows[cut - 1].id), known) == key:
            cut -= 1
        rows = rows[cut:]
        has_more = True
    return rows, has_more


@dataclass
class _CustomSnapshotEntry:
    """One directory's tracked newest-wins custom snapshots, validated by the
    file's inode and size: a same-inode same-size transcript cannot have
    changed (append-only per inode; ``compact_file`` replaces the file)."""

    inode: int
    size: int
    customs: dict[str, dict[str, Any]]
    #: Which types this entry has actually looked for. A miss is only a miss for
    #: a type that was SEARCHED: the reads are per-type now (see ``_scan``), so an
    #: entry built for ``todo_snapshot`` says nothing about ``subagent_roster``
    #: until that type is asked for too.
    scanned: set[str] = field(default_factory=set)


class CustomSnapshotCache:
    """Newest-wins custom snapshots for directories the fold cache does not keep.

    Exists for the deep-roster durable path: ``_durable_projection`` asks every
    child transcript for its todo snapshot, and routing those reads through the
    full fold cache would let an 80-child roster evict the ROOT's fold (the
    cache is bounded) — re-folding a 50 MB transcript on the next open. These
    snapshots are tiny (one details dict per type), so a large separate LRU
    costs almost nothing and keeps the fold cache for actual folds.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _CustomSnapshotEntry] = OrderedDict()
        self._lock = threading.Lock()

    def load(self, directory: Path, custom_type: str) -> dict[str, Any] | None:
        """Details of the newest ``custom_type`` entry, or ``None``.

        Re-stats on every call (one syscall) and re-scans only when the file
        grew or was replaced — a dead child's transcript never changes, so the
        scan runs once per daemon lifetime per child.
        """
        path = directory / TRANSCRIPT_FILENAME
        try:
            stat = os.stat(path)
        except OSError:
            with self._lock:
                self._entries.pop(str(directory), None)
            return None
        key = str(directory)
        with self._lock:
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.inode == stat.st_ino
                and entry.size == stat.st_size
                and custom_type in entry.scanned
            ):
                self._entries.move_to_end(key)
                return entry.customs.get(custom_type)
        details = self._scan(directory, custom_type)
        with self._lock:
            held = self._entries.get(key)
            if held is not None and held.inode == stat.st_ino and held.size == stat.st_size:
                # Another thread asked for a different type of the same version
                # while this one scanned: keep its answer.
                held.customs.update(details)
                held.scanned.add(custom_type)
                self._entries.move_to_end(key)
                return held.customs.get(custom_type)
            self._entries[key] = _CustomSnapshotEntry(
                inode=stat.st_ino,
                size=stat.st_size,
                customs=details,
                scanned={custom_type},
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return details.get(custom_type)

    @staticmethod
    def _scan(directory: Path, custom_type: str) -> dict[str, dict[str, Any]]:
        """The newest details for ONE tracked type, found by BYTES.

        WHY NOT A FORWARD PARSE (review round 1, F2). This used to stream the
        whole transcript, JSON-decoding every row to keep the newest snapshot of
        each tracked type in one pass. That was cheap enough while the roster it
        served had one record; once the roster came from its sidecar (all 255 of
        them), ``_durable_projection`` called this per child and ``_scan`` measured
        **2.15 s** of the phone's 3.31 s cold projection — a 2-3x regression on the
        first open of a real session, against a 300 ms target.

        The question is still "what does the newest row of this type say", so it
        is answered by the same needle reader the fold's roster fallback uses:
        chunked backward over the file, decoding only the candidate rows. Measured
        against the parse on the same children: ~7x cheaper, and proportional to
        BYTES rather than to rows.

        A missing row is still a real absence: the needle is exact for this
        format's spelling (``transcript.find_row_for_custom_type`` documents that
        assumption and keeps the parse walk as the authority for a type it cannot
        spell), and this reader's callers ask for types whose absence is normal.
        """
        located = find_row_for_custom_type(directory, custom_type)
        if located is None:
            return {}
        return {custom_type: dict(located[1].payload.get("details", {}))}


def _read_roster_sidecar(directory: Path) -> dict[str, Any] | None:
    """The roster sidecar's payload for ``directory``, or ``None``.

    The fold's seeding path for the one tracked type whose rows are written once
    and then sit above the replay window forever (see ``_rebuild``). Frozen
    only as an answer for THIS fold: the file is rewritten on every roster move,
    so a later ``load`` of a cached session does not re-read it — the same
    staleness the transcript-row source has (its legacy row is written once
    too), and the live path, not this cache, is what keeps a running
    conversation's roster current.

    Best-effort by construction: the durable projection must degrade to "no
    roster" rather than fail because a sidecar is half-written, absent, or from
    a version this build does not know. A payload that is not the shape the
    writer promises is refused rather than passed on, because the daemon reads
    ``jobs``/``records`` off it directly.
    """
    try:
        with (directory / ROSTER_SIDECAR_FILENAME).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _ROSTER_SIDECAR_VERSION:
        return None
    if not isinstance(payload.get("jobs"), list) or not isinstance(payload.get("records"), list):
        return None
    # The projection's own reader (``daemon._durable_projection``) takes what it
    # needs by key; the whole mapping is kept so the seeded value is the store's
    # record rather than a projection of it that could drop something.
    return dict(payload)


#: One shared store: it is content-addressed under config_dir() and read-only
#: here, so every cached session resolves its image references against the
#: same bytes the Transcript would.
_ATTACHMENT_STORES: dict[str, AttachmentStore] = {}
_ATTACHMENT_LOCK = threading.Lock()


def _attachments() -> AttachmentStore:
    from local_operator.paths import config_dir

    key = str(config_dir())
    with _ATTACHMENT_LOCK:
        store = _ATTACHMENT_STORES.get(key)
        if store is None:
            store = AttachmentStore()
            _ATTACHMENT_STORES[key] = store
        return store
