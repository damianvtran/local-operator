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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Container

from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    TextContent,
)
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import (
    CUSTOM_KIND_CUSTOM,
    ENTRY_COMPACTION,
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    ENTRY_PRUNE,
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
    _entry_to_message,
    _journal_injection_ids,
    read_latest_custom_entry,
    read_replay_suffix,
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
#: WHY A CEILING AT ALL. That lookup is ``read_latest_custom_entry``, bounded by
#: the answer's DISTANCE FROM EOF and by nothing the caller can choose — correct
#: for its own contract (a caller today always gets an answer) and expensive for
#: a cold fold that asks it for a type which is usually ABSENT: measured 457 ms
#: on a 118 MB journal with neither a sidecar nor a roster row, against a 20 ms
#: window read and a 310 ms fold. Paying that per cold phone open would be the
#: exact regression this module's window read exists to remove.
#:
#: WHY 32 MiB. Below it the lookup is bounded by a file that small, so the scan
#: costs what the whole-file parse it replaces costs (~5 ms/MB) and a legacy
#: journal keeps its roster rows. Above it the roster comes from the sidecar,
#: which is the roster's own store and the fresher of the two — the same
#: precedence ``AttachedSession._restore_cold_subagents`` applies. The real store
#: sits below the line: of the 40 largest real journals, every one WITHOUT a
#: sidecar was at most 27.2 MB.
_TRANSCRIPT_LOOKUP_MAX_BYTES = 32 * 1024 * 1024

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
        to a full rebuild, which is still exactly what every request used to
        do before this cache existed.
        """
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
        """
        # Sorted, so the tuple a reader sees is deterministic rather than
        # whatever order a frozenset happened to iterate in.
        suffix = read_replay_suffix(
            state.directory, opportunistic_types=tuple(sorted(_TRACKED_CUSTOM_TYPES))
        )
        entries = list(suffix.entries)
        state.window_bytes = suffix.bytes_read
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
            # Only the ROSTER goes back into the journal, and only for a session
            # small enough that the scan is bounded by the file it scans. It is
            # the one tracked type with a consumer (``daemon._durable_projection``
            # reads it to rebuild the child rows), and a session with no sidecar
            # is a legacy one whose roster row sits near the head of the journal
            # — written once, before the sidecar existed — so nothing nearer than
            # a backward scan can answer it.
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
        state.history = _replay(entries)
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


def _replay(entries: list[TranscriptEntry]) -> list[AgentMessage]:
    """``Transcript.build_llm_history`` semantics over parsed entries.

    Kept beside the cache rather than reused THROUGH a ``Transcript`` instance
    because constructing one mkdirs, takes an asyncio lock, and retains the raw
    entries — everything this module exists to avoid. The semantics are the
    contract: latest compaction wins, preserved user turns re-injected
    verbatim, prunes applied last.
    """
    compaction_index: int | None = None
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].type == ENTRY_COMPACTION:
            compaction_index = i
            break

    start = 0
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
    return out


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
    """
    details: dict[str, Any] = {"summary": compaction.payload.get("summary", "")}
    preserve_data = compaction.payload.get("preserve_data")
    if preserve_data is not None:
        details["preserve_data"] = preserve_data
    prefix: list[AgentMessage] = [
        CustomMessage(
            custom_type="compaction_summary",
            attribution="system",
            details=details,
        )
    ]
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


@dataclass
class _CustomSnapshotEntry:
    """One directory's tracked newest-wins custom snapshots, validated by the
    file's inode and size: a same-inode same-size transcript cannot have
    changed (append-only per inode; ``compact_file`` replaces the file)."""

    inode: int
    size: int
    customs: dict[str, dict[str, Any]]


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
            if entry is not None and entry.inode == stat.st_ino and entry.size == stat.st_size:
                self._entries.move_to_end(key)
                return entry.customs.get(custom_type)
        customs = self._scan(path)
        with self._lock:
            self._entries[key] = _CustomSnapshotEntry(
                inode=stat.st_ino, size=stat.st_size, customs=customs
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return customs.get(custom_type)

    @staticmethod
    def _scan(path: Path) -> dict[str, dict[str, Any]]:
        """One forward pass keeping the newest details per tracked type.

        Streaming and retaining nothing but the answer dicts: the scan must
        not materialize a child's whole transcript just to find one snapshot.
        """
        customs: dict[str, dict[str, Any]] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = TranscriptEntry.from_json(line)
                    if entry is None or entry.type != ENTRY_CUSTOM:
                        continue
                    custom_type = entry.payload.get("custom_type")
                    if custom_type in _TRACKED_CUSTOM_TYPES:
                        customs[str(custom_type)] = dict(entry.payload.get("details", {}))
        except OSError:
            pass
        return customs


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
