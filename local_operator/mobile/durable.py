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

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    TextContent,
)
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import (
    ENTRY_COMPACTION,
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    ENTRY_PRUNE,
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
    _entry_to_message,
)

logger = logging.getLogger(__name__)

#: Custom-entry types the durable projection reads newest-wins (the store's
#: ``latest_custom`` contract). Tracked in the fold cache so a cached session
#: never re-parses its transcript to answer them. Importing the roster
#: constant pulls in the session module, so the literals are restated here and
#: asserted against the source of truth in the unit tests.
_TRACKED_CUSTOM_TYPES = frozenset({"subagent_roster", "todo_snapshot"})

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
    #: which the cache tests assert against a full re-parse.
    entry_count: int = 0
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
        """Full fold from the file: the pre-cache behaviour, run once per
        session per daemon lifetime (then maintained incrementally)."""
        entries: list[TranscriptEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = TranscriptEntry.from_json(line)
                if entry is not None:
                    entries.append(entry)
        # Newest-wins custom snapshots, backward scan like ``latest_custom``.
        latest_customs: dict[str, dict[str, Any]] = {}
        for entry in reversed(entries):
            if entry.type != ENTRY_CUSTOM:
                continue
            custom_type = entry.payload.get("custom_type")
            if custom_type in _TRACKED_CUSTOM_TYPES and custom_type not in latest_customs:
                latest_customs[str(custom_type)] = dict(entry.payload.get("details", {}))
        state.history = _replay(entries)
        state.render = _fold(state.history)
        state.prunes = {
            str(entry.payload.get("target")): str(entry.payload.get("notice", ""))
            for entry in entries
            if entry.type == ENTRY_PRUNE and entry.payload.get("target")
        }
        state.latest_customs = latest_customs
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
        prefix = _compaction_prefix(compaction)
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


def _compaction_prefix(compaction: TranscriptEntry) -> list[AgentMessage]:
    """The marker summary plus preserved user turns, exactly as
    ``build_llm_history`` injects them (see its docstring for why the turns
    ride the payload verbatim)."""
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
        from local_operator.compaction.cutpoint import PRESERVED_USER_TURN_KEY

        for turn in preserved_turns:
            if not isinstance(turn, dict):
                continue
            message = Message.user(str(turn.get("text", "")))
            turn_id = turn.get("id")
            if isinstance(turn_id, str) and turn_id:
                message.id = turn_id
            message.provider_payload = {PRESERVED_USER_TURN_KEY: True}
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
    prefix = _compaction_prefix(entry)
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
