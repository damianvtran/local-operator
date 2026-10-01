"""The session transcript index: checkpoint manifest and message docs, derived on
the backend from the raw transcript journal.

WHY THIS EXISTS. Two desktop features need per-session structure the journal does
not carry explicitly: the checkpoint rail (a tick per user message and per
completed agent turn, with an outcome) and in-thread find (a per-message search
document). Both need it for rows the reader has never LOADED — the whole point on
long conversations — so it is derived on the backend, once per session, and cached
beside the other derived stores under ``cache/transcript_index/``.

WHAT IT DERIVES, and from what:

- **User checkpoints** — one per ``message`` row whose ``role`` is ``user`` and
  which is the operator's own words: not a harness injection
  (``payload.kind == "custom"`` with a ``custom_type`` — the
  ``_journal_injection_ids`` rule), and not harness chrome (the
  ``provider_payload.harness_injected`` stamp, or a text the shared recognisers
  claim — ``_is_harness_user_row``, the folds' own decision). Steers count;
  they are user messages.
- **Completion checkpoints** — one per turn whose span (its opening user row to
  just before the next user checkpoint) contains at least one message row that is
  not the opening user row. The checkpoint's ``id`` is the span's closing ANSWER
  row — its last assistant message row with non-empty content, the collapse
  branch's ``closingAnswerId`` semantics — falling back to the span's last
  message row only when the turn has no answer at all (QA round 1's Q1 ruling;
  the last-message-row rule it replaces put 43 of 67 completion targets on
  tool/inject rows and made ``[model switch]`` notices the hover text).
- **Outcomes**, from ``completion_attention`` markers bound to their runs by
  token through ``attention_started`` (rules below).
- **Message docs** — one per user/assistant message row and per injected row, in
  journal order (the find slice consumes these; injected rows stay searchable and
  are marked so ranking can demote them). Tool rows are never indexed: they are
  machine output, and indexing them makes every path-like query match everything
  (``session_search``'s rule). Harness chrome rows are skipped whole, for the
  user-checkpoint bullet's reason plus find's own: a hit's reveal jump must land
  on a row a surface paints, and the find wire carries no injected flag to
  demote one by.

THE S3 REFINEMENTS, stated where they live (spike S3, 31 real journals at
origin/main 2026-09-28, approved before implementation):

1. **Markers are per-RUN, and a run is not a turn.** ``attention_started`` lands
   BEFORE its user row (the admission pair is ``[start][user]``), and steer user
   rows land mid-run — one run can contain several user rows. A marker therefore
   resolves the turn of the LAST user checkpoint with ordinal in
   ``[start, marker]``; other markers are hub/wake runs with no user row (441 of
   462 markers in the operator's 272 MB session are ``wake_prompt``/
   ``peer_message`` runs) and correctly resolve no turn.
2. **Tail settlement is an ordering fact, not a bracket test.** The tail turn is
   settled iff a marker resolves it, OR the newest marker is newer than every
   start and sits at/after the tail user row. The bracket form
   (``start <= user <= marker``) fails on steer tails and hub tails, both of
   which the store carries in quantity.
3. **Older sessions predate the mechanism** (the 272 MB journal has no attention
   rows in its first ~17k rows): unattached turns get ``outcome`` omitted and are
   still emitted.
4. **``eligible: false`` markers settle their run with no kind** — the checkpoint
   exists, the outcome is omitted (common: 71 of 83 markers in one sampled
   session).

THE CACHE FILE. ``config_dir()/cache/transcript_index/<session_id>.json``, one
file per session — outside the session directory on purpose (``search_index``'s
reason: a sidecar write moves the session directory mtime, and a listing that
ranks by directory recency would treat an index rebuild as user activity).

Schema (``TRANSCRIPT_INDEX_VERSION`` guards it; a version mismatch discards the
scan sections and preserves ``naming`` when its ``prompt_version`` matches)::

    {
      "version": 1,
      "sig": {"size": <covered bytes>, "mtime": <.9f>, "inode": N, "last_id": "<entry-id>"},
      "coverage": {"first_id", "last_id", "complete"},
      "checkpoints": [{"id","kind","turn","ts","seq","text","outcome"?}],
      "messages":    [{"id","ts","role","text","injected","seq"}],
      "naming": {"prompt_version": 1, "items": {}},
      "scan": {"rows": N, "offset": B, "window": {"offset": B2, "rows": N2}}
    }

FIELDS THAT RIDE BEYOND THE DESIGN DOC'S SKETCH, and the frozen invalidation rule
("size grew and last_id still parses at the recorded tail -> incremental append of
the new tail") is why: an append must re-derive only what can still change, and
that needs resume state the sketch did not enumerate. ``scan`` is the resume
point (rows scanned, byte offset covered, and the "window" row the next append
re-derives from); ``seq`` on message docs is the same ordinal the checkpoints
carry, so the cache can be split at the window without re-reading the prefix.
``sig`` also carries the journal's ``inode``: ``compact_file`` REPLACES the
journal (tmp + ``os.replace``), so a rewrite can keep the recorded size — or
net-grow past it — while no byte-level check at the tail can see it, and an
append never changes the inode. All of it stays inside the version gate: a bump
discards the scan sections.

WHY THE WINDOW IS WHERE IT IS. Everything appended can change only the TAIL
turn's completion checkpoint (its span keeps growing until the next user
checkpoint lands) and the outcomes of runs still open at the old tail. A run's
start row precedes its user row, so re-derivation must begin at or before the
last run start that precedes the last user checkpoint — and the region must
carry that start ROW itself, not just begin before it: ``_scan_incremental``
registers starts only from inside the region, and a marker whose
``attention_started`` sat one row outside it was treated as an orphan, silently
dropping the first re-derived turn's outcome on every append (review round 1,
BLOCKER-1). The window is therefore the first re-derived user's run start when
one exists, else that user row (itself the last user checkpoint at or before the
last start that precedes the last user checkpoint, falling back user -> start ->
marker -> 0). Rows before the window are frozen; ``seq < window.rows``
partitions both arrays.

Text bounds: message ``text`` in full up to :data:`DOC_TEXT_CAP` per doc (rare
prose exceeds it; a match beyond the cap is a stated miss, not a silent one);
checkpoint ``text`` flattened to :data:`CHECKPOINT_TEXT_CAP` (the hover card's
own bound, ~8 lines).

CONSUMERS. The desktop route builds its manifest through :func:`checkpoints_view`.
The naming slice (BE-2) owns ``naming`` generation and writes through
:func:`patch_naming`; the find slice (BE-3) reads :class:`MessageDoc`s. Writes
are atomic (pid-suffixed temp + replace, ``search_index``'s convention) and
best-effort: an unwritable cache costs the speed of the next read, never the
read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from local_operator.session.runtime.engagement import TRANSCRIPT_FILENAME

logger = logging.getLogger(__name__)

#: Bumped when the cache's SHAPE changes. An older file is discarded (scan
#: sections) rather than migrated — it is derived and rebuildable, and a
#: migration path for a cache is code that exists to be wrong.
#:
#: 2: ``MessageDoc`` carries ``custom_type`` (the find filter's discriminator).
#: The bump IS the correctness: a version-1 file would load its peer docs with
#: ``custom_type=None`` — the key is simply absent — and slip them past
#: ``transcript_find``'s hidden-cross-session gate, a silent leak rather than
#: the stale-cache miss a version mismatch is allowed to be.
#:
#: 3: role-user rows that are harness chrome are skipped whole (no checkpoint,
#: no doc — F1 of local-operator-ui#670). The bump IS the correctness again:
#: the frozen prefix re-derives only on a full rescan, so without it every
#: version-2 file would keep serving the rows its scan already minted.
TRANSCRIPT_INDEX_VERSION = 3

#: Per-doc cap on stored message text. A match beyond the cap is a stated miss.
DOC_TEXT_CAP = 32 * 1024

#: Per-checkpoint cap on the display text (the hover card's own bound).
CHECKPOINT_TEXT_CAP = 600

#: Version of the ``naming`` section's prompt/schema; the naming slice owns the
#: values, this module owns the gate (a section written under another version is
#: not served).
NAMING_PROMPT_VERSION = 1

KIND_USER = "user"
KIND_COMPLETION = "completion"

OUTCOME_COMPLETE = "complete"
OUTCOME_ERROR = "error"
OUTCOME_INTERRUPTED = "interrupted"
OUTCOME_OPEN = "open"

_INDEX_DIRNAME = "transcript_index"
_ATTENTION_STARTED = "attention_started"
_COMPLETION_ATTENTION = "completion_attention"

#: Read chunk for the scanner.
_CHUNK_BYTES = 1 << 20

#: Bytes of a row's head kept for id/ts/type classification. The journal row
#: format leads with ``{"id":...``, then ``ts``, then ``type``, then the payload
#: head, so this covers every discriminator the scanner needs (``role``,
#: ``kind``, ``custom_type``); a row whose discriminators fall outside it is
#: kept fully rather than guessed at.
_ROW_HEAD_BYTES = 512

#: A skip-eligible row larger than this stops being materialised: its bytes are
#: read and dropped, only the head is kept. A 30 MB tool result costs one pass
#: and no resident copy; rows we must parse are never dropped.
_MAX_KEPT_LINE_BYTES = 2 << 20

#: Backward scan bound for the append verification (the last scanned row must
#: still end at the recorded tail). Rows larger than this fall back to a full
#: rescan — correct, just slower, and the giant-tail case is rare.
_TAIL_VERIFY_LIMIT = 8 << 20

#: How many parsed sessions stay resident in-process. The manifest is re-read
#: once per poll while a rail is open, and re-parsing a tens-of-MB cache on the
#: event loop is the one cost worth avoiding; four is the design's own bound
#: (the find slice's planning note) and is enforced here for both consumers.
_RESIDENT_SESSIONS = 4

#: How long a failed refresh suppresses further attempts (each poll would
#: otherwise retry a broken journal every 1.5 s and log each time).
_FAILURE_COOLDOWN_S = 60.0

#: Waited for a just-started build before answering "building"; the rail's
#: first paint budget (design D3: the loading state must appear within 200 ms).
_FIRST_PAINT_WAIT_S = 0.2


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MessageDoc:
    """One searchable message row (user/assistant/injected)."""

    id: str
    ts: float
    role: str
    text: str
    injected: bool
    seq: int
    #: The injecting custom row's ``custom_type`` (``"peer_message"``,
    #: ``"hub_message"``, ...); ``None`` on genuine user/assistant rows.
    #: Carried so ``transcript_find`` can drop hidden cross-session docs
    #: (``display.hide_cross_session``) without re-reading the journal. The
    #: scanner already reads this discriminator for its injection rule, so
    #: this is a field copy, not new scanning (design §5.3).
    custom_type: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "ts": self.ts,
            "role": self.role,
            "text": self.text,
            "injected": self.injected,
            "seq": self.seq,
        }
        # Omitted when None, like ``Checkpoint.outcome``: genuine rows pay no
        # bytes for a discriminator they do not carry.
        if self.custom_type is not None:
            payload["custom_type"] = self.custom_type
        return payload

    @staticmethod
    def from_payload(raw: Any) -> "MessageDoc | None":
        if not isinstance(raw, dict):
            return None
        try:
            custom_type = raw.get("custom_type")
            return MessageDoc(
                id=str(raw["id"]),
                ts=float(raw["ts"]),
                role=str(raw["role"]),
                text=str(raw["text"]),
                injected=bool(raw["injected"]),
                seq=int(raw["seq"]),
                custom_type=str(custom_type) if custom_type is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class Checkpoint:
    """One rail checkpoint: the user's row, or a turn's closing row."""

    id: str
    kind: str
    turn: int
    ts: float
    seq: int
    text: str
    outcome: str | None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "turn": self.turn,
            "ts": self.ts,
            "seq": self.seq,
            "text": self.text,
        }
        if self.outcome is not None:
            payload["outcome"] = self.outcome
        return payload

    @staticmethod
    def from_payload(raw: Any) -> "Checkpoint | None":
        if not isinstance(raw, dict):
            return None
        try:
            outcome = raw.get("outcome")
            return Checkpoint(
                id=str(raw["id"]),
                kind=str(raw["kind"]),
                turn=int(raw["turn"]),
                ts=float(raw["ts"]),
                seq=int(raw["seq"]),
                text=str(raw["text"]),
                outcome=str(outcome) if outcome is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class ScanState:
    """Resume point for the incremental-append path (see the module docstring)."""

    rows: int
    offset: int
    window_offset: int
    window_rows: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "offset": self.offset,
            "window": {"offset": self.window_offset, "rows": self.window_rows},
        }

    @staticmethod
    def from_payload(raw: Any) -> "ScanState | None":
        if not isinstance(raw, dict):
            return None
        window = raw.get("window")
        if not isinstance(window, dict):
            return None
        try:
            return ScanState(
                rows=int(raw["rows"]),
                offset=int(raw["offset"]),
                window_offset=int(window["offset"]),
                window_rows=int(window["rows"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class TranscriptIndex:
    """The parsed cache: checkpoints, message docs, and the scan's own facts."""

    checkpoints: list[Checkpoint]
    messages: list[MessageDoc]
    sig: dict[str, Any]
    coverage: dict[str, Any]
    naming: dict[str, Any]
    scan: ScanState
    version: int = TRANSCRIPT_INDEX_VERSION


def _index_dir(config_dir: str | Path) -> Path:
    return Path(config_dir) / "cache" / _INDEX_DIRNAME


def index_path(config_dir: str | Path, session_id: str) -> Path:
    """Where one session's index lives."""
    return _index_dir(config_dir) / f"{session_id}.json"


def _journal_path(config_dir: str | Path, session_id: str) -> Path:
    return Path(config_dir) / "sessions" / session_id / TRANSCRIPT_FILENAME


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _read_raw(config_dir: str | Path, session_id: str) -> dict[str, Any] | None:
    """The cache file's raw document, or ``None`` when absent/unreadable/junk.

    Deliberately version-agnostic: parts of it (``naming``) survive a version
    mismatch, so the reader cannot be the thing that discards it.
    """
    path = index_path(config_dir, session_id)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        loaded = json.loads(raw)
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _index_from_raw(raw: dict[str, Any] | None) -> TranscriptIndex | None:
    """Typed view of a cache document, or ``None`` when it is not usable."""
    if not isinstance(raw, dict) or raw.get("version") != TRANSCRIPT_INDEX_VERSION:
        return None
    scan = ScanState.from_payload(raw.get("scan"))
    if scan is None:
        return None
    sig = raw.get("sig")
    coverage = raw.get("coverage")
    naming = raw.get("naming")
    if not isinstance(sig, dict) or not isinstance(coverage, dict):
        return None
    checkpoints = raw.get("checkpoints")
    messages = raw.get("messages")
    if not isinstance(checkpoints, list) or not isinstance(messages, list):
        return None
    parsed_checkpoints = [c for c in (Checkpoint.from_payload(c) for c in checkpoints) if c]
    parsed_messages = [m for m in (MessageDoc.from_payload(m) for m in messages) if m]
    return TranscriptIndex(
        checkpoints=parsed_checkpoints,
        messages=parsed_messages,
        sig=sig,
        coverage=coverage,
        naming=naming if isinstance(naming, dict) else {},
        scan=scan,
    )


def read_index(config_dir: str | Path, session_id: str) -> TranscriptIndex | None:
    """The cached index as a typed view (no freshness check), else ``None``."""
    return _index_from_raw(_read_raw(config_dir, session_id))


def write_index(config_dir: str | Path, session_id: str, index: TranscriptIndex) -> None:
    """Persist the index atomically (pid-temp + replace), best-effort.

    Atomic because a poll may read while a build writes, and the temp name
    carries the writer's PID for ``search_index``'s measured reason: two
    processes building one session wrote the same fixed temp path and produced
    torn documents. Best-effort because an unwritable cache must cost the speed
    of the next read, never the read itself.
    """
    path = index_path(config_dir, session_id)
    payload = {
        # The LIVE constant, not ``index.version``: a cache must be stamped
        # with the version this build writes, and reading it back compares
        # against the same live constant (see ``_index_from_raw``).
        "version": TRANSCRIPT_INDEX_VERSION,
        "sig": index.sig,
        "coverage": index.coverage,
        "checkpoints": [c.to_payload() for c in index.checkpoints],
        "messages": [m.to_payload() for m in index.messages],
        "naming": index.naming,
        "scan": index.scan.to_payload(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.warning("could not write the transcript index for %s", session_id, exc_info=True)


def _sweep_missing(config_dir: str | Path) -> None:
    """Drop cache files whose session directory is gone (design D8's cleanup
    bullet: "cache files for missing sessions cleaned in the same pass that
    builds").

    Runs on the BUILD pass only — a full scan — because a per-session refresh
    cannot see other sessions, and this is the one moment that can prune. A
    session that is merely closed still has its directory; only a deleted one
    loses its cache. Best-effort, like the write: an unreadable directory must
    never fail a build. ``write_index``'s pid-temps are skipped by suffix.
    """
    try:
        entries = list(_index_dir(config_dir).iterdir())
    except OSError:
        return
    sessions = Path(config_dir) / "sessions"
    for entry in entries:
        if entry.suffix != ".json":
            continue
        try:
            if not (sessions / entry.stem).is_dir():
                entry.unlink()
        except OSError:
            continue


def patch_naming(config_dir: str | Path, session_id: str, items: dict[str, Any]) -> bool:
    """Merge ``items`` into the cache's ``naming`` section.

    The naming slice's write path. A cache that does not exist yet gets a
    minimal document with empty scan sections and the EVALUATED-EMPTY signature
    (size -1), so the next refresh treats it as stale, scans the journal, and
    carries the naming forward (see :func:`preserved_naming`). Returns whether
    the write was attempted.
    """
    raw = _read_raw(config_dir, session_id)
    index = _index_from_raw(raw)
    if index is not None and raw is not None:
        naming = raw.get("naming")
        merged = dict(naming) if isinstance(naming, dict) else {}
        current = merged.get("items")
        merged_items = dict(current) if isinstance(current, dict) else {}
        merged_items.update(items)
        merged["prompt_version"] = NAMING_PROMPT_VERSION
        merged["items"] = merged_items
        index.naming = merged
        write_index(config_dir, session_id, index)
        return True
    if raw is not None:
        # A document exists but is unreadable or another version's: leave it to
        # the refresh, which owns version handling, rather than half-rewriting.
        return False
    empty = TranscriptIndex(
        checkpoints=[],
        messages=[],
        sig={"size": -1, "mtime": -1.0, "last_id": ""},
        coverage={"first_id": "", "last_id": "", "complete": False},
        naming={"prompt_version": NAMING_PROMPT_VERSION, "items": dict(items)},
        scan=ScanState(rows=0, offset=0, window_offset=0, window_rows=0),
    )
    write_index(config_dir, session_id, empty)
    return True


def preserved_naming(raw: dict[str, Any] | None, user_checkpoint_ids: set[str]) -> dict[str, Any]:
    """The ``naming`` section to carry across a (re)scan.

    Kept when its ``prompt_version`` matches this build; items are filtered to
    turn keys that still exist (the opening user entry id — D2's key). Names are
    LLM spend, so a scan never silently drops a still-reachable one; an item
    whose turn is gone is unreachable and goes.
    """
    naming = raw.get("naming") if isinstance(raw, dict) else None
    if not isinstance(naming, dict) or naming.get("prompt_version") != NAMING_PROMPT_VERSION:
        return {"prompt_version": NAMING_PROMPT_VERSION, "items": {}}
    items = naming.get("items")
    if not isinstance(items, dict):
        items = {}
    kept = {
        key: value
        for key, value in items.items()
        if key in user_checkpoint_ids and isinstance(value, dict)
    }
    return {"prompt_version": NAMING_PROMPT_VERSION, "items": kept}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One journal row, classified to exactly what the derivation needs."""

    ordinal: int
    offset: int
    end: int
    id: str
    ts: float
    kind: str  # user | assistant | tool | inject | start | marker | other
    text: str = ""
    token: str = ""
    marker_kind: str | None = None
    eligible: bool = True
    #: ``payload.custom_type`` of an inject row — the same payload field the
    #: injection rule at ``_classify`` already inspects; None on other kinds.
    custom_type: str | None = None


def _head_id(head: bytes) -> str:
    """The entry id from a row's head, if it opens with the canonical shape."""
    if not head.startswith(b'{"id":"'):
        return ""
    end = head.find(b'"', 7)
    if end < 0:
        return ""
    return head[7:end].decode("utf-8", "replace")


def _head_ts(head: bytes) -> float:
    """The entry timestamp from a row's head, 0.0 when it cannot be read."""
    marker = head.find(b'"ts":')
    if marker < 0:
        return 0.0
    rest = head[marker + 5 :]
    end = rest.find(b",")
    if end < 0:
        return 0.0
    try:
        return float(rest[:end])
    except ValueError:
        return 0.0


def _skip_eligible(head: bytes) -> bool:
    """Whether a row may be dropped whole once it is large.

    Only rows whose classification needs nothing beyond the head qualify: tool
    results and non-attention custom rows. A row whose discriminators are not
    visible in the head is kept and parsed — a wrong drop would be silent.
    """
    if b'"type":"message"' in head:
        return b'"role":"tool"' in head and b'"kind":"custom"' not in head
    if b'"type":"custom"' in head:
        return (
            _ATTENTION_STARTED.encode() not in head and _COMPLETION_ATTENTION.encode() not in head
        )
    return b'"type":"compaction"' in head or b'"type":"prune"' in head


def _content_text(payload: dict[str, Any]) -> str:
    """The text parts of a message payload's content, in order."""
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


#: The keys an injected row's displayable text has lived under, in priority
#: order. ``text`` is the mechanism's own key; ``body`` is ``peer_message``'s —
#: a peer's message indexed as "" and was therefore unfindable (review round 1,
#: MAJOR-2); ``summary`` is the compaction marker's. Rows carrying none of them
#: (the gate-timeout notice, notices without a body) index as empty,
#: deliberately: find cannot fabricate text a row does not carry.
_INJECT_TEXT_KEYS = ("text", "body", "summary")


def _inject_text(details: Any) -> str:
    """The first non-empty string among the known injected-text keys."""
    if not isinstance(details, dict):
        return ""
    for key in _INJECT_TEXT_KEYS:
        value = details.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _is_harness_user_row(entry_id: str, payload: dict[str, Any], text: str) -> bool:
    """Whether a role-user message row is harness chrome, not the operator's words.

    The SAME both-legs decision every display fold makes, asked from the one
    implementation (:mod:`local_operator.harness.rows`): the structural stamp
    (``provider_payload.harness_injected``) on rows this build minted, and the
    text recognisers — the chrome prompt families and the legacy notice heads —
    for rows written before the stamp existed. Lazy import, like the folds:
    ``harness.rows`` pulls ``compaction.cutpoint`` on first use.

    The entry id rides IN beside the payload because the notice predicate's
    third leg — the legacy elision-id check — reads it, and a message payload
    carries no id (``encode_message_payload`` excludes it; the id is already
    the entry's). Payload-only, that leg is silently unreachable — which is how
    a pre-#857 ``compaction-elision-<count>`` row (journalled by the previous
    revision; the retained ``PRESERVED_TURN_ELISION_ID_PREFIX`` exists so such
    transcripts still parse) stayed filed as the operator's words (QA round 1,
    Q-1).

    WHY THE INDEX ASKS (F1 of local-operator-ui#670): every index product is a
    human readout — the rail's hover card, find's snippets — and a harness row
    filed as "user" paints the harness's words as the operator's. The UI review
    reproduced it on a row carrying the stamp (this module did not read
    ``provider_payload`` at all), and rows written before the stamp existed leak
    by text on top; both legs close here because both reach the index. Skipped
    WHOLE rather than marked injected: the find wire carries no injected flag (a
    marked doc still arrives as a user-role snippet), and a find hit's reveal
    jump must land on a row that renders — these rows render nowhere. The row
    itself stays in the journal and in the model's context, the folds' exact
    contract.
    """
    from local_operator.harness.rows import is_harness_chrome, is_harness_notice_row

    # A row-shaped view: the predicate reads id, provider_payload and content,
    # and the id lives on the entry rather than in the payload.
    return is_harness_notice_row({**payload, "id": entry_id}) or is_harness_chrome(text)


def _classify(
    ordinal: int, line_start: int, line_end: int, head: bytes, line: bytes | None
) -> _Row:
    """One complete line -> a :class:`_Row` (dropped bodies classify from head)."""
    id_ = _head_id(head)
    ts = _head_ts(head)
    if line is None:
        # Body dropped: only tool/other rows are ever dropped, see _skip_eligible.
        kind = "tool" if b'"role":"tool"' in head else "other"
        return _Row(ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind=kind)
    try:
        entry = json.loads(line)
    except ValueError:
        return _Row(ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="other")
    if not isinstance(entry, dict):
        return _Row(ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="other")
    if not id_:
        id_ = str(entry.get("id", ""))
    if not ts:
        try:
            ts = float(entry.get("ts", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
    etype = entry.get("type")
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    if etype == "message":
        if payload.get("kind") == "custom" and payload.get("custom_type"):
            return _Row(
                ordinal=ordinal,
                offset=line_start,
                end=line_end,
                id=id_,
                ts=ts,
                kind="inject",
                text=_inject_text(payload.get("details")),
                custom_type=str(payload.get("custom_type")),
            )
        role = payload.get("role")
        if role == "user" or role == "assistant":
            text = _content_text(payload)
            if role == "user" and _is_harness_user_row(id_, payload, text):
                # Harness chrome, skipped whole: no user checkpoint (the rail
                # would caption it "Your message"), no find doc, no turn content.
                return _Row(
                    ordinal=ordinal,
                    offset=line_start,
                    end=line_end,
                    id=id_,
                    ts=ts,
                    kind="other",
                )
            return _Row(
                ordinal=ordinal,
                offset=line_start,
                end=line_end,
                id=id_,
                ts=ts,
                kind=str(role),
                text=text,
            )
        if role == "tool":
            return _Row(
                ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="tool"
            )
        return _Row(ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="other")
    if etype == "custom":
        custom_type = payload.get("custom_type")
        if custom_type == _ATTENTION_STARTED:
            details = payload.get("details")
            token = ""
            if isinstance(details, dict):
                token = str(details.get("token", ""))
            return _Row(
                ordinal=ordinal,
                offset=line_start,
                end=line_end,
                id=id_,
                ts=ts,
                kind="start",
                token=token,
            )
        if custom_type == _COMPLETION_ATTENTION:
            details = payload.get("details")
            token = ""
            marker_kind: str | None = None
            eligible = True
            if isinstance(details, dict):
                token = str(details.get("token", ""))
                kind = details.get("kind")
                marker_kind = str(kind) if kind is not None else None
                eligible = bool(details.get("eligible", True))
            return _Row(
                ordinal=ordinal,
                offset=line_start,
                end=line_end,
                id=id_,
                ts=ts,
                kind="marker",
                token=token,
                marker_kind=marker_kind,
                eligible=eligible,
            )
    return _Row(ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="other")


class _RowReader:
    """Streams a journal's complete rows from ``start_offset``.

    Reads in chunks and splits lines without copying a line more than once
    (``b"".join(parts)`` at the boundary), so a single huge row costs one pass;
    skip-eligible giant rows stop being materialised at
    :data:`_MAX_KEPT_LINE_BYTES` and only their head is kept.
    """

    def __init__(self, path: Path, start_offset: int, start_ordinal: int) -> None:
        self._path = path
        self._start_offset = start_offset
        self._next_ordinal = start_ordinal
        #: Complete rows yielded.
        self.rows = 0
        #: Byte offset just past the last complete row.
        self.last_end = start_offset
        #: Entry id of the last complete row (``""`` when it could not be read).
        self.last_id = ""
        #: Entry id of the first complete row.
        self.first_id = ""
        #: True when bytes beyond :attr:`last_end` did not form a complete line.
        self.torn = False
        #: Bytes read (the S1 spike's read-cost reading).
        self.input_bytes = 0
        #: The handle's own stat, taken after the read: what was SCANNED, rather
        #: than what the path says now (a concurrent replace lies about that).
        self.st: os.stat_result | None = None
        #: The handle's stat at open, for the replace detection below.
        self.st_open: os.stat_result | None = None

    def __iter__(self) -> Iterator[_Row]:
        with self._path.open("rb") as handle:
            self.st_open = os.fstat(handle.fileno())
            handle.seek(self._start_offset)
            line_start = self._start_offset
            line_len = 0
            parts: list[bytes] = []
            head = b""
            dropped = False
            while True:
                chunk = handle.read(_CHUNK_BYTES)
                if not chunk:
                    break
                self.input_bytes += len(chunk)
                pos = 0
                while True:
                    newline = chunk.find(b"\n", pos)
                    if newline < 0:
                        piece = chunk[pos:]
                        if piece:
                            if len(head) < _ROW_HEAD_BYTES:
                                head = head + piece[: _ROW_HEAD_BYTES - len(head)]
                            line_len += len(piece)
                            if not dropped:
                                parts.append(piece)
                                if line_len > _MAX_KEPT_LINE_BYTES and _skip_eligible(head):
                                    dropped = True
                                    parts.clear()
                        break
                    piece = chunk[pos:newline]
                    pos = newline + 1
                    if piece:
                        if len(head) < _ROW_HEAD_BYTES:
                            head = head + piece[: _ROW_HEAD_BYTES - len(head)]
                        line_len += len(piece)
                        if not dropped:
                            parts.append(piece)
                            if line_len > _MAX_KEPT_LINE_BYTES and _skip_eligible(head):
                                dropped = True
                                parts.clear()
                    row = _classify(
                        self._next_ordinal,
                        line_start,
                        line_start + line_len + 1,
                        head,
                        None if dropped else b"".join(parts),
                    )
                    self.rows += 1
                    self._next_ordinal += 1
                    self.last_end = row.end
                    self.last_id = row.id
                    if self.rows == 1:
                        self.first_id = row.id
                    yield row
                    line_start = self.last_end
                    line_len = 0
                    parts = []
                    head = b""
                    dropped = False
            if line_len > 0:
                self.torn = True
            self.st = os.fstat(handle.fileno())


class _Replaced(Exception):
    """The journal was atomically replaced while it was being read."""


class _Derivation:
    """Accumulates the scan's raw facts and derives checkpoints and docs."""

    def __init__(self) -> None:
        self.users: list[_Row] = []
        self.user_ords: list[int] = []
        self.messages: list[MessageDoc] = []
        self.starts: dict[str, int] = {}
        self.start_ords: list[int] = []
        self.start_offsets: list[int] = []
        self.markers: list[tuple[int, str, str | None, bool]] = []
        self.content: list[bool] = []
        self.last_row: list[_Row | None] = []
        self.last_text: list[str] = []
        self.answer_row: list[_Row | None] = []
        self.answer_text: list[str] = []
        self._last_start: tuple[int, int] | None = None
        self._last_start_le_user: tuple[int, int] | None = None
        self._last_user: tuple[int, int] | None = None
        self._last_marker: tuple[int, int] | None = None

    def consume(self, row: _Row) -> None:
        kind = row.kind
        if kind == "user":
            self.users.append(row)
            self.user_ords.append(row.ordinal)
            self.content.append(False)
            self.last_row.append(None)
            self.last_text.append("")
            self.answer_row.append(None)
            self.answer_text.append("")
            self._last_user = (row.offset, row.ordinal)
            self._last_start_le_user = self._last_start
            self.messages.append(
                MessageDoc(
                    id=row.id,
                    ts=row.ts,
                    role="user",
                    text=row.text[:DOC_TEXT_CAP],
                    injected=False,
                    seq=row.ordinal,
                )
            )
            return
        if kind == "assistant":
            self.messages.append(
                MessageDoc(
                    id=row.id,
                    ts=row.ts,
                    role="assistant",
                    text=row.text[:DOC_TEXT_CAP],
                    injected=False,
                    seq=row.ordinal,
                )
            )
        elif kind == "inject":
            # Injected rows are conversation INPUTS and stay searchable; the
            # find slice demotes them by ``injected`` and drops the hidden
            # cross-session ones by ``custom_type``. Role is "user" because the
            # wire's role vocabulary is user|agent and these are not the agent.
            self.messages.append(
                MessageDoc(
                    id=row.id,
                    ts=row.ts,
                    role="user",
                    text=row.text[:DOC_TEXT_CAP],
                    injected=True,
                    seq=row.ordinal,
                    custom_type=row.custom_type,
                )
            )
        elif kind == "start":
            self.starts[row.token] = row.ordinal
            self.start_ords.append(row.ordinal)
            self.start_offsets.append(row.offset)
            self._last_start = (row.offset, row.ordinal)
            return
        elif kind == "marker":
            self.markers.append((row.ordinal, row.token, row.marker_kind, row.eligible))
            self._last_marker = (row.offset, row.ordinal)
            return
        # Span content: any message row that is not the opening user row. Rows
        # before the first user checkpoint belong to no span (pre-mechanism
        # history and admission scaffolding both look like this).
        if kind in ("assistant", "tool", "inject") and self.users:
            index = len(self.users) - 1
            self.content[index] = True
            self.last_row[index] = row
            if row.text:
                self.last_text[index] = row.text
            # The completion target rule (QA round 1's Q1 ruling): a turn
            # closes on its last ASSISTANT row with non-empty content — the
            # ``closingAnswerId`` semantics D1 cites — and only falls back to
            # the last message row when the turn has no answer. Tracked here,
            # not at emit time, because the region is re-derived on every
            # incremental refresh.
            if kind == "assistant" and row.text:
                self.answer_row[index] = row
                self.answer_text[index] = row.text

    def window(self) -> tuple[int, int]:
        """The ``(offset, ordinal)`` the next append must re-derive from.

        Backs up from the rewind user to the START of its run when one exists:
        the region must contain the start row itself or the first re-derived
        turn's marker resolves against nothing and its outcome is dropped on
        every append (review round 1, BLOCKER-1).
        """
        anchor = (
            self._last_start_le_user or self._last_user or self._last_start or self._last_marker
        )
        if anchor is None:
            return (0, 0)
        rewind = bisect_right(self.user_ords, anchor[1]) - 1
        if rewind >= 0:
            user = self.users[rewind]
            index = bisect_right(self.start_ords, user.ordinal) - 1
            if index >= 0:
                return (self.start_offsets[index], self.start_ords[index])
            return (user.offset, user.ordinal)
        return anchor

    def emit(
        self,
        *,
        prior_checkpoints: list[Checkpoint],
        prior_messages: list[MessageDoc],
        turn_base: int,
    ) -> tuple[list[Checkpoint], list[MessageDoc]]:
        """Derive this region's checkpoints and docs, merged behind the priors."""
        outcomes: dict[int, str | None] = {}
        for ordinal, token, marker_kind, eligible in self.markers:
            start = self.starts.get(token)
            if start is None:
                continue
            # The LAST user checkpoint in [start, marker]; see S3 note 1.
            index = bisect_right(self.user_ords, ordinal) - 1
            if index < 0 or self.user_ords[index] < start:
                continue
            if marker_kind == "closed":
                # A NEUTRAL CLOSURE (v2 directive, 2026-09-29, session
                # 23fc556c3799) is INERT for the rail: it records that a
                # zero-work follow-up run ended after the previous turn's
                # output was delivered, so it may neither claim the tick
                # (``closed`` is not a rail outcome) nor CLEAR it — the
                # earlier ``complete`` marker for the same turn must stand,
                # which is exactly the masking this fix exists to stop.
                continue
            if marker_kind == "retired":
                # RETIRE-FOR-BUILD (2026-09-29): the rail's vocabulary is
                # frozen (``CheckpointOutcome``), so the kind is normalized to
                # an EXISTING cut value — and ``interrupted`` (CircleSlash,
                # warning) is the honest one: ``error`` is the failure framing
                # this arm exists to remove, while the warning slash is the
                # rail's "cut short" mark and matches the row's own warning
                # tier. Eligibility keeps its usual no-claim rule below.
                marker_kind = OUTCOME_INTERRUPTED
            outcomes[index] = marker_kind if eligible else None

        # Only markers bound to a run by token count as evidence: an orphan
        # marker (no ``attention_started``) cannot say a run ended, and the tail
        # rule below leans on that (S3 note 1's token discipline).
        resolvable = [ordinal for ordinal, token, _, _ in self.markers if token in self.starts]
        last_marker_ordinal = resolvable[-1] if resolvable else -1
        last_start_ordinal = max(self.starts.values(), default=-1)
        tail = len(self.users) - 1

        checkpoints = list(prior_checkpoints)
        for index, user in enumerate(self.users):
            checkpoints.append(
                Checkpoint(
                    id=user.id,
                    kind=KIND_USER,
                    turn=turn_base + index + 1,
                    ts=user.ts,
                    seq=user.ordinal,
                    text=_flatten(user.text, CHECKPOINT_TEXT_CAP),
                    outcome=None,
                )
            )
            if not self.content[index]:
                continue
            # The completion TARGET: the turn's closing answer row, falling
            # back to its last message row only when there is no answer (the
            # Q1 ruling above; ``text`` follows the same row).
            answer = self.answer_row[index]
            closing = answer if answer is not None else self.last_row[index]
            if closing is None:
                continue
            closing_text = self.answer_text[index] if answer is not None else self.last_text[index]
            if index in outcomes:
                outcome = outcomes[index]
            elif index == tail and not (
                last_marker_ordinal > last_start_ordinal and last_marker_ordinal >= user.ordinal
            ):
                # No marker resolves the live tail and no newer settled run
                # followed it: the rail draws an in-progress dot (S3 note 2).
                outcome = OUTCOME_OPEN
            else:
                outcome = None
            checkpoints.append(
                Checkpoint(
                    id=closing.id,
                    kind=KIND_COMPLETION,
                    turn=turn_base + index + 1,
                    ts=closing.ts,
                    seq=closing.ordinal,
                    text=_flatten(closing_text, CHECKPOINT_TEXT_CAP),
                    outcome=outcome,
                )
            )
        return checkpoints, list(prior_messages) + self.messages


def _flatten(text: str, limit: int) -> str:
    """Collapse whitespace and cap — the display-text rule for checkpoints."""
    return " ".join(text.split())[:limit]


def _verify_tail(path: Path, end: int, last_id: str) -> bool:
    """Whether the row ending at ``end`` is still the one recorded.

    The append-only cheap check: the last scanned row must still end exactly at
    the recorded coverage end and still parse to ``last_id``. A failure (or a row
    larger than :data:`_TAIL_VERIFY_LIMIT`) means the file cannot be trusted as a
    prefix of what is there now, and the caller rescans.
    """
    if end <= 0 or not last_id:
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(end - 1)
            if handle.read(1) != b"\n":
                return False
            # Walk back to the previous newline; bounded so a giant tail row
            # degrades to a rescan instead of an unbounded read.
            position = end - 1
            start: int | None = None
            scanned = 0
            while position > 0 and scanned <= _TAIL_VERIFY_LIMIT:
                step = min(position, _CHUNK_BYTES)
                position -= step
                scanned += step
                handle.seek(position)
                window = handle.read(step)
                found = window.rfind(b"\n")
                if found >= 0:
                    start = position + found + 1
                    break
            if start is None:
                start = 0
            if end - 1 - start > _TAIL_VERIFY_LIMIT:
                return False
            handle.seek(start)
            raw = handle.read(end - 1 - start)
    except OSError:
        return False
    try:
        entry = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return False
    return isinstance(entry, dict) and str(entry.get("id", "")) == last_id


def _scan_full(path: Path, naming_raw: dict[str, Any] | None) -> TranscriptIndex:
    """A first read (or a rewrite's forced rescan) of the whole journal."""
    reader = _RowReader(path, 0, 0)
    state = _Derivation()
    for row in reader:
        state.consume(row)
    _assert_same_file(reader)
    checkpoints, messages = state.emit(prior_checkpoints=[], prior_messages=[], turn_base=0)
    naming = preserved_naming(naming_raw, {c.id for c in checkpoints if c.kind == KIND_USER})
    window_offset, window_rows = state.window()
    return TranscriptIndex(
        checkpoints=checkpoints,
        messages=messages,
        sig={
            "size": reader.last_end,
            "mtime": reader.st.st_mtime if reader.st else 0.0,
            "inode": reader.st.st_ino if reader.st else 0,
            "last_id": reader.last_id,
        },
        coverage={
            "first_id": reader.first_id,
            "last_id": reader.last_id,
            "complete": not reader.torn,
        },
        naming=naming,
        scan=ScanState(
            rows=reader.rows,
            offset=reader.last_end,
            window_offset=window_offset,
            window_rows=window_rows,
        ),
    )


def _scan_incremental(
    path: Path, previous: TranscriptIndex, naming_raw: dict[str, Any] | None
) -> TranscriptIndex:
    """Re-derive from the resume window and extend through the appended bytes."""
    window_offset = previous.scan.window_offset
    window_rows = previous.scan.window_rows
    keep_checkpoints = [c for c in previous.checkpoints if c.seq < window_rows]
    keep_messages = [m for m in previous.messages if m.seq < window_rows]
    turn_base = sum(1 for c in keep_checkpoints if c.kind == KIND_USER)
    reader = _RowReader(path, window_offset, window_rows)
    state = _Derivation()
    for row in reader:
        state.consume(row)
    _assert_same_file(reader)
    checkpoints, messages = state.emit(
        prior_checkpoints=keep_checkpoints,
        prior_messages=keep_messages,
        turn_base=turn_base,
    )
    naming = preserved_naming(naming_raw, {c.id for c in checkpoints if c.kind == KIND_USER})
    new_window_offset, new_window_rows = state.window()
    return TranscriptIndex(
        checkpoints=checkpoints,
        messages=messages,
        sig={
            "size": reader.last_end,
            "mtime": reader.st.st_mtime if reader.st else 0.0,
            "inode": reader.st.st_ino if reader.st else 0,
            "last_id": reader.last_id,
        },
        coverage={
            "first_id": previous.coverage.get("first_id", reader.first_id),
            "last_id": reader.last_id,
            "complete": not reader.torn,
        },
        naming=naming,
        scan=ScanState(
            rows=reader.rows,
            offset=reader.last_end,
            window_offset=new_window_offset,
            window_rows=new_window_rows,
        ),
    )


def _assert_same_file(reader: _RowReader) -> None:
    """Refuse a scan whose inode was swapped away mid-read."""
    if reader.st is None or reader.st_open is None:
        return
    if (reader.st.st_ino, reader.st.st_dev) != (reader.st_open.st_ino, reader.st_open.st_dev):
        raise _Replaced()


def _sig_matches(sig: dict[str, Any], st: os.stat_result) -> bool:
    """Is the cached scan exactly current for this stat?

    The inode is part of the comparison on purpose: ``compact_file`` REPLACES
    the journal (tmp + ``os.replace``), so a rewrite can preserve the size
    and can even make the file LONGER than the recorded scan (folding a
    one-byte tool body into a long notice, with the prune row itself dropped);
    a size+mtime test alone then happily pairs old ordinals with new bytes.
    An append never changes the inode, so requiring it costs nothing there.
    """
    return (
        sig.get("size") == st.st_size
        and sig.get("mtime") == st.st_mtime
        and sig.get("inode") == st.st_ino
    )


def refresh_index(config_dir: str | Path, session_id: str) -> TranscriptIndex | None:
    """Bring one session's cache up to date; ``None`` when there is no journal.

    The whole invalidation ladder, in order: an exact (size, mtime, inode)
    match reuses; a GROWN file from the SAME inode whose recorded tail still
    verifies increments; anything else — including a compact rewrite, whose
    ``os.replace`` changes the inode — rescans. One retry covers a concurrent
    ``compact_file`` (which replaces the file, changing the inode under the
    read).
    """
    path = _journal_path(config_dir, session_id)
    for attempt in (0, 1):
        try:
            st = path.stat()
        except OSError:
            return None
        raw = _read_raw(config_dir, session_id)
        previous = _index_from_raw(raw)
        if previous is not None and _sig_matches(previous.sig, st):
            return previous
        try:
            if (
                previous is not None
                and st.st_ino == previous.sig.get("inode")
                and st.st_size > previous.scan.offset
                and _verify_tail(path, previous.scan.offset, str(previous.sig.get("last_id", "")))
            ):
                index = _scan_incremental(path, previous, raw)
            else:
                index = _scan_full(path, raw)
                # D8's cleanup bullet rides the build pass, and a build is the
                # only moment that can see the whole cache directory.
                _sweep_missing(config_dir)
        except _Replaced:
            if attempt:
                raise
            continue
        # A naming write can land WHILE this scan runs, and that pairing is by
        # design rather than by accident: the daemon refreshes a stale cache
        # for the manifest route while the session's owner writes a freshly
        # bought name through ``patch_naming`` — a DIFFERENT process, so no
        # process-local lock can span them. The scan carried its section from
        # the document read before it ran, and the whole-document write below
        # would silently drop anything that landed since: the rail then reads
        # pending, the spend is gone, and nothing retries (agent review round
        # 1, MAJOR-1, reproduced exactly here). Merging the on-disk section one
        # last time — the same read-merge-write ``patch_naming`` performs, from
        # the other side — shrinks that window from the scan's whole duration
        # to this copy. Per-key the disk wins: an item there is the newer write
        # for its turn key, while keys only the scan carries (version-bump
        # preservation) stay put.
        fresh_section = preserved_naming(
            _read_raw(config_dir, session_id),
            {c.id for c in index.checkpoints if c.kind == KIND_USER},
        )
        if fresh_section["items"]:
            carried = index.naming.get("items") if isinstance(index.naming, dict) else None
            merged = dict(carried) if isinstance(carried, dict) else {}
            merged.update(fresh_section["items"])
            index.naming = {"prompt_version": NAMING_PROMPT_VERSION, "items": merged}
        write_index(config_dir, session_id, index)
        return index
    return None


# ---------------------------------------------------------------------------
# Probe and orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexProbe:
    """What the cheap read tells the caller before any scan is attempted."""

    state: str  # "missing" | "ready" | "stale"
    index: TranscriptIndex | None
    built_at: float | None


def probe_index(config_dir: str | Path, session_id: str) -> IndexProbe:
    """Stat the journal and read the cache; never scans."""
    path = _journal_path(config_dir, session_id)
    try:
        st = path.stat()
    except OSError:
        return IndexProbe(state="missing", index=None, built_at=None)
    raw = _read_raw(config_dir, session_id)
    index = _index_from_raw(raw)
    cache_path = index_path(config_dir, session_id)
    try:
        built_at = cache_path.stat().st_mtime
    except OSError:
        built_at = None
    if index is not None and _sig_matches(index.sig, st):
        return IndexProbe(state="ready", index=index, built_at=built_at)
    return IndexProbe(state="stale", index=index, built_at=built_at)


#: In-flight refreshes and the resident parsed indexes, keyed by
#: ``(str(config_dir), session_id)``. LOOP-ONLY MUTATION: entries are created,
#: read and dropped from the event loop thread; the worker only computes and
#: returns (``page_cache``'s cross-thread rule). Each entry carries the loop it
#: belongs to so a torn-down test loop reads as absent rather than as a future
#: bound to a dead loop.
_IN_FLIGHT: dict[tuple[str, str], tuple[asyncio.AbstractEventLoop, "asyncio.Task[Any]"]] = {}
_RESIDENT: "OrderedDict[tuple[str, str], TranscriptIndex]" = OrderedDict()

#: The cache file's mtime each resident index was remembered under. The fast
#: path revalidates the JOURNAL's stat; a naming write (``patch_naming``) moves
#: the CACHE file and leaves the journal alone, so without this stamp a resident
#: index keeps answering ``pending`` for as long as the journal stays quiet
#: (BE-2's isolated run measured 87 s of exactly that). A moved stamp falls
#: through to the disk read, which is where the write landed.
_RESIDENT_STAMP: dict[tuple[str, str], float] = {}
_FAILURES: dict[tuple[str, str], float] = {}


def _key(config_dir: str | Path, session_id: str) -> tuple[str, str]:
    return (str(config_dir), session_id)


def _remember(key: tuple[str, str], index: TranscriptIndex, cache_mtime: float | None) -> None:
    """Keep a just-built index resident (loop thread only), with the cache
    file's mtime it must still match to be served from here."""
    _RESIDENT[key] = index
    _RESIDENT.move_to_end(key)
    if cache_mtime is None:
        _RESIDENT_STAMP.pop(key, None)
    else:
        _RESIDENT_STAMP[key] = cache_mtime
    while len(_RESIDENT) > _RESIDENT_SESSIONS:
        old_key, _entry = _RESIDENT.popitem(last=False)
        _RESIDENT_STAMP.pop(old_key, None)


def resident(config_dir: str | Path, session_id: str) -> TranscriptIndex | None:
    """The resident parsed index for a session, when one is held."""
    entry = _RESIDENT.get(_key(config_dir, session_id))
    if entry is not None:
        _RESIDENT.move_to_end(_key(config_dir, session_id))
    return entry


def start_refresh(config_dir: str | Path, session_id: str) -> "tuple[asyncio.Task[Any], bool]":
    """Start (or join) the background refresh for one session.

    Single-flight per session, with the strong reference asyncio tasks need —
    a bare ``create_task`` has only a weak referent and would be collected
    mid-flight (the pattern serving.py uses for the same reason). Callers in
    another event loop than a recorded entry get a fresh task; the stale entry
    is replaced.

    Returns ``(task, started)``: ``started`` is True only when THIS call created
    the task, False when it joined one already in flight. The distinction is
    :func:`checkpoints_view`'s to spend — only the call that STARTS a build
    owes the first-paint wait for it (see the comment there for the measured
    cost of re-paying it on every poll).
    """
    key = _key(config_dir, session_id)
    loop = asyncio.get_running_loop()
    entry = _IN_FLIGHT.get(key)
    if entry is not None:
        entry_loop, task = entry
        if entry_loop is loop and not task.done():
            return task, False
        if entry_loop is loop:
            _IN_FLIGHT.pop(key, None)

    async def _run() -> TranscriptIndex | None:
        return await asyncio.to_thread(refresh_index, config_dir, session_id)

    async def _wrapped() -> TranscriptIndex | None:
        try:
            index = await _run()
        except Exception:
            _FAILURES[key] = time.monotonic()
            logger.warning(
                "transcript index refresh failed for session %s; serving the previous state",
                session_id,
                exc_info=True,
            )
            raise
        _FAILURES.pop(key, None)
        if index is not None:
            cache_mtime = _mtime_or_none(
                await asyncio.to_thread(_stat_or_none, index_path(config_dir, session_id))
            )
            _remember(key, index, cache_mtime)
        return index

    task = loop.create_task(_wrapped(), name=f"transcript-index:{session_id}")
    _IN_FLIGHT[key] = (loop, task)

    def _drop(settled: "asyncio.Task[Any]") -> None:
        current = _IN_FLIGHT.get(key)
        if current is not None and current[1] is settled:
            _IN_FLIGHT.pop(key, None)

    task.add_done_callback(_drop)
    return task, True


def _manifest_state(
    session_id: str, state: str, index: TranscriptIndex | None, built_at: float | None
) -> dict[str, Any]:
    """The D9 wire shape for the checkpoints manifest."""
    checkpoints: list[dict[str, Any]] = []
    if index is not None:
        # The naming slice owns the STATE semantics for its section; this
        # module only serves them. Function-local because the dependency runs
        # the other way at module scope (``checkpoint_naming`` imports this
        # module for the cache API, and a top-level import here would be a
        # cycle). The derive covers all three states: a named item is
        # ``ready``, an item inside ``NAMING_UNAVAILABLE_COOLDOWN_S`` of its
        # ``failed_ts`` is ``unavailable`` (warm writes that marker on a
        # failed call), and anything else is ``pending``.
        from local_operator.session import checkpoint_naming

        naming = index.naming if isinstance(index.naming, dict) else {}
        items = naming.get("items") if naming.get("prompt_version") == NAMING_PROMPT_VERSION else {}
        items = items if isinstance(items, dict) else {}
        user_id_for_turn = {c.turn: c.id for c in index.checkpoints if c.kind == KIND_USER}
        for checkpoint in index.checkpoints:
            entry: dict[str, Any] = {
                "id": checkpoint.id,
                "kind": checkpoint.kind,
                "turn": checkpoint.turn,
                "ts": checkpoint.ts,
                "seq": checkpoint.seq,
                "text": checkpoint.text,
            }
            if checkpoint.outcome is not None:
                entry["outcome"] = checkpoint.outcome
            if checkpoint.kind == KIND_COMPLETION:
                turn_key = user_id_for_turn.get(checkpoint.turn)
                item = items.get(turn_key) if turn_key else None
                item_state = checkpoint_naming.naming_state(item)
                if item_state == "ready" and isinstance(item, dict):
                    entry["naming"] = {
                        "state": "ready",
                        "name": item.get("name"),
                        "summary": item.get("summary") or "",
                    }
                else:
                    entry["naming"] = {
                        "state": item_state,
                        "name": None,
                        "summary": None,
                    }
            checkpoints.append(entry)
    payload: dict[str, Any] = {"session_id": session_id, "index": {"state": state}}
    if built_at is not None:
        payload["index"]["built_at"] = built_at
    payload["checkpoints"] = checkpoints
    return payload


async def checkpoints_view(
    config_dir: str | Path, session_id: str, *, wait_s: float = _FIRST_PAINT_WAIT_S
) -> dict[str, Any]:
    """The ``sessions.checkpoints`` manifest for one local session.

    Fast path: the resident index, revalidated against one stat. Else the disk
    cache decides; a stale/missing cache starts a background build and awaits it
    only for the first-paint budget, answering ``building`` (with whatever the
    previous scan has) when it is still running. Only the poll that STARTS a
    build waits that budget; a poll that JOINS one answers ``building``
    immediately instead of re-awaiting the same build. A failure inside the
    cooldown answers ``error`` without hammering a broken journal.
    """
    key = _key(config_dir, session_id)
    resident_index = _RESIDENT.get(key)
    if resident_index is not None:
        st, cache_st = await asyncio.to_thread(_freshness_pair, config_dir, session_id)
        stamp = _RESIDENT_STAMP.get(key)
        cache_stable = (
            # No cache file at all: an unwritable root never held one, so
            # nothing on disk could have moved — serve the resident rather
            # than rescanning on every poll (measured 1,2,3 scans across
            # successive calls before this clause; 1,1,1 with it, and the
            # module's promise is that an unwritable cache costs the SPEED of
            # the next read, never the read).
            (stamp is None and cache_st is None)
            # A file exists: it may only be served from when it is the same
            # one the entry was remembered under (patch_naming rewrites it).
            or (stamp is not None and cache_st is not None and stamp == cache_st.st_mtime)
        )
        if st is not None and _sig_matches(resident_index.sig, st) and cache_stable:
            _RESIDENT.move_to_end(key)
            return _manifest_state(session_id, "ready", resident_index, _mtime_or_none(cache_st))
    probe = await asyncio.to_thread(probe_index, config_dir, session_id)
    if probe.state == "missing":
        # No journal: a draft or a session with nothing written yet. An empty
        # manifest is the honest answer, matching history's empty page.
        return _manifest_state(session_id, "ready", None, None)
    if probe.state == "ready":
        if probe.index is not None:
            _remember(key, probe.index, probe.built_at)
        return _manifest_state(session_id, "ready", probe.index, probe.built_at)

    failed_at = _FAILURES.get(key)
    if failed_at is not None and (time.monotonic() - failed_at) < _FAILURE_COOLDOWN_S:
        return _manifest_state(session_id, "error", probe.index, probe.built_at)
    task, started = start_refresh(config_dir, session_id)
    if started:
        # FIRST PAINT: only the call that STARTS the build owes this wait — the
        # rail's loading state must be able to appear within the budget (design
        # D3) even while the build runs on.
        done, _pending = await asyncio.wait({task}, timeout=wait_s)
        settled = task in done
    else:
        # JOINED an in-flight build: the starting call already owns the
        # first-paint wait for it, so re-awaiting it here would charge EVERY
        # poll the full budget for the same build (measured: 6 polls over one
        # 3.0 s bda7 scan, ~220 ms each). Answer from the task's own state
        # instead — near-zero wait — and let the shared path below serve a
        # build that settled since this poll joined it.
        settled = task.done()
    if not settled:
        return _manifest_state(session_id, "building", probe.index, probe.built_at)
    error = task.exception()
    if error is not None:
        return _manifest_state(session_id, "error", probe.index, probe.built_at)
    built = task.result()
    if built is not None:
        cache_st = await asyncio.to_thread(_stat_or_none, index_path(config_dir, session_id))
        return _manifest_state(session_id, "ready", built, _mtime_or_none(cache_st))
    return _manifest_state(session_id, "ready", None, None)


def _stat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def _freshness_pair(
    config_dir: str | Path, session_id: str
) -> tuple[os.stat_result | None, os.stat_result | None]:
    """The journal and cache stats the resident ready path needs, ONE hop."""
    return (
        _stat_or_none(_journal_path(config_dir, session_id)),
        _stat_or_none(index_path(config_dir, session_id)),
    )


def _mtime_or_none(st: os.stat_result | None) -> float | None:
    return st.st_mtime if st is not None else None


def _reset_for_tests() -> None:
    """Drop the module's loop state (test isolation; never called in production).

    The in-process caches are keyed by config root, which tests frequently
    re-create under new ``tmp_path``s — but the LRU keeps up to
    :data:`_RESIDENT_SESSIONS` entries alive across them, so a suite must be
    able to start clean.
    """
    _IN_FLIGHT.clear()
    _RESIDENT.clear()
    _RESIDENT_STAMP.clear()
    _FAILURES.clear()
