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
import contextvars
import json
import logging
import os
import threading
import time
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator
from uuid import uuid4

from local_operator.session import open_trace
from local_operator.session.runtime.engagement import TRANSCRIPT_FILENAME

if TYPE_CHECKING:
    from local_operator.session.index_read_model import FrameSelection
    from local_operator.session.transcript import TranscriptPage

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
#: How often the row reader yields the GIL while scanning (see the note there).
_SCAN_YIELD_ROWS = 512

#: The byte marker that says a row is a ``send`` result, for the one case where
#: the name cannot be read from a parsed payload: a row so large its body was
#: dropped (see ``_carries_send_marker``).
_SEND_MARKER = b'"tool_name":"send"'

#: The tool name whose rows the desktop's cross-session filter hides
#: (``cross-session-visibility.ts::visibleRecords``: ``kind === "tool" &&
#: toolName === "send"``, gated on ``display.hide_cross_session``). One spelling
#: here because the per-run split has to agree with that filter exactly.
#: NOTE FOR THE NEXT CONSUMER (review round 2, F12): a hidden ``send`` row's
#: FAILURE stays inside ``failed_count`` — there is no cross-session split of
#: failures, because the client's own filter does not split them either. A surface
#: that wants a hidden-failure count has to ask for one.
CROSS_SESSION_TOOL_NAME = "send"

#: 3: role-user rows that are harness chrome are skipped whole (no checkpoint,
#: no doc — F1 of local-operator-ui#670). The bump IS the correctness again:
#: the frozen prefix re-derives only on a full rescan, so without it every
#: version-2 file would keep serving the rows its scan already minted.
#:
#: 4: the ``runs`` section (the open frame's per-run facts). The bump is the
#: same correctness in a different direction: a version-3 file has no ``runs``
#: key at all, and serving it to the open frame would read as "this session has
#: no runs" — the absence the wire uses for "no facts available" — while the
#: rows to derive them from are all still there. Only a rescan can mint them.
#: 5: all-row journal ordinals and their equality domain. Earlier caches only
#: cover the search/rail subsets; they cannot prove a visible custom row's
#: position. Rebuilding on version mismatch is preferable to fabricating it.
#: 8: binding notice rows no longer contribute display/search documents.
#: Version-7 caches may already hold those now-invisible hits in their frozen
#: prefix. Even an unchanged journal or an incremental append must rederive
#: them under the new visibility rule, not offer jumps to hidden notices.
TRANSCRIPT_INDEX_VERSION = 8

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

#: The marker kinds the desktop paints as a COMPLETED turn
#: (``durableRecord``'s four arms; ``closed``/``retired`` are the neutral-closure
#: and retire-for-build receipts, both of which still say ``complete: true``).
_TERMINAL_MARKER_KINDS = frozenset({"closed", "retired", "error", "interrupted"})

#: The error-level custom the renderer paints from a row of kind ``custom``: the
#: second arm of its terminal vocabulary (``boundaryKindOf``).
_SESSION_INCIDENT = "session_incident"

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

#: First-paint budget (design D3: the loading state must appear within 200 ms).
_FIRST_PAINT_WAIT_S = 0.2

#: What that budget is allowed to spend itself on: a journal scan rate the wait
#: is planned against, so a build that cannot POSSIBLY finish inside the budget
#: does not block a COLD open at all. Measured on this host with
#: ``refresh_index`` over a 20,014-entry / 24,902,439-byte journal: 943.5 ms, so
#: ~26 MB/s while the fleet was loaded — the constant is that measurement, not a
#: fraction of it, and the 200 ms budget therefore covers ~4.8 MB.
#:
#: WHY NOT A CONSERVATIVE FRACTION (agent review round 1, MAJOR-1). The two
#: ways this constant can be wrong are not symmetric. Set too high, the wait is
#: spent on a build that misses the budget: the caller pays 200 ms and gets the
#: same ``building`` it would have got without waiting — which is exactly the
#: pre-0.89 behaviour, so nothing regresses. Set too low, a build that WOULD
#: have settled inside the budget is refused, and a ``building`` answer is not
#: re-read by the current consumer (the rail's poll is armed only while naming
#: ids are pending), so the rail keeps the previous tick set for the whole
#: visit. A first revision used 8 MB/s — a third of the measurement — and QA
#: caught that regression on a 1.9 MB append (round 1, Q1).
#:
#: WHY THIS EXISTS (an UPPER-BOUND measurement, 2026-10-24): a cold open of a
#: 20k-entry conversation spent 819.8 ms in ``GET .../checkpoints`` on a host at
#: ~92% swap, answering ``building`` with an empty manifest. In-process, with
#: the same journal and nothing else running, that path measured 204.0 ms and
#: 213.8 ms on two runs — ~200 ms of it this budget, paid on an open whose
#: answer could never have been ``ready`` (the same journal scans in 943.5 ms).
#: The spread is this host, not the code. The wait is only worth paying when the
#: scan can land inside it.
_FIRST_PAINT_SCAN_BYTES_PER_S = 24 << 20


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
class RunRecord:
    """One RUN of a conversation: the rows between one opening and the next.

    THE UNIT EVERY SURFACE RE-DERIVES AND NONE OF THEM AGREES ON. The desktop
    partitions its loaded rows (``transcript-rows.ts::walkTurns``), the native
    app runs its own condenser over a different window, the relay web folds a
    third, and the TUI keeps a fourth reading — so a bar's action count and its
    "worked for" are computed from whatever rows happen to be loaded, which is
    why a turn heard the operator report ``30 actions`` at open and ``423
    actions`` after thirteen pages. This record is the ONE derivation, from the
    whole journal, on the backend.

    THE PARTITION RULE IS THE DESKTOP'S, deliberately (``walkTurns``), because
    the facts are consumed by a bar built from that partition: a user row opens
    a run when no run is open, when the open run's last painting row is a
    settled assistant row, when a TERMINAL marker has been seen since the run
    opened (a completion the client paints as ``complete``, or a
    ``session_incident``), or when the open run is an empty preamble that has
    done no work. Otherwise the user row is a STEER and stays inside the run —
    the client's own rule, and the reason a steered run's count is its whole
    count rather than its last turn's.

    ``attention_started`` rows do NOT delimit runs here even though they do in
    the store: the desktop never renders one (``durableRecord`` drops every
    ``custom`` that is not ``completion_attention``), so a partition built from
    them would disagree with the client about which rows belong to which run.

    COUNTS ARE EXACT, and the one case that could make them a lower bound says
    so: ``complete`` is False only when the scanner dropped a row body inside
    the run (a row past :data:`_MAX_KEPT_LINE_BYTES`), where a failure or a
    duration may be unknowable. Measured on this machine: 0 of 65,755 tool rows
    across the twelve largest journals exceed that limit.
    """

    #: The run's opening USER row, or ``""`` for a run whose head is cut off
    #: (a wake/hub run with no user row, or pre-mechanism history).
    opening_user_id: str
    #: The row the client elects as this run's closing answer (the completion
    #: checkpoint's own rule), or ``""`` when the run has no answer.
    closing_answer_id: str
    #: The run's last row, whatever it is.
    last_id: str
    first_seq: int
    last_seq: int
    start_ts: float
    end_ts: float
    action_count: int
    failed_count: int
    worked_seconds: float
    #: True when the run is finished — the facts of a live tail would be
    #: corrected by the next row, and the wire refuses to state them (see
    #: ``open_frame``).
    settled: bool
    outcome: str | None
    complete: bool
    #: INTERNAL BOOKKEEPING, carried in the cache so an incremental scan can
    #: resume a run that straddles its window (see ``_scan_incremental``). The
    #: wire never sees these: ``open_frame.publish_runs`` builds its payloads
    #: field by field, which is also what keeps a future internal addition from
    #: silently becoming a client contract.
    last_painter: str | None = None
    saw_terminal: bool = False
    saw_work: bool = False
    opened_with_user_row: bool = True
    #: The ordinal of the run's last WORK row (an assistant or tool row). The
    #: tail run's settlement test needs it (F4): a marker older than the last work
    #: row has been overtaken by work the run is still accumulating.
    last_work_seq: int = -1
    #: THE CLIENT'S OWN KEY for this run's closing record, in the vocabulary
    #: ``walkTurns`` keys records with (``transcript-reducer.ts``: a tool result is
    #: ``tool:<call_id>``, a completion-marker notice is ``prov-<token>``, anything
    #: else is its entry id). It is what a client matches a fact against, so the
    #: wire's ``run_key`` is this and not an entry id: a head-cut INTERRUPTED run
    #: keyed by its marker's entry id never matched, and the bar showed the
    #: placeholder (measured: "50+" for a 150-action run).
    key_id: str = ""
    #: Whether ANY row of the run reported a duration. The client states ``null``
    #: rather than ``0s`` for a run that reported none (``workedSeconds`` in
    #: ``trace-fold-model.ts``), so the wire must too.
    worked_any: bool = False
    #: The subset of this run's counted calls the desktop HIDES when
    #: ``display.hide_cross_session`` is on: ``send`` tool rows, the arm of its own
    #: filter (``cross-session-visibility.ts::visibleRecords``: ``kind === "tool"
    #: && toolName === "send"``). Additive: a client that hides them subtracts
    #: these from the counts so its bar matches its own fold.
    cross_actions: int = 0
    cross_worked: float = 0.0
    #: The answer THIS REGION saw for the run (the index's per-turn rule). Kept
    #: beside ``closing_answer_id`` rather than replacing it: the record's own
    #: field is what other readers see, and this is what the emit pass consults
    #: first, because a carried run must not keep the answer it had before the
    #: append that completed it.
    answer_id: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "opening_user_id": self.opening_user_id,
            "closing_answer_id": self.closing_answer_id,
            "last_id": self.last_id,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "action_count": self.action_count,
            "failed_count": self.failed_count,
            "worked_seconds": self.worked_seconds,
            "settled": self.settled,
            "complete": self.complete,
            "answer_id": self.answer_id,
            "key_id": self.key_id,
            "worked_any": self.worked_any,
            "cross_actions": self.cross_actions,
            "cross_worked": self.cross_worked,
            "last_work_seq": self.last_work_seq,
            "last_painter": self.last_painter,
            "saw_terminal": self.saw_terminal,
            "saw_work": self.saw_work,
            "opened_with_user_row": self.opened_with_user_row,
        }
        if self.outcome is not None:
            payload["outcome"] = self.outcome
        return payload

    @staticmethod
    def from_payload(raw: Any) -> "RunRecord | None":
        if not isinstance(raw, dict):
            return None
        try:
            outcome = raw.get("outcome")
            return RunRecord(
                opening_user_id=str(raw["opening_user_id"]),
                closing_answer_id=str(raw["closing_answer_id"]),
                last_id=str(raw["last_id"]),
                first_seq=int(raw["first_seq"]),
                last_seq=int(raw["last_seq"]),
                start_ts=float(raw["start_ts"]),
                end_ts=float(raw["end_ts"]),
                action_count=int(raw["action_count"]),
                failed_count=int(raw["failed_count"]),
                worked_seconds=float(raw["worked_seconds"]),
                settled=bool(raw["settled"]),
                complete=bool(raw["complete"]),
                outcome=str(outcome) if outcome is not None else None,
                answer_id=str(raw.get("answer_id", "")),
                key_id=str(raw.get("key_id", "")),
                worked_any=bool(raw.get("worked_any", False)),
                cross_actions=int(raw.get("cross_actions", 0)),
                cross_worked=float(raw.get("cross_worked", 0.0)),
                last_work_seq=int(raw.get("last_work_seq", -1)),
                last_painter=(
                    str(raw["last_painter"]) if raw.get("last_painter") is not None else None
                ),
                saw_terminal=bool(raw.get("saw_terminal", False)),
                saw_work=bool(raw.get("saw_work", False)),
                opened_with_user_row=bool(raw.get("opened_with_user_row", True)),
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
class SourceProjectionViews:
    """Body-free inverse views, rebuilt ONCE with this same index, never per read."""

    by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    duplicates: set[str] = field(default_factory=set)
    prunes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    owners: dict[str, list[str]] = field(default_factory=dict)
    results: dict[str, list[str]] = field(default_factory=dict)
    compactions: list[dict[str, Any]] = field(default_factory=list)
    compaction_seqs: list[int] = field(default_factory=list)
    prune_seqs: dict[str, list[int]] = field(default_factory=dict)
    preserved_sets: dict[int, frozenset[str]] = field(default_factory=dict)
    opaque_causal_seqs: list[int] = field(default_factory=list)
    global_unknown_seqs: list[int] = field(default_factory=list)
    unknown_identity_seqs: list[int] = field(default_factory=list)
    origins_complete: bool = False

    @classmethod
    def build(cls, facts: list[dict[str, Any]]) -> "SourceProjectionViews":
        out = cls()
        for fact in facts:
            id_, seq = fact["id"], fact["seq"]
            if id_ in out.by_id:
                out.duplicates.add(id_)
            out.by_id[id_] = fact
            if not id_:
                out.unknown_identity_seqs.append(seq)
            flags = fact.get("flags", [])
            if "global_unknown" in flags:
                out.global_unknown_seqs.append(seq)
            if "opaque_causal" in flags:
                out.opaque_causal_seqs.append(seq)
            for call in fact.get("call_ids", []):
                out.owners.setdefault(call, []).append(id_)
            if fact.get("result_call_id"):
                out.results.setdefault(fact["result_call_id"], []).append(id_)
            if fact["source_class"] == "prune" and fact.get("prune_target"):
                target = fact["prune_target"]
                out.prunes.setdefault(target, []).append(fact)
                out.prune_seqs.setdefault(target, []).append(seq)
            if fact["source_class"] == "compaction":
                out.compactions.append(fact)
                out.compaction_seqs.append(seq)
                out.preserved_sets[seq] = frozenset(fact.get("preserved_candidate_ids", []))
        return out


def source_projection_fact(entry: Any, seq: int) -> dict[str, Any]:
    """Extract IDs/flags before rail classification drops semantic information.

    This receives the scanner's ALREADY parsed JSON. No reparse, arguments,
    prose, prune notices or preserved bodies are retained. Incomplete semantic
    shape fails closed; opaque tool bodies have causal rather than global scope.
    """
    from local_operator.harness.rows import (
        is_ask_gate_divert_row,
        is_hidden_tool_name,
        is_hidden_wake_delivery,
        is_settle_only_ask,
    )

    fact: dict[str, Any] = {
        "id": "",
        "seq": seq,
        "source_class": "invalid",
        "flags": ["global_unknown"],
    }
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
        return fact
    fact["id"] = entry["id"]
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return fact
    etype = entry.get("type")
    fact["flags"] = []
    if etype == "custom":
        fact["source_class"] = "bookkeeping"
    elif etype == "prune":
        fact["source_class"] = "prune"
        target = payload.get("target")
        if isinstance(target, str) and 1 <= len(target) <= 128:
            fact["prune_target"] = target
        else:
            fact["flags"] = ["global_unknown"]
    elif etype == "compaction":
        fact["source_class"] = "compaction"
        kept = payload.get("first_kept_entry_id")
        if kept is not None and (not isinstance(kept, str) or not 1 <= len(kept) <= 128):
            fact["flags"].append("incomplete")
        else:
            fact["first_kept_entry_id"] = kept
        preserved = payload.get("preserved_user_turns", [])
        if (
            not isinstance(preserved, list)
            or len(preserved) > 512
            or any(
                not isinstance(turn, dict)
                or not isinstance(turn.get("id"), str)
                or not 1 <= len(turn["id"]) <= 128
                for turn in preserved
            )
        ):
            fact["flags"].append("incomplete")
        else:
            fact["preserved_candidate_ids"] = [turn["id"] for turn in preserved]
    elif etype == "message":
        kind = payload.get("kind", "message")
        if kind not in ("message", "custom"):
            fact["flags"] = ["global_unknown"]
            return fact
        fact["source_class"] = "custom_message" if kind == "custom" else "plain_message"
        if kind == "custom":
            if payload.get("custom_type"):
                fact["flags"].append("injection")
            else:
                fact["flags"].append("incomplete")
        else:
            role = payload.get("role")
            if role not in ("user", "assistant", "tool"):
                fact["flags"].append("incomplete")
            calls = payload.get("tool_calls", [])
            if (
                not isinstance(calls, list)
                or len(calls) > 512
                or any(
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not 1 <= len(call["id"]) <= 128
                    for call in calls
                )
            ):
                fact["flags"].append("opaque_causal")
            elif calls and role != "assistant":
                fact["flags"].append("opaque_causal")
            elif calls:
                fact["call_ids"] = [call["id"] for call in calls]
                if any(is_settle_only_ask(call.get("name"), queued_engine=True) for call in calls):
                    fact["flags"].append("ask")
                if any(is_hidden_tool_name(call.get("name")) for call in calls):
                    fact["flags"].append("hidden")
            if role == "tool":
                call = payload.get("tool_call_id")
                if isinstance(call, str) and 1 <= len(call) <= 128:
                    fact["result_call_id"] = call
                else:
                    fact["flags"].append("opaque_causal")
                if is_settle_only_ask(payload.get("tool_name"), queued_engine=True):
                    fact["flags"].append("ask")
        if (
            _row_is_invisible(entry)
            or is_ask_gate_divert_row(entry)
            or is_hidden_wake_delivery(payload)
        ):
            fact["flags"].append("hidden")
    else:
        fact["flags"] = ["global_unknown"]
    fact["flags"] = sorted(set(fact["flags"]))
    return fact


def valid_source_projection_facts(facts: Any, scan: ScanState, rows: dict[str, Any]) -> bool:
    """Cache structure/origins are validated, not silently repaired/truncated."""
    # ``ScanState.rows`` is NOT cross-checked against ``len(facts)``: documents
    # written before the total was recorded on the incremental path hold a
    # suffix-only count, and rejecting them would force a rescan of every such
    # cache. Contiguous fact ordinals plus last-ID origins validate the complete
    # table without mistaking a valid append for corruption.
    if not isinstance(facts, list) or not 0 <= scan.window_rows <= len(facts):
        return False
    allowed = {
        "id",
        "seq",
        "source_class",
        "flags",
        "call_ids",
        "result_call_id",
        "prune_target",
        "first_kept_entry_id",
        "preserved_candidate_ids",
    }
    enums = {
        "plain_message",
        "custom_message",
        "prune",
        "compaction",
        "bookkeeping",
        "opaque",
        "invalid",
    }
    flag_enums = {"injection", "hidden", "ask", "incomplete", "opaque_causal", "global_unknown"}
    last = {}
    for seq, fact in enumerate(facts):
        if (
            not isinstance(fact, dict)
            or set(fact) - allowed
            or type(fact.get("seq")) is not int
            or fact["seq"] != seq
        ):
            return False
        if (
            not isinstance(fact.get("id"), str)
            or not isinstance(fact.get("source_class"), str)
            or fact["source_class"] not in enums
        ):
            return False
        flags = fact.get("flags")
        if not isinstance(flags, list) or any(
            not isinstance(flag, str) or flag not in flag_enums for flag in flags
        ):
            return False
        for key in ("call_ids", "preserved_candidate_ids"):
            values = fact.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not 1 <= len(value) <= 128 for value in values
            ):
                return False
            if len(values) > 512:
                return False
        for key in ("result_call_id", "prune_target", "first_kept_entry_id"):
            value = fact.get(key)
            if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 128):
                return False
        cls, flags = fact["source_class"], set(flags)
        if cls == "invalid" and "global_unknown" not in flags:
            return False
        if cls == "opaque" and flags not in ({"opaque_causal"}, {"global_unknown"}):
            return False
        if cls == "prune" and not fact.get("prune_target") and "global_unknown" not in flags:
            return False
        if (
            cls == "compaction"
            and "incomplete" not in flags
            and ("first_kept_entry_id" not in fact or "preserved_candidate_ids" not in fact)
        ):
            return False
        if fact["id"]:
            last[fact["id"]] = seq
    return last == rows


@dataclass
class TranscriptIndex:
    """The parsed cache: checkpoints, message docs, and the scan's own facts."""

    checkpoints: list[Checkpoint]
    messages: list[MessageDoc]
    sig: dict[str, Any]
    coverage: dict[str, Any]
    naming: dict[str, Any]
    scan: ScanState
    #: The run partition and its per-run facts (the open frame's source).
    #: Defaulted so a caller that only wants the rail can build an index
    #: without one — and so the field's arrival is not a second constructor
    #: arity every existing call site has to learn.
    runs: list[RunRecord] = field(default_factory=list)
    #: Opaque scan lineage: append preserves it, rebuild/cache loss remints it.
    #: Equality only; response freshness belongs to the caller's request fence.
    journal_generation: str = ""
    #: ALL complete rows by entry id, not just search/rail rows. Derived during
    #: the scan already paid for those views; page readers do indexed lookups.
    journal_rows: dict[str, int] = field(default_factory=dict)
    # PROTOTYPE: offsets come from the same scanner/witness as ordinals. They
    # retain no payload and do not certify settled replay projection semantics.
    journal_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    source_projection_facts: list[dict[str, Any]] = field(default_factory=list)
    source_views: SourceProjectionViews = field(init=False, repr=False)
    version: int = TRANSCRIPT_INDEX_VERSION
    # Process-only identity of the JSON FD that supplied naming+source facts.
    # Never serialized; delayed migration must not overwrite a newer naming write.
    publication_identity: tuple[int, int, int] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.source_views = SourceProjectionViews.build(self.source_projection_facts)
        self.source_views.origins_complete = {
            id_: fact["seq"] for id_, fact in self.source_views.by_id.items() if id_
        } == self.journal_rows


def journal_coordinates_for_page(
    index: TranscriptIndex, page: "TranscriptPage"
) -> tuple[str, dict[str, int]] | None:
    """Coordinates only when the payload and index share one verified source.

    Feature support does not imply readiness. A stale/building index, raced
    page, unknown domain or missing id has no authority; never fabricate a
    coordinate for it or attach a fresh generation to an old page's payload.
    This is a bounded lookup over the page, with no filesystem/journal scan.
    """
    source = page.source_signature
    indexed = (index.sig.get("inode"), index.sig.get("size"), index.sig.get("mtime"))
    if not index.journal_generation or source is None or source != indexed:
        return None
    if any(entry.id not in index.journal_rows for entry in page.entries):
        return None
    return index.journal_generation, {
        entry.id: index.journal_rows[entry.id] for entry in page.entries
    }


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
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            stamp = os.fstat(handle.fileno())
            raw = handle.read()
    except OSError:
        return None
    try:
        loaded = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(loaded, dict):
        return None
    # Private read receipt, deliberately ignored by write_index's explicit payload.
    loaded["_publication_identity"] = (stamp.st_ino, stamp.st_size, stamp.st_mtime_ns)
    return loaded


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
    raw_ranges = raw.get("journal_ranges")
    raw_rows = raw.get("journal_rows")
    if (
        not isinstance(raw_ranges, dict)
        or not isinstance(raw_rows, dict)
        or raw_ranges.keys() != raw_rows.keys()
    ):
        return None
    if any(
        not isinstance(id_, str) or type(seq) is not int or seq < 0 for id_, seq in raw_rows.items()
    ):
        return None
    # The isinstance narrows for the TYPE CHECKER the same shape the validator
    # already refuses at runtime (it answers False for a non-list): one
    # condition, so the facts map reaching the constructor is provably the
    # list the validator just accepted.
    facts = raw.get("source_projection_facts")
    if not isinstance(facts, list) or not valid_source_projection_facts(facts, scan, raw_rows):
        return None
    # A document an incremental refresh wrote before ``scan.rows`` recorded the
    # journal's total holds a suffix-only count (O1, #3034 QA). The validated
    # facts table has exactly one entry per scanned row on both scan paths, so
    # its length IS the total: restating it here heals those caches on read,
    # with no version bump and no forced rescan of every session's index.
    if scan.rows != len(facts):
        scan = replace(scan, rows=len(facts))
    size = sig.get("size")
    if type(size) is not int or any(
        not isinstance(id_, str)
        or not isinstance(span, (list, tuple))
        or len(span) != 2
        or any(type(n) is not int for n in span)
        or not 0 <= span[0] < span[1] <= size
        for id_, span in raw_ranges.items()
    ):
        return None
    parsed_checkpoints = [c for c in (Checkpoint.from_payload(c) for c in checkpoints) if c]
    parsed_messages = [m for m in (MessageDoc.from_payload(m) for m in messages) if m]
    raw_runs = raw.get("runs")
    parsed_runs = (
        [r for r in (RunRecord.from_payload(r) for r in raw_runs) if r]
        if isinstance(raw_runs, list)
        else []
    )
    return TranscriptIndex(
        checkpoints=parsed_checkpoints,
        messages=parsed_messages,
        publication_identity=raw.get("_publication_identity"),
        sig=sig,
        coverage=coverage,
        naming=naming if isinstance(naming, dict) else {},
        scan=scan,
        runs=parsed_runs,
        journal_generation=(
            raw["journal_generation"] if isinstance(raw.get("journal_generation"), str) else ""
        ),
        source_projection_facts=facts,
        journal_ranges={id_: tuple(span) for id_, span in raw_ranges.items()},
        journal_rows=(
            {
                str(key): value
                for key, value in raw.get("journal_rows", {}).items()
                if isinstance(key, str) and type(value) is int and value >= 0
            }
            if isinstance(raw.get("journal_rows"), dict)
            else {}
        ),
    )


def read_index(config_dir: str | Path, session_id: str) -> TranscriptIndex | None:
    """The cached index as a typed view (no freshness check), else ``None``."""
    return _index_from_raw(_read_raw(config_dir, session_id))


@contextmanager
def _publish_lock(path: Path) -> Iterator[None]:
    """Serialize one session's writers on every platform; readers never lock.

    ``fcntl`` DOES NOT EXIST ON WINDOWS, and this runs inside
    ``refresh_index``/``patch_naming``: an unconditional import made the whole
    cache raise there and left ``checkpoints_view`` at ``error``. Use the POSIX
    advisory lock where it exists, ``msvcrt.locking`` where it does not, and an
    in-process lock either way — so a platform with neither still serialises the
    writers this process has, and the same thread can never deadlock itself.
    Atomic replace is what protects READERS, so they take no lock at all.
    """
    with _PUBLISH_THREADS.setdefault(str(path), threading.Lock()):
        handle = None
        try:
            try:
                import fcntl as _fcntl
            except ImportError:  # pragma: no cover - Windows only
                _fcntl = None  # type: ignore[assignment]
            if _fcntl is not None:
                handle = path.open("a")
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
            else:
                try:
                    import msvcrt as _msvcrt
                except ImportError:  # pragma: no cover - exotic platforms
                    _msvcrt = None  # type: ignore[assignment]
                # ``getattr`` because this module is only importable on Windows:
                # a POSIX type-checker sees a stub without these members, and the
                # lock's absence there is exactly the branch we are NOT in.
                locking = getattr(_msvcrt, "locking", None) if _msvcrt else None
                if _msvcrt is not None and locking is not None:
                    handle = path.open("a")
                    handle.seek(0)
                    locking(handle.fileno(), getattr(_msvcrt, "LK_LOCK", 1), 1)
            yield
        finally:
            if handle is not None:
                handle.close()


def write_index(
    config_dir: str | Path,
    session_id: str,
    index: TranscriptIndex,
    *,
    expected_publication: tuple[int, int, int] | None = None,
    previous: tuple[tuple[int, int, int], dict[str, Any]] | None = None,
) -> None:
    """Persist the index atomically (unique temp + replace), best-effort.

    THIS IS BASE'S WRITE PATH, plus one small locked step. The canonical document
    is written exactly as it always was; the read-model companion is NOT built
    here. Building it on this path cost a measured ~1.3 s per append refresh, so
    the append's own publication landed after the open it was meant to serve.
    ``index_read_model.schedule_materialize`` now derives the companion from the
    completed index afterwards, off this path.

    The one addition inside the lock is the naming merge and its sidecar: naming
    is mutable metadata a concurrent ``patch_naming`` can overwrite, so the merge
    and the document it lands in must be one atomic step. Readers never take the
    lock — atomic replace is what protects them.

    ``expected_publication`` is compare-and-publish for a delayed writer: it
    republishes only the document it actually read.

    ``previous`` is the document the CALLER already parsed, WITH the identity it
    was parsed from (``patch_naming`` reads it to build its merge). Handing it in
    removes a second full JSON decode of an index that is routinely tens of MB —
    the round-3 M3 cost — and it is safe only because the identity is re-checked
    under the lock: if the file moved since that parse, the document is re-read, so
    the merge can never run on a stale view (which is the round-1 R4 defect).

    A FAILURE TO REPLACE IS REPORTED, NEVER SWALLOWED. A previous round fsynced a
    read-only descriptor and then swallowed the ``OSError`` in a trailing
    handler; on Windows that meant the index was rebuilt on every open and never
    written. Durability is taken on the handle that wrote the bytes, and a
    failure that leaves the old document in place is logged at ERROR level.
    """
    from local_operator.session import index_read_model

    path = index_path(config_dir, session_id)
    payload_ok = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with _publish_lock(path.with_suffix(".publish.lock")):
                if expected_publication is not None:
                    current = path.stat()
                    if (
                        current.st_ino,
                        current.st_size,
                        current.st_mtime_ns,
                    ) != expected_publication:
                        return
                    if not _sig_matches(index.sig, _journal_path(config_dir, session_id).stat()):
                        return
                else:
                    latest = _previous_if_current(previous, path) or _read_raw(
                        config_dir, session_id
                    )
                    if latest is not None:
                        merged = preserved_naming(latest, {c.id for c in index.checkpoints})
                        live = {c.id for c in index.checkpoints if c.kind == KIND_USER}
                        own = index.naming.get("items") if isinstance(index.naming, dict) else None
                        # UNION with THIS write winning (see round-1 R4): the disk
                        # read is what a concurrent ``patch_naming`` just landed,
                        # this index is what the caller derived. Neither clobbers
                        # the other, and both are filtered to live turns.
                        items = dict(merged.get("items") or {})
                        if (
                            isinstance(own, dict)
                            and index.naming.get("prompt_version") == NAMING_PROMPT_VERSION
                        ):
                            items.update({k: v for k, v in own.items() if k in live})
                        index.naming = {"prompt_version": NAMING_PROMPT_VERSION, "items": items}
                document = _document(index)
                # The sidecar is keyed by the DERIVED token, so it is correct even
                # if the companion lands later; it costs a few KB, not the body.
                try:
                    index_read_model.write_naming(
                        config_dir, session_id, index_read_model.token_for(index), index.naming
                    )
                except (OSError, ValueError):
                    logger.debug("read-model naming unavailable for %s", session_id)
                with tmp.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps(document, separators=(",", ":")))
                    handle.flush()
                    # fsync the descriptor that WROTE these bytes. A read-only
                    # handle cannot be synced on Windows, and that is exactly the
                    # round-2 Q10 defect.
                    os.fsync(handle.fileno())
                tmp.replace(path)
                payload_ok = True
        finally:
            tmp.unlink(missing_ok=True)
    except OSError:
        if not payload_ok:
            logger.error(
                "could not write the transcript index for %s; the previous document is "
                "still in place and this read serves it",
                session_id,
                exc_info=True,
            )
        return
    # Reclaim outside the lock: unlink cost must not sit in the critical section.
    try:
        index_read_model.collect(config_dir, session_id, index_read_model.token_for(index))
    except OSError:
        pass


def _previous_if_current(
    previous: tuple[tuple[int, int, int], dict[str, Any]] | None, path: Path
) -> dict[str, Any] | None:
    """The caller's parsed document, but only if the file has not moved since.

    One ``stat`` under the lock is what makes reusing a parse sound: the same
    identity means the bytes are the ones the caller read, so the merge sees the
    truth; any motion falls back to a fresh read.
    """
    if previous is None:
        return None
    identity, document = previous
    try:
        current = path.stat()
    except OSError:
        return None
    if (current.st_ino, current.st_size, current.st_mtime_ns) != identity:
        return None
    return document


def _document(index: TranscriptIndex) -> dict[str, Any]:
    """The canonical document, in base's key order (the head reader depends on it)."""
    return {
        # The LIVE constant, not ``index.version``: a cache must be stamped
        # with the version this build writes, and reading it back compares
        # against the same live constant (see ``_index_from_raw``).
        "version": TRANSCRIPT_INDEX_VERSION,
        # THE RESUME HEAD: ``version``, ``sig``, ``scan`` and
        # ``journal_generation`` come FIRST and in this order, so that
        # :func:`_read_resume_head` can say how many bytes the next refresh would
        # re-read from a few KB at the top of the file, with no parse of the
        # whole multi-MB document (the paint hold decides that on the event
        # loop). Readers of the document are order-blind (``json.loads``); only
        # the head reader depends on this, and it falls back to the pre-head
        # rule for any file laid out otherwise. The read model derives its own
        # token from exactly these fields, so publishing a companion needs no
        # field here and no second canonical write.
        "sig": index.sig,
        "scan": index.scan.to_payload(),
        "journal_generation": index.journal_generation,
        "coverage": index.coverage,
        "checkpoints": [c.to_payload() for c in index.checkpoints],
        "messages": [m.to_payload() for m in index.messages],
        "runs": [r.to_payload() for r in index.runs],
        "naming": index.naming,
        "journal_rows": index.journal_rows,
        "journal_ranges": index.journal_ranges,
        "source_projection_facts": index.source_projection_facts,
    }


#: How much of the cache file :func:`_read_resume_head` reads. The head holds
#: four small fields (an id, a uuid and a handful of integers: ~300 bytes
#: measured on the 51.7 MB fixture), so 4 KiB is generous; a head that does not
#: fit simply answers "unknown" and the caller keeps the pre-head rule.
_RESUME_HEAD_BYTES = 4096

_HEAD_KEYS = ("version", "sig", "scan", "journal_generation")


def _read_resume_head(
    config_dir: str | Path, session_id: str
) -> tuple[dict[str, Any], ScanState, str, int] | None:
    """``(sig, scan, journal_generation, document_bytes)`` from the first bytes.

    ``document_bytes`` comes from the SAME open descriptor's ``fstat``. The paint
    hold runs on the event loop, so the hydration charge it needs must not cost a
    second path lookup (``stat``) on that loop: the bounded head read is already
    happening here, and one ``fstat`` on the handle is all this adds.

    WHY (conversation-first-paint R2): the paint hold must decide whether a
    refresh is payable by the bytes it will RE-READ, and that is
    ``journal size - scan.window_offset`` -- a figure that lives inside a cache
    document of up to several MB. Parsing it on the event loop would be the very
    cost the hold exists to keep off the first frame, so :func:`write_index`
    puts these fields first and this decodes only them, value by value, from a
    bounded read.

    ``None`` for anything it cannot vouch for: no file, an unreadable one, the
    wrong version, or a head laid out otherwise (every cache written before
    this). The caller then keeps its pre-head answer, so an old cache costs the
    old behaviour until its next write, never a wrong deferral.
    """
    try:
        with index_path(config_dir, session_id).open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            head = handle.read(_RESUME_HEAD_BYTES)
    except OSError:
        return None
    # A cut multibyte character can only sit at the END of the read, past the
    # fields this decodes; ``ignore`` keeps the prefix intact.
    text = head.decode("utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    values: dict[str, Any] = {}
    position = 1 if text.startswith("{") else -1
    try:
        for expected in _HEAD_KEYS:
            if position < 0:
                return None
            key, position = decoder.raw_decode(text, position)
            if key != expected or text[position : position + 1] != ":":
                return None
            values[key], position = decoder.raw_decode(text, position + 1)
            if text[position : position + 1] != ",":
                return None
            position += 1
    except ValueError:
        return None
    sig = values["sig"]
    scan = ScanState.from_payload(values["scan"])
    generation = values["journal_generation"]
    if (
        values["version"] != TRANSCRIPT_INDEX_VERSION
        or not isinstance(sig, dict)
        or scan is None
        or not isinstance(generation, str)
    ):
        return None
    return sig, scan, generation, size


def _rederive_bytes(
    st: os.stat_result, sig: dict[str, Any], scan: ScanState, generation: str
) -> int:
    """How many journal bytes :func:`refresh_index` would read for this cache.

    The same ladder, in the same order, minus the one check that needs the
    journal's bytes (``_verify_tail``; a tail that fails it rescans, which this
    reports as the cheaper incremental figure -- the pre-R2 direction, and rare):
    current -> nothing; grown on the same inode -> from the resume WINDOW, not
    from the old end, because the incremental path re-derives the window's run
    whole; anything else -> the whole journal.
    """
    if generation and _sig_matches(sig, st):
        return 0
    if generation and st.st_ino == sig.get("inode") and st.st_size > scan.offset:
        return max(0, st.st_size - scan.window_offset)
    return st.st_size


def _cached_refresh_cost(
    config_dir: str | Path, session_id: str, journal: os.stat_result
) -> tuple[bool, int | None, int]:
    """``(has_previous, rederive_bytes, hydration_bytes)`` without hydrating.

    An exact disk HIT avoids journal work, not ``_read_raw``/``_index_from_raw``:
    the cache still contains the entire transcript's derived facts. Price that
    document before starting its uninterruptible JSON step. Both figures come
    from the ONE bounded head read (its ``fstat`` sizes the document), because
    this runs inside the paint hold's event-loop decision and a separate ``stat``
    there would be new loop I/O. Missing/old heads retain the previous
    unknown-rederivation policy, but an existing document is never free.

    This describes a REFRESH, which reads disk even if an older object is held.
    A validated resident HIT bypasses refresh in the consumers; charging it here
    as resident would exempt work that ``refresh_index`` actually still does.
    """
    head = _read_resume_head(config_dir, session_id)
    if head is None:
        # No head means no document this build can vouch for; the pre-head rule
        # (whatever a fresh reader sees) is the caller's fallback, not ours.
        return _stat_or_none(index_path(config_dir, session_id)) is not None, None, 0
    rederive = _rederive_bytes(journal, *head[:3])
    return True, rederive, head[3]


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

    THIS READ IS OUTSIDE THE LOCK, and that is deliberate rather than a claim:
    it only decides which document shape to hand ``write_index`` (an existing
    index, a version-mismatched one, or the empty skeleton). The merge that
    must be atomic — this patch against whatever a concurrent writer landed —
    happens INSIDE ``write_index``, which re-reads ``naming`` under the publish
    lock and unions it with this write winning. An earlier version of this
    docstring claimed the whole function was under that lock; it never was.
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
        # Hand the parse over WITH its identity: write_index would otherwise decode
        # this same document again under the lock (round-3 M3), and the identity
        # check there keeps a concurrent patch from being merged over stale bytes.
        receipt = raw.get("_publication_identity")
        if isinstance(receipt, tuple) and len(receipt) == 3:
            write_index(config_dir, session_id, index, previous=(receipt, raw))
        else:
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
    kind: str  # user | assistant | tool | inject | start | marker | compaction | other
    text: str = ""
    token: str = ""
    marker_kind: str | None = None
    eligible: bool = True
    #: A tool row's ``tool_call_id`` / ``tool_name``, for the client's own key
    #: vocabulary (``tool:<call_id>``) and for its cross-session filter.
    tool_call_id: str = ""
    tool_name: str = ""
    #: A marker row's own ``details.anchor`` — the id the PRODUCER gives the
    #: record (``attention.py``: ``completion-<token>``), which is the id the
    #: desktop's reducer keys it by. See ``RunRecord.key_id``.
    anchor: str = ""
    #: ``payload.custom_type`` of an inject row — the same payload field the
    #: injection rule at ``_classify`` already inspects; None on other kinds.
    custom_type: str | None = None
    #: The four facts a tool row contributes to its run's counters, read from the
    #: payload the same way the desktop's own ``isFailedCall`` reads them (see
    #: ``_tool_failure``): ``is_error`` from the row, and the fault/delivery
    #: vocabulary from ``provider_payload.details``. The three exclusions are why
    #: they are carried rather than inferred: a stopped call, a never-sent call and
    #: a partial ``send`` delivery all carry ``is_error`` while being settled
    #: non-failures, and a bar that counted them would state a failure that never
    #: happened.
    is_error: bool = False
    duration_s: float | None = None
    fault: str | None = None
    delivery: str | None = None
    #: True for a row whose body the scanner dropped (a row past
    #: ``_MAX_KEPT_LINE_BYTES``): the row still counts as an action, but nothing
    #: inside it can be read, so a run containing one is only ever ``complete:
    #: False`` (see :class:`RunRecord`).
    body_dropped: bool = False
    source_fact: dict[str, Any] | None = None
    #: True when this row PAINTS as a completed turn — the desktop's terminal
    #: vocabulary (``boundaryKindOf``/``isTerminalMarker``): a
    #: ``completion_attention`` marker the renderer turns into a ``complete``
    #: notice (an anchor is what makes it renderable at all), or a
    #: ``session_incident``, which the renderer paints at ``level: "error"``.
    terminal: bool = False


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


def _tool_duration(payload: dict[str, Any]) -> float | None:
    """The ``duration_s`` a tool row reports, or ``None`` when it reports none.

    Read from ``provider_payload`` rather than from the row's top level for the
    reason the desktop's own reducer does: the tool facts (``details``,
    ``useless``, ``duration_s``) are a provider envelope, and a row that carries
    no envelope contributes no worked time rather than a ``0`` — the difference
    between "the tool was instant" and "the producer stated nothing", which the
    bar's `Took` line refuses to conflate.
    """
    provider_payload = payload.get("provider_payload")
    if not isinstance(provider_payload, dict):
        return None
    value = provider_payload.get("duration_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _tool_fault_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """The fault and delivery words a tool row's ``details`` carry, if any.

    Both ride inside ``provider_payload.details`` — ``__fault`` for a call the
    harness never ran or the user aborted, ``delivery`` for a ``send`` with a
    partial outcome — and both exist to EXCLUDE a row from a failure count.
    """
    provider_payload = payload.get("provider_payload")
    details = provider_payload.get("details") if isinstance(provider_payload, dict) else None
    if not isinstance(details, dict):
        return {"fault": None, "delivery": None}
    fault = details.get("__fault")
    delivery = details.get("delivery")
    # ``details.delivery`` IS A DICT (``builtin.py`` writes ``{"state": ...}``),
    # and the desktop reads ``details.delivery.state``
    # (``transcript-reducer.ts:414``). Reading it as a string made
    # ``row.delivery`` the repr of the dict, so a partial ``send`` was counted as
    # a FAILURE — measured: the server said 3 where the desktop's own fold said 1.
    # A plain string is what an older writer left behind, so both shapes are
    # accepted rather than the newer one silently failing on an old journal.
    if isinstance(delivery, dict):
        delivery = delivery.get("state")
    return {
        "fault": str(fault) if fault is not None else None,
        "delivery": str(delivery) if delivery is not None else None,
    }


def _tool_failure(row: _Row) -> bool:
    """Whether a tool row is a genuine FAILURE, in the desktop's own terms.

    The client's predicate is ``isFailedCall``
    (``turn-collapse-model.ts``), and the bar's count is only useful if the
    backend states the number the client would have derived itself. Three
    settled non-failures carry ``is_error`` and are excluded there, so they are
    excluded here:

    - a never-run or aborted call (``__fault`` of ``skipped``/``aborted``) —
      ``isInterruptedFault``'s two words;
    - a partial ``send`` delivery (``mailbox``/``unconfirmed``), which draws the
      amber row rather than a red one;
    - and the ``stopped``/``never_sent``/``not_run_reason`` arms, which the
      DURABLE payload does not carry at all: the reducer hard-codes those to
      their defaults on this path, so a row read from a journal cannot be any of
      them.
    """
    if not row.is_error:
        return False
    if row.fault in ("skipped", "aborted"):
        return False
    return row.delivery not in ("mailbox", "unconfirmed")


def _row_is_invisible(entry: dict[str, Any]) -> bool:
    """Whether the SERVED page would never contain this row.

    The predicates live in ``harness.rows`` beside the filter that applies them
    (``visible_transcript_rows``); this is the index asking the same question the
    reader does, in the same place, so the counts and the page cannot disagree
    about which rows exist.

    THE SHAPE MATTERS AND IS NOT UNIFORM: these two read ``row["payload"]``
    themselves, so they take the ENVELOPE — while ``is_harness_injection``
    unwraps a payload and takes the PAYLOAD (``open_frame.strip_entry`` passes it
    that way). Handing either one the other's shape matches nothing and fails
    silently, which is how the first version of this check came to count every
    hidden row it was written to exclude.
    """
    from local_operator.harness.rows import (
        is_ask_gate_divert_row,
        is_binding_notice_row,
        is_hidden_tool_row,
    )

    return (
        is_hidden_tool_row(entry) or is_ask_gate_divert_row(entry) or is_binding_notice_row(entry)
    )


def _carries_send_marker(pieces: list[bytes]) -> bool:
    """Whether a line's bytes, already in hand, say this is a ``send`` result.

    Only used on a row whose body is being DROPPED, and only over bytes the
    reader has streamed anyway, so it costs nothing the read was not already
    paying.
    """
    return any(_SEND_MARKER in piece for piece in pieces)


def is_tool_head(head: bytes) -> bool:
    """Whether a row's HEAD (the bytes before its body) declares a tool result."""
    return b'"role":"tool"' in head


def _classify(
    ordinal: int,
    line_start: int,
    line_end: int,
    head: bytes,
    line: bytes | None,
    dropped_tool_name: str = "",
) -> _Row:
    """One complete line -> a :class:`_Row` (dropped bodies classify from head)."""
    id_ = _head_id(head)
    ts = _head_ts(head)
    if line is None:
        # Body dropped: only tool/compaction/other rows are ever dropped, see
        # _skip_eligible. A compaction keeps its own kind because it is a RECORD
        # on the wire (the marker a transcript pins) while ``other`` is where the
        # rows no client ever sees land — and the run partition must treat the
        # two differently.
        if is_tool_head(head):
            kind = "tool"
        elif b'"type":"compaction"' in head:
            kind = "compaction"
        else:
            kind = "other"
        return _Row(
            ordinal=ordinal,
            offset=line_start,
            end=line_end,
            id=id_,
            ts=ts,
            kind=kind,
            # The name a dropped row still has to carry: see ``dropped_tool_name``
            # and the note at its call site (UI review round 2, M2).
            tool_name=dropped_tool_name if kind == "tool" else "",
            body_dropped=True,
            source_fact={
                "id": id_,
                "seq": ordinal,
                "source_class": "opaque",
                "flags": ["opaque_causal" if kind == "tool" else "global_unknown"],
            },
        )
    fact = source_projection_fact(None, ordinal)

    def make_row(**kwargs: Any) -> _Row:
        captured = {**fact, "id": kwargs.get("id", "")}
        return _Row(**kwargs, source_fact=captured)

    try:
        entry = json.loads(line)
    except ValueError:
        return make_row(
            ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="other"
        )
    if not isinstance(entry, dict):
        return make_row(
            ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="other"
        )
    fact = source_projection_fact(entry, ordinal)
    id_ = entry.get("id") if isinstance(entry.get("id"), str) else id_
    if _row_is_invisible(entry):
        # F5: A ROW THE CLIENT NEVER RECEIVES IS NOT WORK. The page is filtered
        # through ``harness.rows.visible_transcript_rows`` before it is served —
        # hidden wake deliveries, the ``patience`` arm's ledger rows and a
        # diverted ask's result — so a count that included them would state work
        # the reader cannot see, and the client's own fold would contradict it
        # (reproduced: a run of one visible call and one hidden row reported
        # ``action_count=2``, worked 3.5 s, against the fold's 1 and 3.0 s).
        return make_row(
            ordinal=ordinal,
            offset=line_start,
            end=line_end,
            id=id_,
            ts=ts,
            kind="other",
        )
    if not id_:
        id_ = str(entry.get("id", ""))
    if not ts:
        try:
            ts = float(entry.get("ts", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
    etype = entry.get("type")
    if etype == "compaction":
        # Classified before the payload branches: a compaction is a transcript
        # RECORD (the marker a reader pins) and never a message, whatever the
        # payload holds.
        return make_row(
            ordinal=ordinal,
            offset=line_start,
            end=line_end,
            id=id_,
            ts=ts,
            kind="compaction",
        )
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    if etype == "message":
        if payload.get("kind") == "custom" and payload.get("custom_type"):
            inject_type = str(payload.get("custom_type"))
            return make_row(
                ordinal=ordinal,
                offset=line_start,
                end=line_end,
                id=id_,
                ts=ts,
                kind="inject",
                text=_inject_text(payload.get("details")),
                custom_type=inject_type,
                # The desktop paints this one at ``level: "error"``, which is its
                # second terminal-marker arm (``boundaryKindOf``'s ``custom``
                # case). Every other custom is an info-level statement.
                terminal=inject_type == _SESSION_INCIDENT,
            )
        role = payload.get("role")
        if role == "user" or role == "assistant":
            text = _content_text(payload)
            if role == "user" and _is_harness_user_row(id_, payload, text):
                # Harness chrome, skipped whole: no user checkpoint (the rail
                # would caption it "Your message"), no find doc, no turn content.
                return make_row(
                    ordinal=ordinal,
                    offset=line_start,
                    end=line_end,
                    id=id_,
                    ts=ts,
                    kind="other",
                )
            return make_row(
                ordinal=ordinal,
                offset=line_start,
                end=line_end,
                id=id_,
                ts=ts,
                kind=str(role),
                text=text,
            )
        if role == "tool":
            return make_row(
                ordinal=ordinal,
                offset=line_start,
                end=line_end,
                id=id_,
                ts=ts,
                kind="tool",
                is_error=payload.get("is_error") is True,
                duration_s=_tool_duration(payload),
                tool_call_id=str(payload.get("tool_call_id") or ""),
                tool_name=str(payload.get("tool_name") or ""),
                **_tool_fault_fields(payload),
            )
        return make_row(
            ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="other"
        )
    if etype == "custom":
        custom_type = payload.get("custom_type")
        if custom_type == _ATTENTION_STARTED:
            details = payload.get("details")
            token = ""
            if isinstance(details, dict):
                token = str(details.get("token", ""))
            return make_row(
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
            anchor = ""
            marker_kind: str | None = None
            eligible = True
            anchored = False
            if isinstance(details, dict):
                token = str(details.get("token", ""))
                kind = details.get("kind")
                marker_kind = str(kind) if kind is not None else None
                eligible = bool(details.get("eligible", True))
                # THE ANCHOR IS WHAT MAKES THE ROW RENDERABLE, so a run's
                # settlement cannot be read off the marker's kind alone: the
                # renderer returns a notice only for a marker carrying an anchor
                # string (``durableRecord``), and a marker without one paints
                # nothing at all — a boundary the client cannot see is not a
                # boundary.
                anchored = isinstance(details.get("anchor"), str)
                anchor = str(details.get("anchor") or "") if anchored else ""
            return make_row(
                ordinal=ordinal,
                offset=line_start,
                end=line_end,
                id=id_,
                ts=ts,
                kind="marker",
                anchor=anchor,
                token=token,
                marker_kind=marker_kind,
                eligible=eligible,
                terminal=anchored and marker_kind in _TERMINAL_MARKER_KINDS,
            )
    return make_row(ordinal=ordinal, offset=line_start, end=line_end, id=id_, ts=ts, kind="other")


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
            # Per-ROW state, reset after each yield below: it has to exist before
            # the first row streams, which is why it is initialised here too.
            send_attr = False
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
                                    send_attr = send_attr or _carries_send_marker(parts)
                                    dropped = True
                                    parts.clear()
                            elif _SEND_MARKER in piece:
                                send_attr = True
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
                                send_attr = send_attr or _carries_send_marker(parts)
                                dropped = True
                                parts.clear()
                        elif _SEND_MARKER in piece:
                            send_attr = True
                    if dropped and is_tool_head(head) and send_attr:
                        # A DROPPED BODY STILL BELONGS TO A ``send``, and the
                        # desktop hides the row by its NAME — which rides BEYOND
                        # the head on a row this large, so it is not in ``head``.
                        # The stream is already in hand (the reader sees every
                        # chunk of the line while dropping it), so the marker is
                        # caught as it goes past rather than by re-reading the
                        # file. Without it the dropped ``send`` was an action the
                        # facts counted but the cross-session split did not, and
                        # the desktop's subtraction left a hidden action in the
                        # bar (UI review round 2, M2).
                        tool_name_override = CROSS_SESSION_TOOL_NAME
                    else:
                        tool_name_override = ""
                    row = _classify(
                        self._next_ordinal,
                        line_start,
                        line_start + line_len + 1,
                        head,
                        None if dropped else b"".join(parts),
                        dropped_tool_name=tool_name_override,
                    )
                    self.rows += 1
                    self._next_ordinal += 1
                    # YIELD THE GIL, PERIODICALLY, BECAUSE THIS LOOP IS A THREAD
                    # ON THE SERVER'S OWN PROCESS (review/QA round 1, item 1): a
                    # cold scan of a large journal parses JSON back to back, and
                    # while it does, every other session's read on the same server
                    # waits behind it — measured at 322-674 ms of stall for
                    # another session's snapshot during a 118 MB scan. CPython
                    # releases the GIL on ``time.sleep``, so a zero-length sleep
                    # every ``_SCAN_YIELD_ROWS`` rows hands the loop its turn at a
                    # cost no reader can measure.
                    if self.rows % _SCAN_YIELD_ROWS == 0:
                        time.sleep(0)
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
                    send_attr = False
            if line_len > 0:
                self.torn = True
            self.st = os.fstat(handle.fileno())


class _Replaced(Exception):
    """The journal was atomically replaced while it was being read."""


class _RunBuilder:
    """The open run a :class:`_Derivation` is filling, in mutable form.

    Split from :class:`RunRecord` because a run's counters are written once per
    tool row: rebuilding a frozen record per row would make the scan's cost
    proportional to the number of rows rather than the number of runs.
    """

    def __init__(
        self,
        *,
        opening_user_id: str,
        first_seq: int,
        start_ts: float,
        opened_with_user_row: bool,
        last_painter: str | None = None,
        saw_terminal: bool = False,
        saw_work: bool = False,
        last_work_seq: int = -1,
        key_id: str = "",
        worked_any: bool = False,
        cross_actions: int = 0,
        cross_worked: float = 0.0,
        action_count: int = 0,
        failed_count: int = 0,
        worked_seconds: float = 0.0,
        complete: bool = True,
        closing_answer_id: str = "",
    ) -> None:
        self.opening_user_id = opening_user_id
        #: The closing answer the REGION has seen so far, by the index's own
        #: per-turn rule (an assistant row with text in it; the last one wins).
        #: A carried run starts with none, so the region's own rows decide its
        #: answer — seeding it from the prefix's record pinned a run to the answer
        #: it had BEFORE the append that completed it.
        self.answer_id = ""
        self.closing_answer_id = closing_answer_id
        self.first_seq = first_seq
        self.last_seq = first_seq
        self.start_ts = start_ts
        self.end_ts = start_ts
        self.last_id = opening_user_id or ""
        self.action_count = action_count
        self.failed_count = failed_count
        self.worked_seconds = worked_seconds
        self.complete = complete
        self.last_painter = last_painter
        self.saw_terminal = saw_terminal
        self.saw_work = saw_work
        #: The region's last WORK row (F4's settlement test reads it); a carried
        #: run seeds it from the prefix so a marker older than the prefix's work
        #: stays older.
        self.last_work_seq = last_work_seq
        self.key_id = key_id
        self.worked_any = worked_any
        self.cross_actions = cross_actions
        self.cross_worked = cross_worked
        self.opened_with_user_row = opened_with_user_row

    def touch(self, row: _Row) -> None:
        self.last_seq = row.ordinal
        self.end_ts = row.ts
        self.last_id = row.id


class _Derivation:
    """Accumulates the scan's raw facts and derives checkpoints and docs."""

    def __init__(self, carry: RunRecord | None = None) -> None:
        # THE CARRIED RUN (see ``_scan_incremental``): an incremental scan
        # re-derives from a resume window, and a steered run can straddle that
        # window — its opening user row lies above it while the window sits on
        # the steer's own ``attention_started``. The prefix's record is handed in
        # as the OPEN run so the region's rows continue it instead of opening a
        # second head-cut run beside it, which would state half a count twice.
        self.runs: list[RunRecord] = []
        self._open_run: _RunBuilder | None = None
        self._prev_row: _Row | None = None
        if carry is not None:
            self._open_run = _RunBuilder(
                opening_user_id=carry.opening_user_id,
                closing_answer_id=carry.closing_answer_id,
                first_seq=carry.first_seq,
                start_ts=carry.start_ts,
                opened_with_user_row=carry.opened_with_user_row,
                last_painter=carry.last_painter,
                saw_terminal=carry.saw_terminal,
                saw_work=carry.saw_work,
                last_work_seq=carry.last_work_seq,
                key_id=carry.key_id,
                worked_any=carry.worked_any,
                cross_actions=carry.cross_actions,
                cross_worked=carry.cross_worked,
                action_count=carry.action_count,
                failed_count=carry.failed_count,
                worked_seconds=carry.worked_seconds,
                complete=carry.complete,
            )
            self._open_run.last_seq = carry.last_seq
            self._open_run.end_ts = carry.end_ts
            self._open_run.last_id = carry.last_id
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

    def _run_closed(self) -> bool:
        """The client's closure test, evaluated as its own ``walkTurns`` does.

        Three arms, and the desktop's module comment is the authority for each:
        a TERMINAL marker seen since the run opened; a tail whose last painting
        row is a settled assistant row (``settledTail`` — every durable row is
        settled by construction, so that is "the last painter is an assistant");
        and an EMPTY PREAMBLE, a run that opened off a non-user row and has done
        no work of its own, which cannot be steered into.
        """
        run = self._open_run
        if run is None:
            return True
        return (
            run.saw_terminal
            or run.last_painter == "assistant"
            or (not run.opened_with_user_row and not run.saw_work)
        )

    def _open_head_cut_run(self, row: _Row) -> _RunBuilder:
        """Start a run whose head is not in the rows being read."""
        return _RunBuilder(
            opening_user_id="",
            first_seq=row.ordinal,
            start_ts=row.ts,
            opened_with_user_row=False,
        )

    def _ensure_run(self, row: _Row) -> _RunBuilder:
        if self._open_run is None:
            self._open_run = self._open_head_cut_run(row)
        return self._open_run

    def consume(self, row: _Row) -> None:
        kind = row.kind
        self._track_run(row)
        self._prev_row = row
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

    def _track_run(self, row: _Row) -> None:
        """The run partition, in the client's own unit (see :class:`RunRecord`).

        A user row OPENS a run when no run is open or the open one is closed;
        otherwise it is a steer and the run keeps its identity — which is why the
        identity of a steered run is its FIRST user row and its count is the
        whole steered span.
        """
        kind = row.kind
        if kind == "user":
            if self._open_run is None or self._run_closed():
                self._close_run()
                self._open_run = _RunBuilder(
                    opening_user_id=row.id,
                    first_seq=row.ordinal,
                    start_ts=row.ts,
                    opened_with_user_row=True,
                )
            else:
                self._open_run.touch(row)
            return
        if kind in ("start", "other"):
            # NEITHER KIND IS A RECORD THE CLIENT EVER SEES, so neither may open a
            # run NOR extend one. ``attention_started`` is dropped by the
            # renderer's projection and ``other`` is where harness chrome, prune
            # markers and unknown types land. Two failures came from treating
            # them as rows: opening a run on a leading ``attention_started``
            # produced an empty head-cut run at the top of every journal (a fact
            # entry with no rows, no count and no identity), and TOUCHING a run
            # with a young run's start row made the next run's marker resolve
            # against the previous run's span.
            return
        run = self._ensure_run(row)
        run.touch(row)
        # THE CLIENT'S KEY, tracked on every record it keys: a tool result is
        # ``tool:<call_id>``, a marker notice ``prov-<token>``, anything else its
        # entry id (``transcript-reducer.ts:2838``, ``:2979``).
        if kind == "tool":
            run.key_id = f"tool:{row.tool_call_id}" if row.tool_call_id else row.id
        elif kind == "marker":
            # THE PRODUCER'S OWN ANCHOR (QA round 2, Q5): ``attention.py`` names
            # the record ``completion-<token>`` and that is the id the desktop's
            # reducer keys it by, so it is what a run's facts must be published
            # under. ``prov-<token>`` was a spelling invented here; a client never
            # derived it, so on a head-cut interrupted run — the case Q2 exists
            # for — the fact matched nothing and the bar kept its loaded span.
            run.key_id = row.anchor or (f"prov-{row.token}" if row.token else "")
        elif row.id:
            run.key_id = row.id
        if kind in ("assistant", "tool"):
            run.saw_work = True
            run.last_work_seq = row.ordinal
        if kind == "assistant" and row.text:
            # ``paintsSomething``: an assistant row paints only with text in it,
            # so an empty one is not the run's last painter.
            run.last_painter = "assistant"
            run.answer_id = row.id
        elif kind == "tool":
            run.last_painter = "tool"
            run.action_count += 1
            if row.body_dropped:
                # The row counts as an action (the head names the role); what is
                # inside it — a failure, a duration — is unknowable, and the run
                # says so rather than reporting an exact-looking number.
                run.complete = False
            else:
                if _tool_failure(row):
                    run.failed_count += 1
                if row.duration_s is not None:
                    run.worked_seconds += row.duration_s
                    # The client states ``null``, never ``0s``, for a run that
                    # reported no figure at all (``workedSeconds``), so whether
                    # anything reported one is part of the fact.
                    run.worked_any = True
            # THE CROSS-SESSION SPLIT IS OUTSIDE THE ``body_dropped`` TEST (UI
            # review round 2, M2): the desktop hides the ROW, not its body, so a
            # ``send`` whose body the strip removed is still an action the client
            # subtracts. Counting it only when the body survived left the
            # subtraction one short and a hidden action in the bar.
            if row.tool_name == CROSS_SESSION_TOOL_NAME:
                run.cross_actions += 1
                if row.duration_s is not None:
                    run.cross_worked += row.duration_s
        if row.terminal:
            run.saw_terminal = True

    def _close_run(self) -> None:
        """Finalise the open run with the span it actually accumulated.

        NOT extended to "the row before the next user row". A run's span is the
        rows TOUCHED while it was open, and the row between a run's last record
        and the next user row belongs to the NEXT run — a young run's
        ``attention_started`` sits exactly there. Extending the span made one run
        own the next run's start row, which the resolution floor then mis-bound,
        and it is why the span is stated as rows touched rather than as an
        interval between two user rows.
        """
        run = self._open_run
        if run is None:
            return
        self.runs.append(
            RunRecord(
                opening_user_id=run.opening_user_id,
                closing_answer_id=run.closing_answer_id,
                last_id=run.last_id,
                first_seq=run.first_seq,
                last_seq=run.last_seq,
                start_ts=run.start_ts,
                end_ts=run.end_ts,
                action_count=run.action_count,
                failed_count=run.failed_count,
                worked_seconds=run.worked_seconds,
                # Settled and outcome are decided in ``emit``, where every
                # marker of the region is in hand: a run's closure can be
                # proven by a marker that arrived after its last row.
                settled=False,
                outcome=None,
                complete=run.complete,
                answer_id=run.answer_id,
                key_id=run.key_id,
                worked_any=run.worked_any,
                cross_actions=run.cross_actions,
                cross_worked=run.cross_worked,
                last_work_seq=run.last_work_seq,
                last_painter=run.last_painter,
                saw_terminal=run.saw_terminal,
                saw_work=run.saw_work,
                opened_with_user_row=run.opened_with_user_row,
            )
        )
        self._open_run = None

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
        prior_runs: list[RunRecord] | None = None,
    ) -> tuple[list[Checkpoint], list[MessageDoc], list[RunRecord]]:
        """Derive this region's checkpoints, docs and runs, merged behind the priors."""
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
        return (
            checkpoints,
            list(prior_messages) + self.messages,
            self._emit_runs(prior_runs or [], last_marker_ordinal, last_start_ordinal),
        )

    def _emit_runs(
        self,
        prior_runs: list[RunRecord],
        last_marker_ordinal: int,
        last_start_ordinal: int,
    ) -> list[RunRecord]:
        """Close the open run and decide every run's settlement and outcome.

        DECIDED HERE RATHER THAN AT ``_close_run`` because a run's closure can be
        proven by a row that arrives after its own last row: a run whose last row
        is a tool call is closed by the marker that lands when the turn finishes,
        and a record emitted the moment the run stopped growing would keep
        calling that run live for the rest of the journal's life.
        """
        self._close_run()
        runs = list(prior_runs) + self.runs
        carried = len(prior_runs)
        out: list[RunRecord] = []
        for position, run in enumerate(runs):
            is_tail = position == len(runs) - 1
            outcome: str | None = None
            if position < carried and run.settled and not is_tail:
                # A SETTLED PRIOR RUN'S OUTCOME IS FINAL, and this region cannot
                # recompute it. ``_scan_incremental`` keeps only runs whose last
                # row lies above the re-derived region, so their markers and
                # ``attention_started`` rows were consumed by an EARLIER scan and
                # are absent from ``self.markers``/``self.starts``: recomputing
                # here resolved nothing and overwrote every settled run's
                # error/complete with None after any append (measured on a real
                # journal: error,error,complete -> None,None,None) until a full
                # rescan. Settled means the earlier scan already decided the
                # outcome from every marker inside the run's own rows, and a kept
                # run gains no rows, so the stored value is what a full scan
                # derives. An UNSETTLED kept run is the previous scan's open tail
                # (a user-less hub/wake run the window moved past): its stored
                # ``open`` is provisional and its closing marker may sit in this
                # region, so it keeps the recompute below. The tail is excluded
                # defensively: a region with no run of its own leaves a kept run
                # as the tail, and that one must still take the open-tail rule.
                out.append(run)
                continue
            # The run before this one: a marker's ``attention_started`` is written
            # BEFORE its user row, so the start row legitimately lies ABOVE the
            # run's opening user row — requiring ``start >= first_seq`` dropped
            # every outcome for every run whose marker resolved normally. What
            # proves the binding is that the start sits inside this run and not
            # inside the previous one.
            floor = runs[position - 1].last_seq if position > 0 else -1
            for ordinal, token, marker_kind, eligible in self.markers:
                if ordinal < run.first_seq or ordinal > run.last_seq:
                    continue
                start = self.starts.get(token)
                if start is None or start <= floor or start > ordinal:
                    continue
                if marker_kind == "closed":
                    # The neutral closure: the run is over with no outcome to
                    # state, exactly as the checkpoint rule reads it.
                    continue
                if marker_kind == "retired":
                    marker_kind = OUTCOME_INTERRUPTED
                outcome = marker_kind if eligible else None
            # F4: THE TAIL RUN IS SETTLED ONLY WHEN NOTHING CAN JOIN IT ANY
            # MORE, which is stricter than "a marker resolved it once". A run
            # that is resolved and then CONTINUED without a user row — a wake,
            # hub or peer follow-up — keeps the same identity here and in the
            # client's ``walkTurns``, so a count stated for it would move on
            # every row that lands afterwards: reproduced as a run reported
            # ``settled: true, actions: 1`` and then ``actions: 3`` with the
            # same key, after a wake turn appended to it. The rail's own
            # open-tail rule is the test — the newest resolvable marker must be
            # newer than the newest ``attention_started`` AND newer than the
            # run's last WORK row — and a run that fails it is stated as LIVE
            # with no outcome and, on the wire, no counts, so a client keeps
            # its own fold for it rather than trusting a number about to
            # change.
            # A marker bound to the run above is not enough for the TAIL: the
            # region can hold a marker whose work continued afterwards, and the
            # decision must not depend on which loop proved what.
            settled = not is_tail
            if is_tail:
                settled = (
                    last_marker_ordinal > last_start_ordinal
                    and last_marker_ordinal >= run.last_work_seq
                    and last_marker_ordinal >= run.first_seq
                )
                if not settled:
                    outcome = OUTCOME_OPEN
            closing = run.answer_id or run.closing_answer_id
            if not closing:
                closing = self._run_answer_id(run)
            out.append(
                RunRecord(
                    opening_user_id=run.opening_user_id,
                    closing_answer_id=closing,
                    last_id=run.last_id,
                    first_seq=run.first_seq,
                    last_seq=run.last_seq,
                    start_ts=run.start_ts,
                    end_ts=run.end_ts,
                    action_count=run.action_count,
                    failed_count=run.failed_count,
                    worked_seconds=run.worked_seconds,
                    # A RUN THAT IS NOT THE TAIL IS OVER, whatever marker it has:
                    # the next user row closed it (that is what makes it not the
                    # tail), and only the LAST run can still grow. This is the
                    # same rule the checkpoints state for the rail's tail turn,
                    # applied to the unit the bar is drawn over.
                    settled=settled,
                    outcome=outcome,
                    complete=run.complete,
                    answer_id=run.answer_id,
                    key_id=run.key_id,
                    worked_any=run.worked_any,
                    cross_actions=run.cross_actions,
                    cross_worked=run.cross_worked,
                    last_work_seq=run.last_work_seq,
                    last_painter=run.last_painter,
                    saw_terminal=run.saw_terminal,
                    saw_work=run.saw_work,
                    opened_with_user_row=run.opened_with_user_row,
                )
            )
        return out

    def _run_answer_id(self, run: RunRecord) -> str:
        """The id the client would elect as this run's closing answer.

        The COMPLETION CHECKPOINT's own rule — the turn's answer row
        (``answer_row``), or its last message row when the turn has no answer —
        and it lives here rather than in the wire layer so the two cannot drift:
        a run's ``closing_answer_id`` is the id the client's own bar keys on.
        """
        low = bisect_right(self.user_ords, run.first_seq - 1)
        high = bisect_right(self.user_ords, run.last_seq)
        for index in range(high - 1, low - 1, -1):
            if not self.content[index]:
                continue
            answer = self.answer_row[index]
            if answer is not None:
                return answer.id
        return ""


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
    """A first read (or a rewrite's forced rescan) of the whole journal.

    Pauses between rows when run as a background refresh (``_YIELD_TO``, see
    "Yielding to other opens"); the rows and their order are unaffected.
    """
    yield_to = _YIELD_TO.get()
    reader = _RowReader(path, 0, 0)
    state = _Derivation()
    journal_rows: dict[str, int] = {}
    journal_ranges: dict[str, tuple[int, int]] = {}
    source_facts = []
    for row in reader:
        if yield_to is not None:
            yield_to()
        source_facts.append(row.source_fact or source_projection_fact(None, row.ordinal))
        if row.id:
            journal_rows[row.id] = row.ordinal
            journal_ranges[row.id] = (row.offset, row.end)
        state.consume(row)
    _assert_same_file(reader)
    checkpoints, messages, runs = state.emit(prior_checkpoints=[], prior_messages=[], turn_base=0)
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
        journal_generation=uuid4().hex,
        journal_rows=journal_rows,
        journal_ranges=journal_ranges,
        source_projection_facts=source_facts,
        naming=naming,
        runs=runs,
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
    yield_to = _YIELD_TO.get()
    window_offset = previous.scan.window_offset
    window_rows = previous.scan.window_rows
    keep_checkpoints = [c for c in previous.checkpoints if c.seq < window_rows]
    keep_messages = [m for m in previous.messages if m.seq < window_rows]
    turn_base = sum(1 for c in keep_checkpoints if c.kind == KIND_USER)
    # A RUN CAN STRADDLE THE WINDOW, and it is the one section that cannot simply
    # be cut at it: the window is chosen for the RAIL, and the run that contains it
    # may open at a user row ABOVE the window (the window then names a steer's own
    # ``attention_started``), so that run is re-derived WHOLE — its record is
    # handed to the derivation as the open run and the kept prefix stops before
    # it. Cutting the keeps at ``window_rows`` instead would emit the straddling
    # run twice, once truncated and once head-cut, and the two counts would not
    # reconcile with any surface's ledger.
    straddle = next((r for r in previous.runs if r.last_seq >= window_rows), None)
    # ONLY A RUN THE REGION STARTS INSIDE IS CARRIED. When the window sits ABOVE
    # the run's opening user row the region re-derives the whole run from its
    # first row, and carrying the prefix's counters as well would count those
    # rows twice — measured as a run of 2 actions and 5 s reporting 4 and 10 s.
    carry = straddle if straddle is not None and window_rows > straddle.first_seq else None
    cut = carry.first_seq if carry is not None else window_rows
    keep_runs = [r for r in previous.runs if r.last_seq < cut]
    reader = _RowReader(path, window_offset, window_rows)
    state = _Derivation(carry=carry)
    # This is a suffix RE-READ, not merely an append. Retain only the prefix
    # before the reader's ordinal resume point; derive suffix ids in this same
    # verified view so stale suffix mappings cannot survive a re-derivation.
    journal_rows = {id_: seq for id_, seq in previous.journal_rows.items() if seq < window_rows}
    journal_ranges = {
        id_: span for id_, span in previous.journal_ranges.items() if id_ in journal_rows
    }
    source_facts = [fact for fact in previous.source_projection_facts if fact["seq"] < window_rows]
    for row in reader:
        if yield_to is not None:
            yield_to()
        source_facts.append(row.source_fact or source_projection_fact(None, row.ordinal))
        if row.id:
            journal_rows[row.id] = row.ordinal
            journal_ranges[row.id] = (row.offset, row.end)
        state.consume(row)
    _assert_same_file(reader)
    checkpoints, messages, runs = state.emit(
        prior_checkpoints=keep_checkpoints,
        prior_messages=keep_messages,
        turn_base=turn_base,
        prior_runs=keep_runs,
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
        journal_generation=previous.journal_generation,
        journal_rows=journal_rows,
        journal_ranges=journal_ranges,
        source_projection_facts=source_facts,
        naming=naming,
        runs=runs,
        scan=ScanState(
            # THE JOURNAL'S TOTAL, NOT THIS READ'S: the reader re-read only the
            # suffix starting at ordinal ``window_rows``, so the rows before it
            # (kept, not re-read) are added back. ``open_frame.tail_head_reachable``
            # subtracts a run's ``first_seq`` from this count, and the
            # suffix-only count (6,973 against 10,377 on the 51.7 MB fixture)
            # made the head hunt run on, so the frame depended on how the
            # index was refreshed (O1, #3034 QA). Same total ``_scan_full`` records.
            rows=window_rows + reader.rows,
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

    A background refresh started by :func:`start_refresh` runs with
    :data:`_YIELD_TO` set, and this calls it at every pause point -- between rows
    and before each whole-document JSON step -- so a CPU-bound scan in a worker
    thread steps aside while another conversation's first frame is being built
    (see "Yielding to other opens"). It never changes what is built, only when
    the thread holds the GIL. A synchronous caller sees no pause at all.

    The whole invalidation ladder, in order: an exact (size, mtime, inode)
    match reuses; a GROWN file from the SAME inode whose recorded tail still
    verifies increments; anything else — including a compact rewrite, whose
    ``os.replace`` changes the inode — rescans. One retry covers a concurrent
    ``compact_file`` (which replaces the file, changing the inode under the
    read).
    """
    path = _journal_path(config_dir, session_id)
    yield_to = _YIELD_TO.get()
    for attempt in (0, 1):
        # Pause BEFORE the stat (review round 1, MINOR-3): a pause between them
        # would let ``st`` age by up to the yield budget before it is compared
        # with the cache's signature.
        if yield_to is not None:
            yield_to()
        try:
            st = path.stat()
        except OSError:
            return None
        raw = _read_raw(config_dir, session_id)
        previous = _index_from_raw(raw)
        if previous is not None and previous.journal_generation and _sig_matches(previous.sig, st):
            from local_operator.session import index_read_model

            # Exact HITs never reach write_index. Reuse this parsed snapshot for
            # migration; the async refresher schedules it on return from workers.
            index_read_model.schedule_materialize(config_dir, session_id, previous)
            return previous
        try:
            if (
                previous is not None
                and bool(previous.journal_generation)
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
        if yield_to is not None:
            # The last pause sits ABOVE the naming re-read, never between it and
            # the write (review round 1, MAJOR-1): a name saved during a pause
            # there would be overwritten by the document below, reopening the
            # lost-write window the merge exists to close. Pausing here also
            # keeps the ``json.dumps`` (one uninterruptible C call, ~20-60 ms for
            # the 51.7 MB fixture's 4.5 MB index) out of another open's frame.
            yield_to()
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
    #: The journal's size when it could be stat'ed. ``checkpoints_view`` reads it
    #: to decide whether the first-paint budget can pay for the scan at all; the
    #: probe already stats the journal, so this costs no second syscall.
    journal_bytes: int | None = None


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
        return IndexProbe(state="ready", index=index, built_at=built_at, journal_bytes=st.st_size)
    return IndexProbe(state="stale", index=index, built_at=built_at, journal_bytes=st.st_size)


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


def scan_payable_in_budget(
    journal_bytes: int | None,
    *,
    has_previous: bool,
    wait_s: float,
    rederive_bytes: int | None = None,
    hydration_bytes: int = 0,
) -> bool:
    """Whether a refresh can plausibly land inside a first-paint wait of ``wait_s``.

    THE ONE RULE for "is this scan worth paying for on the paint path", shared
    by :func:`checkpoints_view`'s first-paint wager and the paint hold below so
    the two can never disagree about which journals are small enough. Every
    byte figure is planned against :data:`_FIRST_PAINT_SCAN_BYTES_PER_S` (see
    that constant for the measurement and why it is not a fraction of it).

    WITH A PREVIOUS INDEX THE FIGURE IS THE RE-READ, NOT ZERO
    (conversation-first-paint R2, 2026-10-27). The incremental path re-derives
    from the resume window -- the start of the run holding the last user turn --
    so a long agent run puts the window arbitrarily far back: on the 51.7 MB
    fixture a ONE-ROW append re-read 37.9 MB / 6,973 rows at 304-637 ms CPU,
    more than its cold scan, and the old "a previous index is always payable"
    let that run straight through the paint hold (append opens 351-375 ms under
    load; 101 ms with the scan deferred). ``rederive_bytes`` is that re-read;
    ``None`` means the caller could not state it (a cache laid out before the
    resume head) or chose not to (the manifest wager -- see the note at its call
    in :func:`checkpoints_view`), preserving the old unknown-rederivation
    allowance. Callers that have proven a complete request projection can use
    ``hydration_bytes`` to account its avoided disk work. Generic shared-flight
    predicates deliberately retain original admission (see _defer_for_paint).
    """
    # Hydration and rederivation spend ONE budget, not two independent wagers.
    # Use the existing planning rate for both; this is a scheduling estimate,
    # not a promise about wall time on a contended host. An exact disk signature
    # removes only journal work; never infer zero hydration from that signature.
    budget = _FIRST_PAINT_SCAN_BYTES_PER_S * wait_s
    if has_previous:
        return hydration_bytes + (rederive_bytes or 0) <= budget
    return journal_bytes is not None and hydration_bytes + journal_bytes <= budget


# ---------------------------------------------------------------------------
# The paint hold: a cold full scan never competes with a desktop first frame
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS (CORE-PROFILE L2, 2026-10-25). A cold open of an 8-100 MB
# journal started ``_scan_full`` from inside the open itself (``_frame_index``,
# the journal-order correction, the parallel ``/checkpoints``), and that scan
# spent 806-1,277 ms of CPU under the GIL WHILE the snapshot was being built --
# a scan that cannot finish inside the 120 ms snapshot budget anyway, so the
# frame was ``building`` either way and the only effect of starting it early was
# a slower first paint. The desktop ``/events`` route therefore holds the session
# from its entry until the snapshot frame has been written, and a refresh that
# would be a COLD, UNPAYABLE full scan (by :func:`scan_payable_in_budget`, the
# same rule the manifest's wager uses) waits for that release before it starts.
#
# WHAT IT NEVER CHANGES. A payable refresh (small journal, or a previous index
# that makes it incremental) is never deferred, so every frame that was
# ``ready`` on first paint stays ``ready``; an unpayable one was ``building`` on
# that frame already, so deferring it changes no response, only who pays the
# CPU when. Only the desktop events route takes a hold: the TUI, the mobile
# relay and prewarm never do, so in a process with no desktop open nothing here
# runs (``_PAINT_HOLDS`` is empty and the check is one dict lookup).
#
# HOW IT CANNOT WEDGE. Holds are REFCOUNTED per session (two concurrent opens of
# one conversation both hold; the scan starts when the LAST one releases), each
# release function is idempotent, and the deferred task stops waiting after
# :data:`_PAINT_HOLD_CEILING_S` whatever happens -- so a hold leaked by a path
# nobody anticipated costs one deferral of that length, never a scan that never
# starts. No caller ever waits UNBOUNDED on a deferred task while holding: the
# open frame asks :func:`refresh_deferred` and answers without facts instead of
# spending its budget on a task that cannot run until it returns, and the
# manifest never waits on an unpayable build at all.

#: The longest a deferred scan waits for a paint hold to be released. A desktop
#: snapshot lands in tens of ms warm and a few seconds cold on a loaded host;
#: this is the backstop for a hold whose release was somehow missed, not a
#: budget anyone is expected to reach.
_PAINT_HOLD_CEILING_S = 15.0

#: In-process writer serialisation per publish-lock path. USED BY EVERY PLATFORM:
#: the advisory lock is cross-PROCESS, this one is cross-THREAD, and neither
#: replaces the other (a thread cannot flock its own descriptor twice safely).
_PUBLISH_THREADS: dict[str, threading.Lock] = {}

#: Live hold counts per session key. LOOP-ONLY MUTATION, like ``_IN_FLIGHT``.
_PAINT_HOLDS: dict[tuple[str, str], int] = {}
#: When each live hold's count last rose from zero (``time.monotonic``). Only
#: :func:`_yield_to_other_opens` reads it, to ignore a hold too old to be a first
#: frame still being built. Loop-only mutation, like ``_PAINT_HOLDS``.
_PAINT_HOLD_SINCE: dict[tuple[str, str], float] = {}
#: The event a deferred refresh waits on, set when the key's count reaches zero.
#: Keyed with its loop so a torn-down test loop reads as absent.
_PAINT_RELEASED: dict[tuple[str, str], tuple[asyncio.AbstractEventLoop, asyncio.Event]] = {}
#: Keys whose in-flight refresh is currently waiting on a hold (see
#: :func:`refresh_deferred`).
_DEFERRED: set[tuple[str, str]] = set()


def hold_paint(config_dir: str | Path, session_id: str) -> Callable[[], None]:
    """Take a paint hold on one session; returns its idempotent release.

    Call from the event loop. The returned function may be called any number of
    times from any exit path (snapshot written, error, disconnect, generator
    close) and releases exactly once, which is what lets the route call it from
    every one of those paths without counting.
    """
    key = _key(config_dir, session_id)
    loop = asyncio.get_running_loop()
    if not _PAINT_HOLDS.get(key):
        _PAINT_HOLD_SINCE[key] = time.monotonic()
    _PAINT_HOLDS[key] = _PAINT_HOLDS.get(key, 0) + 1
    entry = _PAINT_RELEASED.get(key)
    if entry is None or entry[0] is not loop or entry[1].is_set():
        _PAINT_RELEASED[key] = (loop, asyncio.Event())
    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        remaining = _PAINT_HOLDS.get(key, 1) - 1
        if remaining > 0:
            _PAINT_HOLDS[key] = remaining
            return
        _PAINT_HOLDS.pop(key, None)
        _PAINT_HOLD_SINCE.pop(key, None)
        current = _PAINT_RELEASED.pop(key, None)
        if current is not None:
            current[1].set()

    return release


# ---------------------------------------------------------------------------
# Yielding to other opens: a background scan never starves another first frame
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS (conversation-first-paint, residual 2, 2026-10-27). The paint
# hold above protects an open from ITS OWN cold scan. It does nothing for the
# NEXT conversation: switching from the 51.7 MB journal to the 8 MB legacy one
# released the first hold at its snapshot, the deferred scan started 0.2 ms
# later in a worker thread, and the legacy open -- loop-thread Python the whole
# way (facade build, attention, page reads) -- then fought that thread for the
# GIL: 298-306 ms back-to-back (5/5, real ``lop serve``) against 28-40 ms opened
# first or after a 3 s gap. Attribution, same rig, injected in place of the
# scan's first 0.5 s: ``time.sleep`` (thread and executor slot occupied, GIL
# free) -> legacy 29-32 ms; a busy loop (GIL held) -> 707-739 ms. The executor
# is not the contended resource, and no lock is: the GIL is. Python's 5 ms
# switch interval hands the loop thread only a share of the interpreter, so
# every millisecond of the open stretched by the scan's share. With this rule:
# legacy-after-real 76-122 ms (median 103) against base 308-417 (median 347),
# 6/6 interleaved reps at load ~38; the real open and the frames unchanged, and
# the real index reaches ``ready`` ~50 ms later (median 402 vs 350 ms after its
# snapshot) because it gave way. The open's own ``a.acquire`` fell from 143-182
# to 16-22 ms. The residual above the 28-40 ms floor lies BEFORE the route takes
# its hold: previous snapshot to this route's entry is 51-86 ms (base 117-144),
# which includes the client's new request and the server's parse of it, both
# still sharing the GIL with the scan -- nothing marks an open as in flight
# earlier than ``hold_paint``.
#
# THE RULE. A background refresh pauses at row boundaries (and before the cache
# read and the naming merge + document write) while any paint hold younger than
# :data:`_YIELD_HOLD_STALE_S` is live -- its own session's included -- sleeping
# in short slices that release the GIL.
#
# ITS OWN OPEN TOO (own-open-yield, 2026-10-28). #3026 exempted the scan's own
# session, reasoning that an open waiting on its own refresh would wait on
# itself. The only own-refresh waiter is ``checkpoints_view``'s
# ``_FIRST_PAINT_WAIT_S`` wait, and it is spent only on a PAYABLE scan, which
# the rule below never pauses; an unpayable open answers ``building`` without
# awaiting the scan. So the exemption prevented no cycle, and it put the cost
# #3026 removed from the switch opens onto the WARM RE-OPEN: re-opening a
# conversation whose cold scan was still running shared the GIL with that scan
# (desktop legacy warm-after-miss 12 ms on v0.90.19, 447-664 ms medians after
# #3026; bisect in conversation-first-paint/BISECT-warm-after-miss). Cost: that
# conversation's ``ready`` lands up to the pause later, as for any other open.
#
# ONLY UNPAYABLE SCANS YIELD (review round 1, MAJOR-2). ``checkpoints_view``
# awaits a refresh it starts for up to ``_FIRST_PAINT_WAIT_S`` when the scan is
# PAYABLE in that budget, and its consumers do not re-read a ``building``
# answer (d560b2ec6a8 removed exactly that regression). A payable scan that
# paused would turn those ``ready`` answers into ``building`` or stale ones, so
# a scan yields only when :func:`scan_payable_in_budget` -- the same rule the
# view and the paint hold use -- says no wait could have paid for it anyway. A
# payable scan is short by that rule's own definition; the contention this
# exists for is the cold multi-MB scan (the 51.7 MB journal's deferred one).
# Decided lazily, in the worker, the first time another hold is seen: two
# ``stat`` calls there, none on the loop and none when nobody is opening.
#
# WHAT IT NEVER CHANGES. The rows consumed, their order and the document
# written are identical; only wall time moves, and only while another open is
# mid-first-frame. The task still settles exactly once, through ``_wrapped``.
# It does REORDER work: the next conversation's own deferred scan now starts
# while this one is still running, so this one's ``ready`` lands later (QA
# round 1: cache written ~470 ms later, polled ``ready`` ~280 ms later) while
# both conversations' indexes finish at the same time as before.
#
# HOW IT CANNOT WEDGE. Paint holds are released on every exit of the events
# route, but a leaked one (or a user switching conversations nonstop) must not
# stall indexing: a scan's cumulative pause is capped at :data:`_YIELD_BUDGET_S`,
# and a hold older than :data:`_YIELD_HOLD_STALE_S` is ignored outright, so a
# leak costs at most the scans started inside its first seconds, never every
# later scan until restart (review round 1, MINOR-2).
#
# THREAD SAFETY. The check runs in the worker thread and only READS the hold
# tables: ``len``/``in`` on ``_PAINT_HOLDS`` and a ``dict.copy()`` of
# ``_PAINT_HOLD_SINCE`` -- each a single C-level operation that holds the dict
# for its whole duration, so the loop's concurrent mutation can never be seen
# mid-iteration. It never writes them.

#: Sleep slice while yielding: short enough that the scan resumes within a few
#: ms of the other open's snapshot, long enough to hand the GIL over outright.
_YIELD_SLICE_S = 0.002
#: Ceiling on one refresh's total pause. A desktop first frame is tens to a few
#: hundred ms; this only bounds a leaked hold or a nonstop run of opens.
_YIELD_BUDGET_S = 2.0
#: A hold older than this is not a first frame still being built (a cold one on
#: a loaded host is a few seconds at worst); it is ignored, not yielded to.
_YIELD_HOLD_STALE_S = 5.0


#: The pause callable the CURRENT background refresh honours; ``None`` (every
#: synchronous caller: the TUI, peer cycle display, tests) means never pause.
_YIELD_TO: contextvars.ContextVar[Callable[[], None] | None] = contextvars.ContextVar(
    "transcript_index_yield_to", default=None
)


def _yield_to_other_opens(
    own: tuple[str, str], config_dir: str | Path, session_id: str
) -> Callable[[], None]:
    """The ``yield_to`` callable for one background refresh of ``own``.

    ``own`` is unused since the own-session exemption went ("ITS OWN OPEN TOO");
    the signature is kept so callers and tests that name the refresh's key stay
    unchanged.
    """
    spent = 0.0
    #: ``None`` until another hold is first seen; then whether this scan may yield.
    unpayable: bool | None = None

    def an_open_is_young() -> bool:
        # ``None``, not ``own``: the scan's own warm re-open must win too (see
        # "ITS OWN OPEN TOO"); ``may_yield`` is what keeps a payable scan, the
        # one ``checkpoints_view`` awaits, from ever pausing for it.
        return _young_hold_besides(None)

    def may_yield() -> bool:
        nonlocal unpayable
        if unpayable is None:
            journal = _stat_or_none(_journal_path(config_dir, session_id))
            if journal is None:
                unpayable = False
            else:
                has_previous, rederive, _hydration = _cached_refresh_cost(
                    config_dir, session_id, journal
                )
                unpayable = not scan_payable_in_budget(
                    journal.st_size,
                    has_previous=has_previous,
                    rederive_bytes=rederive,
                    wait_s=_FIRST_PAINT_WAIT_S,
                )
        return unpayable

    def pause() -> None:
        nonlocal spent
        while spent < _YIELD_BUDGET_S and an_open_is_young() and may_yield():
            # Measured, not counted: on a loaded host a 2 ms sleep can overrun
            # several-fold, and the budget is a wall-clock promise.
            started = time.monotonic()
            time.sleep(_YIELD_SLICE_S)
            spent += time.monotonic() - started

    return pause


def _young_hold_besides(own: tuple[str, str] | None) -> bool:
    """Whether a paint hold younger than the staleness cap is live for any key but ``own``."""
    holds = _PAINT_HOLDS
    if len(holds) <= (1 if own is not None and own in holds else 0):
        return False  # the common case: one len, no copy
    cutoff = time.monotonic() - _YIELD_HOLD_STALE_S
    return any(k != own and t > cutoff for k, t in _PAINT_HOLD_SINCE.copy().items())


def yield_to_any_open() -> Callable[[], None]:
    """A pause for work that NO first frame waits on: the library indexer (R3).

    WHY (conversation-first-paint, residual tail R3, 2026-10-27). The library
    indexer's thread reacts to a journal's FSEvents by scanning it at once --
    which is precisely when the user is opening it (a conversation that just
    synced, was copied in, or finished a turn). It is not a ``start_refresh``
    task, so the per-refresh rule above never applied: on a real ``lop serve``
    at load 30-80 it spent a median 264 ms of GIL-holding CPU inside the open
    of the 51.7 MB journal it was scanning (miss shape, 301 ms median; with this
    pause, spike ``libyield``, 166 ms).

    THE SAME RULE, MINUS THE PAYABLE EXEMPTION. Same slice, same cumulative
    :data:`_YIELD_BUDGET_S` per returned pause (one indexer batch), same
    :data:`_YIELD_HOLD_STALE_S` cap on a leaked hold, and -- like the refresh
    rule since "ITS OWN OPEN TOO" -- the open of the very journal being scanned
    wins as well (that is the measured case). The refresh rule's "payable"
    exemption does not apply, deliberately: no
    caller holds a budget on this thread's progress (the request-path ledger
    reads never install a pause, and a lock they lose to a paused batch answers
    from the last published scan, its documented non-blocking behaviour).
    Content is unaffected: lines are consumed in the same order, only later.
    """
    spent = 0.0

    def pause() -> None:
        nonlocal spent
        while spent < _YIELD_BUDGET_S and _young_hold_besides(None):
            started = time.monotonic()
            time.sleep(_YIELD_SLICE_S)
            spent += time.monotonic() - started

    return pause


def refresh_deferred(config_dir: str | Path, session_id: str) -> bool:
    """Whether this session's in-flight refresh is waiting on a paint hold.

    The open frame's question: a task in this state cannot finish before the
    hold its own request owns is released, so awaiting it would only spend the
    snapshot budget idle.
    """
    return _key(config_dir, session_id) in _DEFERRED


def refresh_paused_for_paint(config_dir: str | Path, session_id: str) -> bool:
    """Whether awaiting this session's refresh from inside its own open is futile.

    :func:`refresh_deferred`, plus the case "ITS OWN OPEN TOO" created: a refresh
    ALREADY RUNNING when the same conversation is re-opened. Its scan pauses for
    as long as this open's paint hold is young, so it cannot land inside the
    snapshot budget, and the open frame's ``asyncio.wait`` would idle the whole
    remaining budget for the same ``building`` answer -- measured on a real
    ``lop serve`` as 112-119 ms of the 120 ms budget, the bulk of what the warm
    re-open still cost once the scan stopped taking its GIL.

    Mirrors the pause's own predicate, so the two cannot disagree: a live hold
    for this key younger than :data:`_YIELD_HOLD_STALE_S`, and a scan that
    :func:`scan_payable_in_budget` says no first-paint wait could have paid for
    (a payable scan never pauses, and waiting on it still pays). Loop thread; two
    ``stat`` calls, and only while this session holds a paint.
    """
    if refresh_deferred(config_dir, session_id):
        return True
    key = _key(config_dir, session_id)
    since = _PAINT_HOLD_SINCE.get(key)
    if since is None or time.monotonic() - since > _YIELD_HOLD_STALE_S:
        return False
    entry = _IN_FLIGHT.get(key)
    if entry is None or entry[0] is not asyncio.get_running_loop() or entry[1].done():
        return False
    journal = _stat_or_none(_journal_path(config_dir, session_id))
    if journal is None:
        return False
    has_previous, rederive, _hydration = _cached_refresh_cost(config_dir, session_id, journal)
    return not scan_payable_in_budget(
        journal.st_size,
        has_previous=has_previous,
        rederive_bytes=rederive,
        wait_s=_FIRST_PAINT_WAIT_S,
    )


def _defer_for_paint(config_dir: str | Path, session_id: str) -> asyncio.Event | None:
    """The hold event a NEW refresh must wait on, or ``None`` to start now.

    Decided synchronously when the task is created, so :func:`refresh_deferred`
    is already true for the caller that just started it. Costs nothing when no
    hold is live (one dict lookup -- every non-desktop process); otherwise two
    ``stat`` calls, the same cost ``fresh_resident`` already pays on the loop.
    """
    key = _key(config_dir, session_id)
    if not _PAINT_HOLDS.get(key):
        return None
    entry = _PAINT_RELEASED.get(key)
    if entry is None or entry[0] is not asyncio.get_running_loop():
        return None
    journal = _stat_or_none(_journal_path(config_dir, session_id))
    if journal is None:
        return None
    # Keep ORIGINAL fallback admission here. Hydration is real work (reported
    # separately by the cost helper), but this shared flight does not know which
    # complete facts its callers acquired. A summary/previous successful read is
    # not enough to strand a later corrupt/stale fallback behind its own hold.
    # Ready projection consumers return their materialized result WITHOUT
    # scheduling hydration; only that request has the capability proof.
    has_previous, rederive, _hydration = _cached_refresh_cost(config_dir, session_id, journal)
    if scan_payable_in_budget(
        journal.st_size,
        has_previous=has_previous,
        rederive_bytes=rederive,
        wait_s=_FIRST_PAINT_WAIT_S,
    ):
        return None
    return entry[1]


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


def _journal_order_doc(
    index: TranscriptIndex | FrameSelection, entry_id: str, seq: int, role: str
) -> MessageDoc | None:
    """Exact existing doc at a raw ordinal; no body/history scan or role guess."""
    if type(seq) is not int or seq < 0 or index.journal_rows.get(entry_id) != seq:
        return None
    at = bisect_left(index.messages, seq, key=lambda doc: doc.seq)
    if at == len(index.messages):
        return None
    doc = index.messages[at]
    return (
        doc
        if doc.id == entry_id and doc.seq == seq and doc.role == role and not doc.injected
        else None
    )


def journal_order_opener(
    index: TranscriptIndex | FrameSelection, entry_id: str, seq: int
) -> tuple[str, int] | None:
    """The genuine run opener of an exact source assistant, not a viewer guess.

    Sparse ready corrections previously stated an assistant's coordinate while
    its optimistic opening user still had none. The desktop then correctly put
    the known assistant before the unknown suffix. Close that metadata dependency
    with the user's REAL coordinate; do not invent its slot, re-rank the known
    reply, or infer a parent from a last-user/admission/frame/generation scan.

    MessageDoc.seq, RunRecord.first_seq/last_seq and journal_rows all derive from
    the SAME _Row.ordinal, which counts every complete journal line (including
    invisible/prune/tool rows). These are not display/turn ordinals. Exact role,
    raw ID, membership and opener-first equality are each necessary: run_key can
    name a tool/marker, a steer does not become a new opener, and head-cut/injected
    inputs have no genuine opening user to contribute. The existing source
    signature/freshness gate belongs to the caller; this helper never grants it.

    Both sequences are scan ordered. Keyed bisects read only a bounded number of
    existing records, with no whole-history walk, new cache, body or journal read.
    Missing proof (including an old cache without docs/runs) contributes nothing.
    """
    if not index.journal_generation or type(seq) is not int or seq < 0:
        return None
    if _journal_order_doc(index, entry_id, seq, "assistant") is None:
        return None
    run_at = bisect_right(index.runs, seq, key=lambda run: run.first_seq) - 1
    if run_at < 0:
        return None
    run = index.runs[run_at]
    if not (run.first_seq <= seq <= run.last_seq) or not run.opening_user_id:
        return None
    opener_id = run.opening_user_id
    opener_seq = index.journal_rows.get(opener_id)
    if not isinstance(opener_id, str) or not 1 <= len(opener_id) <= 128:
        return None  # the existing source-metadata raw-ID wire bound, not truncation
    if type(opener_seq) is not int or opener_seq != run.first_seq or not (0 <= opener_seq < seq):
        return None
    if _journal_order_doc(index, opener_id, opener_seq, "user") is None:
        return None
    return opener_id, opener_seq


def journal_order_correction_batches(
    index: TranscriptIndex, rows: list[dict[str, Any]]
) -> Iterator[list[dict[str, Any]]]:
    """Pack existing ready metadata into <=128-row dependency-closed events.

    Call only after every descriptor's original page signature matches this fresh
    index, as DesktopSessionBridge's correction already requires. A newly known
    assistant travels with its exact run-opening user's coordinate in the SAME
    event: flushing 127 rows before a pair is intentional. Re-include the opener
    in each new event rather than globally deduplicating it, since a viewer may
    join/replay from a later receipt. No user body/new record is published here.
    """
    # Also send already-supplied genuine user facts first. A requested steer U2
    # is not the run opener, but splitting its true coordinate behind A2 at a
    # frame boundary would repeat the race. This partitions only bounded INPUT
    # descriptors by exact source proof; it never selects a last user from history.
    # Ordering user rows ahead of the others is deliberate: the viewer orders by
    # the stored journal_seq, not by arrival, so this only matters for the ONE
    # event boundary a flush can fall on — and the opener is re-prepended into
    # each later event, so a split pair is still emitted user-first there too.
    users: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for row in rows:
        entry_id = row.get("id")
        seq = row.get("journal_seq")
        # The `type(seq) is int` guard is required for type narrowing only: the
        # underlying lookup already rejects non-int ordinals, so it adds no
        # authority — it just lets the callers below type-check.
        is_user = (
            isinstance(entry_id, str)
            and type(seq) is int
            and _journal_order_doc(index, entry_id, seq, "user") is not None
        )
        (users if is_user else others).append(row)
    chunk: list[dict[str, Any]] = []
    in_chunk: set[str] = set()
    for row in users + others:
        entry_id = row.get("id")
        seq = row.get("journal_seq")
        opener = (
            journal_order_opener(index, entry_id, seq)
            if isinstance(entry_id, str) and type(seq) is int
            else None
        )
        dependency = (
            {"id": opener[0], "type": "message", "role": "user", "journal_seq": opener[1]}
            if opener is not None
            else None
        )
        unit = (
            [dependency] if dependency is not None and dependency["id"] not in in_chunk else []
        ) + [row]
        if len(chunk) + len(unit) > 128:
            yield chunk
            chunk = []
            in_chunk.clear()
            # A dependency present in the old chunk is not proof in this event.
            unit = ([dependency] if dependency is not None else []) + [row]
        chunk.extend(unit)
        in_chunk.update(item["id"] for item in unit)
    if chunk:
        yield chunk


def apply_journal_order(
    root: Path,
    session_id: str,
    answer: dict[str, Any],
    source: tuple[int, int, float] | None,
    ids: list[str],
) -> bool:
    """Join owner coordinates only when page and resident index share a witness.

    Both desktop-local and peer-stored reads must use this equality check:
    current disk state alone cannot attest rows read before a concurrent rewrite.
    False leaves the page untouched; each transport owns its readiness policy.
    """
    index = fresh_resident(root, session_id)
    if index is None or not index.journal_generation or source is None:
        return False
    if source != (index.sig.get("inode"), index.sig.get("size"), index.sig.get("mtime")):
        return False
    if any(row["id"] not in index.journal_rows for row in answer["entries"]):
        return False
    answer["entries"] = [
        {**row, "journal_seq": index.journal_rows[row["id"]]} for row in answer["entries"]
    ]
    answer["journal_order_state"] = "ready"
    answer["journal_generation"] = index.journal_generation
    answer["journal_ordinals"] = {
        id_: index.journal_rows[id_] for id_ in ids if id_ in index.journal_rows
    }
    # The synchronous ready page has the same sparse-A race as the asynchronous
    # correction. Carry actual opener coordinates as bounded source sidecars,
    # never user bodies: at most one extra ID per bounded page/request identity.
    # Missing-ID reporting still describes ONLY the IDs the caller requested.
    for id_ in dict.fromkeys([row["id"] for row in answer["entries"]] + ids):
        seq = index.journal_rows.get(id_)
        if seq is None:
            continue
        opener = journal_order_opener(index, id_, seq)
        if opener is not None:
            answer["journal_ordinals"][opener[0]] = opener[1]
    answer["journal_missing_ids"] = [id_ for id_ in ids if id_ not in index.journal_rows]
    return True


def fresh_resident(config_dir: str | Path, session_id: str) -> TranscriptIndex | None:
    """The resident index for a session, but only while it is still CURRENT.

    THE HOT-PATH DOOR, and it exists because the open cannot pay a scan. The
    resident index is revalidated against one stat of the journal plus the cache
    file's own mtime — the same two facts ``checkpoints_view``'s fast path reads
    — so a caller on a paint path gets either an index whose ``runs`` describe
    the journal as it is now, or ``None``. It never scans, never reads the cache
    file and never waits: the caller decides what to do with ``None`` (the open
    frame starts a refresh and answers without facts — see ``open_frame``).

    Synchronous on purpose. Its two ``stat`` calls are microseconds, and every
    caller runs in a worker thread (the page builder is invoked through
    ``asyncio.to_thread``), so there is no event loop to block — and keeping it
    sync is what lets it share the revalidation rule with the async callers
    instead of re-implementing it around ``await``.
    """
    key = _key(config_dir, session_id)
    index = _RESIDENT.get(key)
    if index is None:
        return None
    st, cache_st = _freshness_pair(config_dir, session_id)
    stamp = _RESIDENT_STAMP.get(key)
    cache_stable = (stamp is None and cache_st is None) or (
        stamp is not None and cache_st is not None and stamp == cache_st.st_mtime
    )
    if st is not None and _sig_matches(index.sig, st) and cache_stable:
        _RESIDENT.move_to_end(key)
        return index
    return None


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

    hold = _defer_for_paint(config_dir, session_id)
    if hold is not None:
        _DEFERRED.add(key)

    async def _run() -> TranscriptIndex | None:
        if hold is not None:
            # See "The paint hold" above: an unpayable cold scan starts once the
            # desktop's first frame is written, never later than the ceiling.
            try:
                await asyncio.wait_for(hold.wait(), _PAINT_HOLD_CEILING_S)
            except asyncio.TimeoutError:
                logger.warning(
                    "transcript index refresh for session %s waited the paint-hold "
                    "ceiling (%.0f s); starting it anyway",
                    session_id,
                    _PAINT_HOLD_CEILING_S,
                )
            finally:
                _DEFERRED.discard(key)
        if open_trace.enabled():
            # The evidence row for the paint hold: its timestamp against the
            # route's ``a.wire.frame kind=snapshot`` row says whether the scan
            # started before or after the first frame was written.
            open_trace.mark("index.refresh_start", session=session_id, deferred=hold is not None)
        # Set in THIS task's own context, which ``to_thread`` copies into the
        # worker: only this refresh yields, and ``refresh_index``'s signature
        # (which tests and callers substitute) stays two-argument.
        _YIELD_TO.set(_yield_to_other_opens(key, config_dir, session_id))
        result = await asyncio.to_thread(refresh_index, config_dir, session_id)
        if result is not None:
            from local_operator.session import index_read_model

            index_read_model.schedule_materialize(config_dir, session_id, result)
        return result

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
            # A task cancelled before its body ran never reached the ``finally``
            # that clears the deferred mark; a stale mark would make the open
            # frame skip a wait it could pay.
            _DEFERRED.discard(key)

    task.add_done_callback(_drop)
    return task, True


def resolve_naming_items(
    checkpoints: list[Checkpoint], naming: Any
) -> list[tuple[Checkpoint, Any]]:
    """Pair each checkpoint with the raw naming item its turn owns.

    THE ONE SPELLING of the turn policy, shared by the canonical manifest and the
    read model: a completion belongs to the LAST user checkpoint of its turn, and
    only a prompt-version-matching section supplies items at all.
    """
    section = naming if isinstance(naming, dict) else {}
    items = section.get("items") if section.get("prompt_version") == NAMING_PROMPT_VERSION else {}
    items = items if isinstance(items, dict) else {}
    users = {c.turn: c.id for c in checkpoints if c.kind == KIND_USER}
    return [
        (c, items.get(users.get(c.turn)) if c.kind == KIND_COMPLETION else None)
        for c in checkpoints
    ]


def checkpoint_entry(checkpoint: Checkpoint, item: Any) -> dict[str, Any]:
    """Evaluate naming at read time: cooldown expiry needs no publication write."""
    from local_operator.session import checkpoint_naming

    entry = checkpoint.to_payload()
    if checkpoint.kind == KIND_COMPLETION:
        state = checkpoint_naming.naming_state(item)
        entry["naming"] = {
            "state": state,
            "name": item.get("name") if state == "ready" and isinstance(item, dict) else None,
            "summary": (
                (item.get("summary") or "") if state == "ready" and isinstance(item, dict) else None
            ),
        }
    return entry


def _manifest_state(
    session_id: str, state: str, index: TranscriptIndex | None, built_at: float | None
) -> dict[str, Any]:
    """The D9 wire shape for the checkpoints manifest."""
    checkpoints: list[dict[str, Any]] = []
    if index is not None:
        checkpoints = [
            checkpoint_entry(c, item)
            for c, item in resolve_naming_items(index.checkpoints, index.naming)
        ]
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
    from local_operator.session import index_read_model

    projected = await asyncio.to_thread(
        index_read_model.checkpoint_manifest, config_dir, session_id
    )
    if projected.state == "ready":
        return projected.value
    # Legacy, stale, corrupt and archive cases retain the correct prior payload
    # and its cost. A compact optimization is never authority to drop rail facts.
    probe = await asyncio.to_thread(probe_index, config_dir, session_id)
    if probe.state == "missing":
        # No journal: a draft or a session with nothing written yet. An empty
        # manifest is the honest answer, matching history's empty page.
        return _manifest_state(session_id, "ready", None, None)
    if probe.state == "ready":
        if probe.index is not None:
            _remember(key, probe.index, probe.built_at)
            index_read_model.schedule_materialize(config_dir, session_id, probe.index)
        return _manifest_state(session_id, "ready", probe.index, probe.built_at)

    failed_at = _FAILURES.get(key)
    if failed_at is not None and (time.monotonic() - failed_at) < _FAILURE_COOLDOWN_S:
        return _manifest_state(session_id, "error", probe.index, probe.built_at)
    task, started = start_refresh(config_dir, session_id)
    if started:
        # Retrieve a failed build's exception on every path that does not await
        # the task to completion. Both `building` answers do exactly that (the
        # unpayable budget below, and a wager that timed out), and an unobserved
        # failure would surface as a second, misleading traceback at GC. Same
        # shape as `desktop_sessions`' no-wait refresh.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        # FIRST PAINT: only the call that STARTS the build owes this wait — the
        # rail's loading state must be able to appear within the budget (design
        # D3) even while the build runs on. The wait is a wager that the refresh
        # lands inside the budget and this answer can be `ready`.
        #
        # A USABLE PREVIOUS INDEX MAKES THAT WAGER NEARLY FREE (agent review
        # round 1, MAJOR-1): with one, `refresh_index` takes the incremental
        # path for a journal that grew on the same inode (measured: 2.1-2.4 ms
        # for a one-row append), so `asyncio.wait` returns at the build's own
        # time rather than at the budget. Gating that on TOTAL bytes refused a
        # wait that always pays, and answered `building` where the old answer was
        # `ready` — which the consumer does not re-read. So the size gate is for
        # the COLD case only, where the build is a full scan and the wait really
        # does cost the whole budget.
        #
        # NO `rederive_bytes` HERE, DELIBERATELY (conversation-first-paint R2).
        # The paint hold judges an incremental refresh by its re-read, because
        # there a wrong "payable" puts a 300-640 ms scan on the GIL beside a
        # first frame. This wager's errors run the other way: an over-generous
        # wait is an idle `await` (no CPU, no paint) that answers `building` at
        # the budget, while a refused wait answers `building` where the scan
        # might have landed — the regression QA Q1 pinned. And a refresh the
        # hold defers OR pauses is never awaited: the shared pause predicate
        # below uses the same original rederivation admission as the worker.
        # Ready probes still return above; changing their acquisition requires
        # complete witnessed facts, not replacing that ready answer with building.
        payable = scan_payable_in_budget(
            probe.journal_bytes, has_previous=probe.index is not None, wait_s=wait_s
        )
        if payable and not refresh_paused_for_paint(config_dir, session_id):
            done, _pending = await asyncio.wait({task}, timeout=wait_s)
            settled = task in done
        else:
            settled = False
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
    _PAINT_HOLDS.clear()
    _PAINT_HOLD_SINCE.clear()
    _PAINT_RELEASED.clear()
    _DEFERRED.clear()
    # The read model's own in-process state is keyed by the same roots, so a
    # suite that re-creates a tmp_path must not inherit a memo or a remembered
    # publication failure from the previous one.
    from local_operator.session import index_read_model

    index_read_model.reset_for_tests()
