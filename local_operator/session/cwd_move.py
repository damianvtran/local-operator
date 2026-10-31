"""The durable intent behind a session's OWN working-directory move.

WHY A MARKER AND NOT A FIELD. A move is two acts in two processes: this session
ASKS (its tool call returns ``queued`` immediately) and its SUCCESSOR runtime
ACTS, by booting in the destination. The request therefore has to outlive the
asking process — a crash, a lost acknowledgement, or a retire that lands between
the two — so it is a document on disk, not state in memory.

THE STAGES ARE THE RECONCILIATION AXIS, and there are four:

* ``queued``     — asked for, not yet observed. The effective cwd is unchanged.
* ``committed``  — a successor was observed running in the destination.
* ``superseded`` — a later request replaced this one before it committed, OR the
  caller asked to move back to where the session already is, which is this
  module's CANCEL: it is the only request that can undo a queued one without
  minting a third destination (UX round 1, U4).
* ``expired``    — HELD for longer than :data:`QUEUED_MOVE_TTL_S`. A bounded
  life is what keeps an intent whose successor never came from blocking the
  session forever: without it, every boot would re-consume a request nothing
  satisfies (UX round 1, U4; agent review round 1, B2). The clock is the HOLD's
  (``blocked_at``), never the request's: see :func:`hold_lapsed`.

NOTHING HERE RETIRES A RUNTIME. This module owns the DOCUMENT; the runtime that
consumes it (``session/runtime/server._apply_queued_cwd_move``) owns the
retirement, and it refuses to retire toward a move it cannot guarantee a
successor for — leaving the intent ``queued`` is a legal, visible state rather
than a failure.

NOTHING ROLLS A DESTINATION BACK. Once ``committed``, the destination may be the
tree a successor is already running in, and "restoring" the old directory would
point the record at a tree nothing is using. This mirrors the desktop route's
transaction (``server/utils/desktop_sessions.py``: a move writes the marker and
the bridge field BEFORE the retire, and an UNKNOWN owner outcome is left
unconfirmed for reconciliation rather than restored) — the difference is that
this module owns a session's own request, and the successor is the one that has
to confirm it.

THE MARKER REUSES THE SESSION-DIRECTORY CONVENTION rather than inventing a
second format: one JSON document beside the other per-session documents
(``registry.write_stop_marker`` and its siblings), written through the same
staged-write the rest of that family uses, so the 0600-before-publish and
same-directory-rename contract has ONE implementation.

NO PROSE LIVES HERE. Every refusal is a technical ``code`` plus fields; the
schema explains what they mean (AGENTS.md, "Internationalization is mandatory").
That is also why this module needs neither the audited reader nor a translation
pass.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.session.runtime import registry
from local_operator.session.runtime.types import session_dir

#: The document's name in the session's own directory. Deliberately NOT a dotfile:
#: it is read by a successor runtime that has no memory of writing it.
MOVE_MARKER_NAME = "runtime-cwd-move.json"

#: See the module docstring. These strings are a closed set on purpose: a reader
#: that meets an unknown stage must treat the document as unusable rather than
#: guess which way the move went.
STAGE_QUEUED = "queued"
STAGE_COMMITTED = "committed"
STAGE_SUPERSEDED = "superseded"
STAGE_EXPIRED = "expired"
STAGES = frozenset({STAGE_QUEUED, STAGE_COMMITTED, STAGE_SUPERSEDED, STAGE_EXPIRED})

#: How long a ``queued`` intent may stay HELD before it expires.
#:
#: MEASURED FROM THE START OF THE HOLD, not from the request (UX round 2, U10).
#: A move is asked for inside a turn, and an agent turn that runs past half an
#: hour is ordinary here: aged from ``created_at``, the request expired before the
#: first idle point it could ever have acted at (reproduced: busy -> expired ->
#: idle, with nobody told). A runtime that is merely busy is not holding anything,
#: so the clock only runs while a consumer has said WHY it cannot act.
#:
#: BOUNDED ON PURPOSE. A queued intent is consumed by a runtime repeating "is
#: the successor here yet", and a request no successor ever satisfies would
#: otherwise re-arm on every boot for the life of the session — the state that
#: made a session unopenable (agent review round 1, B2). Thirty minutes is longer
#: than a turn the user is watching and shorter than "the operator forgot about
#: it": past that, the honest answer is that the request lapsed, and the message
#: says so rather than the move happening hours later out of nowhere.
QUEUED_MOVE_TTL_S = 30 * 60.0


class CwdMoveRefused(Exception):
    """A move this session may not ask for, as a code plus technical fields.

    Raised rather than returned so the caller cannot forget to check: every
    caller in the tool path has to decide what to report. ``fields`` are the
    same structured keys the result carries — no sentence, no copy.
    """

    def __init__(self, code: str, **fields: Any) -> None:
        self.code = code
        self.fields = fields
        super().__init__(code)


@dataclass(frozen=True)
class MoveIntent:
    """One move request as it is stored, with the stage it is at.

    ``source_cwd`` is the effective cwd AT ENQUEUE, recorded because the commit
    has to be able to tell "the successor is where we asked" from "somebody else
    moved this session meanwhile" — a destination is not proof on its own.
    """

    request_id: str
    session_id: str
    source_cwd: str
    destination_cwd: str
    stage: str
    owner_pid: int
    created_at: float
    #: Why the consumer is holding this intent ``queued`` instead of acting on
    #: it, as a code (``""`` when nothing is blocking), and when it first said
    #: so. A runtime that cannot guarantee a successor must SAY it rather than
    #: retire blind, and this is the surviving half of that notice: a marker is
    #: readable by ``sessions(op='info')`` and by the next operator when no
    #: viewer was attached to paint it (agent review round 1, B2).
    blocked_reason: str = ""
    blocked_at: float = 0.0

    def as_fields(self, *, effective_cwd: str = "") -> dict[str, Any]:
        """The structured result: exactly the keys the tool's contract names."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "requested_cwd": self.destination_cwd,
            "destination_cwd": self.destination_cwd,
            "source_cwd": self.source_cwd,
            "effective_cwd": effective_cwd,
            "stage": self.stage,
            "blocked_reason": self.blocked_reason,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "source_cwd": self.source_cwd,
            "destination_cwd": self.destination_cwd,
            "stage": self.stage,
            "owner_pid": self.owner_pid,
            "created_at": self.created_at,
            "blocked_reason": self.blocked_reason,
            "blocked_at": self.blocked_at,
        }


def marker_path(root: Path, session_id: str) -> Path:
    """Where one session's move intent lives."""
    return session_dir(root, session_id) / MOVE_MARKER_NAME


def _write(root: Path, intent: MoveIntent) -> None:
    """Publish the intent atomically, through the family's staged write."""
    # The same helper the stop marker, exit request and turn journal use: staging
    # in the SAME directory and applying 0600 BEFORE publication is the contract
    # a crash must not be able to break (see registry._staged_write).
    # The session directory is normally there (the transcript lives in it), but
    # a refused/rolled-back earlier move can leave it absent in a fresh tree, and
    # a staged write needs its directory: creating it here is what keeps a queued
    # intent from failing on a filesystem detail.
    marker_path(root, intent.session_id).parent.mkdir(parents=True, exist_ok=True)
    registry._staged_write(
        marker_path(root, intent.session_id),
        intent.to_payload(),
        prefix=f".{MOVE_MARKER_NAME}.",
    )


def read_move_intent(root: Path, session_id: str) -> MoveIntent | None:
    """The stored intent, or ``None`` when there is none or it is unusable.

    Unreadable is "no intent", never a raise: this runs on the reconciliation
    path, where a torn document must not wedge a session.
    """
    try:
        raw = marker_path(root, session_id).read_text()
        payload = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    stage = str(payload.get("stage") or "")
    if stage not in STAGES:
        return None
    try:
        return MoveIntent(
            request_id=str(payload["request_id"]),
            session_id=str(payload["session_id"]),
            source_cwd=str(payload["source_cwd"]),
            destination_cwd=str(payload["destination_cwd"]),
            stage=stage,
            owner_pid=int(payload.get("owner_pid") or 0),
            created_at=float(payload.get("created_at") or 0.0),
            # ADDITIVE, and defaulted: every marker written before these keys
            # existed reads back as "nothing is blocking it", which is what the
            # consumer's own guarantee check decides afresh anyway.
            blocked_reason=str(payload.get("blocked_reason") or ""),
            blocked_at=float(payload.get("blocked_at") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def resolve_destination(requested: str, *, effective_cwd: str) -> str:
    """Resolve a requested destination to an absolute, enterable directory.

    ``~`` and relative paths resolve against the SESSION's cwd, never this
    process's: the caller asked for a path as its session sees it. Every failure
    is a code, because the caller is a model that can read fields and act.
    """
    if not requested or not str(requested).strip():
        raise CwdMoveRefused("sessions_move_cwd_required")
    candidate = Path(os.path.expanduser(str(requested)))
    if not candidate.is_absolute():
        candidate = Path(effective_cwd) / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise CwdMoveRefused(
            "sessions_move_destination_missing", requested_cwd=str(requested)
        ) from None
    if not resolved.is_dir():
        raise CwdMoveRefused(
            "sessions_move_destination_not_a_directory", requested_cwd=str(requested)
        )
    # Enterable, not merely present: a directory the session cannot chdir into is
    # a destination a successor could not boot in, and finding that out at retire
    # time would strand the session between two directories.
    if not os.access(resolved, os.X_OK):
        raise CwdMoveRefused(
            "sessions_move_destination_not_enterable", requested_cwd=str(requested)
        )
    return str(resolved)


def enqueue_self_move(
    root: Path,
    session_id: str,
    *,
    requested_cwd: str,
    effective_cwd: str,
    owner_pid: int,
) -> MoveIntent:
    """Persist a ``queued`` intent for this session; the cwd does NOT change.

    IDEMPOTENT PER REQUEST. A second message asking for the directory already
    queued returns that same request rather than a new one — a retried tool call
    must not look like a new move to the reconciliation. A third destination
    supersedes the queued request (the later intent wins; the earlier one never
    reaches ``committed``), and one naming the directory the caller is ALREADY IN
    is the CANCEL: it supersedes the queued intent and returns it at stage
    ``superseded``, which is how a move is withdrawn without inventing a third
    destination (UX round 1, U4). With nothing queued it stays the refusal it was.
    """
    destination = resolve_destination(requested_cwd, effective_cwd=effective_cwd)
    previous = read_move_intent(root, session_id)
    # THE INTENT IS CONSULTED BEFORE THE CURRENT-DIRECTORY REFUSAL, and that
    # ordering is the cancel path: "move back to where I am" is how a queued
    # request is withdrawn, so a call that would have been refused for pointing
    # at the current directory must first get the chance to mean "cancel"
    # (UX round 1, U4).
    if _same_directory(destination, effective_cwd):
        if previous is not None and previous.stage == STAGE_QUEUED:
            cancelled = _with_stage(previous, STAGE_SUPERSEDED)
            _write(root, cancelled)
            return cancelled
        raise CwdMoveRefused("sessions_move_destination_is_current", requested_cwd=destination)
    if previous is not None and previous.stage == STAGE_QUEUED:
        if previous.destination_cwd == destination:
            return previous
        _write(root, _with_stage(previous, STAGE_SUPERSEDED))
    intent = MoveIntent(
        request_id=uuid.uuid4().hex,
        session_id=session_id,
        source_cwd=str(effective_cwd),
        destination_cwd=destination,
        stage=STAGE_QUEUED,
        owner_pid=owner_pid,
        created_at=time.time(),
    )
    _write(root, intent)
    return intent


def _same_directory(left: str, right: str) -> bool:
    """Whether two paths name one directory, however they are spelled.

    REALPATH ON BOTH SIDES, which is the whole reason this exists as a helper:
    the destination is stored resolved, while a session's cwd arrives from a
    record or a tool context and may reach the same tree through a symlink
    (``/var`` versus ``/private/var`` on macOS, any hand-made link elsewhere).
    Comparing a resolved path with an unresolved one queued a full session
    restart toward the directory the session was already in (agent review m1,
    UX round 1, U7), and :func:`reconcile_move` has applied this rule since the
    commit path was fixed — two spellings of one comparison is how the two ends
    drift apart.
    """
    if not left.strip() or not right.strip():
        return False
    return os.path.realpath(left) == os.path.realpath(right)


def _with_stage(intent: MoveIntent, stage: str, **overrides: Any) -> MoveIntent:
    """A copy of ``intent`` at another stage, keeping the request identity.

    ONE PLACE THAT BUILDS A STAGE TRANSITION: every writer here mints the next
    document from the previous one, and a field forgotten in one of the copies
    is a request that changes identity mid-flight.
    """
    fields: dict[str, Any] = {
        "request_id": intent.request_id,
        "session_id": intent.session_id,
        "source_cwd": intent.source_cwd,
        "destination_cwd": intent.destination_cwd,
        "stage": stage,
        "owner_pid": intent.owner_pid,
        "created_at": intent.created_at,
        "blocked_reason": intent.blocked_reason,
        "blocked_at": intent.blocked_at,
    }
    fields.update(overrides)
    return MoveIntent(**fields)


def hold_lapsed(intent: MoveIntent, *, now: float | None = None) -> bool:
    """Whether a ``queued`` intent has been HELD for longer than the TTL.

    Only a recorded hold ages (``blocked_at``, set by :func:`mark_blocked` when
    the hold BEGINS and kept while it continues): a request that nobody has
    looked at yet, or one waiting on an ordinary busy turn, has not been refused
    anything and must not run out of time (UX round 2, U10). A marker with no
    ``blocked_at`` — every pre-hold document, and a hand-written one — therefore
    never lapses on a guess.
    """
    if intent.stage != STAGE_QUEUED or not intent.blocked_reason or intent.blocked_at <= 0:
        return False
    instant = time.time() if now is None else now
    return instant - intent.blocked_at >= QUEUED_MOVE_TTL_S


def expire_stale_intent(
    root: Path, session_id: str, *, now: float | None = None
) -> MoveIntent | None:
    """Flip a ``queued`` intent whose hold outlived its TTL to ``expired``.

    APPLIED BY EVERY READER THAT REPORTS THE STAGE — the runtime's consumer and
    ``sessions(op='info')`` (agent review round 2, m-e: ``info`` used to read the
    raw marker, so a lapsed request still showed as queued until a runtime
    ticked). A bounded life only bounds anything if every reader applies it. An
    intent whose hold has not lapsed, or one already past ``queued``, is returned
    untouched; ``None`` is still "no intent" rather than an error, for the same
    reason :func:`read_move_intent` documents.
    """
    intent = read_move_intent(root, session_id)
    if intent is None or not hold_lapsed(intent, now=now):
        return intent
    expired = _with_stage(intent, STAGE_EXPIRED)
    _write(root, expired)
    return expired


def mark_blocked(
    root: Path, session_id: str, reason: str, *, now: float | None = None
) -> MoveIntent | None:
    """Record why the consumer is holding the intent, so the hold is visible.

    WRITTEN ONLY WHEN THE REASON CHANGES: the consumer ticks twice a second, and
    a staged write per tick would rewrite the same bytes for as long as the
    session stays busy. ``""`` clears the record (the hold ended) for the same
    reason the file name says nothing about it: the stage is the state, this is
    the explanation attached to it.

    ``blocked_at`` IS THE TTL's CLOCK (:func:`hold_lapsed`), so it is stamped when
    a hold BEGINS and kept when one hold reason gives way to another — a request
    that went from "nobody can follow it" to "its directory is gone" has been
    held the whole time, and resetting the clock on every change of wording
    would let a request that can never happen live forever. Clearing the hold
    resets it.
    """
    intent = read_move_intent(root, session_id)
    if intent is None or intent.stage != STAGE_QUEUED:
        return intent
    if intent.blocked_reason == reason and (reason == "" or intent.blocked_at > 0):
        return intent
    if not reason:
        started = 0.0
    elif intent.blocked_reason and intent.blocked_at > 0:
        started = intent.blocked_at
    else:
        started = time.time() if now is None else now
    updated = _with_stage(intent, STAGE_QUEUED, blocked_reason=reason, blocked_at=started)
    _write(root, updated)
    return updated


def reconcile_move(root: Path, session_id: str, *, effective_cwd: str) -> MoveIntent | None:
    """Settle a queued intent against where the session ACTUALLY is now.

    THE ONLY WAY A ``queued`` INTENT BECOMES ``committed`` WITHOUT A LIVE
    CONFIRMATION: if the session is observed at the destination, the successor
    evidently booted there — the confirmation was lost, not the move. Observed
    anywhere else, the intent stays ``queued`` for the successor to act on; this
    function never infers a failure, because an inference is not evidence.

    ``effective_cwd`` is the directory THE SESSION holds, which is not
    necessarily the one this process runs in: the value arrives from a record (or
    from a caller) and may spell the same directory through a different symlink
    (``/var`` versus ``/private/var`` on macOS), so both sides are compared
    realpath-ed — the destination was stored resolved by
    :func:`resolve_destination`, and a spelling difference is not a different
    directory.
    """
    intent = read_move_intent(root, session_id)
    if intent is None or intent.stage != STAGE_QUEUED:
        return intent
    if not effective_cwd.strip():
        return intent
    if not _same_directory(effective_cwd, intent.destination_cwd):
        return intent
    committed = _with_stage(intent, STAGE_COMMITTED)
    _write(root, committed)
    return committed
