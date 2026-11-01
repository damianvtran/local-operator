"""The durable intent behind a session's OWN working-directory move.

WHY A MARKER AND NOT A FIELD. A move is two acts in two processes: this session
ASKS (its tool call returns ``queued`` immediately) and its SUCCESSOR runtime
ACTS, by booting in the destination. The request therefore has to outlive the
asking process — a crash, a lost acknowledgement, or a retire that lands between
the two — so it is a document on disk, not state in memory.

THE STAGES ARE THE RECONCILIATION AXIS, and there are five:

* ``queued``    — asked for, not yet observed. The effective cwd is unchanged.
  A LIVE runtime consumes it at its next idle point, by retiring.
* ``handoff``   — the destination is now the session's RECORDED TRUTH, written
  just before the asking runtime retires. Nothing has been observed running
  there yet; the next engage boots at the destination and THAT boot promotes
  this stage (see :func:`reconcile_move`). This is what makes a viewer optional
  rather than a precondition: a headless session has nobody to re-engage it, so
  the destination has to survive the gap on its own (design note, option (d)).
* ``committed`` — a successor was OBSERVED running in the destination.
* ``superseded`` — a later request replaced this one before it committed, OR the
  caller asked to move back to where the session already is, which is this
  module's CANCEL: it is the only request that can undo a queued one without
  minting a third destination (UX round 1, U4). A ``handoff`` whose successor
  booted somewhere else is superseded the same way, once, naming that reason.
* ``expired``   — RETIRED AS AN OUTCOME, kept as a value only so an OLD marker
  still reports honestly. It used to mean "held for longer than a TTL", and
  the hold-then-lapse state is exactly the defect the handoff replaces: a
  headless session could never move itself, so its request aged out with the
  cwd unchanged. Nothing writes this stage any more.

WHY A TTL IS GONE. ``QUEUED_MOVE_TTL_S`` bounded a request no successor ever
satisfied, because the loop it guarded re-armed on every boot. The handoff
closes that loop by construction — the boot now lands at the destination, and a
boot that lands elsewhere settles the request as ``superseded`` — so there is
nothing left for a clock to bound. A move is either observed (``committed``) or
ends with an explicit reason; it never lapses into silence.

NOTHING HERE RETIRES A RUNTIME. This module owns the DOCUMENT; the runtime that
consumes it (``session/runtime/server._apply_queued_cwd_move``) owns the
retirement, and it writes :func:`accept_handoff` BEFORE it stops so the
destination outlives the process (design note, crash semantics).

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
STAGE_HANDOFF = "handoff"
STAGE_COMMITTED = "committed"
STAGE_SUPERSEDED = "superseded"
STAGE_EXPIRED = "expired"
STAGES = frozenset({STAGE_QUEUED, STAGE_HANDOFF, STAGE_COMMITTED, STAGE_SUPERSEDED, STAGE_EXPIRED})

#: The payload's schema version. 2 added ``handoff``; a reader does not gate on
#: it (an older marker is still readable), it is here so a DOWNGRADED build can
#: be told, in the document itself, that it is looking at something it predates.
PAYLOAD_VERSION = 2

#: The stages a new request may displace: the two that have not been observed
#: yet. Every other stage is history, and a later request starts a new document
#: rather than editing one that already reached an outcome.
DISPLACEABLE_STAGES = frozenset({STAGE_QUEUED, STAGE_HANDOFF})


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
    """The move request in force for this session, with the stage it reached.

    ``source_cwd`` is the effective cwd AT ENQUEUE, recorded because the commit
    has to be able to tell "the successor is where we asked" from "somebody else
    moved this session meanwhile" — a destination is not proof on its own (and
    the engage guard needs the source to decide whether a caller is asking to go
    where the session already IS, or somewhere new).
    """

    request_id: str
    session_id: str
    source_cwd: str
    destination_cwd: str
    stage: str
    owner_pid: int
    created_at: float
    #: Why the consuming runtime could not act on this request on its last look —
    #: a NOTICE, never a gate: it names the condition the runtime hit, and the
    #: consumer re-checks that condition from scratch on every tick. It stays on
    #: the marker so ``sessions(op='info')`` and the next caller can see why
    #: nothing has happened yet. Only transients land here (a destination that
    #: went away, an unreadable idle boundary); "no viewer can follow this" is
    #: no longer a reason to wait, so it is no longer a reason recorded here.
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
            "version": PAYLOAD_VERSION,
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
    must not look like a new move to the reconciliation. A different destination
    supersedes a request nothing has OBSERVED yet (``queued`` or ``handoff``):
    the later request wins, and the earlier one must never reach ``committed``
    behind it. Naming the directory the caller is ALREADY IN is the CANCEL: it
    supersedes such a request and returns it at stage ``superseded``, which is
    how a move is withdrawn without inventing a third destination (UX round 1,
    U4) — including a ``handoff`` a runtime recorded moments before exiting.
    With nothing supersedable it raises :class:`CwdMoveRefused`, as before.
    """
    destination = resolve_destination(requested_cwd, effective_cwd=effective_cwd)
    previous = read_move_intent(root, session_id)
    # ONLY AN UNRECORDED OUTCOME CAN BE DISPLACED. A committed/superseded/expired
    # document is history: editing it would rewrite what the session is recorded
    # to have done, which is the one thing this file exists to keep straight.
    displaceable = (
        previous if previous is not None and previous.stage in DISPLACEABLE_STAGES else None
    )
    # THE INTENT IS CONSULTED BEFORE THE CURRENT-DIRECTORY REFUSAL, and that
    # ordering is the cancel path: "move back to where I am" is how a queued
    # request is withdrawn, so a call that would have been refused for pointing
    # at the current directory must first get the chance to mean "cancel"
    # (UX round 1, U4).
    if same_directory(destination, effective_cwd):
        if displaceable is not None:
            cancelled = _with_stage(displaceable, STAGE_SUPERSEDED)
            _write(root, cancelled)
            return cancelled
        raise CwdMoveRefused("sessions_move_destination_is_current", requested_cwd=destination)
    if displaceable is not None:
        if displaceable.stage == STAGE_QUEUED and displaceable.destination_cwd == destination:
            return displaceable
        _write(root, _with_stage(displaceable, STAGE_SUPERSEDED))
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


def same_directory(left: str, right: str) -> bool:
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


def mark_blocked(
    root: Path, session_id: str, reason: str, *, now: float | None = None
) -> MoveIntent | None:
    """Record why the consumer could not act on the intent, so it is visible.

    WRITTEN ONLY WHEN THE REASON CHANGES: the consumer ticks twice a second, and
    a staged write per tick would rewrite the same bytes for as long as the
    condition persists. ``""`` clears the record (the condition ended) for the
    same reason the file name says nothing about it: the stage is the state,
    this is the explanation attached to it.

    ``blocked_at`` is when the CONDITION was first seen, kept while one reason
    gives way to another — it is a log of "how long has this not worked" and is
    deliberately not a clock that decides anything any more (the TTL it once
    fed is gone; see the module docstring). Clearing the record resets it.
    """
    intent = read_move_intent(root, session_id)
    if intent is None or intent.stage not in DISPLACEABLE_STAGES:
        return intent
    if intent.blocked_reason == reason and (reason == "" or intent.blocked_at > 0):
        return intent
    if not reason:
        started = 0.0
    elif intent.blocked_reason and intent.blocked_at > 0:
        started = intent.blocked_at
    else:
        started = time.time() if now is None else now
    updated = _with_stage(intent, intent.stage, blocked_reason=reason, blocked_at=started)
    _write(root, updated)
    return updated


def accept_handoff(root: Path, session_id: str, intent: MoveIntent) -> MoveIntent | None:
    """Publish ``intent``'s destination as the session's RECORDED TRUTH, then stop.

    THE ACT THAT MAKES A VIEWER OPTIONAL. The asking runtime calls this in the
    same breath as it retires: from here on the destination is what ANY later
    engage reads, so the session can be re-engaged by a message, a wake or
    ``lop exec`` with nobody attached to follow it. Nothing has been observed
    running there yet — that is :func:`reconcile_move`'s half — and the request
    keeps its identity (``request_id``, ``source_cwd``) so the successor can
    still tell "booted where we asked" from "booted somewhere else".

    Returns ``None`` when the document on disk is no longer THIS request (it was
    cancelled or replaced in the moment between the consumer's read and this
    write): publishing then would resurrect a withdrawn destination, so the
    caller must treat ``None`` as "stop trying, do nothing".
    """
    current = read_move_intent(root, session_id)
    if current is None or current.request_id != intent.request_id:
        return None
    if current.stage not in DISPLACEABLE_STAGES:
        return None
    handoff = _with_stage(current, STAGE_HANDOFF, blocked_reason="", blocked_at=0.0)
    _write(root, handoff)
    return handoff


def rollback_handoff(root: Path, session_id: str, intent: MoveIntent) -> MoveIntent | None:
    """Put a ``handoff`` back to ``queued`` when the retire it was written for is refused.

    THE ONE ROLLBACK, and it is narrow on purpose: a handoff is written before
    the retire latch is claimed (so a crash cannot lose the destination), so a
    latch that refuses — work arrived in that instant — must undo the write or
    the next engage would act on a move this runtime is still here to carry
    out. Everything else stays put; once the session HAS moved nothing here
    rolls it back (see the module docstring).

    Returns the restored intent, or ``None`` when the document is no longer this
    request (nothing to roll back).
    """
    current = read_move_intent(root, session_id)
    if current is None or current.request_id != intent.request_id:
        return None
    if current.stage != STAGE_HANDOFF:
        return None
    queued = _with_stage(current, STAGE_QUEUED)
    _write(root, queued)
    return queued


def withdraw_queued_move(root: Path, session_id: str) -> MoveIntent | None:
    """Supersede a request that nothing has observed yet, so nothing acts on it.

    Used by a caller that has to REFUSE a move it has just recorded (a
    destination outside the session's confinement, a host that can never
    restart it): leaving the request in force would either move the session
    somewhere the caller refused, or sit `queued` for ever, which is exactly the
    silent lapse the handoff design removes. The document keeps its identity at
    stage ``superseded``, so every reader sees the request was withdrawn rather
    than never made.
    """
    current = read_move_intent(root, session_id)
    if current is None or current.stage not in DISPLACEABLE_STAGES:
        return None
    withdrawn = _with_stage(current, STAGE_SUPERSEDED)
    _write(root, withdrawn)
    return withdrawn


class HandoffUnreadable(Exception):
    """A move document exists but cannot be understood, so the destination is unknown.

    Raised only by :func:`pending_handoff`, and only towards callers that are
    about to choose a directory to boot a session in. The two candidate answers
    there are not symmetric: refusing costs one engage a sentence, while
    guessing spawns the session in a directory the document may have said not
    to — a wrong spawn, which is worse than a sentence.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(path)


def pending_handoff(root: Path, session_id: str) -> MoveIntent | None:
    """The handoff waiting for this session's next runtime, or ``None``.

    THE ONE READER THE ENGAGE PATH USES, and it is deliberately stricter than
    :func:`read_move_intent`:

    * no document file at all -> ``None`` (one `stat`, which is the cost paid on
      every ordinary engage),
    * a document that is not a ``handoff`` -> ``None``: every other stage says
      nothing about where the next process should start,
    * a ``handoff`` document that cannot be parsed -> :class:`HandoffUnreadable`,
      never ``None``. Falling through to the caller's own cwd there would spawn
      the session in the directory the move was explicitly leaving.

    The whole intent is returned, not just the destination: a caller that has to
    decide between "substitute the destination" and "this caller named a third
    directory, so the move is cancelled" needs ``source_cwd`` too.
    """
    path = marker_path(root, session_id)
    if not path.is_file():
        return None
    intent = read_move_intent(root, session_id)
    if intent is None:
        raise HandoffUnreadable(str(path))
    if intent.stage != STAGE_HANDOFF:
        return None
    return intent


def reconcile_move(root: Path, session_id: str, *, effective_cwd: str) -> MoveIntent | None:
    """Settle an unobserved intent against where the session ACTUALLY booted.

    THE ONLY WAY A REQUEST BECOMES ``committed`` WITHOUT A LIVE CONFIRMATION: if
    the session is observed at the destination, the successor evidently booted
    there — the confirmation was lost, not the move. This is the promotion rule
    the whole design leans on, so it is one function and it runs at every boot.

    THE TWO UNOBSERVED STAGES SETTLE DIFFERENTLY, and the difference is the loop
    bound:

    * ``queued`` — the runtime that is still running has not acted yet, so a boot
      anywhere means nothing either way: at the destination it commits (the
      request was carried out), anywhere else it stays ``queued`` for that
      runtime's own idle point. Never inferred as a failure.
    * ``handoff`` — the destination was already written as the session's truth
      and the writer has (or had) gone. A boot at the destination commits; a boot
      ANYWHERE ELSE supersedes, once, naming the reason. That single shot is what
      stops the B2 failure (a boot that retires and exits on every open, for
      ever): the request cannot survive a boot that ignored it.

    ``committed`` still means OBSERVED, never inferred from a write.

    ``effective_cwd`` is the directory THE SESSION holds, which is not
    necessarily the one this process runs in: the value arrives from a record (or
    from a caller) and may spell the same directory through a different symlink
    (``/var`` versus ``/private/var`` on macOS), so both sides are compared
    realpath-ed — the destination was stored resolved by
    :func:`resolve_destination`, and a spelling difference is not a different
    directory.
    """
    intent = read_move_intent(root, session_id)
    if intent is None or intent.stage not in DISPLACEABLE_STAGES:
        return intent
    if not effective_cwd.strip():
        return intent
    at_destination = same_directory(effective_cwd, intent.destination_cwd)
    if not at_destination:
        if intent.stage != STAGE_HANDOFF:
            return intent
        superseded = _with_stage(intent, STAGE_SUPERSEDED)
        _write(root, superseded)
        return superseded
    committed = _with_stage(intent, STAGE_COMMITTED, blocked_reason="", blocked_at=0.0)
    _write(root, committed)
    return committed
