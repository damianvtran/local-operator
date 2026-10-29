"""The durable intent behind a session's OWN working-directory move.

WHY A MARKER AND NOT A FIELD. A move is two acts in two processes: this session
ASKS (its tool call returns ``queued`` immediately) and its SUCCESSOR runtime
ACTS, by booting in the destination. The request therefore has to outlive the
asking process — a crash, a lost acknowledgement, or a retire that lands between
the two — so it is a document on disk, not state in memory.

THE STAGES ARE THE RECONCILIATION AXIS, and there are only three:

* ``queued``     — asked for, not yet observed. The effective cwd is unchanged.
* ``committed``  — a successor was observed running in the destination.
* ``superseded`` — a later request replaced this one before it committed.

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
STAGES = frozenset({STAGE_QUEUED, STAGE_COMMITTED, STAGE_SUPERSEDED})


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

    def as_fields(self, *, effective_cwd: str = "") -> dict[str, Any]:
        """The structured result: exactly the keys the tool's contract names."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "requested_cwd": self.destination_cwd,
            "source_cwd": self.source_cwd,
            "effective_cwd": effective_cwd,
            "stage": self.stage,
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

    IDEMPOTENT PER DESTINATION: asking twice for the directory already queued
    returns the SAME request rather than minting a second one, because a retried
    tool call must not look like a new move to the reconciliation. A different
    destination supersedes the queued request — the later intent wins, and the
    earlier one never reaches ``committed``.
    """
    destination = resolve_destination(requested_cwd, effective_cwd=effective_cwd)
    if destination == str(Path(effective_cwd)):
        raise CwdMoveRefused("sessions_move_destination_is_current", requested_cwd=destination)
    previous = read_move_intent(root, session_id)
    if previous is not None and previous.stage == STAGE_QUEUED:
        if previous.destination_cwd == destination:
            return previous
        _write(
            root,
            MoveIntent(
                request_id=previous.request_id,
                session_id=previous.session_id,
                source_cwd=previous.source_cwd,
                destination_cwd=previous.destination_cwd,
                stage=STAGE_SUPERSEDED,
                owner_pid=previous.owner_pid,
                created_at=previous.created_at,
            ),
        )
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


def commit_move(
    root: Path,
    session_id: str,
    request_id: str,
    *,
    effective_cwd: str,
) -> MoveIntent:
    """Record that the successor really is running in the destination.

    THE VALIDATION IS REPEATED HERE, deliberately: the directory can be renamed,
    unmounted or removed between enqueue and the boot, and the commit is the last
    moment the runtime can still stay where it is instead of booting into a path
    that no longer exists.

    Refuses when the request is unknown, superseded, or when the session's source
    cwd is no longer the one the request recorded — that last case means something
    else moved the session, and claiming this request completed would be a lie.
    """
    intent = read_move_intent(root, session_id)
    if intent is None or intent.request_id != request_id:
        raise CwdMoveRefused("sessions_move_request_unknown", request_id=request_id)
    if intent.stage != STAGE_QUEUED:
        raise CwdMoveRefused(
            "sessions_move_request_not_queued", request_id=request_id, stage=intent.stage
        )
    resolve_destination(intent.destination_cwd, effective_cwd=intent.source_cwd)
    if str(Path(effective_cwd)) != intent.source_cwd:
        raise CwdMoveRefused(
            "sessions_move_source_changed",
            request_id=request_id,
            source_cwd=intent.source_cwd,
            effective_cwd=str(effective_cwd),
        )
    committed = MoveIntent(
        request_id=intent.request_id,
        session_id=intent.session_id,
        source_cwd=intent.source_cwd,
        destination_cwd=intent.destination_cwd,
        stage=STAGE_COMMITTED,
        owner_pid=intent.owner_pid,
        created_at=intent.created_at,
    )
    _write(root, committed)
    return committed


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
    if os.path.realpath(effective_cwd) != os.path.realpath(intent.destination_cwd):
        return intent
    committed = MoveIntent(
        request_id=intent.request_id,
        session_id=intent.session_id,
        source_cwd=intent.source_cwd,
        destination_cwd=intent.destination_cwd,
        stage=STAGE_COMMITTED,
        owner_pid=intent.owner_pid,
        created_at=intent.created_at,
    )
    _write(root, committed)
    return committed
