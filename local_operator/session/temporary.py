"""Temporary chats: the per-session mark, its admission rules and its tombstones.

A temporary chat is the user's own conversation that is left out of every
history listing and deleted after a window of inactivity (default 30 minutes).
This module holds the facts every surface shares: how a session is MARKED, how
a mark is READ, who may set one, and how a deleted id is remembered for a week
so a client can say "this temporary chat was deleted" instead of "not found".

WHY A PER-SESSION SIDECAR (``sessions/<id>/temporary.json``), and not the two
obvious alternatives (design record: the temporary-chats contract, §1.1):

* **Not an ``origin.json`` value.** The origin is written once and its verdict
  is memoised as immutable, so "Keep" could never undo it; a session has ONE
  origin, so marking a fork would erase ``fork``; and any non-allow-listed
  origin hides the session everywhere AND moves it into the delegated retention
  class. A temporary chat is the user's own chat.
* **Not a store-level index** (``archived.py``'s shape). That index accepts
  last-writer-wins over the whole file. Here a lost "Keep" write means the
  conversation gets DELETED, so the race fails open on an irreversible act.

THE READER FAILS CLOSED, and closed means RETAINED: a mark counts only when it
is exactly the shape :func:`write_mark` writes. Absent, torn, unreadable, a
non-object, another session's id, a ``marked_at`` far in the future — every one
of them reads as "not temporary", so nothing a bad parse produces can make the
reaper delete a conversation.

NOTHING HERE REMOVES A SESSION. ``session/cleanup.py`` is the only deleter
(``tests/unit/session/test_no_session_deletion.py``); this module's only
removal is its OWN temp file after a failed atomic replace. "Keep" is a write
of ``temporary: false``, not an unlink, for exactly that reason.

Stdlib-only at import, like ``archived.py``: the listing scan calls
:func:`is_marked_path` per user row and the runtime child imports cleanup.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The per-session sidecar. Registered as bookkeeping in ``retention``, as part
#: of the mesh copy set in ``network.sync``, as NOT copied by ``fork``, and as
#: evidence in ``guardrails.classify`` (a shell write of it ranks CRITICAL).
MARK_NAME = "temporary.json"

MARK_VERSION = 1

#: A mark stamped more than this far in the future is not trusted: a clock that
#: far ahead would otherwise read as "never idle" — harmless — but a mark whose
#: time cannot be trusted is not the shape this module writes either.
FUTURE_SKEW_S = 60.0

#: Who set a mark. Recorded for diagnosis only; nothing branches on it.
MARK_SOURCES = frozenset({"cli", "exec", "tui", "desktop", "relay"})

#: ``sessions/.temporary-tombstones.json``: ids and deletion times only (no
#: titles, no content), written by the cleanup pass BEFORE it removes anything.
TOMBSTONES_NAME = ".temporary-tombstones.json"
TOMBSTONE_TTL_S = 7 * 24 * 3600.0
TOMBSTONE_CAP = 1000

#: ``sessions/.temporary-state.json``: the pass's view of each marked session
#: (when it was last held, and by what). Written only by the pass; read by the
#: listings to say "deletes at" / "kept while …". Reads never create it.
STATE_NAME = ".temporary-state.json"

#: The window presets, in minutes. A configured value outside the set is
#: clamped UP to the next preset (kept longer — the fail-closed direction).
IDLE_PRESETS: tuple[int, ...] = (15, 30, 120, 1440)
DEFAULT_IDLE_MINUTES = 30

#: Cadence of the reaping pass. The listings trust a ``hold`` only while the
#: state is fresher than three passes, so a dead sweeper does not leave a stale
#: "kept while running" on screen forever.
TEMPORARY_SWEEP_S = 60.0

#: The closed set of reasons a temporary chat is being kept past its window.
#: The clients key their copy off these codes, so the set is part of the wire.
HOLD_CODES: tuple[str, ...] = (
    "running",
    "on_screen",
    "open_ask",
    "armed_wake",
    "armed_monitor",
    "unread_mail",
    "operator_hold",
    "unseen_result",
    "unsaved_files",
    "linked",
    "cannot_verify",
)

# Refusal codes, shared with the HTTP routes so one spelling reaches the wire.
CODE_TEMPORARY = "session_temporary"
CODE_DELETED = "session_temporary_deleted"
CODE_PINNED = "session_pinned"
CODE_IN_PROJECT = "session_in_project"
CODE_NOT_USER = "session_not_user"
CODE_NOT_FOUND = "session_not_found"
CODE_MARK_FAILED = "temporary_mark_failed"


@dataclass(frozen=True)
class Mark:
    """A mark that read back EXACTLY as written, so it is temporary."""

    session_id: str
    marked_at: float
    by: str


@dataclass(frozen=True)
class MarkResult:
    """The outcome of :func:`write_mark`. ``code`` is empty on success."""

    ok: bool
    code: str = ""


class TemporarySessionError(ValueError):
    """A temporary chat cannot be linked to a project until the user keeps it.

    A ``ValueError`` so every existing caller that already turns a registry
    ``ValueError`` into a refusal keeps working; the routes that know the
    ``code`` answer 409 with it.
    """

    code = CODE_TEMPORARY

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} is a temporary chat and can't be linked to a "
            "project until the user keeps it."
        )


# ---------------------------------------------------------------------------
# Reading the mark
# ---------------------------------------------------------------------------


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def read_mark(session_dir: Path, *, now: float | None = None) -> Mark | None:
    """The session's mark when it is temporary, else ``None``.

    EXACT SCHEMA OR NOTHING: ``v == 1``, ``temporary is True``, ``session_id``
    equal to the directory's own name (a mark copied or planted from another
    session does not count), and a finite ``marked_at`` not more than
    :data:`FUTURE_SKEW_S` ahead of the clock. Everything else — absent,
    unreadable, torn, a Keep (``temporary: false``) — reads as not temporary,
    the direction in which the conversation is retained.

    ``errors="replace"`` for ``resume.session_origin``'s reason: a cut inside a
    multi-byte character must read as "not temporary", not raise past an
    ``except OSError``.
    """
    try:
        raw = (session_dir / MARK_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("v") != MARK_VERSION or payload.get("temporary") is not True:
        return None
    if payload.get("session_id") != session_dir.name:
        return None
    marked_at = _finite(payload.get("marked_at"))
    if marked_at is None:
        return None
    clock = time.time() if now is None else now
    if marked_at > clock + FUTURE_SKEW_S:
        return None
    by = payload.get("by")
    return Mark(session_id=session_dir.name, marked_at=marked_at, by=by if isinstance(by, str) else "")


def is_marked_path(session_dir: str | os.PathLike[str]) -> bool:
    """Whether ``session_dir`` is a temporary chat — one ``stat`` when it is not.

    The listing scan calls this per user row, so the common case (no sidecar)
    costs a single failed ``stat`` and the parse runs only when the file exists.
    """
    try:
        os.stat(os.path.join(session_dir, MARK_NAME))
    except OSError:
        return False
    return read_mark(Path(session_dir)) is not None


def is_marked(config_dir: Path, session_id: str) -> bool:
    """:func:`is_marked_path` by id. A malformed id is never temporary."""
    if not _plain_id(session_id):
        return False
    return is_marked_path(Path(config_dir) / "sessions" / session_id)


def _plain_id(session_id: object) -> bool:
    return (
        isinstance(session_id, str)
        and bool(session_id)
        and session_id not in {".", ".."}
        and not any(ch in "/\\" or ord(ch) < 32 or ord(ch) == 127 for ch in session_id)
    )


# ---------------------------------------------------------------------------
# Writing the mark
# ---------------------------------------------------------------------------


def admission_refusal(config_dir: Path, session_id: str) -> str:
    """Why ``session_id`` may NOT become temporary, as a code; ``""`` when it may.

    * Not a user session, or an origin other than the user's own or ``fork``:
      ``agent-workstream`` is opened by an agent and its lifecycle is the
      agent's; a delegated run already has its own retention class.
    * Pinned: a pin says "keep this", and so does project membership. Both are
      refused here rather than silently undone, so the remedy (unpin, unlink)
      stays the user's act.

    Every check that cannot be EVALUATED refuses too: marking is the step that
    makes a conversation deletable, so "could not tell" must mean "not marked".
    """
    from local_operator.resume import ORIGIN_FORK, is_user_session_origin, session_origin

    if not _plain_id(session_id):
        return CODE_NOT_FOUND
    directory = Path(config_dir) / "sessions" / session_id
    if not directory.is_dir():
        return CODE_NOT_FOUND
    origin = session_origin(directory)
    if not is_user_session_origin(origin) or origin not in ("", ORIGIN_FORK):
        return CODE_NOT_USER
    try:
        from local_operator.session.cleanup import GuardUnavailable, pinned_session_ids

        try:
            if session_id in pinned_session_ids(Path(config_dir)):
                return CODE_PINNED
        except GuardUnavailable:
            return CODE_MARK_FAILED
        if _linked_to_project(Path(config_dir), session_id):
            return CODE_IN_PROJECT
    except Exception:  # noqa: BLE001 — a check that cannot answer refuses the mark
        logger.warning("temporary: could not evaluate admission for %s", session_id, exc_info=True)
        return CODE_MARK_FAILED
    return ""


def _linked_to_project(config_dir: Path, session_id: str) -> bool:
    from local_operator.projects import ProjectRegistry

    return bool(ProjectRegistry(config_dir).projects_for_session(session_id))


def write_mark(
    config_dir: Path,
    session_id: str,
    on: bool,
    *,
    by: str,
    now: float | None = None,
) -> MarkResult:
    """Mark (``on``) or keep (``not on``) a session; read it back before success.

    Marking runs :func:`admission_refusal` first. Keeping has no admission: it
    only ever makes a conversation retained, so it is always allowed for a
    session that exists.

    THE DIRECTORY MTIME IS RESTORED, for ``resume.mark_session_origin``'s
    reason: a mark is bookkeeping ABOUT a session, never activity IN it.

    SUCCESS MEANS THE MARK READS BACK: a create path that asked for a
    temporary chat must refuse the create when the mark did not land, because
    silently retaining the chat breaks the promise the user was given.
    """
    if not _plain_id(session_id):
        return MarkResult(False, CODE_NOT_FOUND)
    directory = Path(config_dir) / "sessions" / session_id
    if not directory.is_dir():
        return MarkResult(False, CODE_NOT_FOUND)
    if on:
        refusal = admission_refusal(config_dir, session_id)
        if refusal:
            return MarkResult(False, refusal)
    clock = time.time() if now is None else now
    payload: dict[str, Any] = {"v": MARK_VERSION, "temporary": bool(on), "session_id": session_id}
    payload["marked_at" if on else "kept_at"] = clock
    payload["by"] = by if by in MARK_SOURCES else "cli"
    if not _write_mark_file(directory, payload):
        return MarkResult(False, CODE_MARK_FAILED)
    landed = read_mark(directory, now=clock) is not None
    if landed != bool(on):
        return MarkResult(False, CODE_MARK_FAILED)
    return MarkResult(True)


def _write_mark_file(directory: Path, payload: dict[str, Any]) -> bool:
    """Same-directory temp file + ``os.replace``; restore the directory mtime."""
    try:
        previous = directory.stat().st_mtime
    except OSError:
        return False
    try:
        handle_fd, temporary = tempfile.mkstemp(dir=directory, prefix=".temporary-", suffix=".tmp")
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            Path(temporary).replace(directory / MARK_NAME)
        except BaseException:
            # Only this call's own temp FILE; the session directory is never a target.
            Path(temporary).unlink(missing_ok=True)
            raise
        os.utime(directory, (previous, previous))
    except OSError:
        logger.warning("temporary: could not write the mark for %s", directory.name, exc_info=True)
        return False
    return True


def pin_refusal(config_dir: Path, session_id: str) -> str:
    """``session_temporary`` when pinning ``session_id`` must be refused, else ``""``.

    One helper for every pin surface (desktop, relay, TUI) so the rule cannot
    drift: a pin is a promise to keep, and a temporary chat is a promise to
    delete; the user picks one by keeping the chat first.
    """
    return CODE_TEMPORARY if is_marked(config_dir, session_id) else ""


def require_not_temporary(config_dir: Path, session_ids: Any) -> None:
    """Raise :class:`TemporarySessionError` for the first temporary id in ``session_ids``."""
    if isinstance(session_ids, str):
        session_ids = [session_ids]
    for session_id in session_ids or ():
        if is_marked(config_dir, session_id):
            raise TemporarySessionError(session_id)


# ---------------------------------------------------------------------------
# Tombstones and pass state (written by session/cleanup.py; read here)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def read_tombstones(config_dir: Path, *, now: float | None = None) -> list[dict[str, Any]]:
    """The tombstone index, newest first, PRUNED AT READ (7 days, 1,000 rows).

    Malformed entries are dropped rather than failing the read: a tombstone
    only ever changes WHICH 404 a client sees, never whether data is kept.
    """
    payload = _read_json(Path(config_dir) / "sessions" / TOMBSTONES_NAME)
    if not isinstance(payload, list):
        return []
    clock = time.time() if now is None else now
    entries: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        session_id = item.get("id")
        deleted_at = _finite(item.get("deleted_at"))
        if not _plain_id(session_id) or deleted_at is None:
            continue
        if clock - deleted_at > TOMBSTONE_TTL_S:
            continue
        entries.append({"id": session_id, "deleted_at": deleted_at})
    return entries[:TOMBSTONE_CAP]


def deleted_at(config_dir: Path, session_id: str, *, now: float | None = None) -> float | None:
    """When the temporary chat ``session_id`` was deleted, or ``None``.

    A value ONLY when the id is tombstoned AND its directory is gone: the pass
    writes the tombstone before it removes, so a removal that was then refused
    leaves a tombstone beside a live session, and that session must never be
    reported as deleted. Reading never creates the file.
    """
    if not _plain_id(session_id):
        return None
    if (Path(config_dir) / "sessions" / session_id).exists():
        return None
    for entry in read_tombstones(config_dir, now=now):
        if entry["id"] == session_id:
            return float(entry["deleted_at"])
    return None


def read_state(config_dir: Path) -> dict[str, Any]:
    """The pass's state file, or an empty state. Never creates it."""
    payload = _read_json(Path(config_dir) / "sessions" / STATE_NAME)
    if not isinstance(payload, dict) or payload.get("v") != 1:
        return {"v": 1, "evaluated_at": None, "sessions": {}}
    sessions = payload.get("sessions")
    return {
        "v": 1,
        "evaluated_at": _finite(payload.get("evaluated_at")),
        "sessions": sessions if isinstance(sessions, dict) else {},
    }


def clamp_idle_minutes(value: Any) -> int:
    """A configured window, clamped UP to the next preset; junk reads as 30.

    Up, because a longer window keeps the chat longer: the fail-closed side.
    """
    number: int | None
    if isinstance(value, bool):
        number = None
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value):
        number = math.ceil(value)
    elif isinstance(value, str):
        try:
            number = int(value.strip())
        except ValueError:
            number = None
    else:
        number = None
    if number is None:
        if value is not None:
            logger.warning(
                "session cleanup: session.cleanup.temporary.idle_minutes=%r is not a number "
                "of minutes; using %d",
                value,
                DEFAULT_IDLE_MINUTES,
            )
        return DEFAULT_IDLE_MINUTES
    for preset in IDLE_PRESETS:
        if number <= preset:
            return preset
    return IDLE_PRESETS[-1]


def row_info(
    session_dir: Path,
    *,
    idle_minutes: int,
    state: dict[str, Any] | None,
    now: float | None = None,
    activity: float | None = None,
) -> dict[str, Any] | None:
    """The wire ``temporary`` object for one row, or ``None`` when not temporary.

    ``{"marked_at", "idle_minutes", "deletes_at", "hold"}``. ``hold`` is taken
    from the pass state only while that state is fresh (three passes): a
    sweeper that stopped must not leave "kept while running" on screen.
    ``deletes_at`` is ``None`` while held.
    """
    mark = read_mark(session_dir, now=now)
    if mark is None:
        return None
    clock = time.time() if now is None else now
    if activity is None:
        from local_operator.session.retention import session_activity

        activity = session_activity(session_dir)
    entry: dict[str, Any] = {}
    evaluated = None
    if state:
        evaluated = state.get("evaluated_at")
        raw = (state.get("sessions") or {}).get(session_dir.name)
        if isinstance(raw, dict):
            entry = raw
    held_at = _finite(entry.get("held_at")) or 0.0
    hold = entry.get("hold") if entry.get("hold") in HOLD_CODES else None
    if not isinstance(evaluated, (int, float)) or clock - evaluated > 3 * TEMPORARY_SWEEP_S:
        hold = None
    idle_since = max(activity or 0.0, mark.marked_at, held_at)
    return {
        "marked_at": mark.marked_at,
        "idle_minutes": idle_minutes,
        "deletes_at": None if hold else idle_since + idle_minutes * 60.0,
        "hold": hold,
    }


def configured_idle_minutes(config_manager: Any = None) -> int:
    """The live window from config, clamped; 30 when it cannot be read."""
    if config_manager is None:
        try:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            config_manager = ConfigManager(config_dir())
        except Exception:  # noqa: BLE001 — unreadable config means the default
            return DEFAULT_IDLE_MINUTES
    getter = getattr(config_manager, "get_nested_value", None)
    if getter is None:
        return DEFAULT_IDLE_MINUTES
    try:
        value = getter(("session", "cleanup", "temporary", "idle_minutes"), DEFAULT_IDLE_MINUTES)
    except Exception:  # noqa: BLE001
        return DEFAULT_IDLE_MINUTES
    return clamp_idle_minutes(value)
