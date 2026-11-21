"""Write-once conversation birth dates, independent of activity and process owners.

Listing is read-only: old stores without a recoverable creation date tie at zero
by session id. Never manufacture a date from a heartbeat, transcript mtime, or
first journal entry (forks inherit that entry and compaction can remove it).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

CREATED_AT_NAME = "created_at.json"

#: THE SESSION'S DURABLE WORKING-DIRECTORY RECORD, ``{"v": 1, "cwd": <absolute>,
#: "at": <epoch>, "source": "birth" | "runtime"}``.
#:
#: WHY IT EXISTS (coordinator retest on v0.111.4): a session started by ``lop exec``
#: or the TUI carries no ``desktop.json`` marker, is in no wake index, and once its
#: runtime exits publishes no discovery record either — so ``lop sessions move
#: --dry-run`` could not tell which folder it worked in. The git worktree holding its
#: unpushed commit and dirty files then showed only as "mentioned", nothing was
#: planned, no confirmation was asked, and a real move would have silently left the
#: work behind. This file is the one record every surface writes: at birth (the
#: transcript constructor, beside :data:`CREATED_AT_NAME`) and again whenever a
#: runtime starts in, or moves to, a different absolute cwd.
#:
#: It is deliberately NOT ``desktop.json``: creating that marker changes how the
#: desktop treats a TUI-created session (``network.carry.mark_moved_in``), and a cwd
#: record must not. Owner-only (0600) and atomic like the marker: it names a path on
#: this device. Readers apply the move's S-2 rule (absolute after ``~``, never the
#: session directory) — see ``network.mobility._session_cwd_at``.
CWD_RECORD_NAME = "cwd.json"
CWD_RECORD_VERSION = 1


def _timestamp(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _stored(directory: Path) -> float | None:
    try:
        return _timestamp(json.loads((directory / CREATED_AT_NAME).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def session_created_at(directory: Path) -> float:
    """Read canonical birth, then legacy fork provenance/filesystem birth, or zero."""
    stored = _stored(directory)
    if stored is not None:
        return stored
    try:
        origin = json.loads((directory / "origin.json").read_text(encoding="utf-8"))
        forked = _timestamp(origin.get("forked_at")) if isinstance(origin, dict) else None
        if isinstance(origin, dict) and origin.get("origin") == "fork" and forked is not None:
            return forked
    except (OSError, ValueError):
        pass
    try:
        return _timestamp(getattr(directory.stat(), "st_birthtime", None)) or 0.0
    except OSError:
        return 0.0


def ensure_session_created_at(directory: Path, proposed: float) -> float:
    """Publish complete bytes without clobbering another owner's winning date.

    A hard link publishes atomically and exclusively, unlike exists+replace or
    opening the destination with O_EXCL before its bytes have been written.
    Read-only/corrupt stores remain usable; this metadata must not lose a turn.
    """
    stored = _stored(directory)
    if stored is not None:
        return stored
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
            temporary = handle.name
            json.dump(proposed, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, directory / CREATED_AT_NAME)
        except FileExistsError:
            pass
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    stored = _stored(directory)
    return stored if stored is not None else session_created_at(directory)


def read_session_cwd(directory: Path) -> str:
    """The ``cwd`` :data:`CWD_RECORD_NAME` records for this session, or ``""``.

    Returns the stored string unvalidated beyond its type: whether it is usable
    (absolute, not the session directory) is the reader's rule, applied in one place
    (``mobility._session_cwd_at``'s ``usable``), so a relative or stale value is
    rejected there rather than silently here. Never raises.
    """
    try:
        document = json.loads((Path(directory) / CWD_RECORD_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    cwd = document.get("cwd") if isinstance(document, dict) else None
    return cwd if isinstance(cwd, str) else ""


def write_session_cwd(directory: Path, cwd: str, *, source: str, at: float | None = None) -> bool:
    """Write :data:`CWD_RECORD_NAME` atomically (temp + ``os.replace``), 0600.

    ``cwd`` may be ``""`` only for the arrival remap's "dropped" case
    (``network.carry.remap_cwd_record``), which must overwrite a foreign path rather
    than leave it. NEVER CREATES THE SESSION DIRECTORY: a speculatively warmed
    runtime (``Transcript(defer_materialise=True)``) must leave nothing on disk, so a
    missing directory is a ``False`` the caller retries later, never a ``mkdir``.
    Never raises: this is metadata, and it must not cost a turn or a boot.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return False
    document = {
        "v": CWD_RECORD_VERSION,
        "cwd": str(cwd),
        "at": float(at if at is not None else time.time()),
        "source": str(source),
    }
    temporary = directory / f"{CWD_RECORD_NAME}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, sort_keys=True))
        os.replace(temporary, directory / CWD_RECORD_NAME)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        return False
    return True


def record_session_cwd(directory: Path, cwd: str | None, *, source: str) -> bool:
    """Record ``cwd`` as this session's working directory when it says something new.

    ``source="birth"`` writes only when no record exists yet: a transcript is
    constructed on EVERY open (resume, viewer, monitor arm), and a later open from
    another folder must not overwrite what the creator recorded — the runtime's own
    record (``source="runtime"``) is what follows a real change of directory.
    ``source="runtime"`` writes when the recorded value differs. A relative or empty
    ``cwd`` is never written (S-2: it would resolve against whatever process reads
    it). Returns whether the file now holds ``cwd``. Never raises.
    """
    if not isinstance(cwd, str) or not cwd.strip():
        return False
    value = os.path.expanduser(cwd.strip())
    if not os.path.isabs(value):
        return False
    existing = read_session_cwd(directory)
    if existing == value:
        return True
    if source == "birth" and (Path(directory) / CWD_RECORD_NAME).exists():
        return False
    return write_session_cwd(directory, value, source=source)


def session_category(*, pending: bool, busy: bool, unseen: bool, kind: str, live: bool) -> int:
    """Decision gates stay first; completed outcomes outrank work in progress.

    A resumed busy turn is not the stale completion it has not acknowledged yet.
    Within each category callers use only birth date and immutable session id.
    """
    if pending:
        return 0
    if not busy and unseen:
        return {"error": 2, "interrupted": 3}.get(kind, 1)
    if busy:
        return 4
    return 5 if live else 6
