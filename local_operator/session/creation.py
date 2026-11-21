"""Write-once conversation birth dates, independent of activity and process owners.

Listing is read-only: old stores without a recoverable creation date tie at zero
by session id. Never manufacture a date from a heartbeat, transcript mtime, or
first journal entry (forks inherit that entry and compaction can remove it).
"""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

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


#: Upper bound on a ``cwd.json`` read. The record is one short JSON object (a path, a
#: version, a timestamp and a source word: well under 1 KiB for any real path, and
#: ``PATH_MAX`` is 4 KiB on Linux, 1 KiB on macOS). 64 KiB leaves room for any path a
#: filesystem accepts while keeping a planted multi-gigabyte file from being read
#: whole by a move's preflight.
_CWD_RECORD_MAX_BYTES = 64 * 1024


def _open_session_dir(directory: Path) -> int | None:
    """A verified handle on the session directory, or ``None``.

    WHY A HANDLE (security review #4403, S3): every later step — the bounded read,
    the private staging file, the publish — is done relative to this descriptor, so
    a session directory swapped for a symlink between steps cannot redirect a write
    elsewhere. ``O_NOFOLLOW`` refuses a symlinked final component, and the ``fstat``
    confirms it is a directory this user owns (a directory another user controls
    could swap entries under us). Never creates anything.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(directory), flags)
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            os.close(descriptor)
            return None
    except OSError:
        os.close(descriptor)
        return None
    return descriptor


def _read_record_at(dir_fd: int) -> dict[str, Any]:
    """The record under ``dir_fd`` as a dict, or ``{}``. Never raises.

    Bounded, no-follow and regular-file-only (S3): a symlink planted as ``cwd.json``
    is refused by ``O_NOFOLLOW``, a FIFO cannot block the read (``O_NONBLOCK``, then
    the ``S_ISREG`` check), and at most :data:`_CWD_RECORD_MAX_BYTES` are read.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(CWD_RECORD_NAME, flags, dir_fd=dir_fd)
    except OSError:
        return {}
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _CWD_RECORD_MAX_BYTES:
            return {}
        raw = os.read(descriptor, _CWD_RECORD_MAX_BYTES + 1)
    except OSError:
        return {}
    finally:
        os.close(descriptor)
    if len(raw) > _CWD_RECORD_MAX_BYTES:
        return {}
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def read_session_cwd(directory: Path) -> str:
    """The ``cwd`` :data:`CWD_RECORD_NAME` records for this session, or ``""``.

    Returns the stored string unvalidated beyond its type: whether it is usable
    (absolute, an existing directory under this home, not a protected one) is the
    reader's rule, applied in one place (``mobility._session_cwd_at``), so a relative
    or stale value is rejected there rather than silently here. The read itself is
    the safe one (:func:`_read_record_at`). Never raises.
    """
    cwd = read_session_cwd_document(directory).get("cwd")
    return cwd if isinstance(cwd, str) else ""


def read_session_cwd_document(directory: Path) -> dict[str, Any]:
    """The whole :data:`CWD_RECORD_NAME` document (``{}`` when absent or unsafe).

    For the arrival remap, which keeps the record's ``source`` while it rewrites the
    path. Same safe read as :func:`read_session_cwd`. Never raises.
    """
    dir_fd = _open_session_dir(Path(directory))
    if dir_fd is None:
        return {}
    try:
        return _read_record_at(dir_fd)
    finally:
        os.close(dir_fd)


def write_session_cwd(
    directory: Path,
    cwd: str,
    *,
    source: str,
    at: float | None = None,
    exclusive: bool = False,
) -> bool:
    """Publish :data:`CWD_RECORD_NAME`, 0600, through private staging. Never raises.

    ``cwd`` may be ``""`` only for the arrival remap's "dropped" case
    (``network.carry.remap_cwd_record``), which must overwrite a foreign path rather
    than leave it. NEVER CREATES THE SESSION DIRECTORY: a speculatively warmed
    runtime (``Transcript(defer_materialise=True)``) must leave nothing on disk, so a
    missing directory is a ``False`` the caller retries later, never a ``mkdir``.

    THE FILESYSTEM-IDENTITY CONTRACT (security review #4403, S3). Everything happens
    relative to one verified directory handle (:func:`_open_session_dir`). The bytes
    go to a randomly named staging file created ``O_EXCL | O_NOFOLLOW`` with mode
    0600, so no pre-existing inode (a planted file or symlink at a predictable name)
    is ever written through. Then:

    * ``exclusive=True`` (a birth) publishes with a hard link, which fails if a
      record already exists — the same no-clobber publication
      ``ensure_session_created_at`` uses — so two racing births cannot overwrite each
      other's winner; returns ``False`` when another writer won;
    * otherwise (a runtime change of directory, the arrival remap) ``os.replace``
      swaps the complete record in atomically.

    The staging name keeps the ``cwd.json.<token>.tmp`` shape that
    ``sync.assert_complete`` already tolerates as a crash leftover.
    """
    dir_fd = _open_session_dir(Path(directory))
    if dir_fd is None:
        return False
    document = {
        "v": CWD_RECORD_VERSION,
        "cwd": str(cwd),
        "at": float(at if at is not None else time.time()),
        "source": str(source),
    }
    temporary = f"{CWD_RECORD_NAME}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    published = False
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=dir_fd)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                os.link(temporary, CWD_RECORD_NAME, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                published = True
            except FileExistsError:
                published = False
        else:
            os.replace(temporary, CWD_RECORD_NAME, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            published = True
    except OSError:
        published = False
    finally:
        # The staging name is this call's own (random, O_EXCL): after a link it is a
        # second name for the record, after a replace it is already gone, and after
        # a failure it is a partial file. Dropping it is never another writer's data.
        try:
            os.unlink(temporary, dir_fd=dir_fd)
        except OSError:
            pass
        os.close(dir_fd)
    return published


def record_session_cwd(directory: Path, cwd: str | None, *, source: str) -> bool:
    """Record ``cwd`` as this session's working directory when it says something new.

    ``source="birth"`` writes only when no record exists yet: a transcript is
    constructed on EVERY open (resume, viewer, monitor arm), and a later open from
    another folder must not overwrite what the creator recorded — the runtime's own
    record (``source="runtime"``) is what follows a real change of directory. The
    birth publish is exclusive (:func:`write_session_cwd`), so the "no record yet"
    check cannot race another writer. ``source="runtime"`` writes when the recorded
    value differs. A relative or empty ``cwd`` is never written (S-2: it would
    resolve against whatever process reads it). Returns whether the file now holds
    ``cwd``. Never raises.
    """
    if not isinstance(cwd, str) or not cwd.strip():
        return False
    value = os.path.expanduser(cwd.strip())
    if not os.path.isabs(value):
        return False
    existing = read_session_cwd(directory)
    if existing == value:
        return True
    if source == "birth":
        return write_session_cwd(directory, value, source=source, exclusive=True)
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
