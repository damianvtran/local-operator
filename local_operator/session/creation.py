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
from pathlib import Path

CREATED_AT_NAME = "created_at.json"


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
