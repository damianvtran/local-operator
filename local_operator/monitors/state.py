"""Per-monitor state files: counters + snapshot blob (contract §10.3).

Two files per monitor under ``<config_dir>/monitors/state/<session_id>/``:

- ``<monitor_id>.json`` — counters/health, ≲ 1 KiB, rewritten every check.
  The per-quiet-tick cost of a monitor is ONE such atomic write.
- ``<monitor_id>.snap`` — the normalized snapshot, ≤ ``snapshotMaxChars``,
  rewritten only on baseline establishment and on change.

The split is why a 30 s monitor does not rewrite 32 KiB 2,880×/day. Neither
file is authoritative: the transcript's ``monitor_schedules`` entry is; both
are cache-like derived state whose loss the next check heals (§7.2).

**Import-light, like the index.** Stdlib only (``json``, ``os``, ``pathlib``,
``tempfile``, ``time``): the same readers that scan the monitor index —
session-cleanup guards, the park marker, the pristine probe, ``lop monitor
status`` — want the state size and shapes without the scheduler. Pinned beside
the wake siblings in ``tests/unit/test_import_graph.py``.

Writes are staged + ``os.replace`` so a reader never sees a torn file; a
failing write is logged and swallowed by the CALLER's contract (the next check
repairs it, and a lost advance costs one re-diff, not a dead monitor).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Subdirectory of ``monitors/`` holding per-session state. The index scan
#: skips anything not ending ``.json``, so state can never be mistaken for an
#: index entry — that is why it lives under its own SUBdirectory (§10.2).
STATE_DIRNAME = "state"

#: Bumped only on an incompatible change; readers skip unknown schemas (the
#: transcript rebuilds both files on the next change anyway).
STATE_SCHEMA = 1


def state_dir(config_dir: Path, session_id: str) -> Path:
    return Path(config_dir) / "monitors" / STATE_DIRNAME / session_id


def counters_path(config_dir: Path, session_id: str, monitor_id: str) -> Path:
    return state_dir(config_dir, session_id) / f"{monitor_id}.json"


def snapshot_path(config_dir: Path, session_id: str, monitor_id: str) -> Path:
    return state_dir(config_dir, session_id) / f"{monitor_id}.snap"


def read_counters(config_dir: Path, session_id: str, monitor_id: str) -> dict[str, Any] | None:
    """One monitor's counters, or ``None`` when absent/unreadable/unknown.

    Unreadable is treated exactly like absent — the next check rebuilds what
    it needs (§7.2's loss semantics) — because a reader that raised would take
    the state rebuild down with one bad file.
    """
    path = counters_path(config_dir, session_id, monitor_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("monitor state: unreadable counters %s; treating as absent", path)
        return None
    if not isinstance(data, dict) or data.get("schema") != STATE_SCHEMA:
        logger.warning("monitor state: skipping counters %s with unknown schema", path)
        return None
    return data


def write_counters(
    config_dir: Path, session_id: str, monitor_id: str, counters: Mapping[str, Any]
) -> Path:
    return _atomic_write_json(counters_path(config_dir, session_id, monitor_id), counters)


def read_snapshot(config_dir: Path, session_id: str, monitor_id: str) -> dict[str, Any] | None:
    """One monitor's snapshot record, or ``None`` when absent/unreadable."""
    path = snapshot_path(config_dir, session_id, monitor_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("monitor state: unreadable snapshot %s; treating as absent", path)
        return None
    if not isinstance(data, dict) or data.get("schema") != STATE_SCHEMA:
        return None
    return data


def snapshot_exists(config_dir: Path, session_id: str, monitor_id: str) -> bool:
    """Cheap probe for the quiet-tick heal (blob-only loss, §7.2)."""
    try:
        return snapshot_path(config_dir, session_id, monitor_id).is_file()
    except OSError:
        return False


def write_snapshot(
    config_dir: Path, session_id: str, monitor_id: str, snapshot: str, *, truncated: bool
) -> Path:
    return _atomic_write_json(
        snapshot_path(config_dir, session_id, monitor_id),
        {
            "schema": STATE_SCHEMA,
            "monitor_id": monitor_id,
            "snapshot": snapshot,
            "snapshot_truncated": bool(truncated),
        },
    )


def remove_monitor_state(config_dir: Path, session_id: str, monitor_id: str) -> None:
    """Drop one monitor's two files (best-effort; a cancel is a user action).

    Then reclaim the session's state directory when this was its last monitor.
    Without the ``rmdir`` a cancel left the empty directory behind forever —
    the live store carried 16 of them, one per armed-then-cancelled session —
    because nothing else sweeps it: the cleanup path and a network carry are
    the only other callers, and neither runs for an ordinary cancel. ``rmdir``
    refuses a non-empty directory, so a sibling monitor's files are never at
    risk; a failure is the expected outcome then, and is swallowed.
    """
    for path in (
        counters_path(config_dir, session_id, monitor_id),
        snapshot_path(config_dir, session_id, monitor_id),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            logger.debug("monitor state: could not remove %s", path, exc_info=True)
    _rmdir_if_empty(state_dir(config_dir, session_id))


def prune_empty_state_dirs(config_dir: Path) -> int:
    """Remove every EMPTY ``state/<session_id>/`` directory; returns the count.

    The sweep half of the cancel-time ``rmdir`` above: directories left by an
    older build (or by a cancel whose ``rmdir`` raced a concurrent write) are
    reclaimed the next time a session opens. ``rmdir`` is the whole safety
    argument — it fails on a directory that holds anything, so this can never
    delete a monitor's state, only the container once its last file is gone.
    """
    root = Path(config_dir) / "monitors" / STATE_DIRNAME
    try:
        children = list(root.iterdir())
    except FileNotFoundError:
        return 0
    except OSError:
        logger.debug("monitor state: could not list %s", root, exc_info=True)
        return 0
    removed = 0
    for child in children:
        if not child.is_dir():
            continue
        if _rmdir_if_empty(child):
            removed += 1
    return removed


def _rmdir_if_empty(directory: Path) -> bool:
    """``rmdir`` one directory, treating a non-empty one as a no-op.

    Returns whether it went. The failure is not logged above ``debug``: a
    sibling monitor still holding files makes "directory not empty" the normal
    answer, not a fault.
    """
    try:
        directory.rmdir()
        return True
    except OSError:
        logger.debug("monitor state: %s not removed (not empty, or gone)", directory)
        return False


def remove_session_state(config_dir: Path, session_id: str) -> None:
    """Drop a whole session's state directory (the cleanup path, §11.5)."""
    directory = state_dir(config_dir, session_id)
    try:
        for child in directory.iterdir():
            if child.is_file():
                try:
                    child.unlink()
                except OSError:
                    logger.debug("monitor state: could not remove %s", child, exc_info=True)
        directory.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        logger.debug("monitor state: could not remove %s", directory, exc_info=True)


def _stage_json_file(directory: Path, name: str) -> tuple[int, str]:
    """Ensure ``directory`` exists and stage a temp file inside it.

    Split out of :func:`_atomic_write_json` so the caller can retry the pair
    as one unit after a directory that vanished mid-cancel.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".tmp")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Staged write + ``os.replace``: a reader never sees a torn file.

    The temp name starts with ``.`` so the index scan (and anything else
    listing the state directory) skips it.

    The ``mkdir`` is retried ONCE: ``remove_monitor_state`` now rmdirs the
    session's directory when its last monitor goes, so a cancel racing a
    sibling's write can delete the directory between the ``mkdir`` and the
    ``mkstemp`` — a ``FileNotFoundError`` from a directory that existed a
    microsecond earlier, which one retry closes. (A genuine mount failure
    still raises on the retry.)
    """
    directory = path.parent
    try:
        fd, tmp = _stage_json_file(directory, path.name)
    except FileNotFoundError:
        fd, tmp = _stage_json_file(directory, path.name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
