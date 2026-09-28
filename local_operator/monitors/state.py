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
    """Drop one monitor's two files (best-effort; a cancel is a user action)."""
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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Staged write + ``os.replace``: a reader never sees a torn file.

    The temp name starts with ``.`` so the index scan (and anything else
    listing the state directory) skips it.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
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
