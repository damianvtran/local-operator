"""The monitor index: one small JSON file per monitor-carrying session.

``<config_dir>/monitors/<session_id>.json`` answers, for a process that has no
session open, *which sessions have monitors, and which are armed?* Absent file
⇒ no monitors.

**Derived, never authoritative** — the exact ``wakes/store.py`` contract. The
transcript's ``monitor_schedules`` entry (written by
``Session._persist_monitor_schedules``) is the source of truth; this file is a
projection the session rewrites immediately after every transcript append AND
on every open. The open-time rewrite is the self-healing property: a deleted,
corrupt or stale entry is repaired the next time the session is built, and an
entry for a session whose list was emptied is removed.

**Why it lives outside the session directory**: the junk-session reap deletes
whole session directories, and a cold scan (cleanup guards, the park marker,
``lop monitor status``) must not open 4,000 session directories to find the
twelve with monitors. Flat, one file per armed session; the state files live
under ``monitors/state/…`` and the index scan skips anything not ending
``.json``.

**Why this module must stay import-light**: its readers include the session
cleanup guards and the pristine probe, which run before any session is built.
Importing asyncio, pydantic, or anything under ``session``/``harness`` here
would put those on their path. So: stdlib only, pins in
``tests/unit/test_import_graph.py`` beside the wake siblings.

**Divergence from ``wakes/store.py``, deliberate (§10.2)**: this index is NOT
rewritten on every quiet tick — a 30 s monitor would rewrite it 2,880×/day for
no reader's benefit; the counters file alone moves, and a cold reader's
``next_due_at`` is best-effort between change events.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Subdirectory of the config dir holding one ``<session_id>.json`` per
#: monitor-carrying session. Flat: see the module docstring.
MONITORS_DIRNAME = "monitors"

#: Bumped only on an incompatible change to the entry shape. Readers skip
#: entries whose schema they do not understand rather than guess; the owning
#: session rewrites the entry on its next open, so a bump heals like a
#: deleted file.
INDEX_SCHEMA = 1


def monitors_dir(config_dir: Path) -> Path:
    return Path(config_dir) / MONITORS_DIRNAME


def entry_path(config_dir: Path, session_id: str) -> Path:
    return monitors_dir(config_dir) / f"{session_id}.json"


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    dump = getattr(row, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise TypeError(f"not a monitor row: {row!r}")


def is_held(entry: Mapping[str, Any] | None) -> bool:
    """Whether an entry is parked: ``stopped_at`` (the stop lever's marker).

    Monitor entries carry no ``held_at`` — there is no Aida engine for
    monitors — so this is the one key. Kept beside the entry shape so every
    reader does not re-spell it.
    """
    if not isinstance(entry, Mapping):
        return False
    return bool(entry.get("stopped_at"))


def has_armed(entry: Mapping[str, Any] | None, now_ms: int) -> bool:
    """Whether any row in the entry can still FIRE (§11.5's cleanup rule).

    A disabled monitor is failed state; an expired one (`until_at` passed) is
    finished. Neither arms the delete refusal — "a marker that cannot fire is
    not pending live work" (``_has_armed_wake``'s rationale) — while a durable
    or future-bounded one does. Tolerates malformed rows: a scan must not die
    on one bad file.
    """
    if not isinstance(entry, Mapping):
        return False
    for raw in entry.get("monitors") or ():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("disabled"):
            continue
        until = raw.get("until_at")
        if isinstance(until, int) and not isinstance(until, bool) and until <= now_ms:
            continue
        return True
    return False


def next_due_at(entry: Mapping[str, Any]) -> int | None:
    """Earliest ``next_due_at`` across an entry's rows, or ``None``.

    Best-effort by contract: the index is only rewritten on change events, so
    a cold reader's value is the last-written one, not the live schedule.
    """
    earliest: int | None = None
    for raw in entry.get("monitors") or ():
        if not isinstance(raw, Mapping) or raw.get("disabled"):
            continue
        due = raw.get("next_due_at")
        if isinstance(due, bool) or not isinstance(due, int):
            continue
        if earliest is None or due < earliest:
            earliest = due
    return earliest


def read_entry(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """One session's entry, or ``None`` when absent or unreadable.

    Unreadable is treated exactly like absent — the transcript is the truth
    and the next open rewrites the file.
    """
    path = entry_path(config_dir, session_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("monitor index: unreadable entry %s; treating as absent", path)
        return None
    if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
        logger.warning("monitor index: skipping entry %s with unknown schema", path)
        return None
    return data


def read_index(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Every readable entry keyed by session id; a missing directory is empty."""
    return read_index_report(config_dir)[0]


def read_index_report(config_dir: Path) -> tuple[dict[str, dict[str, Any]], bool]:
    """``(entries, read_error)`` — plus whether the DIRECTORY could not be listed.

    The two empty answers are different claims ("no monitors" vs "cannot
    know"), the ``wakes/store.py`` distinction kept verbatim.
    """
    directory = monitors_dir(config_dir)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return {}, False
    except OSError:
        logger.warning("monitor index: cannot list %s", directory)
        return {}, True
    index: dict[str, dict[str, Any]] = {}
    for name in names:
        if not name.endswith(".json") or name.startswith("."):
            continue  # staged temp files, stray dotfiles and state/ are never entries
        session_id = name[: -len(".json")]
        entry = read_entry(config_dir, session_id)
        if entry is None:
            continue
        # The filename is the lookup key; a body whose session_id disagrees is
        # a copied file, and the filename wins because that is what the owning
        # session will overwrite.
        entry["session_id"] = session_id
        index[session_id] = entry
    return index, False


def write_entry(
    config_dir: Path,
    session_id: str,
    *,
    cwd: str,
    monitors: Sequence[Any],
    preserve: Mapping[str, Any] | None = None,
    clear: tuple[str, ...] = (),
) -> Path | None:
    """Write (replace) one session's entry, or remove it when empty.

    ``preserve`` keeps unknown keys across the rewrite (today ``stopped_at``,
    which the park marker stamps and the open-time rewrite clears); keys named
    in ``clear`` are dropped even when present in ``preserve``.
    """
    rows = [_row_dict(row) for row in monitors]
    if not rows:
        remove_entry(config_dir, session_id)
        return None
    entry: dict[str, Any] = {}
    if preserve:
        entry.update({k: v for k, v in preserve.items() if k not in clear})
    entry.update(
        {
            "schema": INDEX_SCHEMA,
            "session_id": session_id,
            "cwd": cwd,
            "updated_at": int(time.time() * 1000),
            "monitors": rows,
        }
    )
    path = entry_path(config_dir, session_id)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{session_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(entry, handle, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def remove_entry(config_dir: Path, session_id: str) -> bool:
    """Delete one session's entry. Idempotent: absent is success."""
    path = entry_path(config_dir, session_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
