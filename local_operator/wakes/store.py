"""The wake index: one small JSON file per wake-carrying session.

``<config_dir>/wakes/<session_id>.json`` answers, for a process that has no
session open, the one question a wake trigger needs: *which sessions have
schedules, and when is the next one due?* Absent file ⇒ no wakes.

**Derived, never authoritative.** The transcript's ``wake_schedules`` custom
entry (written by ``Session._persist_wake_schedules``) is the source of truth
for a session's schedules; this file is a projection of it that the session
rewrites immediately after every transcript append AND on every open (after
``_load_wake_schedules``). That second rewrite is the self-healing property:
a deleted, corrupt, or stale index entry is repaired the next time the
session is built, and an entry for a session whose schedules were emptied is
removed. Nothing ever reads this file to decide what a session's schedules
*are* — a reader that finds the file wrong should open the session, not
patch the file.

**Why it lives outside the session directory.** Two reasons, both about the
readers. The junk-session reap deletes whole session directories, and an
index that lived inside one would vanish with the transcript it describes
(correct) but also be invisible to a scan that does not want to open 4,000
session directories to find the twelve with wakes (not acceptable for a
process that ticks every minute). A flat ``wakes/`` directory holds exactly
one file per session that has something scheduled, so the scan is
O(sessions-with-wakes), and the directory is empty on the typical machine.

**Why this module must stay import-light.** Its readers are the wake
supervisor — a ~40 MB always-on process whose whole justification is that it
does NOT load the harness — and the TUI's session picker at boot, before any
session exists. Importing ``asyncio``, ``pydantic``, or anything under
``local_operator.session``/``harness`` here would put those on the
supervisor's resident set and on the picker's startup path, and it would
break the cold-boot rule that the picker opens no session. So: stdlib only.
Schedules are handled as plain dicts (the ``model_dump()`` form the transcript
already stores); the ``WakeSchedule`` type is imported for annotations only.
``tests/unit/test_import_graph.py`` pins this in a fresh interpreter.

Every writer here is best-effort *by contract with its caller*: the session
wraps the call so a failed index write can never break wake persistence — the
transcript entry is what matters, and the next open rebuilds the index.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from local_operator.harness.wake import WakeSchedule

logger = logging.getLogger(__name__)

#: Subdirectory of the config dir holding one ``<session_id>.json`` per
#: wake-carrying session. Flat, not nested under ``sessions/``: see the module
#: docstring for why the index deliberately does not live with the transcript.
WAKES_DIRNAME = "wakes"

#: Bumped only on an incompatible change to the entry shape. Readers skip
#: entries whose schema they do not understand rather than guess, and the
#: owning session rewrites the entry on its next open — so a schema bump
#: heals itself the same way a deleted file does.
INDEX_SCHEMA = 1


def wakes_dir(config_dir: Path) -> Path:
    return Path(config_dir) / WAKES_DIRNAME


def entry_path(config_dir: Path, session_id: str) -> Path:
    return wakes_dir(config_dir) / f"{session_id}.json"


def _schedule_dict(schedule: "WakeSchedule | Mapping[str, Any]") -> dict[str, Any]:
    """Accept either the pydantic model or its already-dumped dict.

    Duck-typed on ``model_dump`` rather than an ``isinstance`` check so this
    module never imports the model at runtime (see the import-light rule).
    """
    if isinstance(schedule, Mapping):
        return dict(schedule)
    dump = getattr(schedule, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise TypeError(f"not a wake schedule: {schedule!r}")


def next_due_at(entry: Mapping[str, Any]) -> int | None:
    """Earliest ``next_due_at`` (epoch ms) across an entry's schedules, or
    ``None`` when the entry carries none. Tolerates malformed rows: a
    supervisor scanning many entries must not die on one bad file."""
    earliest: int | None = None
    for raw in entry.get("schedules") or ():
        if not isinstance(raw, Mapping):
            continue
        due = raw.get("next_due_at")
        if isinstance(due, bool) or not isinstance(due, int):
            continue
        if earliest is None or due < earliest:
            earliest = due
    return earliest


def read_entry(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """One session's entry, or ``None`` when absent or unreadable.

    Unreadable is treated exactly like absent — "no wakes known" — because
    the transcript is the truth and the next open rewrites the file. A
    reader that raised on a half-written file would take down a scan over
    every other session for one bad entry.
    """
    path = entry_path(config_dir, session_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("wake index: unreadable entry %s; treating as absent", path, exc_info=True)
        return None
    if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
        logger.warning("wake index: skipping entry %s with unknown schema", path)
        return None
    return data


def read_index(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Every readable entry keyed by session id. A missing directory is an
    empty index — the common case on a machine that has never set a wake."""
    directory = wakes_dir(config_dir)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return {}
    except OSError:
        logger.warning("wake index: cannot list %s", directory, exc_info=True)
        return {}
    index: dict[str, dict[str, Any]] = {}
    for name in names:
        if not name.endswith(".json") or name.startswith("."):
            continue  # staged temp files and stray dotfiles are never entries
        session_id = name[: -len(".json")]
        entry = read_entry(config_dir, session_id)
        if entry is None:
            continue
        # The filename is the lookup key; a body whose ``session_id`` disagrees
        # is a copied or hand-edited file, and the filename wins because that
        # is what the owning session will overwrite.
        entry["session_id"] = session_id
        index[session_id] = entry
    return index


def write_entry(
    config_dir: Path,
    session_id: str,
    *,
    cwd: str,
    schedules: "Sequence[WakeSchedule | Mapping[str, Any]]",
    preserve: Mapping[str, Any] | None = None,
    clear: tuple[str, ...] = (),
) -> Path | None:
    """Write (replace) one session's entry. Returns the path, or ``None``
    when ``schedules`` is empty — an empty list means *remove* the entry, so
    "no file" and "no wakes" stay the same statement.

    ``preserve`` is an existing entry whose unknown keys should survive the
    rewrite (today: ``stopped_at``, which a later change uses to keep a
    stopped session's wakes dormant). Keys named in ``clear`` are dropped even
    if present in ``preserve`` — the open-time rewrite clears ``stopped_at``
    because opening a session is what un-stops it.

    Staged write + ``os.replace`` so a reader never sees a torn file: the
    supervisor may scan while the session writes, and ``replace`` is atomic
    on every platform we run on. The temp name starts with ``.`` so the
    directory scan in :func:`read_index` skips it.
    """
    rows = [_schedule_dict(s) for s in schedules]
    path = entry_path(config_dir, session_id)
    if not rows:
        remove_entry(config_dir, session_id)
        return None
    entry: dict[str, Any] = {}
    if preserve:
        # Unknown keys ride along untouched; the fields below are rewritten
        # from the caller's (authoritative) values regardless of what the
        # old file said.
        entry.update({k: v for k, v in preserve.items() if k not in clear})
    entry.update(
        {
            "schema": INDEX_SCHEMA,
            "session_id": session_id,
            "cwd": cwd,
            "updated_at": int(time.time() * 1000),
            "schedules": rows,
        }
    )
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
