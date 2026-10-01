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
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Subdirectory of the config dir holding one ``<session_id>.json`` per
#: monitor-carrying session. Flat: see the module docstring.
MONITORS_DIRNAME = "monitors"

#: Bumped only on an incompatible change to the entry shape. Readers skip
#: entries whose schema they do not understand rather than guess; the owning
#: session rewrites the entry on its next open, so a bump heals like a
#: deleted file.
INDEX_SCHEMA = 1

#: How long an entry with no transcript must sit before the ghost sweep
#: removes it. The floor exists for one race: a session is being CREATED (its
#: directory is not written yet, or its transcript not yet appended) while its
#: index entry lands, and deleting that entry would lose a live arm. An hour
#: is far longer than any creation path takes and far shorter than the time an
#: operator waits to notice a ghost row.
GHOST_MIN_AGE_MS = 3_600_000


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


#: The floor under every staleness threshold below: an interval-derived bound
#: can be seconds long, and "overdue" on a 5 s monitor is not a hint.
HEALTH_MIN_STALE_MS = 300_000


def _stale_threshold_ms(row: Mapping[str, Any]) -> int:
    """``max(2 × every, 5 min)`` — when a monitor counts as not running."""
    every = row.get("every_ms")
    if isinstance(every, int) and not isinstance(every, bool) and every > 0:
        return max(2 * every, HEALTH_MIN_STALE_MS)
    return HEALTH_MIN_STALE_MS


def _clock(ms: int) -> str:
    return time.strftime("%H:%M", time.localtime(ms / 1000))


def health_hint(row: Mapping[str, Any], now_ms: int) -> str | None:
    """One line explaining a monitor that is not doing what the reader assumes.

    THE shared helper for every listing surface (the agent tool's rows, the
    CLI, the desktop route, the TUI band), because the failure this closes was
    a DISCOVERABILITY one: the live store held monitors that had never checked
    (their session was never open), monitors with 0 deliveries after many
    checks, and one disabled by a flapping MCP tool — and every surface
    rendered all four as healthy rows.

    Returns ``None`` when there is nothing to say (a normal, working monitor),
    which is the common case and must stay silent. ``format_age`` is
    deliberately NOT used here: this module is stdlib-only and the reading
    surfaces own their own durations.
    """
    if isinstance(row, Mapping) and row.get("disabled"):
        # A disabled row already renders its own reason and count; the shared
        # hint would only restate it.
        return None
    since = row.get("unavailable_since") if isinstance(row, Mapping) else None
    if isinstance(since, int) and not isinstance(since, bool) and since > 0:
        return f"tool unavailable since {_clock(since)} — retrying"
    if not isinstance(row, Mapping):
        return None
    threshold = _stale_threshold_ms(row)
    checks = row.get("checks")
    checks_n = int(checks) if isinstance(checks, int) and not isinstance(checks, bool) else 0
    if checks_n == 0:
        created = row.get("created_at")
        if not isinstance(created, int) or isinstance(created, bool):
            return None
        if now_ms - created > threshold:
            return "never checked — its session was not open since arming"
        return None
    deliveries = row.get("deliveries")
    deliveries_n = (
        int(deliveries) if isinstance(deliveries, int) and not isinstance(deliveries, bool) else 0
    )
    if checks_n >= 3 and deliveries_n == 0:
        # Neutral, not a warning: a watch that has seen nothing may simply be
        # watching something quiet — or watching the wrong thing.
        return (
            f"{checks_n} checks, 0 deliveries — nothing has changed "
            "(confirm the call observes what you expect)"
        )
    return None


def unavailable_since_of(row: Mapping[str, Any]) -> int:
    """The epoch-ms an unavailable episode began, or 0 (the wire's own reader)."""
    value = row.get("unavailable_since") if isinstance(row, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def is_idle(row: Mapping[str, Any], now_ms: int) -> bool:
    """Whether a monitor is overdue because nothing is hosting it (§D6).

    Overdue beyond ``max(2 × every, 5 min)`` and NOT held: a monitor ticks
    only while its session is open, so a dormant row is normal — but a reader
    looking at "next due 4 hours ago" needs to be told which of the two it is.
    """
    if not isinstance(row, Mapping) or row.get("disabled") or is_held(row):
        return False
    due = row.get("next_due_at")
    if not isinstance(due, int) or isinstance(due, bool):
        return False
    return now_ms - due > _stale_threshold_ms(row)


def idle_detail(row: Mapping[str, Any], now_ms: int) -> str:
    """``overdue by 3h — session not open`` for an idle row (see :func:`is_idle`)."""
    due = row.get("next_due_at")
    overdue_ms = now_ms - int(due) if isinstance(due, int) and not isinstance(due, bool) else 0
    return f"overdue by {format_age_ms(overdue_ms)} — session not open"


def format_age_ms(ms: int) -> str:
    """A compact age: ``45s``, ``12m``, ``3h``, ``2d``.

    Local rather than imported: the reading surfaces that call the helpers
    above have their own formatters (``wakes.display.format_age`` lives behind
    a heavier import), and this module's whole contract is that nothing outside
    the stdlib is on its path.
    """
    seconds = max(0, int(ms // 1000))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


def prune_ghost_entries(
    config_dir: Path,
    now_ms: int,
    *,
    session_exists: Callable[[str], bool],
) -> list[str]:
    """Remove index entries whose session no longer exists; returns the ids.

    A GHOST is an entry that can never be engaged: the session directory is
    gone (a reap, a hand-deleted directory, a QA scratch home), so the row
    still lists monitors that cannot fire and each engage burns a whole
    deadline proving it. The live store carried one (``9a7c31e40b22.json``,
    whose ``cwd`` pointed into a scratch home).

    THREE conditions, all required:

    - the session has no transcript. The predicate is INJECTED rather than
      imported so this module stays stdlib-only (``tests/unit/test_import_graph.py``
      pins that, and the supervisor's own guard is the caller's); a caller that
      cannot prove absence must pass a predicate that answers ``True``;
    - ``updated_at`` is older than :data:`GHOST_MIN_AGE_MS` (the creation race
      above);
    - the entry is not HELD (``stopped_at``) — a parked entry belongs to a
      session someone stopped on purpose, and archive is a hide flag rather
      than destruction (design monitor-tool.md §D5).

    Plus its ``state/<session_id>`` directory, through
    :func:`local_operator.monitors.state.remove_session_state`, which unlinks
    files and rmdirs. Callers treat this as maintenance: an unreadable index
    directory prunes nothing rather than raising.
    """
    from local_operator.monitors import state as monitor_state

    index, read_error = read_index_report(config_dir)
    if read_error:
        return []
    removed: list[str] = []
    for session_id, entry in index.items():
        if is_held(entry):
            continue
        updated_at = entry.get("updated_at")
        if not isinstance(updated_at, int) or isinstance(updated_at, bool):
            continue
        if now_ms - updated_at < GHOST_MIN_AGE_MS:
            continue
        if session_exists(session_id):
            continue
        if not remove_entry(config_dir, session_id):
            continue
        monitor_state.remove_session_state(config_dir, session_id)
        removed.append(session_id)
        logger.info("monitor index: pruned the ghost entry for %s", session_id)
    return removed


def remove_entry(config_dir: Path, session_id: str) -> bool:
    """Delete one session's entry. Idempotent: absent is success."""
    path = entry_path(config_dir, session_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
