"""Session transcript retention: nothing is ever deleted.

A session transcript is the only durable record of what a run did — the
work it performed, the decisions it made, and the state a user may need to
resume from hours or months later. Earlier this module enforced age, count,
and byte ceilings over ``sessions/`` and evicted the oldest directories at
startup. That policy destroyed real work in practice: ceilings that looked
generous (200 directories, 128 MiB) were reached on heavy installs, and the
eviction could take out the transcript of a session that was still running
in another process, leaving that run to die on
``FileNotFoundError: .../sessions/<id>/transcript.jsonl`` with no way to
continue. A session in that state loses everything since its last save.

The rule now is absolute: **no sweep, automation, or startup hook deletes a
session transcript, under any circumstance.** Session history is removed
only when the user explicitly disposes of it. Disk pressure is a real
concern — the harness this project is benchmarked against accumulated
5.9 GB of transcripts — but the answer to disk pressure is a tool the user
chooses to run, never a silent ceiling, because the cost asymmetry is
extreme: gigabytes of recoverable disk versus hours of unrecoverable work.

What this module still does, because it is definitionally safe, is reap
EMPTY session directories. An empty directory holds no ``transcript.jsonl``
and no other content — it is left behind by a run that created its session
directory and exited before writing a single turn (23 of 147 directories on
a real install were exactly this). Nothing can be lost by removing a
directory that contains nothing, and an un-empty directory — one byte of
transcript — is never touched, whatever its age, whatever the count of its
neighbours, whatever the total size of the store.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Directory under the config dir holding ephemeral per-run transcripts.
SESSIONS_DIRNAME = "sessions"

#: Config keys of the RETIRED eviction ceilings. They no longer do anything
#: — deleting transcripts by policy is precisely the behaviour this module
#: exists to prevent — but they are still read by :func:`sweep_from_config`
#: so that a config file carrying them produces an honest log line once per
#: startup instead of a silent no-op that looks like the ceilings still run.
MAX_SESSIONS_KEY = "session_retention_max_sessions"
MAX_BYTES_KEY = "session_retention_max_bytes"
MAX_AGE_DAYS_KEY = "session_retention_max_age_days"

#: The ceilings are retired, so the only correct value for each is 0
#: ("disabled"). Exported under the old names because the configuration
#: reference and older call sites import them; they exist to say "no
#: ceiling", never to be tuned.
DEFAULT_MAX_SESSIONS = 0
DEFAULT_MAX_BYTES = 0
DEFAULT_MAX_AGE_DAYS = 0

#: How old an EMPTY session directory must be before the reaper may take it.
#:
#: A session directory is empty from the moment ``_prepare`` creates it until
#: the first turn is appended to ``transcript.jsonl`` — and that window is as
#: long as the user takes to type their first message. ``live_dir`` only
#: protects the sweeping process's OWN session; on an install running many
#: concurrent sessions, a sibling process starting up in that window saw a
#: fresh empty directory, judged it abandoned, and rmtree'd a session another
#: process was about to write into. Its first append then died on
#: ``FileNotFoundError: .../transcript.jsonl`` — observed in production on an
#: install with ~12 concurrent sessions, where new sessions started minutes
#: apart reliably raced each other.
#:
#: One hour is deliberately generous: an empty directory costs nothing while
#: it waits, and a directory abandoned by a crashed run is still reclaimed by
#: any startup an hour later. The transcript's self-healing append (see
#: ``Transcript._append``) backstops the pathological case of a session left
#: idle past the grace window before its first message.
EMPTY_DIR_GRACE_SECONDS = 3600.0


@dataclass(frozen=True)
class SweepResult:
    """What one sweep did. Returned rather than logged so the benchmark and
    the tests can assert on it instead of scraping log lines.

    ``evicted`` counts only empty directories — a non-zero value never means
    a transcript was removed, because transcripts are never removed."""

    scanned: int = 0
    evicted: int = 0
    bytes_freed: int = 0
    bytes_remaining: int = 0
    errors: int = 0

    @property
    def changed(self) -> bool:
        return self.evicted > 0


def _dir_size(directory: Path) -> int:
    """Bytes under ``directory``. Files that vanish mid-walk are skipped: a
    concurrent process disposing its own session is normal, not an error.

    Every file counts, including the origin marker. A directory holding
    only ``origin.json`` is a session that has been claimed — typically a
    child between ``mark_session_origin`` and its first append, or a run
    that aborted in that window. Treating the marker as invisible made
    those directories look empty and the sweep rmtree'd them, which is
    exactly the ``FileNotFoundError: .../transcript.jsonl`` kill this
    module exists to prevent. A 43-byte marker is cheaper than a lost
    session; abandoned markers accumulate and the user can remove them.
    """
    total = 0
    for entry in directory.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def sweep_sessions(
    sessions_dir: Path,
    *,
    live_dir: Path | None = None,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    now: float | None = None,
) -> SweepResult:
    """Reap EMPTY session directories. Never delete anything else.

    The ceiling parameters are accepted for call-site compatibility and
    are IGNORED: no configuration can make this function delete a
    directory that holds content. ``live_dir`` is still honoured as a
    belt: even a literally-empty live directory is skipped, because the
    caller just created it and has not written a turn yet. A transcript
    is removed only when the user explicitly disposes of the session —
    there is no automated path, because the failure mode of an automated
    path (a running session's transcript vanishing underneath it) costs
    the user the whole session, and the benefit (bounded disk) is
    recoverable by hand at any time.

    An empty directory younger than :data:`EMPTY_DIR_GRACE_SECONDS` is also
    skipped, whoever owns it: a fresh empty directory is indistinguishable
    from a sibling process's session that has not received its first message
    yet, and reaping one killed exactly such a session on a real install.
    ``now`` exists so tests can move the clock instead of the filesystem.

    Idempotent and safe to call on every startup: a missing ``sessions_dir``
    is a no-op rather than an error — the first run of a fresh install has
    not created it yet, and a startup path that raises there would be a
    regression traded for nothing.
    """
    if not sessions_dir.is_dir():
        return SweepResult()

    scanned = 0
    evicted = 0
    errors = 0
    bytes_remaining = 0
    moment = time.time() if now is None else now
    live_resolved = live_dir.resolve() if live_dir is not None else None
    try:
        children = [child for child in sessions_dir.iterdir() if child.is_dir()]
    except OSError as exc:
        logger.warning("session retention: cannot scan %s: %s", sessions_dir, exc)
        return SweepResult(errors=1)

    for child in children:
        scanned += 1
        try:
            if live_resolved is not None and child.resolve() == live_resolved:
                # The caller just created this directory and has not written
                # a turn. It is empty by construction and must still survive.
                continue
            size = _dir_size(child)
        except OSError:
            # A directory another process removed mid-scan. Nothing to do.
            continue
        if size > 0:
            bytes_remaining += size
            continue
        try:
            age = moment - child.stat().st_mtime
        except OSError:
            # Removed by another process between the size walk and the stat.
            continue
        if age < EMPTY_DIR_GRACE_SECONDS:
            # Empty but fresh: this is what a sibling's just-created session
            # looks like before its first turn lands. ``live_dir`` cannot
            # protect it (it only names OUR session), so age is the only
            # signal that separates "abandoned" from "about to be written".
            continue
        # Literally empty AND past the grace window: no files at all, not
        # even a marker, and nothing has claimed it for an hour. Deleting
        # it loses nothing, and it was never a real session.
        try:
            shutil.rmtree(child)
        except OSError as exc:
            # Best-effort by construction: reclaiming disk must never be the
            # reason a session fails to start.
            logger.debug("session retention: cannot remove %s: %s", child, exc)
            errors += 1
            continue
        evicted += 1

    result = SweepResult(
        scanned=scanned,
        evicted=evicted,
        bytes_freed=0,  # empty directories free nothing measurable
        bytes_remaining=bytes_remaining,
        errors=errors,
    )
    if result.changed:
        logger.info(
            "session retention: reaped %d empty session directories (of %d)",
            result.evicted,
            result.scanned,
        )
    return result


def sweep_from_config(
    config_manager: object, config_dir: Path, live_dir: Path | None
) -> SweepResult:
    """Run :func:`sweep_sessions` for the store under ``config_dir``.

    ``config_manager`` is duck-typed on ``get_config_value``. The retired
    ceiling keys are read once so that a config still carrying them gets a
    single honest warning — transcripts are never deleted — rather than the
    silence that would let a user believe a ceiling is still protecting (or
    still endangering) anything.
    """

    getter = getattr(config_manager, "get_config_value", None)
    if getter is not None:
        for key in (MAX_SESSIONS_KEY, MAX_BYTES_KEY, MAX_AGE_DAYS_KEY):
            try:
                value = getter(key, 0)
            except (TypeError, ValueError):
                continue
            try:
                configured = int(value)
            except (TypeError, ValueError):
                configured = 0
            if configured:
                logger.warning(
                    "session retention: %s=%s is retired and ignored; "
                    "session transcripts are never deleted automatically",
                    key,
                    value,
                )

    return sweep_sessions(config_dir / SESSIONS_DIRNAME, live_dir=live_dir)
