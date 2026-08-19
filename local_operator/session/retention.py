"""Bounded retention for ephemeral session directories.

``sessions/<hex>/transcript.jsonl`` is written by every non-``--train`` run
and read by nobody afterwards. Nothing deleted it, so the directory grew for
the lifetime of the install — the same shape that let the harness this project
is benchmarked against accumulate 5.9 GB of session transcripts and exhaust a
developer's volume. Capping what enters the context does nothing about that
half; this module is the other half.

Three independent ceilings, any of which may be disabled by setting it to 0:

- **age** — a session directory older than N days;
- **count** — at most N session directories;
- **total bytes** — at most N bytes across all of them.

They compose rather than override: eviction runs age, then count, then bytes,
oldest first, until every ceiling holds. Age alone would let a burst of
activity blow the disk budget inside the window; bytes alone would keep a
single ancient directory forever. Whatever the ceilings say, the LIVE session
is never a candidate — evicting the transcript of the run that is currently
appending to it would take out resume and compaction replay together, which
is a far worse outcome than the disk it reclaims.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from local_operator.resume import ORIGIN_NAME

logger = logging.getLogger(__name__)

#: Directory under the config dir holding ephemeral per-run transcripts.
SESSIONS_DIRNAME = "sessions"

#: Config keys. Read through ``ConfigManager.get_config_value`` so the
#: ceilings are editable with ``local-operator config edit`` like every other
#: setting, rather than needing a second configuration mechanism.
MAX_SESSIONS_KEY = "session_retention_max_sessions"
MAX_BYTES_KEY = "session_retention_max_bytes"
MAX_AGE_DAYS_KEY = "session_retention_max_age_days"

#: Defaults. Measured against real runs: a working session costs ~1.3 KB per
#: turn on disk after the slim encoding, and a heavy 60-turn day is ~80 KB.
#: 200 sessions therefore lands around 16 MB in practice, and the 128 MiB
#: byte ceiling is the backstop for the outlier — a session that dumps
#: megabytes of tool output cannot push the total past it no matter how few
#: directories there are. The worst case a heavy user can reach is exactly
#: 128 MiB plus the live session, versus unbounded before.
DEFAULT_MAX_SESSIONS = 200
DEFAULT_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_AGE_DAYS = 30


@dataclass(frozen=True)
class SweepResult:
    """What one sweep did. Returned rather than logged so the benchmark and
    the tests can assert on it instead of scraping log lines."""

    scanned: int = 0
    evicted: int = 0
    bytes_freed: int = 0
    bytes_remaining: int = 0
    errors: int = 0

    @property
    def changed(self) -> bool:
        return self.evicted > 0


@dataclass(frozen=True)
class _Candidate:
    path: Path
    mtime: float
    size: int


def _dir_size(directory: Path) -> int:
    """Bytes under ``directory``. Files that vanish mid-walk are skipped: a
    concurrent process disposing its own session is normal, not an error.

    The origin marker is EXCLUDED from the total, which is what keeps the
    "empty directories are always reaped" rule meaning what it says. A session
    is stamped before its transcript exists, so a run that aborts in between
    leaves a directory holding nothing but a 43-byte marker. Counting those
    bytes turned a directory that carried nothing to lose into an ordinary
    keep candidate occupying a retention slot — measured: with a count ceiling
    of 3, two aborted children evicted two of the user's real transcripts to
    keep two empty markers. The marker is bookkeeping ABOUT the session, never
    session content, so it is not what the ceilings are budgeting.
    """
    total = 0
    for entry in directory.rglob("*"):
        try:
            if entry.is_file() and entry.name != ORIGIN_NAME:
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _candidates(sessions_dir: Path, live: Path | None) -> list[_Candidate]:
    """Evictable session directories, oldest first.

    ``mtime`` of the directory rather than of ``transcript.jsonl``: a session
    may hold other files, and the directory's own mtime moves whenever any of
    them is created. Directories are sorted oldest-first so every ceiling
    evicts in the same, predictable order.
    """
    live_resolved = live.resolve() if live is not None else None
    out: list[_Candidate] = []
    for child in sessions_dir.iterdir():
        try:
            if not child.is_dir():
                continue
            if live_resolved is not None and child.resolve() == live_resolved:
                continue
            stat = child.stat()
        except OSError:
            continue
        out.append(_Candidate(path=child, mtime=stat.st_mtime, size=_dir_size(child)))
    out.sort(key=lambda candidate: candidate.mtime)
    return out


def sweep_sessions(
    sessions_dir: Path,
    *,
    live_dir: Path | None = None,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    now: float | None = None,
) -> SweepResult:
    """Evict session directories until every enabled ceiling holds.

    Idempotent and safe to call on every startup: a second call over a swept
    directory evicts nothing. A missing ``sessions_dir`` is a no-op rather
    than an error — the first run of a fresh install has not created it yet,
    and a startup path that raises there would be a regression traded for
    disk. ``live_dir`` is excluded from every pass.

    Empty directories are always reaped regardless of the ceilings. They are
    left behind by runs that built a session and exited before writing a
    turn, they carry nothing to lose, and on a real install 23 of 147 session
    directories were exactly this.
    """
    if not sessions_dir.is_dir():
        return SweepResult()

    horizon = (now if now is not None else time.time()) - max_age_days * 86400
    try:
        candidates = _candidates(sessions_dir, live_dir)
    except OSError as exc:
        logger.warning("session retention: cannot scan %s: %s", sessions_dir, exc)
        return SweepResult(errors=1)

    keep: list[_Candidate] = []
    doomed: list[_Candidate] = []
    for candidate in candidates:
        if candidate.size == 0:
            doomed.append(candidate)
        elif max_age_days > 0 and candidate.mtime < horizon:
            doomed.append(candidate)
        else:
            keep.append(candidate)

    # Count then bytes, both oldest-first off the front of ``keep``. Bytes
    # runs last because it is the ceiling that must hold unconditionally:
    # trimming by count first often satisfies it for free.
    if max_sessions > 0 and len(keep) > max_sessions:
        cut = len(keep) - max_sessions
        doomed.extend(keep[:cut])
        keep = keep[cut:]

    if max_bytes > 0:
        total = sum(candidate.size for candidate in keep)
        index = 0
        while total > max_bytes and index < len(keep):
            total -= keep[index].size
            doomed.append(keep[index])
            index += 1
        keep = keep[index:]

    evicted = 0
    freed = 0
    errors = 0
    for candidate in doomed:
        try:
            shutil.rmtree(candidate.path)
        except OSError as exc:
            # A directory another process is writing, or one on a read-only
            # mount. Skipping it keeps the sweep best-effort; refusing to
            # start the session over a failed cleanup would be the wrong
            # trade every time.
            logger.debug("session retention: cannot remove %s: %s", candidate.path, exc)
            errors += 1
            continue
        evicted += 1
        freed += candidate.size

    result = SweepResult(
        scanned=len(candidates),
        evicted=evicted,
        bytes_freed=freed,
        bytes_remaining=sum(candidate.size for candidate in keep),
        errors=errors,
    )
    if result.changed:
        logger.info(
            "session retention: evicted %d of %d session directories, freed %.1f KB",
            result.evicted,
            result.scanned,
            result.bytes_freed / 1024,
        )
    return result


def sweep_from_config(
    config_manager: object, config_dir: Path, live_dir: Path | None
) -> SweepResult:
    """Run :func:`sweep_sessions` with the ceilings from the app config.

    ``config_manager`` is duck-typed on ``get_config_value`` so this module
    stays importable without pulling the config graph onto the session import
    path; anything that cannot be read as an int falls back to the default
    rather than disabling the ceiling, because a typo'd config value must not
    silently restore the unbounded behaviour.
    """

    def _int(key: str, default: int) -> int:
        getter = getattr(config_manager, "get_config_value", None)
        if getter is None:
            return default
        try:
            return int(getter(key, default))
        except (TypeError, ValueError):
            logger.warning("session retention: %s is not an integer; using %d", key, default)
            return default

    return sweep_sessions(
        config_dir / SESSIONS_DIRNAME,
        live_dir=live_dir,
        max_sessions=_int(MAX_SESSIONS_KEY, DEFAULT_MAX_SESSIONS),
        max_bytes=_int(MAX_BYTES_KEY, DEFAULT_MAX_BYTES),
        max_age_days=_int(MAX_AGE_DAYS_KEY, DEFAULT_MAX_AGE_DAYS),
    )
