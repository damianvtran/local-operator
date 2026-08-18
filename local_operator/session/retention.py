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
single ancient directory forever. Whatever the ceilings say, a LIVE session
is never a candidate — evicting the transcript of a run that is currently
appending to it would take out resume and compaction replay together, which
is a far worse outcome than the disk it reclaims.

"Live" means *every* running session, not just the one doing the sweeping.
The sweep runs at startup, and the ``live_dir`` the starting session knows
about is its own; on a machine running several sessions at once (the normal
way this harness is used) every OTHER running session was an ordinary
candidate. Once the count ceiling was reached, each new session evicted the
directory of a session still writing to it, and that session's next turn died
on ``FileNotFoundError: .../sessions/<id>/transcript.jsonl`` — with no way
forward, because every subsequent turn hit the same missing file. Sessions
therefore CLAIM their directory (:func:`claim_session`) for as long as they
are running, and the sweep skips any directory whose claim names a process
that is still alive.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Directory under the config dir holding ephemeral per-run transcripts.
SESSIONS_DIRNAME = "sessions"

#: Written into a session directory while its run is alive; holds the owning
#: process id. Named with a leading dot so it never looks like session content
#: to a reader listing the directory, and excluded from the "is this session
#: empty" test below for the same reason.
LIVE_MARKER_NAME = ".session.pid"

#: How long a session directory is protected from the empty-directory reap
#: purely because it is new. A run creates its directory before it writes the
#: first transcript line, and on a busy machine another session's startup
#: sweep lands in that window: without a grace period it reaped a directory
#: whose owner was seconds away from writing to it. The claim marker covers
#: this too, so the grace period is only the belt to that pair of braces —
#: it also covers a session whose claim could not be written (read-only dir,
#: exotic platform) and one killed before it claimed.
NEW_SESSION_GRACE_S = 300.0

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

    The claim marker is excluded. It is bookkeeping this module wrote itself,
    and counting it would make a directory holding nothing but a claim look
    like it has content — the opposite of what the empty-directory reap needs
    to see once the claim is released.
    """
    total = 0
    for entry in directory.rglob("*"):
        try:
            if entry.is_file() and entry.name != LIVE_MARKER_NAME:
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _process_alive(pid: int) -> bool:
    """Is ``pid`` a process this user can still see?

    ``os.kill(pid, 0)`` is the portable liveness probe: it delivers no signal
    and raises ``ProcessLookupError`` when nothing owns the id. ``PermissionError``
    means the id IS taken (by another user's process), so it counts as alive —
    refusing to evict is always the safe side of this decision.

    Windows has no signal 0; ``os.kill`` there terminates the process instead,
    which would be catastrophic, so that platform reports "alive" and leans on
    the marker's age instead. Being conservative costs at most one stale
    directory per crashed run.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - probe is POSIX-only
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def claim_session(session_dir: Path, pid: int | None = None) -> None:
    """Mark ``session_dir`` as owned by a running process.

    Called once when a session is built, so that every OTHER session's startup
    sweep can tell "history nobody is using" from "the transcript a live run is
    still appending to". Without it the sweep only knew about its own
    ``live_dir`` and happily deleted a concurrent session's directory out from
    under it, breaking that session permanently (see the module docstring).

    Best-effort by design: a claim that cannot be written must not stop a
    session from starting, and the worst consequence of a missing claim is the
    pre-existing behaviour, now additionally guarded by ``NEW_SESSION_GRACE_S``.
    """
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / LIVE_MARKER_NAME).write_text(
            str(os.getpid() if pid is None else pid), encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("session retention: cannot claim %s: %s", session_dir, exc)


def release_session(session_dir: Path) -> None:
    """Drop the claim on ``session_dir``; the run that owned it has finished.

    Wired into session dispose. Not required for correctness — a claim naming
    a dead pid is ignored anyway — but releasing it promptly means a session
    directory becomes evictable at the moment its run ends rather than when
    the operating system happens to reuse the process id.
    """
    try:
        (session_dir / LIVE_MARKER_NAME).unlink()
    except OSError:
        pass


def _is_claimed(directory: Path) -> bool:
    """Does a live process still own ``directory``?

    A marker holding an unparseable value is treated as NOT claimed: it is
    corrupt bookkeeping, and honouring it forever would make the directory
    immortal and quietly disable the ceilings.
    """
    try:
        raw = (directory / LIVE_MARKER_NAME).read_text(encoding="utf-8").strip()
    except OSError:
        return False
    try:
        pid = int(raw)
    except ValueError:
        return False
    return _process_alive(pid)


def _activity_mtime(directory: Path, fallback: float) -> float:
    """When this session was last WRITTEN to, not when its directory was made.

    The directory's own mtime only moves when an entry is created or removed
    inside it, and a session creates ``transcript.jsonl`` once and then appends
    to it for hours. So a directory's mtime is effectively its BIRTH time, and
    ranking by it evicted oldest-*started* rather than least-recently-used:
    the long-running session an operator had been talking to all afternoon
    sorted ahead of a dozen one-shot runs that had opened and exited since.
    Taking the newest mtime among the directory's files ranks by real activity.

    The content's mtime REPLACES the directory's rather than being maxed with
    it, and the claim marker is ignored entirely. Both exclusions are the same
    point: writing or removing the marker moves the directory's mtime to now,
    so folding either one in would refresh the age of every session this module
    touches and silently exempt old history from the age ceiling.

    The directory's own mtime is the fallback for a directory with no content
    yet — there, birth time is the only signal there is, and it is the right
    one: that is a session still starting up.
    """
    newest: float | None = None
    try:
        for entry in directory.rglob("*"):
            try:
                if entry.name == LIVE_MARKER_NAME or not entry.is_file():
                    continue
                stamp = entry.stat().st_mtime
            except OSError:
                continue
            newest = stamp if newest is None else max(newest, stamp)
    except OSError:
        return fallback
    return fallback if newest is None else newest


def _candidates(sessions_dir: Path, live: Path | None) -> list[_Candidate]:
    """Evictable session directories, least-recently-active first.

    Two kinds of directory are never candidates, and both exclusions exist
    because deleting them breaks a RUNNING session rather than reclaiming dead
    history:

    - ``live``, the sweeping session's own directory;
    - any directory still claimed by a live process (:func:`_is_claimed`),
      which is every other concurrently running session.
    """
    live_resolved = live.resolve() if live is not None else None
    out: list[_Candidate] = []
    for child in sessions_dir.iterdir():
        try:
            if not child.is_dir():
                continue
            if live_resolved is not None and child.resolve() == live_resolved:
                continue
            if _is_claimed(child):
                continue
            stat = child.stat()
        except OSError:
            continue
        out.append(
            _Candidate(
                path=child,
                mtime=_activity_mtime(child, stat.st_mtime),
                size=_dir_size(child),
            )
        )
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

    Empty directories are reaped regardless of the ceilings once they are past
    ``NEW_SESSION_GRACE_S``. They are left behind by runs that built a session
    and exited before writing a turn, they carry nothing to lose, and on a real
    install 23 of 147 session directories were exactly this. The grace period
    is what keeps that reap off a session that has just created its directory
    and has not yet written its first line: reaping it there deleted the
    directory a starting run was about to write into, and that run then failed
    on every turn with a missing transcript.
    """
    if not sessions_dir.is_dir():
        return SweepResult()

    moment = now if now is not None else time.time()
    horizon = moment - max_age_days * 86400
    try:
        candidates = _candidates(sessions_dir, live_dir)
    except OSError as exc:
        logger.warning("session retention: cannot scan %s: %s", sessions_dir, exc)
        return SweepResult(errors=1)

    keep: list[_Candidate] = []
    doomed: list[_Candidate] = []
    for candidate in candidates:
        fresh = candidate.mtime > moment - NEW_SESSION_GRACE_S
        if candidate.size == 0:
            # A directory with no content is worth nothing EXCEPT in the window
            # where its owner is still starting up; there, it is the next
            # turn's transcript.
            (keep if fresh else doomed).append(candidate)
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
