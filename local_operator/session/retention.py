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
oldest first, until every ceiling holds over the history it is allowed to
touch. Age alone would let a burst of activity blow the disk budget inside the
window; bytes alone would keep a single ancient directory forever.

**The ceilings bound DEAD history, not the store.** A live session is never a
candidate — evicting the transcript of a run that is currently appending to it
would take out resume and compaction replay together, which is a far worse
outcome than the disk it reclaims. So live sessions sit outside all three
ceilings, and the store's true size is the bounded dead history plus whatever
the live sessions are holding. That residue is unbounded in principle (N
concurrent sessions against append-only transcripts) and small in practice (a
heavy 60-turn day is ~80 KB), but it is real: it is why
:func:`sweep_sessions` logs when the store as a whole sits meaningfully over
the byte ceiling rather than letting the overshoot pass silently, and why
``SweepResult`` reports live bytes separately instead of folding them into a
figure that reads as "what is on disk".

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

#: How long a claim is trusted after the session's last WRITE, where liveness
#: cannot be probed (Windows, see :func:`_process_alive`). On POSIX the pid
#: probe is authoritative and this is never consulted, so a session of any
#: length keeps its protection there.
#:
#: Measured against activity rather than against the marker, this is not a cap
#: on session length — a session writing every few minutes is never stale, and
#: sessions here run for days. It is the answer to "nothing has touched this
#: directory for half a day and we cannot ask the OS whether its owner still
#: exists", where the alternative is trusting a leaked claim forever and
#: silently disabling all three ceilings, one directory per crash.
CLAIM_TRUST_S = 12 * 3600.0

#: What share of the byte ceiling live sessions must hold before the sweep
#: warns. See the warning's own comment in :func:`sweep_sessions` for why the
#: test is on the LIVE share rather than on the store's total size, and for
#: the measurements behind this number.
LIVE_BYTES_WARN_SHARE = 0.75

#: This module's view of the platform. A module-local copy rather than reading
#: ``sys.platform`` at the point of use, so a test can steer the Windows branch
#: by patching THIS name instead of the global ``sys.platform`` — patching that
#: would tell every other thread in the process it is running on Windows for
#: the duration, and this suite runs with threads.
_PLATFORM = sys.platform

#: Whether this platform can actually answer "is that process alive?".
#: Named rather than inlined because two different decisions read it and they
#: must agree: liveness itself, and whether a claim needs an age bound.
_LIVENESS_IS_VERIFIABLE = _PLATFORM != "win32"

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
    #: Bytes of EVICTABLE history left after the sweep — what the ceilings
    #: actually govern. Not the size of the store: live sessions are exempt
    #: and counted in ``bytes_live`` instead. Includes directories the sweep
    #: selected but could not delete, which are still occupying disk.
    bytes_remaining: int = 0
    #: Bytes held by sessions that were skipped because a live process still
    #: owns them. Reported separately so a caller measuring the footprint can
    #: add the two rather than silently under-reading the store by this much.
    bytes_live: int = 0
    errors: int = 0

    @property
    def changed(self) -> bool:
        return self.evicted > 0

    @property
    def bytes_on_disk(self) -> int:
        """Everything the sweep saw: bounded history plus live sessions.

        The figure to quote when the question is "how big is the store";
        ``bytes_remaining`` answers the narrower "how much of it is subject to
        the ceilings".

        Directories the sweep selected but failed to delete are counted in
        ``bytes_remaining``, so a sweep that could remove nothing reports the
        store's real size rather than an empty one.
        """
        return self.bytes_remaining + self.bytes_live


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

    Windows has no signal 0 — ``os.kill`` there TERMINATES the target — so the
    probe is unavailable and this returns ``True``. That answer alone would be
    a disaster: every crash would leak a claim nothing can disprove, and since
    a claimed directory is skipped entirely, the module would silently switch
    itself off one directory at a time. :func:`_is_claimed` therefore bounds a
    claim by its own age wherever liveness cannot be established (see
    ``CLAIM_TRUST_S``); this function answers only the liveness question.

    Pid REUSE is the residual risk on every platform: a leaked marker whose id
    has since been recycled reads as alive. It costs one retained directory
    until that unrelated process exits, never a deleted live one, which is the
    direction this whole module should err in.
    """
    if pid <= 0:
        return False
    if _PLATFORM == "win32":  # pragma: no cover - probe is POSIX-only
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


def _is_session_store_dir(directory: Path) -> bool:
    """Is ``directory`` one of the ephemeral session directories we sweep?

    The marker belongs only under ``sessions/``. With ``--train`` (or a named
    agent) a transcript lives in ``agents/<id>/``, which retention never
    scans, so a marker there protects nothing — and it does not merely waste a
    file: ``AgentRegistry.export_agent`` zips every file in an agent directory
    and that archive is published to the Agent Hub and copied into importing
    users' agent directories.
    """
    return directory.parent.name == SESSIONS_DIRNAME


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

    Confined to ``sessions/`` (see :func:`_is_session_store_dir`): a marker in
    an agent directory protects nothing and escapes into published agent
    bundles.
    """
    if not _is_session_store_dir(session_dir):
        return
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
    the operating system happens to reuse the process id. That matters most
    for a HOST running several sessions in one process, where the owning pid
    stays alive long after an individual session is gone.

    Gated on the directory being under ``sessions/``, for the same reason
    :func:`claim_session` is: an agent directory must never receive this file,
    because ``AgentRegistry.export_agent`` zips every file it finds there and
    publishes the result. The gate lives INSIDE both functions rather than at
    the call sites so the two sides cannot drift apart — they did once, when
    only the claim was gated and dispose still wrote the marker into
    ``agents/<id>/`` on every clean exit.
    """
    if not _is_session_store_dir(session_dir):
        return
    try:
        (session_dir / LIVE_MARKER_NAME).unlink()
    except OSError:
        pass


def _is_claimed(directory: Path, now: float) -> bool:
    """Does a live process still own ``directory``?

    A marker holding an unparseable value is treated as NOT claimed: it is
    corrupt bookkeeping, and honouring it forever would make the directory
    immortal and quietly disable the ceilings.

    Where :func:`_process_alive` can actually probe, its answer is
    authoritative: a session stays protected for as long as it runs, however
    long that is. Where it cannot (Windows), the claim is bounded by
    ``CLAIM_TRUST_S``, measured against the LATER of two clocks — the
    directory's last write and the marker itself. Both are needed, and each
    one alone deletes a live session's transcript:

    - the marker alone measures how long the session has been RUNNING, since
      nothing refreshes it, so a long session that is actively writing expires
      mid-conversation;
    - activity alone ignores that a live process just took ownership, so
      ``--resume`` of an older transcript — fresh claim, old content — reads as
      abandoned from the moment it is claimed until its first turn lands.

    Taking the max is the union of the two: the bound is reached only when
    nothing has written AND no process has claimed the directory within the
    window.

    The residual, deliberately accepted rather than designed away: a live but
    IDLE session on an unverifiable platform — open since Friday, untouched
    since — does satisfy both clocks and becomes evictable. The transcript
    recovery path in :class:`~local_operator.session.transcript.Transcript` is
    what covers that case, and the alternative (trusting a claim forever where
    it cannot be checked) reopens the leak this bound exists to close.
    """
    marker = directory / LIVE_MARKER_NAME
    try:
        raw = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    try:
        pid = int(raw)
    except ValueError:
        return False
    if not _process_alive(pid):
        return False
    if _LIVENESS_IS_VERIFIABLE:
        return True
    try:
        claimed_at = marker.stat().st_mtime
        stamp = max(_activity_mtime(directory, claimed_at), claimed_at)
    except OSError:
        return False
    return now - stamp < CLAIM_TRUST_S


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
    point: the marker is written once at startup and would otherwise peg a
    long-running session's "activity" to its birth time, and folding the
    directory's own mtime in would let claim/release churn refresh the age of
    real history and exempt it from the age ceiling.

    A directory with NO content falls back to its own mtime, and there that
    genuinely is the best signal available. Note the consequence, because it
    is deliberate rather than overlooked: claiming and releasing both touch the
    directory, so a session that started, wrote nothing and exited looks
    freshly created and keeps its ``NEW_SESSION_GRACE_S`` for one more sweep
    cycle than its age warrants. Making that exact would mean preserving the
    claim time across the release, which is a second timestamp to keep honest
    in return for reaping an empty directory a few minutes sooner — the
    tradeoff is not worth it, and the reap still collects it on the next pass.
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


def _candidates(sessions_dir: Path, live: Path | None, now: float) -> tuple[list[_Candidate], int]:
    """Evictable session directories, least-recently-active first.

    Two kinds of directory are never candidates, and both exclusions exist
    because deleting them breaks a RUNNING session rather than reclaiming dead
    history:

    - ``live``, the sweeping session's own directory;
    - any directory still claimed by a live process (:func:`_is_claimed`),
      which is every other concurrently running session.

    Returns the candidates AND the bytes held by those exempt directories, so
    the caller can report the store's real size rather than only the part it
    is allowed to govern.
    """
    live_resolved = live.resolve() if live is not None else None
    out: list[_Candidate] = []
    live_bytes = 0
    for child in sessions_dir.iterdir():
        try:
            if not child.is_dir():
                continue
            exempt = live_resolved is not None and child.resolve() == live_resolved
            if exempt or _is_claimed(child, now):
                live_bytes += _dir_size(child)
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
    return out, live_bytes


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
        candidates, live_bytes = _candidates(sessions_dir, live_dir, moment)
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
    # runs last because it is the ceiling that must hold over everything this
    # sweep is allowed to touch — live sessions are exempt (see the module
    # docstring) — and trimming by count first often satisfies it for free.
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
    stranded = 0
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
            # Selected for eviction and still on disk. Counted, because it is
            # in neither of the other two buckets: without this a sweep that
            # could delete nothing reported an empty store while the volume
            # filled up, which is the exact blind spot the byte accounting
            # exists to close.
            stranded += candidate.size
            continue
        evicted += 1
        freed += candidate.size

    result = SweepResult(
        scanned=len(candidates),
        evicted=evicted,
        bytes_freed=freed,
        bytes_remaining=sum(candidate.size for candidate in keep) + stranded,
        bytes_live=live_bytes,
        errors=errors,
    )
    if result.changed:
        logger.info(
            "session retention: evicted %d of %d session directories, freed %.1f KB",
            result.evicted,
            result.scanned,
            result.bytes_freed / 1024,
        )
    # Live sessions sit outside the ceilings, so the store can exceed the byte
    # budget with nothing evictable left to reclaim. That is the correct trade
    # (never delete a transcript in use) but it must not be SILENT: this is the
    # only signal that the configured bound is not currently being honoured,
    # and without it the failure mode is a full volume with a clean sweep log.
    #
    # Tested on the LIVE SHARE of the ceiling, which is the only figure here
    # that actually distinguishes the healthy state from the unbounded one.
    # Two conditions were tried and both are wrong:
    #
    # - ``live_bytes > max_bytes`` is too narrow: it stayed silent at 1.8x over
    #   budget, the case worth hearing about.
    # - ``bytes_on_disk > max_bytes`` is too wide BY CONSTRUCTION: the byte loop
    #   evicts until governed bytes fall just under the ceiling, so any live
    #   session pushes the total over it. Measured over randomised healthy
    #   stores, the total lands between 1.02x and 1.64x of the ceiling — so a
    #   strict test fires on about half of ordinary startups, and even a 1.25x
    #   margin still fired on 31 of 60. A warning that common is one nobody
    #   reads by the time a real one arrives.
    #
    # The live share separates them cleanly: across those same healthy stores
    # it never exceeded 0.63 of the ceiling, while the pathological case sits
    # at 0.90. That is the honest signal, because it is live sessions — the
    # bytes eviction may never touch — that decide whether the store can come
    # back under budget at all. A big number here means the budget is being
    # consumed by data the ceilings are not allowed to reclaim.
    if max_bytes > 0 and live_bytes > max_bytes * LIVE_BYTES_WARN_SHARE:
        logger.warning(
            "session retention: live sessions hold %.1f MB of the %.1f MB ceiling and are "
            "exempt from eviction (%.1f MB on disk in total); the ceiling cannot bring the "
            "store back under budget while they are running",
            live_bytes / 1024 / 1024,
            max_bytes / 1024 / 1024,
            result.bytes_on_disk / 1024 / 1024,
        )
    # The other way the ceiling can fail to hold, and the one with a cause the
    # operator can act on: the sweep SELECTED directories and could not delete
    # them (read-only mount, permissions, a directory being written). Reported
    # separately from the live-session case because it is a fault rather than a
    # trade, and at warning level because ``errors`` alone is a count nobody
    # reads — the bytes are what say the store is not shrinking.
    if stranded > 0:
        logger.warning(
            "session retention: could not delete %d session %s holding %.1f MB; the store "
            "cannot be brought under its ceiling until that is resolved",
            errors,
            "directory" if errors == 1 else "directories",
            stranded / 1024 / 1024,
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
