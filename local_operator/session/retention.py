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

The one hazard in "reap empty directories" is that every LIVE session is
empty for an instant too — between the ``mkdir`` that creates its directory
and the first append that fills it. On a machine running several sessions at
once (the normal way this harness is used) another session's startup sweep
can land in exactly that window and delete a directory whose owner is about
to write to it, which reproduces the original ``FileNotFoundError`` kill. So
a starting session CLAIMS its directory (:func:`claim_session`) with a
liveness marker before anything else creates it, and the sweep skips any
empty directory a live process still owns (:func:`_is_claimed`). The marker
is liveness, not content: a hard-killed run's leftover marker does not keep
its empty directory alive, so corpses are still reclaimed. As a second belt,
:class:`~local_operator.session.transcript.Transcript` recreates its
directory if it is deleted mid-session rather than dying on the next append.
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
#: to a reader listing the directory.
#:
#: This is a LIVENESS marker, not content: it says "a process is using this
#: directory right now", and it is the one thing that lets a startup sweep tell
#: an empty directory a concurrent session just created and is about to write
#: to from an empty directory a dead run left behind. It is therefore
#: deliberately NOT charged as bytes by :func:`_dir_size` — a session killed
#: hard (SIGKILL when its terminal closes) leaves this marker behind on an
#: otherwise-empty directory, and if the marker counted as content that
#: directory would be immortal. Excluding it means such a corpse reads as empty
#: and is reaped once :func:`_is_claimed` confirms no live process owns it.
LIVE_MARKER_NAME = ".session.pid"

#: Files a session directory may hold that are bookkeeping ABOUT the run rather
#: than content produced BY it, and so are not charged as bytes. Only the
#: liveness marker qualifies: ``origin.json`` is deliberately treated as
#: content (see :func:`_dir_size`) so an aborted child stamped with its origin
#: is protected exactly as #154/#192 intend — the point of the never-delete
#: model. Collected in one place so the size charge and the activity clock
#: agree on the list.
_SIDECAR_NAMES = frozenset({LIVE_MARKER_NAME})

#: This module's view of the platform. A module-local copy rather than reading
#: ``sys.platform`` at the point of use, so a test can steer the Windows branch
#: by patching THIS name instead of the global ``sys.platform`` — patching that
#: would tell every other thread in the process it is on Windows for the
#: duration, and this suite runs with threads.
_PLATFORM = sys.platform

#: Whether this platform can actually answer "is that process alive?". Named
#: rather than inlined because two decisions read it and must agree: liveness
#: itself, and whether a claim needs an age bound.
_LIVENESS_IS_VERIFIABLE = _PLATFORM != "win32"

#: How long a claim is trusted after the session's last WRITE, where liveness
#: cannot be probed (Windows, see :func:`_process_alive`). On POSIX the pid
#: probe is authoritative and this is never consulted, so a session of any
#: length keeps its protection there. Measured against activity rather than the
#: marker, this is not a cap on session length — a session writing every few
#: minutes is never stale — it is the answer to "nothing has touched this
#: directory for half a day and we cannot ask the OS whether its owner still
#: exists", where the alternative is trusting a leaked claim forever.
CLAIM_TRUST_S = 12 * 3600.0

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

    ``origin.json`` counts, deliberately. A directory holding only
    ``origin.json`` is a session that has been stamped — typically a child
    between ``mark_session_origin`` and its first append, or a run that
    aborted in that window. Treating the marker as invisible made those
    directories look empty and the sweep rmtree'd them, which is exactly the
    ``FileNotFoundError: .../transcript.jsonl`` kill this module exists to
    prevent. A 43-byte marker is cheaper than a lost session; abandoned
    markers accumulate and the user can remove them.

    The LIVENESS marker (:data:`LIVE_MARKER_NAME`) is the one file NOT charged,
    read from :data:`_SIDECAR_NAMES`. It is bookkeeping about the run, not
    content: a session killed hard leaves it on an otherwise-empty directory,
    and if it counted as content that corpse would be immortal. Excluded here,
    such a directory reads as empty and the reap in :func:`sweep_sessions`
    removes it once :func:`_is_claimed` confirms no live process owns it —
    while a directory with any real content stays untouchable regardless.
    """
    total = 0
    for entry in directory.rglob("*"):
        try:
            if entry.is_file() and entry.name not in _SIDECAR_NAMES:
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _process_alive(pid: int) -> bool:
    """Is ``pid`` a process this user can still see?

    ``os.kill(pid, 0)`` is the portable liveness probe: it delivers no signal
    and raises ``ProcessLookupError`` when nothing owns the id. ``PermissionError``
    means the id IS taken (by another user's process), so it counts as alive —
    refusing to reap is always the safe side of this decision.

    Windows has no signal 0 — ``os.kill`` there TERMINATES the target — so the
    probe is unavailable and this returns ``True``. That answer alone would be
    a disaster: every crash would leak a claim nothing can disprove, and since
    a claimed directory is skipped, the module would silently stop reaping one
    directory at a time. :func:`_is_claimed` therefore bounds a claim by its
    own age wherever liveness cannot be established (see ``CLAIM_TRUST_S``);
    this function answers only the liveness question.

    Pid REUSE is the residual risk on every platform: a leaked marker whose id
    has since been recycled reads as alive. It costs one retained EMPTY
    directory until that unrelated process exits, never a deleted live one,
    which is the direction this whole module errs in.
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
    """Mark ``session_dir`` as owned by a running process, BEFORE anything else
    creates it.

    Called once when a session is built, so that every OTHER session's startup
    sweep can tell "an empty directory a dead run left behind" from "the empty
    directory a live run just created and is one syscall from writing to". The
    never-delete model already protects any directory with content; this closes
    the residual window that content cannot — the instant between ``mkdir`` and
    the first append, during which the directory is genuinely empty and a
    concurrent sweep would reap it out from under its live owner (the exact
    ``FileNotFoundError`` this module exists to prevent, reachable again once
    the sweep reaps empty directories at all).

    ``claim_session`` creates the directory itself and writes the marker in one
    step, so a caller that claims FIRST leaves no unclaimed-empty window for the
    sweep to catch — this is why call sites claim ahead of their own ``mkdir``.

    Best-effort by design: a claim that cannot be written must not stop a
    session from starting, and the worst consequence of a missing claim is the
    pre-existing behaviour — the sweep treats the empty directory as reapable.

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
    a dead pid is ignored anyway — but releasing it promptly means an EMPTY
    session directory (a run that wrote nothing) becomes reapable at the moment
    its run ends rather than when the operating system happens to reuse the
    process id. That matters most for a HOST running several sessions in one
    process, where the owning pid stays alive long after an individual session
    is gone.

    Gated on the directory being under ``sessions/``, for the same reason
    :func:`claim_session` is: an agent directory must never receive this file,
    because ``AgentRegistry.export_agent`` zips every file it finds there and
    publishes the result. The gate lives INSIDE both functions rather than at
    the call sites so the two sides cannot drift apart.
    """
    if not _is_session_store_dir(session_dir):
        return
    # A leased session releases its compatibility mirror through the lease's
    # compare-token hook. Unconditional unlink here could erase a successor's
    # mirror after an in-process generation handoff.
    if (session_dir / ".execution-lease").exists():
        return
    try:
        (session_dir / LIVE_MARKER_NAME).unlink()
    except OSError:
        pass


def _is_claimed(directory: Path, now: float) -> bool:
    """Does a live process still own ``directory``?

    Consulted only for a directory that reads as EMPTY (see
    :func:`sweep_sessions`): a directory with content is never reaped whatever
    this returns, so the question only matters for the empty startup window and
    for a hard-killed run's leftover marker.

    A marker holding an unparseable value is treated as NOT claimed: it is
    corrupt bookkeeping, and honouring it forever would make the directory
    immortal.

    Where :func:`_process_alive` can actually probe, its answer is
    authoritative: a session stays protected for as long as it runs, however
    long that is. Where it cannot (Windows), the claim is bounded by
    ``CLAIM_TRUST_S``, measured against the LATER of two clocks — the
    directory's last write and the marker itself — so a fresh claim on an old
    resume is not read as abandoned, and a long active session is not expired
    mid-conversation.
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
    """When this session was last WRITTEN to, ignoring the liveness marker.

    Used only on the unverifiable-platform branch of :func:`_is_claimed`, to
    bound how long a claim is trusted after real activity. The liveness marker
    is excluded (via :data:`_SIDECAR_NAMES`) for the same reason it is not
    charged as bytes: it is written once at claim time and would otherwise peg
    "last activity" to the moment the session started, expiring a long but
    quiet session's claim while it is still alive.

    A directory with no non-marker content falls back to ``fallback`` (the
    marker's own mtime), which is the best signal available there.
    """
    newest: float | None = None
    try:
        for entry in directory.rglob("*"):
            try:
                if entry.name in _SIDECAR_NAMES or not entry.is_file():
                    continue
                stamp = entry.stat().st_mtime
            except OSError:
                continue
            newest = stamp if newest is None else max(newest, stamp)
    except OSError:
        return fallback
    return fallback if newest is None else newest


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
    directory that holds content. A transcript is removed only when the
    user explicitly disposes of the session — there is no automated path,
    because the failure mode of an automated path (a running session's
    transcript vanishing underneath it) costs the user the whole session,
    and the benefit (bounded disk) is recoverable by hand at any time.

    Two belts protect the empty directory a session passes through at
    startup, before it has written its first turn:

    - ``live_dir`` — the sweeping session's OWN directory, skipped even
      when empty, because it just created it;
    - the CLAIM marker — any OTHER session's directory that a live process
      still owns (:func:`_is_claimed`), so a concurrent startup does not
      reap the empty directory of a session that is one syscall from
      writing to it. A dead run's leftover marker does NOT protect its
      empty directory, so hard-killed corpses are still reclaimed.

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
    moment = now if now is not None else time.time()
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
            # Any real content (the liveness marker aside) — never touched.
            bytes_remaining += size
            continue
        # Reads as empty: no content, at most a leftover liveness marker.
        #
        # #152 (claim marker) and #235 (grace window) both guard the empty
        # startup window, but they are NOT interchangeable belts to stack
        # blindly: naively skipping on "claimed OR fresh" reopens the exact
        # immortal-empty leak #152 was built to prevent, because a hard-killed
        # run's corpse directory is BOTH freshly created AND carries a dead
        # marker, and the grace window would shield it forever-ish. They are
        # complementary because they answer DIFFERENT questions about
        # DIFFERENT populations, so the marker's presence selects which one is
        # authoritative:
        #
        #   - A directory WITH a marker was claimed by a session that reached
        #     ``claim_session``. Its liveness is knowable — :func:`_is_claimed`
        #     probes the pid (or, where it can't, bounds the claim by age). A
        #     live owner keeps it at ANY age; a dead owner's marker does NOT,
        #     and — crucially — a dead claim is reaped WITHOUT consulting the
        #     grace window, so a just-crashed session's corpse still reaps
        #     immediately instead of lingering an hour. This is what keeps the
        #     never-leak invariant intact once the grace window exists.
        #
        #   - A directory WITHOUT a marker is the case #235 was written for: a
        #     sibling whose claim write lost its race or was never attempted
        #     (``claim_session`` is best-effort — a claim that cannot be
        #     written must never stop a session from starting). There is no
        #     liveness signal at all, so the CREATION-time grace window is the
        #     only thing separating "a session one syscall from its first
        #     append" from "abandoned". Fresh survives; aged-out reaps.
        #
        # Removing either mechanism reopens a real hole: without the claim, a
        # long-running empty-for-now session past the grace window dies; without
        # the grace window, an unclaimed just-created sibling dies. So both stay.
        marker = child / LIVE_MARKER_NAME
        if marker.exists():
            # Claimed at some point: the marker's liveness is authoritative and
            # the grace window is deliberately bypassed so a dead claim reaps at
            # once (no immortal-empty corpse).
            try:
                if _is_claimed(child, moment):
                    continue
            except OSError:
                # Cannot read the marker to decide: err toward keeping it. A
                # spurious retained empty directory costs nothing; a wrongly
                # deleted live one costs the session.
                continue
        else:
            # Unclaimed: fall back to #235's creation-time grace window.
            try:
                # A directory's st_mtime is bumped whenever an entry is added
                # to or removed from it, so for a directory that is still empty
                # this is its CREATION time — exactly the "how long has this
                # been waiting for its first turn" signal the grace window
                # needs. The one edge case: creating and then deleting a
                # transient file inside an otherwise-empty dir resets this
                # clock, extending the grace window rather than shortening it,
                # which stays on the safe side (an unreaped empty dir costs
                # nothing).
                age = moment - child.stat().st_mtime
            except OSError:
                # Removed by another process between the size walk and the stat.
                continue
            if age < EMPTY_DIR_GRACE_SECONDS:
                # Empty, unclaimed, but fresh: indistinguishable from a
                # sibling's just-created session before its first turn lands
                # and before (or without) a claim. Age is the only signal left
                # that separates "abandoned" from "about to be written".
                continue
        # Reapable: either a dead/expired claim, or unclaimed and past the
        # grace window. Either way no live owner and no content — deleting it
        # loses nothing, and it was never a real session.
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
