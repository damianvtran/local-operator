"""Session directory liveness: who owns a session directory right now.

THIS MODULE NEVER DELETES ANYTHING. It used to. Three generations of
"safe" automatic cleanup lived here — age/count/byte ceilings, then an
empty-directory reaper, then an "unused session" backfill (#576) that
removed any directory whose transcript held no ``"role": "user"`` row —
and the last of them destroyed 225 of an operator's 244 named sessions
(296,617 model calls of history) in one night. Its opt-out setting was
written under a nested key and read under a flat one, so the toggle the
user was offered did nothing. The runtime exit path carried a fourth
deleter (``_remove_unwritten_session_dir``, #622) with the same production
evidence.

The invariant this module now states and the test suite enforces
(``tests/unit/session/test_no_session_deletion.py``): **no code path
outside** :mod:`local_operator.session.cleanup` **removes, renames or
replaces a directory under** ``sessions/``. Cleanup is one explicit policy,
OFF by default, that a user turns on in ``/settings`` or runs by hand with
``lop sessions cleanup``. Nothing here — no sweep, no startup hook, no
exit hook, no daemon — deletes a session on its own judgement, whatever
the directory holds, however old it is, however many siblings it has.

What survives here is the CLAIM: a starting session writes a liveness
marker (:func:`claim_session`) naming its pid, and drops it on dispose
(:func:`release_session`). The marker exists so that anything that scans
the store — the cleanup policy when the user enables it, the resume
picker, the mobile daemon — can tell "a live process owns this" from "a
dead run left this behind" without guessing. :func:`_is_claimed` answers
that question and :func:`_process_alive` is the shared liveness probe
(``tools/group_reaper.py`` imports it so the two never disagree).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Directory under the config dir holding ephemeral per-run transcripts.
SESSIONS_DIRNAME = "sessions"

#: Written into a session directory while its run is alive; holds the owning
#: process id. Named with a leading dot so it never looks like session content
#: to a reader listing the directory.
#:
#: This is a LIVENESS marker, not content: it says "a process is using this
#: directory right now". It is what lets anything scanning the store — the
#: user-enabled cleanup policy, the resume picker, the mobile daemon — tell a
#: directory a live session owns from one a dead run left behind, without
#: guessing from mtimes. A hard-killed run (SIGKILL when its terminal closes)
#: leaves the marker behind; :func:`_is_claimed` probes the pid rather than
#: trusting the file, so a leftover marker never protects anything.
LIVE_MARKER_NAME = ".session.pid"

#: The attachment sidecar's name (``resume.ATTACHMENT_SIDECAR_NAME``), spelled
#: here for the same import-weight reason :data:`TRANSCRIPT_FILENAME` is.
ATTACHMENT_SIDECAR_FILENAME = "attachment.json"

#: Files a session directory may hold that are bookkeeping ABOUT the run rather
#: than content produced BY it: the liveness marker, and the attachment sidecar
#: naming the ``/team``, ``/agent`` and ``/goal`` a resume re-stamps. Excluded
#: from the activity clock (:func:`_activity_mtime`) because writing either
#: says nothing about when the user last worked here. Exported so
#: :mod:`local_operator.session.cleanup` uses the same list when it decides
#: what "last activity" means — two lists would drift.
_SIDECAR_NAMES = frozenset({LIVE_MARKER_NAME, ATTACHMENT_SIDECAR_FILENAME})

#: The transcript file's name, exported for the cleanup policy's "has this
#: session got a transcript at all" question.
#:
#: Spelled here rather than imported from
#: ``local_operator.session.transcript``, matching ``resume.TRANSCRIPT_NAME``
#: which keeps its own copy for the same reason. This module is imported on the
#: session-construction path and is deliberately lean (115 modules, 13 ms);
#: importing ``transcript`` for one string literal pulls in the harness types
#: and attachment store with it (259 modules, 90 ms). The literal is stable — it
#: is the on-disk format's name — and a change to it would break far more than
#: this constant.
TRANSCRIPT_FILENAME = "transcript.jsonl"

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


def _process_alive(pid: int) -> bool:
    """Is ``pid`` a process this user can still see?

    ``os.kill(pid, 0)`` is the portable liveness probe: it delivers no signal
    and raises ``ProcessLookupError`` when nothing owns the id. ``PermissionError``
    means the id IS taken (by another user's process), so it counts as alive —
    "alive" is always the safe answer for a caller deciding whether a
    directory is in use.

    Windows has no signal 0 — ``os.kill`` there TERMINATES the target — so the
    probe is unavailable and this returns ``True``. That answer alone would
    make every crashed run's claim permanent, so :func:`_is_claimed` bounds a
    claim by its own age wherever liveness cannot be established (see
    ``CLAIM_TRUST_S``); this function answers only the liveness question.

    Pid REUSE is the residual risk on every platform: a leaked marker whose id
    has since been recycled reads as alive. That over-protects a directory
    until the unrelated process exits; it never under-protects one, which is
    the direction every caller of this probe errs in.
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
    """Is ``directory`` one of the session directories under ``sessions/``?

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

    Called once when a session is built, so that anything scanning the store
    can tell "a directory a dead run left behind" from "a directory a live run
    owns and is one syscall from writing to". Nothing deletes on the strength
    of the answer by default; when the user enables the cleanup policy
    (:mod:`local_operator.session.cleanup`), the claim is one of the guards
    that keep a live session out of its reach whatever else the policy says.

    ``claim_session`` creates the directory itself and writes the marker in one
    step, so a caller that claims FIRST is never observable as unclaimed — this
    is why call sites claim ahead of their own ``mkdir``.

    Best-effort by design: a claim that cannot be written must not stop a
    session from starting. A missing claim means the directory reads as
    unowned to a scanner, which costs nothing while cleanup is off.

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
    a dead pid is ignored anyway — but releasing it promptly means a scanner
    reads the directory as unowned the moment its run ends rather than when
    the operating system happens to reuse the process id. That matters most
    for a HOST running several sessions in one process, where the owning pid
    stays alive long after an individual session is gone.

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

    The one question the liveness marker answers. Consulted by the cleanup
    policy as a hard guard — a claimed directory is never a candidate, whatever
    the policy says — and by anything else that needs to know whether a
    directory is in use.

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

    Used on the unverifiable-platform branch of :func:`_is_claimed` to bound
    how long a claim is trusted after real activity, and by the cleanup policy
    as its definition of "last activity" for the inactivity limit. The liveness marker
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
