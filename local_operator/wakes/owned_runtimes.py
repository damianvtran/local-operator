"""End the session runtimes a wake supervisor spawned in this store.

WHY THIS EXISTS (#3028, real-systemd QA Q-S1). ``KillMode=process`` stops only
the supervisor, deliberately: systemd's cgroup SIGTERM also kills a running
turn's tool child, so a routine supervisor restart cuts healthy work. But a
supervisor that exits without signalling its runtimes leaves an approval-parked
one alive for its day-long gate. The unit's signal must therefore be FORWARDED
to owned runtimes, which then drain their own turns (and deny parked gates)
rather than being cut.

THE MARKER IS A FACT OF THE SPAWN, NOT OF THE STORE. Each supervisor
invocation mints a token (:data:`OWNER_ENV`) and records its own pid
(:data:`SUPERVISOR_PID_ENV`). ``launch._spawn_runtime`` stamps the token ONLY
when the process building the child environment IS that supervisor (pid
equality), and strips every marker variable otherwise. A runtime also drops
the inherited marker from its own environment at boot
(:func:`forget_inherited_marker`). Together these keep the invariant the sweep
relies on — a marker means "a wake supervisor forked this runtime directly" —
so a session a wake turn messaged or spawned, or one started from a wake
runtime's shell, is never swept (#3028 review r4, MAJOR-2).

WHAT A STOP REACHES. Both production callers (the supervisor's stop handler and
uninstall) sweep with ``any_owner=True``: every runtime in THIS config root that
carries a marker, whichever invocation minted it. A runtime outlives the
invocation that spawned it, so the long-lived parked one usually holds an
EARLIER token (review r3, MAJOR-1). A process without a marker — the `lop
serve` daemon's runtimes, interactive and mesh-created sessions — is never
signalled, and another root's runtimes are out of reach, so a scratch/rig
supervisor cannot touch the live install's sessions. ``owner=`` (one exact
token) remains for inventory and tests. The token is not a secret and grants
no runtime authority; it only narrows a local SIGTERM sweep.

Every candidate is re-identified immediately before ``os.kill`` by exact pid:
argv must still be a session runtime, age must not have gone backwards (pid
recycling), and its EXEC-TIME environment (``/proc/<pid>/environ``, ``ps -E``)
must still carry a marker, this config root and a session. Never match
processes by name. Unknown or changed identity means no signal. SIGKILL is
deliberately absent.

EVERY SIGNAL CARRIES A STAGED SANCTION (2026-10-30, slice 1 of
``docs/design-surface-independence.md``): each candidate's involuntary stop marker
is written BEFORE its SIGTERM, because a runtime now REFUSES a terminating signal
that nothing sanctioned (``process._route_signal``). This was the only in-tree
sender of a terminating signal to a runtime that did not stage one, so leaving it
bare would have made this unit's stop silently stop working.
"""

from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Set by ``launch._spawn_runtime`` for the current wake supervisor. It is an
#: environment fact because adoption may replace the forked candidate's pid;
#: recording only that pid would miss the process that actually took the lease.
OWNER_ENV = "LOP_WAKE_SUPERVISOR_OWNER"

#: The supervisor's own pid, set in ITS environment beside the token. The
#: stamp is applied only where ``os.getpid()`` equals it: every descendant
#: inherits the variable but none shares the pid, which is what makes "only
#: the supervisor itself stamps" hold without trusting a child to scrub.
SUPERVISOR_PID_ENV = "LOP_WAKE_SUPERVISOR_PID"

#: The pre-#3028-r4 stamp gate ("1" meant "stamp"). Nothing reads it now; it is
#: still stripped so a runtime started under an older supervisor cannot pass it
#: to a newer launch path.
LEGACY_ACTIVE_ENV = "LOP_WAKE_SUPERVISOR_ACTIVE"

#: Every variable the marker scheme uses: stripped from a child environment
#: unless the supervisor itself is stamping, and from a runtime's own
#: environment at boot.
MARKER_ENVS: tuple[str, ...] = (OWNER_ENV, SUPERVISOR_PID_ENV, LEGACY_ACTIVE_ENV)


def stamp_child_env(env: dict[str, str]) -> None:
    """Stamp ``env`` (a child's environment, edited in place) or strip the marker.

    The token is copied ONLY when THIS process is the supervisor that minted
    it. Any other process — a wake runtime starting a session it messaged, a
    `lop` run from its shell — gets every marker variable removed, so the
    child it starts is not swept as the supervisor's (review r4, MAJOR-2).
    The pid itself is never passed on.
    """
    owner = os.environ.get(OWNER_ENV) or ""
    supervisor_pid = os.environ.get(SUPERVISOR_PID_ENV) or ""
    for name in MARKER_ENVS:
        env.pop(name, None)
    if owner and supervisor_pid == str(os.getpid()):
        env[OWNER_ENV] = owner


def forget_inherited_marker() -> None:
    """Drop the marker from THIS runtime's environment, so no child inherits it.

    Called once at runtime boot. Spawn paths other than ``launch`` (the
    `lop serve` daemon a tool might start, any subprocess) copy
    ``os.environ`` wholesale, and a marked runtime's tool children would carry
    its token into runtimes no supervisor started. The sweep is unaffected: it
    reads the EXEC-TIME environment (``/proc/<pid>/environ`` on Linux,
    ``ps -E`` on macOS), which unsetting a variable in a running process does
    not rewrite.
    """
    for name in MARKER_ENVS:
        os.environ.pop(name, None)


def owner_of(env_text: str) -> str:
    """The spawn token in a process environment, or empty."""
    prefix = OWNER_ENV + "="
    for word in env_text.split():
        if word.startswith(prefix):
            return word[len(prefix) :]
    return ""


def owned_pids(config_dir: Path, *, owner: str) -> list[tuple[int, str, str]]:
    """(pid, session-id, environment-text) candidates, for inventory and tests.

    Signals nothing; ``sweep`` re-identifies every pid at signal time.
    """
    from local_operator.session.runtime import reclaim

    found: list[tuple[int, str, str]] = []
    if not owner:
        return found
    for observed_row in reclaim.runtime_processes():
        row = reclaim.process_row(observed_row.pid)
        if row is None:
            continue
        env = row.command
        if owner_of(env) != owner or reclaim.config_root_of(env) != str(config_dir):
            continue
        session_id = reclaim.session_id_of(env)
        if session_id:
            found.append((row.pid, session_id, env))
    return found


def sweep(
    config_dir: Path,
    *,
    owner: str = "",
    any_owner: bool = False,
    why: str = "supervisor stop",
) -> list[int]:
    """SIGTERM this store's wake-marked runtimes; return the pids actually signalled.

    ``any_owner=True`` is what production uses — the supervisor's stop and
    uninstall alike: every runtime in THIS config root that carries a marker,
    whichever invocation minted it (see the module docstring for why that is
    the right reach). ``owner`` restricts the sweep to one exact token, for
    inventory and tests. Both are scoped to ``config_dir``, so a rig root's
    sweep can never reach the live install's runtimes (the parent lane's
    binding condition 2), and neither touches a process without a marker.

    IT STAGES A SANCTION PER CANDIDATE, BEFORE SIGNALLING IT, and that is now a
    correctness requirement rather than good manners: the runtime refuses a
    terminating signal nothing sanctioned (``process._route_signal``), so a sweep
    that kept signalling bare would silently leave every runtime behind — the
    exact opposite of what this unit's stop is for. This was the one in-tree
    sender of a terminating signal to a runtime that did not stage a marker; the
    ladder (``control.stop_session``), ``reclaim`` and the update paths all
    already do (``note_involuntary_stop``). The marker names this act and its
    actor, so a victim's death reads ``runtime-killed`` by the wake supervisor
    rather than ``unattributed``.

    Best-effort and non-raising: a failed sweep must not turn a supervisor's
    exit into a crash, and each runtime remains stoppable through ``lop stop``.
    The identity checks are deliberately repeated at signal time, not just when
    the candidate list is built.
    """
    from local_operator.session.runtime import reclaim

    signalled: list[int] = []
    if not owner and not any_owner:
        return signalled
    # THE RUN KEY COMES FROM THE PUBLISHED RECORD, never from the census age, and the
    # difference is not cosmetic: coverage is strict to ``_RUN_KEY_TOLERANCE_S`` (1.0
    # s) while ``ps``'s ``etime`` the census holds is 2-s-rounded, so a marker built
    # from ``age_s`` can MISS coverage — and then the runtime refuses this SIGTERM and
    # the supervisor's stop silently becomes a no-op, which is #3028 exactly (an
    # approval-parked runtime outliving its supervisor for the day-long gate).
    fleet = reclaim.read_fleet(config_dir)
    for observed_row in reclaim.runtime_processes():
        pid = observed_row.pid
        # OWNERSHIP IS READ FROM THE PER-PID PROBE, not the census row, and that
        # is not a preference: the macOS census (``ps -eo ... command=``) does
        # not carry the environment at all, so a census-only read finds no
        # marker and the sweep silently signals nothing (measured on a real
        # store: the supervisor exited 0 and its runtime outlived it). The
        # per-pid probe appends the environment on both platforms.
        row = reclaim.process_row(pid)
        if row is None:
            continue
        observed_owner = owner_of(row.command)
        if not observed_owner:
            continue
        if owner and observed_owner != owner:
            continue
        if reclaim.config_root_of(row.command) != str(config_dir):
            continue
        session_id = reclaim.session_id_of(row.command)
        if not session_id:
            continue
        # A pid reused after the census has a YOUNGER process age. Keep the
        # existing two-second rounding slack; never signal a different process.
        if observed_row.age_s and row.age_s < observed_row.age_s - 2.0:
            continue
        # THE MARKER GOES BEFORE THE SIGNAL, always: the victim cannot record
        # anything once the signal lands, and a rung that stages but is then
        # refused must take its marker back (see ``_withdraw_attestation``).
        published = fleet.records.get(pid) or fleet.boots.get(pid)
        started_at = getattr(published, "started_at", None)
        if not isinstance(started_at, (int, float)):
            # NO RUN KEY, SO NO SANCTION, SO NO STOP: the victim would refuse a
            # signal nothing covers, and reporting it as signalled would be a lie a
            # supervisor stop acts on. Say it instead — this is the one case where
            # this unit cannot do its job, and a record that vanished under a live
            # runtime is worth a line in the log.
            logger.warning(
                "wake supervisor: %s; NOT signalling owned runtime %s (session %s): no "
                "published record, so there is no run key to stage a sanction for",
                why,
                pid,
                session_id,
            )
            continue
        record = _SweepRecord(
            session_id=session_id,
            pid=pid,
            started_at=float(started_at),
        )
        attested = _attest_before_signal(record, config_dir, why)
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # THE SIGNAL NEVER LANDED, so the act this marker speaks for was not
            # taken. Leaving it would narrate an unrelated later death as this
            # sweep's, which is the one misreading the marker must not create.
            if attested:
                _withdraw_attestation(record, config_dir)
            continue
        logger.info(
            "wake supervisor: %s; SIGTERM to owned runtime %s (session %s)",
            why,
            pid,
            session_id,
        )
        signalled.append(pid)
    return signalled


@dataclass(frozen=True)
class _SweepRecord:
    """The run key one sweep marker is staged with — all the sweep knows.

    ``control.note_involuntary_stop`` duck-types the record it is handed (it reads
    the session, the pid, the start time and, when present, the build), so this
    carries exactly those facts and invents nothing. ``started_at`` is the
    kernel's own age arithmetic (``ps``'s ``etime``, held on
    :class:`reclaim.RuntimeProcess`) subtracted from now, the same derivation
    ``reclaim._ReclaimAttestation`` uses for the same reason.

    Built ONCE per candidate and reused by the withdrawal, because the marker's
    run key must round-trip exactly: a second ``time.time()`` here would stage a
    marker the withdrawal cannot recognise as its own.
    """

    session_id: str
    pid: int
    started_at: float


#: The involuntary-stop mechanism token this sweep's signals are attested under,
#: rendered by ``incidents.INVOLUNTARY_MECHANISM_LABELS``.
SWEEP_MECHANISM = "wake-sweep"

#: The acting component named on the marker. ``killer`` already records the
#: process (pid + argv0), so this names the PASS.
SWEEP_ATTEST_ACTOR = "wake supervisor"


def _attest_before_signal(record: _SweepRecord, config_dir: Path, why: str) -> bool:
    """Stage the involuntary stop marker for one candidate before its SIGTERM.

    Returns whether a marker is on disk, which the caller needs for exactly one
    decision: a signal that then fails must take its marker back. A candidate
    that could NOT be attested is still signalled — the unit's stop outranks the
    paperwork, as it does for ``reclaim`` and ``update`` — but the gap is SAID,
    because an unattested death is the class this marker exists to end.

    Never raises: an escaped error would abort a loop that has already sent
    signals and lose its report.
    """
    from local_operator.session.runtime.control import note_involuntary_stop

    try:
        attested = note_involuntary_stop(
            record,
            config_dir,
            mechanism=SWEEP_MECHANISM,
            actor=SWEEP_ATTEST_ACTOR,
            command=why,
        )
    except Exception:  # noqa: BLE001 — the sweep outranks the paperwork
        logger.warning(
            "wake supervisor: attesting pid %d raised; its ending will read unattributed",
            record.pid,
            exc_info=True,
        )
        return False
    if not attested:
        logger.warning(
            "wake supervisor: could not attest pid %d (session %s, root %s) before "
            "signalling it — its ending will read unattributed",
            record.pid,
            record.session_id,
            config_dir,
        )
    return attested


def _withdraw_attestation(record: _SweepRecord, config_dir: Path) -> None:
    """Take back a marker whose signal never landed. Best-effort, never raises."""
    from local_operator.session.runtime.control import withdraw_involuntary_stop

    try:
        withdraw_involuntary_stop(record, config_dir, mechanism=SWEEP_MECHANISM)
    except Exception:  # noqa: BLE001 — a withdrawal must not fail the sweep's loop
        logger.warning(
            "wake supervisor: withdrawing the marker for pid %d raised",
            record.pid,
            exc_info=True,
        )


def spawn_token() -> str:
    """One token per supervisor invocation, not a stable installation id."""
    import secrets

    return secrets.token_hex(16)
