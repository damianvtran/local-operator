"""End the session runtimes one wake-supervisor invocation actually owns.

WHY THIS EXISTS (#3028, real-systemd QA Q-S1). ``KillMode=process`` stops only
the supervisor, deliberately: systemd's cgroup SIGTERM also kills a running
turn's tool child, so a routine supervisor restart cuts healthy work. But a
supervisor that exits without signalling its runtimes leaves an approval-parked
one alive for its day-long gate. The unit's signal must therefore be FORWARDED
to owned runtimes, which then drain their own turns (and deny parked gates)
rather than being cut.

OWNERSHIP IS A FACT OF THE SPAWN, NOT OF THE STORE. ``launch`` gives every new
candidate a unique token for this supervisor invocation. It is scrubbed from
other spawn paths, so a runtime this process merely adopted or an unrelated
runtime in the same store is not claimed. This also lets a scratch/rig
supervisor stop without touching the live install's sessions. The token is
not a secret and grants no runtime authority; it only narrows a local SIGTERM
sweep.

Every candidate is re-identified immediately before ``os.kill`` by exact pid:
argv must still be a session runtime, age must not have gone backwards (pid
recycling), and its environment must still name the same owner token, config
root and session. Never match processes by name. Unknown or changed identity
means no signal. SIGKILL is deliberately absent.
"""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path

logger = logging.getLogger(__name__)

#: Set by ``launch._spawn_runtime`` for the current wake supervisor. It is an
#: environment fact because adoption may replace the forked candidate's pid;
#: recording only that pid would miss the process that actually took the lease.
OWNER_ENV = "LOP_WAKE_SUPERVISOR_OWNER"


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
    """SIGTERM this store's wake-owned runtimes; return the pids actually signalled.

    ``owner`` is the running invocation's own token — the precise case, used by
    the supervisor's own stop. ``any_owner`` sweeps every runtime in THIS store
    that any wake supervisor spawned, which is the uninstall case: the store is
    being removed and a dead supervisor cannot name its own token. Both are
    scoped to ``config_dir``, so a rig root's sweep can never reach the live
    install's runtimes (the parent lane's binding condition 2).

    Best-effort and non-raising: a failed sweep must not turn a supervisor's
    exit into a crash, and each runtime remains stoppable through ``lop stop``.
    The identity checks are deliberately repeated at signal time, not just when
    the candidate list is built.
    """
    from local_operator.session.runtime import reclaim

    signalled: list[int] = []
    if not owner and not any_owner:
        return signalled
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
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            continue
        logger.info(
            "wake supervisor: %s; SIGTERM to owned runtime %s (session %s)",
            why,
            pid,
            session_id,
        )
        signalled.append(pid)
    return signalled


def spawn_token() -> str:
    """One token per supervisor invocation, not a stable installation id."""
    import secrets

    return secrets.token_hex(16)
