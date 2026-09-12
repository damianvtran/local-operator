"""The one answer to "is this pid still a process?" — including zombies.

WHY THIS IS ITS OWN MODULE
--------------------------
Three modules ask this question about a pid that some *other* process owns, and
each asks it to decide whether something a dead owner left behind may be taken
over:

- :mod:`local_operator.session_lease` — may this transcript's sole-writer claim
  be acquired, or is its holder still working?
- :mod:`local_operator.session.runtime.registry` — may this discovery record be
  reaped, or is its runtime still live?
- :mod:`local_operator.resume` — should an attach be offered, or refused
  because the session is already open in another process?

They must agree, and the cost of disagreement is not a stale row: discovery
reporting "gone" while the lease reports "held" is a session **no interface can
open and no mechanism can recover**. That is not hypothetical — see
:func:`is_zombie` for the incident that produced this module.

All three callers are on the resume/attach/startup path and each documents that
it must stay stdlib-only and import-light (``registry`` sits on the CLI startup
path; ``session_lease`` and ``resume`` must be consultable without the engine
or the mobile stack). A leaf module with no local imports is therefore the only
home that satisfies all of them at once, which is exactly why the probe was
written twice before and got the zombie case right in only one of the two.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def is_zombie(pid: int) -> bool:
    """Whether this pid is an exited-but-unreaped process.

    **A ZOMBIE IS NOT A LIVE PROCESS, AND ``kill(pid, 0)`` CANNOT TELL YOU
    THAT.** Signal 0 succeeds against a process that has exited but has not
    been reaped yet, so every probe built on it reports such a pid as alive
    for as long as its parent fails to reap it — and for a runtime spawned by
    a long-lived TUI, that parent may never reap it at all. The pid is not
    reused while it lingers, so the wrong answer is stable, not a race.

    Where the wrong answer costs a row: `registry` fixed this for discovery
    records in round 3 (U10) — a SIGKILLed runtime reported ``live`` with
    ``0B`` RSS in `lop sessions`. Where it costs the SESSION: the lease probe
    kept calling such an owner live, so nothing could take its claim over —
    ``acquire_session_lease`` refused every attempt and
    ``reap_proven_dead_session_claim`` declined to remove it, because both
    require the holder to be *proven dead*. An operator's session whose
    runtime was killed while its TUI parent lived on was therefore
    un-attachable from every interface (TUI ``/resume``, ``lop exec
    --resume``, phone attach) with the message "already open in pid N", where
    N was a corpse.

    Fails CLOSED (returns False, i.e. "treat as alive") on any doubt: calling
    a live process dead would let a second writer take a transcript a working
    runtime is still appending to, and a forked trajectory is far worse than
    the unrecovered claim this exists to remove.

    Deliberately NOT ``psutil``: this module is stdlib-only by contract and
    ``/proc`` does not exist on macOS, so on POSIX the fallback is a ``ps``
    fork — measured at 2.4-3.9 ms across runs on an M-series box, against
    ~1 µs for signal-0. Callers therefore spend it only where the answer changes
    what they do. Concretely, these are the places that may pay it:

    - ``session_lease._pid_state`` — every acquisition and reaper decision, and
      the legacy ``.session.pid``-only branch, all of which require a holder to
      be *proven* dead before they move its claim.
    - ``resume.live_runtime_pid`` — after signal 0 has already said "exists",
      because that answer decides whether an interface refuses to open a
      session at all.
    - ``registry.pid_alive(check_zombie=True)`` — ``registry.scan`` spends it on
      a record whose heartbeat has already gone quiet, so a healthy session's
      row stays fork-free.

    Two places deliberately do NOT, and both are latency trades rather than
    safety ones: ``registry.pid_alive``'s default, which keeps ``scan``
    fork-free for healthy records, and ``launch._lease_holder``'s dense-window
    call, where the engage loop polls every 10 ms and the fork would cost more
    than the wait it is trying to shorten. Neither can take a claim — only
    ``session_lease`` does that, and it always asks.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        # Windows has no zombie state at all: a terminated process is
        # immediately reusable, and liveness there is a *handle* question
        # (``OpenProcess``), not a process-table one. Returning False is the
        # same answer the POSIX probe would reach through a doomed
        # ``/bin/ps`` fork, without the fork.
        return False
    try:
        proc_status = Path(f"/proc/{pid}/stat")
        if proc_status.exists():  # Linux: no subprocess needed
            # `comm` can contain spaces and parentheses; state is the field
            # after the LAST ')'.
            data = proc_status.read_text()
            return data.rpartition(")")[2].strip().startswith("Z")
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["/bin/ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except Exception:  # noqa: BLE001 — an unprobeable pid is treated as alive
        return False
    return result.stdout.strip().upper().startswith("Z")
