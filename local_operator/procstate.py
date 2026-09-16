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
from collections.abc import Iterable, Sequence
from pathlib import Path


def _is_zombie_state(state: str) -> bool:
    """Whether a ``ps``/``/proc`` state field names an exited process.

    ONE HOME for the spelling, because two probes read it now (the single-pid
    :func:`is_zombie` and the batched :func:`zombie_states`): ``ps`` prints a
    zombie as ``Z`` and may append flag characters (``Z+``), so the test is a
    prefix rather than an equality.
    """
    return state.strip().upper().startswith("Z")


def _proc_state(pid: int) -> str | None:
    """Linux's own answer, or ``None`` where ``/proc`` does not exist.

    ``comm`` can contain spaces and parentheses, so the state is the field after
    the LAST ``)``.
    """
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return data.rpartition(")")[2].strip()


def _ps_states(pids: Sequence[int]) -> dict[int, bool]:
    """ONE ``ps`` fork answering the zombie question for a whole pid SET.

    ``ps`` takes a pid LIST, which is what makes the batch cost one fork rather
    than one per pid — the difference between a probe that scales with the
    record population and one that does not. A pid ``ps`` does not report (it
    exited between the caller's signal-0 check and this call) is simply ABSENT
    from the result: existence is the caller's question and it has already asked
    it, so inventing a verdict here would answer a different one.

    An unprobeable set answers ``{}`` — the same "treat as alive" failure
    :func:`is_zombie` documents, and the reason this can be a dict rather than a
    raise.
    """
    if not pids:
        return {}
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["/bin/ps", "-o", "pid=,state=", "-p", ",".join(str(pid) for pid in pids)],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except Exception:  # noqa: BLE001 — an unprobeable set is treated as alive
        return {}
    states: dict[int, bool] = {}
    for line in result.stdout.splitlines():
        # ``-o pid=`` suppresses the header, so every line is "<pid> <state>".
        fields = line.split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        states[int(fields[0])] = _is_zombie_state(fields[1])
    return states


def zombie_states(pids: Iterable[int]) -> dict[int, bool]:
    """The zombie verdict for a whole pid SET, in ONE probe.

    WHY THIS EXISTS (review round 1, MINOR 3 / QA Q1). ``registry.classify``'s
    derived policy spends the ``ps`` fork on every record whose heartbeat has
    gone quiet, and the desktop feed re-runs the whole scan on a one-second
    clock — so a population of quiet-but-alive records (the wedged sessions this
    feature most cares about) cost one fork PER RECORD PER SECOND: measured at
    88 forks on each probe with 200 records, a probe of 1.7 s, and the feed's
    10 Hz doorbell collapsing to 0.5 Hz. The question is per pid, but the probe
    need not be: ask it once for the set.

    MEMOISATION WAS THE OTHER CANDIDATE AND WAS REJECTED, deliberately: any
    cache of the "not a zombie" answer delays the alive -> zombie transition by
    its staleness bound, which is exactly the transition the round-3 U10 fix
    exists to catch promptly (a ``kill -9``'d runtime must stop reading as live
    as soon as its beat is quiet). A batched probe keeps every answer as fresh
    as the unbatched one — it changes how many forks the answers cost, never how
    old they are — so the freshness guarantee is preserved rather than traded.

    Pids that cannot be probed are absent from the result; the caller decides
    what absence means (``registry.scan`` reads it as "not a zombie", the same
    fail-closed answer as a failed single probe).

    Windows answers ``{}`` rather than probing: a terminated process is
    immediately reusable there, and liveness is a *handle* question
    (``OpenProcess``), not a process-table one — the answer the POSIX probe
    would reach through a doomed ``/bin/ps`` fork, without the fork.
    """
    wanted = sorted({int(pid) for pid in pids if int(pid) > 0})
    if not wanted or os.name == "nt":
        return {}
    if os.path.isdir("/proc"):
        # Linux: no subprocess needed, and no fork for the whole set.
        scraped: dict[int, bool] = {}
        for pid in wanted:
            state = _proc_state(pid)
            if state is not None:
                scraped[pid] = _is_zombie_state(state)
        return scraped
    return _ps_states(wanted)


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
    fork — measured at 2.4-4.6 ms across runs on an M-series box, tracking
    host load, against ~1 µs for signal-0. Callers therefore spend it only where the answer changes
    what they do. Concretely, these are the places that may pay it:

    - ``session_lease._pid_state`` — every acquisition and reaper decision, and
      the legacy ``.session.pid``-only branch, all of which require a holder to
      be *proven* dead before they move its claim.
    - ``resume.live_runtime_pid`` — after signal 0 has already said "exists",
      because at its user-facing call sites (the TUI's ``/resume``,
      ``lop exec --resume``, the phone's attach) that answer decides whether
      someone is refused the session they asked for. Its ``check_zombie=False``
      mode exists for one caller, the engage loop's dense discovery pass, where
      the same answer can only cost a wait.
    - ``registry.pid_alive(check_zombie=True)`` — ``registry.scan`` spends it on
      a record whose heartbeat has already gone quiet, so a healthy session's
      row stays fork-free. That scan asks through :func:`zombie_states`, i.e.
      ONE fork for the whole quiet set rather than one per quiet record: the
      answer is identical and the caller is the one that has the population in
      hand.

    Two places deliberately do NOT, and both are latency trades rather than
    safety ones: ``registry.pid_alive``'s default, which keeps ``scan``
    fork-free for healthy records, and the engage loop's two probes inside its
    dense 10 ms grid (``find_runtime_record``'s owner lookup and
    ``_lease_holder``), where a fork costs more than the wait it would shorten.
    Neither can take a claim — only ``session_lease`` does that, and it always
    asks.

    The probe is :func:`zombie_states`, so the single-pid answer and the batch a
    whole scan asks for cannot drift apart: this is that function asked about
    one pid.
    """
    return zombie_states([pid]).get(pid, False)
