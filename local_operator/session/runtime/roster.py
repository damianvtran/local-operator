"""The runtime roster: every live session runtime this machine has, reachable or not.

**WHAT IT ANSWERS.** "Is there a runtime for this session, where is it, and did it
answer?" — for EVERY runtime on the machine, in one bounded read. That question
had no answer before this module, and the gap was measured: on 2026-09-17, 34 of
57 live runtimes had no discovery record in any namespace, so every reader in the
product (``lop sessions``, ``lop send``, attach, the desktop app, the wake
supervisor) could not name them at all. Central reachability means a client must
be able to FIND and REACH any runtime without depending on a record, on one
particular port, or on any one process answering — which is why the roster is
composed from four independent sources and states, per row, which of them knew
about it.

**THE FOUR SOURCES, AND WHY EACH IS NEEDED.**

* ``run/mobile`` (via ``registry.scan``) — the discovery records: session id,
  control port, build, live state, heartbeat. The only source that can be
  ATTACHED to.
* ``run/host`` — the boot records: pid, session id, build, install root, parent.
  A boot-time snapshot rather than a liveness signal, so it is used for ATTRIBUTION
  (which session this pid was spawned for) and never for liveness.
* The process census (:mod:`reclaim`) — the process table, which is the only source
  that sees a runtime whose record and boot record are both gone. This is the one
  that makes the roster complete, and the row it produces is the one the operator's
  machine was full of.
* The machine's socket table — the control port of a record-less runtime (so it can
  be DIALED, not merely listed) and whether anything is connected to it right now.
  Read with one bounded ``lsof``, degrading to "unknown" rather than to "none".

Nothing that already works depends on this module. It is a reader: records and boot
records are read with ``reap=False``, the census and the socket table are read-only
external calls, and the only mutation any of it performs is the TCP connect of a
health probe.

**BOUNDED BY CONSTRUCTION.** The cost is O(live runtimes), one process-table fork
plus one batched zombie probe plus one socket-table call plus one bounded connect
per row that names a port. The whole composition runs on a worker thread (the
route hands it to ``asyncio.to_thread``), the probes run in a small fixed pool
with a short per-connect timeout, and the caller's budget is enforced as a
deadline: a row whose probe did not complete inside the budget is reported as
``unknown`` rather than waited for. A dead pid is never dialled at all —
``pid_alive`` (signal-0) answers first, and the row reads ``gone`` — so no wedged
or crashed runtime can make this endpoint hang.

**WHAT THE BUDGET DOES AND DOES NOT COVER.** ``budget_s`` bounds the PROBES — one
batch of loopback connects, abandoned when the deadline passes — and nothing else.
The response's real ceiling is ``budget_s + CENSUS_TIMEOUT_S +
SOCKET_TIMEOUT_S`` (2.5 s + 5 s + 3 s in the worst case, i.e. both external reads
wedged at their own timeouts, which is a machine whose process table and socket
table are both unresponsive). A small ``budget_s`` does not make the response
small: measured at ``budget_s=0.1`` the probe batch is abandoned at 0.1 s but the
``ps`` and ``lsof`` reads in front of it still cost what they cost (~0.14 s +
~0.26 s idle on the reference host, seconds when real load is present), and the
budget bounds only the part of this route that scales with the ROW COUNT — which
is the part that grows. Review round 1 (R1-4) measured the docstrings promising a
bound the code did not keep; this paragraph is the honest version.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.procstate import zombie_states
from local_operator.session.runtime import registry
from local_operator.session.runtime.reclaim import (
    REFUSAL_GONE,
    REFUSAL_NO_CENSUS,
    Fleet,
    RuntimeProcess,
    config_root_of,
    read_fleet,
    runtime_processes,
    session_id_of,
    socket_evidence,
    verdict,
)
from local_operator.session.runtime.types import RUN_DIRNAME, SessionRecord

#: The whole roster's wall budget. Beyond it, a row's probe is reported as
#: ``unknown``: the endpoint's contract is a bounded answer, so an unbounded wait
#: on one slow runtime is the failure mode, not a slow response. Sized well above
#: the measured cost (one ``ps`` ~140 ms + one ``lsof`` ~200 ms + N connects) and
#: well below any client's own request timeout.
ROSTER_BUDGET_S = 2.5

#: One connect's timeout. A loopback connect to a listening socket completes in
#: single-digit milliseconds, and to a process that is ALIVE but not answering
#: (wedged, stopped by the debugger, mid-GC) it does not complete at all — this is
#: the only place the roster can spend time, which is why it is short.
PROBE_TIMEOUT_S = 0.25

#: How many probes run at once. Fixed rather than derived from the row count: a
#: 40-row roster opening 40 sockets at once would dwarf the traffic of every real
#: client, and 8 probes at 0.25 s worst case keeps a fully-wedged fleet inside the
#: budget above.
PROBE_WORKERS = 8

#: How many OTHER stores' record directories one roster will read. A runtime of a
#: sibling root publishes its record into ITS root, so naming what such a runtime is
#: doing costs one scan per distinct foreign root; the reference host has 11 of them
#: (QA cells and sandboxes under /tmp). The cap keeps the composition O(live
#: runtimes) rather than O(stores ever seen) on a machine that has accumulated
#: hundreds of sandboxes, and the row for a runtime beyond the cap is still listed —
#: it simply reports no build, port or heartbeat, exactly as it did before this
#: lookup existed.
FOREIGN_ROOT_LIMIT = 32

#: Reachability verdicts. ``unknown`` exists so the roster can never report
#: absence-of-evidence as evidence-of-absence: a row with no port, a probe outside
#: the budget and a fleet whose process table could not be read are three different
#: situations that would all be a lie under ``unreachable``.
REACHABLE_LIVE = "live"
REACHABLE_UNREACHABLE = "unreachable"
REACHABLE_GONE = "gone"
REACHABLE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class RosterRow:
    """One runtime's row. Every field is measured, and each names its source.

    ``port`` is ``None`` when no source could supply one — a record publishes its
    control port, and a record-less runtime's port comes from the socket table,
    which is unavailable on a machine without ``lsof``. ``None`` is reported as
    such (and then ``reachability`` is ``unknown``) rather than as an empty port
    that a client would try to dial.
    """

    pid: int
    session_id: str
    port: int | None
    build_version: str
    install_root: str
    #: An interface is looking at this runtime right now: a live viewer lease names
    #: its session, or something holds an ESTABLISHED connection to its control
    #: port. Both are evidence of the present rather than a self-report.
    attached: bool
    #: How many live viewer leases name its session. ``0`` with ``attached`` True is
    #: a client on the control socket that is not a viewer (a `lop send`, an
    #: attach in progress).
    observers: int
    #: Seconds since the runtime last beat. ``None`` when no record exists — a
    #: record-less runtime does not beat anywhere readable, and reporting 0.0 for it
    #: would read as "fresh".
    heartbeat_age_s: float | None
    reachability: str
    #: The registry's verdict for a recorded runtime (``live``/``wedged``/``stale``),
    #: ``None`` for a record-less one. Derived ONCE per row, in
    #: :func:`~local_operator.session.runtime.registry.classify`, from the same
    #: pid-liveness answer ``reachability`` is derived from — a row that reported
    #: ``state: stale`` beside ``reachability: live`` was the defect review round 1
    #: (R1-3) measured, one surface disagreeing with itself about whether a process
    #: exists.
    state: str | None
    has_record: bool
    #: The config root whose record supplied this row's facts, or ``""`` when no
    #: record did. ``has_record`` answers the SWEEP's question (is there a record in
    #: the store being swept); ``record_root`` answers the ROSTER's (whose record is
    #: this row's data from). They differ for a runtime of a sibling store, which is
    #: the measured majority of the population that had no record in the operator's
    #: own store.
    record_root: str
    has_boot_record: bool
    busy: bool | None
    leaving: str
    #: The config root the process itself reports, when it could be read.
    config_root: str
    parent_pid: int
    age_s: float | None
    cpu_s: float | None
    #: Whether the residency sweep would end this runtime, and the token saying why
    #: not when it would not. The SAME verdict the sweep acts on, computed from the
    #: same function, so the roster can never describe a runtime differently from
    #: the way the sweep treats it. When there is no census row to judge there is no
    #: verdict either, and the token distinguishes the two reasons for that: the pid
    #: is ``gone``, or it is running and the process table simply did not match it
    #: (``no-census-row``).
    reclaimable: bool
    reclaim_refusal: str
    reclaim_detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "session_id": self.session_id,
            "port": self.port,
            "build_version": self.build_version,
            "install_root": self.install_root,
            "attached": self.attached,
            "observers": self.observers,
            "heartbeat_age_s": (
                None if self.heartbeat_age_s is None else round(self.heartbeat_age_s, 1)
            ),
            "reachability": self.reachability,
            "state": self.state,
            "has_record": self.has_record,
            "record_root": self.record_root,
            "has_boot_record": self.has_boot_record,
            "busy": self.busy,
            "leaving": self.leaving,
            "config_root": self.config_root,
            "parent_pid": self.parent_pid,
            "age_s": None if self.age_s is None else round(self.age_s, 1),
            "cpu_s": None if self.cpu_s is None else round(self.cpu_s, 1),
            "reclaimable": self.reclaimable,
            "reclaim_refusal": self.reclaim_refusal,
            "reclaim_detail": self.reclaim_detail,
        }


def _connect_ok(port: int, timeout_s: float) -> bool:
    """One bounded loopback connect. Any failure — refused, timed out, no route —
    is ``False``; the row it belongs to distinguishes "failed" from "not tried"."""
    if port <= 0:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(timeout_s)
            return probe.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def _probe_ports(
    ports: Mapping[int, int],
    *,
    budget_s: float,
    timeout_s: float = PROBE_TIMEOUT_S,
    workers: int = PROBE_WORKERS,
    connect: Callable[[int, float], bool] = _connect_ok,
    now: Callable[[], float] = time.monotonic,
) -> dict[int, bool]:
    """Probe many ports inside one wall budget. Missing pids mean "not answered".

    THE POOL IS SHUT DOWN WITHOUT WAITING, and that is the whole of the budget's
    enforcement. ``with ThreadPoolExecutor(...)`` calls ``shutdown(wait=True)``,
    which joins the QUEUED work items as well as the running ones — so a deadline
    that abandoned the waiting loop still blocked here for
    ``ceil(rows / PROBE_WORKERS) x timeout_s``, i.e. the budget was a hope on a
    roster bigger than the pool. Measured before the fix: ``budget_s=0.1``
    returned after 1.69 s at the endpoint and 2.04 s on a 64-row micro-repro
    (review round 1, R1-4 / QA Q4). ``cancel_futures`` drops the ones that never started, and the
    ones already inside the kernel are left to finish on their own thread — bounded
    by ``timeout_s``, off the caller's path, and the reason the futures are
    abandoned rather than joined.
    """
    if not ports:
        return {}
    deadline = now() + budget_s
    answered: dict[int, bool] = {}
    pool = ThreadPoolExecutor(max_workers=min(workers, len(ports)))
    try:
        futures = {pid: pool.submit(connect, port, timeout_s) for pid, port in ports.items()}
        for pid, future in futures.items():
            remaining = deadline - now()
            if remaining <= 0:
                # BUDGET SPENT, so the rest are UNKNOWN rather than false: a probe
                # that never ran must not be reported as a runtime that did not
                # answer.
                break
            try:
                answered[pid] = future.result(timeout=remaining)
            except Exception:  # noqa: BLE001 — a probe's failure is a verdict, never an error
                continue
    finally:
        # An already-running connect cannot be cancelled and does not need to be:
        # it holds a socket for at most ``timeout_s`` and no caller is waiting.
        pool.shutdown(wait=False, cancel_futures=True)
    return answered


def build_roster(
    root: Path | None = None,
    *,
    probe: bool = True,
    budget_s: float = ROSTER_BUDGET_S,
    fleet: Fleet | None = None,
    processes: Sequence[RuntimeProcess] | None = None,
    env_of: Callable[[int], str] | None = None,
    connect: Callable[[int, float], bool] | None = None,
    now: float | None = None,
) -> list[RosterRow]:
    """Compose the roster for one config root. Sorted by pid, cheapest source first.

    Every row comes from the union of the three pid-keyed sources, and a pid known
    to only one of them is a row the other two could not produce — which is the
    point of enumerating all three. Rows are sorted by pid so a client polling the
    roster sees a stable order it can diff, and so two runtimes of the same session
    (a successor and a leaving predecessor) read adjacently.
    """
    from local_operator.session.runtime.reclaim import process_envs

    moment = time.time() if now is None else now
    view = fleet if fleet is not None else read_fleet(root, sockets=socket_evidence())
    census = list(processes) if processes is not None else runtime_processes()
    # ONE fork for the whole fleet when the caller did not supply a reader. THE TRADE
    # IS WHICH POPULATION THE COST FOLLOWS: the per-pid form is a fork per row, so it
    # scales with the number of RUNTIMES (measured 12-50 ms each, i.e. ~2-4 s for the
    # 36 record-less rows this roster exists to name), while the batched form is one
    # ``ps`` over the whole process table, so it scales with the MACHINE (~0.8 s for
    # 1260 processes loaded, 0.2 s calm). Bounded by the session count is the property
    # that matters here — "resilient to the number of sessions running" is the whole
    # point of this surface — so the batch is the default and an injected ``env_of``
    # (the tests, or a caller with its own cache) is used as given.
    env_batch: dict[int, str] = {} if env_of is not None else process_envs()
    read_env = env_of if env_of is not None else (lambda pid: env_batch.get(pid, ""))
    # MEMOISED because two consumers need it for the same row — this function, for
    # the session id and root a record cannot supply, and ``verdict``, for the root
    # it attributes the candidate to.
    env_cache: dict[int, str] = {}

    def env_for(pid: int) -> str:
        if pid not in env_cache:
            env_cache[pid] = read_env(pid)
        return env_cache[pid]

    by_pid: dict[int, RuntimeProcess] = {process.pid: process for process in census}
    # A record's pid can be a corpse whose process is gone; it still gets a row, and
    # `reachability` says `gone`. Dropping it would hide exactly the evidence
    # (`stale` record) a reader is asking the roster for.
    pids = sorted(set(by_pid) | set(view.records) | set(view.boots))

    observers: dict[str, int] = {}
    # From the FLEET, which has already read this directory once. A second
    # ``scan_viewers`` here would be a second read of the same state inside one
    # composition — the two could disagree, and the row would then report a number no
    # caller could reproduce.
    for viewer in view.viewers:
        session = getattr(viewer, "current_session", "")
        if session:
            observers[session] = observers.get(session, 0) + 1

    ports: dict[int, int] = {}
    sessions: dict[int, str] = {}
    builds: dict[int, str] = {}
    roots: dict[int, str] = {}
    for pid in pids:
        record = view.records.get(pid)
        boot = view.boots.get(pid)
        if record is not None:
            port = int(getattr(record, "control_port", 0) or 0)
            if port > 0:
                ports[pid] = port
            sessions[pid] = str(getattr(record, "session_id", "") or "")
            builds[pid] = str(getattr(record, "version", "") or "")
            roots[pid] = str(view.root)
            continue
        # NO RECORD. A boot record still attributes the pid to a session and a
        # build — and the fact that it was found under THIS root is itself the
        # attribution of the runtime to this root, since a runtime publishes its
        # boot record into the root it runs under.
        port = view.sockets.ports.get(pid)
        if port:
            # The socket table is the only source of a port for a runtime that
            # published nothing: without it the roster could LIST such a runtime and
            # no client could DIAL it, which is the half of central reachability the
            # measurement said was missing.
            ports[pid] = port
        if isinstance(boot, dict):
            sessions[pid] = str(boot.get("session_id") or "")
            builds[pid] = str(boot.get("build_version") or "")
            roots[pid] = str(view.root)
        else:
            # NO RECORD AND NO BOOT RECORD — a runtime of a root that has been
            # deleted, or one whose instrumentation failed. Its own environment is
            # the last source for both facts, and the only way this roster can name
            # a runtime nothing else on the machine can see.
            env_text = env_for(pid)
            sessions[pid] = session_id_of(env_text)
            roots[pid] = config_root_of(env_text)

    # FOREIGN ROOTS: one bounded scan each, for the runtimes this store cannot
    # describe. See FOREIGN_ROOT_LIMIT for the cap and why it exists.
    foreign: dict[int, Any] = {}
    needed = sorted(
        {
            roots[pid]
            for pid in pids
            if pid not in view.records and roots.get(pid) and roots[pid] != str(view.root)
        }
    )[:FOREIGN_ROOT_LIMIT]
    for root_text in needed:
        # NOT ``is_dir`` BY ACCIDENT: ``registry.scan`` starts at ``registry.run_dir``,
        # which CREATES the directory it is about to read (documented for the swept
        # root, where it is harmless because the caller owns the store). On a SIBLING
        # root that is gone — or whose path is a file, or on a mount that has been
        # unmounted — that mkdir either conjures a directory inside someone else's
        # store or raises ``NotADirectoryError``, and this reader may do neither. So a
        # foreign root is scanned only when it is a directory RIGHT NOW, and any
        # failure below degrades to "no facts from there" rather than to an error on a
        # roster read.
        try:
            # ``Path``, not the string: ``registry.scan`` hands its ``root`` to
            # ``run_dir``, which joins it with ``/`` (a str root is a TypeError there).
            sibling = Path(root_text)
            if not sibling.is_dir():
                continue
            scanned = registry.scan(sibling, RUN_DIRNAME, SessionRecord.from_json, reap=False)
        except (OSError, ValueError):
            continue
        for record, _state in scanned:
            foreign.setdefault(record.pid, record)

    # ONE LIVENESS ANSWER PER ROW, AND ONE ZOMBIE PROBE FOR THE WHOLE ROSTER.
    #
    # Every row's ``state``, ``reachability`` and dial decision come from the answer
    # computed here, because two answers is the defect: review round 1 (R1-3)
    # measured a zombie pid DIALED and reported ``reachability=live`` on a row whose
    # own ``state`` said ``stale`` (the port it dialled belonged to whoever had taken
    # the pid), while a live, answering runtime whose pid the census could not match
    # read ``reclaim_refusal=gone`` — a surface reporting a machine that is not
    # there. There were three rules before (a bare signal-0 in this function,
    # ``registry.scan``'s derived policy behind ``state``, and ``REFUSAL_GONE`` for a
    # pid with no census row) and there is one now:
    #
    #   * ``registry.classify`` owns live/wedged/stale, and it is asked with the
    #     ZOMBIE ANSWER SUPPLIED (the ``zombie`` keyword exists for exactly this:
    #     the policy decides when to SPEND the probe, a caller holding the answer
    #     supplies it), so ``state`` is the registry's verdict under the true
    #     liveness; a record whose owner is a zombie is ``stale`` whatever its
    #     heartbeat says — which is what ``pid_alive(check_zombie=True)`` would
    #     answer too.
    #   * ``reachability`` follows from that same verdict (``stale`` is ``gone``),
    #     so it cannot contradict ``state``.
    #   * a row with NO record has no state to derive from and asks signal-0, plus
    #     the same batched zombie answer.
    #
    # ONE FORK: ``zombie_states`` is a single ``ps`` over a pid LIST (and no fork at
    # all on Linux, where it reads /proc). The list is FILTERED BY SIGNAL-0 first,
    # exactly as ``registry.scan``'s derived path filters its own batch before
    # spending the probe: a pid that is not there is ``stale`` whatever ``ps``
    # would have said, and a store full of dead records is the case that costs
    # nothing. The alternative — ``pid_alive(pid, check_zombie=True)`` per row — is
    # a ``ps`` fork per row, measured at 3.9 ms each, i.e. the ~150 ms this roster
    # cannot spend on the 36 record-less rows it exists to name.
    zombies = zombie_states([pid for pid in pids if registry.pid_alive(pid)])
    states: dict[int, str] = {}
    alive: dict[int, bool] = {}
    for pid in pids:
        # ``classify`` wants the RECORD, and either store's will do: the swept
        # store's first (its row is this roster's own), else the sibling's.
        here = view.records.get(pid)
        source = here if here is not None else foreign.get(pid)
        zombie = bool(zombies.get(pid, False))
        if source is not None:
            states[pid] = registry.classify(source, now=moment, zombie=zombie).state
            alive[pid] = states[pid] != "stale"
        else:
            alive[pid] = registry.pid_alive(pid) and not zombie

    if probe:
        alive_ports: dict[int, int] = {
            pid: port for pid, port in ports.items() if alive.get(pid, False)
        }
        # Resolved through the module attribute rather than as a default bound at
        # definition time, so a caller (or a test) that replaces the connect sees its
        # replacement used.
        answers = _probe_ports(alive_ports, budget_s=budget_s, connect=connect or _connect_ok)
    else:
        answers = {}

    rows: list[RosterRow] = []
    for pid in pids:
        process = by_pid.get(pid)
        record = view.records.get(pid)
        boot = view.boots.get(pid)
        session_id = sessions.get(pid, "")
        port = ports.get(pid)
        # The facts of a SIBLING store's runtime come from ITS record: this store
        # cannot describe it, and a row that lists the pid and says nothing else is
        # the state the roster exists to end.
        elsewhere = foreign.get(pid)
        if record is None and elsewhere is not None:
            if port is None:
                sibling_port = int(getattr(elsewhere, "control_port", 0) or 0)
                port = sibling_port or None
            session_id = session_id or str(getattr(elsewhere, "session_id", "") or "")
            builds.setdefault(pid, str(getattr(elsewhere, "version", "") or ""))
        # THE ROW'S LIVENESS IS DERIVED ONCE, above: ``reachability`` follows the
        # same answer ``state`` was classified with, so a row can no longer report a
        # connected runtime beside a stale one, and a zombie is never dialled.
        alive_now = alive.get(pid, False)
        if not alive_now:
            reachability = REACHABLE_GONE
        elif port is None:
            reachability = REACHABLE_UNKNOWN
        elif pid in answers:
            reachability = REACHABLE_LIVE if answers[pid] else REACHABLE_UNREACHABLE
        else:
            reachability = REACHABLE_UNKNOWN
        heartbeat = None
        state = states.get(pid)
        busy = None
        leaving = ""
        has_record = record is not None
        if record is not None or elsewhere is not None:
            source = record if record is not None else elsewhere
            heartbeat = max(0.0, moment - float(getattr(source, "heartbeat_at", 0.0) or 0.0))
            busy = bool(getattr(source, "busy", False))
            leaving = str(getattr(source, "leaving", "") or "")
        elif isinstance(boot, dict) and boot.get("started_at"):
            # A boot record's heartbeat never moves (it is a boot-time snapshot), so
            # there is no HEARTBEAT age to report for a runtime that published only
            # one — it is left None rather than filled with the process's age, which
            # `age_s` already carries and which would read as a fresh beat.
            heartbeat = None
        item = (
            verdict(
                process,
                view,
                env_of=env_for,
                now=moment,
                root_hint=roots.get(pid, ""),
                session_id_hint=session_id,
            )
            if process is not None
            else None
        )
        rows.append(
            RosterRow(
                pid=pid,
                session_id=session_id,
                port=port,
                build_version=builds.get(pid, ""),
                install_root=str(
                    getattr(record if record is not None else elsewhere, "install_root", "")
                    or (boot.get("install_root") if isinstance(boot, dict) else "")
                    or ""
                ),
                # ATTACHED IS EVIDENCE, NOT A SELF-REPORT: a record's own `detached`
                # bit is a runtime's opinion about its clients, while a viewer lease
                # and an established connection are things the machine can see.
                attached=(
                    observers.get(session_id, 0) > 0
                    or (port is not None and port in view.sockets.attached)
                ),
                observers=observers.get(session_id, 0),
                heartbeat_age_s=heartbeat,
                reachability=reachability,
                state=state,
                has_record=has_record,
                record_root=(
                    str(view.root)
                    if record is not None
                    else roots.get(pid, "") if elsewhere is not None else ""
                ),
                has_boot_record=isinstance(boot, dict),
                busy=busy,
                leaving=leaving,
                config_root=roots.get(pid, ""),
                parent_pid=process.parent_pid if process is not None else 0,
                age_s=process.age_s if process is not None else None,
                cpu_s=process.cpu_s if process is not None else None,
                reclaimable=bool(item is not None and item.may_end()),
                # NO CENSUS ROW, NO VERDICT — and TWO reasons for that, which the
                # roster used to report with one token. ``REFUSAL_GONE`` ("the pid is
                # not running") is only true when the pid really is not there; a pid
                # that IS running but that the census could not match — an argv
                # outside the spawn contract, or a ``ps`` that timed out, which lands
                # EVERY row here — is a different fact and reads ``no-census-row``.
                # Review round 1 (R1-3) measured a live, answering runtime reported as
                # ``gone``; the answer to "why is this not reclaimable" has to be
                # true (see ``REFUSAL_NO_CENSUS``).
                reclaim_refusal=(
                    item.refusal
                    if item is not None
                    else (REFUSAL_GONE if not alive_now else REFUSAL_NO_CENSUS)
                ),
                reclaim_detail=item.detail if item is not None else "",
            )
        )
    return rows
