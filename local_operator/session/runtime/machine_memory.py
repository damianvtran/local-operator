"""The machine memory pass: one reading of the whole fleet, and its sum.

**WHY THIS EXISTS, in one incident.** On 2026-09-28 the operator's desktop
saturated — the session runtimes, the command trees under them (a test runner at
~10 GB, a type checker at ~4 GB) and the desktop around them together exhausted
36 GB of RAM and a 3 GB swap, macOS raised its out-of-application-memory dialog,
and the operator force-quit the app mid-turn. Nothing in the product killed
anything, and nothing could: ``memory_guard``'s ceilings are per-command groups,
and its own scope note says there is no machine-wide aggregate budget. This
module is the pass that reads the SUM and, only when the sum demands it, ends
ONE runaway fragment.

**WHAT IT READS.** Three things, each through an injectable seam so no test
forks a process:

1. The live runtimes of ONE config root — the same records ``lop sessions``
   reads (``session.runtime.registry``). The pass exists for THIS install's
   fleet; another store's processes are that store's business, the same scope
   limit the residency sweep documents.
2. The process table once (``ps -axo pid=,ppid=,pgid=``) for topology, with
   ``pgid`` carried solely for the pre-signal re-check.
3. One batched memory read of the fleet's closure
   (``mobile.resources.session_resource_usage``) — ``ri_phys_footprint`` on
   macOS, which INCLUDES compressed pages, with RSS as the fallback.

**WHAT IT DECIDES.** ``memory_guard.machine_verdict`` grades the sum: ``ok``;
``warn`` (a loud line naming the largest fragments — the operator's chance to
act); ``act`` (the sum is closing on the device, and the single largest
non-runtime fragment at or above ``MACHINE_FRAGMENT_MIN_MB`` is ended). The
stop walks the fragment's SUBTREE — every pid the sum counted, descendants
first and the fragment root last — through ``procstate.terminate_process_tree``,
the same stop primitive the per-command guard uses; a leader's group kill
covers anything the walk missed. Immediately before the first signal EVERY
row the ranking summed is re-read in one batched ``ps``, and any change
withholds the whole stop (see :func:`_fragment_refusal`). Never a session
runtime: a conversation is
not a runaway, the command under it is.

**WHAT IT REFUSES.** Everything fails closed. An unreadable table, a host whose
memory cannot be measured, a fragment below the floor, a withheld kill (the
seat's cooldown), a fragment whose rows changed — or could not be re-read —
between the snapshot and the
signal, or no live runtimes at all: each produces a report with no kill.
Unknown never kills — the per-command sampler's rule, kept here.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from local_operator import memory_guard
from local_operator.mobile.resources import direct_ppid_pgid, session_resource_usage

logger = logging.getLogger(__name__)

#: ``ps`` failure collapses to an empty reading, exactly as the per-command
#: sampler's runner does — a guard tick must never be the thing that crashes.
Runner = memory_guard.Runner
FootprintProbe = memory_guard.FootprintProbe


#: How long ONE ``ps``/``top`` read of THIS pass may run. Longer than the
#: per-command guard's 5 s on purpose, and MEASURED: twelve ``ps -axo`` reads under
#: the 2026-09-30 fleet load took 0.42 s min / 3.51 s mean / 13.60 s max, and every
#: one past 5 s collapsed to "no data" — which is how the pass came to withhold
#: four runaway kills (95-309 GB) with "the fragment's rows could not be re-read".
#: This is a PER-ATTEMPT bound; the total a read may spend is
#: :data:`PASS_READ_BUDGET_S`.
PASS_PROBE_TIMEOUT_S = 15.0

#: The most attempts one read gets. A failure under pressure is not always a slow
#: one (a ``ps`` that cannot fork returns at once), so a retry is worth its fork.
PASS_PROBE_ATTEMPTS = 3

#: The TOTAL a single logical read (the table, or the pre-signal re-check) may
#: spend across all its attempts and pauses. Without it the worst case was three
#: 15 s attempts per read — ~46 s for the table and again for the re-check, plus
#: the memory read and the owner dial, i.e. 100+ s against a 60 s cadence. A
#: ``ps`` that has not answered in 20 s is not going to; the fork-free fallback
#: (re-check) or a fail-closed ``unknown`` (table) is the better use of the pass.
#: The seat arms the next pass from this one's START and never overlaps two
#: (``_MachineMemorySweep.kick``), so a long pass is a LATE pass, not a racing one.
PASS_READ_BUDGET_S = 20.0

#: Pause between attempts. Module-level so a test need not sleep for it.
PASS_PROBE_RETRY_PAUSE_S = 0.5


def _default_runner(argv: list[str]) -> tuple[int, str]:
    """Run one probe with a bounded timeout, swallowing every failure mode.

    Same contract as ``memory_guard._default_runner`` (a missing binary, a
    timeout, or a non-zero exit all collapse to ``(1, "")``), with the budget the
    pass can afford (:data:`PASS_PROBE_TIMEOUT_S`) rather than the one a 250 ms
    poll can. Kept local rather than imported because the two budgets are
    deliberately different numbers for different cadences.
    """
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=PASS_PROBE_TIMEOUT_S, check=False
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _run_with_retry(
    runner: Runner, argv: list[str], *, retry_if: Callable[[int, str], bool] | None = None
) -> tuple[int, str]:
    """``runner(argv)`` retried inside :data:`PASS_READ_BUDGET_S`; the last answer.

    A raising runner counts as a failed attempt (the pass's seams are
    injectable, and a probe must never be the reason a pass dies). No new attempt
    starts once the budget is spent, so the budget is the real bound on the read
    (an attempt in flight still runs to its own timeout).

    ``retry_if`` lets a caller say that a non-zero exit is an ANSWER rather than a
    failure — ``ps -p`` exits 1 when every pid is gone, which is information and
    not a reason to fork twice more.
    """
    deadline = time.monotonic() + PASS_READ_BUDGET_S
    code, out = 1, ""
    for attempt in range(PASS_PROBE_ATTEMPTS):
        try:
            code, out = runner(argv)
        except Exception:  # noqa: BLE001 — a failed probe is "no data"
            code, out = 1, ""
        if code == 0 or (retry_if is not None and not retry_if(code, out)):
            break
        if attempt < PASS_PROBE_ATTEMPTS - 1:
            if time.monotonic() + PASS_PROBE_RETRY_PAUSE_S >= deadline:
                break
            time.sleep(PASS_PROBE_RETRY_PAUSE_S)
    return code, out


def parse_process_table(output: str) -> list[tuple[int, int, int]]:
    """Parse ``ps -axo pid=,ppid=,pgid=`` into ``[(pid, ppid, pgid)]``.

    Unparseable lines are skipped, not fatal: a header or a blank line must not
    sink the reading (the per-command parser's rule, kept). Rows with a
    non-positive pid are dropped at the source; a reading that cannot name a
    process cannot judge one either. ``pgid`` rides along for ONE consumer —
    the pre-signal re-check that compares the fragment's rows (the whole walk,
    batched) against the snapshot before a stop goes out.
    """
    rows: list[tuple[int, int, int]] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pgid = int(parts[2]) if len(parts) >= 3 else 0
        except ValueError:
            continue
        if pid > 0:
            rows.append((pid, ppid, pgid))
    return rows


def _live_runtime_pids(config_dir: Path) -> list[int]:
    """The live runtime pids of ONE config root, in stable order.

    The registry import is deferred: this module is pulled in by the wakes
    supervisor's pass, and the supervisor's contract is that it starts without
    dragging the runtime stack in until a pass needs it (the same reason its own
    sweep defers ``reclaim``).

    ``reap=False`` because a READER has no business deleting another process's
    record; ``stale`` rows are skipped so a record whose pid is proven gone
    never anchors a closure.
    """
    from local_operator.session.runtime import registry

    pids: list[int] = []
    for record, state in registry.scan(config_dir, reap=False):
        if state == "stale":
            continue
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0:
            pids.append(pid)
    return pids


@dataclass(frozen=True)
class MemoryPassReport:
    """One pass's reading and its one decision, in the shape a log line needs."""

    state: str  # "ok" | "warn" | "act" | "unknown" | "empty"
    fleet_mb: int
    runtimes: int
    measured: int
    unmeasured: int
    top: tuple[memory_guard.Fragment, ...] = ()
    killed: memory_guard.Fragment | None = None
    #: The ended fragment's lineage keys (:func:`lineage_keys`) — what the seat's
    #: cooldown remembers. Empty unless a stop was delivered.
    killed_lineage: frozenset[tuple[str, int]] = frozenset()
    kill_withheld: bool = False
    #: WHY a stop was withheld: "cooldown" (the seat's rung) or "changed"
    #: (the snapshot no longer matches). Text that names the wrong cause sends
    #: an operator to the wrong place (round 2, R2-3).
    withheld_cause: str = ""
    reason: str = ""

    def summary(self) -> str:
        """One line, no newline, safe for a log record."""
        parts = [self.reason or self.state]
        if self.killed is not None:
            count = len(self.killed.pids)
            parts.append(
                f"ended the largest fragment: pid {self.killed.pid} "
                f"({self.killed.mb} MB across {count} "
                f"process{'es' if count != 1 else ''})"
            )
        elif self.kill_withheld:
            if self.withheld_cause == "cooldown":
                parts.append("kill withheld: a fragment was ended within the cooldown")
            elif self.withheld_cause == "changed":
                parts.append("kill withheld: the fragment changed before the signal")
            elif self.withheld_cause == "unreadable":
                parts.append("kill withheld: the fragment's rows could not be re-read")
            else:
                parts.append("kill withheld")
        return "; ".join(part for part in parts if part)


def _rank_fragments(
    rows: Sequence[memory_guard.ProcessNode], roots: Sequence[int]
) -> tuple[memory_guard.Fragment, ...]:
    """The ranked fragments as a tuple — one call, one ordering, shared by the
    warning text and the kill choice so the two can never disagree."""
    return tuple(memory_guard.fragments_ranked(rows, roots))


def machine_memory_pass(
    config_dir: Path,
    *,
    apply: bool = True,
    kill_allowed: bool = True,
    runner: Runner | None = None,
    footprint_probe: FootprintProbe | None = None,
    pids_probe: Callable[[Path], list[int]] | None = None,
    kill: Callable[[memory_guard.Fragment], bool] | None = None,
    total_mb: int | None = None,
    identity_probe: "IdentityProbe | None" = None,
    in_cooldown: Callable[[frozenset[tuple[str, int]]], bool] | None = None,
    notify: "OwnerNotifier | None" = None,
) -> MemoryPassReport:
    """Run one aggregate pass; blocking, and the caller hands it to a thread.

    ``apply=False`` measures and grades but never kills (the ``--once`` shape).
    ``kill_allowed=False`` withholds every stop, and ``in_cooldown`` withholds the
    stop of ONE candidate: the seat's cooldown rung, scoped to the fragment (see
    ``wakes.supervisor._MachineMemorySweep.in_cooldown``). Either way the pass
    still warns, and still names the fragment it WOULD have ended.
    ``notify`` tells the owning session about a stop that was delivered
    (:func:`notify_owner_of_kill`). ``total_mb`` is the host
    probe's test seam; production leaves it ``None`` and the verdict measures.

    The kill walks the fragment's SUBTREE (``Fragment.pids``) through
    ``terminate_process_tree(pid, force=True)`` — descendants first, the
    fragment root last. One call on the root alone would signal the GROUP only
    when the root leads one; measured on this host, roughly half of the direct
    children of runtimes are not leaders, and their descendants would survive
    the stop while the log credited the whole fragment. A leader's group kill
    still covers anything the walk missed. Immediately before the first signal
    every row the ranking summed is re-read in one batched ``ps`` and any
    change withholds (:func:`_fragment_refusal`) — the snapshot-to-signal
    window the
    residency pass closes the same way. A guard acting to keep the device alive
    cannot wait on a graceful exit it has no ladder to escalate; if the
    fragment's work cannot take SIGKILL, nothing here could have stopped it
    anyway. The ``kill`` seam receives the chosen :class:`Fragment`, not a bare
    pid: the subtree the sum was over is the unit to end.
    """
    base = runner or _default_runner
    probe = pids_probe or _live_runtime_pids

    code, table = _run_with_retry(base, ["ps", "-axo", "pid=,ppid=,pgid="])
    if code != 0:
        return MemoryPassReport(
            state="unknown",
            fleet_mb=0,
            runtimes=0,
            measured=0,
            unmeasured=0,
            reason="the process table could not be read; the fleet's sum is not judged",
        )
    rows_all = parse_process_table(table)
    rows_by_pid = {pid: (ppid, pgid) for pid, ppid, pgid in rows_all}

    roots = [pid for pid in probe(config_dir) if pid > 0]
    if not roots:
        return MemoryPassReport(
            state="empty",
            fleet_mb=0,
            runtimes=0,
            measured=0,
            unmeasured=0,
            reason="no live runtimes in this store",
        )

    links = memory_guard.child_links(
        [memory_guard.ProcessNode(pid=pid, ppid=ppid, mb=0) for pid, ppid, _pgid in rows_all]
    )
    closure = memory_guard.descendant_closure(roots, links)

    usage = session_resource_usage(sorted(closure), runner=base, footprint_probe=footprint_probe)
    mb_of: dict[int, int] = {}
    unmeasured = 0
    for pid in sorted(closure):
        entry = usage.get(pid)
        value = None
        if entry is not None:
            value = entry.footprint_bytes
            if value is None or value <= 0:
                value = entry.rss_bytes
        if value is None or value <= 0:
            unmeasured += 1
            mb_of[pid] = 0
        else:
            mb_of[pid] = value // (1024 * 1024)

    fleet_mb = sum(mb_of.values())
    verdict = memory_guard.machine_verdict(fleet_mb, total_mb=total_mb, unmeasured=unmeasured)

    nodes = [
        memory_guard.ProcessNode(
            pid=pid,
            ppid=rows_by_pid.get(pid, (0, 0))[0],
            pgid=rows_by_pid.get(pid, (0, 0))[1],
            mb=mb_of.get(pid, 0),
        )
        for pid in sorted(closure)
    ]
    ranked = _rank_fragments(nodes, roots)
    top = tuple(ranked[:3])

    if verdict.state == "ok":
        return MemoryPassReport(
            state="ok",
            fleet_mb=fleet_mb,
            runtimes=len(roots),
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            top=top,
            reason=verdict.reason,
        )

    if verdict.state == "warn":
        logger.warning(
            "machine memory: %s; largest fragments: %s",
            verdict.reason,
            _fragment_line(top),
        )
        return MemoryPassReport(
            state="warn",
            fleet_mb=fleet_mb,
            runtimes=len(roots),
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            top=top,
            reason=verdict.reason,
        )

    if verdict.state != "act":
        # ``unknown`` (the host could not be measured) lands here by
        # construction, and making that true is the state's whole contract: a
        # host that could not be measured is not judged, so nothing may be
        # ended on its reading (``memory_guard.machine_verdict`` says "no
        # reading of it can ever kill"). The report still carries what the
        # pass DID read, so the operator can see why nothing moved.
        return MemoryPassReport(
            state=verdict.state,
            fleet_mb=fleet_mb,
            runtimes=len(roots),
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            top=top,
            reason=verdict.reason,
        )

    # --- act: the sum is closing on the device -----------------------------
    logger.warning(
        "machine memory: %s; largest fragments: %s",
        verdict.reason,
        _fragment_line(top),
    )
    eligible = [
        fragment for fragment in ranked if fragment.mb >= memory_guard.MACHINE_FRAGMENT_MIN_MB
    ]
    if not eligible:
        return MemoryPassReport(
            state="act",
            fleet_mb=fleet_mb,
            runtimes=len(roots),
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            top=top,
            reason=verdict.reason
            + (
                "; no single fragment reaches the "
                f"{memory_guard.MACHINE_FRAGMENT_MIN_MB} MB floor, "
                "so nothing was ended"
            ),
        )
    if not apply:
        return MemoryPassReport(
            state="act",
            fleet_mb=fleet_mb,
            runtimes=len(roots),
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            top=top,
            reason=verdict.reason + f"; this pass does not apply (would end pid {eligible[0].pid})",
        )

    # **THE LARGEST UNHELD FRAGMENT, NOT THE LARGEST FRAGMENT.** The cooldown holds
    # a respawn of what was just ended; it must not also hold an UNRELATED runaway
    # that happens to rank second (the 02:41/02:43 shape of 2026-09-30, one rank
    # down). Held fragments are passed over and NAMED, so the log reads "skipped
    # held pid X; ended pid Y" rather than silently choosing a smaller target.
    # ``kill_allowed=False`` is the whole-pass rung and holds everything.
    lineages = {fragment.pid: lineage_keys(fragment, rows_by_pid, roots) for fragment in eligible}
    held: list[memory_guard.Fragment] = []
    candidate = None
    if kill_allowed:
        for fragment in eligible:
            if in_cooldown is not None and in_cooldown(lineages[fragment.pid]):
                held.append(fragment)
                continue
            candidate = fragment
            break
    else:
        held = list(eligible)
    if candidate is None:
        return MemoryPassReport(
            state="act",
            fleet_mb=fleet_mb,
            runtimes=len(roots),
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            top=top,
            kill_withheld=True,
            withheld_cause="cooldown",
            reason=verdict.reason
            + f"; would end the largest fragment (pid {eligible[0].pid}), kill withheld",
        )
    if held:
        logger.warning(
            "machine memory: skipped held (cooldown) %s; ending pid %s (%s MB) instead",
            _fragment_line(held),
            candidate.pid,
            candidate.mb,
        )
    lineage = lineages[candidate.pid]

    cause, message = _fragment_refusal(candidate, runner=base, identity_probe=identity_probe)
    if cause:
        return MemoryPassReport(
            state="act",
            fleet_mb=fleet_mb,
            runtimes=len(roots),
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            top=top,
            kill_withheld=True,
            withheld_cause=cause,
            reason=verdict.reason + f"; {message}",
        )

    stopper = kill or _default_kill
    delivered = False
    try:
        delivered = bool(stopper(candidate))
    except Exception:  # noqa: BLE001 — a stop path never raises out of a pass
        logger.warning("machine memory: the stop for pid %s raised", candidate.pid, exc_info=True)
    count = len(candidate.pids)
    logger.warning(
        "machine memory: %s; ended the largest fragment pid %s (%s MB across %s "
        "process%s) — delivered=%s",
        verdict.reason,
        candidate.pid,
        candidate.mb,
        count,
        "es" if count != 1 else "",
        delivered,
    )
    if delivered:
        owner_pid = _owner_runtime_pid(candidate, rows_by_pid, roots)
        event = KillEvent(
            fragment=candidate,
            owner_runtime_pid=owner_pid,
            fleet_mb=fleet_mb,
            act_mb=verdict.act_mb,
            total_mb=verdict.total_mb,
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            cause="fleet footprint at or above the act line; largest non-runtime fragment",
        )
        _log_kill_event(event)
        try:
            (notify or notify_owner_of_kill)(config_dir, event)
        except Exception:  # noqa: BLE001 — telling the owner must never undo a stop
            logger.warning("machine memory: the owner notification raised", exc_info=True)
    return MemoryPassReport(
        state="act",
        fleet_mb=fleet_mb,
        runtimes=len(roots),
        measured=len(closure) - unmeasured,
        unmeasured=unmeasured,
        top=top,
        killed=candidate if delivered else None,
        killed_lineage=lineage if delivered else frozenset(),
        reason=verdict.reason,
    )


def lineage_keys(
    fragment: memory_guard.Fragment,
    rows_by_pid: dict[int, tuple[int, int]],
    roots: Sequence[int],
) -> frozenset[tuple[str, int]]:
    """What makes two fragments the SAME runaway, as ``(kind, value)`` keys.

    The cooldown exists to stop a war on one respawning process, so two fragments
    are kin only when they share something a respawn would keep: the root pid, or a
    process group / parent that is NOT a session runtime's. **A runtime's own pid
    and process group are deliberately never keys.** A command fragment is normally
    a direct child of its session runtime, and roughly half of those are not group
    leaders (so they share the runtime's group): keying on either would make every
    fragment of one session kin, i.e. a per-SESSION cooldown, and a second,
    unrelated runaway in the same session would be withheld for ten minutes — the
    52 GB / 350 GB shape of 2026-09-30, reproduced by review and QA on this PR's
    first revision (R2/Q2).
    """
    runtime_pgids = {rows_by_pid[root][1] for root in roots if root in rows_by_pid}
    keys: set[tuple[str, int]] = {("pid", fragment.pid)}
    if fragment.pgid > 1 and fragment.pgid not in runtime_pgids:
        keys.add(("pgid", fragment.pgid))
    if fragment.ppid > 1 and fragment.ppid not in set(roots):
        keys.add(("ppid", fragment.ppid))
    return frozenset(keys)


@dataclass(frozen=True)
class KillEvent:
    """Who / what / when / how big for one delivered stop, in the shape a log line
    and an owner notice both need — built once so the two cannot disagree."""

    fragment: memory_guard.Fragment
    #: The session runtime the fragment ran under, or ``None`` when the walk up
    #: the process table did not reach one (a fragment is rooted below a runtime
    #: by construction, so ``None`` means the table changed under the pass).
    owner_runtime_pid: int | None
    fleet_mb: int
    act_mb: int
    total_mb: int | None
    measured: int
    unmeasured: int
    cause: str
    at: float = 0.0


#: Tells the owning session about a delivered stop. Injectable so a test asserts
#: the event without a registry or a socket.
OwnerNotifier = Callable[[Path, KillEvent], object]


def _owner_runtime_pid(
    fragment: memory_guard.Fragment,
    rows_by_pid: dict[int, tuple[int, int]],
    roots: Sequence[int],
) -> int | None:
    """The runtime above ``fragment``: the first ancestor that is one of ``roots``.

    Walks the snapshot's ``ppid`` links, bounded by the table's size so a cycle in
    a torn read cannot spin it. A fragment is by construction rooted below a root,
    so the usual answer is its direct parent.
    """
    root_set = set(roots)
    pid = fragment.ppid
    for _hop in range(len(rows_by_pid) + 1):
        if pid in root_set:
            return pid
        row = rows_by_pid.get(pid)
        if row is None or row[0] == pid or row[0] <= 0:
            return None
        pid = row[0]
    return None


def _log_kill_event(event: KillEvent) -> None:
    """ONE structured line per delivered stop: who, what, how big, why.

    ``key=value`` pairs so an operator can grep a runtime pid or a footprint out
    of the supervisor log. The counts are the pass's own measured/unmeasured split
    (the same honesty the verdict's reason line keeps), because a sum that
    under-counts should say so on the line that justifies a kill.
    """
    fragment = event.fragment
    logger.warning(
        "machine memory kill: owner_runtime_pid=%s fragment_pid=%s pids=%s footprint_mb=%s "
        "fleet_mb=%s act_mb=%s total_mb=%s measured=%s unmeasured=%s cause=%r",
        event.owner_runtime_pid,
        fragment.pid,
        list(fragment.pids),
        fragment.mb,
        event.fleet_mb,
        event.act_mb,
        event.total_mb,
        event.measured,
        event.unmeasured,
        event.cause,
    )


def owner_notice_text(event: KillEvent) -> str:
    """What the owning session's model is told. One paragraph, plain text.

    Names the numbers (so the retry can be sized), the cause (so it is not read
    as a bug in the command) and the way out; it is NOT a throttle notice — the
    process was ended, not slowed.
    """
    fragment = event.fragment
    gb = fragment.mb / 1024
    count = len(fragment.pids)
    fleet = (
        f"{event.fleet_mb / 1024:.1f} GB of {event.total_mb / 1024:.1f} GB"
        if event.total_mb
        else (f"{event.fleet_mb / 1024:.1f} GB")
    )
    return (
        f"MEMORY GUARD: your process group (pid {fragment.pid}, {count} "
        f"process{'es' if count != 1 else ''}, {gb:.1f} GB footprint) was ended by the "
        f"memory guard. The machine's sessions together held {fleet}, over the "
        f"guard's act line, and yours was the largest command tree. The session is "
        f"fine; the command was not completed. Reduce its peak memory (stream or "
        f"chunk the input, lower the batch size, cap the runtime's heap) and bound any "
        f"test rig you start before retrying."
    )


#: Deadline for dialling a live owner. The pass is on a worker thread once a
#: minute, so seconds are free; a runtime that does not answer in this window is
#: spooled to instead (see :func:`notify_owner_of_kill` for what that does and does
#: not promise).
OWNER_DIAL_DEADLINE_S = 5.0


#: What :func:`notify_owner_of_kill` actually achieved. Three outcomes, kept apart
#: because they are three different promises to the owner:
#:
#: * ``"dialled"`` — the runtime acknowledged the message; it reads it at its next
#:   turn boundary.
#: * ``"spooled"`` — the runtime did not answer, so the row sits in the session's
#:   inbox and is delivered when a runtime next OPENS that session (boot, or the
#:   first turn of an unengaged one). For a long-lived but wedged runtime that may
#:   be a long time, and it must never be reported as "told".
#: * ``"not_told"`` — nothing was written.
NoticeOutcome = str


def notify_owner_of_kill(config_dir: Path, event: KillEvent) -> NoticeOutcome:
    """Tell the session that owns a killed fragment; return what was ACHIEVED.

    **WHY THIS EXISTS.** A machine-pass kill reached the supervisor's log and
    nowhere else (2026-09-30: seven runaway waves, the owning sessions never
    learned why their command vanished). The per-command guard already tells its
    owner in the tool result; this is the machine pass's equivalent.

    **TRANSPORT.** Two steps, both existing mechanisms (no new marker format; the
    stop-attribution marker ``control.note_involuntary_stop`` is for ending a
    RUNTIME's run and does not fit ending a child while its runtime lives):

    1. the runtime is live, so DIAL it (``peer_client.send_peer_message``, the
       ``peer_message`` control op) — the same path ``lop send`` uses. The spool is
       not enough on its own: ``inbox.drain_inbox`` runs at runtime boot and once
       at the first turn, so a row appended to a LIVE runtime's spool is not read
       until it next restarts.
    2. when the dial fails (wedged, refused, no answer), SPOOL the row with
       ``inbox.append_inbox`` under ``sessions/<session_id>/``.

    **A DUPLICATE IS ACCEPTED, A LIE IS NOT.** A dial that times out AFTER the
    runtime acknowledged internally falls through to the spool, so the owner can be
    told twice. That is the at-least-once direction ``peer_client`` documents for
    every sender ("delivery is UNCONFIRMED ... may still arrive"), and a second
    copy of a one-paragraph notice costs a line of context; a notice that was
    silently dropped would cost the owner the reason its command vanished. What is
    NOT accepted is reporting more than happened: the return value and the log
    distinguish ``dialled`` from ``spooled`` (see :data:`NoticeOutcome`), and the
    spool log says when the row will be read.

    Never raises: a failed notice is logged and costs the notice, never the stop
    that already happened.

    # COORD: 54427b7ef091 — the stop-attribution lane may supply a richer path
    # (a typed "child ended" marker). Swap the body here; the adapter's name and
    # signature are the contract the pass depends on.
    """
    owner = event.owner_runtime_pid
    if owner is None:
        logger.warning(
            "machine memory: no owning runtime found for fragment pid %s; the owner was not told",
            event.fragment.pid,
        )
        return "not_told"
    try:
        from local_operator.session.runtime import registry

        record = None
        for candidate, state in registry.scan(config_dir, reap=False):
            if getattr(candidate, "pid", None) == owner and state != "stale":
                record = candidate
                break
        if record is None:
            logger.warning(
                "machine memory: runtime pid %s has no live record; owner not told", owner
            )
            return "not_told"
        session_id = str(getattr(record, "session_id", "") or "")
        if not session_id:
            return "not_told"
        text = owner_notice_text(event)
        sender = {"conversation_name": "machine memory guard"}

        import asyncio

        from local_operator.mobile.peer_client import send_peer_message

        try:
            asyncio.run(
                send_peer_message(
                    record,
                    text=text,
                    mode="mailbox",
                    wake=False,
                    sender=sender,
                    deadline_s=OWNER_DIAL_DEADLINE_S,
                )
            )
            logger.info(
                "machine memory: session %s acknowledged the notice that its fragment was "
                "ended (dialled)",
                session_id,
            )
            return "dialled"
        except Exception as exc:  # noqa: BLE001 — fall through to the spool
            logger.info(
                "machine memory: dial of session %s failed (%s); spooling the notice",
                session_id,
                exc,
            )

        from local_operator.session.runtime.inbox import InboxLine, append_inbox

        written = append_inbox(
            config_dir / "sessions" / session_id,
            InboxLine(text=text, sender=sender, mode="mailbox", written_at=time.time()),
        )
        if written:
            logger.warning(
                "machine memory: the notice for session %s is QUEUED, not delivered: its runtime "
                "(pid %s) did not answer, so it is read the next time that session opens a runtime",
                session_id,
                owner,
            )
            return "spooled"
        return "not_told"
    except Exception:  # noqa: BLE001 — a notice never undoes or fails a stop
        logger.warning("machine memory: could not notify the owner of pid %s", owner, exc_info=True)
        return "not_told"


def _fragment_refusal(
    fragment: memory_guard.Fragment,
    *,
    runner: Runner,
    identity_probe: "IdentityProbe | None" = None,
) -> tuple[str, str]:
    """``("", "")`` when every process of the fragment still matches the snapshot.

    **THE DECISION IS A SNAPSHOT AND THE SIGNAL IS NOT** — the same hazard the
    residency pass closes with ``reclaim.target_changed``, and this pass must
    not ship without it: the table read and the fleet's memory read stand
    ~0.1-5 s in front of the signal, and a pid recycled inside that window
    would take the stop. Here that is worse than a stray SIGTERM: a recycled
    pid that happens to LEAD a group takes ``killpg`` — its whole group.

    **THE WHOLE WALK IS RE-CHECKED, NOT JUST ITS ROOT** (round 2, R2-1): the
    stop signals every pid the ranking summed, so every one of them is a
    potential recycled-pid signal. ONE batched ``ps`` for the fragment's pids
    compares each row's ``(ppid, pgid)`` against the snapshot the ranking
    walked (``Fragment.rows``); any missing row or any change withholds the
    WHOLE stop — refusal, never a correction, ``reclaim.target_changed``'s
    contract.

    Returns ``(cause, message)``: ``cause`` is ``"changed"`` or
    ``"unreadable"`` and feeds ``MemoryPassReport.withheld_cause``, so the
    summary and the reason say WHERE the withhold came from — a stale
    snapshot and a ``ps`` that would not answer send an operator to different
    places (round 3, R3-2).

    **THE RE-CHECK USED TO BE THE FIRST INSTRUMENT TO DIE** (2026-09-30): its one
    ``ps -p`` read, on a 5 s timeout, failed under the memory pressure the pass
    exists for, and four runaway fragments (95-309 GB) were withheld as "rows could
    not be re-read". Two changes, neither of which adds kill authority:

    1. the ``ps`` read is retried inside a longer budget (:func:`_run_with_retry`);
    2. when it STILL cannot be read, a FORK-FREE identity check runs instead
       (:func:`fork_free_identity_refusal`): every row of the walk must still
       exist with the same ``(ppid, pgid)``, read by syscall. A fragment that
       passes it is exactly as verified as one that passed ``ps`` — the same
       fields, compared against the same snapshot — and anything it cannot
       confirm withholds the whole stop, as before.
    """
    if fragment.rows:
        snapshot = fragment.rows
    else:  # nothing carried (a hand-built Fragment); fall back to its root row
        snapshot = ((fragment.pid, fragment.ppid, fragment.pgid),)
    csv = ",".join(str(pid) for pid, _ppid, _pgid in snapshot)
    # ``ps -p`` exits 1 with EMPTY output when none of the pids exist (verified on
    # macOS) — but the runner also collapses a TIMEOUT or a failed fork to the same
    # ``(1, "")``, and those ARE worth a retry. The two are told apart without a
    # subprocess: an empty exit-1 is an ANSWER only when every pid of the walk is
    # really gone (``procstate.pid_alive``), and then it is not retried — the
    # fork-free check below names it "changed" instead of "unreadable".
    from local_operator import procstate

    def _not_all_gone(rc: int, text: str) -> bool:
        if rc == 1 and not text.strip():
            return any(procstate.pid_alive(pid) for pid, _ppid, _pgid in snapshot)
        return True

    code, out = _run_with_retry(
        runner, ["ps", "-o", "pid=,ppid=,pgid=", "-p", csv], retry_if=_not_all_gone
    )
    if code != 0:
        fallback = fork_free_identity_refusal(
            fragment, snapshot, identity_probe=identity_probe or _default_identity_probe
        )
        if fallback is None:
            logger.warning(
                "machine memory: the re-check ps could not be read for the fragment at pid %s; "
                "the fork-free identity check confirmed all %d rows, so the stop proceeds",
                fragment.pid,
                len(snapshot),
            )
            return "", ""
        return fallback
    seen: dict[int, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            seen[int(parts[0])] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    for pid, ppid, pgid in snapshot:
        if seen.get(pid) != (ppid, pgid):
            return (
                "changed",
                f"the fragment changed before the signal (pid {pid}); withheld",
            )
    return "", ""


#: One pid's ``(ppid, pgid)`` by a fork-free read, or ``None`` when it cannot be
#: read (gone, foreign, or no reader on this platform). The seam a test injects.
IdentityProbe = Callable[[int], "tuple[int, int] | None"]


def _default_identity_probe(pid: int) -> tuple[int, int] | None:
    """``(ppid, pgid)`` by a syscall, cross-checked, or ``None`` = cannot confirm.

    The row comes from :func:`~local_operator.mobile.resources.direct_ppid_pgid`
    (``proc_pidinfo`` on macOS, ``/proc/<pid>/stat`` on Linux), which is itself the
    existence proof: a pid that is gone, or that this account may not read, answers
    ``None``, and a process this one cannot read is one it must not judge. For the
    pid's own ``pgid`` the kernel is then asked a second, independent way
    (``os.getpgid``): the two must agree, or the row is treated as unreadable.

    **NO ``os.kill(pid, 0)`` HERE, deliberately.** An earlier draft used it as the
    cheap "is it still there" step. On Windows signal 0 is ``TerminateProcess`` —
    a liveness probe that ENDS the process — and ``scripts/xplat_probe.py``'s
    ``static.posix_attributes`` rightly refused it (CI ``xplat-probe-linux`` and
    ``-windows``). The direct reader makes it redundant on every platform that has
    one, and it answers ``None`` on every platform that does not.
    """
    row = direct_ppid_pgid(pid)
    if row is None:
        return None
    try:
        if os.getpgid(pid) != row[1]:
            return None
    except (OSError, AttributeError):
        return None
    return row


def fork_free_identity_refusal(
    fragment: memory_guard.Fragment,
    snapshot: Sequence[tuple[int, int, int]],
    *,
    identity_probe: IdentityProbe,
) -> tuple[str, str] | None:
    """``None`` when EVERY row still matches the snapshot, else ``(cause, message)``.

    The fallback for a ``ps`` that cannot be read. Each pid of the walk is a
    signal target, so each is compared — ``(ppid, pgid)`` against what the ranking
    walked — exactly as the ``ps`` path does. A pid that is GONE (a finished or
    recycled process) or whose row differs is ``"changed"``; one that exists but
    whose row the kernel would not give us is ``"unreadable"`` — the summary must
    send an operator to the right place. Both withhold; this function can only
    ever say "go" when every row has been positively confirmed, which is why it
    adds no kill authority.
    """
    for pid, ppid, pgid in snapshot:
        try:
            row = identity_probe(pid)
        except Exception:  # noqa: BLE001 — a probe that raises is "unknown"
            row = None
        if row is None:
            from local_operator import procstate

            if not procstate.pid_alive(pid):
                # Gone, not merely unreadable: the snapshot is stale, and the
                # operator should read "changed" (a recycled or finished process)
                # rather than be sent looking for a probe that failed.
                return (
                    "changed",
                    f"the fragment changed before the signal (pid {pid} is gone); withheld",
                )
            return (
                "unreadable",
                f"the fragment's rows could not be re-read, and pid {pid} could not be "
                f"confirmed without a subprocess either (pid set starting {fragment.pid}); "
                "withheld",
            )
        if row != (ppid, pgid):
            return (
                "changed",
                f"the fragment changed before the signal (pid {pid}); withheld",
            )
    return None


def _default_kill(fragment: memory_guard.Fragment) -> bool:
    """End the fragment's whole subtree, one process at a time.

    **WHY NOT ONE CALL ON THE ROOT.** ``terminate_process_tree`` signals the
    GROUP only when the pid leads one; a candidate that is not a group leader
    (measured on this host: roughly half of the direct children of runtimes)
    would be signalled alone, leaving its descendants behind — reparented and
    still burning — while the log credited the whole fragment. So every pid the
    ranking actually summed is signalled, descendants first and the fragment
    root LAST (a parent dying first would reparent the children; they keep
    their pids, but ending the root last keeps the stop on the tree the
    snapshot described). Each signal still goes through the same primitive the
    per-command guard uses, so a leader's descendants die with its group even
    if the walk missed them.

    ``True`` when any stop was delivered; a pid that is already gone is not a
    failure (``terminate_process_tree``'s own contract).
    """
    from local_operator.procstate import terminate_process_tree

    order = [pid for pid in fragment.pids if pid != fragment.pid]
    order.append(fragment.pid)
    delivered = False
    for pid in order:
        delivered = terminate_process_tree(pid, force=True) or delivered
    return delivered


def _fragment_line(fragments: Sequence[memory_guard.Fragment]) -> str:
    """``pid 12345 (4120 MB), pid 678 (1904 MB)`` — or a word when there are none."""
    if not fragments:
        return "none"
    return ", ".join(f"pid {fragment.pid} ({fragment.mb} MB)" for fragment in fragments)
