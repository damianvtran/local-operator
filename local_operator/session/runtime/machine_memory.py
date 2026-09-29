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
2. The process table once (``ps -axo pid=,ppid=``) for topology only.
3. One batched memory read of the fleet's closure
   (``mobile.resources.session_resource_usage``) — ``ri_phys_footprint`` on
   macOS, which INCLUDES compressed pages, with RSS as the fallback.

**WHAT IT DECIDES.** ``memory_guard.machine_verdict`` grades the sum: ``ok``;
``warn`` (a loud line naming the largest fragments — the operator's chance to
act); ``act`` (the sum is closing on the device, and the single largest
non-runtime fragment is ended when it is at or above
``MACHINE_FRAGMENT_MIN_MB`` — through ``procstate.terminate_process_tree``, the
same group-stop primitive the per-command guard uses, and never a session
runtime: a conversation is not a runaway, the command under it is).

**WHAT IT REFUSES.** Everything fails closed. An unreadable table, a host whose
memory cannot be measured, a fragment below the floor, a withheld kill (the
seat's cooldown), or no live runtimes at all: each produces a report with no
kill. Unknown never kills — the per-command sampler's rule, kept here.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from local_operator import memory_guard
from local_operator.mobile.resources import session_resource_usage

logger = logging.getLogger(__name__)

#: ``ps`` failure collapses to an empty reading, exactly as the per-command
#: sampler's runner does — a guard tick must never be the thing that crashes.
Runner = memory_guard.Runner
FootprintProbe = memory_guard.FootprintProbe


def _default_runner(argv: list[str]) -> tuple[int, str]:
    """Run one probe with a short timeout, swallowing every failure mode.

    Same contract as ``memory_guard._default_runner`` (a missing binary, a
    timeout, or a non-zero exit all collapse to ``(1, "")``); kept local rather
    than imported because that one is private to the per-command guard and this
    module's probes differ (one table read, not three).
    """
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=5.0, check=False)
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def parse_process_table(output: str) -> list[tuple[int, int]]:
    """Parse ``ps -axo pid=,ppid=`` into ``[(pid, ppid)]``.

    Unparseable lines are skipped, not fatal: a header or a blank line must not
    sink the reading (the per-command parser's rule, kept). Rows with a
    non-positive pid are dropped at the source; a reading that cannot name a
    process cannot judge one either.
    """
    rows: list[tuple[int, int]] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        if pid > 0:
            rows.append((pid, ppid))
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
    kill_withheld: bool = False
    reason: str = ""

    def summary(self) -> str:
        """One line, no newline, safe for a log record."""
        parts = [self.reason or self.state]
        if self.killed is not None:
            parts.append(
                f"ended the largest fragment: pid {self.killed.pid} "
                f"({self.killed.mb} MB with its descendants)"
            )
        elif self.kill_withheld:
            parts.append("kill withheld: a fragment was ended recently")
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
    kill: Callable[[int], bool] | None = None,
    total_mb: int | None = None,
) -> MemoryPassReport:
    """Run one aggregate pass; blocking, and the caller hands it to a thread.

    ``apply=False`` measures and grades but never kills (the ``--once`` shape).
    ``kill_allowed=False`` is the seat's cooldown rung: the pass still warns,
    and still names a fragment it WOULD have ended. ``total_mb`` is the host
    probe's test seam; production leaves it ``None`` and the verdict measures.

    The kill is ``terminate_process_tree(pid, force=True)`` — SIGKILL to the
    process group when the pid leads one, which is what a spawned command tree
    is (``start_new_session``), and exactly the stop the bash tool's own guard
    delivers on a breach. A guard acting to keep the device alive cannot wait on
    a graceful exit it has no ladder to escalate; if the fragment's work cannot
    take SIGKILL, nothing here could have stopped it anyway.
    """
    base = runner or _default_runner
    probe = pids_probe or _live_runtime_pids

    code, table = base(["ps", "-axo", "pid=,ppid="])
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
    parents = {pid: ppid for pid, ppid in rows_all}

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
        [
            memory_guard.ProcessNode(pid=pid, ppid=ppid, mb=0)
            for pid, ppid in rows_all
        ]
    )
    closure = memory_guard.descendant_closure(roots, links)

    usage = session_resource_usage(
        sorted(closure), runner=base, footprint_probe=footprint_probe
    )
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
    verdict = memory_guard.machine_verdict(
        fleet_mb, total_mb=total_mb, unmeasured=unmeasured
    )

    nodes = [
        memory_guard.ProcessNode(
            pid=pid, ppid=parents.get(pid, 0), mb=mb_of.get(pid, 0)
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

    # --- act: the sum is closing on the device -----------------------------
    logger.warning(
        "machine memory: %s; largest fragments: %s",
        verdict.reason,
        _fragment_line(top),
    )
    candidate = next(
        (
            fragment
            for fragment in ranked
            if fragment.mb >= memory_guard.MACHINE_FRAGMENT_MIN_MB
        ),
        None,
    )
    if candidate is None:
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
            reason=verdict.reason + f"; this pass does not apply (would end pid {candidate.pid})",
        )
    if not kill_allowed:
        return MemoryPassReport(
            state="act",
            fleet_mb=fleet_mb,
            runtimes=len(roots),
            measured=len(closure) - unmeasured,
            unmeasured=unmeasured,
            top=top,
            kill_withheld=True,
            reason=verdict.reason
            + f"; would end the largest fragment (pid {candidate.pid}), kill withheld",
        )

    stopper = kill or _default_kill
    delivered = False
    try:
        delivered = bool(stopper(candidate.pid))
    except Exception:  # noqa: BLE001 — a stop path never raises out of a pass
        logger.warning("machine memory: the stop for pid %s raised", candidate.pid, exc_info=True)
    logger.warning(
        "machine memory: %s; ended the largest fragment pid %s (%s MB with its "
        "descendants) — delivered=%s",
        verdict.reason,
        candidate.pid,
        candidate.mb,
        delivered,
    )
    return MemoryPassReport(
        state="act",
        fleet_mb=fleet_mb,
        runtimes=len(roots),
        measured=len(closure) - unmeasured,
        unmeasured=unmeasured,
        top=top,
        killed=candidate if delivered else None,
        reason=verdict.reason,
    )


def _default_kill(pid: int) -> bool:
    """Stop one fragment: its process group when it leads one, else the pid."""
    from local_operator.procstate import terminate_process_tree

    return terminate_process_tree(pid, force=True)


def _fragment_line(fragments: Sequence[memory_guard.Fragment]) -> str:
    """``pid 12345 (4120 MB), pid 678 (1904 MB)`` — or a word when there are none."""
    if not fragments:
        return "none"
    return ", ".join(f"pid {fragment.pid} ({fragment.mb} MB)" for fragment in fragments)
