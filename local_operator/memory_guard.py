"""A per-command RAM ceiling for the bash tool: the k8s/Docker shape, on one command.

An agent session runs a shell command — a build, a ``pip install``, a job with a
mis-sized batch — and it allocates until the *device* runs out of memory. The
kernel then does not politely fail the command; it picks a victim by its own
heuristic, and on this host the victim is often the ``lop`` runtime or another
session's process, taking the whole desktop down. The agent never learns it was
the cause, so the retry is identical.

This module is the fix, mapped onto one shell command: account for the device's
memory, give the command a ceiling, and when it crosses it **kill the command's
whole process group — never the runtime** — so the tool result can say "your
command exceeded the memory budget" and the model can revise to something that
fits. The design contract is ``docs/design/process-memory-guard.md``; this module
is its §3 (budget), §4 (enforcement) and §8 (interface) made real.

What this deliberately is NOT:

* **Not a throttle.** The soft threshold is the ``memory.high`` analog, but in
  userspace we cannot stall an allocation — there is no ``MemoryHigh`` knob a
  process can set on itself. So the soft threshold is an *advisory* (one live
  line), never a slowdown. Anything that claims otherwise is wrong.
* **Not a dependency.** Stdlib only. ``psutil`` is deliberately not a dependency
  in this repo (see ``mobile/resources.py`` and ``procstate.py``); the platform
  probes below are the ones that module already ships, reused rather than
  reinvented.

**It reduces the blast radius; it does not make allocation safe.** A fast
allocator can outrun a 250 ms poll — one tick can be one allocation burst. The
ceiling is ``0.5 x available minus a reserve``, so a command that blows past it
inside one tick still has roughly the reserve of real RAM before the kernel's own
OOM killer looks at the box. The number is a stopgap between "a slow command" and
"a dead session", not a guarantee.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from local_operator.mobile.resources import session_resource_usage


def _platform() -> str:
    """The host platform string, behind a seam so tests need not fake ``sys``.

    The probes branch on the platform, and a test that wants to exercise the
    macOS parse on a Linux host must be able to say so WITHOUT mutating the
    shared ``sys`` module attribute process-wide for the duration of the test
    (review round 1, n2). This indirection is what `monkeypatch.setattr(mg,
    "_platform", ...)` targets; production never overrides it.
    """
    return sys.platform


# ---------------------------------------------------------------------------
# Constants, next to the code that reads them (the `_consumer_defaults()` rule).
#
# The reserve arithmetic is copied from conftest.py (`_MEMORY_SHARE`,
# `_MEMORY_RESERVE_CAP_MB`, `_MEMORY_RESERVE_FRACTION`) with the same values and
# the same shape, so this machine keeps ONE budget vocabulary rather than two that
# can drift. A reader who understands the pytest worker cap understands this.
# ---------------------------------------------------------------------------

#: Fraction of *available* memory one command may claim. The rest is left for the
#: OS, the editor and the other sessions that make this machine contended.
_MEMORY_SHARE = 0.5

#: Floor held out of the budget entirely, so the fraction is not the only thing
#: between a mis-sized command and zero free memory. `min` rather than
#: subtract-then-halve so a solo command is not charged twice.
_MEMORY_RESERVE_CAP_MB = 2048

#: 1/8 of total scales the reserve down on a small device; the cap above bounds it
#: from above on a large one.
_MEMORY_RESERVE_FRACTION = 8

#: Advisory line fires at this fraction of the ceiling (the `memory.high` analog).
_SOFT_FRACTION = 0.8

#: If free swap is below this, the host has *already* paged itself into a corner,
#: so the effective available memory is the min of the measured arm and the
#: free-swap headroom. Swap is NEVER summed into spendable budget: counting swap
#: as headroom would RAISE the ceiling on exactly the thrashing host we are trying
#: to protect. This is a pressure floor on the reserve, nothing more.
_SWAP_FLOOR_MB = 256

#: Small-device floor, and the DEFAULT floor: the bash tool's. The reserve arithmetic
#: above can drive the ceiling to zero
#: on a tight host (~1 GB available on an 8 GB device: `min(512, 1024 - 1024)` =
#: 0 MB), which would kill every command the instant it started — a guard that is
#: worse than no guard. Measured on this host: the smallest command that actually
#: runs (`bash -c 'sleep & sleep'`) peaks at ~4 MB, a plain `git status` ~3 MB, and
#: a trivial `python3 -c` at ~15 MB of interpreter; the realistic smallest
#: "ordinary" command this fleet runs is a Python interpreter, so the floor is set
#: above that. It is a JUDGEMENT, not a calibrated number (no 8 GB device was
#: available to measure against), and it is deliberately low: its job is to keep
#: ordinary commands (a `git status`, a shell pipeline, a `python -c`) alive on a
#: pressured small host, not to hand a big job a licence to run. A command that
#: genuinely needs more than this asks for it — `memory_mb=`, or `mode=manual`.
#: (At 1.5 GB available the arithmetic gives 512 MB, above this floor — the floor
#: binds at ~1 GB, the case named here and in the contract's §11.)
#:
#: It is only the DEFAULT. A caller whose command is not an ordinary one names its
#: own through ``compute_budget(floor_mb=...)`` — see ``mobile.install``'s
#: ``_STEP_MEMORY_FLOOR_MB``, which is ~2x the measured cost of a package-manager
#: child. The floor is a parameter rather than a second calculation downstream
#: precisely so the reserve arithmetic has ONE owner.
_MIN_CEILING_MB = 64

#: Where the config keys live under `values`, spelled ONCE and shared with the
#: `settings_io` rows so the reader and the writer cannot disagree.
BASH_MEMORY_ENABLED_PATH: tuple[str, ...] = ("bash", "memory", "enabled")
BASH_MEMORY_MODE_PATH: tuple[str, ...] = ("bash", "memory", "mode")
BASH_MEMORY_LIMIT_MB_PATH: tuple[str, ...] = ("bash", "memory", "limit_mb")
BASH_MEMORY_SOFT_FRACTION_PATH: tuple[str, ...] = ("bash", "memory", "soft_fraction")

#: Registry defaults. The default is ENABLED: a config that says nothing gets the
#: protection, because the failure this exists for (a dead desktop) is far worse
#: than the failure it risks (a legitimately big command stopped).
BASH_MEMORY_ENABLED_DEFAULT = True
#: ``auto`` = the derived ceiling; ``manual`` = use ``limit_mb``.
BASH_MEMORY_MODE_DEFAULT = "auto"
#: ``0`` in manual mode means "fall back to the auto ceiling" rather than "zero".
BASH_MEMORY_LIMIT_MB_DEFAULT = 0
BASH_MEMORY_SOFT_FRACTION_DEFAULT = 0.8

#: A runner is the same shape ``mobile.resources`` uses: argv -> (rc, stdout). A
#: test injects a fake so no guard test forks a real ``ps``.
Runner = Callable[[list[str]], "tuple[int, str]"]

#: One pid's phys footprint in bytes, or ``None`` when this host cannot answer.
#: Injectable for the same reason ``Runner`` is (see ``mobile/resources.py``).
FootprintProbe = Callable[[int], "int | None"]


def _default_runner(argv: list[str]) -> tuple[int, str]:
    """Run one probe with a short timeout, swallowing every failure mode.

    A missing binary, a timeout or a non-zero exit all collapse to ``(1, "")``
    so callers treat "no data" uniformly and a guard tick can never crash the
    command it is guarding. The timeout is short on purpose: this runs inside the
    command's own poll loop and must not become the thing that stalls it.
    """
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=5.0, check=False)
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


# ---------------------------------------------------------------------------
# Host memory probes (stdlib only). The macOS arms mirror conftest.py's, and the
# two load-bearing details are restated here so they are not re-derived wrongly.
# ---------------------------------------------------------------------------


def _total_memory_mb() -> int | None:
    """Physical RAM in MB, or ``None`` when it cannot be measured.

    ``os.sysconf`` answers this on both macOS and Linux without a subprocess,
    unlike the ``vm_stat`` probe below.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):  # not POSIX, or name absent
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return (pages * page_size) // (1024 * 1024)


def _available_memory_mb(runner: Runner) -> int | None:
    """Memory a command can take without pushing the machine into swap, in MB.

    ``None`` when it cannot be measured; the caller then degrades to ``disabled``
    rather than guessing. The macOS arm is ``free + speculative + file-backed``:
    NOT ``inactive`` (which is dirty and compressor-backed, reclaimable only by
    paging — the cost being avoided, and which reported 8,137 MB of headroom at a
    moment this host had 452 MB free) and NOT consumed swap (which is cumulative
    and never ratchets back down). ``File-backed pages`` is the subset ``vm_stat``
    itself identifies as clean and droppable, so it needs no invented discount.
    """
    if _platform() == "darwin":
        code, out = runner(["vm_stat"])
        if code != 0:
            return None

        header = re.search(r"page size of (\d+) bytes", out)
        if header is None:
            return None
        page_size = int(header.group(1))
        counts: dict[str, int] = {}
        for label in ("Pages free", "Pages speculative"):
            match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.", out, re.MULTILINE)
            if match is None:
                return None
            counts[label] = int(match.group(1))
        # "File-backed pages" is absent on some macOS versions; a miss is 0, not
        # a probe failure (the free-page estimate is still usable).
        file_backed = re.search(r"^File-backed pages:\s+(\d+)\.", out, re.MULTILINE)
        counts["File-backed pages"] = int(file_backed.group(1)) if file_backed else 0
        per_mb = page_size / (1024 * 1024)
        return max(0, int(sum(counts.values()) * per_mb))

    if _platform().startswith("linux"):
        try:
            # MemAvailable is the kernel's own estimate of what can be handed out
            # without swapping — strictly better than MemFree, which ignores
            # reclaimable page cache.
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError):
            return None
        return None

    return None


def _free_swap_mb(runner: Runner) -> int | None:
    """Free swap in MB, or ``None`` when it cannot be measured.

    The INSTANTANEOUS term only. macOS swap ``used`` is cumulative (pages stay in
    the swap file until faulted back or reboot), so it reads "this host swapped
    since boot" rather than "this host is swapping now"; only ``free`` recovers,
    so only ``free`` is a pressure signal.
    """
    if _platform() == "darwin":
        code, out = runner(["sysctl", "-n", "vm.swapusage"])
        if code != 0:
            return None

        # `total = 5120.00M  used = 4011.25M  free = 1108.75M`
        match = re.search(r"free\s*=\s*([\d.]+)([MGT])", out)
        if match is None:
            return None
        value = float(match.group(1))
        unit = match.group(2)
        scale = {"M": 1, "G": 1024, "T": 1024 * 1024}[unit]
        return int(value * scale)

    if _platform().startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("SwapFree:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError):
            return None
        return None

    return None


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """The result of one budget computation. All MB unless noted.

    ``source`` is the provenance a reader needs to know WHY a command was (or was
    not) bounded:

    * ``auto`` — the derived ceiling from measured host memory.
    * ``manual`` — a configured ``limit_mb`` was used instead.
    * ``override`` — the per-call ``memory_mb`` argument was used.
    * ``disabled`` — no ceiling; the command runs unguarded. ``reason`` says why.
    """

    ceiling_mb: int
    soft_mb: int
    available_mb: int | None
    total_mb: int | None
    reserve_mb: int | None
    source: str  # "auto" | "manual" | "override" | "disabled"
    reason: str  # one line, for the result text and the ledger


def compute_budget(
    *,
    mode: str = BASH_MEMORY_MODE_DEFAULT,
    limit_mb: int = BASH_MEMORY_LIMIT_MB_DEFAULT,
    soft_fraction: float = BASH_MEMORY_SOFT_FRACTION_DEFAULT,
    override_mb: float | None = None,
    enabled: bool = BASH_MEMORY_ENABLED_DEFAULT,
    floor_mb: int = _MIN_CEILING_MB,
    runner: Runner | None = None,
) -> Budget:
    """Resolve the per-command ceiling from config + host memory.

    Pure except for the injectable ``runner`` (defaults to a subprocess runner).
    Never raises; an unmeasurable host degrades to ``source="disabled"`` rather
    than to a guess, which is the pre-guard behaviour (no kill, no exception).

    Resolution order, first match wins:

    * ``override_mb == 0`` — disabled for this call, whatever the config says.
    * ``enabled=False`` — the machine-wide MASTER SWITCH (contract §6/§7). It is
      checked BEFORE a positive override, so ``enabled=False, override_mb=64``
      returns ``source="disabled"``: the off switch wins. A positive override
      only *chooses a ceiling*; it does not overrule the switch that says there
      is to be no ceiling.
    * a positive ``override_mb`` — the per-call ceiling.
    * ``mode="manual"`` with a positive ``limit_mb``.
    * else the auto ceiling.

    The override and manual arms do NOT probe host memory: an explicit ceiling
    needs no measurement (the caller, or the config, already said what the
    number is), so their ``available_mb`` is honestly ``None`` rather than a
    ``vm_stat`` subprocess spent only to populate a cosmetic field.

    ``floor_mb`` is the small-device floor, and it is a PARAMETER because the
    caller owns the judgement of what "ordinary" means for the command it is
    bounding. ``_MIN_CEILING_MB`` (the default) is calibrated against the smallest
    things the bash tool is asked to run — a ``git status`` at ~3 MB, a bare
    interpreter at ~15 MB — while a package-manager build child is two orders of
    magnitude larger (measured: ~121 MB for a plain ``pnpm --version``; see
    ``mobile.install._STEP_MEMORY_FLOOR_MB``). Defaulting rather than adding a
    second budget calculation downstream is the point: the reserve arithmetic has
    ONE owner, so a caller can price its own command without a copy of these
    numbers that nobody would notice had drifted.
    """
    base = runner or _default_runner

    def disabled(reason: str) -> Budget:
        return Budget(
            ceiling_mb=0,
            soft_mb=0,
            available_mb=None,
            total_mb=None,
            reserve_mb=None,
            source="disabled",
            reason=reason,
        )

    def _zero_override() -> bool:
        # `0` disables for THIS command; `None` means "use config". Both `0.0`
        # and `0` arrive here as a float from the per-call argument.
        return override_mb is not None and override_mb == 0

    def _positive_override() -> int | None:
        if override_mb is None or override_mb <= 0:
            return None
        return int(override_mb)

    if _zero_override():
        return disabled("disabled for this call (memory_mb=0)")
    if not enabled:
        return disabled("disabled in settings (bash.memory.enabled=false)")

    total_mb = _total_memory_mb()
    # Assigned by the AUTO arm only; the override/manual arms leave it None
    # rather than spending a `vm_stat` probe on a cosmetic field (m4). Named
    # here so the closure below reads it without a NameError on those arms.
    available_mb: int | None = None

    def soft_of(ceiling: int) -> int:
        return int(max(0, ceiling) * max(0.0, min(1.0, soft_fraction)))

    def with_ceilings(ceiling: int, source: str, reason: str, reserve: int | None) -> Budget:
        ceiling = max(0, int(ceiling))
        return Budget(
            ceiling_mb=ceiling,
            soft_mb=soft_of(ceiling),
            available_mb=available_mb,
            total_mb=total_mb,
            reserve_mb=reserve,
            source=source,
            reason=reason,
        )

    override_ceiling = _positive_override()
    if override_ceiling is not None:
        # No host probe: the caller already said what the ceiling is, so this arm
        # needs no measurement (see the docstring). `available_mb` is left None
        # rather than spending a `vm_stat` subprocess on a cosmetic field.
        return with_ceilings(
            override_ceiling,
            "override",
            f"per-call memory_mb={override_ceiling}",
            None,
        )

    if mode == "manual" and limit_mb and limit_mb > 0:
        # Same reasoning as the override arm: the config named the ceiling.
        return with_ceilings(
            limit_mb,
            "manual",
            f"configured bash.memory.limit_mb={int(limit_mb)}",
            None,
        )

    # --- auto: the derived ceiling -----------------------------------------
    available_mb = _available_memory_mb(base)
    if available_mb is None or total_mb is None:
        # F8/F11: no measurable host (non-POSIX, or every probe failed). Degrade
        # to pre-guard behaviour rather than guessing a number that kills.
        return disabled("host memory could not be measured on this platform")

    # Pressure floor: a host whose free swap has bottomed out has already paged
    # itself into a corner, so the ceiling must not stand on a figure that counts
    # pages it can no longer keep resident. NOT summed into spendable budget.
    free_swap_mb = _free_swap_mb(base)
    floor_reason = ""
    effective_available = available_mb
    if free_swap_mb is not None and free_swap_mb < _SWAP_FLOOR_MB:
        effective_available = min(available_mb, free_swap_mb + _SWAP_FLOOR_MB)
        floor_reason = f"; swap-pressure floor (free swap {free_swap_mb} MB)"

    budget_mb = effective_available * _MEMORY_SHARE
    reserve_mb = min(_MEMORY_RESERVE_CAP_MB, total_mb // _MEMORY_RESERVE_FRACTION)
    budget_mb = max(0, min(budget_mb, effective_available - reserve_mb))
    ceiling_mb = int(budget_mb)

    floor_mb = max(0, int(floor_mb))
    floored = False
    if ceiling_mb < floor_mb:
        # §10's small-device hazard made concrete: the reserve arithmetic can land
        # on 0, which would kill every command on the first tick. The floor keeps
        # ordinary commands alive; it is not a licence for a big job.
        ceiling_mb = floor_mb
        floored = True

    reason = (
        f"auto ceiling: {_MEMORY_SHARE:g} x {effective_available} MB available "
        f"minus {reserve_mb} MB reserve{floor_reason}"
    )
    if floored:
        reason += f" (raised to the {floor_mb} MB floor for this command)"
    return with_ceilings(ceiling_mb, "auto", reason, reserve_mb)


# ---------------------------------------------------------------------------
# Sampling a group
# ---------------------------------------------------------------------------


def _parse_group_rss(output: str, pgid: int) -> dict[int, int]:
    """Parse ``ps -axo pid=,pgid=,rss=`` (rss in KiB) into bytes for ``pgid``.

    Returns pid -> bytes for every process whose ``pgid`` column equals ``pgid``.
    Unparseable lines are skipped, not fatal: a stray header or blank line must
    not sink the whole batch.
    """
    result: dict[int, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            row_pgid = int(parts[1])
            rss_kib = int(parts[2])
        except ValueError:
            continue
        if row_pgid == pgid:
            result[pid] = rss_kib * 1024
    return result


def _group_pids(output: str, pgid: int) -> set[int]:
    """The pids in ``ps -axo pid=,pgid=,rss=`` output belonging to ``pgid``."""
    return set(_parse_group_rss(output, pgid))


def group_rss_bytes(pgid: int, *, runner: Runner | None = None) -> int | None:
    """Sum RSS (bytes) of every process in ``pgid`` in ONE ``ps`` pass.

    ``None`` when the group is gone or a probe failed — the caller treats an
    unknown as "do not kill" (fail closed). ``pgid`` is the id the CALLER spawned;
    this function never discovers a group of its own, so it cannot read or kill
    the runtime's own group by accident.

    The fast arm is one ``ps`` per tick, independent of group size, and it catches
    grandchildren that ``sh -c`` spawned into the same group. Measured on this
    host (36 GB, ~12 load, 719 procs): ~30-40 ms for the full table on a quiet
    read, ~47 ms mean under 8 concurrent guarded commands. `ps -g PGID` is ~8x
    cheaper here but is NOT portable-equivalent — Linux procps reads ``-g`` as an
    e-group/session selector, not ``pgid`` — so the pgid filter is done in Python
    over the portable full-table read.
    """
    members = _read_group_members(pgid, runner=runner)
    if not members:
        return None
    return sum(members.values())


def _read_group_members(pgid: int, *, runner: Runner | None = None) -> dict[int, int] | None:
    """One ``ps`` pass -> ``{pid: rss_bytes}`` for ``pgid``, or ``None``.

    The shared spine of :func:`group_rss_bytes` and the guard's fidelity arm:
    returning the membership (not just the sum) lets a refined tick reuse the
    pids this read already resolved instead of forking a second full-table
    ``ps`` to re-derive them (review round 1, n3). ``None`` on any failure,
    same as a vanished group.
    """
    run = runner or _default_runner
    try:
        code, out = run(["ps", "-axo", "pid=,pgid=,rss="])
    except Exception:  # noqa: BLE001 — any probe failure is just "no data"
        return None
    if code != 0:
        return None
    members = _parse_group_rss(out, pgid)
    return members or None


@dataclass
class Sample:
    """One tick's reading of a guarded group."""

    pgid: int
    bytes_used: int | None
    bytes_soft: int
    bytes_hard: int
    over_soft: bool
    over_hard: bool
    #: True once the per-pid fidelity read was spent this tick.
    refined: bool = False


class Guard:
    """Per-command memory guard. One instance per spawned command group.

    Constructed by the tool AFTER the child is spawned and its pgid is known
    (``spawned_pgid``), so the guard is bound to exactly one group and can never
    sample or kill any other. It never enumerates ``os.getpid()``, the session's
    pgid or its own — a guard tick reads a group id it was HANDED.

    A failed sample (no ``ps``, unparseable output, a vanished pid) yields ``None``
    and the tick is a no-op: the guard only kills on a *measured* reading at or
    above the ceiling. Unknown never kills. A guard that killed on a failed read
    would kill every command the moment ``ps`` hiccupped.
    """

    def __init__(
        self,
        pgid: int,
        budget: Budget,
        *,
        runner: Runner | None = None,
        footprint_probe: FootprintProbe | None = None,
        tick_s: float = 0.25,
    ) -> None:
        self.pgid = pgid
        self.budget = budget
        self.runner = runner
        self.footprint_probe = footprint_probe
        self.tick_s = tick_s
        self._advised = False
        #: The largest MEASURED group charge seen this guard's life, for the
        #: result text and details. ``None`` until a reading lands.
        self.peak_bytes: int | None = None

    @property
    def hard_bytes(self) -> int:
        return max(0, self.budget.ceiling_mb) * 1024 * 1024

    @property
    def soft_bytes(self) -> int:
        return max(0, self.budget.soft_mb) * 1024 * 1024

    async def sample(self) -> Sample:
        """Read the group's usage off the event loop (``asyncio.to_thread``).

        Cheap arm first; refines with the per-pid footprint read only when the
        cheap sum is inside the watch band near the soft line. The whole probe
        runs in a worker thread: a ``ps`` read on the loop thread would stall the
        TUI frame, which is exactly what this guard must not do.
        """
        return await asyncio.to_thread(self._sample_sync)

    def sample_sync(self) -> Sample:
        """The same reading as :meth:`sample`, for a caller that has no event loop.

        The bash tool polls from the session's loop, so it hops into a worker
        thread to keep a ``ps`` read off the frame. ``mobile.install``'s build step
        is plain synchronous code (the install path is not async), so there is no
        loop for the hop to be scheduled on and the honest spelling is the read
        itself. Same call, same :class:`Sample`, same never-raise contract: an
        unmeasurable tick yields ``None`` usage and never a kill.
        """
        return self._sample_sync()

    def _sample_sync(self) -> Sample:
        soft = self.soft_bytes
        hard = self.hard_bytes
        members = _read_group_members(self.pgid, runner=self.runner)
        used = sum(members.values()) if members else None
        refined = False

        # Fidelity arm, near the decision only: `ri_phys_footprint`/Pss is the
        # honest number (compressor-inclusive on macOS), but it is a per-pid read
        # whose fallback (`top` dump) can cost seconds. Spend it only when the
        # cheap sum is inside the watch band just below the soft line, where the
        # under-read could change the verdict.
        #
        # The pids come from the SAME `ps` read the cheap sum used (n3): the
        # fidelity arm re-uses this membership rather than forking a second
        # full-table `ps` to re-derive it.
        if used is not None and soft > 0:
            watch_band = max(0, soft - max(used // 8, hard // 20))
            if used >= watch_band:
                honest = self._footprint_total(sorted(members) if members else [])
                if honest is not None and honest > used:
                    used = honest
                refined = True

        if used is not None:
            if self.peak_bytes is None or used > self.peak_bytes:
                self.peak_bytes = used
        return Sample(
            pgid=self.pgid,
            bytes_used=used,
            bytes_soft=soft,
            bytes_hard=hard,
            over_soft=used is not None and soft > 0 and used >= soft,
            over_hard=used is not None and hard > 0 and used >= hard,
            refined=refined,
        )

    def _footprint_total(self, pids: list[int]) -> int | None:
        """Honest footprint sum for ``pids``, or ``None`` when unmeasurable.

        Reuses ``mobile.resources.session_resource_usage`` with OUR runner so a
        guard tick never pays the ``top`` fallback's unbounded cost. ``pids``
        are the membership the tick's OWN ``ps`` read already parsed (n3) — this
        arm never forks its own table read; an empty list yields ``None`` and the
        caller keeps the cheap sum.
        """
        if not pids:
            return None
        try:
            usage = session_resource_usage(
                pids, runner=self.runner, footprint_probe=self.footprint_probe
            )
        except Exception:  # noqa: BLE001 — an unknown footprint is not a kill
            return None
        total = 0
        for entry in usage.values():
            value = entry.footprint_bytes
            if value is not None and value > 0:
                total += value
        return total or None

    def should_kill(self, sample: Sample) -> bool:
        """True iff ``sample.over_hard`` on a MEASURED reading.

        ``None`` usage is never a kill (F6). Pure, so the decision is a unit
        target with no process in sight.
        """
        return sample.bytes_used is not None and sample.over_hard

    def soft_notice(self, sample: Sample) -> str | None:
        """The one-shot live advisory line, or ``None``.

        Latched: returns a string once per guard instance, then ``None``, so a
        group sitting over the soft line does not spam the stream. ADVISORY ONLY —
        see the module docstring: userspace cannot throttle an allocation.

        "One-shot" is about not RE-FIRING, not about how long it stays visible:
        the caller carries this as a field on every subsequent live update (the
        card paints it as a state line beside the running header), so it remains
        on screen until the command ends even though this method returns it only
        once.
        """
        if self._advised or not sample.over_soft:
            return None
        self._advised = True
        used_gb = (sample.bytes_used or 0) / (1024**3)
        hard_gb = sample.bytes_hard / (1024**3)
        return f"memory {used_gb:.1f}/{hard_gb:.1f} GB — approaching the command budget"

    def over_budget_message(self, sample: Sample) -> str:
        """The tool-result text from §5. Plain text, no markdown, no backticks.

        The numbers are the MEASURED group peak and the ceiling; naming them is
        what lets the model size the retry, and it must not name a "safe" number
        it cannot know. Plain Text because the tool card paints Text — backticks
        would land literally.

        ONE paragraph, no internal newline (design review D3), and its DEVICE
        clause is the short `(on a N GB host)` form (design review D6). The tool
        card claims only the FIRST result line for its wrapping reason block and
        bounds that block at :data:`REASON_MAX_CELLS` (432) — a whole sentence
        outside the budget gets its TAIL dropped behind a marker, and the tail is
        the escape hatch (`memory_mb`/`bash.memory.limit_mb`). The old
        `(N GB total, M MB available)` clause pushed the message to 433 cells on
        a 36 GB host and 434 on a 128 GB host — i.e. it cropped at EVERY width,
        and cropped SOONER the more RAM the host had, which is exactly backwards:
        the reminder a big host most needs is the one it lost first. The short
        form measures 405 cells on a 128 GB host, so the whole sentence, escape
        hatch included, fits inside the reason block at every width.
        """
        peak = sample.bytes_used if sample.bytes_used is not None else self.peak_bytes
        peak_gb = (peak or sample.bytes_hard) / (1024**3)
        hard_gb = sample.bytes_hard / (1024**3)
        device = ""
        if self.budget.total_mb is not None and self.budget.available_mb is not None:
            device = f" (on a {self.budget.total_mb / 1024:.0f} GB host)"
        return (
            f"MEMORY LIMIT EXCEEDED: this command's process group reached "
            f"{peak_gb:.1f} GB, over the {hard_gb:.1f} GB budget for one command"
            f"{device}. The command was killed; the session is fine. Reduce peak "
            f"memory and retry: stream instead of loading all rows, lower the batch "
            f"size, or process the input in chunks. To allow a deliberately large "
            f"command, pass memory_mb on the bash call or raise bash.memory.limit_mb "
            f"in settings."
        )


#: ``MemoryGuard`` is the name the design doc uses in prose (§2/§9); ``Guard`` is
#: the name §8's signature block fixes. Both resolve to the same class so an
#: importer may use either without a second, drifting definition.
MemoryGuard = Guard
