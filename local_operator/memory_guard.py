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
auto ceiling responds to the available-memory and swap-pressure sample for each
command, retains an absolute reserve, and is capped at a fraction of physical RAM
for unusually idle hosts. Each guard applies to one command process group only;
there is no machine-wide aggregate budget, so several concurrent commands can
consume several such ceilings. This is a stopgap between "a slow command" and
"a dead session", not a guarantee.

"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from local_operator.mobile.resources import direct_footprint_bytes, direct_group_pids


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
# Auto-budget constants, next to the code that reads them. These intentionally
# differ from pytest's worker-pool values: a memory test should not silently tune
# the resource limit of one live command group.
# ---------------------------------------------------------------------------

#: Share of measured available memory a single command may claim before other
#: limits apply. Recomputing from current availability keeps it pressure-sensitive.
_MEMORY_SHARE = 0.75

#: Absolute headroom held back, scaled down on small hosts by the second arm.
_MEMORY_RESERVE_CAP_MB = 1024
_MEMORY_RESERVE_FRACTION = 16

#: Per-command cap as a fraction of physical RAM, limiting unusually idle hosts.
#: It is not an aggregate ceiling across concurrently running command groups.
_MEMORY_PHYSICAL_CAP_FRACTION = 0.25

#: Advisory line fires at this fraction of the ceiling (the `memory.high` analog).
_SOFT_FRACTION = 0.8

#: If free swap is below this, lower effective available memory; never count swap
#: as spendable headroom because that would raise limits on a thrashing host.
_SWAP_FLOOR_MB = 256

#: Low default floor prevents the reserve arithmetic from killing ordinary commands
#: on small, pressured hosts; it is not a licence for a large command to run.
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

#: The pids of one process group by a fork-free read, ``[]`` when the group is
#: empty, ``None`` when this host cannot answer. Injectable like the others, and a
#: test that fakes the other two MUST fake this one too: a fake pgid (``100``) is a
#: real group on a live host, and the default reader would answer for it.
GroupProbe = Callable[[int], "list[int] | None"]


def _run_probe(argv: list[str], timeout_s: float) -> tuple[int, str]:
    """One probe subprocess, every failure mode collapsed to ``(1, "")``.

    A missing binary, a timeout or a non-zero exit all read as "no data" so
    callers treat an unknown uniformly and a guard tick can never crash the
    command it is guarding.
    """
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _default_runner(argv: list[str]) -> tuple[int, str]:
    """Run one probe with a short timeout, swallowing every failure mode.

    The timeout is short on purpose for the HOST-BUDGET probes (``vm_stat``,
    ``sysctl``): they run once per command, at spawn, and must not become the
    thing that stalls it. The per-tick MEMBERSHIP read does NOT use this runner —
    see :func:`_membership_runner` for why a 5 s bound was the wrong number there.
    """
    return _run_probe(argv, 5.0)


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

    reserve_mb = min(_MEMORY_RESERVE_CAP_MB, total_mb // _MEMORY_RESERVE_FRACTION)
    # Bound one group by all three independent constraints: a responsive share of
    # memory available at launch, headroom left for the OS/other work, and a
    # physical-RAM fraction for very idle hosts. This is not an aggregate governor;
    # concurrent command groups each calculate their own ceiling.
    physical_cap_mb = total_mb * _MEMORY_PHYSICAL_CAP_FRACTION
    budget_mb = max(
        0,
        min(
            effective_available * _MEMORY_SHARE,
            effective_available - reserve_mb,
            physical_cap_mb,
        ),
    )
    ceiling_mb = int(budget_mb)

    floor_mb = max(0, int(floor_mb))
    # Floors keep tiny ordinary commands alive, but must not punch through the
    # physical-RAM bound (including callers with a larger command-specific floor).
    bounded_floor_mb = min(floor_mb, int(physical_cap_mb))
    floored = False
    if ceiling_mb < bounded_floor_mb:
        ceiling_mb = bounded_floor_mb
        floored = True

    reason = (
        f"auto ceiling: min({_MEMORY_SHARE:g} x {effective_available} MB available, "
        f"{effective_available} MB available minus {reserve_mb} MB reserve, "
        f"{_MEMORY_PHYSICAL_CAP_FRACTION:g} x {total_mb} MB physical cap)"
        f"{floor_reason}"
    )
    if floored:
        reason += f" (raised to the {bounded_floor_mb} MB floor for this command)"
        if bounded_floor_mb < floor_mb:
            reason += f"; requested floor {floor_mb} MB limited by physical cap"
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


def _read_group_table(pgid: int, *, runner: Runner | None = None) -> dict[int, int] | None:
    """One ``ps`` pass -> ``{pid: rss_bytes}`` for ``pgid``; ``None`` = probe FAILED.

    Unlike :func:`_read_group_members` this keeps the two empty readings apart:
    ``None`` means the ``ps`` itself failed (timeout, non-zero exit, raised) and
    is worth retrying, while ``{}`` means ``ps`` answered and the group has no
    members — the command ended — and a retry would only spend a second table
    read to learn the same thing.
    """
    run = runner or _default_runner
    try:
        code, out = run(["ps", "-axo", "pid=,pgid=,rss="])
    except Exception:  # noqa: BLE001 — any probe failure is just "no data"
        return None
    if code != 0:
        return None
    return _parse_group_rss(out, pgid)


def _read_group_members(pgid: int, *, runner: Runner | None = None) -> dict[int, int] | None:
    """One ``ps`` pass -> ``{pid: rss_bytes}`` for ``pgid``, or ``None``.

    The shared spine of :func:`group_rss_bytes`: returning the membership (not
    just the sum) lets a caller reuse the pids this read already resolved instead
    of forking a second full-table ``ps`` to re-derive them (review round 1, n3).
    ``None`` on any failure, same as a vanished group.
    """
    return _read_group_table(pgid, runner=runner) or None


#: How long ONE ``ps`` discovery read may take in total, across its attempts.
#: MEASURED, not chosen: twelve ``ps -axo pid=,pgid=,rss=`` reads under the
#: 2026-09-30 fleet load took 0.42 s min / 3.51 s mean / 13.60 s max, and the 5 s
#: bound this replaces turned that tail into a ``None`` reading ("unknown never
#: kills") on 3 of 8 consecutive ticks while a child held 4 GB against a 3 GB
#: ceiling. The read runs on its OWN thread (see :meth:`Guard._poll_ps`), so a
#: long one delays only the next discovery, never a tick.
_MEMBERSHIP_BUDGET_S = 12.0

#: ``ps`` attempts inside that budget. A fast failure (non-zero exit, a ``ps``
#: that could not fork under pressure) is retried; a timeout spends most of the
#: budget on its own and leaves the second attempt only what is left.
_MEMBERSHIP_ATTEMPTS = 2

#: The longest a single membership ``ps`` may run.
_MEMBERSHIP_ATTEMPT_TIMEOUT_S = 8.0

#: The slowest the ``ps`` read is re-run after one that ANSWERED. It is the
#: fallback membership source (the fork-free group listing is primary), so it
#: only runs where that listing is unavailable.
_MEMBERSHIP_REFRESH_S = 1.0

#: How long to wait before re-running a ``ps`` that FAILED. Longer than the
#: refresh on purpose: a ``ps`` that fails under pressure should not be hammered
#: by the guard (measured before this: 40 forks in 5 s from a failing ``ps``, ~8
#: a second, aimed at the instrument that is already dying).
_MEMBERSHIP_FAILURE_BACKOFF_S = 2.0

#: How long a tick waits for the ``ps`` thread it just started. The breach check
#: has already run by then, so this only buys a same-tick answer from a ``ps``
#: that is fast; a slow or hung one is picked up on a LATER tick and never delays
#: the kill clock (measured before: a hung ``ps`` made every tick 12 s long, so
#: the fork-free check ran once per 12 s instead of per 250 ms).
_PS_GRACE_S = 0.2


def _membership_runner(deadline: float) -> Runner:
    """A runner whose per-call timeout is whatever the tick's budget has left.

    Built per tick (not a module constant) because the budget is a DEADLINE: the
    second attempt must get the remainder of it, not a fresh allowance, or two
    slow reads would stretch one tick past its stated bound.
    """

    def run(argv: list[str]) -> tuple[int, str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 1, ""
        return _run_probe(argv, min(_MEMBERSHIP_ATTEMPT_TIMEOUT_S, remaining))

    return run


@dataclass
class Sample:
    """One tick's reading of a guarded group.

    ``bytes_used`` is ``max(rss_bytes, footprint_bytes)`` of whatever was
    measured: RSS is only the RESIDENT subset of an owned set the kernel may have
    compressed or swapped (measured: a 4 GB hold read 37 MB of RSS and 4,113 MB of
    ``ri_phys_footprint`` on the same process), so the decision keys on the larger
    of the two and never on RSS alone. The extra fields are ADDITIVE — callers
    that predate them read ``bytes_used``/``over_*`` exactly as before.
    """

    pgid: int
    bytes_used: int | None
    bytes_soft: int
    bytes_hard: int
    over_soft: bool
    over_hard: bool
    #: True once a per-pid footprint read contributed to this tick.
    refined: bool = False
    #: The two instruments, kept apart so a result can say which one tripped.
    rss_bytes: int | None = None
    footprint_bytes: int | None = None
    #: Where the pid set came from: ``"group"`` (the fork-free group listing),
    #: ``"ps"`` (a table read), ``"cached"`` (the last known members, each pid
    #: re-verified fork-free) or ``"leader"`` (only the group leader we spawned) —
    #: ``"none"`` when nothing could be read.
    membership: str = "none"


class Guard:
    """Per-command memory guard. One instance per spawned command group.

    Constructed by the tool AFTER the child is spawned and its pgid is known
    (``spawned_pgid``), so the guard is bound to exactly one group and can never
    sample or kill any other. It never enumerates ``os.getpid()``, the session's
    pgid or its own — a guard tick reads a group id it was HANDED.

    **IT KEYS ON FOOTPRINT, NOT RSS.** ``ps`` RSS counts only the pages a process
    has resident right now; ``ri_phys_footprint`` (Pss on Linux) counts the whole
    owned set, compressed and swapped pages at their original size. Measured
    2026-09-30: a node child held 4,096 MB while ``ps`` read 37-2,498 MB and
    ``proc_pid_rusage`` read 4,113 MB on every sample, and the 2026-09-30 incident
    process owned ~198 GB while ``ps`` read ~1.4 GB. A guard that decides on RSS —
    or only looks at the honest number when RSS is already near the line, which is
    what this class used to do — is structurally blind to that class. So every tick
    reads the footprint of every group member through a FORK-FREE per-pid reader
    (:func:`~local_operator.mobile.resources.direct_footprint_bytes`), and the
    decision is ``max(rss_sum, footprint_sum)``.

    **THE SUBPROCESS IS THE FIRST INSTRUMENT TO DIE, SO NOTHING THAT DECIDES A
    KILL DEPENDS ON ONE.** ``ps`` was measured at 0.42-13.6 s under the same load
    that makes a guard necessary, against a 5 s timeout, and an unreadable ``ps``
    used to mean "no reading, so no kill". Membership therefore comes, in order,
    from: (1) the kernel's own group listing
    (:func:`~local_operator.mobile.resources.direct_group_pids`:
    ``proc_listpids(PROC_PGRP_ONLY)`` on macOS, a ``/proc`` scan on Linux — no fork,
    and it sees EVERY member, so ``timeout 900 node ...`` and a late-spawned
    allocator are covered with ``ps`` dead); (2) where that cannot answer, the
    last-known members (each re-verified with ``os.getpgid``) plus the leader we
    spawned, extended by a ``ps`` discovery read that runs on its own thread with
    backoff and never delays a tick. The breach check always runs BEFORE any
    ``ps`` is consulted. ``ps`` supplies RSS only where no footprint reader exists.

    **UNKNOWN STILL NEVER KILLS.** A tick where neither RSS nor any footprint could
    be measured yields ``None`` usage and no kill; an EMPTY group listing is the
    command having ended, not a reading. What counts as measured is the fork-free
    footprint of a pid in the pgid this guard was handed.
    """

    def __init__(
        self,
        pgid: int,
        budget: Budget,
        *,
        runner: Runner | None = None,
        footprint_probe: FootprintProbe | None = None,
        group_probe: GroupProbe | None = None,
        tick_s: float = 0.25,
    ) -> None:
        self.pgid = pgid
        self.budget = budget
        self.runner = runner
        self.footprint_probe = footprint_probe
        self.group_probe = group_probe
        self.tick_s = tick_s
        self._advised = False
        #: The largest MEASURED group charge seen this guard's life, for the
        #: result text and details. ``None`` until a reading lands.
        self.peak_bytes: int | None = None
        #: The last known membership (from the group listing or a ``ps`` read).
        #: Where the listing cannot answer it is reused, each pid re-verified to
        #: still be in OUR group before it is charged (:meth:`_verified_members`).
        self._members: set[int] = set()
        #: The single-flight ``ps`` discovery thread, its latest UNCONSUMED answer
        #: (``None`` = nothing new; ``{}`` = answered, group empty; a dict =
        #: pid -> rss), and the earliest monotonic time another may start.
        self._ps_thread: threading.Thread | None = None
        self._ps_answer: dict[int, int] | None = None
        self._ps_failed = False
        self._ps_only = False
        self._ps_next_at = 0.0

    @property
    def hard_bytes(self) -> int:
        return max(0, self.budget.ceiling_mb) * 1024 * 1024

    @property
    def soft_bytes(self) -> int:
        return max(0, self.budget.soft_mb) * 1024 * 1024

    async def sample(self) -> Sample:
        """Read the group's usage off the event loop (``asyncio.to_thread``).

        Fork-free arms first, ``ps`` only where they cannot answer. The whole
        probe runs in a worker thread: a read on the loop thread would stall the
        TUI frame, which is exactly what this guard must not do.
        """
        return await asyncio.to_thread(self._sample_sync)

    def sample_sync(self) -> Sample:
        """The same reading as :meth:`sample`, for a caller that has no event loop.

        The bash tool polls from the session's loop, so it hops into a worker
        thread to keep a read off the frame. ``mobile.install``'s build step
        is plain synchronous code (the install path is not async), so there is no
        loop for the hop to be scheduled on and the honest spelling is the read
        itself. Same call, same :class:`Sample`, same never-raise contract: an
        unmeasurable tick yields ``None`` usage and never a kill.
        """
        return self._sample_sync()

    def _read_footprint(self, pid: int) -> int | None:
        """One pid's footprint through the injected probe or the fork-free reader.

        A raising probe is "unknown", never an exception out of a tick; a value
        of zero (a zombie reads 0) is not a measurement.
        """
        probe = self.footprint_probe or direct_footprint_bytes
        try:
            value = probe(pid)
        except Exception:  # noqa: BLE001 — an unknown footprint is not a kill
            return None
        return value if value is not None and value > 0 else None

    def _list_group(self) -> list[int] | None:
        """The group's members by the fork-free listing, ``None`` = cannot say."""
        probe = self.group_probe or direct_group_pids
        try:
            return probe(self.pgid)
        except Exception:  # noqa: BLE001 — an unanswerable listing is not a kill
            return None

    def _verified_members(self) -> set[int]:
        """Last-known members that are still in OUR group, plus the leader.

        ``os.getpgid`` is a syscall, so this costs no fork, and it is what makes
        a CACHED pid safe to charge: a pid the kernel has since recycled to an
        unrelated process answers a different group and is dropped, so a stale
        membership can never put a stranger's footprint on this command. The
        leader (``pgid`` itself) is always in: it is the process this guard's
        caller spawned and is bound to by construction.
        """
        live = {self.pgid}
        for pid in self._members:
            if pid == self.pgid:
                continue
            try:
                if os.getpgid(pid) == self.pgid:
                    live.add(pid)
            except (OSError, AttributeError):  # gone, or no getpgid on this platform
                continue
        return live

    def _ps_worker(self) -> None:
        """One bounded ``ps`` discovery read, run on :attr:`_ps_thread`.

        Retried inside :data:`_MEMBERSHIP_BUDGET_S`. Stamps WHEN the next read may
        start, success or failure — the earlier version stamped only on success,
        so a failing ``ps`` was re-forked on every tick.
        """
        deadline = time.monotonic() + _MEMBERSHIP_BUDGET_S
        run = self.runner or _membership_runner(deadline)
        table: dict[int, int] | None = None
        for _attempt in range(_MEMBERSHIP_ATTEMPTS):
            table = _read_group_table(self.pgid, runner=run)
            if table is not None or time.monotonic() >= deadline:
                break
        self._ps_failed = table is None
        self._ps_answer = table
        # ``ps``-only hosts read on every tick after a SUCCESS, but a failure always
        # backs off: hammering the instrument that is failing is what made a dead
        # ``ps`` cost ~8 forks a second.
        if table is None:
            backoff = _MEMBERSHIP_FAILURE_BACKOFF_S
        else:
            backoff = 0.0 if self._ps_only else _MEMBERSHIP_REFRESH_S
        self._ps_next_at = time.monotonic() + backoff

    def _poll_ps(self, *, every_tick: bool) -> dict[int, int] | None:
        """Start a ``ps`` read if one is due; return the newest unconsumed answer.

        Single-flight and non-blocking beyond :data:`_PS_GRACE_S`: the thread is
        started, given a short grace to answer (a fast ``ps`` lands the same tick),
        and otherwise left to finish while ticks carry on without it.
        ``every_tick`` is for a host where ``ps`` is the ONLY source (no footprint
        reader and no group listing): there it is the whole reading, so a read that
        ANSWERED is followed by another on the next tick. A read that FAILED backs
        off either way.
        """
        self._ps_only = every_tick
        thread = self._ps_thread
        if thread is not None and not thread.is_alive():
            thread = self._ps_thread = None
        if thread is None and time.monotonic() >= self._ps_next_at:
            thread = threading.Thread(target=self._ps_worker, name="memory-guard-ps", daemon=True)
            self._ps_thread = thread
            thread.start()
            thread.join(_PS_GRACE_S)
        answer, self._ps_answer = self._ps_answer, None
        return answer

    def _sample_sync(self) -> Sample:
        soft = self.soft_bytes
        hard = self.hard_bytes
        footprints: dict[int, int | None] = {}

        def footprint_of(pids: Iterable[int]) -> int:
            total = 0
            for pid in pids:
                if pid not in footprints:
                    footprints[pid] = self._read_footprint(pid)
                value = footprints[pid]
                if value is not None:
                    total += value
            return total

        # 1. MEMBERSHIP WITHOUT A FORK. The kernel's group listing is exact and
        #    instantaneous; where it cannot answer, the last-known members (each
        #    re-verified) plus the leader stand in.
        listed = self._list_group()
        if listed is not None:
            known = set(listed)
            membership = "group"
            self._members = set(known)
        else:
            known = self._verified_members()
            membership = "cached" if len(known) > 1 else "leader"

        # 2. BREACH CHECK, BEFORE ANY ``ps``. A fork-free footprint at or above the
        #    ceiling is final: nothing below can lower it, so the tick does not
        #    wait on the instrument most likely to be dying.
        known_fp = footprint_of(known)
        readable = any(value is not None for value in footprints.values())
        decided = hard > 0 and known_fp >= hard

        # 3. ``ps``: discovery where the listing could not answer, RSS where no
        #    footprint reader exists. Never needed (and never forked) on the
        #    steady path of a host with both fork-free readers.
        table: dict[int, int] | None = None
        if not decided and (listed is None or not readable) and (listed is None or known):
            table = self._poll_ps(every_tick=not readable)
            if table:
                if listed is None:
                    known = known | set(table)
                    self._members = set(known)
                    membership = "ps"
            elif table is not None and listed is None:
                # ``ps`` answered and the group is empty: the command has ended.
                known = set()
                self._members = set()
                footprints.clear()

        footprint_sum = footprint_of(known) if known else 0
        rss_sum = sum(table.values()) if table else None
        measured_fp = footprint_sum if footprint_sum > 0 else None

        used: int | None
        if rss_sum is None:
            used = measured_fp
        elif measured_fp is None:
            used = rss_sum
        else:
            used = max(rss_sum, measured_fp)
        refined = measured_fp is not None

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
            rss_bytes=rss_sum,
            footprint_bytes=measured_fp,
            membership=membership if used is not None else "none",
        )

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


# ---------------------------------------------------------------------------
# The machine budget: the aggregate arm
# ---------------------------------------------------------------------------
#
# WHY THIS ARM EXISTS, in one incident. On 2026-09-28 this machine saturated:
# the session runtimes, the command trees under them and the desktop around
# them together held the device (36 GB RAM, only 3 GB swap) until macOS put up
# the out-of-application-memory dialog and the operator force-quit the app,
# ending live turns. No guard fired, and none could: every ceiling above is PER
# GROUP, and this module's own scope note says there is "no machine-wide
# aggregate budget". This section is that missing arm — ONE reading of the
# whole fleet, and a decision about the SUM.
#
# WHAT IT IS NOT. It guarantees nothing: a fleet can saturate inside one
# sampling period, and the pass that uses this arithmetic runs on a cadence of
# its own (see ``wakes.supervisor``). It does not throttle anything. And it
# ends a fragment only when ONE fragment is big enough to be worth ending — a
# diffuse overshoot is warned about, never "solved" by killing a scattering of
# small processes, because that frees nothing and destroys turns.

#: Share of physical RAM the fleet's aggregate footprint may reach before the
#: pass warns. Deliberately below the act line: the warning is the operator's
#: chance to act before the guard does.
MACHINE_WARN_FRACTION = 0.75

#: Share of physical RAM the aggregate may reach before the pass may end a
#: runaway fragment. Between warn and act, the per-command ceilings still bound
#: new work but the sum is closing on the device; at or above act, something has
#: to give.
MACHINE_ACT_FRACTION = 0.85

#: The smallest fragment worth ending. Ending means killing a process tree and
#: the work in it; below this floor the kill frees less than the noise it makes,
#: so a diffuse overshoot only ever warns. 1 GiB is calibrated against the
#: measured offenders of 2026-09-28 (a test runner at ~10 GB, a type checker at
#: ~4 GB, several at 1-2 GB): the population this arm exists for is all far
#: above it, and the processes under it are the ones it must never touch.
MACHINE_FRAGMENT_MIN_MB = 1024


@dataclass(frozen=True)
class ProcessNode:
    """One row of the aggregate pass's process table: pid, parent, MB.

    ``mb`` is the process's OWN footprint in MB (never its subtree's); subtree
    sums are derived by :func:`fragments_ranked` so a reader of one row cannot
    mistake which number it is holding.
    """

    pid: int
    ppid: int
    mb: int
    #: The row's process-group id. Carried for ONE consumer: the pass's
    #: signal-time re-check compares it against a fresh read before a stop goes
    #: out (a recycled pid that now leads a group would take its whole group).
    #: ``0`` means "not read" — callers that never re-check leave it at that.
    pgid: int = 0


@dataclass(frozen=True)
class MachineVerdict:
    """The aggregate's standing against the machine budget.

    ``state`` is ``ok`` | ``warn`` | ``act`` | ``unknown``. ``unknown`` is the
    fail-closed state: a host whose memory cannot be measured is not judged (the
    same degradation the per-command budget makes), so no reading of it can
    ever kill.
    """

    state: str
    fleet_mb: int
    warn_mb: int
    act_mb: int
    total_mb: int | None
    reason: str


def machine_verdict(
    fleet_mb: int,
    *,
    total_mb: int | None = None,
    unmeasured: int = 0,
) -> MachineVerdict:
    """Compare the fleet's aggregate footprint against the machine budget.

    ``total_mb=None`` measures the host through the same stdlib probe the
    per-command budget uses (``_total_memory_mb``); an explicit ``total_mb`` is
    the test seam and must not be used to "fix" a host that cannot answer.

    ``unmeasured`` counts the fleet processes neither footprint nor RSS could be
    read for. They are NOT guessed at (a guess would move the verdict), but the
    count rides the reason line so a reading that under-counts says so on its
    own face — the honesty rule the per-command sampler follows for unknowns,
    applied to the aggregate.

    Boundaries are inclusive — at or above a line is over it — matching the
    per-command guard's ``over_hard`` (``>= ceiling``): an operator reading
    "act at 31,457 MB" must see the act fire when the fleet sits exactly there.
    """
    measured_total = _total_memory_mb() if total_mb is None else total_mb
    if measured_total is None or measured_total <= 0:
        return MachineVerdict(
            state="unknown",
            fleet_mb=max(0, int(fleet_mb)),
            warn_mb=0,
            act_mb=0,
            total_mb=None,
            reason="host memory could not be measured; the fleet's sum is not judged",
        )
    fleet_mb = max(0, int(fleet_mb))
    warn_mb = int(measured_total * MACHINE_WARN_FRACTION)
    act_mb = int(measured_total * MACHINE_ACT_FRACTION)
    suffix = (
        f"; {unmeasured} of the fleet's processes could not be measured" if unmeasured > 0 else ""
    )
    if fleet_mb >= act_mb:
        state = "act"
    elif fleet_mb >= warn_mb:
        state = "warn"
    else:
        state = "ok"
    reason = (
        f"fleet {fleet_mb} MB of {measured_total} MB physical "
        f"(warn at {warn_mb} MB, act at {act_mb} MB){suffix}"
    )
    return MachineVerdict(state, fleet_mb, warn_mb, act_mb, measured_total, reason)


def child_links(rows: Iterable[ProcessNode]) -> dict[int, list[int]]:
    """parent pid -> child pids, for parents that are themselves rows.

    A row whose parent is not in the reading (a runtime's parent may be the
    launchd it was detached from, or a pid outside the fleet's slice) links to
    nothing rather than to a phantom — the caller's closure is what decides
    which parents exist. Self-parents are dropped: a pid cannot be its own
    ancestor, and keeping the edge would make every walk below rely on its
    cycle guard instead of its data.
    """
    known = {row.pid for row in rows}
    links: dict[int, list[int]] = {}
    for row in rows:
        if row.ppid == row.pid or row.ppid not in known:
            continue
        links.setdefault(row.ppid, []).append(row.pid)
    return links


def descendant_closure(roots: Iterable[int], links: Mapping[int, list[int]]) -> set[int]:
    """``roots`` plus everything reachable from them under ``links``.

    Cycle-safe by a visited set, not by assumption: pids are recycled and a
    table read mid-recycle can hand back an edge that points up. The closure is
    the SET the aggregate is summed over, so a walk that terminates is the
    difference between a reading and a stack trace.
    """
    seen: set[int] = set()
    stack = [pid for pid in roots if pid > 0]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(links.get(pid, ()))
    return seen


@dataclass(frozen=True)
class Fragment:
    """One candidate: a process and everything under it, in MB and in pids.

    ``pids`` is the exact set the sum is over, so a reaper can walk what the
    number counted. It is ROOT-AWARE: a live session runtime found inside the
    subtree is neither summed into ``mb`` nor listed here nor traversed past —
    a runtime and everything under it is not the guard's to count or to end,
    the same judgement the candidate rule makes. ``ppid``/``pgid`` are the
    root's row, carried for the pre-signal re-check.
    """

    pid: int
    mb: int
    pids: tuple[int, ...] = ()
    ppid: int = 0
    pgid: int = 0
    #: The snapshot row (pid, ppid, pgid) for EVERY pid in ``pids``, in the same
    #: order — what the pre-signal re-check compares a fresh batched read
    #: against. The WHOLE walk is carried, not just its root: every pid in it is
    #: a signal target, and a recycled descendant that leads a group would take
    #: that group (round 2, R2-1).
    rows: tuple[tuple[int, int, int], ...] = ()


def fragment_closure(start: int, links: Mapping[int, list[int]], roots: Iterable[int]) -> set[int]:
    """``start`` plus everything under it that is not a session runtime.

    **WHY THE WALK STOPS AT A ROOT.** The guard may count and end a COMMAND
    tree; a session runtime inside a candidate's subtree — and everything under
    it — is neither (see :func:`fragments_ranked` for the judgement). Stopping
    the traversal, rather than only skipping the pid, is what makes the SUM and
    the STOP agree: a number that included a runtime's workers would credit a
    kill with memory it can never free.

    Cycle-safe by a visited set, the same contract as
    :func:`descendant_closure`. A root is never entered, so the returned set
    contains no root pid and no pid reachable only through one.
    """
    root_set = set(roots)
    seen: set[int] = set()
    stack = [start] if start > 0 else []
    while stack:
        pid = stack.pop()
        if pid in seen or pid in root_set:
            continue
        seen.add(pid)
        stack.extend(links.get(pid, ()))
    return seen


def fragments_ranked(
    rows: Iterable[ProcessNode],
    roots: Iterable[int],
) -> list[Fragment]:
    """Every non-root fragment in ``rows``, largest first (pid breaks ties).

    WHY ROOTS ARE NOT CANDIDATES. A root here is a session runtime — a
    conversation. The runaway this arm exists for is the COMMAND under a
    runtime, and ending a runtime ends turns; that judgement belongs to the
    residency policy (idle runtimes) and to the operator, never to a memory
    pass. A fragment rooted below a runtime is a command tree: fair game, and
    the same unit the per-command guard already ends at its own ceiling.

    Each fragment's number and pid set come from :func:`fragment_closure`, so
    both are ROOT-AWARE: a runtime nested inside a candidate's subtree is
    neither summed nor listed nor walked past — the number and the stop agree
    on what the guard may touch. The sort is total (``(-mb, pid)``) so two
    fragments of equal size rank deterministically — a guard whose choice
    flickers between equal candidates is a guard whose log cannot be read.
    """
    row_list = list(rows)
    links = child_links(row_list)
    mb_of = {row.pid: max(0, row.mb) for row in row_list}
    root_set = set(roots)
    by_pid = {row.pid: row for row in row_list}
    ranked: list[Fragment] = []
    for row in row_list:
        if row.pid in root_set:
            continue
        subtree = tuple(sorted(fragment_closure(row.pid, links, root_set)))
        ranked.append(
            Fragment(
                pid=row.pid,
                mb=sum(mb_of.get(pid, 0) for pid in subtree),
                pids=subtree,
                ppid=row.ppid,
                pgid=row.pgid,
                rows=tuple((pid, by_pid[pid].ppid, by_pid[pid].pgid) for pid in subtree),
            )
        )
    ranked.sort(key=lambda fragment: (-fragment.mb, fragment.pid))
    return ranked
