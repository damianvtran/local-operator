"""Root conftest: cap xdist's ``-n auto`` worker count to what the machine can afford.

This file exists for exactly one hook. It has to live at the **rootdir** rather
than in ``tests/``: ``pytest_xdist_auto_num_workers`` is consulted while the
controller is deciding how many workers to spawn, which happens before the
``tests/`` package conftest is loaded, so a copy under ``tests/`` is never
called.

WHY A CAP AT ALL
----------------
``addopts`` asks for ``-n auto``. With ``psutil`` absent (it is deliberately not
a dependency here), xdist falls through its provider chain to ``os.cpu_count()``,
which on this class of machine means **one worker per core** — 14 on a 14-core
box. That is a fine default for a single checkout on an idle machine and a bad
one here, because this repo is worked through many concurrent git worktrees and
several agent sessions run suites at the same time.

Measured on a 14-core / 36 GB host, all numbers from real runs of
``pytest tests/unit``:

* ``-n auto`` resolves to 14 workers. Measuring the xdist workers themselves
  (matched by their execnet command line, on a subset that forks no children of
  its own, so the numbers describe workers and nothing else), 3 interleaved
  rounds: **14 workers = 3,661 MB** of worker RSS, **7 workers = 1,579 MB** -
  57% less, with per-worker RSS flat at 226-262 MB across both. Whole-tree peak
  figures are deliberately not quoted here: they conflate workers with
  subprocesses the tests themselves spawn and are not a function of ``-n``.
* Three suites running concurrently drove load average to **98-128** on 14 cores
  and consumed **6.1 GB of 7.2 GB** of swap. At that point everything on the
  machine is slower, not just the tests.
* Interleaved A/B on ``tests/unit/server`` under that contention, 3 rounds:
  ``-n 14`` took 11.3s / 8.5s / 7.8s, ``-n 4`` took 5.3s / 5.4s / 5.9s. **Fewer
  workers were faster and far more stable.** The suite is dominated by Textual
  pilot tests that wait on the event loop rather than burning CPU, so extra
  workers past a handful buy no throughput and only add context-switch and swap
  pressure.

So the cap is derived from two independent constraints and takes the smaller:
a CPU share that leaves headroom for the rest of the machine, and a memory
budget computed from **available** (not total) memory, so a host already under
pressure from sibling worktrees backs off on its own instead of adding to the
pile-up.

WHAT THIS DOES NOT AFFECT
-------------------------
* ``-n0`` and an explicit ``-n N`` **bypass this hook entirely** — xdist only
  consults it when ``-n`` is ``auto`` or ``logical``. Serialising for a debugger
  (``-n0 --pdb -s``) and forcing a wide run (``-n 12``) both behave exactly as
  before.
* ``PYTEST_XDIST_AUTO_NUM_WORKERS`` still wins, see below.
* **CI keeps every core.** A hosted runner is a dedicated, single-purpose box:
  nothing else competes for it, there are no sibling worktrees, and it is torn
  down after the job. The entire reason for the CPU share is contention that
  does not exist there, and applying it anyway measurably HALVED CI parallelism
  (a 4-vCPU runner resolved to 2 workers instead of 4) - a regression paid on
  every PR. So the share is skipped when ``CI`` is set; the memory budget and
  the 2..8 clamp still apply, because a runner that runs out of memory fails
  exactly the way a laptop does.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings

import pytest

#: Divisor for the memory budget. This is a deliberately CONSERVATIVE ENVELOPE,
#: not the measured per-worker RSS - do not "correct" it to the measured figure.
#: Cleanly measured xdist workers (matched by their execnet command line, on a
#: subset that spawns no subprocesses of its own) sit at 226-262 MB. 600 is a
#: ~2.5x margin over that, and the margin is intentional on three counts: a
#: worker's RSS depends on which tests it draws (the worst observed single
#: worker was ~1,090 MB), some suites fork their own children that the budget
#: still has to cover (the eval-tool tests spawn kernel subprocesses), and the
#: controller's own footprint is charged to no worker. The asymmetry justifies
#: it: under-provisioning costs a little wall time, over-provisioning costs the
#: whole machine a swap storm.
_MB_PER_WORKER = 600

#: Fraction of macOS ``Pages inactive`` treated as genuinely reclaimable. Those
#: pages are a mix of clean (free on demand) and dirty/compressor-backed
#: (reclaimable only by paging), and vm_stat does not split them. Counting all
#: of it is what let the probe report 8 GB of headroom on a swapping machine;
#: counting none of it reports near-zero on a healthy one with a warm cache.
#: A quarter is a deliberate middle: pessimistic enough that pressure moves the
#: number, generous enough not to starve an idle host.
_INACTIVE_RECLAIM_FRACTION = 0.25

#: Fraction of available memory the suite may claim. The rest is left for the
#: editor, the agent sessions and the OS page cache that are the reason this
#: machine is contended in the first place.
_MEMORY_SHARE = 0.5

#: Fraction of cores to claim on a developer machine. Leaving half idle is what
#: keeps a second worktree's suite from turning into a swap storm; the A/B above
#: shows we lose nothing. Deliberately NOT applied on CI - see the module
#: docstring: a dedicated runner has no contention to protect against, and
#: halving its workers only makes every PR slower.
_CPU_SHARE = 0.5

#: Hard bounds. Below 2 the suite stops being parallel at all (and a one-worker
#: xdist run is strictly worse than ``-n0``); above 8 buys nothing measurable on
#: a wait-bound suite and is precisely what produced the load-128 thrash.
_MIN_WORKERS = 2
_MAX_WORKERS = 8

#: Safe answer for any path that cannot measure the machine. Small enough to be
#: harmless on a laptop already under load, large enough to keep the suite
#: parallel.
_FALLBACK_WORKERS = 4


def _swap_used_mb() -> float:
    """Consumed swap in MB, or 0.0 when it cannot be read.

    Subtracted from apparent headroom: a machine that is already paging has less
    room than its free-page count suggests, and this term is what makes the
    budget shrink as real pressure rises. Failing to 0.0 (rather than to None)
    keeps a missing/unswapped host on the free-page estimate alone.
    """
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
        match = re.search(r"used\s*=\s*([\d.]+)M", out)
        return float(match.group(1)) if match else 0.0
    except Exception:  # a missing probe must not break the budget
        return 0.0


def _available_memory_mb() -> int | None:
    """Memory the suite can take without pushing the machine into swap, in MB.

    ``None`` when it cannot be measured; the caller then degrades to the
    CPU-only cap rather than guessing. ``psutil`` would answer this in one call
    and is intentionally NOT a dependency, so each platform is probed through
    what ships with the OS.

    The macOS arm is deliberately NOT a naive "free + inactive". A first cut
    counted ``Pages inactive`` as available and reported 8,137 MB of headroom on
    this host at the exact moment it had 452 MB genuinely free and 6.1 GB of 7.2
    GB of swap consumed - so the arm never became the binding constraint under
    the pressure it exists to detect. On macOS ``inactive`` is not Linux's
    ``MemAvailable``: much of it is dirty and compressor-backed, reclaimable
    only by paging, which is the very cost being avoided. Two corrections:

    * ``inactive`` is DISCOUNTED (only a fraction counts as reclaimable) rather
      than dropped, because dropping it entirely reports near-zero on a healthy
      machine with a warm page cache and would pin a fine host to the floor.
    * Consumed swap is SUBTRACTED. A machine already paging has negative real
      headroom, and this is what makes the probe fall as pressure rises.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5, check=True
            ).stdout
            # Page size is read from the header, never assumed: this host uses
            # 16K pages, so a hardcoded 4096 would under-report by 4x. A miss
            # returns None like every other failure in this function, rather
            # than silently pinning the cap to the floor.
            header = re.search(r"page size of (\d+) bytes", out)
            if header is None:
                return None
            page_size = int(header.group(1))

            counts = {}
            for label in ("Pages free", "Pages inactive", "Pages speculative"):
                match = re.search(rf"^{label}:\s+(\d+)\.", out, re.MULTILINE)
                if match is None:
                    return None
                counts[label] = int(match.group(1))

            per_mb = page_size / (1024 * 1024)
            free_mb = (counts["Pages free"] + counts["Pages speculative"]) * per_mb
            inactive_mb = counts["Pages inactive"] * per_mb * _INACTIVE_RECLAIM_FRACTION
            available = free_mb + inactive_mb - _swap_used_mb()
            return max(0, int(available))

        if sys.platform.startswith("linux"):
            # MemAvailable is the kernel's own estimate of what can be handed out
            # without swapping — strictly better than MemFree, which ignores
            # reclaimable page cache and would badly understate a warm container.
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
            return None
    except Exception:  # a probe must never break collection
        return None

    return None


@pytest.hookimpl
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    """Resolve ``-n auto`` to a worker count this machine can actually sustain.

    Returning a value here takes priority over xdist's own default provider
    (pluggy ``firstresult``), which is why the ``PYTEST_XDIST_AUTO_NUM_WORKERS``
    handling below is re-implemented rather than delegated: xdist reads that
    variable inside the hook we are displacing, so without this branch the
    documented override would silently stop working.

    Never raises. A suite that cannot start because the worker-count heuristic
    threw would be a far worse failure than a suboptimal worker count.
    """
    try:
        override = os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS")
        if override:
            try:
                # An explicit operator override is honoured UNCLAMPED. Someone who
                # types a number has a reason (a dedicated machine, or bisecting a
                # parallelism-sensitive failure at 1); second-guessing it would
                # make the escape hatch useless.
                return int(override)
            except ValueError:
                warnings.warn(
                    f"PYTEST_XDIST_AUTO_NUM_WORKERS is not a number: {override!r}. Ignoring it.",
                    stacklevel=2,
                )

        cpus = os.cpu_count() or 1
        # `CI` is set by GitHub Actions and essentially every other provider.
        # On a dedicated runner take all the cores; the share exists only to
        # protect a shared developer machine.
        on_ci = bool(os.environ.get("CI"))
        cap = cpus if on_ci else max(1, int(cpus * _CPU_SHARE))

        available_mb = _available_memory_mb()
        if available_mb is not None:
            # Budget from AVAILABLE memory, so a machine already hosting three
            # sibling suites hands this run a smaller cap automatically.
            cap = min(cap, int(available_mb * _MEMORY_SHARE) // _MB_PER_WORKER)

        return max(_MIN_WORKERS, min(_MAX_WORKERS, cap))
    except Exception:  # see docstring: never break the run
        return _FALLBACK_WORKERS
