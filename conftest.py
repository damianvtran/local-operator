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

* ``-n auto`` (14 workers): **7,438 MB peak RSS** across 19 processes; roughly
  **600 MB per worker** including the controller (``-n 6`` peaked at 3,598 MB,
  so the growth is linear).
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
* CI is unchanged in practice: a 4-vCPU runner already resolves below the cap.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings

import pytest

#: Peak RSS a single worker costs, including its share of the controller.
#: Measured on this suite (7,438 MB / 14 workers and 3,598 MB / 6 workers both
#: land here). Used as the divisor for the memory budget; deliberately generous,
#: because under-provisioning costs a little wall time and over-provisioning
#: costs the whole machine to swap.
_MB_PER_WORKER = 600

#: Fraction of available memory the suite may claim. The rest is left for the
#: editor, the agent sessions and the OS page cache that are the reason this
#: machine is contended in the first place.
_MEMORY_SHARE = 0.5

#: Fraction of cores to claim. Leaving half idle is what keeps a second worktree's
#: suite from turning into a swap storm; the A/B above shows we lose nothing.
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


def _available_memory_mb() -> int | None:
    """Available (not total) physical memory in MB, or ``None`` if unmeasurable.

    ``psutil`` would answer this in one call and is intentionally NOT a
    dependency of this project, so each platform is probed through what ships
    with the OS. Every probe is best-effort: any failure returns ``None`` and the
    caller degrades to the CPU-only cap rather than guessing.
    """
    try:
        if sys.platform == "darwin":
            # `vm_stat` reports counts of 16K pages. "Available" for our purpose
            # is free + inactive + speculative: inactive pages are clean and
            # reclaimable on demand, and excluding them would understate a
            # healthy machine's headroom by many gigabytes.
            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5, check=True
            ).stdout
            page_size = 4096
            header = re.search(r"page size of (\d+) bytes", out)
            if header:
                page_size = int(header.group(1))
            pages = 0
            for label in ("Pages free", "Pages inactive", "Pages speculative"):
                match = re.search(rf"^{label}:\s+(\d+)\.", out, re.MULTILINE)
                if match is None:
                    return None
                pages += int(match.group(1))
            return pages * page_size // (1024 * 1024)

        if sys.platform.startswith("linux"):
            # MemAvailable is the kernel's own estimate of what can be handed out
            # without swapping — strictly better than MemFree, which ignores
            # reclaimable page cache and would badly understate a warm container.
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
            return None
    except Exception:  # noqa: BLE001 - a probe must never break collection
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
                    stacklevel=1,
                )

        cpus = os.cpu_count() or 1
        cap = max(1, int(cpus * _CPU_SHARE))

        available_mb = _available_memory_mb()
        if available_mb is not None:
            # Budget from AVAILABLE memory, so a machine already hosting three
            # sibling suites hands this run a smaller cap automatically.
            cap = min(cap, int(available_mb * _MEMORY_SHARE) // _MB_PER_WORKER)

        return max(_MIN_WORKERS, min(_MAX_WORKERS, cap))
    except Exception:  # noqa: BLE001 - see docstring: never break the run
        return _FALLBACK_WORKERS
