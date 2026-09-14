"""Root conftest: cap xdist's ``-n auto`` worker count, and guard the
operator's real session store against the suite.

Two unrelated jobs live here because both need the **rootdir**:

1. ``pytest_xdist_auto_num_workers`` is consulted while the controller is
   deciding how many workers to spawn, which happens before the ``tests/``
   package conftest is loaded, so a copy under ``tests/`` is never called.
2. The real-store guard (:func:`_guard_real_session_store`) has to capture
   the developer's ORIGINAL ``HOME`` before any fixture redirects it, and has
   to be installed before any test module is imported — a test that sets its
   own ``HOME=`` or bypasses ``isolate_environment`` is exactly the case it
   exists for.

THE REAL-STORE GUARD
--------------------
225 of an operator's 244 named sessions vanished from ``~/.local-operator/
sessions`` during a period when several whole-suite ``pytest tests/unit``
runs were executing session-retention tests under heavy load. No
local-operator reaper could account for the loss, and the suite's own
isolation (``tests/conftest.py::isolate_environment``) redirects ``HOME``
per test — but a redirect is only as good as the test that honours it, and
nothing verified afterwards that the real store was untouched. So, from the
original ``HOME`` captured at import:

* ``shutil.rmtree``, ``os.rmdir``, ``os.removedirs``, ``os.rename``,
  ``os.replace``, ``os.renames``, ``shutil.move``, ``pathlib.Path.rmdir``,
  ``pathlib.Path.rename``, ``pathlib.Path.replace``, and the file removers
  ``os.unlink``, ``os.remove``, ``pathlib.Path.unlink`` are WRAPPED for the
  whole session to raise :class:`RealStoreTouched` on any argument that
  resolves under the real store, whatever ``HOME`` says at the time.
* At session start the store's entry NAMES are snapshotted (read-only, one
  ``scandir``); at session end the snapshot must still be a subset of the
  live listing. Entries may be ADDED by the operator's own sessions running
  alongside; none may vanish. A shrink fails the run with the missing ids.
* If the real store does not exist (CI, a fresh machine) both are no-ops.

This is defence against ANY actor in the process — a fixture teardown, a
``tmp_path`` computed from a stale ``HOME``, a test's own ``rmtree`` — not
only against the harness's own code, which the AST test in
``tests/unit/session/test_no_session_deletion.py`` covers separately.

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

WHAT THIS SUITE REPORTS, AND WHY IT DOES NOT DIVIDE BY THE FLEET
---------------------------------------------------------------
Each of the two arms above is a statement about ONE suite - "half the cores",
"half of what remains" - so N sibling suites claim N x 50% and the fleet's total
claim is unbounded by either. Dividing both arms by the number of live sibling
suites is the obvious fix, and it WAS IMPLEMENTED AND MEASURED on 2026-09-13 and
is deliberately NOT what this file does. Recorded here so nobody re-tries it
without the data:

* **It cannot help on this host.** ``available`` memory has been 4,700-6,100 MB
  for hours, where the memory arm alone already resolves 2-4 workers; the
  divisor resolved **2, the floor**, for every instance - it removes parallelism
  the memory arm says is affordable without removing any fleet total.
* **It costs wall time.** Interleaved A/B at real fleet depth (3 concurrent
  instances of ``tests/unit/tui/test_slash_echo.py``, a boot-bound slice, with 11
  sibling suites live): at 3 workers per instance the instances took 219.8 /
  216.3 / 209.8 s; at 2 workers - what the divisor resolves - 338.4 / 314.5 /
  308.1 s. **+45% wall time for the same work**, at effectively identical CPU
  (sum 105.7 s vs 103.1 s) and with the swap counter flat (13,432.6 -> 13,441.1
  MB across the wave). The suite is wait-bound, so fewer workers means less
  latency hiding, not less contention.
* **Fleet depth is not what makes these suites slow.** The same slice took 176.8 s
  at 1 instance / 4 workers against ~207 s per instance at 3 instances / 3
  workers each: three concurrent suites cost each instance ~10-15% while
  tripling aggregate throughput. What turns a ~57-minute suite into hours is the
  per-suite worker count, not the number of suites.

So nothing here divides. What the file does instead is REPORT the resolution,
one line on stderr, because the terms behind a count are otherwise invisible
from outside the process and a session can spend hours guessing which one bound:

    pytest worker cap: 6 (bound by memory, cpu arm 7, memory arm 6,
    available 5,313 MB, reserve 2,048 MB, siblings 11)

That line, and the read-only sibling probe it uses, are part of the change from
that investigation; the count itself is computed as it always was, with the two
constants below moved to where the measurement says they belong.

THE TITRATION
-------------
If the cap is what binds, the next question is which count this slice actually
wants - because "3 workers" on a 14-core box at 0-2.5% CPU is a memory verdict,
not a compute one. Measured 2026-09-13/14 with interleaved waves of 3 concurrent
instances of ``tests/unit/tui/test_slash_echo.py`` (71 tests, ~58 Textual app
boots, the boot-bound shape the whole suite is 82% of), staged 25 s apart, at
real fleet depth - 11-13 sibling suites live throughout. Median per-instance
wall time, and the three instances' total CPU:

* cap 3 (what this host resolved): 313.3 s, 225.2 s over two rounds; 118.4 s and
  107.8 s of CPU.
* cap 6: 188.9 s, 161.6 s, 188.7 s over three rounds; 129.2 s, 123.9 s, 129.5 s.
  **28-40% faster per instance for ~12% more CPU.**
* cap 8: 171.3 s (one paired round); 150.2 s of CPU. 9% faster than cap 6 for
  16% more CPU - it does NOT clear the >=15% bar that would justify a wider run,
  so the 2..8 clamp and the 0.5 CPU share are untouched.

Swap and free memory are NOT the discriminator here and are reported honestly as
such: across those waves ``swap_used`` moved -2,173, -36 and +480 MB on the raised
arm and -331 and +57 MB on the current one, while the ~13 sibling suites move the
same counter by ~500 MB on their own. No wave collapsed free memory (the lowest
reading, 17 MB, was recovered within the next wave), and no raised-arm wave grew
swap on more than one of three rounds - so the raised arm is bounded, not
proven harmless, and the per-instance wall time is what carries the decision.

The two constants that produce 6 at this host's chronic 5,000-5,500 MB available
are ``_MB_PER_WORKER`` (600 -> 400 MB) and the reserve (3,072 -> 2,048 MB); see
their own comments for why each moved and what margin is retained.

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

  **One exception, and it is our own doing.** local-operator's bash tool sets
  ``CI=1`` on every agent-run command so CLIs behave non-interactively. That
  made every agent-run suite on the operator's own laptop look like a
  dedicated runner and take all 14 cores - disabling the share exactly where
  it was needed. The tool now also sets ``LOCAL_OPERATOR_AGENT_SHELL=1`` and
  this hook denies on it, so ``CI`` keeps meaning "dedicated runner" for every
  real provider while the harness stops lying to itself. Deliberately a
  denylist: an allowlist of provider variables would silently halve
  parallelism on every provider nobody remembered to add.

* **A memory reserve is held back** on top of the fraction, scaled per host
  (``min(2048, total // 18)`` MB). It exists because the fraction claims a share
  of what REMAINS, so sibling suites converge toward zero free memory rather
  than toward a floor. Note its real reach before tuning it: because the shape
  is ``min(share, available - reserve)``, it binds only below twice itself (~4 GB
  free on a 36 GB box) and is invisible above that - and this host has been
  BELOW that boundary for hours at a time, which is why the reserve, not the
  share, is the term to look at when a suite feels slow.
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import shutil
import subprocess
import sys
import warnings

import pytest

from tests import shard_stall_watchdog

# ---------------------------------------------------------------------------
# Real-store guard
# ---------------------------------------------------------------------------

#: The developer's real store, resolved from the ORIGINAL environment at
#: import time — before ``isolate_environment`` or any test can move HOME.
#: ``LOCAL_OPERATOR_CONFIG_DIR`` is honoured if the developer runs with one;
#: otherwise ``~/.local-operator``. ``None`` when there is no such store.
_REAL_STORE: pathlib.Path | None = None
_REAL_STORE_ENTRIES: frozenset[str] | None = None
#: The operator's real ``config.yml``, byte-for-byte at session start. A test
#: process rewrote it once — the cleanup migration ran from ConfigManager's
#: load path, and an un-isolated probe constructed one on the real dir (PR
#: #645, round 5) — so a changed config at session end fails the run the same
#: way a shrunken store does.
_REAL_CONFIG: pathlib.Path | None = None
_REAL_CONFIG_BYTES: bytes | None = None


def _resolve_real_store() -> pathlib.Path | None:
    override = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR")
    base = (
        pathlib.Path(override)
        if override
        else pathlib.Path(os.path.expanduser("~")) / ".local-operator"
    )
    store = base / "sessions"
    try:
        return store.resolve(strict=True) if store.is_dir() else None
    except OSError:
        return None


class RealStoreTouched(RuntimeError):
    """A test tried to remove, rename or replace something under the real store."""


def _under_real_store(candidate: object) -> bool:
    if _REAL_STORE is None:
        return False
    try:
        if isinstance(candidate, pathlib.Path):
            path = candidate
        elif isinstance(candidate, (str, bytes, os.PathLike)):
            path = pathlib.Path(os.fsdecode(candidate))
        else:
            return False
        resolved = path.resolve()
    except (TypeError, ValueError, OSError):
        return False
    return resolved == _REAL_STORE or _REAL_STORE in resolved.parents


def _guarded(original, *, positions: tuple[int, ...]):
    """Wrap ``original`` so the arguments at ``positions`` are checked first."""

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        for index in positions:
            if index < len(args) and _under_real_store(args[index]):
                raise RealStoreTouched(
                    f"refusing {original.__module__}.{original.__name__} on {args[index]!s}: "
                    f"it is under the operator's real session store {_REAL_STORE}. "
                    "Tests must never touch it; fix the test's isolation."
                )
        return original(*args, **kwargs)

    return wrapper


def _install_real_store_guard() -> None:
    global _REAL_STORE, _REAL_STORE_ENTRIES, _REAL_CONFIG, _REAL_CONFIG_BYTES
    _REAL_STORE = _resolve_real_store()
    if _REAL_STORE is None:
        return
    config = _REAL_STORE.parent / "config.yml"
    try:
        _REAL_CONFIG_BYTES = config.read_bytes()
        _REAL_CONFIG = config
    except OSError:
        _REAL_CONFIG = None
    try:
        with os.scandir(_REAL_STORE) as entries:
            _REAL_STORE_ENTRIES = frozenset(entry.name for entry in entries)
    except OSError:
        _REAL_STORE_ENTRIES = None
    # Both the source (a session directory being moved away) and the target
    # (something being moved onto it) are checked for the two-argument forms.
    shutil.rmtree = _guarded(shutil.rmtree, positions=(0,))
    shutil.move = _guarded(shutil.move, positions=(0, 1))
    os.rmdir = _guarded(os.rmdir, positions=(0,))
    os.removedirs = _guarded(os.removedirs, positions=(0,))
    os.rename = _guarded(os.rename, positions=(0, 1))
    os.replace = _guarded(os.replace, positions=(0, 1))
    os.renames = _guarded(os.renames, positions=(0, 1))
    # ``Path`` methods take ``self`` at position 0 and the target at 1.
    pathlib.Path.rmdir = _guarded(pathlib.Path.rmdir, positions=(0,))
    pathlib.Path.rename = _guarded(pathlib.Path.rename, positions=(0, 1))
    pathlib.Path.replace = _guarded(pathlib.Path.replace, positions=(0, 1))
    # File removers too (QA round 1, Q5): a test that unlinks a transcript
    # inside a real session directory empties the session without changing
    # the entry count, so the tripwire below would never notice.
    os.unlink = _guarded(os.unlink, positions=(0,))
    os.remove = _guarded(os.remove, positions=(0,))
    pathlib.Path.unlink = _guarded(pathlib.Path.unlink, positions=(0,))


_install_real_store_guard()


# ---------------------------------------------------------------------------
# Shard stall watchdog
# ---------------------------------------------------------------------------
# Four `test (3.12, N)` jobs have been CANCELLED at the workflow's 20-minute cap
# after a silent ~6-minute tail, which fails the PR. That tail is invisible in
# the log by construction: pytest prints one progress line per 72 completed
# tests, so the final partial line is withheld until its slowest item finishes
# and a single stuck test is indistinguishable from a uniformly slow batch. The
# only way to recover the culpable test is to watch from inside the run, so this
# records in-flight node ids on the controller and dumps worker stacks with the
# C-level timer. Inert unless LOCAL_OPERATOR_SHARD_STALL_SECONDS is set, which
# only the shard job in ci.yml does; tests/shard_stall_watchdog.py carries the
# reasoning, the safety argument and the measurement behind the design.


def pytest_configure(config: pytest.Config) -> None:
    shard_stall_watchdog.install(config)


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    shard_stall_watchdog.note_start(nodeid)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    shard_stall_watchdog.note_report(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run if any pre-existing entry of the real store is gone."""
    shard_stall_watchdog.shutdown()
    if _REAL_STORE is None or _REAL_STORE_ENTRIES is None:
        return
    try:
        with os.scandir(_REAL_STORE) as entries:
            now = frozenset(entry.name for entry in entries)
    except OSError as exc:
        # The store became unreadable during the run. Loud, not silent — but
        # a directory that cannot be listed is not proof of loss, so warn.
        warnings.warn(f"real-store tripwire: cannot re-list {_REAL_STORE}: {exc}", stacklevel=1)
        return
    if _REAL_CONFIG is not None and _REAL_CONFIG_BYTES is not None:
        try:
            after = _REAL_CONFIG.read_bytes()
        except OSError:
            after = None
        if after != _REAL_CONFIG_BYTES:
            message = (
                f"REAL CONFIG CHANGED DURING THIS TEST RUN: {_REAL_CONFIG} is not byte-identical "
                "to its session-start snapshot. Something in this run (or running alongside "
                "it) rewrote the operator's config; a test must never construct a "
                "ConfigManager on the real dir, and loading must never write."
            )
            print(f"\n{message}", file=sys.stderr)
            pytest.exit(message, returncode=pytest.ExitCode.TESTS_FAILED)
    missing = sorted(_REAL_STORE_ENTRIES - now)
    if missing:
        message = (
            f"REAL SESSION STORE SHRANK DURING THIS TEST RUN: {len(missing)} entr"
            f"{'y' if len(missing) == 1 else 'ies'} of {_REAL_STORE} vanished: "
            f"{', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}. "
            "Something in this run (or running alongside it) removed them."
        )
        # ``pytest.exit`` is the one exception ``wrap_session`` catches around
        # this hook and turns into the exit status; anything else is reported
        # as an internal error and the message is buried. Runs on the xdist
        # controller AND every worker (each imports this conftest), so a
        # shrink is reported by whichever process notices it first.
        print(f"\n{message}", file=sys.stderr)
        pytest.exit(message, returncode=pytest.ExitCode.TESTS_FAILED)


#: Divisor for the memory budget. This is a deliberately CONSERVATIVE ENVELOPE,
#: not the measured per-worker RSS - do not "correct" it to the measured figure.
#: Cleanly measured xdist workers (matched by their execnet command line, on a
#: subset that spawns no subprocesses of its own) sit at 226-262 MB; 400 is a
#: 1.5-1.8x margin over that, and the margin is retained on purpose: a worker's
#: RSS depends on which tests it draws (the worst observed single worker was
#: ~1,090 MB, which no budget shape here covers anyway - the value of this
#: constant is that it bounds the COMMON case, not the tail), some suites fork
#: their own children that the budget still has to cover (the eval-tool tests
#: spawn kernel subprocesses), and the controller's own footprint is charged to
#: no worker. The asymmetry still justifies the margin: under-provisioning costs
#: wall time, over-provisioning costs the whole machine a swap storm.
#:
#: WHY IT MOVED FROM 600 (2026-09-14): 600 against a measured 226-262 MB was a
#: 2.5x margin, and on a host that has been sitting at 4,500-5,500 MB of
#: *available* memory for hours it is the term that decides everything - the
#: budget is ``min(0.5 * available, available - reserve)``, so an inflated charge
#: per worker turns ~5 GB of free memory into 3 workers on 14 cores that are
#: 0-2.5% busy. The titration in the module docstring is what justifies the new
#: number: cap 6 beat cap 3 by 28-40% wall time per instance for ~12% more CPU at
#: this host's chronic availability, and 400 MB/worker is what lets 6 workers fit
#: in ~2.5 GB of budget. The measured fleet aggregate for 18 workers across three
#: concurrent instances peaked at 2,462 MB INCLUDING controllers, children and
#: the app processes the suite spawns - i.e. the envelope is still ~1.6x above
#: what was actually observed at that width.
_MB_PER_WORKER = 400

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

#: Environment marker set by local-operator's own bash tool
#: (``local_operator/tools/builtin.py``, ``NON_INTERACTIVE_ENV``) on every
#: agent-run command. Its presence means ``CI`` below is NOT trustworthy as a
#: "dedicated runner" signal: the harness sets ``CI=1`` to make CLIs
#: non-interactive, on a shared laptop that may be running several agent
#: sessions and their suites at once.
#:
#: Measured 2026-09-08 on the 14-core / 36 GB host: six concurrent agent-run
#: suites resolved to 8/6/5/4/4/2 = 29 workers, ~11.2 GB of worker RSS, load
#: average 220, 8.67 of 10 GB swap consumed, 448 MB free. The leading ``8`` is
#: how the inversion was found - it is unreachable with the share applied
#: (``int(14 * 0.5) = 7``), so only a lifted CPU arm can produce it.
#:
#: Denylist, not allowlist, and the distinction is load-bearing: narrowing the
#: check to ``GITHUB_ACTIONS`` and friends would silently drop every provider
#: not on the list (Jenkins, Azure, Travis, self-hosted runners that set only
#: ``CI``) to the developer share - the measured 4-vCPU-resolves-to-2 halving
#: this file already documents as an unacceptable regression. The set of CI
#: providers is open; the set of harnesses lying about ``CI`` on this machine
#: is closed and ours.
_AGENT_SHELL_ENV = "LOCAL_OPERATOR_AGENT_SHELL"

#: Memory held back from the budget entirely, in MB, computed per host.
#:
#: WHY: ``_MEMORY_SHARE`` claims a fraction of what REMAINS, so N sibling
#: suites each halve the remainder - 1 suite leaves 50% free, 3 leave 12.5%,
#: 6 leave 1.6%. Every suite is individually polite and the fleet still walks
#: into swap. Holding an absolute floor out of the budget first is what stops
#: the walk being purely geometric.
#:
#: SHAPE - ``min(share, available - reserve)``, NOT ``(available - reserve) *
#: share``. Subtracting first then halving charges two independent politeness
#: terms to the same memory, costing several workers even when this suite is the
#: ONLY one running: a solo developer at 6 GB free would drop 7 workers to 4.
#: The ``min`` form expresses "leave the reserve free" without that penalty.
#:
#: THE CONSEQUENCE, which is the single most useful thing to know before
#: tuning these constants: ``min(a * 0.5, a - reserve) == a * 0.5`` for all
#: ``a >= 2 * reserve``. So this term BINDS ONLY BELOW ~2x itself (4,096 MB
#: here) and is invisible above that. It is a floor under a single suite's
#: appetite when memory is genuinely short - NOT a fleet-total lever. An
#: earlier draft claimed a ~3.1 GB fleet reclaim, but that reclaim came
#: entirely FROM the double-charge removed above; the two corrections cancel.
#: READ THIS BEFORE TUNING IT AGAIN: this host spends hours at a time BELOW
#: that 4 GB boundary, so the reserve is not a rarely-armed safety net here - it
#: is the ordinary binding term, and its size is the common-case worker count.
#:
#: DELIBERATELY NOT CALLED AN ASYMPTOTE, because it is not one.
#: ``_MIN_WORKERS`` keeps charging a flat ~2-worker tax once the memory arm is
#: exhausted, so a deep enough fleet still walks to zero; the reserve
#: DISPLACES that point rather than preventing it. Modelled at 395 MB/worker
#: from 12.4 GB available, the trough improves by ~395 MB at every fleet depth
#: (n=6: 1,735 -> 2,130 MB) and n=10 still goes negative. Six simultaneous
#: suites remain unbounded by this term - and dividing the budget by the live
#: fleet size was implemented and REJECTED as the fix, because on this host it
#: resolves to the 2-worker floor and costs ~45% wall time for the same work
#: (the module docstring carries the measurement). The tension this leaves is
#: recorded rather than solved: under chronic pressure these constants, not the
#: fleet size, are what bind, and they are the lever a future titration has to
#: move. A cross-process budget would be a truer lever for the fleet total,
#: deliberately not built - see ``local_operator/harness/group_reaper.py`` on
#: wedged ``flock`` holders propagating a freeze between sessions, which is not a
#: hazard worth importing into every ``pytest`` startup.
#:
#: SCALED, not flat: a flat 2 GB would reserve most of a 4 GB CI container.
#: ``total // 18`` gives 227 MB on a 4 GB runner and 2,048 MB on this 36 GB
#: laptop, where the 2048 cap then binds. Traced on the runners that matter, all
#: unchanged: 2 vCPU/4 GB -> 2, 2 vCPU/8 GB -> 2, 4 vCPU/16 GB -> 4,
#: 8 vCPU/32 GB -> 8. The memory arm and the 2..8 clamp apply on CI (see the
#: module docstring), so the reserve applies there too - intentionally, because a
#: runner that runs out of memory fails exactly the way a laptop does, and the
#: scaling is what makes that safe.
#:
#: WHY IT MOVED FROM min(3072, total // 8) (2026-09-14): on this host total // 8
#: exceeds the cap, so the suite held a flat 3,072 MB per process out of a budget
#: that is 50% of *available* memory - and with `available` chronically at
#: 4,500-5,500 MB the reserve was binding on every single run (the crossover is
#: 2x the reserve = 6,144 MB, and this host lives below it). That is how 5 GB of
#: free memory became `min(2,570, 2,068)` = 2,068 MB = 3 workers. 2,048 MB still
#: holds a real floor out of the budget - 18 GB of this host's memory cannot be
#: claimed by a suite, and a CI container keeps the same proportional protection
#: it had - while letting the share be the binding term in the regime the fleet
#: actually runs in. See the titration in the module docstring: with this reserve
#: plus `_MB_PER_WORKER = 400`, the measured 5,313 MB of available memory
#: resolves 6 workers, which is the count that measured 28-40% faster.
#:
#: SOFTER THAN IT READS on macOS: ``_available_memory_mb`` counts file-backed
#: page cache, which the kernel would evict under pressure anyway, so this
#: reserves some memory that was never really at risk.
_MEMORY_RESERVE_CAP_MB = 2048
_MEMORY_RESERVE_FRACTION = 18

#: Hard bounds. Below 2 the suite stops being parallel at all (and a one-worker
#: xdist run is strictly worse than ``-n0``); above 8 buys nothing measurable on
#: a wait-bound suite and is precisely what produced the load-128 thrash.
_MIN_WORKERS = 2
_MAX_WORKERS = 8

#: Safe answer for any path that cannot measure the machine. Small enough to be
#: harmless on a laptop already under load, large enough to keep the suite
#: parallel.
_FALLBACK_WORKERS = 4

# ---------------------------------------------------------------------------
# The live fleet (REPORTED, never used to budget)
# ---------------------------------------------------------------------------
# Everything below answers one question, once per ``pytest`` process: how many
# OTHER pytest suites are running on this machine right now? The answer goes in
# the one-line decision report (see `_report_worker_cap`) and NOWHERE else - the
# divisor that would have used it to shrink this suite's cap was implemented,
# measured and rejected (module docstring). It is a READ of the machine, never a
# claim on shared state, and it must never be able to break a run: every failure
# path returns ``None``, i.e. "unknown", which is reported as unknown rather than
# as a comforting zero.

#: Sibling suites as first observed: a count, or ``None`` when the probe failed.
#: Cached for the process lifetime because the hook is consulted once, right
#: before workers spawn, and a fleet's depth moves on a scale of minutes; a second
#: probe could only disagree with the first about a suite that is still starting
#: up. The cache is also what makes the visibility line cheap enough to leave on
#: by default.
_SIBLING_SUITES: int | None = None

#: Distinguishes "not probed yet" from "probed, unknown" - `None` alone cannot,
#: and conflating the two would report a failed probe as zero siblings.
_SIBLING_PROBED = False

#: Set this to suppress the one-line decision report (see
#: `_report_worker_cap`). Cheap enough to be worth having: a CI log or a
#: transcript being parsed for output wants to opt out, and a one-line rule is
#: easier to trust than a "comment it out" convention.
_QUIET_ENV = "PYTEST_QUIET_WORKER_CAP"

#: An interpreter name as ``ps`` renders it: ``python``, ``python3``,
#: ``python3.12``, ``python3.12t``. Used to separate a pytest run from a shell
#: whose command line merely MENTIONS pytest - the distinction matters because a
#: ``bash -c "... python -m pytest ... | tail"`` wrapper is not a suite, and
#: treating it as one would make its child look like a nested run.
_PYTHON_EXECUTABLE = re.compile(r"python[0-9.]*[a-z]?")

#: ``env`` flags that consume the following token. Needed because the documented
#: way to run a TUI test here starts ``env -u NO_COLOR TERM=xterm-256color
#: .venv/bin/python -m pytest``, and the executable is then two to four tokens in.
_ENV_VALUE_FLAGS = frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"})

#: A controller's own request for xdist workers, or for none. ``-n0`` and
#: ``-n 0`` mean "serialise for a debugger": that run occupies one process and
#: claims no worker share, so it must not shrink every other suite on the box.
_NUMPROCESSES = re.compile(r"(?:^|\s)(?:-n|--numprocesses)(?:=|\s*)([^\s]+)")

#: What an xdist worker looks like in ``ps``. xdist spawns each worker through
#: execnet as ``python -u -c "import sys; exec(eval(sys.stdin.readline()))"`` -
#: matched on the bootstrap rather than on the word "execnet" because that word
#: does not appear in the command line at all (measured on macOS and, for the
#: procps form of the same call, on Linux).
_XDIST_WORKER = re.compile(r"execnet|exec\(eval\(sys\.stdin\.readline\(\)\)\)")


def _is_pytest_controller(command: str) -> bool:
    """Is ``command``, as ``ps`` renders it, a pytest CONTROLLER?

    Two shapes count: an interpreter invoked with ``-m pytest``, and a ``pytest``
    executable. Both are matched on the EXECUTABLE position, after skipping a
    leading ``env`` and its arguments, because the string  ``pytest`` appears as
    an ARGUMENT in every shell wrapper that launches a suite here
    (``bash -c "cd X && ... python -m pytest ... | tail"``). Matching the string
    anywhere would make those wrappers controllers, and the real controller a
    "nested" run of its own wrapper - which would hide every suite on the box.
    """
    tokens = command.split()
    index = 0
    if tokens and tokens[0].rsplit("/", 1)[-1] == "env":
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if "=" in token and not token.startswith("-"):
                index += 1  # NAME=value
            elif token in _ENV_VALUE_FLAGS:
                index += 2  # the flag and the argument it consumes
            elif token.startswith("-"):
                index += 1  # a valueless flag such as -i
            else:
                break
    if index >= len(tokens):
        return False
    executable = tokens[index].rsplit("/", 1)[-1]
    rest = tokens[index + 1 :]
    if _PYTHON_EXECUTABLE.fullmatch(executable):
        return any(
            token == "-m" and rest[position + 1] == "pytest"
            for position, token in enumerate(rest[:-1])
        )
    return executable == "pytest"


def _asks_for_serial_run(command: str) -> bool:
    """Does this controller's own command line ask for a serial (``-n0``) run?

    Only the explicitly-serialised form is recognised. A run that says ``auto``
    or ``N >= 2`` is a fleet member whether or not it has spawned its workers
    yet, and that matters because the hook runs 1-3 s into the process: without
    this arm a burst of suites launched together would each REPORT a fleet of
    one, which is exactly the reading a person cannot correct by hand later.
    A ``-n0`` peer is excluded on the other side because it occupies one process
    rather than a share of the machine, and the measured fleet is full of them
    (a QA session running three directories at ``-n0`` is one of the loads that
    prompted this work).
    """
    match = _NUMPROCESSES.search(command)
    return match is not None and match.group(1) == "0"


def _count_live_sibling_suites() -> int | None:
    """How many OTHER pytest suites are live on this machine, ``None`` if unknown.

    One read-only ``ps``, one pass over its output, no lock, no file, no shared
    state: a probe that cannot block and cannot wedge another session.

    A process counts when ALL of these hold:

    1. Its command line is a pytest controller (see `_is_pytest_controller`).
    2. It is not this process.
    3. No ancestor of it is also a controller. This is what excludes the nested
       ``pytest`` runs the suite itself spawns (they are descendants of our own
       controller, through its workers), and it does so as a PROPERTY rather
       than as a guess about age - the suggested shape for this probe was "ignore
       controllers younger than ~15 s", which would also discard the genuine peer
       started 5 s ago, and it would discard it for the whole run because the
       answer is cached. A nested run that double-forks and loses its ancestry is
       counted, which over-reports the fleet rather than flattering it.
    4. It either has an xdist worker child already, or its command line does not
       ask for a serial run. The second half is the burst case explained in
       `_asks_for_serial_run`.

    Any failure - no ``ps``, unparsable output, a timeout, an exception anywhere
    - returns ``None``, and the report then says ``siblings unknown``. It is
    never rounded to zero: a fleet the probe could not count is the last thing
    that should look like an empty machine.
    """
    try:
        dump = subprocess.run(
            # `-w -w` for unlimited width: macOS truncates `command` to the
            # window width otherwise, and a worker's bootstrap is ~88 characters
            # in, so a truncated line would look like a controller with no
            # workers. `pid=,ppid=,command=` suppresses the header, and the form
            # is accepted by both BSD and procps `ps`.
            ["ps", "-A", "-w", "-w", "-o", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except Exception:
        return None

    if not dump.strip():
        # `ps -A` always lists this process at least, so an empty table is a
        # broken probe rather than an empty machine. Reporting 0 here would be a
        # lie in exactly the direction that wastes a session's time.
        return None

    try:
        parents: dict[int, int] = {}
        commands: dict[int, str] = {}
        for line in dump.splitlines():
            pid_text, ppid_text, command = line.split(None, 2)
            parents[int(pid_text)] = int(ppid_text)
            commands[int(pid_text)] = command
        me = os.getpid()
        controllers = {pid for pid, command in commands.items() if _is_pytest_controller(command)}
        children: dict[int, list[int]] = {}
        for pid, ppid in parents.items():
            children.setdefault(ppid, []).append(pid)

        def _is_nested(pid: int) -> bool:
            seen = {pid}
            parent = parents.get(pid)
            while parent is not None and parent not in seen and parent > 1:
                seen.add(parent)
                if parent in controllers:
                    return True
                parent = parents.get(parent)
            return False

        def _has_worker_child(pid: int) -> bool:
            return any(
                _XDIST_WORKER.search(commands.get(child, "")) for child in children.get(pid, ())
            )

        return sum(
            1
            for pid in controllers
            if pid != me
            and not _is_nested(pid)
            and (_has_worker_child(pid) or not _asks_for_serial_run(commands[pid]))
        )
    except Exception:
        return None


def _live_sibling_suites() -> int | None:
    """Sibling suites for this process, probed once and cached."""
    global _SIBLING_SUITES, _SIBLING_PROBED
    if not _SIBLING_PROBED:
        _SIBLING_SUITES = _count_live_sibling_suites()
        _SIBLING_PROBED = True
    return _SIBLING_SUITES


def _report_worker_cap(
    *,
    workers: int,
    bound: str,
    cpu_arm: int,
    memory_arm: int | None,
    available_mb: int | None,
    reserve_mb: int | None,
    sibling_count: int | None,
    on_ci: bool,
) -> None:
    """Print ONE stderr line saying how this run's worker count was decided.

    WHY: the number is a product of four inputs the process never shows anyone,
    and the failure mode of that opacity is expensive in a way this repo has
    already paid for - AGENTS.md records a session bisecting a
    parallelism-sensitive failure without knowing what parallelism the run
    actually had, and a cap of 3 on a 14-core machine looks like a bug to the
    reader who has to guess which term produced it. This line also records the
    fleet size, which is the one number that has to be gathered by hand
    otherwise (a live `ps` audit is how the fleet evidence behind this change was
    built).

    One line, on stderr so it cannot be confused with test output (``-q`` parses
    stdout), and never more than one. Suppressed by ``PYTEST_QUIET_WORKER_CAP``.
    Wrapped so that a broken stream cannot decide a worker count: the caller has
    already computed `workers`, and this function is only allowed to describe it.
    """
    if os.environ.get(_QUIET_ENV):
        return
    try:
        memory = "unmeasurable" if memory_arm is None else str(memory_arm)
        available = "unknown" if available_mb is None else f"{available_mb:,} MB"
        reserve = "not applied" if reserve_mb is None else f"{reserve_mb:,} MB"
        siblings = "unknown" if sibling_count is None else str(sibling_count)
        ci = ", CI" if on_ci else ""
        print(
            f"pytest worker cap: {workers} "
            f"(bound by {bound}, cpu arm {cpu_arm}, memory arm {memory}, "
            f"available {available}, reserve {reserve}, siblings {siblings}{ci})",
            file=sys.stderr,
        )
    except Exception:  # a diagnostic must never decide anything
        return


def _total_memory_mb() -> int | None:
    """Physical RAM in MB, or ``None`` when it cannot be measured.

    Used only to SCALE the reserve, so an unmeasurable host degrades to no
    reserve at all rather than to a guess - the pre-existing budget, which is
    the behaviour this file shipped with. ``os.sysconf`` answers this on both
    macOS and Linux without a subprocess, unlike the ``vm_stat`` probe below;
    ``psutil`` remains deliberately not a dependency.
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


def _available_memory_mb() -> int | None:
    """Memory the suite can take without pushing the machine into swap, in MB.

    ``None`` when it cannot be measured; the caller then degrades to the
    CPU-only cap rather than guessing. ``psutil`` would answer this in one call
    and is intentionally NOT a dependency, so each platform is probed through
    what ships with the OS.

    The macOS arm is ``free + speculative + file-backed``, and each of those two
    design choices was made against a measured failure of the obvious
    alternative. Both alternatives are recorded because each looks correct until
    it is measured, and a future reader will otherwise re-introduce one:

    * **Not ``Pages inactive``.** Counting it as available reported 8,137 MB of
      headroom on this host at the exact moment it had 452 MB genuinely free and
      6.1 GB of 7.2 GB of swap consumed, so the arm never bound under the
      pressure it exists to detect. On macOS ``inactive`` is not Linux's
      ``MemAvailable``: much of it is dirty and compressor-backed, reclaimable
      only by paging, which is the cost being avoided. ``File-backed pages`` is
      the subset vm_stat itself identifies as clean and cheaply reclaimable
      (page cache, backed by a file, droppable without a write), so it needs no
      invented discount fraction - an earlier revision multiplied ``inactive``
      by an asserted 0.25, which was a guess dressed as a measurement.
    * **Not consumed swap as a pressure term.** Subtracting
      ``vm.swapusage``'s ``used`` looks like the natural way to notice a machine
      that is paging, but that counter is CUMULATIVE - macOS does not decrement
      it when pressure clears, since pages stay in the swap file until faulted
      back in or the machine reboots. It therefore reads "this host swapped at
      some point since boot", not "this host is swapping now". Measured: with
      the OS reporting 78% free and load down from 155 to 21, a stale 3,315 MB
      swap figure still drove the hook to 2 workers, the floor, where the CPU
      arm alone would have given 7. A term that only ever ratchets down is worse
      than no term, because it silently makes the cap independent of actual
      conditions. The page counts used here are all instantaneous and recover on
      their own.

    The compressor is deliberately NOT subtracted: pages it occupies are already
    excluded from both free and file-backed, so subtracting would double-count.
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
            # "File-backed pages" is not present on every macOS version; treat a
            # miss as 0 rather than as a probe failure, which degrades to the
            # free-page estimate instead of discarding a usable measurement.
            for label in ("Pages free", "Pages speculative"):
                match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.", out, re.MULTILINE)
                if match is None:
                    return None
                counts[label] = int(match.group(1))
            file_backed = re.search(r"^File-backed pages:\s+(\d+)\.", out, re.MULTILINE)
            counts["File-backed pages"] = int(file_backed.group(1)) if file_backed else 0

            per_mb = page_size / (1024 * 1024)
            return max(0, int(sum(counts.values()) * per_mb))

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
        # `CI` is set by GitHub Actions and essentially every other provider,
        # so it stays the signal - EXCEPT for the one liar we control. Our own
        # bash tool injects `CI=1` into every agent-run command to make CLIs
        # non-interactive, which told this hook "dedicated runner, take all 14
        # cores" on the shared laptop the share exists to protect, and only
        # there. Denying that marker keeps every real provider at full
        # parallelism; see `_AGENT_SHELL_ENV` for why this is not an allowlist.
        on_ci = bool(os.environ.get("CI")) and not os.environ.get(_AGENT_SHELL_ENV)

        cpu_arm = cpus if on_ci else max(1, int(cpus * _CPU_SHARE))
        cap = cpu_arm

        available_mb = _available_memory_mb()
        memory_arm: int | None = None
        reserve_mb: int | None = None
        if available_mb is not None:
            # Budget from AVAILABLE memory, so a machine already hosting three
            # sibling suites hands this run a smaller cap automatically.
            budget_mb = available_mb * _MEMORY_SHARE
            total_mb = _total_memory_mb()
            if total_mb is not None:
                # Hold a floor out of the budget entirely, so the fraction is
                # not the only thing standing between six sibling suites and
                # zero free memory. `min` rather than subtract-then-halve so a
                # solo run is not charged twice for the same memory.
                #
                # `max(0, ...)` keeps the intermediate meaningful when free
                # memory is below the reserve and the subtraction goes
                # negative. It is NOT what guarantees a usable worker count -
                # the `max(_MIN_WORKERS, ...)` clamp on the return does that,
                # and would do it from a negative budget too. Kept because a
                # negative "budget" is nonsense to read in a debugger and
                # invites a later reader to divide by it somewhere new.
                reserve_mb = min(_MEMORY_RESERVE_CAP_MB, total_mb // _MEMORY_RESERVE_FRACTION)
                budget_mb = max(0, min(budget_mb, available_mb - reserve_mb))
            memory_arm = int(budget_mb) // _MB_PER_WORKER
            cap = min(cap, memory_arm)

        # Which arm decided the number, for the report only. Reported rather than
        # logged at debug level because "3 workers on 14 cores" is unactionable
        # without it: the reader needs to know whether to look at the share, the
        # reserve, or the host's memory, and a floor or a clamp can mask all of
        # them. (This is how the 2026-09 investigation found that the memory arm
        # binds at ~5,140 MB available - 2,068 MB of budget = 3 workers - while
        # the machine sat at 0-2.5% CPU.)
        raw = cpu_arm if memory_arm is None else min(cpu_arm, memory_arm)
        if raw < _MIN_WORKERS:
            bound = "the 2-worker floor"
        elif raw > _MAX_WORKERS:
            bound = "the 8-worker cap"
        elif memory_arm is not None and memory_arm <= cpu_arm:
            bound = "memory"
        else:
            bound = "cpu"

        workers = max(_MIN_WORKERS, min(_MAX_WORKERS, cap))
        _report_worker_cap(
            workers=workers,
            bound=bound,
            cpu_arm=cpu_arm,
            memory_arm=memory_arm,
            available_mb=available_mb,
            reserve_mb=reserve_mb,
            sibling_count=_live_sibling_suites(),
            on_ci=on_ci,
        )
        return workers
    except Exception:  # see docstring: never break the run
        return _FALLBACK_WORKERS
