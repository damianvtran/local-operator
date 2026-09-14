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

    pytest worker cap: 4 (bound by memory, cpu arm 7, memory arm 4,
    available 5,313 MB, reserve 2,048 MB, siblings 11)

That line, and the read-only sibling probe it uses, are part of the change from
that investigation. The count is computed as it always was; of the constants
below, only the reserve cap moved, and it moved because the A/B says the
reserve was the term doing the binding (see THE TITRATION and
``_MEMORY_RESERVE_CAP_MB``).

THE TITRATION
-------------
If the cap is what binds, the next question is which count this slice actually
wants - because "3 workers" on a 14-core box at 0-2.5% CPU is a memory verdict,
not a compute one. Measured 2026-09-13/14 with interleaved waves of 3 concurrent
instances of ``tests/unit/tui/test_slash_echo.py`` (71 tests, ~58 Textual app
boots, the boot-bound shape the whole suite is 82% of), staged 25 s apart, at
real fleet depth - 11-13 sibling suites live throughout. Per-instance wall time,
and the three instances' total CPU:

* cap 3 (what this host resolved): 313.3 s, 225.2 s over two rounds; 118.4 s and
  107.8 s of CPU.
* cap 6: 188.9 s, 161.6 s, 188.7 s over three rounds; 129.2 s, 123.9 s, 129.5 s.
  **28-40% faster per instance for ~12% more CPU**, in that run.
* cap 8: 171.3 s (one paired round); 150.2 s of CPU. 9% faster than cap 6 for
  16% more CPU - it does NOT clear the >=15% bar that would justify a wider run,
  so the 2..8 clamp and the 0.5 CPU share are untouched.

**The DIRECTION is what reproduces; the MAGNITUDE is host-dependent, and the
smaller figure is the one to plan with.** An independent A/B (review round 1,
2026-09-14) ran fresh interleaved waves on a host carrying 24-36 sibling pytest
processes. It reproduced the ordering - cap 6 was never worse than cap 3, pooled
per-instance medians 139.3 s against 165.9 s - but at **15-16%, not 28-40%**, and
against a different baseline (its cap-3 arm measured 157-182 s where the waves
above measured 225-313 s). Absolute wall times taken minutes apart on a box with
a live fleet of unknown depth are not comparable, which is exactly why the
28-40% range was not reproducible and is gone from the docs. What both runs
agree on, and all that is claimed here: 6 beats 3 on this host, and the CPU the
raised arm spends is real - ~11% more in the waves above, ~33% more in the
independent one. The mechanism is the reason to believe it at all: the suite is
wait-bound, so extra workers hide latency until the machine stops absorbing
them.

**What ships is a 3 -> 4 raise on this host, not a 3 -> 6 one.** The titration
above shows the OLD resolution was too tight and that more workers pay on this
wait-bound suite; it does not measure 4. An earlier revision bought the count 6
by halving ``_MB_PER_WORKER``, which the per-worker RSS measurement then
refused - the charge would have sat BELOW the peak tree it has to cover - so the
whole raise comes from the reserve cap instead and lands one worker higher than
before. Stated plainly because the smaller step is the honest one: the
measurement supports 6 workers being faster than 3, and it supports the reserve
cap having been the binding term; it does not support a 400 MB charge, so the
charge did not move.

Swap and free memory are NOT the discriminator, and are no longer quoted as one.
``vm.swapusage``'s counter moves GIGABYTES in either direction on its own under
this fleet - the same file calls that counter unusable as a pressure signal
below, and a paragraph quoting ``+480`` / ``-2,173`` MB deltas next to it cannot
stand. So the deltas are dropped, and what is claimed is only what the arms
actually showed: no wave collapsed free memory (the lowest reading, 17 MB,
recovered within the next wave), and in the independent run the raised arm's
free-memory minima were not lower than the current arm's (4,499 / 4,595 MB at cap
6 against 4,518 / 4,160 MB at cap 3). CPU time and the free-memory minima carry
the decision; the raised arm is bounded, not proven harmless.

At this host's chronic 4,500-5,500 MB of available memory the resolution is
therefore **4** (5,313 MB buys ``min(2,656, 3,265) // 600``), up from the 3 the
released reserve cap produced. Only the reserve CAP moved (3,072 -> 2,048 MB;
the 1/8 fraction is unchanged, so every small-host floor is exactly what it was);
``_MB_PER_WORKER`` stays at 600 because the measurement says it must. See their
own comments, and `_MB_PER_WORKER`'s for what bounds the ~1,090 MB outlier.

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
  (``min(2048, total // 8)`` MB). It exists because the fraction claims a share
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


#: Marker attribute set on every wrapper :func:`_guarded` installs. It is what
#: makes ``_install_real_store_guard`` idempotent: this module is executed twice
#: in one process (pytest loads the root conftest as a plugin, and
#: ``tests/unit/test_xdist_worker_budget.py`` imports a private copy to reach the
#: hook), and an unconditional install wrapped the PREVIOUS wrapper, so every
#: later ``os.remove``/``unlink``/``rmtree`` in that worker ran N stacked
#: ``Path.resolve()`` checks - N reaching ~28 before this marker existed, on a
#: suite whose subject is wall time.
_GUARD_MARKER = "_local_operator_real_store_guard"


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

    setattr(wrapper, _GUARD_MARKER, True)
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
    if getattr(shutil.rmtree, _GUARD_MARKER, False):
        # The wrappers are already in place, installed by an earlier execution of
        # this module in this process. Installing again would only add a layer:
        # ``_guarded`` closes over _REAL_STORE and the real store cannot have
        # moved within the process, so the layer already there is the whole
        # guard. This early return is why the ~28 module loads the worker-budget
        # tests perform cost 28 snapshot reads and zero extra wrapper layers.
        return
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
#: subset that spawns no subprocesses of its own) sit at 226-262 MB, and a
#: worker's RSS depends on which tests it draws, so the constant has to cover a
#: draw, not an average. Measured 2026-09-14 (review round 1) on the suite's
#: known heavy corner - ``tests/unit/tools/test_eval_tool.py`` (real kernel
#: subprocesses), ``tests/unit/evaluation/adapters/osworld`` (builds the shipped
#: adapter wheel once, then a real copied interpreter per spawn),
#: ``runner/test_episode_subprocess.py``, ``adapters/test_supervisor.py`` and the
#: boot-bound ``tests/unit/tui/test_slash_echo.py``, 685 tests at ``-n 6``, 120 s,
#: sampling every worker's whole process tree with ``ps`` every 1.5 s:
#:
#:   * peak per-worker process tree: **441.5 MB** (269.9 MB of it the worker
#:     itself, 171.6 MB its descendants); the next two peaked at 405.3 and
#:     360.7 MB. Peak per-worker RSS alone was 302.5 MB.
#:   * peak aggregate across all 6 workers: **1,647.3 MB**, including the
#:     children and app processes the suite spawns.
#:   * a narrower run of the same spawn-heavy files without the TUI slice peaked
#:     at 284 MB per worker and 1,240 MB aggregate, so the TUI slice is what
#:     moves the per-worker peak.
#:
#: 600 is therefore **1.36x over the measured peak tree** (600 / 441.5), which is
#: the margin to reason about - NOT the 2.5x a "226-262 MB clean worker" figure
#: suggests, because that figure excludes exactly the draws this constant has to
#: survive.
#:
#: WHY IT IS 600 AND NOT 400 (review round 1). An earlier revision of this change
#: lowered it to 400 on evidence that measured a 137 MB/worker average from ONE
#: boot-bound 71-test file - a slice lighter than the 226-262 MB the envelope was
#: built on - with the A/B arms pinning the worker count through
#: ``PYTEST_XDIST_AUTO_NUM_WORKERS``, so neither this constant nor the reserve was
#: exercised by the measurement at all. 400 is BELOW the peak this constant has
#: to cover (441.5 MB; two further workers peaked above 400 as well), i.e. it
#: stops bounding the common case and starts under-charging for it. The raise to
#: a wider run is now taken from the reserve cap alone (see
#: `_MEMORY_RESERVE_CAP_MB`), which is the term that was actually binding on this
#: host, and this constant is left where it was.
#:
#: THE ~1,090 MB OUTLIER, and what protects the host when it is drawn. This
#: constant does not cover it and never did; what bounds the run is the aggregate.
#: A count of N is only ever produced on a host with **at least 1,200 x N MB of
#: available memory** - the budget is ``min(0.5*available, available - reserve)``,
#: so ``budget >= 600N`` forces ``available >= 1,200N`` - which is >= 9,600 MB at
#: the 2..8 clamp's ceiling of 8 workers. A worker drawing the 1,090 MB case at 8
#: workers puts ~1,090 + 7 x 250 MB of worker RSS on a host that had >= 9.6 GB
#: free when it chose 8, and the memory arm is recomputed per run from *current*
#: pressure, so a shrinking host hands out fewer workers instead of more. What is
#: NOT claimed is that the charge covers that case.
#:
#: The asymmetry is what keeps the envelope where it is: under-provisioning costs
#: wall time on a wait-bound suite, over-provisioning costs the whole machine a
#: swap storm.
_MB_PER_WORKER = 600

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
#: ``total // 8`` gives 512 MB on a 4 GB runner, 1,024 MB on an 8 GB one and
#: 2,048 MB on this 36 GB laptop, where the 2048 cap then binds. Traced on the
#: runners that matter, all unchanged: 2 vCPU/4 GB -> 2, 2 vCPU/8 GB -> 2,
#: 4 vCPU/16 GB -> 4, 8 vCPU/32 GB -> 8. The memory arm and the 2..8 clamp apply
#: on CI (see the module docstring), so the reserve applies there too -
#: intentionally, because a runner that runs out of memory fails exactly the way
#: a laptop does, and the scaling is what makes that safe.
#:
#: WHY THE FRACTION IS 8 AND NOT 18 (review round 1). An earlier revision of this
#: change cut the fraction to 1/18, which silently cut the anti-swap floor on the
#: hosts with the least headroom by ~2.25x - 4 GB: 512 -> 227 MB, 8 GB: 1,024 ->
#: 455 MB, 16 GB: 2,048 -> 910 MB - while ``min(2048, ...)`` held this laptop's own
#: floor at 2,048 MB. Two constants moving in the same direction compounded on
#: exactly the hosts that cannot absorb it, and no traced host showed it, because
#: the four documented CI shapes above are CPU-bound and the reserve's net effect
#: on them is ZERO by construction: their worker count is set by the CPU arm
#: either way, so the trace cannot see a reserve change at all. 1/8 restores every
#: small-host floor to exactly what it was, and this host still resolves 2,048 MB,
#: so the titration below is untouched. What actually changed versus the released
#: file is the CAP only (3,072 -> 2,048 MB), which binds wherever ``total // 8``
#: exceeds it, i.e. on hosts with more than 16 GB of RAM; the fraction - and
#: therefore the proportional protection a CI container gets - is the same 1/8 it
#: always was.
#:
#: WHY THE CAP MOVED FROM 3,072 MB (2026-09-14): on this host total // 8 exceeds
#: the cap either way, so the suite held a flat 3,072 MB per process out of a
#: budget that is 50% of *available* memory - and with `available` chronically at
#: 4,500-5,500 MB the reserve was binding on every single run (the crossover is
#: 2x the reserve = 6,144 MB, and this host lives below it). That is how 5 GB of
#: free memory became `min(2,570, 2,068)` = 2,068 MB = 3 workers. 2,048 MB still
#: holds a real floor out of the budget - it is 1/18 of this host's 36 GB, i.e.
#: about 2 GB, NOT the "18 GB" an earlier revision of this comment claimed (the
#: 2,048 in ``min(2048, total // 8)`` is a megabyte figure, and 2,048 MB is 1/18
#: OF 36 GB; reading it as "18 GB" was a 9x overstatement) - while letting the
#: share be the binding term in the regime the fleet actually runs in.
#:
#: THE RESOLUTION IT BUYS, stated so nobody over-claims it: with the cap at 2,048
#: MB and ``_MB_PER_WORKER`` left at 600, the measured 5,313 MB of available
#: memory resolves **4** workers (``min(2,656, 3,265) // 600``), where the
#: released cap of 3,072 gave 3. One worker, from this term alone - the term that
#: was binding on every run in this regime. Getting to 6 needed the per-worker
#: charge to drop too, and the per-worker RSS measurement refused that (see
#: `_MB_PER_WORKER`: 400 would sit below the measured 441.5 MB peak tree).
#:
#: SOFTER THAN IT READS on macOS: ``_available_memory_mb`` counts file-backed
#: page cache, which the kernel would evict under pressure anyway, so this
#: reserves some memory that was never really at risk.
_MEMORY_RESERVE_CAP_MB = 2048
_MEMORY_RESERVE_FRACTION = 8

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
#:
#: Read as a flag, not as "set at all": ``PYTEST_QUIET_WORKER_CAP=0`` (or
#: ``false``/``no``/``off``, any case) leaves the line ON, matching the
#: documented ``=1`` and every other env flag in this tree. The earlier
#: any-non-empty test made ``=0`` silence it, which is the one value a reader is
#: most likely to reach for when they want the opposite.
_QUIET_ENV = "PYTEST_QUIET_WORKER_CAP"

#: Values of an env flag that mean "not set". A flag is ON for any other
#: non-empty string, so `PYTEST_QUIET_WORKER_CAP=true` still silences.
_FALSY_ENV_VALUES = frozenset({"0", "false", "no", "off"})


#: Env var that replaces the whole calculation with a typed-in number
#: (documented in AGENTS.md, honoured unclamped).
_OVERRIDE_ENV = "PYTEST_XDIST_AUTO_NUM_WORKERS"


def _env_flag(name: str) -> bool:
    """True when ``name`` holds a value that means ON (see `_FALSY_ENV_VALUES`)."""
    return os.environ.get(name, "").strip().lower() not in (_FALSY_ENV_VALUES | {""})


#: An interpreter name as ``ps`` renders it: ``python``, ``python3``,
#: ``python3.12``, ``python3.12t``. Used to separate a pytest run from a shell
#: whose command line merely MENTIONS pytest - the distinction matters because a
#: ``bash -c "... python -m pytest ... | tail"`` wrapper is not a suite, and
#: treating it as one would make its child look like a nested run.
#:
#: Case-sensitive on purpose, and that is a known false negative: a macOS
#: framework/Xcode interpreter renders its ``comm`` as ``Python3.12``, so a
#: controller running from one of those is invisible to the probe and the fleet
#: count is low by that many suites. Left as-is because the field is reporting
#: only - a low count never changes a worker cap, and widening the pattern to
#: ``re.IGNORECASE`` would also start matching shell aliases and wrapper names
#: that merely capitalise the word.
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

    COST, because it runs before any test in every suite: one ``ps -A -w -w``
    over a full process table, measured 64-430 ms here (median 90 ms across 15
    runs, 973 processes on the box at load ~140) and 0.42-1.36 s during review
    round 1's heavier stretch. So the figure to plan with is a few hundred ms on
    a loaded machine, NOT the sub-100 ms an idle host suggests - and it is why
    the answer is cached per process rather than recomputed, and why the probe
    failure path degrades to ``unknown`` instead of retrying.

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
    memory = "unmeasurable" if memory_arm is None else str(memory_arm)
    available = "unknown" if available_mb is None else f"{available_mb:,} MB"
    reserve = "not applied" if reserve_mb is None else f"{reserve_mb:,} MB"
    siblings = "unknown" if sibling_count is None else str(sibling_count)
    ci = ", CI" if on_ci else ""
    _emit_worker_cap_line(
        f"pytest worker cap: {workers} "
        f"(bound by {bound}, cpu arm {cpu_arm}, memory arm {memory}, "
        f"available {available}, reserve {reserve}, siblings {siblings}{ci})"
    )


def _report_worker_override(workers: int) -> None:
    """Report a cap chosen by the operator via ``PYTEST_XDIST_AUTO_NUM_WORKERS``.

    WHY this exists separately: the override returns before the calculation, so
    it used to print NOTHING - and the run it printed nothing on is the one where
    the number is hardest to explain, because the hook did not choose it. A
    silent 12-worker run is also indistinguishable from a ``-n0``/explicit-``-n``
    run, where the silence is honest (the hook genuinely never ran and has
    nothing to say). Naming the override as the binding term is the whole line:
    the arms, the reserve and the fleet have no say in this number, so reporting
    them here would misrepresent it.
    """
    _emit_worker_cap_line(f"pytest worker cap: {workers} (bound by {_OVERRIDE_ENV}, unclamped)")


def _emit_worker_cap_line(line: str) -> None:
    """Write `line` to stderr unless silenced. A broken stream changes nothing.

    The only place either report reaches the terminal, so the quiet flag and the
    never-raise guarantee are stated once and cannot drift apart between the two
    callers. The count is already computed by the time this runs; a diagnostic
    that raised here would be caught by the hook's outer ``except`` and turn a
    good resolution into ``_FALLBACK_WORKERS``.
    """
    if _env_flag(_QUIET_ENV):
        return
    try:
        print(line, file=sys.stderr)
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
        override = os.environ.get(_OVERRIDE_ENV)
        if override:
            try:
                # An explicit operator override is honoured UNCLAMPED. Someone who
                # types a number has a reason (a dedicated machine, or bisecting a
                # parallelism-sensitive failure at 1); second-guessing it would
                # make the escape hatch useless.
                workers = int(override)
            except ValueError:
                warnings.warn(
                    f"{_OVERRIDE_ENV} is not a number: {override!r}. Ignoring it.",
                    stacklevel=2,
                )
            else:
                # Reported, not silent: this is the one path where the count is
                # NOT this hook's decision, so a run that prints nothing is
                # indistinguishable from a `-n0`/explicit-`-n` run (where the
                # hook never ran at all) and the reader has no way to tell that a
                # 12-worker run on a 4-worker machine was typed in by hand.
                _report_worker_override(workers)
                return workers

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
