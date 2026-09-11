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
  (``min(3072, total // 8)`` MB). It exists because the fraction claims a share
  of what REMAINS, so sibling suites converge toward zero free memory rather
  than toward a floor. Note its real reach before tuning it: because the shape
  is ``min(share, available - reserve)``, it binds only below twice itself and
  is a floor under one suite's appetite, not a cap on the fleet total. Six
  simultaneous suites are still not bounded by it.
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
#: subset that spawns no subprocesses of its own) sit at 226-262 MB. 600 is a
#: ~2.5x margin over that, and the margin is intentional on three counts: a
#: worker's RSS depends on which tests it draws (the worst observed single
#: worker was ~1,090 MB), some suites fork their own children that the budget
#: still has to cover (the eval-tool tests spawn kernel subprocesses), and the
#: controller's own footprint is charged to no worker. The asymmetry justifies
#: it: under-provisioning costs a little wall time, over-provisioning costs the
#: whole machine a swap storm.
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
#: terms to the same memory, costing ~2.6 workers even when this suite is the
#: ONLY one running: a solo developer at 6 GB free would drop 5 workers to 2.
#: The ``min`` form expresses "leave the reserve free" without that penalty.
#:
#: THE CONSEQUENCE, which is the single most useful thing to know before
#: tuning these constants: ``min(a * 0.5, a - reserve) == a * 0.5`` for all
#: ``a >= 2 * reserve``. So this term BINDS ONLY BELOW ~2x itself (6,144 MB
#: here) and is invisible above that. It is a floor under a single suite's
#: appetite when memory is genuinely short - NOT a fleet-total lever. An
#: earlier draft claimed a ~3.1 GB fleet reclaim, but that reclaim came
#: entirely FROM the double-charge removed above; the two corrections cancel.
#:
#: DELIBERATELY NOT CALLED AN ASYMPTOTE, because it is not one.
#: ``_MIN_WORKERS`` keeps charging a flat ~2-worker tax once the memory arm is
#: exhausted, so a deep enough fleet still walks to zero; the reserve
#: DISPLACES that point rather than preventing it. Modelled at 395 MB/worker
#: from 12.4 GB available, the trough improves by ~395 MB at every fleet depth
#: (n=6: 1,735 -> 2,130 MB) and n=10 still goes negative. Six simultaneous
#: suites remain genuinely unbounded by this term; the durable lever for that
#: would be a cross-process budget, deliberately not built - see
#: ``local_operator/harness/group_reaper.py`` on wedged ``flock`` holders
#: propagating a freeze between sessions, which is not a hazard worth
#: importing into every ``pytest`` startup.
#:
#: SCALED, not flat: a flat 3 GB would reserve three quarters of a 4 GB CI
#: container. ``total // 8`` gives 512 MB on a 4 GB runner and 4.5 GB on this
#: 36 GB laptop, where the 3072 cap then binds. Traced on the runners that
#: matter, all unchanged from today: 2 vCPU/4 GB -> 2, 2 vCPU/8 GB -> 2,
#: 4 vCPU/16 GB -> 4, 8 vCPU/32 GB -> 8. The memory arm and the 2..8 clamp
#: apply on CI (see the module docstring), so the reserve applies there too -
#: intentionally, because a runner that runs out of memory fails exactly the
#: way a laptop does, and the scaling is what makes that safe.
#:
#: SOFTER THAN IT READS on macOS: ``_available_memory_mb`` counts file-backed
#: page cache, which the kernel would evict under pressure anyway, so this
#: reserves some memory that was never really at risk.
_MEMORY_RESERVE_CAP_MB = 3072
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
        cap = cpus if on_ci else max(1, int(cpus * _CPU_SHARE))

        available_mb = _available_memory_mb()
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
            cap = min(cap, int(budget_mb) // _MB_PER_WORKER)

        return max(_MIN_WORKERS, min(_MAX_WORKERS, cap))
    except Exception:  # see docstring: never break the run
        return _FALLBACK_WORKERS
