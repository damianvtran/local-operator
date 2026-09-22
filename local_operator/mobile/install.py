"""Install, supervise and introspect the mobile daemon.

One supervised unit re-runs THIS interpreter's package as ``python -m
local_operator.mobile.service`` — re-entering the installed code rather than
a hardcoded binary path means an upgrade (``lop-update``) changes what the
agent runs with no reinstall, and ``restart`` picks it up. That is the omp
mobile lesson applied to a Python entry point.

THREE SUPERVISORS, ONE SHAPE PER PLATFORM (see :mod:`local_operator.supervisors`):
a LaunchAgent plist on macOS, a ``systemd --user`` unit on Linux, and a Task
Scheduler task on Windows. The launchd half is unchanged and is the shape every
other arm is modelled on; the Linux and Windows arms exist because "the daemon
is portable, only the supervisor is macOS-specific" used to mean the relay could
not be installed AT ALL on two of the three platforms, and the CLI crashed
rather than saying so (``FileNotFoundError: launchctl``, ``AttributeError: os.getuid``).

The unit name is fixed (``com.local-operator.mobile`` / ``local-operator-mobile``
/ ``Local Operator Mobile``): it owns the port, so a second daemon cannot
split-brain the control plane — it fails to bind and exits loudly.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from local_operator import launchd, memory_guard, procname, procstate, supervisors
from local_operator.mobile.auth import (
    generate_password,
    load_password,
    store_description,
    store_password,
)
from local_operator.mobile.daemon import DEFAULT_PORT
from local_operator.paths import CONFIG_DIR_ENV, config_dir, log_dir

LABEL = "com.local-operator.mobile"

#: Linux user unit for this daemon. Deliberately NOT suffixed per config root:
#: the label owns the port on every platform, and a per-root unit name would let
#: two units fight over 4098 instead of one failing to bind loudly.
SYSTEMD_UNIT = "local-operator-mobile.service"

#: Task Scheduler task name for this daemon (Windows), same reasoning.
TASK_NAME = "Local Operator Mobile"

#: The refusal every entry point shares when no supervisor exists at all.
NO_SUPERVISOR_ERROR = supervisors.no_supervisor_error("lop mobile serve")

#: The SPA the daemon serves. ``web/dist`` is gitignored, so a source
#: checkout has no bundle until something builds it; a pip/uv wheel ships it
#: via package-data but an in-place source install never does. Install must
#: be able to make it, or every such machine shows "bundle not built".
#: Functions below take an optional ``web_dir`` for a tree that is not this
#: one — the snapshot updater builds the tree it is about to install.
_WEB_DIR = Path(__file__).parent / "web"

#: What a failed build's output is scanned for, and how much of it is quoted
#: back. A bound rather than a formality: the string lands in `lop mobile
#: install`'s step list and in whatever bug report copies it.
_BUILD_ERROR_HINT = re.compile(r"error|ERR_|cannot|not found|failed", re.IGNORECASE)
#: pnpm's own wrapper lines — the `$ <script>` echo it writes to stderr before
#: running a script, and the lifecycle/exit-code summary it writes after one
#: fails. "The last line of stderr" WAS one of these, which is how the operator
#: and every bug report saw `pnpm build failed: $ tsc -b && vite build` while
#: the compiler's actual reason sat unread one stream away.
_BUILD_WRAPPER_NOISE = re.compile(
    r"^(\$ |Progress: |WARN )|\[ELIFECYCLE\]|Command failed with exit code", re.IGNORECASE
)
_BUILD_DETAIL_LINES = 3
_BUILD_DETAIL_CHARS = 500

#: How long one bundle-build step (``pnpm install`` / ``pnpm build``) may run
#: before its whole process GROUP is signalled. The value this module always
#: used; what changed is how much of the tree the bound reaches.
_BUILD_STEP_TIMEOUT = 600.0

#: Seconds a signalled build group gets between SIGTERM and SIGKILL. Deliberately
#: short: the group this bounds is the pnpm self-install recursion documented on
#: :func:`_pin_mismatch`, which reached 3.2 GB of RSS in its first 8 seconds in
#: the reproduction of that incident — 0.2-0.4 GB/s across the measurements taken
#: here — so a graceful window sized for an ordinary build would cost gigabytes it
#: exists to save. A real pnpm build is never signalled on this path unless the
#: bound fired.
_BUILD_KILL_GRACE = 5.0

#: How long the ``--version`` probe in :func:`_runner_reports` may run. It is a
#: one-line answer from a runner that is working; a runner that has started
#: resolving a pin instead of answering does not finish (measured: past 120 s).
_PIN_PROBE_TIMEOUT = 20.0

#: How long to wait between samples while a signalled group dies.
_GROUP_POLL_SECONDS = 0.05

#: The floor under the computed MEMORY ceiling for one build step, in MB.
#: Judgement, not a calibration, and it is a SECOND floor rather than
#: ``memory_guard``'s own for a measured reason: that one (64 MB) is sized against
#: the smallest things the bash tool runs, and a package-manager child is two orders
#: of magnitude past it — measured on this host 2026-09-21, a bare ``node -e ''``
#: peaks at 11.6 MB while ``pnpm --version`` in an empty directory peaks at
#: **120.8 MB**. A ceiling below a real step's own interpreter would kill every
#: build on a pressured host and report it as "reduce peak memory", which is not a
#: thing the reader can do to a pnpm. 256 MB is ~2x that measurement, and still an
#: order of magnitude under the incident's growth (+5 GB per 25 s), so the kill
#: lands while the reserve is still holding the machine up.
#:
#: THE WHOLE-STEP PEAK IS DELIBERATELY NOT WHAT THIS WAS DOUBLED FROM, and that is
#: a stated limitation rather than a claim: no real ``pnpm install
#: --frozen-lockfile`` + ``vite build`` was measured, because running one is the
#: hazard this change exists to bound (agent review round 1, MINOR-3). What that
#: costs, said plainly: the floor binds whenever available falls below ≈2.05 GB on
#: a 36 GB host, and a legitimate step peaking past 256 MB on a host that tight is
#: killed and told to free memory. The alternative is worse in the same direction —
#: with no floor the reserve arithmetic can land on 0 and kill EVERY step on its
#: first tick — and a killed step is attributable and retryable, where the incident
#: was neither.
_STEP_MEMORY_FLOOR_MB = 256

#: Seconds a build group that has breached the MEMORY ceiling gets between SIGTERM
#: and SIGKILL. Deliberately NOT :data:`_BUILD_KILL_GRACE`: that window is priced for
#: a cooperative exit, and at the incident's measured rate (0.2-0.4 GB/s, see
#: :func:`_pin_mismatch`) five seconds spends up to 2 GB — the whole of the 2,048 MB
#: reserve the ceiling already held back, i.e. exactly the margin the kill exists to
#: keep. One group poll is ~20 MB at the worst measured rate, and a group that is
#: over budget has no cooperative exit left to make.
_BUILD_MEMORY_KILL_GRACE = _GROUP_POLL_SECONDS

#: The header a memory kill leads with. The SAME word the bounded bash command
#: reports with (``memory_guard``'s §5 contract): the failure is the same failure
#: whichever path hit it, and an operator greps for one string.
_STEP_MEMORY_HEADER = "MEMORY LIMIT EXCEEDED"

#: How long ``corepack enable`` may run before its group is signalled. Short by
#: design: it writes shims and prints a summary, and the thing it can wait on —
#: a network fetch of a package manager — is not something an install should sit
#: behind (see :func:`_corepack_enable`).
_COREPACK_ENABLE_TIMEOUT = 30.0

#: The setting pnpm's switch to a ``packageManager``-pinned version is gated on.
#: Set to ``false`` in the child environment of every package-manager command
#: this module runs, which makes the ``pnpm add pnpm@<pin>`` fetch UNREACHABLE
#: rather than merely unlikely (``switchCliVersion`` -> ``installPnpmToTools`` in
#: the shipped bundle). Defence in depth BEHIND the refusal, never a substitute
#: for it: on its own it does not fail closed (see :func:`_package_manager_env`).
_MANAGE_PM_VERSIONS_ENV = "npm_config_manage_package_manager_versions"

#: The two settings that turn a PIN MISMATCH into a failure instead of a warning:
#: pnpm's version check only runs at all when the first is set, and only throws
#: when the second is on. See :func:`_package_manager_env` for the measurements
#: that put them here — with the disarm alone a wrong pnpm builds the tree and
#: exits 0, which is the silent shape this guard exists to prevent.
_STRICT_PM_VERSION_ENV = "npm_config_package_manager_strict_version"
_STRICT_PM_ENV = "npm_config_package_manager_strict"

#: Corepack asks before it downloads a package manager, and a build child's stdin
#: is not a terminal — it can never answer. Off in every child this module spawns.
_COREPACK_DOWNLOAD_PROMPT_ENV = "COREPACK_ENABLE_DOWNLOAD_PROMPT"

#: Whether this platform can signal a whole process GROUP. ``os.killpg`` and
#: ``os.getpgid`` are "Availability: Unix", so the group half of the bound is
#: POSIX-only and Windows goes through ``procstate.terminate_process_tree``
#: (``taskkill /T``). Same spelling — and same reason — as
#: ``clipboard._SUPPORTS_PROCESS_GROUPS``.
_SUPPORTS_PROCESS_GROUPS = hasattr(os, "killpg") and hasattr(os, "getpgid")

#: The signals that must not leave a running step behind when they arrive from
#: OUTSIDE this process (see :func:`_arm_step_handlers`). SIGINT is deliberately
#: absent: Python turns it into ``KeyboardInterrupt`` in the main thread, which
#: the abort arm of :func:`_run_build_step` already reaps on the way out. SIGHUP
#: and SIGQUIT are the other two ``scripts/run_bounded.py`` forwards — a terminal
#: or ssh teardown, and a `kill -QUIT`, must not leave a group running either.
_STEP_ORPHANING_SIGNALS = tuple(
    signum
    for signum in (
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGHUP", None),
        getattr(signal, "SIGQUIT", None),
    )
    if signum is not None
)


def _failure_detail(result: subprocess.CompletedProcess[str]) -> str:
    """The actionable part of a failed command's output, on one bounded line.

    BOTH streams, because the reason is not always on stderr: `tsc -b` writes
    `error TS2307: …` to ITS stdout while pnpm writes the script echo to
    stderr, so reading ``stderr or stdout`` and keeping the final line reported
    the echo and nothing an operator could act on. Prefer lines that name an
    error, fall back to the tail when the tool failed without printing one, and
    never raise — a diagnostic must not become the failure.
    """
    lines = [
        line.strip()
        for stream in (result.stdout, result.stderr)
        for line in (stream or "").splitlines()
        if line.strip()
    ]
    errors = [
        line
        for line in lines
        if _BUILD_ERROR_HINT.search(line) and not _BUILD_WRAPPER_NOISE.match(line)
    ]
    detail = " | ".join((errors or lines)[-_BUILD_DETAIL_LINES:])
    return detail[:_BUILD_DETAIL_CHARS] if detail else "no output"


def _bundle_state(web_dir: Path | None = None) -> str:
    """built / buildable / missing-sources — what install can do about dist."""
    web_dir = _WEB_DIR if web_dir is None else web_dir
    if _dist_index(web_dir).exists():
        return "built"
    if (web_dir / "package.json").exists():
        return "buildable"
    return "missing-sources"


def _windows_shim_argv(resolved: str) -> list[str]:
    """The argv prefix that runs an npm shim on Windows, given its PATH.

    Split out from :func:`_shim_argv` so the Windows SPELLING is reachable from
    a test on any host — the alternative is patching ``os.name`` process-wide,
    which ``pathlib`` reads at call time.
    """
    # ``cmd.exe`` from ``COMSPEC`` rather than the literal: that variable is how
    # the console says where its interpreter is, and a host that has moved it
    # means it. The fallback is the one path the platform guarantees.
    return [os.environ.get("COMSPEC") or "cmd.exe", "/c", "call", resolved]


def _shim_argv(name: str) -> list[str] | None:
    """The argv PREFIX that launches the npm shim ``name``, or ``None`` if absent.

    POSIX: the name itself. ``shutil.which`` has already proved it is on PATH,
    and ``execve`` runs the ``#!/bin/sh`` wrapper directly.

    WINDOWS: through the command interpreter, because a bare name CANNOT work
    there and that is invisible from the POSIX path (audit C10). npm installs
    ``pnpm`` as a ``.CMD`` batch file, and

      * ``CreateProcess`` appends only ``.exe`` when a name carries no
        extension, so ``subprocess.run(["pnpm", ...])`` never finds the shim at
        all: it raises ``FileNotFoundError``, which this module turns into a
        bundle that "failed to build" with no stated reason;
      * handing it the resolved ``.CMD`` path is not the supported route either.
        MSDN's ``CreateProcess``: "To run a batch file, you must start the
        command interpreter; set lpApplicationName to cmd.exe and set
        lpCommandLine to the following arguments: /c plus the name of the batch
        file." It does sometimes work anyway — the JDK's ``ProcessImpl`` leans
        on that undocumented behaviour — which is a reason to use the documented
        form rather than to depend on the other.

    Hence ``<comspec> /c call <resolved shim>``: ``call`` rather than a bare
    ``/c``, because cmd strips the outer quotes of a command line it cannot
    disambiguate and the default install location,
    ``C:\\Program Files\\nodejs\\pnpm.CMD``, is exactly such a line.
    """
    found = shutil.which(name)
    if found is None:
        return None
    if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
        return _windows_shim_argv(found)
    return [name]


def _dist_index(web_dir: Path) -> Path:
    """The file whose presence means "there is a UI to serve"."""
    return _dist_dir(web_dir) / "index.html"


def _dist_dir(web_dir: Path) -> Path:
    return web_dir / "dist"


def _discard_bundle(web_dir: Path) -> None:
    """Drop a ``dist/`` the bundle guard refused.

    ``_bundle_state`` reads ANY ``index.html`` as "built" and vite writes
    ``dist/`` before npm's ``postbuild`` judges it, so a refused build leaves a
    directory that reports as servable and would be served — unstyled, with no
    error — on every later install. One owner for that removal, so the three
    callers cannot drift in how they do it and the session-deletion guard has
    one call site to argue for. The path is always this tree's ``dist``: this
    install's ``_WEB_DIR`` or the snapshot the updater is about to install.
    """
    shutil.rmtree(_dist_dir(web_dir), ignore_errors=True)


def _package_runner(
    web_dir: Path, *, env: Mapping[str, str] | None = None
) -> tuple[list[str] | None, str | None]:
    """pnpm (or corepack's pnpm) for this tree, or the tool that is missing.

    ``(runner, None)`` when a build can run, else ``(None, "node"|"pnpm")``.
    The missing TOOL rather than a sentence, because two callers word the same
    absence differently — an install step explains what to do, an updater
    status line has to stay one line — and one shared sentence reads wrong in
    one of them. Both arms go through :func:`_shim_argv`, so the Windows
    spelling of each launcher comes from the ONE place that owns it (audit
    C10); ``corepack enable`` runs against ``web_dir`` because the snapshot
    updater prepares a tree that is not this install's.

    The FIRST arm is the ``pnpm`` already on PATH and it is returned as-is, even
    when it is not the pinned version: judging a pin is :func:`_pin_mismatch`'s
    job, and it reports the refusal in the terms of the runner actually chosen.

    The LAST arm — neither ``pnpm`` nor ``corepack`` resolves — consults a
    pinned pnpm that is ALREADY installed where pnpm keeps managed versions
    (see :func:`_seeded_pnpm`) before giving up, because a machine that has one
    can build and refusing it names a remedy it does not need. Nothing is
    fetched to find it: the candidate must be present AND answer.

    ``env`` is the armed child environment (:func:`_package_manager_env`),
    passed by a caller that already built it so the ``corepack enable`` child
    and the steps that follow share one; omitted, it is built here.
    """
    if shutil.which("node") is None:
        return None, "node"
    env = _package_manager_env(web_dir) if env is None else env
    runner = _shim_argv("pnpm")
    if runner is not None:
        return runner, None
    corepack = _shim_argv("corepack")
    if corepack is None:
        pinned = _pinned_pnpm(web_dir)
        seeded = None if pinned is None else _seeded_pnpm(pinned)
        return (seeded, None) if seeded is not None else (None, "pnpm")
    _corepack_enable(corepack, web_dir, env=env)
    return [*corepack, "pnpm"], None


def _corepack_enable(corepack: Sequence[str], web_dir: Path, *, env: Mapping[str, str]) -> None:
    """Install corepack's shims, through the same bound and child environment as a step.

    WHY through :func:`_run_build_step` rather than the plain ``subprocess.run``
    this arm used: ``corepack enable`` IS a package-manager child, and the one
    thing this module has learned about those is that a bound reaching only the
    direct child is not a bound (see :func:`_run_build_step`) — it is bounded and
    its whole group is reaped on every path. It also runs in the ARMED
    environment, so the shims it writes and the pnpm it later launches resolve
    against the shared homes rather than re-fetching (see
    :func:`_package_manager_env`).

    Failure propagates exactly as it did before: the runner that follows is what
    reports whether pnpm can actually run, and a run that cannot even prepare
    corepack should say so now rather than three steps later.
    """
    _run_build_step([*corepack, "enable"], web_dir, timeout=_COREPACK_ENABLE_TIMEOUT, env=env)


def _step_group(proc: subprocess.Popen[str]) -> int | None:
    """The process group to signal for a build step, REMEMBERED at spawn.

    ``procstate.detached_popen_kwargs`` gives the child its own session on POSIX,
    so its group id equals its pid. Remembering it here rather than looking it up
    when the kill is needed is what makes the SWEEP possible at all:
    ``os.getpgid`` raises once the leader has been reaped, which is exactly the
    case a leaked descendant survives in — the same reasoning that makes
    ``clipboard._kill_tree`` take its pgid from the spawn rather than the lookup.

    ``None`` on Windows, where there is no group id to signal, and that makes the
    third path narrower there than the other two: the bound and abort rungs go
    through :func:`local_operator.procstate.terminate_process_tree`
    (``taskkill /T``), which walks the tree from the LEADER and so needs the
    leader still alive. See :func:`_sweep_step_group` for why the post-exit sweep
    is POSIX-only rather than a ``taskkill`` aimed at a reaped pid.
    """
    return proc.pid if _SUPPORTS_PROCESS_GROUPS else None


def _signal_step_group(proc: subprocess.Popen[str], pgid: int | None, *, force: bool) -> None:
    """Signal everything a build step spawned.

    Signals, and reports nothing — including nothing about whether anything was
    there. Every caller is on a path where an exception would replace the failure
    it is already reporting (the bound, an abort, the post-exit sweep), and an
    already-empty group is the ordinary case there rather than an error, so a
    "did anything die?" flag would be a value no caller could act on.

    POSIX signals the remembered GROUP. Windows has no group id, so it goes
    through ``procstate.terminate_process_tree`` (``taskkill /T``), which walks
    the tree from the leader — meaningful on the bound and abort rungs, where the
    leader is still alive, and not on the post-exit sweep (see
    :func:`_sweep_step_group`).
    """
    if pgid is None:  # pragma: no cover - exercised on Windows hosts
        procstate.terminate_process_tree(proc.pid, force=force)
        return
    try:
        os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
    except OSError:  # the group is already gone, or is not ours to signal
        pass


def _reap_step_group(
    proc: subprocess.Popen[str], pgid: int | None, grace: float | None = None
) -> None:
    """SIGTERM the step's group, escalate to SIGKILL, then let its pipes drain.

    Two rungs because a bounded tree that ignores SIGTERM would otherwise be
    waited out (the same lesson as ``scripts/run_bounded.py``, whose semantics
    this follows): SIGTERM gives a cooperative build a chance to exit, and the
    SIGKILL after ``grace`` is what actually clears the case this exists for — a
    pnpm that is forking more of itself and will never exit on its own.

    ``grace`` defaults to :data:`_BUILD_KILL_GRACE` READ AT CALL TIME rather than
    bound into the signature, so the constant stays one place a test (or an
    operator tuning the host) can move.

    ``communicate`` afterwards is not tidiness: it reaps the leader (no zombie
    behind a `lop mobile install`) and drains the pipes, which a later reader of
    ``result.stdout`` depends on.
    """
    grace = _BUILD_KILL_GRACE if grace is None else grace
    _signal_step_group(proc, pgid, force=False)
    deadline = time.monotonic() + grace
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(_GROUP_POLL_SECONDS)
    _signal_step_group(proc, pgid, force=True)
    try:
        proc.communicate(timeout=grace)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL cannot be ignored
        proc.kill()
        proc.communicate()


def _sweep_step_group(proc: subprocess.Popen[str], pgid: int | None) -> None:
    """SIGKILL whatever is still in the step's group after the leader exited.

    NOT redundant with the bound, and this is the incident's own mechanism: the
    recursion regrew AFTER the first kill, because that kill signalled only the
    processes it could see. A bound covers the timeout; an ordinary exit covers
    nothing at all, and a descendant forked in the instant the leader exited is
    the shape that survives one. Several passes because such a child can land
    after the previous pass resolved the group; the whole sweep is a quarter of a
    second against a build that has already finished.

    The pgid is the one REMEMBERED at spawn (:func:`_step_group`), so this still
    targets the right group once the leader is gone — the whole reason it is
    remembered rather than looked up.

    TWO LIMITS, stated because neither is visible from the name and both are
    reachable:

    * **POSIX only.** Windows has no group id, and ``taskkill /T`` walks the tree
      from the LEADER — which is exactly what is gone here. Windows therefore
      gets the bound and abort rungs (:func:`_reap_step_group`, where the leader
      is still alive) and no post-exit sweep. Sweeping a reaped leader's children
      there would mean a parent-pid walk, and this repo deliberately takes no
      ``psutil`` dependency (see ``conftest.py``'s memory probe) — so the honest
      answer is the narrow one, not a ``taskkill`` aimed at a pid that may since
      have been recycled.
    * **The pipe-holding descendant is the BOUND's case, not this one.** A
      descendant that inherits the captured stdout/stderr keeps ``communicate()``
      in :func:`_run_build_step` waiting until the bound fires, so that shape ends
      as a bounded ``TimeoutExpired`` reaped by :func:`_reap_step_group`, not as an
      instant sweep. What this does cover is the DETACHING shape — the one the
      tests plant and the one the field incident showed: the leader exits on its
      own and its descendants never touch our pipes.
    """
    if pgid is None:  # pragma: no cover - Windows, see the first limit above
        return
    for _ in range(5):
        _signal_step_group(proc, pgid, force=True)
        time.sleep(_GROUP_POLL_SECONDS)


def _step_stop_handler(
    proc_box: list[subprocess.Popen[str] | None], pgid_box: list[int | None], signum: int
) -> Callable[[int, object], None]:
    """The signal handler for ``signum``, closed over the step's spawn boxes.

    A factory rather than a lambda in the install loop so the signal and the two
    boxes are captured explicitly: the boxes are read at DELIVERY time, which is
    what lets the handlers be armed before the spawn (see
    :func:`_arm_step_handlers`).
    """

    def handler(_received: int, _frame: object) -> None:
        _reap_step_and_die(proc_box, pgid_box, signum)

    return handler


def _arm_step_handlers(
    proc_box: list[subprocess.Popen[str] | None], pgid_box: list[int | None]
) -> dict[int, Any]:
    """Make an EXTERNAL stop signal reap the step's group before we die.

    WHY (the opening move of the incident — QA round 1, Q-1). The step is its OWN
    session, so a signal sent to THIS process — a `kill`, a supervisor winding
    down, a `lop-update` wrapper being stopped — never reaches it. And because
    nothing is raised in this process when someone ELSE sends SIGTERM, the arms
    inside :func:`_run_build_step` never run either: the default disposition is
    to terminate, and the group is left running. That is exactly how the
    incident began — its first install was SIGTERM'd and the tree kept growing
    afterwards, with nothing left that owned it.

    Armed BEFORE the spawn, because the window between `Popen` and an installed
    handler is one a signal can slip through (the ordering
    `tests/unit/scripts/test_run_bounded.py` pins for the same reason). Only
    where there is a group to reap: Windows has none (see
    :func:`_sweep_step_group`), and a caller on a non-main thread cannot install
    handlers at all — those callers keep the exception arms and nothing else.
    Returns the handlers it replaced, for :func:`_disarm_step_handlers`.
    """
    if not _SUPPORTS_PROCESS_GROUPS:
        return {}
    previous: dict[int, Any] = {}
    for signum in _STEP_ORPHANING_SIGNALS:
        try:
            previous[signum] = signal.signal(signum, _step_stop_handler(proc_box, pgid_box, signum))
        except (OSError, ValueError):  # pragma: no cover - not installable here
            continue
    return previous


def _disarm_step_handlers(previous: dict[int, Any]) -> None:
    """Put back exactly the handlers that were there before the step.

    A library function that leaves a handler installed would change how the whole
    process dies afterwards, so this runs in a ``finally`` on every path out of
    the step — including the ones that are already raising.
    """
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):  # pragma: no cover - see _arm_step_handlers
            continue


def _reap_step_and_die(
    proc_box: list[subprocess.Popen[str] | None], pgid_box: list[int | None], signum: int
) -> None:
    """The handler body: reap the step's group, then die of the signal that asked.

    Runs between bytecodes in the main thread, so it is short, silent, and never
    raises on the way out — and it NEVER returns normally. Returning would let
    `lop mobile install` carry on behind a signal that was meant to stop it, so
    the default disposition is restored and the signal is re-delivered: the
    process then reports the platform's own status (killed by ``signum``),
    exactly what a process that had installed no handler would have reported.

    A signal that arrives before the spawn has no group to reap yet; it is
    re-delivered all the same, so that window is a delay, never a leak.
    """
    try:
        signal.signal(signum, signal.SIG_DFL)
    except (OSError, ValueError):  # pragma: no cover - see _arm_step_handlers
        pass
    proc, pgid = proc_box[0], pgid_box[0]
    if proc is not None:
        _reap_step_group(proc, pgid)
    try:
        os.kill(os.getpid(), signum)
    except OSError:  # pragma: no cover - no other way to honour the signal
        raise SystemExit(128 + signum) from None


class StepMemoryExceeded(subprocess.SubprocessError):
    """A build step whose process GROUP crossed the memory ceiling for one step.

    Public, unlike the helpers around it, because it is part of what
    :func:`_run_build_step` promises: this is the THIRD way a step can end, beside
    "finished" and ``subprocess.TimeoutExpired``.

    A ``SubprocessError`` on purpose — the base ``subprocess.TimeoutExpired`` has —
    so the two bounds are one shape to a caller, and two consequences of that are
    load-bearing. :func:`_runner_reports` already answers ``None`` for a step it
    could not finish (``except (OSError, subprocess.SubprocessError)``), which is
    what keeps :func:`_pin_mismatch`'s fail-open intact when the ceiling is what
    stopped a probe; and :func:`_build_bundle` words it as a failed step exactly as
    it words the time bound. ``args[0]`` is the operator-facing sentence.
    """


def _mb_text(value_bytes: int) -> str:
    """A memory figure a reader can compare at a glance (GB above 1 GB)."""
    if value_bytes >= 1024 * 1024 * 1024:
        return f"{value_bytes / (1024**3):.1f} GB"
    return f"{value_bytes / (1024 * 1024):.0f} MB"


def _step_memory_report(sample: memory_guard.Sample, budget: memory_guard.Budget) -> str:
    """What an operator reads when a build step was killed for memory.

    The same SHAPE the guarded bash command reports
    (:meth:`memory_guard.Guard.over_budget_message`): the header word first, then
    the MEASURED peak against the ceiling and the device it was derived from —
    naming the numbers is what lets the reader size the retry.

    The REMEDY clause is this path's own, and that is deliberate. The bash copy
    closes with ``memory_mb``/``bash.memory.limit_mb``; neither exists on
    ``lop mobile install``, and naming a knob the reader cannot turn is the defect
    this module has been fixed for twice already (see the remedy sentences in
    :func:`_build_bundle`). The honest remedy here is the one that does work: the
    ceiling is a function of what is FREE, so it rises on its own once the host has
    room — free memory, then re-run. ONE line, because every caller shows it inside
    a summary line of its own and :func:`_failure_detail` reads line by line.
    """
    device = f"on a {budget.total_mb} MB device" if budget.total_mb else "on this device"
    # The reserve is rendered only when the computation produced one. The override
    # and manual arms answer ``None`` — an explicit ceiling needs no reserve
    # arithmetic — and the sentence would otherwise read "minus a None MB reserve"
    # the moment an operator override exists on this path (QA round 1, Q-1: not
    # reachable today only because :func:`_step_memory_budget` takes no arguments).
    reserve = f" minus a {budget.reserve_mb} MB reserve" if budget.reserve_mb is not None else ""
    return (
        f"{_STEP_MEMORY_HEADER}: this build step's process group reached "
        f"{_mb_text(sample.bytes_used or 0)}, over the {budget.ceiling_mb} MB ceiling "
        f"for one build step {device}. The group was killed; the install is fine and "
        "nothing else on the machine was touched. A package manager that cannot "
        "finish an install fans out until the machine dies, so free memory and "
        f"re-run `lop mobile install` — the ceiling is half of what is available"
        f"{reserve}, so it rises on its own once the host has room"
    )


def _step_memory_budget() -> memory_guard.Budget:
    """The memory ceiling ONE build step gets, from the SHARED budget arithmetic.

    WHY SHARED, and this is the whole reason the build path is a small change
    rather than a second guard: ``0.5 x available minus reserve`` and the reserve
    constants are the ONE budget vocabulary on this machine
    (``docs/design/process-memory-guard.md`` §3.2, which mirrors ``conftest.py``),
    and a second copy of them here is a number nobody would notice had drifted —
    the failure mode that doc exists to prevent. §10 of that doc names this path as
    phase 2's first consumer ("the ``Guard`` class is deliberately reusable so phase
    2 is wiring, not design"), i.e. wiring, which is what this is.

    The FLOOR is the one thing the caller names (:data:`_STEP_MEMORY_FLOOR_MB`),
    because "ordinary command" and "one package-manager build step" are different
    sizes; the arithmetic itself is untouched.

    The config keys are deliberately NOT read here, unlike the bash tool's
    ``_configured_memory_budget``. ``ConfigManager(config_dir)`` WRITES a
    default ``config.yml`` when none exists, and this path runs on fresh clones,
    in containers, and from ``lop update``'s snapshot builder — a memory bound is
    not a good enough reason to start creating the operator's config from a build.
    A config that says nothing therefore gets the protection, which is the
    documented default anyway.
    """
    return memory_guard.compute_budget(floor_mb=_STEP_MEMORY_FLOOR_MB)


def _step_memory_guard(pgid: int | None, budget: memory_guard.Budget) -> memory_guard.Guard | None:
    """A guard bound to THIS step's group, or ``None`` when there is nothing to bound.

    ``None`` on a host with no process groups to sample (Windows: ``os.getpgid``
    does not exist there, so :func:`_step_group` already answered ``None``) and on
    a host whose memory probes cannot answer (``source == "disabled"``), which is
    also where an operator's ``bash.memory.enabled=false`` would land if this path
    read it (it deliberately does not — see :func:`_step_memory_budget`). Both are
    the pre-guard behaviour and neither may raise: an install must not fail because
    a probe could not describe the host.

    The BUDGET is passed in rather than resolved here on purpose: resolving it
    forks the host probes, and this function runs with the child already spawned,
    where a fork is a window an abort can escape through. Constructing a
    :class:`memory_guard.Guard` itself spawns nothing. See :func:`_run_build_step`.

    The guard only ever reads and kills the pgid it is HANDED here, which is the
    one :func:`_step_group` remembered from this step's own spawn — it never
    enumerates a group of its own, so it cannot touch ``lop`` or a sibling process
    (``memory_guard`` §6/F5).
    """
    if pgid is None:
        return None
    if budget.source == "disabled":
        return None
    return memory_guard.Guard(pgid, budget)


def _wait_for_step(
    proc: subprocess.Popen[str],
    guard: memory_guard.Guard | None,
    bound: float,
) -> tuple[str, str, str | None]:
    """Wait up to ``bound`` seconds, sampling the group's memory while it runs.

    Returns ``(stdout, stderr, report)``: ``report`` is the operator-facing sentence
    when the group crossed its ceiling and the caller must kill the group, or
    ``None`` when the step ended on its own. Raises ``subprocess.TimeoutExpired``
    when the TIME bound elapses first — the behaviour this replaced
    (``proc.communicate(timeout=bound)``) and the only other way out.

    WHY A SLICE LOOP, and this is the point of the whole change. One blocking
    ``communicate(timeout=bound)`` cannot see the group it waits for, so a memory
    bound bolted beside it would never fire in time to matter: the incident's growth
    (+100 processes and +5 GB every 25 s) is orders of magnitude past any sane
    ceiling long before a 600 s wait returns — the host is dead first. So the wait
    is taken in ``guard.tick_s`` slices with a sample between them, and the ceiling
    is the wall that actually fires; the time bound is the backstop.

    A failed sample is never a kill (``Guard.should_kill`` requires a MEASURED
    reading), so a hiccupping ``ps`` leaves the step alone — F6, and the reason this
    cannot make an install flakier than it was. The SOFT line is not consumed here
    either: the bash tool paints that advisory on the live update stream, and a
    build step has no such stream, so showing it would be a line nobody sees.

    On the memory arm the returned streams are EMPTY, deliberately: the reason the
    step ended is the report, and the caller is about to reap the group (which
    drains those pipes) rather than report a partial build log as a failure detail.

    Re-entering ``communicate`` is documented ("Catching this exception and
    retrying communication will not lose any output") and the OTHER half of that
    property — that it does not DUPLICATE what it already read — is not in the
    docs, so both halves are pinned by a test
    (``test_a_sliced_wait_captures_the_same_streams_as_one_blocking_call``). The
    captured streams are what :func:`_failure_detail` reads, and a bound that cost
    the step's own error message would be a worse install, not a safer one.
    """
    if guard is None:
        stdout, stderr = proc.communicate(timeout=bound)
        return stdout, stderr, None
    deadline = time.monotonic() + bound
    last: subprocess.TimeoutExpired | None = None
    while True:
        # A slice that RUNS OUT rather than one that blocks past the deadline. The
        # zero slice is NOT the "exits exactly at the deadline" case this comment
        # first described: `communicate(timeout=0)` is an immediate poll that raises
        # `TimeoutExpired` without reading anything, even for a child that has
        # already exited with output pending — measured on this interpreter (agent
        # review round 1, MINOR-1). So zero is reachable only once the deadline has
        # passed, and a step that exits AT the deadline is caught by the positive
        # slice that ends on it.
        slice_s = max(0.0, min(guard.tick_s, deadline - time.monotonic()))
        try:
            stdout, stderr = proc.communicate(timeout=slice_s)
        except subprocess.TimeoutExpired as exc:
            last = exc
        else:
            return stdout, stderr, None
        if time.monotonic() >= deadline:
            # The same shape the single blocking call raised — ``cmd`` is the argv
            # and ``timeout`` the bound, which is what _build_bundle words as
            # "Command '...' timed out after N seconds" — and it carries the last
            # slice's partial streams, because the call it replaced carried them,
            # and a reader of ``exc.output`` must not find the field quietly
            # emptied. No caller reads it today; that is exactly when a lost field
            # goes unnoticed (agent review round 1, MINOR-1).
            raise subprocess.TimeoutExpired(
                proc.args,
                bound,
                output=last.output if last is not None else None,
                stderr=last.stderr if last is not None else None,
            )
        sample = guard.sample_sync()
        if guard.should_kill(sample):
            return "", "", _step_memory_report(sample, guard.budget)


def _run_build_step(
    argv: Sequence[str],
    cwd: Path,
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bundle-build command in its OWN process group, and reap the group.

    WHY (incident, 2026-09-21 — this is the defect that took the machine down).
    ``subprocess.run(..., timeout=600)`` bounds the DIRECT CHILD only: when the
    bound fires, Python kills the leader and returns, and everything the leader
    forked keeps running. On a fresh clone the portal build reached that state
    through a ``packageManager`` pin the global pnpm did not match (see
    :func:`_pin_mismatch`): pnpm resolves such a pin by installing that pnpm WITH
    pnpm, and the tree GREW BACK after the first kill because the survivors kept
    spawning — +100 processes and +5 GB every 25 s until the host had 0.1 GB free
    and no swap, and was rebooted.

    THE TIME BOUND CANNOT BE THE WALL FOR THAT FAILURE, and this is the arithmetic
    the second bound below exists for. The recorded rate is +5 GB per 25 s = 0.2 GB/s
    steady, with bursts to 0.4 GB/s (3.2 GB in the first 8 s of the reproduction),
    so a 600 s bound permits on the order of ``0.2 GB/s x 600 s = 120 GB`` of growth
    before it fires — against a 36 GB host and a 32 GB device. The host is dead
    three orders of magnitude before the clock helps. So the step now runs under a
    MEMORY bound too (:func:`_wait_for_step`), reusing ``memory_guard``'s per-command
    ceiling and its group sampler rather than inventing a second budget — the
    arithmetic and the reserve vocabulary have ONE owner on this machine
    (``docs/design/process-memory-guard.md`` §10 names this path as phase 2's first
    consumer, so this is wiring rather than design). The time bound stays as the
    backstop, which is all it can honestly be.

    So the step runs in its own session/group and the GROUP is signalled on every
    path that can end the wait:

    * the memory ceiling firing (the third rung, added with the second bound);
    * the time bound firing;
    * an abort raised IN this process (Ctrl-C -> ``KeyboardInterrupt``);
    * an EXTERNAL stop signal (SIGTERM/SIGHUP/SIGQUIT), which raises nothing here
      and is why :func:`_arm_step_handlers` exists;
    * and the ordinary exit, where a descendant can outlive its leader.

    Two of those are narrower than they sound, and both are stated where a reader
    asking "what exactly is covered" ends up: the last is POSIX-only, and the
    pipe-holding shape of the third is the bound's case rather than the sweep's
    (see :func:`_sweep_step_group`).

    NOT ``scripts/run_bounded.py``: this follows that wrapper's semantics
    deliberately, but the wrapper lives in the repository's ``scripts/`` tree,
    which is not shipped in the wheel — an installed ``lop`` has no such file to
    run. The inline spelling therefore reuses ``procstate`` for the one part
    that must not be hand-rolled (the platform decision).

    ``env`` is passed to the child verbatim (``None`` inherits this process's
    environment, which is what every caller did before the parameter existed).
    It exists so the package-manager children can be given the armed environment
    in :func:`_package_manager_env` — the appending of it belongs to this ONE
    spawn site, so no caller can arm the environment and then forget to hand it
    to the child (the shape agent review round 1 caught: an armed environment
    that nothing passed on).
    """
    #: Resolved BEFORE anything else, and the ordering is a fix rather than
    #: tidiness. The budget forks host probes (``vm_stat`` for available memory,
    #: ``sysctl -n vm.swapusage`` for free swap), so resolving it once a child
    #: exists opens an instant in which an abort — a real Ctrl-C, which
    #: :func:`_arm_step_handlers` deliberately does not cover — escapes with the
    #: step's group live and no arm left to reap it. That was measured, not
    #: theorised: the guard call sat above the arms and the step leader survived an
    #: abort, re-parented to 1, its descendant with it (agent review round 1,
    #: BLOCKER-1 — 6/6 red on the head, clean on the base whose spawn-to-arm gap is
    #: three pure-Python statements). With no child yet there is no group to leak,
    #: and the reading is taken with the host undisturbed by the step.
    memory_budget = _step_memory_budget()
    #: Boxes rather than locals: the handlers are armed BEFORE the spawn and read
    #: these at delivery time, so they see the process once it exists.
    proc_box: list[subprocess.Popen[str] | None] = [None]
    pgid_box: list[int | None] = [None]
    handlers = _arm_step_handlers(proc_box, pgid_box)
    bound = _BUILD_STEP_TIMEOUT if timeout is None else timeout
    try:
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            list(argv),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            **procstate.detached_popen_kwargs(),
        )
        proc_box[0] = proc
        pgid = _step_group(proc)
        pgid_box[0] = pgid
        try:
            # Inside the arms as well, and for the same reason: from the line
            # above the group EXISTS, so every statement between here and the wait
            # has to be interruptible INTO a reap. Constructing the guard is pure
            # Python — it probes nothing (see :func:`_step_memory_guard`) — and it
            # is in here so that a future change cannot quietly move work back into
            # the gap this fix closed.
            guard = _step_memory_guard(pgid, memory_budget)
            stdout, stderr, memory_report = _wait_for_step(proc, guard, bound)
        except subprocess.TimeoutExpired:
            _reap_step_group(proc, pgid)
            raise
        except BaseException:
            # Any abort raised HERE — the operator's Ctrl-C above all: the step is
            # its own session, so the terminal does not deliver SIGINT to it and
            # the group would otherwise carry on building behind a `lop mobile
            # install` that has already exited. An external SIGTERM/SIGHUP raises
            # nothing here at all; that is `_arm_step_handlers`' half.
            _reap_step_group(proc, pgid)
            raise
        if memory_report is not None:
            # The group is over its ceiling and still growing, so the SIGTERM
            # window is one poll rather than _BUILD_KILL_GRACE (see that constant).
            # Reaped BEFORE raising, for the same reason the time-bound arm reaps
            # before re-raising: whatever the caller does with the exception, the
            # tree is not left behind.
            _reap_step_group(proc, pgid, grace=_BUILD_MEMORY_KILL_GRACE)
            raise StepMemoryExceeded(memory_report)
        _sweep_step_group(proc, pgid)
    finally:
        _disarm_step_handlers(handlers)
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


#: ``packageManager`` is spelled ``name@version``, optionally with a Corepack
#: integrity suffix (``pnpm@11.22.0+sha512.…``), which is not part of the version.
_PACKAGE_MANAGER_PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)@(?P<version>[^+\s]+)")

#: An EXACT version, which is all equality can judge — see
#: :func:`_dev_engines_pin` for why ``devEngines``' ranges are not enforced.
_EXACT_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")


def _dev_engines_pin(manifest: dict[str, Any]) -> str | None:
    """The pnpm version ``devEngines.packageManager`` names, if it is EXACT.

    pnpm 11 added this second spelling (pnpm 11.0.0, and it is the one whose
    ``onFail`` decides whether a miss errors, warns or DOWNLOADS), so it resolves
    a mismatch through the same managed-version path the guard exists to refuse —
    hence it is read rather than ignored.

    A RANGE is deliberately not a pin here, and the cost of that is real rather
    than hypothetical: a range gets NO protection from this guard. ``^11.5.1`` is
    not a version, so equality cannot judge it, and whatever pnpm the host has is
    used as-is — measured on this host, a pnpm 10.30.3 that does not satisfy
    ``^11.5.1`` is let through by this guard. (What pnpm then does with the
    mismatch is pnpm's behaviour, not this guard's, and no pnpm was run to learn
    it.) Judging a range needs a semver implementation, which is not what this fix
    is, so an unjudgeable range stays unenforced rather than being refused
    wholesale; ``packageManager``'s exact pins are what these trees carry and what
    is decidable here.
    """
    dev = manifest.get("devEngines")
    entry = dev.get("packageManager") if isinstance(dev, dict) else None
    if not isinstance(entry, dict):
        return None
    name, version = entry.get("name"), entry.get("version")
    if not isinstance(name, str) or name.lower() != "pnpm" or not isinstance(version, str):
        return None
    candidate = version.strip()
    return candidate if _EXACT_VERSION.match(candidate) else None


def _pinned_pnpm(web_dir: Path) -> str | None:
    """The pnpm version ``web_dir``'s ``package.json`` pins, or ``None``.

    BOTH SPELLINGS pnpm reads, because both resolve a mismatch through the same
    managed-version path: the ``packageManager`` field (``pnpm@11.22.0``), and
    ``devEngines.packageManager`` when the first is absent (see
    :func:`_dev_engines_pin`). ``packageManager`` WINS when both are present — it
    is the field Corepack resolves first and the one these trees carry — and a
    manifest whose two pins disagree is a broken manifest rather than a guess
    this guard should make; it fails loudly in pnpm's own hands either way.

    ``None`` covers every "there is no pin to enforce" case — no manifest,
    unreadable JSON, neither field, a pin naming a different manager, a
    ``devEngines`` range — because refusing to build a tree that pins nothing
    would break the snapshot updater's older trees for no gain: this guard is an
    extra wall, not the wall (the group bound around every step is what makes a
    bad build survivable).
    """
    try:
        manifest = json.loads((web_dir / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    pin = manifest.get("packageManager")
    if isinstance(pin, str):
        match = _PACKAGE_MANAGER_PIN.match(pin.strip())
        if match is not None and match.group("name").lower() == "pnpm":
            return match.group("version")
    return _dev_engines_pin(manifest)


def _probe_env() -> dict[str, str]:
    """The environment for a ``--version`` probe: the fetch disarmed, nothing moved.

    Deliberately NOT :func:`_package_manager_env`. That one relocates
    ``COREPACK_HOME``/``PNPM_HOME`` at the shared homes so a build child resolves
    against them, and doing that in a PROBE would make the guard itself fetch a
    package manager on a host whose ``pnpm`` is a corepack shim with a cold
    cache — the one thing a probe must never cause. The probe wants the opposite
    property, and it has it: an empty directory and this setting.

    ``COREPACK_ENABLE_DOWNLOAD_PROMPT=0`` is set here too because a probe's stdin
    is a pipe, so a version resolution that decided to download would block on a
    confirmation nobody can answer until the bound fires.
    """
    env = dict(os.environ)
    env[_MANAGE_PM_VERSIONS_ENV] = "false"
    env[_COREPACK_DOWNLOAD_PROMPT_ENV] = "0"
    return env


def _pnpm_home() -> Path:
    """Where pnpm keeps the versions it manages for itself, resolved as pnpm does.

    Mirrors ``getDataDir`` in pnpm's own CLI (read from the shipped 10.30.3
    bundle): ``PNPM_HOME``, else ``XDG_DATA_HOME/pnpm``, else the platform default
    (``~/Library/pnpm`` on macOS, ``%LOCALAPPDATA%\\pnpm`` on Windows,
    ``~/.local/share/pnpm`` elsewhere). Mirroring rather than inventing is the
    point: ``PNPM_HOME`` also decides where pnpm's content-addressable STORE lives
    (``<PNPM_HOME>/store``, the tree every install in this repo shares through
    hard links), so a location of our own choosing would silently repopulate a
    second store — hundreds of megabytes on a machine that already has one.

    The module needs the value for one reason: pnpm's version switch looks for a
    pinned pnpm under ``<PNPM_HOME>/.tools/pnpm/<version>``, so a machine that has
    already seeded one has to be recognised as seeded rather than refused.
    """
    override = os.environ.get("PNPM_HOME")
    if override:
        return Path(override)
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "pnpm"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "pnpm"
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "pnpm"
    return Path.home() / ".local" / "share" / "pnpm"


def _corepack_home() -> Path:
    """Where corepack caches what it downloads, resolved as corepack does.

    ``COREPACK_HOME``, else ``%LOCALAPPDATA%\\node\\corepack`` on Windows and
    ``$HOME/.cache/node/corepack`` everywhere else — the defaults corepack
    documents. Both are named explicitly in the build child's environment so one
    download serves every clone and worktree of this repo on the machine; the
    home-derived default moves with a relocated ``HOME``, which is what a
    container, a CI runner and an isolated test run each have.
    """
    override = os.environ.get("COREPACK_HOME")
    if override:
        return Path(override)
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "node" / "corepack"
    return Path.home() / ".cache" / "node" / "corepack"


def _package_manager_env(web_dir: Path) -> dict[str, str]:
    """The child environment for the build path, one owner for all of it.

    Three things beyond the ambient environment, and the first is the one that
    matters:

      * ``npm_config_manage_package_manager_versions=false`` in a tree that pins an
        exact pnpm. pnpm reaches its ``installPnpmToTools`` (the ``pnpm add
        pnpm@<pin>`` fetch, and the fork chain) only through ``switchCliVersion``,
        which is gated on exactly this setting — so turning it off makes the fetch
        UNREACHABLE rather than merely unlikely. (pnpm sets the same flag in the
        child it re-executes after a switch, which is the same idea one layer in and
        is where this setting's meaning was read from.) This is defence in depth
        behind the refusal in :func:`_pin_mismatch`, never a substitute for it: on
        its own it does not fail closed, and a shared home that is unwritable or
        misconfigured would put an unguarded install straight back into the loop.
        With it set, a pnpm that does not match the pin does NOT fail on its own —
        measured on 2026-09-21 with pnpm 10.30.3 against a tree pinning 11.22.0:
        ``pnpm build`` and ``pnpm install --lockfile-only`` both exited **0**, and
        the build script reported ``pnpm/10.30.3``. pnpm's version check compares
        the pin with the running pnpm but only THROWS when ``packageManagerStrict``
        is on, and only compares at all when ``packageManagerStrictVersion`` is on
        (both default to warn-and-continue for the version, which is why an earlier
        revision of this docstring claimed a loud failure the code did not
        deliver). So those two settings are added with the same exact-pin gate, and
        with both the same two commands exit **1** with ``ERROR This project is
        configured to use v11.22.0 of pnpm. Your current pnpm is v10.30.3``.
        ``package_manager_strict`` is set explicitly rather than relied on as a
        default because an operator's ``.npmrc`` may turn it off, and this layer's
        job is to fail CLOSED; neither setting can fire on a build that the pin
        itself runs, which is every route this module chooses.

        The EXACTNESS test is this function's own, and it is not redundant with
        :func:`_pinned_pnpm`: that one reports the ``packageManager`` spelling's
        version verbatim, a range included, while pnpm's switch returns early for
        anything ``semver.valid`` rejects — ``switchCliVersion`` warns ``Cannot
        switch to pnpm@^11: "^11" is not a valid version`` and RETURNS, measured in
        the shipped 10.30.3 bundle — so a range is not a fetch, and disarming one
        would turn pnpm's warn-and-continue into an error on a tree that works
        today. The strict pair is gated the same way, and for the reason the
        MEASUREMENT gives rather than a plausible-looking one: the pair does not fire
        on its own. The version check it controls is only reached in the ELSE of
        pnpm's switch decision — ``if (config.managePackageManagerVersions &&
        config.wantedPackageManager?.name === "pnpm" …) switchCliVersion(config); else
        … checkPackageManager(…)`` in the shipped bundle — and for a range pin the
        switch branch owns the call and returns early with its warning, so the pair
        is inert on a range until something makes that else-branch run. That
        something is this function's own disarm: set together, they turn a range tree
        that is fine into a failure, measured on a tree pinning ``pnpm@^11`` with
        both (``pnpm install --lockfile-only`` → rc 1, ``ERROR This project is
        configured to use v^11 of pnpm. Your current pnpm is v10.30.3``). Gating both
        on an exact pin is what keeps that from happening; the range case is handled
        by the refusal instead, which is why the disarm is gated with them.
      * ``PNPM_HOME``/``COREPACK_HOME`` at the locations resolved above — the
        shared, pre-populated homes — so a seeded machine finds the pinned manager
        already there instead of fetching it.
      * ``COREPACK_ENABLE_DOWNLOAD_PROMPT=0``: whatever the corepack arm does may
        download, and a child whose stdin is a pipe can never answer the
        confirmation prompt that would otherwise guard it.
    """
    env = dict(os.environ)
    env["PNPM_HOME"] = str(_pnpm_home())
    env["COREPACK_HOME"] = str(_corepack_home())
    env[_COREPACK_DOWNLOAD_PROMPT_ENV] = "0"
    pin = _pinned_pnpm(web_dir)
    if pin is not None and _EXACT_VERSION.fullmatch(pin) is not None:
        env[_MANAGE_PM_VERSIONS_ENV] = "false"
        env[_STRICT_PM_VERSION_ENV] = "true"
        env[_STRICT_PM_ENV] = "true"
    return env


def _seeded_pnpm(pin: str) -> list[str] | None:
    """The pinned pnpm already sitting in the shared home, if it is really there.

    pnpm's switch treats the mere EXISTENCE of ``<PNPM_HOME>/.tools/pnpm/<pin>/bin``
    as "the pinned version is installed" and re-executes whatever is inside
    (``alreadyExisted`` in its ``installPnpmToTools``). That directory-existence
    test is deliberately NOT repeated here: the candidate has to ANSWER with the
    pinned version, because a directory left behind by a killed or half-written
    fetch is exactly what turns one bad install into a chain that never converges.
    This host is the standing example — ``~/Library/pnpm/.tools/pnpm/`` carries
    ~14,800 ``11.22.0_tmp_<pid>`` stage directories from the incident and no
    completed ``11.22.0`` at all, so existence alone would have "found" the pin in
    any of them.
    """
    bin_dir = _pnpm_home() / ".tools" / "pnpm" / pin / "bin"
    candidate = bin_dir / ("pnpm.cmd" if os.name == "nt" else "pnpm")
    if not candidate.exists():
        return None
    argv = _windows_shim_argv(str(candidate)) if os.name == "nt" else [str(candidate)]
    return argv if _runner_reports(argv) == pin else None


def _runner_reports(runner: Sequence[str]) -> str | None:
    """What ``runner`` answers for ``--version``, or ``None`` if it cannot.

    Run in a THROWAWAY directory, and that is load-bearing rather than tidy: pnpm
    reads the nearest ``package.json`` above its cwd, so a probe run INSIDE the
    pinned tree does not answer at all — it starts RESOLVING the pin, which is
    the recursion this guard exists to refuse. Measured on this host: `pnpm
    --version` under a ``packageManager: pnpm@11.22.0`` manifest had not returned
    after 120 s and was spawning ``pnpm add pnpm@11.22.0`` children (28 processes,
    3.2 GB RSS in 8 s), while the same command in an empty directory answered in
    under a second. Through :func:`_run_build_step` so the probe is itself
    bounded and group-reaped: a probe that can hang must not hang the install. It
    runs with the fetch DISARMED in the child and nothing else moved — see
    :func:`_probe_env`, and :func:`_package_manager_env` for why the shared homes
    are deliberately not the probe's.
    """
    with tempfile.TemporaryDirectory(prefix="lop-pnpm-probe-") as probe_dir:
        try:
            result = _run_build_step(
                [*runner, "--version"],
                Path(probe_dir),
                timeout=_PIN_PROBE_TIMEOUT,
                env=_probe_env(),
            )
        except (OSError, subprocess.SubprocessError):
            return None
    if result.returncode != 0:
        return None
    lines = (result.stdout or "").strip().splitlines()
    return lines[0].strip() if lines else None


def _corepack_shaped(runner: Sequence[str]) -> bool:
    """Whether ``runner`` is Corepack's pnpm rather than a global one.

    Two shapes reach here: the argv :func:`_package_runner`'s Corepack arm
    returns (``corepack pnpm``), and a ``pnpm`` that Corepack itself installed as
    a shim on PATH — a script that re-executes Corepack. Both RESOLVE the pin,
    which is the whole point of Corepack and a route :func:`_pin_mismatch` offers
    wherever a corepack launcher resolves, so comparing their reported version
    against the pin would refuse a build that was about to be handled correctly.
    Checked by reading the shim, and only ever to AVOID refusing: a wrong answer
    here costs one unguarded build, never a refused good one.
    """
    if not runner:
        return False
    if any(
        Path(part).name.lower() in {"corepack", "corepack.cmd", "corepack.exe"} for part in runner
    ):
        return True
    resolved = shutil.which(runner[0])
    if resolved is None:
        return False
    try:
        with open(resolved, "rb") as shim:
            return b"corepack" in shim.read(4096).lower()
    except OSError:
        return False


def _pin_mismatch(runner: Sequence[str], web_dir: Path) -> str | None:
    """Refuse a build whose pnpm is not the version the tree pins, or ``None``.

    WHY (the incident's other half). pnpm does not fail when the version on PATH
    disagrees with ``packageManager``: it resolves the pin by INSTALLING that
    pnpm with pnpm (``pnpm add pnpm@11.22.0``), and on a machine that cannot
    finish that install each attempt forks more of the same — measured here as 28
    processes and 3.2 GB RSS in 8 s, and as +100 processes / +5 GB every 25 s in
    the incident until the host was out of memory. The bound in
    :func:`_run_build_step` turns that into a bounded failure; this guard is what
    makes it not start at all.

    The comparison is against what the RESOLVED runner reports, never against
    the pin's own claim, because the pin is the thing in doubt. A runner that
    cannot answer (no output, non-zero exit, the probe's own timeout, or a memory
    kill — see below) is NOT refused: the bound protects that case, and refusing an
    install because a probe was inconclusive would trade a rare hang for a common
    false refusal on machines whose pnpm is entirely fine.

    AND THAT GROUNDS IS NOW TRUE, which it was not when it was written — stated
    here because the fail-open above is justified BY the bound, so the fail-open is
    only as good as the bound. A probe that has started resolving the pin instead of
    answering grows at the measured 0.4 GB/s (3.2 GB in its first 8 s), and
    ``_PIN_PROBE_TIMEOUT`` is 20 s: a TIME bound therefore permitted roughly another
    8 GB before it could fire, on a host that had 0.1 GB free. The probe now runs
    through :func:`_run_build_step` like every other step, so it is under the same
    memory-sampled wait and is stopped at the ceiling a few ticks in.

    WHERE THAT HOLDS, because the paragraph above is only as good as the bound and
    the bound is a function of the HOST. It holds on a host in pressure, which is
    the case this change exists for (the incident's host: 0.1 GB free, so the
    256 MB floor WAS the ceiling, ~1 s) and on a host like this one (≈2.8 GB from
    ≈5.6 GB available, ≈7-14 s at the recorded rate). It is NOT universal, and the
    two limits belong where a reader deciding whether to lean on this fail-open
    will look: a roomy host — roughly 16 GB+ available — has a ceiling that 20 s of
    the recorded growth cannot reach, so ``_PIN_PROBE_TIMEOUT`` still fires first
    there and roughly another 8 GB can be spent; and on Windows, or on any host
    whose memory probes cannot answer, :func:`_step_memory_guard` answers ``None``,
    so the probe is on the clock alone and this paragraph does not apply at all.

    The fail-open posture is deliberately LEFT ALONE: it can now rest on a bound
    that holds at the recorded site, and flipping it to fail-closed would cost
    every user with an inconclusive probe an install for a hang the ceiling
    already stops.
    """
    pinned = _pinned_pnpm(web_dir)
    if pinned is None or _corepack_shaped(runner):
        return None
    reported = _runner_reports(runner)
    if reported is None or reported == pinned:
        return None
    # The route this copy names must WORK ON THE MACHINE PRINTING IT, which is
    # the defect D8/D14 fixed in the no-pnpm arm above and this arm kept: it
    # offered `corepack enable` unconditionally, and Corepack stopped shipping
    # with Node at v25 — this host runs Node v26.5.0, whose install has no
    # corepack file and no `corepack` on PATH, so a reader here would run the
    # remedy, get "command not found", and read a second refusal back.
    #
    # NO SINGLE ROUTE IS UNIVERSAL, and an earlier attempt at this fix simply
    # promoted the next candidate: `npm install -g pnpm@<pin>` cannot land where
    # another manager owns the `pnpm` on PATH either — measured here, npm
    # 11.17.0's global prefix is `/opt/homebrew`, exactly the directory Homebrew's
    # own pnpm symlink lives in, and npm refuses to clobber a shim it does not own
    # (reproduced offline on a scratch prefix: `npm error code EEXIST … Remove the
    # existing file and try again, or run npm with --force`; `--force` then
    # succeeds and repoints the link). So the copy states the CONDITION that has
    # to hold, names the routes that exist, and carries each route's obstacle with
    # it instead of promising that one of them will just work. Which routes exist
    # is asked of the host, through the ONE place that owns each launcher's argv
    # (see :func:`_shim_argv`), because a route named where it is absent is the
    # same defect one step along.
    routes: list[str] = []
    if _shim_argv("npm") is not None:
        routes.append(
            f"`npm install -g pnpm@{pinned}` (add `--force`, or remove the existing "
            "shim first, if another manager owns it)"
        )
    if _shim_argv("corepack") is not None:
        routes.append("`corepack enable`")
    routes.append("whatever manager installed the pnpm already on PATH")
    if len(routes) == 1:
        how = routes[0]
    elif len(routes) == 2:
        how = f"{routes[0]} or {routes[1]}"
    else:
        how = ", ".join(routes[:-1]) + f", or {routes[-1]}"
    return (
        f"the pnpm on PATH is {reported} but {web_dir / 'package.json'} pins "
        f"pnpm@{pinned}: PATH's pnpm has to report {pinned} before this build can "
        f"start -- install that version with {how}, then re-run "
        "`lop mobile install`. A pnpm that does not match its own pin resolves it "
        "by installing pnpm WITH pnpm, which fans out until the machine dies"
    )


def _runner_or_refusal(
    runner: list[str], web_dir: Path, *, env: Mapping[str, str]
) -> tuple[list[str], str | None]:
    """``(runner, None)``, or the guard's own refusal when no route can satisfy the pin.

    THE POLICY THIS BRANCH ADDS TO #1394's GUARD, and the one place it changes
    shipped behaviour: main refuses whenever the runner's reported version is not
    the pin, while this prefers a runner that CAN supply the pin and refuses only
    when none can. The order, and why each clause is where it is:

      1. the runner PATH resolved, when :func:`_pin_mismatch` accepts it. That call
         is also the ONLY version probe on a machine with nothing else to offer,
         so the process list #1394's tests assert is unchanged — this function adds
         no probe to the happy path, and ``_pin_mismatch`` still owns the sentence
         and the policy for a runner that cannot answer at all (not a mismatch;
         the group bound is the wall there);
      2. a pinned pnpm VERIFIED where pnpm keeps managed versions
         (:func:`_seeded_pnpm`) — local, so it downloads nothing, and verified by
         ASKING it rather than by its directory existing, so the residue a killed
         fetch leaves behind is not mistaken for a pin;
      3. corepack (:func:`_corepack_enable` then ``corepack pnpm``) — a bounded
         tarball fetch that converges, run through the same bounded, armed runner
         as every other child, and pnpm's own switch is unreachable on this route
         (``isExecutedByCorepack``). This is the route a machine that already
         relied on corepack keeps: dropping the arm refused hosts that used to
         build, which is the regression this arm closes;
      4. nothing — the runner is handed back WITH the guard's refusal, so the
         sentence a reader ends up acting on is still the one #1394 ships.

    Arms 2 and 3 are consulted only when the guard WOULD refuse, which is what
    keeps a machine whose PATH pnpm already satisfies the pin from having a
    package manager downloaded for it: there is nothing to fix, so nothing is
    fetched. (:func:`_package_runner`'s no-pnpm arm keeps #1394's own order —
    corepack first — because a host with no pnpm on PATH has no runner to compare
    and corepack is the route it already has; the seeded pin is that arm's
    fallback.)
    """
    mismatch = _pin_mismatch(runner, web_dir)
    if mismatch is None:
        return runner, None
    pin = _pinned_pnpm(web_dir)
    if pin is None:
        return runner, mismatch
    seeded = _seeded_pnpm(pin)
    if seeded is not None:
        return seeded, None
    corepack = _shim_argv("corepack")
    if corepack is None:
        return runner, mismatch
    _corepack_enable(corepack, web_dir, env=env)
    return [*corepack, "pnpm"], None


def _build_bundle(web_dir: Path | None = None, runner: list[str] | None = None) -> str | None:
    """Build the SPA in place. Returns an error string, or None on success.

    pnpm only — the lockfile and packageManager pin are pnpm's, and mixing
    npm here would write a second, unreviewed lockfile. Corepack is tried
    first so a machine with only Node (no global pnpm) still self-heals;
    the packageManager field pins the exact pnpm corepack fetches.

    Both candidates are launched through :func:`_shim_argv` (see
    :func:`_package_runner`), which is what makes this work on Windows at all:
    there the resolvable ``pnpm`` is a ``.CMD`` batch file that a bare argv
    never reaches (audit C10).

    ``web_dir`` defaults to this install's tree; the snapshot updater passes
    the tree it is about to install (see :func:`snapshot_bundle`), so both
    paths share this one builder rather than carrying a second copy of the
    pnpm invocation. ``runner`` is passed by a caller that already resolved it
    (so ``corepack enable`` runs once, not twice).

    Every step runs through :func:`_run_build_step`, which bounds and reaps the
    step's whole process GROUP, and the pinned pnpm is checked by
    :func:`_pin_mismatch` first. Both exist because of one incident: see those
    two docstrings, and note that this builder is shared with the updater, so a
    bound that only covered `lop mobile install` would leave `lop update`'s
    snapshot build unbounded.

    Every package-manager child here — the steps, and the ``corepack enable``
    :func:`_package_runner` may run — gets the ARMED environment
    (:func:`_package_manager_env`), built once at the top of this function so one
    owner supplies it to all of them. Before the pin is judged, a runner that
    cannot satisfy the pin is offered every other route this host has
    (:func:`_runner_or_refusal`), and only its refusal ends the build.
    """
    web_dir = _WEB_DIR if web_dir is None else web_dir
    env = _package_manager_env(web_dir)
    try:
        if runner is None:
            runner, missing = _package_runner(web_dir, env=env)
            if missing == "node":
                # Named with its REMEDY rather than with its mechanism. This is the one
                # refusal an operator meets on a fresh Linux or Windows box, and the
                # previous text ("the bundle needs a one-time `pnpm build`") named a
                # command that cannot be run without the thing that is missing —
                # measured in the Ubuntu and Mint containers, where `mobile.install`
                # failed with exactly that and the container reading could not say what
                # to install. The wheel ships the built bundle, so this is a source
                # checkout (a container, a dev machine), and Node is a one-time cost
                # there rather than a runtime dependency of the daemon.
                # THE REMEDY HAS TO BE ONE THE READER CAN ACTUALLY RUN (design round 2,
                # D7). This sentence used to lead with `apt install nodejs`, which was
                # measured wrong on the very host this branch's own leg runs on: Ubuntu
                # 24.04's archive package is `nodejs 18.19.1`, and Debian freezes it at
                # the distro release, so the operator runs the remedy and gets this
                # identical refusal back. What the reader needs is the version check and
                # the routes that give them a current Node.
                return (
                    # The remedy leads, because the reader's premise is that they have
                    # no node: telling them to run `node --version` first is telling
                    # them to run a command this arm exists because it is missing
                    # (design round 4, D18). The version requirement follows the
                    # instruction it constrains rather than preceding it.
                    "node is not installed, and the portal bundle is built once with "
                    "it: install a current Node from https://nodejs.org (or, if you "
                    "use nvm, `nvm install 22`), then re-run `lop mobile install`. "
                    "Node >=22 is required -- a distro package is often older, and "
                    "Ubuntu 24.04 ships 18"
                )
            if runner is None:
                # SAME DEFECT AS THE NODE ARM ABOVE (design round 2, D8), and this
                # sentence then repeated it in its own replacement (design round
                # 3, D14). The old text offered only `pnpm build` -- the command
                # that cannot run BECAUSE pnpm is missing -- and the first fix
                # named `corepack enable` with the assurance "it ships with
                # Node". That assurance is FALSE from Node 25 on: Corepack is no
                # longer distributed with it (the v26.9.0 tarball ships no
                # corepack file at all, while v24.21.0 ships bin/corepack
                # 0.36.0), so a reader on Current would run the remedy and get a
                # different refusal back -- exactly the class D8 was raised for.
                #
                # So the sentence names the ONE route that works on every Node:
                # pnpm's own install page. Corepack stays in the CODE below,
                # where it is genuinely useful -- `_shim_argv` found it, so the
                # self-heal runs -- but it is no longer promised in copy.
                return (
                    "neither pnpm nor corepack is on PATH, and the portal bundle "
                    "is built with pnpm: install pnpm "
                    "(https://pnpm.io/installation), then re-run "
                    "`lop mobile install`"
                )
        # Before any pnpm BUILD child is started: a runner whose version is not
        # the pin is the recursion's engine (see :func:`_pin_mismatch`), and
        # refusing here is the difference between an install that stops with a
        # sentence and one that stops when the machine runs out of memory. The one
        # probe this costs is `_pin_mismatch`'s own and it cannot fetch (see
        # :func:`_probe_env`); the routes that can supply the pin when PATH cannot
        # are :func:`_runner_or_refusal`'s business.
        runner, mismatch = _runner_or_refusal(runner, web_dir, env=env)
        if mismatch is not None:
            return mismatch
        for args in (["install", "--frozen-lockfile"], ["build"]):
            try:
                result = _run_build_step(
                    [*runner, *args], web_dir, timeout=_BUILD_STEP_TIMEOUT, env=env
                )
            except StepMemoryExceeded as exc:
                # Named with the STEP, not only with the guard's sentence: which
                # step was too big is half of what the operator has to act on
                # (`install` and `build` are different workloads, and the retry is
                # a different command for each), and the sentence is the other
                # half. Still inside the outer `try`, so a memory kill can never
                # escape this function as a traceback.
                return f"pnpm {' '.join(args)} failed: {exc}"
            if result.returncode != 0:
                # A failed `build` can leave a dist/ behind: vite writes it
                # before npm's `postbuild` guard judges it, and the guard
                # failing is the common case here. Any index.html is enough
                # for `_bundle_state` to call the install "built" and serve an
                # unstyled phone with no error, so a bundle that does not pass
                # the guard is removed rather than left reading as servable.
                # Only one that FAILS the guard is removed, though: a failure
                # in `tsc -b` happens before vite writes anything, and the
                # previous dist is still a good bundle.
                if _dist_index(web_dir).exists() and _verify_bundle(web_dir) is not None:
                    _discard_bundle(web_dir)
                return f"pnpm {' '.join(args)} failed: {_failure_detail(result)}"
    except (OSError, subprocess.SubprocessError) as exc:
        # ``SubprocessError`` rather than ``TimeoutExpired``: the two bounds
        # _run_build_step enforces are one shape (see StepMemoryExceeded), so a
        # step the ceiling stopped on a path that does not name its own step — the
        # `corepack enable` in _package_runner above all — is worded here rather
        # than raised out of an install.
        return f"bundle build failed: {exc}"
    if not _dist_index(web_dir).exists():
        return "build ran but dist/index.html is still missing"
    # Exit 0 is not proof the bundle is servable — see ``_verify_bundle``.
    error = _verify_bundle(web_dir)
    if error is not None:
        _discard_bundle(web_dir)
        return error
    return None


def _verify_bundle(web_dir: Path | None = None) -> str | None:
    """Run the bundle's own guard over a freshly built ``dist``.

    The guard ships with the web tree (``scripts/check-bundle.mjs``) and runs
    as npm's ``postbuild``, so ``pnpm build`` already fails on a degenerate
    bundle. The installer runs it AGAIN because it cannot assume the
    package.json it just used carries that hook: a source snapshot taken
    before the hook existed builds green and installs a stylesheet carrying no
    utilities at all — the silent half of this defect, where the phone renders
    unstyled, vite exits 0, and nothing in any log says why.

    No node, or an older tree without the script, is not an error here: there
    is nothing to run, and refusing to install would be worse than the blind
    spot. Returns an error string, or None.
    """
    web_dir = _WEB_DIR if web_dir is None else web_dir
    script = web_dir / "scripts" / "check-bundle.mjs"
    if not script.exists() or shutil.which("node") is None:
        return None
    try:
        result = subprocess.run(
            ["node", str(script)],
            cwd=web_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"bundle check failed: {exc}"
    if result.returncode == 0:
        return None
    return f"bundle check failed: {_failure_detail(result)}"


def snapshot_bundle(web_dir: Path) -> str:
    """Build a snapshot's web bundle in place, before it is installed.

    Returns the one-line status both updaters print. The host script
    ``~/.local/bin/lop-update`` has always done this for the trees it prepares;
    the in-package updater (``lop update --from-snapshot``) did NOT, so the
    snapshot it installed carried the web SOURCES and no ``dist/`` — the
    bundle globs matched nothing, every authed GET answered 503 "bundle not
    built", and the only repair was ``lop mobile install`` on that machine.

    Never raises and never fails the update. No Node is a documented skip:
    the daemon self-heals at ``lop mobile install`` on a host that has it, and
    the wording matches the host script's so the two paths read as one log.

    A snapshot that ARRIVES with a ``dist/`` is not thereby a snapshot with a
    usable one: the present bundle is put through the same guard
    ``ensure_bundle`` applies, and a refused one is dropped and rebuilt. The
    alternative — trusting the file's existence — installs a tree packed with
    a pre-fix (utility-less) stylesheet and reports success, which is the
    silent half of the defect this whole path exists to close.
    """
    state = _bundle_state(web_dir)
    if state == "built":
        rejected = _verify_bundle(web_dir)
        if rejected is None:
            return "already built"
        _discard_bundle(web_dir)
    if state == "missing-sources":
        return "skipped (no web sources in snapshot)"
    try:
        runner, missing = _package_runner(web_dir)
    except (OSError, subprocess.SubprocessError):
        # ``SubprocessError``, matching the twin in :func:`_build_bundle`: a memory
        # kill during ``corepack enable`` is a ``StepMemoryExceeded``, which is a
        # sibling of ``TimeoutExpired`` and NOT one, so the narrow catch let it
        # escape this function and arrive at ``lop update``'s snapshot step as a
        # traceback (agent review round 1, MAJOR-1 — the twin worded it, this one
        # raised). Every ``_package_runner``/``_run_build_step`` call site was
        # swept: the other two are :func:`_build_bundle`'s (already widened) and
        # :func:`_verify_bundle`'s, which runs a plain ``subprocess.run`` and so
        # cannot raise this at all.
        return "skipped (pnpm could not be prepared; build at `lop mobile install`)"
    if missing == "node":
        return "skipped (node not installed; build at `lop mobile install`)"
    if runner is None:
        return "skipped (neither pnpm nor corepack; build at `lop mobile install`)"
    error = _build_bundle(web_dir, runner)
    return "built" if error is None else f"FAILED ({error})"


def ensure_bundle(*, build: bool = True, web_dir: Path | None = None) -> tuple[bool, str]:
    """Guarantee the daemon has a UI to serve. (ok, detail-for-status).

    The three states, in the order a fresh machine hits them: a wheel ships
    dist and this is a no-op; a source checkout is buildable and we build
    it; a broken install has neither and we say so rather than serving the
    503 the daemon would show every authed GET.

    A dist that is PRESENT and fails the guard is not one of those three: it
    is what a failed build leaves behind (vite writes dist/ before npm's
    postbuild judges it), and ``_bundle_state`` reads any index.html as
    "built" — so trusting it here would report "bundle present" on every later
    install while the phone rendered unstyled. It is dropped and rebuilt
    instead.
    """
    state = _bundle_state(web_dir)
    # Defaulted HERE rather than in each helper: the helpers below take a real
    # path, and one that silently tolerates None is one that crashes on the
    # default install path instead — measured on the real CLI (TypeError on
    # ``_dist_dir(None)``), which only a run through `lop mobile install` sees.
    web_dir = _WEB_DIR if web_dir is None else web_dir
    if state == "built":
        rejected = _verify_bundle(web_dir)
        if rejected is None:
            return True, "bundle present"
        if not build:
            return False, rejected
        _discard_bundle(web_dir)
        state = "buildable"
    if state == "missing-sources":
        return False, "bundle and web sources both missing from the install"
    if not build:
        return False, "bundle missing (web sources present; build skipped)"
    error = _build_bundle(web_dir)
    if error is not None:
        return False, error
    return True, "built the web bundle"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def systemd_path() -> Path:
    """The Linux user unit this daemon is registered as."""
    return supervisors.systemd_unit_path(SYSTEMD_UNIT)


def task_record_path() -> Path:
    """On Windows, our own copy of the task definition we registered.

    Task Scheduler keeps its registration in its own store (the registry), not
    in a file we own, so this is a RECORD rather than the registration itself:
    it is what ``create_task`` was handed, written where a user can read it, and
    it is what makes "did an install ever run here?" answerable without a
    subprocess. The authoritative question is still asked of ``schtasks``.
    """
    return config_dir() / "supervisor" / "mobile-task.xml"


def log_path() -> Path:
    return log_dir() / "mobile.log"


def render_plist(port: int = DEFAULT_PORT) -> dict[str, object]:
    """The whole supervised-unit plan in one pure function — every consumer
    (install, status, tests) reads the same rendering.

    ``launchd_job`` rather than ``launchd_program``: it sets ``Program`` to the
    branded interpreter image and ``ProgramArguments[0]`` to this daemon's role
    label, which is what stops the four supervised daemons reading as one
    indistinguishable ``Local Operator`` row. macOS names a background item by
    the basename of ``ProgramArguments[0]``, so the pre-branding bare
    ``sys.executable`` is what made ``lop mobile install`` notify that 'python3
    is running in the background'. The trade-off that shape accepts is recorded
    in ``procname.launchd_job``; a machine where no link can be planted gets
    byte-for-byte this plist as it was before.

    ``Program`` is the STABLE shim on a machine with the generation layout, and
    the role label does not move with it — see ``procname.supervised_image`` for
    why a path inside this venv is unsafe for a unit launchd may restart.

    ``EnvironmentVariables`` CARRIES THE STORE, and its absence was a bug on all
    three platforms rather than a Windows one. ``install`` stores the portal
    password under ``config_dir()`` and the daemon reads it back with the same
    function, but a supervised process does NOT inherit the installer's
    environment: with ``LOCAL_OPERATOR_CONFIG_DIR`` set, the installer wrote a
    password into that store and the daemon looked in the default one, then
    refused to start with "no mobile password set. Run `lop mobile install`" —
    naming the store it had just written to. The Windows probe battery found it
    (`mobile.install`, PASS=20/FAIL=1) and it is the same latent bug on macOS and
    Linux. ``wakes`` and ``tunnels`` have always passed the store this way; this
    daemon is the one that did not.
    """
    return {
        "Label": LABEL,
        **procname.launchd_job(
            "local_operator.mobile.service",
            "--port",
            str(port),
            label=procname.branded_argv0(procname.LABEL_MOBILE, port=port),
        ),
        "RunAtLoad": True,
        # Restart on crash, throttled by launchd's own 10s floor; an
        # exit-code-2 (no password) stays down because KeepAlive keys on
        # successful exit only being false — crashes restart, refusals don't
        # flap.
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path()),
        "StandardErrorPath": str(log_path()),
        # SSE-holding daemons must not be App Nap'd into suspending timers.
        "ProcessType": "Interactive",
        "EnvironmentVariables": {CONFIG_DIR_ENV: str(config_dir())},
    }


def refresh_plist_if_stale() -> launchd.PlistRefresh:
    """Bring this daemon's LaunchAgent up to date, and restart it if it changed.

    THE REPAIR ``lop-update`` NEVER HAD FOR THIS DAEMON. ``install`` renders the
    current plist and nothing ever rewrote one written by an older build, so a
    daemon installed before branding shows a bare ``python3.14`` in Activity
    Monitor for the rest of its life — the colleague's symptom, measured on the
    operator's machine as ``com.local-operator.mobile.plist`` (mtime Sep 5)
    running ``python3.14 -m local_operator.mobile.service --port 4098``.

    Deliberately NARROW: it rewrites the plist and restarts the job, and does
    not touch the password, the Keychain or the web bundle, because the wheel
    already ships the bundle and the upgrade path must not pay for a rebuild —
    that is why this is not routed through ``install``.

    Never raises, and never acts outside the real home:
    ``launchd.is_own_plist`` refuses a redirected ``HOME`` because
    ``launchctl`` addresses the REAL user's session whatever the plist path
    says, so a sandboxed run would otherwise restart the operator's daemon.
    """
    name = "mobile"
    try:
        if supervisors.supervisor() != supervisors.LAUNCHCTL:
            # NOT "not-addressable": there is no plist on this platform to
            # repair at all. `is_supported()` was the wrong gate once it began
            # answering True on Linux and Windows, where this function would
            # otherwise have gone looking for a LaunchAgent that cannot exist.
            return launchd.PlistRefresh(name=name, kind="unsupported")
        path = plist_path()
        if not launchd.is_own_plist(path, LABEL):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        # The port comes off the plist being REPLACED: a repair must not move a
        # daemon someone installed on a non-default port back to the default.
        port = launchd.int_arg(launchd.load(path), "--port", DEFAULT_PORT)
        outcome = launchd.rewrite_if_stale(name=name, path=path, rendered=render_plist(port))
        if outcome.kind != "repaired":
            return outcome
        # bootout + bootstrap through the shared helper, NOT kickstart -k:
        # measured on macOS with a scratch label, a kickstart after a rewrite
        # restarts the job from launchd's in-memory definition and keeps running
        # the OLD argv. See :mod:`local_operator.launchd`.
        reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
        if not reloaded.ok:
            # Names the recovery, because the job is DOWN at this point: see
            # `launchd.reload_failure`.
            return reloaded.as_refresh_failure(name=name, path=path, recovery="lop mobile install")
        return outcome
    except Exception as exc:  # noqa: BLE001 — a repair must never fail an upgrade
        return launchd.PlistRefresh(name=name, kind="failed", detail=str(exc))


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


def _supervised_pid() -> int | None:
    """The pid this daemon's supervisor currently runs, or ``None``.

    launchd prints ``pid = <n>`` only while a process is alive behind the label;
    systemd answers ``MainPID`` (``0`` when the unit is not running). Task
    Scheduler publishes no pid through ``schtasks`` at all, which is why the
    Windows answer below is "the task reports Running" rather than a pid.

    ASKS NOTHING ABOUT A JOB THIS RUN DOES NOT OWN. The label is a fixed module
    constant while the plist path moves with ``$HOME``, so
    ``launchctl print gui/<uid>/<label>`` from a redirected home is a question
    about the OPERATOR's daemon — read-only, but it is still their job being
    inspected, and ``_our_daemon_listening`` is the path a sandboxed install
    takes to decide whether to skip its reload (QA round 2, Q-1: the last
    unguarded call left on this axis). The identity test lives HERE rather than
    in the caller so no future caller of this probe can reintroduce the read.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        if not launchd.is_own_plist(plist_path(), LABEL):
            return None
        printed = _launchctl("print", f"{_domain()}/{LABEL}")
        if printed.returncode != 0:
            return None
        match = re.search(r"pid = (\d+)", printed.stdout)
        return int(match.group(1)) if match else None
    if kind == supervisors.SYSTEMCTL:
        shown = supervisors.systemctl_user("show", "-p", "MainPID", "--value", SYSTEMD_UNIT)
        value = shown.stdout.strip() if shown.returncode == 0 else ""
        return int(value) if value.isdigit() and value != "0" else None
    return None


def _listening_pids(port: int) -> set[str] | None:
    """The pids listening on ``port``, or ``None`` when this box cannot be asked.

    ``lsof`` is the mechanism this repo already uses for exactly this question
    (``session/runtime/control.py``); a machine without it — a slim container —
    answers ``None``, and the caller says what it could not check instead of
    pretending it did.
    """
    binary = shutil.which("lsof")
    if binary is None:
        return None
    try:
        listeners = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [binary, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return set(listeners.stdout.split())


def _our_daemon_listening(port: int) -> bool:
    """True when the process bound to ``port`` is the one this unit starts.

    Health alone cannot answer this: a stale foreground daemon on the same
    port passes every check while the supervised one fails to bind. Ask the
    supervisor which pid it is running, then ask lsof who owns the port.

    macOS behaviour is unchanged — this is the same ``launchctl print`` + lsof
    pair it always was, and the same False when either says no. The two
    additions are the systemd arm (same question, ``MainPID``) and the
    no-lsof case, which used to escape as an uncaught ``FileNotFoundError`` out
    of ``install()`` and now falls back to the supervisor's own word, with the
    gap named in the install steps.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.SCHTASKS:
        # No pid to cross-check: Task Scheduler's own Running status is the
        # strongest signal schtasks gives, and health + the auth gate below
        # still have to pass before an install reports success.
        #
        # ``running is not False`` rather than a truth test, because the status
        # word is localized: an unrecognised spelling comes back as ``None``
        # rather than as a definite "not running", and treating that as a
        # refusal made this loop unable to ever pass on such a host — a daemon
        # that was up and serving was reported as "daemon did not come up
        # healthy". Where the state cannot be read, the two signals that CAN be
        # — the unauth health endpoint and the auth gate — are what decide.
        registered, running, _detail = supervisors.task_state(TASK_NAME)
        return registered and running is not False
    pid = _supervised_pid()
    if pid is None:
        return False
    owners = _listening_pids(port)
    if owners is None:
        return True
    return str(pid) in owners


def _domain() -> str:
    """``gui/<uid>`` — the launchd domain every caller feeds to ``launchctl``.

    The platform guard is INSIDE this function rather than at its call sites,
    each of which is behind ``if kind == supervisors.LAUNCHCTL``: ``os.getuid``
    does not exist off POSIX, so an unguarded call was an ``AttributeError``
    waiting for any arm that forgot the check, and a guard at the call site is
    invisible to a reader — and to the static scan that grades this branch —
    which cannot see that the arm is unreachable. Guarded here, the function is
    safe to call anywhere on its own merits.
    """
    if procstate.is_windows():
        raise RuntimeError("launchd domains exist only on macOS")
    return f"gui/{os.getuid()}"


def is_supported() -> bool:
    """Whether this machine has a user-level supervisor this installer can drive.

    Binary-guarded rather than platform-guarded (see
    :mod:`local_operator.supervisors`): on a Linux without systemd — Devuan,
    Alpine, most containers, WSL2 without systemd — the answer is False, which
    the entry points turn into a sentence naming the foreground command. Before
    this, that machine got ``FileNotFoundError: 'launchctl'`` out of the CLI and
    a Windows one got ``AttributeError: module 'os' has no attribute 'getuid'``,
    because only ``install()`` was guarded and ``uninstall``/``service_action``
    were not.
    """
    return supervisors.supervisor() is not None


def registration_present() -> bool:
    """Whether this platform's supervisor has a registration for this daemon.

    The unit FILE for launchd/systemd; a ``schtasks /Query`` for Windows, whose
    registration lives in Task Scheduler's own store. The subprocess is why this
    is a status-time call and not something any hot path may use.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        return plist_path().exists()
    if kind == supervisors.SYSTEMCTL:
        return systemd_path().exists()
    if kind == supervisors.SCHTASKS:
        registered, _running, _detail = supervisors.task_state(TASK_NAME)
        return registered
    return False


def render_systemd(port: int = DEFAULT_PORT) -> str:
    """The Linux user unit, in the shape all four daemons' units share.

    ``procname.supervised_image`` when this machine has the generation layout,
    else this interpreter: a unit is re-executed on every restart, and a path
    inside a tree a flip or a prune replaced is a daemon that dies at load
    (see that function for the 2026-09-15 incident).
    """
    image = procname.supervised_image() or Path(sys.executable)
    return supervisors.render_systemd_unit(
        description="Local Operator mobile daemon",
        # The image is quoted for systemd's own grammar, not shell quoting: this
        # is the same rule ``tunnels.install`` uses, hoisted into
        # ``supervisors.quoted``. Measured on systemd 255: an interpreter under
        # a path with a space produced a unit that could never start ("Command
        # /home/a is not executable").
        exec_start=(
            f"{supervisors.quoted(str(image))} -m local_operator.mobile.service --port {port}"
        ),
        # ``Restart=on-failure``: an exit-code-2 (no password) stays down so a
        # refused start does not flap, exactly what the plist's
        # KeepAlive{SuccessfulExit: false} buys on macOS.
        #
        # ``Environment=`` carries the STORE, the ``EnvironmentVariables``
        # analogue in the plist above and the same two reasons its siblings
        # document: the store is part of the contract, and an UNQUOTED
        # assignment truncates it at the first space — silently, so the daemon
        # would read a different store than the installer wrote to and refuse to
        # start.
        post_lines=[
            f"Environment={supervisors.quoted(f'{CONFIG_DIR_ENV}={config_dir()}')}",
            *supervisors.output_redirect_lines(log_path()),
        ],
    )


def render_task_xml(port: int = DEFAULT_PORT) -> str:
    """The Windows Task Scheduler task for this daemon.

    ``environment=`` CARRIES THE STORE. It used to say that this daemon's plist
    recorded no config dir either and the store was the default one — which was
    true of the plist and wrong about the consequence: no supervised process
    inherits the installer's environment, so an install from a store override
    registered a task whose daemon read the DEFAULT store, found no password
    there, and exited 2. The Windows battery reported exactly that
    ("no mobile password set. Run `lop mobile install`" naming the store the
    installer had just written to), and it is the same omission the plist and the
    unit had. Task Scheduler has no native environment element, so
    ``supervisors.render_task_xml`` expresses it through the launcher; see that
    function for the shape.
    """
    image = procname.supervised_image() or Path(sys.executable)
    return supervisors.render_task_xml(
        description="Local Operator mobile daemon (serve the phone portal)",
        image=str(image),
        argv=["-m", "local_operator.mobile.service", "--port", str(port)],
        environment={CONFIG_DIR_ENV: str(config_dir())},
        log=log_path(),
        user_id=supervisors.current_user_id(),
    )


def health(port: int = DEFAULT_PORT, timeout: float = 3.0) -> dict[str, object] | None:
    """Probe the daemon's unauthenticated liveness endpoint."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=timeout
        ) as response:
            return json.loads(response.read().decode())  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001 — health is a probe, absence is the answer
        return None


def gate_closed(port: int = DEFAULT_PORT, timeout: float = 3.0) -> bool:
    """Assert the AUTH gate, not mere liveness: a daemon that serves the API
    without a cookie is a boundary failure, however healthy its process."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/sessions")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 401
    except urllib.error.HTTPError as exc:
        return exc.code == 401
    except Exception:  # noqa: BLE001
        return False


def _serving(port: int) -> bool:
    """The supervised daemon is up, answering, and still gating the API.

    The triple rather than ``health`` alone, because a leftover FOREGROUND
    daemon on the same port passes a health probe while the supervised one
    fails to bind — see :func:`_our_daemon_listening`. Spelled once so the
    install's skip decision and its verification loop cannot drift apart.
    """
    return _our_daemon_listening(port) and health(port) is not None and gate_closed(port)


def _plist_is_current(path: Path, wanted: dict[str, object]) -> bool:
    """Whether the file already says exactly what ``wanted`` says.

    Content, not existence: a plist from an older build names a different
    interpreter and must still be replaced. An unreadable or unparseable file
    answers ``False``, which is the rewrite direction — see
    :func:`local_operator.wakes.install.ensure_supervisor_installed`, which
    repairs by content for the same reason.
    """
    try:
        return plistlib.loads(path.read_bytes()) == wanted
    except (OSError, ValueError):
        return False


def install(port: int = DEFAULT_PORT, *, dry_run: bool = False) -> dict[str, object]:
    """Idempotent one-shot: bundle, password (kept if present), unit, load, verify.

    One shared prefix — guarantee the UI bundle, generate/keep the password —
    then one arm per supervisor. The arms differ only in HOW the daemon is made
    to run at login; the verification at the end is shared, because "a unit file
    exists" was never the question.

    The BUNDLE comes first, before anything is written. It is the only step that
    can fail for a reason the operator has to go and fix (no Node on a source
    checkout), and running it after the password left a store mutated by an
    install that then reported failure — visible in the Ubuntu/Mint container
    readings, whose FAIL detail was the password progress line rather than the
    reason. A preflight that refuses changes nothing on disk.
    """
    steps: list[str] = []
    kind = supervisors.supervisor()
    if kind is None:
        return {"ok": False, "steps": [], "error": NO_SUPERVISOR_ERROR}

    # The UI is half the product. A missing bundle means every authed GET
    # 503s, so install builds it rather than leaving the phone on a dead
    # page — the wheel normally ships it, a source checkout does not.
    bundle_ok, bundle_detail = ensure_bundle(build=not dry_run)
    if not bundle_ok:
        # THE FAILURE TEXT IS THE ERROR, NOT ALSO A STEP (design round 2, D11).
        # Appending it here and repeating it in `error` printed the same
        # multi-line sentence twice -- once as progress, once as the red
        # failure -- so a reader saw one problem scroll past and then read the
        # second copy as a second one. Steps are what SUCCEEDED.
        return {"ok": False, "steps": steps, "error": bundle_detail}
    steps.append(bundle_detail)

    password = load_password()
    if password is None:
        password = generate_password()
        if not dry_run:
            store_password(password)
        steps.append(f"generated a new portal password ({store_description()})")
    else:
        steps.append(f"kept the existing portal password ({store_description()})")

    if kind == supervisors.LAUNCHCTL:
        plist_path().parent.mkdir(parents=True, exist_ok=True)
        # WRITE ONLY WHEN IT WOULD SAY SOMETHING NEW, and reload only when the
        # file changed or the daemon is not serving. `wakes.install` has
        # compared-then-skipped since it shipped; this is the same shape.
        #
        # WHAT THIS DOES AND DOES NOT CLAIM (review round 1, R-3 — an earlier
        # revision of this comment overclaimed): an install that would change
        # nothing no longer writes the plist or bounces the job, so the two
        # signals an EDR reads as "Persistence: launchd job / plist file
        # modification" (MITRE T1543.001) no longer come from THIS path. It is
        # NOT an explanation of the 2026-09-19 incident's plist modification:
        # that child was `[daemons] refresh`, whose repair goes through
        # `launchd.rewrite_if_stale` — which already returned "current" without
        # writing, pre-existing and not in this diff — and the plist it did
        # rewrite was genuinely stale (the pre-branding shape). A real repair,
        # not an identical-bytes rewrite.
        current = _plist_is_current(plist_path(), render_plist(port))
        if not dry_run and not current:
            plist_path().write_bytes(plistlib.dumps(render_plist(port)))
        steps.append(f"wrote {plist_path()}" if not current else "LaunchAgent already current")
        if not dry_run:
            # The reload, not a bare pair: it tolerates an absent job, waits for
            # launchd to release the label, retries the bootstrap past the
            # measured teardown race, and verifies the job is registered
            # afterwards — so the steps below are reporting a daemon that really
            # is loaded. See :mod:`local_operator.launchd`.
            #
            # SKIPPED WHEN THERE IS NOTHING TO LOAD: the file is already current
            # AND the supervised daemon is serving, which is the common case for
            # a re-run (see the write above for why that churn is an EDR signal,
            # not tidiness). A loaded-but-DEAD job is the state the old
            # unconditional reload used to repair, and it is still repaired
            # here: `kickstart` is the narrower operation, and it does not
            # briefly unregister the label — the precedent is
            # `wakes.install.ensure_supervisor_installed`. If the label is not
            # registered at all, kickstart fails and the full reload below runs,
            # which is exactly today's behaviour.
            reload_needed = True
            if current and _serving(port):
                reload_needed = False
                steps.append("LaunchAgent already current and serving; left it loaded")
            elif current and launchd.kickstart(label=LABEL, path=plist_path(), run=_launchctl):
                reload_needed = False
                steps.append("restarted the loaded LaunchAgent (its file was already current)")
            if reload_needed:
                reloaded = launchd.reload_job(label=LABEL, path=plist_path(), runner=_launchctl)
                if not reloaded.ok:
                    return {"ok": False, "steps": steps, "error": reloaded.detail[:300]}
                steps.append("loaded the LaunchAgent")
    elif kind == supervisors.SYSTEMCTL:
        loaded, detail = _install_systemd(port, dry_run=dry_run, steps=steps)
        if not loaded:
            return {"ok": False, "steps": steps, "error": detail}
    else:  # schtasks
        registered, detail = _install_task(port, dry_run=dry_run, steps=steps)
        if not registered:
            return {"ok": False, "steps": steps, "error": detail}

    if dry_run:
        steps.append("dry run: skipped load and verification")
        return {"ok": True, "steps": steps}

    # Shared verification: the supervisor must own the port (a stale foreground
    # daemon passes a bare health check while the supervised one fails to bind)
    # AND the auth gate must be closed — never installed-but-unauthenticated.
    deadline = time.time() + 20
    while time.time() < deadline:
        if _serving(port):
            steps.append("health check passed and the auth gate is closed")
            return {"ok": True, "steps": steps}
        time.sleep(0.5)
    return {
        "ok": False,
        "steps": steps,
        "error": f"daemon did not come up healthy; see {log_path()}",
    }


def _install_systemd(port: int, *, dry_run: bool, steps: list[str]) -> tuple[bool, str]:
    """Write and enable the Linux user unit. ``(ok, error)``.

    The unit file is written wherever ``systemd_path()`` says — that half is
    fully testable under a redirected home — while the user manager is only
    ADDRESSED when that is the path the real home owns. ``systemctl --user``
    has no sandbox: it reaches the calling user's live instance whatever
    ``$HOME`` says, so without the guard an isolated run would enable a real
    unit pointed at a store that vanishes when the sandbox ends. Same guard, and
    same incident, as the plist half in :mod:`local_operator.launchd`.
    """
    unit = systemd_path()
    unit.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        unit.write_text(render_systemd(port), encoding="utf-8")
    steps.append(f"wrote {unit}")
    if dry_run:
        return True, ""
    if not supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
        return False, (
            f"{unit} is not the unit the real home owns; refusing to enable it "
            "from a redirected home"
        )
    # Lingering BEFORE enable --now: without a user manager the enable itself
    # fails with the bus error, and enabling linger is what spawns one.
    # Best-effort, so a refusal does not fail the install.
    if supervisors.enable_linger():
        steps.append("enabled lingering so the daemon survives logout and reboot")
    else:
        steps.append(
            "could not enable lingering; the daemon may not survive logout "
            f"(run: loginctl enable-linger {os.environ.get('USER', '$USER')})"
        )
    supervisors.systemctl_user("daemon-reload")
    loaded = supervisors.systemctl_user("enable", "--now", SYSTEMD_UNIT)
    if loaded.returncode:
        return False, supervisors.translate_systemctl_error(loaded.stderr)
    steps.append(f"enabled the systemd user service ({SYSTEMD_UNIT})")
    return True, ""


def _install_task(port: int, *, dry_run: bool, steps: list[str]) -> tuple[bool, str]:
    """Register and start the Windows scheduled task. ``(ok, error)``.

    The XML is written under the config root first and kept: Task Scheduler
    stores its copy in the registry, so this is the only readable record of what
    was registered, and it is what makes a later diff possible.
    """
    xml = render_task_xml(port)
    record = task_record_path()
    if not dry_run:
        record.parent.mkdir(parents=True, exist_ok=True)
        # Task Scheduler has no stdout redirection, so the log the other two
        # platforms' supervisors create comes from the task's own command line
        # (see supervisors.render_task_xml): this is what keeps log_path() the
        # one place every log surface points at on every platform.
        log_path().parent.mkdir(parents=True, exist_ok=True)
        record.write_text(xml, encoding="utf-8")
    steps.append(f"wrote the task definition to {record}")
    if dry_run:
        return True, ""
    ok, detail = supervisors.create_task(TASK_NAME, xml)
    if not ok:
        return False, f"schtasks could not register the task: {detail}"
    steps.append(f"registered the scheduled task ({TASK_NAME})")
    started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
    if started.returncode:
        return False, (started.stderr or started.stdout or "").strip()[:300] or (
            "schtasks could not start the task"
        )
    steps.append("started the task")
    return True, ""


def uninstall(*, purge: bool = False, dry_run: bool = False) -> dict[str, object]:
    """Stop and deregister this daemon on whichever platform it was installed.

    EVERY verb is guarded here, not only ``install``: ``uninstall`` used to
    call ``launchctl`` (and ``_domain()``, which needs ``os.getuid``) with no
    platform check, so on Linux it raised ``FileNotFoundError: 'launchctl'``
    and on Windows ``AttributeError: module 'os' has no attribute 'getuid'`` —
    a stack trace where the user asked to remove something. An unsupported
    platform now removes what this installer could have written and says so.
    """
    steps: list[str] = []
    kind = supervisors.supervisor()
    if not dry_run:
        if kind == supervisors.LAUNCHCTL:
            _launchctl("bootout", _domain(), str(plist_path()))
            plist_path().unlink(missing_ok=True)
            steps.append("removed the LaunchAgent")
        elif kind == supervisors.SYSTEMCTL:
            if supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
                supervisors.systemctl_user("disable", "--now", SYSTEMD_UNIT)
                steps.append("disabled the systemd user service")
            else:
                steps.append("left the unit loaded: not addressable from a redirected home")
            systemd_path().unlink(missing_ok=True)
            steps.append(f"removed {systemd_path()}")
        elif kind == supervisors.SCHTASKS:
            # ``/Delete`` deregisters the task WITHOUT interrupting a running
            # program — Microsoft's own wording for the verb is "This command
            # doesn't delete the program that the task runs or interrupt a
            # running program" — so an uninstall that only deletes leaves the
            # portal daemon serving on its port with the password still loaded.
            # ``/End`` first is the Windows spelling of the ``bootout`` the
            # launchd arm issues and of Linux's ``disable --now``; it is the
            # shape ``service_action`` already uses for ``stop``.
            supervisors.schtasks(*supervisors.task_end_args(TASK_NAME))
            deleted, detail = supervisors.delete_task(TASK_NAME)
            if not deleted:
                # REFUSED rather than quietly successful: the daemon is still
                # registered, so reporting ``ok: True`` told a caller the
                # opposite of the state it left behind.
                steps.append(detail)
                return {"ok": False, "steps": steps, "error": detail}
            steps.append(f"removed the scheduled task ({detail})")
            task_record_path().unlink(missing_ok=True)
        else:
            # REFUSED rather than quietly successful: nothing here could have
            # registered this daemon, so there is nothing to remove, and the
            # step line carries the same sentence `install` gives. (The CLI's
            # uninstall prints `steps` and not `error`, which is why the refusal
            # is in both.)
            steps.append(NO_SUPERVISOR_ERROR)
            return {"ok": False, "steps": steps, "error": NO_SUPERVISOR_ERROR}
    if purge:
        if not dry_run:
            from local_operator.mobile.auth import delete_password

            delete_password()
        steps.append(f"deleted the portal password ({store_description()})")
    return {"ok": True, "steps": steps}


def service_action(action: str) -> dict[str, object]:
    """start|stop|restart, in each platform's own vocabulary.

    The launchd arm's kickstart family fails with "Could not find service" when
    the plist exists but the agent was never bootstrapped (or was booted out and
    not re-loaded). Bootstrap it on demand so the control commands work from
    whatever state launchd is in, not just the state `install` left behind.
    systemd has the same class of gap and closes it differently —
    ``systemctl --user start`` on an unloaded unit fails with "Unit not found",
    which is true and actionable, so no repair is invented here.
    """
    kind = supervisors.supervisor()
    if kind is None:
        return {"ok": False, "error": NO_SUPERVISOR_ERROR}
    if kind == supervisors.SYSTEMCTL:
        if not supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
            return {
                "ok": False,
                "error": (
                    f"{systemd_path()} is not the unit the real home owns; not "
                    "addressing the user manager from a redirected home"
                ),
            }
        result = supervisors.systemctl_user(action, SYSTEMD_UNIT)
        ok = result.returncode == 0
        return {
            "ok": ok,
            "error": "" if ok else supervisors.translate_systemctl_error(result.stderr)[:300],
        }
    if kind == supervisors.SCHTASKS:
        if action in ("stop", "restart"):
            # `/End` on a task that is not running exits non-zero, which for a
            # stop is the state the caller asked for; only a task that is not
            # REGISTERED is a real failure, and its stderr says so.
            ended = supervisors.schtasks(*supervisors.task_end_args(TASK_NAME))
            if ended.returncode:
                detail = (ended.stderr or ended.stdout or "").strip()
                if "running" not in detail.lower():
                    return {"ok": False, "error": detail[:300] or "schtasks could not stop it"}
        if action in ("start", "restart"):
            started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
            ok = started.returncode == 0
            return {
                "ok": ok,
                "error": "" if ok else ((started.stderr or started.stdout or "").strip()[:300]),
            }
        return {"ok": True, "error": ""}
    # THE LAUNCHD ARM BEGINS HERE (every arm above returned). `bootstrap
    # <domain> <plist>` is resolved by launchd to the Label INSIDE the file, so a
    # redirected home's `lop mobile restart` EVICTS and replaces the operator's
    # daemon rather than merely restarting it — the measurement
    # `browser_bridge._root_suffix` records — and `LABEL` is a fixed constant
    # here while `plist_path()` moves with `$HOME` (review round 2, R-8). The
    # same guard `reload_job` applies to this installer's install path, applied
    # to the verbs; the launchd counterpart of the systemd refusal above.
    if not launchd.is_own_plist(plist_path(), LABEL):
        return {"ok": False, "error": launchd.not_our_job_error(plist_path(), LABEL)}
    if action in ("start", "restart") and plist_path().exists():
        printed = _launchctl("print", f"{_domain()}/{LABEL}")
        if printed.returncode != 0:
            bootstrap = _launchctl("bootstrap", _domain(), str(plist_path()))
            if bootstrap.returncode != 0:
                return {"ok": False, "error": bootstrap.stderr.strip()[:300]}
    if action == "start":
        result = _launchctl("kickstart", f"{_domain()}/{LABEL}")
    elif action == "stop":
        result = _launchctl("kill", "SIGTERM", f"{_domain()}/{LABEL}")
    else:  # restart
        result = _launchctl("kickstart", "-k", f"{_domain()}/{LABEL}")
    ok = result.returncode == 0
    return {"ok": ok, "error": "" if ok else result.stderr.strip()[:300]}


def status(port: int = DEFAULT_PORT) -> dict[str, object]:
    """What a human (or `lop mobile status`) needs: install state, live
    health, gate state, registered sessions, log path."""
    from local_operator.session.runtime import registry

    probe = health(port)
    records = registry.scan()
    kind = supervisors.supervisor()
    return {
        "installed": registration_present(),
        # Which supervisor that answer came FROM, and where its registration
        # lives on this platform: "installed: yes" with no way to see whether it
        # is launchd, systemd or Task Scheduler is not a diagnosable state.
        "supervisor": kind or "none",
        "registration": str(_registration_path()) if _registration_path() else "",
        "password_set": load_password() is not None,
        "password_store": store_description(),
        "bundle": _bundle_state(),
        "healthy": probe is not None,
        "gate_closed": gate_closed(port),
        "health": probe,
        "port": port,
        "log": str(log_path()),
        "sessions": [
            {
                "pid": record.pid,
                "kind": record.kind,
                "session_id": record.session_id,
                "conversation_name": record.conversation_name,
                "model_label": record.model_label,
                "state": state,
            }
            for record, state in records
        ],
    }


def _registration_path() -> Path | None:
    """Where this platform's registration lives, for status output.

    ``None`` on a platform with no supervisor; on Windows it is the record this
    installer wrote, because Task Scheduler keeps its own copy in the registry.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        return plist_path()
    if kind == supervisors.SYSTEMCTL:
        return systemd_path()
    if kind == supervisors.SCHTASKS:
        return task_record_path()
    return None
