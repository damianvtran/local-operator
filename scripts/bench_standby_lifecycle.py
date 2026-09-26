#!/usr/bin/env python3
"""Standby lifecycle bench: does a warming console ALWAYS keep a spare in hand?

WHAT THIS PROVES
================
The supervisor (``local_operator/session/runtime/standby.py``) promises
invariant P: a warming console is, at every instant, in one of "a ready spare in
hand", "a warm in flight", "a refill scheduled at a deadline <= RETRY_CEIL_S".
The module's docstring names the five guards this replaces (S1-S6/S8); one of
them — "exited at once; not re-warming" — was reproduced on an isolated daemon
as 90 s with no replacement and the flock slot still held.

This script warms ONE real console (this process, ``daemon=True``: the desktop
daemon's slot) with real standby children, and drives every way the spare can be
lost:

  boot-warm               a spare must be READY within the readiness budget
                          (strict 10 s via ``--ready-budget 10``; the default
                          ``auto`` scales it to this host's load -- see below)
                          of ``enable_warming``.
  kill>5s -> replace      SIGTERM a spare that lived past REWARM_MIN_LIFE_S: a
                          replacement must FORK within warm_p95+1 s of the exit
                          and be READY within READY_BUDGET_S.
  kill<5s -> replace      kill a spare within REWARM_MIN_LIFE_S of its fork
                          (the reproduced dead-end): the refill must be
                          SCHEDULED and appear — never dropped.
  config-touch -> replace os.utime the root's ``config.yml``: the ready spare
                          retires itself as ``config-moved`` on its next poll
                          (<= 30 s) and the supervisor replaces it.
  adoption -> replace     a real engage (``AttachedSession.bind_runtime``)
                          adopts the ready spare; the pool must refill and a
                          spare become READY WITHOUT any further engage.
  next engage -> safe     a second engage must adopt a spare and must NEVER
                          signal the first chat's runtime (the measured defect:
                          the next engage SIGTERMed the still-live adopted
                          runtime ~1 s in, 7.8 s engage against 6.5 s cold).
  concurrent engages      TWO engages at the same instant (two threads released
                          by one barrier -- the daemon's own overlapping shape):
                          neither adopted runtime may be signalled, and neither
                          may appear in the retire ledger (agent review round 1,
                          B1); with two ready spares the loser of the first race
                          moves to the second, so BOTH engages adopt.
  daemon idle-keep        the daemon slot's spare carries
                          ``STANDBY_IDLE_ENV=0`` (keep warm; read back from the
                          live child's environment) and survives an idle window.
                          A TUI slot carries the 900 s window instead. (The
                          900 s reap is not raced — the policy is asserted where
                          it is decided, plus an idle window on the real child.)

THE INVARIANT, MEASURED: a 0.5 Hz census samples which standby children of this
console exist, plus the console's own pool, and the run fails when any interval
with NO spare (nothing ready and nothing warming) exceeds
``RETRY_CEIL_S + warm_p95``. "Replacement within warm_p95+1 s" is measured to
the FORK (the new child exists): readiness is asserted separately against the
readiness budget -- strict 10 s with ``--ready-budget 10``, the default ``auto``
scaled to this host's load at start -- and folding the warm into the fork bound
would charge it twice. Runtime liveness is read from the runtime's own Popen
EXIT STATUS, never from ``kill(pid, 0)``, which a zombie answers (Q1).
``warm_p95`` is computed from THIS run's observed warm times (max() for n<5, so
a small sample is not flattered).

ISOLATION
=========
Fresh ``HOME`` + ``LOCAL_OPERATOR_CONFIG_DIR`` under a session-unique temp root
(AGENTS.md "Isolating a run"), ``LOP_*``/``CMUX_*``/``XPC_FLAGS`` stripped,
notifications suppressed. Every process this script forks is reaped by exact
pid: the spares (ppid + ``-m <module>`` argv shape re-verified immediately
before each signal) and the engaged runtimes (their records, plus a direct
SIGTERM fallback). The root is removed at the end.

USAGE
=====
    env -u XPC_FLAGS .venv/bin/python scripts/bench_standby_lifecycle.py \\
        --json /tmp/lifecycle.json

``--measured-tree`` works as in the sibling scripts (``bench_tree``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bench_tree  # noqa: E402

_STRIPPED_PREFIXES = ("LOP_", "CMUX_")

#: The STRICT acceptance number: a READY spare within this of "boot" and of every
#: refill. The bench's DEFAULT scales it to this host's load at start
#: (``--ready-budget auto``): warm times of 14.7-34.8 s were measured at fleet
#: load >= 120 (agent review round 1, M1/Q2), and a red check for weather tells
#: nobody anything. Quote this number WITH the load whenever it is the claim.
READY_BUDGET_S = 10.0

#: Census cadence for the child samples and the dead-window measurement.
SAMPLE_S = 0.5

#: Backstop for event waits that should take seconds (a config retire is the
#: slowest at one stale-poll interval, <= 30 s).
EXIT_WAIT_S = 45.0


def _strip_inherited() -> None:
    for key in list(os.environ):
        if key.startswith(_STRIPPED_PREFIXES):
            del os.environ[key]
    # NIT1 (agent review round 1): this docstring said XPC_FLAGS is stripped and
    # only the usage line made that true. An inherited XPC_FLAGS breaks
    # getaddrinfo for CHILDREN (the fleet fact), so the console this bench IS
    # would hand it to every standby and runtime it forks; drop it here too.
    os.environ.pop("XPC_FLAGS", None)


def _load1() -> float:
    return round(os.getloadavg()[0], 1)


def _children(pid: int) -> dict[int, str]:
    """``pid -> command`` for every live child of ``pid`` (one wide ``ps``)."""
    out = subprocess.run(
        ["ps", "-eww", "-o", "pid=,ppid=,command="], capture_output=True, text=True
    ).stdout
    kids: dict[int, str] = {}
    for line in out.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 3:
            continue
        try:
            child, parent = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if parent == pid:
            kids[child] = fields[2]
    return kids


def _standby_pids(console_pid: int) -> list[int]:
    """This console's live ``[standby]`` children, by exact pid.

    ``-eww`` and the ``-m <module>`` anchor, both mandatory (the same trap
    ``bench_standby_engage`` caught in round 8, R8-1): a bare substring match on
    a truncated column reports zero on a busy host, and matching anything that
    merely mentions the module name would signal the wrong process.
    """
    from local_operator.session.runtime import standby

    return sorted(
        child
        for child, command in _children(console_pid).items()
        if f"-m {standby.STANDBY_MODULE}" in command
    )


def _kill_standby(pid: int, console_pid: int) -> bool:
    """SIGTERM one pid, only after re-verifying it is OUR child and a standby."""
    from local_operator.session.runtime import standby

    command = _children(console_pid).get(pid, "")
    if f"-m {standby.STANDBY_MODULE}" not in command:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class _RuntimeLedger:
    """Every ``subprocess.Popen`` this process forks for a spare or a runtime.

    WHY (QA round 1, Q1): ``os.kill(pid, 0)`` answers happily for a
    dead-but-unreaped child (a zombie), so the old liveness check false-passed on
    zombies and false-failed once an unidentified observer reaped one first.
    Holding the Popen lets a check read the EXIT STATUS — ``None`` while running,
    ``0`` for the runtime's own clean exit, negative for a signal — which is the
    evidence the check's name promises, and it leaves exactly one owner to reap
    the child.

    SPARES ARE RECORDED ON PURPOSE: an adopted runtime STARTS as a standby — it
    is forked during warming, long before any engage — so the only moment its
    Popen is ours to keep is its spawn as a spare. Patches ``subprocess.Popen``
    for its lifetime; every call passes through, and only receivers whose argv
    names the standby or runtime module word are retained.

    ONE OWNER IS ALSO THE READING'S PRECONDITION (agent review round 2, NIT3):
    the exit status is reap-order-independent only while THIS process's held
    Popen is the reaper. A third-party reaper would make ``poll()`` answer ``0``
    (CPython maps the ECHILD dead-state to a clean exit), so a signalled runtime
    reaped elsewhere would read as clean. Nothing else in this bench waits on
    these pids — the ledger is the only reaper by construction — so the caveat is
    a precondition to preserve, not a hole to plug; if that ever changes, this
    is the paragraph to re-read.
    """

    #: The two module words the real spawns carry.
    _MODULE_WORDS = ("runtime.process", "runtime.standby")

    def __init__(self) -> None:
        self.popens: dict[int, subprocess.Popen[Any]] = {}
        self._real = subprocess.Popen

    def start(self) -> "_RuntimeLedger":
        subprocess.Popen = self._recording  # type: ignore[assignment]
        return self

    def stop(self) -> None:
        subprocess.Popen = self._real  # type: ignore[assignment]

    def __enter__(self) -> "_RuntimeLedger":
        return self.start()

    def __exit__(self, *exc: object) -> bool:
        self.stop()
        return False

    def _recording(self, *args: Any, **kwargs: Any) -> Any:
        pop = self._real(*args, **kwargs)
        argv = args[0] if args else kwargs.get("args") or []
        try:
            joined = " ".join(str(word) for word in argv)
        except TypeError:  # a str/bytes argv rather than a sequence of words
            joined = str(argv)
        if any(word in joined for word in self._MODULE_WORDS):
            self.popens[pop.pid] = pop
        return pop

    def status(self, pid: int | None) -> tuple[str, bool]:
        """``(evidence, ok)`` for one runtime pid.

        ``ok`` is False only for a SIGNAL (rc < 0) — the harm the check exists
        for. A clean self-exit (rc 0, no viewer/lease yet) is not a signal and
        passes with its evidence named; a pid with no Popen on file says so
        instead of guessing from a liveness read that cannot see zombies. The
        status is authoritative while this process's held Popen is the only
        reaper (the class docstring's NIT3 caveat).
        """
        if pid is None:
            return "no runtime pid recorded", False
        pop = self.popens.get(pid)
        if pop is None:
            return (
                f"status unavailable (no Popen captured; kill(pid,0)={_alive(pid)})",
                False,
            )
        rc = pop.poll()
        if rc is None:
            return f"alive at check (popen.poll()=None; kill(pid,0)={_alive(pid)})", True
        if rc == 0:
            return "exited cleanly before the check (popen rc=0) -- not signalled", True
        return f"SIGNALLED (popen rc={rc})", False

    def reap(self) -> None:
        """``poll()`` everything held, so a finished child is not left a zombie.

        A held Popen that nobody polls keeps its exited child in ``<defunct>`` —
        visible in the cleanup child map, and this bench's whole point is that
        "gone" is measured truthfully. Nobody else may reap these (the ledger is
        the single owner), so the bench does it here, once, at the end.
        """
        for pop in self.popens.values():
            try:
                pop.poll()
            except Exception:  # noqa: BLE001 — a reap is best-effort, never a verdict
                pass


def _warm_p95(warms: list[float]) -> float:
    """p95 of the warm times observed so far, ``max()`` for fewer than 5 samples.

    A small sample must not be flattered: with one observation, "p95" is that
    observation and the fork bound built on it stays honest.
    """
    if not warms:
        return 1.0
    ordered = sorted(warms)
    if len(ordered) < 5:
        return ordered[-1]
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _pool() -> list[Any]:
    from local_operator.session.runtime import standby

    return list(standby._POOL)


def _wait_ready(timeout: float) -> float | None:
    """Seconds until this console has a READY spare, or None on timeout."""
    from local_operator.session.runtime import standby

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if standby.adoption_possible():
            return time.monotonic() - t0
        time.sleep(0.1)
    return None


def _wait_ready_pid(exclude: set[int], timeout: float) -> tuple[int | None, float | None]:
    """Seconds until a spare OUTSIDE ``exclude`` is READY."""
    from local_operator.session.runtime import standby

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        for warm in _pool():
            if warm.proc.pid in exclude:
                continue
            if standby._ready(warm):
                return warm.proc.pid, time.monotonic() - t0
        time.sleep(0.1)
    return None, None


def _wait_new_standby(exclude: set[int], timeout: float) -> tuple[int | None, float | None]:
    """Seconds until a standby child appears that is not in ``exclude``."""
    console_pid = os.getpid()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        for pid in _standby_pids(console_pid):
            if pid not in exclude:
                return pid, time.monotonic() - t0
        time.sleep(0.1)
    return None, None


def _wait_pid_ready(pid: int, timeout: float) -> float | None:
    """Seconds until the SPECIFIC spare ``pid`` is READY, or None on timeout.

    Used where "a spare is ready" is not the claim: a replacement must be ready,
    and an older spare that was never killed would otherwise satisfy the check
    without the replacement ever warming.
    """
    from local_operator.session.runtime import standby

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        for warm in _pool():
            if warm.proc.pid == pid and standby._ready(warm):
                return time.monotonic() - t0
        time.sleep(0.1)
    return None


class _Census(threading.Thread):
    """0.5 Hz child census + pool/ready state, and the dead-window measurement."""

    def __init__(self, console_pid: int) -> None:
        super().__init__(name="bench-standby-census", daemon=True)
        self.console_pid = console_pid
        # NOT ``self._stop``: ``threading.Thread`` uses that name for its own
        # shutdown method, and an Event there breaks ``join()`` (TypeError).
        self._halt = threading.Event()
        self.rows: list[dict[str, Any]] = []
        self.warms: list[float] = []
        self.max_dead_s = 0.0
        self._seen_ready: set[int] = set()
        self._dead_since: float | None = None
        self._t0 = time.monotonic()

    def run(self) -> None:
        from local_operator.session.runtime import standby

        while not self._halt.wait(SAMPLE_S):
            now = time.monotonic()
            try:
                pool = list(standby._POOL)
            except Exception:  # noqa: BLE001 - a sample must never kill the census
                pool = []
            ready = standby.adoption_possible()
            for warm in pool:
                pid = warm.proc.pid
                if warm.ready and pid not in self._seen_ready:
                    self._seen_ready.add(pid)
                    self.warms.append(round(now - warm.spawned_at, 2))
            # "NEITHER READY NOR WARMING" is measured on ALIVE spares, not on the
            # pool: a dead spare sits in the pool until the supervisor's next tick
            # prunes it, and counting it as a live spare would hide exactly the
            # window this metric exists to catch.
            alive = [warm.proc.pid for warm in pool if warm.proc.poll() is None]
            if alive:
                self._dead_since = None
            else:
                if self._dead_since is None:
                    self._dead_since = now
                self.max_dead_s = max(self.max_dead_s, now - self._dead_since)
            self.rows.append(
                {
                    "t": round(now - self._t0, 2),
                    "load": _load1(),
                    "standby": _standby_pids(self.console_pid),
                    "pool": [warm.proc.pid for warm in pool],
                    "alive": alive,
                    "ready": ready,
                    "dead_s": round(self.max_dead_s, 2),
                }
            )

    def close(self) -> None:
        self._halt.set()
        self.join(timeout=3)


def _check(checks: dict[str, Any], name: str, ok: bool, **detail: Any) -> None:
    checks[name] = {"ok": bool(ok), **detail}
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail if detail else ''}", flush=True)


def _record(notes: dict[str, Any], name: str, **detail: Any) -> None:
    notes[name] = detail
    print(f"  ...   {name}  {detail}", flush=True)


def _kill_scoped_standbys(console_pid: int) -> None:
    for pid in _standby_pids(console_pid):
        _kill_standby(pid, console_pid)


def _cleanup(state: dict[str, Any], root: Path, ledger: "_RuntimeLedger | None" = None) -> None:
    from local_operator.session.runtime import standby

    console_pid = os.getpid()
    _kill_scoped_standbys(console_pid)
    try:
        from scripts.bench_cold_send_http import _kill_runtime
    except Exception:  # noqa: BLE001
        _kill_runtime = None  # type: ignore[assignment]
    for sid in state.get("sids", []):
        if _kill_runtime is not None:
            try:
                _kill_runtime(root, sid)
            except Exception:  # noqa: BLE001 - nothing left to clean is not a failure
                pass
    children = _children(console_pid)
    for pid in state.get("runtime_pids", []):
        if pid and pid in children and "runtime.process" in children[pid]:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    standby.reset_for_tests()
    # NIT2 (agent review round 1): the "never leaves a standby behind" rule has to
    # be SELF-ENFORCING, not a printed map everyone reads past. Wait out the
    # SIGTERMs just sent, then assert emptiness — this console's own children only,
    # since another session's standby is neither this rig's to kill nor to judge.
    leftover_wait = time.monotonic() + 10.0
    while time.monotonic() < leftover_wait and _standby_pids(console_pid):
        time.sleep(0.2)
    if ledger is not None:
        ledger.reap()
    leftovers = _standby_pids(console_pid)
    print(f"  ...   cleanup: children now {_children(console_pid)}", flush=True)
    assert not leftovers, f"the bench left standby children alive: {leftovers}"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--json", default="", help="write the full report here")
    parser.add_argument(
        "--idle-probe-s",
        type=float,
        default=20.0,
        help="idle window a daemon spare must survive (the 900 s reap is not raced)",
    )
    parser.add_argument(
        "--ready-budget",
        default="auto",
        help=(
            "seconds a spare must be READY within, after boot and after every loss. "
            "'auto' (the default) scales the strict 10 s acceptance to this host's "
            "load at start: max(10, 0.30 * load). A starved host warms in tens of "
            "seconds (14.7-34.8 s measured at load >= 120), and a red check for "
            "weather tells nobody anything -- but a PASS at auto is not the design "
            "claim: pass --ready-budget 10 for the strict number, and quote the load "
            "beside it"
        ),
    )
    parser.add_argument(
        "--spare-wait",
        type=float,
        default=90.0,
        help="seconds to wait for a spare before a phase that needs one (weather, not an assert)",
    )
    parser.add_argument("--measured-tree", default="", help="commit whose tree this ran against")
    args = parser.parse_args()

    # The readiness budget: 'auto' scales the strict acceptance to the load this
    # run starts under, and the source string travels into every check detail so
    # a PASS can never be quoted as the 10 s claim without the caveat attached.
    load_at_start = _load1()
    if str(args.ready_budget).strip().lower() == "auto":
        budget = max(READY_BUDGET_S, round(0.30 * load_at_start, 1))
        budget_source = f"auto: max(10, 0.30 * {load_at_start})"
    else:
        budget = float(args.ready_budget)
        budget_source = "explicit"

    tree = bench_tree.describe(args.measured_tree)
    print(bench_tree.format_banner(tree), flush=True)

    _strip_inherited()
    from local_operator.tui.notify import suppress_notifications_for_process

    suppress_notifications_for_process("standby lifecycle benchmark")

    base = Path(tempfile.mkdtemp(prefix="lop-standby-lifecycle-"))
    root = base / ".local-operator"
    root.mkdir(parents=True)
    os.environ["HOME"] = str(base)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)

    from local_operator.config import ConfigManager
    from local_operator.session.runtime import standby

    ConfigManager(config_dir=root).update_config({"hosting": "test", "model_name": "test-model"})

    checks: dict[str, Any] = {}
    notes: dict[str, Any] = {}
    state: dict[str, Any] = {"sids": [], "runtime_pids": [], "seen": set()}
    census: _Census | None = None

    print(
        f"console pid {os.getpid()} root {root} load {_load1()} "
        f"ready-budget {budget}s ({budget_source})",
        flush=True,
    )
    # The ledger is installed BEFORE the first fork and stays for this process's
    # lifetime (Q1): an adopted runtime STARTS as a standby spare — forked during
    # warming, long before any engage — so the only moment its Popen is ours to
    # keep is its spawn. Holding it is what lets the checks read an exit STATUS
    # later; nothing else in this bench reaps those children.
    ledger = _RuntimeLedger()
    ledger.start()
    try:
        # ---- boot-warm ----------------------------------------------------
        t0 = time.monotonic()
        standby.enable_warming(root, daemon=True)
        census = _Census(os.getpid())
        census.start()
        ready_s = _wait_ready(budget)
        _check(
            checks,
            "boot_warm_ready_within_budget",
            ready_s is not None,
            ready_s=None if ready_s is None else round(ready_s, 2),
            budget_s=budget,
            budget_source=budget_source,
            load=_load1(),
        )
        boot_pids = _standby_pids(os.getpid())
        state["seen"].update(boot_pids)
        _record(notes, "boot", pids=boot_pids, gap_s=round(time.monotonic() - t0, 2))

        # ---- kill>5s -> replace -------------------------------------------
        # Precondition, not an assert: on a starved host the warm itself can take
        # tens of seconds, and that weather must not read as a lifecycle failure.
        _record(notes, "pre-aged-kill", ready_after_s=_wait_ready(args.spare_wait))
        warm = next((w for w in _pool() if w.ready), None)
        if warm is not None:
            while time.monotonic() - warm.spawned_at < standby.REWARM_MIN_LIFE_S + 0.2:
                time.sleep(0.2)
            exclude = (
                set(state["seen"]) | {w.proc.pid for w in _pool()} | set(_standby_pids(os.getpid()))
            )
            killed = _kill_standby(warm.proc.pid, os.getpid())
            fork_pid, fork_s = _wait_new_standby(exclude, EXIT_WAIT_S)
            bound = _warm_p95(census.warms) + 1.0
            _check(
                checks,
                "aged_kill_replacement_forks_in_time",
                killed and fork_pid is not None and fork_s is not None and fork_s <= bound,
                killed=killed,
                fork_pid=fork_pid,
                fork_s=None if fork_s is None else round(fork_s, 2),
                bound_s=round(bound, 2),
                load=_load1(),
            )
            if fork_pid is not None:
                state["seen"].add(fork_pid)
            ready_after = _wait_pid_ready(fork_pid, budget) if fork_pid else None
            _check(
                checks,
                "aged_kill_replacement_ready_in_budget",
                ready_after is not None,
                ready_pid=fork_pid,
                ready_s=None if ready_after is None else round(ready_after, 2),
                budget_s=budget,
                budget_source=budget_source,
                load=_load1(),
            )
        else:
            _check(
                checks, "aged_kill_replacement_forks_in_time", False, error="no ready spare to kill"
            )

        # ---- kill<5s -> replace (the reproduced dead-end) ------------------
        _record(notes, "pre-young-kill", ready_after_s=_wait_ready(args.spare_wait))
        warm = next((w for w in _pool() if w.ready), None)
        young_pid: int | None = None
        young_age: float | None = None
        if warm is not None:
            # EVERY pid this console has a spare on, snapshotted BEFORE the kill:
            # a replacement is by definition not in here, so "a new pid appeared"
            # cannot be satisfied by a spare that was already there. (The snapshot
            # must precede the kill: the supervisor can fork within a tick.)
            pre_kill = (
                set(state["seen"]) | {w.proc.pid for w in _pool()} | set(_standby_pids(os.getpid()))
            )
            age = time.monotonic() - warm.spawned_at
            if age < standby.REWARM_MIN_LIFE_S:
                young_pid, young_age = warm.proc.pid, age
                _kill_standby(warm.proc.pid, os.getpid())
            else:
                # The ready spare is old: kill it (>5 s class, already covered) and
                # young-kill its fresh replacement instead — exactly the state the
                # old guard ended its loop on.
                _kill_standby(warm.proc.pid, os.getpid())
                fork_watched_at = time.monotonic()
                fresh, _dt = _wait_new_standby(pre_kill, EXIT_WAIT_S)
                if fresh is not None:
                    # Age as measured from the moment the fork was first observed;
                    # the kill lands well inside REWARM_MIN_LIFE_S either way.
                    young_pid = fresh
                    young_age = time.monotonic() - fork_watched_at
                    _kill_standby(fresh, os.getpid())
            young_expected = pre_kill | ({young_pid} if young_pid else set())
        else:
            young_expected = set()
        if young_pid is not None:
            state["seen"].add(young_pid)
            got_pid, got_s = _wait_new_standby(young_expected, EXIT_WAIT_S)
            bound = max(_warm_p95(census.warms) + 1.0, 2.0)
            _check(
                checks,
                "young_kill_replacement_is_scheduled_and_forks",
                got_pid is not None and got_s is not None and got_s <= bound,
                young_pid=young_pid,
                young_age_s=None if young_age is None else round(young_age, 2),
                fork_pid=got_pid,
                fork_s=None if got_s is None else round(got_s, 2),
                bound_s=round(bound, 2),
                load=_load1(),
            )
            if got_pid is not None:
                state["seen"].add(got_pid)
            ready_after = _wait_pid_ready(got_pid, budget) if got_pid else None
            _check(
                checks,
                "young_kill_replacement_ready_in_budget",
                ready_after is not None,
                ready_pid=got_pid,
                ready_s=None if ready_after is None else round(ready_after, 2),
                budget_s=budget,
                budget_source=budget_source,
                load=_load1(),
            )
        else:
            _check(checks, "young_kill_replacement_is_scheduled_and_forks", False, error="no spare")

        # ---- adoption -> replacement, and the next engage must be safe ------
        _record(notes, "pre-adoption", ready_after_s=_wait_ready(args.spare_wait))
        # ---- adoption -> replacement, and the next engage must be safe ------
        _adoption_scenarios(root, checks, notes, state, budget, args.spare_wait, ledger)

        # ---- concurrent engages (B1 regression) -----------------------------
        _concurrent_engage_scenario(root, checks, notes, state, budget, args.spare_wait, ledger)

        # ---- config-touch -> retire -> replace ------------------------------
        _record(notes, "pre-config-touch", ready_after_s=_wait_ready(args.spare_wait))
        warm = next((w for w in _pool() if w.ready), None)
        if warm is not None:
            victim = warm.proc.pid
            cfg = root / "config.yml"
            # Snapshot BEFORE the touch: the replacement is forked the moment the
            # supervisor prunes the retired spare, which can be within its 0.25 s
            # tick — possibly before this loop notices the exit. A snapshot taken
            # after the exit would contain the replacement and the wait below
            # would then see nothing new.
            pre_touch = set(state["seen"]) | set(_standby_pids(os.getpid())) | {victim}
            os.utime(cfg, None)
            t_touch = time.monotonic()
            exited = False
            while time.monotonic() - t_touch < EXIT_WAIT_S:
                if not _alive(victim) or victim not in _children(os.getpid()):
                    exited = True
                    break
                time.sleep(0.2)
            fork_pid, fork_s = _wait_new_standby(pre_touch, EXIT_WAIT_S)
            bound = _warm_p95(census.warms) + 1.0
            _check(
                checks,
                "config_touch_retires_and_replaces",
                exited and fork_pid is not None,
                victim=victim,
                exited=exited,
                fork_pid=fork_pid,
                fork_s=None if fork_s is None else round(fork_s, 2),
                bound_s=round(bound, 2),
                load=_load1(),
            )
            if fork_pid is not None:
                state["seen"].add(fork_pid)
            ready_after = _wait_pid_ready(fork_pid, budget) if fork_pid else None
            _check(
                checks,
                "config_touch_replacement_ready_in_budget",
                ready_after is not None,
                ready_pid=fork_pid,
                ready_s=None if ready_after is None else round(ready_after, 2),
                budget_s=budget,
                budget_source=budget_source,
                load=_load1(),
            )
        else:
            _check(checks, "config_touch_retires_and_replaces", False, error="no ready spare")

        # ---- daemon idle-keep ------------------------------------------------
        _wait_ready(args.spare_wait)
        idle_daemon = standby._idle_for(standby.SLOT_DAEMON)
        idle_tui = standby._idle_for(standby.SLOT_TUI)
        live = _standby_pids(os.getpid())
        env_text = ""
        if live:
            from local_operator.session.runtime.reclaim import pid_environment

            env_text = pid_environment(live[0])
        _check(
            checks,
            "daemon_slot_policy_is_keep_warm",
            idle_daemon <= 0 and idle_tui == standby.IDLE_REAP_S,
            idle_daemon_s=idle_daemon,
            idle_tui_s=idle_tui,
        )
        _check(
            checks,
            "daemon_spare_carries_keep_warm_to_the_child",
            f"{standby.STANDBY_IDLE_ENV}=0" in env_text,
            pid=live[0] if live else None,
            env_has=standby.STANDBY_IDLE_ENV,
        )
        before_idle = set(_standby_pids(os.getpid()))
        time.sleep(args.idle_probe_s)
        still = sorted(before_idle & set(_standby_pids(os.getpid())))
        _check(
            checks,
            "daemon_spare_survives_the_idle_window",
            bool(still),
            idle_s=args.idle_probe_s,
            alive_pids=still,
            load=_load1(),
        )

        # ---- the invariant itself, from the census ---------------------------
        assert census is not None
        warm_p95 = _warm_p95(census.warms)
        dead_bound = standby.RETRY_CEIL_S + warm_p95
        _check(
            checks,
            "no_dead_window_beyond_ceiling_plus_warm_p95",
            census.max_dead_s <= dead_bound,
            max_dead_s=round(census.max_dead_s, 2),
            bound_s=round(dead_bound, 2),
            warm_p95_s=round(warm_p95, 2),
            samples=len(census.rows),
        )

        report = {
            "tree": tree,
            "ready_budget_s": budget,
            "ready_budget_source": budget_source,
            "load_at_start": load_at_start,
            "checks": checks,
            "notes": notes,
            "warms_s": census.warms,
            "warm_p95_s": round(warm_p95, 2),
            "census_samples": len(census.rows),
            "timeline": census.rows,
        }
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        failures = [name for name, value in checks.items() if not value.get("ok")]
        verdict = f"{len(checks) - len(failures)}/{len(checks)} checks"
        print(
            f"\n{'FAIL' if failures else 'PASS'}: {verdict}"
            + (f" FAILED: {', '.join(failures)}" if failures else ""),
            flush=True,
        )
        return 1 if failures else 0
    except BaseException as exc:  # noqa: BLE001 - report and still clean up
        print(f"bench error: {exc!r}", flush=True)
        return 1
    finally:
        # The sampler stops FIRST: it forks ``ps`` every tick, and a cleanup that
        # runs beside it reports its own processes as leftovers.
        try:
            if census is not None:
                census.close()
        except Exception:  # noqa: BLE001 - closing the sampler must not break cleanup
            pass
        try:
            _cleanup(state, root, ledger)
        finally:
            shutil.rmtree(base, ignore_errors=True)


def _adoption_scenarios(
    root: Path,
    checks: dict[str, Any],
    notes: dict[str, Any],
    state: dict[str, Any],
    budget: float,
    spare_wait: float,
    ledger: _RuntimeLedger,
) -> None:
    """A real engage adopts; the replacement must arrive with no further engage;
    the next engage must adopt a spare and never signal the first runtime."""

    async def _run() -> None:
        from local_operator.mobile.attach_client import find_runtime_record
        from local_operator.session.attached import AttachedSession

        async def _no_takeover() -> None:
            raise RuntimeError("benchmark has no takeover")

        async def _engage() -> tuple[Any, str, int | None]:
            sid = uuid.uuid4().hex[:12]
            remote = await AttachedSession.cold(
                sid,
                config_dir=root,
                cwd=str(root.parent),
                takeover_factory=_no_takeover,
                surface="desktop",
            )
            await remote.bind_runtime()
            record, _owner = await asyncio.to_thread(find_runtime_record, root, sid)
            return remote, sid, getattr(record, "pid", None)

        _wait_ready(spare_wait)
        ready_before = {w.proc.pid for w in _pool() if w.ready}
        remote1, sid1, runtime1 = await _engage()
        state["sids"].append(sid1)
        if runtime1:
            state["runtime_pids"].append(runtime1)
        _check(
            checks,
            "first_engage_adopts_the_ready_spare",
            runtime1 is not None and runtime1 in ready_before,
            runtime_pid=runtime1,
            ready_pids=sorted(ready_before),
            load=_load1(),
        )
        # (E) WITHOUT another engage the pool must refill to the daemon depth and a
        # spare that was NOT ready before the engage must become READY. ``refilled``
        # alone is the D1 discriminator (the defect left no refill at all); the fresh
        # readiness is what the next engage's adoption depends on.
        t_adopt = time.monotonic()
        refilled = False
        while time.monotonic() - t_adopt < budget:
            if len(_pool()) >= 2:
                refilled = True
                break
            await asyncio.sleep(0.2)
        fresh_pid, fresh_s = _wait_ready_pid(ready_before, budget)
        _check(
            checks,
            "adoption_replacement_without_another_engage",
            refilled and fresh_pid is not None,
            refilled=refilled,
            fresh_ready_pid=fresh_pid,
            ready_before=sorted(ready_before),
            ready_s=None if fresh_s is None else round(fresh_s, 2),
            load=_load1(),
        )

        # (F) the next engage: adopts a spare, and must not touch the first runtime.
        standbys_before = {w.proc.pid for w in _pool()} | set(_standby_pids(os.getpid()))
        remote2, sid2, runtime2 = await _engage()
        state["sids"].append(sid2)
        if runtime2:
            state["runtime_pids"].append(runtime2)
        # Q1 (agent review round 1): the verdict must come from the runtime's EXIT
        # STATUS, not from ``kill(pid, 0)`` — a dead-but-unreaped child answers
        # kill(0) happily, so the old read false-passed on zombies and false-failed
        # once an observer reaped one first (both seen on this head). The ledger has
        # held the runtime's Popen since before the engage, so the status survives
        # either way, and the evidence string names which reading decided.
        first_status, first_ok = ledger.status(runtime1)
        _check(
            checks,
            "second_engage_adopts_and_never_signals_the_first_runtime",
            runtime2 is not None and runtime2 in standbys_before and first_ok,
            runtime2=runtime2,
            adopted=runtime2 in standbys_before if runtime2 else False,
            first_runtime_pid=runtime1,
            first_runtime_status=first_status,
            first_runtime_evidence="popen exit status (rc<0 == signalled)",
            load=_load1(),
        )
        _record(notes, "adoption", runtime1=runtime1, runtime2=runtime2, sid1=sid1, sid2=sid2)

        await remote1.dispose()
        await remote2.dispose()

    asyncio.run(_run())


def _concurrent_engage_scenario(
    root: Path,
    checks: dict[str, Any],
    notes: dict[str, Any],
    state: dict[str, Any],
    budget: float,
    spare_wait: float,
    ledger: _RuntimeLedger,
) -> None:
    """B1 regression: TWO engages at the same instant on one warming console.

    The race needs two attempts overlapping inside ``try_adopt``. The daemon gets
    that shape for free (engages run via ``asyncio.to_thread``, one per session);
    here two threads each run their own loop, released by one barrier. At depth 2
    both spares can be ready, which is also the shape that proves the loser's
    fix: pre-fix, the loser of the first spare's handshake SIGTERMed the winner's
    runtime and fell back to a cold spawn; post-fix it moves to the second spare
    and BOTH engages adopt.

    The verdict reads two ledgers: per-runtime Popen exit status (never a
    signal) and the wholesale ``_retire`` ledger (no runtime pid may appear, no
    matter which thread or pass did the retiring).
    """
    from local_operator.mobile.attach_client import find_runtime_record
    from local_operator.session.attached import AttachedSession
    from local_operator.session.runtime import standby

    def _ready_pids() -> set[int]:
        return {w.proc.pid for w in _pool() if w.ready}

    _wait_ready(spare_wait)
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline and len(_ready_pids()) < 2:
        time.sleep(0.2)
    ready_before = _ready_pids()
    awaited_two = len(ready_before) >= 2

    retires: list[dict[str, Any]] = []
    real_retire = standby._retire

    def spy_retire(spare: Any) -> None:
        retires.append({"pid": spare.proc.pid, "alive_at_retire": spare.proc.poll() is None})
        return real_retire(spare)

    standby._retire = spy_retire
    results: list[dict[str, Any]] = []
    release = threading.Event()
    barrier = threading.Barrier(2)

    async def _engage_hold(slot: int) -> None:
        async def _no_takeover() -> None:
            raise RuntimeError("benchmark has no takeover")

        sid = uuid.uuid4().hex[:12]
        remote = await AttachedSession.cold(
            sid,
            config_dir=root,
            cwd=str(root.parent),
            takeover_factory=_no_takeover,
            surface="desktop",
        )
        await remote.bind_runtime()
        record, _owner = await asyncio.to_thread(find_runtime_record, root, sid)
        pid = getattr(record, "pid", None)
        results.append(
            {
                "slot": slot,
                "sid": sid,
                "runtime": pid,
                "adopted": pid in ready_before if pid else False,
            }
        )
        state.setdefault("sids", []).append(sid)
        if pid:
            state.setdefault("runtime_pids", []).append(pid)
        # Hold the remote — and this loop — until the caller has read the
        # evidence: the runtime's Popen lives in the ledger, and disposing the
        # remote on THIS loop is the only clean teardown.
        await asyncio.to_thread(release.wait, 60)
        await remote.dispose()

    def _thread(slot: int) -> None:
        barrier.wait(timeout=60)
        try:
            asyncio.run(_engage_hold(slot))
        except BaseException as exc:  # noqa: BLE001 — the verdict is the checks, not a crash
            results.append({"slot": slot, "error": repr(exc)})

    threads = [threading.Thread(target=_thread, args=(slot,)) for slot in (1, 2)]
    try:
        for thread in threads:
            thread.start()
        # Wait for both engages to have RECORDED their runtime — not for the
        # threads to END: each one holds its remote until ``release``, and the
        # evidence reads stronger while those remotes still hold the runtimes
        # up ("alive at check"), so it is read HERE and the release happens in
        # the finally. The old order joined first, which made every thread burn
        # its full hold-wait and read statuses only after ``dispose()`` (agent
        # review round 2, NIT5).
        wait_deadline = time.monotonic() + 240
        while time.monotonic() < wait_deadline and len(results) < 2:
            time.sleep(0.1)
        runtime_pids = [r["runtime"] for r in results if r.get("runtime")]
        statuses = {pid: ledger.status(pid) for pid in runtime_pids}
    finally:
        release.set()
        for thread in threads:
            thread.join(30)
        standby._retire = real_retire

    admitted = [r["runtime"] for r in results if r.get("adopted")]
    signalled = {pid: st for pid, (st, ok) in statuses.items() if not ok}
    retired = [r for r in retires if r["pid"] in set(runtime_pids)]
    _check(
        checks,
        "concurrent_engages_never_signal_an_adopted_runtime",
        len(runtime_pids) == 2
        and not signalled
        and not retired
        and (len(admitted) == 2 or not awaited_two),
        runtimes=runtime_pids,
        adopted=admitted,
        ready_before=sorted(ready_before),
        awaited_two_ready=awaited_two,
        statuses={str(pid): st for pid, (st, _ok) in statuses.items()},
        retire_ledger=retires,
        load=_load1(),
    )
    _record(notes, "concurrent-engage", results=results, retires=retires)


if __name__ == "__main__":
    sys.exit(main())
