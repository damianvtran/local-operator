#!/usr/bin/env python3
"""End-to-end reproduction of the orphaned-process-group reaper.

Drives the REAL ``execute_bash`` tool path (not a hand-rolled ``Popen``) to
spawn detached ``sleep`` groups, then exercises every branch of the reaper
against them, asserting liveness the hard way — ``os.killpg(pgid, 0)`` /
child ``wait()`` — so each result is proof the group truly lived or died
rather than a green unit assertion.

No 10-hour process is needed: every scenario is deterministic. Owner "hard
death" is simulated by writing a ledger keyed to a genuinely-dead pid (a
just-reaped child), which is byte-identical to what a SIGKILLed ``lop`` process
leaves behind — the ledger file outlives the process precisely because it is on
disk, which is the whole point of the mechanism.

Run (from the worktree, with its venv on PYTHONPATH so the new source is used):

    PYTHONPATH=/tmp/lop-reaper .venv/bin/python tests/e2e_group_reaper_repro.py

``PYTHONPATH`` here only forces the WORKTREE source to be exercised instead of
the installed package; it is a test-harness convenience, not a production
requirement. In production, register/sweep/kill all run in-process — there is no
child ``python -m local_operator`` that needs ``local_operator`` on its path (the
only children this script spawns are ``/bin/sh`` sleeps and, in scenario E, a
``python -c`` that imports from the same interpreter the parent already resolved).

A: hard death, owner DEAD  -> group survived owner, sweep REAPS it.
B: owner ALIVE             -> sweep does NOT reap (anti-regression, live runner).
C: PID-reuse decoys        -> alive-wrong-token and dead-pid both REAP;
   C-inverse: pgid recycled onto an innocent live group -> sweep DROPS, innocent lives.
D: soft death              -> kill_own_groups reaps own group, spares a sibling owner.
E: SIGINT vs SIGTERM scope -> a real child installs the soft-death hook and holds
   a live registered group; SIGINT (the headless-REPL turn abort) leaves the
   group ALIVE, a following SIGTERM (real termination) REAPS it. This is the R1
   regression: reaping on SIGINT would kill a live owner's background job.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import ToolContext
from local_operator.tools import builtin, group_reaper


def _token(pid: int) -> str:
    """A start token that is statically known non-``None`` for the caller.

    ``group_reaper._owner_start_token`` returns ``str | None`` (``None`` when the
    pid is gone or ``ps`` fails), but every call site here passes a pid that is
    demonstrably alive at the moment of the call, so a ``None`` is a genuine test
    failure rather than an expected branch. Asserting here keeps the type checker
    honest without scattering ``# type: ignore`` across the scenarios.
    """
    token = group_reaper._owner_start_token(pid)
    assert token is not None, f"no start token for live pid {pid}"
    return token


def _killpg_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # pgid recycled onto a process we cannot signal -> our group is gone.
        return False


def _dead_pid() -> int:
    """A genuinely-dead pid: spawn a trivial child and reap it."""
    child = subprocess.Popen(["/bin/sh", "-c", "true"])
    child.wait()
    return child.pid


async def _spawn_via_real_bash(config_dir: Path, cmd: str) -> tuple[AsyncJobManager, int]:
    """Spawn ``cmd`` detached through the REAL execute_bash path.

    Returns the job manager (owning the background job) and the pgid the reaper
    recorded in the owner's ledger — read back from disk to prove registration
    happened on the real path, not from a side channel.
    """
    manager = AsyncJobManager()
    ctx = ToolContext(cwd=str(config_dir), session_id="e2e", jobs=manager)
    tool = builtin.build_bash_tool()
    result = await tool.execute(
        "e2e-call",
        {"command": cmd, "background": True, "timeout": 300},
        None,
        None,
        ctx,
    )
    assert result.is_error is False, result
    # The ledger the reaper wrote under the REAL config dir for this process.
    my_pid = os.getpid()
    my_start = _token(my_pid)
    ledger = group_reaper._ledger_path(config_dir, my_pid, my_start)
    deadline = time.time() + 5
    while time.time() < deadline:
        if ledger.exists() and ledger.read_text().strip():
            break
        await asyncio.sleep(0.02)
    entries = group_reaper._read_ledger(ledger)
    assert entries, "no ledger line written by the real bash path"
    pgid = entries[-1]["pgid"]
    assert isinstance(pgid, int), f"ledger pgid was not an int: {pgid!r}"
    return manager, pgid


def _check(label: str, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        raise SystemExit(f"e2e assertion failed: {label}")


async def scenario_a(config_dir: Path) -> None:
    print("A) hard death, owner DEAD -> group survives owner, sweep REAPS")
    # Spawn through the real bash path; point config at the SAME dir the reaper
    # resolves for this process by temporarily overriding the env var.
    manager, pgid = await _spawn_via_real_bash(config_dir, "sleep 600")
    _check("group is alive after spawn (killpg 0 succeeds)", _killpg_alive(pgid))

    # Simulate the OWNER hard-dying: rewrite the ledger keyed to a dead pid,
    # exactly what a SIGKILLed lop leaves on disk. (Our real process stays
    # alive as the test driver, so we re-key rather than kill ourselves.)
    my_pid = os.getpid()
    my_start = _token(my_pid)
    live_ledger = group_reaper._ledger_path(config_dir, my_pid, my_start)
    entry = group_reaper._read_ledger(live_ledger)[-1]
    live_ledger.unlink()
    dead = _dead_pid()
    entry = {**entry, "owner_pid": dead, "owner_start": "dead-owner-token"}
    dead_ledger = group_reaper._ledger_path(config_dir, dead, "dead-owner-token")
    dead_ledger.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    _check("group SURVIVED the owner's death (still alive)", _killpg_alive(pgid))
    result = group_reaper.sweep_orphan_groups(config_dir)
    _check("sweep reaped exactly one group", result.reaped_groups == 1)
    time.sleep(0.1)
    _check(
        "group is REAPED after sweep (killpg raises ProcessLookupError)",
        not _killpg_alive(pgid),
    )
    _check("dead owner's ledger unlinked", not dead_ledger.exists())
    # Best-effort teardown: older AsyncJobManager builds expose ``shutdown``;
    # getattr keeps this forward/backward compatible without a static attr error.
    _shutdown = getattr(manager, "shutdown", None)
    if callable(_shutdown):
        _shutdown()


async def scenario_b(config_dir: Path) -> None:
    print("B) owner ALIVE -> sweep does NOT reap (live-runner anti-regression)")
    manager, pgid = await _spawn_via_real_bash(config_dir, "sleep 600")
    _check("group alive after spawn", _killpg_alive(pgid))
    # Owner (this process) is alive and its ledger is intact/matching.
    result = group_reaper.sweep_orphan_groups(config_dir, self_pid=os.getpid() + 9_000_000)
    _check("sweep reaped nothing", result.reaped_groups == 0)
    _check("sweep skipped a live owner", result.skipped_live_owners == 1)
    _check("group SURVIVES (live owner's long-runner untouched)", _killpg_alive(pgid))
    # cleanup
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    my_start = _token(os.getpid())
    group_reaper._ledger_path(config_dir, os.getpid(), my_start).unlink(missing_ok=True)


async def scenario_c(config_dir: Path) -> None:
    print("C) PID-reuse decoys -> both REAP; C-inverse -> innocent SURVIVES")

    # C1: owner_pid alive (this process) but start token WRONG -> reap (row 3).
    _, pgid1 = await _spawn_via_real_bash(config_dir, "sleep 600")
    my_pid = os.getpid()
    my_start = _token(my_pid)
    lg1 = group_reaper._ledger_path(config_dir, my_pid, my_start)
    e1 = group_reaper._read_ledger(lg1)[-1]
    lg1.unlink()
    wrong = group_reaper._ledger_path(config_dir, my_pid, "WRONG-TOKEN")
    wrong.write_text(json.dumps({**e1, "owner_start": "WRONG-TOKEN"}) + "\n", encoding="utf-8")

    # C2: owner_pid DEAD -> reap (row 1). Separate real group + dead owner.
    proc2 = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 600"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.1)
    pgid2 = os.getpgid(proc2.pid)
    leader2 = group_reaper._owner_start_token(pgid2)
    dead = _dead_pid()
    group_reaper._ledger_path(config_dir, dead, "d").write_text(
        json.dumps(
            {
                "pgid": pgid2,
                "owner_pid": dead,
                "owner_start": "d",
                "grp_leader_start": leader2,
                "cmd": "sleep 600",
                "ts": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = group_reaper.sweep_orphan_groups(config_dir, self_pid=my_pid + 9_000_000)
    _check("both decoys reaped (row1 dead-pid + row3 wrong-token)", result.reaped_groups == 2)
    time.sleep(0.1)
    _check("C1 alive-but-wrong-token group REAPED", not _killpg_alive(pgid1))
    _check("C2 dead-owner group REAPED", not _killpg_alive(pgid2))
    proc2.wait(timeout=2)

    # C-inverse: dead owner, but pgid now recycled onto an INNOCENT live group
    # whose leader-start differs -> sweep must DROP the line, innocent survives.
    innocent = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 600"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.1)
    innocent_pgid = os.getpgid(innocent.pid)
    dead2 = _dead_pid()
    group_reaper._ledger_path(config_dir, dead2, "d2").write_text(
        json.dumps(
            {
                "pgid": innocent_pgid,  # recycled onto the innocent group
                "owner_pid": dead2,
                "owner_start": "d2",
                "grp_leader_start": "STALE-DOES-NOT-MATCH-INNOCENT",
                "cmd": "sleep 600",
                "ts": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = group_reaper.sweep_orphan_groups(config_dir, self_pid=my_pid + 9_000_000)
    _check("C-inverse: sweep reaped nothing (victim gate held)", result.reaped_groups == 0)
    _check("C-inverse: innocent live group SURVIVES", _killpg_alive(innocent_pgid))
    try:
        os.killpg(innocent_pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    innocent.wait(timeout=2)


async def scenario_d(config_dir: Path) -> None:
    print("D) soft death -> kill_own_groups reaps own group, spares sibling owner")
    manager, my_pgid = await _spawn_via_real_bash(config_dir, "sleep 600")

    # Pre-seed a SIBLING owner (a different, live pid) with a live group.
    sibling = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 600"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.1)
    sib_pgid = os.getpgid(sibling.pid)
    sib_leader = _token(sib_pgid)
    sib_owner = os.getpid() + 5_000_000
    sib_ledger = group_reaper._ledger_path(config_dir, sib_owner, "sib")
    sib_ledger.write_text(
        json.dumps(
            {
                "pgid": sib_pgid,
                "owner_pid": sib_owner,
                "owner_start": "sib",
                "grp_leader_start": sib_leader,
                "cmd": "sleep 600",
                "ts": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    group_reaper.kill_own_groups(config_dir=config_dir)
    time.sleep(0.1)
    _check("own group REAPED by soft death", not _killpg_alive(my_pgid))
    _check("sibling owner's group SURVIVES (scope proof)", _killpg_alive(sib_pgid))
    my_start = _token(os.getpid())
    _check(
        "own ledger unlinked",
        not group_reaper._ledger_path(config_dir, os.getpid(), my_start).exists(),
    )
    _check("sibling ledger intact", sib_ledger.exists())
    try:
        os.killpg(sib_pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    sibling.wait(timeout=2)


# A real child that installs the soft-death hook, holds one live registered
# group, and models the headless REPL's SIGINT contract: Ctrl-C aborts the
# "turn" (caught KeyboardInterrupt) and KEEPS running rather than exiting. The
# parent drives real SIGINT then SIGTERM at it to prove the reaper's signal
# scope end-to-end, not just by inspecting the handler table.
_SIGINT_CHILD = r"""
import json, os, signal, subprocess, sys, time
from local_operator import cli
from local_operator.tools import group_reaper

# Own group so the grandchild is a real killable process group, exactly like a
# background bash job the user asked to keep alive across a turn abort.
child = subprocess.Popen(
    ["/bin/sh", "-c", "sleep 600"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
)
for _ in range(200):
    try:
        pgid = os.getpgid(child.pid); break
    except ProcessLookupError:
        time.sleep(0.01)
group_reaper.register_group(pgid, "sleep 600")
cli._install_group_reaper_soft_death()
# Hand the parent the pgid to watch, then flush so it is readable immediately.
sys.stdout.write(json.dumps({"pgid": pgid}) + "\n"); sys.stdout.flush()
# Model _run_headless_repl: SIGINT -> KeyboardInterrupt -> abort the turn, keep
# the session (and its background jobs) alive. Only a real exit reaps.
while True:
    try:
        time.sleep(3600)
    except KeyboardInterrupt:
        sys.stderr.write("turn aborted, session alive\n"); sys.stderr.flush()
"""


async def scenario_e(config_dir: Path) -> None:
    print("E) SIGINT (turn abort) spares live group; SIGTERM (real stop) REAPS")
    proc = subprocess.Popen(
        [sys.executable, "-c", _SIGINT_CHILD],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={**os.environ, "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir)},
    )
    assert proc.stdout is not None  # PIPE was requested above
    line = proc.stdout.readline()  # blocks until the child registered its group
    pgid = json.loads(line)["pgid"]
    time.sleep(0.1)
    _check("child's background group alive after spawn", _killpg_alive(pgid))

    # SIGINT: the headless turn abort. The group MUST survive and the child live.
    proc.send_signal(signal.SIGINT)
    time.sleep(0.3)
    _check("group ALIVE after SIGINT (Ctrl-C spared the background job)", _killpg_alive(pgid))
    _check("REPL child still running after SIGINT", proc.poll() is None)

    # SIGTERM: a real termination. The soft-death handler reaps, then chains.
    proc.terminate()
    proc.wait(timeout=5)
    time.sleep(0.2)
    _check("group REAPED after SIGTERM (real stop cleaned up)", not _killpg_alive(pgid))
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def main() -> None:
    # Real config dir override so execute_bash's register lands where the sweep
    # reads. Each scenario gets its own clean dir.
    for name, fn in (
        ("A", scenario_a),
        ("B", scenario_b),
        ("C", scenario_c),
        ("D", scenario_d),
        ("E", scenario_e),
    ):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td)
            os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(cfg)
            await fn(cfg)
        print()
    print("ALL SCENARIOS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
