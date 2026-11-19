"""Escaped descendants: cleanup is never `proven-empty` while one may live.

WHY THESE EXIST
---------------
lopdev fd36 (2026-11-19) saw an rc=124 run whose Chrome -- its own pgid AND
session -- outlived the reap while the completion record said
`cleanup=proven-empty`. Reproduced on origin/main a7ed05d55 with a child that
forks, `setsid()`s and sleeps: rc 124, `cleanup=proven-empty`, sleeper alive
afterwards under `systemd --user`. The owned-group certificate could not see a
process outside its group, and an orphan re-parented away from the wrapper was
outside every census.

The real-process cases drive the wrapper the way the incident did and assert on
what is alive afterwards; the synthetic cases pin the attribution and pid-reuse
rules without ever handing a synthetic pid to a real signal.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_operator import bounded_contract as contracts
from local_operator import escaped_descendants as ed

REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "scripts" / "run_bounded.py"

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="cleanup-required certifies only on Linux (/proc backend + subreaper)",
)

#: The incident shape: a setsid'd child outlives a leader that hits the bound.
_SETSID_ESCAPE = """
import os, sys, time
if os.fork() == 0:
    os.setsid()
    open(sys.argv[1], "w").write(str(os.getpid()))
    time.sleep(60)
    os._exit(0)
time.sleep(30)
"""

#: A daemon: the middle process exits at once, so the setsid'd grandchild is an
#: orphan before any sample could see its parent edge. Without the subreaper it
#: re-parents to init/systemd and no ppid walk from the wrapper reaches it.
_DOUBLE_FORK = """
import os, sys, time
if os.fork() == 0:
    if os.fork() == 0:
        os.setsid()
        open(sys.argv[1], "w").write(str(os.getpid()))
        time.sleep(60)
        os._exit(0)
    os._exit(0)
while not os.path.exists(sys.argv[1]):
    time.sleep(0.02)
"""


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().rpartition(")")[2].split()[0] not in ("Z", "X")
    except OSError:
        return False


def _run(
    tmp_path: Path, code: str, *args: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, object], int]:
    pidfile = tmp_path / "escapee.pid"
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    env["LOCAL_OPERATOR_SCRATCHPAD"] = str(tmp_path / "sp")
    env.pop("LOCAL_OPERATOR_RESOLVED_MEMORY_MB", None)
    escapee = 0
    try:
        with (tmp_path / "err").open("w") as err:
            done = subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER),
                    *args,
                    "--",
                    sys.executable,
                    "-c",
                    code,
                    str(pidfile),
                ],
                stdout=subprocess.DEVNULL,
                stderr=err,
                env=env,
                timeout=60,
                check=False,
            )
        stderr = (tmp_path / "err").read_text()
        escapee = int(pidfile.read_text())
        completion = next((tmp_path / "sp" / "run-bounded-runs").glob("*.completion.json"))
        row = json.loads(completion.read_text())
        return subprocess.CompletedProcess(done.args, done.returncode, "", stderr), row, escapee
    finally:
        # A leak test must not become a leak: reap the escapee by exact pid
        # whatever the wrapper did (it is gone already when the fix holds).
        if not escapee and pidfile.exists():
            escapee = int(pidfile.read_text() or 0)
        if escapee and _alive(escapee):
            os.kill(escapee, signal.SIGKILL)


@linux_only
@pytest.mark.slow
def test_a_setsid_escapee_is_reaped_before_the_bound_reports_proven_empty(tmp_path):
    done, row, escapee = _run(tmp_path, _SETSID_ESCAPE, "--timeout", "2", "--grace", "1")
    assert done.returncode == 124, done.stderr
    assert f"SIGKILL to escaped descendant(s) {escapee}" in done.stderr, done.stderr
    assert not _alive(escapee)
    assert row["cleanup_status"] == "proven-empty"
    assert row["descendant_cleanup"] == "proven-empty"
    assert row["escaped_residuals"] == []


@linux_only
@pytest.mark.slow
def test_a_double_forked_orphan_is_reaped_through_the_subreaper(tmp_path):
    done, row, escapee = _run(tmp_path, _DOUBLE_FORK, "--timeout", "20")
    assert done.returncode == 0, done.stderr
    assert f"SIGKILL to escaped descendant(s) {escapee}" in done.stderr, done.stderr
    assert not _alive(escapee)
    assert row["cleanup_status"] == "proven-empty"
    assert row["descendant_cleanup"] == "proven-empty"


@linux_only
@pytest.mark.slow
def test_an_in_group_orphan_corpse_does_not_block_the_terminal_census(tmp_path):
    # The subreaper makes the wrapper the parent of an in-group orphan; its
    # corpse must be reaped before the owned group's terminal census, or every
    # `cmd &` command would turn rc 0 into 126.
    code = (
        "import subprocess, sys; "
        "p = subprocess.Popen(['sleep', '60']); open(sys.argv[1], 'w').write(str(p.pid))"
    )
    done, row, escapee = _run(tmp_path, code, "--timeout", "20")
    assert done.returncode == 0, done.stderr
    assert not _alive(escapee)
    assert row["cleanup_status"] == "proven-empty"


# --------------------------------------------------------------------------
# Synthetic rows: attribution, pid reuse, and the honest-unproven branches.
# --------------------------------------------------------------------------

ROOT = 10_000_001
OWNED = 10_000_002
ESCAPEE = 10_000_003


def _row(pid, *, ppid=ROOT, pgid=None, sid=None, state="S", birth="100", threads=1):
    return ed.Row(
        pid=pid,
        state=state,
        ppid=ppid,
        pgid=pgid if pgid is not None else pid,
        sid=sid if sid is not None else pid,
        threads=threads,
        birth=birth,
    )


@pytest.fixture
def no_real_signals(monkeypatch):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(ed.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(ed.os, "waitpid", lambda pid, flags: (0, 0))
    return sent


def _tracker(subreaper=True, baseline=frozenset()):
    tracker = ed.Tracker(root=ROOT, owned_pgid=OWNED, subreaper=subreaper, own_sid=1)
    tracker.baseline_sids = baseline
    return tracker


def test_a_recycled_pid_is_never_signalled(monkeypatch, no_real_signals):
    scanned = _row(ESCAPEE, birth="100")
    monkeypatch.setattr(ed, "_scan", lambda: {ESCAPEE: scanned})
    # Same pid, different birth: the escapee died and the number was reused.
    monkeypatch.setattr(ed, "_read_row", lambda pid: _row(pid, birth="999"))
    result = _tracker().sweep(deadline=time.monotonic() + 0.1)
    assert no_real_signals == []
    assert not result.proven and result.remaining == [ESCAPEE]


def test_an_unchanged_escapee_is_signalled_and_then_proven_gone(monkeypatch, no_real_signals):
    scans = iter([{ESCAPEE: _row(ESCAPEE)}])
    monkeypatch.setattr(ed, "_scan", lambda: next(scans, {}))
    monkeypatch.setattr(ed, "_read_row", lambda pid: _row(pid))
    result = _tracker().sweep(deadline=time.monotonic() + 2)
    assert no_real_signals == [(ESCAPEE, signal.SIGKILL)]
    assert result.proven and result.signalled == [ESCAPEE]


def test_wrapper_session_owned_group_and_baseline_are_not_attributed(monkeypatch, no_real_signals):
    rows = {
        1: _row(1, sid=1),  # the wrapper's own helper (shares its session)
        2: _row(2, pgid=OWNED, sid=OWNED),  # owned group: OwnedGroup's authority
        3: _row(3, sid=77),  # a session the caller already had before spawn
        4: _row(4, ppid=3, sid=77),  # ...and its later children
    }
    monkeypatch.setattr(ed, "_scan", lambda: rows)
    result = _tracker(baseline=frozenset({77})).sweep(deadline=time.monotonic() + 1)
    assert no_real_signals == []
    assert result.proven


def test_no_subreaper_is_unproven_even_when_nothing_is_visible(monkeypatch, no_real_signals):
    monkeypatch.setattr(ed, "_scan", lambda: {})
    result = _tracker(subreaper=False).sweep(deadline=time.monotonic() + 1)
    assert not result.proven
    assert "subreaper" in result.reason


def test_an_unreadable_census_is_unproven_not_absent(monkeypatch, no_real_signals):
    def unreadable():
        raise ed.CensusUnknown("unreadable /proc row")

    monkeypatch.setattr(ed, "_scan", unreadable)
    result = _tracker().sweep(deadline=time.monotonic() + 0.1)
    assert not result.proven and "unreadable" in result.reason


def test_the_module_imports_without_sigkill_and_sweeps_unproven(monkeypatch, no_real_signals):
    # Windows has no signal.SIGKILL. The xplat probe imports every module there,
    # so nothing Unix-only may be evaluated at import (a default argument is).
    # Load a fresh copy with SIGKILL removed rather than reloading ``ed``, which
    # would swap the classes other tests in this file hold.
    import importlib.util

    monkeypatch.delattr(signal, "SIGKILL")
    name = "escaped_descendants_without_sigkill"
    spec = importlib.util.spec_from_file_location(name, ed.__file__)
    # spec_from_file_location returns ModuleSpec | None, and its loader is
    # Optional too. Asserting BOTH narrows them for the type checker rather than
    # leaving pyright to flag an unchecked optional at three call sites.
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)  # dataclasses resolve here
    spec.loader.exec_module(module)

    monkeypatch.setattr(module.os, "kill", lambda pid, sig: no_real_signals.append((pid, sig)))
    monkeypatch.setattr(module, "_scan", lambda: {ESCAPEE: _row(ESCAPEE)})
    tracker = module.Tracker(root=ROOT, owned_pgid=OWNED, subreaper=True, own_sid=1)
    tracker.baseline_sids = frozenset()
    result = tracker.sweep(deadline=time.monotonic() + 0.1)
    assert not result.proven and "SIGKILL" in result.reason
    assert no_real_signals == []


def test_a_threaded_zombie_leader_still_counts_as_live():
    assert not _row(ESCAPEE, state="Z", threads=3).dead
    assert _row(ESCAPEE, state="Z", threads=1).dead


def test_an_unproven_sweep_never_publishes_proven_empty(tmp_path):
    owner = type("Owner", (), {"accepted_terminal": lambda self, verdict: True})()
    result = ed.SweepResult(subreaper=True, remaining=[4242], reason="still alive")
    outcome = contracts.Outcome()
    outcome.command_exit_code = 0
    outcome.wrapper_exit_code = 0
    outcome.termination_cause = "child-exit"
    outcome.accept_terminal(owner, object(), escaped=result)
    receipt = contracts.Receipt(tmp_path, "unproven")
    row = receipt.publish("cleanup-required", outcome)
    assert row["cleanup_status"] == "unproven"
    assert row["descendant_cleanup"] == "unproven"
    assert row["escaped_residuals"] == [4242]
    assert "4242" in str(row["cleanup_reason"])


@linux_only
def test_a_pre_existing_foreign_session_child_is_left_alone(tmp_path):
    # An in-process caller (a test) may already have children in sessions of
    # their own; the baseline keeps the sweep from ever attributing them.
    before = subprocess.Popen(["sleep", "30"], start_new_session=True)
    after = None
    try:
        tracker = ed.Tracker.before_spawn(subreaper=False)
        tracker.bind(-1)
        after = subprocess.Popen(["sleep", "30"], start_new_session=True)
        tracker.sweep(deadline=time.monotonic() + 3)
        assert before.poll() is None, "baseline child was signalled"
        # The sweep reaped the corpse itself (it was our direct child), so
        # Popen cannot read a status; liveness is the assertion.
        assert not _alive(after.pid)
    finally:
        for proc in (before, after):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()


@linux_only
def test_subreaper_is_restored_for_in_process_callers():
    enabled, previous = ed.enable_subreaper()
    try:
        assert enabled
        assert Path(f"/proc/{os.getpid()}/status").exists()
    finally:
        ed.restore_subreaper(previous)
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c") or None)
    value = ctypes.c_int(-1)
    assert libc.prctl(ed.PR_GET_CHILD_SUBREAPER, ctypes.byref(value), 0, 0, 0) == 0
    assert bool(value.value) == bool(previous)
