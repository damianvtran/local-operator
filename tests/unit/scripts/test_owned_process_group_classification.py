"""Classification at the REAL syscall boundary, and what it costs a passing gate.

Every other authority test injects a ``signal_group`` that returns a verdict, so
the mapping from a real ``os.killpg`` failure to a verdict was exercised nowhere:
A1 (ESRCH read as refusal) was invisible to the suite. These tests keep the
classification inside the real module and move only the kernel facts.

The deletion controls assert on EXIT CODE and SIGNAL COUNT, never on a message:
a relabelled path would still pass a string match.
"""
from __future__ import annotations

import io
import os
import signal
import subprocess
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401


class World:
    """Synthetic kernel facts. Nothing here is a host capability."""

    def __init__(self, pgid=10, parent=1, birth="leader", scheme="fixture"):
        self.pgid, self.parent, self.birth, self.scheme = pgid, parent, birth, scheme
        self.rows = {parent: (0, parent, parent, "creator"), pgid: (parent, pgid, pgid, birth)}
        self.now = 0.0
        self.calls = []

    def clock(self):
        self.now += 0.001
        return self.now

    def census(self, pgid, deadline):
        return sorted(pid for pid, row in self.rows.items() if row[1] == pgid)

    def samples(self, pids):
        return {pid: owned.procstate.ProcessSample(False, self.rows[pid][3])
                for pid in pids if pid in self.rows}

    def identity(self, pid):
        return self.rows[pid][:3]

    def capture(self, *, send, pgid=None):
        return owned.OwnedGroup.capture(
            pgid or self.pgid, parent_pid=self.parent, census=self.census,
            samples=self.samples, identity=self.identity, scheme=lambda: self.scheme,
            clock=self.clock, signal_group=send)

    def forget(self, pgid=None):
        self.rows.pop(pgid or self.pgid, None)


def _free_process_group() -> int:
    """A process group that provably does not exist. Signal 0 delivers nothing."""
    for candidate in range(900_000, 900_400):
        try:
            os.killpg(candidate, 0)
        except ProcessLookupError:
            return candidate
        except PermissionError:
            continue
    pytest.skip("no free process-group id available for the ESRCH control")


@pytest.fixture
def wrapper(memory_wrapper, monkeypatch, tmp_path):
    """The real wrapper module, isolated scratch, real capability object."""
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return memory_wrapper


def _fake_process(poll_status):
    return NS(pid=10, poll=lambda: poll_status, wait=lambda **_: poll_status)


def _install(wrapper, monkeypatch, world, send, process):
    cap = world.capture(send=send)
    monkeypatch.setattr(wrapper, "owned_process_group",
                        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict))
    monkeypatch.setattr(wrapper, "subprocess",
                        NS(Popen=lambda *a, **k: process, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(wrapper, "_memory_budget",
                        lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture"))
    return cap


def test_real_killpg_esrch_is_observed_absent_not_a_refusal():
    """The classification under test is the REAL one; only the facts are fake."""
    pgid = _free_process_group()
    world = World(pgid=pgid, parent=os.getpid(), birth="leader")
    # Deliberately NOT injecting signal_group: this is os.killpg itself, and the
    # group it is asked about provably does not exist, so the kernel answers ESRCH.
    cap = world.capture(send=os.killpg)
    assert cap.last.status == "VERIFIED"
    verdict = cap.signal(signal.SIGKILL, deadline=time_budget())
    assert verdict.status == "OBSERVED_EMPTY", verdict.reason
    assert "ESRCH" in verdict.reason


def time_budget():
    import time
    return time.monotonic() + 1.0


def test_permission_error_from_the_checked_boundary_stays_a_refusal():
    world = World()

    def denied(_pgid, _signum):
        raise PermissionError(1, "Operation not permitted")
    cap = world.capture(send=denied)
    assert cap.signal(signal.SIGKILL, deadline=time_budget()).status == "REFUSED"


def test_a_passing_gate_keeps_rc_zero_when_the_group_is_already_gone(
        wrapper, monkeypatch):
    """A1's reachable window: the leader passed, the residual group exited first.

    RED on the reviewed head: the ESRCH was mapped to REFUSED, so a green gate
    returned 125 ("did not run"). The assertion is the EXIT CODE.
    """
    world = World()

    def gone(pgid, _signum):
        world.forget(pgid)
        raise ProcessLookupError(3, "No such process")
    _install(wrapper, monkeypatch, world, gone, _fake_process(0))
    output = io.StringIO()
    rc = wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=output)
    assert rc == 0, output.getvalue()


def test_deleting_the_identity_check_fails_on_exit_code_and_signal_count(wrapper, monkeypatch):
    """The mutation control: remove the identity check and BOTH assertions break.

    With the check: the final refresh sees a foreign birth and refuses, so no
    syscall is attempted and the passed gate reports 126 (cleanup unproven).
    Without it: the same run would signal a stranger (count 1) and return 0.
    """
    world = World()
    cap = _install(wrapper, monkeypatch, world, lambda *a: world.calls.append(a), _fake_process(0))

    def reaped():
        world.rows[world.pgid] = (world.parent, world.pgid, world.pgid, "FOREIGN")
        return 0

    wrapper.subprocess = NS(Popen=lambda *a, **k: NS(pid=10, poll=reaped,
                                                     wait=lambda **_: 0),
                            TimeoutExpired=subprocess.TimeoutExpired)
    output = io.StringIO()
    rc = wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=output)
    assert world.calls == [], "a lost identity must not reach the syscall"
    assert rc == 126, (rc, output.getvalue())


def test_a_refused_cleanup_after_a_nonzero_gate_keeps_the_child_status(wrapper, monkeypatch):
    """Only a PASSING gate is re-labelled; a failing one keeps its own code."""
    world = World()
    cap = world.capture(send=world.calls.append)
    world.rows[world.pgid] = (world.parent, world.pgid, world.pgid, "FOREIGN")
    monkeypatch.setattr(wrapper, "owned_process_group",
                        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict))
    monkeypatch.setattr(wrapper, "subprocess",
                        NS(Popen=lambda *a, **k: _fake_process(1),
                           TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(wrapper, "_memory_budget",
                        lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture"))
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=io.StringIO()) == 1
    assert world.calls == []
