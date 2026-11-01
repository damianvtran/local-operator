"""The reservation predicate: ownership by pinned group number, not by membership.

Every fake here is a SYNTHETIC kernel fact map — no pid reaches an OS probe and
no signal leaves this module. The worlds stub ``procstate.birth_scheme`` because
``ProcessSample.is_birth`` compares against the PLATFORM scheme: a fixture that
invents ``scheme="fixture"`` without stubbing it refuses every proof, which is
how the earlier harnesses silently tested nothing.
"""

from __future__ import annotations

import io
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from local_operator.procstate import ProcessSample
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401

SCHEME = "reservation-fixture"


class Kernel:
    """Synthetic /proc: rows of (parent, pgid, sid, birth), plus a corpse flag."""

    def __init__(self, leader=10, parent=1, birth="leader"):
        self.leader, self.parent = leader, parent
        self.rows = {
            parent: (0, parent, parent, "creator"),
            leader: (parent, leader, leader, birth),
        }
        self.corpses: set[int] = set()
        self.now = 0.0
        self.censuses = 0
        self.sent: list[tuple[int, int]] = []

    def clock(self):
        self.now += 0.001
        return self.now

    def census(self, pgid, deadline):
        self.censuses += 1
        return sorted(pid for pid, row in self.rows.items() if row[1] == pgid)

    def samples(self, pids):
        return {
            pid: ProcessSample(pid in self.corpses, self.rows[pid][3])
            for pid in pids
            if pid in self.rows
        }

    def identity(self, pid):
        return self.rows[pid][:3]

    def send(self, pgid, signum):
        self.sent.append((pgid, signum))
        self.rows.pop(pgid, None)

    def cap(self, *, send=None, leader=None):
        return owned.OwnedGroup.capture(
            leader or self.leader,
            parent_pid=self.parent,
            census=self.census,
            samples=self.samples,
            identity=self.identity,
            scheme=lambda: SCHEME,
            clock=self.clock,
            signal_group=send or self.send,
            deadline=self.clock() + 5.0,
        )


@pytest.fixture
def wrapper(request, monkeypatch, tmp_path):
    """The real wrapper module, resolved by name so nothing shadows the import."""
    module = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return module


@pytest.fixture(autouse=True)
def _platform_scheme(monkeypatch):
    """`is_birth` compares against the platform scheme — make them agree."""
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: SCHEME)


def test_a_zombie_leader_holds_the_pin_and_its_members_are_ours():
    """C6, the incident shape in miniature: leader is a corpse, a child lives."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "SIGNALED", verdict.reason
    assert verdict.anchor == "leader-reservation"
    assert kernel.sent == [(10, signal.SIGKILL)]
    assert {row["pid"] for row in cap.probe["proof"]} == {10, 20}


def test_a_reaped_leader_is_pin_lost_and_nothing_is_signalled():
    kernel = Kernel()
    cap = kernel.cap()
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")
    del kernel.rows[kernel.leader]  # reaped: the number is no longer reserved
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == "pin_lost"
    assert kernel.sent == []


def test_a_live_leader_is_live_leader_only_and_still_signals():
    kernel = Kernel()
    cap = kernel.cap()
    verdict = cap.signal(signal.SIGTERM, deadline=kernel.clock() + 5.0)
    assert verdict.status == "SIGNALED"
    assert verdict.anchor == "leader-live"


def test_c1_a_zombie_leader_the_census_cannot_see_is_refused_not_assumed():
    """Darwin scope: no visible corpse means no claimed reservation."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    original = kernel.census

    def without_leader(pgid, deadline):
        return [pid for pid in original(pgid, deadline) if pid != kernel.leader]

    cap.census = without_leader
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == "pin_lost"
    assert "live-leader-only" in verdict.reason
    assert kernel.sent == []


def test_a_pid_in_only_one_of_the_two_censuses_is_included_not_refused():
    """The union, not set-equality: a pid seen once was seen inside our number."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "transient")
    original = kernel.census
    calls = {"n": 0}

    def flickering(pgid, deadline):
        calls["n"] += 1
        found = original(pgid, deadline)
        return found if calls["n"] % 2 else [pid for pid in found if pid != 20]

    cap.census = flickering
    verdict = cap.signal(signal.SIGTERM, deadline=kernel.clock() + 5.0)
    assert verdict.status in {"SIGNALED", "OBSERVED_EMPTY"}, verdict.reason
    if verdict.status == "SIGNALED":
        assert 20 in verdict.members


def test_a_corpse_only_group_is_observed_empty_not_held_open():
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    verdict = cap.signal(signal.SIGTERM, deadline=kernel.clock() + 5.0)
    assert verdict.status == "OBSERVED_EMPTY", verdict.reason
    assert kernel.sent == []


def test_c2_sigchld_ignored_refuses_with_a_named_reason(monkeypatch):
    """No corpse, no reservation: refuse rather than pretend the pin exists."""
    kernel = Kernel()
    monkeypatch.setattr(owned.signal, "getsignal", lambda _sig: owned.signal.SIG_IGN)
    cap = kernel.cap()
    assert cap.last.status == "REFUSED"
    assert cap.last.refusal_class == "no_corpse_sigchld"
    assert kernel.sent == []


def test_c4_pid_one_and_our_own_group_are_never_signalled(monkeypatch):
    for pgid in (1, 4242):
        kernel = Kernel(leader=pgid, parent=1)
        cap = kernel.cap()
        monkeypatch.setattr(owned.os, "getpgrp", lambda pgid=pgid: pgid if pgid == 4242 else 99)
        monkeypatch.setattr(owned.os, "getpid", lambda: 77)
        verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
        assert verdict.status == "REFUSED", (pgid, verdict.reason)
        assert kernel.sent == []


RESIDENCY_REMOVED = (
    "if self._leader_reservation(deadline) is None:",
    "if False:  # mutation: residency re-check removed",
)


def _mutant():
    source = Path(owned.__file__).read_text()
    before, after = RESIDENCY_REMOVED
    assert before in source, "the residency re-check moved; update this control"
    module = ModuleType("owned_process_group_no_residency")
    module.__file__ = owned.__file__
    sys.modules[module.__name__] = module  # @dataclass resolves its module on 3.12
    exec(compile(source.replace(before, after), module.__name__, "exec"), module.__dict__)
    return module


def test_c3_removing_the_residency_re_check_lets_a_stranger_die():
    """The mutation must fail in the RIGHT direction — a stranger dies."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    original = kernel.samples

    def reaped_mid_proof(pids):
        # Proof passes; the corpse disappears exactly at the pre-syscall check.
        if kernel.sent == [] and capped["calls"] >= 3:
            return {}
        capped["calls"] += 1
        return original(pids)

    capped = {"calls": 0}
    cap.samples = reaped_mid_proof
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == "pin_lost"
    assert kernel.sent == []

    mutant = _mutant()
    kernel2 = Kernel()
    cap2 = mutant.OwnedGroup.capture(
        kernel2.leader,
        parent_pid=kernel2.parent,
        census=kernel2.census,
        samples=kernel2.samples,
        identity=kernel2.identity,
        scheme=lambda: SCHEME,
        clock=kernel2.clock,
        signal_group=lambda *a: kernel2.sent.append(a),
        deadline=kernel2.clock() + 5.0,
    )
    kernel2.corpses.add(kernel2.leader)
    capped2 = {"calls": 0}

    def reaped_mid_proof2(pids):
        if capped2["calls"] >= 3:
            return {}
        capped2["calls"] += 1
        return kernel2.samples(pids)

    cap2.samples = reaped_mid_proof2
    mutated = cap2.signal(signal.SIGKILL, deadline=kernel2.clock() + 5.0)
    assert mutated.status == "SIGNALED", mutated.reason
    assert kernel2.sent == [(10, signal.SIGKILL)]


def test_the_grace_loop_costs_no_census_and_signals_cost_at_most_two(
    wrapper, monkeypatch, tmp_path
):
    """Condition: census_passes <= 2 per signal, zero per grace tick."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "child")
    monkeypatch.setattr(
        wrapper,
        "owned_process_group",
        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict),
    )
    monkeypatch.setattr(
        wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
    )

    class Process:
        pid = 10
        polls = 0

        def poll(self):
            Process.polls += 1
            return 0

        def wait(self, **_):
            return 0

    monkeypatch.setattr(
        wrapper,
        "subprocess",
        NS(Popen=lambda *a, **k: Process(), TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    before = kernel.censuses
    out = io.StringIO()
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=out) == 0, out.getvalue()
    # One refresh proves, then each signal re-proves: two complete strict scans
    # per proof. Nothing here runs a census once per poll or per grace tick.
    assert kernel.censuses - before <= 2 * (1 + len(kernel.sent) + 1), (
        kernel.censuses - before,
        kernel.sent,
    )


def _wrapper_owner(wrapper, monkeypatch, cap, process):
    monkeypatch.setattr(
        wrapper,
        "owned_process_group",
        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict),
    )
    monkeypatch.setattr(
        wrapper,
        "subprocess",
        NS(Popen=lambda *a, **k: process, TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.setattr(
        wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
    )


def test_the_leader_is_not_reaped_before_the_terminal_census(wrapper, monkeypatch, tmp_path):
    """Design test 3: poll()/wait() must not run while the pin is still needed."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "child")

    class Process:
        pid = 10

        def poll(self):
            assert kernel.sent, "poll() would reap the leader before the census"
            return 0

        def wait(self, **_):
            assert kernel.sent, "wait() would reap the leader before the census"
            return 0

    _wrapper_owner(wrapper, monkeypatch, cap, Process())
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    out = io.StringIO()
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=out) == 0, out.getvalue()
    assert kernel.sent, "the run must have signalled before any reap"


def test_sigchld_ignored_at_entry_is_refused_with_a_named_reason(monkeypatch):
    """Design test 4: no corpse, no pin — refuse by name, never proceed."""
    kernel = Kernel()
    monkeypatch.setattr(
        owned,
        "signal",
        NS(
            SIGCHLD=signal.SIGCHLD, SIG_IGN=signal.SIG_IGN, getsignal=lambda _signum: signal.SIG_IGN
        ),
    )
    cap = kernel.cap()
    assert cap.last.status == "REFUSED", cap.last.reason
    assert cap.last.refusal_class == "no_corpse_sigchld"
    assert kernel.censuses == 0, "the premise is void before any census runs"


def test_a_zombie_grandchild_does_not_hold_the_cleanup_unproven(wrapper, monkeypatch, tmp_path):
    """Design test 8, the 0 arm: corpses hold no code, so they do not hold the run."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.update({kernel.leader, 20})
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")

    class Process:
        pid = 10

        def poll(self):
            return 0

        def wait(self, **_):
            return 0

    _wrapper_owner(wrapper, monkeypatch, cap, Process())
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    out = io.StringIO()
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=out) == 0, out.getvalue()
    # Nothing live: the run reports clean, and it did so without a signal.
    assert kernel.sent == []
