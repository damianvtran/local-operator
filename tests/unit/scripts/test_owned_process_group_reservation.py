"""The reservation predicate: ownership by pinned group number, not by membership.

Every fake here is a SYNTHETIC kernel fact map — no pid reaches an OS probe and
no signal leaves this module. The worlds stub ``procstate.birth_scheme`` because
``ProcessSample.is_birth`` compares against the PLATFORM scheme: a fixture that
invents ``scheme="fixture"`` without stubbing it refuses every proof, which is
how the earlier harnesses silently tested nothing.
"""

from __future__ import annotations

import io
import json
import os
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

    def __init__(self, leader=10, parent=None, birth="birth"):
        # C1: the witness only means anything if WE are the child's parent, so the
        # fixtures model a real child of this process. Tests that need the refusal
        # arm pass an explicit foreign parent.
        self.leader = leader
        self.parent = os.getpid() if parent is None else parent
        self.rows = {
            self.parent: (0, self.parent, self.parent, "creator"),
            leader: (self.parent, leader, leader, birth),
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
        """Delivery marks the group terminal; it does NOT reap (the wrapper's wait does).

        The earlier fake popped the row on delivery, which made every post-signal
        census describe a number nobody held and turned healthy runs into
        pin_lost -- a fake that could not express the state the design is built on.
        """
        self.sent.append((pgid, signum))
        self.corpses.update(pid for pid, row in self.rows.items() if row[1] == pgid)

    def reap(self, pid):
        """What subprocess.wait() does to the real world: give up the corpse."""
        self.rows.pop(pid, None)
        self.corpses.discard(pid)

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


def test_a_missing_leader_row_is_unreadable_not_pin_lost():
    """CONTRACT: unreadable column. A missing map entry is the READER failing.

    It is not proof of a reap, and calling it `pin_lost` was the collapse the
    contract forbids: only a readable foreign birth proves loss.
    """
    kernel = Kernel()
    cap = kernel.cap()
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")
    del kernel.rows[kernel.leader]
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == "unproven", verdict.reason
    assert kernel.sent == []


def test_a_readable_foreign_birth_is_pin_lost_and_nothing_is_signalled():
    """CONTRACT: reaped-gone / readable foreign birth -> REFUSED(pin_lost), every column."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")
    kernel.rows[kernel.leader] = (kernel.parent, kernel.leader, kernel.leader, "foreign")
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == "pin_lost", verdict.reason
    assert kernel.sent == []


def test_a_live_leader_is_live_leader_only_and_still_signals():
    kernel = Kernel()
    cap = kernel.cap()
    verdict = cap.signal(signal.SIGTERM, deadline=kernel.clock() + 5.0)
    assert verdict.status == "SIGNALED"
    assert verdict.anchor == "leader-live"


def test_a_zombie_leader_omitted_from_the_census_still_signals():
    """CONTRACT: held-zombie + live members -> VERIFIED+SIGNAL.

    Census residence is NOT the predicate: a corpse the enumeration omits while it
    lists the live grandchild is a diagnostic gap, not lost authority. The held
    creation receipt is the witness.
    """
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")
    original = kernel.census

    def without_leader(pgid, deadline):
        return [pid for pid in original(pgid, deadline) if pid != kernel.leader]

    cap.census = without_leader
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "SIGNALED", verdict.reason
    assert verdict.anchor == "leader-reservation"


def test_the_same_omitted_corpse_with_a_foreign_birth_still_refuses():
    """The discriminating arm: the SAME census, a readable foreign birth -> pin_lost."""
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")
    kernel.rows[kernel.leader] = (kernel.parent, kernel.leader, kernel.leader, "foreign")
    original = kernel.census

    def without_leader(pgid, deadline):
        return [pid for pid in original(pgid, deadline) if pid != kernel.leader]

    cap.census = without_leader
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == "pin_lost", verdict.reason
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
    """pid 1 is excluded BY CONSTRUCTION; our own group is excluded at the syscall.

    Adjudicated 2026-11-01: the test used to build authority for pid 1 through
    `capture`, which refuses to construct one at all ("numeric PGID alone cannot
    construct signal authority") -- so the exclusion had already fired a level
    earlier and the assertion could never be reached. The construction refusal IS
    the pid-1 exclusion, and it is asserted here instead.
    """
    with pytest.raises(ValueError):
        Kernel(leader=1, parent=1).cap()

    kernel = Kernel(leader=4242, parent=1)
    cap = kernel.cap()
    monkeypatch.setattr(owned.os, "getpgrp", lambda: 4242)
    monkeypatch.setattr(owned.os, "getpid", lambda: 77)
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    assert kernel.sent == []


#: The C3 marker is what makes this mutation take ONE check: `refresh` holds a
#: second, differently-motivated `_leader_reservation` test, and a bare
#: `str.replace` of the call deleted BOTH -- so the control proved less than its
#: comment claimed. The slice below runs from the marker to the end of the
#: residency block, and nothing outside it.
RESIDENCY_REMOVED = (
    "        # WITNESS RE-CHECK IMMEDIATELY BEFORE THE SYSCALL -- THE MUTATION ANCHOR",
    "        # MUTANT: the pre-send witness re-check is deleted here; refresh keeps its own.\n"
    "        if False:\n"
    "            pass",
)


def _mutant_source(source):
    """Return `source` with ONLY the signal-path residency check deleted."""
    before, after = RESIDENCY_REMOVED
    marker = source.index(before)
    check = source.index("        final_state, _ = self._leader_witness(deadline)", marker)
    tail = source.index("            return self.last\n", check) + len(
        "            return self.last\n"
    )
    return source[:marker] + after + "\n" + source[tail:]


def _mutant():
    source = Path(owned.__file__).read_text()
    mutated = _mutant_source(source)
    assert mutated != source, "the residency re-check moved; update this control"
    module = ModuleType("owned_process_group_no_residency")
    module.__file__ = owned.__file__
    sys.modules[module.__name__] = module  # @dataclass resolves its module on 3.12
    exec(compile(mutated, module.__name__, "exec"), module.__dict__)
    return module


def test_c3_removing_the_residency_re_check_lets_a_stranger_die():
    """The mutation must fail in the RIGHT direction — a stranger dies."""
    kernel = Kernel()
    cap = kernel.cap()
    # Adjudicated 2026-11-01: the control used a CORPSE-ONLY census, which the
    # three-way rule now refuses on its own (no live member and no reservation is
    # a lost pin, not an observation). The mutation has to isolate the PRE-SYSCALL
    # check, so the group is alive and the leader's sample disappears only on the
    # fourth `_samples` call -- the one `signal` makes immediately before the send.
    original = kernel.samples

    def reaped_mid_proof(pids):
        # The witness disappears EXACTLY at the pre-send re-check: every earlier
        # sample is real, so only that one check can catch it. Anchored on the
        # check's semantics, not on a call count.
        # Semantic anchor: the witness is unreadable only when the LEADER is asked
        # about on its own, after the proof has already been made once. A call-count
        # anchor drifts every time the predicate asks for one more sample.
        if kernel.sent == [] and pids == [kernel.leader] and capped["calls"] >= 3:
            return {}
        capped["calls"] += 1
        return original(pids)

    capped = {"calls": 0}
    cap.samples = reaped_mid_proof
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    # An unreadable witness at the last instant is the READER failing: `unproven`,
    # never `pin_lost` (that word is reserved for a readable foreign birth).
    assert verdict.refusal_class == "unproven", verdict.reason
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
    capped2 = {"calls": 0}

    def reaped_mid_proof2(pids):
        if capped2["calls"] >= 4:
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
    kernel.corpses.add(kernel.leader)
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
    # TERMINAL, not a timeout: the corpse leader makes `leader_terminal` true on the
    # first tick, so the run reaches the grace loop deterministically instead of
    # spinning for ever -- `poll` is no longer the exit driver, and a live-leader
    # `timeout=None` shape would never leave this loop. It also exercises the
    # `inf` horizon B2 fixed.
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=out) == 0, out.getvalue()
    # One refresh proves, then each signal re-proves: two complete strict scans
    # per proof. Nothing here runs a census once per poll or per grace tick.
    # FIVE proofs, and 2 strict censuses each, is the whole cost:
    #   acquire (capture), the required-refresh, the signal's own re-proof, the
    #   finally refresh, and the finally signal's re-proof. Ten censuses were
    #   observed against a bound of six because the earlier bound forgot the
    #   signal and the finally pass -- the bound was wrong, not the run cheap.
    # What the bound still proves is the property this test exists for: no census
    # per poll tick and none per grace tick, only per proof.
    assert kernel.censuses - before <= 2 * 5, (kernel.censuses - before, kernel.sent)


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
    kernel.corpses.update({kernel.leader, 20})
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "child")

    class Process:
        pid = 10

        def poll(self):
            # The census under the pin must have run BEFORE the corpse is given
            # up; `poll` is the reap. Ordering is the property, not "a signal
            # happened": a run whose members are already corpses sends nothing.
            assert kernel.censuses >= 2, "poll() reaped the leader before the terminal census"
            kernel.reap(10)
            return 0

        def wait(self, **_):
            assert kernel.censuses >= 2, "wait() reaped the leader before the terminal census"
            kernel.reap(10)
            return 0

    _wrapper_owner(wrapper, monkeypatch, cap, Process())
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    out = io.StringIO()
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=out) == 0, out.getvalue()
    assert kernel.censuses >= 2, "the terminal census must have run"


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


def test_a_darwin_shaped_identity_provider_does_not_refuse_the_incident_shape():
    """B1: the corpse is the RESERVATION, never an identity proof.

    Darwin's `kernel_identity` raises for a zombie (`getsid` fails on a corpse),
    so proving PG/SID over the CENSUS UNION refused a corpse leader plus a live
    grandchild on the operator's platform -- rc 126 with the child still running,
    the exact bug this PR exists to kill. The live set is what carries an identity
    proof; the corpse carries the reservation.
    """
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")

    def darwin_identity(pid):
        if pid in kernel.corpses:
            raise owned.Unknown("actual parent/PG/SID unavailable (zombie)")
        return kernel.rows[pid][:3]

    cap.identity = darwin_identity
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "SIGNALED", verdict.reason
    assert verdict.anchor == "leader-reservation"
    assert kernel.sent == [(10, signal.SIGKILL)]


def test_a_member_sample_that_cannot_be_read_refuses_as_unproven():
    """CONTRACT: unreadable column — an unresolved member sample is not absence.

    "The member exited" and "we could not read the member" are different facts,
    and only the first is allowed to be non-blocking. Inferring departure from a
    missing sample is how an unobserved member becomes false absence.
    """
    kernel = Kernel()
    cap = kernel.cap()
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")
    original = kernel.samples

    def vanishing(pids):
        return {pid: sample for pid, sample in original(pids).items() if pid != 20}

    cap.samples = vanishing
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status == "REFUSED", verdict.reason
    assert verdict.refusal_class == "unproven", verdict.reason
    assert kernel.sent == []


def test_a_member_seen_by_one_census_and_gone_from_the_next_is_recorded():
    """The UNION's amnesty survives: a pid seen once was inside our number.

    Distinct from the case above: here the member is genuinely gone between the
    two CENSUSES (which the union tolerates and records), not unreadable at
    sample time.
    """
    kernel = Kernel()
    cap = kernel.cap()
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")
    scans = {"n": 0}
    original = kernel.census

    def two_scans(pgid, deadline):
        scans["n"] += 1
        rows = original(pgid, deadline)
        if scans["n"] == 2:
            return [pid for pid in rows if pid != 20]
        return rows

    cap.census = two_scans
    verdict = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5.0)
    assert verdict.status in {"SIGNALED", "OBSERVED_EMPTY"}, verdict.reason
    assert 20 in cap.probe["members"]


def test_a_run_without_a_timeout_no_longer_compares_against_none(wrapper, monkeypatch, tmp_path):
    """B2 control: `--timeout`-less is `inf`, so `now >= deadline` compares floats.

    On the `None` form this run raises `TypeError: '>=' not supported between
    'float' and 'NoneType'` in `leader_terminal`, and every such invocation dies.
    """
    kernel = Kernel()
    cap = kernel.cap()
    for pid in (kernel.leader,):
        kernel.corpses.add(pid)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "child")
    kernel.corpses.add(20)

    class Process:
        pid = 10

        def poll(self):
            kernel.reap(10)
            return 0

        def wait(self, **_):
            kernel.reap(10)
            return 0

    _wrapper_owner(wrapper, monkeypatch, cap, Process())
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    out = io.StringIO()
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=out) == 0, out.getvalue()


LOSS_OF_AUTHORITY_SIGNAL_GUARD = (
    "                if proof_failed:\n"
    "                    break  # no automatic signal retry after loss of authority\n"
)


def test_the_discarded_authority_guard_has_an_executed_deletion_control(
    wrapper, monkeypatch, tmp_path
):
    """The :754 guard, discriminated: a latched loss must never be followed by a send.

    Intact, a run whose authority is refused mid-run sends NOTHING afterwards even
    though every later refresh would verify again -- `proof_failed` is the memory
    that the refusal happened. The mutant deletes that memory and signals anyway.
    """
    ASK = {"armed": True}

    def once_then_verified(kernel):
        def refresh(*, deadline):
            if ASK["armed"]:
                ASK["armed"] = False
                return owned.Verdict(
                    "REFUSED", "authority refused mid-run", (10,), refusal_class="unproven"
                )
            return owned.Verdict("VERIFIED", "verifies again", (10,), anchor="leader-live")

        return refresh

    def run_one(module, kernel, cap):
        monkeypatch.setattr(
            module,
            "owned_process_group",
            NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict),
        )
        monkeypatch.setattr(
            module, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
        )

        class Process:
            pid = 10

            def poll(self):
                return 0

            def wait(self, **_):
                return 0

        monkeypatch.setattr(
            module,
            "subprocess",
            NS(Popen=lambda *a, **k: Process(), TimeoutExpired=subprocess.TimeoutExpired),
        )
        monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
        return module.run(["synthetic"], timeout=None, grace=0.05, stream=io.StringIO())

    kernel = Kernel()
    cap = kernel.cap()
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "child")
    cap.refresh = once_then_verified(kernel)
    assert run_one(wrapper, kernel, cap) != 0, "the refusal must be reported, not swallowed"
    assert kernel.sent == [], "a latched loss of authority was followed by a signal"

    source = Path(wrapper.__file__).read_text()
    assert LOSS_OF_AUTHORITY_SIGNAL_GUARD in source, "the guard moved; update this control"
    module = ModuleType("run_bounded_no_authority_guard")
    module.__file__ = wrapper.__file__
    sys.modules[module.__name__] = module
    exec(
        compile(
            source.replace(
                LOSS_OF_AUTHORITY_SIGNAL_GUARD,
                "                if False:  # MUTANT\n                    break\n",
            ),
            module.__name__,
            "exec",
        ),
        module.__dict__,
    )
    kernel2 = Kernel()
    cap2 = kernel2.cap()
    kernel2.rows[20] = (kernel2.leader, kernel2.leader, kernel2.leader, "child")
    cap2.refresh = once_then_verified(kernel2)
    ASK["armed"] = True
    run_one(module, kernel2, cap2)
    assert kernel2.sent, "the mutant must signal after the latched loss, or it proves nothing"


def test_a_terminal_observed_empty_still_records_the_anchor_and_the_proof(
    wrapper, monkeypatch, tmp_path
):
    """B1: `proof` and `anchor` are RUN-level evidence and outlive the last verdict.

    The final refresh of a successful run is the terminal one, and it returns
    OBSERVED_EMPTY. Resetting the probe per verdict therefore published
    `anchor: null` and `proof: []` on every run that worked -- the object three
    downstream lanes bind to, and the one the incident shape asserts.
    """
    kernel = Kernel()
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "grandchild")

    class Process:
        pid = 10

        def poll(self):
            kernel.reap(10)
            return 0

        def wait(self, **_):
            kernel.reap(10)
            return 0

    _wrapper_owner(wrapper, monkeypatch, cap, Process())
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    out = io.StringIO()
    assert wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=out) == 0, out.getvalue()
    records = sorted(Path(tmp_path).rglob("*.exit.json"))
    assert records, "a spawned run must leave an exit record"
    ownership = json.loads(records[-1].read_text())["ownership"]
    assert ownership["anchor"] == "leader-reservation", ownership
    assert {row["pid"] for row in ownership["proof"]} == {10, 20}, ownership["proof"]
    # The terminal verdict owns no declined set of its own, and must not inherit one.
    assert ownership["residuals"] == [], ownership["residuals"]


def test_c1_a_child_that_is_not_ours_refuses_as_not_exclusive_reaper():
    """C1: exclusive reap ownership is ESTABLISHED, or named as absent.

    The witness only means something if nobody else can consume the corpse. A
    leader whose kernel-recorded parent is another process cannot be witnessed by
    us, and the refusal names the reason instead of implying a proof we lack.
    """
    kernel = Kernel(parent=1)  # observe a foreign parent, not our own pid
    cap = kernel.cap()
    # ACQUISITION refuses: there is no witness to hold, so nothing downstream may
    # pretend to have proved anything. Asserted on the capture verdict, because a
    # later `signal` would only restate that no receipt exists.
    assert cap.last.status == "REFUSED", cap.last.reason
    assert cap.last.refusal_class == "not_exclusive_reaper", cap.last.reason
    assert kernel.sent == []


def test_c2_same_second_reuse_is_encoded_unsupported_not_waived():
    """C2: the pin cannot distinguish a same-second reuse, and says so.

    `ps` lstart has second resolution, so two processes with the same pid, parent
    and second collide in the birth token. The honest outcome is a named
    UNSUPPORTED, not a silent claim of safety and not a fabricated discriminator.
    """
    assert owned.SECOND_RESOLUTION_REUSE == "UNSUPPORTED"
    doc = owned.__doc__ or ""
    assert "SECOND-RESOLUTION" in doc.upper(), "the scope must be documented"
    assert "does not cover" in doc.lower(), "C3: say what the pin does NOT cover"
