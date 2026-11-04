"""The cleanup-proof deadline is sized for the census it runs, and stays bounded.

#3365 CI 41684135750 (job 126086404095): the incident-shape test exited 126 with
``finally refused: Unknown:strict /proc row deadline`` -- attempt 1, phase
``row-deadline``, 167 ``/proc`` rows, 0.3 s wall. The finally, terminal and kill
phases shared ONE ``begin_cleanup(0.25)`` allowance, a number inherited from the
pre-#3274 five-sleep sweep and never sized for a census. That allowance now has
to pay for ~10 full strict ``/proc`` scans (each refresh is two censuses, and
``signal()`` refreshes again before its syscall). A preempted xdist worker ran
out of it mid-scan, the SIGKILL was refused, and the grandchild the sweep exists
to kill survived: the 2026-10-30 incident shape.

These tests run the REAL ``strict_linux`` row loop over a synthetic ``/proc`` on
a fake clock that advances only when a row is read or the wrapper sleeps. That
makes "a slow host" a precise, reproducible quantity rather than a hope about
CI load. No pid reaches an OS probe and no signal leaves this module, so the
file runs identically on Darwin and Linux.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from local_operator.procstate import ProcessSample
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401

SCHEME = "cleanup-deadline-fixture"
LEADER = 10
GRANDCHILD = 20
#: The row count the failing CI census reported (``opening_count 167``).
ROWS = 167
#: Cleanup scans one successful leader-exit sweep performs (verify+kill pass,
#: observed-empty pass, terminal refresh). The ruling counted ~10; the rig below
#: counts them rather than trusting that number.
SWEEP_SCANS = 10
#: Fake seconds per full scan for the "slow but finishing" host: see test 1.
SLOW_SCAN = 0.08


class SlowProc:
    """A synthetic ``/proc`` whose every row read costs ``row_cost`` fake seconds.

    Rows are ``(parent, pgid, sid, birth)``; ``corpses`` are zombies (still listed,
    still holding their number until ``reap``). Filler rows in a foreign group pad
    the namespace to ``ROWS`` so a scan costs what the CI scan cost.
    """

    def __init__(self, *, leader_ignores_term=False, leader_exited=True):
        self.parent = os.getpid()
        self.rows = {
            self.parent: (0, self.parent, self.parent, "creator"),
            LEADER: (self.parent, LEADER, LEADER, "leader-birth"),
            GRANDCHILD: (LEADER, LEADER, LEADER, "grandchild-birth"),
        }
        filler = 1000
        while len(self.rows) < ROWS:
            self.rows[filler] = (1, 1, 1, "filler")
            filler += 1
        self.corpses = {LEADER} if leader_exited else set()
        #: Signals the leader's group survives. Only SIGKILL cannot be ignored.
        self.ignored = {signal.SIGTERM} if leader_ignores_term else set()
        self.now = 0.0
        self.row_cost = 0.0
        self.scans = 0
        self.sent: list[int] = []
        #: Fault injection for the must-still-refuse matrix.
        self.unreadable: int | None = None
        self.churn = False

    # -- the clock: moves only on row reads and wrapper sleeps -----------------
    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)

    # -- the /proc surface strict_linux reads ----------------------------------
    def listdir(self, _path):
        names = [str(pid) for pid in self.rows]
        if self.churn:
            # A pid that appears between opening and closing listings every time:
            # namespace churn the six-attempt limit must refuse, never wait out.
            self.now += 1e-6
            names.append(str(900000 + int(self.now * 1e6)))
        return names

    def read(self, path):
        self.now += self.row_cost
        pid = int(path.split("/")[2])
        if pid == self.unreadable:
            raise PermissionError(path)
        if pid not in self.rows:
            raise FileNotFoundError(path)
        parent, group, sid, _birth = self.rows[pid]
        state = "Z" if pid in self.corpses else "S"
        return f"{pid} (fixture) {state} {parent} {group} {sid} 0 0\n"

    def stat(self, path):
        if int(path.split("/")[2]) not in self.rows:
            raise FileNotFoundError(path)

    def census(self, pgid, deadline):
        self.scans += 1
        pids, complete, seen = owned.strict_linux(
            pgid,
            deadline,
            listdir=self.listdir,
            read=self.read,
            stat=self.stat,
            clock=self.clock,
        )
        assert complete is True
        return owned.AuthoritativeCensus(tuple(pids), True, bool(seen))

    # -- identity and the checked send -----------------------------------------
    def samples(self, pids):
        return {
            pid: ProcessSample(pid in self.corpses, self.rows[pid][3])
            for pid in pids
            if pid in self.rows
        }

    def identity(self, pid):
        return self.rows[pid][:3]

    def send(self, pgid, signum):
        self.sent.append(signum)
        if signum in self.ignored:
            return
        # The group dies: the leader becomes our unreaped corpse (the pin), the
        # grandchild -- whose parent is that corpse -- is reparented and reaped.
        for pid, row in list(self.rows.items()):
            if row[1] == pgid and pid != LEADER:
                self.rows.pop(pid)
        self.corpses.add(LEADER)

    def reap(self, pid):
        self.rows.pop(pid, None)
        self.corpses.discard(pid)

    def cap(self):
        return owned.OwnedGroup.capture(
            LEADER,
            parent_pid=self.parent,
            census=self.census,
            samples=self.samples,
            identity=self.identity,
            scheme=lambda: SCHEME,
            clock=self.clock,
            signal_group=self.send,
            deadline=self.clock() + 5.0,
        )


@pytest.fixture(autouse=True)
def _platform_scheme(monkeypatch):
    """`is_birth` compares against the platform scheme -- make them agree."""
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: SCHEME)


@pytest.fixture
def wrapper(request, monkeypatch, tmp_path):
    module = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return module


def _per_scan(seconds):
    """Row cost that makes ONE full scan of ``ROWS`` rows take ``seconds``."""
    return seconds / ROWS


def _run(wrapper, monkeypatch, world, *, timeout=None, grace=0.05, keep_budget=False):
    """Drive the real ``run`` against ``world``; return (rc, stream, cleanup wall).

    Capture is free (``row_cost`` is applied only after it) because this file is
    about the cleanup allowance, not the capture bound. ``cleanup wall`` is fake
    time from the moment the loop decided to clean up to the return of ``run``.
    """
    cost, world.row_cost = world.row_cost, 0.0
    cap = world.cap()
    world.row_cost = cost
    world.scans = 0
    marks = {}

    class Clock:
        def monotonic(self):
            return world.clock()

        def sleep(self, seconds):
            world.sleep(seconds)

    class Process:
        returncode = 0
        pid = LEADER

        def poll(self):
            world.reap(LEADER)
            return 0

        def wait(self, **_):
            world.reap(LEADER)
            return 0

    original_refresh = cap.refresh

    def refresh(**kwargs):
        marks.setdefault("first", world.now)
        return original_refresh(**kwargs)

    monkeypatch.setattr(wrapper, "time", Clock())
    monkeypatch.setattr(
        wrapper,
        "owned_process_group",
        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict),
    )
    monkeypatch.setattr(
        wrapper,
        "subprocess",
        NS(Popen=lambda *a, **k: Process(), TimeoutExpired=subprocess.TimeoutExpired),
    )
    if not keep_budget:
        monkeypatch.setattr(
            wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
        )
    cap.refresh = refresh
    out = io.StringIO()
    rc = wrapper.run(["synthetic"], timeout=timeout, grace=grace, stream=out)
    return rc, out.getvalue(), world.now - marks.get("first", world.now)


def _ownership(tmp_path):
    records = sorted(Path(tmp_path).rglob("*.exit.json"))
    assert records, "a spawned run must leave an exit record"
    return json.loads(records[-1].read_text())["ownership"]


def test_a_slow_but_finishing_census_reclaims_the_grandchild(wrapper, monkeypatch, tmp_path):
    """Test 1 (the defect): ~10 scans of 167 rows, ~0.8 s in all, must finish.

    0.08 s a scan is the smallest round cost at which the KILL's own pre-send
    refresh (scans 3-4 of the sweep) outruns 0.25 s -- the CI shape: refused
    mid-row-scan, no SIGKILL, grandchild alive. Below 0.0625 s a scan the 0.25 s
    base still fails, but later (the pass-2 witness sample), with the kill sent.
    Sized for the census, the sweep kills it and the record names the pin.
    """
    world = SlowProc()
    world.row_cost = _per_scan(SLOW_SCAN)
    rc, out, _wall = _run(wrapper, monkeypatch, world)
    assert rc == 0, out
    assert "row deadline" not in out, out
    assert signal.SIGKILL in world.sent, "the sweep must actually send the kill"
    assert GRANDCHILD not in world.rows, "the grandchild the sweep exists to kill survived"
    assert _ownership(tmp_path)["anchor"] == "leader-reservation"
    # The rig is the evidence only if the work it priced actually ran.
    assert world.scans >= SWEEP_SCANS - 2, world.scans


def test_a_census_that_never_finishes_is_still_refused_within_the_bound(
    wrapper, monkeypatch, tmp_path
):
    """Test 2 (still bounded): when the census cannot finish, refuse -- promptly.

    Each scan costs 0.5 s, so the sweep would need ~5 s. The allowance is a
    wall-time bound on the instrument, not a grace period: the run refuses with
    ``row-deadline`` and spends at most the bound plus one row read doing so.
    """
    world = SlowProc()
    world.row_cost = _per_scan(0.5)
    rc, out, wall = _run(wrapper, monkeypatch, world)
    assert rc == wrapper.EXIT_CLEANUP_UNPROVEN, out
    assert "strict /proc row deadline" in out, out
    assert wall <= wrapper._CLEANUP_PROOF_SECONDS + world.row_cost + 1e-6, wall
    del tmp_path


def test_a_timeout_with_a_term_ignoring_leader_still_kills_and_certifies(
    wrapper, monkeypatch, tmp_path
):
    """Test 3: rc 124 wins, the SIGKILL lands, and the terminal phase certifies.

    The kill, finally and terminal phases share the slice past the grace wait.
    On 0.25 s that slice could not pay for the kill's own pre-send refresh at
    this census cost, so the TERM-ignoring leader's group was never certified.
    The grace (1.0 s) is deliberately wider than one refresh: the SIGTERM proof
    is bounded by the grace window by design, and that is not what this tests.
    """
    world = SlowProc(leader_ignores_term=True, leader_exited=False)
    world.row_cost = _per_scan(SLOW_SCAN)
    rc, out, _wall = _run(wrapper, monkeypatch, world, timeout=0.5, grace=1.0)
    assert rc == wrapper.EXIT_TIMEOUT, out
    assert world.sent[:1] == [signal.SIGTERM], (world.sent, out)
    assert signal.SIGKILL in world.sent, out
    assert GRANDCHILD not in world.rows, out
    assert "not observed empty" not in out, out
    assert "refused" not in out, out
    del tmp_path


def test_an_unreadable_row_is_refused_at_once_not_waited_out(wrapper, monkeypatch, tmp_path):
    """Test 4a: a larger allowance buys NO time for a broken row.

    The refusal is immediate: the cleanup spends at most the rows read before the
    bad one, never the allowance.
    """
    world = SlowProc()
    world.row_cost = _per_scan(0.001)
    world.unreadable = GRANDCHILD
    rc, out, wall = _run(wrapper, monkeypatch, world)
    assert rc == wrapper.EXIT_CLEANUP_UNPROVEN, out
    assert "unreadable /proc row" in out, out
    assert GRANDCHILD in world.rows
    assert wall < 0.05, wall
    del tmp_path


def test_churn_past_six_attempts_is_still_unstable(wrapper, monkeypatch, tmp_path):
    """Test 4b: time is not attempts -- churn still exhausts the six-attempt limit."""
    world = SlowProc()
    world.row_cost = _per_scan(0.001)
    world.churn = True
    rc, out, wall = _run(wrapper, monkeypatch, world)
    assert rc == wrapper.EXIT_CLEANUP_UNPROVEN, out
    assert "unstable /proc census" in out, out
    assert signal.SIGKILL not in world.sent, world.sent
    assert wall < wrapper._CLEANUP_PROOF_SECONDS, wall
    del tmp_path


def test_a_foreign_leader_birth_is_still_pin_lost_whatever_the_bound(
    wrapper, monkeypatch, tmp_path
):
    """Test 4c: a readable foreign birth on the leader number refuses and sends nothing.

    Elapsed time is not authority: with the full allowance available, a reused
    number still loses the pin before any SIGKILL leaves the module.
    """
    world = SlowProc()
    world.row_cost = _per_scan(0.001)
    original_cap = world.cap

    def cap_then_reuse():
        cap = original_cap()
        parent, group, sid, _birth = world.rows[LEADER]
        world.rows[LEADER] = (parent, group, sid, "a-stranger")
        world.corpses.discard(LEADER)
        return cap

    world.cap = cap_then_reuse
    rc, out, _wall = _run(wrapper, monkeypatch, world)
    assert rc == wrapper.EXIT_CLEANUP_UNPROVEN, out
    assert signal.SIGKILL not in world.sent, world.sent
    assert GRANDCHILD in world.rows
    # The refusal must be the PIN loss, not any refusal: an unreadable row or an
    # expired bound would also send nothing, and this test is about reuse.
    records = [json.loads(path.read_text()) for path in sorted(Path(tmp_path).rglob("*.json"))]
    classes = [r["ownership"].get("refusal_class") for r in records if "ownership" in r]
    assert "pin_lost" in classes, (classes, out)
    assert "unproven" not in classes, (classes, out)


def test_a_member_alive_after_the_kill_is_still_refused(wrapper, monkeypatch, tmp_path):
    """Test 4d: the larger bound cannot certify a group whose member survived SIGKILL."""
    world = SlowProc()
    world.row_cost = _per_scan(0.001)
    world.send = lambda pgid, signum: world.sent.append(signum)  # delivers, kills nothing
    rc, out, _wall = _run(wrapper, monkeypatch, world)
    assert rc == wrapper.EXIT_CLEANUP_UNPROVEN, out
    assert signal.SIGKILL in world.sent, world.sent
    assert GRANDCHILD in world.rows
    residual = sorted(Path(tmp_path).rglob("*.residual-census.json"))
    assert residual, out
    verdict = json.loads(residual[-1].read_text())["verdict"]
    assert verdict["status"] == "REFUSED", verdict
    assert verdict["reason"].startswith("group not observed empty"), verdict
    assert GRANDCHILD in verdict["members"], verdict


def test_a_proof_that_ends_past_the_deadline_is_refused_before_the_syscall():
    """Test 4e: ``checked >= deadline`` still refuses at the send, after a VERIFIED proof.

    Exercised on the owner directly: the refresh verifies, but the clock has
    passed the deadline by the time the syscall would run, so nothing is sent.
    """
    world = SlowProc()
    cap = world.cap()
    deadline = world.now + 1.0
    original = cap.refresh

    def verifies_late(**kwargs):
        verdict = original(**kwargs)
        world.now = deadline
        return verdict

    cap.refresh = verifies_late
    verdict = cap.signal(signal.SIGKILL, deadline=deadline)
    assert verdict.status == "REFUSED", verdict
    assert verdict.reason == "expired before syscall", verdict
    assert world.sent == [], world.sent


def test_a_small_grace_still_proves_the_term_and_kills(wrapper, monkeypatch, tmp_path):
    """Test 5 (QA Q1, #3368 c2): ``--grace 0.2`` must not starve the SIGTERM proof.

    The SIGTERM was proved against ``start + grace``, and the required refresh
    just before it spent most of that window: at 0.09 s a scan the TERM was
    refused mid-row-scan. A refused proof is a lost authority, after which no
    signal is retried (the ``finally`` rule) -- so the SIGKILL was never even
    attempted and the grandchild outlived an rc 124 run. ``--grace`` is how long
    the group gets to exit AFTER the SIGTERM, not a bound on proving it.
    """
    world = SlowProc(leader_ignores_term=True, leader_exited=False)
    world.row_cost = _per_scan(0.09)
    rc, out, _wall = _run(wrapper, monkeypatch, world, timeout=0.5, grace=0.2)
    assert rc == wrapper.EXIT_TIMEOUT, out
    assert world.sent[:2] == [signal.SIGTERM, signal.SIGKILL], (world.sent, out)
    assert GRANDCHILD not in world.rows, out
    assert "refused" not in out, out
    del tmp_path


def test_a_forwarded_sigint_with_a_small_grace_still_kills(wrapper, monkeypatch, tmp_path):
    """Test 6 (R1-m1): the signal path forwards, TERMs, waits ``grace``, then KILLs.

    The group ignores both the forwarded SIGINT and the SIGTERM, so only the
    checked SIGKILL can end it. SIGINT is latched by calling the handler the
    wrapper installed (no OS signal leaves this test), once the run is polling.
    """
    world = SlowProc(leader_ignores_term=True, leader_exited=False)
    world.ignored.add(signal.SIGINT)
    world.row_cost = _per_scan(SLOW_SCAN)
    sleep = world.sleep

    def sleep_then_interrupt(seconds):
        sleep(seconds)
        if world.now >= 0.2 and not world.sent:
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler), "the wrapper must have installed its forwarder"
            handler(signal.SIGINT, None)

    world.sleep = sleep_then_interrupt
    rc, out, _wall = _run(wrapper, monkeypatch, world, grace=0.2)
    assert rc == 128 + signal.SIGINT, out
    assert world.sent[:3] == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL], (world.sent, out)
    assert GRANDCHILD not in world.rows, out
    assert "refused" not in out, out
    del tmp_path


def test_a_memory_kill_at_census_cost_reclaims_the_grandchild(wrapper, monkeypatch, tmp_path):
    """Test 7 (R1-m1): the memory-kill site opens the cleanup bound itself.

    No timeout or signal has opened it first, so its allowance is the one the
    kill, finally and terminal phases share -- it must pay for the census.
    """
    world = SlowProc(leader_exited=False)
    world.row_cost = _per_scan(SLOW_SCAN)

    class Guard:
        tick_s = 0.25

        def __init__(self, *_a, **_k):
            pass

        def sample_sync(self):
            return NS(bytes_used=1)

        def should_kill(self, _sample):
            return world.now >= 0.2

        def over_budget_message(self, *_a, **_k):
            return "over budget (fixture)"

        def soft_notice(self, _sample):
            return None

    monkeypatch.setattr(wrapper.memory_guard, "Guard", Guard)
    budget = (NS(ceiling_mb=64, reason="fixture"), "fixture")
    monkeypatch.setattr(wrapper, "_memory_budget", lambda _: budget)
    monkeypatch.setattr(wrapper, "_child_environment", lambda _b: {})
    rc, out, _wall = _run(wrapper, monkeypatch, world, keep_budget=True)
    assert rc == wrapper.EXIT_MEMORY, out
    assert world.sent[:1] == [signal.SIGKILL], (world.sent, out)
    assert GRANDCHILD not in world.rows, out
    assert "refused" not in out, out
    del tmp_path
