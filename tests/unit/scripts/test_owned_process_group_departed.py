"""A census member reaped BETWEEN the two refresh censuses is dropped, narrowly.

The shape under test is the measured healthy-gate failure (CI 41665666039, job
126020333214): censuses ``[[2716,2739,2740,2741],[2716,2741]]`` with 2739/2740
failing their member sample at ENOENT. Before this rule every such run ended 126
("census member sample unreadable") although nothing of ours was alive.

The rule (architect 641ae72fe5d1, security e8f8f83bc578) is a conjunction; each
counterexample below removes exactly one conjunct and must still refuse. The
synthetic world is the real `_proc_samples` reader over an intercepted
``/proc/<pid>/stat``, so the ENOENT diagnostic is produced by product code, not
asserted by the fixture.
"""

import errno
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from local_operator import procstate
from tests.unit.scripts.test_owned_process_group_reservation import SCHEME, Kernel
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401
from tests.unit.scripts.test_run_bounded_run_record import (  # noqa: F401
    _install,
    _records,
    wrapper,
)

LEADER = 10


def esrch(pid):
    raise ProcessLookupError(errno.ESRCH, "synthetic: no such process")


def ours_namespace():
    return str(os.getpid())


class World:
    """Scripted censuses plus synthetic /proc rows, with per-PID read failures."""

    def __init__(self, monkeypatch, first, second, *, errors=None, live=(), script=None):
        monkeypatch.setattr(procstate, "birth_scheme", lambda: SCHEME)
        monkeypatch.setattr(owned, "sys", NS(platform="linux"))
        self.monkeypatch = monkeypatch
        self.kernel = Kernel(leader=LEADER, birth="100")
        self.cap = self.kernel.cap()
        self.kernel.corpses.add(LEADER)
        for pid in live:
            self.kernel.rows[pid] = (LEADER, LEADER, LEADER, "200")
        self.first, self.second = list(first), list(second)
        #: Optional census-by-call sequence for multi-cycle histories; past its
        #: end the plain first/second alternation resumes.
        self.script = [list(view) for view in (script or ())]
        #: pid -> exception raised (or short row returned) by its stat read.
        self.errors = dict(errors or {})
        self.calls = 0
        self.member_sampled = False
        #: What the leader's stat read does AFTER the member sample.
        self.leader_after = None
        self.cap.census = self.census
        self.cap.samples = None
        self.cap._departure = esrch
        self.cap._proc_self = ours_namespace

    def census(self, pgid, deadline):
        if self.calls < len(self.script):
            view = self.script[self.calls]
        else:
            view = self.first if self.calls % 2 == 0 else self.second
        self.calls += 1
        return owned.AuthoritativeCensus(tuple(view), True, LEADER in view)

    def read(self, path):
        pid = int(str(path).split("/")[-2])
        if pid != LEADER:
            self.member_sampled = True
        elif self.member_sampled and self.leader_after is not None:
            if isinstance(self.leader_after, BaseException):
                raise self.leader_after
            row = self.kernel.rows[LEADER]
            self.kernel.rows[LEADER] = (*row[:3], self.leader_after)
        error = self.errors.get(pid)
        if isinstance(error, BaseException):
            raise error
        if error == "short-row":
            return f"{pid} (synthetic) R 10 10 10"
        parent, group, sid, birth = self.kernel.rows[pid]
        state = "Z" if pid in self.kernel.corpses else "R"
        tail = [state, str(parent), str(group), str(sid)] + ["0"] * 15 + [birth]
        return f"{pid} (synthetic) " + " ".join(tail)

    def refresh(self, terminal=False):
        with self.monkeypatch.context() as scoped:
            scoped.setattr(Path, "read_text", lambda path, *a, **k: self.read(path))
            verdict = self.cap.refresh(deadline=self.kernel.clock() + 5, terminal=terminal)
        assert self.kernel.sent == []
        return verdict


def gone(pid):
    return FileNotFoundError(errno.ENOENT, "synthetic PID %d reaped" % pid)


def bbed(monkeypatch, **overrides):
    """The measured shape: 20/21 in census 1 only, ENOENT; 22 stays live."""
    kwargs = {
        "first": [LEADER, 20, 21, 22],
        "second": [LEADER, 22],
        "errors": {20: gone(20), 21: gone(21)},
        "live": (22,),
    }
    kwargs.update(overrides)
    return World(monkeypatch, kwargs.pop("first"), kwargs.pop("second"), **kwargs)


def assert_refused(world, verdict):
    assert verdict.status == "REFUSED", verdict
    assert verdict.reason == "Unknown:census member sample unreadable", verdict
    assert "vanished" not in world.cap.probe


# 13: positive control on the measured shape -------------------------------


def test_bbed_shape_with_a_live_member_is_verified_without_the_departed():
    world = bbed(monkeypatch=pytest.MonkeyPatch())
    try:
        verdict = world.refresh()
    finally:
        world.monkeypatch.undo()
    assert verdict.status == "VERIFIED", verdict
    assert verdict.members == (LEADER, 22)
    assert [row["pid"] for row in world.cap.probe["proof"]] == [LEADER, 22]
    assert world.cap.probe["residuals"] == []
    assert world.cap.probe["vanished"] == [
        {
            "pid": pid,
            "reason": "read-error",
            "errno": errno.ENOENT,
            "getpgid": "ProcessLookupError",
            "census": "first-only",
        }
        for pid in (20, 21)
    ]


def test_bbed_corpse_only_shape_is_observed_empty(monkeypatch):
    world = bbed(monkeypatch, first=[LEADER, 20, 21], second=[LEADER], live=())
    verdict = world.refresh(terminal=True)
    assert verdict.status == "OBSERVED_EMPTY", verdict
    assert verdict.members == (LEADER,)
    assert world.cap.accepted_terminal(verdict)
    assert [row["pid"] for row in world.cap.probe["vanished"]] == [20, 21]


# 1: the late-fork guard ----------------------------------------------------


def test_member_in_census_two_then_enoent_is_refused(monkeypatch):
    world = bbed(monkeypatch, second=[LEADER, 20, 21, 22])
    assert_refused(world, world.refresh())
    # The in-both shape is re-run, never amnestied: the trigger persisted
    # through every cycle, so the refusal came only after all of them.
    assert world.calls == 2 * owned._SAMPLE_CYCLES


# 2: anything but this attempt's ENOENT diagnostic ---------------------------


@pytest.mark.parametrize(
    "failure",
    [
        PermissionError(errno.EACCES, "synthetic"),
        OSError(errno.EIO, "synthetic"),
        "short-row",
    ],
    ids=["EACCES", "EIO", "short-row"],
)
def test_non_enoent_sample_failure_is_refused(monkeypatch, failure):
    world = bbed(monkeypatch, errors={20: gone(20), 21: failure})
    assert_refused(world, world.refresh())
    assert world.cap.probe["unresolved"] == [20, 21]


def test_missing_diagnostic_is_refused(monkeypatch):
    world = bbed(monkeypatch)
    original = procstate._proc_samples
    monkeypatch.setattr(procstate, "_proc_samples", lambda pids, **kwargs: original(pids))
    assert_refused(world, world.refresh())


def test_seventeenth_member_past_the_diagnostic_cap_is_refused(monkeypatch):
    many = list(range(20, 37))
    world = World(
        monkeypatch,
        [LEADER, *many],
        [LEADER],
        errors={pid: gone(pid) for pid in many},
    )
    assert_refused(world, world.refresh())
    assert world.cap.probe["sample_failure"]["unresolved"][-1] == {
        "pid": 35,
        "reason": "read-error",
        "errno": errno.ENOENT,
    }
    assert world.cap.probe["sample_failure"]["unresolved_count"] == 17


# 3/4: the kernel must answer ESRCH, and only ESRCH ------------------------


def _raises(exc):
    def probe(pid):
        raise exc

    return probe


@pytest.mark.parametrize(
    "probe",
    [
        lambda pid: LEADER,
        lambda pid: 4242,
        _raises(PermissionError(errno.EPERM, "synthetic LSM denial")),
        _raises(PermissionError(errno.EACCES, "synthetic LSM denial")),
        _raises(OSError(errno.EINVAL, "synthetic")),
        _raises(RuntimeError("synthetic")),
    ],
    ids=["our-pgid", "other-pgid", "EPERM", "EACCES", "EINVAL", "other"],
)
def test_getpgid_other_than_process_lookup_error_is_refused(monkeypatch, probe):
    world = bbed(monkeypatch)
    world.cap._departure = probe
    assert_refused(world, world.refresh())


# 5: census authority is unchanged ------------------------------------------


@pytest.mark.parametrize("shape", ["incomplete", "raises", "legacy"])
def test_unauthoritative_census_still_refuses(monkeypatch, shape):
    world = bbed(monkeypatch)

    def census(pgid, deadline):
        view = world.first if world.calls % 2 == 0 else world.second
        world.calls += 1
        if world.calls < 2:
            return owned.AuthoritativeCensus(tuple(view), True, True)
        if shape == "incomplete":
            return owned.AuthoritativeCensus(tuple(view), False, True)
        if shape == "raises":
            raise owned.Unknown("synthetic census failure")
        return owned.CensusResult(tuple(view), True, True)

    world.cap.census = census
    verdict = world.refresh()
    assert verdict.status == "REFUSED"
    assert "vanished" not in world.cap.probe


# 6: the witness is never amnestied -----------------------------------------


@pytest.mark.parametrize("pid", ["leader", "pgid"])
def test_leader_or_pgid_is_never_departed(monkeypatch, pid):
    world = bbed(monkeypatch)
    number = world.cap.leader if pid == "leader" else world.cap.pgid
    row = {"pid": number, "reason": "read-error", "errno": errno.ENOENT}
    departed, unresolved = world.cap._departed([number], [number, 22], [22], {number: row})
    assert departed == {}
    assert unresolved == [number]


# 7: pruning to nothing never reaches an emptiness verdict ------------------


def test_pruning_every_member_refuses(monkeypatch):
    world = World(monkeypatch, [20], [], errors={20: gone(20)})
    # A leader the census never lists is allowed only after a send we made.
    world.cap.terminating_sends.append((LEADER, 15))
    verdict = world.refresh(terminal=True)
    assert verdict.status == "REFUSED"
    assert verdict.reason == "Unknown:every census member departed: no member to prove"
    assert verdict.members == (20,)
    assert "vanished" not in world.cap.probe
    assert not world.cap.accepted_terminal(verdict)


# 8: an injected sampler never qualifies -------------------------------------


def test_injected_samples_path_never_qualifies_by_default(monkeypatch):
    world = bbed(monkeypatch)
    world.cap._departure = None  # the product default, not a positive control
    # The fixture sampler simply omits 20/21 (they have no rows): exactly the
    # missing-entry shape, but with no per-PID errno this module produced.
    world.cap.samples = world.kernel.samples
    verdict = world.refresh()
    assert_refused(world, verdict)


# 9: witness checks and revocation are unchanged -----------------------------


@pytest.mark.parametrize(
    "after, status, refusal",
    [("999", "REFUSED", "pin_lost"), (OSError(errno.EIO, "synthetic"), "REFUSED", "unproven")],
    ids=["reaped", "unreadable"],
)
def test_witness_lost_after_the_sample_still_refuses(monkeypatch, after, status, refusal):
    world = bbed(monkeypatch, first=[LEADER, 20, 21], second=[LEADER], live=())
    world.leader_after = after
    verdict = world.refresh(terminal=True)
    assert verdict.status == status
    assert verdict.refusal_class == refusal
    assert "vanished" not in world.cap.probe


def test_cached_verified_after_revocation_is_refused(monkeypatch):
    world = bbed(monkeypatch)
    assert world.refresh().status == "VERIFIED"
    assert world.cap.probe["vanished"]
    world.cap.revoke()
    later = world.refresh()
    assert later.status == "REFUSED"
    assert later.refusal_class == "pin_lost"
    assert "vanished" not in world.cap.probe


# 10: /proc must be our PID namespace (C10) ---------------------------------


@pytest.mark.parametrize(
    "proc_self",
    [lambda: "1", lambda: str(os.getpid() + 1), _raises(OSError(errno.ENOENT, "no /proc/self"))],
    ids=["foreign-1", "foreign-other", "unreadable"],
)
def test_foreign_or_unreadable_proc_self_refuses(monkeypatch, proc_self):
    world = bbed(monkeypatch)
    world.cap._proc_self = proc_self
    assert_refused(world, world.refresh())


# 11: the decision uses the snapshot taken right after the member sample (C11)


def test_witness_reread_before_the_decision_does_not_change_it(monkeypatch):
    world = bbed(monkeypatch)
    original = world.cap._departed
    rereads = []

    def interleaved(absent, first, second, failures):
        # A leader-only `_samples` call resets `_sample_failures`; the decision
        # must already hold its own copy of the member evidence.
        rereads.append(world.cap._leader_witness(world.kernel.clock() + 5)[0])
        assert world.cap._sample_failures == []
        return original(absent, first, second, failures)

    world.cap._departed = interleaved
    verdict = world.refresh()
    assert rereads == ["held"]
    assert verdict.status == "VERIFIED"
    assert verdict.members == (LEADER, 22)


# 12: the evidence reaches the run record (C12) -------------------------------


def test_departed_evidence_reaches_the_run_record(request, monkeypatch, tmp_path):
    runner = request.getfixturevalue("wrapper")
    world = bbed(monkeypatch, first=[LEADER, 20, 21], second=[LEADER], live=())
    _install(runner, monkeypatch, world.cap)
    output = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", lambda path, *a, **k: world.read(path))
        rc = runner.run(
            ["synthetic"],
            timeout=1,
            grace=0,
            memory_mb=0,
            stream=output,
            run_id="synthetic-departed",
        )
    text = output.getvalue()
    assert rc == 0, text
    assert "cleanup=proven-empty" in text
    assert "census member sample unreadable" not in text
    # A successful run writes only its `exit` record; the finally-refresh
    # record exists solely for refusals, and there must be none.
    assert _records(tmp_path, "finally-refresh") == []
    receipt = json.loads(_records(tmp_path, "exit")[0].read_text())
    assert receipt["verdict"]["status"] == "OBSERVED_EMPTY"
    ownership = receipt["ownership"]
    assert [row["pid"] for row in ownership["vanished"]] == [20, 21]
    assert {row["getpgid"] for row in ownership["vanished"]} == {"ProcessLookupError"}


# 14: a both-census member that dies before its sample re-runs the cycle ------
#
# CI 41748872018 (job 126359450123) refused a healthy fast-forking command:
# censuses [[2425,2443,2444,2445,2446],[2425,2444,2445,2446]] with 2444 -- in
# BOTH -- failing its sample at ENOENT. `_departed` must not drop such a PID (a
# member seen in census 2 may have forked a live child), so the capability
# discards the whole cycle and proves again from fresh censuses. Nothing from a
# discarded cycle reaches the verdict.


def in_both(monkeypatch, **overrides):
    """21 is in both censuses and reads ENOENT; 22 stays live."""
    kwargs = {
        "first": [LEADER, 21, 22],
        "second": [LEADER, 21, 22],
        "errors": {21: gone(21)},
        "live": (22,),
    }
    kwargs.update(overrides)
    return World(monkeypatch, kwargs.pop("first"), kwargs.pop("second"), **kwargs)


def test_measured_both_census_shape_converges_on_a_fresh_cycle(monkeypatch):
    world = World(
        monkeypatch,
        [LEADER, 22],
        [LEADER, 22],
        errors={20: gone(20), 21: gone(21)},
        live=(22,),
        script=[[LEADER, 20, 21, 22], [LEADER, 21, 22], [LEADER, 22], [LEADER, 22]],
    )
    verdict = world.refresh()
    assert verdict.status == "VERIFIED", verdict
    assert verdict.members == (LEADER, 22)
    assert world.calls == 4
    assert world.cap.probe["members"] == [LEADER, 22]
    assert world.cap.probe["sample_retries"] == [
        {
            "cycle": 0,
            "rerun": [21],
            "rerun_count": 1,
            "censuses": [[LEADER, 20, 21, 22], [LEADER, 21, 22]],
        }
    ]
    # The verdict rests on the final cycle only: its censuses departed nobody,
    # so the first cycle's departure of 20 is not published as evidence.
    assert "vanished" not in world.cap.probe


def test_late_fork_survivor_is_found_by_the_fresh_census(monkeypatch):
    # The hole `_departed` guards: 21 forks a live 30 after census 2, then dies.
    # A re-run must see 30 and keep it, never certify the group empty.
    world = World(
        monkeypatch,
        [LEADER, 30],
        [LEADER, 30],
        errors={21: gone(21)},
        live=(30,),
        script=[[LEADER, 21], [LEADER, 21], [LEADER, 30], [LEADER, 30]],
    )
    verdict = world.refresh(terminal=True)
    assert verdict.status == "VERIFIED", verdict
    assert 30 in verdict.members
    assert world.calls == 4


def test_a_trigger_persisting_through_every_cycle_still_refuses(monkeypatch):
    world = in_both(monkeypatch)
    verdict = world.refresh()
    assert_refused(world, verdict)
    assert world.calls == 6
    retries = world.cap.probe["sample_failure"]["sample_retries"]
    assert [row["cycle"] for row in retries] == [0, 1]
    assert all(row["rerun"] == [21] for row in retries)


def test_a_cycle_that_would_not_fit_the_deadline_is_not_started(monkeypatch):
    world = in_both(monkeypatch)
    census = world.census

    def slow(pgid, deadline):
        # Each census costs 1.5s of the 5s budget, so the first cycle takes ~3s
        # and twice that no longer fits: no second cycle may start.
        world.kernel.now += 1.5
        return census(pgid, deadline)

    world.cap.census = slow
    verdict = world.refresh()
    assert_refused(world, verdict)
    assert world.calls == 2
    assert world.cap.probe["sample_failure"]["sample_retries"] == []


def _counterexample(name):
    """Each removes exactly one `_rerunnable` conjunct from the in-both shape."""

    def errors(failure):
        return lambda world: world.errors.update({21: failure})

    def departure(probe):
        return lambda world: setattr(world.cap, "_departure", probe)

    def missing_diagnostic(world):
        original = procstate._proc_samples
        world.monkeypatch.setattr(procstate, "_proc_samples", lambda pids, **kwargs: original(pids))

    def injected_sampler(world):
        world.cap._departure = None
        world.cap.samples = world.kernel.samples

    return {
        "EACCES": errors(PermissionError(errno.EACCES, "synthetic")),
        "EIO": errors(OSError(errno.EIO, "synthetic")),
        "short-row": errors("short-row"),
        "missing-diagnostic": missing_diagnostic,
        "getpgid-our-pgid": departure(lambda pid: LEADER),
        "getpgid-other-pgid": departure(lambda pid: 4242),
        "getpgid-EPERM": departure(_raises(PermissionError(errno.EPERM, "synthetic"))),
        "getpgid-EACCES": departure(_raises(PermissionError(errno.EACCES, "synthetic"))),
        "getpgid-EINVAL": departure(_raises(OSError(errno.EINVAL, "synthetic"))),
        "getpgid-other": departure(_raises(RuntimeError("synthetic"))),
        "foreign-proc-self": lambda world: setattr(world.cap, "_proc_self", lambda: "1"),
        "unreadable-proc-self": lambda world: setattr(
            world.cap, "_proc_self", _raises(OSError(errno.ENOENT, "no /proc/self"))
        ),
        "injected-sampler": injected_sampler,
    }[name]


COUNTEREXAMPLES = [
    "EACCES",
    "EIO",
    "short-row",
    "missing-diagnostic",
    "getpgid-our-pgid",
    "getpgid-other-pgid",
    "getpgid-EPERM",
    "getpgid-EACCES",
    "getpgid-EINVAL",
    "getpgid-other",
    "foreign-proc-self",
    "unreadable-proc-self",
    "injected-sampler",
]


@pytest.mark.parametrize("name", COUNTEREXAMPLES)
def test_each_missing_conjunct_refuses_without_a_retry(monkeypatch, name):
    world = in_both(monkeypatch)
    _counterexample(name)(world)
    assert_refused(world, world.refresh())
    assert world.calls == 2
    assert "sample_retries" not in world.cap.probe


def test_past_the_diagnostic_cap_refuses_without_a_retry(monkeypatch):
    many = list(range(20, 37))
    world = World(
        monkeypatch,
        [LEADER, *many],
        [LEADER, *many],
        errors={pid: gone(pid) for pid in many},
    )
    assert_refused(world, world.refresh())
    assert world.calls == 2
    assert world.cap.probe["sample_failure"]["unresolved_count"] == 17


def test_one_unexplained_member_refuses_the_whole_retry(monkeypatch):
    # 21 qualifies, 23 is EACCES: a retry needs EVERY unresolved member.
    world = in_both(
        monkeypatch,
        first=[LEADER, 21, 22, 23],
        second=[LEADER, 21, 22, 23],
        errors={21: gone(21), 23: PermissionError(errno.EACCES, "synthetic")},
    )
    assert_refused(world, world.refresh())
    assert world.calls == 2


def test_a_retry_record_never_leaks_into_a_later_refresh(monkeypatch):
    world = World(
        monkeypatch,
        [LEADER, 22],
        [LEADER, 22],
        errors={21: gone(21)},
        live=(22,),
        script=[[LEADER, 21, 22], [LEADER, 21, 22]],
    )
    assert world.refresh().status == "VERIFIED"
    assert world.cap.probe["sample_retries"]
    assert world.refresh().status == "VERIFIED"
    assert "sample_retries" not in world.cap.probe


def _diagnostic(pid):
    return {pid: {"pid": pid, "reason": "read-error", "errno": errno.ENOENT}}


# Conjuncts `_departed`'s ordering keeps from deciding through `refresh` are
# pinned on the predicate itself, each against the same qualifying control.


@pytest.mark.parametrize("pid", ["leader", "pgid"])
def test_rerunnable_never_reruns_on_the_witness(monkeypatch, pid):
    world = in_both(monkeypatch)
    number = world.cap.leader if pid == "leader" else world.cap.pgid
    assert world.cap._rerunnable([21], [LEADER, 21], _diagnostic(21)) is True
    assert world.cap._rerunnable([number], [number, 21], _diagnostic(number)) is False


def test_rerunnable_requires_census_two(monkeypatch):
    world = in_both(monkeypatch)
    assert world.cap._rerunnable([21], [LEADER, 21], _diagnostic(21)) is True
    assert world.cap._rerunnable([21], [LEADER], _diagnostic(21)) is False


def test_rerunnable_refuses_an_injected_sampler_even_with_a_diagnostic(monkeypatch):
    # The `injected-sampler` counterexample above refuses for want of a row;
    # here the row is present and ESRCH answers, so the sampler alone decides.
    world = in_both(monkeypatch)
    assert world.cap._rerunnable([21], [LEADER, 21], _diagnostic(21)) is True
    world.cap.samples = world.kernel.samples
    assert world.cap._rerunnable([21], [LEADER, 21], _diagnostic(21)) is False


@pytest.mark.parametrize(
    "budget, calls",
    # One cycle costs exactly 2.0 on a frozen clock, ending at 2.0. A further
    # cycle needs 2.0 + 2 * 2.0 = 6.0 STRICTLY below the deadline: at 6.0 it is
    # refused (pins `<`, not `<=`), and a 1x factor (4.0 < 6.0) would retry.
    [(6.0, 2), (6.25, 4)],
    ids=["equal-refuses", "just-fits-reruns-once"],
)
def test_the_budget_needs_twice_the_last_cycle_strictly(monkeypatch, budget, calls):
    world = in_both(monkeypatch)
    now = [0.0]
    census = world.census

    def timed(pgid, deadline):
        now[0] += 1.0
        return census(pgid, deadline)

    world.cap.clock = lambda: now[0]
    world.cap.census = timed
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", lambda path, *a, **k: world.read(path))
        verdict = world.cap.refresh(deadline=budget)
    assert_refused(world, verdict)
    assert world.calls == calls


@pytest.mark.parametrize("shape", ["raises", "incomplete"])
def test_a_census_failure_after_a_retry_refuses_on_its_own_reason(monkeypatch, shape):
    world = in_both(monkeypatch)
    census = world.census

    def failing(pgid, deadline):
        if world.calls < 2:
            return census(pgid, deadline)
        world.calls += 1
        if shape == "raises":
            raise owned.Unknown("synthetic census failure")
        return owned.AuthoritativeCensus((LEADER, 22), False, True, "synthetic gap")

    world.cap.census = failing
    verdict = world.refresh()
    assert verdict.status == "REFUSED"
    assert (
        verdict.reason
        == {
            "raises": "Unknown:synthetic census failure",
            "incomplete": "Unknown:census incomplete: synthetic gap",
        }[shape]
    )
    assert world.calls == 3
    assert verdict.members == ()
    assert world.cap.probe["residuals"] == []
    # The discarded cycle 0 is recorded only as a retry row; its members and
    # observations must not sit beside cycle 1's refusal as if they were its own.
    assert [row["cycle"] for row in world.cap.probe["sample_retries"]] == [0]
    assert "members" not in world.cap.probe
    assert "observed" not in world.cap.probe
    assert "vanished" not in world.cap.probe
