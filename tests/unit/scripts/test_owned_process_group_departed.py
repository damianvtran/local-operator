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

    def __init__(self, monkeypatch, first, second, *, errors=None, live=()):
        monkeypatch.setattr(procstate, "birth_scheme", lambda: SCHEME)
        monkeypatch.setattr(owned, "sys", NS(platform="linux"))
        self.monkeypatch = monkeypatch
        self.kernel = Kernel(leader=LEADER, birth="100")
        self.cap = self.kernel.cap()
        self.kernel.corpses.add(LEADER)
        for pid in live:
            self.kernel.rows[pid] = (LEADER, LEADER, LEADER, "200")
        self.first, self.second = list(first), list(second)
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
