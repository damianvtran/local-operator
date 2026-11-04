"""Synthetic Linux finalization histories: diagnostic facts never grant authority."""

import errno
import io
import json
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


def exercise(monkeypatch, transition, diagnostics=True, run=None):
    monkeypatch.setattr(procstate, "birth_scheme", lambda: SCHEME)
    kernel = Kernel(birth="100")
    cap = kernel.cap()
    kernel.corpses.add(kernel.leader)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "200")
    censuses = 0
    reads = []

    def read(path):
        pid = int(str(path).split("/")[-2])
        reads.append(pid)
        if censuses >= 2 and pid == 20:
            if transition == "departed":
                kernel.rows.pop(20, None)
                raise FileNotFoundError(errno.ENOENT, "synthetic PID departed")
            if transition == "denied":
                raise PermissionError(errno.EACCES, "synthetic unreadable live PID")
            if transition == "io-error":
                raise OSError(errno.EIO, "synthetic unreadable live PID")
            if transition == "short-row":
                return "20 (synthetic) R 10 10 10"
        parent, group, sid, birth = kernel.rows[pid]
        state = "Z" if pid in kernel.corpses else "R"
        tail = [state, str(parent), str(group), str(sid)] + ["0"] * 15 + [birth]
        return f"{pid} (synthetic) " + " ".join(tail)

    def census(pgid, deadline):
        nonlocal censuses
        pids, complete, leader_seen = owned.strict_linux(
            pgid,
            deadline,
            listdir=lambda path: [str(pid) for pid in kernel.rows],
            read=read,
            stat=lambda path: None,
            clock=kernel.clock,
        )
        censuses += 1
        return owned.AuthoritativeCensus(tuple(pids), complete, leader_seen)

    def samples(pids):
        result = procstate._proc_samples(list(pids))
        if censuses >= 2 and transition == "foreign-leader" and 20 in pids:
            row = kernel.rows[kernel.leader]
            kernel.rows[kernel.leader] = (*row[:3], "999")
        return result

    if not diagnostics:
        original = procstate._proc_samples
        monkeypatch.setattr(procstate, "_proc_samples", lambda pids, **kwargs: original(pids))
    cap.census = census
    if transition == "foreign-leader":
        cap.samples = samples
    else:
        cap.samples = None
        monkeypatch.setattr(owned, "sys", NS(platform="linux"))
    # Only synthetic /proc reads are intercepted, for the exact call under test.
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", lambda path, *args, **kwargs: read(path))
        if run is not None:
            result = run(cap)
            assert kernel.sent == []
            return result
        verdict = cap.refresh(deadline=kernel.clock() + 5, terminal=True)
        if transition == "foreign-leader":
            kernel.rows[kernel.leader] = (kernel.parent, kernel.leader, kernel.leader, "100")
            later = cap.refresh(deadline=kernel.clock() + 5, terminal=True)
            assert later.status == "REFUSED"
            assert later.refusal_class == "pin_lost"
    assert kernel.sent == []
    print(
        transition,
        verdict.status,
        verdict.reason,
        "unresolved=",
        cap.probe.get("unresolved"),
        "member-present=",
        20 in kernel.rows,
        "censuses=",
        censuses,
    )
    return verdict, cap


@pytest.mark.parametrize("transition", ["departed", "denied", "io-error", "short-row"])
def test_histories_collapse_to_same_refusal(monkeypatch, transition):
    verdict, cap = exercise(monkeypatch, transition)
    assert verdict.status == "REFUSED"
    assert verdict.reason == "Unknown:census member sample unreadable"
    assert cap.probe["unresolved"] == [20]
    detail = cap.probe["sample_failure"]
    assert detail["phase"] == "terminal-member-sample"
    assert detail["censuses"] == [[10, 20], [10, 20]]
    assert detail["witness"] == "held"
    reason = "short-stat-row" if transition == "short-row" else "read-error"
    error = {
        "departed": errno.ENOENT,
        "denied": errno.EACCES,
        "io-error": errno.EIO,
        "short-row": None,
    }[transition]
    assert detail["unresolved"] == [{"pid": 20, "reason": reason, "errno": error}]


def test_readable_foreign_leader_revokes_irreversibly(monkeypatch):
    verdict, _ = exercise(monkeypatch, "foreign-leader")
    assert verdict.status == "REFUSED"
    assert verdict.refusal_class == "pin_lost"


def test_live_member_control_remains_verified(monkeypatch):
    verdict, _ = exercise(monkeypatch, "live")
    assert verdict.status == "VERIFIED"


@pytest.mark.parametrize(
    "transition", ["departed", "denied", "io-error", "short-row", "live", "foreign-leader"]
)
def test_diagnostics_do_not_change_verdicts(monkeypatch, transition):
    with monkeypatch.context() as scoped:
        before, _ = exercise(scoped, transition, diagnostics=False)
    with monkeypatch.context() as scoped:
        after, _ = exercise(scoped, transition)
    assert before == after


def test_old_failures_do_not_survive_a_new_sample_attempt(monkeypatch):
    verdict, cap = exercise(monkeypatch, "denied")
    assert verdict.status == "REFUSED"
    cap.samples = lambda pids: {pid: procstate.ProcessSample(True, "100") for pid in pids}
    cap.census = lambda pgid, deadline: owned.AuthoritativeCensus((10,), True, True)
    later = cap.refresh(deadline=cap.clock() + 5, terminal=True)
    assert later.status == "OBSERVED_EMPTY"
    assert "sample_failure" not in cap.probe
    assert "unresolved" not in cap.probe


def test_revocation_does_not_relabel_old_sampling_evidence(monkeypatch):
    _, cap = exercise(monkeypatch, "denied")
    assert "sample_failure" in cap.probe
    cap.revoke()
    later = cap.refresh(deadline=cap.clock() + 5, terminal=True)
    assert later.status == "REFUSED"
    assert later.refusal_class == "pin_lost"
    assert "sample_failure" not in cap.probe
    assert "unresolved" not in cap.probe


def test_broken_optional_sink_does_not_change_missing_sample(monkeypatch):
    class BrokenSink(list[dict[str, object]]):
        def append(self, value):
            raise RuntimeError("synthetic diagnostic sink unavailable")

    def unreadable(path, *args, **kwargs):
        raise PermissionError(errno.EACCES, "synthetic")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", unreadable)
        assert procstate._proc_samples([20]) == {}
        assert procstate._proc_samples([20], diagnostics=BrokenSink()) == {}


@pytest.mark.parametrize("transition", ["departed", "denied", "io-error", "short-row"])
def test_run_refusal_retains_same_rc_and_stderr_receipt_evidence(
    request, monkeypatch, tmp_path, transition
):
    runner = request.getfixturevalue("wrapper")

    def run(cap):
        _install(runner, monkeypatch, cap)
        output = io.StringIO()
        rc = runner.run(
            ["synthetic"],
            timeout=1,
            grace=0,
            memory_mb=0,
            stream=output,
            run_id="synthetic-sample-diagnostics",
        )
        return rc, output.getvalue()

    rc, output = exercise(monkeypatch, transition, run=run)
    # A departed member is later omitted by a NEW complete census, so the
    # retained corpse can certify no live members. Persistent unreadability
    # remains unknown. Do not turn the first missing sample into absence.
    assert rc == (0 if transition == "departed" else 126), output
    assert ("cleanup=proven-empty" if transition == "departed" else "cleanup=unknown") in output
    assert "finally-refresh refused: Unknown:census member sample unreadable" in output
    lines = [line for line in output.splitlines() if "sample_failure=" in line]
    assert lines, output
    detail = json.loads(lines[0].split("sample_failure=", 1)[1])
    assert detail["phase"] == "member-sample"
    assert detail["unresolved"][0]["pid"] == 20
    assert detail["unresolved"][0]["reason"] == (
        "short-stat-row" if transition == "short-row" else "read-error"
    )
    receipt = json.loads(_records(tmp_path, "finally-refresh")[0].read_text())
    assert receipt["ownership"]["sample_failure"] == detail
    with monkeypatch.context() as scoped:
        rc_without, output_without = exercise(scoped, transition, diagnostics=False, run=run)
    assert rc_without == rc
    assert "finally-refresh refused: Unknown:census member sample unreadable" in output_without


def test_optional_sink_bounds_rows_without_extra_kernel_reads(monkeypatch):
    reads = []

    def unreadable(path, *args, **kwargs):
        reads.append(str(path))
        raise OSError(errno.EIO, "synthetic text must not enter diagnostics")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", unreadable)
        assert procstate._proc_samples(range(20, 40)) == {}
        original_reads = list(reads)
        reads.clear()
        detail = []
        assert procstate._proc_samples(range(20, 40), diagnostics=detail) == {}
    assert reads == original_reads
    assert len(detail) == 16
    assert detail == [
        {"pid": pid, "reason": "read-error", "errno": errno.EIO} for pid in range(20, 36)
    ]


def test_serialized_diagnostics_are_bounded(request):
    runner = request.getfixturevalue("wrapper")
    cap = NS(probe={"sample_failure": {"unresolved": ["x" * 5000]}})
    assert runner._sample_failure_detail(cap) == {"diagnostic_truncated": True}
    cap.probe["sample_failure"] = {"bad": object()}
    assert runner._sample_failure_detail(cap) == {"diagnostic_truncated": True}
