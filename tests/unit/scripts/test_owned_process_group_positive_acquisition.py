"""Linux creation is a positive receipt, never population or send authority.

All kernel identities, samples, censuses and sends below are synthetic. These
controls intentionally preserve strict failures after acquisition.
"""

from __future__ import annotations

import io
import json
import os
import signal
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from local_operator.procstate import ProcessSample
from tests.unit.scripts.test_owned_process_group_reservation import (
    SCHEME,
    Kernel,
    _as_typed_census,
)
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401
from tests.unit.scripts.test_run_bounded_run_record import _install


@pytest.fixture(autouse=True)
def linux(monkeypatch):
    monkeypatch.setattr(owned, "sys", NS(platform="linux"))
    monkeypatch.setattr(owned.procstate, "birth_scheme", lambda: SCHEME)
    monkeypatch.setattr(
        owned,
        "signal",
        NS(
            SIGCHLD=signal.SIGCHLD,
            SIG_DFL=signal.SIG_DFL,
            SIG_IGN=signal.SIG_IGN,
            SIGKILL=signal.SIGKILL,
            SIGTERM=signal.SIGTERM,
            getsignal=lambda _: signal.SIG_DFL,
        ),
    )


def capture(kernel, *, census=None, deadline=5):
    def unstable(pgid, until):
        kernel.censuses += 1
        raise owned.Unknown("unstable /proc census")

    return owned.OwnedGroup.capture(
        kernel.leader,
        parent_pid=kernel.parent,
        census=census or unstable,
        samples=kernel.samples,
        identity=kernel.identity,
        scheme=lambda: SCHEME,
        clock=kernel.clock,
        signal_group=kernel.send,
        deadline=deadline,
    )


def test_live_creation_survives_unstable_census_without_population_claim():
    kernel = Kernel()
    sample_calls, identity_calls = [], []
    original_samples, original_identity = kernel.samples, kernel.identity

    def samples(pids):
        sample_calls.append(set(pids))
        return original_samples(pids)

    def identity(pid):
        identity_calls.append(pid)
        return original_identity(pid)

    kernel.samples, kernel.identity = samples, identity
    cap = capture(kernel)
    assert cap.last.status == "ACQUIRED_POSITIVE", cap.last.reason
    assert sample_calls == [{kernel.leader, kernel.parent}] * 2
    assert identity_calls == [kernel.leader] * 2
    assert kernel.censuses == 0
    assert cap.last.members == ()
    assert cap.probe["members"] == []
    assert cap.probe["proof"] == []
    assert cap.last.acquisition_kind == "linux-live-creation"
    assert cap.last.population_coverage == "not-established"
    assert cap.last.creation_receipt is cap.receipts[kernel.leader]
    assert cap.last.creation_receipt.birth == "birth"
    assert cap.last.creation_receipt.parent == os.getpid()
    assert (
        cap.probe["exclusive_reap_precondition"] == "caller retains exclusive wait/reap ownership"
    )
    assert cap._terminal_result is None
    assert cap.terminating_sends == []
    assert not cap.accepted_terminal(cap.last)
    assert kernel.sent == []


def test_acquisition_never_substitutes_for_fresh_send_or_terminal_proof():
    kernel = Kernel()
    cap = capture(kernel)
    assert cap.last.status == "ACQUIRED_POSITIVE"
    sent = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5)
    assert sent.status == "REFUSED", sent.reason
    assert "unstable /proc census" in sent.reason
    assert kernel.censuses > 0
    assert kernel.sent == []
    terminal = cap.refresh(deadline=kernel.clock() + 5, terminal=True)
    assert terminal.status == "REFUSED"
    assert not cap.accepted_terminal(terminal)
    assert cap._terminal_result is None


def test_later_complete_proof_checks_arrivals_then_sends_and_certifies():
    kernel = Kernel()
    cap = capture(kernel)
    kernel.rows[20] = (kernel.leader, kernel.leader, kernel.leader, "arrival")
    cap.census = _as_typed_census(kernel.census, lambda: kernel.rows)
    sent = cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5)
    assert sent.status == "SIGNALED", sent.reason
    assert set(sent.members) == {10, 20}
    assert kernel.censuses >= 2
    assert kernel.sent == [(10, signal.SIGKILL)]
    terminal = cap.refresh(deadline=kernel.clock() + 5, terminal=True)
    assert terminal.status == "OBSERVED_EMPTY", terminal.reason
    assert cap.accepted_terminal(terminal)
    assert not cap.accepted_terminal(owned.Verdict("OBSERVED_EMPTY", terminal.reason))
    cap.revoke()
    assert not cap.accepted_terminal(terminal)


@pytest.mark.parametrize(
    "boundary", ["foreign-birth", "unreadable", "revoked", "expired", "wrong-arrival", "incomplete"]
)
def test_acquisition_cannot_recover_later_lost_or_unreadable_authority(boundary):
    kernel = Kernel()
    cap = capture(kernel)
    cap.census = _as_typed_census(kernel.census, lambda: kernel.rows)
    if boundary == "foreign-birth":
        row = kernel.rows[10]
        kernel.rows[10] = (*row[:3], "foreign")
    elif boundary == "unreadable":
        original = kernel.samples
        cap.samples = lambda pids: {
            pid: value for pid, value in original(pids).items() if pid != 10
        }
    elif boundary == "revoked":
        cap.revoke()
    elif boundary == "expired":
        kernel.now = 10
    elif boundary == "wrong-arrival":
        kernel.rows[20] = (10, 10, 99, "foreign-session")
    else:
        cap.census = lambda pgid, until: owned.CensusResult((), True, True)
    verdict = cap.signal(signal.SIGKILL, deadline=5)
    assert verdict.status == "REFUSED", verdict.reason
    assert kernel.sent == []
    if boundary == "foreign-birth":
        kernel.rows[10] = (kernel.parent, 10, 10, "birth")
        assert cap.signal(signal.SIGKILL, deadline=5).status == "REFUSED"
        assert cap._revoked


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_positive_capture_requires_finite_unexpired_deadline(deadline):
    kernel = Kernel()
    cap = capture(kernel, deadline=deadline)
    assert cap.last.status == "REFUSED", cap.last.reason
    assert not cap.receipts
    assert kernel.sent == []


@pytest.mark.parametrize(
    "boundary",
    [
        "nonchild",
        "wrong-group",
        "wrong-session",
        "missing",
        "foreign-birth",
        "parent-terminal",
        "leader-transition",
        "ignored",
        "custom",
        "handler-transition",
        "self",
        "own-group",
        "expiry-at-publication",
    ],
)
def test_positive_creation_preconditions_refuse_without_grants(monkeypatch, boundary):
    kernel = Kernel()
    original = kernel.samples
    calls = 0

    def samples(pids):
        nonlocal calls
        calls += 1
        result = original(pids)
        if boundary == "missing":
            result.pop(10, None)
        elif boundary == "foreign-birth" and calls >= 2:
            result[10] = ProcessSample(False, "foreign")
        elif boundary == "parent-terminal":
            result[kernel.parent] = ProcessSample(True, "creator")
        elif boundary == "leader-transition" and calls >= 2:
            result[10] = ProcessSample(True, "birth")
        elif boundary == "handler-transition" and calls >= 2:
            monkeypatch.setattr(owned.signal, "getsignal", lambda _: (lambda *args: None))
        elif boundary == "expiry-at-publication" and calls >= 2:
            kernel.now = 6
        return result

    kernel.samples = samples
    if boundary in ("nonchild", "wrong-group", "wrong-session"):
        parent, group, sid, birth = kernel.rows[10]
        kernel.rows[10] = (
            parent + 1 if boundary == "nonchild" else parent,
            99 if boundary == "wrong-group" else group,
            99 if boundary == "wrong-session" else sid,
            birth,
        )
    elif boundary in ("ignored", "custom"):
        monkeypatch.setattr(
            owned.signal,
            "getsignal",
            lambda _: 1 if boundary == "ignored" else (lambda *args: None),
        )
    elif boundary in ("self", "own-group"):
        monkeypatch.setattr(
            owned,
            "os",
            NS(
                getpid=lambda: 10 if boundary == "self" else kernel.parent,
                getpgrp=lambda: 10 if boundary == "own-group" else 99999,
            ),
        )
    cap = capture(kernel)
    assert cap.last.status == "REFUSED", (boundary, cap.last.reason)
    assert not cap.receipts
    assert cap.probe["proof"] == []
    assert cap._terminal_result is None
    assert kernel.sent == []


def test_positive_acquire_serializes_no_population_and_final_strict_failure(
    request, monkeypatch, tmp_path
):
    kernel = Kernel()
    cap = capture(kernel)
    runner = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    _install(runner, monkeypatch, cap)
    kernel.corpses.add(10)
    output = io.StringIO()
    assert (
        runner.run(
            ["synthetic"],
            timeout=10,
            grace=0,
            memory_mb=0,
            stream=output,
            run_id="positive-acquisition",
        )
        == 126
    )
    assert "finally-refresh refused" in output.getvalue()
    record = json.loads(next((tmp_path / "run-bounded-runs").glob("*.spawn.json")).read_text())
    assert record["verdict"]["status"] == "ACQUIRED_POSITIVE"
    for value in (record["verdict"], record["ownership"]):
        assert value["acquisition_kind"] == "linux-live-creation"
        assert value["population_coverage"] == "not-established"
        assert value["creation_receipt"]["birth"] == "birth"
    assert record["ownership"]["proof"] == []
    assert record["verdict"]["members"] == []
    assert kernel.sent == []


@pytest.mark.parametrize("cached", [False, True])
def test_already_due_first_refresh_is_not_postponed_or_allowed_to_reuse_acquisition(
    request, monkeypatch, tmp_path, cached
):
    kernel = Kernel()
    cap = capture(kernel)
    if cached:
        acquired = cap.last
        cap.refresh = lambda **kwargs: acquired
    runner = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    _install(runner, monkeypatch, cap)
    ticks = 0

    def now():
        nonlocal ticks
        ticks += 1
        return 1000 if ticks == 1 else 1002

    monkeypatch.setattr(runner, "time", NS(monotonic=now, sleep=lambda _: None))
    cap.leader_terminal = lambda until: pytest.fail("required refresh is already due")
    output = io.StringIO()
    assert (
        runner.run(
            ["synthetic"], timeout=10, grace=0, memory_mb=0, stream=output, run_id="already-due"
        )
        == 126
    )
    assert "required-refresh refused" in output.getvalue()
    if cached:
        assert "ACQUIRED_POSITIVE" in output.getvalue()
    else:
        assert kernel.censuses > 0
    assert kernel.sent == []


@pytest.mark.parametrize(
    "boundary",
    ["parent-missing", "birth-unreadable", "parent-reuse", "identity-transition", "corpse"],
)
def test_additional_creation_observations_cannot_publish_positive(boundary):
    kernel = Kernel()
    original_samples, original_identity = kernel.samples, kernel.identity
    samples_count = 0
    identity_count = 0

    def samples(pids):
        nonlocal samples_count
        samples_count += 1
        result = original_samples(pids)
        if boundary == "parent-missing":
            result.pop(kernel.parent, None)
        elif boundary == "birth-unreadable":
            result[10] = ProcessSample(False, None)
        elif boundary == "parent-reuse" and samples_count >= 2:
            result[kernel.parent] = ProcessSample(False, "other-parent")
        return result

    def identity(pid):
        nonlocal identity_count
        identity_count += 1
        return (
            (kernel.parent, 99, 99)
            if boundary == "identity-transition" and identity_count >= 2
            else original_identity(pid)
        )

    kernel.samples, kernel.identity = samples, identity
    if boundary == "corpse":
        kernel.corpses.add(10)
    cap = capture(kernel)
    assert cap.last.status == "REFUSED", cap.last.reason
    assert not cap.receipts
    assert cap.probe["proof"] == []
    assert cap._terminal_result is None
    assert kernel.sent == []


def test_revocation_during_samples_cannot_publish_creation(monkeypatch):
    instances = []
    original_group = owned.OwnedGroup

    class RecordedGroup(original_group):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(owned, "OwnedGroup", RecordedGroup)
    kernel = Kernel()
    original = kernel.samples
    count = 0

    def samples(pids):
        nonlocal count
        count += 1
        if count == 2:
            instances[0].revoke()
        return original(pids)

    kernel.samples = samples
    cap = capture(kernel)
    assert cap.last.status == "REFUSED"
    assert cap._revoked
    assert not cap.receipts
    assert cap.signal(signal.SIGKILL, deadline=kernel.clock() + 5).status == "REFUSED"
    assert kernel.sent == []


def test_acquisition_cached_at_send_phase_is_refused(request, monkeypatch, tmp_path):
    kernel = Kernel()
    cap = capture(kernel)
    acquired = cap.last
    cap.census = _as_typed_census(kernel.census, lambda: kernel.rows)
    cap.signal = lambda *args, **kwargs: acquired
    # Isolate runner phase admission: the terminal observer says finished while
    # the synthetic kernel remains live, so strict terminal cleanup cannot mask
    # the rejected cached send result with a later valid empty certificate.
    cap.leader_terminal = lambda until: True
    runner = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    _install(runner, monkeypatch, cap)
    output = io.StringIO()
    assert (
        runner.run(
            ["synthetic"],
            timeout=10,
            grace=0,
            memory_mb=0,
            stream=output,
            run_id="cached-send-acquisition",
        )
        == 126
    )
    assert "finally refused: unexpected ACQUIRED_POSITIVE in finally" in output.getvalue()
    assert kernel.sent == []


def test_unknown_acquire_status_is_not_monitor_admission(request, monkeypatch, tmp_path):
    kernel = Kernel()
    cap = capture(kernel)
    cap.last = owned.Verdict("UNRECOGNIZED", "synthetic unknown status")
    cap.leader_terminal = lambda until: pytest.fail("unknown status must not admit monitoring")
    runner = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    _install(runner, monkeypatch, cap)
    output = io.StringIO()
    assert (
        runner.run(
            ["synthetic"],
            timeout=10,
            grace=0,
            memory_mb=0,
            stream=output,
            run_id="unknown-acquisition",
        )
        == 126
    )
    assert "acquire refused: unexpected UNRECOGNIZED in acquire" in output.getvalue()
    assert kernel.sent == []


@pytest.mark.parametrize("boundary", ["expired", "nonfinite-clock", "revoked", "custom-handler"])
def test_local_receipt_preparation_cannot_outlive_publication_preconditions(monkeypatch, boundary):
    instances = []
    original_group = owned.OwnedGroup

    class RecordedGroup(original_group):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(owned, "OwnedGroup", RecordedGroup)
    kernel = Kernel()
    original = owned.asdict

    def prepare(receipt):
        view = original(receipt)
        if boundary == "expired":
            kernel.now = 6
        elif boundary == "nonfinite-clock":
            kernel.now = float("nan")
        elif boundary == "revoked":
            instances[0].revoke()
        else:
            monkeypatch.setattr(owned.signal, "getsignal", lambda _: (lambda *args: None))
        return view

    monkeypatch.setattr(owned, "asdict", prepare)
    cap = capture(kernel)
    assert cap.last.status == "REFUSED", cap.last.reason
    assert not cap.receipts
    assert "acquisition_kind" not in cap.probe
    assert cap._terminal_result is None
    assert kernel.sent == []
