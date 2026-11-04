"""Synthetic contract and policy regressions; real CLI coverage lives separately.

Deleting the authority-type guard must break the observation-only promotion test,
not merely change diagnostic text. Environment tests intentionally use the real
builder: replacing it with a stub hid a missing definition in the POC.
"""

from __future__ import annotations

import importlib.util
import io
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator import bounded_contract as contracts
from local_operator import owned_process_group as groups


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_bounded.py"
    spec = importlib.util.spec_from_file_location("bounded_contract_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forbidden(*args, **kwargs):
    pytest.fail("forbidden spawn, group signal, handler mutation, or strict fallback")


def receipt_row(receipt):
    return json.loads(receipt.path.read_text(encoding="utf-8"))


class Available:
    @staticmethod
    def available():
        return True


def invoke(
    tmp_path,
    *,
    contract="command-status",
    direct=None,
    budget=0,
    timeout=None,
    strict=forbidden,
    backend=None,
):
    receipt = contracts.Receipt(tmp_path, "caller-reused-id")
    stream = io.StringIO()
    rc = contracts.execute(
        ["synthetic-command"],
        contract=contract,
        timeout=timeout,
        grace=0.1,
        budget=SimpleNamespace(ceiling_mb=budget),
        child_env={},
        receipt=receipt,
        stream=stream,
        strict=strict,
        backend=backend or contracts.ObservationOnlyBackend(),
        direct=direct or forbidden,
        boundary=Available,
    )
    return rc, receipt, stream.getvalue()


def test_default_refuses_before_spawn_and_handlers(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner, "_memory_budget", lambda mb: (SimpleNamespace(ceiling_mb=0), "test")
    )
    monkeypatch.setattr(runner, "_record_dir", lambda: tmp_path)
    monkeypatch.setattr(runner, "_child_environment", lambda budget: {})
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    monkeypatch.setattr(contracts.signal, "signal", forbidden)
    monkeypatch.setattr(runner, "_signal_group", forbidden)
    stream = io.StringIO()
    assert (
        runner.run(
            ["never-start"],
            timeout=1,
            grace=0.1,
            stream=stream,
            _backend=contracts.ObservationOnlyBackend(),
        )
        == 125
    )
    row = json.loads(next(tmp_path.glob("*.completion.json")).read_text())
    assert row["requested_contract"] == "cleanup-required"
    assert row["termination_cause"] == "unsupported-backend"
    assert row["command_exit_code"] is None
    assert row["cleanup_status"] == "unknown"
    assert "pgid" not in row
    assert not list(tmp_path.glob("*.spawn.json"))


def leader_factory(log, *, status=0, terminal=lambda: True, hook=lambda point: None):
    """A scripted reserved leader: every observe/signal/reap lands in ``log``.

    ``send_group`` refuses after revocation exactly like the real boundary, so a
    test that signals after the consuming reap fails instead of passing silently.
    """

    class Leader:
        observer = "synthetic"

        def __init__(self, proc):
            self.proc = proc
            self.revoked = False

        def observe(self):
            log.append("observe")
            hook("observe")
            return terminal()

        def send_group(self, signum):
            if self.revoked:
                raise RuntimeError("reserved-leader authority irrevocably revoked")
            log.append(("killpg", self.proc.pid, signum))
            return True

        def reap(self):
            self.revoked = True
            log.append("reap")
            hook("reap")
            return status

        def revoke(self):
            self.revoked = True
            hook("revoke")

    return Leader


class Child:
    pid = 43210
    returncode = None

    def poll(self):
        pytest.fail("Popen.poll maps ECHILD to 0 and is forbidden on this path")

    def wait(self, *args, **kwargs):
        pytest.fail("Popen.wait is forbidden on this path")


def direct_with(leader, **extra):
    """A ``direct`` seam running the real supervisor over a scripted leader."""

    def direct(argv, **kwargs):
        kwargs.update(extra)
        contracts.supervise_direct(
            argv,
            **kwargs,
            spawn=lambda *a, **k: Child(),
            boundary=leader,
            group_pids=lambda pgid: [pgid],
        )

    return direct


@pytest.mark.parametrize("child_rc", [0, 7, 124, 125, 126, 130, -signal.SIGTERM])
def test_command_status_uses_direct_child_and_never_group_sender(monkeypatch, tmp_path, child_rc):
    monkeypatch.setattr(groups.OwnedGroup, "capture", forbidden)
    monkeypatch.setattr(contracts.os, "killpg", forbidden)
    log = []
    rc, receipt, text = invoke(tmp_path, direct=direct_with(leader_factory(log, status=child_rc)))
    # Independent oracle: sharing shell_status here let its negative-status
    # mutation survive by changing both the implementation and expectation.
    assert rc == (128 + abs(child_rc) if child_rc < 0 else child_rc)
    row = receipt_row(receipt)
    assert row["command_exit_code"] == rc
    assert row["termination_cause"] == ("child-signal" if child_rc < 0 else "child-exit")
    assert row["cleanup_status"] == "unknown"
    assert row["signal_scope"] == "reserved-leader-group"
    assert "contract=command-status" in text and "cleanup=unknown" in text
    assert "spawned=no" not in text
    # A lone leader exit needs no group signal: the listing showed nobody else.
    assert log == ["observe", "reap"]
    assert not contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )


def test_deadline_is_policy_outcome_not_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(contracts.os, "killpg", forbidden)
    now = [0.0]
    log = []
    # The child dies only once the group has actually been signalled.
    leader = leader_factory(log, terminal=lambda: any(isinstance(e, tuple) for e in log))
    rc, receipt, text = invoke(
        tmp_path,
        timeout=0.01,
        direct=direct_with(
            leader,
            clock=lambda: now[0],
            sleep=lambda dt: now.__setitem__(0, now[0] + dt),
        ),
    )
    row = receipt_row(receipt)
    assert rc == 124 and row["termination_cause"] == "deadline"
    assert [e for e in log if isinstance(e, tuple)] == [("killpg", 43210, signal.SIGTERM)]
    assert log[-1] == "reap"
    assert row["command_exit_code"] == 0 and row["cleanup_status"] == "unknown"
    assert row["memory_guarantee"].startswith("none:")
    assert row["deadline_guarantee"].startswith("none:")
    assert "that is the BOUND" in text


def test_positive_grant_cannot_be_disabled_or_widened(runner, monkeypatch, tmp_path):
    monkeypatch.setenv(runner.memory_guard.RESOLVED_MEMORY_MB_ENV, "128")
    monkeypatch.setattr(runner, "_record_dir", lambda: tmp_path)
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    for requested in (0, 129):
        stream = io.StringIO()
        assert (
            runner.run(
                ["never-start"],
                timeout=None,
                grace=0.1,
                memory_mb=requested,
                contract="command-status",
                stream=stream,
            )
            == 125
        )
        text = stream.getvalue()
        # R8: the refusal names its receipt like every other path, and D1/D4:
        # a human reason line precedes a status line that says nothing ran.
        assert "refused before spawn: --memory-mb cannot disable or widen" in text
        assert "spawned=no" in text and "invocation_id=" in text and "completion=" in text
    # A positive valid grant is ENFORCED by command-status, not refused: the
    # supervisor receives the budget and builds the guard on its own group.
    seen = {}

    def direct(argv, **kwargs):
        seen["ceiling"] = kwargs["budget"].ceiling_mb

    rc, receipt, text = invoke(tmp_path, budget=128, direct=direct)
    row = receipt_row(receipt)
    assert seen == {"ceiling": 128}
    assert row["memory_ceiling_mb"] == 128
    assert row["memory_guarantee"].startswith("sampled owned-group policy")
    assert "memory budget 128 MiB" in text


def test_observation_complete_flag_is_not_authority():
    cap = object.__new__(groups.OwnedGroup)
    # __new__ bypasses the constructor's diagnostic ledger, which _members
    # clears before each attempt; model that invariant without granting proof.
    cap.probe = {}
    cap.clock = lambda: 0
    cap.pgid = 43210
    cap.terminating_sends = []
    # Mutation target: changing AuthoritativeCensus back to CensusResult would
    # accept this positive complete flag and make this test fail.
    cap.census = lambda *args: groups.CensusResult((43210,), True, True)
    with pytest.raises(groups.Unknown):
        cap._members(1)
    cap.census = lambda *args: groups.AuthoritativeCensus((43210,), True, True)
    assert cap._members(1) == [43210]


def test_sentinel_cannot_borrow_authority_from_empty_rows():
    cap = object.__new__(groups.OwnedGroup)
    cap.clock, cap.pgid, cap.terminating_sends = lambda: 0, 43210, [signal.SIGKILL]
    cap.census = contracts.ObservationOnlyBackend().census
    with pytest.raises(groups.Unknown):
        cap._terminal_absence(1)


def test_revocation_beats_cached_verdict_at_mutation_boundary(monkeypatch):
    cap = object.__new__(groups.OwnedGroup)
    cap._revoked = False
    cap.pgid = 43210
    cap.clock = lambda: 0
    cap._send = forbidden
    verified = groups.Verdict("VERIFIED", "cached", (43210,))
    cap.refresh = lambda **kwargs: verified
    cap._leader_witness = lambda deadline: ("held", None)
    cap.revoke()
    # Even replacing refresh with a cached positive control cannot restore send.
    assert cap.signal(signal.SIGKILL, deadline=1).status == "REFUSED"
    cap.revoke()
    assert cap._revoked is True


def test_only_exact_accepted_terminal_can_construct_proven_empty():
    cap = object.__new__(groups.OwnedGroup)
    cap._revoked = False
    cap.last = groups.Verdict("OBSERVED_EMPTY", "synthetic authoritative terminal")
    cap._terminal_result = None
    outcome = contracts.Outcome()
    outcome.accept_terminal(cap, cap.last)
    assert outcome.cleanup_status == "unknown"
    cap._terminal_result = cap.last
    outcome.accept_terminal(cap, cap.last)
    assert outcome.cleanup_status == "proven-empty"
    cap.revoke()
    second = contracts.Outcome()
    second.accept_terminal(cap, cap.last)
    assert second.cleanup_status == "unknown"


def test_missing_stale_and_command_status_receipts_never_admit(tmp_path):
    receipt = contracts.Receipt(tmp_path, "same-id")
    assert not contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )
    outcome = contracts.Outcome(0, 0, "child-exit", "proven-empty", "synthetic test certificate")
    receipt.publish("cleanup-required", outcome)
    assert contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )
    fresh = contracts.Receipt(tmp_path, "same-id")
    assert fresh.invocation_id != receipt.invocation_id
    assert not contracts.admit_cleanup(
        receipt.path, expected_invocation_id=fresh.invocation_id, expected_directory=tmp_path
    )
    fresh.publish("command-status", outcome)  # Forged status alone cannot admit.
    assert not contracts.admit_cleanup(
        fresh.path, expected_invocation_id=fresh.invocation_id, expected_directory=tmp_path
    )
    before = receipt.path.read_bytes()
    with pytest.raises(FileExistsError):
        receipt.publish("cleanup-required", outcome)
    assert receipt.path.read_bytes() == before


def test_failed_fsync_never_publishes_admissible_success(monkeypatch, tmp_path):
    def fail(fd):
        raise OSError("synthetic storage failure")

    monkeypatch.setattr(contracts.os, "fsync", fail)

    def strict(outcome):
        outcome.command_exit_code, outcome.wrapper_exit_code = 0, 0
        outcome.termination_cause = "child-exit"
        outcome.cleanup_status = "proven-empty"

    rc, receipt, text = invoke(
        tmp_path,
        contract="cleanup-required",
        strict=strict,
        backend=contracts.Backend("synthetic-authoritative", True, "test"),
    )
    assert rc == 126 and not receipt.path.exists()
    assert "publication failed" in text
    pending = receipt.path.with_suffix(".pending")
    # Complete strict-success JSON survives fsync failure. JSON fields alone
    # would admit this SAME invocation, so this kills the missing-path guard.
    row = json.loads(pending.read_text())
    assert row["wrapper_exit_code"] == row["command_exit_code"] == 0
    assert row["cleanup_status"] == "proven-empty"
    assert row["requested_contract"] == "cleanup-required"
    assert not contracts.admit_cleanup(
        pending, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )
    assert not contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )


def test_supported_strict_retains_existing_gate_and_nonzero_cause(tmp_path):
    def strict(outcome):
        outcome.wrapper_exit_code = 137
        outcome.termination_cause = "memory-policy"
        outcome.cleanup_reason = "terminal proof failed"

    rc, receipt, text = invoke(
        tmp_path,
        contract="cleanup-required",
        strict=strict,
        backend=contracts.Backend("synthetic-authoritative", True, "test"),
    )
    assert rc == 137
    assert receipt_row(receipt)["termination_cause"] == "memory-policy"


def test_strict_child_success_requires_terminal_certificate(tmp_path):
    def strict(outcome):
        outcome.command_exit_code = outcome.wrapper_exit_code = 0
        outcome.termination_cause = "child-exit"

    rc, receipt, text = invoke(
        tmp_path,
        contract="cleanup-required",
        strict=strict,
        backend=contracts.Backend("synthetic-authoritative", True, "test"),
    )
    assert rc == 126
    row = receipt_row(receipt)
    assert row["command_exit_code"] == 0 and row["cleanup_status"] == "unknown"
    assert row["termination_cause"] == "child-exit"


def test_revoking_wait_that_raises_cannot_restore_direct_send(monkeypatch):
    def failing_waitpid(pid, options):
        raise ChildProcessError("synthetic consuming-wait failure")

    child = contracts.ReservedLeader(Child(), observer="waitid-wnowait")
    monkeypatch.setattr(contracts.os, "killpg", forbidden)
    monkeypatch.setattr(contracts.os, "waitpid", failing_waitpid)
    with pytest.raises(ChildProcessError):
        child.reap()
    with pytest.raises(RuntimeError, match="revoked"):
        child.send_group(signal.SIGTERM)
    child.revoke()
    with pytest.raises(RuntimeError, match="revoked"):
        child.send_group(signal.SIGKILL)
    with pytest.raises(RuntimeError, match="revoked"):
        child.observe()


def test_try_reap_that_consumes_a_status_revokes_direct_send(monkeypatch):
    """R3.1: once try_reap consumed the leader's status the group number is free
    for reuse, so a later killpg could hit a stranger. Revocation must hold."""
    child = contracts.ReservedLeader(Child(), observer="waitid-wnowait")
    monkeypatch.setattr(contracts.os, "killpg", forbidden)
    monkeypatch.setattr(contracts.os, "waitpid", lambda pid, options: (pid, 9))
    assert child.try_reap() == -9
    assert child.proc.returncode == -9
    with pytest.raises(RuntimeError, match="revoked"):
        child.send_group(signal.SIGKILL)


def test_unseen_observation_does_not_reach_terminal_promotion():
    cap = object.__new__(groups.OwnedGroup)
    cap.leader = cap.pgid = 43210
    cap._revoked = False
    cap._terminal_result = None
    cap.receipts = {43210: SimpleNamespace()}
    cap.probe = {}
    cap.terminating_sends = [signal.SIGKILL]
    cap._leader_witness = lambda deadline: ("unseen", None)
    cap.census = contracts.ObservationOnlyBackend().census
    cap._terminal_absence = forbidden
    # Mutation target P1: an observation that claims complete=True and omits
    # the leader must not reach the otherwise-accepted terminal path.
    assert cap.refresh(deadline=1, terminal=True).status == "REFUSED"


@pytest.mark.parametrize(
    "cause,status", [("deadline", 124), ("interruption", 130), ("memory-policy", 137)]
)
def test_established_nonzero_survives_supervision_and_publication_failure(tmp_path, cause, status):
    class BrokenReceipt(contracts.Receipt):
        def publish(self, contract, outcome):
            raise OSError("synthetic publication failure")

    def strict(outcome):
        outcome.wrapper_exit_code = status
        outcome.termination_cause = cause
        raise RuntimeError("synthetic cleanup failure")

    rc = contracts.execute(
        ["synthetic"],
        contract="cleanup-required",
        timeout=1,
        grace=0,
        budget=SimpleNamespace(ceiling_mb=0),
        child_env={},
        receipt=BrokenReceipt(tmp_path, "same-id"),
        stream=io.StringIO(),
        strict=strict,
        backend=contracts.Backend("synthetic-authoritative", True, "test"),
    )
    assert rc == status


def test_interruption_is_not_confused_with_child_returning_130(tmp_path):
    log = []
    inner = direct_with(leader_factory(log, status=0))

    def direct(argv, **kwargs):
        kwargs["stop"]["signum"] = signal.SIGINT
        inner(argv, **kwargs)

    rc, receipt, text = invoke(tmp_path, direct=direct)
    row = receipt_row(receipt)
    assert rc == 130 and row["command_exit_code"] == 0
    assert row["termination_cause"] == "interruption" and row["cleanup_status"] == "unknown"
    assert log.index(("killpg", 43210, signal.SIGINT)) < log.index("reap")


def test_summary_names_default_contract(tmp_path):
    rc, receipt, text = invoke(tmp_path, contract="cleanup-required")
    assert rc == 125
    assert "contract=cleanup-required" in text and "cleanup=unknown" in text


def strict_success(outcome):
    outcome.command_exit_code = outcome.wrapper_exit_code = 0
    outcome.termination_cause = "child-exit"
    outcome.cleanup_status = "proven-empty"


def test_strict_success_never_claims_descendant_cleanup(tmp_path):
    rc, receipt, _ = invoke(
        tmp_path,
        contract="cleanup-required",
        strict=strict_success,
        backend=contracts.Backend("synthetic-authoritative", True, "test"),
    )
    row = receipt_row(receipt)
    assert rc == 0 and row["cleanup_status"] == "proven-empty"
    assert row["descendant_cleanup"] == "unknown"
    assert contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )
    # Even a successfully linked pending inode is NOT the published pathname.
    assert not contracts.admit_cleanup(
        receipt.path.with_suffix(".pending"),
        expected_invocation_id=receipt.invocation_id,
        expected_directory=tmp_path,
    )
    assert not contracts.admit_cleanup(
        receipt.path,
        expected_invocation_id=receipt.invocation_id,
        expected_directory=tmp_path / "other",
    )


def test_final_collision_is_independent_of_pending_collision(tmp_path):
    receipt = contracts.Receipt(tmp_path, "same-id")
    original = b"existing final must not be overwritten\n"
    receipt.path.write_bytes(original)
    pending = receipt.path.with_suffix(".pending")
    assert not pending.exists()
    outcome = contracts.Outcome()
    strict_success(outcome)
    with pytest.raises(FileExistsError):
        receipt.publish("cleanup-required", outcome)
    # Kills replacing os.link with os.replace: pending creation succeeds here,
    # unlike the old second-publish test, which failed before reaching the link.
    assert receipt.path.read_bytes() == original
    assert json.loads(pending.read_text())["cleanup_status"] == "proven-empty"


def test_errant_direct_supervisor_cannot_promote_group_cleanup(tmp_path):
    def direct(argv, *, outcome, **kwargs):
        strict_success(outcome)

    rc, receipt, _ = invoke(tmp_path, direct=direct)
    row = receipt_row(receipt)
    assert rc == 0 and row["command_exit_code"] == 0
    assert row["cleanup_status"] == "unknown"
    assert row["descendant_cleanup"] == "unknown"


@pytest.mark.parametrize("point", ["observe", "reap", "revoke"])
def test_direct_late_interruption_is_not_child_success(point):
    stop = {"signum": 0}
    outcome = contracts.Outcome()

    def hook(where):
        if where == point:
            stop["signum"] = signal.SIGINT

    contracts.supervise_direct(
        ["synthetic"],
        timeout=None,
        grace=0,
        env={},
        outcome=outcome,
        stop=stop,
        stream=io.StringIO(),
        spawn=lambda *a, **k: Child(),
        boundary=leader_factory([], status=0, hook=hook),
        group_pids=lambda pgid: [pgid],
    )
    assert outcome.command_exit_code == 0
    assert outcome.wrapper_exit_code == 130
    assert outcome.termination_cause == "interruption"


@pytest.mark.parametrize("point", ["return", "restore"])
def test_execute_reconciles_direct_return_and_handler_restore(monkeypatch, tmp_path, point):
    handlers = {}
    previous = object()

    def fake_signal(sig, handler):
        if handler is previous:
            if point == "restore" and sig == signal.SIGINT:
                handlers[sig](signal.SIGINT, None)
        else:
            handlers[sig] = handler
        return previous

    monkeypatch.setattr(contracts.signal, "signal", fake_signal)

    def direct(argv, *, outcome, stop, **kwargs):
        strict_success(outcome)
        if point == "return":
            stop["signum"] = signal.SIGINT

    rc, receipt, _ = invoke(tmp_path, direct=direct)
    row = receipt_row(receipt)
    assert rc == 130 and row["command_exit_code"] == 0
    assert row["termination_cause"] == "interruption"


@pytest.fixture
def strict_boundary(runner, monkeypatch, tmp_path):
    """Integrated strict runner with NO real spawn, census, wait or handlers.

    The wait's assertion discriminates revocation-before-consumption, including
    TimeoutExpired; merely checking eventual owner state misses that ordering.
    """
    state = {
        "point": None,
        "wait_timeout": False,
        "wait_calls": 0,
        "wait_revoked": [],
        "child_status": 0,
        "raise_at_point": False,
        "restore_attempts": [],
        "refuse_nonterminal": False,
    }
    handlers = {}
    previous = object()

    def interrupt():
        handlers[signal.SIGINT](signal.SIGINT, None)

    def fake_signal(sig, handler):
        if handler is previous:
            state["restore_attempts"].append(sig)
            if state["point"] == "restore" and sig == signal.SIGINT:
                interrupt()
                if state["raise_at_point"]:
                    raise RuntimeError("synthetic restoration failure")
        else:
            handlers[sig] = handler
        return previous

    class Owner:
        last = groups.Verdict("VERIFIED", "synthetic held reservation")
        terminating_sends = []
        revoked = False
        terminal = None

        def leader_terminal(self, deadline):
            return True

        def refresh(self, *, deadline, terminal=False):
            if not terminal and state["refuse_nonterminal"]:
                # Linux fork churn: an unstable census refuses a non-terminal check.
                self.last = groups.Verdict("REFUSED", "churn", refusal_class="unproven")
                return self.last
            if terminal and state["point"] == "census":
                interrupt()
                if state["raise_at_point"]:
                    raise RuntimeError("synthetic final census failure")
            self.last = groups.Verdict("OBSERVED_EMPTY", "synthetic group only")
            if terminal:
                self.terminal = self.last
            return self.last

        def accepted_terminal(self, verdict):
            return not self.revoked and verdict is self.terminal

        def revoke(self):
            self.revoked = True

    owner = Owner()
    # Ensure terminal=True is used in final census, without any real send.
    owner.terminating_sends = [signal.SIGKILL]

    class Child:
        pid = 43210
        returncode = None

        def poll(self):
            pytest.fail("a second consuming poll bypassed the wait boundary")

        def wait(self, *, timeout):
            state["wait_calls"] += 1
            # Observe outside execute's exception handler too: otherwise an
            # AssertionError here can masquerade as the expected timeout rc126.
            state["wait_revoked"].append(owner.revoked)
            assert owner.revoked, "strict wait consumed before authority revocation"
            if state["point"] == "wait":
                interrupt()
                if state["raise_at_point"]:
                    raise RuntimeError("synthetic consuming wait failure")
            if state["wait_timeout"]:
                raise runner.subprocess.TimeoutExpired("synthetic", timeout)
            self.returncode = state["child_status"]
            return self.returncode

    monkeypatch.setattr(runner.signal, "signal", fake_signal)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(groups.OwnedGroup, "capture", lambda *a, **k: owner)
    monkeypatch.setattr(runner, "_signal_group", lambda *a, **k: owner.last)
    monkeypatch.setattr(runner, "_record_run", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_record_dir", lambda: tmp_path)
    monkeypatch.setattr(runner, "_child_environment", lambda budget: {})
    monkeypatch.setattr(
        runner,
        "_memory_budget",
        lambda mb: (SimpleNamespace(ceiling_mb=0, reason="synthetic zero grant"), "test"),
    )
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(runner.time, "sleep", forbidden)

    def run():
        rc = runner.run(
            ["synthetic"],
            timeout=None,
            grace=0,
            stream=io.StringIO(),
            _backend=contracts.Backend("synthetic-authoritative", True, "test"),
        )
        path = next(tmp_path.glob("*.completion.json"))
        row = json.loads(path.read_text())
        admitted = contracts.admit_cleanup(
            path, expected_invocation_id=row["invocation_id"], expected_directory=tmp_path
        )
        return rc, row, admitted

    return state, run


@pytest.mark.parametrize("point", [None, "census", "wait", "restore"])
def test_strict_finalization_reconciles_late_interruptions(strict_boundary, point):
    state, run = strict_boundary
    state["point"] = point
    rc, row, admitted = run()
    assert state["wait_calls"] == 1
    assert row["command_exit_code"] == 0 and row["cleanup_status"] == "proven-empty"
    assert row["descendant_cleanup"] == "unknown"
    assert rc == (0 if point is None else 130)
    assert row["termination_cause"] == ("child-exit" if point is None else "interruption")
    assert admitted is (point is None)


@pytest.mark.parametrize("wait_timeout", [False, True])
def test_strict_wait_revokes_before_success_or_timeout(strict_boundary, wait_timeout):
    state, run = strict_boundary
    state["wait_timeout"] = wait_timeout
    rc, row, admitted = run()
    assert state["wait_calls"] == 1
    assert state["wait_revoked"] == [True]
    assert rc == (126 if wait_timeout else 0)
    assert row["command_exit_code"] == (None if wait_timeout else 0)
    assert admitted is (not wait_timeout)


@pytest.mark.parametrize("child_status,expected", [(-signal.SIGTERM, 143), (7, 7)])
def test_strict_child_status_has_independent_shell_oracle(strict_boundary, child_status, expected):
    state, run = strict_boundary
    state["child_status"] = child_status
    rc, row, admitted = run()
    assert state["wait_revoked"] == [True]
    assert rc == expected and row["command_exit_code"] == expected
    assert row["termination_cause"] == ("child-signal" if child_status < 0 else "child-exit")
    assert not admitted


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other-schema"),
        ("invocation_id", "0" * 32),
        ("requested_contract", "command-status"),
        ("record_phase", "pending"),
        ("cleanup_scope", "descendants"),
        ("cleanup_status", "unknown"),
        ("wrapper_exit_code", 130),
        ("command_exit_code", 7),
        ("termination_cause", "interruption"),
    ],
)
def test_admission_checks_each_success_predicate_independently(tmp_path, field, value):
    receipt = contracts.Receipt(tmp_path, "same-id")
    # Make the invocation mismatch deterministic, not probability-dependent.
    receipt.invocation_id = "1" * 32
    receipt.path = tmp_path / f"{receipt.invocation_id}.completion.json"
    outcome = contracts.Outcome()
    strict_success(outcome)
    row = receipt.publish("cleanup-required", outcome)
    assert contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )
    row[field] = value
    receipt.path.write_text(json.dumps(row))
    assert not contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )


@pytest.mark.parametrize("mismatch", ["terminal", "last", "argument"])
def test_terminal_authority_requires_both_identities_and_status(mismatch):
    cap = object.__new__(groups.OwnedGroup)
    cap._revoked = False
    accepted = groups.Verdict("OBSERVED_EMPTY", "synthetic terminal")
    equal_but_not_same = groups.Verdict("OBSERVED_EMPTY", "synthetic terminal")
    cap._terminal_result = cap.last = accepted
    supplied = accepted
    if mismatch == "terminal":
        cap._terminal_result = equal_but_not_same
    elif mismatch == "last":
        cap.last = equal_but_not_same
    else:
        # Both identities match but the status is not terminal absence.
        supplied = cap._terminal_result = cap.last = groups.Verdict("VERIFIED", "live")
    outcome = contracts.Outcome()
    outcome.accept_terminal(cap, supplied)
    assert outcome.cleanup_status == "unknown"


def test_canonical_name_cannot_be_a_symlink_to_unpublished_pending(tmp_path):
    receipt = contracts.Receipt(tmp_path, "same-id")
    pending = receipt.path.with_suffix(".pending")
    outcome = contracts.Outcome()
    strict_success(outcome)
    # Construct strict-success content without claiming publication succeeded.
    row = {
        "schema": contracts.SCHEMA,
        "invocation_id": receipt.invocation_id,
        "requested_contract": "cleanup-required",
        "record_phase": "complete",
        "cleanup_scope": "owned-process-group",
        "cleanup_status": "proven-empty",
        "wrapper_exit_code": 0,
        "command_exit_code": 0,
        "termination_cause": "child-exit",
    }
    pending.write_text(json.dumps(row))
    receipt.path.symlink_to(pending)
    assert not contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )


@pytest.fixture(autouse=True)
def synthetic_handler_registry(monkeypatch):
    # These synthetic invoke() cases must never install OS handlers; per-test
    # boundary fakes may override. Real CLI cells execute in a separate process.
    handlers = {}

    def register(sig, handler):
        previous = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(contracts.signal, "signal", register)


@pytest.mark.parametrize("point", ["census", "wait", "restore"])
def test_strict_finalization_signal_plus_exception_preserves_interruption(strict_boundary, point):
    state, run = strict_boundary
    state["point"] = point
    state["raise_at_point"] = True
    rc, row, admitted = run()
    assert rc == 130 and row["wrapper_exit_code"] == 130
    assert row["termination_cause"] == "interruption"
    assert row["finalization_status"] == "failed"
    stage = "handler-restoration" if point == "restore" else "strict-finalization"
    assert row["finalization_errors"] == [f"{stage}:RuntimeError"]
    assert row["command_exit_code"] == (0 if point == "restore" else None)
    assert state["restore_attempts"] == list(contracts.FORWARDED)
    assert not admitted


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize(
    "cause,status", [("child-exit", 0), ("deadline", 124), ("memory-policy", 137)]
)
def test_execute_restoration_exception_publishes_failure_and_attempts_remaining_handlers(
    monkeypatch, tmp_path, interrupt, cause, status
):
    handlers = {}
    previous = object()
    attempts = []

    def fake_signal(sig, handler):
        if handler is previous:
            attempts.append(sig)
            if sig == signal.SIGINT:
                if interrupt:
                    handlers[sig](signal.SIGINT, None)
                raise RuntimeError("synthetic restoration failure")
        else:
            handlers[sig] = handler
        return previous

    monkeypatch.setattr(contracts.signal, "signal", fake_signal)

    def direct(argv, *, outcome, **kwargs):
        outcome.command_exit_code = 0
        outcome.wrapper_exit_code = status
        outcome.termination_cause = cause

    rc, receipt, text = invoke(tmp_path, direct=direct)
    row = receipt_row(receipt)
    expected = 130 if interrupt else (126 if cause == "child-exit" else status)
    expected_cause = (
        "interruption"
        if interrupt
        else ("finalization-failure" if cause == "child-exit" else cause)
    )
    assert rc == expected and row["wrapper_exit_code"] == expected
    assert row["termination_cause"] == expected_cause
    assert row["command_exit_code"] == 0
    assert row["finalization_status"] == "failed"
    assert row["finalization_errors"] == ["handler-restoration:RuntimeError"]
    assert attempts == list(contracts.FORWARDED)
    assert f"rc={expected} cause={expected_cause}" in text
    assert not contracts.admit_cleanup(
        receipt.path, expected_invocation_id=receipt.invocation_id, expected_directory=tmp_path
    )


@pytest.mark.parametrize(
    "malformed_id",
    ["a" * 31, "a" * 33, "g" + "a" * 31],
    ids=["short-hex", "long-hex", "non-hex-correct-length"],
)
def test_admission_rejects_malformed_expected_id_despite_matching_path_and_json(
    tmp_path, malformed_id
):
    # All other predicates agree. Deleting JUST the length or character guard
    # admits its corresponding malformed case; a mismatched path cannot mask it.
    receipt = contracts.Receipt(tmp_path, "same-id")
    receipt.invocation_id = malformed_id
    receipt.path = tmp_path / f"{malformed_id}.completion.json"
    outcome = contracts.Outcome()
    strict_success(outcome)
    row = receipt.publish("cleanup-required", outcome)
    assert row["invocation_id"] == malformed_id
    assert receipt.path == tmp_path / f"{malformed_id}.completion.json"
    assert not contracts.admit_cleanup(
        receipt.path, expected_invocation_id=malformed_id, expected_directory=tmp_path
    )


@pytest.mark.parametrize("ceiling", [0, 128, 4096])
def test_real_child_environment_carries_resolved_budget_and_default_gate(
    runner, monkeypatch, ceiling
):
    monkeypatch.setenv(runner.memory_guard.RESOLVED_MEMORY_MB_ENV, "99999")
    monkeypatch.delenv(runner._NO_NOTIFICATIONS_ENV, raising=False)
    monkeypatch.setenv("BOUNDED_UNRELATED", "preserved")
    before = dict(runner.os.environ)
    env = runner._child_environment(SimpleNamespace(ceiling_mb=ceiling))
    assert env[runner.memory_guard.RESOLVED_MEMORY_MB_ENV] == str(ceiling)
    assert env[runner._NO_NOTIFICATIONS_ENV] == "1"
    assert env["BOUNDED_UNRELATED"] == "preserved"
    assert dict(runner.os.environ) == before, "building a child's policy must not mutate the parent"


@pytest.mark.parametrize("gate", ["0", "1", ""])
def test_real_child_environment_preserves_explicit_notification_policy(runner, monkeypatch, gate):
    monkeypatch.setenv(runner._NO_NOTIFICATIONS_ENV, gate)
    assert (
        runner._child_environment(SimpleNamespace(ceiling_mb=64))[runner._NO_NOTIFICATIONS_ENV]
        == gate
    )


@pytest.mark.parametrize(
    "options, append",
    [
        (None, True),
        ("", True),
        ("--trace-warnings", True),
        ("--trace-warnings --max-old-space-size=16", True),
        ("--max_old_space_size=+8192 --trace-warnings", False),
        ('"--max-old-space-size=8192" --trace-warnings', False),
        ("--max-old-space-size=8192 --max-old-space-size=32", True),
        ("--max-old-space-size=32 --max-old-space-size=8192", False),
        ('--title="contains --max-old-space-size=8192"', True),
        ('--title="unclosed', True),
    ],
)
def test_real_child_environment_heap_floor_preserves_unrelated_flags(
    runner, monkeypatch, options, append
):
    if options is None:
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
    else:
        monkeypatch.setenv("NODE_OPTIONS", options)
    env = runner._child_environment(SimpleNamespace(ceiling_mb=128))
    expected = options or ""
    if append:
        expected = (
            expected + " " if expected else ""
        ) + f"--max-old-space-size={runner.NODE_HEAP_MB}"
    assert runner.NODE_HEAP_MB == 6144
    assert env["NODE_OPTIONS"] == expected
    # The V8 heap floor is deliberately not the group admission ceiling.
    assert env[runner.memory_guard.RESOLVED_MEMORY_MB_ENV] == "128"


def test_real_child_environment_heap_floor_boundary(runner, monkeypatch):
    for heap in (runner.NODE_HEAP_MB - 1, runner.NODE_HEAP_MB, runner.NODE_HEAP_MB + 1):
        options = f"--trace-warnings --max-old-space-size={heap}"
        monkeypatch.setenv("NODE_OPTIONS", options)
        result = runner._child_environment(SimpleNamespace(ceiling_mb=0))["NODE_OPTIONS"]
        assert result == (
            options + f" --max-old-space-size={runner.NODE_HEAP_MB}"
            if heap < runner.NODE_HEAP_MB
            else options
        )


def test_environment_extraction_is_one_definition_and_two_real_call_sites(runner):
    import ast

    # Direct source delivery has no ordered patch applier. Enforce its replacement
    # invariant on the integrated AST: one policy body, both execution paths wired
    # to it, and no second inline copy that a later extraction could edit instead.
    tree = ast.parse(Path(runner.__file__).read_text())
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_child_environment"
    ]
    assert len(definitions) == 1
    callers = set()
    policy_owners = set()
    for function in [node for node in tree.body if isinstance(node, ast.FunctionDef)]:
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_child_environment"
            ):
                callers.add(function.name)
            if isinstance(node, ast.Constant) and node.value == "NODE_OPTIONS":
                policy_owners.add(function.name)
    assert callers == {"run", "_run_cleanup_required"}
    assert policy_owners == {"_child_environment"}


def test_integrated_sources_have_no_undefined_names_and_checker_rejects_missing_builder(runner):
    from pyflakes.api import check
    from pyflakes.reporter import Reporter

    # flake8 (the existing CI + check-changed gate) includes F821. Check the
    # integrated files too, and prove this is load-bearing with the exact POC
    # defect: a call whose definition was described but never emitted.
    paths = [Path(runner.__file__), Path(contracts.__file__), Path(groups.__file__), Path(__file__)]
    for path in paths:
        errors = io.StringIO()
        assert check(path.read_text(), str(path), Reporter(errors, errors)) == 0, errors.getvalue()
    source = Path(runner.__file__).read_text()
    assert source.count("def _child_environment(") == 1
    mutant = source.replace("def _child_environment(", "def _missing_environment_builder(")
    errors = io.StringIO()
    assert check(mutant, str(runner.__file__), Reporter(errors, errors)) >= 2
    assert errors.getvalue().count("undefined name '_child_environment'") == 2


def test_broken_summary_preserves_original_supervision_error(tmp_path):
    output = io.StringIO()
    output.close()
    receipt = contracts.Receipt(tmp_path, "broken-summary")

    def strict(outcome):
        raise RuntimeError("original monitor error")

    with pytest.raises(RuntimeError, match="original monitor error"):
        contracts.execute(
            ["synthetic"],
            contract="cleanup-required",
            timeout=None,
            grace=0,
            budget=SimpleNamespace(ceiling_mb=0),
            child_env={},
            receipt=receipt,
            stream=output,
            strict=strict,
            backend=contracts.Backend("synthetic-authoritative", True, "test"),
        )
    row = receipt_row(receipt)
    assert row["supervision_error"] == "RuntimeError"
    assert row["wrapper_exit_code"] == 126
    assert row["cleanup_status"] == "unknown"


class FakeKernel:
    """The OS calls ReservedLeader makes, recorded in one ordered event log.

    The leader turns into a zombie at ``dies_at`` ticks of ``waitid`` or when it
    is SIGKILLed (unless ``unkillable``: a leader stuck in uninterruptible
    sleep); ``stragglers`` stay in the group listing until signalled. The log is
    the oracle for the architect's ordering invariant: every killpg names
    pgid == proc.pid and precedes the single consuming waitpid.

    The kernel also owns the fake clock. Its ``sleep`` is CAPPED (R2.2): a
    supervisor regression that never leaves its loop used to grow ``log``
    until a memory guard killed the run (1.6 GB bounded, 15.3 GB unbounded);
    now it trips ``runaway`` and the caller fails by assertion instead.
    """

    #: Far above any legitimate run here (the longest, a 10 s grace at 50 ms
    #: ticks, needs ~200), far below where the log costs real memory.
    MAX_SLEEPS = 10_000

    def __init__(
        self,
        monkeypatch,
        *,
        dies_at=2,
        stragglers=(),
        observe_error=None,
        unkillable=False,
        eperm_listing=False,
    ):
        self.log = []
        self.ticks = 0
        self.dies_at = dies_at
        self.zombie = False
        self.reaped = False
        self.members = list(stragglers)
        self.observe_error = observe_error
        self.unkillable = unkillable
        self.now = 0.0
        self.sleeps = 0
        self.runaway = False
        self.leader: contracts.ReservedLeader | None = None
        self.signalled_at = []
        # ``eperm_listing`` (anything but False) makes every killpg answer
        # EPERM, as Darwin does for a group holding only our zombie leader,
        # and is what the kernel's own group listing then reports.
        self.eperm_listing = eperm_listing
        monkeypatch.setattr(contracts.os, "killpg", self.killpg)
        monkeypatch.setattr(contracts.os, "waitpid", self.waitpid)
        monkeypatch.setattr(contracts.os, "waitid", self.waitid, raising=False)
        for name, value in (("P_PID", 1), ("WEXITED", 4), ("WNOHANG", 1), ("WNOWAIT", 32)):
            monkeypatch.setattr(contracts.os, name, value, raising=False)
        if eperm_listing is not False:
            from local_operator.mobile import resources

            monkeypatch.setattr(resources, "direct_group_pids", lambda pgid: eperm_listing)

    def clock(self):
        return self.now

    def sleep(self, dt):
        self.sleeps += 1
        if self.sleeps > self.MAX_SLEEPS:
            self.runaway = True
            raise AssertionError(f"supervisor loop exceeded {self.MAX_SLEEPS} ticks")
        self.now += dt

    def killpg(self, pgid, signum):
        assert not self.reaped, "signal after the consuming reap"
        self.log.append(("killpg", pgid, signum))
        self.signalled_at.append((signum, self.now))
        if self.eperm_listing is not False:
            raise PermissionError(1, "Operation not permitted")
        self.members = []
        if signum == signal.SIGKILL and not self.unkillable:
            self.zombie = True

    def waitid(self, idtype, pid, options):
        assert options & 32, "observation must be non-consuming (WNOWAIT)"
        self.log.append(("waitid", pid))
        if self.observe_error is not None:
            raise self.observe_error
        self.ticks += 1
        if self.ticks >= self.dies_at:
            self.zombie = True
        return SimpleNamespace(si_pid=pid) if self.zombie else None

    def waitpid(self, pid, options):
        assert options in (0, 1), options
        if not self.zombie:
            # A blocking wait on a live leader never returns: that is the bug.
            assert options == 1, "blocking waitpid on a leader not known dead"
            self.log.append(("waitpid-nohang-empty", pid))
            return 0, 0
        self.log.append(("waitpid", pid))
        self.reaped = True
        return pid, 0

    def group(self, pgid):
        return [pgid, *self.members]

    def boundary(self, proc):
        self.leader = contracts.ReservedLeader(proc, observer="waitid-wnowait")
        return self.leader


def run_supervised(kernel, *, timeout=None, grace=0.2, clock=None, **extra):
    outcome = contracts.Outcome()
    contracts.supervise_direct(
        ["synthetic"],
        timeout=timeout,
        grace=grace,
        env={},
        outcome=outcome,
        stop={"signum": 0},
        stream=io.StringIO(),
        spawn=lambda *a, **k: Child(),
        clock=clock or kernel.clock,
        sleep=kernel.sleep,
        boundary=kernel.boundary,
        group_pids=kernel.group,
        **extra,
    )
    # supervise_direct swallows Exception into a 126 outcome, so the cap's
    # AssertionError is re-asserted here, where it fails the test.
    assert not kernel.runaway, "supervisor never left its loop (R2.2 cap tripped)"
    return outcome


def test_every_group_signal_targets_the_leader_pgid_and_precedes_the_reap(monkeypatch):
    # Leader exits with a straggler still in its group: the sweep must signal
    # BEFORE the consuming wait releases the group number (mutant: reap above
    # sweep fails here because killpg would follow waitpid).
    kernel = FakeKernel(monkeypatch, dies_at=2, stragglers=[43299])
    outcome = run_supervised(kernel)
    kills = [i for i, e in enumerate(kernel.log) if e[0] == "killpg"]
    reaps = [i for i, e in enumerate(kernel.log) if e[0] == "waitpid"]
    assert kills, kernel.log
    assert len(reaps) == 1 and all(i < reaps[0] for i in kills), kernel.log
    assert {e[1] for e in kernel.log if e[0] in ("killpg", "waitpid", "waitid")} == {43210}
    assert outcome.wrapper_exit_code == 0 and outcome.termination_cause == "child-exit"
    assert outcome.cleanup_status == "unknown"


def test_deadline_escalation_also_signals_only_before_the_reap(monkeypatch):
    kernel = FakeKernel(monkeypatch, dies_at=10**9)  # ignores SIGTERM
    outcome = run_supervised(kernel, timeout=0.1, grace=0.2)
    sigs = [e[2] for e in kernel.log if e[0] == "killpg"]
    assert sigs == [signal.SIGTERM, signal.SIGKILL]
    assert kernel.log[-1] == ("waitpid", 43210)
    assert outcome.wrapper_exit_code == 124 and outcome.termination_cause == "deadline"


def test_unreadable_observer_kills_and_never_reports_success(monkeypatch):
    kernel = FakeKernel(monkeypatch, observe_error=OSError("synthetic EINVAL"))
    outcome = run_supervised(kernel)
    assert ("killpg", 43210, signal.SIGKILL) in kernel.log
    assert kernel.log[-1] == ("waitpid", 43210)
    assert outcome.wrapper_exit_code == 126
    assert outcome.termination_cause == "supervision-failure"
    assert outcome.cleanup_status == "unknown"


def test_foreign_reap_revokes_instead_of_fabricating_status(monkeypatch):
    # ECHILD means someone else consumed the leader: the number may be reused,
    # so no further signal is allowed and the outcome is a supervision failure.
    kernel = FakeKernel(monkeypatch, observe_error=ChildProcessError("synthetic ECHILD"))
    outcome = run_supervised(kernel)
    assert not [e for e in kernel.log if e[0] in ("killpg", "waitpid")]
    assert outcome.wrapper_exit_code == 126
    assert outcome.command_exit_code is None


def test_darwin_sysctl_observer_reads_zombie_without_consuming(monkeypatch):
    from local_operator import procstate

    samples = iter(
        [
            {43210: SimpleNamespace(zombie=False, birth="b1")},
            None,  # unreadable tick is unknown, not terminal
            {43210: SimpleNamespace(zombie=True, birth="b1")},
            {43210: SimpleNamespace(zombie=True, birth="stranger")},
        ]
    )
    monkeypatch.setattr(procstate, "_sysctl_samples", lambda pids: next(samples))
    leader = contracts.ReservedLeader(Child(), observer="darwin-sysctl-zombie")
    assert leader.observe() is False
    assert leader.observe() is None
    assert leader.observe() is True
    with pytest.raises(ChildProcessError):
        leader.observe()  # birth changed: reaped elsewhere and reused
    with pytest.raises(RuntimeError, match="revoked"):
        leader.send_group(signal.SIGTERM)


def test_darwin_sysctl_absent_pid_revokes_instead_of_reading_unknown(monkeypatch):
    """R2.3: a readable sysctl result WITHOUT our pid means someone else reaped
    the leader. That is revocation, not "unreadable": the unreadable branch
    would SIGKILL a group number that may already belong to a stranger."""
    from local_operator import procstate

    samples = iter([{43210: SimpleNamespace(zombie=False, birth="b1")}, {}])
    monkeypatch.setattr(procstate, "_sysctl_samples", lambda pids: next(samples))
    monkeypatch.setattr(contracts.os, "killpg", forbidden)
    leader = contracts.ReservedLeader(Child(), observer="darwin-sysctl-zombie")
    assert leader.observe() is False
    with pytest.raises(ChildProcessError):
        leader.observe()
    with pytest.raises(RuntimeError, match="revoked"):
        leader.send_group(signal.SIGTERM)


def test_non_default_sigchld_still_refuses_before_spawn(monkeypatch, tmp_path):
    monkeypatch.setattr(contracts.signal, "getsignal", lambda sig: (lambda *a: None))
    assert "SIGCHLD is not default" in (contracts.ReservedLeader.unavailable_reason() or "")
    receipt = contracts.Receipt(tmp_path, "sigchld")
    stream = io.StringIO()
    rc = contracts.execute(
        ["synthetic"],
        contract="command-status",
        timeout=None,
        grace=0,
        budget=SimpleNamespace(ceiling_mb=0),
        child_env={},
        receipt=receipt,
        stream=stream,
        strict=forbidden,
        direct=forbidden,
    )
    text = stream.getvalue()
    assert rc == 125 and receipt_row(receipt)["termination_cause"] == "unsupported-boundary"
    assert "refused before spawn:" in text and "spawned=no" in text


def synthetic_guard(*, kill_on_sample=1):
    """A guard factory whose ``kill_on_sample``-th sample breaches the grant."""

    class Guard:
        tick_s = 0.05

        def __init__(self, pgid, budget):
            assert pgid == 43210 and budget.ceiling_mb == 64
            self.samples = 0

        def sample_sync(self):
            self.samples += 1
            return self.samples

        def should_kill(self, sample):
            return sample >= kill_on_sample

        def over_budget_message(self, sample, **kwargs):
            return "MEMORY LIMIT EXCEEDED: synthetic"

    return Guard


def test_memory_breach_kills_group_before_reap_with_policy_cause(monkeypatch):
    kernel = FakeKernel(monkeypatch, dies_at=10**9)
    outcome = run_supervised(
        kernel,
        grace=10,
        budget=SimpleNamespace(ceiling_mb=64),
        guard_factory=synthetic_guard(),
    )
    assert kernel.log[0] == ("killpg", 43210, signal.SIGKILL)
    assert kernel.log[-1] == ("waitpid", 43210)
    assert outcome.wrapper_exit_code == 137 and outcome.termination_cause == "memory-policy"
    assert outcome.cleanup_status == "unknown"


def test_grant_stays_enforced_through_a_deadline_grace(monkeypatch):
    """R2.4: a TERM-ignoring group may not balloon while we wait out --grace.

    timeout=0 latches the deadline on the first tick (kill_at = grace = 10 s);
    the guard breaches on its SECOND sample, one 50 ms tick later. SIGKILL must
    come from the guard then, not from the grace expiry at 10 s.
    """
    kernel = FakeKernel(monkeypatch, dies_at=10**9)  # ignores SIGTERM
    outcome = run_supervised(
        kernel,
        timeout=0,
        grace=10,
        budget=SimpleNamespace(ceiling_mb=64),
        guard_factory=synthetic_guard(kill_on_sample=2),
    )
    assert [sig for sig, _ in kernel.signalled_at] == [signal.SIGTERM, signal.SIGKILL]
    kill_time = kernel.signalled_at[1][1]
    assert kill_time < 10, kernel.signalled_at  # before kill_at = 0 + grace
    assert kernel.log[-1] == ("waitpid", 43210)
    # The deadline latched first, so it keeps the cause and status.
    assert outcome.wrapper_exit_code == 124 and outcome.termination_cause == "deadline"


@pytest.mark.parametrize(
    ("listing", "accepted"),
    [([43210], True), ([43210, 43299], False), (None, False)],
    ids=["leader-only", "leader-and-other", "unreadable"],
)
def test_darwin_eperm_is_accepted_only_when_the_listing_proves_leader_only(
    monkeypatch, listing, accepted
):
    """R2.1: Darwin answers killpg with EPERM when the only member is our
    zombie leader. That reading is accepted ONLY when the kernel listing says
    exactly that; an EPERM with another member (e.g. a setuid exec we may not
    signal) or with an unreadable listing is authority loss: revoke, 126."""
    kernel = FakeKernel(monkeypatch, dies_at=2, stragglers=[43299], eperm_listing=listing)
    outcome = run_supervised(kernel)
    assert kernel.log[0] == ("waitid", 43210)
    first_signal = next(e for e in kernel.log if e[0] == "killpg")
    assert first_signal == ("killpg", 43210, signal.SIGTERM)
    if accepted:
        # The sweep's EPERMs were "nobody signalable", so the corpse is reaped
        # and the child's own status stands.
        assert kernel.log[-1] == ("waitpid", 43210)
        assert outcome.wrapper_exit_code == 0 and outcome.termination_cause == "child-exit"
    else:
        leader = kernel.leader
        assert leader is not None and leader._revoked
        with pytest.raises(RuntimeError, match="revoked"):
            leader.send_group(signal.SIGTERM)
        # Exactly one signal attempt, and no reap: the number may hold a
        # member we cannot signal, so nothing more is sent or consumed.
        assert [e for e in kernel.log if e[0] == "killpg"] == [first_signal]
        assert not [e for e in kernel.log if e[0] == "waitpid"]
        assert outcome.wrapper_exit_code == 126
        assert outcome.termination_cause == "supervision-failure"
        assert outcome.command_exit_code is None
    assert outcome.cleanup_status == "unknown"


def test_unreadable_observer_never_blocks_on_a_leader_that_will_not_die(monkeypatch):
    """R2.5: after SIGKILL the unreadable branch polls with WNOHANG until
    KILL_SETTLE_SECONDS; a leader stuck in uninterruptible sleep yields 126
    with no status instead of an unbounded blocking waitpid."""
    kernel = FakeKernel(
        monkeypatch, dies_at=10**9, unkillable=True, observe_error=OSError("synthetic EINVAL")
    )
    outcome = run_supervised(kernel)
    assert ("killpg", 43210, signal.SIGKILL) in kernel.log
    polls = [e for e in kernel.log if e[0] == "waitpid-nohang-empty"]
    assert polls and not [e for e in kernel.log if e[0] == "waitpid"]
    # R3.2: giving up on the leader must also end signal authority over its group.
    assert kernel.leader is not None and kernel.leader._revoked
    killed_at = kernel.signalled_at[-1][1]
    assert contracts.KILL_SETTLE_SECONDS <= kernel.now - killed_at < 3
    assert outcome.wrapper_exit_code == 126
    assert outcome.termination_cause == "supervision-failure"
    assert outcome.command_exit_code is None
    assert "termination not established" in (outcome.supervision_error or "")


def test_windows_shaped_signal_module_still_imports(monkeypatch):
    # R1: the cross-platform probe imports every module; Windows has no SIGHUP.
    import importlib.util
    import types

    fake = types.ModuleType("signal")
    for name in ("SIGINT", "SIGTERM", "SIG_DFL", "signal", "getsignal"):
        setattr(fake, name, getattr(signal, name))
    monkeypatch.setitem(__import__("sys").modules, "signal", fake)
    spec = importlib.util.find_spec("local_operator.bounded_contract")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.FORWARDED == (signal.SIGINT, signal.SIGTERM)


def test_accepted_terminal_certificate_outranks_an_earlier_transient_refusal(strict_boundary):
    """CI run 41535590707 (Linux churn): the finally-loop refresh refused, then the
    terminal census proved the group empty under the retained witness. The receipt
    said proven-empty yet rc was 126. The accepted certificate IS the proof."""
    state, run = strict_boundary
    state["refuse_nonterminal"] = True
    rc, row, admitted = run()
    assert row["cleanup_status"] == "proven-empty"
    assert rc == 0 and admitted


def test_strict_on_uncertifying_backend_leads_with_the_backend_refusal(
    runner, monkeypatch, tmp_path
):
    """D2.2: both refusals are known before spawn. Leading with the memory one
    sent users through two fixes when the answer was --contract=command-status."""
    monkeypatch.setenv("LOCAL_OPERATOR_RESOLVED_MEMORY_MB", "128")
    monkeypatch.setattr(runner, "_record_dir", lambda: tmp_path)
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    stream = io.StringIO()
    rc = runner.run(
        ["never-start"],
        timeout=None,
        grace=0.1,
        memory_mb=0,
        stream=stream,
        _backend=contracts.ObservationOnlyBackend(),
    )
    lines = stream.getvalue().splitlines()
    assert rc == 125
    assert "--contract=command-status" in lines[0]
    assert lines[1].startswith("[run_bounded] refused before spawn, also: --memory-mb")
    assert "cause=unsupported-backend" in lines[-1] and "spawned=no" in lines[-1]
    row = json.loads(next(tmp_path.glob("*.completion.json")).read_text())
    assert row["termination_cause"] == "unsupported-backend"


@pytest.mark.parametrize(
    ("inherited", "expected"),
    [
        (
            "128",
            "raise --memory-mb (up to the inherited 128 MiB), or memory_mb on the outer bash call",
        ),
        (None, "raise --memory-mb, or pass --memory-mb 0 to disable the guard"),
    ],
)
def test_wrapper_memory_escape_names_its_own_flag_first(runner, monkeypatch, inherited, expected):
    # D2.1: the bash tool's `memory_mb` cannot move a --memory-mb ceiling.
    if inherited is None:
        monkeypatch.delenv("LOCAL_OPERATOR_RESOLVED_MEMORY_MB", raising=False)
    else:
        monkeypatch.setenv("LOCAL_OPERATOR_RESOLVED_MEMORY_MB", inherited)
    assert runner._memory_escape() == expected


def test_a_breach_names_escaped_descendants_the_guard_could_not_verify(monkeypatch):
    """Review M1 on PR #3372: every macOS run_bounded run goes through
    supervise_direct (cleanup-required refuses there), so the "never silent"
    disclosure for an escaped pid no parentage source could verify must be
    printed HERE, not only on the cleanup-required path."""
    base = synthetic_guard()

    class Guard(base):
        def stop_escaped(self, signum):
            self.unverified = [77001, 77002]
            self.unsignalled = []
            return []

    kernel = FakeKernel(monkeypatch, dies_at=10**9)
    stream = io.StringIO()
    outcome = contracts.Outcome()
    contracts.supervise_direct(
        ["synthetic"],
        timeout=None,
        grace=0.2,
        env={},
        outcome=outcome,
        stop={"signum": 0},
        stream=stream,
        spawn=lambda *a, **k: Child(),
        clock=kernel.clock,
        sleep=kernel.sleep,
        boundary=kernel.boundary,
        group_pids=kernel.group,
        budget=SimpleNamespace(ceiling_mb=64),
        guard_factory=Guard,
    )
    assert not kernel.runaway
    text = stream.getvalue()
    assert outcome.termination_cause == "memory-policy"
    assert (
        "[run_bounded] could not verify parentage of escaped descendant(s) "
        "(unreadable, or re-parented to pid 1), not signalled: 77001, 77002"
    ) in text, text
