"""Explicit runner contracts, not a process-tree supervisor.

No observation, successful send, wait, or residual list is cleanup evidence.
The cleanup-required runner supplies its accepted authoritative terminal result
to this boundary; direct-child status can never substitute for that proof.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCHEMA = "run-bounded-completion/v1"
CONTRACTS = ("cleanup-required", "command-status")
FORWARDED = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


@dataclass(frozen=True)
class Backend:
    name: str
    authoritative: bool
    reason: str


def selected_backend() -> Backend:
    # This instrument on this installed OS, not a theorem about macOS: a
    # stronger backend can be added later only with its own coverage argument.
    if sys.platform.startswith("linux"):
        return Backend("strict-procfs", True, "strict /proc accounting")
    return Backend(
        sys.platform, False, "this instrument cannot certify coverage on this installed OS"
    )


class ObservationOnlyBackend:
    """Sentinel census: even legacy complete=True cannot confer authority."""

    name = "observation-only-sentinel"
    authoritative = False
    reason = "sentinel cannot construct authoritative coverage"

    def census(self, pgid, deadline):
        from local_operator.owned_process_group import CensusResult

        return CensusResult((), True, False, self.reason)


@dataclass
class Outcome:
    # Child status is shell-normalized, separately from the wrapper disposition.
    command_exit_code: int | None = None
    wrapper_exit_code: int = 126
    termination_cause: str = "supervision-failure"
    cleanup_status: str = "unknown"
    cleanup_reason: str = "no accepted authoritative terminal result"
    supervision_error: str | None = None
    # Finalization failures are published, not allowed to escape and erase the
    # receipt. Interruption retains precedence, with this independent diagnostic.
    finalization_errors: tuple[str, ...] = ()

    def finalization_failed(self, stage: str, exc: Exception) -> None:
        # A failed restore must not discard the structured outcome or hide behind
        # child success. Keep established policy causes; interruption wins later.
        self.finalization_errors += (f"{stage}:{type(exc).__name__}",)
        if self.termination_cause not in {"deadline", "memory-policy", "interruption"}:
            self.wrapper_exit_code = 126
            self.termination_cause = "finalization-failure"

    def accept_terminal(self, owner, verdict) -> None:
        # A string status is not a certificate. The owner must have accepted
        # this exact terminal object before irreversible authority revocation.
        if owner.accepted_terminal(verdict):
            self.cleanup_status = "proven-empty"
            self.cleanup_reason = verdict.reason


def shell_status(status: int) -> int:
    return 128 - status if status < 0 else status


class Receipt:
    """One fresh invocation, even if a caller reuses its correlation run_id.

    Exclusive create prevents replacing an earlier certificate. A random
    invocation_id is a challenge returned to the caller, NOT inferred from a
    reused run_id. Admission requires the caller's expected fresh challenge.
    JSON is eligible for admission only after final publication; an interrupted
    write is missing/invalid evidence, never a provisional success.
    """

    def __init__(self, base: Path, run_id: str):
        self.invocation_id = uuid.uuid4().hex
        self.run_id = run_id
        self.path = base / f"{self.invocation_id}.completion.json"

    def publish(self, contract: str, outcome: Outcome) -> dict:
        record = {
            "schema": SCHEMA,
            "invocation_id": self.invocation_id,
            "run_id": self.run_id,
            "requested_contract": contract,
            "command_exit_code": outcome.command_exit_code,
            "wrapper_exit_code": outcome.wrapper_exit_code,
            "termination_cause": outcome.termination_cause,
            "finalization_status": "failed" if outcome.finalization_errors else "complete",
            "finalization_errors": list(outcome.finalization_errors),
            "cleanup_scope": "owned-process-group",
            "cleanup_status": outcome.cleanup_status,
            "cleanup_reason": outcome.cleanup_reason,
            "signal_scope": (
                "direct-child" if contract == "command-status" else "owned-process-group"
            ),
            "record_phase": "complete",
            "supervision_error": outcome.supervision_error,
            # These are policy/guarantee disclosures, not timing measurements.
            "deadline_guarantee": "none: cooperative observation and signal attempts only",
            "memory_guarantee": (
                "none: no hard RAM bound; descendants not bounded"
                if contract == "command-status"
                else "sampled owned-group policy, not a hard allocation bound"
            ),
            # An empty owned group says nothing about descendants that escaped
            # it. Neither contract has authority over that wider population.
            "descendant_cleanup": "unknown",
        }
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # x mode never truncates an earlier invocation. Failure, including fsync,
        # propagates: publication failure must not become wrapper success.
        pending = self.path.with_suffix(".pending")
        with pending.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A complete-looking JSON file must not survive a failed flush/fsync as
        # admissible evidence. Hard-link only AFTER writing and closing; unlike
        # replace/rename, link refuses an existing final certificate. The pending
        # inode is intentionally retained; this protocol does not claim directory
        # crash durability or bounded filesystem latency.
        os.link(pending, self.path)
        return record


def admit_cleanup(path: Path, *, expected_invocation_id: str, expected_directory: Path) -> bool:
    """Proof-dependent consumers must possess this run's fresh challenge.

    Requires a trusted local filesystem and no competing writer throughout
    publication and admission. This binds a lexical pathname and fresh challenge,
    NOT publication provenance or resolved filesystem identity. Only the final
    component is checked for symlinks; directory/ancestor symlinks are allowed.
    A competing writer can hard-link matching pending JSON to the expected name;
    the symlink check/read pair is not race-resistant. Authentication against
    hostile writers is out of scope. An old run_id alone confers no authority.
    """
    # The trusted caller supplies the publication directory, not just a nonce
    # learned from a candidate file. In particular the fsynced pending inode is
    # not a publication, even though its JSON is byte-identical to the final one.
    if (
        len(expected_invocation_id) != 32
        or any(c not in "0123456789abcdef" for c in expected_invocation_id)
        or path != expected_directory / f"{expected_invocation_id}.completion.json"
        or path.is_symlink()
    ):
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(row, dict) and all(
        (
            row.get("schema") == SCHEMA,
            bool(expected_invocation_id),
            row.get("invocation_id") == expected_invocation_id,
            row.get("requested_contract") == "cleanup-required",
            row.get("record_phase") == "complete",
            row.get("cleanup_scope") == "owned-process-group",
            row.get("cleanup_status") == "proven-empty",
            row.get("wrapper_exit_code") == 0,
            row.get("command_exit_code") == 0,
            row.get("termination_cause") == "child-exit",
        )
    )


class DirectChild:
    """Exclusive reaper in one thread; handlers only latch intent.

    WNOWAIT observes our unreaped child without releasing its PID. Every
    consuming wait revokes future sends BEFORE attempting it, even on error.
    No threads, Popen.poll/send_signal, unrelated waiter, SIGCHLD=SIG_IGN,
    numeric group sender, or census is permitted inside this boundary.
    """

    def __init__(self, proc):
        self.proc = proc
        self._revoked = False

    @staticmethod
    def unavailable_reason() -> str | None:
        # CPython exposes waitid on macOS only from 3.13; platform alone cannot
        # select this capability while the package still supports Python 3.12.
        missing = [
            f"os.{name}"
            for name in ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
            if not hasattr(os, name)
        ]
        if missing:
            return "exclusive-reaper observation unavailable: missing " + ", ".join(missing)
        # A custom SIGCHLD handler may be another reaper, not merely a notifier.
        # Default disposition and no other waiter are embedding preconditions,
        # not a kernel-enforced capability against concurrent consuming waits.
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            return "exclusive-reaper observation unavailable: SIGCHLD is not default"
        return None

    @staticmethod
    def available() -> bool:
        return DirectChild.unavailable_reason() is None

    def terminal(self) -> bool:
        # ECHILD is not absence proof; it terminates supervision without send.
        try:
            return (
                os.waitid(os.P_PID, self.proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
            )
        except BaseException:
            self._revoked = True
            raise

    def send(self, signum: int) -> None:
        # Mutation-boundary revocation, not a cached verdict or census check.
        # Our sole reaper cannot run between this guard and kill in this thread.
        if self._revoked:
            raise RuntimeError("direct-child authority irrevocably revoked")
        try:
            os.kill(self.proc.pid, signum)
        except BaseException:
            self._revoked = True
            raise

    def reap(self) -> int:
        self._revoked = True
        return self.proc.wait(timeout=0)

    def revoke(self) -> None:
        self._revoked = True


def supervise_direct(
    argv,
    *,
    timeout,
    grace,
    env,
    outcome,
    stop,
    spawn=subprocess.Popen,
    clock=time.monotonic,
    sleep=time.sleep,
    boundary=DirectChild,
):
    """No group authority is created, queried, or spent on this path.

    A deadline requests TERM then KILL of the direct child only. Uninterruptible
    kernel sleep, scheduler delay, filesystem stalls and surviving descendants
    prevent any honest hard wall-time/RAM promise. We return after a finite
    observation allowance even if the child cannot be reaped; cleanup is UNKNOWN.
    """
    started = clock()
    try:
        proc = spawn(list(argv), start_new_session=True, env=env)
    except OSError as exc:
        outcome.wrapper_exit_code = 125
        outcome.termination_cause = "spawn-refused"
        outcome.supervision_error = type(exc).__name__
        return
    child = boundary(proc)
    deadline = math.inf if timeout is None else started + timeout
    escalating = None
    kill_at = finish_at = math.inf
    killed = False
    forwarded = False
    try:
        while True:
            now = clock()
            if escalating is None:
                # Latch cause independently: a child can itself return 124,
                # 125 or 126. Those remain child-exit in the completion record.
                if stop["signum"]:
                    escalating = stop["signum"]
                    outcome.wrapper_exit_code = 128 + escalating
                    outcome.termination_cause = "interruption"
                elif now >= deadline:
                    escalating = signal.SIGTERM
                    outcome.wrapper_exit_code = 124
                    outcome.termination_cause = "deadline"
                if escalating is not None:
                    kill_at, finish_at = now + grace, now + grace + 0.25
            if child.terminal():
                raw_status = child.reap()
                outcome.command_exit_code = shell_status(raw_status)
                # Select the wrapper result only after final revocation below;
                # both consuming wait and finalization can latch interruption.
                return
            if escalating is not None:
                if not killed and now >= kill_at:
                    child.send(signal.SIGKILL)
                    killed = True
                elif not forwarded and now < kill_at:
                    child.send(escalating)
                    # One forwarding attempt, not one send per tick.
                    forwarded = True
                if now >= finish_at:
                    outcome.supervision_error = "direct-child termination not established"
                    return
            sleep(0.05)
    except Exception as exc:
        # Preserve established deadline/interruption outcomes, never convert a
        # failed wait/send into successful command status or group cleanliness.
        outcome.supervision_error = type(exc).__name__
        if outcome.termination_cause not in {"deadline", "interruption"}:
            outcome.wrapper_exit_code = 126
            outcome.termination_cause = "supervision-failure"
    finally:
        child.revoke()
        if stop["signum"]:
            outcome.wrapper_exit_code = 128 + stop["signum"]
            outcome.termination_cause = "interruption"
        elif outcome.command_exit_code is not None and escalating is None:
            outcome.wrapper_exit_code = outcome.command_exit_code
            outcome.termination_cause = "child-signal" if raw_status < 0 else "child-exit"


def execute(
    argv,
    *,
    contract,
    timeout,
    grace,
    budget,
    child_env,
    receipt,
    stream,
    strict: Callable,
    backend=None,
    direct=supervise_direct,
    boundary=DirectChild,
) -> int:
    """Called after canonical _memory_budget, before any handler or Popen.

    Supported strict gates retain their existing monitor and sampled memory
    policy. Command-status cannot enforce that GROUP grant; refusing a positive
    grant is intentional, not an invitation to disable an inherited ceiling.
    The canonical resolver has already rejected disabling/widening that grant.
    Finalization exceptions (Exception, not process-level BaseException) produce
    a failure receipt and integer status, subject to publication/stream I/O. All
    handler restores are attempted; failure is recorded even when interruption
    or an established policy retains its cause/status. A failed restoration may
    leave our handler installed: this diagnostic is not a restoration guarantee.
    """
    backend = selected_backend() if backend is None else backend
    outcome = Outcome()
    supervision_exception = None
    previous = {}
    stop = {"signum": 0}
    try:
        if contract not in CONTRACTS:
            raise ValueError("unknown execution contract")
        if (
            not argv
            or timeout is not None
            and (not math.isfinite(timeout) or timeout < 0)
            or not math.isfinite(grace)
            or grace < 0
        ):
            outcome.wrapper_exit_code, outcome.termination_cause = 125, "invalid-request"
        elif contract == "cleanup-required" and not backend.authoritative:
            # Known inability refuses BEFORE handler mutation and before spawn.
            outcome.wrapper_exit_code, outcome.termination_cause = 125, "unsupported-backend"
            outcome.cleanup_reason = backend.reason
        elif contract == "command-status" and budget.ceiling_mb > 0:
            outcome.wrapper_exit_code, outcome.termination_cause = 125, "unsupported-policy"
            outcome.cleanup_reason = (
                "direct-child contract cannot enforce a positive group memory grant"
            )
        elif contract == "command-status" and not boundary.available():
            outcome.wrapper_exit_code, outcome.termination_cause = 125, "unsupported-boundary"
            explain = getattr(boundary, "unavailable_reason", None)
            outcome.cleanup_reason = (
                explain() if explain else "exclusive-reaper observation unavailable"
            )
        elif contract == "cleanup-required":
            strict(outcome)
        else:

            def latch(signum, frame):
                stop["signum"] = signum

            for sig in FORWARDED:
                previous[sig] = signal.signal(sig, latch)
            outcome.cleanup_reason = (
                "command-status never certifies descendant or owned-group cleanup"
            )
            direct(argv, timeout=timeout, grace=grace, env=child_env, outcome=outcome, stop=stop)
    except Exception as exc:
        supervision_exception = exc
        outcome.supervision_error = type(exc).__name__
        if outcome.termination_cause not in {
            "deadline",
            "memory-policy",
            "interruption",
            "child-exit",
            "child-signal",
            "finalization-failure",
        }:
            outcome.wrapper_exit_code, outcome.termination_cause = 126, "supervision-failure"
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except Exception as exc:
                outcome.finalization_failed("handler-restoration", exc)
    # Also close the direct supervisor's return/finally window. Successful
    # restoration returns later signals to the caller's disposition. Failed
    # restoration is recorded; it cannot promise to close later latch delivery.
    if stop["signum"]:
        outcome.wrapper_exit_code = 128 + stop["signum"]
        outcome.termination_cause = "interruption"
    if contract == "command-status":
        # Even an injected/errant direct supervisor cannot promote its waits,
        # sends or observations into a group-cleanup certificate.
        outcome.cleanup_status = "unknown"
    elif outcome.wrapper_exit_code == 0 and outcome.cleanup_status != "proven-empty":
        outcome.wrapper_exit_code = 126
    try:
        try:
            receipt.publish(contract, outcome)
        except (OSError, ValueError, TypeError) as exc:
            # No record can attest to its own failure to publish. Stderr carries the
            # failure; admission fails closed on missing/invalid/mismatched receipt.
            # A nonzero established outcome remains nonzero with its original cause.
            stream.write(f"[run_bounded] completion publication failed: {type(exc).__name__}\n")
            if outcome.wrapper_exit_code == 0:
                outcome.wrapper_exit_code = 126
        stream.write(
            f"[run_bounded] contract={contract} rc={outcome.wrapper_exit_code} "
            f"cause={outcome.termination_cause} cleanup={outcome.cleanup_status} "
            f"invocation_id={receipt.invocation_id} completion={receipt.path}\n"
        )
        stream.flush()
    except (OSError, ValueError) as reporting_error:
        if supervision_exception is not None:
            # The receipt keeps the structured failure; a broken diagnostic sink
            # must not replace the original monitor error with "closed stderr".
            raise supervision_exception from reporting_error
        raise
    return outcome.wrapper_exit_code
