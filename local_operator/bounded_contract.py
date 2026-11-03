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
from typing import Any, Callable

SCHEMA = "run-bounded-completion/v1"
CONTRACTS = ("cleanup-required", "command-status")
# Windows has no SIGHUP/SIGQUIT. The module must still IMPORT there (the
# cross-platform probe imports every package module); execution on such a host
# refuses by name through ReservedLeader.unavailable_reason (no os.killpg).
FORWARDED = tuple(
    getattr(signal, name)
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")
    if hasattr(signal, name)
)
#: Causes that mean no command process was ever created. The stderr status line
#: says ``spawned=no`` for them, so a pre-spawn ``cleanup=unknown`` cannot be
#: misread as "a run happened and may have leaked" (design review D4).
PRE_SPAWN_CAUSES = frozenset(
    {
        "invalid-request",
        "unsupported-backend",
        "unsupported-boundary",
        "memory-policy-refused",
        "spawn-refused",
        "spawn-failure",
    }
)


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
    # Policy disclosures copied into the receipt. ``terminal_observer`` names the
    # instrument command-status used to see the leader die without reaping it.
    memory_ceiling_mb: int | None = None
    memory_budget_source: str | None = None
    terminal_observer: str | None = None

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

    def publish(self, contract: str, outcome: Outcome) -> dict[str, object]:
        record: dict[str, object] = {
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
            # command-status signals its group by killpg only while the leader is
            # unreaped (the reservation); strict signals through census authority.
            "signal_scope": (
                "reserved-leader-group" if contract == "command-status" else "owned-process-group"
            ),
            "record_phase": "complete",
            "supervision_error": outcome.supervision_error,
            # These are policy/guarantee disclosures, not timing measurements.
            "deadline_guarantee": "none: cooperative observation and signal attempts only",
            "memory_guarantee": (
                "sampled owned-group policy, not a hard allocation bound"
                if (outcome.memory_ceiling_mb or 0) > 0
                else "none: no memory ceiling for this run"
            ),
            "memory_ceiling_mb": outcome.memory_ceiling_mb,
            "memory_budget_source": outcome.memory_budget_source,
            "terminal_observer": outcome.terminal_observer,
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


class ReservedLeader:
    """The unreaped leader IS the owned group's reservation; one thread reaps it.

    ``start_new_session=True`` makes the child lead a fresh group whose number is
    its pid, and the kernel cannot hand that number to anyone else while the
    leader is unreaped. So every ``killpg(proc.pid)`` issued BEFORE our single
    consuming wait reaches only processes carrying our group number, and none is
    ever issued after it. Observation is therefore non-consuming: ``waitid``
    with ``WNOWAIT`` where CPython exposes it (macOS only from 3.13), else the
    Darwin kernel's own zombie flag via ``sysctl(KERN_PROC_PID)``. An unreadable
    observation is ``None``, never terminal. This is signal authority, not a
    census: it can never certify that the group is empty.

    Preconditions this class cannot enforce (embedding contract): default
    SIGCHLD disposition and no other waiter for this pid. ``Popen.poll``/
    ``wait`` are forbidden here because ``_try_wait`` maps ECHILD to status 0,
    turning a foreign reap into a fabricated child success.
    """

    #: Primitives the WNOWAIT observer needs; their absence selects the fallback.
    _WAITID = ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")

    def __init__(self, proc, *, observer: str | None = None):
        self.proc = proc
        self._revoked = False
        self.observer = observer or self.select_observer()
        # The leader's kernel birth, recorded on first sysctl read: a pid whose
        # birth changes was reaped by someone else and reused by a stranger.
        self._birth: str | None = None

    @classmethod
    def select_observer(cls) -> str | None:
        if all(hasattr(os, name) for name in cls._WAITID):
            return "waitid-wnowait"
        if sys.platform == "darwin":
            return "darwin-sysctl-zombie"
        return None

    @classmethod
    def unavailable_reason(cls) -> str | None:
        if not hasattr(os, "killpg"):
            return (
                "owned-group signalling unavailable: os.killpg is missing (POSIX process "
                "groups are required)"
            )
        if cls.select_observer() is None:
            missing = ", ".join(f"os.{name}" for name in cls._WAITID if not hasattr(os, name))
            return (
                "no non-consuming terminal observer: missing "
                f"{missing}, and no Darwin sysctl fallback on {sys.platform}"
            )
        # A custom SIGCHLD handler may be another reaper, not merely a notifier.
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            return "exclusive-reaper observation unavailable: SIGCHLD is not default"
        return None

    @classmethod
    def available(cls) -> bool:
        return cls.unavailable_reason() is None

    def observe(self) -> bool | None:
        """True once the leader is a corpse; None when the observer cannot say.

        Never consumes the status. A pid that is no longer our child (ECHILD) or
        whose identity changed means someone else reaped it: authority is
        revoked and the error propagates, because the group number may already
        belong to a stranger.
        """
        if self._revoked:
            raise RuntimeError("reserved-leader authority irrevocably revoked")
        pid = self.proc.pid
        if self.observer == "waitid-wnowait":
            waitid = getattr(os, "waitid")
            flags = getattr(os, "WEXITED") | getattr(os, "WNOHANG") | getattr(os, "WNOWAIT")
            try:
                return waitid(getattr(os, "P_PID"), pid, flags) is not None
            except ChildProcessError:
                self._revoked = True
                raise
            except OSError:
                return None
        if self.observer == "darwin-sysctl-zombie":
            from local_operator import procstate

            samples = procstate._sysctl_samples([pid])
            if samples is None:
                return None
            sample = samples.get(pid)
            if sample is None or (
                self._birth is not None and sample.birth is not None and sample.birth != self._birth
            ):
                self._revoked = True
                raise ChildProcessError(f"leader {pid} was reaped outside this supervisor")
            if self._birth is None:
                self._birth = sample.birth
            return sample.zombie
        return None

    def send_group(self, signum: int) -> bool:
        """killpg the owned group; False when it has no signalable member.

        The return value is diagnostic only: no supervisor branches on it,
        because "nobody signalable" is advice, never cleanup evidence.

        ESRCH is not authority loss: the leader is still unreaped by us, so the
        number is still ours and simply holds nobody live. Any other failure
        revokes, so a later tick cannot retry into an unknown state.
        """
        if self._revoked:
            raise RuntimeError("reserved-leader authority irrevocably revoked")
        try:
            os.killpg(self.proc.pid, signum)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Darwin answers EPERM, not ESRCH, when the group's only member is
            # our unreaped (zombie) leader (measured 2026-11-02, macOS 27.2).
            # Accept that reading only when the kernel's own listing agrees the
            # group holds nobody else; any other EPERM is real authority loss.
            from local_operator.mobile.resources import direct_group_pids

            members = direct_group_pids(self.proc.pid)
            if members is not None and all(pid == self.proc.pid for pid in members):
                return False
            self._revoked = True
            raise
        except BaseException:
            self._revoked = True
            raise
        return True

    def reap(self) -> int:
        """Revoke, then consume the leader's status directly (never Popen.poll)."""
        self._revoked = True
        _, status = os.waitpid(self.proc.pid, 0)
        code = os.waitstatus_to_exitcode(status)
        self.proc.returncode = code
        return code

    def try_reap(self) -> int | None:
        """Revoke, then consume the status only if the leader is ALREADY dead.

        For a leader we have SIGKILLed but cannot observe: a blocking waitpid
        there has no bound (uninterruptible sleep, and PEP 475 retries it
        through our latched signals). Revoking first is safe because no signal
        may follow this call whether or not it consumed anything.
        """
        self._revoked = True
        pid, status = os.waitpid(self.proc.pid, os.WNOHANG)
        if pid == 0:
            return None
        code = os.waitstatus_to_exitcode(status)
        self.proc.returncode = code
        return code

    def revoke(self) -> None:
        self._revoked = True


#: Silence from the terminal observer longer than this is a broken instrument,
#: not a slow child: the leader is killed and the run reports 126.
OBSERVER_UNREADABLE_SECONDS = 2.0
#: How long a SIGKILLed leader may take to become observable as a corpse. A
#: multi-GB heap takes real time to tear down; past this, we stop waiting.
KILL_SETTLE_SECONDS = 2.0
#: Upper bound on the leader-exit sweep's SIGTERM grace, whatever --grace says:
#: the command has already finished, so only stragglers are being negotiated.
SWEEP_GRACE_CAP_SECONDS = 2.0


def _sweep(leader, *, grace, clock, sleep, group_pids, report) -> None:
    """Leader is a corpse, still unreaped: clear stragglers BEFORE giving it up.

    The group listing is advice about WHETHER to signal, never cleanup proof. An
    unreadable listing (None) is treated as "maybe members", so we signal.
    """
    pgid = leader.proc.pid

    def stragglers() -> bool:
        members = group_pids(pgid)
        return members is None or any(pid != pgid for pid in members)

    if not stragglers():
        return
    report(f"[run_bounded] leader exited with members left in group {pgid} — SIGTERM")
    leader.send_group(signal.SIGTERM)
    until = clock() + min(grace, SWEEP_GRACE_CAP_SECONDS)
    while clock() < until:
        if not stragglers():
            return
        sleep(0.05)
    report(f"[run_bounded] group {pgid} outlived the sweep's SIGTERM — SIGKILL")
    leader.send_group(signal.SIGKILL)


def supervise_direct(
    argv,
    *,
    timeout,
    grace,
    env,
    outcome,
    stop,
    budget=None,
    stream=None,
    spawn: Callable[..., Any] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    boundary: Callable[[Any], Any] | None = None,
    guard_factory: Callable[..., Any] | None = None,
    group_pids: Callable[[int], list[int] | None] | None = None,
    memory_escape: str | None = None,
):
    """Command status plus an ENFORCED owned-group memory grant; no cleanup proof.

    One loop, one thread, one consuming wait. Per tick: latch interruption or
    deadline (requested to the owned group, SIGKILL after ``grace``), sample the
    memory guard while the leader is unreaped, then observe without consuming.
    A terminal leader is swept and only then reaped; no signal follows the reap.
    Uninterruptible sleep, scheduler delay and escaped (setsid) descendants still
    prevent any honest hard wall-time/RAM promise, so cleanup stays UNKNOWN.
    """
    write = stream.write if stream is not None else sys.stderr.write
    boundary = ReservedLeader if boundary is None else boundary
    if group_pids is None:
        from local_operator.mobile.resources import direct_group_pids as group_pids
    make_guard: Callable[..., Any]
    if guard_factory is None:
        import functools

        from local_operator.memory_guard import Guard

        # Charge descendants that setsid() out of the group too; see
        # memory_guard.Guard._escaped_descendants.
        make_guard = functools.partial(Guard, descendants=True)
    else:
        make_guard = guard_factory

    def report(text: str) -> None:
        # A broken diagnostic sink must not cost the group its signals.
        try:
            write(text + "\n")
        except (OSError, ValueError):
            pass

    started = clock()
    try:
        proc = spawn(list(argv), start_new_session=True, env=env)
    except OSError as exc:
        outcome.wrapper_exit_code = 125
        outcome.termination_cause = "spawn-refused"
        outcome.supervision_error = type(exc).__name__
        report(f"[run_bounded] could not start {argv[0]!r}: {exc.strerror or exc}")
        return
    leader = boundary(proc)
    outcome.terminal_observer = getattr(leader, "observer", None)
    pgid = proc.pid
    ceiling = getattr(budget, "ceiling_mb", 0) or 0
    guard = make_guard(pgid, budget) if budget is not None and ceiling > 0 else None
    next_sample = started
    deadline = math.inf if timeout is None else started + timeout
    escalating = None
    kill_at = settle_until = math.inf
    termed = killed = False
    unreadable_since = None
    raw_status: int | None = None
    forced_failure = False

    def kill_group(now: float) -> None:
        nonlocal killed, settle_until
        leader.send_group(signal.SIGKILL)
        killed = True
        settle_until = now + KILL_SETTLE_SECONDS

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
                    report(f"[run_bounded] signal {escalating} received — signalling group {pgid}")
                elif now >= deadline:
                    escalating = signal.SIGTERM
                    outcome.wrapper_exit_code = 124
                    outcome.termination_cause = "deadline"
                    report(f"[run_bounded] timeout after {timeout:g}s — SIGTERM to group {pgid}")
                    report(
                        f"[run_bounded] that is the BOUND ({timeout:g}s) firing, not the gate "
                        "failing — re-run it, or raise it (--timeout, or "
                        "`make type-check BOUND_TIMEOUT=<seconds>`)"
                    )
                if escalating is not None:
                    kill_at = now + grace
            if escalating is not None and not killed:
                if not termed:
                    leader.send_group(escalating)
                    termed = True  # One forwarding attempt, not one per tick.
                elif now >= kill_at:
                    report(f"[run_bounded] group {pgid} outlived {grace:g}s grace — SIGKILL")
                    kill_group(now)
            # The grant stays enforced through a deadline/interrupt grace too: a
            # SIGTERM-ignoring group must not balloon while we wait to SIGKILL it.
            if guard is not None and not killed and now >= next_sample:
                sample = guard.sample_sync()
                next_sample = now + guard.tick_s
                # Disclosed ONCE per pid (review R1): unreadable by credential
                # is named, never counted toward fail-closed.
                new_np = getattr(guard, "new_not_permitted", None)
                fresh = new_np(sample) if new_np else []
                if fresh:
                    from local_operator.memory_guard import not_permitted_line

                    report(f"[run_bounded] {not_permitted_line(sample, fresh)}")
                if guard.should_kill(sample):
                    if getattr(sample, "unaccounted", None):
                        from local_operator.memory_guard import unaccounted_line

                        report(f"[run_bounded] {unaccounted_line(sample)}")
                    # MiB plus the caller's own knob (design D2.1); None keeps
                    # the guard's default wording for any other embedder.
                    if memory_escape is None:
                        message = guard.over_budget_message(sample)
                    else:
                        message = guard.over_budget_message(
                            sample, unit="MiB", escape=memory_escape
                        )
                    report(f"[run_bounded] {message}")
                    # Escaped descendants FIRST: their parent edge is the only
                    # ownership proof, and it is gone once the group dies.
                    stop_escaped = getattr(guard, "stop_escaped", None)
                    stopped = stop_escaped(signal.SIGKILL) if stop_escaped else []
                    if stopped:
                        report(
                            "[run_bounded] SIGKILL to escaped descendant(s) "
                            + ", ".join(str(pid) for pid in stopped)
                        )
                    unsignalled = getattr(guard, "unsignalled", None)
                    if unsignalled:
                        from local_operator.memory_guard import unsignalled_line

                        report(f"[run_bounded] {unsignalled_line(unsignalled)}")
                    if escalating is None:
                        outcome.wrapper_exit_code = 128 + signal.SIGKILL
                        outcome.termination_cause = "memory-policy"
                        escalating = signal.SIGKILL
                    termed = True
                    kill_group(now)
            terminal = leader.observe()
            if terminal is None:
                unreadable_since = now if unreadable_since is None else unreadable_since
                if now - unreadable_since >= OBSERVER_UNREADABLE_SECONDS:
                    # Unknown is never terminal and never success: stop the run.
                    report("[run_bounded] terminal observer unreadable for 2s — SIGKILL")
                    forced_failure = True
                    outcome.supervision_error = "terminal observer unreadable"
                    outcome.wrapper_exit_code = 126
                    outcome.termination_cause = "supervision-failure"
                    if not killed:
                        kill_group(now)
                    # Poll, never block: the leader may not die (R2.5). An
                    # unsettled leader still reports 126, with no status.
                    while True:
                        reaped = leader.try_reap()
                        if reaped is not None:
                            raw_status = reaped
                            outcome.command_exit_code = shell_status(reaped)
                            return
                        if clock() >= settle_until:
                            outcome.supervision_error = (
                                "terminal observer unreadable; leader termination not established"
                            )
                            return
                        sleep(0.05)
            else:
                unreadable_since = None
            if terminal:
                _sweep(
                    leader,
                    grace=grace,
                    clock=clock,
                    sleep=sleep,
                    group_pids=group_pids,
                    report=report,
                )
                status: int = leader.reap()
                raw_status = status
                outcome.command_exit_code = shell_status(status)
                if status < 0 and escalating is None:
                    report(f"[run_bounded] child killed by signal {-status}")
                return
            if killed and now >= settle_until:
                outcome.supervision_error = "leader termination not established"
                return
            sleep(0.05)
    except Exception as exc:
        # Preserve established deadline/interruption/memory outcomes; never
        # convert a failed wait/send into command success or group cleanliness.
        outcome.supervision_error = type(exc).__name__
        if outcome.termination_cause not in {"deadline", "interruption", "memory-policy"}:
            outcome.wrapper_exit_code = 126
            outcome.termination_cause = "supervision-failure"
    finally:
        leader.revoke()
        if stop["signum"]:
            outcome.wrapper_exit_code = 128 + stop["signum"]
            outcome.termination_cause = "interruption"
        elif (
            raw_status is not None
            and outcome.command_exit_code is not None
            and escalating is None
            and not forced_failure
        ):
            outcome.wrapper_exit_code = outcome.command_exit_code
            outcome.termination_cause = "child-signal" if raw_status < 0 else "child-exit"


def _report_budget(stream, ceiling: int, source: str | None, budget) -> None:
    # The effective ceiling and its provenance, the line AGENTS.md tells readers
    # to check; printed for command-status here and by the strict runner there.
    try:
        if ceiling > 0:
            stream.write(
                f"[run_bounded] memory budget {ceiling} MiB ({source or 'unknown'}); "
                "sampled owned child group and escaped descendants\n"
            )
        else:
            stream.write(disabled_budget_line(source, budget) + "\n")
    except (OSError, ValueError):
        pass


def disabled_budget_line(source: str | None, budget) -> str:
    """The guard-off line in run_bounded's own vocabulary (design D2.3).

    memory_guard's reason is written for the bash tool (`memory_mb=0`) and
    repeated the word "disabled"; the wrapper's user typed `--memory-mb 0`.
    """
    if source == "explicit":
        return "[run_bounded] memory guard disabled by --memory-mb 0"
    if source == "inherited: guard off":
        return "[run_bounded] memory guard disabled: the inherited budget is 0 (guard off)"
    reason = getattr(budget, "reason", "") or "no ceiling"
    return f"[run_bounded] memory guard disabled ({source or 'unknown'}): {reason}"


def status_line(contract: str, outcome: Outcome, receipt) -> str:
    """The one machine-greppable summary every path ends with."""
    spawned = " spawned=no" if outcome.termination_cause in PRE_SPAWN_CAUSES else ""
    return (
        f"[run_bounded] contract={contract} rc={outcome.wrapper_exit_code} "
        f"cause={outcome.termination_cause} cleanup={outcome.cleanup_status}{spawned} "
        f"invocation_id={receipt.invocation_id} completion={receipt.path}"
    )


def refusal_advice(outcome: Outcome, contract: str, *, argv, backend) -> str | None:
    """One human sentence per pre-spawn refusal: the reason AND the next step.

    The receipt keeps the structured cause; this is for the person reading a
    failed ``make`` line, who otherwise sees only a token (design review D1).
    Advice never points back at the option that just refused (D2).
    """
    cause = outcome.termination_cause
    if cause == "invalid-request":
        if not argv:
            return "no command was given; pass it after a `--` separator"
        return "--timeout and --grace must be finite, nonnegative seconds"
    if cause == "unsupported-backend":
        return (
            f"cleanup-required cannot certify owned-group cleanup on {backend.name} "
            f"({backend.reason}). Run this gate on Linux/CI, or pass "
            "--contract=command-status if the command's own status is enough (it "
            "enforces the memory grant but never certifies cleanup)"
        )
    if cause == "unsupported-boundary":
        return (
            f"command-status cannot supervise here: {outcome.cleanup_reason}. It needs "
            "POSIX process groups, a non-consuming terminal observer and the default "
            "SIGCHLD disposition; use a POSIX host and do not install a SIGCHLD handler"
        )
    return None


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
    strict: Callable[[Outcome], None],
    backend=None,
    direct: Callable[..., None] = supervise_direct,
    boundary: Any = ReservedLeader,
    budget_source: str | None = None,
    memory_escape: str | None = None,
) -> int:
    """Called after canonical _memory_budget, before any handler or Popen.

    Supported strict gates retain their existing monitor and sampled memory
    policy. Command-status ENFORCES the same positive group grant with the same
    memory guard on its own unreaped-leader group (architect ruling, option B):
    a grant is never silently dropped by choosing the weaker contract. The
    canonical resolver has already rejected disabling/widening that grant.
    Finalization exceptions (Exception, not process-level BaseException) produce
    a failure receipt and integer status, subject to publication/stream I/O. All
    handler restores are attempted; failure is recorded even when interruption
    or an established policy retains its cause/status. A failed restoration may
    leave our handler installed: this diagnostic is not a restoration guarantee.
    """
    backend = selected_backend() if backend is None else backend
    ceiling = getattr(budget, "ceiling_mb", 0) or 0
    outcome = Outcome(memory_ceiling_mb=ceiling, memory_budget_source=budget_source)
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
            _report_budget(stream, ceiling, budget_source, budget)
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

            _report_budget(stream, ceiling, budget_source, budget)
            for sig in FORWARDED:
                previous[sig] = signal.signal(sig, latch)
            outcome.cleanup_reason = (
                "command-status never certifies descendant or owned-group cleanup"
            )
            direct(
                argv,
                timeout=timeout,
                grace=grace,
                env=child_env,
                outcome=outcome,
                stop=stop,
                budget=budget,
                stream=stream,
                memory_escape=memory_escape,
            )
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
        advice = refusal_advice(outcome, contract, argv=argv, backend=backend)
        if advice:
            stream.write(f"[run_bounded] refused before spawn: {advice}\n")
        stream.write(status_line(contract, outcome, receipt) + "\n")
        stream.flush()
    except (OSError, ValueError) as reporting_error:
        if supervision_exception is not None:
            # The receipt keeps the structured failure; a broken diagnostic sink
            # must not replace the original monitor error with "closed stderr".
            raise supervision_exception from reporting_error
        raise
    return outcome.wrapper_exit_code
