#!/usr/bin/env python3
"""Run a gate command in its OWN process group with checked signal authority.

WHY THIS EXISTS
---------------
`pyright` (the pip wrapper) runs the real analyzer as an npm/node child, and the
local gate used to be spelled `timeout 900 .venv/bin/python -m pyright …`.

On this host's GNU coreutils, `timeout(1)` turns out to signal the child's whole
process group, so the tidy causal story — "the bound fires, the wrapper dies,
node survives as an orphan" — is NOT the mechanism, and it could not be
reproduced on demand: `timeout 6`, `timeout -k 2 6`, `timeout -s KILL 6` and the
job-control shape all left nothing behind on the machines this was tried on.
What the fleet DID show is the symptom no such run explains:

    $ ps -o pid,ppid,pgid,rss,etime,command -p <node pid>
      pid   ppid  pgid   rss  etime  command
    92099      1 92098  2.28G  03:41  node …/pyright/index.js --outputjson …

`ppid 1` is the defect: the orphan is re-parented to launchd and keeps its heap
for as long as its work takes — measured at 2.28 GB and 1.50 GB on one host, ten
analyzers alive at once, 5.1 GB total, one alive 81 minutes after its parent
died. Sessions queue more pyrights behind a host those orphans are already
saturating, more of them hit the bound, and the leak compounds. A bound that
leaves a gigabyte running is not a bound.

Two failure modes ARE reproducible here, and they are enough to justify this
wrapper on their own:

* a descendant still alive when the leader exits BY ITSELF — an ordinary exit is
  not a bound at all, and nothing else reaps the group;
* a group that ignores SIGTERM, where a bare `timeout` waits it out (measured over
  a SIGTERM-ignoring tree: `timeout 3` fires at 3 s and then waits the TREE out —
  30 s for a 30 s tree, 300 s in an earlier run for a 300 s one — where this
  wrapper's SIGKILL escalation clears the same tree in 5 s with `--grace 2`; the
  shipped `--grace` default of 10 would take ~13 s for it).

So the wrapper puts the command in its own session (`start_new_session=True`, so
it leads a fresh process group its children inherit) and requests a checked group signal on
every exit path: the timeout, a forwarded signal, and the ordinary exit where the
leader is gone but its descendants are not.

What `timeout(1)` cannot do is the third case: by then it HAS exited, so no
spelling of it can reap anything. `setsid` does not exist on macOS, which is the
platform this bites hardest.

Node heap headroom belongs here too: whole-tree pyright OOMed at node's ~4 GB
old-space cap under fleet load (rc=250 from its Python wrapper). Both the scoped
runner and Makefile use this wrapper, so sizing only the Makefile leaves the
scoped fallback to whole-tree analysis exposed. Match ci.yml's heap floor without
lowering a larger operator value. NODE_OPTIONS is inert for non-node children,
so passing it to every wrapped command is harmless.

USAGE
-----
    python scripts/run_bounded.py --timeout 1800 -- <command> [args…]
    python scripts/run_bounded.py -- <command> [args…]      # no bound, checked cleanup

Run it through the interpreter (`.venv/bin/python scripts/run_bounded.py …`),
never as a console script — the #423 shebang rule applies to this file too.

EXIT STATUS
-----------
The command's own status; `124` when the bound fired (what `timeout(1)` reports,
so existing callers keep reading the same code); `128 + signum` when a signal
ended it — forwarded to the group here, or delivered to the child by someone
else (a `kill -9`, the OOM killer, a job runner), which a shell would report the
same way; `137` for a measured owned-group memory breach, identified by the
`MEMORY LIMIT EXCEEDED` diagnostic (status 137 alone also covers an external
SIGKILL); `125` when the command could not start or its memory budget was refused
before spawn, or a previously successful gate whose cleanup was REFUSED.
Timeout/RAM/signal/nonzero child statuses are retained with secondary residual
receipts on refusal; a requested signal is never reported as proven reaping.

SAFETY BOUNDARY
---------------
Fresh COMPLETE membership, canonical birth/parent-birth and actual PG/SID are
required before EVERY signal. New members enrol only through a live verified
creation chain before polling can reap the leader. After leader exit a previously
authenticated live survivor must anchor continuity. Unknown/reused/mixed/terminal-
only groups refuse with a nonzero outcome and a durable residual receipt.
User-space census-to-killpg remains NON-ATOMIC; this is checked owner-alive, not a
kernel-held capability or a hard-owner-SIGKILL immediate-cleanup guarantee.
Escaped/new-session descendants are outside this group's signal authority.
No keeper, tokenless fallback, or existing monitor-policy changes are introduced.

DIAGNOSTICS go to stderr, never stdout: the lint/format gates' stdout is
parseable output and the suite's `-q` output is parsed too (the same rule the
pytest worker cap follows in `conftest.py`).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Sequence

# This script is also called with Python -S/from outside the checkout. Load the
# canonical stdlib-only guard from THIS script's tree, not a stale tool install
# or another worktree's editable finder. No dependency installation is needed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_operator import memory_guard, owned_process_group  # noqa: E402

#: Exit status for a fired bound. `timeout(1)` uses the same number.
EXIT_TIMEOUT = 124

#: Keep in sync with .github/workflows/ci.yml's type-check NODE_OPTIONS: both
#: local gate spellings need the analyzer headroom, not just the Makefile recipe.
NODE_HEAP_MB = 6144

#: The command never started (missing interpreter, bad path). Distinguished from
#: any status the command itself can produce, so a gate that did not run cannot
#: be read as a passing gate.
EXIT_NOT_STARTED = 125

#: A measured owned-group memory breach is SIGKILL, not the timeout 124 path.
EXIT_MEMORY = 128 + signal.SIGKILL

#: How long a signalled process group gets before SIGKILL. The analyzer needs a
#: moment to unwind; a group that ignores SIGTERM entirely must still not outlive
#: the gate, which is the failure this module exists to remove.
DEFAULT_GRACE_SECONDS = 10.0

#: Signals forwarded to the child's group. SIGHUP and SIGQUIT are here for the
#: same reason as the other two — a dangling terminal, a tmux/ssh teardown or a
#: `kill -HUP` must not leave the analyzer running. Measured with only
#: SIGINT/SIGTERM forwarded: SIGHUP killed this wrapper and the node child
#: survived it at `ppid 1`, which is the same leak by another road.
FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)

#: Polling interval while waiting for the group leader. Deliberately a poll loop
#: rather than `Popen.wait()`: PEP 475 auto-retries the blocking waitpid after a
#: signal handler returns, so a handler that only records the signal could not
#: get control back in time to bound the group.
_POLL_SECONDS = 0.05


def _signal_group(owner, signum: int, *, deadline: float):
    """One capability boundary; numeric groups cannot be supplied as authority."""
    return owner.signal(signum, deadline=deadline)


def _report(stream, text):
    # A closed pipe is not permission to skip cleanup or mask the original error.
    try:
        stream.write(text + "\n")
        stream.flush()
    except (OSError, ValueError):
        pass


def _refusal_receipt(owner, phase, verdict, stream):
    """Durable residual facts only: no argv, environment, capability or nonce."""
    import json
    import uuid
    from dataclasses import asdict

    base = Path(os.environ.get("LOCAL_OPERATOR_SCRATCHPAD") or
                os.environ.get("LOCAL_OPERATOR_CONFIG_DIR") or
                str(Path.home() / ".local-operator")) / "run-bounded-refusals"
    try:
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = base / (uuid.uuid4().hex + ".json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump({"phase": phase, "pgid": owner.pgid,
                       "verdict": asdict(verdict),
                       "receipts": [asdict(r) for r in owner.receipts.values()],
                       "monotonic": time.monotonic(),
                       "cleanup": "REFUSED; residuals not claimed absent"}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _report(stream, f"[run_bounded] safety refusal ({phase}); residual receipt: {path}")
    except (OSError, ValueError) as exc:
        # This is already a nonzero safety failure, not a recovered cleanup. Do
        # not replace an original monitor exception with a second capture error.
        _report(stream, f"[run_bounded] residual capture failed: {type(exc).__name__} errno={getattr(exc, 'errno', None)}")


def _reap(owner, grace, poll, stream, *, reason, send, deadline):
    """TERM and KILL EACH obtain fresh authority; refusal never enters a wait."""
    _report(stream, f"[run_bounded] {reason} — requesting checked SIGTERM")
    verdict = send(signal.SIGTERM, reason + ":term", deadline)
    if verdict.status == "REFUSED":
        return
    while time.monotonic() < deadline:
        owner.refresh(deadline=deadline)  # retain ancestry BEFORE reaping leader
        if poll() is not None:
            return
        time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
    send(signal.SIGKILL, reason + ":kill", deadline + 0.25)


def _memory_budget(override_mb: int | None) -> tuple[memory_guard.Budget, str]:
    """Resolve a cooperative child ceiling without widening its parent grant.

    Missing means canonical auto policy; zero is an explicit, disclosed disable.
    Invalid metadata is an error, never a spelling of unlimited. This only
    prices the owned child PG, not the wrapper+child aggregate or later setsid.
    """
    raw = os.environ.get(memory_guard.RESOLVED_MEMORY_MB_ENV)
    inherited: int | None = None
    if raw is not None:
        if re.fullmatch(r"[0-9]+", raw) is None:
            raise ValueError(
                f"{memory_guard.RESOLVED_MEMORY_MB_ENV} must be nonnegative integer MiB"
            )
        inherited = int(raw)
    if override_mb is not None and (type(override_mb) is not int or override_mb < 0):
        raise ValueError("--memory-mb must be nonnegative integer MiB")
    if inherited is not None and inherited > 0:
        if override_mb is not None and (override_mb == 0 or override_mb > inherited):
            raise ValueError(
                f"--memory-mb cannot disable or widen inherited {inherited} MiB; "
                f"choose a positive value no higher than {inherited}, or raise the "
                "granting caller's ceiling (memory_mb on the outer bash tool call)"
            )
        ceiling = inherited if override_mb is None else override_mb
        source = "inherited" if override_mb is None else "explicit, capped by inherited"
    elif override_mb is not None:
        ceiling = override_mb
        source = "explicit"
    elif inherited is not None:
        ceiling = inherited
        source = "inherited: guard off"
    else:
        budget = memory_guard.compute_budget()
        if budget.source == "disabled":
            raise ValueError(
                "auto memory budget unavailable; set --memory-mb N to an explicit positive "
                "MiB ceiling (--memory-mb 0 deliberately disables protection)"
            )
        return budget, "auto"
    return memory_guard.compute_budget(override_mb=ceiling), source


def run(
    argv: Sequence[str],
    *,
    timeout: float | None,
    grace: float,
    memory_mb: int | None = None,
    stream: IO[str] = sys.stderr,
) -> int:
    """Run `argv` as a fresh process group and return its status.

    `timeout` is None for an unbounded run: the group is still reaped on exit,
    because the orphan in this module's docstring is produced by an ordinary exit
    as well as by a bound (the wrapper exits, the node child does not).
    """
    if not argv:
        stream.write("[run_bounded] no command was given\n")
        return EXIT_NOT_STARTED
    try:
        budget, budget_source = _memory_budget(memory_mb)
        if budget.ceiling_mb > 0 and not hasattr(os, "killpg"):
            raise ValueError("a child process-group memory ceiling requires POSIX process groups")
    except ValueError as exc:
        stream.write(f"[run_bounded] memory budget refused before spawn: {exc}\n")
        return EXIT_NOT_STARTED
    if budget.ceiling_mb > 0:
        stream.write(
            f"[run_bounded] memory budget {budget.ceiling_mb} MiB ({budget_source}); "
            "sampled owned child group only\n"
        )
    else:
        stream.write(f"[run_bounded] memory guard disabled ({budget_source}): {budget.reason}\n")
    stream.flush()
    started = time.monotonic()
    # A signal to this wrapper has to reach the group: the child leads its own
    # session, so the terminal's SIGINT goes to the wrapper alone and an
    # unforwarded Ctrl-C would leave the analyzer running.
    #
    # The handlers are installed BEFORE the child is spawned, and they forward
    # through a slot the spawn fills in. Installing them afterwards leaves a
    # window in which a signal kills the wrapper outright — default disposition —
    # and nothing forwards it or reaps the group; that window was measured on the
    # Linux CI leg as `rc=-15` with an empty stderr, i.e. a wrapper that died
    # before it could report anything.
    stop: dict[str, int] = {"signum": 0}

    def _forward(signum: int, _frame: object) -> None:
        stop["signum"] = signum
        # Handlers latch intent only; probes and syscalls run in the main loop.

    child_env = os.environ.copy()
    # Nested cooperative wrappers inherit the effective (possibly tightened)
    # ceiling, not the original wider grant. env -i must carry it explicitly.
    child_env[memory_guard.RESOLVED_MEMORY_MB_ENV] = str(budget.ceiling_mb)
    node_options = child_env.get("NODE_OPTIONS", "")
    # Match whole options after unquoting: a raw regex misses quoted heap flags
    # and can mistake a heap-looking substring in --title's value for a flag.
    try:
        option_tokens = shlex.split(node_options)
    except ValueError:
        # Leave malformed caller text for the child to diagnose, but don't let
        # an unmatched quote trick us into recognising a larger heap option.
        option_tokens = []
    heaps = [
        match.group(1)
        for token in option_tokens
        if (match := re.fullmatch(r"--max[-_]old[-_]space[-_]size=(\+?\d+)", token))
    ]
    # V8 resolves repeated flags last-wins: append after a smaller effective
    # value, preserving unrelated options and leaving a larger caller cap alone.
    if not heaps or int(heaps[-1]) < NODE_HEAP_MB:
        child_env["NODE_OPTIONS"] = (
            f"{node_options} " if node_options else ""
        ) + f"--max-old-space-size={NODE_HEAP_MB}"

    previous = {sig: signal.signal(sig, _forward) for sig in FORWARDED_SIGNALS}
    try:
        proc = subprocess.Popen(list(argv), start_new_session=True, env=child_env)
    except OSError as exc:
        # A gate that never ran must never look green (#423's lesson).
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        stream.write(f"[run_bounded] could not start {argv[0]!r}: {exc}\n")
        return EXIT_NOT_STARTED
    pgid = proc.pid
    deadline = None if timeout is None else started + timeout
    # Provisional numeric identity grants NO signal authority. Capture before ANY
    # poll/wait can reap this leader; failed capture remains a bounded refusal.
    owner = owned_process_group.OwnedGroup.capture(
        proc.pid, deadline=min(deadline if deadline is not None else float("inf"),
                               time.monotonic() + 3.0))
    refused = False
    cleanup_deadline = None

    def begin_cleanup(allowance):
        nonlocal cleanup_deadline
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + allowance
        return cleanup_deadline

    def send(signum, phase, until):
        nonlocal refused
        result = _signal_group(owner, signum, deadline=until)
        if result.status == "REFUSED":
            refused = True
            _refusal_receipt(owner, phase, result, stream)
        return result

    def bounded_wait(until):
        try:
            proc.wait(timeout=0 if refused else max(0.0, until - time.monotonic()))
        except subprocess.TimeoutExpired:
            # Waiting is not cleanup proof. No PID-only or numeric-PG fallback.
            return False
        return True

    rc = EXIT_NOT_STARTED
    try:
        guard = memory_guard.Guard(pgid, budget) if budget.ceiling_mb > 0 else None
        next_sample = started
        unmeasured_reported = False
        while True:
            # Enrolment needs a still-live creation chain; do this before poll
            # can reap the only authenticated ancestor. Unknown stays unknown.
            owner.refresh(deadline=min(deadline if deadline is not None else float("inf"),
                                       time.monotonic() + 3.0))
            signum = stop["signum"]
            if signum:
                until = begin_cleanup(grace + 0.25)
                send(signum, "forward", until)
                _reap(owner, grace, proc.poll, stream, reason=f"signal {signum} received",
                      send=send, deadline=until - 0.25)
                rc = 128 + signum
                break
            status = proc.poll()
            if status is not None:
                rc = 128 - status if status < 0 else status
                break
            if deadline is not None and time.monotonic() >= deadline:
                until = begin_cleanup(grace + 0.25)
                _reap(owner, grace, proc.poll, stream, reason=f"timeout after {timeout:g}s",
                      send=send, deadline=until - 0.25)
                rc = EXIT_TIMEOUT
                _report(stream, f"[run_bounded] BOUND ({timeout:g}s) fired, not the gate failing; --timeout sets the bound")
                break
            now = time.monotonic()
            if guard is not None and now >= next_sample:
                sample = guard.sample_sync()
                next_sample = now + guard.tick_s
                if guard.should_kill(sample):
                    _report(stream, f"[run_bounded] {guard.over_budget_message(sample)}")
                    send(signal.SIGKILL, "memory", begin_cleanup(0.25))
                    rc = EXIT_MEMORY
                    break
                notice = guard.soft_notice(sample)
                if notice:
                    _report(stream, f"[run_bounded] {notice} (advisory only)")
                if sample.bytes_used is None and not unmeasured_reported:
                    _report(stream, "[run_bounded] memory charge unavailable; no measured cap proof")
                    unmeasured_reported = True
            time.sleep(_POLL_SECONDS)
    except Exception as exc:
        _report(stream, f"[run_bounded] owned-child monitor failed: {type(exc).__name__}")
        raise
    finally:
        try:
            # Retain the existing quarter-second final allowance, not a new
            # grace period per pass. Never claim a requested signal means reaped.
            until = begin_cleanup(0.25)
            for attempt in range(5):
                owner.refresh(deadline=until)
                proc.poll()  # refresh before reaping; then strictly observe again
                result = send(signal.SIGKILL, "finally", until)
                if result.status in {"REFUSED", "OBSERVED_EMPTY"}:
                    break
                if time.monotonic() >= until:
                    break
                if attempt < 4:
                    time.sleep(min(_POLL_SECONDS, max(0, until - time.monotonic())))
            terminal = owner.refresh(deadline=until, enrol=False)
            if terminal.status != "OBSERVED_EMPTY":
                refused = True
                terminal = owned_process_group.Verdict(
                    "REFUSED", "group not observed empty: " + terminal.reason, terminal.members)
                _refusal_receipt(owner, "residual-census", terminal, stream)
            if not bounded_wait(until):
                refused = True
                result = owned_process_group.Verdict("REFUSED", "leader wait exceeded remaining cleanup allowance")
                _refusal_receipt(owner, "bounded-wait", result, stream)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    if refused and rc == 0:
        rc = EXIT_NOT_STARTED
    _report(stream, f"[run_bounded] {os.path.basename(argv[0])} rc={rc} wall={time.monotonic() - started:.1f}s")
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_bounded.py",
        description="Run a gate command in its own process group and reap the group.",
        # The short description deliberately does not use the whole module doc;
        # callers consulting --help still need the diagnostic that distinguishes
        # our memory stop from another process delivering SIGKILL (same status).
        epilog=(
            "Exit status: child status normally; 124 on timeout; 125 on start or "
            "budget refusal; 137 with MEMORY LIMIT EXCEEDED on a measured memory "
            "breach. Status 137 alone can also mean external SIGKILL; inspect stderr."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "seconds before the whole process group is signalled (default: no "
            "bound). The local typed gate ships 1800 — twice ci.yml's 15-minute "
            "provision, because a whole-tree pyright measures 508-1170 s on a "
            "loaded host here and timing out a slow host is worse than waiting for "
            "it — and `make type-check BOUND_TIMEOUT=<seconds>` raises it for one "
            "run."
        ),
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=None,
        help=(
            "child process-group ceiling in MiB (default: bash tool's resolved budget, "
            "otherwise auto); positive inherited budgets can only be tightened; "
            "0 disables only without a positive inherited budget"
        ),
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE_SECONDS,
        help=(
            "seconds between SIGTERM and SIGKILL for a group that ignores "
            f"SIGTERM (default {DEFAULT_GRACE_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the command to run, after a `--` separator",
    )
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    return run(command, timeout=args.timeout, grace=args.grace, memory_mb=args.memory_mb)


if __name__ == "__main__":
    sys.exit(main())
