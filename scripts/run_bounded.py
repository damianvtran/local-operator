#!/usr/bin/env python3
"""Run a gate command in its OWN process group and reap the whole group.

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
it leads a fresh process group its children inherit) and signals the GROUP on
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
    python scripts/run_bounded.py -- <command> [args…]      # no bound, still reaped

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
before spawn — never 0 for a gate that did not run.

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
from local_operator import memory_guard  # noqa: E402

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


def _signal_group(pgid: int, signum: int) -> bool:
    """Signal every process in `pgid`. False when the group is already gone."""
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _reap(
    pgid: int,
    grace: float,
    poll,
    stream: IO[str],
    *,
    reason: str,
) -> bool:
    """SIGTERM the group, escalate to SIGKILL, and report what happened.

    Returns True when the escalation was needed, which is the fact worth
    printing: a group that ignores SIGTERM is exactly the case an unbounded
    orphan comes from.
    """
    stream.write(f"[run_bounded] {reason} — SIGTERM to process group {pgid}\n")
    stream.flush()
    _signal_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while poll() is None and time.monotonic() < deadline:
        time.sleep(_POLL_SECONDS)
    if poll() is not None:
        return False
    stream.write(f"[run_bounded] group {pgid} outlived SIGTERM by {grace:g}s — SIGKILL\n")
    stream.flush()
    _signal_group(pgid, signal.SIGKILL)
    return True


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
    state: dict[str, int] = {}
    stop: dict[str, int] = {"signum": 0}

    def _forward(signum: int, _frame: object) -> None:
        stop["signum"] = signum
        pgid = state.get("pgid")
        if pgid:
            _signal_group(pgid, signum)

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
    # The group id IS the child's pid: `start_new_session=True` makes the child a
    # session leader, so its process group is its own. Asking the kernel
    # (`os.getpgid`) instead RACES a command that exits quickly — measured: the
    # simplest gate, `python -c "print(...)"`, raised `ProcessLookupError` from
    # `getpgid` on most runs, because the pid was already gone and a session
    # leader's group dissolves with it. That turned a passing gate into `rc=1`,
    # which is the worst thing a wrapper like this can do.
    pgid = state["pgid"] = proc.pid

    rc: int
    try:
        guard = memory_guard.Guard(pgid, budget) if budget.ceiling_mb > 0 else None
        next_sample = started
        unmeasured_reported = False
        deadline = None if timeout is None else started + timeout
        while True:
            status = proc.poll()
            signum = stop["signum"]
            if status is not None:
                # A forwarded signal ends the child as well, and the child's
                # status then arrives as `-signum` (Python renders that as 241).
                # The wrapper reports the SIGNAL instead, the way a shell does, so
                # a caller reading `128 + signum` sees the same code either way,
                # and it says so — a gate that was signalled must not look like a
                # gate that failed on its own.
                if signum:
                    rc = 128 + signum
                    note = f"signal {signum} received — process group {pgid} signalled"
                elif status < 0:
                    # Nobody here sent this signal (a `kill -9` by hand, the OOM
                    # killer, a job runner). `sys.exit(-9)` is 247 and the
                    # diagnostic line would read `rc=-9`: neither is in the EXIT
                    # STATUS contract, and both are unreadable in a log. A shell
                    # reports 128 + the signal for a child killed this way, so
                    # this does too.
                    rc = 128 - status
                    note = f"child killed by signal {-status}"
                else:
                    rc = status
                    note = None
                if note:
                    stream.write(f"[run_bounded] {note} (rc={rc})\n")
                    stream.flush()
                break
            if signum:
                _reap(pgid, grace, proc.poll, stream, reason=f"signal {signum} received")
                proc.wait()
                rc = 128 + signum
                break
            if deadline is not None and time.monotonic() >= deadline:
                _reap(pgid, grace, proc.poll, stream, reason=f"timeout after {timeout:g}s")
                # Say what this is and what to do about it, on THIS path too: the
                # bound is a per-host judgement, a loaded host is the one that
                # legitimately reaches it, and `make type-check` prints nothing
                # else — a developer there would otherwise learn what happened but
                # not what to do (review round 2, M3 / QA Q4).
                stream.write(
                    f"[run_bounded] that is the BOUND ({timeout:g}s) firing, not the gate "
                    "failing — re-run it, or raise it (--timeout, or "
                    "`make type-check BOUND_TIMEOUT=<seconds>`)\n"
                )
                stream.flush()
                proc.wait()
                rc = EXIT_TIMEOUT
                break
            now = time.monotonic()
            if guard is not None and now >= next_sample:
                sample = guard.sample_sync()
                next_sample = now + guard.tick_s
                if guard.should_kill(sample):
                    stream.write(f"[run_bounded] {guard.over_budget_message(sample)}\n")
                    stream.flush()
                    _signal_group(pgid, signal.SIGKILL)
                    proc.wait()
                    rc = EXIT_MEMORY
                    break
                notice = guard.soft_notice(sample)
                if notice:
                    stream.write(f"[run_bounded] {notice} (advisory only)\n")
                    stream.flush()
                if sample.bytes_used is None and not unmeasured_reported:
                    stream.write("[run_bounded] memory charge unavailable; no measured cap proof\n")
                    stream.flush()
                    unmeasured_reported = True
            time.sleep(_POLL_SECONDS)
    except Exception as exc:
        stream.write(f"[run_bounded] owned-child monitor failed: {type(exc).__name__}: {exc}\n")
        stream.flush()
        raise
    finally:
        try:
            # The ordinary-exit half of the defect: the leader is gone, and anything
            # it spawned is still holding its heap. SIGKILL, because there is nothing
            # left to negotiate with — the gate has already finished.
            #
            # Several passes, because a child forked in the instant the leader exited
            # can land after the kernel resolved the previous one. Bounded and cheap:
            # the whole sweep is a quarter of a second, against a gate that has
            # already finished.
            reaped = False
            for _ in range(5):
                reaped = _signal_group(pgid, signal.SIGKILL) or reaped
                time.sleep(_POLL_SECONDS)
            if reaped:
                stream.write(
                    f"[run_bounded] reaped processes left behind in group {pgid} "
                    "after the gate exited\n"
                )
                stream.flush()

        finally:
            try:
                # Also reap the leader when an unexpected monitor error, not
                # normal poll/timeout, brought us here. Keep forwarding signals
                # until cleanup finishes rather than reopening an orphan window.
                proc.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

    stream.write(
        f"[run_bounded] {os.path.basename(argv[0])} rc={rc} "
        f"wall={time.monotonic() - started:.1f}s\n"
    )
    stream.flush()
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
