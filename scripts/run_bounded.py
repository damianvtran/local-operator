#!/usr/bin/env python3
"""Run a gate command in its OWN process group and reap the whole group.

WHY THIS EXISTS
---------------
`pyright` (the pip wrapper) runs the real analyzer as an npm/node child. The
local gate used to be spelled `timeout 900 .venv/bin/python -m pyright …`, and
`timeout(1)` signals only the process it started — the Python wrapper. Nothing
forwards that signal to the node child, so when the bound fires the gate is
over but the analyzer is not:

    $ timeout 8 .venv/bin/python -m pyright --pythonpath .venv/bin/python .
    $ ps -o pid,ppid,pgid,rss,etime,command -p <node pid>
      pid   ppid  pgid   rss  etime  command
    92099      1 92098  2.28G  03:41  node …/pyright/index.js --outputjson …

`ppid 1` is the whole defect: the orphan is re-parented to launchd and keeps its
heap for as long as its work takes — measured at 2.28 GB and 1.50 GB on one
host, ten analyzers alive at once, 5.1 GB total. Sessions queue more pyrights
behind a host those orphans are already saturating, more of them hit the bound,
and the leak compounds. A bound that leaves a gigabyte running is not a bound.

The fix is NOT a longer timeout. It is to put the command in its own session
(`start_new_session=True`, so it leads a fresh process group that its children
inherit) and to signal the GROUP on every exit path: the timeout, a signal to
this wrapper, and the ordinary exit where the leader is gone but its
descendants are not.

`timeout(1)` cannot do this portably — it has no `--kill-group`, and `setsid`
does not exist on macOS, which is the platform this bites hardest.

USAGE
-----
    python scripts/run_bounded.py --timeout 900 -- <command> [args…]
    python scripts/run_bounded.py -- <command> [args…]      # no bound, still reaped

Run it through the interpreter (`.venv/bin/python scripts/run_bounded.py …`),
never as a console script — the #423 shebang rule applies to this file too.

EXIT STATUS
-----------
The command's own status; `124` when the bound fired (what `timeout(1)` reports,
so existing callers keep reading the same code); `128 + signum` when a forwarded
signal ended it; `125` when the command could not be started at all — never 0
for a gate that did not run.

DIAGNOSTICS go to stderr, never stdout: the lint/format gates' stdout is
parseable output and the suite's `-q` output is parsed too (the same rule the
pytest worker cap follows in `conftest.py`).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import IO, Sequence

#: Exit status for a fired bound. `timeout(1)` uses the same number.
EXIT_TIMEOUT = 124

#: The command never started (missing interpreter, bad path). Distinguished from
#: any status the command itself can produce, so a gate that did not run cannot
#: be read as a passing gate.
EXIT_NOT_STARTED = 125

#: How long a signalled process group gets before SIGKILL. The analyzer needs a
#: moment to unwind; a group that ignores SIGTERM entirely must still not outlive
#: the gate, which is the failure this module exists to remove.
DEFAULT_GRACE_SECONDS = 10.0

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


def run(
    argv: Sequence[str],
    *,
    timeout: float | None,
    grace: float,
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

    previous = {sig: signal.signal(sig, _forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        proc = subprocess.Popen(list(argv), start_new_session=True)
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
                    stream.write(
                        f"[run_bounded] signal {signum} received — process group "
                        f"{pgid} signalled (rc={128 + signum})\n"
                    )
                    stream.flush()
                rc = 128 + signum if signum else status
                break
            if signum:
                _reap(pgid, grace, proc.poll, stream, reason=f"signal {signum} received")
                proc.wait()
                rc = 128 + signum
                break
            if deadline is not None and time.monotonic() >= deadline:
                _reap(pgid, grace, proc.poll, stream, reason=f"timeout after {timeout:g}s")
                proc.wait()
                rc = EXIT_TIMEOUT
                break
            time.sleep(_POLL_SECONDS)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
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
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "seconds before the whole process group is signalled (default: no "
            "bound). CI's own bound for this gate is 15 minutes, so 900 mirrors it."
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
    return run(command, timeout=args.timeout, grace=args.grace)


if __name__ == "__main__":
    sys.exit(main())
