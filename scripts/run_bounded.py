#!/usr/bin/env python3
"""Run an explicit cleanup-required (default) or command-status contract.

The census/certification claims below describe cleanup-required ONLY.
Explicit --contract=command-status runs the command as its own process group,
signals that GROUP (deadline, forwarded signal, leader-exit sweep) only while
the leader is still unreaped -- so the group number cannot have been reused --
and ENFORCES the same positive memory grant with the same sampled guard. It
reports the command's status and ALWAYS reports group/descendant cleanup
UNKNOWN: it never certifies cleanup, and a deadline is a signal request, not a
hard wall-time bound. Status-only gates (`make type-check`, `make
check-changed`) use it; cleanup-required refuses before spawn on hosts with no
authoritative census backend (Darwin today).

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
before spawn; `126` when the command ran but its owned group's cleanup could not
be proved — kept apart from 125 precisely so an interrupted cleanup is never
reported as a gate that did not run.
Timeout/RAM/signal/nonzero child statuses are retained with secondary residual
receipts on refusal; a requested signal is never reported as proven reaping.

SAFETY BOUNDARY
----------------
OWNERSHIP IS A RESERVATION, NOT A MEMBER SET. The leader is held UNREAPED, which
keeps its process-group number reserved in the kernel, so every process carrying
that number is ours and a census only describes what the reservation holds. Two
complete strict censuses are unioned per check; each LIVE member is re-verified
against the recorded group and session, and the reservation is re-sampled between
that check and the send (the corpse is read from the census plus the recorded
birth, never through a live session query -- `getsid` fails on a zombie on Darwin,
which is exactly the process the pin keeps). The leader is reaped only AFTER the
terminal census, and a REAPED leader is refused by name: giving up the corpse
releases the number.
User-space census-to-send remains NON-ATOMIC; this is checked owner-alive, not a
kernel-held capability or a hard-owner-SIGKILL immediate-cleanup guarantee.
Unknown/reused/mixed/terminal-only groups refuse with a nonzero outcome and a
durable residual receipt naming births. Escaped/new-session descendants are
outside this group's signal authority.
No keeper, tokenless fallback, or existing monitor-policy changes are introduced.

DIAGNOSTICS go to stderr, never stdout: the lint/format gates' stdout is
parseable output and the suite's `-q` output is parsed too (the same rule the
pytest worker cap follows in `conftest.py`).

RUN RECORD
----------
Every SPAWNED cleanup-required run writes two identity JSON records under
`$LOCAL_OPERATOR_SCRATCHPAD` (else `$LOCAL_OPERATOR_CONFIG_DIR`, else
`~/.local-operator`) + `/run-bounded-runs/`:
`<run_id>.spawn.json` at spawn, before the leader can be reaped, and
`<run_id>.exit.json` before the wrapper returns. Both carry
`schema: run-bounded-run/v1`, `run_id`, `wrapper_pid`, `pgid`, the command's
basename, `wall_clock_start_utc` (and end/elapsed/exit code at exit), `verdict`,
`actors` (each enrolled receipt: pid, scheme, birth, parent, parent_birth, pgid,
sid) and `believed` (what was attempted/observed). No argv, environment or
capability descriptor is written, and stdout stays the child's alone.

A caller binds its evidence with `--run-id TOKEN` (or `$LOCAL_OPERATOR_RUN_ID`);
the token is sanitised to `[A-Za-z0-9._-]` and capped at 64 characters, then
reused for both filenames and the `span run_id=... pgid=...` stderr line emitted
at spawn. An acquisition failure still writes both records — with `actors` empty
and `believed` naming the attempted identities — so "refused with a record" is
distinguishable from "no record", which means no run was spawned. The record
filesystem work is bounded by `_RECORD_BUDGET_SECONDS`, so a wedged home
directory costs that budget rather than the gate.

Those spawn/exit identity records are cleanup-required only. EVERY invocation,
of either contract and including pre-spawn refusals, publishes
``<invocation_id>.completion.json``: the authoritative outcome record, named on
the final stderr line. ``invocation_id`` is a fresh random challenge per
invocation; ``run_id`` is the caller's correlation token and may repeat. Read the
``.completion.json``; the hard-linked ``.completion.pending`` beside it is a
publication artifact (fsync-before-link), byte-identical and never admissible.
Nothing prunes this directory: retention belongs to whoever owns the record
directory (the session scratchpad's lifecycle, else the operator).
Each invocation's ``run-bounded-completion/v1`` record is published even for
unsupported-backend and policy refusals that never spawned a child.
Identity records and integer status are not cleanup certificates; admission
requires this invocation's expected challenge and directory, strict success, and
an accepted authoritative owned-group terminal proof. Descendant cleanup is
always unknown. Completion publication uses fsync and non-overwriting hard-link
publication, but promises neither directory crash durability nor a filesystem
deadline; this is separate from the bounded legacy identity writer above.
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
from local_operator import (  # noqa: E402
    bounded_contract,
    memory_guard,
    owned_process_group,
)

#: Exit status for a fired bound. `timeout(1)` uses the same number.
EXIT_TIMEOUT = 124

#: Keep in sync with .github/workflows/ci.yml's type-check NODE_OPTIONS: both
#: local gate spellings need the analyzer headroom, not just the Makefile recipe.
NODE_HEAP_MB = 6144

#: The command never started (missing interpreter, bad path, refused budget).
#: Distinguished from any status the command itself can produce, so a gate that
#: did not run cannot be read as a passing gate.
EXIT_NOT_STARTED = 125

#: The command RAN and produced its own status, but the owned group could not be
#: proved cleaned up. Deliberately distinct from 125: "did not run" and "ran,
#: residuals unknown" call for different follow-up, and conflating them made an
#: interrupted cleanup look like a gate that never executed (the same conflation
#: turned an already-gone group into a false red).
EXIT_CLEANUP_UNPROVEN = 126

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

#: How often authority is re-proved while waiting for the leader. NOT the poll
#: cadence: one refresh runs two complete strict censuses (Darwin: a syscall;
#: Linux: a full ``/proc`` scan), so refreshing on every 50 ms poll was ~40
#: censuses a second and made ordinary fork churn on Linux read as lost
#: authority. Signals still re-prove inside ``signal()``; this paces enrolment
#: between them.
_AUTHORITY_REFRESH_SECONDS = 1.0

#: Wall-clock ceiling for one run record's filesystem work. Every other step of a
#: refusal is deadline-bounded; a wedged home directory must not be the one
#: unbounded operation on a path that is already refusing.
_RECORD_BUDGET_SECONDS = 0.25

#: Environment fallback for a caller that wants its own run identifier without
#: changing argv.
RUN_ID_ENV = "LOCAL_OPERATOR_RUN_ID"


#: Bound on ONE probe wait inside the run loop. The child's lifetime is not
#: bounded -- a `--timeout`-less run may wait for it forever -- but no single
#: instrument call may block the loop past this.
_NO_NOTIFICATIONS_ENV = "LOCAL_OPERATOR_NO_NOTIFICATIONS"

#: Bound on ONE probe wait inside the run loop.
_PROBE_BUDGET_SECONDS = 0.5
#: Consecutive unanswered probes before the instrument is called BROKEN. At the
#: 50 ms poll cadence this is ~2 s of silence, so a busy sampler is not broken.
_PROBE_STREAK_LIMIT = 40


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


def _utc_now() -> str:
    import datetime

    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _record_dir() -> Path:
    return (
        Path(
            os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")
            or os.environ.get("LOCAL_OPERATOR_CONFIG_DIR")
            or str(Path.home() / ".local-operator")
        )
        / "run-bounded-runs"
    )


def _safe_token(value: object, limit: int = 64) -> str:
    """A filename/phase-safe token: no separators, no traversal, bounded length.

    Returns ``""`` on failure instead of a placeholder: a sentinel such as
    ``"phase"`` would collide with a caller who legitimately chose
    ``--run-id phase``, so each caller decides what an unusable token means.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", str(value))[:limit].strip(".-")


def _resolve_run_id(supplied: str | None) -> str:
    """The caller's ``--run-id``, else ``$LOCAL_OPERATOR_RUN_ID``, else generated."""
    import uuid

    raw = supplied if supplied is not None else os.environ.get(RUN_ID_ENV)
    if raw:
        token = _safe_token(raw)
        if token:
            return token
    return "run-" + uuid.uuid4().hex[:12]


def _verdict_record(verdict):
    """The verdict exactly as it was: four fields, unchanged for existing readers.

    ``Verdict`` gained ``anchor`` and ``refusal_class``; serialising it with
    ``asdict`` would silently change a published object, so the record spells the
    old shape out and the new facts live in ``ownership``.
    """
    if verdict is None:
        return None
    return {
        "status": verdict.status,
        "reason": verdict.reason,
        "members": list(verdict.members),
        "evidence_age_s": verdict.evidence_age_s,
    }


def _ownership_record(owner, verdict):
    """How ownership was established, as a SIBLING object callers opt into.

    Keep the existing identity schema stable for the run-record and incident
    fixtures. These observations do not certify cleanup: proof-dependent callers
    must admit the fresh completion receipt. An earlier comment referred to three
    unnamed downstream lanes; no production readers identifying them were found.

    ``proof`` is the exact set a signal was proven against; ``residuals`` is the
    set that was SEEN and declined, and ``refusal_class`` names why (``pin_lost``,
    ``unproven``, ``excluded``, or null when nothing was refused). All of them are
    carried on every run, including a refusal, because "what did we decline to
    touch, and on what evidence" is the question a later reader actually asks.
    """
    probe = getattr(owner, "probe", {}) or {}
    anchor = getattr(verdict, "anchor", None) or probe.get("anchor")
    refused = verdict is not None and verdict.status == "REFUSED"
    return {
        "schema": "run-bounded-ownership/v1",
        "anchor": anchor,
        "pin": (
            "leader held unreaped: the group number stays reserved, so its "
            "members are ours by construction"
        ),
        "refusal_class": getattr(verdict, "refusal_class", None),
        "proof": probe.get("proof", []),
        "residuals": probe.get("residuals", []),
        # Members the census saw that had already exited by the sample: gone, not
        # declined, and without this the record cannot tell the two apart.
        "vanished": probe.get("vanished", []),
        "cleanup": (
            "residuals named here, never claimed absent"
            if refused
            else "record only; this object claims no cleanup outcome"
        ),
        "excluded": [
            "pid 1",
            "this process",
            "this process group",
            "any group outside the recorded session",
        ],
    }


def _record_run(
    owner,
    stream,
    run_id: str,
    phase: str,
    *,
    command=None,
    verdict=None,
    start_utc=None,
    end_utc=None,
    elapsed=None,
    exit_code=None,
):
    """Durable, bounded evidence binding THIS run to the group it drove.

    WHY A RECORD AND NOT JUST A REFUSAL RECEIPT: a caller that invoked the wrapper
    has no other way to learn which process group it owned. A refused run needs it
    to name residuals; a SUCCESSFUL run needs it for admission — the consuming
    lanes are expected to pass, and a green gate that emits no ownership record
    leaves them nothing to bind. There is therefore a record at spawn and one at
    exit, both keyed by the same caller-supplied run id, both on disk before the
    wrapper returns, and both announced on stderr as they are written.

    Schema ``run-bounded-run/v1``: ``run_id``, ``wrapper_pid``, ``pgid``, the
    command's basename, ``wall_clock_start_utc``/``wall_clock_end_utc``,
    ``elapsed_seconds``, ``exit_code``, ``phase``, ``verdict``, ``actors`` (each
    enrolled receipt: pid, scheme, birth, parent, parent_birth, pgid, sid) and
    ``believed`` (what was attempted/observed even when proof failed). Path
    convention: ``<record dir>/<run_id>.<phase>.json``. No argv, no environment
    and no capability descriptor is written.

    NOTHING HERE MAY RAISE: this path can already be a safety failure, and a second
    capture error must not replace the original one — only warn.

    A4: the record must never be the thing that hangs. The JSON body is serialised
    before any filesystem call, and the deadline is then checked immediately before
    each step that STARTS — mkdir, open, write, flush and fsync. An in-flight
    syscall is not preemptible from user space, so the promise is "no new step
    begins past the budget", which is exactly what this code does and no more.
    """
    import json
    from dataclasses import asdict

    deadline = time.monotonic() + _RECORD_BUDGET_SECONDS
    started = start_utc or _utc_now()
    payload = {
        "schema": "run-bounded-run/v1",
        "run_id": run_id,
        "wrapper_pid": os.getpid(),
        "pgid": getattr(owner, "pgid", None),
        "command": command,
        "phase": phase,
        "wall_clock_start_utc": started,
        "wall_clock_end_utc": end_utc,
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "actors": [asdict(receipt) for receipt in getattr(owner, "receipts", {}).values()],
        "believed": getattr(owner, "probe", {}) or {},
        "verdict": _verdict_record(verdict),
        "ownership": _ownership_record(owner, verdict),
        "cleanup": (
            "REFUSED; residuals named here, not claimed absent"
            if verdict is not None
            else "record only; cleanup outcome not claimed"
        ),
    }
    base = _record_dir()
    try:
        # Serialise BEFORE any filesystem call: after this the write itself is the
        # only step left, and every step that STARTS is checked against the budget.
        body = json.dumps(payload, sort_keys=True)
        if time.monotonic() >= deadline:
            raise TimeoutError("record budget spent before write")
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = base / f"{run_id}.{_safe_token(phase) or 'record'}.json"
        if time.monotonic() >= deadline:
            raise TimeoutError("record budget spent before open")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Same run id, same phase: a retry or two wrappers sharing a caller
            # id. Keep both rather than overwrite one lane's evidence.
            path = base / f"{run_id}.{_safe_token(phase)}.{os.getpid()}.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            if time.monotonic() >= deadline:
                raise TimeoutError("record budget spent before write")
            handle.write(body)
            if time.monotonic() >= deadline:
                raise TimeoutError("record budget spent before flush")
            handle.flush()
            if time.monotonic() < deadline:
                os.fsync(handle.fileno())
        _report(stream, f"[run_bounded] record ({phase}) run_id={run_id} path={path}")
        return path
    except (OSError, TypeError, ValueError) as exc:
        # ValueError/TypeError: a payload that cannot be serialised is a bug, not a
        # reason to take the run's status down; OSError (TimeoutError included, it
        # IS an OSError) is the filesystem. Either way: warn and carry on.
        _report(
            stream,
            f"[run_bounded] record write failed ({phase}): "
            f"{type(exc).__name__} errno={getattr(exc, 'errno', None)}",
        )
        return None


def _reap(owner, grace, stream, *, reason, send, require, deadline):
    """TERM then KILL; the grace wait asks ONE cheap question per tick.

    A grace tick asks only "has the leader gone terminal?" through the pin seam —
    it costs one sample, not the two complete strict censuses the old per-tick
    ``refresh`` ran 40 times a second. Authority is re-proven where it is spent:
    inside ``send`` at the SIGTERM and again at the SIGKILL decision.
    """
    _report(stream, f"[run_bounded] {reason} — requesting checked SIGTERM")
    # ``send`` is ``require(...)``: a refusal raises AuthorityRefused at this call
    # site, so this function never enters its wait with a lost capability. The
    # exception is the enforcement point; a returned-REFUSED branch here read like
    # one and could never run.
    send(signal.SIGTERM, reason + ":term", deadline)
    while time.monotonic() < deadline:
        if owner.leader_terminal(deadline):
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


def _child_environment(budget):
    # Source integration required two independent edits: define this builder,
    # then replace the old inline policy with a call. There is no atomic "move".
    # The integrated-AST and undefined-name regressions enforce one definition
    # shared by strict execution and the dispatcher, without an ordered applier.
    child_env = os.environ.copy()
    # Nested cooperative wrappers inherit the effective (possibly tightened)
    # ceiling, not the original wider grant. env -i must carry it explicitly.
    child_env[memory_guard.RESOLVED_MEMORY_MB_ENV] = str(budget.ceiling_mb)
    # THE GATE TRAVELS WITH THE ENVIRONMENT WE BUILD. This builder hands the child a
    # mapping it assembled rather than the ambient one, so inheriting the switch by
    # accident is not a property it can rely on: the next builder may start from
    # scratch. Same carrier the other builders set (tui/notify.ENV_DISABLE).
    child_env.setdefault(_NO_NOTIFICATIONS_ENV, "1")
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
    return child_env


def _run_cleanup_required(
    argv: Sequence[str],
    *,
    timeout: float | None,
    grace: float,
    memory_mb: int | None = None,
    stream: IO[str] = sys.stderr,
    run_id: str | None = None,
    _budget=None,
    _outcome=None,
) -> int:
    """Run `argv` as a fresh process group and return its status.

    `timeout` is None for an unbounded run: the group is still reaped on exit,
    because the orphan in this module's docstring is produced by an ordinary exit
    as well as by a bound (the wrapper exits, the node child does not).

    `run_id` binds the returned evidence to this invocation: it is sanitised into
    a filename token, recorded at spawn and again at exit under
    ``run-bounded-runs/<run_id>.<phase>.json``, and echoed on stderr. Callers that
    pass none get ``$LOCAL_OPERATOR_RUN_ID`` or a generated id.
    """
    if not argv:
        stream.write("[run_bounded] no command was given\n")
        return EXIT_NOT_STARTED
    try:
        budget, budget_source = _memory_budget(memory_mb) if _budget is None else _budget
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

    child_env = _child_environment(budget)

    previous = {sig: signal.signal(sig, _forward) for sig in FORWARDED_SIGNALS}
    try:
        proc = subprocess.Popen(list(argv), start_new_session=True, env=child_env)
    except OSError as exc:
        # A gate that never ran must never look green (#423's lesson).
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        stream.write(f"[run_bounded] could not start {argv[0]!r}: {exc}\n")
        if _outcome is not None:
            _outcome.termination_cause = "spawn-failure"
        return EXIT_NOT_STARTED
    pgid = proc.pid
    # ONE REPRESENTATION for the horizon, normalised here where it enters: the
    # unbounded case is `inf`, never `None`. Every downstream reader -- the loop's
    # `now >= deadline`, the probe windows, and `leader_terminal(deadline)` -- then
    # compares floats, and a `--timeout`-less run cannot reach `float >= None`.
    deadline = float("inf") if timeout is None else started + timeout
    run_id = _resolve_run_id(run_id)
    run_started_utc = _utc_now()
    # The caller-readable binding, emitted the moment the group exists: an
    # invocation that never returns still leaves this id/pgid pairing in its log.
    _report(
        stream,
        f"[run_bounded] span run_id={run_id} wrapper_pid={os.getpid()} "
        f"pgid={pgid} utc={run_started_utc} command={os.path.basename(argv[0])}",
    )
    # Provisional numeric identity grants NO signal authority. Capture before ANY
    # poll/wait can reap this leader; failed capture remains a bounded refusal.
    owner = owned_process_group.OwnedGroup.capture(
        proc.pid,
        deadline=min(deadline, time.monotonic() + 3.0),
    )
    # Written even when capture failed: the acquisition-failure run is the one
    # whose evidence matters most, and a record carrying only the believed
    # identity is still distinguishable from "no record at all".
    _record_run(
        owner, stream, run_id, "spawn", command=os.path.basename(argv[0]), start_utc=run_started_utc
    )
    refused = False
    proof_failed = False
    cleanup_deadline = None
    # Whether the owner accepted THIS run's exact terminal verdict before
    # revocation -- the one certificate strict cleanup admits (see below).
    certified = False

    class AuthorityRefused(Exception):
        """Local control transfer: refusal must not leave timeout=None polling."""

    def fail_proof(result, phase):
        nonlocal refused, proof_failed
        refused = proof_failed = True
        _record_run(
            owner,
            stream,
            run_id,
            phase,
            command=os.path.basename(argv[0]),
            verdict=result,
            start_utc=run_started_utc,
        )

    def require(result, phase):
        if result.status == "REFUSED":
            fail_proof(result, phase)
            raise AuthorityRefused
        return result

    def begin_cleanup(allowance):
        nonlocal cleanup_deadline
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + allowance
        return cleanup_deadline

    def send(signum, phase, until):
        return require(_signal_group(owner, signum, deadline=until), phase)

    def bounded_wait(until):
        # All consuming waits revoke FIRST, including failed/timed-out attempts.
        # The accepted terminal proof, if any, was captured before this boundary.
        owner.revoke()
        try:
            proc.wait(timeout=0 if refused else max(0.0, until - time.monotonic()))
        except subprocess.TimeoutExpired:
            # Waiting is not cleanup proof. No PID-only or numeric-PG fallback.
            return False
        return True

    # The command is spawned: from here every failure names residuals, never
    # "did not run". A forwarded signal still carries its own status.
    # NO DEFAULT TERMINAL CODE. This used to start at EXIT_CLEANUP_UNPROVEN, so any
    # exit that did not reach its own assignment published 126 -- a refusal code
    # overwriting the caller's actual outcome (timeout 124, memory 137, a monitor
    # error's raise). The code is now decided by the path that produced it.
    rc = 128 + stop["signum"] if stop["signum"] else None
    if _outcome is not None and stop["signum"]:
        _outcome.termination_cause = "interruption"
        _outcome.wrapper_exit_code = rc
    exited = False
    unanswered = 0
    try:
        require(owner.last, "acquire")
        guard = memory_guard.Guard(pgid, budget) if budget.ceiling_mb > 0 else None
        next_sample = started
        next_probe = started + _AUTHORITY_REFRESH_SECONDS
        unmeasured_reported = False
        while True:
            now = time.monotonic()
            signum = stop["signum"]
            timed_out = now >= deadline
            if signum:
                rc = 128 + signum
                if _outcome is not None:
                    _outcome.termination_cause = "interruption"
                    _outcome.wrapper_exit_code = rc
            elif timed_out:
                rc = EXIT_TIMEOUT
                if _outcome is not None:
                    _outcome.termination_cause = "deadline"
                    _outcome.wrapper_exit_code = rc
            # Enrolment needs a still-live creation chain, and any signal re-proves
            # inside `signal()`. Between signals the census runs on a bounded
            # cadence rather than once per poll: see _AUTHORITY_REFRESH_SECONDS.
            if signum or timed_out or now >= next_probe:
                probe_until = (
                    begin_cleanup(grace + 0.25) if signum or timed_out else min(deadline, now + 3.0)
                )
                require(owner.refresh(deadline=probe_until), "required-refresh")
                next_probe = now + _AUTHORITY_REFRESH_SECONDS
            if signum:
                until = begin_cleanup(grace + 0.25)
                send(signum, "forward", until)
                _reap(
                    owner,
                    grace,
                    stream,
                    reason=f"signal {signum} received",
                    send=send,
                    require=require,
                    deadline=until - 0.25,
                )
                rc = 128 + signum
                break
            if timed_out:
                until = begin_cleanup(grace + 0.25)
                _reap(
                    owner,
                    grace,
                    stream,
                    reason=f"timeout after {timeout:g}s",
                    send=send,
                    require=require,
                    deadline=until - 0.25,
                )
                _report(
                    stream,
                    f"[run_bounded] BOUND ({timeout:g}s) fired, not the gate failing; "
                    "--timeout sets the bound",
                )
                break
            # EACH ITERATION BOUNDS ITS OWN WAIT, and a probe that cannot ANSWER
            # for a sustained streak refuses BY NAME instead of spinning. An
            # unbounded wait for a live CHILD is the user's contract and stays;
            # what must never happen is an unbounded wait on a broken instrument,
            # which is how this loop became unreachable by any signal boundary
            # (precedent: the probe-streak rule in session/runtime/process.py).
            probe_until = min(deadline, time.monotonic() + _PROBE_BUDGET_SECONDS)
            try:
                terminal_now = owner.leader_terminal(probe_until)
            except Exception:
                unanswered += 1
                if unanswered >= _PROBE_STREAK_LIMIT:
                    require(
                        owned_process_group.Verdict(
                            "REFUSED",
                            "owned-group probe unanswered for %d consecutive attempts" % unanswered,
                        ),
                        "probe-streak",
                    )
                terminal_now = False
            else:
                unanswered = 0
            if terminal_now:
                # TERMINAL, NOT REAPED. The leader is deliberately left a corpse:
                # reaping it frees the group number and destroys the reservation
                # the cleanup below depends on. The status is read after the
                # terminal census, once cleanup has decided to give the pin up.
                exited = True
                break
            now = time.monotonic()
            if guard is not None and now >= next_sample:
                sample = guard.sample_sync()
                next_sample = now + guard.tick_s
                if guard.should_kill(sample):
                    _report(stream, f"[run_bounded] {guard.over_budget_message(sample)}")
                    rc = EXIT_MEMORY
                    if _outcome is not None:
                        _outcome.termination_cause = "memory-policy"
                        _outcome.wrapper_exit_code = rc
                    send(signal.SIGKILL, "memory", begin_cleanup(0.25))
                    break
                notice = guard.soft_notice(sample)
                if notice:
                    _report(stream, f"[run_bounded] {notice} (advisory only)")
                if sample.bytes_used is None and not unmeasured_reported:
                    _report(
                        stream, "[run_bounded] memory charge unavailable; no measured cap proof"
                    )
                    unmeasured_reported = True
            time.sleep(_POLL_SECONDS)
    except AuthorityRefused:
        # The existing error/timeout/RAM/signal status is retained. In particular
        # timeout=None cannot leave this wrapper polling an unowned live child.
        pass
    except Exception as exc:
        _report(stream, f"[run_bounded] owned-child monitor failed: {type(exc).__name__}")
        # This run reached a group and then failed: it must not vanish from the
        # record set just because it leaves through an exception.
        _record_run(
            owner,
            stream,
            run_id,
            "monitor-error",
            command=os.path.basename(argv[0]),
            verdict=owner.last,
            start_utc=run_started_utc,
        )
        raise
    finally:
        try:
            # Retain the existing quarter-second final allowance, not a new
            # grace period per pass. Never claim a requested signal means reaped.
            until = begin_cleanup(0.25)
            for attempt in range(5):
                if proof_failed:
                    break  # no automatic signal retry after loss of authority
                check = owner.refresh(deadline=until)
                if check.status == "REFUSED":
                    fail_proof(check, "finally-refresh")
                    break
                try:
                    result = send(signal.SIGKILL, "finally", until)
                except AuthorityRefused:
                    break
                if result.status in {"REFUSED", "OBSERVED_EMPTY"}:
                    break
                if time.monotonic() >= until:
                    break
                if attempt < 4:
                    time.sleep(min(_POLL_SECONDS, max(0, until - time.monotonic())))
            # TERMINAL ONLY: the leader is legitimately gone once we have killed
            # the group, so the end may be reported from the census. `signal()`
            # never passes this flag -- the send path stays witness-first.
            # DERIVED FROM THE OWNER, not a runner-side flag: `terminating_sends`
            # records only the signals this code's ladder uses to END a group, so
            # there is one source of truth and the call site cannot claim a kill
            # that never happened (round 9, R2).
            terminal = owner.refresh(deadline=until, terminal=bool(owner.terminating_sends))
            certified = bool(owner.accepted_terminal(terminal))
            if _outcome is not None:
                _outcome.accept_terminal(owner, terminal)
            if terminal.status != "OBSERVED_EMPTY":
                refused = True
                # KEEP THE CLASS. Dropping it made the record say only "not observed
                # empty", which is how three separate causes (lost pin, unreadable
                # census, live member) all looked identical in CI.
                terminal = owned_process_group.Verdict(
                    "REFUSED",
                    "group not observed empty: " + terminal.reason,
                    terminal.members,
                    refusal_class=terminal.refusal_class,
                )
                _record_run(
                    owner,
                    stream,
                    run_id,
                    "residual-census",
                    command=os.path.basename(argv[0]),
                    verdict=terminal,
                    start_utc=run_started_utc,
                )
            waited = bounded_wait(until)
            if _outcome is not None and waited and proc.returncode is not None:
                _outcome.command_exit_code = bounded_contract.shell_status(proc.returncode)
            # Wrapper status is selected after handler restoration below.
            if not waited:
                refused = True
                # No reaped status means no child result: a certificate about the
                # group cannot turn an unknown command status into a passing gate.
                certified = False
                result = owned_process_group.Verdict(
                    "REFUSED", "leader wait exceeded remaining cleanup allowance"
                )
                _record_run(
                    owner,
                    stream,
                    run_id,
                    "bounded-wait",
                    command=os.path.basename(argv[0]),
                    verdict=result,
                    start_utc=run_started_utc,
                )
        except Exception as exc:
            if _outcome is not None:
                _outcome.finalization_failed("strict-finalization", exc)
            raise
        finally:
            try:
                for sig, handler in previous.items():
                    try:
                        signal.signal(sig, handler)
                    except Exception as exc:
                        if _outcome is None:
                            raise
                        _outcome.finalization_failed("handler-restoration", exc)
            finally:
                # Export the final latch before ANY exception crosses execute's
                # separate stop dictionary. Normal-return code cannot do this.
                if stop["signum"]:
                    rc = 128 + stop["signum"]
                    if _outcome is not None:
                        _outcome.termination_cause = "interruption"
                        _outcome.wrapper_exit_code = rc
                elif _outcome is not None and _outcome.finalization_errors:
                    rc = _outcome.wrapper_exit_code

    # The inner finally exported interruption even on exceptional exits. Only
    # an otherwise unselected normal result may now adopt the child's status.
    if rc is None and exited and waited and proc.returncode is not None:
        rc = bounded_contract.shell_status(proc.returncode)
        if _outcome is not None:
            _outcome.termination_cause = "child-signal" if proc.returncode < 0 else "child-exit"
            _outcome.wrapper_exit_code = rc
    if rc is None:
        # Nothing determined a terminal code: either the child ended on its own (0)
        # or the path could not say (cleanup unproven).
        rc = 0 if exited else EXIT_CLEANUP_UNPROVEN
    if refused and rc == 0 and not certified:
        # The gate ran and passed; what is missing is proof of cleanup. Say that,
        # rather than reusing 125, which callers read as "the gate did not run".
        # An ACCEPTED terminal certificate is that proof: a transient refusal of
        # an earlier, non-terminal refresh (Linux fork churn during the final
        # sweep) must not override a census that then proved the group empty
        # under a retained witness. Measured on CI run 41535590707: records
        # `finally-refresh` (refused) then `cleanup=proven-empty`, rc 126. A
        # failed bounded wait clears `certified` (no child status), and
        # bounded_contract.execute independently demotes rc 0 without proof.
        rc = EXIT_CLEANUP_UNPROVEN
    if _outcome is not None:
        _outcome.wrapper_exit_code = rc
    # The record a SUCCESSFUL run must leave behind: callers bind admission to it,
    # so it is written before the wrapper returns and after the leader is gone.
    _record_run(
        owner,
        stream,
        run_id,
        "exit",
        command=os.path.basename(argv[0]),
        verdict=owner.last,
        start_utc=run_started_utc,
        end_utc=_utc_now(),
        elapsed=time.monotonic() - started,
        exit_code=rc,
    )
    _report(
        stream,
        f"[run_bounded] {os.path.basename(argv[0])} rc={rc} wall={time.monotonic() - started:.1f}s",
    )
    return rc


def run(
    argv,
    *,
    timeout,
    grace,
    memory_mb=None,
    stream=sys.stderr,
    run_id=None,
    contract="cleanup-required",
    _backend=None,
):
    # Resolve the inherited grant BEFORE contract dispatch. Neither contract can
    # disable or widen a positive grant, and both ENFORCE the resolved ceiling.
    receipt = bounded_contract.Receipt(_record_dir(), _resolve_run_id(run_id))
    try:
        resolved = _memory_budget(memory_mb)
    except ValueError as exc:
        # The structured receipt supplements the old actionable refusal; hiding
        # the grant/override explanation made a correctly refused gate opaque.
        # The refusal line names the reason and the next step (design D1); the
        # status line then carries invocation_id/completion like every other
        # path, so the caller can locate the receipt proving the refusal (R8).
        _report(stream, f"[run_bounded] refused before spawn: memory budget refused: {exc}")
        outcome = bounded_contract.Outcome(
            wrapper_exit_code=125,
            termination_cause="memory-policy-refused",
            cleanup_reason=str(exc),
        )
        try:
            receipt.publish(contract, outcome)
        except (OSError, ValueError, TypeError) as publication_error:
            _report(
                stream,
                f"[run_bounded] completion publication failed: {type(publication_error).__name__}",
            )
        _report(stream, bounded_contract.status_line(contract, outcome, receipt))
        return 125
    budget, budget_source = resolved

    def strict(outcome):
        outcome.wrapper_exit_code = _run_cleanup_required(
            argv,
            timeout=timeout,
            grace=grace,
            memory_mb=memory_mb,
            stream=stream,
            run_id=receipt.run_id,
            _budget=resolved,
            _outcome=outcome,
        )

    return bounded_contract.execute(
        argv,
        contract=contract,
        timeout=timeout,
        grace=grace,
        budget=budget,
        child_env=_child_environment(budget),
        receipt=receipt,
        stream=stream,
        strict=strict,
        backend=_backend,
        budget_source=budget_source,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_bounded.py",
        description=(
            "cleanup-required (default): command success plus certified owned-group"
            " cleanup (refuses before spawn where no authoritative backend exists,"
            " e.g. Darwin). command-status: the command's own status; enforces the"
            " memory grant and signals the owned group, but NEVER certifies cleanup."
        ),
        # The short description deliberately does not use the whole module doc;
        # callers consulting --help still need the diagnostic that distinguishes
        # our memory stop from another process delivering SIGKILL (same status).
        epilog=(
            "Exit status: child status normally; 124 on timeout; 125 on start or "
            "budget refusal (also backend/policy refusals before spawn: the "
            "command never ran); 126 when the command ran but its owned group's "
            "cleanup could not be proved under cleanup-required, or command-status"
            " supervision failed (command-status child success remains 0 with "
            "cleanup UNKNOWN); 137 with MEMORY LIMIT "
            "EXCEEDED on a measured "
            "memory breach. Status 137 alone can also mean external SIGKILL; "
            "inspect stderr. Every invocation (either contract, refusals "
            "included) writes the authoritative record "
            "run-bounded-runs/<invocation_id>.completion.json, named on the final "
            "stderr line; invocation_id is fresh per call, unlike the reusable "
            "--run-id. Read .completion.json; the hard-linked .completion.pending "
            "is a publication artifact. Spawned cleanup-required runs also write "
            "legacy <run_id>.spawn.json/.exit.json identity records. Nothing "
            "prunes run-bounded-runs/: its directory owner (the session "
            "scratchpad, else you) retains or removes it."
        ),
    )
    parser.add_argument(
        "--contract",
        choices=bounded_contract.CONTRACTS,
        default="cleanup-required",
        help=(
            "cleanup-required (default) needs certified group cleanup; "
            "command-status reports the command's status, enforces the same "
            "memory grant on its own process group and signals that group only "
            "while its leader is unreaped; cleanup is always UNKNOWN (never "
            "certified) and the deadline is not a hard wall-time bound"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "seconds before requesting a stop of the owned group (SIGTERM, then "
            "SIGKILL after --grace; not a hard wall-time promise; default: no "
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
            "sampled owned-group ceiling in MiB, enforced by both contracts "
            "(default: bash tool's resolved budget, otherwise auto); positive "
            "inherited budgets can only be tightened; 0 disables only without a "
            "positive inherited budget. The first stderr line names the "
            "effective ceiling and its source"
        ),
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE_SECONDS,
        help=(
            "seconds between SIGTERM and SIGKILL requests to the owned "
            "group when it ignores "
            f"SIGTERM (default {DEFAULT_GRACE_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "caller-supplied identifier recorded with this run's owned-group "
            "evidence (filename token and stderr line; default "
            f"${RUN_ID_ENV}, else generated). Sanitised to [A-Za-z0-9._-], max 64."
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
    return run(
        command,
        timeout=args.timeout,
        grace=args.grace,
        memory_mb=args.memory_mb,
        run_id=args.run_id,
        contract=args.contract,
    )


if __name__ == "__main__":
    sys.exit(main())
