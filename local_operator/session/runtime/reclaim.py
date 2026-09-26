"""The external residency sweep: end the session runtimes nothing can reach.

**WHY THIS EXISTS, in two measurements.** On 2026-09-17 this machine ran 57-61
session runtimes. **34 of them had no discovery record in the operator's own
store** (``~/.local-operator/run/mobile`` — what ``lop sessions``, ``lop send``,
attach and the desktop app all read), the oldest had been alive 11.3 h, and
between them they held ~810 MB of RSS that no surface in the product could name.
The second measurement is the one that changes what to do about them: asking each
process for the config root it ACTUALLY uses — its own environment, ``ps -Eww`` on
macOS and ``/proc/<pid>/environ`` on Linux (:func:`pid_environment`) — found every
one holding a heartbeat-fresh record in its OWN root — sibling QA stores under
``/tmp`` and
``/private/tmp``, 32 of the 36 publishing ``busy: true``. So the population is not
a set of corpses: it is runtimes of other stores that this store cannot see, and a
sweep scoped to one store has no business ending any of them. Both facts are why
this module exists and why its refusals are as wide as they are: the census is
what makes such a runtime NAMABLE (`roster`), and the refusal ladder is what keeps
the sweep from ending a runtime another store is still using.

**THE ONE PROCESS IT CAN END IS THE ONE NOTHING CAN REACH.** A candidate is a
running process whose command line is the session runtime's own
(``-m local_operator.session.runtime.process``), whose own config root — read from
its environment, not assumed — is either the root being swept or a path that NO
LONGER EXISTS. Everything else is refused: a runtime of a live sibling root is its
own root's business (that root's supervisor sees it and can sweep it), and a
machine-wide sweep that ended it would be reaching across a boundary it cannot see
the far side of. A root that is gone is the one case with no such far side at all:
no client, viewer, wake or supervisor of that root can exist, because every one of
them resolves through a path that is gone. (A runtime whose store is merely
deleted often recreates it — ``registry.publish`` makes the directories it needs —
so this class is the store that CANNOT be written: a gone mount, a read-only
volume, a replaced path. Measured: a plain deletion holds for under 15 s.)

**HOW THE SWEEP KNOWS A CANDIDATE IS IDLE — it does not, and it does not have to.**
The sweep never forces an exit. It sends **SIGTERM**, whose handler in the runtime
is work-aware by construction (``process.amain``'s ``_on_signal``): with nothing in
flight it leaves immediately, and with a turn, a compaction, a subagent or a
parked gate in flight it COMMITS to leaving and finishes that work first, bounded
by ``types.SIGNAL_DRAIN_S`` (120 s). So a runtime that is mid-turn is not cut off
by a sweep — it finishes its turn and then leaves, which is exactly the behaviour
``lop stop``'s second rung already relies on. SIGKILL is deliberately absent from
this module and no escalation rung exists: an unrefusable signal is the one thing
that could destroy a turn, and nothing here needs it.

What the sweep DOES establish, for every candidate, is that nothing outside can be
depending on that runtime right now:

* **No record ⇒ no attach, no send, no engage.** A runtime publishes its discovery
  record before it listens and keeps it while it works; attach, `lop send`, the
  desktop roster and the wake supervisor's own engage path all resolve a runtime
  through that record. A record-less runtime cannot be reached, and cannot be given
  new work by anything.
* **The record's absence is CONFIRMED over a window** (:data:`CONFIRM_S`), from two
  independent sightings of the process table, and any record that appears in
  between drops the candidate. That is what makes the sweep unable to race an
  attach that has not happened yet: an attach is possible only through a record,
  and a record that exists at confirm time is a refusal (``record-present``).
* **Nothing is attached to it either.** A live viewer lease naming its session, or
  an ESTABLISHED connection to its control port (from the machine's own socket
  table, when ``lsof`` is available), is a refusal: something is looking at it.
* **It is not merely still booting.** A candidate younger than
  :data:`MIN_AGE_S` is refused, which covers the whole construction window of a
  spawn whose record is not published yet (measured at 2.1 s warm, and the engage
  deadline the supervisor sizes for is 180 s). The age is the YOUNGER of the
  process's own and its boot record's (:func:`effective_age`), because an adopted
  runtime's interpreter is started minutes before the session it constructs.
* **It is not burning CPU.** Cumulative CPU is compared across the two sightings
  and a candidate that spent more than :data:`BUSY_CPU_FRACTION` of the window
  burning is refused for another window. This test can only ever REFUSE — a quiet
  runtime may still be waiting on a model API call, so low CPU proves nothing —
  and it exists so the one candidate class the sweep ends is never a runtime that
  is visibly doing work. It needs a window long enough to mean anything, which is
  :data:`MIN_ACTIONABLE_CONFIRM_S`: below it the floor dominates the fraction and
  the rung cannot fire whatever the runtime does, so a caller may not ask for one.
* **It is still the process the pass decided about.** The census and the fleet are
  read once at the top of a pass and the signal goes out later — measured on this
  host, 19 ms to 1161 ms later across a real 74-row census — so the candidate's own
  row is re-read immediately before ``os.kill`` and the signal is withheld if the
  pid was recycled, changed its argv, moved to another root or published a record
  in the interval (:func:`target_changed`).

**PARENTAGE IS NOT EVIDENCE, IN EITHER DIRECTION.** A session runtime is spawned
detached (``start_new_session=True``), so ``ppid=1`` is the NORMAL state of a
healthy runtime whose spawner has moved on, and it is equally the state of the
orphans this module exists for. A dead parent therefore says nothing about whether
a runtime is wanted, and nothing here keys on it.

**WHAT IT WILL NOT DO.** It never signals this process or any of its ancestors
(a sweep started from inside a session must not end that session). It never
touches a record — reading is all it does to ``run/mobile`` and ``run/host``
(``reap=False`` throughout), because a sweep that reaped evidence while deciding
would destroy the answer to "why did this die". And it never ends a runtime it
cannot attribute to a root.

**AND IT DOES NOTHING WHEN ITS EVIDENCE IS INCOMPLETE.** If the machine's socket
table cannot be read — no ``lsof``, or it timed out — then the refusal that lives
only there (a client attached to a record-less runtime's control port) cannot be
evaluated, and the pass refuses the candidates it would otherwise admit
(:data:`REFUSAL_SOCKETS_UNKNOWN`) rather than signalling through the gap. The
consequence is that the sweep reclaims NOTHING on such a machine, which is the
posture this module's own test file states: a sweep that ends one runtime it
should not have is worse than a sweep that reclaims nothing. The report carries
``sockets_available=False``, the CLI prints it and the roster's ``socket_table``
says so, so the reason is never a mystery.

The module is stdlib-only and off the model-facing path by the same contract as
``registry`` and ``viewers``: the wake supervisor imports it, and the supervisor's
whole justification is staying cheap.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from local_operator.paths import CONFIG_DIR_ENV, DEFAULT_CONFIG_DIRNAME, config_dir
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import (
    HOST_RUN_DIRNAME,
    RUN_DIRNAME,
    RUNTIME_MODULE,
)
from local_operator.session.runtime.viewers import scan_viewers

logger = logging.getLogger(__name__)

# ``RUNTIME_MODULE`` — the ``-m`` target of a session runtime process, and
# therefore the only thing a census may match on — is imported above from
# ``types``, which is where THE SPAWN CONTRACT lives: this module matches the
# string that ``launch`` and ``mobile.daemon`` write into an argv, so the two
# halves cannot drift into a census that silently matches nothing. It is
# matched as a whole argv word after a ``-m``, never as a substring: a human
# running ``grep local_operator.session.runtime.process`` (or this module's own
# tests) would otherwise appear in the census as a runtime.

#: How old a record-less runtime must be before a sweep may consider it. Sized
#: against the spread of the spawn path rather than against its median: the
#: measured cold engage publishes a record in 2.1 s warm and 5.1 s resuming a
#: 34 MB transcript, but the wake supervisor's own engage deadline — the budget a
#: cold runtime is allowed to take before its record is expected — is 180 s. 300 s
#: is that budget plus a wide margin, and it is spent as a delay on a process that
#: is already invisible rather than as a risk to a process that is still booting.
MIN_AGE_S = 300.0

#: The minimum separation between the two sightings a reclaim decision needs.
#: Long enough that a record written by a just-arriving attach is seen (the
#: supervisor re-reads the index every 10 s; a client that attaches writes its
#: record immediately), and short enough that a real orphan is gone within a
#: couple of passes on a 60 s cadence.
CONFIRM_S = 60.0

#: The CPU share of the confirm window above which a candidate is refused for
#: another window. A ONE-WAY TEST: a healthy idle runtime measures 0.0-0.3 % of a
#: core lifetime on this machine, while the two runtimes that were observed
#: mid-work measured 22.8 % and 77.0 %, so 2 % separates "visibly working" from
#: "quiet" with a wide margin. It cannot acquit anything — a runtime waiting on a
#: model API call is quiet — which is why the acquittal comes from the SIGTERM
#: drain above and this test only ever removes a candidate from the pass.
BUSY_CPU_FRACTION = 0.02

#: Floor on the CPU budget, so a window shortened by a caller (a test, or a pass
#: that ran back-to-back) cannot make the test fire on a single scheduler tick.
BUSY_CPU_FLOOR_S = 1.0

#: The shortest confirm window in which the CPU rung can still fire at all, and
#: therefore the shortest window a caller may ASK FOR.
#:
#: Derived rather than restated, because the two constants above are what make a
#: window meaningful: the budget is ``max(BUSY_CPU_FLOOR_S, BUSY_CPU_FRACTION *
#: elapsed)``, so below ``BUSY_CPU_FLOOR_S / BUSY_CPU_FRACTION`` the floor
#: dominates and the measured fraction can never reach it whatever the runtime
#: does. QA round 1 (Q2) measured the consequence end to end: at
#: ``--confirm-s 0`` a process that had burned 90.4 s of CPU (6.30 s per 60 s
#: against a 1.2 s budget at the default window) was admitted and SIGTERMed,
#: because the window it was judged over was too short for the test to see it.
#: A window shorter than this is not a faster sweep — it is a sweep with no CPU
#: rung, which is one of the four rungs the safety argument is built from, so the
#: CLI refuses it rather than quietly dropping it (``cli._confirm_window``).
MIN_ACTIONABLE_CONFIRM_S = BUSY_CPU_FLOOR_S / BUSY_CPU_FRACTION

#: How long each external tool may take before the pass gives up on it. The
#: census and the socket table are read on a supervisor's slice, next to wakes
#: that are due; a wedged ``ps`` must cost one slice, not the loop.
CENSUS_TIMEOUT_S = 5.0
SOCKET_TIMEOUT_S = 3.0

#: How long a caller that watches (the CLI, and the evidence harness) waits for a
#: signalled runtime to actually leave. The drain it may have to finish is bounded
#: by ``types.SIGNAL_DRAIN_S``, so a wait shorter than that reports "signalled, not
#: yet gone" for a runtime that is doing the right thing.
EXIT_WAIT_S = 150.0

#: One ``lsof`` row: COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME[(STATE)].
#: Anchored on the fixed column layout rather than on a split, because the DEVICE
#: column carries its own space on macOS (``0xb3af…  0t0``) and a positional split
#: drifts by one field on exactly the rows that matter.
_SOCKET_ROW = re.compile(
    r"^\S+\s+(?P<pid>\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+TCP\s+(?P<name>\S+)(?P<tail>.*)$"
)


# ---------------------------------------------------------------------------
# The evidence: what the machine says about the runtimes it is running
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeProcess:
    """One live ``-m local_operator.session.runtime.process`` from the process table.

    ``age_s`` and ``cpu_s`` are both from the kernel's own accounting (``ps``'s
    ``etime`` and ``time``), read for the whole census in one fork. The age is what
    :data:`MIN_AGE_S` tests, and the CPU is what the confirm window differences —
    neither is derived from a file the candidate writes, so both survive a
    candidate whose config root has been deleted.
    """

    pid: int
    parent_pid: int
    age_s: float
    cpu_s: float
    command: str


@dataclass(frozen=True)
class SocketEvidence:
    """The machine's TCP table, reduced to the two facts a verdict needs.

    ``available`` is False when ``lsof`` is missing, refused or timed out — the
    normal state on a machine without it — in which case :func:`verdict` REFUSES
    every candidate whose only remaining rung is this one
    (:data:`REFUSAL_SOCKETS_UNKNOWN`) and :func:`reclaim_runtimes` reports that the
    socket half was unknown. There is no third possibility: a fleet with no
    control sockets and an unread socket table look identical, and the ambiguity is
    named, never resolved by guessing — a guess here is a signal sent through
    missing evidence.
    """

    ports: Mapping[int, int] = field(default_factory=dict)
    attached: frozenset[int] = frozenset()
    available: bool = False


@dataclass(frozen=True)
class Verdict:
    """Why a candidate may or may not be ended, and everything the log quotes."""

    process: RuntimeProcess
    #: The config root the process itself reports, or ``""`` when it could not be
    #: read. Empty is a refusal (:data:`REFUSAL_UNATTRIBUTABLE`), never an orphan.
    config_root: str
    session_id: str = ""
    port: int | None = None
    #: ``""`` when the runtime may be ended; otherwise one of the ``REFUSAL_*``
    #: tokens. A token rather than a sentence because three readers consume it:
    #: this module's log line, the CLI's table and the roster's ``reclaimable``
    #: flag, and a stable token is what lets them agree.
    refusal: str = ""
    detail: str = ""

    def may_end(self) -> bool:
        return not self.refusal


#: A candidate in a state another party already owns, or in a state the sweep
#: cannot attribute. Each token has exactly one meaning, and the roster shows the
#: same token for the same runtime, so there is one answer to "why is this still
#: up" across every surface.
REFUSAL_SELF = "self"
REFUSAL_YOUNG = "young"
REFUSAL_UNATTRIBUTABLE = "unattributable"
REFUSAL_FOREIGN_ROOT = "foreign-root"
REFUSAL_RECORD_PRESENT = "record-present"
REFUSAL_OBSERVED = "observed"
REFUSAL_UNCONFIRMED = "unconfirmed"
REFUSAL_BUSY_CPU = "busy-cpu"
#: The target stopped being the process the pass decided about. Every other token
#: names a reason the LADDER refused a candidate, evaluated against the snapshot
#: the pass was handed; this one is produced at SIGNAL time, from the re-read that
#: closes the window between the snapshot and ``os.kill`` (see
#: :func:`target_changed`). It means the pid is gone, was recycled by something
#: whose argv is not the runtime's, reports a different config root, or is younger
#: than the row the pass measured — and it is a REFUSAL in every one of those
#: cases, because a candidate that cannot be re-identified is not a candidate this
#: pass may signal.
REFUSAL_CHANGED = "changed"
#: The socket table could not be read, so the strongest refusal — a client holding
#: a record-less runtime's control port — cannot be evaluated. FAIL-CLOSED, and
#: deliberately so: the alternative is the posture QA round 1 (Q5) measured, where
#: a candidate with a live client on its control port was admitted and SIGTERMed
#: when ``lsof`` failed and correctly left alone when it did not. A sweep that ends
#: one runtime it should not have is worse than a sweep that reclaims nothing (see
#: this module's test file), so an unreadable table ends the pass instead of
#: relaxing it. The report carries ``sockets_available=False`` and the roster shows
#: ``socket_table: false``, so the reason is visible wherever the consequence is.
REFUSAL_SOCKETS_UNKNOWN = "sockets-unknown"
#: The ROSTER'S token, never this module's verdict: a row whose process is not
#: running. The sweep cannot produce it because its candidate list IS the process
#: table — a record whose pid is gone is a row to report, not a candidate — so this
#: exists only so the roster never answers "not reclaimable" with no reason at all.
REFUSAL_GONE = "gone"
#: The ROSTER'S second token, never this module's verdict: the pid IS running but
#: the process table has no row for it, so the sweep's ladder never judged it and
#: "gone" would be a lie about a process a client is talking to. Two ways to reach
#: it, and the same answer covers both: the pid is not one the census RECOGNISES
#: (its argv is not the spawn contract), or the census itself could not be read
#: (``ps`` timed out at :data:`CENSUS_TIMEOUT_S`, in which case EVERY row lands
#: here). Review round 1 (R1-3) measured a live, answering runtime reported as
#: ``reclaim_refusal: gone`` for exactly this reason.
REFUSAL_NO_CENSUS = "no-census-row"


#: Whether this host's ``ps`` is procps (Linux) or the BSD/macOS implementation.
#: A NAMED CONSTANT rather than an inline ``sys.platform`` test for two reasons:
#: :func:`pid_environment` is the single place the difference is resolved, and both
#: arms are then drivable from either host by patching this name (the reason
#: ``tools/group_reaper`` patches a name rather than ``sys.platform``), so the
#: Linux spelling is exercised on the macOS developer machines too.
_IS_LINUX = sys.platform.startswith("linux")

#: Failure signatures (:func:`_report_failure`) already reported by this process.
#: Keyed by (program, what went wrong) rather than by the whole argv, because the
#: per-pid readers fork once per candidate and their argv carries the pid: a
#: per-argv key would print one line per candidate for ONE broken instrument.
_REPORTED_FAILURES: set[tuple[str, str]] = set()


def _stderr_excerpt(stderr: str | None) -> str:
    """The first thing a failed command said on stderr, trimmed, or ``""``."""
    for line in (stderr or "").splitlines():
        text = line.strip()
        if text:
            return text[:200]
    return ""


def _report_failure(command: Sequence[str], detail: str) -> None:
    """Say ONCE, on the log, that a diagnostic command failed — and what it said.

    WHY THIS EXISTS. :func:`_run_command` tolerates failure by design, but it used
    to tolerate it SILENTLY: it never read ``returncode`` and never saw the tool's
    own stderr, so "the instrument ran and found nothing" and "the instrument does
    not exist on this platform" produced the same ``""``. That is how ``ps -Eww``
    — an option procps does not have — kept every environment reader answering
    nothing on Linux while every test stayed green. The failure has to be legible
    somewhere, and this log line is the only place a diagnostic can put it without
    raising through the supervisor's slice.

    ONCE PER (program, failure kind) for the life of the process: a fleet-wide
    breakage is one line, not one per pid. The first occurrence carries the argv
    and the tool's own message; every repeat would be the same fact.
    """
    if not command:
        return
    key = (str(command[0]), detail)
    if key in _REPORTED_FAILURES:
        return
    _REPORTED_FAILURES.add(key)
    logger.warning(
        "residency reader %s failed (%s) — reading it as 'no evidence'",
        " ".join(str(part) for part in command),
        detail,
    )


def _run_command(command: Sequence[str], timeout_s: float) -> str:
    """Run a read-only command and return its stdout, or ``""`` on any failure.

    Never raises: every caller here is a diagnostic whose failure mode is
    "evidence unavailable", and a sweep that raised because ``ps`` was missing
    would take the supervisor's slice with it.

    STILL NEVER RAISES, BUT NO LONGER SILENT. Two shapes are reported, once each,
    through :func:`_report_failure`:

    * a command that could not be run at all — ``OSError`` (no such binary, no
      permission) or ``SubprocessError`` (the :data:`CENSUS_TIMEOUT_S` timeout, or
      a signal);
    * a non-zero exit **that came with a message on stderr**, which is the shape a
      platform rejecting the argv takes (``ps -Eww`` on Linux: an invalid option,
      non-zero, with the complaint on stderr).

    A non-zero exit with NOTHING on stderr is deliberately NOT reported: that is
    this module's normal, expected answer. ``ps -p <pid>`` on a pid that has gone
    and ``lsof`` with an empty table both exit non-zero in silence, and a warning
    printed on that ordinary "the target moved" path is one nobody reads by the
    time a real one arrives.

    The RETURN VALUE is unchanged on every path, including both reported ones: the
    six call sites' contract ("stdout or ``""``, never a raise") is not this fix's
    business, and widening it into a raiser — or into "failed means empty" — would
    change behaviour on paths no test here drives.
    """
    try:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            list(command), capture_output=True, text=True, timeout=timeout_s
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _report_failure(command, f"{type(exc).__name__}: {exc}")
        return ""
    if done.returncode != 0:
        stderr = _stderr_excerpt(done.stderr)
        if stderr:
            _report_failure(command, f"exit {done.returncode}: {stderr}")
    return done.stdout or ""


def proc_environ_text(pid: int) -> str:
    """``/proc/<pid>/environ`` as whitespace-separated ``NAME=value`` text, or ``""``.

    THE LINUX HALF of :func:`pid_environment`, and it needs no fork: the file is a
    NUL-separated snapshot of the environment the process was STARTED with, which
    is what ``ps -Eww`` reports on macOS (macOS ``ps(1)``: "This does not reflect
    changes in the environment after process launch").

    "Cannot read it" and "it says nothing" collapse into ``""`` here, the same way
    they do on the macOS route: the caller's only question is which config root
    this process names, and an empty answer is refused identically to a pid that is
    gone. The two ways to get ``""`` are an exited pid and a process owned by
    another user — the SAME limit ``ps -Eww`` has, because both read the same
    kernel-side environment.
    """
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return ""
    return raw.decode("utf-8", "replace").replace("\0", " ").strip()


def pid_environment(
    pid: int,
    *,
    run: Callable[[Sequence[str], float], str] = _run_command,
    timeout_s: float = CENSUS_TIMEOUT_S,
) -> str:
    """One pid's environment as text, from THIS platform's own source.

    TWO SPELLINGS, AND IT IS NOT A FLAG DANCE — ``-E`` DOES NOT EXIST IN procps.
    This module ran ``ps -Eww`` everywhere and called it "appends the process's
    environment", which is true of BSD/macOS ``ps(1)`` (``-E``: "Display the
    environment as well") and false of Linux's: procps-ng's manual lists no ``-E``
    option at all, so ``ps -Eww`` there fails with an invalid-option complaint and
    — before :func:`_report_failure` existed — every reader of a process's
    environment answered "no evidence" instead. On Linux the source is the kernel's
    own ``/proc/<pid>/environ`` (:func:`proc_environ_text`): NUL-separated, the same
    snapshot ``ps`` itself would print, and free of a fork.

    A SINGLE SPELLING WAS CONSIDERED AND NOT USED, AND THERE IS NONE TO USE.
    procps documents the BSD-style modifier ``e`` ("Show the environment after the
    command"), and ``ps eww -p PID -o command=`` is the one form that looks as if it
    would serve both families. It does not. macOS ``ps(1)`` documents NO bare ``e``
    modifier at all: its environment spelling is ``-E`` ("Display the environment as
    well. This does not reflect changes in the environment after process launch"),
    its DASHED ``-e`` is "Identical to -A" — a process-SELECTION option — and the
    ``-e`` that measures the environment exists only under LEGACY DESCRIPTION,
    beside a redefined ``-g``/``-l``/``-u``. So the
    "portable" spelling is a documented modifier on one family and an undocumented
    one on the other; and on the family that has it, using it means mixing BSD
    syntax with the Unix-style ``-p``/``-o`` this module needs — that same manual:
    "Options of different types may be freely mixed, but conflicts can appear", and
    BSD options "must not be used with a dash". A spelling whose correctness rests
    on undefined mixing rules is the same class of assumption that made ``-E`` look
    portable in the first place, and (unlike ``/proc``) it cannot be exercised from
    a macOS developer machine.

    The NULs become single spaces so the text has the same SHAPE as the macOS
    route's — whitespace-separated ``NAME=value`` — which is what the patterns in
    :func:`config_root_of` and :func:`session_id_of` read. A value containing a
    space is ambiguous in both spellings, equally, and nothing here reads anything
    but those two names.
    """
    if _IS_LINUX:
        return proc_environ_text(pid)
    return run(["ps", "-Eww", "-p", str(pid), "-o", "command="], timeout_s)


def etime_seconds(text: str) -> float:
    """``ps``'s ``etime``/``time`` (``[[dd-]hh:]mm:ss``) as seconds, or 0.0.

    An unparseable field reads as 0.0 — "no evidence" — which is the safe direction
    for both callers: an age of 0.0 is refused as too young, and a CPU figure of
    0.0 is inside every budget.
    """
    days = 0
    body = text.strip()
    if "-" in body:
        head, _, body = body.partition("-")
        try:
            days = int(head)
        except ValueError:
            return 0.0
    parts = body.split(":")
    if not parts or len(parts) > 3:
        return 0.0
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return 0.0
    while len(values) < 3:
        values.insert(0, 0.0)
    return days * 86400.0 + values[0] * 3600.0 + values[1] * 60.0 + values[2]


def parse_process_row(line: str) -> RuntimeProcess | None:
    """One ``ps`` row in this module's census format, or ``None`` if it is not a
    live session runtime.

    ONE HOME for the row shape, because two readers must agree about what a
    runtime IS: the whole-fleet census that a pass decides from, and the
    single-pid re-read that decides whether the thing about to be signalled is
    still the thing the pass measured (:func:`process_row`). A second copy of
    the argv match could drift, and the drift would be the dangerous direction —
    the re-read would stop recognising the target and every pass would refuse.

    The columns are read positionally from a fixed format string, and a row that
    does not parse is skipped rather than guessed at.
    """
    fields = line.split(None, 4)
    if len(fields) < 5:
        return None
    pid_text, ppid_text, age_text, cpu_text, command = fields
    # THE MATCH IS THE SPAWN CONTRACT, not a substring. ``launch`` starts a
    # runtime as ``<interpreter> -P -m local_operator.session.runtime.process``,
    # so the ``-m`` immediately before the module name is what identifies one.
    # Matching the module name alone would also match a person running
    # ``grep local_operator.session.runtime.process`` (and this module's own
    # tests), i.e. a census that reports the searcher as a runtime — the
    # single misidentification that would make a sweep signal a stranger.
    words = command.split()
    if not any(
        words[index] == "-m" and words[index + 1] == RUNTIME_MODULE
        for index in range(len(words) - 1)
    ):
        return None
    try:
        pid = int(pid_text)
        parent_pid = int(ppid_text)
    except ValueError:
        return None
    return RuntimeProcess(
        pid=pid,
        parent_pid=parent_pid,
        age_s=etime_seconds(age_text),
        cpu_s=etime_seconds(cpu_text),
        command=command,
    )


def runtime_processes(
    *,
    run: Callable[[Sequence[str], float], str] = _run_command,
    timeout_s: float = CENSUS_TIMEOUT_S,
) -> list[RuntimeProcess]:
    """Every live session runtime on this machine, from one ``ps`` fork.

    ONE fork for the whole census, with no ``per-pid`` probe: at 57 runtimes a
    per-pid ``ps`` (3.9 ms each) would cost 220 ms of the supervisor's slice for
    data the process table already prints in a single call.

    ``-ww`` IS MANDATORY AND IT IS NOT COSMETIC. ``command`` is this row's LAST
    column, and ``ps(1)`` extends a last column to the edge of the display — so when
    ps cannot determine the display width, as when its output is piped (every caller
    here), the width on Linux "is undefined (it may be 80, unlimited, determined by
    the TERM variable, and so on)". A truncated row severs the ``-m <module>`` pair
    :func:`parse_process_row` matches, so the census reports FEWER runtimes than
    exist — or none — as a NUMBER rather than as an error, and an empty census reads
    downstream as "nothing to sweep". ``-ww`` is unlimited width.

    ``-Eww`` would widen the row too, and it is still the wrong flag even though the
    three sibling readers in this module use it. ``-E`` appends each process's own
    environment to that same ``command`` column, which this reader does not need — the
    environment is read per CANDIDATE by :func:`process_env`/:func:`process_envs`, and
    the whole-fleet form measured 1.7 MB against 385 KB without it (see
    :func:`process_env`) — and which it must not have, because ``parse_process_row``
    word-splits that column and matches the spawn contract in it: environment words in
    the field being matched are words an env var could put there, and admitting a
    stranger as a runtime is the one misidentification this module's parser comment
    exists to forbid.
    """
    output = run(["ps", "-ww", "-eo", "pid=,ppid=,etime=,time=,command="], timeout_s)
    found: list[RuntimeProcess] = []
    for line in output.splitlines():
        row = parse_process_row(line)
        if row is not None:
            found.append(row)
    return found


def process_row(
    pid: int,
    *,
    run: Callable[[Sequence[str], float], str] = _run_command,
    timeout_s: float = CENSUS_TIMEOUT_S,
) -> RuntimeProcess | None:
    """ONE pid's census row AND the environment it reports, in ONE ``ps`` fork.

    THE SIGNAL-TIME RE-READ (see :func:`target_changed`). The environment is
    appended to the ``command`` column, so the same call that says "this pid is
    still a runtime, this old, this much CPU" also carries the config root the
    verdict attributed the candidate to — two of the four facts the
    re-identification rests on, at the cost of the single per-candidate fork the
    pass already pays for ``env_of``.

    WHERE THAT ENVIRONMENT COMES FROM IS PER PLATFORM (:func:`pid_environment`):
    on macOS ``-Eww`` appends it inside this very fork, and on Linux ``-E`` is not
    an option at all, so the row comes from ``ps -ww`` and the environment is
    appended from ``/proc/<pid>/environ`` — still ONE fork, because a ``/proc``
    read forks nothing. ``-ww`` on the Linux argv is not cosmetic either: procps
    leaves the ``command`` column's width UNDEFINED when its output is piped, and
    the argv match in :func:`parse_process_row` is what says "still a runtime".

    ``None`` means the pid is not a live session runtime NOW: gone, or a
    different process wearing the pid. Both are refusals at signal time.
    """
    # The one place the two spellings differ; the reason is in :func:`pid_environment`.
    env_flag = "-ww" if _IS_LINUX else "-Eww"
    output = run(
        ["ps", env_flag, "-p", str(pid), "-o", "pid=,ppid=,etime=,time=,command="],
        timeout_s,
    )
    for line in output.splitlines():
        row = parse_process_row(line)
        if row is not None and row.pid == pid:
            if not _IS_LINUX:
                return row
            # ``ps`` cannot carry the environment here, so append it from /proc:
            # same column, same shape for every reader of ``row.command``, and the
            # same "" when it cannot be read (see :func:`proc_environ_text`).
            env_text = pid_environment(pid, run=run, timeout_s=timeout_s)
            return replace(row, command=f"{row.command} {env_text}".strip())
    return None


def record_file(root: Path, pid: int) -> Path:
    """Where a runtime's discovery record WOULD be, creating nothing.

    ``registry.record_path`` is the owner of the ``<pid>.json`` spelling, but it
    goes through ``registry.run_dir``, which CREATES the directory it names — and
    a re-read at signal time may not. The candidate's own root can be a path that
    is GONE (the class this sweep exists for), and conjuring it back would be the
    sweep inventing the very store the runtime could then publish into, i.e.
    manufacturing the reachability it was checking for. Note the same reasoning
    in ``roster``, which refuses to scan a sibling root that is not a directory
    right now for the same reason.
    """
    return Path(root) / RUN_DIRNAME / f"{pid}.json"


#: How much YOUNGER than the row the pass measured a re-read row may be before it
#: is a different process. ``ps``'s ``etime`` has one-second resolution and only
#: ever counts up, so a recycled pid (whose life starts at zero) is separated from
#: its predecessor by the whole confirm window — 60 s by default — and the slack is
#: there for the rounding, not for the discrimination.
_AGE_SLACK_S = 2.0


def target_changed(
    item: Verdict,
    *,
    row_of: Callable[[int], RuntimeProcess | None] = process_row,
    roots: Sequence[Path] = (),
    age_slack_s: float = _AGE_SLACK_S,
) -> str:
    """``""`` when the candidate is still the process the pass decided about.

    **THE DECISION IS A SNAPSHOT AND THE SIGNAL IS NOT.** The census and the
    fleet are read once at the top of a pass, and ``os.kill`` fires later from
    that snapshot — measured on this host's real 74-row census: first SIGTERM
    19 ms after the pass began, last one **1161 ms** after. Two things can change
    inside that window and neither is visible to the ladder, because the ladder
    has already run:

    * **(a) the pid is recycled.** It exits and the kernel reuses the number; the
      signal then lands on a stranger. Nothing in the ladder can see the
      substitution — the ``young``/``foreign``/``self`` rungs were all evaluated
      against the EARLIER census. A runtime reusing the pid would be refused as
      ``young`` if it were re-read, but the hazard that survives is any OTHER
      process: a build, a test runner, a daemon, receiving SIGTERM.
    * **(b) a record appears.** A candidate whose record was deleted re-publishes
      on its next heartbeat, which makes it reachable — the exact race the
      confirm window exists to prevent, moved inside the pass. This one is
      contained by the runtime's work-aware SIGTERM drain, but it should not need
      to be.

    Both are closed here, by re-reading the ONE row immediately before the signal
    and refusing on any change: the pid must still exist and still be a runtime by
    its argv (:func:`process_row` returns ``None`` otherwise), it must not be
    YOUNGER than the row the pass measured (a recycled pid starts its life at
    zero), it must report the same config root (the attribution the verdict rests
    on), and there must be no discovery record for it now in any of ``roots``.

    Refusal, never a correction: this is not a second verdict and it can neither
    admit a candidate the ladder turned away nor rescue one it did not judge —
    only withhold a signal the snapshot authorised.
    """
    pid = item.process.pid
    row = row_of(pid)
    if row is None:
        return REFUSAL_CHANGED
    if row.age_s < item.process.age_s - age_slack_s:
        return REFUSAL_CHANGED
    if config_root_of(row.command) != item.config_root:
        return REFUSAL_CHANGED
    if any(record_file(root, pid).is_file() for root in roots):
        return REFUSAL_RECORD_PRESENT
    return ""


def socket_evidence(
    *,
    run: Callable[[Sequence[str], float], str] = _run_command,
    timeout_s: float = SOCKET_TIMEOUT_S,
) -> SocketEvidence:
    """The machine's TCP table: each pid's listening port, and which are occupied.

    ONE machine-wide ``lsof`` rather than one per runtime. Measured on this
    machine at ~200 ms for 664 lines, against ~186 ms PER PID for the per-pid form
    — which at 57 runtimes would be 10.6 s of a 10 s supervisor slice and would
    make the roster endpoint's whole budget one row deep.

    Both facts come from the same call because both are properties of the same
    table: a ``(LISTEN)`` row names a pid's port, and an ``ESTABLISHED`` row whose
    LOCAL side is that port means something is connected to it right now. The
    second is the sweep's strongest per-candidate evidence that an interface is
    attached, and it is evidence a record cannot supply at all — the orphans this
    module exists for have no record to ask.
    """
    output = run(["lsof", "-nP", "-iTCP"], timeout_s)
    if not output.strip():
        return SocketEvidence()
    ports: dict[int, int] = {}
    occupied: set[int] = set()
    for line in output.splitlines():
        match = _SOCKET_ROW.match(line)
        if match is None:
            continue
        name = match.group("name")
        tail = match.group("tail")
        if "(LISTEN)" in tail:
            port_text = name.rsplit(":", 1)[-1]
            if port_text.isdigit():
                ports.setdefault(int(match.group("pid")), int(port_text))
        elif "(ESTABLISHED)" in tail and "->" in name:
            # The LOCAL side is what a portrait of this runtime needs: the peer is
            # a client, and its own port says nothing about which runtime it holds.
            local = name.split("->", 1)[0]
            port_text = local.rsplit(":", 1)[-1]
            if port_text.isdigit():
                occupied.add(int(port_text))
    # ---- ATTACHED IS AN INTERSECTION, NOT A UNION. An ESTABLISHED row on its own
    # says only that SOME process has a connection; only a connection whose LOCAL
    # side is a port this table shows in LISTEN state is a client sitting on a
    # runtime's control socket. A plain outbound connection (every browser tab on
    # the machine) would otherwise be read as an interface attached to a runtime.
    return SocketEvidence(
        ports=ports, attached=frozenset(occupied & set(ports.values())), available=True
    )


def process_env(pid: int, *, run: Callable[[Sequence[str], float], str] = _run_command) -> str:
    """One runtime's environment as text, or ``""`` when it cannot be read.

    Read per CANDIDATE and never for the whole fleet: one census with the
    environment attached measured 1.7 MB of output for 1140 processes against
    385 KB without it, and the only question this answers — which config root does
    this runtime use — is asked of a handful of rows after the cheap filters have
    run.

    The platform's own source, and why it is no longer ``ps -Eww`` everywhere, is
    in :func:`pid_environment`. The macOS route needs ``-ww``, not as cosmetics:
    ``ps`` truncates the environment to the terminal width by default, which on a
    narrow or piped stdout silently drops the tail of the very variable being
    looked for. The Linux route needs no width flag at all — ``/proc`` is not a
    terminal.
    """
    return pid_environment(pid, run=run, timeout_s=CENSUS_TIMEOUT_S)


def process_envs(
    *,
    run: Callable[[Sequence[str], float], str] = _run_command,
    timeout_s: float = CENSUS_TIMEOUT_S,
) -> dict[int, str]:
    """EVERY process's environment, from ONE ``ps`` fork, keyed by pid.

    The per-pid form (:func:`process_env`) is a fork each, and a caller that wants
    this for the whole fleet pays it per runtime: the roster measured ~2-4 s for 36
    read-only reads on a loaded machine, against ~0.8 s for the same data in one
    call. The per-pid reader stays for the single-candidate path (a sweep looks at a
    handful) and for the fallback below; this one is for the fleet.

    ONE ``ps`` fork on both platforms, and on Linux that is the whole extra cost:
    the environment per row comes from ``/proc/<pid>/environ``
    (:func:`proc_environ_text`), which is a file read and not a fork, so the
    fleet-wide shape and its measured budget are unchanged. ``-Eww`` there would be
    the macOS spelling only — ``-E`` is not a procps option, see
    :func:`pid_environment` — and every value would then be the command line alone,
    i.e. a config root nobody could read. ``-ww`` IS passed on both, because procps
    leaves the ``command`` column's width undefined whenever its output is piped,
    and the command prefix is half of what a caller is handed.

    A pid ``ps`` does not print is ABSENT rather than empty, so a caller can tell "no
    environment readable" from "an environment with nothing in it" — the first is
    what a candidate must be refused for, and the second does not exist.
    """
    argv = (
        ["ps", "-ww", "-eo", "pid=,command="]
        if _IS_LINUX
        else ["ps", "-Eww", "-eo", "pid=,command="]
    )
    output = run(argv, timeout_s)
    envs: dict[int, str] = {}
    for line in output.splitlines():
        text = line.strip()
        if not text:
            continue
        pid_text, _, command = text.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if _IS_LINUX:
            # The command prefix is KEPT so a caller sees the shape it sees on
            # macOS, and the environment is appended from /proc: it is the part
            # any caller reads (:func:`config_root_of`, :func:`session_id_of`).
            envs[pid] = f"{command.strip()} {proc_environ_text(pid)}".strip()
        else:
            envs[pid] = command.strip()
    return envs


def config_root_of(env_text: str) -> str:
    """The config root a runtime's own environment names, or ``""`` if it does not.

    ``LOCAL_OPERATOR_CONFIG_DIR`` first, then the documented default under the
    process's own ``HOME`` — the same two-step ``paths.config_dir`` performs in the
    runtime itself, re-derived here because the runtime's answer is the only one
    that matters: a sweep must decide against the root that process is actually
    using, not against the root the sweep happens to be running under.
    """
    match = re.search(rf"(?:^|\s){re.escape(CONFIG_DIR_ENV)}=(\S+)", env_text)
    if match:
        return match.group(1)
    home = re.search(r"(?:^|\s)HOME=(\S+)", env_text)
    if home:
        return str(Path(home.group(1)) / DEFAULT_CONFIG_DIRNAME)
    return ""


def session_id_of(env_text: str) -> str:
    """The session id a runtime was spawned for, from the spawn contract.

    ``LOP_MOBILE_CHILD_RESUME`` is set by ``launch`` for every spawned runtime and
    is the session it hosts; read only as a fallback for a candidate whose boot
    record is unreadable (a deleted root), because the boot record is the artifact
    whose whole purpose is to name the pid's session after the fact.
    """
    match = re.search(r"(?:^|\s)LOP_MOBILE_CHILD_RESUME=(\S+)", env_text)
    return match.group(1) if match else ""


def ancestor_pids(
    pid: int | None = None, *, run: Callable[[Sequence[str], float], str] = _run_command
) -> frozenset[int]:
    """This process, its parents, and its grandparents — the pids a sweep may not touch.

    A sweep can be run from inside a session: ``lop sessions reclaim`` typed into a
    TUI, or an agent's own shell. Signalling an ancestor would end the very session
    that asked, so the whole chain is excluded, read from ``ps`` in one fork (the
    walk uses ``ppid=`` per hop because there is no stdlib call for it, and it is
    bounded because a cycle would otherwise spin).
    """
    chain: set[int] = {os.getpid() if pid is None else pid}
    current = os.getpid() if pid is None else pid
    for _ in range(64):
        output = run(["ps", "-o", "ppid=", "-p", str(current)], CENSUS_TIMEOUT_S).strip()
        try:
            parent = int(output.splitlines()[0]) if output else 0
        except (IndexError, ValueError):
            break
        if parent <= 0 or parent in chain:
            break
        chain.add(parent)
        current = parent
    return frozenset(chain)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fleet:
    """One read of everything a verdict needs, taken once per pass.

    Bundled rather than passed as four arguments because the four are read
    together and must agree: a record read in one pass and a socket table in
    another would let a runtime be refused for an attach that had already ended.
    """

    root: Path
    records: Mapping[int, Any]
    boots: Mapping[int, Any]
    viewers: Sequence[Any]
    sockets: SocketEvidence
    #: ``pid -> the boot record's own ``started_at````, for :func:`effective_age`.
    #: A SECOND, LATER age for one process, and the only way the young rung can
    #: see it: an ADOPTED runtime is an interpreter that was started minutes
    #: before the session it is now constructing (see
    #: ``session/runtime/standby.py``), so its ``ps`` age says "long-lived" while
    #: its session is seconds old.
    boot_starts: Mapping[int, float] = field(default_factory=dict)
    own_pids: frozenset[int] = frozenset()
    #: The roots this pass may act in, when a caller NAMED them. ``None`` is the
    #: production rule in the module docstring — the swept root, plus any root that
    #: no longer exists — and a named set REPLACES it, which is what lets a caller
    #: sweep a store that is not its own without the ladder reading that store's
    #: runtimes as someone else's business. The evidence harness and the tests are
    #: the callers that need it: a sweep run inside a sandbox must not be able to
    #: act on another store's runtimes at all.
    scope: frozenset[Path] | None = None


def read_fleet(root: Path | None = None, *, sockets: SocketEvidence | None = None) -> Fleet:
    """Read the disposition of the swept root: records, boot records, viewers, sockets.

    EVERY READ HERE IS A READER'S READ: ``registry.scan(..., reap=False)`` and
    ``scan_viewers(..., reap=False)`` return the same verdicts while removing
    nothing. A sweep that reaped a stale record as a side effect of deciding would
    be the process that destroyed the evidence for the death it was looking at.

    The boot records are read directly rather than through ``journal``'s helpers
    because they are looked up BY PID for every candidate, which is a mapping, not
    the single-record question ``read_boot_record`` answers.
    """
    from local_operator.session.runtime.types import SessionRecord

    config_root = Path(root) if root is not None else config_dir()
    scanned = registry.scan(config_root, RUN_DIRNAME, SessionRecord.from_json, reap=False)
    records = {record.pid: record for record, _state in scanned}
    boots: dict[int, Any] = {}
    boot_starts: dict[int, float] = {}
    host_dir = config_root / HOST_RUN_DIRNAME
    try:
        boot_paths = sorted(host_dir.glob("*.json"))
    except OSError:
        boot_paths = []
    for path in boot_paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        pid = data.get("pid") if isinstance(data, dict) else None
        if isinstance(pid, int):
            boots.setdefault(pid, data)
            started = data.get("started_at")
            if isinstance(started, (int, float)):
                boot_starts.setdefault(pid, float(started))
    return Fleet(
        root=config_root,
        records=records,
        boots=boots,
        viewers=scan_viewers(config_root, reap=False),
        sockets=sockets if sockets is not None else SocketEvidence(),
        boot_starts=boot_starts,
        own_pids=ancestor_pids(),
    )


def effective_age(process: RuntimeProcess, fleet: Fleet, *, now: float | None = None) -> float:
    """How long this runtime has been doing what a caller cares about.

    The YOUNGER of two observations, and both are stamped by the kernel or by a
    file's writer:

    * the process's own age from the process table (``ps`` ``etime``), valid
      however broken the runtime's own instrumentation is;
    * the age of its boot record (``run/host/<pid>.json``) — written by the
      runtime at its boot boundary, or by an ADOPTED standby the moment it takes
      a spawn.

    WHY THE SECOND IS NEEDED AT ALL (agent review round 1, R1-3). An adopted
    runtime is a pre-imported interpreter this machine started up to
    :data:`IDLE_REAP_S` BEFORE the session it is constructing. Its ``etime``
    therefore reports "15 minutes old" while its construction is seconds old, so
    the young rung — the one rung that covers the whole construction window —
    would not fire, and a sweep could signal a runtime that has not published
    yet. The MINIMUM is what makes that impossible, and for a forked runtime the
    two agree to within the boot boundary, so the reading is unchanged.
    """
    started = fleet.boot_starts.get(process.pid)
    if started is None:
        return process.age_s
    stamp = time.time() if now is None else now
    return min(process.age_s, max(0.0, stamp - started))


def verdict(
    process: RuntimeProcess,
    fleet: Fleet,
    *,
    env_of: Callable[[int], str] = process_env,
    min_age_s: float = MIN_AGE_S,
    now: float | None = None,
    root_hint: str = "",
    session_id_hint: str = "",
) -> Verdict:
    """Whether this runtime may be ended, and the reason it may not.

    The ladder is ordered so that the cheapest and most decisive refusals come
    first, and every rung is evidence about the PROCESS rather than about a file it
    may have stopped updating:

    1. ``self`` — this process or an ancestor. The one refusal that must never be
       skipped, and the one a caller cannot be trusted to state.
    2. ``young`` — newer than :data:`MIN_AGE_S`, i.e. possibly still constructing
       and not yet publishing. The process table's own age, or its boot record's
       (whichever is YOUNGER — see :func:`effective_age`), so it holds however
       broken the runtime's own instrumentation is, and it covers an adopted
       runtime whose interpreter predates its session.
    3. ``unattributable`` — its config root could not be read, so no statement can
       be made about which records could exist for it.
    4. ``foreign-root`` — its root is neither the swept root nor a deleted path. A
       runtime of a live root this sweep is not scoped to is that root's business.
    5. ``record-present`` — it published a record, so it has a reader-facing
       existence and its own reaper's attention. A record whose heartbeat is stale
       is still a refusal: a heartbeat is authored by the runtime's own event loop,
       so a quiet one covers a frozen process AND a healthy one starved by a long
       turn (measured false positives: 105.8 s and 205.8 s), which is why
       ``wedged_runtime`` refuses to end a runtime on that evidence and so does
       this.
    6. ``observed`` — a live viewer lease names its session, or its control port
       has an ESTABLISHED peer. Something is looking at it.
    7. ``sockets-unknown`` — the machine's socket table could not be read at all,
       so the half of rung 6 that lives ONLY there cannot be evaluated. Placed
       last because every stronger refusal has already had its say, and it fires
       only for a candidate that would otherwise be ADMITTED: losing the strongest
       refusal must not become a permission (see :data:`REFUSAL_SOCKETS_UNKNOWN`).
    """
    if process.pid in fleet.own_pids:
        return Verdict(process=process, config_root="", refusal=REFUSAL_SELF)
    age_s = effective_age(process, fleet, now=now)
    if age_s < min_age_s:
        return Verdict(
            process=process,
            config_root="",
            refusal=REFUSAL_YOUNG,
            detail=f"{age_s:.0f}s < {min_age_s:.0f}s",
        )
    # THE HINTS ARE FOR A CALLER THAT ALREADY KNOWS — the roster, which has just
    # composed the row and must not be able to describe a runtime differently from
    # the way this function decides about it. A hint is STRONGER evidence than the
    # environment (a record found in this root, a boot record found in this root),
    # so supplying one skips the fork rather than second-guessing it; with no hint,
    # the process's own environment is the only source and it is read here.
    env_text = env_of(process.pid) if not (root_hint and session_id_hint) else ""
    config_root = root_hint or config_root_of(env_text)
    if not config_root:
        return Verdict(
            process=process, config_root="", refusal=REFUSAL_UNATTRIBUTABLE, detail="no config root"
        )
    candidate_root = Path(config_root)
    if fleet.scope is not None:
        in_scope = candidate_root in fleet.scope
    else:
        in_scope = candidate_root == fleet.root or not candidate_root.exists()
    if not in_scope:
        return Verdict(process=process, config_root=config_root, refusal=REFUSAL_FOREIGN_ROOT)
    if process.pid in fleet.records:
        return Verdict(
            process=process,
            config_root=config_root,
            refusal=REFUSAL_RECORD_PRESENT,
            detail="discovery record published",
        )
    session_id = session_id_hint
    boot = fleet.boots.get(process.pid)
    if not session_id and isinstance(boot, dict):
        session_id = str(boot.get("session_id") or "")
    if not session_id:
        session_id = session_id_of(env_text)
    port = fleet.sockets.ports.get(process.pid)
    if session_id:
        for viewer in fleet.viewers:
            if getattr(viewer, "current_session", "") == session_id:
                return Verdict(
                    process=process,
                    config_root=config_root,
                    session_id=session_id,
                    port=port,
                    refusal=REFUSAL_OBSERVED,
                    detail=f"viewer {getattr(viewer, 'pid', '?')} is showing it",
                )
    if port is not None and port in fleet.sockets.attached:
        return Verdict(
            process=process,
            config_root=config_root,
            session_id=session_id,
            port=port,
            refusal=REFUSAL_OBSERVED,
            detail=f"connection established on control port {port}",
        )
    # FAIL-CLOSED ON MISSING EVIDENCE. Everything above is evidence that something
    # IS there; this is the absence of the evidence that something is ATTACHED, and
    # absence-of-evidence must never be read as permission on the rung whose only
    # job is to notice an attached client: QA round 1 (Q5) measured two identical
    # record-less candidates with a live client on the control port — the one whose
    # pass could read the socket table was correctly ``observed`` and left alive,
    # the one whose ``lsof`` failed was admitted and SIGTERMed. Refusing here costs
    # the whole software-reclaim class on a machine without ``lsof`` (the module's
    # docstring says so), which is the trade this module's own test file states:
    # a sweep that ends one runtime it should not have is worse than a sweep that
    # reclaims nothing.
    if not fleet.sockets.available:
        return Verdict(
            process=process,
            config_root=config_root,
            session_id=session_id,
            port=port,
            refusal=REFUSAL_SOCKETS_UNKNOWN,
            detail="the socket table could not be read, so an attach cannot be ruled out",
        )
    return Verdict(process=process, config_root=config_root, session_id=session_id, port=port)


# ---------------------------------------------------------------------------
# The confirm window, and the sweep
# ---------------------------------------------------------------------------


@dataclass
class _Sighting:
    at: float
    cpu_s: float


class Sightings:
    """The sweep's memory ACROSS passes: when each candidate was last seen, and its CPU.

    Held by the caller rather than by this module, because the two callers need
    different lifetimes: the supervisor keeps one for the life of its process (so a
    60 s confirm window spans two of its slices), while the CLI makes one per
    invocation and waits the window out inside it. A lost memory is not a hazard —
    the next pass simply re-establishes the window — which is what makes an
    in-memory structure the right shape for a reaper whose whole job is done
    cheaply and repeated.
    """

    def __init__(self) -> None:
        self._seen: dict[int, _Sighting] = {}

    def confirm(self, candidate: Verdict, *, now: float, confirm_s: float = CONFIRM_S) -> str:
        """``""`` when the window has elapsed, else the refusal that applies now.

        The CPU test lives here because it needs the two sightings: the difference
        between the cumulative CPU at the first sighting and at this one, against
        the window's own budget. A candidate that fails either test re-arms the
        window from NOW, so a runtime that is busy for an hour is re-tested on every
        pass rather than accumulating credit towards a reclaim.
        """
        process = candidate.process
        previous = self._seen.get(process.pid)
        self._seen[process.pid] = _Sighting(at=now, cpu_s=process.cpu_s)
        if previous is None:
            return REFUSAL_UNCONFIRMED
        elapsed = now - previous.at
        if elapsed < confirm_s:
            return REFUSAL_UNCONFIRMED
        budget = max(BUSY_CPU_FLOOR_S, BUSY_CPU_FRACTION * elapsed)
        spent = process.cpu_s - previous.cpu_s
        if spent > budget:
            return REFUSAL_BUSY_CPU
        return ""

    def forget(self, pids: Iterable[int]) -> None:
        """Drop pids that are no longer candidates, so the memory stays bounded.

        THE ITERABLE IS MATERIALISED ONCE, and that is not a micro-optimisation:
        the only production caller passes a GENERATOR (``reclaim_runtimes``
        calls ``seen.forget(process.pid for process in census)``), and a
        ``set(pids)`` written INSIDE the loop re-evaluates it — first iteration
        drains the generator, every later iteration compares against an EMPTY
        set and drops the entry. The result was a sweep that could confirm
        exactly one runtime per pass (the first in census order, the same one
        every time) and read every other candidate as ``unconfirmed`` forever,
        on the two paths the feature exists for: the wake supervisor's pass and
        ``lop sessions reclaim``. The whole-fleet fix for 34-40 unreachable
        runtimes was one process. The parameter is typed ``Iterable``, so the
        one-shot form is a legal call and this is where it must be handled.
        """
        keep = set(pids)
        for pid in list(self._seen):
            if pid not in keep:
                self._seen.pop(pid, None)


@dataclass
class ReclaimReport:
    """What one pass saw, refused, and ended. The only account of a sweep that acts."""

    root: Path
    #: Live session runtimes on the machine — the census, before any filtering.
    census: int = 0
    #: Candidates the verdict admitted (no refusal) AND the confirm window cleared.
    reclaimed: list[Verdict] = field(default_factory=list)
    #: Candidates the verdict admitted that the confirm window has NOT cleared,
    #: carrying the token that deferred them (``unconfirmed``/``busy-cpu``). One
    #: candidate appears in exactly one of these lists: they used to be appended to
    #: ``refused`` as well, so one summary line reported the same 59 rows on both
    #: halves of itself and :meth:`refusals` counted rows the pass had not refused —
    #: the roster read "59 awaiting the window, refused [unconfirmed 59]" as two
    #: facts about two populations (review round 1, NIT 8).
    pending: list[Verdict] = field(default_factory=list)
    #: Every runtime the sweep declined to end, with its token. Counted by token in
    #: :meth:`summary` so a pass over a healthy machine reads as "nothing but
    #: record-present", which is what makes a real orphan visible as a delta.
    refused: list[Verdict] = field(default_factory=list)
    #: Candidates signalled this pass (empty on a dry run).
    signalled: list[Verdict] = field(default_factory=list)
    #: Of the signalled, the ones confirmed gone inside the caller's wait.
    exited: list[Verdict] = field(default_factory=list)
    applied: bool = False
    sockets_available: bool = True
    #: The confirm window this pass used. Carried on the report rather than read from
    #: the module constant, because a caller (the CLI, the tests, the evidence
    #: harness) may shorten it and a summary that quoted the default would describe a
    #: different run from the one it is reporting.
    confirm_s: float = CONFIRM_S

    def refusals(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.refused:
            counts[item.refusal] = counts.get(item.refusal, 0) + 1
        return counts

    def deferrals(self) -> dict[str, int]:
        """Why the pending candidates are waiting, by token.

        The counterpart of :meth:`refusals`, and separate from it because a
        candidate awaiting the window is not a refusal: the ladder admitted it and
        the next pass may well end it. Reported with the same tokens so one
        operator reading either line is reading the same vocabulary.
        """
        counts: dict[str, int] = {}
        for item in self.pending:
            counts[item.refusal] = counts.get(item.refusal, 0) + 1
        return counts

    def summary(self) -> str:
        """One operator-grade line: the census, the verdicts, and what was ended."""
        mode = "reclaimed" if self.applied else "would reclaim (dry run)"
        refused = ", ".join(f"{token} {count}" for token, count in sorted(self.refusals().items()))
        deferred = ", ".join(
            f"{token} {count}" for token, count in sorted(self.deferrals().items())
        )
        return (
            f"runtime residency: {self.census} live session runtimes, "
            f"{len(self.reclaimed)} {mode}, {len(self.pending)} awaiting the "
            f"{self.confirm_s:.0f}s confirm window"
            + (f" [{deferred}]" if deferred else "")
            + (f", refused [{refused}]" if refused else "")
            + (
                ""
                if self.sockets_available
                else " (socket table unavailable: lsof missing or timed out)"
            )
        )

    def to_json(self) -> dict[str, Any]:
        def row(item: Verdict) -> dict[str, Any]:
            return {
                "pid": item.process.pid,
                "session_id": item.session_id,
                "config_root": item.config_root,
                "age_s": round(item.process.age_s, 1),
                "cpu_s": round(item.process.cpu_s, 1),
                "port": item.port,
                "refusal": item.refusal,
                "detail": item.detail,
            }

        return {
            "root": str(self.root),
            "census": self.census,
            "applied": self.applied,
            "sockets_available": self.sockets_available,
            "reclaimed": [row(item) for item in self.reclaimed],
            "pending": [row(item) for item in self.pending],
            "signalled": [row(item) for item in self.signalled],
            "exited": [row(item) for item in self.exited],
            "refused": [row(item) for item in self.refused],
            "refusals": self.refusals(),
        }


def _record_roots(view: Fleet, item: Verdict) -> list[Path]:
    """The roots whose record directory could hold this candidate's record NOW.

    TWO, because attribution and publication are different questions: the pass
    judged the candidate against the SWEPT root's records (``verdict``'s rung 5),
    while a runtime publishes its record into ITS OWN root — which is how a root
    that no longer exists is in scope at all. A record appearing in either place
    makes the candidate reachable, and reachability is the whole of what this
    sweep must not race, so both are checked at signal time rather than the one a
    single ``Fleet`` happens to have been read from.
    """
    roots = [view.root]
    own = Path(item.config_root) if item.config_root else None
    if own is not None and own != view.root:
        roots.append(own)
    return roots


def reclaim_runtimes(
    root: Path | None = None,
    *,
    apply: bool = False,
    sightings: Sightings | None = None,
    confirm_s: float = CONFIRM_S,
    min_age_s: float = MIN_AGE_S,
    roots: Iterable[Path] | None = None,
    processes: Sequence[RuntimeProcess] | None = None,
    sockets: SocketEvidence | None = None,
    env_of: Callable[[int], str] = process_env,
    row_of: Callable[[int], RuntimeProcess | None] = process_row,
    fleet: Fleet | None = None,
    kill: Callable[[int, int], None] = os.kill,
    wait_s: float = 0.0,
    now: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ReclaimReport:
    """One residency pass: decide, and (with ``apply``) SIGTERM what may be ended.

    ``roots`` narrows the pass to the named config roots, which is what the
    evidence harness and the tests drive so a sweep run inside a sandbox can never
    act on another store's runtimes. The production default (``None``) is the
    scoped rule in this module's docstring: the swept root, plus any root that no
    longer exists.

    ``apply=False`` is the default for every caller that has not been asked to
    act — the pass still does the full census and the full verdict and reports what
    it WOULD end, which is what makes a dry run evidence rather than a promise.

    ``row_of`` is the SIGNAL-TIME reader (:func:`process_row`, one fork for one
    pid), and it is injected for the same reason ``env_of`` is: it is the only
    place this pass touches the process table a second time, and a test that
    drives a synthetic census has to be able to answer for it. It is used only
    when ``apply`` is set, and only for a candidate the window has just cleared —
    see :func:`target_changed` for why the snapshot a decision came from is not
    authoritative by the time the signal goes out.
    """
    moment = time.time() if now is None else now
    # THE SOCKET TABLE IS READ HERE, NOT LEFT TO THE CALLER. It is one of the two
    # pieces of evidence that can refuse a candidate nothing else can refuse (a
    # runtime with no record, no viewer lease and an open client on its control
    # port), so a pass that skipped it would be a pass whose strongest refusal never
    # fires — which is exactly what the supervisor's own call site would do if the
    # default were "no evidence". Measured cost on the reference host: ~200 ms for
    # the whole table, once per pass.
    view = (
        fleet
        if fleet is not None
        else read_fleet(root, sockets=sockets if sockets is not None else socket_evidence())
    )
    if roots is not None:
        # ``replace`` rather than a second Fleet so every rung downstream sees the
        # same object: the verdict's scope test and this pass's filter must agree, or
        # a named root would be reported as a refusal in one place and skipped in the
        # other.
        view = replace(view, scope=frozenset(Path(item) for item in roots))
    seen = sightings if sightings is not None else Sightings()
    census = list(processes) if processes is not None else runtime_processes()
    report = ReclaimReport(
        root=view.root,
        census=len(census),
        applied=apply,
        sockets_available=view.sockets.available,
        confirm_s=confirm_s,
    )
    named = view.scope
    for process in census:
        item = verdict(process, view, env_of=env_of, min_age_s=min_age_s, now=moment)
        if named is not None and (not item.config_root or Path(item.config_root) not in named):
            # NAMED ROOTS ARE A FILTER, NOT A VERDICT: a caller that asked for one
            # store does not want the other 60 rows of refusals rendered back at it.
            # The verdict still ran (so nothing here can overrule the ladder), the row
            # is simply not part of this pass.
            continue
        if item.refusal:
            report.refused.append(item)
            continue
        refusal = seen.confirm(item, now=moment, confirm_s=confirm_s)
        if refusal:
            # DEFERRED, NOT REFUSED, and reported ONCE: ``pending`` carries the
            # token that deferred it, so nothing is counted in two places (see the
            # field's comment). A ``busy-cpu`` candidate lands here too — it is a
            # candidate the next window may end, and the CLI lists it as one.
            report.pending.append(
                Verdict(
                    process=item.process,
                    config_root=item.config_root,
                    session_id=item.session_id,
                    port=item.port,
                    refusal=refusal,
                    detail=f"cpu {item.process.cpu_s:.1f}s",
                )
            )
            continue
        if apply:
            # THE SNAPSHOT IS NOT AUTHORITATIVE BY THE TIME THE SIGNAL GOES OUT.
            # See ``target_changed``: the row is re-read here, one fork, and a
            # candidate that cannot be re-identified is refused rather than
            # signalled. Absent for a dry run, whose whole output is the decision
            # itself and which signals nothing for the race to endanger.
            changed = target_changed(item, row_of=row_of, roots=_record_roots(view, item))
            if changed:
                report.refused.append(
                    Verdict(
                        process=item.process,
                        config_root=item.config_root,
                        session_id=item.session_id,
                        port=item.port,
                        refusal=changed,
                        detail="the target changed between the decision and the signal",
                    )
                )
                continue
        report.reclaimed.append(item)
        if not apply:
            continue
        logger.warning(
            "runtime residency: ending an unreferenced session runtime (pid %d, session %s, "
            "root %s, alive %.0fs, cpu %.1fs, port %s) — no record, nothing attached, and "
            "no root it can be reached through",
            item.process.pid,
            item.session_id or "<unknown>",
            item.config_root or "<unknown>",
            item.process.age_s,
            item.process.cpu_s,
            item.port if item.port is not None else "<none>",
        )
        try:
            # SIGTERM ONLY, AND NEVER ESCALATED. The runtime's own handler is
            # work-aware: it finishes any turn in flight before leaving, bounded by
            # SIGNAL_DRAIN_S. SIGKILL has no handler and would destroy that turn,
            # which is the one thing this module must never do.
            kill(item.process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
        report.signalled.append(item)
    if apply and wait_s > 0:
        deadline = time.monotonic() + wait_s
        for item in list(report.signalled):
            while time.monotonic() < deadline and registry.pid_alive(
                item.process.pid, check_zombie=True
            ):
                sleep(0.25)
            if not registry.pid_alive(item.process.pid, check_zombie=True):
                report.exited.append(item)
    if view.scope is None:
        seen.forget(process.pid for process in census)
    return report
