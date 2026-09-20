"""Bound a runtime's OWN stall: dump every thread, then leave.

**WHY THIS EXISTS, in one measurement.** On 2026-09-20 five of this machine's
session runtimes were found frozen at once, for 1.5 to 7.2 hours each, and each
had to be reaped by hand with ``lop stop --force``. ``sample <pid> 3`` on every
one of them put 100% of the event-loop main thread inside
``_sre_SRE_Pattern_search`` -> ``sre_search`` -> ``sre_ucs1_match`` — a bare
``.search()`` — while CPU kept advancing at ~0.9 core and the transcript took
zero writes. The linear cost model does not explain a freeze of hours: the
repo's own ``scrub_secrets`` runs over those very transcripts in 1.75-2.07 s.
The scan simply was not converging, and **nothing in the process could say which
line it was stuck on**, because the one instrument that would have
(``LOP_RUNTIME_DEBUG_STACKS``' SIGUSR1 asyncio dump, ``process.amain``) was not
switched on. There is no automatic recovery path either: ``lop sessions
reclaim`` refused all five (``record-present``) and ``session.cleanup.enabled``
is off.

So a runtime now bounds its own stall. The operator's directive, verbatim: "Yes
force stop it then continue with instructions to not repeat that."

WHY A C TIMER AND NOTHING IN PYTHON
-----------------------------------
A ``threading.Thread`` watchdog, an ``asyncio`` timer and a signal handler all
fail here, and ``tests/e2e/watchdog.py`` documents the measurements: a hung
thread never releases the GIL, so a thread watchdog cannot run; a Python signal
handler runs only between bytecodes, so ``pytest-timeout``'s default method
never fires; and a loop parked in a C call never schedules the coroutine that
would report the sample. (The same GIL is why the SIGUSR1 dump above cannot
answer this class of question: its handler is an asyncio signal handler and
needs the loop that is blocked.)

``faulthandler.dump_traceback_later`` is the one mechanism that survives it. It
arms a timer in a dedicated **C** thread that writes every thread's stack with
``write(2)`` straight to a file descriptor and then calls ``_exit(1)`` — no GIL,
no Python frames and no interpreter state required.

WHAT "STALL" MEANS HERE: NO PROGRESS, NOT SLOW WORK
---------------------------------------------------
The bound is not a stopwatch on a turn. The timer is armed once and **re-armed
by the runtime's own loops as they run** (``beat``), so it measures the gap
since the last sign of life *inside this process* — the same signal the record's
heartbeat publishes. A turn that awaits for ten minutes keeps beating; a loop
parked in one C call does not.

TWO LOOPS BEAT, DELIBERATELY. A ``daemon``/``exec`` runtime runs its workload
and its serving plane on separate threads (``RuntimeServer.start``), and only
the process as a whole is what an operator experiences as frozen. The timer is
process-global — there is exactly one — so the honest reset for it is progress
from ANY of the process's own loops: ``RuntimeServer._heartbeat_loop`` beats,
and so does the workload half (``process._beat_stall_watchdog``). A stall
confined to one plane therefore does NOT trip this bound (that shape shows up as
a ``busy``/lane reading instead), which is the residual this design accepts
rather than killing a runtime that is demonstrably still working.

WHY THE EXIT IS THE BLUNT ONE, AND WHAT IT COSTS
------------------------------------------------
Past the bound the runtime must stop being a multi-hour freeze, and the graceful
rungs cannot be reached from the state being detected: ``_drain_for_signal``,
``_commit_to_leaving`` and ``_leave_overdue`` are all coroutines on the loop that
is blocked, and the ``SIGTERM`` handler that reaches them is a loop handler that
needs bytecodes the stuck thread never executes. That is the same fact that
makes ``lop stop`` refuse a silent-socket runtime without ``--force``. So the
timer is armed with ``exit=True``: faulthandler writes the dump and calls
``_exit(1)`` from its own C thread. Write-then-act is therefore structural
rather than something this module has to sequence.

WHAT IS LOST: the in-flight turn's uncommitted step — the same loss a SIGKILL
inflicts, because a turn commits its transcript at each step and the step in
flight has not committed. What SURVIVES: everything already committed, i.e. the
conversation, which a successor can be engaged on. What IS LEFT BEHIND: the
record, whose pid is then gone — ``registry.classify`` already reads that as
``stale`` and ``reclaim`` already sweeps it, so no new vocabulary is needed. The
one asymmetry worth stating in the PR: until this ships, the automatic paths
refuse a loop-blocked runtime (``record-present``) and the only working recovery
is a hand-run ``lop stop --pid <pid> --force``.

THE FILE IS THE EVIDENCE, AND ITS ABSENCE MEANS NOTHING FIRED
------------------------------------------------------------
The dump is written to ``<log dir>/runtime-stall-<pid>.log`` — beside
``runtime.log``, in the directory ``paths.log_dir`` already resolves and
``lop mobile logs`` already points a reader at. Its EXISTENCE is the signal: a
clean exit (``disarm``) cancels the timer and removes the file, so a header-only
file that survives means the process died without disarming (a SIGKILL), and a
file carrying :data:`FIRED_MARKER` means this bound actually fired. That is the
same shape ``tests/e2e/watchdog.py`` uses, and for the same reason: faulthandler
writes with a raw descriptor from a C thread, so the header cannot be deferred
to the moment it fires.

NO CREDENTIAL CAN LAND IN IT. ``faulthandler`` prints frames — file, line and
function name — and never local variables, which is the property this file
depends on to be written into a log directory at all. That is asserted, not
assumed: see ``test_the_dump_carries_no_local_values``.

ONE PROCESS-GLOBAL TIMER, AND THE INTERLOCK IS THE ARMING SITE
-------------------------------------------------------------
``faulthandler``'s timer is process-wide, shared with ``tests/e2e/watchdog.py``
and ``tests/shard_stall_watchdog.py``, and displacing either would silently
remove a CI stage's only bound (pinned by
``tests/unit/test_shard_stall_watchdog.py::test_no_ci_job_runs_both_watchdogs_in_one_process``).
Nothing interlocks the timers here, and nothing has to: this module is armed
ONLY from the runtime child's own entry point
(``process.__main__`` — reachable by ``python -m ...``, which is how every
spawn names it: ``launch._spawn_runtime``, ``mobile/daemon.py``), while those two
harnesses arm theirs in the pytest process. ``arm`` is deliberately NOT called
from ``RuntimeServer`` or any constructor: ``start_in_process`` is used by
in-process hosts and by the test suite, so arming there would arm a timer inside
a pytest worker — which is exactly how the e2e stage would lose its bound. That
is an invariant with a test, not a comment: see
``tests/unit/session/runtime/test_runtime_stall_watchdog.py``.

OUT OF SCOPE, NAMED SO A READER DOES NOT ASSUME COVERAGE. The TUI hosts a
``RuntimeServer(kind="tui")`` registrant of its OWN in ``tui/app.py``; that
process is a person's terminal, so whether a stalled one should exit (and what
that does to the terminal) is its own design conversation, and a bound armed
from a library constructor there is the interlock hazard above. The detached
``-m`` child is the shape this covers — 16 of 16 records in the operator's own
store were ``kind=daemon`` when this was written.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import threading
import time
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

#: The bound, in seconds, as an operator or a test overrides it. Unset means
#: :data:`DEFAULT_STALL_S`; ``0``/``off`` disables the watchdog outright.
ENV_SECONDS = "LOP_RUNTIME_STALL_SECONDS"

#: Seconds of NO PROGRESS before the runtime dumps every thread and leaves.
#:
#: Sized against the two numbers the record already publishes, and against the
#: one measured false-positive class:
#:
#: * ``HEARTBEAT_INTERVAL_S`` (15 s) is how often a healthy runtime reports;
#:   ``HEARTBEAT_TIMEOUT_S`` (45 s) is when ANOTHER process starts calling it
#:   ``wedged``. Both are too tight to exit on — a ``wedged`` reading is
#:   degraded-responsiveness EVIDENCE, and this host has produced 105.8 s and
#:   205.8 s beat gaps on sessions whose CPU time was advancing (see
#:   ``types.HEARTBEAT_INTERVAL_S`` and ``registry.classify``). Killing a
#:   session for that would be the bug this module exists to prevent, wearing
#:   the other hat.
#: * 300 s sits above the largest legitimate gap ever measured here (205.8 s,
#:   1.5x) and far below the freezes the operator actually hit (1.5-7.2 h, and
#:   four of the five with ZERO writes in that window). A single synchronous
#:   step that holds the GIL for five unbroken minutes is not slow work; it is
#:   the state that made five sessions unreachable at once.
#:
#: A runtime is expected to be armed for its whole life, so this number is a
#: CEILING on one uninterrupted stall, not a limit on how long a session may
#: live or how long a turn may take: a turn that keeps yielding keeps beating.
DEFAULT_STALL_S = 300.0

#: The largest bound ``faulthandler`` can hold. Its timeout becomes a signed
#: 64-bit count of nanoseconds, so a larger value raises ``OverflowError:
#: timestamp out of range for platform time_t`` inside a C call — and here that
#: would happen in the runtime's own entry point. Any bound a person could mean
#: is minutes; a value at or above this is a typo and is treated as one.
MAX_BOUND_S = 2**63 / 1e9

#: Filename prefix of one process's dump. Keyed by pid, like every other runtime
#: artifact here: one process writes exactly one file, and a reader holding a
#: record's pid can name the file it would have written without being told
#: anything else. A pid recycled by a later runtime truncates that file, which
#: is the correct outcome — the evidence is about whoever holds the pid now.
DUMP_PREFIX = "runtime-stall"

#: What ``faulthandler`` itself writes when the C timer actually fires
#: (``Timeout (0:05:00)!``). Presence of this line is the single definition of
#: "this file is evidence", so a header-only file left by a SIGKILL is never
#: read as a fired bound.
FIRED_MARKER = "Timeout ("

#: The header written when the timer is ARMED. It reads like a claim but is not
#: one; :func:`fired_dumps` is what distinguishes the two.
ARM_MARKER = "[stall watchdog] "

#: The re-arm cadence the CALLERS are expected to keep, for the benefit of a
#: reader of this module only — nothing here enforces it. It is the serving
#: plane's ``HEARTBEAT_INTERVAL_S`` (15 s) and the workload loop's matching
#: tick, both of which sit an order of magnitude inside the bound.
BEAT_INTERVAL_HINT_S = 15.0


class _Armed:
    """One process's armed timer: the open file and the bound it was armed for.

    The handle is held OPEN for the process's life, and that is a requirement
    rather than tidiness: ``faulthandler`` writes with the bare descriptor, so
    closing the object would leave the timer firing at a closed fd.
    """

    __slots__ = ("path", "handle", "seconds", "pid")

    def __init__(self, path: Path, handle: IO[str], seconds: float, pid: int) -> None:
        self.path = path
        self.handle = handle
        self.seconds = seconds
        self.pid = pid


#: The process's armed timer, or ``None``. Module state rather than an object a
#: caller threads through, because the two beats come from two different loops
#: (a serving thread and the workload's own loop) that share no owner — and
#: because "is this armed" has to be answerable without one:
#: :func:`is_armed` is what pins that no library path armed anything.
_ARMED: _Armed | None = None

#: Serializes arm/beat/disarm. ``beat`` is called from two threads, and while
#: ``faulthandler`` tolerates a re-arm over a live timer (measured: no
#: exception, the new bound replaces the old), a ``cancel`` from one thread
#: interleaved with an ``arm`` from the other could leave the timer disarmed —
#: i.e. silently remove this runtime's own bound. The lock is held for two C
#: calls, once per beat.
_LOCK = threading.Lock()


def bound_seconds() -> float | None:
    """The configured bound, or ``None`` when the watchdog is switched off.

    Unreadable, non-numeric and out-of-range values fall back to
    :data:`DEFAULT_STALL_S` rather than raising: this runs in the runtime's
    entry point, where an exception would take the session down over a
    diagnostic, and the honest failure of a typo is the default bound rather
    than no bound at all. An explicit ``0``/``off`` is the one spelling that
    means "do not arm", which an operator (or a test that deliberately blocks a
    loop) needs.
    """
    raw = os.environ.get(ENV_SECONDS)
    if raw is None:
        return DEFAULT_STALL_S
    text = raw.strip().lower()
    if text in ("", "off", "no", "false"):
        return None
    try:
        seconds = float(text)
    except ValueError:
        logger.warning("%s=%r is not a number; using %.0fs", ENV_SECONDS, raw, DEFAULT_STALL_S)
        return DEFAULT_STALL_S
    if seconds == 0:
        return None
    if seconds < 0:
        # A NEGATIVE bound is a typo rather than a switch: ``0`` is the written
        # spelling for "do not arm", and nothing else can be meant by -1.
        logger.warning("%s=%r is negative; using %.0fs", ENV_SECONDS, raw, DEFAULT_STALL_S)
        return DEFAULT_STALL_S
    if seconds >= MAX_BOUND_S:
        logger.warning(
            "%s=%r is beyond the C timer's range; using %.0fs", ENV_SECONDS, raw, DEFAULT_STALL_S
        )
        return DEFAULT_STALL_S
    return seconds


def dump_path(pid: int | None = None, directory: Path | None = None) -> Path:
    """Where this process's dump goes: ``<log dir>/runtime-stall-<pid>.log``.

    A function rather than an inline join because three readers name the same
    file — the runtime's own log line, the tests, and an operator (or an agent)
    who has a record's pid and nothing else.
    """
    from local_operator.paths import log_dir

    base = directory if directory is not None else log_dir()
    return base / f"{DUMP_PREFIX}-{pid or os.getpid()}.log"


def arm(
    *,
    seconds: float | None = None,
    directory: Path | None = None,
    pid: int | None = None,
) -> bool:
    """Arm the process's stall bound. Returns whether it is now armed.

    Called from the runtime child's entry point and NOWHERE else — see the
    module docstring for why that is the interlock with the two pytest-side
    watchdog timers. Idempotent in the sense that re-arming replaces the bound,
    but it is written to be called once: a second call leaks the first handle.

    Never raises. A diagnostic that cannot be armed must leave the runtime
    otherwise untouched: an unwritable log directory is a reading of "no dump
    possible", not a reason to fail a session's boot.
    """
    global _ARMED
    with _LOCK:
        if _ARMED is not None:
            return True
        bound = bound_seconds() if seconds is None else seconds
        if bound is None:
            logger.info("stall watchdog disabled by %s", ENV_SECONDS)
            return False
        target = dump_path(pid, directory)
        try:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = target.open("w", encoding="utf-8")
        except OSError:
            logger.warning("stall watchdog could not open %s; no dump will be written", target)
            return False
        try:
            # THE HEADER IS WRITTEN BEFORE THE TIMER IS ARMED, never deferred:
            # faulthandler writes with a raw descriptor from a C thread, so the
            # file and its header have to exist first. Everything below this
            # line is what a reader finds when the bound fires, in this order.
            handle.write(
                f"{ARM_MARKER}pid {pid or os.getpid()} armed for {bound:g}s at {time.time():.0f} "
                f"({time.strftime('%Y-%m-%d %H:%M:%S')}); the runtime's own loops re-arm this "
                f"timer, so a dump below means this process made no progress for {bound:g}s.\n"
            )
            handle.flush()
            faulthandler.dump_traceback_later(bound, file=handle, exit=True)
        except (OSError, ValueError, OverflowError, RuntimeError):
            logger.warning("stall watchdog could not arm; no dump will be written", exc_info=True)
            try:
                handle.close()
                target.unlink()
            except OSError:
                pass
            return False
        _ARMED = _Armed(target, handle, bound, pid or os.getpid())
        return True


def beat(*args: object, **kwargs: object) -> None:
    """Report progress: restart the bound.

    Called by the runtime's own loops as they run — the serving plane's
    heartbeat and the workload loop's matching tick. A no-op when nothing is
    armed, which is what keeps this safe to call from an in-process host (a TUI
    or a test) that never armed the process timer.

    The signature takes and ignores arguments so a caller can pass whatever it
    has to hand (a loop, a handle) without this module learning about it.
    """
    with _LOCK:
        armed = _ARMED
        if armed is None:
            return
        try:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(armed.seconds, file=armed.handle, exit=True)
        except (OSError, ValueError, RuntimeError):
            # A beat that cannot re-arm must not take the loop down with it: the
            # timer is still armed from the previous beat, so the worst case is
            # a bound that expires sooner than intended.
            logger.warning("stall watchdog could not re-arm its timer", exc_info=True)


def disarm() -> None:
    """Cancel the bound and remove the file: this process left on its own terms.

    Removing the file is what makes its existence mean something — see the
    module docstring. Unconditionally safe to call, and called on every clean
    exit path.
    """
    global _ARMED
    with _LOCK:
        armed = _ARMED
        _ARMED = None
        if armed is None:
            return
        try:
            faulthandler.cancel_dump_traceback_later()
        except (OSError, RuntimeError):
            logger.debug("stall watchdog could not cancel its timer", exc_info=True)
        try:
            armed.handle.close()
        except OSError:
            pass
        try:
            armed.path.unlink()
        except OSError:
            pass


def is_armed() -> bool:
    """Whether THIS process holds the C timer through this module."""
    with _LOCK:
        return _ARMED is not None


def announce() -> None:
    """Name the dump file in the runtime's log, once the log exists.

    The arming happens in the entry point, before ``configure_file_logging``
    runs, so the path cannot be logged at that moment. It is logged here
    instead — from ``main``'s first lines — so an operator reading
    ``logs/runtime.log`` after a freeze finds the file to open next to it
    rather than having to reconstruct the naming convention. A no-op when
    nothing is armed, which is the whole in-process case.
    """
    with _LOCK:
        armed = _ARMED
        if armed is None:
            return
        logger.info(
            "stall watchdog armed: %.0fs of no progress dumps every thread to %s and exits",
            armed.seconds,
            armed.path,
        )


def fired_dumps(directory: Path | None = None) -> list[Path]:
    """Every dump this store holds that a bound actually fired into.

    ``FIRED_MARKER`` at the start of a line is the test, exactly as
    ``tests/e2e/watchdog.py`` defines it: a file whose header exists but whose
    timer never fired is a SIGKILL's leavings, not a freeze report.
    """
    from local_operator.paths import log_dir

    base = directory if directory is not None else log_dir()
    found: list[Path] = []
    try:
        candidates = sorted(base.glob(f"{DUMP_PREFIX}-*.log"))
    except OSError:
        return found
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(line.startswith(FIRED_MARKER) for line in text.splitlines()):
            found.append(path)
    return found
