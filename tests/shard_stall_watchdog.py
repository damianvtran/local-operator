"""Name the test a CI shard stalls on, instead of being cancelled by the cap.

WHY THIS EXISTS
---------------
``test (3.12, N)`` intermittently reaches the workflow's ``timeout-minutes: 20``
and is reported CANCELLED, which fails the PR. The signature is the same in
every occurrence and it is not a failure of any test: the run reaches 95-99%,
then produces **no output at all** for 5m55s-7m18s until the cap kills it. No
assertion, no traceback, no name.

That silence is the whole problem. ``-q`` prints one line per 72 completed
tests, so the final partial line is withheld until the LAST item in the batch
finishes: a single stuck test and a uniformly slow tail are indistinguishable
from the log. Measured over four cancels (runs 34545290432, 34546963294,
34547820936, 34551028930) the tail is 6 minutes with the run's own progress
information already exhausted, so the culpable item can only be recovered by
watching from *inside* the run rather than from its output.

The partition was ruled out first, with evidence: at ``origin/main`` the
cancelled shard was among the LIGHTEST by measured per-file cost while a
heavier shard finished in 14m14s, and the cancels have been observed on shard
indices 0, 1 and 4. So this module does not rebalance anything. It reports.

WHAT IT REPORTS, AND WHY THOSE TWO THINGS
-----------------------------------------
1. **The master names the tests that are in flight when progress stops.** The
   xdist controller receives ``pytest_runtest_logstart`` and every
   ``pytest_runtest_logreport``, so it knows which node ids have started and not
   finished. A node id turns "a 6-minute silent tail" into a test name. This is
   the part that answers the question, and it needs no cooperation from the
   stalled process at all -- which matters, because the stalled process is the
   thing that cannot tell anyone anything.
2. **Each worker dumps every thread's stacks with the C-level timer.** Named
   tests alone cannot distinguish "doing legitimate work slowly" from "parked
   inside a syscall", and the difference decides whether the fix is a bound or a
   deadlock. :mod:`tests.e2e.watchdog` already carries the why at length: only
   ``faulthandler.dump_traceback_later`` survives the #401 class of failure,
   because it arms a timer in a dedicated C thread that writes with ``write(2)``
   and needs neither the GIL nor a live interpreter. It is reused here rather
   than re-derived, and for the same reason the timer is armed around ONE test
   item rather than around the session: a session-wide timer cannot tell a stall
   from a healthy long run.

WHY IT IS SAFE TO LEAVE ON: IT NEVER FAILS A RUN
------------------------------------------------
Nothing here exits, kills or times out a process. ``exit=False`` on the worker
timer means a fired dump is a snapshot and the test keeps running; the master
prints and keeps polling. A run that is merely slow finishes normally and the
dump files are removed by the master's own cleanup, so a fired dump is the only
thing that survives -- the same "the file's existence is the signal" discipline
``tests.e2e.watchdog`` uses, and for the same reason: a diagnostic that can turn
a green run red gets disabled the first time it is wrong.

"Never fails a run" has to include failing to do its own work, because these
hooks run in every pytest process: no filesystem or timer operation here is
allowed to raise either. An unusable dump directory, or a variable unset
between install and the first test, disables the instrument and the run
continues -- measured before that was true, an occupied ``TMPDIR`` was an
``INTERNALERROR`` that failed a shard with ``no tests ran``. The reading is the
same one :func:`enabled_seconds` takes of a malformed bound: disable, never fail.

The workspace is deliberately NOT under ``tmp_path``: the master and its workers
are separate processes and the controller has to be able to read what a wedged
worker wrote, which means a path both can compute from the environment. It is
also fixed rather than unique per session so that the workflow's ``always()``
step can find the files after a job is cancelled, when no process is left to
tell it where they are.

WHAT A CANCELLED JOB LEAVES BEHIND
----------------------------------
The controller prints straight into the job log, but a job can also die with no
report at all, so the workflow keeps an ``always()`` step that prints the dump
directory. That step asks :func:`report_dumps` to do the printing rather than
``cat``-ing the files itself, because a worker writes its header at ARM time,
one per test, and the raw file is therefore mostly claims about tests that ran
in milliseconds. The first real CI occurrence (job 103135419093, shard 2) is the
measurement: the raw step printed 3088 ``exceeded 240s`` lines across four
files, exactly four of them backed by a real :data:`FIRED_MARKER`, and the one
genuine timeout was indistinguishable among the rest. :data:`FIRED_MARKER` is
the single source of truth for "this is evidence", and it is read through one
function (:func:`_fired_lines`) that the live controller report, the cleanup and
this step's report all call -- so the rule cannot drift between them.

WHAT IT DOES NOT INSTRUMENT, AND WHY THAT IS ACCEPTABLE
-------------------------------------------------------
Only a worker arms a C timer; the controller's sole witness is its Python poll
thread. That is weaker than it sounds -- the poll thread does the printing too,
so a worker wedged inside a syscall is fully covered without the controller's
main thread making progress. The mirror case, a park *inside the controller
process*, is left unwired deliberately, but for a narrower reason than
"impossible". A report-time arm WOULD cover a park that releases the GIL: the
report is emitted by the reporter thread, which therefore has a live frame at
that moment, and the dump would add the parked main thread's stacks. What it
cannot cover is a park that holds the GIL -- but then no report is emitted
either, which is the separately acknowledged "no report at all" path. So the
report-time arm buys the GIL-releasing half of a case the shard job does not
reach (the controller runs no test body; its only Python is pytest/xdist
orchestration), at the cost of a second timer and a new file class in the
report. That trade is not obviously worth taking, but it is a judgement, not an
impossibility -- if a controller-side park is ever seen in CI, a report-time arm
is the repair. A wall-clock controller timer armed at startup is NOT an
alternative: it would fire on every healthy long run and keep a file, breaking
the rule that a file's existence means something fired.

THE OTHER C TIMER IN THIS REPO
------------------------------
``faulthandler``'s timer is process-global, so ``tests.e2e.watchdog.bounded``
-- a second C timer, for tests whose failure mode is a hang -- displaces the
worker timer for the rest of whichever test opens it, leaving that test with no
stacks. Nothing interlocks them because nothing runs them together: the shard
job sets :data:`ENV_SECONDS` and deselects ``e2e``, while the ``tui-e2e`` job
runs ``-n0`` (one process, no worker branch) without the variable. Both halves
of that invariant are pinned by the unit guards rather than left to prose.

WHERE IT IS ENABLED
-------------------
Only when ``LOCAL_OPERATOR_SHARD_STALL_SECONDS`` is set, which the ``test`` job
in ``.github/workflows/ci.yml`` does and nothing else does. The default absence
is what keeps a developer run and the ``-n0`` e2e stage (which has its own,
tighter bound) untouched; it also means the number is a CI budget rather than a
wall-clock assertion about anyone's machine, which is the only kind of number
AGENTS.md allows to be calibrated from CI.
"""

from __future__ import annotations

import contextlib
import faulthandler
import math
import os
import sys
import threading
import time
from pathlib import Path

#: Seconds of NO COMPLETED TEST before the controller reports. Set by the shard
#: job. Unset (and therefore inert) everywhere else.
ENV_SECONDS = "LOCAL_OPERATOR_SHARD_STALL_SECONDS"

#: Shared by every xdist worker and readable by the controller -- see the module
#: docstring for why this is not a ``tmp_path`` fixture directory.
DUMP_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "lo-shard-stall"

#: ``faulthandler`` writes this when the C timer actually fires
#: (``Timeout (0:04:00)!``). Distinguishing a real dump from a header written at
#: arm time is what lets cleanup delete the harmless ones and keep the evidence.
FIRED_MARKER = "Timeout ("

#: The header a worker writes when it ARMS the timer, i.e. when the test starts.
#: It reads like a timeout claim but is not one; :func:`report_dumps` recognises
#: it so a reader is never handed "<nodeid> exceeded 240s" for a test that ran in
#: milliseconds. Kept here rather than spelled out at the write site and again in
#: the report so the two cannot drift apart.
ARM_MARKER = "[shard stall] "

#: How often the controller repeats an ongoing report. A stall is reported more
#: than once on purpose: the cap can arrive mid-stall, and the last report
#: before it is the one a reader will see.
REPORT_INTERVAL_S = 30.0

#: The largest bound the C timer can hold. ``faulthandler``'s timeout becomes a
#: signed 64-bit count of nanoseconds, so 2**63 ns is the ceiling: a larger (or
#: infinite) value raises ``OverflowError: timestamp out of range for platform
#: time_t`` the first time a test starts, i.e. inside a pytest hook, where it is
#: an ``INTERNALERROR`` that fails the shard with ``no tests ran``. Any bound CI
#: could mean is minutes, so a value at or above this is a typo and is treated
#: as one -- see :func:`enabled_seconds`.
MAX_BOUND_S = 2**63 / 1e9

#: Poll granularity. Small enough that the first report lands promptly, large
#: enough that the thread is invisible next to a 4-worker test run.
POLL_INTERVAL_S = 2.0

#: Cap on the lines of stack printed per worker. One snapshot of a busy
#: xdist worker (which carries the session runtime, the viewer endpoint and
#: execnet's own threads) is comfortably under this; the cap only exists so a
#: pathological dump cannot bury the report.
STACK_EXCERPT_LINES = 400


def _fired_lines(text: str) -> list[str] | None:
    """The first complete stack dump in ``text``, or ``None`` if none fired.

    This is the single definition of "this file is evidence", so the live
    controller report, the cleanup and the workflow's report all agree by
    construction.

    The marker is matched at the START OF A LINE, not anywhere in the text.
    ``faulthandler`` always writes it as its own line, while a worker's header
    embeds a node id -- a path plus a test name -- which in principle could
    contain the literal marker text; anchoring the match is what makes it
    impossible for a header-only file to be classified as fired.

    ``repeat=True`` appends another full dump every interval, so a file that
    tripped early holds several near-identical snapshots. One snapshot is the run
    of lines from its marker up to the next marker (or the end), and taking
    exactly that keeps it whole: the thread that matters is not at a predictable
    end, because the session runtime and the viewer endpoint have threads of
    their own and faulthandler groups them its own way. The result is capped at
    :data:`STACK_EXCERPT_LINES` so a file full of repeated dumps cannot push the
    live report past the log's tail.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(FIRED_MARKER)), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and not lines[end].startswith(FIRED_MARKER):
        end += 1
    return lines[start:end][:STACK_EXCERPT_LINES]


def enabled_seconds() -> float | None:
    """The configured stall bound, or ``None`` when this module is inert.

    A malformed value disables the module rather than raising: this runs inside
    ``pytest_configure``, so a typo in a workflow env var must not be able to
    take out the whole suite. It also cannot silently become a *shorter* bound
    than intended, because a value that does not parse is not used at all.

    "Malformed" includes a value that parses but no timer can hold. ``inf`` and
    ``nan`` parse as floats, and any finite value at or above :data:`MAX_BOUND_S`
    overflows the ``time_t`` the C timer converts to -- an ``INTERNALERROR`` from
    inside the first test's hook, which is the same failure this function exists
    to prevent, arriving through the one door a parse-only check leaves open.
    """
    raw = os.environ.get(ENV_SECONDS, "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds <= 0 or seconds >= MAX_BOUND_S:
        return None
    return seconds


def _dump_path(tag: str) -> Path:
    """The dump path for ``tag``; best-effort ensures the directory.

    The ``mkdir`` is guarded even though its result is ignored, because this is
    reached from ``pytest_configure`` (via :class:`_WorkerTimer`), where an
    exception is an ``INTERNALERROR`` that fails the whole shard before a single
    test runs. An unwritable ``TMPDIR``, or a file occupying the path, was
    measured to do exactly that -- ``NotADirectoryError`` and ``FileExistsError``
    respectively -- which is the one way a diagnostic could turn a green run red.
    Failing to create it instead disables the writes that follow, which is the
    same "disable, never fail" reading :func:`enabled_seconds` takes of a
    malformed bound.
    """
    with contextlib.suppress(OSError):
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
    return DUMP_DIR / f"{tag}.log"


class _WorkerTimer:
    """The C-level per-item timer, armed in an xdist worker.

    ``repeat=True`` because the bound is a floor, not a schedule: a test that
    runs for three times the bound should leave three snapshots, and the last
    one before the cap is the one with the most useful stacks.

    The bound is captured once, here, rather than re-read from the environment
    for every test: the value is known to be usable at install time, whereas
    ``note_start`` runs inside ``pytest_runtest_logstart``, where an unset
    variable reaching ``dump_traceback_later(None)`` raised ``TypeError`` and
    became an ``INTERNALERROR``.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._handle = None
        self._path = _dump_path(f"worker-{os.getpid()}")
        self._armed = False
        # Set when the dump target itself is unusable, so nothing retries a
        # failed open once per test for the rest of the run.
        self._disabled = False

    def arm(self, nodeid: str) -> None:
        """Start the C timer for ``nodeid``; disable quietly if it cannot.

        Every failure here is swallowed rather than raised. This runs inside a
        pytest hook, so an exception is an ``INTERNALERROR`` that fails the
        shard -- the diagnostic is not allowed to be the reason a run goes red,
        and a run whose dump directory is unusable is still a run worth having.

        ``OverflowError`` is in the tuple even though :func:`enabled_seconds`
        already refuses a bound no timer can hold: the value can also arrive
        here from a direct construction, and this is the one call whose failure
        mode is the whole shard. Belt and braces, because the cost of the extra
        name is zero and the cost of missing it is a red run.
        """
        if self._armed or self._disabled:
            return
        try:
            # The handle must exist and stay open for the life of the process:
            # the dump is written from a C thread with a raw descriptor, so the
            # file cannot be opened at the moment it fires.
            if self._handle is None or self._handle.closed:
                self._handle = self._path.open("w", encoding="utf-8")
            self._handle.write(
                f"{ARM_MARKER}{nodeid} exceeded {self.seconds:g}s; every thread follows.\n"
            )
            self._handle.flush()
            faulthandler.dump_traceback_later(
                self.seconds, file=self._handle, repeat=True, exit=False
            )
        except (OSError, ValueError, TypeError, OverflowError):
            self._disabled = True
            return
        self._armed = True

    def disarm(self) -> None:
        if not self._armed:
            return
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            faulthandler.cancel_dump_traceback_later()
        self._armed = False


class _Controller:
    """Tracks in-flight tests on the xdist controller and reports a stall once.

    Thread-safe because the tracker is mutated by pytest hooks on the main
    thread while the reporter reads it from its own.
    """

    def __init__(self, seconds: float, sink=None) -> None:
        self.seconds = seconds
        self._lock = threading.Lock()
        self._in_flight: dict[str, float] = {}
        self._last_progress = time.monotonic()
        self._reported_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Injectable so the local proof can capture a report without racing the
        # terminal. The default writes to the controller's stderr, which is
        # measured to land in the CI job log WHILE a worker is still busy --
        # pytest's fd capture is suspended on the controller, which is why the
        # `-q` progress of a cancelled job is readable at all. That measurement
        # is what makes a report-only design viable here: nothing has to be
        # printed by a later step that a cancelled job would never run.
        self._sink = sink or (lambda text: print(text, file=sys.stderr, flush=True))

    def started(self, nodeid: str) -> None:
        with self._lock:
            self._in_flight[nodeid] = time.monotonic()

    def note(self, nodeid: str, when: str) -> None:
        """A report arrived. Any report is progress; only teardown completes.

        The distinction is load-bearing and was measured the hard way: xdist
        sends a report with ``when="setup"`` as soon as a test's fixtures are
        done and BEFORE its call phase runs. Counting that as completion
        emptied the in-flight set for the whole duration of the test body, so a
        real stall reported "0 still in flight" -- the exact opposite of the
        answer wanted, and it read as a healthy quiet run rather than a bug.
        """
        with self._lock:
            self._last_progress = time.monotonic()
            if when == "teardown":
                self._in_flight.pop(nodeid, None)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll, name="shard-stall", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _poll(self) -> None:
        while not self._stop.wait(POLL_INTERVAL_S):
            self.report_if_stalled()

    def report_if_stalled(self) -> bool:
        """Emit one report per :data:`REPORT_INTERVAL_S` while a stall persists.

        Returns whether a report was emitted, which is what the local proof
        asserts on; a real run only cares about the text reaching the job log.
        """
        now = time.monotonic()
        with self._lock:
            idle = now - self._last_progress
            # Reported even with an EMPTY in-flight set, which is not an edge
            # case but a distinct diagnosis: the controller stops hearing from a
            # worker it believes is running something, so an empty set beside a
            # stall means the worker died or wedged without reporting, not that
            # the suite finished. Suppressing the report there would hide the
            # only case where the controller is the last process alive.
            if idle < self.seconds:
                return False
            if now - self._reported_at < REPORT_INTERVAL_S:
                return False
            self._reported_at = now
            in_flight = sorted(self._in_flight.items(), key=lambda kv: kv[1])
        self._sink(self._format(idle, in_flight))
        return True

    def _format(self, idle: float, in_flight: list[tuple[str, float]]) -> str:
        now = time.monotonic()
        lines = [
            "",
            "!" * 72,
            f"SHARD STALL: no test has completed for {idle:.0f}s "
            f"(bound {self.seconds:g}s). {len(in_flight)} still in flight:",
        ]
        for nodeid, started in in_flight:
            lines.append(f"  {now - started:7.0f}s  {nodeid}")
        lines.extend(self._stack_excerpts())
        lines.append("!" * 72)
        lines.append("")
        return "\n".join(lines)

    def _stack_excerpts(self) -> list[str]:
        """The head of each worker dump that actually fired.

        Read on the controller rather than printed by the worker, because the
        worker is the process that may be unable to write to the job log at all.
        A file without :data:`FIRED_MARKER` is a header from an armed-but-not-
        fired timer and is skipped, so the report never claims a timeout for a
        test that was merely running.
        """
        out: list[str] = []
        try:
            candidates = sorted(DUMP_DIR.glob("worker-*.log"))
        except OSError:
            return out
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = _fired_lines(text)
            if lines is None:
                continue
            out.append(f"--- {path} (first snapshot, {len(lines)} lines) ---")
            out.extend(lines)
        return out

    def cleanup(self) -> None:
        """Remove armed-but-unfired dumps; keep anything that actually fired.

        Runs on the controller, which outlives its workers. A fired dump is
        evidence and is kept even on a green run -- a test that took four
        minutes and still passed is worth knowing about.
        """
        try:
            paths = sorted(DUMP_DIR.glob("*.log"))
        except OSError:
            return
        for path in paths:
            try:
                if _fired_lines(path.read_text(encoding="utf-8", errors="replace")) is not None:
                    continue
            except OSError:
                pass
            try:
                path.unlink()
            except OSError:
                pass


#: One controller per controller process; None on a worker.
_CONTROLLER: _Controller | None = None
#: One timer per worker process; None on the controller.
_WORKER: _WorkerTimer | None = None


def install(config) -> None:
    """Wire the module up in whichever process this is. Idempotent.

    ``hasattr(config, "workerinput")`` is xdist's own discriminator between a
    worker and the controller, and it is the only one that is correct *before*
    a session exists -- the two processes need different instrumentation (one
    watches, the other is watched), so the branch has to happen here rather
    than inside a hook.
    """
    global _CONTROLLER, _WORKER
    seconds = enabled_seconds()
    if seconds is None:
        return
    if hasattr(config, "workerinput"):
        if _WORKER is None:
            _WORKER = _WorkerTimer(seconds)
    elif _CONTROLLER is None:
        _CONTROLLER = _Controller(seconds)
        _CONTROLLER.start()


def note_start(nodeid: str) -> None:
    if _CONTROLLER is not None:
        _CONTROLLER.started(nodeid)
    elif _WORKER is not None:
        # Inertness is decided once, at install: re-reading the variable here
        # and passing it on was how an unset value could reach
        # ``dump_traceback_later(None)`` from inside a hook.
        _WORKER.arm(nodeid)


def note_report(report) -> None:
    if _CONTROLLER is not None:
        _CONTROLLER.note(report.nodeid, report.when)
    elif _WORKER is not None and report.when == "teardown":
        _WORKER.disarm()


def shutdown() -> None:
    if _CONTROLLER is not None:
        _CONTROLLER.stop()
        _CONTROLLER.cleanup()
    if _WORKER is not None:
        _WORKER.disarm()


def _install_controller_for_test(seconds: float, sink) -> _Controller:
    """Test seam: a controller wired to a capturing sink, no env var needed."""
    global _CONTROLLER
    _CONTROLLER = _Controller(seconds, sink=sink)
    return _CONTROLLER


def _armed_nodeids(text: str) -> list[str]:
    """Node ids from arm-time headers, in the order the worker armed them.

    The last entry is the test that worker had started most recently, which is
    the most useful thing an unfired file can say once the job is gone. Reading
    stops at the first fired marker, because a worker keeps arming timers after
    a dump: on a fired file the last id before the marker is the test the timer
    fired on, while ids after it belong to later tests and would misattribute
    the snapshot.
    """
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith(FIRED_MARKER):
            break
        if line.startswith(ARM_MARKER):
            out.append(line[len(ARM_MARKER) :].split(" exceeded ", 1)[0].strip())
    return out


def _report_file(path: Path) -> list[str]:
    """One dump file for :func:`report_dumps`, labelled for what it is.

    Every block names the test it is about, because that is the whole point of
    the instrument: the stacks say what was parked, but a reader coming to a
    cancelled run needs the node id first. The arm headers are the only record
    of it, and the file already has them -- in the fired case the header above
    the marker is the test that was running when the timer fired, in the unfired
    case the last one is the test in flight at the cancel.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"===== {path} : unreadable ({exc.__class__.__name__}) ====="]
    armed = _armed_nodeids(text)
    snapshot = _fired_lines(text)
    if snapshot is not None:
        who = armed[-1] if armed else "(unrecorded)"
        return [
            f"===== {path} (fired, first snapshot, {len(snapshot)} lines) =====",
            f"in flight when the timer fired: {who}",
            *snapshot,
        ]
    if not armed:
        return [f"===== {path} : empty (armed, no test recorded) ====="]
    return [
        f"===== {path} : timer armed but never fired "
        f"({len(armed)} tests started in this worker) =====",
        f"in flight at cancel, not fired: {armed[-1]}",
    ]


def report_dumps(directory: Path | None = None) -> str:
    """A filtered, human-readable report over the dump files in ``directory``.

    This is what the workflow's ``always()`` step prints, and it lives here
    rather than in the YAML on purpose. The step used to ``cat`` every ``*.log``,
    which contradicted this module's own discipline: a worker writes its header
    at ARM time, one per test, so a cancelled shard's files held thousands of
    lines all reading "<nodeid> exceeded 240s" with the one real timeout
    indistinguishable among them (measured: 3088 lines, 4 of them fired, on this
    module's first real CI occurrence). :data:`FIRED_MARKER` is the single source
    of truth for "this snapshot is evidence", read through one function
    (:func:`_fired_lines`) that the live report, the cleanup and this report all
    call.

    An empty or missing directory is not an error: the step runs on every shard,
    including the ones with nothing to report.
    """
    root = DUMP_DIR if directory is None else directory
    try:
        paths = sorted(root.glob("*.log"))
    except OSError:
        paths = []
    if not paths:
        return "no shard stall report: nothing stalled long enough to trip the watchdog\n"
    blocks: list[str] = []
    for path in paths:
        blocks.extend(_report_file(path))
    return "\n".join(blocks) + "\n"


def main(argv: list[str] | None = None) -> int:
    """``python -m tests.shard_stall_watchdog [dir]`` -- print the dumps.

    The optional directory lets the workflow point at its own
    ``${TMPDIR:-/tmp}/lo-shard-stall``; the default is :data:`DUMP_DIR`, computed
    from the same variable, so the two agree without the caller knowing the name.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    sys.stdout.write(report_dumps(Path(args[0]) if args else None))
    return 0


if __name__ == "__main__":  # pragma: no cover - operator/CI aid
    raise SystemExit(main())
