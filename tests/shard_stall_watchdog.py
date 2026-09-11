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

The workspace is deliberately NOT under ``tmp_path``: the master and its workers
are separate processes and the controller has to be able to read what a wedged
worker wrote, which means a path both can compute from the environment. It is
also fixed rather than unique per session so that the workflow's ``always()``
step can find the files after a job is cancelled, when no process is left to
tell it where they are.

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

import faulthandler
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

#: How often the controller repeats an ongoing report. A stall is reported more
#: than once on purpose: the cap can arrive mid-stall, and the last report
#: before it is the one a reader will see.
REPORT_INTERVAL_S = 30.0

#: Poll granularity. Small enough that the first report lands promptly, large
#: enough that the thread is invisible next to a 4-worker test run.
POLL_INTERVAL_S = 2.0

#: Cap on the lines of stack printed per worker. One snapshot of a busy
#: xdist worker (which carries the session runtime, the viewer endpoint and
#: execnet's own threads) is comfortably under this; the cap only exists so a
#: pathological dump cannot bury the report.
STACK_EXCERPT_LINES = 400


def _first_snapshot(text: str) -> list[str]:
    """The first complete stack dump in ``text``, as lines.

    ``repeat=True`` appends another full dump every interval, so a file that
    tripped early holds several near-identical snapshots. Taking everything
    would repeat the same stacks until the cap, and taking a fixed number of
    lines from one end is not enough either: the thread that matters is not at
    a predictable end -- faulthandler groups threads its own way, and the
    worker under test is usually NOT the first block, because the session
    runtime and the viewer endpoint have threads of their own. Splitting on the
    marker keeps ONE snapshot whole, so every thread is present.
    """
    parts = text.split(FIRED_MARKER)
    if len(parts) < 2:
        return text.splitlines()
    body = FIRED_MARKER + parts[1]
    lines = body.splitlines()
    return lines[:STACK_EXCERPT_LINES]


def enabled_seconds() -> float | None:
    """The configured stall bound, or ``None`` when this module is inert.

    A malformed value disables the module rather than raising: this runs inside
    ``pytest_configure``, so a typo in a workflow env var must not be able to
    take out the whole suite. It also cannot silently become a *shorter* bound
    than intended, because a value that does not parse is not used at all.
    """
    raw = os.environ.get(ENV_SECONDS, "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _dump_path(tag: str) -> Path:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    return DUMP_DIR / f"{tag}.log"


class _WorkerTimer:
    """The C-level per-item timer, armed in an xdist worker.

    ``repeat=True`` because the bound is a floor, not a schedule: a test that
    runs for three times the bound should leave three snapshots, and the last
    one before the cap is the one with the most useful stacks.
    """

    def __init__(self) -> None:
        self._handle = None
        self._path = _dump_path(f"worker-{os.getpid()}")
        self._armed = False

    def arm(self, nodeid: str, seconds: float) -> None:
        if self._armed:
            return
        # The handle must exist and stay open for the life of the process: the
        # dump is written from a C thread with a raw descriptor, so the file
        # cannot be opened at the moment it fires.
        if self._handle is None or self._handle.closed:
            self._handle = self._path.open("w", encoding="utf-8")
        self._handle.write(f"[shard stall] {nodeid} exceeded {seconds:g}s; every thread follows.\n")
        self._handle.flush()
        faulthandler.dump_traceback_later(seconds, file=self._handle, repeat=True, exit=False)
        self._armed = True

    def disarm(self) -> None:
        if not self._armed:
            return
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
            if FIRED_MARKER not in text:
                continue
            lines = _first_snapshot(text)
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
                if FIRED_MARKER in path.read_text(encoding="utf-8", errors="replace"):
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
            _WORKER = _WorkerTimer()
    elif _CONTROLLER is None:
        _CONTROLLER = _Controller(seconds)
        _CONTROLLER.start()


def note_start(nodeid: str) -> None:
    if _CONTROLLER is not None:
        _CONTROLLER.started(nodeid)
    elif _WORKER is not None:
        _WORKER.arm(nodeid, enabled_seconds())  # type: ignore[arg-type]


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
