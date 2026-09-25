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
worker wrote, which means a path both can compute from the environment. Its
ROOT is fixed for the same reason -- the workflow's ``always()`` step finds the
files after a job is cancelled, when no process is left to tell it where they
are -- while the run's own directory beneath it is unique (see
:func:`run_dir`), so two concurrent runs on one host never read, or delete, each
other's evidence.

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

THE OTHER C TIMERS IN THIS REPO
------------------------------
``faulthandler``'s timer is process-global, so ``tests.e2e.watchdog.bounded``
-- a second C timer, for tests whose failure mode is a hang -- displaces the
worker timer for the rest of whichever test opens it, leaving that test with no
stacks. Nothing interlocks them because nothing runs them together: the shard
job sets :data:`ENV_SECONDS` and deselects ``e2e``, while the ``tui-e2e`` job
runs ``-n0`` (one process, no worker branch) without the variable. Both halves
of that invariant are pinned by the unit guards rather than left to prose.

A THIRD armed timer exists in the product -- ``session/runtime/stall_watchdog``,
the runtime's own bound on a wedged interpreter -- and it is outside that
invariant for a structural reason rather than a job's configuration: it arms in
the runtime child's ``__main__`` branch, which only ``python -m`` reaches, so no
pytest process can arm it however the CI jobs are arranged. That placement is
the interlock (arming it from a runtime constructor would let any in-process
boot in this suite displace the worker timer), and it is pinned as a fact about
the source AND about behaviour in
``tests/unit/session/runtime/test_runtime_stall_watchdog.py``.

WHERE IT IS ENABLED
-------------------
``LOCAL_OPERATOR_SHARD_STALL_SECONDS`` is the named bound, and the ``test`` job
in ``.github/workflows/ci.yml`` sets it. When it is absent the module falls back
to :data:`LOCAL_DEFAULT_SECONDS` on a host that is not CI, because the same
silence that costs a CI shard costs a local whole-tree run an evening: measured
2026-09-24, a local run with no bound at all was watched by hand for four hours
and killed mid-progress, and its log tail (a wall-clock ``timeout`` killing the
group, then orphaned workers finishing their own ``pytest_sessionfinish`` against
a dead channel) read exactly like the CI stall signature -- a ``PluggyTeardownRaisedWarning``
and an ``OSError: cannot send (already closed?)`` with no test named. Nothing was
stuck; nobody could tell that from the artefact. The local default is a REPORTING
bound (:data:`LOCAL_DEFAULT_SECONDS` explains the number), and it stays off on CI
that did not ask for it, which is what keeps the ``-n0`` e2e stage -- its own
tighter bound, no worker branch -- out of it.

THE OPT-IN PER-TEST HARD BOUND
------------------------------
Reporting alone cannot end a run that is genuinely parked, so the worker's C
timer can be armed with ``exit=True`` instead: make the process die, and let
xdist's own crash reporting name the item. That is OFF everywhere by default and
the timers are report-only (``exit=False``) unless
``LOCAL_OPERATOR_TEST_TIMEOUT_SECONDS`` asks for it, because a fired bound kills
a worker carrying unrelated tests -- the same reason the e2e stage runs ``-n0``
(see below). ``pytest-timeout`` is not a dependency here and is not added for
this: :mod:`faulthandler` already has the only mechanism that survives the #401
class of park, and it is the one this module is built on.

The per-test bound is SIZED, never typed in: :func:`sized_bound_seconds` reads
``tests/durations.json`` -- the same committed per-file weights the CI sharder
balances on -- and allows :data:`BOUND_SLACK` times a whole FILE's measured total,
floored at :data:`BOUND_FLOOR_S`. Read that as what it is: a hang catcher, not a
slow-test detector. A single item cannot trip it unless it alone outran four
times what its entire file was measured to cost, which is why it is safe against
a legitimately slow test under fleet load and why it still catches a park (a park
does not finish at all).
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

#: Seconds of NO COMPLETED TEST before the controller reports. Set by the shard
#: job; absent on a developer machine, where :func:`local_seconds` supplies the
#: fallback above.
ENV_SECONDS = "LOCAL_OPERATOR_SHARD_STALL_SECONDS"

#: The local run's own bound, and its off switch. Absent on a local run means
#: :data:`LOCAL_DEFAULT_SECONDS`; a value that means OFF disables the instrument
#: entirely (no thread, no dump directory), which a debugger session needs.
LOCAL_ENV_SECONDS = "LOCAL_OPERATOR_LOCAL_STALL_SECONDS"

#: The bound a LOCAL run reports at when nothing overrides it: 900s (15 min) of
#: no completed test.
#:
#: Sized ABOVE every legitimate item rather than tightly, because the cost of
#: being too tight is a false alarm on a healthy run -- which is how a diagnostic
#: gets switched off and then is not there when it matters. The reference point is
#: The worst legitimate test in a full local run, from the C-timer comment in
#: ``.github/workflows/ci.yml`` (which sizes the e2e bound from the same figure),
#: is 81s; this host runs the suite at load average 80-200, so even a 5x
#: load-slowed item is ~400s. CI's 240s is NOT the local number and is not a
#: candidate for one: it is priced against a 20-minute job cap that a local run
#: does not have.
LOCAL_DEFAULT_SECONDS = 900.0

#: Opt-in per-test HARD bound (kills the process, naming the item through xdist's
#: crash report). Unset -> report-only everywhere, which is also what CI runs.
#: ``1``/``true``/``yes`` -> enable it, sized by :func:`sized_bound_seconds`; a
#: positive number -> that many seconds for every test. See the module docstring.
TEST_TIMEOUT_ENV = "LOCAL_OPERATOR_TEST_TIMEOUT_SECONDS"

#: Values of an env flag that mean "not set" / "off". Same reading as the root
#: conftest's ``_FALSY_ENV_VALUES``; kept in step with it by the guard in
#: ``tests/unit/test_shard_stall_watchdog.py``.
FALSY_ENV_VALUES = frozenset({"0", "false", "no", "off"})

#: The committed per-file weights the CI sharder balances on, reused here as the
#: input to :func:`sized_bound_seconds`. It is measured on a developer machine
#: (see the manifest's own comment), which is exactly the right calibration for a
#: bound that has to hold on one.
MANIFEST_PATH = Path(__file__).with_name("durations.json")

#: How many times a whole FILE's measured total one item may take before the
#: hard bound fires. 4x a file total is deliberately generous: the manifest is
#: per FILE, so an item's own cost is not in the data, and the bound must not
#: fail a test whose file was measured on an idle host while the fleet runs at
#: load 200.
BOUND_SLACK = 4.0

#: Floor for the hard bound, in seconds: a file whose measured total is 1.5s still
#: gets a bound in minutes, because a park is measured in minutes and a small
#: file's tests still have to boot a Textual app.
BOUND_FLOOR_S = 300.0

#: The base directory every run's dumps live under. The workflow's report step
#: points here, and :func:`report_dumps` walks one level into it, so the CI reader
#: still sees every shard's evidence while each run stays in its own directory.
DUMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "lo-shard-stall"

#: The variable the controller exports so that its workers write into THIS run's
#: directory. A worker process is spawned after ``pytest_configure`` and inherits
#: the controller's environment -- measured on this tree, a value set in the
#: controller's configure was visible inside a ``-n 2`` worker -- which is what
#: makes one directory per run possible without a lock file shared between runs.
ENV_RUN_DIR = "LOCAL_OPERATOR_SHARD_STALL_DIR"

#: This process's run directory, resolved at install and cached. ``None`` until
#: then, and on any process that never installed.
_RUN_DIR: Path | None = None


def run_dir() -> Path:
    """The dump directory for THIS run.

    One directory per run rather than one per user. This module is on by default
    for local runs now and this host runs ~25 concurrent sessions, where a shared
    directory makes two failures reachable at once: a report can read a
    concurrent run's stack excerpt as its own evidence, and a run that finishes
    can delete an armed dump that a concurrent run's worker still holds open and
    is still writing to -- so the other run's fired dump lands on an unlinked
    inode and its report loses the one thing that decides bound-versus-deadlock.
    Both halves were measured against the shared directory before this change.

    The pid fallback is for a process that never installed (an unusual gateway, a
    direct import in a test): it still cannot collide with a concurrent run.
    """
    if _RUN_DIR is not None:
        return _RUN_DIR
    explicit = os.environ.get(ENV_RUN_DIR, "").strip()
    if explicit:
        return Path(explicit)
    return DUMP_ROOT / f"pid-{os.getpid()}"


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


def local_seconds(on_ci: bool) -> float | None:
    """The stall bound a LOCAL run gets, or ``None`` when the instrument is off here.

    Resolution, first match wins:

    * ``LOCAL_OPERATOR_LOCAL_STALL_SECONDS`` parsing as a usable bound -> that
      value.
    * a value that means OFF (``0``/``false``/``no``/``off``) -> ``None``. The
      off switch has to exist: this instrument now arms itself on every local
      run, and an operator stepping through a debugger has tests that look
      stalled for as long as they sit there.
    * unset, on CI -> ``None``. A CI job that did not ask for the shard bound
      stays exactly as inert as it was before this default existed. That is not
      politeness: it is what keeps the ``-n0`` e2e stage (one process, its own
      tighter bound) out of this module's worker branch, because CI's own
      explicit value is the only thing that can enable it there.
    * unset, not CI -> :data:`LOCAL_DEFAULT_SECONDS`.

    ``on_ci`` is passed in rather than read here because the caller already owns
    that judgement: the root conftest denies ``CI`` when its own bash tool set
    it, so a subagent running the suite on the operator's laptop must NOT be told
    it is on a dedicated runner -- a mistake that hook has made once already (see
    ``_in_agent_shell`` in ``conftest.py``).
    """
    raw = os.environ.get(LOCAL_ENV_SECONDS, "").strip()
    fallback = None if on_ci else LOCAL_DEFAULT_SECONDS
    if raw:
        if raw.lower() in FALSY_ENV_VALUES:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            return fallback
        if not math.isfinite(seconds) or seconds <= 0 or seconds >= MAX_BOUND_S:
            return fallback
        return seconds
    return fallback


def _manifest_seconds(nodeid: str) -> float | None:
    """The measured seconds of ``nodeid``'s FILE, or ``None`` when unknown.

    Read through a fresh read rather than a cached import-time load: this runs
    once per test start in a worker, the file is ~30 KB, and a run that someone
    regenerates mid-suite (``scripts/gen_test_durations.py``) should be believed
    rather than shadowed by a snapshot taken at process start. Any failure -- a
    missing manifest, a malformed one, a file pytest is running but the manifest
    does not mention -- is ``None``, i.e. "use the floor", never an exception
    from inside a pytest hook.
    """
    path = nodeid.split("::", 1)[0]
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    durations = raw.get("durations") if isinstance(raw, dict) else None
    if not isinstance(durations, dict):
        return None
    value = durations.get(path)
    return float(value) if isinstance(value, (int, float)) else None


def sized_bound_seconds(nodeid: str) -> float:
    """The per-test hard bound for ``nodeid``: :data:`BOUND_SLACK` x its file, floored.

    See the module docstring for why the manifest is the right input and why the
    answer is a hang catcher. The floor is what an unmeasured file gets, so a new
    test file (or one the manifest is missing) is bounded rather than exempt.
    """
    measured = _manifest_seconds(nodeid)
    if measured is None:
        return BOUND_FLOOR_S
    return max(BOUND_FLOOR_S, measured * BOUND_SLACK)


def per_test_bound() -> tuple[float | None, bool] | None:
    """The per-test HARD bound as ``(seconds, exit)``, or ``None`` when not asked for.

    ``seconds`` is ``None`` for the sized bound (:func:`sized_bound_seconds`, one
    bound per test file) and a number when the operator typed one in, which is the
    escape hatch for someone who has already seen the sized bound misfire.

    The bound kills the process that hits it -- that is its whole purpose, and the
    only way to turn a park into a failure rather than a silence -- which is why
    it is OFF unless asked for and why a malformed value resolves to ``None``
    (report-only). Note the asymmetry with :func:`local_seconds`: an unparseable
    LOCAL bound falls through to the local default, which only prints, while an
    unparseable HARD bound falls back to not killing anything.
    """
    raw = os.environ.get(TEST_TIMEOUT_ENV, "").strip()
    if not raw or raw.lower() in FALSY_ENV_VALUES:
        return None
    if raw.lower() in {"1", "true", "yes", "on"}:
        return (None, True)
    try:
        seconds = float(raw)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds <= 0 or seconds >= MAX_BOUND_S:
        return None
    return (seconds, True)


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
    target = run_dir()
    with contextlib.suppress(OSError):
        target.mkdir(parents=True, exist_ok=True)
    return target / f"{tag}.log"


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

    def __init__(
        self, seconds: float | None, hard: tuple[float | None, bool] | None = None
    ) -> None:
        #: The report-only bound, or ``None`` when the operator silenced the
        #: REPORTER and left only the hard bound on. The two knobs are
        #: independent on purpose: the announcement for the report names the
        #: variable that switches it off, so the combination is not exotic, and
        #: the bound is exactly what an operator silencing the report still
        #: wants (round-1 review MAJOR-1).
        self.seconds = seconds
        #: The opt-in per-test HARD bound, or ``None`` for the report-only default.
        #: Captured at install for the same reason ``seconds`` is: the value is
        #: known to be usable there, whereas ``note_start`` runs inside a hook.
        self.hard = hard
        self._handle = None
        self._path = _dump_path(f"worker-{os.getpid()}")
        self._armed = False
        # Set when the dump target itself is unusable, so nothing retries a
        # failed open once per test for the rest of the run.
        self._disabled = False

    def bound_for(self, nodeid: str) -> tuple[float, bool]:
        """The ``(seconds, exit)`` to arm for ``nodeid``.

        ONE place decides this, because two things must agree on it: the timer,
        and the arm-time header that records what the timer was armed at. A header
        naming a bound the timer was not armed at is the mislabelling the report
        filters exist to prevent (see :data:`ARM_MARKER`).
        """
        if self.hard is None:
            # install() refuses to build a timer with neither knob set, so a
            # reporting bound is present here; the assertion is for the type.
            assert self.seconds is not None
            return (self.seconds, False)
        seconds, _exit = self.hard
        return (sized_bound_seconds(nodeid) if seconds is None else seconds, True)

    def arm(self, nodeid: str) -> None:
        """Start the C timer for ``nodeid``; disable quietly if it cannot.

        One arm per item, because :meth:`disarm` runs on that item's teardown: the
        timer's unit really is one test, in both modes. That is also the price of
        the instrument -- one ``faulthandler`` thread create/cancel pair per test,
        which is why the LOCAL default announces itself rather than being silent.

        Should an arm ever arrive while one is still running (a test that reported
        no teardown, i.e. a crash or a park), the stale timer is cancelled first --
        and the reason is stronger than tidiness: ``faulthandler`` does NOT allow
        several timers, it REPLACES the live one when ``dump_traceback_later`` is
        called again (measured on this host: the last-armed bound fires, the
        earlier one never does; the same rule is recorded in
        ``local_operator/session/runtime/stall_watchdog.py``). Two consequences
        follow, both of them why the cancel is here rather than optional: an arm
        silently replaces any other in-process user of the C timer (the runtime
        watchdog when a test drives it in-process), and :meth:`disarm` cancels
        whatever is armed rather than only this module's, so the one-arm-per-item
        invariant has to hold for the whole process.

        Every failure here is swallowed rather than raised. This runs inside a
        pytest hook, so an exception is an ``INTERNALERROR`` that fails the
        shard -- the diagnostic is not allowed to be the reason a run goes red,
        and a run whose dump directory is unusable is still a run worth having.

        ``OverflowError`` is in the tuple even though the resolvers already refuse
        a bound no timer can hold: the value can also arrive here from a direct
        construction, and this is the one call whose failure mode is the whole
        shard. Belt and braces, because the cost of the extra name is zero and the
        cost of missing it is a red run.
        """
        if self._disabled:
            return
        bound, hard = self.bound_for(nodeid)
        try:
            # The handle must exist and stay open for the life of the process:
            # the dump is written from a C thread with a raw descriptor, so the
            # file cannot be opened at the moment it fires.
            if self._handle is None or self._handle.closed:
                self._handle = self._path.open("w", encoding="utf-8")
            self._handle.write(f"{ARM_MARKER}{nodeid} exceeded {bound:g}s; every thread follows.\n")
            self._handle.flush()
            if self._armed:
                faulthandler.cancel_dump_traceback_later()
            # ``repeat`` is the difference between the two modes, not a detail: a
            # report-only bound is a floor and wants a fresh snapshot every
            # interval, while a hard bound ENDS the process on the first firing
            # (there is nothing left to repeat with), which is also why the two
            # cannot be requested together.
            faulthandler.dump_traceback_later(bound, file=self._handle, repeat=not hard, exit=hard)
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

    def __init__(self, seconds: float | None, sink=None) -> None:
        #: ``None`` means "no stall reporting" -- the mode an operator gets from
        #: ``LOCAL_OPERATOR_LOCAL_STALL_SECONDS=0`` with a per-test hard bound
        #: left on. The controller still runs, because its other job is the
        #: cleanup that keeps each run's dump directory from accumulating one
        #: armed file per test; only the reporting thread is skipped.
        self.seconds = seconds
        self._lock = threading.Lock()
        self._in_flight: dict[str, float] = {}
        self._last_progress = time.monotonic()
        #: When this controller started, and how many tests have COMPLETED since.
        #: Reported with every stall, because a silence has two explanations and
        #: the artifact cannot tell them apart without them: a run that is parked
        #: and a run that is grinding through slow tests look identical in `-q`
        #: output (one progress line per 72 completed tests, so the last partial
        #: line is withheld until its slowest item finishes). Measured 2026-09-24:
        #: a local whole-tree run whose log stood still at 69% for 29 minutes was
        #: neither -- it had been killed by the wall-clock `timeout` its operator
        #: passed, and was progressing at a healthy rate until that second.
        self._started_at = time.monotonic()
        self._completed = 0
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
                self._completed += 1
                self._in_flight.pop(nodeid, None)

    def start(self) -> None:
        if self.seconds is None:
            return
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
        if self.seconds is None:
            # Reporting silenced with a per-test bound left on: the controller
            # exists for its cleanup, not for this, and comparing against a bound
            # that is not there would raise inside a hook.
            return False
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
        elapsed = now - self._started_at
        rate = self._completed / elapsed if elapsed > 0 else 0.0
        lines = [
            "",
            "!" * 72,
            f"SHARD STALL: no test has completed for {idle:.0f}s "
            f"(bound {self.seconds:g}s). {len(in_flight)} still in flight:",
            # The rate is what turns "silent" into a diagnosis. A run at 0.2/s
            # that has been quiet for a minute is working; the same numbers with
            # a rate near zero are a park, or a dead worker set. Printed BEFORE
            # the node ids because it decides how to read them.
            f"  {self._completed} tests completed in {elapsed:.0f}s ({rate:.2f}/s)",
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
            candidates = sorted(run_dir().glob("worker-*.log"))
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

        Runs on the controller, which outlives its workers, and ONLY inside this
        run's own directory -- a shared directory made this call delete a
        concurrent run's armed dump, which is how that run's report lost its
        stack excerpt.

        A fired dump is evidence and is kept even on a green run -- a test that
        took four minutes and still passed is worth knowing about. The run's
        directory is then removed if it is empty, so a healthy run leaves nothing
        behind while a run with evidence leaves its own directory standing.
        """
        directory = run_dir()
        try:
            paths = sorted(directory.glob("*.log"))
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
        with contextlib.suppress(OSError):
            directory.rmdir()


#: One controller per controller process; None on a worker.
_CONTROLLER: _Controller | None = None
#: One timer per worker process; None on the controller.
_WORKER: _WorkerTimer | None = None


def _bound_detail(hard: tuple[float | None, bool] | None) -> str:
    """How the per-test HARD bound is expressed in an announcement line.

    One definition, because two lines print it (:func:`_announce` when reporting
    is on, :func:`_announce_hard_only` when it is silenced) and a reader has to be
    able to compare them.
    """
    if hard is None:
        return "reporting only"
    if hard[0] is None:
        return "per-test hard bound ON, sized per file"
    return f"per-test hard bound {hard[0]:g}s"


def _announce(seconds: float, hard: tuple[float | None, bool] | None, source: str) -> None:
    """One stderr line saying this reporter turned ITSELF on, and how to stop it.

    Printed only for the LOCAL default, never for an explicit value: a CI shard or
    an operator who set the variable already knows, and a per-worker line would be
    spam (every xdist worker imports this conftest). The silence knob is named in
    the line itself, because an instrument that reports on a healthy run has to be
    switchable off by the person reading it -- and never raises, for the same
    reason nothing else here does.

    The line also says what the silence knob does NOT switch off. The two knobs
    are independent, and the announcement is where an operator learns the name of
    the off switch, so a reader who followed it and then set a per-test bound must
    not be left believing: they were off.
    """
    with contextlib.suppress(Exception):
        print(
            f"{source} report: ON ({seconds:g}s of no completed test), "
            f"{_bound_detail(hard)}. "
            f"Silence the report with {LOCAL_ENV_SECONDS}=0 "
            f"(a per-test bound survives that).",
            file=sys.stderr,
            flush=True,
        )


def _announce_hard_only(hard: tuple[float | None, bool]) -> None:
    """The mirror line: reporting silenced, a per-test bound still armed.

    Printed because the hard bound kills a worker, and a run that can be killed
    must say so once. Without it the combination is silent in both directions:
    no report from the controller, and the first thing anyone learns about the
    bound is a worker dying (round-1 review MAJOR-1).
    """
    with contextlib.suppress(Exception):
        print(
            f"local stall report: OFF ({LOCAL_ENV_SECONDS}=0); "
            f"{_bound_detail(hard)} still ON -- a test over it kills its worker.",
            file=sys.stderr,
            flush=True,
        )


def install(config, on_ci: bool = False) -> None:
    """Wire the module up in whichever process this is. Idempotent.

    ``hasattr(config, "workerinput")`` is xdist's own discriminator between a
    worker and the controller, and it is the only one that is correct *before*
    a session exists -- the two processes need different instrumentation (one
    watches, the other is watched), so the branch has to happen here rather
    than inside a hook.

    Only a WORKER ever arms a C timer, which is the invariant that keeps this
    module out of the e2e stage's way: that stage runs ``-n0`` (no worker
    process exists) with its own ``tests/e2e/watchdog`` timer, and
    ``faulthandler``'s timer is process-global.

    The two knobs are resolved independently, and that ORDER is load-bearing: an
    early return on a silenced reporter used to skip the hard bound entirely, so
    the exact combination the announcement invites -- silence the report, set the
    bound -- ran unbounded and passed (round-1 review MAJOR-1, measured twice).
    The controller also creates this run's dump directory here and exports it, so
    that every worker writes into it: install runs before any worker exists.
    """
    global _CONTROLLER, _WORKER, _RUN_DIR
    hard = per_test_bound()
    seconds = enabled_seconds()
    # Announced only when the fallback -- rather than an explicit bound -- is what
    # enabled this, so a CI shard and a set variable stay silent.
    local_default = seconds is None
    if local_default:
        seconds = local_seconds(on_ci)
    if seconds is None and hard is None:
        return
    is_worker = hasattr(config, "workerinput")
    if not is_worker:
        # Made here rather than lazily on the first dump, so that the directory a
        # reader (or the workflow's report step) is pointed at exists for the
        # whole run. An empty one is removed by :meth:`_Controller.cleanup`.
        _RUN_DIR = DUMP_ROOT / f"run-{os.getpid()}"
        with contextlib.suppress(OSError):
            _RUN_DIR.mkdir(parents=True, exist_ok=True)
        os.environ[ENV_RUN_DIR] = str(_RUN_DIR)
    if is_worker:
        if _WORKER is None:
            _WORKER = _WorkerTimer(seconds, hard)
    elif _CONTROLLER is None:
        _CONTROLLER = _Controller(seconds)
        _CONTROLLER.start()
        if seconds is None:
            # Reached only with the bound set (the early return above), which is
            # what the assertion states for the type checker's benefit.
            assert hard is not None
            _announce_hard_only(hard)
        elif local_default:
            _announce(seconds, hard, "local stall")


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


def _install_controller_for_test(seconds: float | None, sink) -> _Controller:
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

    Each run writes into its own subdirectory (see :func:`run_dir`), so the walk
    goes one level down and labels the nested blocks with the run they came from:
    two concurrent runs' evidence has to be attributable, which is the other half
    of why the directories are separate.
    """
    root = DUMP_ROOT if directory is None else directory
    try:
        paths = sorted(root.glob("*.log"))
    except OSError:
        paths = []
    runs: list[Path] = []
    with contextlib.suppress(OSError):
        runs = sorted(path for path in root.iterdir() if path.is_dir())
    if not paths and not runs:
        return "no shard stall report: nothing stalled long enough to trip the watchdog\n"
    blocks: list[str] = []
    for path in paths:
        blocks.extend(_report_file(path))
    for run in runs:
        try:
            nested = sorted(run.glob("*.log"))
        except OSError:
            continue
        if not nested:
            continue
        blocks.append(f"----- run {run.name} -----")
        for path in nested:
            blocks.extend(_report_file(path))
    if not blocks:
        return "no shard stall report: nothing stalled long enough to trip the watchdog\n"
    return "\n".join(blocks) + "\n"


def main(argv: list[str] | None = None) -> int:
    """``python -m tests.shard_stall_watchdog [dir]`` -- print the dumps.

    The optional directory lets the workflow point at its own
    ``${TMPDIR:-/tmp}/lo-shard-stall``; the default is :data:`DUMP_ROOT`, computed
    from the same variable, so the two agree without the caller knowing the name.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    sys.stdout.write(report_dumps(Path(args[0]) if args else None))
    return 0


if __name__ == "__main__":  # pragma: no cover - operator/CI aid
    raise SystemExit(main())
