"""Name, and then end, a ``lop serve`` event loop that has stopped turning.

WHY THIS EXISTS (2026-11-11, pids 11196 and 9633). The desktop app's daemon on
``:1111`` wedged twice in one day, on 0.101.43 and again on 0.101.46: ``/health``
timed out for tens of minutes, ``lop services status`` reported "daemon stopped
reporting", and the app's own recovery deliberately refuses to replace a live
process on a failed probe ("a live process is not this app's to replace"). The
read-only evidence that could be collected (``sample 9633``) showed every
GIL-needing thread — the event-loop main thread included — parked in
``take_gil`` behind one default-executor thread (``run_in_executor`` /
``asyncio.to_thread``) iterating pure Python (``FOR_ITER_TUPLE``), with the
process at ~0% CPU, 839 MiB footprint but ~25 MiB resident on a host swapping
10.6 of 12 GB. ``py-spy`` needs root on macOS, and serve had no in-process
traceback hook, so **nobody could say which Python function was running**: the
same gap ``session/runtime/stall_watchdog.py`` closed for session runtimes.

WHAT IT DOES
------------
* A heartbeat coroutine on the loop stamps ``time.monotonic()`` every
  :data:`BEAT_S`. A plain daemon THREAD (never the loop: a blocked loop cannot
  report itself) reads the stamp every :data:`CHECK_S`.
* Lag past :data:`DUMP_LAG_S` writes ``faulthandler.dump_traceback(all_threads=
  True)`` to ``<config>/logs/serve-stall-<pid>.log`` (and names that file on the
  serve log), once per stall EPISODE plus once more each :data:`REDUMP_S` while
  it persists (backing off after :data:`MAX_EPISODE_DUMPS`, nothing past
  :data:`MAX_LOG_BYTES`), so a second dump shows whether the hot frame moved.
  An episode opens at that first crossing and closes only after
  :data:`RECOVERY_S` of continuously healthy beats, so a GIL convoy that lets
  the loop beat now and then is still ONE episode (QA Q-1).
* An episode :data:`RESTART_LAG_S` old, still degraded on
  :data:`CONFIRM_CHECKS` consecutive checks, re-execs this process ONCE (per
  episode, at most :data:`MAX_RESTARTS` per :data:`RESTART_WINDOW_S` across
  execs) onto the current build
  through :func:`local_operator.server.reload.emergency_restart` — the same
  pid, the same listening socket, the same bearer handoff an operator's
  ``lop services restart`` uses — so the app's registered pid stays valid and
  the port never unbinds.

INVARIANTS
----------
* NEVER DURING A RELOAD DRAIN. While a requested reload is pending or its
  admission fence is closed, the reload owns the process's exit; the watchdog
  only dumps (``reload.restart_blocker``).
* NEVER OVER UNRECOVERABLE DAEMON-OWNED WORK. The restart is gated on the same
  inventory a reload drains (``reload.work_blocker``): a runtime mid-spawn,
  scheduler-owned runs, legacy jobs, hub work, the Aida bootstrap. The ONE term
  left out is the in-flight request count, on purpose: a wedged loop's in-flight
  requests are the very thing that cannot finish, so counting them would make
  the restart refuse in exactly the state it exists for. Those requests are
  plain HTTP exchanges a client retries; a blocker that IS present keeps the
  daemon wedged-but-alive with dumps (fail closed), as before this module.
* FAIL CLOSED. An unreadable inventory, a missing listener (a ``--reload``
  child), an unresolvable interpreter, or an exec that fails all leave the
  daemon running exactly as it would have without this module, logged.
* NAMED GAP: this is a Python thread, so it needs a GIL turn to act. A thread
  executing bytecode yields the GIL every ``sys.getswitchinterval()`` (5 ms), so
  the measured wedge class (a pure-Python loop on an executor thread) is
  covered; a thread holding the GIL inside ONE C call is not, and neither is a
  process the kernel is not scheduling at all. ``faulthandler.dump_traceback_
  later`` would cover the first, but re-arming it is itself a GIL-held block
  (see ``stall_watchdog.py``'s docstring) — so it is not used here. What does
  cover both for DIAGNOSIS is ``kill -USR2 <pid>``: ``faulthandler.register``
  dumps all threads from C into the stall log with no GIL turn needed (review
  R4). Nothing restarts those two classes; the operator's reclaim still does.

THRESHOLDS — measured (PR body, "Measurements"). Healthy loop round-trip lag
of a real ``lop serve`` on the reporting host at load 38-75: n=120, median
0.3 ms, p99 4.5 ms, max 13.1 ms. One GIL-holding pure-Python executor thread
raised ``/health`` from a median ~8 ms to 183-489 ms (n=40) — the GIL convoy,
seconds-scale only once more threads queue on ``take_gil`` — and never past
1 s. :data:`DUMP_LAG_S` sits ~750x over the healthy max and 20x over the
convoy max; :data:`RESTART_LAG_S` sits above the desktop app's 90 s
``DETACHED_AFTER_MS`` ("reporting it as stopped"), so a restart only ends a
stall the operator already sees as an outage. ``time.monotonic`` excludes
system sleep on macOS and Linux, so a lid-close is not a stall.
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import re
import signal
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("local_operator.server.loop_watchdog")

#: Heartbeat period. Lag is measured as ``now - last_beat``, so a healthy loop
#: reads between 0 and ``BEAT_S`` plus its scheduling jitter.
BEAT_S = 0.5
#: How often the watchdog thread looks. Bounds detection latency, costs one
#: ``Event.wait`` wake-up a second.
CHECK_S = 1.0
#: Lag that earns a traceback dump; see THRESHOLDS above for the distribution.
DUMP_LAG_S = 10.0
#: While one stall persists, dump again this often so successive dumps show
#: whether the hot frame is moving (slow) or fixed (spinning/blocked).
REDUMP_S = 60.0
#: A stall EPISODE that has lasted this long earns a self-restart; above the
#: app's 90 s "stopped" verdict. Measured from the stall's first missed beat,
#: so a fully blocked loop restarts at the same 120 s of silence as before.
RESTART_LAG_S = 120.0
#: Lag below this is a loop that is turning normally. A healthy loop reads
#: between 0 and ``BEAT_S`` plus jitter (measured jitter max 13.1 ms), so 1 s is
#: two beat periods and ~37x the jitter max over ``BEAT_S``; anything at or above
#: it inside an episode is the loop still being starved (QA Q1d's convoy read
#: lags of 20-80 s with brief turns in between).
HEALTHY_LAG_S = 1.0
#: An episode ends only after the loop has stayed under :data:`HEALTHY_LAG_S`
#: for this long, continuously. Under QA's Q1d convoy (``setswitchinterval(10)``)
#: the loop got turns now and then, and ``/health`` stayed dead for 222 s; the old
#: "any beat resets" rule let each turn restart the 120 s clock. 30 s is 60 beats
#: in a row on time, which a convoy that leaves ``/health`` timing out at 5 s
#: does not produce, and Q2's 200 s of normal load never opens an episode at all
#: (it never reaches :data:`DUMP_LAG_S`), so this cannot create a false fire.
RECOVERY_S = 30.0
#: Consecutive degraded checks needed before a restart. One huge sample followed
#: by a beat is a PAUSE (SIGSTOP, a debugger attach, review R2), not a wedge:
#: after SIGCONT the watchdog's expired wait reads the stale stamp before the
#: loop's overdue timer runs. The second check, one :data:`CHECK_S` later, gives
#: the loop that turn; a real wedge still restarts one tick later.
CONFIRM_CHECKS = 2
#: Per episode, after this many dumps the redump cadence backs off to
#: :data:`REDUMP_BACKOFF_S`: a stall held by a blocker for an hour would
#: otherwise write a dump a minute (QA Q-2: ~6 KB per dump at rig thread counts).
MAX_EPISODE_DUMPS = 10
REDUMP_BACKOFF_S = 3600.0
#: A stall log at or over this size takes no more watchdog dumps (the operator
#: signal still writes): the file is evidence, not a journal.
MAX_LOG_BYTES = 1024 * 1024
#: ``serve-stall-*.log`` files kept at start (newest by mtime); one per pid, and
#: nothing else ever removes them.
KEEP_LOGS = 5
#: Restart budget (review R3): at most this many self-restarts in
#: :data:`RESTART_WINDOW_S`, then dump-only. A hog that returns on every boot
#: (startup indexing on a swapping host) would otherwise re-exec every ~2.5 min
#: forever (QA Q4: 3 re-execs in 7.5 min), each dropping in-flight requests.
MAX_RESTARTS = 3
RESTART_WINDOW_S = 1800.0
#: Where the budget survives the exec: ``reload._exec`` passes ``os.environ``
#: explicitly, so the successor image inherits it. Comma-separated wall epochs
#: (wall, not monotonic, because the value is only compared within one boot of
#: the host anyway and a wall stamp reads in a ``ps eww``).
RESTARTS_ENV = "LOP_SERVE_WATCHDOG_RESTARTS"
#: The operator's lever for a daemon this thread cannot reach (review R4): the
#: GIL held inside one C call, or a convoy that starves this thread too.
#: ``faulthandler``'s handler runs in C, no GIL needed. SIGUSR1 is serve's reload
#: signal; nothing in the tree sends SIGUSR2 to a serve pid, and its default
#: action (terminate) becomes a dump, which is strictly safer for a stray one.
DUMP_SIGNAL_NAME = "SIGUSR2"

#: ``app.state`` attribute holding the running watchdog, for teardown and tests.
STATE_ATTR = "serve_loop_watchdog"

_STALL_LOG = re.compile(r"serve-stall-\d+\.log")


def read_restart_history(environ: Mapping[str, str], now: float) -> list[float]:
    """The budget carried across exec: restart stamps still inside the window."""
    stamps: list[float] = []
    for raw in environ.get(RESTARTS_ENV, "").split(","):
        try:
            stamp = float(raw)
        except ValueError:
            continue  # an unparsable entry is no restart at all
        if now - RESTART_WINDOW_S < stamp <= now:
            stamps.append(stamp)
    return stamps


def prune_stall_logs(
    logs_dir: Path, keep: int = KEEP_LOGS, current: Path | None = None
) -> list[Path]:
    """Delete all but the newest ``keep`` stall logs; never ``current``. Never raises."""
    try:
        logs = [p for p in logs_dir.iterdir() if _STALL_LOG.fullmatch(p.name) and p != current]
        logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    # The current pid's file counts toward ``keep`` whether or not it exists yet.
    doomed = logs[max(keep - 1, 0) :] if current is not None else logs[keep:]
    removed = []
    for path in doomed:
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            pass
    return removed


class LoopWatchdog:
    """Heartbeat + observer pair. ``check`` is the pure, testable step.

    THE EPISODE (QA Q-1). A stall episode opens when lag first reaches
    ``dump_lag_s`` and closes only after ``recovery_s`` of continuously healthy
    lag (< ``healthy_lag_s``). The restart keys on the episode's AGE, not on one
    unbroken gap, so a convoy that lets the loop beat now and then still ends.
    Within one episode the restart is ATTEMPTED AT MOST ONCE (review R1): a
    refusal or a failed exec falls back to the dump cadence instead of retrying
    (and re-dumping) every tick. ``restart=None`` is a dump-only watchdog (a
    process with no listener of its own, e.g. a ``--reload`` child).
    """

    def __init__(
        self,
        *,
        dump_path: Path,
        restart: Callable[[], str | None] | None,
        restart_blocker: Callable[[], str | None] = lambda: None,
        dump_lag_s: float = DUMP_LAG_S,
        restart_lag_s: float = RESTART_LAG_S,
        redump_s: float = REDUMP_S,
        healthy_lag_s: float = HEALTHY_LAG_S,
        recovery_s: float = RECOVERY_S,
        confirm_checks: int = CONFIRM_CHECKS,
        max_episode_dumps: int = MAX_EPISODE_DUMPS,
        redump_backoff_s: float = REDUMP_BACKOFF_S,
        max_log_bytes: int = MAX_LOG_BYTES,
        max_restarts: int = MAX_RESTARTS,
        restart_window_s: float = RESTART_WINDOW_S,
        restart_history: list[float] | None = None,
        environ: MutableMapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.dump_path = dump_path
        self._restart = restart
        self._restart_blocker = restart_blocker
        self.dump_lag_s = dump_lag_s
        self.restart_lag_s = restart_lag_s
        self.redump_s = redump_s
        self.healthy_lag_s = healthy_lag_s
        self.recovery_s = recovery_s
        self.confirm_checks = confirm_checks
        self.max_episode_dumps = max_episode_dumps
        self.redump_backoff_s = redump_backoff_s
        self.max_log_bytes = max_log_bytes
        self.max_restarts = max_restarts
        self.restart_window_s = restart_window_s
        #: Wall stamps of earlier self-restarts (carried across exec by ``install``).
        self.restart_history = list(restart_history or [])
        #: Where the history is written before an exec; ``None`` keeps it in memory
        #: (tests). ``install`` passes ``os.environ``.
        self._environ = environ
        self._clock = clock
        self._wall_clock = wall_clock
        self.last_beat = clock()
        self.max_lag = 0.0
        self.dumps = 0
        self.restarts_requested = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: Kept open for the process's life: faulthandler's signal handler writes
        #: to its fd. Never closed and never unregistered (``stall_watchdog``:
        #: ``unregister`` restores a stale disposition).
        self.signal_handle: Any = None
        self.dump_signal: int | None = None
        self._budget_logged = False
        self._cap_logged = False
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._episode_start: float | None = None
        self._healthy_since: float | None = None
        self._degraded_streak = 0
        self._episode_dumps = 0
        self._last_dump: float | None = None
        self._last_refusal: str | None = None
        self._restart_attempted = False
        self.max_lag = 0.0

    @property
    def in_episode(self) -> bool:
        return self._episode_start is not None

    # -- the loop side -------------------------------------------------------
    async def heartbeat(self) -> None:
        while True:
            self.last_beat = self._clock()
            await asyncio.sleep(BEAT_S)

    # -- the thread side -----------------------------------------------------
    def check(self, now: float | None = None) -> str | None:
        """One observation. Returns ``"dump"``, ``"restart"``, ``"refused"`` or ``None``."""
        now = self._clock() if now is None else now
        lag = now - self.last_beat
        if self._episode_start is None:
            if lag < self.dump_lag_s:
                return None
            # The stall began at the last beat, so a fully blocked loop's episode
            # age equals its lag: the 120 s bound is unchanged for that class.
            self._episode_start = now - lag
        self.max_lag = max(self.max_lag, lag)
        if lag < self.healthy_lag_s:
            self._degraded_streak = 0
            if self._healthy_since is None:
                self._healthy_since = now
            if now - self._healthy_since >= self.recovery_s:
                logger.warning(
                    "serve event loop recovered after a %.1fs stall episode "
                    "(max lag %.1fs; traceback in %s)",
                    self._healthy_since - self._episode_start,
                    self.max_lag,
                    self.dump_path,
                )
                self._reset_episode()
            return None
        self._healthy_since = None
        self._degraded_streak += 1
        verdict: str | None = None
        if (
            self._restart is not None
            and not self._restart_attempted
            and now - self._episode_start >= self.restart_lag_s
            and self._degraded_streak >= self.confirm_checks
        ):
            verdict = self._try_restart(now, lag, now - self._episode_start)
            if verdict == "restart":
                return verdict
        if self._dump_due(now):
            self._dump(lag, "stalled", now - self._episode_start)
            self._last_dump = now
            self._episode_dumps += 1
            return "dump"
        return verdict

    def _dump_due(self, now: float) -> bool:
        if self._last_dump is None:
            return True
        cadence = (
            self.redump_s if self._episode_dumps < self.max_episode_dumps else self.redump_backoff_s
        )
        return now - self._last_dump >= cadence

    def _try_restart(self, now: float, lag: float, age: float) -> str | None:
        """Gate, budget, then at most one attempt; ``"restart"``/``"refused"``/``None``."""
        reason = self._restart_blocker()
        if reason is not None:
            # A blocker is not an attempt: it is re-read each tick (no dump, no
            # exec) and the restart goes ahead the tick it clears.
            if reason != self._last_refusal:
                logger.warning(
                    "serve event loop stalled %.0fs; self-restart withheld: %s",
                    age,
                    reason,
                )
                self._last_refusal = reason
            return "refused"
        wall = self._wall_clock()
        self.restart_history = [
            t for t in self.restart_history if wall - self.restart_window_s < t <= wall
        ]
        if len(self.restart_history) >= self.max_restarts:
            self._restart_attempted = True
            if not self._budget_logged:
                logger.warning(
                    "serve loop watchdog: %d self-restarts in the last %.0f min; "
                    "dump-only from now on "
                    "(the stall keeps coming back, a restart is not curing it)",
                    len(self.restart_history),
                    self.restart_window_s / 60,
                )
                self._budget_logged = True
            return "refused"
        # Once per episode, decided BEFORE the attempt: whatever happens next,
        # this episode does not try again (review R1).
        self._restart_attempted = True
        self._dump(lag, "restarting", age)
        # Counts as this tick's dump: a failed exec must not dump twice.
        self._last_dump = now
        self._episode_dumps += 1
        self.restarts_requested += 1
        previous = list(self.restart_history)
        self.restart_history.append(wall)
        self._persist_history()
        # Returns only on refusal/failure: a successful exec ends this process.
        restart = self._restart
        assert restart is not None  # ``check`` gates on it
        reason = restart()
        if reason is None:
            return "restart"
        self.restart_history = previous  # a restart that never happened spends no budget
        self._persist_history()
        logger.warning(
            "serve event loop stalled %.0fs; self-restart failed, dump-only for this episode: %s",
            age,
            reason,
        )
        self._last_refusal = reason
        return "refused"

    def _persist_history(self) -> None:
        if self._environ is None:
            return
        if self.restart_history:
            self._environ[RESTARTS_ENV] = ",".join(f"{t:.0f}" for t in self.restart_history)
        else:
            self._environ.pop(RESTARTS_ENV, None)

    def _dump(self, lag: float, verdict: str, age: float) -> None:
        try:
            if self.dump_path.stat().st_size >= self.max_log_bytes:
                if not self._cap_logged:
                    logger.warning(
                        "serve loop watchdog: %s is at its %d-byte cap; "
                        "no further dumps (send %s for one)",
                        self.dump_path,
                        self.max_log_bytes,
                        DUMP_SIGNAL_NAME,
                    )
                    self._cap_logged = True
                return
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("serve loop watchdog could not stat %s", self.dump_path, exc_info=True)
        try:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dump_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"\n=== serve loop {verdict}: no heartbeat for {lag:.1f}s, episode {age:.1f}s "
                    f"(pid {os.getpid()}, {time.strftime('%Y-%m-%dT%H:%M:%S%z')}) ===\n"
                )
                handle.flush()
                # C-level, all threads, no allocation in the hot path: it names
                # the executor thread's frames, which is the evidence missing
                # from the 9633 wedge.
                faulthandler.dump_traceback(file=handle, all_threads=True)
            self.dumps += 1
        except OSError:
            logger.warning("serve loop watchdog could not write %s", self.dump_path, exc_info=True)
            return
        logger.warning(
            "serve event loop %s: no heartbeat for %.1fs (episode %.1fs); "
            "all-thread traceback in %s",
            verdict,
            lag,
            age,
            self.dump_path,
        )

    def register_dump_signal(self) -> int | None:
        """Make ``kill -USR2 <pid>`` write every thread's frames to the stall log.

        ``chain=False`` and no ``unregister``, both measured in
        ``session/runtime/stall_watchdog.py`` (a chained registration segfaulted a
        runtime). Never raises: a missing lever leaves the watchdog as it was.
        """
        signum = getattr(signal, DUMP_SIGNAL_NAME, None)
        register = getattr(faulthandler, "register", None)  # absent on Windows
        if signum is None or register is None:
            return None
        try:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            self.signal_handle = open(
                self.dump_path, "a", encoding="utf-8"
            )  # noqa: SIM115 — lives forever
            register(signum, file=self.signal_handle, all_threads=True, chain=False)
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "serve loop watchdog could not register %s", DUMP_SIGNAL_NAME, exc_info=True
            )
            return None
        self.dump_signal = int(signum)
        return self.dump_signal

    def _run(self) -> None:
        while not self._stop.wait(CHECK_S):
            try:
                self.check()
            except Exception:  # noqa: BLE001 — the observer must outlive any one check
                logger.warning("serve loop watchdog check failed", exc_info=True)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="serve-loop-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def install(app: Any, config_dir: str | Path) -> tuple[LoopWatchdog, asyncio.Task[None]]:
    """Start the watchdog for this daemon. Called from the lifespan, on the loop."""
    from local_operator.server import reload as serve_reload

    dump_path = Path(config_dir) / "logs" / f"serve-stall-{os.getpid()}.log"
    prune_stall_logs(dump_path.parent, current=dump_path)
    # No listener of its own (a ``--reload`` child): ``plan_for`` would refuse
    # every attempt, so register no restart at all and stay dump-only (R1). The
    # same predicate decides the record's ``reloadable`` field.
    restart = (
        (lambda: serve_reload.emergency_restart(app)) if serve_reload.is_reloadable(app) else None
    )
    watchdog = LoopWatchdog(
        dump_path=dump_path,
        restart=restart,
        restart_blocker=lambda: serve_reload.restart_blocker(app),
        restart_history=read_restart_history(os.environ, time.time()),
        environ=os.environ,
    )
    signum = watchdog.register_dump_signal()
    beat = asyncio.get_running_loop().create_task(watchdog.heartbeat(), name="serve-loop-heartbeat")
    watchdog.start()
    setattr(app.state, STATE_ATTR, watchdog)
    logger.info(
        "serve loop watchdog armed (%s; %d prior self-restart(s) in window)%s",
        "restart enabled" if restart is not None else "dump-only: no listener of its own",
        len(watchdog.restart_history),
        (
            f"; kill -{DUMP_SIGNAL_NAME[3:]} {os.getpid()} dumps all threads to {dump_path}"
            if signum
            else ""
        ),
    )
    return watchdog, beat
