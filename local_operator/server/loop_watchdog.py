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
  serve log), once per stall plus once more each :data:`REDUMP_S` while the stall
  persists, so a second dump shows whether the hot frame moved.
* Lag past :data:`RESTART_LAG_S` re-execs this process onto the current build
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
  (see ``stall_watchdog.py``'s docstring) — so it is not used here.

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
import threading
import time
from collections.abc import Callable
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
#: Lag that earns a self-restart; above the app's 90 s "stopped" verdict.
RESTART_LAG_S = 120.0

#: ``app.state`` attribute holding the running watchdog, for teardown and tests.
STATE_ATTR = "serve_loop_watchdog"


class LoopWatchdog:
    """Heartbeat + observer pair. ``check`` is the pure, testable step."""

    def __init__(
        self,
        *,
        dump_path: Path,
        restart: Callable[[], str | None],
        restart_blocker: Callable[[], str | None] = lambda: None,
        dump_lag_s: float = DUMP_LAG_S,
        restart_lag_s: float = RESTART_LAG_S,
        redump_s: float = REDUMP_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dump_path = dump_path
        self._restart = restart
        self._restart_blocker = restart_blocker
        self.dump_lag_s = dump_lag_s
        self.restart_lag_s = restart_lag_s
        self.redump_s = redump_s
        self._clock = clock
        self.last_beat = clock()
        self._last_dump: float | None = None
        self._last_refusal: str | None = None
        self.max_lag = 0.0
        self.dumps = 0
        self.restarts_requested = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
        if lag < self.dump_lag_s:
            if self._last_dump is not None:
                logger.warning(
                    "serve event loop recovered after a %.1fs stall (traceback in %s)",
                    self.max_lag,
                    self.dump_path,
                )
            self._last_dump = None
            self._last_refusal = None
            self.max_lag = 0.0
            return None
        self.max_lag = max(self.max_lag, lag)
        if lag >= self.restart_lag_s:
            reason = self._restart_blocker()
            if reason is None:
                self._dump(lag, "restarting")
                self.restarts_requested += 1
                # Returns only on refusal/failure: a successful exec ends this process.
                reason = self._restart()
                if reason is None:
                    return "restart"
            if reason != self._last_refusal:
                logger.warning(
                    "serve event loop stalled %.0fs; self-restart withheld: %s",
                    lag,
                    reason,
                )
                self._last_refusal = reason
            # Fall through: keep dumping on the REDUMP cadence while refused.
        if self._last_dump is None or now - self._last_dump >= self.redump_s:
            self._dump(lag, "stalled")
            self._last_dump = now
            return "dump"
        return "refused" if self._last_refusal else None

    def _dump(self, lag: float, verdict: str) -> None:
        try:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dump_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"\n=== serve loop {verdict}: no heartbeat for {lag:.1f}s "
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
            "serve event loop %s: no heartbeat for %.1fs; all-thread traceback in %s",
            verdict,
            lag,
            self.dump_path,
        )

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

    watchdog = LoopWatchdog(
        dump_path=Path(config_dir) / "logs" / f"serve-stall-{os.getpid()}.log",
        restart=lambda: serve_reload.emergency_restart(app),
        restart_blocker=lambda: serve_reload.restart_blocker(app),
    )
    beat = asyncio.get_running_loop().create_task(watchdog.heartbeat(), name="serve-loop-heartbeat")
    watchdog.start()
    setattr(app.state, STATE_ATTR, watchdog)
    return watchdog, beat
