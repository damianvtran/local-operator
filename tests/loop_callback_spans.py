"""Bounded, test-only attribution for the CPU accumulated between probe ticks.

A timer gap includes every callback ahead of its wake, not just one task step.
Keep the cumulative liveness verdict separate from these explanatory spans: a
queue of short callbacks can starve a timer too. No frames, arguments or callback
reprs are retained, since they may contain conversation content or credentials.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from types import TracebackType
from unittest.mock import patch


@dataclass(frozen=True)
class CallbackSpan:
    name: str
    cpu_ms: float
    wall_ms: float


@dataclass
class CallbackWindow:
    count: int = 0
    cpu_ms: float = 0.0
    wall_ms: float = 0.0
    top_cpu: list[CallbackSpan] = field(default_factory=list)
    top_wall: list[CallbackSpan] = field(default_factory=list)


@dataclass
class ProbeWindow:
    cpu_gap_ms: float
    wall_gap_ms: float
    callbacks: CallbackWindow


class CallbackSpans:
    """Attribute completed Handle callbacks on ONE loop/thread, within a context.

    Handle._run is shared process-wide, so unrelated loops/threads must pass
    straight through and the original must be restored even when the test fails.
    Only five CPU/wall leaders and three worst windows per clock survive; callback
    count and sums include dropped leaders. This is diagnostic instrumentation,
    not a replacement clock or a threshold/calibration input.
    """

    TOP_N = 5
    WINDOW_N = 3

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        cpu_clock: Callable[[], float] = time.thread_time,
        wall_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.loop = loop
        self.thread_id = threading.get_ident()
        self._cpu_clock = cpu_clock
        self._wall_clock = wall_clock
        self._current = CallbackWindow()
        self._cpu_windows: list[ProbeWindow] = []
        self._wall_windows: list[ProbeWindow] = []
        self._patch = patch.object(asyncio.Handle, "_run", self._run_wrapper())

    def _run_wrapper(self) -> Callable[[asyncio.Handle], None]:
        original = asyncio.Handle._run

        def run(handle: asyncio.Handle) -> None:
            if handle._loop is not self.loop or threading.get_ident() != self.thread_id:
                original(handle)
                return
            callback = handle._callback
            owner = getattr(callback, "__self__", None)
            # Coroutine qualnames identify task steps without repr(task/coro),
            # whose rendered arguments can leak the very payload under test.
            if isinstance(owner, asyncio.Task):
                name = getattr(owner.get_coro(), "__qualname__", type(callback).__name__)
            else:
                name = getattr(callback, "__qualname__", type(callback).__name__)
            cpu = self._cpu_clock()
            wall = self._wall_clock()
            try:
                original(handle)
            finally:
                span = CallbackSpan(
                    name,
                    (self._cpu_clock() - cpu) * 1000,
                    (self._wall_clock() - wall) * 1000,
                )
                window = self._current
                window.count += 1
                window.cpu_ms += span.cpu_ms
                window.wall_ms += span.wall_ms
                window.top_cpu.append(span)
                window.top_cpu.sort(key=lambda item: item.cpu_ms, reverse=True)
                del window.top_cpu[self.TOP_N :]
                window.top_wall.append(span)
                window.top_wall.sort(key=lambda item: item.wall_ms, reverse=True)
                del window.top_wall[self.TOP_N :]

        return run

    def __enter__(self) -> CallbackSpans:
        self._patch.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._patch.stop()

    def take_window(self) -> CallbackWindow:
        """Swap buckets at a probe wake; the probe's own span ends after this.

        Consequently callback sums are explanatory, not an exact partition of
        the tick clock: the previous probe's tail is in this bucket and this
        probe's tail is in the next. Other callbacks cannot straddle a wake on
        this single thread. Never subtract these sums from the asserted gap.
        """
        window, self._current = self._current, CallbackWindow()
        return window

    def observe_window(self, cpu_gap_ms: float, wall_gap_ms: float) -> None:
        window = ProbeWindow(cpu_gap_ms, wall_gap_ms, self.take_window())
        self._cpu_windows.append(window)
        self._cpu_windows.sort(key=lambda item: item.cpu_gap_ms, reverse=True)
        del self._cpu_windows[self.WINDOW_N :]
        self._wall_windows.append(window)
        self._wall_windows.sort(key=lambda item: item.wall_gap_ms, reverse=True)
        del self._wall_windows[self.WINDOW_N :]

    def summary(self) -> dict[str, object]:
        return {
            "process_id": os.getpid(),
            "loop_thread_id": self.thread_id,
            "clock": "thread_time CPU / perf_counter wall, milliseconds",
            "attribution": "completed callbacks; probe-tail boundary mismatch, not verdict input",
            "worst_cpu_windows": [asdict(window) for window in self._cpu_windows],
            "worst_wall_windows": [asdict(window) for window in self._wall_windows],
        }
