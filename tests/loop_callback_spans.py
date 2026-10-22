"""Bounded, test-only attribution for the CPU accumulated between probe ticks.

A timer gap includes every callback ahead of its wake, not just one task step.
Keep the cumulative liveness verdict separate from these explanatory spans: a
queue of short callbacks can starve a timer too. No frames, arguments or callback
reprs are retained, since they may contain conversation content or credentials.

A Textual timer step is a coroutine resume, so its span name is only
``Timer._run_timer`` for every timer the app has. The timer patches here add the
one thing that name cannot say -- WHICH timer ran: its name, its bound callback
and the product frame that armed it -- and GC attribution separates a
collection that landed in a callback from the callback's own work.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from types import TracebackType
from unittest.mock import patch

from textual.timer import Timer


@dataclass(frozen=True)
class CallbackSpan:
    name: str
    cpu_ms: float
    wall_ms: float
    #: Which production timer ran, when ``name`` is a Textual ``Timer`` step.
    #: Empty for every other callback: the summary must never invent detail it
    #: did not observe, and a blank here is a fact ("not a timer step").
    detail: str = ""


def _callback_qualname(callback: object) -> str:
    """Name a timer callback without touching its arguments or repr.

    ``functools.partial`` is how several timers are armed (a delay plus an
    argument); ``type(callback).__qualname__`` alone would report them all as
    "partial" and lose the only useful half. The wrapped function's qualname is
    a code attribute, not payload.
    """
    if isinstance(callback, functools.partial):
        inner = getattr(callback.func, "__qualname__", type(callback.func).__qualname__)
        return f"partial({inner})"
    return getattr(callback, "__qualname__", type(callback).__qualname__)


def _timer_site() -> str:
    """The application frame that armed a Textual timer, else its library site.

    ``Timer`` is created inside ``MessagePump.set_interval``/``set_timer``, so
    the innermost frames are Textual's. The first frame outside Textual names
    the product call site (e.g. ``tui/app.py:12291``); when a timer is Textual's
    own (the screen repaint), no such frame exists and the innermost non-asyncio
    frame is reported instead.
    """
    candidates = [
        frame
        for frame in reversed(traceback.extract_stack()[:-2])
        if "/asyncio/" not in frame.filename.replace("\\", "/")
    ]
    library = ""
    for frame in candidates:
        path = frame.filename.replace("\\", "/")
        if path.endswith("/tests/loop_callback_spans.py"):
            break  # our own wrapper: beyond it is the harness, not a caller
        if "/textual/" not in path:
            return f"{path}:{frame.lineno}"
        library = f"{path}:{frame.lineno}"
    return library or (f"{candidates[0].filename}:{candidates[0].lineno}" if candidates else "")


@dataclass
class CallbackWindow:
    count: int = 0
    cpu_ms: float = 0.0
    wall_ms: float = 0.0
    top_cpu: list[CallbackSpan] = field(default_factory=list)
    top_wall: list[CallbackSpan] = field(default_factory=list)
    #: Collections that STARTED inside a tracked callback. The S2 witness
    #: cannot tell "the callback's own work" from "a collection it triggered";
    #: this is that distinction, measured rather than inferred. A pause is
    #: attributed to the callback that happened to be running when it began,
    #: which is a property of the allocation count, not of that callback --
    #: read it as "this pause landed here", never as "this callback caused it".
    gc: list[CallbackSpan] = field(default_factory=list)


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
        # A ``Timer._run_timer`` task step is just "a coroutine resumed". The
        # step's own name therefore cannot say WHICH timer ran; these two
        # patches supply that: the timer's name, its bound callback and the
        # site that armed it. Diagnostic only -- no clock, no behaviour.
        self._tick_serial = 0
        self._tick_detail = ""
        self._timer_init_patch = patch.object(Timer, "__init__", self._timer_init_wrapper())
        self._timer_tick_patch = patch.object(Timer, "_tick", self._timer_tick_wrapper())
        # GC attribution: a collection that starts inside a tracked callback is
        # recorded with the callback it landed in and the generation it
        # collected. gc.callbacks is process-global, so the probe is registered
        # for the context's lifetime and removed in __exit__.
        self._in_callback = False
        self._gc_callback_name = ""
        self._gc_cpu = 0.0
        self._gc_wall = 0.0
        self._gc_generation = -1

    def _timer_init_wrapper(self) -> Callable[..., None]:
        original = Timer.__init__

        def init(timer: Timer, *args: object, **kwargs: object) -> None:
            original(timer, *args, **kwargs)
            callback = getattr(timer, "_callback", None)
            # Set on the instance rather than in a module global: timers are
            # long-lived and concurrently scheduled, so the label must travel
            # with the object the tick belongs to.
            timer._callback_spans_label = (  # type: ignore[attr-defined]
                f"{getattr(timer, 'name', 'timer')}:"
                f"{_callback_qualname(callback)} @ {_timer_site()}"
                if callback is not None
                else ""
            )

        return init

    def _timer_tick_wrapper(self) -> Callable[..., object]:
        original = Timer._tick

        async def tick(timer: Timer, *, next_timer: float, count: int) -> object:
            # Recorded before the await: the callback runs inside this very
            # step, and a later step of the same timer must not inherit it.
            self._tick_serial += 1
            self._tick_detail = getattr(timer, "_callback_spans_label", "")
            return await original(timer, next_timer=next_timer, count=count)

        return tick

    def _gc_started(
        self, phase: str, info: dict[str, object]
    ) -> None:  # pragma: no cover - exercised through the context below
        # gc.callbacks is PROCESS-global, while a span is thread-local: a
        # worker thread collecting while the loop is inside a callback must not
        # be charged to that callback (QA round 1, Q1). The loop's own
        # ``thread_time`` never contained the worker's work; recording it here
        # would have attributed a concurrent pause to the loop step.
        if not self._in_callback or threading.get_ident() != self.thread_id:
            return
        if phase == "start":
            self._gc_cpu = self._cpu_clock()
            self._gc_wall = self._wall_clock()
            self._gc_generation = int(info.get("generation", -1))  # type: ignore[arg-type]
        else:
            window = self._current
            window.gc.append(
                CallbackSpan(
                    f"gc.gen{self._gc_generation}",
                    (self._cpu_clock() - self._gc_cpu) * 1000,
                    (self._wall_clock() - self._gc_wall) * 1000,
                    self._gc_callback_name,
                )
            )
            window.gc.sort(key=lambda item: item.cpu_ms, reverse=True)
            del window.gc[self.TOP_N :]

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
            serial = self._tick_serial
            previous_in_callback = self._in_callback
            self._in_callback = True
            self._gc_callback_name = name
            try:
                original(handle)
            finally:
                self._in_callback = previous_in_callback
                span = CallbackSpan(
                    name,
                    (self._cpu_clock() - cpu) * 1000,
                    (self._wall_clock() - wall) * 1000,
                    # Only a step that actually entered ``_tick`` gets the
                    # timer's identity; the sleep half of the same task does
                    # not, and must not be attributed to the last timer seen.
                    self._tick_detail if self._tick_serial != serial else "",
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
        self._timer_init_patch.start()
        self._timer_tick_patch.start()
        gc.callbacks.append(self._gc_started)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            gc.callbacks.remove(self._gc_started)
        finally:
            try:
                self._timer_tick_patch.stop()
            finally:
                try:
                    self._timer_init_patch.stop()
                finally:
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
