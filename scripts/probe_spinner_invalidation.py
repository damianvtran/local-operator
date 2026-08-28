"""Inspect Textual invalidation from one running subagent spinner row."""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textual._compositor import ChopsUpdate, LayoutUpdate  # noqa: E402

from local_operator.harness.jobs import AsyncJob  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from scripts.benchmark_residual_child_lag import CanonicalFakeSession  # noqa: E402
from tests.unit.tui.test_app_pilot import _factory  # noqa: E402

BLOCKS = 1_457
TICKS = 40


def _summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    values = sorted(values)
    return (
        f"n={len(values)} median={statistics.median(values) * 1000:.3f}ms "
        f"p95={values[int(.95 * (len(values) - 1))] * 1000:.3f}ms "
        f"max={max(values) * 1000:.3f}ms"
    )


class Probe:
    def __init__(self, app: OperatorApp) -> None:
        self.app = app
        self.counts: Counter[str] = Counter()
        self.bytes: list[int] = []
        self.spans: list[int] = []
        self.rows: list[int] = []
        self.layout_times: list[float] = []
        self.display_times: list[float] = []
        self._install()

    def _install(self) -> None:
        screen = self.app.screen
        original_layout = screen._refresh_layout

        def layout(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_layout(*args, **kwargs)
            finally:
                self.counts["layout"] += 1
                self.layout_times.append(time.perf_counter() - started)

        screen._refresh_layout = layout  # type: ignore[method-assign]
        original_display = self.app._display

        def display(screen_arg: Any, renderable: Any) -> None:
            started = time.perf_counter()
            try:
                if isinstance(renderable, ChopsUpdate):
                    self.counts["partial"] += 1
                    self.spans.append(len(renderable.spans))
                    self.rows.append(len({y for y, _x1, _x2 in renderable.spans}))
                    self.bytes.append(len(renderable.render_segments(self.app.console).encode()))
                elif isinstance(renderable, LayoutUpdate):
                    self.counts["full"] += 1
                    self.rows.append(renderable.region.height)
                    self.bytes.append(len(renderable.render_segments(self.app.console).encode()))
                elif renderable is None:
                    self.counts["none"] += 1
                else:
                    self.counts[type(renderable).__name__] += 1
            finally:
                self.display_times.append(time.perf_counter() - started)
            original_display(screen_arg, renderable)

        self.app._display = display  # type: ignore[method-assign]

    def reset(self) -> None:
        self.counts.clear()
        self.bytes.clear()
        self.spans.clear()
        self.rows.clear()
        self.layout_times.clear()
        self.display_times.clear()

    def report(self) -> str:
        return (
            f"counts={dict(self.counts)} bytes={self.bytes} spans={self.spans} rows={self.rows} "
            f"layout={_summary(self.layout_times)} display={_summary(self.display_times)}"
        )


async def _mounted_app(job_count: int) -> tuple[OperatorApp, Any, Any]:
    session = CanonicalFakeSession(running=True)
    now = time.time()
    if job_count > len(session.jobs.list()):
        historical = [
            AsyncJob(
                id=f"history-{index}",
                type="task",
                label=f"historical child {index}",
                status="completed",
                start_time=now - 10_000 + index,
                started_at=now - 10_000 + index,
                settled_at=now - 9_000 + index,
                agent_role="coder",
            )
            for index in range(job_count - 1)
        ]
        active = session.jobs.list()[-1]
        session.jobs._jobs = {job.id: job for job in [*historical, active]}
    app = OperatorApp(lambda: _factory(session))
    context = app.run_test(size=(120, 40))
    pilot = await context.__aenter__()
    for _ in range(80):
        await pilot.pause()
        if app._session is session:
            break
    view = app._transcript_view()
    with view.batch_append():
        for index in range(BLOCKS):
            view.append_block(NoticeBlock(f"retained event {index} " + "word " * 12, "info"))
    await pilot.pause()
    await pilot.pause()
    app._refresh_band()
    await pilot.pause()
    panel = app._subagent_panel
    assert panel is not None
    panel._stop_spinner()
    # Prevent the app's 1 Hz poll from rearming the native timer while this
    # probe drives exact manual ticks; queued UI work is drained before capture.
    panel._start_spinner = lambda: None  # type: ignore[method-assign]
    await pilot.pause()
    await pilot.pause()
    return app, pilot, context


async def scenario(name: str, *, mode: str, job_count: int = 4) -> None:
    app, pilot, context = await _mounted_app(job_count)
    panel = app._subagent_panel
    assert panel is not None
    try:
        if mode == "hidden":
            panel.display = False
        elif mode == "unmounted":
            await panel.remove()
        if mode in {"hidden", "unmounted"}:
            # Display/remove changes post several resize/layout messages. Drain
            # the transition fully so capture contains steady-state ticks only.
            for _settle in range(30):
                await asyncio.sleep(0)
                app.screen._on_timer_update()
        # Flush the transition before installing capture. Each measured sample
        # then consists of exactly one explicit tick and one compositor drain;
        # wall-clock Textual timers cannot add unrelated cursor/status frames.
        app.screen._on_timer_update()
        probe = Probe(app)
        probe.reset()
        tick_times: list[float] = []
        for _ in range(TICKS):
            started = time.perf_counter()
            if mode != "disabled":
                panel._tick()
            # Static.update posts Prompt to the row pump; a few zero-time yields
            # drain row → screen invalidation without Pilot.pause's 50 ms sleep.
            for _yield in range(5):
                await asyncio.sleep(0)
            app.screen._on_timer_update()
            tick_times.append(time.perf_counter() - started)
        print(
            f"{name}: jobs={job_count} transcript_blocks={BLOCKS} ticks={TICKS} "
            f"tick_roundtrip={_summary(tick_times)} {probe.report()}"
        )
    finally:
        await context.__aexit__(None, None, None)


async def main() -> None:
    await scenario("spinner-visible-exact", mode="visible", job_count=4)
    await scenario("spinner-disabled-exact", mode="disabled", job_count=4)
    await scenario("spinner-hidden-exact", mode="hidden", job_count=4)
    await scenario("spinner-unmounted-exact", mode="unmounted", job_count=4)
    await scenario("spinner-visible-100-jobs", mode="visible", job_count=100)


if __name__ == "__main__":
    asyncio.run(main())
