"""Measure parent TUI latency while a bounded child trajectory streams.

This benchmark uses synthetic content with the same shape as the reported long
session. It never opens or writes a retained user session.

Run from the checkout with its interpreter:

    env -u NO_COLOR TERM=xterm-256color \
      .venv/bin/python scripts/benchmark_residual_child_lag.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.harness.jobs import (  # noqa: E402
    AsyncJob,
    AsyncJobManager,
    JobStatus,
)
from local_operator.harness.types import SubagentProgressEvent  # noqa: E402
from local_operator.session.frontend_state import (  # noqa: E402
    FrontendSessionState,
    FrontendStateStore,
    JobState,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import SubagentProgress  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

TRANSCRIPT_BLOCKS = 1_457
RETAINED_TRAJECTORY = 500
STREAM_INTERVAL_S = 0.05
STREAM_UPDATES = 80


def _summary(samples: list[float]) -> str:
    if not samples:
        return "n=0"
    ordered = sorted(samples)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return (
        f"n={len(samples)} median={statistics.median(ordered) * 1_000:.3f}ms "
        f"p95={p95 * 1_000:.3f}ms max={max(ordered) * 1_000:.3f}ms"
    )


def _events(count: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    index = 0
    while len(events) + 5 <= count:
        text = f"child message {index} " + "x" * 240
        events.extend(
            [
                {"type": "message_start", "message": {"role": "assistant", "id": f"m{index}"}},
                {
                    "type": "message_update",
                    "message": {"role": "assistant", "id": f"m{index}"},
                    "delta": text,
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "id": f"m{index}",
                        "content": [{"type": "text", "text": text}],
                    },
                },
                {
                    "type": "tool_execution_start",
                    "tool_call_id": f"t{index}",
                    "tool_name": "read",
                    "args": {"path": f"fixture/module_{index}.py"},
                },
                {
                    "type": "tool_execution_end",
                    "tool_call_id": f"t{index}",
                    "tool_name": "read",
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                },
            ]
        )
        index += 1
    return events[:count]


class CanonicalFakeSession(FakeSession):
    """Fake provider/session edges around the production frontend-state contract."""

    def __init__(self, *, running: bool) -> None:
        super().__init__()
        trajectory = _events(RETAINED_TRAJECTORY) if running else []
        now = time.time()
        jobs = [
            JobState(
                id=f"child-{index}",
                type="task",
                label=f"historical child {index}",
                status="completed",
                start_time=now - 600 - index,
                started_at=now - 600 - index,
                settled_at=now - 500 - index,
                prompt="bounded historical instruction",
                agent_role="coder",
            )
            for index in range(3)
        ]
        jobs.append(
            JobState(
                id="active-child",
                type="task",
                label="streaming child",
                status="running" if running else "completed",
                start_time=now - 300,
                started_at=now - 300,
                settled_at=None if running else now - 200,
                latest_details={"progress": "responding"},
                trajectory=trajectory,
                prompt="bounded active instruction",
                agent_role="architect",
            )
        )
        self._jobs_refresh_scheduled = False

        def job_changed() -> None:
            if self._jobs_refresh_scheduled:
                return
            self._jobs_refresh_scheduled = True
            asyncio.get_running_loop().call_later(0.05, self._flush_jobs)

        self.jobs = AsyncJobManager(on_job_change=job_changed)
        self.jobs._jobs = {
            job.id: AsyncJob(
                id=job.id,
                type="task",
                status=cast(JobStatus, job.status),
                start_time=job.start_time,
                started_at=job.started_at,
                settled_at=job.settled_at,
                label=job.label,
                latest_details=(
                    cast(dict[str, Any], dict(job.latest_details))
                    if isinstance(job.latest_details, dict)
                    else None
                ),
                trajectory=list(job.trajectory or []),
                prompt=job.prompt,
                agent_role=job.agent_role,
            )
            for job in jobs
        }  # synthetic retained ledger
        self._frontend_state_store = FrontendStateStore(
            FrontendSessionState(
                session_id=self.session_id,
                epoch="benchmark",
                jobs=jobs,
                conversation_title="Residual child lag fixture",
            )
        )
        self.raw_progress: Callable[[SubagentProgressEvent], None] | None = None

    def _flush_jobs(self) -> None:
        self._jobs_refresh_scheduled = False
        self._frontend_state_store.refresh_jobs(self)

    @property
    def frontend_state(self) -> FrontendSessionState:
        return self._frontend_state_store.state

    def subscribe_frontend(self, callback: Callable[[Any], None]):  # noqa: ANN201
        return self._frontend_state_store.subscribe(callback)

    def refresh_frontend_usage(self) -> None:
        return None

    def stream_edge(self, index: int) -> None:
        jobs = self.jobs.list()
        active = jobs[-1]
        event = {
            "type": "message_update",
            "message": {"role": "assistant", "id": "live"},
            "delta": f" edge-{index}",
        }
        trajectory = [*(active.trajectory or []), event]
        if len(trajectory) > RETAINED_TRAJECTORY:
            trajectory = trajectory[-RETAINED_TRAJECTORY:]
        changed = active.model_copy(
            update={
                "trajectory": trajectory,
                "latest_details": {"progress": f"stream edge {index}"},
            }
        )
        self.jobs._jobs[active.id] = changed
        self.jobs._progress_fn(active.id)(f"stream edge {index}")
        progress = SubagentProgressEvent(
            job_id=active.id,
            label=active.label,
            progress=f"stream edge {index}",
        )
        # This is Session._emit's ordering: canonical observation first, then
        # EventController fan-out to the owner TUI's raw event handler.
        self._frontend_state_store.observe_event(self, progress)
        if self.raw_progress is not None:
            self.raw_progress(progress)


class Timings:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def wrap(self, owner: Any, name: str, label: str | None = None) -> None:
        original = getattr(owner, name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.values[label or name].append(time.perf_counter() - started)

        setattr(owner, name, measured)


async def _event_loop_probe(stop: asyncio.Event, samples: list[float]) -> None:
    target = time.perf_counter() + 0.01
    while not stop.is_set():
        await asyncio.sleep(max(0.0, target - time.perf_counter()))
        now = time.perf_counter()
        samples.append(max(0.0, now - target))
        target += 0.01


async def _stream(session: CanonicalFakeSession, edge_costs: list[float]) -> None:
    for index in range(STREAM_UPDATES):
        await asyncio.sleep(STREAM_INTERVAL_S)
        started = time.perf_counter()
        session.stream_edge(index)
        edge_costs.append(time.perf_counter() - started)


async def scenario(name: str, *, running: bool, child_visible: bool) -> None:
    session = CanonicalFakeSession(running=running)
    app = OperatorApp(lambda: _factory(session))
    timings = Timings()
    loop_lag: list[float] = []
    edge_costs: list[float] = []
    input_costs: list[float] = []
    frame_costs: list[float] = []
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is session:
                break
        view = app._transcript_view()
        with view.batch_append():
            for index in range(TRANSCRIPT_BLOCKS):
                view.append_block(NoticeBlock(f"retained event {index} " + "word " * 12, "info"))
        await pilot.pause()
        await pilot.pause()

        panel = app._subagent_panel
        if panel is not None:
            timings.wrap(panel, "sync", "panel.sync")
            timings.wrap(panel, "_tick", "panel.tick")
            timings.wrap(panel, "_paint_all", "panel.paint_all")
        timings.wrap(app, "_apply_frontend_state", "app.apply_frontend_state")
        timings.wrap(app, "_refresh_band", "app.refresh_band")
        timings.wrap(session._frontend_state_store, "mutate", "store.mutate")
        timings.wrap(
            session._frontend_state_store,
            "refresh_from_session",
            "store.refresh_from_session",
        )
        timings.wrap(session._frontend_state_store, "refresh_jobs", "store.refresh_jobs")
        session.raw_progress = lambda event: app.on_subagent_progress(
            SubagentProgress(event.job_id, event.label, event.progress)
        )

        if child_visible:
            app._open_subagent_view("active-child")
            await pilot.pause()
            await pilot.pause()
            child = app._subagent_view
            if child is not None:
                timings.wrap(child, "show", "child.show")
                timings.wrap(child, "_tick", "child.tick")

        stop = asyncio.Event()
        probe = asyncio.create_task(_event_loop_probe(stop, loop_lag))
        streamer = asyncio.create_task(_stream(session, edge_costs)) if running else None

        for index in range(50):
            started = time.perf_counter()
            await pilot.pause()
            frame_costs.append(time.perf_counter() - started)
            if not child_visible:
                started = time.perf_counter()
                await pilot.press(chr(ord("a") + index % 26))
                input_costs.append(time.perf_counter() - started)
            await asyncio.sleep(0.05)

        if streamer is not None:
            await streamer
        else:
            await asyncio.sleep(STREAM_INTERVAL_S * STREAM_UPDATES - 2.5)
        stop.set()
        await probe
        await pilot.pause()

    print(f"\n{name}")
    print("  event_loop_lag:", _summary(loop_lag))
    print("  pilot_frame:", _summary(frame_costs))
    if input_costs:
        print("  keypress_roundtrip:", _summary(input_costs))
    if edge_costs:
        print("  stream_edge_sync:", _summary(edge_costs))
    for label in sorted(timings.values):
        print(f"  {label}:", _summary(timings.values[label]))


async def main() -> None:
    print(
        f"fixture transcript_blocks={TRANSCRIPT_BLOCKS} jobs=4 "
        f"trajectory={RETAINED_TRAJECTORY} edges={STREAM_UPDATES}@{STREAM_INTERVAL_S:.2f}s"
    )
    await scenario("idle parent visible", running=False, child_visible=False)
    await scenario("running child, parent visible", running=True, child_visible=False)
    await scenario("running child, child visible", running=True, child_visible=True)


if __name__ == "__main__":
    asyncio.run(main())
