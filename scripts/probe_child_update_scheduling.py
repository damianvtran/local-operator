"""Count redundant parent work for relayed child progress boundaries."""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.harness.jobs import AsyncJobManager  # noqa: E402
from local_operator.harness.subagent import _make_relay  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    Message,
    MessageStartEvent,
    SubagentProgressEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.session.frontend_state import (  # noqa: E402
    FrontendSessionState,
    FrontendStateStore,
)

EVENTS = 60


def _summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    values = sorted(values)
    return (
        f"n={len(values)} median={statistics.median(values) * 1000:.3f}ms "
        f"p95={values[int(.95 * (len(values) - 1))] * 1000:.3f}ms "
        f"max={max(values) * 1000:.3f}ms"
    )


async def main() -> None:
    counts = {
        "job_change_callback": 0,
        "persist_callback": 0,
        "coalesce_scheduled": 0,
        "parent_progress_emitted": 0,
        "observe_event": 0,
        "refresh_from_session": 0,
        "refresh_jobs_delayed": 0,
        "frontend_updates": 0,
        "ui_event_band_refresh": 0,
        "ui_frontend_apply_band_refresh": 0,
    }
    scheduled = False

    def persist() -> None:
        counts["persist_callback"] += 1

    def job_change() -> None:
        nonlocal scheduled
        counts["job_change_callback"] += 1
        if not scheduled:
            scheduled = True
            counts["coalesce_scheduled"] += 1

    manager = AsyncJobManager(on_roster_change=persist, on_job_change=job_change)
    job_id = manager.register("task", "streaming child", lambda *_: asyncio.sleep(3600))
    # Enter the registered coroutine before the synchronous burst so cancelling
    # it at teardown cannot leave an un-awaited coroutine object behind.
    await asyncio.sleep(0)
    job = manager.get(job_id)
    assert job is not None
    job.prompt = "bounded instruction"
    job.trajectory = []
    store = FrontendStateStore(FrontendSessionState(session_id="bench", epoch="bench", jobs=[]))
    session = SimpleNamespace(
        jobs=manager,
        _subagent_comms=None,
        model=None,
        effective_model=None,
        session_id="bench",
        cwd="/tmp",
        queued_steering=lambda: [],
        conversation_name="benchmark",
        goal="",
        active_agent="",
        active_team_name="",
        wake_scheduler=None,
        mcp_manager=None,
        mcp_startup=None,
    )
    store.refresh_from_session(session, initial=True)
    store.subscribe(
        lambda _update: counts.__setitem__("frontend_updates", counts["frontend_updates"] + 1)
    )

    original_refresh = store.refresh_from_session
    fallback_costs: list[float] = []

    def measured_refresh(*args: Any, **kwargs: Any):  # noqa: ANN201
        counts["refresh_from_session"] += 1
        started = time.perf_counter()
        try:
            return original_refresh(*args, **kwargs)
        finally:
            fallback_costs.append(time.perf_counter() - started)

    store.refresh_from_session = measured_refresh  # type: ignore[method-assign]

    async def emit(event: Any) -> None:
        counts["parent_progress_emitted"] += 1
        if isinstance(event, SubagentProgressEvent):
            counts["observe_event"] += 1
            before = store.state.sequence
            store.observe_event(session, event)
            after = store.state.sequence
            # EventController still posts the raw edge, but the owner handler is
            # intentionally a no-op; the delayed canonical jobs publication is
            # the owner/follower-common visual invalidation.
            if after != before:
                counts["ui_frontend_apply_band_refresh"] += 1

    relay = _make_relay(
        job_id,
        "streaming child",
        job,
        manager,
        emit,
        manager._progress_fn(job_id),
        {},
    )

    relay_costs: list[float] = []
    for index in range(EVENTS):
        phase = index % 3
        if phase == 0:
            event = MessageStartEvent(message=Message(role="assistant", content=[], id=f"m{index}"))
        elif phase == 1:
            event = ToolExecutionStartEvent(
                tool_call_id=f"t{index}", tool_name="read", args={"path": "fixture.py"}
            )
        else:
            prior = index - 1
            event = ToolExecutionEndEvent(
                tool_call_id=f"t{prior}",
                tool_name="read",
                result=ToolResult(tool_call_id=f"t{prior}", tool_name="read", content=[]),
            )
        started = time.perf_counter()
        await relay(event)
        relay_costs.append(time.perf_counter() - started)

    # One 50 ms coalesced callback services the whole same-loop burst. It still
    # rebuilds the job projection even though observe_event's defensive fallback
    # already refreshed canonical state once per progress event.
    if scheduled:
        scheduled = False
        counts["refresh_jobs_delayed"] += 1
        started = time.perf_counter()
        before = store.state.sequence
        store.refresh_jobs(session)
        delayed_cost = time.perf_counter() - started
        if store.state.sequence != before:
            counts["ui_frontend_apply_band_refresh"] += 1
    else:
        delayed_cost = 0.0

    manager._tasks[job_id].cancel()
    await asyncio.gather(manager._tasks[job_id], return_exceptions=True)
    print(f"events={EVENTS} counts={counts}")
    print("relay_total:", _summary(relay_costs))
    print("fallback_refresh_from_session:", _summary(fallback_costs))
    print(f"delayed_refresh_jobs: {delayed_cost * 1000:.3f}ms")


if __name__ == "__main__":
    asyncio.run(main())
