"""Controlled scheduled work inside the real CLI/lifespan, never a live daemon.

Only the task body is replaced: SchedulerService._spawn_trigger owns the task,
and the real lifespan's shutdown must cancel it on the unsafe baseline. No SSE
or desktop watch lease hides that cancellation from the regression test.
"""

import asyncio
import os
from pathlib import Path

from local_operator.cli import main
from local_operator.scheduler_service import SchedulerService

root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
original_start = SchedulerService.start


async def barrier(
    self: SchedulerService, agent_id_str: str, schedule_id_str: str, prompt: str
) -> None:
    (root / "task-started").touch()
    try:
        while not (root / "task-release").exists():
            await asyncio.sleep(0.05)
        with (root / "task-completed").open("a") as completed:
            completed.write("completed\n")
    except asyncio.CancelledError:
        (root / "task-cancelled").touch()
        raise


async def start(self: SchedulerService) -> None:
    await original_start(self)
    self._spawn_trigger("isolated-agent", "isolated-schedule", "controlled barrier")


SchedulerService._trigger_agent_task = barrier
SchedulerService.start = start

if __name__ == "__main__":
    main()
