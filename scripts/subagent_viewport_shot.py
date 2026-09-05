"""Capture opening a child from a 100-job expanded roster at a requested size."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.subagent_panel import SubagentPanel  # noqa: E402
from local_operator.tui.widgets.subagent_view import SubagentView  # noqa: E402
from tests.unit.tui.test_band_panels import (  # noqa: E402
    FakeSession,
    _async_factory,
    _fake_jobs,
)
from tests.unit.tui.test_subagent_view import TRAJECTORY, _job_with  # noqa: E402


async def capture(output: Path, size: tuple[int, int]) -> None:
    jobs = [_job_with(TRAJECTORY, status="completed") for _ in range(100)]
    for index, job in enumerate(jobs):
        job.id = f"sub-{index:03d}"
        job.label = f"task {index:03d}"
    session = FakeSession()
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=size) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        app._open_subagent_view(jobs[-1].id)
        for _ in range(4):
            await pilot.pause()
        panel = app.query_one(SubagentPanel)
        view = app.query_one(SubagentView)
        save_capture(app, str(output))
        print(
            f"size={size[0]}x{size[1]} roster_expanded={panel._expanded} "
            f"view={view.size} body={view._body.size} body_virtual={view._body.virtual_size} "
            f"screen={app.screen.size} screen_virtual={app.screen.virtual_size}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("size", nargs="?", default="120x40")
    args = parser.parse_args()
    width, height = (int(part) for part in args.size.lower().split("x", 1))
    asyncio.run(capture(args.output, (width, height)))


if __name__ == "__main__":
    main()
