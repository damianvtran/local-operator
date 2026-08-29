"""Render the subagent page in the duplicated-notice state.

Drives the REAL ``OperatorApp`` (so ``local_operator.tcss`` applies) and puts
the page in exactly the state the operator screenshotted: a busy child whose
trajectory has reached ``TRAJECTORY_CAP``, with one error notice that the
front-eviction keeps re-keying on every 1 Hz refresh.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/subagent_dupe_shot.py out.svg
Run from the repo root (or a worktree of it) — ``tests.`` must import.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, cast

sys.path.insert(0, os.getcwd())

from local_operator.harness.comms import SubagentComms  # noqa: E402
from local_operator.session.session import Session  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.subagent_view import (  # noqa: E402
    TRAJECTORY_MAX_EVENTS as CAP,
)
from local_operator.tui.widgets.subagent_view import SubagentView  # noqa: E402
from tests.unit.tui.test_band_panels import (  # noqa: E402
    FakeSession,
    _async_factory,
    _fake_jobs,
    _Job,
)

ERR = "Invalid arguments: argument 'edits' does not match type array"

#: How many 1 Hz refreshes to simulate. Each one appends a child event and
#: evicts the oldest, which is what re-spells the notice's positional key.
TICKS = 10


def _filler(index: int) -> dict[str, Any]:
    """An event that occupies a trajectory slot but paints no row.

    A ``message_start`` with no delta folds to an empty stream, and the fold
    spends no row on one — so the frame shows the notice defect rather than a
    wall of scaffolding.
    """
    return {"type": "message_start", "message": {"role": "assistant", "id": f"f{index}"}}


def _seed() -> list[dict[str, Any]]:
    # Exactly CAP events, because the defect only exists AT the cap: below it
    # nothing is evicted, every event keeps its offset, and the positional key
    # is accidentally stable. The four real events below occupy four slots.
    events: list[dict[str, Any]] = [_filler(i) for i in range(CAP - 4)]
    events += [
        {"type": "message_start", "message": {"role": "assistant", "id": "m-visible"}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "id": "m-visible",
                "content": [{"type": "text", "text": "Applying the review fixes to the widget."}],
            },
        },
        {
            "type": "tool_execution_start",
            "tool_call_id": "call-edit",
            "tool_name": "edit",
            "intent": "Applying the review fixes",
            "args": {"path": "local_operator/tui/widgets/subagent_view.py"},
        },
        {"type": "notice", "kind": "error", "text": ERR},
    ]
    return events


async def main() -> None:
    job = _Job("child-1", "coder", status="running")
    # The app's own 1 Hz poll recomputes the header clock from ``start_time``
    # against the wall clock, so a fixture's fixed epoch renders as "100d+" and
    # differs run to run. Anchored to now so the before/after pair differs by
    # the fix under test and nothing else.
    job.start_time = time.time() - 72.0
    job.started_at = job.start_time
    job.trajectory = _seed()
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    comms = SubagentComms(cast(Session, session))
    comms.record_launch(job.id, job.label)
    session._subagent_comms = comms

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._open_subagent_view(job.id)
        await pilot.pause()
        view = app.query_one(SubagentView)

        # The 1 Hz poll: every refresh re-folds the SAME retained window after
        # the engine has evicted from its front. Nothing new happened; only
        # the notice's offset moved.
        for tick in range(TICKS):
            assert job.trajectory is not None
            job.trajectory.append(_filler(10_000 + tick))
            overflow = len(job.trajectory) - CAP
            if overflow > 0:
                del job.trajectory[:overflow]
            view.show(
                job_id=job.id,
                label=job.label,
                status="running",
                queued=False,
                elapsed="1m 12s",
                events=job.trajectory,
                prompt="Fix the duplicate notice rows in the subagent view.",
                progress="Applying the review fixes",
                agent_role="coder",
            )
            await pilot.pause()

        await pilot.pause()
        rows = [row for row in view.rendered_rows() if ERR in row]
        print(f"error rows painted: {len(rows)}")
        app.save_screenshot(sys.argv[1])


asyncio.run(main())
