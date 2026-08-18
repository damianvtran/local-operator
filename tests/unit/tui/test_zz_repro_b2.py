"""B2 exact false-finish: capacity filled by bash, a `task` child parked queued.

Then _job_count('task') (the ROUND-1 gate) == 0 -> 'Task complete' fires over a
delegated child that has not started. Driven through the REAL app handler with
the REAL Notifier, on a session backed by the REAL AsyncJobManager.
"""

import asyncio

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import TurnEnded
from local_operator.tui.notify import Notifier
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_notify_wiring import _boot

GHOSTTY = {"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty"}


class RealJobsSession(FakeSession):
    def __init__(self, mgr):
        super().__init__()
        self.jobs = mgr


@pytest.mark.asyncio
async def test_b2_real_manager():
    mgr = AsyncJobManager(max_running=2)

    async def slow(job_id, signal, progress):
        await asyncio.sleep(30)

    # Fill capacity with BASH jobs, then a `task` child arrives and parks.
    for i in range(2):
        mgr.register(type="bash", label=f"bg{i}", run=slow, queued=mgr.at_capacity())
    task_q = mgr.at_capacity()
    mgr.register(type="task", label="child", run=slow, queued=task_q)
    await asyncio.sleep(0.15)

    for j in mgr.list():
        print(f"  {j.label}: type={j.type} status={j.status} queued={j.queued}")
    print("task child registered queued =", task_q)

    session = RealJobsSession(mgr)
    session.set_conversation_name("S")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        writes = []
        n = Notifier(writes.append, enabled=True, env=dict(GHOSTTY), platform="darwin")
        n.set_focused(False)  # user is away
        app._notifier = n

        print("ROUND-1 gate  _job_count('task')        =", app._job_count("task"))
        print("ROUND-2 gate  _outstanding_delegated_jobs =", app._outstanding_delegated_jobs())

        app.on_turn_ended(TurnEnded(aborted=False, error=None))
        await pilot.pause()
        print("writes after turn end:", writes)
        print("_completion_deferred:", app._completion_deferred)

        assert writes == [], "B2 NOT FIXED: false finish over a parked child"
        print("B2 RESULT: suppressed, deferred -> FIXED")

    for j in mgr.list():
        await mgr.cancel(j.id)
