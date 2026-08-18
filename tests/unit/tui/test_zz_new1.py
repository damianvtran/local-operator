"""NEW-defect probe: a long-running backgrounded bash job vs. completions.

M2's fix made `_outstanding_delegated_jobs()` count bash. Two questions:
  (a) does a long-lived bash job (npm run dev) silence EVERY later completion?
  (b) is there any flush path for a bash-caused deferral? `SubagentEnded` is
      posted only by harness/subagent.py, i.e. only for `task` jobs.
"""

import asyncio

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import SubagentEnded, TurnEnded, TurnStarted
from local_operator.tui.notify import Notifier
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_notify_wiring import _boot

GHOSTTY = {"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty"}


class RealJobsSession(FakeSession):
    def __init__(self, mgr):
        super().__init__()
        self.jobs = mgr


@pytest.mark.asyncio
async def test_long_running_bash_silences_completions():
    mgr = AsyncJobManager(max_running=15)

    async def forever(job_id, signal, progress):
        await asyncio.sleep(300)

    # The user backgrounds `npm run dev` early in the session and forgets it.
    mgr.register(type="bash", label="npm run dev", run=forever)
    await asyncio.sleep(0.15)

    session = RealJobsSession(mgr)
    session.set_conversation_name("S")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        writes = []
        n = Notifier(writes.append, enabled=True, env=dict(GHOSTTY), platform="darwin")
        n.set_focused(False)
        app._notifier = n

        print("outstanding =", app._outstanding_delegated_jobs())

        # Three completely unrelated later turns, none of which delegate.
        for i in range(3):
            app.on_turn_started(TurnStarted())
            await pilot.pause()
            app.on_turn_ended(TurnEnded(aborted=False, error=None))
            await pilot.pause()
            print(f"turn {i+1} ended -> writes={len(writes)} deferred={app._completion_deferred}")

        print("\nRESULT writes:", writes)
        print("A) 3 unrelated turns produced", len(writes), "toasts (expected 3)")

        # (b) Is there a flush path? bash never posts SubagentEnded.
        print("\n_completion_deferred stuck at:", app._completion_deferred)
        print("SubagentEnded is posted only for `task` jobs (harness/subagent.py)")

    for j in mgr.list():
        await mgr.cancel(j.id)
