"""B1 re-reproduction: question raised WHILE FOCUSED, then user tabs away.

Round 1 defect: the latch recorded 'announced' for a toast the focus gate
suppressed, so the question was never announced -- indefinitely.
Uses the REAL Notifier (real focus gate, real wire) with a capturing sink.
"""

import asyncio
import sys

import pytest
from textual.events import AppBlur, AppFocus

from local_operator.tui.app import OperatorApp
from local_operator.tui.notify import Notifier
from local_operator.tui.widgets.approval import ApprovalBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_notify_wiring import JobsSession, _boot

GHOSTTY = {"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty"}


def real_notifier():
    writes = []
    n = Notifier(writes.append, enabled=True, env=dict(GHOSTTY), platform="darwin")
    return n, writes


@pytest.mark.asyncio
async def test_b1():
    session = JobsSession()
    session.set_conversation_name("S")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        notifier, writes = real_notifier()
        app._notifier = notifier
        print("starts focused:", notifier.focused)

        # Approval raised WHILE FOCUSED
        app._approval = ApprovalBlock("bash", "rm -rf /tmp/x", on_answer=lambda _: None)
        app._refresh_working_activity()
        await pilot.pause()
        print("focused, approval raised -> writes=%d latch=%r" % (len(writes), app._waiting_kind))

        # User tabs away
        app.on_app_blur(AppBlur())
        await pilot.pause()
        print("after blur                -> writes=%d latch=%r" % (len(writes), app._waiting_kind))
        print("  bytes:", writes)

        # Repaints on the SAME unanswered question must not re-notify
        for _ in range(5):
            app._refresh_working_activity()
        await pilot.pause()
        print("after 5 repaints          -> writes=%d" % len(writes))

        assert len(writes) == 1, "B1 NOT FIXED or double-notify"
        print("B1 RESULT: announced exactly once on blur -> FIXED")
