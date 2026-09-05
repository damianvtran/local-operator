"""Capture the approval prompt over a populated transcript.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/approval_shot.py OUT.svg [COLSxROWS] [focus]

The optional third argument reproduces the reported defect: with the composer
focused, the approval's answer keys are typed into the composer instead of
answering the prompt.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    steal_focus = len(sys.argv) > 3 and sys.argv[3] == "focus"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # The app reads the developer's REAL approval mode from
        # ~/.local-operator, and this machine has it on `auto` — which
        # short-circuits the gate and captures a frame with no prompt in it at
        # all. Forced off here so the shot shows the surface under test rather
        # than the operator's saved preference.
        app._set_approve_all(False)
        app._approvals_default_auto = False
        app.query_one(Editor).cursor_blink = False
        for turn in range(1, 7):
            app._append_block(UserBlock(f"Turn {turn}: clean up the stale rows please"))
            prose = AssistantBlock()
            prose.update_text(
                f"Answer {turn}: I will drop them once the audit log backfill is confirmed."
            )
            app._append_block(prose)
        await pilot.pause()

        task = asyncio.create_task(app.request_tool_approval("bash", "rm -rf ./build/stale-cache"))
        for _ in range(8):
            await pilot.pause()

        if steal_focus:
            # The reported defect: the user is looking at the composer, presses
            # the answer key, and it lands in the buffer instead.
            app.query_one(Editor).focus()
            await pilot.pause()
            await pilot.press("y")
            for _ in range(4):
                await pilot.pause()

        save_capture(app, out)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


asyncio.run(main())
