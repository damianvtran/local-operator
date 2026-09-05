"""Capture the ask surface over a populated transcript, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/ask_shot.py OUT.svg [COLSxROWS]

Fills the transcript with enough conversation to make the regression under
test visible: if the ask surface covers the conversation, none of the seeded
user/assistant turns are readable behind it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.harness.types import AskOption, AskQuestion  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

QUESTION = AskQuestion(
    id="rollout",
    question="Which rollout should the stale-row migration take?",
    options=[
        AskOption(label="Drop the rows", description="nothing reads the column any more"),
        AskOption(label="Backfill from the audit log", description="slower, keeps history"),
        AskOption(label="Dual-write for a week", description="safest, needs a follow-up MR"),
        AskOption(label="Leave them and add a filter", description="cheapest, hides the problem"),
    ],
    recommended=1,
)


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # Seed a conversation so "can the user still read the transcript?" is
        # an answerable question rather than an empty-screen no-op.
        for turn in range(1, 7):
            app._append_block(UserBlock(f"Turn {turn}: what should we do about the stale rows?"))
            prose = AssistantBlock()
            prose.update_text(
                f"Answer {turn}: the audit log still has every row, so a backfill"
                " is possible. Nothing else reads that column today."
            )
            app._append_block(prose)
        await pilot.pause()

        task = asyncio.create_task(app.request_user_choice([QUESTION]))
        for _ in range(10):
            await pilot.pause()
        save_capture(app, out)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


asyncio.run(main())
