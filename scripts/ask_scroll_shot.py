"""Capture the ask picker with a LONG option list, for scroll visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ask_scroll_shot.py OUT.svg [COLSxROWS] [MOVES] [key] [reveal]

MOVES = number of arrow-down presses before the shot (default 0).
key   = 'down' (default) or 'pagedown' — which key MOVES presses.
reveal= append 'reveal' to press ctrl+e first.

Twelve options, each with a two-line-ish description, so the list windows and
the scrollbar thumb is painted.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.harness.types import AskOption, AskQuestion  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

QUESTION = AskQuestion(
    id="rollout",
    question="Which rollout should the stale-row migration take?",
    options=[
        AskOption(
            label="Drop the rows outright",
            description="Nothing reads the column any more, so the fastest path is to drop the rows and move on.",  # noqa: E501
        ),
        AskOption(
            label="Backfill from the audit log",
            description="Slower, but it keeps history intact by replaying every recorded mutation.",
        ),
        AskOption(
            label="Dual-write for a week",
            description="Safest of the options; needs a follow-up MR to remove the shim once traffic settles.",  # noqa: E501
        ),
        AskOption(
            label="Leave them and add a filter",
            description="Cheapest, but it hides the problem behind a query filter instead of fixing it.",  # noqa: E501
        ),
        AskOption(
            label="Archive to cold storage",
            description="Move the rows to the archive table and keep a pointer for any late audit request.",  # noqa: E501
        ),
        AskOption(
            label="Soft-delete with a tombstone",
            description="Flag the rows deleted and let the nightly compaction reap them out of band.",  # noqa: E501
        ),
        AskOption(
            label="Partition and detach",
            description="Detach the stale partition so the drop is instant and the table stays online.",  # noqa: E501
        ),
        AskOption(
            label="Migrate to the new schema",
            description="Fold the cleanup into the pending schema migration so it ships in one change.",  # noqa: E501
        ),
        AskOption(
            label="Export then truncate",
            description="Dump the rows to object storage first, then truncate the table in a single step.",  # noqa: E501
        ),
        AskOption(
            label="Rewrite via a shadow table",
            description="Build a clean shadow table and swap it in once the row copy has caught up.",  # noqa: E501
        ),
        AskOption(
            label="Throttle a background delete",
            description="Delete in small batches on a background worker to avoid a long lock on the table.",  # noqa: E501
        ),
        AskOption(
            label="Defer to the next release",
            description="Do nothing now and revisit once the read path that once used the column is gone.",  # noqa: E501
        ),
    ],
    recommended=2,
)


async def main() -> None:
    out = sys.argv[1]
    size = (120, 24)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    moves = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    key = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] in ("down", "pagedown") else "down"
    reveal = "reveal" in sys.argv[3:]

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        for turn in range(1, 5):
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
        for _ in range(moves):
            await pilot.press(key)
            await pilot.pause()
        if reveal:
            await pilot.press("ctrl+e")
            await pilot.pause()
        app.save_screenshot(out)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


asyncio.run(main())
