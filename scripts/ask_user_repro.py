"""Reproduce the user's reported frame: four options, paragraph-long prose.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ask_user_repro.py OUT.svg [COLSxROWS] [ROW] [reveal]

The descriptions here are the ones from the reported screenshot: several
sentences each, far longer than the two-clause consequences the card was
originally measured against.
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
    question="Which rollout strategy should we use for the analytics recorder migration?",
    options=[
        AskOption(
            label="Migrate in place on open",
            description=(
                "The store upgrades itself the first time a session opens it: `_migrate` runs an"
                " idempotent sequence of `ALTER TABLE ADD COLUMN` statements for every column in"
                " `_MIGRATION_COLUMNS` that is not already present, each carrying a DEFAULT so rows"
                " written by older releases read back as a sane value rather than NULL. This is the"
                " path the cost columns already took, so it is well-trodden here, and it means a"
                " user who upgrades mid-week never has to think about their ledger at all — the"
                " first turn after the upgrade quietly widens the table and everything downstream"
                " keeps working. The cost is that the migration runs on the writer thread while a"
                " real turn may be in flight, so a pathological schema change on a very large"
                " ledger could stall the recorder's queue for a noticeable interval; in practice"
                " the ALTERs are metadata-only in SQLite and complete in well under a millisecond"
                " even on a multi-megabyte database, which is why the existing code takes this"
                " route and why it remains the default recommendation for this migration."
            ),
        ),
        AskOption(
            label="Rebuild the ledger into a fresh file",
            description=(
                "Create a new database alongside the old one, copy rows across with the new columns"
                " computed rather than defaulted, then atomically rename it into place once the"
                " copy has been verified. This is the only option that can backfill a column whose"
                " correct historical value cannot be expressed as a constant DEFAULT — for example"
                " a per-call cost that has to be recomputed from the stored token counts against a"
                " price table the old release never had. It is also the slowest and the most"
                " dangerous: several `lop` sessions write to the one file concurrently, so the"
                " rename has to be coordinated against live writers or a session that held the old"
                " connection will keep committing into the old file, and the loss is silent."
            ),
        ),
        AskOption(
            label="Version the columns and read both shapes",
            description=(
                "Leave every existing database exactly as it is and teach the read path to tolerate"
                " either shape: the aggregation queries select the new columns only when `PRAGMA"
                " table_info` reports them, and fall back to a computed expression otherwise."
                " Nothing is ever written to an old ledger that it did not already understand, so"
                " downgrade is free and a user who rolls back a release loses nothing. The price is"
                " paid forever afterwards in the read path, which grows a branch per historical"
                " shape and becomes progressively harder to reason about."
            ),
        ),
    ],
    recommended=0,
)


async def main() -> None:
    out = sys.argv[1]
    size = (190, 50)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    moves = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    reveal = len(sys.argv) > 4 and sys.argv[4] == "reveal"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._append_block(
            UserBlock("give me an ask tool call where the description is really long")
        )
        prose = AssistantBlock()
        prose.update_text("Sure — here is one with paragraph-length options.")
        app._append_block(prose)
        await pilot.pause()

        task = asyncio.create_task(app.request_user_choice([QUESTION]))
        for _ in range(10):
            await pilot.pause()
        for _ in range(moves):
            await pilot.press("down")
            await pilot.pause()
        if reveal:
            await pilot.press("ctrl+e")
            await pilot.pause()
        save_capture(app, out)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


asyncio.run(main())
