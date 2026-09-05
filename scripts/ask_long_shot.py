"""Capture the ask surface carrying a LONG question and long descriptions.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ask_long_shot.py OUT.svg [COLSxROWS] [ROW] [reveal]

The third argument moves the cursor down ``ROW`` times before the shot, which
is what shows whether the selected row can be read in full.

The fourth presses ``ctrl+e`` first, capturing the REVEALED card. Pass it with
a ``ROW`` to check the property the reveal rests on: the card is the same
height whichever row the cursor is on, so two shots at different rows differ in
their text and in nothing else.

This is the reproduction for the truncation report: every option's description
is far wider than any terminal, so a card that only ever draws one line per
description ends every row in `…` with no way to read the rest.
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
    id="canary_v3_direction",
    question=(
        "For the next iteration of the model regression canary battery (v2 currently lives at"
        " ~/workspace/model-canary/canary-eval.md, with the 11-item CORE block fully saturated"
        " at 11/11 for both models across every run logged so far, and essentially all of the"
        " discriminating signal concentrated in the four self-constraint items 13, 16, 17 and 18"
        " of the HARD block), which direction would you like the v3 item-recruitment effort to"
        " take, bearing in mind the calibration result already recorded in the workspace notes"
        " that mechanical difficulty does not discriminate between current frontier models?"
    ),
    options=[
        AskOption(
            label="Double down on self-constraint items",
            description=(
                "Recruit eight to twelve new v3 items drawn exclusively from the self-monitoring"
                " family that items 13, 16, 17 and 18 already occupy — exact word counts under"
                " simultaneous lexical constraints, sentences that must state their own character"
                " length, paragraphs forbidden from containing a letter that the instruction"
                " itself contains — then pilot each candidate three times per model and keep only"
                " those landing in the partial-failure band."
            ),
        ),
        AskOption(
            label="Add an orthogonal instruction-following block",
            description=(
                "Keep CORE and HARD byte-stable exactly as the versioning rule requires, and"
                " append a brand-new third scored block with its own denominator that targets"
                " multi-turn instruction adherence and negative constraints, on the theory that"
                " self-constraint failures and delayed-instruction failures are two faces of the"
                " same weakness."
            ),
        ),
        AskOption(
            label="Fix measurement before adding items",
            description=(
                "Leave the item set alone entirely for now and spend the effort on statistical"
                " power instead: raise the runner from two core / three hard runs per model to ten"
                " or more, record per-item pass rates rather than only block totals, and add"
                " variance bands to runs.md."
            ),
        ),
        AskOption(
            label="Retire the canary in its current form",
            description=(
                "Accept that a coarse hand-graded battery has reached the end of its useful life"
                " now that CORE is saturated and HARD is carried by four items, and either fold"
                " the surviving discriminating items into a proper harness or stop maintaining it."
            ),
        ),
    ],
    recommended=0,
)


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    moves = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    reveal = len(sys.argv) > 4 and sys.argv[4] == "reveal"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
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
        for _ in range(moves):
            await pilot.press("down")
            await pilot.pause()
        if reveal:
            # Through the KEY rather than by writing the state, so the shot
            # captures what a user's keypress produces — including the card
            # refusing the mode at sizes where it would show nothing new.
            await pilot.press("ctrl+e")
            await pilot.pause()
        save_capture(app, out)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


asyncio.run(main())
