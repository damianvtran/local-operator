"""Measure and capture the `/help` command table, where the flag clauses live.

Why this exists separate from ``scripts/help_keys_shot.py``: that script jumps to
the END of the transcript, because the KEY reference it was written for sits
under the whole command table. The finding this script serves is about the TABLE
itself — `/loop`'s description was sized against a 100-column terminal, so at the
common 80 columns its tail wrapped onto a continuation line that starts in the
command column and reads as a command named ``--clear`` (design D2, the
phantom-command shape #402's round 1 warned about). So it presses `/help`, reads
the painted rows through the block's own line API, and captures the table from
the TOP.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/shot_slash_help_row.py OUT.svg COLSxROWS

The painted rows are printed to stderr: one 75-cell line for `/loop` at 80
columns AFTER, and `... --stop or` + `  --clear` (the orphan) BEFORE. The row is
read through the real compositor (``RichBlock.render_line``), never through a
Rich width arithmetic of our own — the `/help` ceiling is `terminal width - 6`,
four separately-declared reservations, and the app's own comment records how a
subtraction-derived figure went wrong once already.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

for _key in [key for key in os.environ if key.startswith(("CMUX_", "LOP_"))]:
    # An inherited CMUX_WORKSPACE_ID has let a headless run rename the operator's
    # real cmux workspaces (AGENTS.md, "Isolating a run"); drop them before any
    # product import.
    os.environ.pop(_key)

# This script lives in ``scripts/``, one level under the repo root — the same
# depth the sibling shot scripts assume. Retarget this if it ever moves again.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    RichBlock,
    TranscriptView,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The rows the flag clauses ride. `loop` is the one this change edits; `goal` is
#: its sibling, so both are printed whatever the case.
TRACKED = ("/goal", "/loop")


async def main() -> None:
    out = sys.argv[1]
    cols, rows = (int(part) for part in (sys.argv[2] if len(sys.argv) > 2 else "80x30").split("x"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(cols, rows)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/help"
        await pilot.pause()
        await pilot.press("enter")
        # The block mounts and lays out over several frames; one pause captures
        # a half-rendered table.
        for _ in range(6):
            await pilot.pause()
        blocks = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, RichBlock)
        ]
        assert blocks, "the /help block did not mount"
        block = blocks[-1]
        painted = [block.render_line(y).text for y in range(block.size.height)]
        for index, line in enumerate(painted):
            if any(name in line for name in TRACKED) or line.lstrip().startswith("--"):
                print(
                    f"grid={cols} painted[{index}]={line.rstrip()!r} "
                    f"cells={len(line.rstrip())}",
                    file=sys.stderr,
                )
        # The table is at the TOP of the block, so take the transcript to its
        # oldest row through the app's own chord (`ctrl+home`,
        # `action_transcript_home`) — a bare `home` press goes to the focused
        # composer's caret instead, which frames the bottom of the key reference.
        app.action_transcript_home()
        for _ in range(6):
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
