"""Capture the `/help` key reference, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/help_keys_shot.py OUT.svg [COLSxROWS]

The rows under test (`ctrl+c` and its continuation, `ctrl+v`) sit at the very
BOTTOM of the help block, under the whole command table, so a naive capture
frames the commands and misses them entirely. This drives `/help` for real and
then jumps to the end of the transcript, which is the only way the key
reference is in shot.

Uses the real ``OperatorApp`` so ``local_operator.tcss`` applies: the
lightweight hosts in the test files declare no ``CSS_PATH`` and cannot show a
stylesheet change at all (see AGENTS.md, "Visual validation").
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "help_keys.svg"
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    cols, rows = (int(part) for part in size.split("x"))

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
        await pilot.press("end")
        for _ in range(4):
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
