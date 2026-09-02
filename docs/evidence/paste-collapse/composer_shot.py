"""Capture a composer frame after pasting a large log.

env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/paste_shot.py out.svg
Run from the worktree root.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())  # run from the repo root

from textual import events  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

LOG = "\n".join(
    f"[{i:04d}] ERROR  build failed in module {i}: unresolved symbol 'widget_{i}'"
    for i in range(500)
)


async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        # Wait for the session to EXIST, or the boot state (model picker /
        # login banner) is what gets painted instead of the composer.
        import time as _t

        deadline = _t.monotonic() + 15.0
        while _t.monotonic() < deadline:
            await pilot.pause()
            if app._session is not None:
                break
        assert app._session is not None, "session never booted"
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.text = "why does this build fail? here is the log:"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        app.post_message(events.Paste("\n" + LOG))
        await pilot.pause()
        await pilot.pause()
        app.save_screenshot(sys.argv[1])
        doc_lines = editor.document.line_count
        print(f"document lines: {doc_lines}")
        print(f"buffer tail: {editor.text[-80:]!r}")


asyncio.run(main())
