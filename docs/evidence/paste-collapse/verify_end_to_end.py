"""Manager's independent end-to-end check of paste collapse.

env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/final_verify.py
Run from the worktree root.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from textual import events  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

LOG = "\n".join(f"[{i:04d}] ERROR build failed in module {i}" for i in range(500))
MYPY = "\n".join(f"/usr/local/lib/pkg/mod{i}.py:{i}: error: bad type" for i in range(200))
DRAG_300 = "\n".join(f"/Users/ben/Pictures/photo_{i}.png" for i in range(300))
DRAG_20K = "\n".join(f"/Users/ben/Pictures/photo_{i}.png" for i in range(20000))
FIND = DRAG_20K


async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            await pilot.pause()
            if app._session is not None:
                break
        assert app._session is not None, "session never booted"
        ed = app.query_one(Editor)
        ed.focus()
        await pilot.pause()

        async def probe(label, payload, expect_chip):
            ed.text = ""
            ed._attachments.clear()
            await pilot.pause()
            app.post_message(events.Paste(payload))
            await pilot.pause()
            await pilot.pause()
            chipped = bool(ed._attachments)
            ok = "OK " if chipped == expect_chip else "FAIL"
            print(f"  {ok} {label:34s} chip={chipped!s:5s} rows={ed.document.line_count}")
            return chipped == expect_chip

        print("== collapse decision ==")
        results = []
        results.append(await probe("500-line ERROR log", LOG, True))
        results.append(await probe("200-line mypy log (path-prefixed)", MYPY, True))
        results.append(await probe("300-file refused drag (D5)", DRAG_300, False))
        results.append(await probe("20000-file refused drag (D5)", DRAG_20K, False))
        results.append(await probe("20000-line find dump (accepted cost)", FIND, False))
        results.append(await probe("short 3-line snippet", "a\nb\nc", False))

        print("\n== end to end: payload reaches the model ==")
        ed.text = ""
        ed._attachments.clear()
        await pilot.pause()
        ed.text = "why does this build fail?"
        ed.move_cursor(ed._end_of_buffer())
        app.post_message(events.Paste("\n" + LOG))
        await pilot.pause()
        await pilot.pause()
        composer = ed.text
        await pilot.press("enter")
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            await pilot.pause()
            if app._session.prompts:
                break
        sent = app._session.prompts[-1]
        print(f"  composer  : {composer[:70]!r}")
        print(f"  composer rows: {ed.document.line_count if ed.text else 0}")
        print(f"  sent chars: {len(sent)}  lines: {sent.count(chr(10)) + 1}")
        print(f"  payload intact : {LOG in sent}")
        print(f"  no chip leaked : {'[Paste #' not in sent}")
        print(f"  question kept  : {'why does this build fail?' in sent}")
        results.append(LOG in sent and "[Paste #" not in sent)

        print("\nRESULT:", "ALL PASS" if all(results) else "FAILURES PRESENT")


asyncio.run(main())
