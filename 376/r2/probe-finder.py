"""Finder Cmd+C on an oversized image: does the user get ANY response?

That is the exact shape of issue #372 — a textless pasteboard, a gesture, and
possibly nothing on screen. Sample the toast every pause rather than only at
the end, so a card that appeared and expired is not mistaken for silence.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402
from textual import events  # noqa: E402

from local_operator.clipboard import ClipboardContents  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets import editor as editor_module  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT = Path(sys.argv[1])
F = {}


def big_png(w=3024, h=1964):
    b = io.BytesIO()
    Image.frombytes("RGB", (w, h), os.urandom(w * h * 3)).save(b, "PNG")
    return b.getvalue()


async def main():
    p = OUT / "finder2.png"
    p.write_bytes(big_png())
    F["file_mb"] = round(p.stat().st_size / 1048576, 2)
    editor_module.read_clipboard = lambda *a, **k: ClipboardContents(paths=(str(p),))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        for _ in range(400):
            if app._session is not None:
                break
            await pilot.pause()
        ed = app.query_one(Editor)
        ed.focus()
        await pilot.pause()
        toast = app.query_one(Toast)

        t0 = time.perf_counter()
        app.post_message(events.Paste(""))
        seen = []
        for _ in range(400):
            await pilot.pause()
            if toast.message and (not seen or seen[-1][1] != toast.message):
                seen.append((round(time.perf_counter() - t0, 2), toast.message))
        F["toast_samples"] = seen
        F["final_toast"] = toast.message
        F["buffer"] = ed.text
        F["attached"] = len(ed.referenced_images())
        F["elapsed_s"] = round(time.perf_counter() - t0, 2)
        app.save_screenshot(str(OUT / "new8-finder-copy-oversized.svg"))
    p.unlink(missing_ok=True)
    print(json.dumps(F, indent=2))
    (OUT / "finder.json").write_text(json.dumps(F, indent=2))


asyncio.run(main())
