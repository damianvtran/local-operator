"""Two residuals of the round-1 remediation.

A) D1 residual: whitespace the user really copied is still discarded when an
   image happens to be on the clipboard at the same time, with no notice.
B) D2 residual: the Finder Cmd+C route raises "…or paste its file path" for a
   file the user reached BY its path, and both named remedies are wrong there.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402
from textual import events  # noqa: E402

from local_operator.clipboard import ClipboardContents, ClipboardImage  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets import editor as editor_module  # noqa: E402
from local_operator.tui.widgets.editor import (  # noqa: E402
    MAX_ATTACHMENT_BYTES,
    Editor,
)
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT = Path(sys.argv[1])
F = {}


def png(w=1568, h=200):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (30, 30, 40)).save(b, "PNG")
    return b.getvalue()


def big_png():
    w, h = 3024, 1964
    b = io.BytesIO()
    Image.frombytes("RGB", (w, h), os.urandom(w * h * 3)).save(b, "PNG")
    return b.getvalue()


def stub(image=None, paths=()):
    editor_module.read_clipboard = lambda *a, **k: ClipboardContents(
        image=image, paths=tuple(paths)
    )


async def boot(pilot, app):
    await pilot.pause()
    for _ in range(400):
        if app._session is not None:
            break
        await pilot.pause()
    ed = app.query_one(Editor)
    ed.focus()
    await pilot.pause()
    return ed


async def paste(app, pilot, text, n=200):
    app.post_message(events.Paste(text))
    for _ in range(n):
        await pilot.pause()


async def residual_a():
    """The user copies a four-space indent while a screenshot is still on the
    clipboard from earlier. Which one lands?"""
    rows = {}
    for label, payload in [("4-space indent", "    "), ("tab", "\t"), ("newline", "\n")]:
        stub(image=ClipboardImage(png(), "image/png"))
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            ed = await boot(pilot, app)
            ed.text = "def handler():\n"
            ed.move_cursor(ed.document.end)
            await pilot.pause()
            await paste(app, pilot, payload)
            rows[label] = {
                "pasted": payload,
                "buffer": ed.text,
                "whitespace_kept": payload in ed.text,
                "toast": app.query_one(Toast).message,
            }
            if label == "4-space indent":
                app.save_screenshot(str(OUT / "new7-indent-lost-to-image.svg"))
    F["A_whitespace_vs_image"] = rows


async def residual_b():
    """Finder Cmd+C on a 17 MB screenshot file: textless pasteboard, a
    public.file-url, and the path route's 4 MB stat gate refuses it."""
    p = OUT / "finder.png"
    p.write_bytes(big_png())
    F["B_file_mb"] = round(p.stat().st_size / 1048576, 2)
    F["B_cap_mb"] = MAX_ATTACHMENT_BYTES / 1048576
    stub(paths=(str(p),))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, "")
        F["B_finder_copy"] = {
            "toast": app.query_one(Toast).message,
            "buffer": ed.text,
            "attached": len(ed.referenced_images()) == 1,
        }
        app.save_screenshot(str(OUT / "new8-finder-copy-advice.svg"))
    p.unlink(missing_ok=True)


async def main():
    await residual_a()
    await residual_b()
    print(json.dumps(F, indent=2))
    (OUT / "residual.json").write_text(json.dumps(F, indent=2))


asyncio.run(main())
