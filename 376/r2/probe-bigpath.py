"""The headline case for the "unattachable" copy: a big screenshot.

"Try a smaller one, or paste its file path." Both remedies are named for the
size case. Test the size case end to end on the SAME image:
  1. on the clipboard  -> does it attach, or raise the notice?
  2. saved to disk and pasted as a path -> does the advice work?
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


def retina_screenshot():
    """A dense 3024x1964 PNG: the shape `screencapture -c` produces on a 16in
    Retina display, high-entropy so it does not compress under the cap by
    accident (the mistake behind U1)."""
    w, h = 3024, 1964
    img = Image.frombytes("RGB", (w, h), os.urandom(w * h * 3))
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def stub(image=None):
    editor_module.read_clipboard = lambda *a, **k: ClipboardContents(image=image)


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


async def main():
    data = retina_screenshot()
    F["source_mb"] = round(len(data) / 1048576, 2)
    F["attachment_cap_mb"] = MAX_ATTACHMENT_BYTES / 1048576
    F["on_disk_over_cap"] = len(data) > MAX_ATTACHMENT_BYTES

    stub(ClipboardImage(data, "image/png"))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, "")
        F["clipboard_route"] = {
            "attached": len(ed.referenced_images()) == 1,
            "toast": app.query_one(Toast).message,
            "buffer": ed.text,
        }

    p = OUT / "retina.png"
    p.write_bytes(data)
    stub()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, str(p))
        F["path_route_same_image"] = {
            "attached": len(ed.referenced_images()) == 1,
            "toast": app.query_one(Toast).message,
            "inserted_as_plain_text": ed.text.strip() == str(p),
        }
        app.save_screenshot(str(OUT / "new6-path-route-big-screenshot.svg"))

    p.unlink(missing_ok=True)
    print(json.dumps(F, indent=2))
    (OUT / "bigpath.json").write_text(json.dumps(F, indent=2))


asyncio.run(main())
