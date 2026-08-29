"""Which failures reach the "unattachable" copy, and does its advice work?

The message is: "Couldn't attach that image. Try a smaller one, or paste its
file path."  Two remedies. The path route runs the SAME `_attach_image_bytes`
tail, so "paste its file path" can only help when the refusal came from the
CLIPBOARD read rather than from the image. Enumerate the reachable causes and
run both remedies against each.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402
from textual import events  # noqa: E402

from local_operator.clipboard import ClipboardContents, ClipboardImage  # noqa: E402
from local_operator.imaging import bound_image_for_model, sniff_image  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets import editor as editor_module  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
F = {}
Image.MAX_IMAGE_PIXELS = None


def flat_png(w, h):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (12, 90, 140)).save(b, "PNG")
    return b.getvalue()


def stub(image=None, paths=(), refused_remote=False):
    editor_module.read_clipboard = lambda *a, **k: ClipboardContents(
        image=image, paths=tuple(paths), refused_remote=refused_remote
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


async def paste(app, pilot, text, n=140):
    app.post_message(events.Paste(text))
    for _ in range(n):
        await pilot.pause()


async def run(name, data, shot_clip=None, shot_path=None):
    """Paste `data` via the clipboard, then via its file path, and record both."""
    row = {}
    # what the pipeline itself says
    info = sniff_image(data)
    row["sniff"] = None if info is None else {
        "mime": info.mime_type, "sendable": info.sendable,
        "w": info.width, "h": info.height,
    }
    try:
        payload, _m, _s = bound_image_for_model(data, info) if info else (None, None, None)
        row["bound"] = "ok" if payload else "n/a"
    except Exception as exc:  # noqa: BLE001
        row["bound"] = f"{type(exc).__name__}: {str(exc)[:70]}"

    stub(image=ClipboardImage(data, "image/png"))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, "")
        row["clipboard_toast"] = app.query_one(Toast).message
        row["clipboard_buffer"] = ed.text
        if shot_clip:
            app.save_screenshot(str(OUT / shot_clip))

    # now the remedy the card names: paste the same image's FILE PATH
    p = OUT / f"{name}.png"
    p.write_bytes(data)
    stub()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, str(p))
        row["path_toast"] = app.query_one(Toast).message
        row["path_attached"] = len(ed.referenced_images()) == 1
        row["path_buffer_is_literal_path"] = ed.text.strip() == str(p)
        if shot_path:
            app.save_screenshot(str(OUT / shot_path))
    row["advice_paste_file_path_works"] = row["path_attached"]
    F[name] = row


async def main():
    # cause 1: a decompression bomb (> IMAGE_MAX_PIXELS = 50 MP)
    await run(
        "bomb_56MP",
        flat_png(8000, 7000),
        shot_clip="new4-unattachable-clipboard.svg",
        shot_path="new5-unattachable-path-remedy.svg",
    )
    # cause 2: bytes that sniff as PNG and will not decode
    await run("corrupt_png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 6000)

    (OUT / "advice.json").write_text(json.dumps(F, indent=2))
    print(json.dumps(F, indent=2))


asyncio.run(main())
