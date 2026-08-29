"""Does the remedy the new copy NAMES retire the card that named it?

Two messages tell the user to "paste its file path" / "Paste a file path
instead". That remedy goes through the PATH branch, and `EditorPasteAttached`
is documented as posted only by the clipboard route. So: does a deferred
notice survive a successful path paste and get replayed?
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
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets import editor as editor_module  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
F = {}


def png(w=1568, h=200):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (30, 30, 40)).save(b, "PNG")
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


async def paste(app, pilot, text, n=60):
    app.post_message(events.Paste(text))
    for _ in range(n):
        await pilot.pause()


def g(app):
    t = app.query_one(Toast)
    return {
        "showing": t.message,
        "display": bool(t.display),
        "deferred": None if t._deferred is None else str(t._deferred[0]),
    }


async def case_a():
    """SHOWING notice + successful CLIPBOARD paste (implementer's deliberate
    non-fix). Worst case is the message that names a remedy."""
    shot = OUT / "s.png"
    shot.write_bytes(png())
    # An image that is found and refused -> "Couldn't attach that image.
    # Try a smaller one, or paste its file path."
    stub(image=ClipboardImage(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096, "image/png"))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, "")
        first = g(app)
        # user does the remedy: a smaller image, pasted from the clipboard
        stub(image=ClipboardImage(png(), "image/png"))
        await paste(app, pilot, "")
        F["A_showing_then_clipboard_success"] = {
            "after_failure": first,
            "after_success": dict(g(app), buffer=ed.text),
        }
        app.save_screenshot(str(OUT / "new1-showing-over-attachment.svg"))


async def case_b():
    """DEFERRED notice + successful PATH paste — the exact remedy both
    actionable messages name. Does EditorPasteAttached fire?"""
    shot = OUT / "s.png"
    shot.write_bytes(png())
    stub(refused_remote=True)  # "Clipboard images aren't read over SSH..."
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        toast = app.query_one(Toast)
        toast.show("MCP github failed: command not found: gh", duration_ms=10000)
        await pilot.pause()
        await paste(app, pilot, "")
        deferred = g(app)
        # The user follows the advice and pastes the FILE PATH. It attaches.
        await paste(app, pilot, str(shot))
        after_path = dict(g(app), buffer=ed.text)
        # the MCP card expires
        toast.dismiss_toast()
        await pilot.pause()
        await pilot.pause()
        F["B_deferred_then_path_success"] = {
            "after_ssh_notice": deferred,
            "after_path_attach": after_path,
            "after_mcp_expiry": dict(g(app), buffer=ed.text),
        }
        app.save_screenshot(str(OUT / "new2-replay-after-path-remedy.svg"))


async def case_c():
    """Same as B but no MCP card: the SSH notice is SHOWING, the user pastes a
    path successfully, and the card that told them to do it stays up."""
    shot = OUT / "s.png"
    shot.write_bytes(png())
    stub(refused_remote=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, "")
        showing = g(app)
        await paste(app, pilot, str(shot))
        F["C_showing_ssh_then_path_success"] = {
            "after_ssh_notice": showing,
            "after_path_attach": dict(g(app), buffer=ed.text),
        }
        app.save_screenshot(str(OUT / "new3-ssh-card-over-attached-path.svg"))


async def main():
    await case_a()
    await case_b()
    await case_c()
    (OUT / "remedy.json").write_text(json.dumps(F, indent=2))
    print(json.dumps(F, indent=2))


asyncio.run(main())
