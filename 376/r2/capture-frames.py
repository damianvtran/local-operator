"""Design round 2 frame capture for PR 376 on head 33ddf1c0.

Drives the REAL OperatorApp under run_test (so local_operator.tcss applies,
per AGENTS.md "Visual validation"), puts it into each state under review, and
exports a Textual SVG per frame. The clipboard READ is stubbed at the
editor module seam -- the same seam the unit suite uses -- because what is
under design review is what the composer and the toast PAINT, not whether
osascript works.

Usage:
  env -u NO_COLOR TERM=xterm-256color .venv/bin/python shots_r2.py <outdir>
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
FACTS: dict[str, object] = {}


def png(width: int = 1568, height: int = 200, colour=(30, 30, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, "PNG")
    return buf.getvalue()


def stub(image=None, paths=(), refused_remote=False):
    def read_clipboard(*a, **k):
        return ClipboardContents(
            image=image, paths=tuple(paths), refused_remote=refused_remote
        )

    editor_module.read_clipboard = read_clipboard


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


async def paste(app, pilot, text, settle=60):
    app.post_message(events.Paste(text))
    for _ in range(settle):
        await pilot.pause()


def toast_geom(app):
    t = app.query_one(Toast)
    return {
        "message": t.message,
        "display": bool(t.display),
        "region": [t.region.x, t.region.y, t.region.width, t.region.height],
        "actionable": getattr(t, "_actionable", None),
        "deferred": (
            None if t._deferred is None else [str(t._deferred[0]), t._deferred[1]]
        ),
        "virtual_size": list(app.screen.virtual_size),
        "size": list(app.screen.size),
        "vscrollbar": bool(app.screen.show_vertical_scrollbar),
    }


async def shot(app, pilot, name):
    await pilot.pause()
    app.save_screenshot(str(OUT / f"{name}.svg"))


# --------------------------------------------------------------------------
async def d1_whitespace(size=(100, 30)):
    """D1: an indent pasted under `def handler():` must land visibly, with no
    card about images the user was not pasting."""
    stub()  # nothing on the clipboard at all
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        ed = await boot(pilot, app)
        ed.text = "def handler():\n"
        ed.move_cursor(ed.document.end)
        await pilot.pause()
        await shot(app, pilot, "d1-before-paste")
        await paste(app, pilot, "    ")
        FACTS["d1_buffer"] = ed.text
        FACTS["d1_toast"] = toast_geom(app)
        await shot(app, pilot, "d1-whitespace-paste")
        # a second consecutive frame: nothing may reflow after paint
        await pilot.pause()
        await pilot.pause()
        app.save_screenshot(str(OUT / "d1-whitespace-paste-settled.svg"))


async def d1_variants():
    """The payload table, measured through the real app rather than the Host."""
    rows = {}
    for label, payload in [
        ("4-space indent", "    "),
        ("2-space indent", "  "),
        ("single space", " "),
        ("tab", "\t"),
        ("blank line", "\n"),
        ("two blank lines", "\n\n"),
        ("indent inside blanks", "\n    \n"),
        ("EMPTY (terminal signal)", ""),
        ("ordinary text", "hello"),
    ]:
        stub()
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            ed = await boot(pilot, app)
            ed.text = "X"
            ed.move_cursor(ed.document.end)
            await pilot.pause()
            await paste(app, pilot, payload)
            rows[label] = {
                "pasted": payload,
                "buffer": ed.text,
                "toast": app.query_one(Toast).message,
            }
    FACTS["d1_table"] = rows


async def d1_whitespace_with_image():
    """The other half of the judgement call: whitespace pasted while a
    screenshot IS on the clipboard. Does the whitespace or the image win?"""
    stub(image=ClipboardImage(png(), "image/png"))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        ed.text = "def handler():\n"
        ed.move_cursor(ed.document.end)
        await pilot.pause()
        await paste(app, pilot, "    ")
        FACTS["d1_ws_with_image"] = {
            "buffer": ed.text,
            "toast": app.query_one(Toast).message,
            "images": len(ed.referenced_images()),
        }
        await shot(app, pilot, "d1-whitespace-with-image")


# --------------------------------------------------------------------------
async def d2_notices():
    """D2/D4/D6: each reason code's card, at 100x30, rendered."""
    cases = {
        "nothing": dict(),
        "unattachable": dict(image=ClipboardImage(png(9000, 9000), "image/png")),
        "remote": dict(refused_remote=True),
    }
    out = {}
    for reason, kw in cases.items():
        if reason == "unattachable":
            # An image that is FOUND and then refused. A 9000x9000 flat PNG
            # bounds down fine, so force the refusal the way the composer sees
            # it: an undecodable payload sniffed as an image.
            stub(image=ClipboardImage(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096, "image/png"))
        else:
            stub(**kw)
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            ed = await boot(pilot, app)
            await paste(app, pilot, "")
            out[reason] = toast_geom(app)
            out[reason]["buffer"] = ed.text
            await shot(app, pilot, f"d2-{reason}-100x30")
    FACTS["d2"] = out


async def d2_narrow():
    """Re-check the NEW copy at 60 and 30 columns, and 40 for continuity with
    round 1's table."""
    out = {}
    for reason, kw in [
        ("nothing", dict()),
        ("remote", dict(refused_remote=True)),
        ("unattachable", None),
    ]:
        for w, h in [(60, 24), (40, 20), (30, 18)]:
            if kw is None:
                stub(image=ClipboardImage(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096, "image/png"))
            else:
                stub(**kw)
            app = OperatorApp(lambda: _factory(FakeSession()))
            async with app.run_test(size=(w, h)) as pilot:
                await boot(pilot, app)
                await paste(app, pilot, "")
                out[f"{reason}-{w}x{h}"] = toast_geom(app)
                await shot(app, pilot, f"d2-{reason}-{w}x{h}")
    FACTS["d2_narrow"] = out


# --------------------------------------------------------------------------
async def d3_replay():
    """D3: the exact round-1 replay sequence. MCP failure claims the slot, an
    empty paste defers the notice, a screenshot then attaches, the failure
    expires. The held card must be gone."""
    stub()  # step 2: clipboard empty
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        toast = app.query_one(Toast)
        steps = []

        # 1. an actionable MCP failure claims the slot
        toast.show("MCP github failed: command not found: gh", duration_ms=10000)
        await pilot.pause()
        steps.append(("1 mcp claims slot", toast_geom(app)))
        await shot(app, pilot, "d3-1-mcp-holds-slot")

        # 2. Cmd+V with an empty clipboard -> notice deferred, not shown
        await paste(app, pilot, "")
        steps.append(("2 empty paste defers", toast_geom(app)))
        await shot(app, pilot, "d3-2-notice-deferred")

        # 3. user copies a screenshot, Cmd+V again -> attaches
        stub(image=ClipboardImage(png(), "image/png"))
        await paste(app, pilot, "")
        steps.append(("3 screenshot attaches", dict(toast_geom(app), buffer=ed.text)))
        await shot(app, pilot, "d3-3-attached")

        # 4. the MCP toast expires
        toast.dismiss_toast()
        await pilot.pause()
        await pilot.pause()
        steps.append(("4 mcp expires", dict(toast_geom(app), buffer=ed.text)))
        await shot(app, pilot, "d3-4-after-expiry")
        FACTS["d3"] = steps


async def d3_showing_case():
    """The deliberate non-fix: a notice that is SHOWING (not held) when a later
    paste succeeds. The implementer chose drop_deferred over withdraw, so this
    frame is what the user sees. Judge it."""
    stub()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, "")
        showing = toast_geom(app)
        stub(image=ClipboardImage(png(), "image/png"))
        await paste(app, pilot, "")
        FACTS["d3_showing"] = {
            "after_first_paste": showing,
            "after_successful_paste": dict(toast_geom(app), buffer=ed.text),
        }
        await shot(app, pilot, "d3-showing-not-withdrawn")


# --------------------------------------------------------------------------
async def marker_routes():
    """The load-bearing round-1 clean result, re-checked because the shared
    tail was refactored again: the clipboard route and the path route must
    paint a byte-identical marker."""
    data = png(1568, 200)
    tmp = OUT / "shot.png"
    tmp.write_bytes(data)

    stub(image=ClipboardImage(data, "image/png"))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, "")
        clip_text = ed.text
        clip_img = ed.referenced_images()[0]
        await shot(app, pilot, "marker-clipboard-route")

    stub()  # path route: clipboard read finds nothing, the text carries a path
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, str(tmp))
        path_text = ed.text
        path_img = ed.referenced_images()[0]
        await shot(app, pilot, "marker-path-route")

    FACTS["markers"] = {
        "clipboard_text": clip_text,
        "path_text": path_text,
        "text_equal": clip_text == path_text,
        "bytes_equal": clip_img.data == path_img.data,
        "mime_equal": clip_img.mime_type == path_img.mime_type,
    }


async def multi_image_wrap():
    """A populated composer: two images plus prose, soft-wrapped."""
    data = png(1568, 200)
    a = OUT / "a.png"
    a.write_bytes(data)
    b = OUT / "b.png"
    b.write_bytes(png(3000, 2000, (60, 30, 30)))
    stub()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, f"{a} {b}")
        ed.insert("what changed between these two screenshots of the dashboard header?")
        await pilot.pause()
        FACTS["multi"] = {"buffer": ed.text, "toast": toast_geom(app)}
        await shot(app, pilot, "populated-multi-image")


async def main() -> None:
    await d1_whitespace()
    await d1_variants()
    await d1_whitespace_with_image()
    await d2_notices()
    await d2_narrow()
    await d3_replay()
    await d3_showing_case()
    await marker_routes()
    await multi_image_wrap()
    (OUT / "facts.json").write_text(json.dumps(FACTS, indent=2, default=str))
    print(json.dumps(FACTS, indent=2, default=str))


asyncio.run(main())
