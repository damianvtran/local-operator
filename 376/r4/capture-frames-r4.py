"""Design round 4 frame capture for PR 376 on head 70b3e878.

Drives the REAL OperatorApp under run_test (so local_operator.tcss applies,
per AGENTS.md "Visual validation"), puts it into each state under review, and
exports a Textual SVG per frame plus the geometry numbers behind it.

Scope this round: D12 (both halves), D13, and the frame-level invariants the
composer's attach path could have regressed.

Usage:
  env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/r4-capture.py <outdir>
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/tmp/dsgn376-r4")

from PIL import Image  # noqa: E402
from textual import events  # noqa: E402

from local_operator.clipboard import (  # noqa: E402
    MAX_CLIPBOARD_READ_BYTES,
    ClipboardContents,
    ClipboardImage,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets import editor as editor_module  # noqa: E402
from local_operator.tui.widgets.editor import MAX_ATTACHMENT_BYTES, Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ART = Path("/tmp/r4-art")
FACTS: dict[str, object] = {}

BIG203 = ART / "big203.png"  # 20.29 MB valid PNG
BIG86 = ART / "big86.png"  # 8.60 MB valid PNG


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


async def paste(app, pilot, text, settle=80):
    app.post_message(events.Paste(text))
    for _ in range(settle):
        await pilot.pause()


def toast_geom(app):
    t = app.query_one(Toast)
    return {
        "message": str(t.message),
        "display": bool(t.display),
        "region": [t.region.x, t.region.y, t.region.width, t.region.height],
        "deferred": (
            None if t._deferred is None else [str(t._deferred[0]), t._deferred[1]]
        ),
        "virtual_size": list(app.screen.virtual_size),
        "size": list(app.screen.size),
        "vscrollbar": bool(app.screen.show_vertical_scrollbar),
        "hscrollbar": bool(app.screen.show_horizontal_scrollbar),
    }


async def shot(app, name):
    app.save_screenshot(str(OUT / f"{name}.svg"))


# ---------------------------------------------------------------- D12 mechanism
async def d12_two_routes():
    """The contradiction: the SAME valid PNG via clipboard image bytes and via
    a Finder-style file URL. Both must produce an identical marker, no notice.

    Three routes are captured, because the user reaches this composer by three
    gestures that must agree:
      A. clipboard IMAGE BYTES   (Cmd+V on a screenshot)
      B. clipboard FILE URL      (Finder Cmd+C -> contents.paths)  <- D12's half
      C. path in the paste TEXT  (drag-in / cmux)                  <- same gate
    """
    rows = {}
    for label, src in [("20.3MB", BIG203), ("8.6MB", BIG86)]:
        data = src.read_bytes()
        per = {"source_bytes": len(data)}

        # A. clipboard image bytes
        stub(image=ClipboardImage(data, "image/png"))
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            ed = await boot(pilot, app)
            t0 = time.perf_counter()
            await paste(app, pilot, "")
            per["A_bytes_route"] = {
                "buffer": ed.text,
                "images": len(ed.referenced_images()),
                "attached_bytes": (
                    len(ed.referenced_images()[0].data) if ed.referenced_images() else 0
                ),
                "toast": toast_geom(app),
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
            if ed.referenced_images():
                per["A_img"] = ed.referenced_images()[0]
            await shot(app, f"d12-{label}-A-clipboard-bytes")

        # B. clipboard file URL (Finder Cmd+C)
        stub(paths=(str(src),))
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            ed = await boot(pilot, app)
            t0 = time.perf_counter()
            await paste(app, pilot, "")
            per["B_fileurl_route"] = {
                "buffer": ed.text,
                "images": len(ed.referenced_images()),
                "attached_bytes": (
                    len(ed.referenced_images()[0].data) if ed.referenced_images() else 0
                ),
                "toast": toast_geom(app),
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
            if ed.referenced_images():
                per["B_img"] = ed.referenced_images()[0]
            await shot(app, f"d12-{label}-B-finder-fileurl")

        # C. path in the paste text
        stub()
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            ed = await boot(pilot, app)
            await paste(app, pilot, str(src))
            per["C_pathtext_route"] = {
                "buffer": ed.text,
                "images": len(ed.referenced_images()),
                "toast": toast_geom(app),
            }
            if ed.referenced_images():
                per["C_img"] = ed.referenced_images()[0]
            await shot(app, f"d12-{label}-C-path-text")

        a = per.pop("A_img", None)
        b = per.pop("B_img", None)
        c = per.pop("C_img", None)
        per["identical_marker_A_vs_B"] = (
            per["A_bytes_route"]["buffer"].replace(str(src), "")
            == per["B_fileurl_route"]["buffer"]
        ) or (per["A_bytes_route"]["buffer"] == per["B_fileurl_route"]["buffer"])
        per["marker_A"] = per["A_bytes_route"]["buffer"]
        per["marker_B"] = per["B_fileurl_route"]["buffer"]
        per["marker_C"] = per["C_pathtext_route"]["buffer"]
        per["payload_equal_A_B"] = bool(a and b and a.data == b.data)
        per["payload_equal_B_C"] = bool(b and c and b.data == c.data)
        per["mime_equal"] = bool(a and b and c) and (
            a.mime_type == b.mime_type == c.mime_type
        )
        rows[label] = per
    FACTS["d12_routes"] = rows


# ------------------------------------------------------------------- D12 copy
async def d12_copy_causes(tmp: Path):
    """Every failure that still reaches the `unreadable` branch, rendered.

    The question the new sentence has to answer honestly: "Couldn't attach that
    file. It may be too large, or not an image."
    """
    tmp.mkdir(parents=True, exist_ok=True)
    notes = tmp / "notes.txt"
    notes.write_text("just some prose\n")
    heic = tmp / "shot.heic"
    # A real-enough HEIC: ftyp/heic brand is what `sniff_image` keys on.
    heic.write_bytes(
        b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1" + b"\x00" * 512
    )
    good = tmp / "ok.png"
    good.write_bytes(png(400, 300))
    missing = tmp / "gone.png"
    over = tmp / "over-ingest.png"
    over.write_bytes(png(64, 64))
    with over.open("r+b") as fh:
        fh.seek(0, 2)
        fh.truncate(MAX_CLIPBOARD_READ_BYTES + 1)

    cases = {
        "non-image file": (str(notes),),
        "HEIC": (str(heic),),
        "mixed selection (png + txt)": (str(good), str(notes)),
        "missing path": (str(missing),),
        "over the 64MB ingest ceiling": (str(over),),
    }
    out = {}
    for label, paths in cases.items():
        stub(paths=paths)
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            ed = await boot(pilot, app)
            await paste(app, pilot, "")
            out[label] = dict(toast_geom(app), buffer=ed.text, images=len(ed.referenced_images()))
            slug = label.split()[0].lower().strip("(")
            await shot(app, f"d12-copy-{slug}")
    # narrow widths for the new sentence
    narrow = {}
    for w, h in [(60, 24), (30, 18)]:
        stub(paths=(str(notes),))
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(w, h)) as pilot:
            await boot(pilot, app)
            await paste(app, pilot, "")
            narrow[f"{w}x{h}"] = toast_geom(app)
            await shot(app, f"d12-copy-unreadable-{w}x{h}")
    FACTS["d12_copy"] = out
    FACTS["d12_copy_narrow"] = narrow


# ------------------------------------------------------------------------ D13
async def d13_showing_withdrawn():
    """A SHOWING notice must be withdrawn by the paste that answers it."""
    stub()  # first Cmd+V: clipboard empty -> notice shows
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        await paste(app, pilot, "")
        before = toast_geom(app)
        await shot(app, "d13-1-notice-showing")
        # user copies a screenshot, pastes again
        stub(image=ClipboardImage(png(), "image/png"))
        t0 = time.perf_counter()
        await paste(app, pilot, "")
        after = dict(toast_geom(app), buffer=ed.text, images=len(ed.referenced_images()))
        after["elapsed_s"] = round(time.perf_counter() - t0, 3)
        await shot(app, "d13-2-after-attach-settled")
        await pilot.pause()
        await pilot.pause()
        await shot(app, "d13-3-second-consecutive-frame")
        FACTS["d13"] = {"notice_showing": before, "after_attach": after}


async def d3_d8_deferred_still_retired():
    """The D3/D8 sequence the old drop_deferred was written for: an MCP failure
    owns the slot, an empty paste DEFERS the notice, an image then attaches,
    the MCP card expires. The held card must never surface."""
    stub()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        ed = await boot(pilot, app)
        toast = app.query_one(Toast)
        toast.show("MCP server 'github' failed to start", duration_ms=60000)
        await pilot.pause()
        steps = [("1 mcp owns slot", toast_geom(app))]
        await shot(app, "d3-1-mcp-owns-slot")

        await paste(app, pilot, "")
        steps.append(("2 empty paste defers", toast_geom(app)))
        await shot(app, "d3-2-notice-deferred")

        stub(image=ClipboardImage(png(), "image/png"))
        await paste(app, pilot, "")
        steps.append(("3 screenshot attaches", dict(toast_geom(app), buffer=ed.text)))
        await shot(app, "d3-3-attached")

        toast.dismiss_toast()
        await pilot.pause()
        await pilot.pause()
        steps.append(("4 mcp expires", dict(toast_geom(app), buffer=ed.text)))
        await shot(app, "d3-4-after-expiry")
        FACTS["d3_d8"] = steps


# ------------------------------------------------------------ frame invariants
async def invariants():
    """No reflow between first paint and settled; no overflow at any size."""
    out = {}
    data = BIG86.read_bytes()
    for w, h in [(100, 30), (60, 24), (30, 18)]:
        stub(image=ClipboardImage(data, "image/png"))
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(w, h)) as pilot:
            ed = await boot(pilot, app)
            app.post_message(events.Paste(""))
            # first paint after the attach lands
            for _ in range(80):
                await pilot.pause()
                if ed.referenced_images():
                    break
            await pilot.pause()
            app.save_screenshot(str(OUT / f"inv-{w}x{h}-first.svg"))
            first = toast_geom(app)
            for _ in range(30):
                await pilot.pause()
            app.save_screenshot(str(OUT / f"inv-{w}x{h}-settled.svg"))
            settled = toast_geom(app)
            out[f"{w}x{h}"] = {
                "buffer": ed.text,
                "first": first,
                "settled": settled,
                "reflow": first["virtual_size"] != settled["virtual_size"]
                or first["size"] != settled["size"],
                "overflow": settled["virtual_size"][1] > settled["size"][1],
            }
    FACTS["invariants"] = out


async def main() -> None:
    await d12_two_routes()
    await d12_copy_causes(Path("/tmp/r4-art/causes"))
    await d13_showing_withdrawn()
    await d3_d8_deferred_still_retired()
    await invariants()
    FACTS["caps"] = {
        "MAX_ATTACHMENT_BYTES": MAX_ATTACHMENT_BYTES,
        "MAX_CLIPBOARD_READ_BYTES": MAX_CLIPBOARD_READ_BYTES,
    }
    (OUT / "facts.json").write_text(json.dumps(FACTS, indent=2, default=str))
    print(json.dumps(FACTS, indent=2, default=str))


asyncio.run(main())
