"""Design-round 2 capture matrix for PR #1193 (`/links`) — scratch only.

Drives the real OperatorApp (production local_operator.tcss) over a seeded
transcript, types `/links` into the real editor, and saves a native SVG frame
per case plus a digest of the geometry and the exact ink each painted row was
given.

    env -u NO_COLOR -u CMUX_WORKSPACE_ID -u CMUX_SURFACE_ID TERM=xterm-256color \
      LO_ROOT=<tree> .venv/bin/python capture.py OUTDIR [CASE...]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get("LO_ROOT", "/Users/damian/lo-wt/open-links-r3")).resolve()
sys.path.insert(0, str(ROOT))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()  # BEFORE app imports

from local_operator.tui import theme as theme_mod  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.link_picker import LinkPickerScreen  # noqa: E402
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

LONG_URL = (
    "https://example.test/reports/2026/09/" + "very-long-segment/" * 4 + "index.html"
)
SEVERAL = (
    "Sources: https://one.test/a, https://two.test/b and https://three.test/c "
    "all say the same thing."
)
MANY = "\n\n".join(f"Source {n}: https://example.test/source/{n}" for n in range(1, 25))

CASES: list[dict] = [
    # D2 — the meta layer: footer glosses, counter words, every row's sender.
    dict(name="meta", size=(100, 30), answer=SEVERAL),
    dict(name="many", size=(100, 30), answer=MANY),
    dict(name="floor", size=(38, 9), answer=SEVERAL),
    # D1 — the too-small notice, the clipping band and the whole-form band.
    dict(name="tiny30x8", size=(30, 8), answer=SEVERAL, transcript_first=True),
    dict(name="tiny36x8", size=(36, 8), answer=SEVERAL),
    dict(name="tiny38x8", size=(38, 8), answer=SEVERAL),
    # The exact switch: 35 content cells hold the long form, 33 do not.
    dict(name="tiny37x8", size=(37, 8), answer=SEVERAL),
    dict(name="tiny35x8", size=(35, 8), answer=SEVERAL),
    # D4/D7 — the sender column at its widest gap and in its shed form.
    dict(name="long", size=(100, 30), answer=f"Here it is: {LONG_URL}\n\nand the short one is https://example.test/short"),
    dict(name="narrow44", size=(44, 24), answer=f"Here it is: {LONG_URL}"),
    # The delta's other half: a body's brackets are now a PAIR.
    dict(
        name="bracket",
        size=(100, 30),
        answer=(
            "See [https://a.test/x] for docs, and the wiki form is "
            "[[https://a.test/y]] as well."
        ),
    ),
    dict(
        name="labelurl",
        size=(100, 30),
        answer="See [https://a.test/x](https://a.test/x) for the source.",
    ),
    # D5 — what the transcript paints for a bare URL, looked at, not read.
    dict(name="transcript_d5", size=(100, 30), answer="Here it is: https://example.test/reports/2026/09/index.html and [the docs](https://docs.example.test/x)."),
]

#: Cases that also get a frame BEFORE `/links` is typed — the transcript's own ink.
TRANSCRIPT_FIRST = {
    "tiny30x8",
    "transcript_d5",
    "bracket",
    "long",
}


def _answer(text: str) -> AssistantBlock:
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _seed(app: OperatorApp, case: dict) -> None:
    app._append_block(UserBlock("how do I roll this out?", fold_width=100))
    app._append_block(_answer(case["answer"]))


class _Opener:
    """Spy for the browser: records the URL, answers with a fixed result."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    async def __call__(self, url: str) -> bool:
        self.urls.append(url)
        return True


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(60):
        await pilot.pause()
        if app._session is not None:
            return


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _style_hex(style) -> str | None:
    color = getattr(style, "color", None)
    if color is None:
        return None
    try:
        return "#%02x%02x%02x" % color.get_truecolor()
    except Exception:  # noqa: BLE001
        return str(color)


def _runs(line) -> list[dict]:
    out = []
    for span in line.spans:
        text = line.plain[span.start : span.end]
        out.append(
            {"text": text, "ink": _style_hex(span.style), "cells": len(text)}
        )
    return out


def _relative_luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    chan = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]


def _ratio(fg: str, bg: str) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _digest(app: OperatorApp, screen, out: Path, extra: dict) -> dict:
    info: dict = {"frame": out.name, "size": tuple(app.screen.size), **extra}
    info["virtual"] = tuple(app.screen.virtual_size)
    info["screen_vscroll"] = app.screen.show_vertical_scrollbar
    info["screen_content"] = [app.screen.size.width, app.screen.size.height]
    try:
        view = app.query_one(TranscriptView)
        info["transcript"] = {
            "region": [view.region.x, view.region.y, view.region.width, view.region.height],
            "size": [view.size.width, view.size.height],
            "virtual": [view.virtual_size.width, view.virtual_size.height],
            "vscroll": view.show_vertical_scrollbar,
        }
    except Exception as exc:  # noqa: BLE001
        info["transcript_error"] = repr(exc)

    if isinstance(screen, LinkPickerScreen):
        body = screen._body
        card = body.parent if body is not None else None
        notice = getattr(screen, "_too_small", None)
        notice_text = None
        if notice is not None and notice.is_mounted and notice.display:
            try:
                rendered = notice.render()
                notice_text = getattr(rendered, "plain", rendered)
                notice_text = str(notice_text)
            except Exception as exc:  # noqa: BLE001
                notice_text = repr(exc)
        info["card"] = {
            "region": None if card is None else [card.region.x, card.region.y, card.region.width, card.region.height],
            "styles_height": None if card is None else str(card.styles.height),
            "styles_padding": None if card is None else str(card.styles.padding),
            "drawable": screen.is_drawable(),
            "row_budget": screen._row_budget(),
            "card_width": screen._card_width(),
            "body_region": None if body is None else [body.region.x, body.region.y, body.region.width, body.region.height],
            "body_size": None if body is None else [body.size.width, body.size.height],
            "selected": screen._selected,
            "offset": screen._offset,
            "targets": [t.url for t in screen._targets],
            "too_small_text": notice_text,
            "too_small_cells": None if notice_text is None else len(notice_text),
        }
        card_text = screen._card_text()
        info["rows"] = [
            {"plain": line.plain, "cells": len(line.plain), "runs": _runs(line)}
            for line in card_text.split("\n")
        ]
    return info


async def run_case(case: dict, outdir: Path, results: list[dict]) -> None:
    name = case["name"]
    size = case["size"]
    opener = _Opener()
    app = OperatorApp(lambda: _factory(FakeSession()))
    with patch("local_operator.mcp.auth.open_browser_quietly", opener):
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await _boot(pilot, app)
            _seed(app, case)
            await pilot.pause()
            await pilot.pause()

            before = None
            if name in TRANSCRIPT_FIRST:
                before = outdir / f"{name}-transcript.svg"
                save_capture(app, str(before))

            await _submit(pilot, app, "/links")
            screen = app.screen
            frame1 = outdir / f"{name}.svg"
            save_capture(app, str(frame1))
            await pilot.pause()
            frame2 = outdir / f"{name}.f2.svg"
            save_capture(app, str(frame2))
            settled = frame2.read_bytes() == frame1.read_bytes()
            if settled:
                frame2.unlink()

            info = _digest(
                app,
                screen,
                frame1,
                {
                    "case": name,
                    "settled_after_one_pause": settled,
                    "second_frame": None if settled else frame2.name,
                    "before_frame": None if before is None else before.name,
                    "sha256": hashlib.sha256(frame1.read_bytes()).hexdigest()[:16],
                },
            )
            results.append(info)
            print(f"[{name}] {json.dumps({k: v for k, v in info.items() if k != 'rows'})}")


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[2:]) or None
    results: list[dict] = []

    # The theme's own numbers, from the theme module rather than a paste.
    dark = theme_mod.BRAND_TOKENS["dark"]
    light = theme_mod.BRAND_TOKENS["light"]
    palette = {
        "dark": {
            k: dark[k] for k in ("overlay", "fg", "muted", "dim", "faint", "accent", "signal", "raised", "amber")
        },
        "light": {
            k: light[k] for k in ("overlay", "ink", "muted", "dim", "faint", "accent", "signal", "raised")
        },
    }
    for ramp, toks in palette.items():
        ground = toks["overlay"]
        print(f"--- {ramp} ramp, ground overlay={ground}")
        for name, hexv in toks.items():
            if name == "overlay":
                continue
            print(f"    {name:8s} {hexv}  {_ratio(hexv, ground):.2f}:1")
    print(f"    faint on dark overlay: {_ratio(dark['faint'], dark['overlay']):.2f}:1")
    print(f"    dim on dark overlay:   {_ratio(dark['dim'], dark['overlay']):.2f}:1")

    for case in CASES:
        if wanted and case["name"] not in wanted:
            continue
        await run_case(case, outdir, results)

    (outdir / "digest.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {outdir / 'digest.json'} ({len(results)} cases)")


asyncio.run(main())
