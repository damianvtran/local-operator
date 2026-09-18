"""Design-round capture matrix for PR #1193 (`/links`), PR #1193, /tmp only.

Drives the real OperatorApp (production CSS) over a seeded transcript, types
`/links` into the real editor, and saves a native SVG frame per case plus a
digest of the geometry and the exact ink each row was painted in.

    env -u NO_COLOR -u CMUX_WORKSPACE_ID -u CMUX_SURFACE_ID TERM=xterm-256color \
      LO_ROOT=<tree> .venv/bin/python /tmp/lo-design-1193/capture.py OUTDIR [CASE...]
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
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

# ---------------------------------------------------------------- conversations

LONG_URL = "https://example.test/reports/2026/09/" + "very-long-segment/" * 4 + "index.html"

CASES: list[dict] = [
    dict(
        name="bare",
        size=(100, 30),
        prompt="where is the report?",
        answer="Here it is: https://example.test/reports/2026/09/index.html\n\nThat is the only one.",
    ),
    dict(
        name="prose",
        size=(100, 30),
        prompt="how do I roll this out?",
        answer=(
            "Read the docs at https://docs.example.test/rollout (the first section), "
            "then file the change with the reviewer."
        ),
    ),
    dict(
        name="mdlink",
        size=(100, 30),
        prompt="where is the change?",
        answer=(
            "The change is [the pull request]"
            "(https://gitlab.com/minervaai/core-svc/-/merge_requests/412) and it is merged."
        ),
    ),
    dict(
        name="labelurl",
        size=(100, 30),
        prompt="which source?",
        answer="See [https://a.test/x](https://a.test/x) for the source.",
    ),
    dict(
        name="labelurl_mixed",
        size=(100, 30),
        prompt="which source?",
        answer="See [see https://a.test/x](https://b.test/y) for the target.",
    ),
    dict(
        name="paren2",
        size=(100, 30),
        prompt="which case is that?",
        answer=(
            "The case is https://en.wikipedia.org/wiki/Foo_(bar_(baz)) and the "
            "plain one is https://a.test/plain."
        ),
    ),
    dict(
        name="paren4",
        size=(100, 30),
        prompt="how deep does it go?",
        answer="Deep: https://a.test/a_(b_(c_(d_(e))))_f and then stopped.",
    ),
    dict(
        name="ipv6",
        size=(100, 30),
        prompt="which host?",
        answer="It is https://[2001:db8::1]/path on the internal net.",
    ),
    dict(
        name="several",
        size=(100, 30),
        prompt="collect them",
        answer=(
            "Sources: https://one.test/a, https://two.test/b and https://three.test/c "
            "all say the same thing."
        ),
    ),
    dict(
        name="long",
        size=(100, 30),
        prompt="where is that report?",
        answer=f"Here it is: {LONG_URL}\n\nand the summary is https://example.test/short",
    ),
    dict(
        name="many",
        size=(100, 30),
        prompt="collect the links for the review",
        answer="\n\n".join(
            f"Source {n}: https://example.test/source/{n}" for n in range(1, 25)
        ),
    ),
    dict(
        name="narrow60",
        size=(60, 30),
        prompt="collect them",
        answer=(
            "Sources: https://one.test/a, https://two.test/b and https://three.test/c "
            "all say the same thing."
        ),
    ),
    dict(
        name="narrow44",
        size=(44, 24),
        prompt="where is that report?",
        answer=f"Here it is: {LONG_URL}",
    ),
    dict(
        name="tiny",
        size=(30, 8),
        prompt="where?",
        answer="Here it is: https://example.test/reports/index.html",
    ),
    dict(
        name="empty",
        size=(100, 30),
        prompt="anything there?",
        answer="Nothing here is a web address, I am afraid.",
    ),
]

#: Cases that also get a frame BEFORE `/links` is typed — the transcript's own ink.
TRANSCRIPT_FIRST = {"bare", "mdlink", "paren4", "long", "several", "narrow44"}


def _answer(text: str) -> AssistantBlock:
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _seed(app: OperatorApp, case: dict) -> None:
    app._append_block(UserBlock(case["prompt"], fold_width=100))
    app._append_block(_answer(case["answer"]))


class _Opener:
    """Spy for the browser: records the URL, answers with a fixed result."""

    def __init__(self, *, ok: bool) -> None:
        self.urls: list[str] = []
        self._ok = ok

    async def __call__(self, url: str) -> bool:
        self.urls.append(url)
        return self._ok


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
    except Exception:  # noqa: BLE001 — a named/ANSI colour string is still a fact
        return str(color)


def _runs(line) -> list[dict]:
    """The line's spans as (text, colour) runs, so coverage is exact."""
    out = []
    for span in line.spans:
        out.append(
            {
                "text": line.plain[span.start : span.end],
                "ink": _style_hex(span.style),
                "cells": len(line.plain[span.start : span.end]),
            }
        )
    return out


def _relative_luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    chan = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)
    ]
    return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]


def _ratio(fg: str, bg: str) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _digest(app: OperatorApp, screen, out: Path, extra: dict) -> dict:
    info: dict = {"frame": out.name, "size": tuple(app.screen.size), **extra}
    info["virtual"] = tuple(app.screen.virtual_size)
    info["screen_vscroll"] = app.screen.show_vertical_scrollbar
    info["screen_padding"] = None
    try:
        view = app.query_one(TranscriptView)
        info["transcript"] = {
            "region": [view.region.x, view.region.y, view.region.width, view.region.height],
            "size": [view.size.width, view.size.height],
            "virtual": [view.virtual_size.width, view.virtual_size.height],
            "vscroll": view.show_vertical_scrollbar,
            "scroll_y": view.scroll_y,
        }
        notices = [b.text() for b in view.blocks() if isinstance(b, NoticeBlock)]
        info["notices"] = notices
    except Exception as exc:  # noqa: BLE001
        info["transcript_error"] = repr(exc)

    if isinstance(screen, LinkPickerScreen):
        body = screen._body
        card = body.parent if body is not None else None
        info["card"] = {
            "region": None
            if card is None
            else [card.region.x, card.region.y, card.region.width, card.region.height],
            "styles_height": None if card is None else str(card.styles.height),
            "styles_width": None if card is None else str(card.styles.width),
            "padding": None if card is None else str(card.styles.padding),
            "drawable": screen.is_drawable(),
            "row_budget": screen._row_budget(),
            "card_width": screen._card_width(),
            "content_size": screen._content_size(),
            "body_region": None
            if body is None
            else [body.region.x, body.region.y, body.region.width, body.region.height],
            "body_size": None if body is None else [body.size.width, body.size.height],
            "selected": screen._selected,
            "offset": screen._offset,
            "targets": [t.url for t in screen._targets],
        }
        lines = screen._card_text().split("\n")
        info["rows"] = [
            {"plain": line.plain, "cells": len(line.plain), "runs": _runs(line)}
            for line in lines
        ]
    return info


async def run_case(case: dict, outdir: Path, results: list[dict]) -> None:
    name = case["name"]
    size = case["size"]
    opener_ok = case.get("opener_ok", True)
    opener = _Opener(ok=opener_ok)
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

            await _submit(pilot, app, case.get("cmd", "/links"))
            screen = app.screen
            frame1 = outdir / f"{name}.svg"
            save_capture(app, str(frame1))
            await pilot.pause()
            frame2 = outdir / f"{name}.f2.svg"
            save_capture(app, str(frame2))
            settled = frame2.read_bytes() == frame1.read_bytes()
            if settled:
                frame2.unlink()

            pressed = case.get("press", None)
            info = _digest(
                app,
                screen,
                frame1,
                {
                    "case": name,
                    "kind": case.get("kind", "picker"),
                    "settled_after_one_pause": settled,
                    "second_frame": None if settled else frame2.name,
                    "before_frame": None if before is None else before.name,
                    "opened": list(opener.urls),
                },
            )
            if pressed:
                await pilot.press(*pressed)
                for _ in range(6):
                    await pilot.pause()
                after = outdir / f"{name}-after.svg"
                save_capture(app, str(after))
                await pilot.pause()
                after2 = outdir / f"{name}-after.f2.svg"
                save_capture(app, str(after2))
                if after2.read_bytes() == after.read_bytes():
                    after2.unlink()
                    info["after_settled"] = True
                else:
                    info["after_settled"] = False
                info["after_frame"] = after.name
                view = app.query_one(TranscriptView)
                nb = [b for b in view.blocks() if isinstance(b, NoticeBlock)]
                info["after_notice"] = [b.text() for b in nb]
                info["after_notice_region"] = [
                    [b.region.x, b.region.y, b.region.width, b.region.height] for b in nb
                ]
                info["opened"] = list(opener.urls)
                info["after_screen"] = type(app.screen).__name__
            results.append(info)
            print(json.dumps(info), flush=True)


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[2:])
    cases = [c for c in CASES if not only or c["name"] in only]
    if "copysib" in only:
        cases = [
            dict(name="copy30x8", size=(30, 8), prompt="anything?", answer="It is https://a.test/x and more text", cmd="/copy", kind="sibling"),
            dict(name="copy38x8", size=(38, 8), prompt="anything?", answer="It is https://a.test/x and more text", cmd="/copy", kind="sibling"),
            dict(name="links38x8", size=(38, 8), prompt="anything?", answer="It is https://a.test/x and more text", kind="sibling"),
            dict(name="links30x8", size=(30, 8), prompt="anything?", answer="It is https://a.test/x and more text", kind="sibling"),
        ]
    if only == {"sweep"}:
        cases = [
            dict(
                name=f"sz{c}x{r}",
                size=(c, r),
                prompt="where is it?",
                answer="It is https://a.test/x",
                kind="size-sweep",
            )
            for c, r in [(30, 8), (34, 8), (36, 8), (38, 8), (38, 9), (40, 10), (44, 8), (44, 9), (50, 12)]
        ]
    if not only:
        cases.append(
            dict(
                name="open_ok",
                size=(100, 30),
                prompt="where is the change?",
                answer="The change is [the pull request]"
                "(https://gitlab.com/minervaai/core-svc/-/merge_requests/412) and it is merged.",
                press=("enter",),
                kind="after-press",
            )
        )
        cases.append(
            dict(
                name="open_fail",
                size=(100, 30),
                prompt="where is the change?",
                answer="The change is [the pull request]"
                "(https://gitlab.com/minervaai/core-svc/-/merge_requests/412) and it is merged.",
                press=("enter",),
                kind="after-press",
                opener_ok=False,
            )
        )
        cases.append(
            dict(
                name="open_ok_narrow",
                size=(60, 30),
                prompt="where is the change?",
                answer="See https://gitlab.com/minervaai/core-svc/-/merge_requests/412689 for it.",
                press=("enter",),
                kind="after-press",
            )
        )

    print(
        json.dumps(
            {
                "root": str(ROOT),
                "theme_tokens": sorted(theme_mod.SEMANTIC_TOKENS),
                "link_targets_file": getattr(
                    sys.modules.get("local_operator.tui.link_targets"), "__file__", None
                ),
                "link_targets_sha256": hashlib.sha256(
                    (ROOT / "local_operator/tui/link_targets.py").read_bytes()
                ).hexdigest(),
                "app_file": sys.modules["local_operator.tui.app"].__file__,
                "tokens": {
                    k: theme_mod.semantic_color(k) for k in theme_mod.SEMANTIC_TOKENS
                },
            }
        ),
        flush=True,
    )
    results: list[dict] = []
    for case in cases:
        await run_case(case, outdir, results)
    (outdir / "digest.json").write_text(json.dumps(results, indent=1))


asyncio.run(main())
