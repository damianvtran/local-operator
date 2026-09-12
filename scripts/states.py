"""The states a pagination change can break, rendered: fits, empty, narrow, ends.

Each state boots the real app, presses the pagination KEY, and compares the frame
before with the frame after. A no-op gesture (a home at the top, a pagedown at
the bottom, a page on a report that fits) must be byte-identical — a change that
made the key paint without moving anything would show up here. A moving gesture
must land in ONE complete frame whose every body row is the composed line for
its content index.

Usage: env -u NO_COLOR TERM=xterm-256color .venv/bin/python states.py <repo-root> <out-dir>
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/tmp/des994")

import scripts.probe_isolation  # noqa: F401,E402

from svgrows import svg_rows  # noqa: E402

from scripts.visual_capture import save_capture  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from tests.unit.tui.test_analytics_panel import _agg, _tall_report_agg  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
RESULTS: list[dict[str, Any]] = []


def norm(text: str) -> str:
    return text.replace("\u00a0", " ").rstrip()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def rows_of(path: Path) -> dict[int, str]:
    return svg_rows(str(path))


class Frames:
    """Minimal compositor-frame counter (rows/cells/scroll_y at paint time)."""

    def __init__(self, body: Any) -> None:
        self.body = body
        self.frames: list[dict[str, Any]] = []
        self._orig = None

    def __enter__(self) -> "Frames":
        from textual._compositor import ChopsUpdate, Compositor

        self._orig = Compositor.render_update
        orig = self._orig
        body = self.body

        def patched(comp, full=False, screen_stack=None, simplify=False):
            out = orig(comp, full=full, screen_stack=screen_stack, simplify=simplify)
            if isinstance(out, ChopsUpdate):
                spans = sorted(out.spans)
                rows = sorted({y for y, _, _ in spans})
                if rows and set(rows) <= set(range(body.region.y, body.region.y + body.region.height)):
                    self.frames.append(
                        {
                            "scroll_y": float(body.scroll_offset.y),
                            "rows": len(rows),
                            "cells": sum(max(0, x2 - x1) for _, x1, x2 in spans),
                        }
                    )
            return out

        Compositor.render_update = patched  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        from textual._compositor import Compositor

        Compositor.render_update = self._orig  # type: ignore[method-assign]


async def state(label: str, agg: Any, size: tuple[int, int], keys: list[str], out: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = AnalyticsScreen(agg)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        body = screen._scroll
        before_svg = out / f"{label}-before.svg"
        save_capture(app, str(before_svg))
        geo = {
            "grid": [app.size.width, app.size.height],
            "body_region": list(body.region),
            "viewport_height": body.scrollable_content_region.height,
            "max_scroll_y": float(body.max_scroll_y),
            "scroll_y": float(body.scroll_offset.y),
            "content_lines": len(body._lines),
            "scrollbar_visible": body.show_vertical_scrollbar,
        }
        recorder = Frames(body)
        with recorder:
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        after_svg = out / f"{label}-after.svg"
        save_capture(app, str(after_svg))
        lines = [norm(line.plain) for line in body.lines_for_test()]
        rows = rows_of(after_svg)
        x, y, h = body.region.x, body.region.y, body.region.height
        painted = [norm(rows.get(y + i, "")[x:]) for i in range(h)]
        start = int(body.scroll_offset.y)
        mismatches = []
        width = body.scrollable_content_region.width  # content cells; the gutter is outside it
        for i, got in enumerate(painted):
            index = start + i
            past_end = not (0 <= index < len(lines))
            expected = "" if past_end else lines[index]
            # the last cell(s) of the body are the scrollbar gutter; drop them
            if got[:width].rstrip() != expected[:width].rstrip():
                mismatches.append(
                    {
                        "row": i,
                        "content_index": index,
                        "past_end": past_end,
                        "painted": got,
                        "expected": expected,
                    }
                )
        result = {
            "label": label,
            "keys": keys,
            "geometry": geo,
            "geometry_after": {
                "scroll_y": float(body.scroll_offset.y),
                "max_scroll_y": float(body.max_scroll_y),
                "viewport_height": body.scrollable_content_region.height,
            },
            "body_frames": recorder.frames,
            "before_md5": md5(before_svg),
            "after_md5": md5(after_svg),
            "unchanged": md5(before_svg) == md5(after_svg),
            "row_mismatches": mismatches,
            "painted_rows": painted,
        }
        RESULTS.append(result)
        print(
            f"{label:34s} {size} keys={keys} y {geo['scroll_y']}->{result['geometry_after']['scroll_y']} "
            f"body_frames={len(recorder.frames)} {recorder.frames} unchanged={result['unchanged']} "
            f"row_mismatches={len(mismatches)} scrollbar={geo['scrollbar_visible']}"
        )
        for m in mismatches[:3]:
            print(f"      row {m['row']} content {m['content_index']}")
            print(f"        painted |{m['painted'][:110]}|")
            print(f"        expect  |{m['expected'][:110]}|")


async def main() -> None:
    await state("tall-120x45-pagedown", _tall_report_agg(), (120, 45), ["pagedown"], OUT)
    await state("tall-120x45-end-then-pagedown", _tall_report_agg(), (120, 45), ["end", "pagedown"], OUT)
    await state("tall-120x45-home-then-pageup", _tall_report_agg(), (120, 45), ["home", "pageup"], OUT)
    await state("small-fits-120x45-pagedown", _agg(), (120, 45), ["pagedown"], OUT)
    await state("small-fits-120x45-end", _agg(), (120, 45), ["end"], OUT)
    await state("tall-narrow-60x24-pagedown", _tall_report_agg(), (60, 24), ["pagedown"], OUT)
    await state("tall-narrow-60x24-end", _tall_report_agg(), (60, 24), ["end"], OUT)
    (OUT / "states-result.json").write_text(json.dumps(RESULTS, indent=2) + "\n")


asyncio.run(main())
