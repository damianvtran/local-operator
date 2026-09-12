"""Designer round 1: rendered frames for the /analytics pagination fix (PR #994).

Reads the frames it captures: every still is parsed back into the rows that were
painted (``svgrows.svg_rows``), so the claims this round makes are about pixels
the frame actually contains, not about the source that produced them.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python seamshot.py \
        <repo-root> <out-dir> <tag> <route:key|container>

``route`` selects the path the pagedown KEY took on the tree being captured
(recorded in the JSON): on the unfixed tree a key press reaches the container's
own ``ScrollView`` binding (``Widget.action_page_down``), on the fixed tree the
screen's ``priority=True`` action. Capturing both trees on one route would
either show the fixed tree animating or miss the unfixed tree's animation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
TAG = sys.argv[3]
ROUTE = sys.argv[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scripts.probe_isolation  # noqa: F401,E402  — FIRST: re-home HOME, drop CMUX_*

from svgrows import svg_rows  # noqa: E402

from scripts.visual_capture import save_capture  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from tests.unit.tui.test_analytics_panel import _tall_report_agg  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
RECORD: dict[str, Any] = {"tag": TAG, "repo": str(REPO), "route": ROUTE, "frames": [], "events": []}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class FrameRecorder:
    """Wrap the compositor's ``render_update`` and describe every frame it writes."""

    def __init__(self, body: Any) -> None:
        self.body = body
        self.body_rows: range | None = None
        self.frames: list[dict[str, Any]] = []
        self._orig = None
        self._t0 = 0.0

    def __enter__(self) -> "FrameRecorder":
        from textual._compositor import ChopsUpdate, Compositor, LayoutUpdate

        self._orig = Compositor.render_update
        orig = self._orig

        def patched(comp, full=False, screen_stack=None, simplify=False):
            t_start = time.perf_counter()
            out = orig(comp, full=full, screen_stack=screen_stack, simplify=simplify)
            ms = (time.perf_counter() - t_start) * 1000
            rec: dict[str, Any] = {
                "t_ms": (t_start - self._t0) * 1000,
                "render_ms": round(ms, 2),
                "scroll_y": float(self.body.scroll_offset.y),
            }
            if isinstance(out, ChopsUpdate):
                spans = sorted(out.spans)
                rows = sorted({y for y, _, _ in spans})
                rec.update(
                    kind="chops",
                    rows=rows,
                    row_count=len(rows),
                    cells=sum(max(0, x2 - x1) for _, x1, x2 in spans),
                    x_span=[min(s[1] for s in spans), max(s[2] for s in spans)] if spans else None,
                )
            elif isinstance(out, LayoutUpdate):
                rec.update(kind="full", rows=[], row_count=0, cells=None, x_span=None)
            else:
                rec.update(kind="none", rows=[], row_count=0, cells=None, x_span=None)
            self.frames.append(rec)
            return out

        Compositor.render_update = patched  # type: ignore[method-assign]
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        from textual._compositor import Compositor

        Compositor.render_update = self._orig  # type: ignore[method-assign]

    def body_frames(self) -> list[dict[str, Any]]:
        rows = self.body_rows
        if rows is None:
            return []
        return [
            f
            for f in self.frames
            if f["kind"] == "chops" and f["rows"] and set(f["rows"]) <= set(rows)
        ]

    def summary(self) -> dict[str, Any]:
        body = self.body_frames()
        return {
            "all_frames": len(self.frames),
            "body_frames": len(body),
            "rows_per_body_frame": sorted(f["row_count"] for f in body),
            "cells_per_body_frame": sorted(f["cells"] for f in body),
            "scroll_y_per_body_frame": [f["scroll_y"] for f in body],
            "x_span_per_body_frame": [f["x_span"] for f in body],
        }


def frame_entry(svg: Path, label: str, body_box: tuple[int, int], extra: dict[str, Any]) -> dict[str, Any]:
    rows = svg_rows(str(svg))
    body_y, body_h = body_box
    painted = [rows.get(body_y + offset, "") for offset in range(body_h)]
    entry = {
        "label": label,
        "svg": str(svg),
        "svg_md5": md5(svg),
        "painted_body_rows": painted,
    }
    entry.update(extra)
    RECORD["frames"].append(entry)
    return entry


def body_box(screen: AnalyticsScreen) -> tuple[int, int]:
    body = screen._scroll
    return body.region.y, body.region.height


def geometry(screen: AnalyticsScreen, label: str) -> dict[str, Any]:
    body = screen._scroll
    grid = screen.app.size
    geo = {
        "label": label,
        "grid": [grid.width, grid.height],
        "body_region": list(body.region),
        "body_size": [body.size.width, body.size.height],
        "viewport_height": body.scrollable_content_region.height,
        "viewport_width": body.scrollable_content_region.width,
        "virtual_size": [body.virtual_size.width, body.virtual_size.height],
        "max_scroll_y": float(body.max_scroll_y),
        "scroll_y": float(body.scroll_offset.y),
        "content_lines": len(body._lines),
        "scrollbar_visible": body.show_vertical_scrollbar,
    }
    RECORD["events"].append({"geometry": geo})
    return geo


async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.pause()
        screen = AnalyticsScreen(_tall_report_agg())
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        body = screen._scroll
        box = body_box(screen)
        RECORD["report_lines"] = screen.render_lines_for_test()
        RECORD["geometry_start"] = geometry(screen, "start")

        start = OUT / f"{TAG}-start.svg"
        save_capture(app, str(start))
        frame_entry(start, "start", box, {"scroll_y": float(body.scroll_offset.y)})

        # --- ONE pagedown, first frame it writes ---------------------------------
        rec = FrameRecorder(body)
        rec.body_rows = range(body.region.y, body.region.y + body.region.height)
        if ROUTE == "key":
            with rec:
                await pilot.press("pagedown")
                await pilot.pause()
        else:
            with rec:
                body.action_page_down()
                await pilot.pause()
        after = OUT / f"{TAG}-pagedown-first.svg"
        save_capture(app, str(after))
        entry = frame_entry(
            after,
            "pagedown-first",
            box,
            {"scroll_y": float(body.scroll_offset.y), "frame_summary": rec.summary()},
        )
        RECORD["events"].append({"pagedown_first": entry["frame_summary"]})

        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        settled = OUT / f"{TAG}-pagedown-settled.svg"
        save_capture(app, str(settled))
        frame_entry(settled, "pagedown-settled", box, {"scroll_y": float(body.scroll_offset.y)})

        # --- end, then home ------------------------------------------------------
        rec_end = FrameRecorder(body)
        rec_end.body_rows = range(body.region.y, body.region.y + body.region.height)
        if ROUTE == "key":
            with rec_end:
                await pilot.press("end")
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            end_svg = OUT / f"{TAG}-end.svg"
            save_capture(app, str(end_svg))
            frame_entry(
                end_svg, "end", box, {"scroll_y": float(body.scroll_offset.y), "frame_summary": rec_end.summary()}
            )
            RECORD["geometry_end"] = geometry(screen, "end")

            with rec_end:
                await pilot.press("home")
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            home_svg = OUT / f"{TAG}-home.svg"
            save_capture(app, str(home_svg))
            frame_entry(
                home_svg, "home", box, {"scroll_y": float(body.scroll_offset.y), "frame_summary": rec_end.summary()}
            )
            RECORD["geometry_home"] = geometry(screen, "home")
        else:
            with rec_end:
                body.action_scroll_end()
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            end_svg = OUT / f"{TAG}-end.svg"
            save_capture(app, str(end_svg))
            frame_entry(
                end_svg, "end", box, {"scroll_y": float(body.scroll_offset.y), "frame_summary": rec_end.summary()}
            )
            with rec_end:
                body.action_scroll_home()
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            home_svg = OUT / f"{TAG}-home.svg"
            save_capture(app, str(home_svg))
            frame_entry(
                home_svg, "home", box, {"scroll_y": float(body.scroll_offset.y), "frame_summary": rec_end.summary()}
            )

        # --- a RESIZE, then a page move on the new geometry ----------------------
        body.scroll_to(y=0, animate=False)
        await pilot.pause()
        await pilot.resize_terminal(100, 35)
        for _ in range(3):
            await pilot.pause()
        resized = OUT / f"{TAG}-resize.svg"
        save_capture(app, str(resized))
        box_after = body_box(screen)
        frame_entry(resized, "resize", box_after, {"scroll_y": float(body.scroll_offset.y)})
        RECORD["geometry_resize"] = geometry(screen, "resize")

        rec_resize = FrameRecorder(body)
        rec_resize.body_rows = range(body.region.y, body.region.y + body.region.height)
        if ROUTE == "key":
            with rec_resize:
                await pilot.press("pagedown")
                await pilot.pause()
        else:
            with rec_resize:
                body.action_page_down()
                await pilot.pause()
        rp = OUT / f"{TAG}-resize-pagedown-first.svg"
        save_capture(app, str(rp))
        entry = frame_entry(
            rp,
            "resize-pagedown-first",
            box_after,
            {"scroll_y": float(body.scroll_offset.y), "frame_summary": rec_resize.summary()},
        )
        RECORD["events"].append({"resize_pagedown_first": entry["frame_summary"]})

        # --- seam analysis on the frames themselves ------------------------------
        seam: dict[str, Any] = {}
        first = next(f for f in RECORD["frames"] if f["label"] == "pagedown-first")
        start_rows = RECORD["frames"][0]["painted_body_rows"]
        first_rows = first["painted_body_rows"]
        start_y = float(RECORD["frames"][0]["scroll_y"])
        first_y = float(first["scroll_y"])
        vh = RECORD["geometry_start"]["viewport_height"]
        start_content = list(range(int(start_y), int(start_y) + vh))
        first_content = list(range(int(first_y), int(first_y) + vh))
        seam["start_content_rows"] = start_content
        seam["first_content_rows"] = first_content
        seam["content_rows_in_common"] = sorted(set(start_content) & set(first_content))
        seam["start_last_content_row"] = start_content[-1]
        seam["first_first_content_row"] = first_content[0]
        seam["gap_rows_skipped"] = first_content[0] - start_content[-1] - 1
        seam["painted_rows_repeated_across_the_two_frames"] = sorted(
            {r.strip() for r in start_rows if r.strip()} & {r.strip() for r in first_rows if r.strip()}
        )
        seam["first_painted_top_row"] = first_rows[0]
        seam["start_painted_last_row"] = start_rows[-1]
        seam["step_rows"] = first_y - start_y
        RECORD["seam"] = seam

        (OUT / f"{TAG}-record.json").write_text(json.dumps(RECORD, indent=2) + "\n")

    print(json.dumps({k: v for k, v in RECORD.items() if k != "report_lines"}, indent=2))


asyncio.run(main())
