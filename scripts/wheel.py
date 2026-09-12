"""The wheel path on both trees: unchanged by the fix, and it must stay that way.

Drives the REAL handler (``Widget._on_mouse_scroll_down``, the one Textual calls
for a wheel-notch event) ten times over the body and records the compositor
frames it writes; the landed frame is compared across the two trees.

Usage: env -u NO_COLOR TERM=xterm-256color .venv/bin/python wheel.py <repo-root> <out.svg>
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(REPO))

import scripts.probe_isolation  # noqa: F401,E402

from scripts.visual_capture import save_capture  # noqa: E402

from textual import events  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from tests.unit.tui.test_analytics_panel import _tall_report_agg  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.pause()
        screen = AnalyticsScreen(_tall_report_agg())
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        body = screen._scroll

        from textual._compositor import ChopsUpdate, Compositor

        orig = Compositor.render_update
        frames: list[dict] = []
        body_rows = set(range(body.region.y, body.region.y + body.region.height))

        def patched(comp, full=False, screen_stack=None, simplify=False):
            out = orig(comp, full=full, screen_stack=screen_stack, simplify=simplify)
            if isinstance(out, ChopsUpdate):
                spans = sorted(out.spans)
                rows = sorted({y for y, _, _ in spans})
                frames.append(
                    {
                        "scroll_y": float(body.scroll_offset.y),
                        "rows": len(rows),
                        "cells": sum(max(0, x2 - x1) for _, x1, x2 in spans),
                        "inside_body": set(rows) <= body_rows,
                    }
                )
            return out

        Compositor.render_update = patched  # type: ignore[method-assign]
        try:
            for _ in range(10):
                event = events.MouseScrollDown(
                    widget=body,
                    x=10.0,
                    y=5.0,
                    delta_x=0,
                    delta_y=1,
                    button=0,
                    shift=False,
                    meta=False,
                    ctrl=False,
                )
                body._on_mouse_scroll_down(event)
                await pilot.pause()
        finally:
            Compositor.render_update = orig  # type: ignore[method-assign]

        path = OUT
        save_capture(app, str(path))
        print(
            json.dumps(
                {
                    "repo": str(REPO),
                    "scroll_y": float(body.scroll_offset.y),
                    "body_frames": len([f for f in frames if f["inside_body"]]),
                    "frames": frames,
                    "svg_md5": hashlib.md5(path.read_bytes()).hexdigest(),
                },
                indent=2,
            )
        )


asyncio.run(main())
