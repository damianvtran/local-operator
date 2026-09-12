"""One page move on the fixed tree, every compositor frame it writes, with the
wall time of each; plus the ``t`` metric toggle's visible effect.

Usage: env -u NO_COLOR TERM=xterm-256color .venv/bin/python landing.py <repo-root>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO))

import scripts.probe_isolation  # noqa: F401,E402

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

        title_before = screen._title_text().plain
        await pilot.press("t")
        await pilot.pause()
        title_after = screen._title_text().plain
        print(f"t: metric {screen._metric} title {title_before!r} -> {title_after!r}")
        await pilot.press("t")
        await pilot.pause()
        print(f"t again: metric {screen._metric}")

        from textual._compositor import ChopsUpdate, Compositor, LayoutUpdate

        orig = Compositor.render_update
        frames: list[dict] = []
        body_rows = set(range(body.region.y, body.region.y + body.region.height))

        def patched(comp, full=False, screen_stack=None, simplify=False):
            t0 = time.perf_counter()
            out = orig(comp, full=full, screen_stack=screen_stack, simplify=simplify)
            ms = (time.perf_counter() - t0) * 1000
            rec: dict = {"ms": round(ms, 2), "scroll_y": float(body.scroll_offset.y)}
            if isinstance(out, ChopsUpdate):
                spans = sorted(out.spans)
                rows = sorted({y for y, _, _ in spans})
                rec.update(
                    kind="chops",
                    rows=len(rows),
                    cells=sum(max(0, x2 - x1) for _, x1, x2 in spans),
                    inside_body=set(rows) <= body_rows,
                    x_span=[min(s[1] for s in spans), max(s[2] for s in spans)] if spans else None,
                )
            elif isinstance(out, LayoutUpdate):
                rec.update(kind="layout", rows=None, cells=None)
            else:
                rec.update(kind="none", rows=None, cells=None)
            frames.append(rec)
            return out

        Compositor.render_update = patched  # type: ignore[method-assign]
        try:
            body.scroll_to(y=0, animate=False)
            for _ in range(3):
                await pilot.pause()
            frames.clear()
            t0 = time.perf_counter()
            await pilot.press("pagedown")
            await pilot.pause()
            wall_ms = (time.perf_counter() - t0) * 1000
        finally:
            Compositor.render_update = orig  # type: ignore[method-assign]

        print(f"pagedown y -> {body.scroll_offset.y}; frames after the gesture ({wall_ms:.1f} ms wall):")
        for f in frames:
            print(f"   {f}")

        Path("/tmp/des994/landing-result.json").write_text(json.dumps({"frames": frames, "wall_ms": wall_ms}, indent=2) + "\n")


asyncio.run(main())
