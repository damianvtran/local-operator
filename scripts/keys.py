"""The keys the hint line advertises, driven for real, plus the frame cost of the
landing frame.

The hint under the report says ``esc back · ↑↓ row · enter expand · e all`` and
``t cost/tokens`` (in the tiers that fit it). A pagination change that pinned
four bindings with ``priority=True`` could shadow a sibling key, and the surface
that would show it is the hint's own promise. This drives each key on the fixed
tree and records what changed, then measures the wall time of the compositor
frame the page move writes (reported, never asserted — a bound from this laptop
is a bet on machine load).

Usage: env -u NO_COLOR TERM=xterm-256color .venv/bin/python keys.py <repo-root>
"""

from __future__ import annotations

import asyncio
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

        def state() -> dict[str, Any]:
            return {
                "scroll_y": float(body.scroll_offset.y),
                "cursor": getattr(screen, "_cursor", None),
                "expanded": len(getattr(screen, "_expanded", set()) or set()),
                "metric": getattr(screen, "_metric", None),
                "screens": len(app.screen_stack),
            }

        results: dict[str, Any] = {}
        for key in ("enter", "e", "t", "down", "up", "pagedown", "pageup", "home", "end"):
            before = state()
            await pilot.press(key)
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            after = state()
            changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
            results[key] = {"before": before, "after": after, "changed": changed}
            print(f"{key:10s} changed={changed}")

        # esc last: it pops the screen
        before = state()
        await pilot.press("escape")
        await pilot.pause()
        after = state()
        print(f"{'escape':10s} changed={{'screens': ({before['screens']}, {after['screens']})}}")

        # render cost of the landing frame
        rec: list[tuple[float, float]] = []
        from textual._compositor import ChopsUpdate, Compositor

        orig = Compositor.render_update
        body_rows = set(range(body.region.y, body.region.y + body.region.height))

        def patched(comp, full=False, screen_stack=None, simplify=False):
            t0 = time.perf_counter()
            out = orig(comp, full=full, screen_stack=screen_stack, simplify=simplify)
            ms = (time.perf_counter() - t0) * 1000
            if isinstance(out, ChopsUpdate):
                rows = {y for y, _, _ in out.spans}
                if rows and rows <= body_rows:
                    rec.append((ms, float(body.scroll_offset.y)))
            return out

        Compositor.render_update = patched  # type: ignore[method-assign]
        try:
            body.scroll_to(y=0, animate=False)
            await pilot.pause()
            await pilot.pause()
            rec.clear()
            await pilot.press("pagedown")
            await pilot.pause()
        finally:
            Compositor.render_update = orig  # type: ignore[method-assign]
        print(f"landing frame(s): {[f'{ms:.1f}ms at y={y}' for ms, y in rec]}")
        results["landing_frame_ms"] = rec

    Path("/tmp/des994/keys-result.json").write_text(json.dumps(results, indent=2, default=str) + "\n")


asyncio.run(main())
