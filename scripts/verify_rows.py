"""Verify every captured frame paints EXACTLY the content rows its offset implies.

A stale row, a half-painted band or a clipped row all show up as a mismatch
between a painted body row and the body's own composed line at that content
index. ``ReportView.render_line(y)`` serves ``scroll_offset.y + y``, so the
expected row for painted body row ``i`` is ``lines[scroll_y + i]``.

Usage: env -u NO_COLOR TERM=xterm-256color .venv/bin/python verify_rows.py <repo-root>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/tmp/des994")

import scripts.probe_isolation  # noqa: F401,E402

from analyse import load  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from tests.unit.tui.test_analytics_panel import _tall_report_agg  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def norm(text: str) -> str:
    return text.replace("\u00a0", " ").rstrip()


async def main() -> None:
    out: dict[str, list[str]] = {}
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.pause()
        screen = AnalyticsScreen(_tall_report_agg())
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        out["120x45"] = [norm(line.plain) for line in screen._scroll.lines_for_test()]
        await pilot.resize_terminal(100, 35)
        for _ in range(3):
            await pilot.pause()
        out["100x35"] = [norm(line.plain) for line in screen._scroll.lines_for_test()]

    report = {
        "lines": out,
        "checks": {},
    }
    for tag, grid in (("head", "120x45"), ("base", "120x45")):
        record = load(tag)
        lines = out[grid if grid in out else "120x45"]
        for frame in record["frames"]:
            key = f"{tag}:{frame['label']}"
            box = record["geometry_resize"]["body_region"] if frame["label"].startswith("resize") else record["geometry_start"]["body_region"]
            body_y, body_h = box[1], box[3]
            if frame["label"].startswith("resize"):
                lines_for_frame = out["100x35"]
            else:
                lines_for_frame = out["120x45"]
            start = int(frame["scroll_y"])
            mismatches = []
            for i, painted in enumerate(frame["painted_body_rows"]):
                index = start + i
                expected = lines_for_frame[index] if 0 <= index < len(lines_for_frame) else "<past end of report>"
                if norm(painted) != expected:
                    mismatches.append({"row": i, "content_index": index, "painted": norm(painted), "expected": expected})
            report["checks"][key] = {
                "scroll_y": frame["scroll_y"],
                "rows_checked": len(frame["painted_body_rows"]),
                "mismatches": mismatches,
                "verdict": "EXACT" if not mismatches else f"{len(mismatches)} MISMATCH",
            }
            print(f"{key:28s} y={int(frame['scroll_y']):3d} rows={len(frame['painted_body_rows']):3d} :: {report['checks'][key]['verdict']}")
            for m in mismatches[:4]:
                print(f"     row {m['row']} (content {m['content_index']})")
                print(f"       painted : |{m['painted'][:100]}|")
                print(f"       expected: |{m['expected'][:100]}|")
    Path("/tmp/des994/row-verify.json").write_text(json.dumps(report, indent=2) + "\n")


asyncio.run(main())
