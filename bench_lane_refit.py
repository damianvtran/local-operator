"""Cost of the lane funnel on a ~900-row ledger (the architect's ask for R2).

Measures, on the real app:

  A. a layout pass where the lane did NOT move  -> expected: no row rebuilds
  B. a real lane change (sidebar toggle)        -> rebuilds + end-to-end ms,
                                                   run against BOTH trees so the
                                                   funnel's marginal cost is a
                                                   difference, not a total
  C. the funnel's own pass, isolated            -> the O(rows) walk itself

Run from the worktree root (works on the fixed tree and on the base one):

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python bench_lane_refit.py [ROWS]
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
from pathlib import Path

for _k in tuple(os.environ):
    if _k.startswith("CMUX_"):
        os.environ.pop(_k)

sys.path.insert(0, str(Path.cwd()))

from scripts.visual_capture import isolate_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import TranscriptView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 900


def ledger(view: TranscriptView) -> list[ToolCard]:
    return [
        b for b in view.blocks() if isinstance(b, ToolCard) and b.LEDGER_ROW and b.parent is view
    ]


async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(130, 36)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        started = time.perf_counter()
        for i in range(ROWS):
            card = ToolCard(f"c{i}", "bash", {"command": f"echo row{i}_ " + "x" * 40})
            view.append_block(card)
            card.mark_done("done")
        await pilot.pause()
        await pilot.pause()
        rows = ledger(view)
        print(f"seeded {len(rows)} rows in {time.perf_counter() - started:.2f}s")

        rebuilt: list[str] = []
        original = ToolCard._refresh_row
        # The lane funnel gives this method an optional width; the base tree does
        # not have it, and both trees are measured with the same script.
        takes_width = "width" in inspect.signature(original).parameters

        def counting(self: ToolCard, width: int | None = None) -> None:
            rebuilt.append(self.tool_call_id)
            if takes_width:
                original(self, width)
            else:
                original(self)

        ToolCard._refresh_row = counting  # type: ignore[method-assign]
        try:
            rebuilt.clear()
            view.scroll_relative(y=3, animate=False)
            await pilot.pause()
            app.screen._refresh_layout(scroll=True)
            await pilot.pause()
            print(f"A. scroll pass, lane unchanged: {len(rebuilt)} rows rebuilt")

            rebuilt.clear()
            started = time.perf_counter()
            app.action_toggle_sidebar()
            # 900 rows is a heavy reflow; settle properly before reading counts.
            for _ in range(6):
                await pilot.pause()
            print(
                f"B. lane change to {view.scrollable_content_region.width}: "
                f"{len(rebuilt)} rebuilds, {1000 * (time.perf_counter() - started):.1f} ms "
                "end to end (sidebar + reflow + settle)"
            )

            refit = getattr(view, "_refit_ledger_lane", None)
            if refit is not None:
                rebuilt.clear()
                view._row_width_applied = -1  # force the funnel to do its walk
                started = time.perf_counter()
                refit()
                print(
                    f"C. funnel pass alone: {len(rebuilt)} rebuilds, "
                    f"{1000 * (time.perf_counter() - started):.1f} ms"
                )
            if refit is not None:
                # Worst case: every row still stale, i.e. a lane change where not
                # one row re-fitted itself off its own `Resize`. Forced by
                # clearing the authored width, which is exactly what the missing
                # `Resize` leaves behind.
                rebuilt.clear()
                for block in rows:
                    block._built_width = -1
                view._row_width_applied = -1
                started = time.perf_counter()
                refit()
                print(
                    f"D. funnel pass, all rows stale: {len(rebuilt)} rebuilds, "
                    f"{1000 * (time.perf_counter() - started):.1f} ms"
                )
            print(
                f"authored widths: {sorted({b._built_width for b in rows})} "
                f"lane {view.scrollable_content_region.width}"
            )
        finally:
            ToolCard._refresh_row = original  # type: ignore[method-assign]


asyncio.run(main())
