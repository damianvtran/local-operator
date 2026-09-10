"""Capture the row-cursor states of ``/analytics`` that the frames must prove.

Three things the collapse feature claims and only a rendered frame can settle:

1. The cursor on an EXPANDABLE root — the caret and the ``▸`` glyph have to be
   legible together, and the dim ``+N subagents`` count must read as metadata
   rather than as the tail of the session's name (review D3).
2. The cursor on a CHILDLESS root — it must NOT look interactive: blank
   disclosure cells, no count, and ``enter`` leaves the frame unchanged.
3. The hint line at the two widths where it used to wrap to an unpainted second
   line, losing exactly the copy that documents ``enter``/``e`` (review U3).

The frames are named after the state they show, and the state is ASSERTED before
the capture rather than reached by a fixed number of key presses. Round 1's
artifacts had the expandable and childless frames swapped precisely because
press count was used as a proxy for row kind, and which row N presses reaches
depends on the seeded ledger's ordering.

Usage: ``python -m scripts.analytics_cursor_shot <out-dir>``
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import scripts.probe_isolation  # noqa: F401  — isolates the config dir and clears CMUX_*
from local_operator.analytics.store import AnalyticsStore
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen
from scripts.analytics_collapse_shot import (
    _factory_for,
    _make_session,
    _walk_cursor_to,
    seed,
)
from scripts.visual_capture import save_capture
from tests.unit.tui.test_slash_echo import _submit


async def _open_analytics(pilot, app) -> AnalyticsScreen:
    await pilot.pause()
    await _submit(pilot, app, "/analytics")
    await app.workers.wait_for_complete()
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, AnalyticsScreen), f"expected /analytics, got {type(screen)}"
    # Page down to the session table: it sits below the totals, both charts and
    # the input attribution, and the cursor only addresses table rows.
    for _ in range(4):
        await pilot.press("pagedown")
        await pilot.pause()
    return screen


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    store = AnalyticsStore()
    seed(store)
    store.close()

    states: list[dict[str, object]] = []

    # -- the two cursor states, at the width the PR's other frames use ---------
    app = OperatorApp(lambda: _factory_for(_make_session()))
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_analytics(pilot, app)

        for name, expandable in (("cursor-expandable", True), ("cursor-childless", False)):
            session_id = await _walk_cursor_to(pilot, screen, expandable=expandable)
            row = screen._cursor_row()
            assert row is not None and row.expandable is expandable
            before = "\n".join(screen.render_lines_for_test())
            await pilot.press("enter")
            await pilot.pause()
            after = "\n".join(screen.render_lines_for_test())
            save_capture(app, str(out / f"{name}.svg"))
            states.append(
                {
                    "state": name,
                    "session_id": session_id,
                    "expandable": row.expandable,
                    "descendants": row.descendants,
                    # A childless row must be inert: the blank gutter is the
                    # whole signal, so the frame must not change on the press.
                    "enter_changed_the_frame": before != after,
                }
            )
            if expandable:
                # Put it back so the next state starts from the collapsed table.
                await pilot.press("enter")
                await pilot.pause()

    # -- the hint at the widths where it used to wrap --------------------------
    for width, height in ((50, 16), (60, 20)):
        app = OperatorApp(lambda: _factory_for(_make_session()))
        async with app.run_test(size=(width, height)) as pilot:
            screen = await _open_analytics(pilot, app)
            hint = screen._hint_text(scrollable=True).plain
            box = screen._hint.content_size.width
            save_capture(app, str(out / f"hint-{width}col.svg"))
            states.append(
                {
                    "state": f"hint-{width}col",
                    "hint": hint,
                    "hint_cells": len(hint),
                    "hint_box_cells": box,
                    "fits_one_line": len(hint) <= box,
                    "names_expand_keys": "enter" in hint and "e all" in hint,
                }
            )

    (out / "cursor-states.json").write_text(json.dumps(states, indent=2) + "\n")
    for state in states:
        print(json.dumps(state))


if __name__ == "__main__":
    asyncio.run(main())
