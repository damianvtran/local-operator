"""Deterministic reproduction: the ledger's RIGHT-EDGE tear.

The operator's frame: some tool rows fill the transcript lane and some stop short
of it, leaving the right-aligned status (`✓ 0.5s`) stranded mid-row, while the
row's own background slab spans the full lane — so the WIDGET is full width and
only the AUTHORED content is narrow. Hovering one row repairs that one row.

This script drives the real app through the operator's sequence and reports, for
every mounted ledger row, the width its content was AUTHORED at against the
width the reconciler laid it out at:

    built  = ToolCard._built_width   (what the summary/status were built for)
    outer  = Widget.outer_size.width (what the paint and the slab use)

`region`/`size` are deliberately NOT the reference: they are a compositor map
lookup, and in the frame under test the map already answers with the NEW lane,
which is why "the row was built at its region width" cannot see this defect.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python repro_ledger_lane_tear.py

Exit code is 1 when any row is torn, so it can be used as a before/after check.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

for _k in tuple(os.environ):
    if _k.startswith("CMUX_"):
        os.environ.pop(_k)

sys.path.insert(0, str(Path.cwd()))

from scripts.visual_capture import isolate_capture  # noqa: E402

isolate_capture()

from textual._compositor import Compositor  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import TranscriptView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

SIZE = (130, 36)
ROWS = 40
COMMAND = (
    'curl -s https://openrouter.ai/api/v1/models | python3 -c "import json,sys; '
    "ms=json.load(sys.stdin)['data']; print(len(ms))\""
)

_orig_reflow = Compositor.reflow
_orig_reflow_visible = Compositor.reflow_visible


def _reflow(self, parent, size):
    result = _orig_reflow(self, parent, size)
    print(
        f"  [reflow]        size={size.width} shown={len(result.shown)} "
        f"resized={len(result.resized)}"
    )
    return result


def _reflow_visible(self, parent, size):
    exposed = _orig_reflow_visible(self, parent, size)
    print(f"  [reflow_VISIBLE] size={size.width} exposed={len(exposed)}")
    return exposed


Compositor.reflow = _reflow
Compositor.reflow_visible = _reflow_visible


def ledger(view: TranscriptView) -> list[ToolCard]:
    return [b for b in view.blocks() if getattr(b, "LEDGER_ROW", False) and b.parent is view]


def report(view: TranscriptView, label: str) -> int:
    """Print the authored-vs-laid-out widths; return the number of torn rows."""
    groups: dict[int | None, int] = {}
    torn: list[tuple[str, int | None, int]] = []
    for block in ledger(view):
        built = getattr(block, "_built_width", None)
        outer = block.outer_size.width
        groups[built] = groups.get(built, 0) + 1
        if built != outer:
            torn.append((block.tool_call_id, built, outer))
    comp = view.screen._compositor
    print(
        f"--- {label}: lane={view.scrollable_content_region.width} rows={len(ledger(view))} "
        f"authored_widths={groups}\n"
        f"    full_map_invalidated={comp._full_map_invalidated} "
        f"visible_map={'set' if comp._visible_map is not None else 'None'} "
        f"layout_widgets={len(view.screen._layout_widgets)}"
    )
    print(f"    TORN (authored != laid out): {len(torn)} {torn[:4]}")
    return len(torn)


async def main() -> int:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        cards = []
        for i in range(ROWS):
            card = ToolCard(f"c{i}", "bash", {"command": f"{COMMAND}  # {i}"})
            view.append_block(card)
            card.mark_done("done")
            cards.append(card)
        await pilot.pause()
        await pilot.pause()
        report(view, f"seeded {ROWS} rows at {SIZE[0]}x{SIZE[1]}")

        print("== 1. sidebar OPEN: the lane narrows")
        await pilot.press("ctrl+b")
        await pilot.pause()
        await pilot.pause()
        report(view, "sidebar open")

        print("== 2. a wheel frame at that lane (captures the visible-only arrangement)")
        view.scroll_relative(y=-6, animate=False)
        await pilot.pause()
        app.screen._refresh_layout(scroll=True)
        await pilot.pause()
        report(view, "after the scroll frame")

        print("== 3. sidebar CLOSED with the scroll pass still pending")
        print(f"   layout_widgets={dict(app.screen._layout_widgets)} (empty: the Layout message is queued)")
        app.action_toggle_sidebar()
        app.screen._refresh_layout(scroll=True)
        print("   the visible-only pass above resized only newly exposed widgets")
        app.screen._compositor.full_map
        print("   lazy full arrangement read: `_full_map` now holds the NEW lane")
        await pilot.pause()
        await pilot.pause()
        torn = report(view, "after the corrective full reflow")

        if torn:
            target = card = cards[20]
            print(f"== 4. hover {target.tool_call_id} (the operator's manual cure)")
            card._hovered = True
            card._refresh_row()
            await pilot.pause()
            report(view, "after hovering one row")

        print(f"\nRESULT: {torn} torn rows" + (" -> FAIL" if torn else " -> PASS"))
        return 1 if torn else 0


sys.exit(asyncio.run(main()))
