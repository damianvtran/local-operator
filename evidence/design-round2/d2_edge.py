"""D2 edge: where does the ladder's own claim break?

The docstring asserts: "The narrowest rung is 22 cells, so the exit survives
every width the picker paints at." Measured: the narrowest rung is 25 cells.
Find the width at which the exit word actually stops surviving, and quote the
budget arithmetic so the cause is visible, not just the symptom.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/tmp/d891r2")

from probe_common import SECRET, assert_instrument, frame_text, new_app, save_capture, type_text  # noqa: E402

OUT = "/tmp/d891r2/frames"


def notice_row(app) -> str:
    for ln in frame_text(app).splitlines():
        s = ln.strip()
        if "Esc" in s or "masked" in s or "Enter chips" in s:
            return s
    return ""


async def main() -> None:
    from local_operator.tui.app import CREDENTIAL_TYPING_NOTICE_RUNGS as RUNGS
    from local_operator.tui.widgets.command_picker import (
        _EDGE_MARGIN,
        _GUTTER_CELLS,
        CommandPicker,
        cell_len,
    )

    print("=== the ladder's own numbers ===")
    print(f"  narrowest rung        : {RUNGS[-1]!r}")
    print(f"  its cell_len          : {cell_len(RUNGS[-1])}   (docstring says 22)")
    print(f"  _GUTTER_CELLS={_GUTTER_CELLS}  _EDGE_MARGIN={_EDGE_MARGIN}")
    need = cell_len(RUNGS[-1]) + _GUTTER_CELLS + _EDGE_MARGIN
    print(f"  screen width needed for the narrowest rung uncropped: {need}")
    print()

    print("=== per width: is the full exit phrase 'Esc cancels' intact? ===")
    print(f"{'w':>4} {'picker w':>8} {'budget':>6}  {'exit intact':>11}  painted")
    first_loss = None
    for w in range(24, 46):
        app = new_app()
        async with app.run_test(size=(w, 30)) as pilot:
            await type_text(pilot, "/credential ")
            await type_text(pilot, SECRET)
            await pilot.pause()
            await pilot.pause()
            assert_instrument(app, "\u2022", f"w={w}")
            picker = app.query_one(CommandPicker)
            pw = picker.size.width
            budget = max(1, pw - _GUTTER_CELLS - _EDGE_MARGIN)
            row = notice_row(app)
            intact = "Esc cancels" in row
            if not intact and first_loss is None:
                first_loss = w
            print(f"{w:>4} {pw:>8} {budget:>6}  {str(intact):>11}  {row!r}")
            if w in (28, 30, 34, 35):
                save_capture(app, f"{OUT}/d2-edge-{w}.svg")
    print()
    print(f"  'Esc cancels' first INCOMPLETE at screen width : {first_loss}")
    print(f"  docstring claim: 'exit survives every width the picker paints at'")


asyncio.run(main())
