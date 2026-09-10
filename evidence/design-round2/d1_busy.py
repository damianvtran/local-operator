"""D1 in a REALISTIC frame: a busy transcript, and narrow widths.

The empty-transcript capture flatters the warning — it lands as the only row on
screen. An operator using /credential mid-session has a full transcript, so the
warning arrives at the BOTTOM of a scrolling log, one row above a composer that
looks ordinary. Capture that, and capture it narrow.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/tmp/d891r2")

from probe_common import (  # noqa: E402
    SECRET,
    assert_instrument,
    editor_of,
    frame_text,
    new_app,
    save_capture,
    type_text,
)

OUT = "/tmp/d891r2/frames"


async def main() -> None:
    for w in (100, 80, 45):
        app = new_app()
        async with app.run_test(size=(w, 30)) as pilot:
            ed = editor_of(app)
            # seed a real-looking session so the notice competes for attention
            for i in range(12):
                app._notice(f"ran a tool call and it returned result {i}", "info")
            await pilot.pause()
            await type_text(pilot, "/credential ")
            await type_text(pilot, SECRET)
            await pilot.pause()
            save_capture(app, f"{OUT}/d1-busy-{w}-masked.svg")

            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            assert_instrument(app, "/credential", f"busy w={w}")
            frame = frame_text(app)
            rows = [ln for ln in frame.splitlines() if ln.strip()]
            warn_idx = next((i for i, r in enumerate(rows) if "PLAIN TEXT" in r), None)
            comp_idx = next((i for i, r in enumerate(rows) if SECRET in r), None)
            print(f"--- width {w} ---")
            print(f"  warning row index : {warn_idx}")
            print(f"  composer row index: {comp_idx}")
            print(f"  rows between      : "
                  f"{None if warn_idx is None or comp_idx is None else comp_idx - warn_idx - 1}")
            print(f"  warning still visible: {warn_idx is not None}")
            if warn_idx is not None:
                print(f"  warning as painted: {rows[warn_idx].strip()!r}")
                from local_operator.tui.widgets.command_picker import cell_len
                print(f"  its cell_len      : {cell_len(rows[warn_idx].strip())}")
                print(f"  ellipsized?       : {chr(0x2026) in rows[warn_idx]}")
            print("  tail of frame:")
            for r in rows[-6:]:
                print(f"    |{r}")
            print()
            save_capture(app, f"{OUT}/d1-busy-{w}-esc.svg")

            # settle across consecutive frames
            f1 = frame_text(app)
            await pilot.pause()
            f2 = frame_text(app)
            print(f"  settle: frame(n)==frame(n+1) ? {f1 == f2}\n")


asyncio.run(main())
