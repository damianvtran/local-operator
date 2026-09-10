"""D5: the picker description row, per width, as painted.

Author claims 44 cells and 'survives every width where the row paints', with a
non-monotonic budget explained by the transcript gutter. Verify by painting it,
and check the U5 claim that the row now names the SPACE in the deciding frame.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/tmp/d891r2")

from probe_common import assert_instrument, frame_text, new_app, save_capture, type_text  # noqa: E402

OUT = "/tmp/d891r2/frames"


async def main() -> None:
    from local_operator.tui.widgets.command_picker import cell_len

    print(f"{'width':>5}  {'space named':>11} {'cropped':>7}  painted picker row")
    for w in (45, 56, 60, 66, 68, 69, 80, 100, 120, 140):
        app = new_app()
        async with app.run_test(size=(w, 30)) as pilot:
            await type_text(pilot, "/credential")
            await pilot.pause()
            await pilot.pause()
            assert_instrument(app, "/credential", f"picker w={w}")
            frame = frame_text(app)
            row = ""
            for ln in frame.splitlines():
                if "/credential" in ln and ("secret" in ln or "Type or paste" in ln):
                    row = ln.strip()
                    break
            names_space = "space" in row.lower()
            cropped = "\u2026" in row
            print(f"{w:>5}  {str(names_space):>11} {str(cropped):>7}  {row!r}")
            if w in (60, 80, 100, 120):
                save_capture(app, f"{OUT}/d5-picker-{w}.svg")

    # what the constant actually is
    print()
    import local_operator.tui.app as am

    for name in dir(am):
        if "CREDENTIAL" in name and isinstance(getattr(am, name), str):
            val = getattr(am, name)
            if "Type or paste" in val or "secret" in val:
                print(f"  {name}: cell_len={cell_len(val)}  {val!r}")


asyncio.run(main())
