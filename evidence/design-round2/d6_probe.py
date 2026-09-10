"""D6 close-up: the usage block, WHOLE frame, several widths.

My first pass filtered rows containing 'credential' and so hid the gesture
line — a self-inflicted false absence. Dump the whole notice block instead,
and check whether the lead FOLDS (author's claim) or CROPS at 80.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/tmp/d891r2")

from probe_common import assert_instrument, frame_text, new_app, save_capture, type_text  # noqa: E402

OUT = "/tmp/d891r2/frames"


async def main() -> None:
    from local_operator.variables import CREDENTIAL_USAGE
    from local_operator.tui.widgets.command_picker import cell_len

    print("=== CREDENTIAL_USAGE as a constant ===")
    for i, line in enumerate(CREDENTIAL_USAGE.splitlines()):
        print(f"  {i}: cell_len={cell_len(line):3d}  {line!r}")
    print()

    for w in (60, 70, 80, 90, 100, 120):
        app = new_app()
        async with app.run_test(size=(w, 30)) as pilot:
            await type_text(pilot, "/credential ")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert_instrument(app, "test/model", f"usage w={w}")
            frame = frame_text(app)
            print(f"--- width {w}: whole notice region ---")
            started = False
            for ln in frame.splitlines():
                if "No credentials stored" in ln:
                    started = True
                if started and ln.strip():
                    if "test/model" in ln or "Message Local" in ln:
                        break
                    print(f"  |{ln}")
            lead_intact = "then the secret" in frame
            folded = "then the" in frame and "then the secret" not in frame
            print(f"  lead phrase 'then the secret' present anywhere: {lead_intact}")
            print(f"  ellipsis in block: {chr(0x2026) in frame}")
            print()
            save_capture(app, f"{OUT}/d6-usage-full-{w}.svg")


asyncio.run(main())
