"""Prove the frame-reading instrument works BEFORE trusting any absence.

Positive controls: the masked frame must contain mask cells; the composer must
paint its own buffer somewhere. If frame_text() cannot see a string we KNOW is
on screen, every "absent" reading from it is worthless.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/tmp/d891r2")

from probe_common import SECRET, editor_of, frame_text, new_app, type_text  # noqa: E402


async def main() -> None:
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        ed = editor_of(app)
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.pause()

        f = frame_text(app)
        print(f"frame_text length      : {len(f)}")
        print(f"frame non-blank rows   : {len([l for l in f.splitlines() if l.strip()])}")
        print("--- POSITIVE CONTROLS on the MASKED frame ---")
        print(f"  mask char '\u2022' in frame : {'\u2022' in f}   count={f.count('\u2022')}")
        print(f"  '/credential' in frame : {'/credential' in f}")
        print(f"  secret in frame        : {SECRET in f}")
        print("--- raw frame dump ---")
        for i, line in enumerate(f.splitlines()):
            if line.strip():
                print(f"  {i:3d}| {line}")

        # Independent instrument: export_screenshot (what save_capture writes)
        svg = app.export_screenshot()
        print("--- export_screenshot cross-check ---")
        print(f"  svg length             : {len(svg)}")
        print(f"  '/credential' in svg   : {'/credential' in svg}")
        print(f"  secret in svg          : {SECRET in svg}")

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        f2 = frame_text(app)
        svg2 = app.export_screenshot()
        print("--- AFTER Esc, both instruments ---")
        print(f"  buffer                 : {ed.text!r}")
        print(f"  frame_text sees secret : {SECRET in f2}")
        print(f"  svg sees secret        : {SECRET in svg2}")
        print(f"  frame_text sees PLAIN  : {'PLAIN TEXT' in f2}")
        print(f"  svg sees PLAIN TEXT    : {'PLAIN TEXT' in svg2}")
        print("--- raw frame dump AFTER Esc ---")
        for i, line in enumerate(f2.splitlines()):
            if line.strip():
                print(f"  {i:3d}| {line}")


asyncio.run(main())
