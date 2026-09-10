"""D1 depth: the plaintext STATE persists; does the CUE?

Round 1's defect was "the recovery is silent and terminal-looking". The fix is
a one-shot transcript notice. But the composer keeps holding the plaintext
secret indefinitely. So: after the notice, does anything in the frame still say
the composer holds a secret? Test by continuing to work after Esc.
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


def cues(app):
    frame = frame_text(app)
    rows = frame.splitlines()
    return {
        "warning_row_present": any("PLAIN TEXT" in r for r in rows),
        "secret_in_composer": any(SECRET in r for r in rows),
        "amber_marker": any(r.strip().startswith("\u2022") for r in rows),
        "chevron_marker": any("\u276f" in r for r in rows),
        "mask_cells": frame.count("\u2022"),
    }


async def main() -> None:
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        ed = editor_of(app)
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert_instrument(app, "/credential", "just after Esc")
        print("== immediately after Esc ==")
        print(f"   buffer: {ed.text!r}")
        for k, v in cues(app).items():
            print(f"   {k:22s}: {v}")
        save_capture(app, f"{OUT}/d1-8-cue-at-esc.svg")

        # The operator carries on: they came back to fix a typo, so they edit.
        # Every one of these keystrokes is a thing an operator does next.
        await pilot.press("home")
        await pilot.pause()
        print("== after caret move (home) ==")
        for k, v in cues(app).items():
            print(f"   {k:22s}: {v}")

        await type_text(pilot, "please store ")
        await pilot.pause()
        await pilot.pause()
        print("== after typing prose in front of it ==")
        print(f"   buffer: {ed.text!r}")
        for k, v in cues(app).items():
            print(f"   {k:22s}: {v}")
        assert_instrument(app, "test/model", "after prose")
        save_capture(app, f"{OUT}/d1-9-cue-after-prose.svg")

        # And with enough transcript traffic to scroll the notice off.
        for i in range(4):
            app._notice(f"unrelated activity line {i}", "info")
        await pilot.pause()
        await pilot.pause()
        print("== after 4 unrelated notices ==")
        for k, v in cues(app).items():
            print(f"   {k:22s}: {v}")
        print("   visible rows:")
        for ln in frame_text(app).splitlines():
            if ln.strip():
                print(f"     | {ln}")
        save_capture(app, f"{OUT}/d1-10-cue-scrolled.svg")


asyncio.run(main())
