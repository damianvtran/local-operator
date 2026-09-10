"""D1: does the Esc unredact ANNOUNCE itself, and is the silent path closed?

Round 1 proved HEAD-after-Esc-then-Enter byte-identical to BASE-after-Enter.
This re-runs the exact keystrokes on 349cdd270 and captures the frames.
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
    notice_row_text,
    save_capture,
    type_text,
)

OUT = "/tmp/d891r2/frames"


async def main() -> None:
    # ---------------- typed secret, then Esc ----------------
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        ed = editor_of(app)
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.pause()

        print("== BEFORE Esc ==")
        print(f"   buffer      : {ed.text!r}")
        print(f"   typing      : {ed.credential_typing()}   armed: {ed.credential_armed()}")
        assert_instrument(app, "/credential", "masked state")
        print(f"   mask cells in frame    : {frame_text(app).count(chr(0x2022))}")
        print(f"   secret painted in frame: {SECRET in frame_text(app)}")
        save_capture(app, f"{OUT}/d1-1-masked.svg")

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        frame = frame_text(app)
        assert_instrument(app, "/credential", "after Esc")
        print("== AFTER Esc ==")
        print(f"   buffer      : {ed.text!r}")
        print(f"   typing      : {ed.credential_typing()}   armed: {ed.credential_armed()}")
        print(f"   secret PAINTED IN COMPOSER: {SECRET in frame}")
        # POSITIVE CONTROL for the grep: the secret IS present, so an absent
        # warning cannot be an empty-container false clean.
        warn_rows = [
            ln.strip()
            for ln in frame.splitlines()
            if "PLAIN TEXT" in ln or "plain text" in ln.lower()
        ]
        print(f"   warning rows: {warn_rows}")
        print(f"   any '!' notice row: {[ln.strip() for ln in frame.splitlines() if ln.strip().startswith('!')]}")
        save_capture(app, f"{OUT}/d1-2-esc-warned.svg")

        # settle: consecutive frames after the notice
        f_a = frame_text(app)
        await pilot.pause()
        f_b = frame_text(app)
        print(f"   settle: frame(n) == frame(n+1) ? {f_a == f_b}")
        save_capture(app, f"{OUT}/d1-3-esc-settled.svg")

    # ---------------- EMPTY Esc: must announce nothing ----------------
    app2 = new_app()
    async with app2.run_test(size=(100, 30)) as pilot:
        ed = editor_of(app2)
        await type_text(pilot, "/credential ")
        await pilot.pause()
        print("== EMPTY span, before Esc ==")
        print(f"   buffer {ed.text!r} typing={ed.credential_typing()} armed={ed.credential_armed()}")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        frame = frame_text(app2)
        assert_instrument(app2, "/credential", "empty Esc")
        print("== EMPTY span, after Esc ==")
        print(f"   buffer {ed.text!r} typing={ed.credential_typing()} armed={ed.credential_armed()}")
        print(f"   PLAIN TEXT warning present? {'PLAIN TEXT' in frame}  (must be False)")
        save_capture(app2, f"{OUT}/d1-4-empty-esc.svg")

        # and prose after an empty escape must be ordinary text
        await type_text(pilot, "is broken please fix")
        await pilot.pause()
        print(f"   prose buffer: {ed.text!r}")
        print(f"   typing now  : {ed.credential_typing()}")
        save_capture(app2, f"{OUT}/d1-5-prose-after-empty-esc.svg")

    # ---------------- Esc then ENTER: the re-leak path ----------------
    app3 = new_app()
    async with app3.run_test(size=(100, 30)) as pilot:
        ed = editor_of(app3)
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        frame = frame_text(app3)
        print("== Esc THEN Enter (the round-1 leak completion) ==")
        print(f"   buffer after enter: {ed.text!r}")
        print(f"   secret in frame   : {SECRET in frame}")
        hits = [ln.strip() for ln in frame.splitlines() if SECRET in ln]
        print(f"   rows carrying it  : {hits}")
        save_capture(app3, f"{OUT}/d1-6-esc-then-enter.svg")


asyncio.run(main())
