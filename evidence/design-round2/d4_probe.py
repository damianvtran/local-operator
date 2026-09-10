"""D4/D3/D5/D6/D7: verify the accepted-with-measurement claims.

D4 is the notable one: the author wrote a docstring claiming ARMED_NOTICE
survives in a pre-space window, measured six routes, found it paints on NONE,
and corrected the docstring. Verify the CORRECTED claim is true — i.e. that
both strings really are unreachable — with a POSITIVE CONTROL proving the
detector can see a notice at all.
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


async def state(app, pilot, label):
    from local_operator.tui.app import (
        CREDENTIAL_ARMED_NOTICE,
        CREDENTIAL_HELD_NOTICE,
        CREDENTIAL_PLACEHOLDER,
        CREDENTIAL_TYPING_NOTICE_RUNGS,
    )

    await pilot.pause()
    await pilot.pause()
    ed = editor_of(app)
    frame = frame_text(app)
    armed_paints = CREDENTIAL_ARMED_NOTICE in frame
    ph_paints = CREDENTIAL_PLACEHOLDER in frame
    held_paints = CREDENTIAL_HELD_NOTICE in frame
    typing_paints = any(r in frame for r in CREDENTIAL_TYPING_NOTICE_RUNGS)
    print(
        f"  {label:<38} armed={ed.credential_armed()!s:<5} "
        f"typing={ed.credential_typing()!s:<5} | "
        f"ARMED_paints={armed_paints!s:<5} PLACEHOLDER_paints={ph_paints!s:<5} "
        f"HELD_paints={held_paints!s:<5} TYPING_paints={typing_paints}"
    )
    return frame


async def main() -> None:
    from local_operator.tui.app import (
        CREDENTIAL_ARMED_NOTICE,
        CREDENTIAL_PLACEHOLDER,
        CREDENTIAL_TYPING_NOTICE_RUNGS,
    )

    print("=== D4: six routes, does ARMED_NOTICE or PLACEHOLDER ever PAINT? ===")
    print(f"  ARMED_NOTICE = {CREDENTIAL_ARMED_NOTICE!r}")
    print(f"  PLACEHOLDER  = {CREDENTIAL_PLACEHOLDER!r}")
    print()

    # POSITIVE CONTROL FIRST: the detector must be able to see a rung paint.
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        f = await state(app, pilot, "POSITIVE CONTROL (typing)")
        assert any(r in f for r in CREDENTIAL_TYPING_NOTICE_RUNGS), "DETECTOR DEAD"
        print("  -> detector CAN see a notice paint. Absences below are real.\n")

    # route 1: /credential with no space
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential")
        await state(app, pilot, "1. /credential (no space)")
        save_capture(app, f"{OUT}/d4-1-no-space.svg")

    # route 2: + space
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential ")
        await state(app, pilot, "2. /credential + space")

    # route 3: + space + '-' (the flag disarm)
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential -")
        await state(app, pilot, "3. + space + '-' (flag disarm)")

    # route 4: caret left out of span
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("home")
        await state(app, pilot, "4. caret left out of span")

    # route 5: chip minted, then re-arm
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("enter")
        await pilot.pause()
        await type_text(pilot, " /credential ")
        await state(app, pilot, "5. chip minted, then re-arm")

    # route 6: backspace back to the token
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential ")
        await type_text(pilot, "abc")
        for _ in range(4):
            await pilot.press("backspace")
        await state(app, pilot, "6. backspace back to the token")

    # empty composer -> is the PLACEHOLDER shown?
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await state(app, pilot, "7. empty composer (placeholder?)")
        ed = editor_of(app)
        print(f"     editor.placeholder attribute = {ed.placeholder!r}")

    # ---------- D3: glyph collision ----------
    print("\n=== D3: marker vs mask glyph ===")
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.pause()
        assert_instrument(app, "\u2022", "d3")
        frame = frame_text(app)
        row = next(ln for ln in frame.splitlines() if "/credential" in ln)
        print(f"  composer row      : {row.strip()!r}")
        print(f"  U+2022 in row     : {row.count(chr(0x2022))}  (typed {len(SECRET)})")
        print(f"  marker cell       : {row.strip()[0]!r}")
        save_capture(app, f"{OUT}/d3-marker-collision.svg")

    # ---------- D7 / D6: Enter on an armed-but-empty span ----------
    print("\n=== D7/D6: Enter on an armed-but-EMPTY span (usage block) ===")
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        ed = editor_of(app)
        await type_text(pilot, "/credential ")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert_instrument(app, "test/model", "d7")
        print(f"  buffer after enter: {ed.text!r}")
        for ln in frame_text(app).splitlines():
            if ln.strip():
                print(f"    | {ln}")
        save_capture(app, f"{OUT}/d7-empty-enter-usage.svg")

    # narrow, since D6 was about wrapping under a prefix
    for w in (80, 100, 120):
        app = new_app()
        async with app.run_test(size=(w, 30)) as pilot:
            await type_text(pilot, "/credential ")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            rows = [ln for ln in frame_text(app).splitlines() if "credential" in ln.lower()]
            print(f"\n  -- usage block at width {w} --")
            for ln in rows:
                print(f"    | {ln}")
            save_capture(app, f"{OUT}/d6-usage-{w}.svg")


asyncio.run(main())
