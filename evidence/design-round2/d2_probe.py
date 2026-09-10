"""D2: verify the laddered notice PER WIDTH, with my own captures.

Widths the brief names: 45, 56, 60, 66, 68, 69, 80 (plus 120 and the extremes
the ladder claims to cover). Then a mid-capture resize, which is the property
the paint-time resolution exists for.

Read off the COMPOSITED frame, not the constant: the question is what the
operator sees, and the author's numbers are a starting point only.
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
WIDTHS = [30, 35, 40, 45, 50, 56, 60, 66, 68, 69, 72, 80, 100, 120]
CAPTURE_AT = {45, 56, 66, 68, 69, 80}


def notice_row(app) -> str:
    """The painted guidance row: the one carrying the mode/keys copy."""
    for ln in frame_text(app).splitlines():
        s = ln.strip()
        if "Esc" in s or "masked" in s or "Enter chips" in s or "chip" in s:
            return s
    return ""


async def main() -> None:
    print(f"{'width':>5}  {'Esc?':>5} {'Enter?':>6} {'rung#':>5}  painted row")
    from local_operator.tui.app import CREDENTIAL_TYPING_NOTICE_RUNGS as RUNGS

    results = []
    for w in WIDTHS:
        app = new_app()
        async with app.run_test(size=(w, 30)) as pilot:
            ed = editor_of(app)
            await type_text(pilot, "/credential ")
            await type_text(pilot, SECRET)
            await pilot.pause()
            await pilot.pause()
            # positive control per width: the composer row must be painted,
            # else an empty notice reading is a dead instrument not a finding.
            assert_instrument(app, "\u2022", f"width {w}")
            row = notice_row(app)
            has_esc = "Esc" in row
            has_enter = "Enter" in row or "Enter chips" in row
            rung = next((i for i, r in enumerate(RUNGS) if r.strip() in row), None)
            cropped = "\u2026" in row
            print(
                f"{w:>5}  {str(has_esc):>5} {str(has_enter):>6} "
                f"{str(rung):>5}  {row!r}{'  <-- ELLIPSIZED' if cropped else ''}"
            )
            results.append((w, has_esc, has_enter, row, cropped, ed.credential_typing()))
            if w in CAPTURE_AT:
                save_capture(app, f"{OUT}/d2-width-{w}.svg")

    print()
    lost_esc = [w for w, e, _, _, _, _ in results if not e]
    lost_enter = [w for w, _, n, _, _, _ in results if not n]
    ellipsized = [w for w, _, _, _, c, _ in results if c]
    not_typing = [w for w, _, _, _, _, t in results if not t]
    print(f"widths LOSING 'Esc'    : {lost_esc or 'none'}")
    print(f"widths LOSING 'Enter'  : {lost_enter or 'none'}")
    print(f"widths ELLIPSIZED      : {ellipsized or 'none'}")
    print(f"widths NOT in typing   : {not_typing or 'none'}")

    # ---------------- mid-capture resize ----------------
    print("\n== MID-CAPTURE RESIZE (the property paint-time resolution is for) ==")
    app = new_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.pause()
        assert_instrument(app, "\u2022", "resize start 120")
        print(f"   opened at 120 : {notice_row(app)!r}")
        save_capture(app, f"{OUT}/d2-resize-1-at120.svg")

        for target in (60, 45, 100):
            app._size = None  # force a real resize path
            await pilot.resize_terminal(target, 30)
            await pilot.pause()
            await pilot.pause()
            assert_instrument(app, "\u2022", f"resized to {target}")
            row = notice_row(app)
            print(
                f"   resized to {target:>3} : {row!r}   "
                f"Esc={'Esc' in row} ellipsized={chr(0x2026) in row}"
            )
            save_capture(app, f"{OUT}/d2-resize-2-at{target}.svg")

        # settle across consecutive frames at the narrow width
        await pilot.resize_terminal(45, 30)
        await pilot.pause()
        f1 = frame_text(app)
        await pilot.pause()
        f2 = frame_text(app)
        print(f"   settle at 45: frame(n)==frame(n+1) ? {f1 == f2}")


asyncio.run(main())
