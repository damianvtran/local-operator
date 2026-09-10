"""Sixth probe: the exact width at which each half of the typing notice crops.

The source comment beside ``CREDENTIAL_TYPING_NOTICE`` asserts:

    "Same width budget as the held notice (~82 cells at 120 columns, and the
     row ellipsizes its TAIL): the tail here is 'Esc cancels', the way out,
     which is exactly the half that must not be the part that crops."

This MEASURES that claim rather than taking it: for every width it prints the
row exactly as painted, and reports the first width at which each half is lost.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.tui.app import CREDENTIAL_TYPING_NOTICE, OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

SECRET = "hunter2-typed-test"
OUT = Path("/tmp/design891-frames/probe6")


def rows(app):
    return [strip.text for strip in app.screen._compositor.render_strips()]


async def type_text(pilot, text):
    for ch in text:
        await pilot.press(ch)
    for _ in range(3):
        await pilot.pause()


async def at_width(width, save):
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(width, 22)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        for _ in range(5):
            await pilot.pause()
        if save:
            save_capture(app, OUT / f"typing-notice-{width}.svg")
        for line in rows(app):
            if "masked as you type" in line:
                return width, line.strip()
    return width, "<NOT PAINTED>"


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("SOURCE CLAIM under test: 'the tail here is Esc cancels, the way out,")
    print("  which is exactly the half that must not be the part that crops.'")
    print(f"\nFULL STRING ({len(CREDENTIAL_TYPING_NOTICE)} cells): {CREDENTIAL_TYPING_NOTICE!r}\n")
    print(f"{'width':>5}  {'Esc?':>5} {'chip?':>6} {'Enter?':>7}  painted row")
    saved = {45, 55, 60, 66, 70, 80}
    first_esc_lost = None
    first_chip_lost = None
    for width in range(90, 39, -2):
        w, row = await at_width(width, width in saved)
        has_esc = "Esc cancels" in row
        has_chip = "chip" in row
        has_enter = "Enter" in row
        if not has_esc and first_esc_lost is None:
            first_esc_lost = width
        if not has_chip and first_chip_lost is None:
            first_chip_lost = width
        print(f"{w:>5}  {has_esc!s:>5} {has_chip!s:>6} {has_enter!s:>7}  {row!r}")
    print(f"\n  'Esc cancels' first LOST at width : {first_esc_lost}")
    print(f"  'chip' first LOST at width        : {first_chip_lost}")


if __name__ == "__main__":
    asyncio.run(main())
