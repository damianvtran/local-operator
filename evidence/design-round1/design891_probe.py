"""Follow-up probes for design round 1 on PR #891.

Each probe drives the REAL app and prints the exact state it reached, plus the
frame rows that carry the guidance copy. No assertions: the printed values are
the evidence, and the command that produced them is this file.
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
from local_operator.tui.app import (  # noqa: E402
    CREDENTIAL_ARMED_NOTICE,
    CREDENTIAL_PLACEHOLDER,
    CREDENTIAL_TYPING_NOTICE,
    OperatorApp,
)
from local_operator.variables import CREDENTIAL_USAGE  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

SECRET = "hunter2-typed-test"


def painted(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def type_text(pilot, text: str) -> None:
    for char in text:
        await pilot.press(char)
    for _ in range(3):
        await pilot.pause()


async def run(name: str, drive, size=(120, 36)) -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        print(f"\n=== {name} ({size[0]}x{size[1]}) ===")
        await drive(pilot, app, editor, session)


async def main() -> None:
    print("STRINGS UNDER JUDGEMENT (as defined on this head)")
    print(f"  PLACEHOLDER    : {CREDENTIAL_PLACEHOLDER!r} ({len(CREDENTIAL_PLACEHOLDER)} cells)")
    print(f"  ARMED_NOTICE   : {CREDENTIAL_ARMED_NOTICE!r} ({len(CREDENTIAL_ARMED_NOTICE)} cells)")
    print(f"  TYPING_NOTICE  : {CREDENTIAL_TYPING_NOTICE!r} ({len(CREDENTIAL_TYPING_NOTICE)} cells)")
    print("  USAGE:")
    for line in CREDENTIAL_USAGE.splitlines():
        print(f"    |{line}|  ({len(line)} cells)")

    async def esc_then_enter(pilot, app, editor, session):
        """Esc unredacts. What does the NEXT Enter then do with that line?"""
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        print(f"  masked buffer : {editor.text!r}")
        await pilot.press("escape")
        for _ in range(6):
            await pilot.pause()
        print(f"  after Esc     : {editor.text!r}")
        print(f"  SECRET visible: {SECRET in painted(app)}")
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause()
        print(f"  after Enter   : {editor.text!r}")
        frame = painted(app)
        print(f"  SECRET on screen after Enter: {SECRET in frame}")
        for line in frame.splitlines():
            if SECRET in line or "credential" in line.lower():
                print(f"    row: {line.strip()!r}")
        print(f"  store names   : {session.variables.credential_names()}")

    async def notice_rows(pilot, app, editor, session):
        """Which notice string is ACTUALLY PAINTED at each step of the gesture."""
        for step, keys in (
            ("after '/credential' (no space)", "/credential"),
            ("after the arming SPACE", " "),
            ("after 1 typed char", "h"),
            ("after 4 more typed chars", "unte"),
        ):
            await type_text(pilot, keys)
            frame = painted(app)
            print(f"  -- {step}")
            print(f"     ARMED_NOTICE painted : {CREDENTIAL_ARMED_NOTICE in frame}")
            print(f"     TYPING_NOTICE painted: {CREDENTIAL_TYPING_NOTICE in frame}")
            print(f"     PLACEHOLDER painted  : {CREDENTIAL_PLACEHOLDER in frame}")
            hits = [ln.strip() for ln in frame.splitlines() if "masked" in ln or "armed" in ln]
            print(f"     guidance rows        : {hits}")

    async def picker_row(pilot, app, editor, session):
        """The command-picker row, opened the way the operator opens it."""
        await type_text(pilot, "/")
        await type_text(pilot, "cred")
        for _ in range(8):
            await pilot.pause()
        frame = painted(app)
        rows = [ln.rstrip() for ln in frame.splitlines() if "credential" in ln.lower()]
        print(f"  picker open   : {editor.picker.display if editor.picker else None}")
        for row in rows:
            print(f"    row ({len(row.strip())} cells): {row.strip()!r}")

    async def usage_error(pilot, app, editor, session):
        """The usage block as PAINTED after a real parse error."""
        await type_text(pilot, "/credential --nope")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(25):
            await pilot.pause()
        frame = painted(app)
        for line in frame.splitlines():
            if "Usage" in line or "/credential" in line or "masked" in line:
                print(f"    row: {line.rstrip()!r}")

    async def backspace(pilot, app, editor, session):
        """Backspace inside the masked span: does the count follow?"""
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        print(f"  before bs     : {editor.text!r}")
        for _ in range(5):
            await pilot.press("backspace")
        for _ in range(4):
            await pilot.pause()
        print(f"  after 5 bs    : {editor.text!r}")
        for _ in range(20):
            await pilot.press("backspace")
        for _ in range(4):
            await pilot.pause()
        print(f"  after 20 more : {editor.text!r}")
        print(f"  armed         : {editor.credential_armed()}  typing: {editor.credential_typing()}")
        frame = painted(app)
        print(f"  TYPING_NOTICE painted: {CREDENTIAL_TYPING_NOTICE in frame}")

    async def empty_enter(pilot, app, editor, session):
        """Enter with NOTHING typed after the arming space."""
        await type_text(pilot, "/credential ")
        await pilot.press("enter")
        for _ in range(15):
            await pilot.pause()
        print(f"  buffer        : {editor.text!r}")
        frame = painted(app)
        hits = [ln.strip() for ln in frame.splitlines() if ln.strip()][-6:]
        for hit in hits:
            print(f"    tail row: {hit!r}")

    await run("A. Esc unredacts, then Enter", esc_then_enter)
    await run("B. Which notice paints at each step", notice_rows)
    await run("C. Picker row copy", picker_row)
    await run("D. Usage block on a parse error", usage_error)
    await run("E. Backspace inside the mask", backspace)
    await run("F. Enter on an EMPTY masked span", empty_enter)


if __name__ == "__main__":
    asyncio.run(main())
