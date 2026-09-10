"""Third probe round for the design review of PR #891.

Nails down four things the frames raised but a stripped string hid:

1. The exact GLYPH the composer paints in its left gutter while masked, and
   whether it collides with the mask character.
2. The UNSTRIPPED rows, so the indent of the notice against the masked span is
   measurable rather than eyeballed.
3. The bare-gesture Esc->Enter path end to end: what the transcript, the store
   and the status bar actually receive.
4. Whether ``Enter`` does what the typing notice promises when ZERO characters
   have been typed.
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
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import CREDENTIAL_MASK_CHAR  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

SECRET = "hunter2-typed-test"
OUT = Path("/tmp/design891-frames/probe3")


def rows(app):
    return [strip.text for strip in app.screen._compositor.render_strips()]


async def type_text(pilot, text):
    for ch in text:
        await pilot.press(ch)
    for _ in range(3):
        await pilot.pause()


async def q1_glyphs():
    print("\n=== Q1: the gutter glyph vs the mask glyph ===")
    print(f"  CREDENTIAL_MASK_CHAR = {CREDENTIAL_MASK_CHAR!r} U+{ord(CREDENTIAL_MASK_CHAR):04X}")
    for label, keys in (("unarmed", "hello"), ("masked", "/credential " + SECRET)):
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(120, 36)) as pilot:
            await _boot(pilot, app)
            editor = app._editor()
            editor.focus()
            await type_text(pilot, keys)
            for _ in range(4):
                await pilot.pause()
            for line in rows(app):
                if "hello" in line or CREDENTIAL_MASK_CHAR in line or "/credential" in line:
                    prefix = line[:6]
                    print(f"  {label:8s} row prefix {prefix!r} -> " + " ".join(
                        f"U+{ord(c):04X}" for c in prefix
                    ))
                    total = line.count(CREDENTIAL_MASK_CHAR)
                    print(f"           mask glyphs in that ROW: {total} (typed {len(SECRET)})")
                    break


async def q2_indent():
    print("\n=== Q2: unstripped rows — the indent of the notice vs the masked span ===")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        for _ in range(4):
            await pilot.pause()
        for line in rows(app):
            if CREDENTIAL_MASK_CHAR in line or "masked as you type" in line:
                lead = len(line) - len(line.lstrip())
                print(f"  indent={lead:2d}  {line.rstrip()!r}")


async def q3_bare_esc_enter():
    print("\n=== Q3: bare gesture, Esc, then Enter — the full consequence ===")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()
        print(f"  after Esc  : {editor.text!r}")
        save_capture(app, OUT / "bare-esc-unredacted.svg")
        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause()
        save_capture(app, OUT / "bare-esc-then-enter.svg")
        print(f"  after Enter: {editor.text!r}")
        print(f"  store      : {session.variables.credential_names()}")
        for line in rows(app):
            if line.strip():
                print(f"    {line.rstrip()!r}")


async def q4_empty_enter_promise():
    print("\n=== Q4: the notice promises 'Enter turns it into a chip' — with 0 chars typed ===")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/credential ")
        for _ in range(6):
            await pilot.pause()
        notice = [ln.strip() for ln in rows(app) if "masked as you type" in ln]
        print(f"  notice shown with 0 chars typed: {notice}")
        print(f"  buffer before Enter            : {editor.text!r}")
        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause()
        save_capture(app, OUT / "empty-enter-result.svg")
        print(f"  buffer after Enter             : {editor.text!r}")
        print(f"  a chip was minted              : {'[Credential' in editor.text}")
        print("  what Enter ACTUALLY produced:")
        for line in rows(app):
            if line.strip():
                print(f"    {line.rstrip()!r}")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    await q1_glyphs()
    await q2_indent()
    await q3_bare_esc_enter()
    await q4_empty_enter_promise()


if __name__ == "__main__":
    asyncio.run(main())
