"""BEFORE-frames on the base commit (6eacd91e2), for the design round on #891.

Drives the same gestures as the after-capture. The base tree has no typed
capture at all, so these frames show what the operator saw when they typed
their secret: plaintext, and a hint naming only the paste route.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
for _k in tuple(os.environ):
    if _k.startswith("CMUX_"):
        os.environ.pop(_k)
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import asyncio  # noqa: E402
import scripts.probe_isolation  # noqa: E402,F401
from local_operator.tui.app import CREDENTIAL_ARMED_NOTICE, CREDENTIAL_PLACEHOLDER, OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

SECRET = "hunter2-typed-test"
OUT = Path("/tmp/design891-frames/before")

def painted(app):
    return "\n".join(s.text for s in app.screen._compositor.render_strips())

async def type_text(pilot, text):
    for ch in text:
        await pilot.press(ch)
    for _ in range(3):
        await pilot.pause()

async def shoot(name, drive, size=(120, 36)):
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await drive(pilot, app, editor, session)
        for _ in range(4):
            await pilot.pause()
        save_capture(app, OUT / f"{name}.svg")
        frame = painted(app)
        print(f"\n=== BEFORE {name} ({size[0]}x{size[1]}) ===")
        print(f"  buffer            : {editor.text!r}")
        print(f"  SECRET on screen  : {SECRET in frame}   <-- the defect")
        print(f"  ARMED notice      : {CREDENTIAL_ARMED_NOTICE in frame}")
        print(f"  PLACEHOLDER       : {CREDENTIAL_PLACEHOLDER in frame}")
        print(f"  store             : {session.variables.credential_names()}")
        rows = [ln.strip() for ln in frame.splitlines() if SECRET in ln or "armed" in ln]
        for r in rows:
            print(f"    row: {r!r}")

async def main():
    OUT.mkdir(parents=True, exist_ok=True)

    async def typed(pilot, app, editor, session):
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)

    async def typed_then_enter(pilot, app, editor, session):
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("enter")
        for _ in range(30):
            await pilot.pause()

    async def picker(pilot, app, editor, session):
        await type_text(pilot, "/cred")
        for _ in range(8):
            await pilot.pause()
        frame = painted(app)
        rows = [ln.rstrip().strip() for ln in frame.splitlines() if "/credential" in ln]
        for r in rows:
            print(f"    PICKER ROW ({len(r)} cells): {r!r}")

    async def usage(pilot, app, editor, session):
        await type_text(pilot, "/credential --nope")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(25):
            await pilot.pause()
        for ln in painted(app).splitlines():
            if "Usage" in ln or "credential" in ln:
                print(f"    USAGE ROW: {ln.rstrip()!r}")

    await shoot("b10-typed-PLAINTEXT", typed)
    await shoot("b14-typed-then-enter-LEAK", typed_then_enter)
    await shoot("b17-picker-description", picker)
    await shoot("b18-usage-error", usage)
    for w in (45, 80, 100):
        await shoot(f"b17-picker-{w}", picker, size=(w, 30))

asyncio.run(main())
