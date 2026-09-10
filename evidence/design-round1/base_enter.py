"""BASE-tree behaviour of the exact keystrokes HEAD's Esc exit lands on."""
from __future__ import annotations
import os, sys
from pathlib import Path
for _k in tuple(os.environ):
    if _k.startswith("CMUX_"):
        os.environ.pop(_k)
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import asyncio  # noqa: E402
import scripts.probe_isolation  # noqa: E402,F401
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402
SECRET = "hunter2-typed-test"
def rows(app):
    return [s.text for s in app.screen._compositor.render_strips()]
async def go(prefix, label):
    session = FakeSession(); app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app); ed = app._editor(); ed.focus()
        for ch in f"{prefix}/credential {SECRET}":
            await pilot.press(ch)
        for _ in range(6): await pilot.pause()
        print(f"\n== BASE {label} ==")
        print(f"   before Enter: {ed.text!r}")
        await pilot.press("enter")
        for _ in range(30): await pilot.pause()
        frame = "\n".join(rows(app))
        print(f"   after Enter : {ed.text!r}")
        print(f"   secret painted        : {SECRET in frame}")
        print(f"   secret AS A KEY NAME  : {SECRET.upper().replace('-','_') in frame}")
        print(f"   store: {session.variables.credential_names()}")
        for ln in rows(app):
            if ln.strip() and (SECRET in ln or "Paste the value" in ln or "▌" in ln):
                print(f"     row: {ln.strip()!r}")
async def main():
    await go("", "bare")
    await go("please store ", "prose")
asyncio.run(main())
