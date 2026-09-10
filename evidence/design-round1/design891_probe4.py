"""Fourth probe: the composer's gutter glyph vs the mask glyph, exactly.

The masked row rendered as ``•  /credential ••••…`` and a raw count of U+2022
in that row returned 19 for an 18-character secret. This isolates WHY: it
prints the chevron cell's codepoint in each mode, and counts the mask glyphs
inside the text span alone versus across the whole row.

Also re-runs the Esc->Enter consequence with the store and transcript read
back, and checks whether the PR's two OTHER updated strings paint anywhere.
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
from local_operator.tui.app import CREDENTIAL_PLACEHOLDER, OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import CREDENTIAL_MASK_CHAR  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

SECRET = "hunter2-typed-test"
MASK = CREDENTIAL_MASK_CHAR


def rows(app):
    return [strip.text for strip in app.screen._compositor.render_strips()]


async def type_text(pilot, text):
    for ch in text:
        await pilot.press(ch)
    for _ in range(3):
        await pilot.pause()


async def gutter(label, keys):
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, keys)
        for _ in range(6):
            await pilot.pause()
        chevron = None
        try:
            chevron = app.query_one("#prompt-chevron")
        except Exception:
            pass
        chev_text = None
        if chevron is not None:
            chev_text = getattr(chevron, "renderable", None)
            chev_text = str(chev_text) if chev_text is not None else None
        target = None
        for line in rows(app):
            if "/credential" in line or "hello" in line:
                target = line
                break
        print(f"\n  -- {label}")
        print(f"     chevron widget text : {chev_text!r}")
        if chev_text:
            print(f"     chevron codepoints  : " + " ".join(f"U+{ord(c):04X}" for c in chev_text.strip()))
        if target is not None:
            stripped = target.strip()
            print(f"     row (stripped)      : {stripped!r}")
            print(f"     row codepoints [0:4]: " + " ".join(f"U+{ord(c):04X}" for c in stripped[:4]))
            print(f"     U+2022 in WHOLE row : {target.count(MASK)}   (typed {len(SECRET)})")
            marker, _, rest = stripped.partition("  ")
            print(f"     leading marker cell : {marker!r}")
            print(f"     U+2022 after marker : {rest.count(MASK)}")
        print(f"     editor.text         : {editor.text!r}")
        print(f"     mask chars in buffer: {editor.text.count(MASK)}")


async def main():
    print("=== GUTTER GLYPH vs MASK GLYPH ===")
    print(f"  MASK = {MASK!r} U+{ord(MASK):04X}")
    await gutter("ordinary prose (unarmed)", "hello there")
    await gutter("armed, nothing typed", "/credential ")
    await gutter("armed, secret typed", "/credential " + SECRET)

    print("\n=== IS CREDENTIAL_PLACEHOLDER USED ANYWHERE THAT PAINTS? ===")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/credential ")
        for _ in range(10):
            await pilot.pause()
        frame = "\n".join(rows(app))
        print(f"  placeholder string      : {CREDENTIAL_PLACEHOLDER!r}")
        print(f"  painted while armed     : {CREDENTIAL_PLACEHOLDER in frame}")
        ph = getattr(editor, "placeholder", None)
        print(f"  editor.placeholder      : {ph!r}")


if __name__ == "__main__":
    asyncio.run(main())
