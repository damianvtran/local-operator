"""Width sweep + Esc-path probe for design round 1 on PR #891.

Prints, for each terminal width, the EXACT painted row for the picker
description, the typing notice and the armed notice — so a crop is read off
the frame rather than inferred from a string length.
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
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

SECRET = "hunter2-typed-test"
WIDTHS = (140, 120, 100, 90, 80, 70, 60, 50, 45, 40)


def painted(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def type_text(pilot, text: str) -> None:
    for char in text:
        await pilot.press(char)
    for _ in range(3):
        await pilot.pause()


async def one_width(width: int, outdir: Path) -> None:
    # -- the picker row -------------------------------------------------------
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(width, 30)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/cred")
        for _ in range(8):
            await pilot.pause()
        rows = [ln.rstrip() for ln in painted(app).splitlines() if "/credential" in ln]
        picker_row = rows[0].strip() if rows else "<NO ROW>"
        save_capture(app, outdir / f"picker-{width}.svg")

    # -- the masked state + its notice ---------------------------------------
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(width, 30)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        frame = painted(app)
        notice_rows = [
            ln.rstrip().strip()
            for ln in frame.splitlines()
            if "masked" in ln or "armed" in ln or "chip" in ln
        ]
        save_capture(app, outdir / f"masked-{width}.svg")
        mask_cells = frame.count("\u2022")
        full_typing = CREDENTIAL_TYPING_NOTICE in frame

    print(f"\n--- {width} columns ---")
    print(f"  picker row        : {picker_row!r}")
    print(f"  TYPING notice full: {full_typing}")
    print(f"  notice rows       : {notice_rows}")
    print(f"  mask cells painted: {mask_cells} (typed {len(SECRET)})")


async def esc_path(outdir: Path) -> None:
    """What Esc ACTUALLY leaves on screen, and what the next Enter does."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "please store /credential ")
        await type_text(pilot, SECRET)
        save_capture(app, outdir / "esc-0-masked.svg")
        print("\n=== ESC PATH ===")
        print(f"  masked        : {editor.text!r}")
        await pilot.press("escape")
        for _ in range(6):
            await pilot.pause()
        save_capture(app, outdir / "esc-1-unredacted.svg")
        frame = painted(app)
        print(f"  after Esc     : {editor.text!r}")
        print(f"  SECRET painted: {SECRET in frame}")
        rows = [ln.strip() for ln in frame.splitlines() if SECRET in ln]
        print(f"  plaintext rows: {rows}")
        guidance = [ln.strip() for ln in frame.splitlines() if "masked" in ln or "armed" in ln]
        print(f"  guidance now  : {guidance}")
        await pilot.press("enter")
        for _ in range(25):
            await pilot.pause()
        save_capture(app, outdir / "esc-2-after-enter.svg")
        frame = painted(app)
        print(f"  after Enter   : {editor.text!r}")
        print(f"  SECRET painted after Enter: {SECRET in frame}")
        print(f"  store         : {session.variables.credential_names()}")
        tail = [ln.strip() for ln in frame.splitlines() if ln.strip()][-8:]
        for row in tail:
            print(f"    tail: {row!r}")


async def backspace_to_zero(outdir: Path) -> None:
    """Backspace to EXACTLY zero typed characters: what does the row say?"""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        for _ in range(len(SECRET)):
            await pilot.press("backspace")
        for _ in range(6):
            await pilot.pause()
        frame = painted(app)
        save_capture(app, outdir / "backspace-to-zero.svg")
        print("\n=== BACKSPACE TO ZERO TYPED CHARS ===")
        print(f"  buffer        : {editor.text!r}")
        print(f"  armed         : {editor.credential_armed()}")
        print(f"  typing        : {editor.credential_typing()}")
        print(f"  TYPING notice : {CREDENTIAL_TYPING_NOTICE in frame}")
        print(f"  ARMED notice  : {CREDENTIAL_ARMED_NOTICE in frame}")
        print(f"  PLACEHOLDER   : {CREDENTIAL_PLACEHOLDER in frame}")
        rows = [ln.strip() for ln in frame.splitlines() if "masked" in ln or "armed" in ln]
        print(f"  guidance rows : {rows}")


async def is_armed_notice_reachable(outdir: Path) -> None:
    """Can the ARMED notice paint AT ALL on this head? Try every route."""
    print("\n=== IS CREDENTIAL_ARMED_NOTICE REACHABLE? ===")
    routes = {
        "/credential + space": ["/credential "],
        "/cred + space": ["/cred "],
        "mid-line + space": ["deploy with /credential "],
    }
    for label, keys in routes.items():
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(120, 36)) as pilot:
            await _boot(pilot, app)
            editor = app._editor()
            editor.focus()
            for chunk in keys:
                await type_text(pilot, chunk)
            for _ in range(8):
                await pilot.pause()
            frame = painted(app)
            print(
                f"  {label:24s} armed={editor.credential_armed()} "
                f"typing={editor.credential_typing()} "
                f"ARMED_painted={CREDENTIAL_ARMED_NOTICE in frame} "
                f"TYPING_painted={CREDENTIAL_TYPING_NOTICE in frame} "
                f"PLACEHOLDER_painted={CREDENTIAL_PLACEHOLDER in frame}"
            )


async def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/design891-frames/widths")
    outdir.mkdir(parents=True, exist_ok=True)
    print("PICKER + NOTICE WIDTH SWEEP")
    for width in WIDTHS:
        await one_width(width, outdir)
    await esc_path(outdir)
    await backspace_to_zero(outdir)
    await is_armed_notice_reachable(outdir)


if __name__ == "__main__":
    asyncio.run(main())
