"""Second probe round for design review of PR #891.

Answers four questions the first round raised, each by driving the real app:

1. Does ``CREDENTIAL_PLACEHOLDER`` paint on ANY route? (it is updated by this
   PR, and no arming route painted it)
2. Does ``CREDENTIAL_ARMED_NOTICE`` paint on ANY route?
3. What does Esc leave, and what does the NEXT Enter do with it — including
   what reaches the transcript and the status bar.
4. What do the masked span and the chip look like at NARROW widths.
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
    CREDENTIAL_HELD_NOTICE,
    CREDENTIAL_PLACEHOLDER,
    CREDENTIAL_TYPING_NOTICE,
    OperatorApp,
)
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

SECRET = "hunter2-typed-test"
OUT = Path("/tmp/design891-frames/probe2")


def painted(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def type_text(pilot, text: str) -> None:
    for char in text:
        await pilot.press(char)
    for _ in range(3):
        await pilot.pause()


async def app_ctx(size=(120, 36)):
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    return session, app


async def q1_placeholder() -> None:
    """Every route that could plausibly show the placeholder."""
    print("\n=== Q1/Q2: does the updated PLACEHOLDER (or ARMED notice) EVER paint? ===")
    routes = {
        "space (typed)": ["/credential "],
        "picker row accepted with tab": ["/cred", "\t"],
        "legacy /credential <KEY>": ["/credential MY_KEY\n"],
        "space then Esc": ["/credential ", "\x1b"],
        "space then a char then bs": ["/credential ", "h", "\x08"],
    }
    for label, chunks in routes.items():
        session, app = await app_ctx()
        async with app.run_test(size=(120, 36)) as pilot:
            await _boot(pilot, app)
            editor = app._editor()
            editor.focus()
            for chunk in chunks:
                for ch in chunk:
                    if ch == "\t":
                        await pilot.press("tab")
                    elif ch == "\n":
                        await pilot.press("enter")
                    elif ch == "\x1b":
                        await pilot.press("escape")
                    elif ch == "\x08":
                        await pilot.press("backspace")
                    else:
                        await pilot.press(ch)
                for _ in range(6):
                    await pilot.pause()
            for _ in range(12):
                await pilot.pause()
            frame = painted(app)
            print(
                f"  {label:32s} PLACEHOLDER={CREDENTIAL_PLACEHOLDER in frame} "
                f"ARMED={CREDENTIAL_ARMED_NOTICE in frame} "
                f"TYPING={CREDENTIAL_TYPING_NOTICE in frame} "
                f"HELD={CREDENTIAL_HELD_NOTICE in frame}"
            )


async def q3_esc_consequence() -> None:
    """Esc, then Enter: exactly what reaches the transcript and status bar."""
    print("\n=== Q3: the consequence of 'Esc cancels' ===")
    for label, prefix in (("bare gesture", ""), ("prose around it", "please store ")):
        session, app = await app_ctx()
        async with app.run_test(size=(120, 36)) as pilot:
            await _boot(pilot, app)
            editor = app._editor()
            editor.focus()
            await type_text(pilot, f"{prefix}/credential ")
            await type_text(pilot, SECRET)
            await pilot.press("escape")
            for _ in range(8):
                await pilot.pause()
            frame_after_esc = painted(app)
            warned = any(
                word in frame_after_esc.lower()
                for word in ("cancel", "unmask", "plaintext", "no longer", "visible")
            )
            print(f"  -- {label}")
            print(f"     buffer after Esc      : {editor.text!r}")
            print(f"     secret PAINTED        : {SECRET in frame_after_esc}")
            print(f"     any warning row       : {warned}")
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
            frame = painted(app)
            save_capture(app, OUT / f"esc-enter-{label.replace(' ', '-')}.svg")
            rows = [ln.strip() for ln in frame.splitlines() if SECRET in ln]
            print(f"     secret painted AFTER Enter: {SECRET in frame}")
            for row in rows:
                print(f"       row: {row!r}")
            print(f"     store                 : {session.variables.credential_names()}")


async def q4_narrow() -> None:
    """The masked span and the chip at narrow widths."""
    print("\n=== Q4: masked span and chip at narrow widths ===")
    for width in (45, 50, 60, 80):
        # masked
        session, app = await app_ctx()
        async with app.run_test(size=(width, 24)) as pilot:
            await _boot(pilot, app)
            editor = app._editor()
            editor.focus()
            await type_text(pilot, "/credential ")
            await type_text(pilot, SECRET)
            frame = painted(app)
            save_capture(app, OUT / f"narrow-masked-{width}.svg")
            rows = [
                ln.rstrip()
                for ln in frame.splitlines()
                if "\u2022" in ln or "masked" in ln or "Enter" in ln
            ]
            print(f"  -- {width} cols, MASKED")
            for row in rows:
                print(f"     {row.strip()!r}")
        # chipped
        session, app = await app_ctx()
        async with app.run_test(size=(width, 24)) as pilot:
            await _boot(pilot, app)
            editor = app._editor()
            editor.focus()
            await type_text(pilot, "/credential ")
            await type_text(pilot, SECRET)
            await pilot.press("enter")
            for _ in range(10):
                await pilot.pause()
            await type_text(pilot, " use this for the staging deploy")
            frame = painted(app)
            save_capture(app, OUT / f"narrow-chipped-{width}.svg")
            rows = [ln.rstrip() for ln in frame.splitlines() if "Credential" in ln]
            print(f"  -- {width} cols, CHIPPED   buffer={editor.text!r}")
            for row in rows:
                print(f"     {row.strip()!r}")


async def q5_empty_enter() -> None:
    """Enter immediately after the arming space, with nothing typed."""
    print("\n=== Q5: Enter on an EMPTY masked span ===")
    session, app = await app_ctx()
    async with app.run_test(size=(120, 36)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        await type_text(pilot, "/credential ")
        await pilot.press("enter")
        for _ in range(25):
            await pilot.pause()
        frame = painted(app)
        save_capture(app, OUT / "empty-enter.svg")
        print(f"  buffer : {editor.text!r}")
        rows = [ln.strip() for ln in frame.splitlines() if ln.strip()]
        for row in rows[:14]:
            print(f"    {row!r}")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    await q1_placeholder()
    await q3_esc_consequence()
    await q4_narrow()
    await q5_empty_enter()


if __name__ == "__main__":
    asyncio.run(main())
