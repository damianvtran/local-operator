"""D1 follow-up: check the Esc->Enter leak in BOTH forms.

The base leak reaches the transcript normalized: '-'->'_' and upper-cased.
A literal-string grep calls that clean. Round 1's finding was proved with the
key-name form, so round 2 must check it the same way.
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
NORMALIZED = SECRET.replace("-", "_").upper()  # HUNTER2_TYPED_TEST


async def run(label, keys_after_esc, path):
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        ed = editor_of(app)
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("escape")
        await pilot.pause()
        warned = "PLAIN TEXT" in frame_text(app)
        for k in keys_after_esc:
            await pilot.press(k)
        await pilot.pause()
        await pilot.pause()
        assert_instrument(app, "test/model", f"{label} after keys")
        frame = frame_text(app)
        print(f"== {label} ==")
        print(f"   warned at Esc      : {warned}")
        print(f"   buffer             : {ed.text!r}")
        print(f"   LITERAL   {SECRET!r:<24} in frame: {SECRET in frame}")
        print(f"   NORMALIZED {NORMALIZED!r:<23} in frame: {NORMALIZED in frame}")
        hits = [ln.strip() for ln in frame.splitlines() if SECRET in ln or NORMALIZED in ln]
        print(f"   rows carrying either: {hits}")
        # store readback: was anything minted?
        try:
            names = app.session_credential_names()
        except Exception as exc:  # pragma: no cover
            names = f"<{exc}>"
        print(f"   session store names : {names}")
        save_capture(app, path)
        return frame


async def main() -> None:
    # positive control: the normalizer form DOES appear when the key-name
    # route is genuinely taken, so an absence below is not a dead grep.
    app = new_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("escape")
        await pilot.pause()
        f = frame_text(app)
        print("== POSITIVE CONTROL: literal secret is on screen after Esc ==")
        print(f"   literal in frame   : {SECRET in f}   (must be True)")
        print(f"   normalized in frame: {NORMALIZED in f}")

    await run("Esc then ENTER", ["enter"], f"{OUT}/d1-6-esc-then-enter.svg")
    await run("Esc then Enter twice", ["enter", "enter"], f"{OUT}/d1-7-esc-enter-twice.svg")


asyncio.run(main())
