"""Capture the inline ``/credential`` ARMED states after the D1/D2 remediation.

Usage: python scripts/credential_armed_shot.py OUTDIR [120x36]

Re-shoots the four frames the design round asked for by name — ``03-armed``,
``17-armed-tab-accepted``, ``24-tab-then-paste-PLAINTEXT``,
``40-forget-all-destroyed-it`` — plus the new frame of the armed affordance that
answers D2, at both the leading and the mid-line arm, and the two silent
disarms D2 reproduced (a newline and a typed word).

The three keystroke sequences that USED to be destructive are driven exactly as
the designer drove them, so a frame that still showed the old behaviour would
show it here: arm then Enter twice (which wiped a seeded store), arm then Tab
(which accepted the ``--forget-all`` ghost and consumed the arming token), and
arm-Tab-paste (which landed the secret in plaintext). Each capture prints the
state assertions beside the frame it writes, because the frame proves what is
on screen and the numbers prove what is in the store and the buffer.

No provider request, live session or operator config is used: ``CMUX_*`` is
cleared before any application import and ``probe_isolation`` sandboxes HOME,
so a headless run cannot touch the operator's real sessions or workspaces.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot must not rename the operator's real workspace through inherited CMUX IDs.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

from textual import events  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

#: A realistic 64-character token. Never printed, only length-checked and
#: searched for in the rendered frames — a frame containing it is a leak.
SECRET = "ghp_A1b2C3d4E5f6G7h8J9k0L1m2N3p4Q5r6S7t8U9v0WxYz1234567890abcd"

#: Seeded before the destructive sequences so a wipe is VISIBLE in the store
#: readout rather than being a no-op on an empty store.
LIVE_KEY = "LOP_SECRET_FSD7W3NK"


def painted(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def type_text(pilot, text: str) -> None:
    for char in text:
        await pilot.press(char)
    for _ in range(3):
        await pilot.pause()


async def shoot(outdir: Path, size: tuple[int, int], name: str, drive) -> None:
    """Boot a fresh app, run ``drive``, and save the frame plus its assertions."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        session.variables.store_credential(LIVE_KEY, "seeded-live-value", "command")
        notes = await drive(pilot, app, editor, session)
        for _ in range(3):
            await pilot.pause()
        frame = painted(app)
        save_capture(app, outdir / f"{name}.svg")
        leaked = SECRET in frame
        print(f"\n=== {name} ===")
        print(f"  store:        {session.variables.credential_names()}")
        print(f"  armed:        {editor.credential_armed()}")
        print(f"  buffer:       {editor.text!r}")
        print(f"  SECRET on screen: {leaked}   <-- must be False")
        for note in notes or []:
            print(f"  {note}")
        assert not leaked, f"{name} put the secret on screen"


async def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cred-frames")
    dims = sys.argv[2] if len(sys.argv) > 2 else "120x36"
    width, height = (int(part) for part in dims.split("x"))
    size = (width, height)
    outdir.mkdir(parents=True, exist_ok=True)

    async def armed(pilot, app, editor, session):
        """03-armed: the leading arm, which used to preselect --forget-all."""
        await type_text(pilot, "/credential ")
        return [
            f"picker highlighted row: {editor.picker.highlighted_name()!r} (was '--forget-all')"
        ]

    async def armed_midline(pilot, app, editor, session):
        """The D2 answer at the form that had NO ink at all before."""
        await type_text(pilot, "deploy staging with /credential ")
        runs = editor._slash_runs()
        return [f"slash runs (was None mid-line): {runs}"]

    async def tab_accepted(pilot, app, editor, session):
        """17: Tab used to accept the --forget-all ghost and consume the arm."""
        await type_text(pilot, "deploy with /credential ")
        await pilot.press("tab")
        for _ in range(4):
            await pilot.pause()
        return ["'--forget-all' in buffer: " + str("--forget-all" in editor.text)]

    async def tab_then_paste(pilot, app, editor, session):
        """24: the leak — Tab disarmed, so the next paste landed in plaintext."""
        await type_text(pilot, "deploy with /credential ")
        await pilot.press("tab")
        for _ in range(4):
            await pilot.pause()
        app.post_message(events.Paste(SECRET))
        for _ in range(4):
            await pilot.pause()
        return ["SECRET in buffer: " + str(SECRET in editor.text)]

    async def forget_all(pilot, app, editor, session):
        """40: arm + Enter + Enter wiped the store with no confirm and no undo."""
        await type_text(pilot, "/credential ")
        await pilot.press("enter")
        await pilot.press("enter")
        for _ in range(8):
            await pilot.pause()
        return [f"live credential survived: {LIVE_KEY in session.variables.credential_names()}"]

    async def disarm_newline(pilot, app, editor, session):
        """D2: shift+Enter after the arm used to disarm silently."""
        await type_text(pilot, "deploy with /credential ")
        await pilot.press("shift+enter")
        for _ in range(3):
            await pilot.pause()
        return ["armed after a newline (was False): " + str(editor.credential_armed())]

    async def disarm_word(pilot, app, editor, session):
        """D2: one more typed word used to disarm silently."""
        await type_text(pilot, "deploy with /credential ")
        await type_text(pilot, "the prod key ")
        return ["armed after a typed word (was False): " + str(editor.credential_armed())]

    async def captured(pilot, app, editor, session):
        """The happy path end state: chip in place, arm released, secret gone."""
        await type_text(pilot, "deploy with /credential ")
        app.post_message(events.Paste(SECRET))
        for _ in range(4):
            await pilot.pause()
        await type_text(pilot, "this is the staging deploy key")
        return [f"chip: {'[Credential #1' in editor.text}"]

    async def blank_paste(pilot, app, editor, session):
        """D3: a whitespace paste while armed used to be a silent no-op."""
        await type_text(pilot, "deploy with /credential ")
        app.post_message(events.Paste("   \n  "))
        for _ in range(6):
            await pilot.pause()
        return ["still armed: " + str(editor.credential_armed())]

    async def big_edit_then_capture(pilot, app, editor, session):
        """R7/QA Q5: ONE large edit above the token, then a capture.

        The frame that has to show a MARKER CHIP rather than ``ghp_…``. Under
        the old ``_ARM_DRIFT = 64`` window this single ``delete_line`` moved
        the token past the bound, ``_relocate_armed_token`` returned ``None``,
        and the arm was dropped SILENTLY while the token was still in the
        buffer — so this same paste painted the secret here in plaintext.
        """
        await type_text(pilot, "x" * 120)
        await pilot.press("shift+enter")
        await type_text(pilot, "deploy /credential ")
        armed_before = editor.credential_armed()
        editor.move_cursor((0, 120))
        await pilot.pause()
        await pilot.press("ctrl+shift+k")
        for _ in range(3):
            await pilot.pause()
        armed_after = editor.credential_armed()
        editor.move_cursor(editor._end_of_buffer())
        app.post_message(events.Paste(SECRET))
        for _ in range(6):
            await pilot.pause()
        await type_text(pilot, " the staging deploy key")
        return [
            f"armed before the 121-char delete_line: {armed_before}",
            f"armed after it (was False, the leak):  {armed_after}",
            f"chip in buffer (was the raw secret):   {'[Credential #1' in editor.text}",
        ]

    frames = [
        ("03-armed", armed),
        ("03b-armed-midline", armed_midline),
        ("17-armed-tab-accepted", tab_accepted),
        ("24-tab-then-paste-PLAINTEXT", tab_then_paste),
        ("40-forget-all-destroyed-it", forget_all),
        ("50-armed-affordance-newline", disarm_newline),
        ("51-armed-affordance-typed-word", disarm_word),
        ("52-captured", captured),
        ("53-blank-paste-while-armed", blank_paste),
        ("54-big-edit-then-capture", big_edit_then_capture),
    ]
    for name, drive in frames:
        await shoot(outdir, size, name, drive)
    print(f"\nwrote {len(frames)} frames to {outdir}")


if __name__ == "__main__":
    asyncio.run(main())
