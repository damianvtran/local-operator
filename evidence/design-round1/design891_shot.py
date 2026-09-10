"""Design round 1 capture for PR #891 — typed credential redaction.

Usage: python scripts/design891_shot.py OUTDIR [WIDTHxHEIGHT]

Drives the REAL ``OperatorApp`` (the host that loads ``local_operator.tcss``),
puts it in each state under judgement, and saves a native-cell SVG plus the
geometry sidecar. Nothing here asserts; the frames are the evidence and the
printed readouts are the numbers behind them.

CMUX_* is cleared before any application import and ``probe_isolation``
sandboxes HOME, so a headless run cannot reach the operator's real sessions,
config or cmux workspaces.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

from PIL import Image  # noqa: E402
from textual import events  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
import local_operator.tui.widgets.editor as editor_module  # noqa: E402
from local_operator.clipboard import ClipboardContents, ClipboardImage  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _boot  # noqa: E402

#: Synthetic only. Never a real credential.
SECRET = "hunter2-typed-test"
LONG_SECRET = "hunter2-typed-test-" + "x" * 45


def painted(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


def _png_bytes(width: int = 1000, height: int = 200) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 40)).save(buffer, "PNG")
    return buffer.getvalue()


def stub_clipboard_image() -> None:
    """Make the next empty paste attach an image, as a real screenshot does."""

    def read_clipboard(*_a, **_kw):
        return ClipboardContents(
            image=ClipboardImage(_png_bytes(), "image/png"),
            paths=(),
            text="",
            refused_remote=False,
        )

    editor_module.read_clipboard = read_clipboard


async def type_text(pilot, text: str) -> None:
    for char in text:
        await pilot.press(char)
    for _ in range(3):
        await pilot.pause()


async def shoot(outdir: Path, size, name: str, drive, *, frames: int = 1) -> None:
    """Boot a fresh app, run ``drive``, save ``frames`` CONSECUTIVE captures.

    ``frames`` > 1 saves one capture per ``pilot.pause()`` so a first painted
    frame that differs from the settled one is visible as motion.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        editor.focus()
        notes = await drive(pilot, app, editor, session)
        saved = []
        for index in range(frames):
            suffix = "" if frames == 1 else f"-f{index}"
            save_capture(app, outdir / f"{name}{suffix}.svg")
            saved.append(painted(app))
            await pilot.pause()
        frame = saved[-1]
        leaked = SECRET in frame or LONG_SECRET[:30] in frame
        print(f"\n=== {name} ({size[0]}x{size[1]}) ===")
        print(f"  buffer:            {editor.text!r}")
        print(f"  armed:             {editor.credential_armed()}")
        typing = getattr(editor, "credential_typing", None)
        print(f"  typing:            {typing() if typing else 'n/a'}")
        print(f"  SECRET on screen:  {leaked}   <-- must be False")
        if frames > 1:
            for index in range(1, frames):
                same = saved[index] == saved[index - 1]
                print(f"  frame {index-1} == frame {index}: {same}")
        for note in notes or []:
            print(f"  {note}")


async def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/design891-frames")
    dims = sys.argv[2] if len(sys.argv) > 2 else "120x36"
    width, height = (int(part) for part in dims.split("x"))
    size = (width, height)
    outdir.mkdir(parents=True, exist_ok=True)

    async def masked_typing(pilot, app, editor, session):
        """THE headline state: keystrokes rendering as mask cells."""
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        rows = [line for line in painted(app).splitlines() if "\u2022" in line]
        return [
            f"mask cells on screen: {painted(app).count(chr(0x2022))} (typed {len(SECRET)})",
            f"masked row: {rows[0].strip()!r}" if rows else "NO masked row found",
        ]

    async def masked_typing_frames(pilot, app, editor, session):
        """Same state, captured as consecutive frames to check settling."""
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        return []

    async def masked_one_char(pilot, app, editor, session):
        """The FIRST keystroke after the arming space — a single mask cell."""
        await type_text(pilot, "/credential ")
        await type_text(pilot, "h")
        return [f"mask cells: {painted(app).count(chr(0x2022))}"]

    async def armed_only(pilot, app, editor, session):
        """Armed but nothing typed yet — the notice before any mask exists."""
        await type_text(pilot, "/credential ")
        return [f"picker notice row present: {'armed' in painted(app)}"]

    async def chip_family(pilot, app, editor, session):
        """ALL THREE CHIP FAMILIES IN ONE FRAME, for side-by-side comparison.

        An image chip, a large-text paste chip and a credential chip in the
        same composer at the same time. A second, near-miss idiom is only
        visible when the three sit together.
        """
        stub_clipboard_image()
        app.post_message(events.Paste(""))
        for _ in range(6):
            await pilot.pause()
        app.post_message(events.Paste("lorem ipsum dolor sit amet\n" * 60))
        for _ in range(6):
            await pilot.pause()
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()
        return [f"buffer: {editor.text!r}"]

    async def chipped(pilot, app, editor, session):
        """Enter mints the chip and leaves the caret in the composer."""
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()
        return [f"chip minted: {'[Credential #1' in editor.text}"]

    async def chipped_settle(pilot, app, editor, session):
        """Consecutive frames ACROSS the Enter transition (mask -> chip)."""
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("enter")
        return []

    async def cancelled(pilot, app, editor, session):
        """Esc unredacts — the characters come back as ordinary text."""
        await type_text(pilot, "/credential ")
        await type_text(pilot, SECRET)
        await pilot.press("escape")
        for _ in range(6):
            await pilot.pause()
        return [f"buffer after Esc: {editor.text!r}"]

    async def long_secret(pilot, app, editor, session):
        """A 64-char secret: does the masked span wrap, and how does it read?"""
        await type_text(pilot, "/credential ")
        await type_text(pilot, LONG_SECRET)
        return [f"mask cells: {painted(app).count(chr(0x2022))} (typed {len(LONG_SECRET)})"]

    async def picker(pilot, app, editor, session):
        """The command-picker row carrying the new description."""
        await type_text(pilot, "/cred")
        return [f"rows: {len(editor.picker._choices)}"]

    async def usage_error(pilot, app, editor, session):
        """CREDENTIAL_USAGE as the operator actually sees it, on a parse error."""
        await type_text(pilot, "/credential --bogus")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        return []

    singles = [
        ("10-masked-typing", masked_typing),
        ("11-masked-one-char", masked_one_char),
        ("12-armed-only", armed_only),
        ("13-chip-family-side-by-side", chip_family),
        ("14-chipped", chipped),
        ("15-cancelled-esc", cancelled),
        ("16-long-secret-masked", long_secret),
        ("17-picker-description", picker),
        ("18-usage-error", usage_error),
    ]
    for name, drive in singles:
        await shoot(outdir, size, name, drive)

    multi = [
        ("20-masked-settle", masked_typing_frames),
        ("21-enter-transition-settle", chipped_settle),
    ]
    for name, drive in multi:
        await shoot(outdir, size, name, drive, frames=4)

    print(f"\nwrote frames to {outdir}")


if __name__ == "__main__":
    asyncio.run(main())
