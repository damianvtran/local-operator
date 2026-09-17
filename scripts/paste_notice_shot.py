"""Capture the frame a paste notice leaves behind, for the ctrl+v fix.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/paste_notice_shot.py OUT.svg VARIANT [COLSxROWS]

Variants, one per frame worth looking at:

* ``read-no-space`` — the notice for a clipboard that was never read because
  there was no room to stage it. The operator's case (2026-09-17): the data
  volume filled, and ``ctrl+v`` on a screenshot killed the session.
* ``read-failed`` — the notice for a clipboard that was never read for any
  other reason.
* ``attached`` — the healthy path, which must NOT change: an image on the
  clipboard still attaches ``[Image #1, WxH]`` and raises no notice. Captured
  by the same script so the before/after pair is not comparing two harnesses.

WHY THIS IS A FRAME AND NOT A LOG. The reported defect is a keystroke that
produced no visible response and then ended the process, so the evidence has to
be what the user is looking at: the notice, in the composer's own card, over a
transcript. The crash itself has no frame to capture — the app exits — which is
why the before-side evidence for the full-volume case is the PTY run and its
traceback, and the after-side is these stills.

WHAT IS STUBBED, and what is not. The clipboard READ is replaced at
``local_operator.tui.widgets.editor.read_clipboard``, the same seam
``tests/unit/tui/test_paste_clipboard.py`` uses, so each variant is reachable
deterministically on any host. Everything downstream is real: the real
``OperatorApp`` with its production stylesheet, the real ``ctrl+v`` key, the
composer's real routing of the result, and the app's real toast. The clipboard
module's own behaviour against a genuinely full volume is separate evidence —
see the mounted-full-volume PTY run recorded on the PR — and is not what this
script renders.

The transcript is seeded first, as in ``mcp_toast_attach_shot.py``: a card
against an empty splash understates the interruption the notice is, and hides
whether it is readable over the prose it lands on.

Every wait POLLS for the condition rather than spending a fixed number of
ticks: a fixed budget races the notice's own raise, and the next person
re-capturing reads a script flake as a product regression.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.clipboard import ClipboardContents, ClipboardImage  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets import editor as editor_module  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_paste_clipboard import _png_bytes  # noqa: E402

#: What the clipboard reports for each variant. The two ``read_failed`` values
#: are the module's own constants verbatim (``no-space`` / ``unavailable``), so
#: the composer's mapping is exercised rather than assumed.
VARIANTS: dict[str, ClipboardContents] = {
    "read-no-space": ClipboardContents(read_failed="no-space"),
    "read-failed": ClipboardContents(read_failed="unavailable"),
    "attached": ClipboardContents(image=ClipboardImage(_png_bytes(320, 180), "image/png")),
}

#: The card each failure variant must be showing, verbatim from the app. Pinned
#: here so the capture cannot silently be of the WRONG notice — the whole point
#: of the pair is which sentence the user gets.
EXPECTED: dict[str, str] = {
    "read-no-space": "Clipboard not read — no temp space left. Free up space.",
    "read-failed": "Clipboard not read. Try ctrl+v again.",
}


async def _until(pilot, predicate) -> bool:  # type: ignore[no-untyped-def]
    """Pause until ``predicate()`` holds, or the budget runs out."""
    for _ in range(200):
        await pilot.pause()
        if predicate():
            return True
    return False


async def main() -> None:
    out = sys.argv[1]
    variant = sys.argv[2]
    if variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {sorted(VARIANTS)}, not {variant!r}")
    size = (100, 30)
    if len(sys.argv) > 3:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))

    contents = VARIANTS[variant]

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        for turn in range(1, 4):
            app._append_block(UserBlock(f"Turn {turn}: paste that screenshot into the prompt."))
            prose = AssistantBlock()
            prose.update_text(
                "Answer: the clipboard read runs on the keystroke, before anything "
                "reaches the provider, so the composer is the only surface that can "
                "report what it found."
            )
            app._append_block(prose)
        await pilot.pause()

        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        toast = app.query_one(Toast)
        # An actionable card riding the boot would DEFER this notice (see
        # `Toast.show`'s `yield_to_actionable`), so the slot is cleared first and
        # the frame is guaranteed to be the notice under test.
        if toast.display:
            toast.dismiss_toast()
            await pilot.pause()

        editor_module.read_clipboard = lambda *a, **k: contents
        await pilot.press("ctrl+v")

        if variant == "attached":
            matched = await _until(pilot, lambda: editor.text.startswith("[Image"))
        else:
            matched = await _until(
                pilot, lambda: toast.display and toast.message == EXPECTED[variant]
            )
        # One more tick so the card has settled before the still is taken.
        await pilot.pause()

        save_capture(app, out)
        print(
            f"terminal={size[0]}x{size[1]} variant={variant} matched={matched} "
            f"composer={editor.text!r} toast_display={toast.display} "
            f"toast_message={toast.message!r}"
        )
        assert matched, "the frame is of the wrong state; the run is not evidence"
        assert app.is_running, "a paste must not be able to end the app"


asyncio.run(main())
