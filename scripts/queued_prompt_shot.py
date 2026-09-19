"""Capture what a message sent to a LEAVING session does to the composer.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/queued_prompt_shot.py OUT.svg [queued|refused] [COLSxROWS]

The two modes are the two outcomes a draining runtime can give the owner's own
prompt, and the pair is the point of the frame:

* ``refused`` — the fallback (nothing to spool to, or an attachment the inbox
  row cannot carry). The draft comes back into the composer, the row painted for
  it is withdrawn, and the sentence says this session is handing over.
* ``queued`` — the drain spooled the message for the successor
  (``inbox.SPOOL_RECEIPT_PROMPT``). Nothing comes back, the row STANDS because
  the message is real and the successor runs it, and the row says where it went.

Both frames first paint the standing drain notice the app puts up the moment the
``retiring`` frame arrives (``OperatorApp._on_runtime_draining``, the real
entry point), because that is the screen the operator is actually looking at
when they type: a warning that a new message will not start a turn here.

Printed alongside the SVG, for the numbers-behind-the-pixels check
(AGENTS.md "Visual validation" §4): the screen's virtual size against its actual
size, whether a scrollbar appeared, and the composer's geometry.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.session.errors import RuntimeRetiring  # noqa: E402
from local_operator.session.runtime.types import LEAVING_FOR_BUILD  # noqa: E402

try:  # the receipt this change adds, absent in the tree a "before" comes from
    from local_operator.session.runtime.inbox import SPOOL_RECEIPT_PROMPT  # noqa: E402
except ImportError:  # pragma: no cover - only a pre-change tree takes this
    SPOOL_RECEIPT_PROMPT = ""
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

MESSAGE = "now summarise the build staleness fix"


def _session(mode: str) -> FakeSession:
    """A session whose runtime answers exactly as the mode names."""
    session = FakeSession()

    async def prompt(text: str, images: Any = None, **kwargs: Any) -> Any:
        if mode == "queued":
            return SPOOL_RECEIPT_PROMPT
        raise RuntimeRetiring(trigger=RuntimeRetiring.BUILD)

    session.prompt = prompt  # type: ignore[assignment]
    return session


async def _settle(pilot: Any, wanted: Any) -> None:
    """Wait on the ROW, never on a clock: the outcome is painted by a worker."""
    for _ in range(150):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if wanted():
            return


async def main() -> None:
    out = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "queued"
    size = (100, 30)
    if len(sys.argv) > 3:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))

    app = OperatorApp(lambda: _factory(_session(mode)))
    async with app.run_test(size=size) as pilot:
        editor = app.query_one(Editor)
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        editor.cursor_blink = False
        # A little context, so the frame answers "can I still read the
        # conversation while this happens" rather than showing an empty app.
        app._append_block(UserBlock("Turn 1: is the install marker settled?"))
        prose = AssistantBlock()
        prose.update_text("It is: the tree answers the new stamp and no writes are pending.")
        app._append_block(prose)
        app._append_block(UserBlock("Turn 2: then hand this session to the new build please"))
        reply = AssistantBlock()
        reply.update_text(
            "This session is still finishing three subagents; it will hand over when they end."
        )
        app._append_block(reply)
        await pilot.pause()

        # The standing notice, from the app's own entry point for the frame.
        app._on_runtime_draining(LEAVING_FOR_BUILD)
        await pilot.pause()

        editor.focus()
        editor.text = MESSAGE
        await pilot.pause()
        await pilot.press("enter")
        await _settle(pilot, lambda: len(app.query("NoticeBlock")) >= 2)

        # Settled, not mid-paint: a second capture must be identical to this one.
        save_capture(app, out)
        editor.cursor_blink = False
        print(f"mode                 {mode}")
        print(f"composer text        {editor.text!r}")
        print(f"screen size          {app.screen.size}")
        print(f"screen virtual_size  {app.screen.virtual_size}")
        print(f"vertical scrollbar   {app.screen.show_vertical_scrollbar}")
        print(f"composer size        {editor.size} content-box, styles {editor.styles.height}")

        # The app's own frame must finish shutting down inside this context —
        # see the note in tests/e2e/test_runtime_refresh_e2e.py.
        app.exit()


asyncio.run(main())
