"""Capture the assistant rail against everything it has to be told apart from.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/assistant_rail_shot.py OUT.svg [COLSxROWS] [THEME]

The seeded tree is chosen so one frame answers every question the treatment can
be wrong about, because a frame of an assistant message ALONE cannot show any of
them:

* a ``UserBlock`` prompt directly above — the rail borrows that rule's GEOMETRY
  and deliberately not its colour, so the two bars have to be visible in one
  frame or "they are still told apart" is an assertion rather than an
  observation;
* a settled tool card — the ledger spine is the other vertical ink on the
  screen, and a rail that reads as a third spine is a regression in the
  transcript's structure even when it is correct per-block;
* prose containing a BLOCKQUOTE, a bullet list and a fenced code block — the
  blockquote is the load-bearing one, because Rich paints its bar with the very
  glyph the rail uses, and the frame is where "two bars, and you can tell which
  is which" is checked;
* a multi-paragraph answer, so the blank separator rows show the rail running
  CONTINUOUSLY rather than breaking into one segment per paragraph.

``THEME`` (default ``dark``) selects the palette, because the rail's ``label``
ink moves per theme and the decision that it stays legible and stays distinct
from the prompt's ``signal`` is a claim about every palette, not about one.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    save_capture,
    settle_status_line,
)

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: Multi-paragraph, and every construct that paints its own furniture. The
#: blockquote is here for the glyph collision; the fence is here because a code
#: line beginning with a digit is the case the copy path refuses to strip.
ANSWER = (
    "The ingest path reads each source once and writes a manifest, so a "
    "re-run is cheap.\n"
    "\n"
    "Three things are worth knowing before you change it:\n"
    "\n"
    "- a source that fails three times is quarantined, never dropped\n"
    "- the manifest is what the reconciler reads on the next pass\n"
    "- retries are backed off, so a flapping source cannot spin the loop\n"
    "\n"
    "> The quarantine is deliberate: losing a source silently is worse than\n"
    "> stopping loudly.\n"
    "\n"
    "Re-run a single source with:\n"
    "\n"
    "```sh\n"
    "1 ingest --source billing --force\n"
    "```\n"
    "\n"
    "That is the whole loop."
)

SHORT = "Both deliveries are accounted for, and the manifest is current."


def _answer(text: str) -> AssistantBlock:
    """A SETTLED answer — ``finalize_text`` is what the stream does at message
    end, and an unsettled block keeps its live-turn ink, which would make this
    a capture of the streaming treatment rather than of the rail."""
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _tool(app: OperatorApp, call_id: str, name: str, args: dict[str, object], result: str) -> None:
    """A FINISHED tool row, so the ledger spine beside the rail is settled ink
    rather than the live-turn accent."""
    card = ToolCard(call_id, name, args)
    app._append_block(card)
    card.mark_done(result)


def _seed(app: OperatorApp) -> None:
    app._append_block(UserBlock("how does the ingest path handle a failing source?"))
    _tool(app, "t1", "read", {"path": "src/ingest/manifest.py"}, "412 lines")
    app._append_block(_answer(ANSWER))
    app._append_block(UserBlock("and the wake deliveries?"))
    app._append_block(_answer(SHORT))


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    theme = sys.argv[3] if len(sys.argv) > 3 else None

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if theme is not None:
            # Through the APP's own path, and only once it is running.
            # ``OperatorApp.__init__`` sets the theme from config, so a
            # ``theme_mod.set_theme`` before construction is silently overridden
            # and every frame comes out in the default ramp — which looks like a
            # working capture and is not one. ``_apply_theme`` is what the theme
            # picker calls, so this frame is the one a user would see.
            app._apply_theme(theme)
            await pilot.pause()
        _seed(app)
        await pilot.pause()
        await pilot.pause()

        # A second settled frame: a first paint that differs from this one is a
        # reflow the user sees as motion (AGENTS.md, "Animation and multi-frame
        # changes"). The status band is waited on too, so a capture of this tree
        # differs from another capture of the same tree in the ledger and
        # nothing else.
        await settle_status_line(pilot, app)
        screen = app.screen
        print(
            f"size={screen.size} virtual={screen.virtual_size} "
            f"vscroll={screen.show_vertical_scrollbar}",
            file=sys.stderr,
        )
        save_capture(app, out)


asyncio.run(main())
