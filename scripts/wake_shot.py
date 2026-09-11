"""Capture a scheduled-wake receipt sitting between real tool rows.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/wake_shot.py OUT.svg [COLSxROWS] [SHAPE] [ICONS]

``SHAPE`` is ``collapsed`` (default), ``expanded`` (the delivered prompt shown
under the headline) or ``catchup`` (a multi-bullet catch-up, the shape whose
bullets wrap and hang), because the collapsed row and the expansion are judged
on different properties — one row that reads as a ledger line, and a body that
stays indented under it.

``ICONS`` is ``nerd`` (default) or ``plain``; same gate and same reasoning as
``peer_message_shot.py``.

The seeded tree deliberately brackets the wake with TOOL rows. The wake card
shares the tool ledger's spine — the same name column, the same left inset on
the summary row, the same air above and below — and a frame of the wake alone
cannot show whether it agrees with the rows it sits between.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()


def _seed_env(mode: str) -> None:
    """Force the icon gate into ``mode`` BEFORE the app is imported.

    Same marker list and same reasoning as ``peer_message_shot.py``: the gate
    reads the environment at row-build time and this capture runs under an
    isolated HOME, so the marker has to be seeded here or every frame shows the
    ASCII fallback rather than the frame the operator sees in cmux.
    """
    for var in (
        "GHOSTTY_RESOURCES_DIR",
        "GHOSTTY_BIN",
        "KITTY_WINDOW_ID",
        "WEZTERM_PANE",
        "WEZTERM_EXECUTABLE",
        "TERM_PROGRAM",
        "LOCAL_OPERATOR_NO_NERD_ICONS",
    ):
        os.environ.pop(var, None)
    if mode == "nerd":
        os.environ["GHOSTTY_BIN"] = "/usr/local/bin/ghostty"
    else:
        os.environ["TERM_PROGRAM"] = "Apple_Terminal"


_seed_env(sys.argv[4] if len(sys.argv) > 4 else "nerd")

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock, WakeBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: A live fire: the envelope the model gets, then the prompt it delivers.
LIVE_TEXT = (
    "(alarm) Scheduled wake w3 (3/8, every 1h) — "
    'cancel with wake({op:"cancel",id:"w3"}) once its goal is met.'
    "\n\ncheck the build"
)
#: A catch-up: several missed schedules as bullets, which is the shape that
#: exercises the expansion's two-cell hanging indent.
CATCHUP_TEXT = (
    "(alarm) The session resumed after being closed; the following scheduled "
    "wake(s) came due while it was down.\n"
    "\n- w1 (due 09:00): missed while the session was down.\n"
    "  Message: check the backup\n"
    "\n- w2 (due 10:00): missed while the session was down.\n"
    "  Message: re-run the nightly audit"
)


def _answer(text: str) -> AssistantBlock:
    """A SETTLED answer — `finalize_text` is what the stream does at message
    end, and an unsettled block keeps its live-turn ink."""
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _tool(app: OperatorApp, call_id: str, name: str, args: dict[str, object], result: str) -> None:
    """Append a FINISHED tool row, so the ledger around the receipt is settled
    ink rather than the live-turn accent."""
    card = ToolCard(call_id, name, args)
    app._append_block(card)
    card.mark_done(result)


def _seed(app: OperatorApp, shape: str) -> list[WakeBlock]:
    app._append_block(UserBlock("did the wake fire while I was away?"))
    app._append_block(_answer("The scheduler delivered one a moment ago. Checking what it left."))
    _tool(app, "t1", "read", {"path": "src/build.py"}, "412 lines")

    live = WakeBlock(LIVE_TEXT)
    app._append_block(live)

    _tool(app, "t2", "bash", {"command": "make -C build test"}, "exit status 0")
    catchup = WakeBlock(CATCHUP_TEXT, catchup=True)
    app._append_block(catchup)

    app._append_block(_answer("Both deliveries are accounted for."))
    if shape == "expanded":
        return [live]
    if shape == "catchup":
        return [catchup]
    return [live, catchup]


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    shape = sys.argv[3] if len(sys.argv) > 3 else "collapsed"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        blocks = _seed(app, shape)
        await pilot.pause()

        if shape in ("expanded", "catchup"):
            for block in blocks:
                block.toggle_expanded()
            await pilot.pause()

        # A second settled frame: a first paint that differs from this one is a
        # reflow the user sees as motion (AGENTS.md, "Animation and multi-frame
        # changes").
        await pilot.pause()
        screen = app.screen
        print(
            f"size={screen.size} virtual={screen.virtual_size} "
            f"vscroll={screen.show_vertical_scrollbar}",
            file=sys.stderr,
        )
        save_capture(app, out)


asyncio.run(main())
