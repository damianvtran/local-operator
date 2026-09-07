"""Capture an inbound peer-message receipt in a realistic transcript.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/peer_message_shot.py OUT.svg [COLSxROWS] [SHAPE] [ICONS]

``SHAPE`` selects what the peer receipt is doing, because the two states are
what the card has to be judged on together — a collapsed row that reads well
but hides the message, or an expansion that buries the ledger, are each a
failure the other frame does not show:

    collapsed  (default)  every receipt closed: the ONE-ROW guarantee
    expanded              the long receipt opened: sender detail + full body

``ICONS`` is ``nerd`` (default) or ``plain``. The gate reads the environment at
row-build time and this capture runs under an isolated HOME, so a marker has to
be seeded here or every frame silently shows the ASCII fallback — which is not
the frame the operator sees in cmux (it embeds ghostty).

The seeded tree deliberately puts TOOL rows on both sides of the peer receipt.
The card claims to share the ledger spine — same name column, same air above
and below — and a frame with nothing to align against cannot show whether it
does.
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

    Same technique (and the same marker list) as ``nerd_glyph_shot.py``: the
    gate has no interactive probe, it reads exactly these variables, and the
    first glyph lookup caches nothing but is resolved against whatever the env
    says at that moment.
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
from local_operator.tui.widgets.transcript import (  # noqa: E402
    PeerMessageBlock,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The shape the operator reported: a multi-paragraph release-window
#: announcement from a peer session. Verbatim-length prose, because the whole
#: complaint is how many rows one of these costs at body weight.
LONG_PEER_BODY = """I am claiming the current release window as owner (pid 48213).

Window contents so far: #744 (fix: retainable cost estimate), #751 (fix: \
tombstone dead OAuth grants instead of re-spending them), and #747 once its \
QA round lands. If your PR merged since v0.50.3 and is not on that list, send \
me the PR number, the merge SHA and your Release: line and I will fold it in.

I am arguing patch for the whole window: none of the three clears the \
step-function bar on its own, and three patches merged in the same hour are \
still one patch release. Say so now if you disagree — once the bump PR is \
open the number is decided.

Do not start a second release. If you were about to cut one, close your bump \
PR, delete its branch, and hand me the contents."""

SHORT_PEER_BODY = "gates are green on #751 — merging now"

LONG_SENDER = {
    "pid": 48213,
    "conversation_name": "lo-release-window",
    "model_label": "anthropic/claude-opus-5",
}
SHORT_SENDER = {
    "pid": 1174,
    "conversation_name": "lo-mcp-oauth-tombstone",
    "model_label": "anthropic/claude-opus-5",
}


def _answer(text: str) -> AssistantBlock:
    """A SETTLED answer — `finalize_text` is what the stream does at message
    end, and an unsettled block keeps its live-turn ink."""
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _tool(app: OperatorApp, call_id: str, name: str, args: dict[str, object], result: str) -> None:
    """Append a FINISHED tool row, so the ledger around the receipt is settled
    ink rather than the live-turn accent. ``mark_done`` runs after the mount
    because a card settles in place (the pattern ``theme_preview.py`` uses)."""
    card = ToolCard(call_id, name, args)
    app._append_block(card)
    card.mark_done(result)


def _seed(app: OperatorApp) -> list[PeerMessageBlock]:
    app._append_block(UserBlock("what is left before we can cut the release?"))
    app._append_block(
        _answer("Two PRs are still in review. Checking what has merged since the tag.")
    )
    _tool(
        app,
        "t1",
        "bash",
        {"command": "git log --oneline v0.50.3..origin/main"},
        "2 commits",
    )

    long_block = PeerMessageBlock(LONG_PEER_BODY, LONG_SENDER)
    app._append_block(long_block)

    _tool(
        app,
        "t2",
        "send",
        {"target": "lo-release-window", "message": "ack — folding #751 into your window"},
        "delivered",
    )

    short_block = PeerMessageBlock(SHORT_PEER_BODY, SHORT_SENDER)
    app._append_block(short_block)

    app._append_block(_answer("A peer already owns the window, so I have handed it our PR."))
    return [long_block, short_block]


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
        blocks = _seed(app)
        await pilot.pause()

        if shape == "expanded":
            # `toggle_expanded` exists only once the receipt is a ledger card.
            # Tolerating its absence is what lets the SAME script take the
            # before-frame from a checkout that predates the card.
            toggle = getattr(blocks[0], "toggle_expanded", None)
            if callable(toggle):
                toggle()
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
