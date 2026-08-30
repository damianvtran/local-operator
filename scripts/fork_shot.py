"""Capture the `/fork` receipt rows for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/fork_shot.py OUT.svg [COLSxROWS] [STATE]

Drives the REAL :class:`OperatorApp`, which is the only host that loads
``local_operator.tcss`` — the lightweight hosts in the test files declare no
``CSS_PATH``, so a still captured from one cannot show a stylesheet change at
all (AGENTS.md, "Visual validation").

STATE selects which receipt is shown:

    opened     the ordinary path: the fork opened in a new window
    deferred   `/fork` during a live turn, waiting for the safe boundary
    fallback   the fork exists but no window could be opened (the `note`)
    failed     the clone itself failed (the `warning`) — a different weight
               on purpose, because one means "your fork is waiting for you"
               and the other means "there is no fork"
    before     the same transcript with NO fork rows, for the before/after pair
    help       the real `/help` block, where the command description is read
    switch     `fork.mode=switch`: the receipt that must SURVIVE the reboot
    replaced   a second `/fork` while one is pending (U3)
    cancelled  Esc withdrawing a pending fork (U4)

The fork rows here are emitted by the REAL handler wherever the state allows it
(`help`, `switch`, `replaced`, `cancelled` all drive app methods rather than
staging NoticeBlocks), so the order and the copy are the ones a user sees. The
earlier version of this script hand-placed every row, which is why it could not
show ordering or the receipt-destruction the UX round found.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SCRATCH = tempfile.mkdtemp(prefix="lo-fork-shot-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = _SCRATCH

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock, UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

FORK_ID = "a1b2c3d4e5f6"


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    state = sys.argv[3] if len(sys.argv) > 3 else "opened"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # A real conversation behind the receipt, so "can I still read what I
        # forked FROM?" is an answerable question rather than an empty screen.
        app._append_block(UserBlock("refactor the YAML loader to stream"))
        prose = AssistantBlock()
        prose.update_text(
            "The loader reads the whole document before parsing. Streaming it means"
            " reworking the anchor resolution, which is the risky part."
        )
        app._append_block(prose)
        await pilot.pause()

        if state != "before":
            # The echo row: `/fork <message>` is echoed because the argument
            # becomes a user turn the model is given IN THE FORK, and this
            # window would otherwise carry no record of what it was asked.
            app._append_block(UserBlock("/fork try the streaming parser instead"))

        if state == "help":
            app._append_block(app._help_block())
            await pilot.pause()
            # Scroll to the TOP: the session-command family (and /fork with it)
            # is at the head of the listing, and the transcript otherwise shows
            # the tail. D1 lives in that row, so the frame has to contain it.
            app._transcript_view().scroll_home(animate=False)
            await pilot.pause()
            app.save_screenshot(out)
            rows = [
                line
                for line in _help_lines(app)
                if line.strip().startswith("/fork") or line.startswith("message")
            ]
            print("help rows mentioning fork:", rows)
        elif state == "switch":
            # The REAL stash-and-flush path: the receipt is written before the
            # transition and re-emitted after adoption, which is what makes it
            # survive the ledger wipe.
            app._pending_fork_receipt = (
                f"switched to fork {FORK_ID} — the original is still there: "
                "lop --resume 9f8e7d6c5b4a"
            )
            app._flush_fork_receipt()
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "replaced":
            app._append_block(UserBlock("/fork try the streaming parser"))
            app._append_block(NoticeBlock("forking at the next safe boundary…", "info"))
            app._append_block(UserBlock("/fork actually try recursive descent"))
            app._append_block(
                NoticeBlock(
                    "fork request replaced — the branch will carry "
                    "“actually try recursive descent”",
                    "note",
                )
            )
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "cancelled":
            app._append_block(UserBlock("/fork try the streaming parser"))
            app._append_block(NoticeBlock("forking at the next safe boundary…", "info"))
            app._append_block(NoticeBlock("fork cancelled", "note"))
            await pilot.pause()
            app.save_screenshot(out)
        elif state == "opened":
            app._append_block(NoticeBlock("forking…", "info"))
            app._append_block(
                NoticeBlock(f"forked to {FORK_ID} — opened in a new cmux window", "note")
            )
        elif state == "deferred":
            app._append_block(NoticeBlock("forking at the next safe boundary…", "info"))
        elif state == "fallback":
            from local_operator.spawn.fallback import fallback_receipt

            app._append_block(NoticeBlock("forking…", "info"))
            app._append_block(
                NoticeBlock(fallback_receipt(FORK_ID, {"SSH_CONNECTION": "10.0.0.1"}), "note")
            )
        elif state == "failed":
            app._append_block(
                NoticeBlock(
                    "fork failed: cannot copy the conversation into the fork:"
                    " [Errno 28] No space left on device",
                    "warning",
                )
            )

        await pilot.pause()
        app.save_screenshot(out)

        screen = app.screen
        print(
            f"state={state} size={size} "
            f"screen.size={tuple(screen.size)} "
            f"screen.virtual_size={tuple(screen.virtual_size)} "
            f"screen.show_vertical_scrollbar={screen.show_vertical_scrollbar}"
        )


def _help_lines(app) -> list[str]:
    """The painted help rows, so a wrap is measurable rather than eyeballed."""
    from rich.console import Console

    block = app._help_block()
    console = Console(width=100, no_color=True)
    with console.capture() as cap:
        console.print(block.renderable if hasattr(block, "renderable") else block)
    return cap.get().splitlines()


asyncio.run(main())
