"""Capture the `/copy` picker over a populated transcript, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/copy_shot.py OUT.svg [COLSxROWS] [SHAPE]

``STATE`` (optional, after SHAPE) drives the picker before the frame is taken:

    rest   (default)  as opened
    hover             pointer over a row that is not the selected one
    preview           the preview scrolled down, for its position cues
    tree              the tree window clipped at BOTH ends, for `↑ N`/`↓ N more`
    moved             one row past `tree`, to diff a frame against it

``SHAPE`` selects the tree the picker is opened over, because the layout maths
differ by case and each one has its own way of looking wrong:

    mixed  (default)  many messages, code and quotes — the long-tree window
    short             a two-node tree, which must NOT sit in a half-height pane
    code              one code-heavy answer, for the syntax-highlighted preview
    long              one very long answer, for the `… N more lines` overflow
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual import events  # noqa: E402

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.copy_targets import build_copy_targets  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.copy_picker import (  # noqa: E402
    HEADER_ROWS,
    CopyPickerScreen,
)
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CODE_ANSWER = """Here is the migration, and the rollback beside it.

```python
def migrate(rows):
    for row in rows:
        if row.stale:
            row.drop()
    return len(rows)
```

The audit log keeps every row, so this is reversible:

```sql
SELECT * FROM audit WHERE table_name = 'rows' AND op = 'delete';
```

> Worth noting: the backfill is slower but keeps history.
> Nothing else reads that column today.
"""

LONG_ANSWER = "An answer long enough to overflow the preview pane.\n\n" + "\n".join(
    f"Line {index} of a deliberately long explanation that keeps going." for index in range(1, 120)
)


def _answer(text: str) -> AssistantBlock:
    """A SETTLED answer. `finalize_text` is what the stream does when a message
    ends; without it the block stays mutable and the picker correctly ignores
    it, which is how the first capture came back with an empty tree."""
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _seed(app: OperatorApp, shape: str) -> None:
    if shape == "short":
        # Two nodes exactly: one message, no drillable blocks.
        app._append_block(UserBlock("what about the stale rows?"))
        block = _answer("Drop them; nothing reads that column.")
        app._append_block(block)
        app._append_block(_answer("Confirmed against the audit log."))
        return

    if shape == "code":
        app._append_block(UserBlock("show me the migration"))
        app._append_block(_answer(CODE_ANSWER))
        return

    if shape == "long":
        app._append_block(UserBlock("explain it fully"))
        app._append_block(_answer(LONG_ANSWER))
        return

    # mixed: enough messages to overflow the tree window, with blocks to nest
    # and one truncated answer so the `truncated ·` hint is in frame.
    for turn in range(1, 7):
        app._append_block(UserBlock(f"Turn {turn}: what should we do about the stale rows?"))
        if turn % 3 == 0:
            app._append_block(_answer(CODE_ANSWER))
        else:
            app._append_block(
                _answer(
                    f"Answer {turn}: the audit log still has every row, so a backfill is"
                    " possible.\n\n> Nothing else reads that column today."
                )
            )
    cut = _answer("This answer was interrupted while it was still")
    cut.mark_truncated()
    app._append_block(cut)


async def _drive(screen: CopyPickerScreen, state: str, pilot) -> None:
    """Put the picker in ``state`` before the frame is taken.

    Hover is posted as a `MouseMove` rather than driven through a pilot helper
    because the card is ONE `Static`: the coordinate has to be computed against
    the body's region and the screen's own row map, which is the same routing
    the widget itself does.
    """
    if state == "hover":
        # A row that is NOT the selected one, so the frame carries both states
        # at once — the contrast is the thing under review.
        region = screen._body.region
        screen.post_message(
            events.MouseMove(
                widget=screen._body,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=0,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=region.x + 6,
                screen_y=region.y + HEADER_ROWS + 2,
            )
        )
    elif state == "preview":
        for _ in range(8):
            await pilot.press("shift+down")
    elif state in ("tree", "moved"):
        # Far enough in that the window is clipped at BOTH ends, which is the
        # case a down-only cue would misreport.
        for _ in range(12):
            await pilot.press("down")
        if state == "moved":
            await pilot.press("down")
    await pilot.pause()
    await pilot.pause()


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    shape = sys.argv[3] if len(sys.argv) > 3 else "mixed"
    state = sys.argv[4] if len(sys.argv) > 4 else "rest"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        _seed(app, shape)
        await pilot.pause()

        targets = build_copy_targets(app._transcript_view().blocks())
        screen = CopyPickerScreen(targets)
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        await _drive(screen, state, pilot)
        save_capture(app, out)

        # The numbers behind the pixels: a still shows the symptom, these show
        # the cause. Two of them are this app's standing invariants — a
        # scrollable Screen is always a bug here, and a scrollbar silently
        # costs two cells of width and reflows the transcript behind.
        print(
            f"{Path(out).name}: size={tuple(app.screen.size)} "
            f"virtual={tuple(app.screen.virtual_size)} "
            f"vscroll={app.screen.show_vertical_scrollbar} "
            f"drawable={screen.is_drawable} split={screen._split_rows()} "
            f"card_w={screen._card_width()} min_flat={screen._min_flat_width()} "
            f"gutter={screen._gutter_drawn(screen._card_width())} "
            f"sel={screen._selected} preview_off={screen._preview_offset}"
        )


asyncio.run(main())
