"""Capture the `/copy` picker over a populated transcript, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/copy_shot.py OUT.svg [COLSxROWS] [SHAPE]

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

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.copy_targets import build_copy_targets  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.copy_picker import CopyPickerScreen  # noqa: E402
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


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    shape = sys.argv[3] if len(sys.argv) > 3 else "mixed"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        _seed(app, shape)
        await pilot.pause()

        targets = build_copy_targets(app._transcript_view().blocks())
        app.push_screen(CopyPickerScreen(targets))
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
