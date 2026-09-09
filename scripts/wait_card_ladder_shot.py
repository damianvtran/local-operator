"""Capture the tool-card state ladder, including the restored-live row.

The row this PR newly reaches — a replayed card the owner is still executing —
has no clock, so the question the frames answer is whether it still says it is
alive, and whether saying so keeps the row STILL when the tool returns.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/wait_card_ladder_shot.py OUT.svg [COLSxROWS] [settled]

``settled`` captures the same ladder with the restored row settled to success,
which is the pair that shows whether the summary beside it moved.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: One call shape across the whole ladder, so a difference between rows is a
#: difference of STATE and never of content.
ARGS = {"job_id": "7a73c97ffc54", "wait_ms": 1800000}
TOOL = "await_job"


def _card(call_id: str) -> ToolCard:
    return ToolCard(call_id, TOOL, dict(ARGS))


async def main() -> None:
    out = sys.argv[1]
    size = (120, 24)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    settled = len(sys.argv) > 3 and sys.argv[3] == "settled"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app.query_one(Editor).cursor_blink = False

        # 1: natively live — the only row with a real start time, so the only
        # one entitled to a clock.
        native = _card("call-native")
        app._append_block(native)
        native._started = time.monotonic() - 34.0

        # 2: the state this PR newly reaches. Built the way the app builds it:
        # replay paints `interrupted`, then `_mark_pending_tool_rows` repaints.
        restored = _card("call-restored")
        app._append_block(restored)
        restored.restore(state="interrupted")
        restored.restore(state="running")

        waiting = _card("call-waiting")
        app._append_block(waiting)
        waiting.mark_waiting()

        success = _card("call-success")
        app._append_block(success)
        success._started = time.monotonic() - 0.4
        success.mark_done("job 7a73c97ffc54 finished")

        error = _card("call-error")
        app._append_block(error)
        error._started = time.monotonic() - 12.0
        error.mark_failed("exit status 1", "exit status 1")

        interrupted = _card("call-interrupted")
        app._append_block(interrupted)
        interrupted._started = time.monotonic() - 5.0
        interrupted.mark_interrupted()

        await pilot.pause()
        if settled:
            # The settle leg of the SAME row: this is the pair that shows
            # whether the summary moved when the outcome arrived.
            restored._started = time.monotonic() - 1800.0
            restored.mark_done("job 7a73c97ffc54 still running after 1800000ms")
            await pilot.pause()
        await pilot.pause()
        save_capture(app, out)
        print(f"wrote {out}")
        for label, card in (
            ("native   ", native),
            ("restored ", restored),
            ("waiting  ", waiting),
            ("success  ", success),
            ("error    ", error),
            ("interrupt", interrupted),
        ):
            runs = [text for text, _style in card._status_runs()]
            print(f"{label} state={card._state:<12} status_runs={runs}")


asyncio.run(main())
