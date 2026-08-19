"""Capture the Esc stop ladder over a populated transcript.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/stop_ladder_shot.py OUT.svg <state> [COLSxROWS]

``state`` selects the frame:

``running``   a turn in flight with two delegated children — the "before" shot.
``offer``     after the first Esc: the parent turn is stopped and the notice
              offers the wider stop ("esc again to stop them"). This line is
              load-bearing; a stop that silently leaves children running is the
              original bug wearing a hat.
``stopped``   after the second Esc inside the window: the children are stopped
              and the stale offer is REPLACED rather than stacked, because a
              "still running" row left above "stopped them" reads as current.
``nochildren`` first Esc with nothing delegated: the subagent sentence must not
              appear at all.
``expired``   a first Esc whose 4s window has since closed: the row keeps the
              FACT that children are running but drops the escalation promise,
              because an instruction no key will honour must not stand (D4).
``late``      a second Esc that arrives after the window: it cannot escalate,
              so the re-armed offer states the constraint rather than
              repainting an identical row and reading as a dropped key (D1).
``finished``  a second Esc landing after the children finished on their own:
              the confirmation must not flatly deny the offer just read (D2).
``spared``    a stop that leaves a backgrounded `bash` job running, which it
              names because escalating reads as "stop all of it" (D3).
``draft``     a half-typed prompt in the composer (Ctrl+C "before").
``cleared``   after Ctrl+C: the draft is cleared into prompt history and the
              exit ladder is NOT armed, so no exit hint is shown.

Uses the real ``OperatorApp``, which loads ``local_operator.tcss`` — a bare test
host does not, and would capture an unstyled frame that proves nothing about
what the user sees.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.tui.app import DOUBLE_STOP_WINDOW_S, OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_steering_approval import (  # noqa: E402
    SteerableSession,
    _boot,
    _factory,
)


async def main() -> None:
    out = sys.argv[1]
    state = sys.argv[2]
    size = (100, 30)
    if len(sys.argv) > 3:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))

    session = SteerableSession()
    # Two delegated children, except for the state that exists to show none.
    session.running_children = 0 if state == "nochildren" else 2

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).cursor_blink = False

        app._append_block(UserBlock("split the backfill across two agents and summarise"))
        prose = AssistantBlock()
        prose.update_text(
            "Delegating: one agent walks the 2019-2022 partitions, the other "
            "the 2023-2025 partitions. I will reconcile the two reports."
        )
        app._append_block(prose)
        await pilot.pause()

        if state == "spared":
            session.running_bash_jobs = 1

        if state in {"draft", "cleared"}:
            editor = app.query_one(Editor)
            editor.focus()
            editor.text = "a half-typed prompt I do not want to lose"
            await pilot.pause()
            if state == "cleared":
                await pilot.press("ctrl+c")
                await pilot.pause()
        else:
            # A turn in flight is the state the user presses Esc in.
            session.streaming = True
            await pilot.pause()
            if state in {
                "offer",
                "stopped",
                "nochildren",
                "expired",
                "late",
                "finished",
                "spared",
            }:
                await pilot.press("escape")
                await pilot.pause(0.1)
            if state in {"stopped", "spared"}:
                # Inside DOUBLE_STOP_WINDOW_S, so this is the escalation press.
                app._stop_offered_at = time.monotonic()
                await pilot.press("escape")
                await pilot.pause(0.1)
            if state == "finished":
                # The children settle on their own before the second press.
                session.running_children = 0
                app._stop_offered_at = time.monotonic()
                await pilot.press("escape")
                await pilot.pause(0.1)
            if state == "late":
                # Past the window: this press cannot escalate.
                app._stop_offered_at = time.monotonic() - (DOUBLE_STOP_WINDOW_S + 1)
                await pilot.press("escape")
                await pilot.pause(0.1)
            if state == "expired":
                # Let the real timer retire the promise.
                await pilot.pause(DOUBLE_STOP_WINDOW_S + 0.5)

        for _ in range(4):
            await pilot.pause()
        app.save_screenshot(out)


asyncio.run(main())
