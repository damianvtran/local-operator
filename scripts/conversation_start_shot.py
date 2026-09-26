"""Capture a New-chat's first send and what a FAILED send leaves behind.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/conversation_start_shot.py OUT_DIR [STATE] [COLSxROWS]

States, the S4 evidence set:

- ``boot``   — the empty welcome splash (control; unchanged by this change).
- ``press``  — Enter pressed with the send gated: the user row is up, the
               composer is empty, and no turn has started yet. Behaviour 2's
               frame ("paint at send; the composer is not held").
- ``turn``   — the same send after ``TurnStarted``: the working line is up
               under the row (behaviours 1 and 4).
- ``failed`` — a dead runtime answers the send: the row STAYS and the failure
               notice carries ``send again enter · edit e``; the composer is empty.
               This is the behaviour-3 change. Run the SAME script from the
               base commit for the before frame — there the row is withdrawn,
               the text is back in the composer, and the notice says so; with
               the two stills side by side the inversion is the report.
- ``overflow`` — a tall transcript with the viewport at the tail: the numbers
               behind J3 (scroll offset, ``max_scroll_y``, follow state) when
               the content actually overflows.

Every state also prints the geometry behind the pixels (AGENTS.md "Visual
validation" §4): the screen's size against its virtual size, the transcript's
scroll offset against its max and its follow state, the composer's box, and the
first user row's top edge relative to the transcript's content region — the
behaviour-1 property ("top inset + 1, not just above the composer").

This script is deliberately runnable against BOTH the base and the changed
tree: the wait after a failed send is a pump, not a read of the new record, so
the same command captures the before and the after frames.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

STATES = ("boot", "press", "turn", "failed", "overflow")


class GatedSession(FakeSession):
    """A session whose ``prompt`` parks, so the in-flight frame holds still.

    The pending states are the point of the capture: a prompt that returned
    instantly could not show the row-without-a-turn window (behaviours 1/2/4).
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def prompt(self, text, images=None, **kwargs):  # noqa: ANN001, ANN201
        await self.gate.wait()


class DeadSession(FakeSession):
    """The shape the round-3 report recorded: a record claims a live owner."""

    async def prompt(self, text, images=None, **kwargs):  # noqa: ANN001, ANN201
        raise ConnectionError("owner socket unreachable: [Errno 61] Connect call failed")


def _metrics(app: OperatorApp, state: str, size: tuple[int, int]) -> dict[str, Any]:
    """The numbers behind the pixels, read off the live widgets and compositor."""
    view = app.query_one(TranscriptView)
    editor = app.query_one(Editor)
    blocks = view.blocks()
    rows = [block for block in blocks if isinstance(block, UserBlock)]
    notices = [block for block in blocks if isinstance(block, NoticeBlock)]
    region = view.scrollable_content_region
    first = rows[0] if rows else None
    return {
        "state": state,
        "grid": list(size),
        "screen_size": list(app.screen.size),
        "screen_virtual_size": list(app.screen.virtual_size),
        "transcript": {
            "scroll_offset": view.scroll_offset.y,
            "max_scroll_y": view.max_scroll_y,
            "is_following_tail": view.is_following_tail,
            "size": [view.size.width, view.size.height],
            "virtual_size": [view.virtual_size.width, view.virtual_size.height],
            "content_region": [region.x, region.y, region.width, region.height],
        },
        "composer": {"text": editor.text, "size": [editor.size.width, editor.size.height]},
        "first_row": {
            "top": None if first is None else first.region.y,
            "top_in_content": None if first is None else first.region.y - region.y,
        },
        "user_rows": [block.text() for block in rows],
        "notices": [block._text for block in notices],
    }


async def _settle(pilot, *, turns: int = 150) -> None:  # noqa: ANN001
    """Pump roughly 1.5 s: enough for a refusal worker on any tree."""
    for _ in range(turns):
        await pilot.pause()
        await asyncio.sleep(0.01)


async def main() -> None:
    if len(sys.argv) < 2:
        print((__doc__ or "").splitlines()[0], file=sys.stderr)
        raise SystemExit(2)
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = sys.argv[2] if len(sys.argv) > 2 else "failed"
    grid = sys.argv[3] if len(sys.argv) > 3 else "100x30"
    if state not in STATES:
        print(f"unknown state {state!r}; expected one of {', '.join(STATES)}", file=sys.stderr)
        raise SystemExit(2)
    try:
        cols, rows = (int(part) for part in grid.split("x"))
    except ValueError:
        print(f"invalid grid {grid!r}; expected e.g. 100x30", file=sys.stderr)
        raise SystemExit(2) from None
    size = (cols, rows)

    session: FakeSession = DeadSession() if state == "failed" else GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        if state == "overflow":
            # The real append path, many rows: what the tail does once the
            # content overflows the viewport is the frames' subject.
            for index in range(60):
                app._append_block(UserBlock(f"message {index + 1}: what does this show?"))
            await pilot.pause()
            await _settle(pilot, turns=10)
        elif state != "boot":
            editor.text = "what does this show?"
            await pilot.pause()
            await pilot.press("enter")
            await _settle(pilot)
            if state == "turn":
                # The same widget `on_turn_started` mounts, called directly:
                # the capture opens no real turn, and posting TurnStarted here
                # would leave one OPEN for teardown to abandon against a screen
                # that is already gone.
                app._start_working_block()
                await pilot.pause()
                await asyncio.sleep(1.0)
                await pilot.pause()
        await pilot.pause()
        save_capture(app, str(out / f"conversation-start-{state}.svg"))
        metrics = _metrics(app, state, size)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
