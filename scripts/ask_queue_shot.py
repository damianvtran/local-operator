"""Capture the queued-ask surfaces over a populated transcript (§5.1's B recipes).

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ask_queue_shot.py OUT.svg [COLSxROWS] [MODE] [ROW]

MODE is one of (the design note's frame list, one frame per run so a still is
always the state its filename claims):

    bar          MINIMIZED, one open ask — the default presentation
    bar3         MINIMIZED, three open asks (the count and the head question)
    bar-timeout  MINIMIZED with nothing open but one timed-out ask still
                 answerable — the case where the count is not what is drawn
    list         EXPANDED onto the list of the three, with a timed-out row
    card         EXPANDED onto one ask's picker
    response     the ask_response card COLLAPSED
    response-open the same card with its Q&A open (ROW is ignored)
    timeout      the ask_timeout card collapsed, then opened with ROW=1

WHY THE TRANSCRIPT IS SEEDED FIRST. Every frame here has to answer "can the
user still read the conversation behind this surface?" — the bar is one row in
a dock that also carries the composer, and the list/card are panels above it.
An empty app would make that question unanswerable.

The same reason `ask_shot.py` gives for its own seeding: the seeded turns are
what make a still a picture of a STATE rather than of an empty screen.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.asks import policy  # noqa: E402
from local_operator.tui import theme as theme_mod  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.ask_queue import ask_rows  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    AskResponseBlock,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The flag is an env seam (D9). Set for the CAPTURE only: these frames are the
#: flag-ON surface, and the before-frames on `origin/main` are the flag-off app.
policy.NONBLOCKING_ASK = True


def _row(
    ask_id: str, question: str, *, status: str = "open", urgent: bool = False
) -> dict[str, Any]:
    return {
        "ask_id": ask_id,
        "created_at": 1_700_000_000_000,
        "expires_at": 1_700_003_600_000,
        "timeout_s": 3600,
        "urgent": urgent,
        "status": status,
        "delivered": False,
        "questions": [
            {
                "id": "q1",
                "question": question,
                "options": [
                    {"label": "Drop the rows", "description": "nothing reads the column"},
                    {
                        "label": "Backfill from the audit log",
                        "description": "slower, keeps history",
                    },
                ],
                "multi": False,
                "recommended": None,
                "secret": False,
                "persist": False,
            }
        ],
    }


THREE = [
    _row("a1", "Which rollout should the stale-row migration take?"),
    _row("a2", "Rotate the deploy key before the cutover?", urgent=True),
    _row("a3", "Which region do we fail over to?", status="timed_out"),
]

RESPONSE = {
    "ask_id": "a1",
    "status": "answered",
    "questions": [
        {
            "id": "q1",
            "question": "Which rollout should the stale-row migration take?",
            "options": [],
            "multi": False,
            "recommended": None,
            "secret": False,
            "persist": False,
        },
        {
            "id": "deploy_key",
            "question": "Paste the deploy key",
            "options": [],
            "multi": False,
            "recommended": None,
            "secret": True,
            "persist": False,
        },
    ],
    "answers": {"q1": ["Backfill from the audit log"], "deploy_key": ["[DEPLOY_KEY]"]},
    "text": "Answered — the whole ask was answered in one write.",
}

TIMEOUT = {
    "ask_id": "a3",
    "status": "timed_out",
    "waited_s": 3600,
    "urgent": False,
    "lapsed_while_stopped": False,
    "questions": [
        {
            "id": "q1",
            "question": "Which region do we fail over to?",
            "options": [],
            "multi": False,
            "recommended": None,
            "secret": False,
            "persist": False,
        }
    ],
    "text": "The deadline passed while the agent kept working.",
}


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    mode = "bar"
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    if len(sys.argv) > 3:
        mode = sys.argv[3]
    reveal = False
    if len(sys.argv) > 4:
        reveal = sys.argv[4].strip() in {"1", "reveal", "open", "true"}

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # `ASK_QUEUE_SHOT_THEME=light` renders the same state on the paper ramp.
        # An env seam rather than an argument, because the frame's FILENAME is
        # what a reader compares against: threading a theme through every mode's
        # name is how two runs of "the same" frame end up labelled alike.
        wanted = os.environ.get("ASK_QUEUE_SHOT_THEME", "").strip()
        if wanted:
            theme_mod.set_theme(wanted)
            app.refresh_css()
            await pilot.pause()
        for turn in range(1, 5):
            app._append_block(UserBlock(f"Turn {turn}: what should we do about the stale rows?"))
            prose = AssistantBlock()
            prose.update_text(
                f"Answer {turn}: the audit log still has every row, so a backfill is possible. "
                "Nothing else reads that column today."
            )
            app._append_block(prose)
        await pilot.pause()

        if mode == "bar":
            app._sync_ask_surface(ask_rows([THREE[0]]))
        elif mode == "bar3":
            app._sync_ask_surface(ask_rows(THREE))
        elif mode == "bar-timeout":
            app._sync_ask_surface(ask_rows([THREE[2]]))
        elif mode == "list":
            app._sync_ask_surface(ask_rows(THREE))
            await pilot.pause()
            app._expand_asks()
        elif mode == "card":
            app._sync_ask_surface(ask_rows([THREE[0]]))
            await pilot.pause()
            app._expand_asks()
        elif mode in {"response", "response-open", "timeout"}:
            details = TIMEOUT if mode == "timeout" else RESPONSE
            kind = "timeout" if mode == "timeout" else "response"
            block = AskResponseBlock(details, kind=kind)
            app._append_block(block)
            await pilot.pause()
            if reveal:
                block.action_activate()
                # The expansion opens BELOW the last line of a transcript that is
                # already at the viewport's end, so without this the still shows
                # the collapsed row and the Q&A sits off-screen — two frames
                # that differ only in the widget's state, not in the pixels.
                # Explicit rather than left to the tail anchor: the anchor
                # follows content that GROWS while the reader is at the bottom,
                # and this is the capture confirming it did.
                try:
                    app._transcript_view().scroll_end(animate=False)
                except Exception:  # pragma: no cover - harness shape only
                    pass
        else:  # pragma: no cover - a typo must not produce a silent empty frame
            raise SystemExit(f"unknown mode {mode!r}")
        for _ in range(12):
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
