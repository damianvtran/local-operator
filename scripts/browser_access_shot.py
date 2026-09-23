"""Capture the browser access receipt, as the TUI transcript paints it.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/browser_access_shot.py OUT.svg [CASE] [COLSxROWS]

``CASE`` selects the state whose receipt is captured, because the change to
these strings has to be judged against the shapes it must not alter:

    attached            (default) a pending prompt on an attached session —
                        the 15-minute budget, the repeated `await_access`
                        calls and the bounded re-request
    unattached          the same, on a session with no pane attached, whose
                        advice is "proceed with what you have rather than
                        blocking the turn" — the pair that must agree
    timeout             the `await_access` timeout arm, attached
    timeout_unattached  the same arm on a detached session, which used to tell
                        every caller to keep waiting

**Why a script rather than an assertion.** The card paints ONE ROW PER RAW
LINE and clips each to the measure (94 cells at 100 columns), so a paragraph
authored as one long line loses its tail on the frame while the model still
receives every byte: a passing test cannot show that "proceed with what you
have" or "15 MINUTES" survived the card. Design review round 1 (D6) measured
four of these rows clipped; this script is what makes "re-render and look"
a one-command check for the next round.

The text is taken from the SHIPPED builder (``_access_result_text``) rather
than hand-typed, so a frame cannot argue about a sentence the code no longer
produces. The card is expanded before capture: the collapsed card is one cell
tall by design, and the rows under review are the expanded body.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot must not rename the operator's real workspace through inherited CMUX IDs.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tools import builtin  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: A real origin from the incident's own flow, long enough that the `where`
#: sentence and the origin do not wrap for free.
URL = "https://www.linkedin.com/feed/"

#: Attachment per case, so the two directions of one arm can be compared in
#: two frames rather than two paragraphs.
ATTACHED = {"attached": True, "unattached": False, "timeout": True, "timeout_unattached": False}


def receipt(case: str) -> str:
    """The shipped text for one case, exactly as a tool result would carry it."""
    attached = ATTACHED[case]
    if case.startswith("timeout"):
        return builtin._access_result_text(
            "await_timeout",
            URL,
            host="bridge",
            attached=attached,
            notify="a short message, or `ask`" if attached else "a short message",
            total_s=240.0,
        )
    return builtin._access_result_text(
        "pending",
        URL,
        position=1,
        pending_count=1,
        host="bridge",
        attached=attached,
        notify="a short message, or `ask`" if attached else "a short message",
    )


async def main() -> None:
    out = sys.argv[1]
    case = sys.argv[2] if len(sys.argv) > 2 else "attached"
    if case not in ATTACHED:
        raise SystemExit(f"unknown CASE {case!r}; expected one of {sorted(ATTACHED)}")
    size = (100, 32)
    if len(sys.argv) > 3:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # A seeded conversation, so the frame is "does this still let me read
        # the transcript" rather than a screenshot of an empty app.
        app._append_block(UserBlock(f"approve {URL} so I can read the feed"))
        app._append_block(AssistantBlock())
        await pilot.pause()
        card = ToolCard("t1", "browser", {"action": "request_access", "url": URL})
        app._append_block(card)
        card.mark_done(receipt(case), {"origin": URL, "state": "pending"}, measured_s=0.4)
        await pilot.pause()
        if not card.can_expand():
            raise AssertionError("the card cannot expand: the frame would show one row")
        card.toggle_expanded()
        for _ in range(6):
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
