"""Capture the frame a SIDEBAR ATTACH leaves behind, for the MCP startup toast.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/mcp_toast_attach_shot.py OUT.svg [COLSxROWS]

The reported defect is a frame, not a log line: "MCP ready: 12 servers, 425
tools" painted over the user's transcript on EVERY session attach, including
every click in the session sidebar. So the evidence for the fix has to be the
frame after an attach — the toast present on the pre-fix tree, absent on the
fixed one — which is what this script renders.

WHAT IT DOES, and why in this order. It boots the app on a session carrying a
startup outcome, lets the boot toast appear, DISMISSES it (the boot announce is
wanted and is not what is under test), then adopts a SECOND session carrying an
outcome with identical content — which is what a sidebar click does, since MCP
servers are process-wide and every session in the list shares one set. The
frame captured is the one the user is looking at a beat after that click.

The transcript is seeded first so the toast has something to paint OVER: an
overlay against an empty splash understates the interruption the report is
about. The status band is in frame deliberately — it carries the live
``⊙ 3 MCP`` segment on both trees, and that segment continuing to re-state MCP
state on every attach is precisely what makes silencing the repeat toast safe.

Set ``LO_MCP_TOAST_SHOT_STAGE=boot`` to capture the FIRST loadup instead, which
is the half of the rule that must not change: the fix suppresses the repeat, not
the announce. A before/after pair of the attach frame alone cannot show that,
since both trees are expected to differ there — the boot pair proves the two
trees still agree where they should.

Deterministic: the outcome is fixed, the toast's own dismissal timer is far
longer than the capture, and nothing here animates.

Every wait here POLLS for the condition rather than spending a fixed number of
ticks. A fixed budget raced the boot announce and tripped this script's own
assertion in 2 of 11 runs on an idle machine (UX round 1, U6) — the app is fine,
but the next person re-capturing evidence reads a script flake as a product
regression. AGENTS.md's timing section: wait on the event, never on the clock.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.session.mcp_status import McpStartupOutcome  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeMcpManager,
    McpSession,
    _band,
    _factory,
)

#: The reported shape, shrunk to fit a 100-column frame: several servers up and
#: a four-figure tool tally, exactly the sentence the operator saw repeated.
OUTCOME = McpStartupOutcome(
    configured=("github", "linear", "slack"),
    connected=("github", "linear", "slack"),
    tool_count=425,
)


class _IdentifiedMcpSession(McpSession):
    """``McpSession`` with a distinct id, so the two sessions are not one."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            FakeMcpManager(list(OUTCOME.configured), list(OUTCOME.connected)),
            OUTCOME,
        )
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id


async def _until(pilot, predicate) -> bool:  # type: ignore[no-untyped-def]
    """Pause until ``predicate()`` holds, or the budget runs out.

    Returns whether it held, so a caller asserting the NEGATIVE — a toast that
    must stay absent — can still spend the full budget looking for it instead
    of concluding from a frame that had not settled yet.
    """
    for _ in range(200):
        await pilot.pause()
        if predicate():
            return True
    return False


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    first = _IdentifiedMcpSession("aaaaaaaaaaa1")
    second = _IdentifiedMcpSession("aaaaaaaaaaa2")

    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=size) as pilot:
        toast = app.query_one(Toast)
        assert await _until(
            pilot, lambda: toast.display
        ), "the BOOT announce is expected on both trees"
        boot_message = toast.message

        if os.environ.get("LO_MCP_TOAST_SHOT_STAGE") == "boot":
            # First loadup, over the splash the user actually sees on launch —
            # no seeded transcript, because that is not the state a boot is in.
            save_capture(app, out)
            print(
                f"terminal={size[0]}x{size[1]} stage=boot "
                f"toast_display={toast.display} toast_message={boot_message!r}"
            )
            return

        toast.dismiss_toast()
        await pilot.pause()

        for turn in range(1, 6):
            app._append_block(UserBlock(f"Turn {turn}: can you check the release window?"))
            prose = AssistantBlock()
            prose.update_text(
                f"Answer {turn}: the merge queue is clear, so the window is open. "
                "Nothing else is waiting on a tag right now."
            )
            app._append_block(prose)
        await pilot.pause()

        # THE ATTACH under test: a sidebar click onto a sibling session that
        # shares this process's MCP servers.
        app._adopt_session(second, replay_history=False)
        # The expected outcome here is ABSENCE, which no condition can confirm
        # early — so this one spends the whole budget on purpose, and captures
        # whatever the toast is doing by the end of it. On the pre-fix tree the
        # poll trips as soon as the re-announce lands, which is the frame that
        # script exists to capture.
        await _until(pilot, lambda: toast.display)

        save_capture(app, out)
        band = _band(app)
        print(
            f"terminal={size[0]}x{size[1]} "
            f"boot_toast={boot_message!r} "
            f"toast_after_attach_display={toast.display} "
            f"toast_after_attach_message={toast.message!r} "
            f"band_has_mcp={'MCP' in band} "
            f"band={band.strip()!r}"
        )


asyncio.run(main())
