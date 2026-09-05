"""Capture the terminal TITLE a failed turn leaves behind, beside the frame.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/failed_title_shot.py out.svg after
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/failed_title_shot.py out.svg before

Drives the REAL ``OperatorApp`` (the one that loads ``local_operator.tcss``)
through a turn that dies without an ``agent_end`` — the reported incident's
route — and renders the resulting tab title as a banner row above the
transcript, because the title is the surface under review and a screenshot of
the app cannot otherwise show it.

``before`` renders the title the pre-fix code produced: ``TitleState`` had no
``failed`` member, so a session whose turn died with an error rendered
``lo › name``, identical to one that finished cleanly. That is the whole defect
— the surface a user scans to find what needs them could not distinguish the
two — so the pair is what makes the change legible. It is derived by asking
``build_title`` for the ``idle`` state rather than by reverting the app, so the
"before" string is the real renderer's output and not a hand-typed caption.

The title bytes are also printed to stdout, so the capture doubles as the OSC
trace: the run asserts that no ``lo ›`` ("finished cleanly") write is emitted on
a turn that failed, which is the ordering claim D6/U8 raised.
"""

import asyncio
import sys
from typing import Any

sys.path.insert(0, ".")

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import TurnStarted  # noqa: E402
from local_operator.tui.terminal_title import TerminalTitle, build_title  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock, UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The reported incident: an MCP grant that expired mid-turn, which is what
#: makes ``prompt()`` raise and so produces a turn with no ``agent_end``.
ERROR = "MCP error: MCP server 'linear' refused the connection (401)"


class _DyingSession(FakeSession):
    """``prompt()`` raises, so no ``agent_end`` ever reaches the app."""

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        raise RuntimeError(ERROR)


async def main() -> None:
    out, mode = sys.argv[1], sys.argv[2]
    app = OperatorApp(lambda: _factory(_DyingSession()))
    writes: list[str] = []
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        assert app._status is not None
        # The headless pilot attaches no TerminalTitle (there is no terminal to
        # title), so attach a real one over a recording sink: the object is the
        # product's, only its writer is ours.
        title = TerminalTitle(writes.append, enabled=True)
        title.start()
        app._status.set_terminal_title(title)
        app._status.update(conversation_name="audit the release")
        await pilot.pause()

        app._append_block(UserBlock("Run the audit"))
        app.post_message(TurnStarted())
        await pilot.pause()
        writes.clear()  # the window under test is the TURN, not the attach

        app._start_turn("Run the audit")
        for _ in range(10):
            await pilot.pause()
            await asyncio.sleep(0.02)

        rendered = [w.split("\x07")[0].split(";", 1)[1] for w in writes if "]0;" in w]
        after = rendered[-1] if rendered else build_title("audit the release", "failed")
        # The pre-fix renderer for the same settled state: no `failed` member,
        # so a died-with-an-error session was written as idle.
        before = build_title("audit the release", "idle")
        shown = before if mode == "before" else after

        # The title is not part of the app's own frame, so it is banner-rowed in
        # to make the still self-describing.
        app._append_block(
            NoticeBlock(f"tab title, {mode}:   {shown}", "error" if mode == "after" else "info")
        )
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out)

    print(f"title writes during the turn: {rendered}")
    print(f"before: {before!r}")
    print(f"after:  {after!r}")
    assert not any(
        t.startswith("lo ›") for t in rendered
    ), f"a failed turn wrote 'finished cleanly': {rendered}"


asyncio.run(main())
