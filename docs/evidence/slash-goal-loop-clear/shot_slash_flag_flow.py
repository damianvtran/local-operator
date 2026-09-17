"""Drive the flag row with REAL keys: one Enter FILLS, the second RUNS.

The sibling script (``shot_slash_flag_picker.py``) captures the row standing
still. This one captures what the row does when it is accepted, because that is
the behaviour round 1 turned on and the property a still cannot show.

Why it exists: the row is the PRE-SELECTED row and usually the only match, so
``Editor._picker_choice_is_unambiguous`` ran it on one Enter — which turned
``/goal `` + Enter, the keystroke ``/goal``'s own description teaches for
*reading* the standing goal, into a clear (design D1 / UX U1 / code MAJOR-1).
``ArgumentChoice.alert`` is the app's existing gate for "accepting this row
removes something"; with it set the first Enter FILLS the buffer with
``/goal --clear`` and the second runs it. Both states are frames here.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        docs/evidence/slash-goal-loop-clear/shot_slash_flag_flow.py OUTDIR \
        [goal-fill|goal-run|loop-fill|loop-run|narrow]

Every case also prints the observed state to stderr (the numbers behind the
frame, AGENTS.md §4) — including the buffer the fill wrote and the goal the run
left behind, which is what the frame is evidence OF.

``narrow`` is the U5 case: the row must ELLIPSISE its label rather than drop it
when the terminal is narrow. It renders the row at six widths and captures the
40-column frame, which is where the base tree showed ``❯  --clear`` with no
label at all.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

GOAL = "land the OAuth refresh fix"
SIZE = (100, 30)
NARROW_SIZE = (40, 30)
#: The widths the row is measured at. 40 is where the base tree dropped the label
#: (`DESCRIPTION_COLLAPSE_WIDTH`); 38 is the width the UX round reported; 44 is
#: the narrowest at which the base still showed it.
MEASURED_WIDTHS = (100, 80, 60, 44, 40, 38)


async def _boot(app: OperatorApp, pilot, session: FakeSession) -> None:
    """Settle until the app has adopted its session — the gate reads its goal."""
    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
    else:  # pragma: no cover — a boot that never adopts is a different failure
        raise RuntimeError("the app never adopted its session")
    assert app._session is session


async def _seed(app: OperatorApp, pilot) -> None:
    """A settled conversation behind the composer, so the frame is not an empty app."""
    for text in ("kick off the refresh work", f"/goal {GOAL}"):
        app.query_one(Editor).text = text
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _picker_rows(app: OperatorApp, width: int) -> list[str]:
    return [row.plain.rstrip() for row in app.query_one(Editor).picker.render_rows(width)]


def _report(app: OperatorApp, session: FakeSession, note: str) -> None:
    editor = app.query_one(Editor)
    screen = app.screen
    print(
        f"{note}: buffer={editor.text!r} goal={session.goal!r} "
        f"prompts={session.prompts!r} loop_cancelled={app._loop_cancelled} "
        f"row={_picker_rows(app, screen.size.width)} "
        f"screen={screen.size} virtual={screen.virtual_size} "
        f"vscroll={screen.show_vertical_scrollbar} "
        f"transcript={app.query_one(TranscriptView).size} "
        f"notices={_notices(app)[-2:]!r}",
        file=sys.stderr,
    )


async def _type_row(pilot, app: OperatorApp, command: str) -> None:
    """Type `/go`/`/lo` and Tab, which is the taught route onto the row.

    Typed a character at a time rather than assigned, because the row is raised
    by the composer's own caret-anchored parse: an assignment would test the
    fixture instead of the path.
    """
    editor = app.query_one(Editor)
    editor.focus()
    for char in command:
        await pilot.press(char)
        await pilot.pause()
    await pilot.press("tab")
    await pilot.pause()


async def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    case = sys.argv[2] if len(sys.argv) > 2 else "goal-fill"

    session = FakeSession()
    session.set_goal(GOAL)
    app = OperatorApp(lambda: _factory(session))
    size = NARROW_SIZE if case == "narrow" else SIZE
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _seed(app, pilot)
        await _boot(app, pilot, session)
        editor = app.query_one(Editor)
        if case == "narrow":
            # The row is raised FIRST, then measured: `render_rows` is empty while
            # the list is closed, and the widths below are the question.
            await _type_row(pilot, app, "/go")
            for _ in range(3):
                await pilot.pause()
            for width in MEASURED_WIDTHS:
                print(f"width={width}: rows={_picker_rows(app, width)}", file=sys.stderr)
            _report(app, session, "narrow")
            save_capture(app, out_dir / "picker-40x30.svg")
            await pilot.pause()
            save_capture(app, out_dir / "picker-40x30-settled.svg")
            return
        loop = case.startswith("loop")
        app._loop_running = loop
        await _type_row(pilot, app, "/lo" if loop else "/go")
        for _ in range(3):
            await pilot.pause()
        print(f"before the first Enter: row={_picker_rows(app, size[0])}", file=sys.stderr)
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        _report(app, session, "after ONE Enter")
        if case.endswith("-fill"):
            save_capture(app, out_dir / ("loop-fill.svg" if loop else "goal-fill.svg"))
            await pilot.pause()
            save_capture(
                app, out_dir / ("loop-fill-settled.svg" if loop else "goal-fill-settled.svg")
            )
            assert editor.text == ("/loop --stop" if loop else "/goal --clear"), editor.text
            assert app._loop_cancelled is False
            assert session.goal == GOAL, "the fill ran the row"
            return
        # The second Enter accepts the completed row.
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        for _ in range(3):
            await pilot.pause()
        _report(app, session, "after the second Enter")
        save_capture(app, out_dir / ("loop-run.svg" if loop else "goal-run.svg"))
        await pilot.pause()
        save_capture(app, out_dir / ("loop-run-settled.svg" if loop else "goal-run-settled.svg"))
        if loop:
            assert app._loop_cancelled is True
        else:
            assert session.goal == ""


asyncio.run(main())
