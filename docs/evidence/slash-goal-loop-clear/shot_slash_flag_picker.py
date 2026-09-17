"""Capture the `/goal` and `/loop` argument pickers when their flag row is offered.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        docs/evidence/slash-goal-loop-clear/shot_slash_flag_picker.py OUT.svg [COLSxROWS] [CASE]

``CASE`` selects the live state the suggestion is gated on, because the row's
whole contract is that it appears only when the state it would change exists:

    goal-set      (default) a standing goal IS set and the caret sits on an EMPTY
                  `/goal ` argument, which is where `--clear` is offered.
    goal-none     no standing goal; the same buffer must offer NOTHING, or the
                  palette would advertise a clear for a goal that is not there.
    goal-typed    `/goal ship it` — free text in the argument region. The row is
                  gone and the buffer is still an ordinary goal.
    loop-running  a loop IS running in this app and the caret sits on an EMPTY
                  `/loop ` argument, which is where `--stop` is offered.
    loop-idle     no loop running; the same buffer must offer NOTHING.

The fixture (transcript, session, size) is IDENTICAL across cases, so a frame
difference is the state and not the data — the ONE exception being the app's own
"goal restored" notice, which a set goal produces by itself.

Drives the REAL :class:`OperatorApp`, the only host that loads
``local_operator.tcss`` (AGENTS.md, "Visual validation"), through the real
editor: the buffer is assigned and the picker re-derives on the same path a
keystroke takes. ``isolate_capture`` re-homes HOME, config and caches first, and
every ``CMUX_*`` variable is dropped before any application import, so this
never reads or writes the operator's own config or cmux workspaces.

The property to read off the frame is the ONE dim row under the composer: it
completes to the flag the case is about, and it is the only row (a second row
would mean free text had become a value list). The geometry numbers printed to
stderr are the other half of the evidence — the picker's content box against its
pinned height, and whether the overlay pushed the screen into scrolling.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The state each case runs in. ``goal`` seeds the standing objective the
#: `--clear` row would take away; ``loop`` is this app's own loop flag, the
#: state `--stop` would end.
CASES: dict[str, dict[str, object]] = {
    "goal-set": {"goal": "land the OAuth refresh fix", "buffer": "/goal ", "loop": False},
    "goal-none": {"goal": "", "buffer": "/goal ", "loop": False},
    "goal-typed": {"goal": "", "buffer": "/goal ship it", "loop": False},
    "loop-running": {"goal": "", "buffer": "/loop ", "loop": True},
    "loop-idle": {"goal": "", "buffer": "/loop ", "loop": False},
}


async def _seed(app: OperatorApp, pilot) -> None:
    """A settled conversation behind the composer.

    An empty app would make "can this surface still be read?" unanswerable, and
    the picker's own row count is measured against a frame with the transcript
    scrolled to the bottom, which is where it is docked.
    """
    app._append_block(UserBlock("the loop keeps re-running the flaky job — can you stop it?"))
    prose = AssistantBlock()
    prose.update_text(
        "It is a bounded loop this terminal owns, so `/loop --stop` ends it after the "
        "current turn. The standing goal survives that, which is what `/goal --clear` "
        "is for."
    )
    app._append_block(prose)
    app._append_block(NoticeBlock("goal set", "info"))
    await pilot.pause()
    await pilot.pause()


def _report_geometry(app: OperatorApp, editor: Editor) -> None:
    """The numbers behind the frame — a still shows the symptom, these the cause."""
    picker = editor.picker
    screen = app.screen
    rows = picker.render_rows(screen.size.width)
    print(
        f"picker: mode={picker.mode.value} open={picker.is_open()} "
        f"noticed={bool(picker._notice)} rows={len(rows)} "
        f"content={picker.size} pinned={picker.styles.height} "
        f"display={picker.display}",
        file=sys.stderr,
    )
    print(
        f"screen: size={screen.size} virtual={screen.virtual_size} "
        f"vscroll={screen.show_vertical_scrollbar} "
        f"composer={editor.size} buffer={editor.text!r} "
        f"argument_command={editor._argument_command!r} "
        f"transcript={app.query_one(TranscriptView).size} "
        f"rows={[row.plain for row in rows]}",
        file=sys.stderr,
    )


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    case = sys.argv[3] if len(sys.argv) > 3 else "goal-set"
    state = CASES[case]

    session = FakeSession()
    session.set_goal(str(state["goal"]))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _seed(app, pilot)
        editor = app.query_one(Editor)
        # The app's own loop flag: the same attribute `_cmd_loop` reads, so the
        # frame is captured in the state the gate is written against.
        app._loop_running = bool(state["loop"])
        editor.text = str(state["buffer"])
        # The caret goes to the END of the buffer, which is the position the whole
        # contract is about: `slash_argument` is caret-anchored, so a caret left
        # at offset 0 (where assigning ``text`` leaves it) is not "inside the
        # argument region" at all and the picker would never be asked.
        editor.move_cursor(editor._end_of_buffer())
        # Three frames: the fill lands one message-loop tick after the keystroke
        # that opened the list, and the pair written below must then match.
        for _ in range(3):
            await pilot.pause()
        _report_geometry(app, editor)
        save_capture(app, out)
        # A SECOND frame one pause later: identical frames are the proof that the
        # row settled rather than still painting (AGENTS.md §5 "Animation and
        # multi-frame changes"). A first frame that differs from this one is a
        # reflow the user would see as motion.
        await pilot.pause()
        save_capture(app, out.replace(".svg", "-settled.svg"))


asyncio.run(main())
