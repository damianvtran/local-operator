"""Capture the `/resume` picker over a store that has an ARCHIVED session.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/picker_archive_shot.py OUT.svg [COLSxROWS] [reveal]

Three states are worth a frame, and they are the three the feature's claims rest
on:

* **default** — the archived conversation is not offered, and the toggle row says
  it exists ("Archived (1) hidden").
* **hover-toggle** — the toggle line with the pointer on it (D5; design round 2, D9),
  for the same store as the default frame.
* **reveal** — the same store with `ctrl+a` pressed: the row is back, carrying its
  `[archived]` mark, and the toggle reads "shown".
* **no-archive** — a store where nothing is archived, i.e. what every user of the
  previous release has: no toggle row at all.

`ARCHIVED=0` in the environment produces that last state from the same script, so
the three frames are the same code path with one input changed, and
``THEME=<palette>`` renders any of them in another palette (``THEME=light`` for
the paper ramp).

THE SCRIPT RUNS ON THE PRE-FEATURE TREE TOO, which is what makes the before/after
pair honest: ``set_archived`` is imported tolerantly, so a worktree checked out at
the parent commit draws the same store with no archived rows and no toggle — the
only difference between the frames is the change under test.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.paths import config_dir  # noqa: E402
from local_operator.resume import write_session_title  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

try:  # the pre-feature tree draws the same store and simply has nothing to show
    from local_operator.session.archived import set_archived  # noqa: E402
except ImportError:  # pragma: no cover - only reached in a before-frame worktree
    set_archived = None  # type: ignore[assignment]

#: Three conversations with names, so the frame answers "can I tell which is
#: which" rather than showing three untitled rows.
SESSIONS = [
    ("a1b2c3d4e5f6", "Retention sweep for the analytics ledger"),
    ("b2c3d4e5f6a1", "Q3 pricing model review"),
    ("c3d4e5f6a1b2", "Parser crash on nested frontmatter"),
]


def _seed(root: Path) -> None:
    for session_id, name in SESSIONS:
        directory = root / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "type": "message",
                    "payload": {"role": "user", "content": f"{name}. Let's pick this up."},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (directory / "created_at.json").write_text(str(1700000000 + len(session_id)))
        # Through the REAL writer: the sidecar's shape is what ``session_name``
        # reads, and a hand-written copy that agreed with a wrong implementation
        # would draw three "(unnamed session)" rows — a frame that answers
        # "can I tell which is which" with no.
        write_session_title(directory, name, user_set=True, past_names=[])
    if set_archived is not None and os.environ.get("ARCHIVED", "1") != "0":
        set_archived(root, SESSIONS[1][0], True)


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2 and "x" in sys.argv[2]:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    reveal = "reveal" in sys.argv[2:]
    #: ``THEME=<palette name>`` renders the same store in that palette (``light``
    #: is the paper ramp). Every frame this PR shipped was dark, and the archive
    #: mark's ink is exactly the kind of choice that has to be looked at on paper
    #: too — D3 was measured in both themes before it was fixed.
    theme = os.environ.get("THEME", "")

    root = Path(config_dir())
    _seed(root)

    async def resume_factory(_session_id: str | None):
        return FakeSession()

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        # Through the real path: the typed command, the submit handler, the row
        # build and the screen. A frame taken from a hand-pushed widget would not
        # show the rows the app actually offers.
        editor.text = "/resume"
        editor.cursor_location = (0, len("/resume"))
        await pilot.pause()
        if editor._picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if app.screen.__class__.__name__ == "SessionPickerScreen":
                break
        await pilot.pause()
        if theme:
            from local_operator.tui import theme as theme_mod

            theme_mod.set_theme(theme)
            app.refresh_css()
            await pilot.pause()
        if reveal:
            await pilot.press("ctrl+a")
            await pilot.pause()
        if "hover-toggle" in sys.argv[2:]:
            # D5's own frame (design round 2, D9): the toggle line had no hover
            # state while the rows did, and a fix to a MOUSE affordance cannot be
            # evidenced by a still that does not move a mouse. The offset is the
            # toggle row's hit box, the same one the suite drives.
            # The results pane is addressed by its CSS id (the suite's own
            # selector constant lives in the test module, not in the widget).
            # ``app.screen`` rather than a local, so the widget is fetched from
            # the screen the loop above waited for.
            body = app.screen.query_one("#session-picker-results")
            await pilot.hover(body, offset=(4, 0))
            await pilot.pause()
        # SETTLE THE LIST BEFORE CAPTURING (design round 1, D1). The pane is
        # composed once, against the box it has at that moment, and a keystroke is
        # what recomposes it (`_repaint` is the only caller of `_results.update`).
        # Without this, the frame under test is the picker's OPENING TRANSIENT —
        # names truncated to the transient pane width — and a hidden/revealed pair
        # is not comparable, because the reveal frame was settled only by accident
        # (`ctrl+a` is itself a keypress). Two cursor moves, net zero, recompose at
        # the settled width and leave the selection where it was.
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        save_capture(app, out)


if __name__ == "__main__":
    asyncio.run(main())
