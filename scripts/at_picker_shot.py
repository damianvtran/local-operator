"""Capture the ``@`` reference picker open over a seeded conversation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/at_picker_shot.py OUT.svg [COLSxROWS]

The picker is user-visible, so a passing test is not evidence that it looks
right (AGENTS.md, "Visual validation"). What has to be judged from a frame and
cannot be judged from an assertion: that the rows sit under the composer
without towering over the transcript, that a directory row reads as a directory
(the trailing slash is the only signal), and that the file list is legible
beside the command and skill lists it shares a widget with.

Drives the real ``OperatorApp`` rather than a bare widget host on purpose: the
lightweight hosts in the test files declare no ``CSS_PATH``, so
``local_operator.tcss`` never applies to them and a still captured from one
cannot show what the user sees.

The tree it lists is BUILT HERE, in a temporary directory, rather than pointed
at the repo: a capture of whatever happened to be in the developer's cwd is not
comparable between two runs, and the before/after pair is only evidence if the
only thing that changed is the code.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402

#: The draft the frame is captured on. A question with the token mid-line, not
#: at the start, because that is the realistic shape and it is the one that
#: exercises inline detection.
DRAFT = "what does @src/"


def _seed_tree(root: Path) -> None:
    """A directory with the three row kinds the frame needs to distinguish."""
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("app\n")
    (root / "src" / "editor.py").write_text("editor\n")
    (root / "src" / "session.py").write_text("session\n")
    (root / "src" / "widgets").mkdir()
    (root / "README.md").write_text("readme\n")


async def _capture(out: Path, size: tuple[int, int]) -> None:
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text(DRAFT)
        editor.move_cursor(editor._end_of_buffer())
        # Two pauses: the app answers `FileQueryOpened` one message-loop tick
        # after the keystroke, so a single pause captures the list mid-fill.
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out)


def main() -> None:
    out = Path(sys.argv[1]).expanduser().resolve()
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].lower().split("x")
        size = (int(cols), int(rows))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_tree(root)
        import os

        os.chdir(root)
        asyncio.run(_capture(out, size))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
