"""Capture the `/move` picker and the band it changes, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/move_shot.py OUT.svg [COLSxROWS] [STATE]

``STATE`` selects the frame:

* ``open`` (default) — the picker as `/move` opens it: the current directory
  marked, recents, home, /tmp, the parent, then the children.
* ``filter`` — a word typed, narrowing the suggestions.
* ``path`` — a path typed, COMPLETING against the filesystem. This is the
  frame worth looking at hardest: the header has to say which of the two modes
  is in force, because an empty list means different things in each.
* ``empty`` — a path that matches nothing, so the "no directory matches" state
  is judgeable rather than inferred.
* ``band-before`` / ``band-after`` — the status band on either side of a move.
  The pair is the point: the band is the surface that must never disagree with
  where the session actually works, so a single "looks fine" frame cannot show
  the property under test.

Drives the real ``OperatorApp`` rather than a bare widget host on purpose: the
lightweight hosts in the test files declare no ``CSS_PATH``, so
``local_operator.tcss`` never applies to them and a still captured from one
cannot show what the user sees (AGENTS.md, "Visual validation").

The directory tree is SYNTHETIC and built under a temporary root, so the frames
are stable across machines and do not leak the operator's real paths into an
image attached to a PR. The suggestions are assembled by the real
``suggest_targets`` over that tree, which is what makes the frame evidence
rather than decoration: only the code under test can change what it shows.
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
from local_operator.tui.move_targets import (  # noqa: E402
    complete_path,
    remember_recent,
    suggest_targets,
)
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.move_picker import MovePickerScreen  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def build_tree(root: Path) -> tuple[Path, Path]:
    """A believable working tree: a home, two checkouts, and some children."""
    home = root / "home"
    workspace = home / "workspace" / "repos"
    project = workspace / "lo-move-cmd"
    for name in ("local_operator", "tests", "scripts", "docs", "benchmarks"):
        (project / name).mkdir(parents=True, exist_ok=True)
    (workspace / "lo-session-sidebar").mkdir(parents=True, exist_ok=True)
    (home / "oss" / "oh-my-pi").mkdir(parents=True, exist_ok=True)
    config = root / "config"
    config.mkdir(exist_ok=True)
    # Two remembered directories, so the `recent` tier is populated and its
    # ordering against home/tmp/parent is visible in the frame.
    remember_recent(config, str(home / "oss" / "oh-my-pi"))
    remember_recent(config, str(workspace / "lo-session-sidebar"))
    return project, config


def seed_conversation(app: OperatorApp) -> None:
    """So "does this overlay still let me read the conversation?" is an
    answerable question rather than a screenshot of an empty app."""
    for turn in range(1, 6):
        app._append_block(UserBlock(f"Turn {turn}: can you check the migration script?"))
        prose = AssistantBlock()
        prose.update_text(
            f"Answer {turn}: the script reads every row once and writes the"
            " backfill in batches, so it is safe to re-run."
        )
        app._append_block(prose)


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    state = sys.argv[3] if len(sys.argv) > 3 else "open"

    with tempfile.TemporaryDirectory(prefix="lop-move-shot-") as raw_root:
        root = Path(raw_root)
        project, config = build_tree(root)
        home = root / "home"

        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            seed_conversation(app)
            await pilot.pause()

            if state in ("band-before", "band-after"):
                # The band pair. `_push_cwd_to_band` is the one call the move
                # makes to keep the band honest, so driving it directly is
                # exactly the surface under test.
                app._push_cwd_to_band(str(project))
                if state == "band-after":
                    for _ in range(2):
                        await pilot.pause()
                    app._push_cwd_to_band(str(home / "oss" / "oh-my-pi"))
                for _ in range(4):
                    await pilot.pause()
                assert app._status is not None
                print(f"band: {app._status.render_text(size[0]).plain}", file=sys.stderr)
                save_capture(app, out)
                return

            targets = suggest_targets(project, config_dir=config, home=home)
            screen = MovePickerScreen(
                targets,
                current=str(project),
                complete=lambda query: complete_path(query, cwd=project, home=home),
            )
            app.push_screen(screen)
            # Let the overlay's layout SETTLE before the capture: the first
            # painted frame is still sized to the pre-push geometry, and rows
            # come out truncated in a way the running app never shows.
            for _ in range(6):
                await pilot.pause()

            if state == "filter":
                screen.set_query("repos")
            elif state == "path":
                screen.set_query(f"{project}/")
            elif state == "empty":
                screen.set_query(f"{project}/zzz")
            for _ in range(4):
                await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            screen._repaint()
            await pilot.pause()

            # The numbers behind the frame: the stills show the symptom, the
            # geometry shows the cause. A card whose virtual height exceeds the
            # screen's own size has made the screen scrollable, which silently
            # costs two cells of width and reflows the transcript behind it.
            print(f"state={state} size={size}", file=sys.stderr)
            print(
                f"  card_width={screen._card_width()} page_rows={screen._page_rows()} "
                f"rows={len(screen.visible_rows)} path_mode={screen.is_path_query}",
                file=sys.stderr,
            )
            print(
                f"  screen.size={screen.size} virtual_size={screen.virtual_size} "
                f"scrollbar={screen.show_vertical_scrollbar}",
                file=sys.stderr,
            )
            for line in screen.render_lines_for_test():
                print(f"  |{line}|", file=sys.stderr)
            save_capture(app, out)


if __name__ == "__main__":
    asyncio.run(main())
