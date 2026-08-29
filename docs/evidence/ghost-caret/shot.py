"""Capture composer frames for the ghost-caret change.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        docs/evidence/ghost-caret/shot.py . /tmp/ghost-after

Drives the REAL ``OperatorApp`` (which declares CSS_PATH, so `local_operator.tcss`
applies) via ``run_test``, puts the composer in each state under test, and saves
an SVG per state plus a JSON of geometry numbers.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO = sys.argv[1]
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, REPO)

# Pin the cwd the app renders in its banner and status line.
#
# The before/after frames are shot from two different worktrees, so the REAL
# cwd differs between them — and the app puts it in two places. That made every
# pair differ in the banner and footer for a reason that has nothing to do with
# the change, so "diff the pair and see nothing move" did not work as a check,
# which is the whole purpose of the full-screen frames. At the 18-column
# narrow-width size the two paths even wrapped to different line counts and
# pushed the composer down by 434px, so the pair could not be compared as-shot
# at all (design review round 1, D3/D4).
#
# Stubbing `os.getcwd` is the honest fix rather than cropping the banner away:
# the frames still show the real surface, and any residual difference between a
# pair is now attributable to the change under test. Applied before the app is
# imported so the module-level readers see it too.
_SHOT_CWD = "/private/tmp/lop-ghost-shot"
os.getcwd = lambda: _SHOT_CWD  # type: ignore[assignment]

from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig  # noqa: E402
from local_operator.session.mcp_status import McpStartupOutcome  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeMcpManager,
    McpSession,
    _factory,
)


def _configs() -> dict[str, Any]:
    return {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        ),
        "notion": MCPHttpServerConfig(
            url="https://mcp.notion.com/mcp", auth=MCPAuthConfig(type="oauth")
        ),
    }


def _app() -> OperatorApp:
    manager = FakeMcpManager(["linear", "notion"], ["linear"])
    manager._configs = _configs()
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    return OperatorApp(lambda: _factory(session))


async def _settle(pilot: Any, times: int = 8) -> None:
    for _ in range(times):
        await pilot.pause()


async def _type(pilot: Any, keys: str) -> None:
    for char in keys:
        await pilot.press("space" if char == " " else char)
        await _settle(pilot, 5)


#: ``(name, size, seed, typed)``. ``seed`` is placed on the buffer directly (the
#: faithful shortcut for "the user already got here"), ``typed`` goes in as real
#: presses so the picker and the ghost derivation run for real.
CASES = [
    # The user's screenshot #1: a fully typed command whose only completion is
    # the trailing space -> whitespace-only ghost.
    ("resume-full", (100, 24), "", "/resume"),
    # A real multi-character ghost on the command word.
    ("resum-partial", (100, 24), "", "/resum"),
    # An enum-tail ARGUMENT ghost.
    ("mcp-lo", (100, 24), "/mcp ", "lo"),
    # A compound ARGUMENT-slot ghost (the server rows).
    ("mcp-login-n", (100, 24), "/mcp login ", "n"),
    # The user's screenshot #2: a command typed mid-draft, caret at the end.
    # Seeded SHORT ("hi ") rather than the original "check this " so the caret
    # cell stays inside the same crop window every other state uses; at the
    # longer seed the block fell on the window's right edge and was clipped in
    # the before panel and lost entirely in the after one (design review round
    # 1, D5). The state under test is "a command token with a draft in front of
    # it", which the shorter prefix expresses identically.
    # NOTE: this state is NOT a control — `hi /resume` yields the same
    # whitespace-only ghost as a bare `/resume`, so gate 4 changes it too (D1).
    # no ghost. Must be unchanged by this work.
    ("mid-draft", (100, 24), "hi ", "/resume"),
    # Narrow terminal: gate 2 withholds the ghost (measured: at w=18 the
    # composer's content box is 10 cells and `/analytic` + `s ` needs 11).
    # Must be unchanged.
    ("narrow-width", (18, 24), "", "/analytic"),
    # Ordinary prose, no command token at all: the no-ghost baseline the
    # ghost-bearing frames are read against.
    ("prose-no-ghost", (100, 24), "", "hello there"),
]


def _cells(editor: Editor, strip: Any) -> dict[str, Any]:
    """Which painted COLUMN carries the cursor ground, and which the dim ink.

    The cell index is what the defect is about (the block one cell left of the
    insertion point), and a strip's ``cell_length`` cannot show it because the
    composer pads the row to its box width. Walking the segments gives the
    column of every cell painted with the cursor background and with the
    suggestion foreground, which is the number the frames are read against.
    """
    cursor_style = editor.get_component_rich_style("text-area--cursor")
    ghost_style = editor.get_component_rich_style("text-area--suggestion")
    cursor: list[Any] = []
    ghost: list[Any] = []
    row = ""
    column = 0
    for seg in strip._segments:
        for char in seg.text:
            row += char
            style = seg.style
            if style is not None and style.bgcolor == cursor_style.bgcolor:
                cursor.append([column, char])
            if style is not None and style.color == ghost_style.color:
                ghost.append([column, char])
            column += 1
    return {"row": row.rstrip(), "cursor": cursor, "ghost": ghost}


async def main() -> None:
    geometry: dict[str, Any] = {}
    for name, size, seed, typed in CASES:
        app = _app()
        async with app.run_test(size=size) as pilot:
            await _settle(pilot, 6)
            editor = app.query_one(Editor)
            editor.focus()
            with patch(
                "local_operator.mcp.config.load_all_mcp_configs",
                return_value=(_configs(), {}),
            ):
                if seed:
                    editor.text = seed
                    editor.move_cursor(editor._end_of_buffer())
                    await _settle(pilot, 5)
                await _type(pilot, typed)
                await _settle(pilot, 12)

                ghost = editor.suggestion
                strip = editor.render_line(0)
                with_ghost = strip.cell_length
                cells = _cells(editor, strip)
                # The same row with the ghost forced off, so anything
                # attributable to the preview is measured rather than inferred.
                editor.suggestion = ""
                await _settle(pilot, 4)
                bare = editor.render_line(0)
                without_ghost = bare.cell_length
                bare_cells = _cells(editor, bare)
                editor.suggestion = ghost
                await _settle(pilot, 6)

                geometry[name] = {
                    "size": list(size),
                    "text": editor.text,
                    "ghost": ghost,
                    "picker_open": editor.picker.is_open(),
                    "strip_cell_length_with_ghost": with_ghost,
                    "strip_cell_length_without_ghost": without_ghost,
                    "delta": with_ghost - without_ghost,
                    "caret_document_column": list(editor.selection.end),
                    "painted_row": cells["row"],
                    "cursor_cells": cells["cursor"],
                    "ghost_cells": cells["ghost"],
                    "cursor_cells_without_ghost": bare_cells["cursor"],
                }
                app.save_screenshot(str(OUT / f"{name}.svg"))
                # A consecutive frame: if it differs, the row is reflowing
                # after paint, which the user sees as motion.
                await _settle(pilot, 6)
                app.save_screenshot(str(OUT / f"{name}.frame2.svg"))

    (OUT / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(json.dumps(geometry, indent=2))


asyncio.run(main())
