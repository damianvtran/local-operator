"""Capture a transcript of tool rows to prove the Nerd-icon autodetect gate.

Run from the worktree root, ONCE per env so the gate is resolved against that
env (the gate reads ``os.environ`` and the settings cache at row-build time):

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/nerd_glyph_shot.py OUT.svg {nerd|plain} [COLSxROWS]

- ``nerd``  seeds a ghostty marker (GHOSTTY_BIN) so autodetect enables the
  expanded Font Awesome glyphs.
- ``plain`` seeds Apple_Terminal and strips every bundling-emulator marker so
  autodetect falls back to the ASCII table (no tofu).

Both frames render the SAME seeded tool rows, so the only difference between
the two SVGs is the icon column — which is the whole point of the fix.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _seed_env(mode: str) -> None:
    """Force the process env into the terminal we want the gate to detect.

    Done BEFORE importing the app so the very first glyph lookup sees it. The
    gate has no interactive probe; it reads exactly these markers.
    """
    # Strip every emulator marker so neither frame inherits the real terminal.
    for var in (
        "GHOSTTY_RESOURCES_DIR",
        "GHOSTTY_BIN",
        "KITTY_WINDOW_ID",
        "WEZTERM_PANE",
        "WEZTERM_EXECUTABLE",
        "TERM_PROGRAM",
        "LOCAL_OPERATOR_NO_NERD_ICONS",
    ):
        os.environ.pop(var, None)
    if mode == "nerd":
        # cmux/ghostty case: the marker enables glyphs even under TERM=dumb.
        os.environ["GHOSTTY_BIN"] = "/opt/ghostty/bin"
    elif mode == "plain":
        os.environ["TERM_PROGRAM"] = "Apple_Terminal"
    else:
        raise SystemExit(f"unknown mode {mode!r}; want 'nerd' or 'plain'")


#: (tool_name, args) pairs spanning every icon category the fix touches.
#: Args typed ``dict[str, object]`` to match ``ToolCard.__init__``'s parameter;
#: without the annotation pyright infers ``dict[str, str]`` and rejects the call.
_ROWS: list[tuple[str, dict[str, object]]] = [
    ("bash", {"command": "pytest -q"}),
    ("read", {"path": "local_operator/tui/glyphs.py"}),
    ("edit", {"path": "local_operator/tui/glyphs.py"}),
    ("grep", {"query": "nerd_icons_enabled"}),
    ("task", {"prompt": "run the review gate"}),
    # display_name() strips the ``mcp__linear_`` prefix to the call segment,
    # so pick one whose call fits the 8-cell name spine without truncation
    # (``search`` -> ``search``); a longer call would clip and muddy the shot.
    ("mcp__linear_search", {"query": "tofu on Apple Terminal"}),
]


async def main() -> None:
    out = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "nerd"
    _seed_env(mode)

    size = (100, 24)
    if len(sys.argv) > 3:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))

    # Imported AFTER _seed_env so module-level env reads (if any) see our env.
    from local_operator.tui import settings as settings_mod  # noqa: E402
    from local_operator.tui.app import OperatorApp  # noqa: E402
    from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
    from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
    from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
    from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

    # Config is unset in this scratch checkout, so the gate is in AUTO — but
    # drop the cache anyway in case a prior run in the same interpreter primed
    # it, so detection resolves against the env we just seeded.
    settings_mod.settings_reload()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("Run the checks and open the review."))
        prose = AssistantBlock()
        prose.update_text("Working through the tool ledger below.")
        app._append_block(prose)
        for name, args in _ROWS:
            app._append_block(ToolCard("t", name, args))
        # Settle so every row paints its final icon before the capture.
        for _ in range(6):
            await pilot.pause()
        app.save_screenshot(out)


asyncio.run(main())
