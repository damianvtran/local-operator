"""Capture the ``sessions`` tool's transcript rows, nerd and plain.

Run from the worktree root, ONCE per env so the icon gate is resolved against
that env (the gate reads ``os.environ`` and the settings cache at row-build
time):

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/sessions_tool_shot.py OUT.svg {nerd|plain} [settled] [clone] [COLSxROWS]

- ``nerd``  seeds a ghostty marker so autodetect draws the Font Awesome table.
- ``plain`` seeds Apple_Terminal and strips every bundling marker, so the frame
  is the WGL4 fallback a bare terminal gets.
- ``settled`` marks every card done, so the rows paint the settled ink — the
  ``tool.row.name_meta`` category colour a running-only frame cannot show.
- ``clone`` repaints ONLY the sessions Nerd glyph as the ALTERNATE candidate
  (``nf-fa-clone``) on a tree that already carries the change; the design round
  picks between the two candidates on rendered frames, and the guard means a
  base tree cannot be asked for a frame its source never produced.

Both modes render the SAME rows, so two frames differ only in the icon column
and the summaries the tree's own ``_summary_from_args`` produces — which is the
whole point of the pair: this script run on the base revision and on the change
is the before/after evidence. The script prints each card's own painted row
(``_build_row``) and the resolved ``tool_icon("sessions")`` beside the stills,
because the frame shows the symptom and the numbers show the cause.

Card args are the shapes the tool's own schema carries (spawn with a name and
without; the four addressed ops). ``peek`` is painted ahead of its merge: its
summary branch ships in this change so PR B's op lands into a working row, and
the frame records what that branch draws.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot must not rename the operator's real workspace through inherited CMUX IDs.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A stable cwd for the capture: the status band paints ``os.getcwd()``, and a
# before frame taken from another checkout would differ in that band and
# nowhere else — a pair that differs in an unrelated row cannot be read as
# "the rows changed".
_CAPTURE_CWD = "/tmp/lo-sessions-tool-shot"
os.makedirs(_CAPTURE_CWD, exist_ok=True)
os.chdir(_CAPTURE_CWD)

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    save_capture,
    settle_status_line,
)

isolate_capture()


def _seed_env(mode: str) -> None:
    """Force the process env into the terminal the gate should detect.

    Done BEFORE importing the app so the first glyph lookup sees it. The gate
    has no interactive probe; it reads exactly these markers.
    """
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
        os.environ["GHOSTTY_BIN"] = "/opt/ghostty/bin"
    elif mode == "plain":
        os.environ["TERM_PROGRAM"] = "Apple_Terminal"
    else:
        raise SystemExit(f"unknown mode {mode!r}; want 'nerd' or 'plain'")


def _warn_if_no_nerd_font(mode: str) -> None:
    """One line when the nerd frame cannot show its own glyphs.

    The frame is rasterized by librsvg against the host's fontconfig; a host
    with no Nerd face paints every Font Awesome codepoint as tofu, so the
    still cannot evidence the glyph pick even though the row, summaries and
    spacing in it are real (design round 1, TUI D2). The check is against the
    font LIST rather than the capture profile's resolved face: the PUA glyph
    would be drawn by whatever fallback owns the codepoint, so what matters is
    whether any installed face carries it — and in practice that is a family
    whose NAME says "Nerd" (a bare "Symbols" match is not enough: macOS's own
    ``.CJK Symbols Fallback`` families match it and carry no FA glyphs).
    """
    if mode != "nerd":
        return
    lister = shutil.which("fc-list")
    installed = ""
    if lister is not None:
        try:
            installed = subprocess.run(
                [lister], capture_output=True, text=True, timeout=10, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            installed = ""
    if "nerd" in installed.casefold():
        return
    print(
        "warning: no Nerd font resolves on this host (fc-list): every nerd "
        "glyph in this frame will render as a replacement box — judge the "
        "glyph on a specimen, not on this frame (design round 1, TUI D2).",
        file=sys.stderr,
    )


#: (tool_name, args) rows. The two neighbours above the sessions block are the
#: collision witnesses: `task` owns nf-fa-users, `send` owns nf-fa-paper_plane,
#: and a sessions glyph that leaned on either would read as a duplicate in the
#: very frame the design round judges.
_ROWS: list[tuple[str, dict[str, object]]] = [
    ("task", {"prompt": "run the review gate"}),
    ("send", {"target": "release cutter", "message": "gates are green"}),
    (
        "sessions",
        {
            "op": "spawn",
            "name": "night-audit",
            "prompt": "audit the release window and report",
            "visibility": "workstream",
        },
    ),
    (
        "sessions",
        {"op": "spawn", "prompt": "fix the flaky shard in isolation", "visibility": "ephemeral"},
    ),
    ("sessions", {"op": "stop", "target": "release-crew"}),
    ("sessions", {"op": "resume", "session": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"}),
    ("sessions", {"op": "peek", "target": "release-crew", "steps": 12}),
    ("sessions", {"op": "list", "include_stored": True, "query": "flaky shard"}),
]


async def main() -> None:
    out = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "nerd"
    _seed_env(mode)
    _warn_if_no_nerd_font(mode)

    size = (100, 24)
    settled = False
    variant = ""
    for token in sys.argv[3:]:
        if token == "settled":
            settled = True
        elif token == "clone":
            variant = "clone"
        elif "x" in token:
            cols, rows = token.split("x")
            size = (int(cols), int(rows))
        else:
            raise SystemExit(f"unknown option {token!r}; want 'settled', 'clone' or COLSxROWS")

    # Imported AFTER _seed_env so module-level env reads (if any) see our env.
    from local_operator.tui import glyphs as glyph_mod  # noqa: E402
    from local_operator.tui import settings as settings_mod  # noqa: E402
    from local_operator.tui.app import OperatorApp  # noqa: E402
    from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
    from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
    from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
    from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

    settings_mod.settings_reload()

    # The alternate candidate, ONLY on a tree that carries the entry: patching
    # the safe table on the base revision would paint a frame the source never
    # produced, which is exactly what a before/after pair must not do.
    if variant == "clone":
        if "sessions" not in glyph_mod._SAFE_NERD_ICONS:
            raise SystemExit("clone variant asked for on a tree without the sessions glyph")
        glyph_mod._SAFE_NERD_ICONS["sessions"] = "\uf24d"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("Audit the release window and keep it visible."))
        prose = AssistantBlock()
        prose.update_text("Working through the tool ledger below.")
        app._append_block(prose)
        cards = [ToolCard("t", name, args) for name, args in _ROWS]
        for card in cards:
            app._append_block(card)
        if settled:
            for card in cards:
                card.mark_done("ok")
        for _ in range(6):
            await pilot.pause()
        await settle_status_line(pilot, app)

        icon = glyph_mod.tool_icon("sessions")
        print(
            f"mode={mode} variant={variant or 'primary'} "
            f"sessions_icon={icon!r} U+{ord(icon):04X}",
            file=sys.stderr,
        )
        for card in cards:
            width = card.size.width if card.size.width else size[0]
            print(f"  [{card.tool_name}] {card._build_row(width).plain!r}", file=sys.stderr)
        print(
            f"  screen={app.screen.size} virtual={app.screen.virtual_size} "
            f"vscroll={app.screen.show_vertical_scrollbar}",
            file=sys.stderr,
        )
        save_capture(app, out)


asyncio.run(main())
