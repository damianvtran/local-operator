"""Capture the ledger's shared name column against the rows a PAGE brings in.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ledger_paged_row_shot.py OUT.svg [COLSxROWS] [CASE]

``CASE`` selects which authoring order the paged row arrives by, because that is
the variable this script exists to show:

    hint  (default)  the row is authored the way the replay and paging paths
                     author one: ``set_fold_hint`` with the width it is about to
                     be given, content written while it is still parentless, then
                     appended. Paged in beside a row that is already on screen
                     and already on the shared column.
    fallback         the same row authored with no hint at all, on a terminal
                     whose transcript content region IS ``FALLBACK_WIDTH`` — the
                     width a parentless build falls back to, so the row is again
                     laid out at exactly the width it authored at.

Both cases put a short-named row (``bash``) on screen beside rows that share a
wider column, so the frame answers one question: is every row's summary in the
same cell? A frame where the paged row's summary starts six cells left of its
neighbours' is the tear — two `bash` rows at different offsets, which a pointer
crossing that row used to repair one row at a time.

The numbers are printed beside the frame (``--geometry`` on stderr) because the
stills show the symptom and the cells show the cause. No provider request, live
session or operator config is used: ``isolate_capture`` redirects HOME, config
and caches first, and every inherited ``CMUX_*`` variable is dropped before the
application is imported.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.glyphs import display_name  # noqa: E402
from local_operator.tui.widgets.tool_card import FALLBACK_WIDTH, ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import TranscriptView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: Rows already on screen, one of them naming the column. `list_variables` is the
#: long one, so the shared column sits well above the floor before the page lands.
SEEDED_ROWS = [
    ("i1", "list_variables", "list them", "4 variables"),
    ("i2", "bash", "git status --short", "3 files"),
]

#: The paged row: short name, so the page does NOT move the column — which is
#: exactly the case a repaint broadcast cannot rescue.
PAGED_ROW = ("p1", "bash", "rg -n 'column' src/ | head -9")

#: The terminal whose transcript content region is the fallback width.
FALLBACK_TERMINAL = (84, 24)


def _settled(view: TranscriptView, call_id: str, name: str, command: str, result: str) -> ToolCard:
    """A row appended the LIVE way: built detached, appended, then settled.

    The settle lands while the row is mounted, so its content is re-authored with
    a ledger to ask — which is why the seeded rows are the reference in the frame.
    """
    card = ToolCard(call_id, name, {"command": command})
    view.append_block(card)
    card.mark_done(result)
    return card


def _paged(name: str, command: str, destination: int | None) -> ToolCard:
    """A row authored the way a restored or paged row is authored.

    ``destination`` is the fold hint — the width the row is about to be given,
    which is what keeps it from folding twice — and ``None`` leaves the parentless
    build to fall back, which is the same trap on a terminal of that width.
    """
    card = ToolCard("p1", name, {"command": command})
    if destination is not None:
        card.set_fold_hint(destination)
    card.mark_done("paged in")
    return card


def _geometry(app: OperatorApp, view: TranscriptView, rows: list[ToolCard]) -> None:
    """Print each painted row's summary cell, straight off the compositor."""
    strips = list(app.screen._compositor.render_strips())
    print(
        f"  size={app.screen.size} content={view.scrollable_content_region.width} "
        f"col={view.tool_name_col} applied={view._name_col_applied}",
        file=sys.stderr,
    )
    columns = {}
    for card in rows:
        text = strips[card.region.y].text
        label = display_name(card.tool_name)
        at = text.find(getattr(card, "region_marker", ""))
        if at < 0:
            at = text.find(label)
            tail = text[at + len(label) :] if at >= 0 else ""
            at = at + len(label) + (len(tail) - len(tail.lstrip(" "))) if at >= 0 else -1
        columns.setdefault(at, []).append(card.tool_call_id)
        print(
            f"    {card.tool_call_id} summary@{at} built_width={card._built_width}",
            file=sys.stderr,
        )
    if len(columns) > 1:
        print(
            f"  !! the ledger is torn: rows sit at different summary cells: {columns}",
            file=sys.stderr,
        )


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    case = sys.argv[3] if len(sys.argv) > 3 else "hint"
    if case == "fallback":
        size = FALLBACK_TERMINAL

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        seeded = [_settled(view, *row) for row in SEEDED_ROWS]
        for card, row in zip(seeded, SEEDED_ROWS):
            card.region_marker = row[2]  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.pause()

        destination = view.scrollable_content_region.width if case == "hint" else None
        paged = _paged(PAGED_ROW[1], PAGED_ROW[2], destination)
        paged.region_marker = PAGED_ROW[2]  # type: ignore[attr-defined]
        if case == "fallback":
            # The fallback case only traps when the row's own width IS the
            # fallback; say so rather than capturing a frame that proves nothing.
            assert destination is None
            assert view.scrollable_content_region.width == FALLBACK_WIDTH, (
                "the fallback case needs a transcript content region of "
                f"{FALLBACK_WIDTH}; this terminal gives "
                f"{view.scrollable_content_region.width}"
            )
        view.insert_blocks(0, [paged])
        await pilot.pause()
        await pilot.pause()

        print(f"case={case}", file=sys.stderr)
        _geometry(app, view, [paged, *seeded])
        save_capture(app, out)


asyncio.run(main())
