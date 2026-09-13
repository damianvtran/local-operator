"""Capture the tool ledger's shared name column against every row that shares it.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ledger_names_shot.py OUT.svg [COLSxROWS] [CASE]

``CASE`` selects the event that MOVES the column, because the defect was that
one event moved it for some rows and left the others where they were:

    append  (default)  a longer-named row appended while older rows are already
                       painted. Reached the way a live session reaches it: the
                       derived column is invalidated first by a block paged in
                       from older history, which is the state the growth path
                       read as "nothing to do".
    reveal             an older page carrying a longer MCP tool name paged in
                       above rows already on screen.
    promote            a call that STARTS: a row dictated under its announce name
                       is promoted to `running` with the execution's name, which
                       is the moment its name starts counting towards the spine.

The fixture is the SAME in both cases, so the two frames differ in the event and
not in the data. Every row is a settled receipt, and the names are chosen so the
column has to grow: `bash` and `read` sit at the floor, `list_variables` and
`create_initiative` (from `mcp__linear_create_initiative`) do not.

The property to read off the frame is that every row's summary starts in the
SAME cell — that is the shared spine. A frame where one row's text starts two or
more cells right of its neighbours' is the tear the pointer used to repair one
row at a time.

Capture the BEFORE frame from a checkout that predates the fix with this same
script and the same fixture; no provider request, live session or operator
config is used (``isolate_capture`` redirects HOME, config and caches first).
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
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import TranscriptView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The rows already painted when the column moves. Short names, so the ledger
#: sits at the floor of the column and the growth is entirely the newcomer's.
SEEDED_ROWS = [
    ("t1", "bash", "rg -n 'name_col' local_operator/ | head -9", "9 matches"),
    ("t2", "read", "local_operator/tui/widgets/transcript.py", "4308 lines"),
]

#: An older LOAD-BEARING row paged in first. Its own name is short — the point
#: is not that it widens the column but that paging history in invalidates the
#: derived width, which is the state the append lands in.
OLDER_SHORT_ROW = ("old1", "bash", "git log --oneline -3", "3 commits")

#: An older row whose name DOES widen the column: the `reveal` case.
OLDER_WIDE_ROW = ("old2", "mcp__linear_create_initiative", "release window", "created")

#: The newcomer. `list_variables` is longer than every name above it.
APPENDED_ROW = ("t3", "list_variables", "list_variables {}", "4 variables")

#: The promoted row of the `promote` case: dictated as `list_variables` while it
#: was composing, then started. Its summary is the one the execution carries.
PROMOTED_ROW = ("live", "list_variables", "list them")


def _tool(app: OperatorApp, rows: list[tuple[str, str, str, str]]) -> list[ToolCard]:
    """Append FINISHED tool rows, so the ledger is settled ink rather than the
    live-turn accent. ``mark_done`` runs after the mount because a card settles
    in place (the pattern ``peer_message_shot.py`` uses)."""
    cards = []
    for call_id, name, command, result in rows:
        card = ToolCard(call_id, name, {"command": command})
        app._append_block(card)
        card.mark_done(result)
        cards.append(card)
    return cards


#: call id -> the text its summary starts with, so the frame's own geometry can
#: be read back: the cell the summary starts in IS the shared name column's far
#: edge, and every row has to agree on it.
SUMMARIES = {
    SEEDED_ROWS[0][0]: SEEDED_ROWS[0][2],
    SEEDED_ROWS[1][0]: SEEDED_ROWS[1][2],
    OLDER_SHORT_ROW[0]: OLDER_SHORT_ROW[2],
    OLDER_WIDE_ROW[0]: OLDER_WIDE_ROW[2],
    APPENDED_ROW[0]: APPENDED_ROW[2],
    PROMOTED_ROW[0]: PROMOTED_ROW[2],
}


def _summary_column(row: str, label: str, marker: str) -> str:
    """Where this row's summary starts, measured off the frame's own text.

    The marker is the direct answer while the frame is wide enough to paint it.
    A narrower frame clips the summary — and that frame is exactly what this
    script exists to document, so raising there wrote no frame at all. The
    fallback measures the same number off the NAME field instead: the label
    ends and the padding that follows belongs to the name column, so the first
    cell of the summary is the first non-space after the label. A row narrow
    enough to clip the label as well is reported as exactly that, rather than as
    a number nothing backs.
    """
    at = row.find(marker)
    if at >= 0:
        return f"summary@{at}"
    name_at = row.find(label)
    if name_at < 0:
        return f"name clipped at this width (neither {marker!r} nor {label!r} painted)"
    tail = row[name_at + len(label) :]
    return f"summary@{name_at + len(label) + len(tail) - len(tail.lstrip(' '))}"


def _report_geometry(app: OperatorApp, view: TranscriptView) -> None:
    """Print the summary column each painted row starts in, straight off the
    compositor. The stills show the tear; these numbers say which rows were in
    which column, and the two together are the evidence."""
    strips = list(app.screen._compositor.render_strips())
    for block in view.blocks():
        call_id = getattr(block, "tool_call_id", "")
        marker = SUMMARIES.get(call_id)
        if marker is None:
            continue
        label = display_name(getattr(block, "tool_name", ""))
        column = _summary_column(strips[block.region.y].text, label, marker)
        print(f"  {call_id:>6} {column}", file=sys.stderr)


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    case = sys.argv[3] if len(sys.argv) > 3 else "append"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        painted = _tool(app, SEEDED_ROWS)
        await pilot.pause()

        if case == "reveal":
            blocks = [
                ToolCard("old1", OLDER_SHORT_ROW[1], {"command": OLDER_SHORT_ROW[2]}),
                ToolCard("old2", OLDER_WIDE_ROW[1], {"command": OLDER_WIDE_ROW[2]}),
            ]
            for block, result in zip(blocks, (OLDER_SHORT_ROW[3], OLDER_WIDE_ROW[3])):
                block.mark_done(result)
            view.insert_blocks(0, blocks)
        elif case == "promote":
            # Dictate first, then start: the two events a real turn produces, in
            # the order that moves the column on the SECOND one.
            call_id, name, summary = PROMOTED_ROW
            live = ToolCard(call_id, "bash", {})
            view.append_block(live)
            live.set_composing(12, name)
            await pilot.pause()
            live.begin_running(name, {"command": summary}, None)
        else:
            # Page one older block in, then append the longer-named row: the
            # column is invalidated by the page and moved by the append.
            older_block = ToolCard(
                OLDER_SHORT_ROW[0], OLDER_SHORT_ROW[1], {"command": OLDER_SHORT_ROW[2]}
            )
            older_block.mark_done(OLDER_SHORT_ROW[3])
            view.insert_blocks(0, [older_block])
            painted.extend(_tool(app, [APPENDED_ROW]))

        # Two settled frames: a first paint that differs from this one is a
        # reflow the user sees as motion (AGENTS.md, "Animation and multi-frame
        # changes"). The pointer never enters the frame in either case — the
        # rows have to be right without it.
        await pilot.pause()
        await pilot.pause()
        screen = app.screen
        print(
            f"case={case} size={screen.size} virtual={screen.virtual_size} "
            f"vscroll={screen.show_vertical_scrollbar} col={view.tool_name_col}",
            file=sys.stderr,
        )
        _report_geometry(app, view)
        save_capture(app, out)


asyncio.run(main())
