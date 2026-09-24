"""Design-review round-2 frames for PR #1436 (agent workstream visibility)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

WT = Path(os.environ["WT"])
sys.path.insert(0, str(WT))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.paths import config_dir  # noqa: E402
from local_operator.resume import (  # noqa: E402
    ORIGIN_AGENT_WORKSTREAM,
    mark_session_origin,
    write_session_title,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.session_sidebar import SessionSidebar  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

WORKSTREAM = "bbbb00000001"
ALL_NULL = "cdcd00000001"
OPERATOR = "dddd00000001"
TRUNCATED = "eeee00000001"
ROLE = os.environ.get("ROLE", "coder")
HOVER = os.environ.get("HOVER", WORKSTREAM)

SESSIONS = [
    (OPERATOR, "Migration checklist for the invoices table", None, 10),
    (WORKSTREAM, "Fan-out audit of the ingest pipeline",
     {"agent": ROLE, "label": "1436 r2 probe", "session": "aaaa00000001"}, 1),
    (ALL_NULL, "Harden lop secret get output", {}, 2),
]


def _entry(role: str, text: str, seconds_ago: float) -> str:
    """One transcript line in the shape ``preview.verbose_entries`` reads.

    ``payload.kind == "message"`` and ``content`` as a LIST of blocks are both
    load-bearing (``condense_entries`` predicates on the first, ``_block_text``
    iterates the second), so a hand-written line missing either draws
    "(no prose in this transcript)" instead of the turn under test.
    """
    return json.dumps(
        {
            "type": "message",
            "ts": time.time() - seconds_ago,
            "payload": {"kind": "message", "role": role, "content": [{"text": text}]},
        }
    ) + "\n"


def _seed(root: Path) -> None:
    now = time.time()
    for session_id, title, opener, minutes in SESSIONS:
        directory = root / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        prompt = f"{title}. Let's pick this up."
        (directory / "transcript.jsonl").write_text(
            _entry("user", prompt, minutes * 60.0 + 2)
            + _entry("assistant", "On it. Reading the pipeline config first.", minutes * 60.0 + 1),
            encoding="utf-8",
        )
        (directory / "created_at.json").write_text(str(now - minutes * 60), encoding="utf-8")
        write_session_title(directory, title, user_set=True, past_names=[])
        if opener is not None:
            mark_session_origin(directory, ORIGIN_AGENT_WORKSTREAM, opened_by=opener)
        stamp = now - minutes * 60
        os.utime(directory / "transcript.jsonl", (stamp, stamp))
    if os.environ.get("EXTRA") == "1":
        directory = root / "sessions" / TRUNCATED
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "transcript.jsonl").write_text(
            _entry("user", "Sweep the retention ledger. Let's pick this up.", 1.0),
            encoding="utf-8",
        )
        write_session_title(directory, "Retention sweep for the analytics ledger", user_set=True, past_names=[])
        (directory / "origin.json").write_text('{"origin": "agent-workstr', encoding="utf-8")
        stamp = now - 180
        os.utime(directory / "transcript.jsonl", (stamp, stamp))


async def _sidebar_frame(app, pilot, out):
    from textual.widgets import Tooltip

    await pilot.press("ctrl+b")
    sidebar = app.query_one(SessionSidebar)
    row_y = None
    for _ in range(80):
        await pilot.pause()
        for y in range(1, sidebar.size.height):
            entry = sidebar._entry_at(y)
            if entry is not None and entry.id == HOVER:
                row_y = y
                break
        if row_y is not None:
            break
    print("rows on screen:", [e.id for y in range(0, sidebar.size.height) if (e := sidebar._entry_at(y)) is not None])
    print("hovered row y:", row_y, "id:", HOVER)
    assert row_y is not None, "the hovered row never appeared in the sidebar"
    # Retire the catalog poll so the real catalog cannot land mid-capture.
    if app._sidebar_timer is not None:
        app._sidebar_timer.pause()
    app._sidebar_refresh_generation += 1
    await pilot.pause()
    before = _painted(sidebar)
    await pilot.hover("#session-sidebar", offset=(8, row_y))
    await asyncio.sleep(float(app.TOOLTIP_DELAY) + 0.2)
    await pilot.pause()
    sidebar._show_tooltip_now()
    await pilot.pause()
    tooltip = app.screen.get_child_by_type(Tooltip)
    print("tooltip display:", tooltip.display)
    print("tooltip render:", repr(str(tooltip.render())))
    print("tooltip region:", tooltip.region, "screen:", app.screen.region)
    after = _painted(sidebar)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    print("row lines the hover changed:", differing, "of", len(before))
    print("screen virtual:", app.screen.virtual_size, "size:", app.screen.size, "scrollbar:", app.screen.show_vertical_scrollbar)
    save_capture(app, str(out))


def _painted(widget):
    from textual.geometry import Region

    return [
        "".join(segment.text for segment in line)
        for line in widget.render_lines(Region(0, 0, widget.size.width, widget.size.height))
    ]


async def _preview_frame(app, pilot, out, row):
    editor = app.query_one(Editor)
    editor.text = "/resume"
    editor.cursor_location = (0, len("/resume"))
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    for _ in range(60):
        await pilot.pause()
        if app.screen.__class__.__name__ == "SessionPickerScreen":
            break
    await pilot.pause()
    await pilot.press("down")
    await pilot.pause()
    await pilot.press("up")
    await pilot.pause()
    for _ in range(row):
        await pilot.press("down")
        await pilot.pause()
    screen = app.screen
    rows = getattr(screen, "_rows", None)
    cursor = getattr(screen, "_cursor_index", getattr(screen, "_cursor", None))
    try:
        print("picked row id:", rows[cursor].id if rows is not None and cursor is not None else "?")
    except Exception as exc:
        print("picked row: ?", exc)
    for line in screen.render_lines_for_test():
        print("RES |", line)
    for line in screen.render_preview_for_test():
        print("PRE |", line)
    print("screen virtual:", app.screen.virtual_size, "size:", app.screen.size, "scrollbar:", app.screen.show_vertical_scrollbar)
    save_capture(app, str(out))


async def main():
    out = Path(sys.argv[1])
    mode = sys.argv[2]
    size = (120, 32)
    if len(sys.argv) > 3 and "x" in sys.argv[3]:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))
    row = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    _seed(Path(config_dir()))

    # The splash's update probe is a NETWORK check (``update.check_latest``) whose
    # notice lands in one capture and not the next — measured: ``! latest is
    # v0.62.19`` was painted in one sidebar frame and absent in the identical one
    # after it, shifting every splash row below it by one. Unrelated to the
    # surface under test, so it is retired the way ``liveness_shot`` retires the
    # catalog poll, rather than left as noise inside a before/after pair.
    OperatorApp._check_for_update = lambda self: None

    async def resume_factory(_session_id):
        return FakeSession()

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=size, tooltips=True) as pilot:
        await pilot.pause()
        if mode == "sidebar":
            await _sidebar_frame(app, pilot, out)
        else:
            await _preview_frame(app, pilot, out, row)
    print("wrote", out)


asyncio.run(main())
