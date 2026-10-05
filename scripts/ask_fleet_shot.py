"""Capture the FLEET ASK surfaces: the sidebar's marks and its total.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ask_fleet_shot.py OUT.svg [COLSxROWS] [MODE]

The fleet half of the queued-ask feature (design §5, amendment A2/A3/A5)
lives on surfaces that are NOT the ask list:

* **A mark on every session's row that holds an outstanding ask** — not only
  the current conversation's, which is what the operator's own report found
  missing: "if I wasn't at my desk I wouldn't have been able to see you had a
  question."
* **The fleet TOTAL on the sidebar's footer note**, only while it is > 0.

Both are fed from the cross-session index (``asks.store.read_index``), so this
capture seeds REAL index entries in the isolated capture root and then calls
the app's own painter — the frame is the product's, not a mock of it.

MODE is one of:

    marks    the sidebar with marks on TWO other sessions and the total in the
             footer (the default)
    none     the same sidebar with no outstanding ask anywhere: no mark, no
             note — absence is not emptiness (a zero is a statement the footer
             never made)
    fleet    the marks frame, then a press on the footer note: the ONE list
             opened on the fleet scope (the door, exercised end to end)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    refuse_flag_shaped_argument,
    save_capture,
)

isolate_capture()

from local_operator.asks import policy, store  # noqa: E402
from local_operator.paths import config_dir  # noqa: E402
from local_operator.resume import SessionRow  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_catalog import (  # noqa: E402
    CatalogEntry,
    SidebarSettings,
)
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

policy.NONBLOCKING_ASK = True

NOW = time.time()

#: ``(session id, name, minutes ago)`` — the sidebar's own rows. ``s-aida`` is
#: the CURRENT conversation, so the frame shows the mark landing on the two
#: rows that are NOT the one on screen, which is the whole point of the frame.
ROWS = (
    ("s-aida", "Stale-row migration rollout", 4),
    ("s-pergamon", "Enrichment backfill review", 9),
    ("s-tools", "omp phone portal deploy", 26),
)

CURRENT_ID = ROWS[0][0]

#: The index entries the marks come from: ``s-pergamon`` (two outstanding —
#: one open, one moved on) and ``s-tools`` (one open). ``s-aida`` is deliberately
#: ABSENT: the current session's mark is the app's live wire count, unioned in
#: by the painter, which is the union the frame has to show.
INDEX = {
    "s-pergamon": ("/Users/damian/pergamon", ["open", "timed_out"]),
    "s-tools": ("/Users/damian/tools", ["open"]),
}


def _ask(ask_id: str, status: str) -> dict:
    now = int(NOW * 1000)
    return {
        "ask_id": ask_id,
        "created_at": now - 120_000,
        "expires_at": now + 2_400_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": status,
        "delivered": False,
        "questions": [
            {
                "id": "q1",
                "question": "Which rollout should the stale-row migration take?",
                "options": [],
                "multi": False,
                "recommended": None,
                "secret": False,
                "persist": False,
            }
        ],
    }


def _seed_index() -> None:
    """Real index entries in the isolated capture root, sessions included.

    The session DIRECTORY matters: ``read_index`` sweeps an entry whose session
    directory is gone, because nothing could ever answer its asks — so a seed
    without one would be swept away by the very read this frame is about.
    """
    root = Path(config_dir())
    for session_id, (cwd, statuses) in INDEX.items():
        (root / "sessions" / session_id).mkdir(parents=True, exist_ok=True)
        store.write_entry(
            root,
            session_id,
            cwd=cwd,
            asks=[_ask(f"{session_id[-2:]}-{i}", status) for i, status in enumerate(statuses)],
        )


def _entries() -> list[CatalogEntry]:
    return [
        CatalogEntry(SessionRow(id=session_id, mtime=NOW - age * 60, name=name))
        for session_id, name, age in ROWS
    ]


class _CurrentSession(FakeSession):
    """The adopted session, with the one queued ask the current row must mark."""

    @property
    def session_id(self) -> str:
        return CURRENT_ID

    def respond_ask(self, ask_id, answers, *, by="unknown"):
        return {"ok": True}


def _current_row() -> dict:
    return _ask("a1", "open")


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: ask_fleet_shot.py OUT.svg [COLSxROWS] [marks|none|fleet]")
    refuse_flag_shaped_argument(sys.argv[1], what="OUT")
    out = sys.argv[1]
    size = (100, 30)
    mode = "marks"
    for arg in sys.argv[2:]:
        refuse_flag_shaped_argument(arg, what="argument")
        if "x" in arg:
            cols, rows = arg.split("x")
            size = (int(cols), int(rows))
        elif arg in {"marks", "none", "fleet"}:
            mode = arg
        else:
            raise SystemExit(f"unknown argument {arg!r}: expected a WxH size or marks|none|fleet")

    if mode != "none":
        _seed_index()

    session = _CurrentSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        app._session = session
        await pilot.pause()
        app._sidebar_settings = SidebarSettings(False, "left")
        for turn in range(1, 6):
            app._append_block(UserBlock(f"Turn {turn}: what should we do about the stale rows?"))
            prose = AssistantBlock()
            prose.update_text(
                "Answer: the audit log still has every row, so a backfill is possible."
            )
            app._append_block(prose)
        await pilot.pause()
        await pilot.press("f9")
        await pilot.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(_entries())
        sidebar.current_id = CURRENT_ID
        sidebar.cursor_id = ROWS[1][0]
        if mode != "none":
            # The CURRENT session's live count, exactly as the wire feeds it.
            from local_operator.tui.widgets.ask_queue import ask_rows

            app._sync_ask_surface(ask_rows([_current_row()]))
            # The index tally, through the app's own reader and painter — the
            # production path, not a hand-set map.
            marks, total = app._read_fleet_asks()
            app._ask_marks = dict(marks)
            app._ask_fleet_total = total
            app._paint_sidebar_asks()
        await pilot.pause()
        await pilot.pause()
        if mode == "fleet":
            app.action_open_fleet_asks()
            for _ in range(8):
                await pilot.pause()
        for _ in range(6):
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
