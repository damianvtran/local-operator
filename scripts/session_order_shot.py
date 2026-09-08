"""Capture activity refreshes through the real app's sidebar, without live sessions."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import scripts.probe_isolation  # noqa: F401 — isolate before app imports
from local_operator.harness.types import Message
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_catalog import (
    CatalogEntry,
    SidebarSettings,
    cached_session_rows,
)
from local_operator.tui.widgets.transcript import UserBlock
from scripts.visual_capture import save_capture
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_session_sidebar import _quiesce_sidebar_refresh


async def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    states = {}
    cfg = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    for index, (sid, title, state, kind) in enumerate(
        [
            ("done-new", "Completed newest", "idle", "complete"),
            ("done-old", "Completed older", "idle", "complete"),
            ("error", "Failed review", "idle", "error"),
            ("interrupt", "Interrupted research", "idle", "interrupted"),
            ("busy-a", "Working Alpha", "busy", ""),
            ("busy-b", "Working Beta", "busy", ""),
            ("busy-c", "Working Gamma", "busy", ""),
        ]
    ):
        directory = cfg / "sessions" / sid
        transcript = Transcript(directory)
        await transcript.append_message(Message.user(title))
        (directory / "created_at.json").write_text(str(1000 - index))
        os.utime(transcript.path, (1000 - index, 1000 - index))
        states[sid] = (state, kind)
    app = OperatorApp(lambda: _factory(FakeSession()))
    log = []
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("Keep Working Beta selected while sessions publish activity."))
        app._sidebar_settings = SidebarSettings(False, "left")
        await pilot.press("ctrl+b")
        _quiesce_sidebar_refresh(app)
        sidebar = app._session_sidebar
        sidebar.current_id = "busy-b"
        sidebar.cursor_id = "busy-b"
        for tick in range(4):
            if tick:
                target = ("busy-c", "busy-a", "busy-b")[tick - 1]
                os.utime(cfg / "sessions" / target / "transcript.jsonl", (2000 + tick, 2000 + tick))
            entries = [
                CatalogEntry(
                    row._replace(live_state=states[row.id][0]),
                    unseen=bool(states[row.id][1]),
                    completion_kind=states[row.id][1],
                )
                for row in cached_session_rows(cfg)
                if row.id in states
            ]
            sidebar.set_entries(entries)
            if sidebar._timer is not None:
                sidebar._timer.stop()
                sidebar._timer = None
            sidebar._frame = 2
            await pilot.pause()
            save_capture(app, out / f"tui-{tick}.svg")
            log.append(
                {
                    "tick": tick,
                    "order": [e.id for e in sidebar.entries],
                    "cursor": sidebar.cursor_id,
                    "current": sidebar.current_id,
                }
            )
        (out / "tui-order.json").write_text(json.dumps(log, indent=2) + "\n")
        print(json.dumps(log))


if __name__ == "__main__":
    asyncio.run(main())
