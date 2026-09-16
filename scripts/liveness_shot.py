"""Capture what the surfaces SAY about a session that has stopped reporting.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/liveness_shot.py \
        OUT.svg [COLSxROWS] [sidebar|info]

``info`` (the default) captures the ``/info`` screen; ``sidebar`` captures the
session sidebar with the quiet row's hover TOOLTIP showing, because that is the
only place the sidebar's words appear — a still frame without the hover shows
the glyph and nothing else, which is why a pair of those frames cannot evidence
a wording change.

**The scenario is a real one, built through the real code.** Two discovery
records are written into an isolated config root: one whose heartbeat stopped
243 s ago on a pid that is still ALIVE (the reported shape — an in-process
runtime busy in a long turn, whose own event loop writes the beat), and one
working session beside it as the control. Nothing is stubbed: ``registry.scan``
classifies them, and the age the surfaces print is the one the classifier
derived. The frames therefore show what a machine in that state renders, not a
hand-set field.

**What this cannot prove.** Both surfaces say the owner has not REPORTED; the
cause of the silence is not observable from here (the beat is authored by the
runtime's own event loop, so a long turn and a deadlock look identical), and
this script makes no claim about either. It is a capture of the wording.

The frame is deterministic for a given tree: the catalog is fixed and the age
is fixed at capture time, so a before/after pair differs only where the change
does. Run it once per tree — the "before" frame comes from the same script on
the pre-change build.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.resume import SessionRow  # noqa: E402
from local_operator.session.catalog import CatalogEntry, decorate_rows  # noqa: E402
from local_operator.session.runtime import registry  # noqa: E402
from local_operator.session.runtime.types import SessionRecord  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_catalog import SidebarSettings  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The reported shape: `HEARTBEAT_TIMEOUT_S` is 45 s, so this is well past it
#: and past the 205.8 s the live machine actually measured.
QUIET_AGE_S = 243.0
QUIET_ID = "quietowner01"
WORKING_ID = "working00001"


def _write(pid: int, session_id: str, name: str, age_s: float, *, busy: bool) -> None:
    """One discovery record, stamped as its own runtime would have left it.

    Written directly rather than through ``RecordPublisher``: ``publish``
    stamps a fresh heartbeat by design, and a quiet owner is exactly one whose
    beat stopped arriving.
    """
    record = SessionRecord(
        pid=pid,
        kind="daemon",
        session_id=session_id,
        conversation_name=name,
        cwd="/tmp",
        model_label="anthropic/claude-opus-5",
        control_port=12345,
        control_key="k" * 64,
        started=True,
        busy=busy,
    )
    record.heartbeat_at = time.time() - age_s
    directory = registry.run_dir()
    path = directory / f"{pid}.json"
    path.write_text(json.dumps(record.to_json()))
    os.chmod(path, 0o600)


async def main() -> None:
    out = Path(sys.argv[1])
    size = (110, 30)
    surface = "info"
    for argument in sys.argv[2:]:
        if "x" in argument:
            cols, rows = argument.split("x")
            size = (int(cols), int(rows))
        else:
            surface = argument

    # Real, live pids: the point of the scenario is a stale BEAT on a process
    # that is still there, and a synthetic pid would classify as `stale` and be
    # reaped by the scan. Neither pid may be this SCRIPT's own — the app's own
    # runtime publishes a record at `<pid>.json` for its session, and a capture
    # that claimed that pid would be overwritten by it mid-run (observed: the
    # quiet row vanished from the frame). The quiet owner is therefore a real
    # child process that does nothing but exist.
    idle = subprocess.Popen(["sleep", "300"])
    try:
        _write(idle.pid, QUIET_ID, "Quiet owner (stale beat)", QUIET_AGE_S, busy=True)
        _write(os.getppid(), WORKING_ID, "Working session", 2.0, busy=True)

        now = time.time()
        rows = decorate_rows(
            registry.config_dir(),
            [
                SessionRow(
                    QUIET_ID, now - 300.0, "Quiet owner (stale beat)", created_at=now - 300.0
                ),
                SessionRow(WORKING_ID, now - 60.0, "Working session", created_at=now - 60.0),
            ],
        )
        # ``getattr`` for the age: this script is run against the PRE-change
        # build too (that is where the "before" frame comes from), and on that
        # tree ``SessionRow`` has no such field.
        print(
            "decorated:",
            [(row.id, row.live_state, getattr(row, "heartbeat_age_s", None)) for row in rows],
        )
        entries = [CatalogEntry(row) for row in rows]
        print("statuses:", [(entry.id, entry.status_code, entry.status) for entry in entries])

        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size, tooltips=True) as pilot:
            await pilot.pause()

            if surface == "sidebar":
                app._sidebar_settings = SidebarSettings(True, "right")
                await pilot.press("ctrl+b")
                await pilot.pause()
                # Retire BOTH catalog-refresh paths so the real catalog cannot
                # land on top of these rows mid-capture (the sidebar tests' own
                # recipe).
                if app._sidebar_timer is not None:
                    app._sidebar_timer.pause()
                app._sidebar_refresh_generation += 1
                app._session_sidebar.set_entries(entries)
                await pilot.pause()
                # The words live in the hover tooltip and nowhere else.
                sidebar = app._session_sidebar
                hovered = next(
                    y
                    for y in range(1, sidebar.size.height)
                    if (entry := sidebar._entry_at(y)) is not None and entry.row.id == QUIET_ID
                )
                await pilot.hover("#session-sidebar", offset=(8, hovered))
                await asyncio.sleep(float(app.TOOLTIP_DELAY) + 0.2)
                await pilot.pause()
                sidebar._show_tooltip_now()
                await pilot.pause()
            else:
                from local_operator.tui.widgets.editor import Editor

                editor = app.query_one(Editor)
                editor.text = "/info"
                await pilot.pause()
                if editor._picker.is_open():
                    await pilot.press("escape")
                    await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                # The sections this change touches are below the fold on a
                # 30-row terminal, and the screen's own binding is the real way
                # to reach them. The count is an argument because the offset is
                # a function of the viewport WIDTH (the body wraps), so a
                # capture at 70 columns needs a different number of pages than
                # one at 110.
                for _ in range(int(os.environ.get("LO_LIVENESS_SHOT_PAGES", "2"))):
                    await pilot.press("pagedown")
                    await pilot.pause()

            save_capture(app, str(out))
            print("screenshot:", out)
    finally:
        idle.terminate()
        idle.wait(timeout=10)


asyncio.run(main())
