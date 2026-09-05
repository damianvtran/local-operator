"""Render the ``/settings`` page with the cursor on the Session section.

    .venv/bin/python scripts/settings_session_shot.py out.svg [100x40] [toggle]

Uses the real ``OperatorApp`` (so ``local_operator.tcss`` is applied) over a
throwaway HOME/config dir that ``isolate_capture`` provisions before any app
import, so the developer's real config is never read. ``toggle`` presses the
master switch on before the shot, so the dependent rows can be seen in both
states.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()


async def main() -> None:
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    out = sys.argv[1]
    size = tuple(int(part) for part in (sys.argv[2] if len(sys.argv) > 2 else "100x40").split("x"))
    toggle = len(sys.argv) > 3 and sys.argv[3] == "toggle"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:  # type: ignore[arg-type]
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app._settings_view
        assert view is not None
        # Land on the LAST cleanup row first so the whole block scrolls into
        # view, then back up to the master switch.
        view._select_setting_row("session.cleanup.remove_empty")
        view._scroll_to_selection(immediate=True)
        await pilot.pause()
        view._select_setting_row("session.cleanup.enabled")
        view._land()
        await pilot.pause()
        if toggle:
            view.action_toggle_bool()
            await pilot.pause()
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
