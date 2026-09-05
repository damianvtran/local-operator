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
    mode = sys.argv[3] if len(sys.argv) > 3 else ""
    toggle = mode in ("toggle", "stale")
    # `stale`: the design-round state — limits typed in, then the master
    # switched OFF, so the sub-rows hold values the policy will not apply.
    if mode == "stale":
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        ConfigManager(config_dir()).update_config(
            {
                "session": {
                    "cleanup": {"enabled": False, "max_sessions": 200, "max_inactive_days": 30}
                }
            }
        )
        toggle = False
    if mode in ("armed", "child"):
        # SEEDED, never empty: an empty-store frame showed `would remove 0`
        # and proved nothing (design round 2, D7). 15 real 40-day transcripts
        # + 3 empties, `max_inactive_days: 7` -> the 10 most recent are
        # spared, so the honest number is 5.
        import os
        import time

        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir
        from local_operator.session.cleanup import mark_store

        sessions = config_dir() / "sessions"
        mark_store(sessions)
        old = time.time() - 40 * 86400
        for index in range(15):
            directory = sessions / f"s{index:02d}"
            directory.mkdir(exist_ok=True)
            (directory / "transcript.jsonl").write_text('{"type":"message"}\n')
            os.utime(directory / "transcript.jsonl", (old + index, old + index))
        for index in range(3):
            (sessions / f"e{index:02d}").mkdir(exist_ok=True)
        ConfigManager(config_dir()).update_config(
            {
                "session": {
                    "cleanup": (
                        {"enabled": True, "max_inactive_days": 7}
                        if mode == "armed"
                        else {"enabled": True, "max_sessions": 5}
                    )
                }
            }
        )
        toggle = False
    if mode == "expand":
        toggle = False

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
        if mode == "expand":
            view.action_activate()
            await pilot.pause()
            await pilot.pause()
        if mode == "stale":
            # Land on a sub-row so its "inert" clause is the one on screen.
            view._select_setting_row("session.cleanup.max_sessions")
            view._land()
            await pilot.pause()
        if mode == "child":
            # A child at a NON-default value under an armed master: the row
            # whose detail must show help AND `default: 0` at 110 cols (UX
            # round 2 U12 / design round 3 D12 — the key path sheds first).
            view._select_setting_row("session.cleanup.max_sessions")
            view._land()
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
