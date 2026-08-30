"""Reproduce the two ``/settings`` bugs reported against v0.43.0 (PR #387).

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/settings_repro.py

BUG 1 — opening ``/settings`` from the boot/splash screen compresses the page.
``Screen.boot`` is a whole separate layout (docked, centred, width-clamped input
card plus ``align-vertical: bottom`` on the transcript). Opening the settings
mode only ADDED ``Screen.settings``, so both layouts applied at once and the
page got the leftovers above a boot card that was still holding its rows.

BUG 2 — the arrow keys wrap from the bottom of the settings list back to the
top. The wheel path (``_scroll_rows``) already clamps, so the two gestures
disagreed on the same list.

The probe drives the REAL :class:`OperatorApp` (the only host that loads
``local_operator.tcss``) against a scratch config dir, so it measures the
shipped layout rather than a stylesheet-less test host.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SCRATCH = tempfile.mkdtemp(prefix="lo-settings-repro-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = _SCRATCH

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.settings_view import SettingsView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


_LOG: list[str] = []


def log(line: str = "") -> None:
    """Collected, not printed: Textual redirects stdout while the app runs."""
    _LOG.append(line)


async def main() -> None:
    size = (100, 30)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()

        # ---- BUG 1: opened from the BOOT state (no conversation yet) -------
        screen = app.screen
        dock = app.query_one("#input-dock")
        log(f"boot class before: {screen.has_class('boot')}")
        app._open_settings_view()
        await pilot.pause()
        await pilot.pause()
        view = app.query_one(SettingsView)
        log(f"boot class AFTER /settings: {screen.has_class('boot')}")
        log(f"settings class: {screen.has_class('settings')}")
        log(f"dock display: {dock.display} | dock height: {dock.size.height}")
        log(f"settings view height: {view.size.height} | screen height: {screen.size.height}")
        log(
            f"screen.size={tuple(screen.size)} virtual_size={tuple(screen.virtual_size)} "
            f"vscrollbar={screen.show_vertical_scrollbar}"
        )

        # ---- BUG 2: arrow movement at the ends of the list -----------------
        indices = view._selectable()
        log(f"\nselectable rows: {len(indices)} | selected: {view._selected}")
        view.action_jump(1)
        bottom = view._selected
        log(f"at bottom -> selected: {bottom} (last is {indices[-1]})")
        view.action_move(1)
        log(f"one MORE down -> selected: {view._selected} | wrapped to top? "
              f"{view._selected == indices[0]}")
        view.action_jump(1)
        view._scroll_rows(1)
        log(f"wheel at bottom -> selected: {view._selected} | clamped? "
              f"{view._selected == bottom}")
        view.action_jump(0)
        top = view._selected
        view.action_move(-1)
        log(f"at top, one MORE up -> selected: {view._selected} | wrapped to bottom? "
              f"{view._selected == indices[-1]}")
        view.action_jump(0)
        view._scroll_rows(-1)
        log(f"wheel at top -> selected: {view._selected} | clamped? {view._selected == top}")

        # ---- leaving restores the boot layout ------------------------------
        app._close_settings_view()
        await pilot.pause()
        await pilot.pause()
        welcome = app._welcome
        log(
            f"\nafter leaving: boot={screen.has_class('boot')} "
            f"settings={screen.has_class('settings')} "
            f"boot-card={screen.has_class('boot-card')} "
            f"welcome.display={welcome.display if welcome is not None else None} "
            f"welcome.height={welcome.size.height if welcome is not None else None} "
            f"dock.height={dock.size.height}"
        )

        # ---- do the OTHER full-page modes share bug 1? ---------------------
        # The subagent view's deferred refresh wants a real job manager, which
        # the fake session has not got; the probe measures LAYOUT, so the
        # refresh is stubbed out rather than the mode being skipped.
        app._refresh_subagent_view = lambda *_args, **_kw: None  # type: ignore[method-assign]
        for label, opener, closer in (
            ("org-chart", lambda: app._open_org_chart_view("lopdev"), app._close_org_chart_view),
            ("subagent", lambda: app._open_subagent_view("job-1"), app._close_subagent_view),
        ):
            try:
                opener()
            except Exception as error:  # a missing team/job is not what we measure
                log(f"{label}: could not open ({error})")
                continue
            await pilot.pause()
            await pilot.pause()
            log(
                f"{label}: boot={screen.has_class('boot')} "
                f"dock.height={dock.size.height} "
                f"view.height={_mode_height(app, label)} screen.height={screen.size.height}"
            )
            closer()
            await pilot.pause()

    for line in _LOG:
        print(line)


def _mode_height(app: OperatorApp, label: str) -> int:
    view = app._org_chart_view if label == "org-chart" else app._subagent_view
    return view.size.height if view is not None else -1


asyncio.run(main())
