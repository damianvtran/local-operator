"""§5.6 in the TUI: the agent-messages settings section is dynamically absent.

Driven through the REAL :class:`OperatorApp` (the only host that loads the
stylesheet), because this file is about what the page RENDERS, not about the
row builder in isolation. The gate itself is the process-local availability
record — see ``tests/unit/server/test_server_settings_visibility.py`` for the
desktop drainer and ``local_operator/team_bridge/availability.py`` for the rule.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator.team_bridge import availability
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.settings_view import SettingsView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from local_operator.tui.settings import settings_reload

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    settings_reload()
    availability.clear()
    yield tmp_path
    availability.clear()


def _section_names(view: SettingsView) -> list[str]:
    view._repaint()
    return [
        row.section.name for row in view._rows if row.kind == "header" and row.section is not None
    ]


@pytest.mark.asyncio
async def test_the_agent_messages_section_appears_only_after_an_eligible_read() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)

        # Never observed in this process: ABSENT — no header, no rows.
        names = _section_names(view)
        assert "team_bridge" not in names
        keys = [
            row.setting.key
            for row in view._rows
            if row.setting is not None and row.setting.key.startswith("team_bridge.")
        ]
        assert keys == []

        # A boundary read records eligibility: the section comes back on the
        # next paint (the row cache keys on the visibility, so it rebuilds).
        availability.record(view._manager.config_dir, visible=True, state="ready", checked_at=1.0)
        names = _section_names(view)
        assert "team_bridge" in names
        keys = [row.setting.key for row in view._rows if row.setting is not None]
        assert "team_bridge.enabled" in keys
        assert "team_bridge.session" in keys

        # And a later read that observes dark/ineligible hides it again.
        availability.record(
            view._manager.config_dir, visible=False, state="unavailable", checked_at=2.0
        )
        assert "team_bridge" not in _section_names(view)
