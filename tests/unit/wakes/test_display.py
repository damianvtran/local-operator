"""Local clock rendering never changes a wake's absolute scheduled instant."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.settings_io import BY_KEY, write_setting
from local_operator.tui.settings import settings_reload
from local_operator.wakes.display import format_wake_time


@pytest.fixture(autouse=True)
def local_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the process timezone requires tzset")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    for key in list(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    settings_reload()
    yield
    monkeypatch.undo()
    time.tzset()
    settings_reload()


def ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


@pytest.mark.parametrize(
    ("due", "expected"),
    [
        ("2026-09-09T02:52:00+00:00", "7:52 PM PDT"),
        ("2026-09-08T07:00:00+00:00", "12:00 AM PDT"),
        ("2026-09-08T19:00:00+00:00", "12:00 PM PDT"),
        ("2026-09-09T07:00:00+00:00", "Sep 09 12:00 AM PDT"),
        ("2027-01-01T08:00:00+00:00", "Jan 01 2027 12:00 AM PST"),
        ("2026-11-01T08:30:00+00:00", "Nov 01 1:30 AM PDT"),
        ("2026-11-01T09:30:00+00:00", "Nov 01 1:30 AM PST"),
    ],
)
def test_local_clock_dates_and_dst(due: str, expected: str) -> None:
    # UTC is already Sep 09, but the viewer's local date is still Sep 08.
    now = datetime(2026, 9, 9, 1, tzinfo=UTC)
    assert format_wake_time(ms(due), now=now) == expected


@pytest.mark.parametrize("value,expected", [("24h", "00:00 PDT"), ("12h", "12:00 AM PDT")])
def test_persisted_preference(value: str, expected: str, tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config")
    write_setting(manager, BY_KEY["display.time_format"], value)
    settings_reload()
    assert (
        format_wake_time(ms("2026-09-08T07:00:00+00:00"), now=datetime(2026, 9, 8, 20, tzinfo=UTC))
        == expected
    )
    assert ConfigManager(tmp_path / "config").get_config_value("display.time_format") == value


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [100, 60])
async def test_settings_choice_persists_and_refreshes_panel(tmp_path: Path, columns: int) -> None:
    from types import SimpleNamespace

    from local_operator.harness.wake import WakeSchedule
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.settings_view import SettingsView
    from tests.unit.tui.test_app_pilot import FakeSession, _factory
    from tests.unit.tui.test_settings_view import _select

    session: Any = FakeSession()
    schedule = WakeSchedule(
        id="w1",
        message="Check the build and report every failing stage",
        next_due_at=ms("2026-09-09T02:52:00+00:00"),
        created_at=1,
    )
    session.wake_scheduler = SimpleNamespace(schedules=[schedule])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(columns, 30)) as pilot:
        async with asyncio.timeout(5):
            while app._session is None:
                await pilot.pause()
        assert app._wake_panel is not None
        app._refresh_band()
        assert app._wake_panel._shown is not None
        assert "7:52 PM PDT" in app._wake_panel._shown[0][0][1]
        await pilot.pause()
        assert app._wake_panel.size.width <= app.screen.size.width
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select(view, "display.time_format")
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        # Opening the enum starts on the current 12h choice; Down selects 24h.
        assert ConfigManager(tmp_path / "config").get_config_value("display.time_format") == "24h"
        await pilot.press("escape")
        # Let the production timer perform the repaint, not a test-only sync.
        async with asyncio.timeout(5):
            while "19:52 PDT" not in app._wake_panel._shown[0][0][1]:
                await pilot.pause()
        assert "19:52 PDT" in app._wake_panel._shown[0][0][1]
        assert schedule.next_due_at == ms("2026-09-09T02:52:00+00:00")
        assert not app.screen.show_vertical_scrollbar
        await pilot.resize_terminal(40, 30)
        async with asyncio.timeout(5):
            while app._wake_panel._shown[2] != app.screen.size.width - 2:
                await pilot.pause()
        await pilot.pause()
        assert app._wake_panel.size.width <= app.screen.size.width
        from rich.cells import cell_len

        lines = app._wake_panel._build(app._wake_panel._shown[0]).plain.splitlines()
        assert all(cell_len(line) <= app.screen.size.width - 2 for line in lines)
        assert any(line.endswith("…") for line in lines)


def test_cli_create_list_and_json_preserve_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse
    import json
    from types import SimpleNamespace

    from local_operator.cli import _wake_create, wake_command
    from local_operator.wakes.store import read_entry
    from tests.unit.wakes.test_wake_create_cli import _args, _session

    root = tmp_path / "config"
    _session(root, "wakecreate01")
    due = ms("2026-09-09T02:52:00+00:00")
    monkeypatch.setattr(time, "time", lambda: due / 1000 - 120)
    # A CLI display test must never install or restart the machine's daemon.
    monkeypatch.setattr(
        "local_operator.wakes.install.ensure_supervisor_installed",
        lambda _: SimpleNamespace(reason="test: supervisor not installed"),
    )
    assert _wake_create(_args(json=False)) == 0
    assert "7:52 PM PDT" in capsys.readouterr().out
    entry = read_entry(root, "wakecreate01")
    assert entry is not None and entry["schedules"][0]["next_due_at"] == due
    args = argparse.Namespace(wake_command="list", json=False)
    assert wake_command(args) == 0
    assert "7:52 PM PDT" in capsys.readouterr().out
    args.json = True
    assert wake_command(args) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["next_due_at"] == due
    write_setting(ConfigManager(root), BY_KEY["display.time_format"], "24h")
    args.json = False
    assert wake_command(args) == 0
    assert "19:52 PDT" in capsys.readouterr().out


def test_missed_wake_receipt_uses_local_clock() -> None:
    from local_operator.harness.wake import WakeSchedule
    from local_operator.session.session import Session

    due = ms("2026-09-09T02:52:00+00:00")
    schedule = WakeSchedule(id="w1", message="build", next_due_at=due, created_at=1)
    text = Session._format_missed_wake_catchup(
        [{"schedule": schedule, "occurrences": 1, "due": 1}], due + 60_000
    )
    assert "7:52 PM PDT" in text
    assert schedule.next_due_at == due


def test_unknown_preference_falls_back_to_am_pm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_operator.tui.settings.settings_get", lambda *_: "bogus")
    assert "12:00 PM PDT" in format_wake_time(ms("2026-09-08T19:00:00+00:00"))


@pytest.mark.asyncio
async def test_tool_create_list_and_panel_share_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.harness.types import ToolContext
    from local_operator.tools.builtin import execute_wake
    from local_operator.tui.widgets.wake_panel import WakePanel
    from tests.unit.tools.test_builtin_tools import _FakeScheduler

    due = ms("2026-09-09T02:52:00+00:00")
    monkeypatch.setattr("local_operator.tools.builtin.time.time", lambda: due / 1000 - 60)
    scheduler = _FakeScheduler()
    context = ToolContext(cwd=str(tmp_path), session_id="clock-test", wake_scheduler=scheduler)
    created = await execute_wake(
        "c", {"op": "create", "message": "check build", "in": "1m"}, None, None, context
    )
    assert not created.is_error
    assert "7:52 PM PDT" in created.text
    assert scheduler.schedules[0].next_due_at == due
    listed = await execute_wake("l", {"op": "list"}, None, None, context)
    assert "7:52 PM PDT" in listed.text
    assert "7:52 PM PDT" in WakePanel._fingerprint(scheduler.schedules[0])[1]
    write_setting(ConfigManager(tmp_path / "config"), BY_KEY["display.time_format"], "24h")
    listed = await execute_wake("l", {"op": "list"}, None, None, context)
    assert "19:52 PDT" in listed.text
    assert "19:52 PDT" in WakePanel._fingerprint(scheduler.schedules[0])[1]
    assert scheduler.schedules[0].next_due_at == due
