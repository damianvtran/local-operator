"""``OperatorApp`` reacts to a ``config.yml`` change from another process.

The session applies the values (``tests/unit/session/test_config_live.py``);
the app's job is the user-facing half — say what changed and why behaviour
just moved — plus the two groups only the TUI owns, ``display.*`` and
``tui.theme``, when the write came from ANOTHER process. A write from this
process is silent: the page already told the user.

The watcher is driven by ``poll_now()`` here rather than by its timer, so the
tests are bound by loop turns, never by the 2 s cadence.
"""

from __future__ import annotations

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _write_elsewhere(config_dir, key: str, value) -> None:
    """A write shaped like another process's: below the notify hook."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io._store(ConfigManager(config_dir), setting.path, value)


def _notices(app) -> list[str]:
    return [block.text() or "" for block in app.query(NoticeBlock)]


async def _adopted(app, pilot) -> None:
    for _ in range(200):
        if app._session is not None and app._unsubscribe_config_watch is not None:
            return
        await pilot.pause()
    raise AssertionError("the app never adopted a session / subscribed to config")


@pytest.mark.asyncio
async def test_a_change_from_another_process_is_announced_once_with_its_keys(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        watcher = process_watcher(tmp_path)
        _write_elsewhere(tmp_path, "compaction.threshold_percent", 0.5)
        _write_elsewhere(tmp_path, "retry.maxRetries", 3)
        watcher.poll_now()
        await pilot.pause()
        notices = [n for n in _notices(app) if "config.yml changed" in n]
        assert notices == [
            "config.yml changed: compaction.threshold_percent, retry.maxRetries — applied"
        ]


@pytest.mark.asyncio
async def test_a_non_live_key_is_named_as_taking_effect_on_new(monkeypatch, tmp_path) -> None:
    """``tool_approval_mode`` is deliberately build-time; the notice must not
    claim it applied."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        _write_elsewhere(tmp_path, "tool_approval_mode", "auto")
        _write_elsewhere(tmp_path, "compaction.enabled", False)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        notices = [n for n in _notices(app) if "config.yml changed" in n]
        assert notices == [
            "config.yml changed: compaction.enabled — applied; "
            "tool_approval_mode takes effect on /new"
        ]


@pytest.mark.asyncio
async def test_a_write_from_this_process_is_silent(monkeypatch, tmp_path) -> None:
    """The page or command here already showed its result; a second line
    would be the same news twice."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        setting = settings_io.resolve_key("compaction.enabled")
        assert setting is not None
        settings_io.write_setting(ConfigManager(tmp_path), setting, False)
        await pilot.pause()
        assert not [n for n in _notices(app) if "config.yml changed" in n]
        # And the fingerprint was recorded: the next tick has nothing to say.
        assert process_watcher(tmp_path).poll_now() is None


@pytest.mark.asyncio
async def test_a_metadata_only_rewrite_produces_no_line(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        ConfigManager(tmp_path).update_config({}, write=True)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert not [n for n in _notices(app) if "config.yml changed" in n]


@pytest.mark.asyncio
async def test_a_theme_written_by_another_process_is_applied_here(monkeypatch, tmp_path) -> None:
    """Closes the cross-process gap for the one LIVE group the session does
    not own: ``/theme`` in pane A repaints pane B."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        before = theme_mod.current_theme()
        target = next(name for name in theme_mod.available_themes() if name != before)
        _write_elsewhere(tmp_path, "tui.theme", target)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert theme_mod.current_theme() == target
        assert any("tui.theme — applied" in n for n in _notices(app))
    theme_mod.set_theme(before)


@pytest.mark.asyncio
async def test_an_unknown_theme_on_disk_is_reported_not_raised(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        before = theme_mod.current_theme()
        # Below the validating facade on purpose: a hand edit is exactly how
        # an unknown name reaches the file.
        manager = ConfigManager(tmp_path)
        manager.set_config_value("tui", {"theme": "no-such-theme"})
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert theme_mod.current_theme() == before
        assert any("unknown theme" in n for n in _notices(app))


@pytest.mark.asyncio
async def test_a_display_flag_from_another_process_drops_the_paint_cache(
    monkeypatch, tmp_path
) -> None:
    from local_operator.tui import settings as tui_settings

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        tui_settings.settings_reload()
        assert tui_settings.settings_get("display.terminal_title") is True  # primes the cache
        _write_elsewhere(tmp_path, "display.terminal_title", False)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert tui_settings.settings_get("display.terminal_title") is False
    tui_settings.settings_reload()


@pytest.mark.asyncio
async def test_unmount_unsubscribes_the_app(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        watcher = process_watcher(tmp_path)
        assert len(watcher._listeners) == 1
    assert watcher._listeners == []
    assert app._unsubscribe_config_watch is None
