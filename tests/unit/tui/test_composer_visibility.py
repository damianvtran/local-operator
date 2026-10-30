"""``display.composer.*`` — widget visibility applied live from ``config.yml``.

The registry rows are `tests/unit/test_settings_io.py`'s subject and the
segment rendering is `test_status_line.py`'s; THIS module owns the app half,
the one the operator's request named: a config edit must reach a RUNNING TUI
without a relaunch. Two deliveries have to work and they are not the same
code path —

- a write from ANOTHER process (the ``lop config edit`` an agent runs, a hand
  edit, another pane): the config watcher ticks, the display cache is
  reloaded, then the widgets and band are applied;
- a write from THIS process (the ``/settings`` page's ``write_setting``): the
  facade notifies the watcher locally and the same apply runs off that call —
  if only the disk branch applied, the page would paint a LIVE section whose
  toggles did nothing until a relaunch.

The watcher is driven by ``poll_now()`` rather than by its timer, so the tests
are bound by loop turns, never by the 2 s cadence (same harness as
``test_config_change_notice.py``).
"""

from __future__ import annotations

import pytest
from rich.text import Text

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.tui.app import Chrome, OperatorApp
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


def _write_here(config_dir, key: str, value) -> None:
    """A write shaped like the human's own: the settings facade, this process."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io.write_setting(ConfigManager(config_dir), setting, value)


def _painted_band(app) -> str:
    """The band's painted row as text (the pixels, not the receipt).

    ``Static.content`` is typed as the general renderable union; the band always
    updates it with a rich ``Text``, so narrow here rather than cast at each
    assertion.
    """
    content = app.query_one("#status-band", Chrome).content
    assert isinstance(content, Text), type(content).__name__
    return content.plain


async def _adopted(app, pilot) -> None:
    for _ in range(200):
        if app._session is not None and app._unsubscribe_config_watch is not None:
            return
        await pilot.pause()
    raise AssertionError("the app never adopted a session / subscribed to config")


def _boot(monkeypatch, tmp_path) -> OperatorApp:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    ConfigManager(tmp_path).set_config_value("hosting", "")
    return OperatorApp(lambda: _factory(FakeSession()))


@pytest.mark.asyncio
async def test_the_band_and_chevron_flip_on_a_disk_change_and_come_back(
    monkeypatch, tmp_path
) -> None:
    """The two whole-widget keys, both directions, on a running app.

    Hidden removes the widget from the layout (Textual ``display``), and a
    later ``true`` must restore it — a visibility flag is a choice, not a
    one-shot.
    """
    app = _boot(monkeypatch, tmp_path)
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        band = app.query_one("#status-band", Chrome)
        chevron = app.query_one("#prompt-chevron", Chrome)
        assert band.display is True and chevron.display is True  # the shipped shape

        _write_elsewhere(tmp_path, "display.composer.band", False)
        _write_elsewhere(tmp_path, "display.composer.chevron", False)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert band.display is False
        assert chevron.display is False

        _write_elsewhere(tmp_path, "display.composer.band", True)
        _write_elsewhere(tmp_path, "display.composer.chevron", True)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert band.display is True
        assert chevron.display is True


@pytest.mark.asyncio
async def test_a_disk_change_leaves_a_hidden_segment_out_of_the_painted_band(
    monkeypatch, tmp_path
) -> None:
    """A segment key arrives through the same delivery, and the band REPAINTS.

    Both halves are asserted on the real app: ``is_showing`` (the app-level
    receipt) and the band's own painted content (the pixels). The pixels are
    the half that matters — the segment keys are read at paint time, so a
    listener that only updated state would leave the old row on screen next to
    the notice saying the key moved.
    """
    app = _boot(monkeypatch, tmp_path)
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        painted = _painted_band(app)
        assert "test/model" in painted and "◆" in painted, painted

        _write_elsewhere(tmp_path, "display.composer.model", False)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert app._status is not None
        assert app._status.is_showing("model") is False
        repainted = _painted_band(app)
        assert "◆" not in repainted and "test/model" not in repainted, repainted
        assert "⌂" in repainted, "the siblings the freed cells went to are still there"

        _write_elsewhere(tmp_path, "display.composer.model", True)
        process_watcher(tmp_path).poll_now()
        await pilot.pause()
        assert "test/model" in _painted_band(app)


@pytest.mark.asyncio
async def test_the_local_path_applies_on_the_settings_pages_own_call_stack(
    monkeypatch, tmp_path
) -> None:
    """``settings_io.write_setting`` — the /settings page's write — applies at once.

    No ``poll_now()`` anywhere in this test: the facade's local notification is
    synchronous on the loop thread, and the listener has to apply the composer
    flags on that branch too, or the page's LIVE label would be a promise the
    app does not keep.
    """
    app = _boot(monkeypatch, tmp_path)
    async with app.run_test(size=(100, 24)) as pilot:
        await _adopted(app, pilot)
        band = app.query_one("#status-band", Chrome)
        chevron = app.query_one("#prompt-chevron", Chrome)
        assert chevron.display is True

        _write_here(tmp_path, "display.composer.chevron", False)
        await pilot.pause()
        assert chevron.display is False

        _write_here(tmp_path, "display.composer.cost", False)
        await pilot.pause()
        assert app._status is not None
        assert app._status.is_showing("cost") is False

        _write_here(tmp_path, "display.composer.chevron", True)
        _write_here(tmp_path, "display.composer.cost", True)
        await pilot.pause()
        assert chevron.display is True
        assert app._status.is_showing("cost") is True
        assert band.display is True


async def _band_shows(app, pilot, needle: str, *, present: bool = True) -> str:
    """Pump the loop until the band does (or does not) paint ``needle``.

    The command runs in a worker on a real subprocess, so its output lands a
    few loop turns after the write; bounded by turns, never by the clock.
    """
    painted = ""
    for _ in range(400):
        await pilot.pause()
        painted = _painted_band(app)
        if (needle in painted) is present:
            return painted
    raise AssertionError(f"band never {'showed' if present else 'dropped'} {needle!r}: {painted}")


@pytest.mark.asyncio
async def test_the_status_command_paints_live_and_clears_when_removed(
    monkeypatch, tmp_path
) -> None:
    """``display.composer.command`` reaches a running TUI without a relaunch.

    Set from another process: the command runs AT ONCE (not after a full
    interval) and its first line lands on the band. A new command replaces the
    old reading, and removing the command clears the segment and the timer.
    The script reads its stdin payload, so the payload path is exercised too.
    """
    app = _boot(monkeypatch, tmp_path)
    async with app.run_test(size=(140, 24)) as pilot:
        await _adopted(app, pilot)
        assert app._status_command_timer is None

        _write_elsewhere(
            tmp_path,
            "display.composer.command",
            "python3 -c 'import json,sys; "
            'print("acct", json.load(sys.stdin)["model"]["display_name"])\'',
        )
        process_watcher(tmp_path).poll_now()
        await _band_shows(app, pilot, "acct test/model")
        assert app._status_command_timer is not None

        _write_elsewhere(tmp_path, "display.composer.command", "echo second")
        process_watcher(tmp_path).poll_now()
        painted = await _band_shows(app, pilot, "second")
        assert "acct" not in painted

        _write_elsewhere(tmp_path, "display.composer.command", "")
        process_watcher(tmp_path).poll_now()
        await _band_shows(app, pilot, "second", present=False)
        assert app._status_command_timer is None
        assert app._status_command_armed is None


@pytest.mark.asyncio
async def test_an_unrelated_composer_edit_does_not_rearm_the_command(monkeypatch, tmp_path) -> None:
    """Every ``display.composer.*`` edit lands in the same apply; only a change
    to the command or its interval may reset the cadence or fork a shell."""
    app = _boot(monkeypatch, tmp_path)
    async with app.run_test(size=(140, 24)) as pilot:
        await _adopted(app, pilot)
        _write_here(tmp_path, "display.composer.command_interval", 60)
        _write_elsewhere(tmp_path, "display.composer.command", "echo steady")
        process_watcher(tmp_path).poll_now()
        await _band_shows(app, pilot, "steady")
        timer = app._status_command_timer
        assert app._status_command_armed == ("echo steady", 60)

        _write_here(tmp_path, "display.composer.cost", False)
        await pilot.pause()
        assert app._status_command_timer is timer

        _write_here(tmp_path, "display.composer.command_interval", 120)
        await pilot.pause()
        assert app._status_command_timer is not timer
        assert app._status_command_armed == ("echo steady", 120)
