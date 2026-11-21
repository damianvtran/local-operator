"""A mounted /info must repaint through both real language-setting routes."""
import os
from pathlib import Path

import pytest

from local_operator.i18n import text
from local_operator.i18n.keys import tui_small
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.info_panel import InfoScreen
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_info_panel import _live, _snapshot
from tests.unit.tui.test_language_live import _boot, _write_elsewhere


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["command", "watcher"])
async def test_mounted_info_repaints_without_refetching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: str
) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    monkeypatch.setenv("LOP_LANG", "en")
    _reset_for_tests()
    text.set_active_locale("en")
    app = OperatorApp(lambda: _factory(FakeSession()))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app, watcher=True)
            # Seed only in-memory diagnostic facts: no registry/provider probes.
            screen = InfoScreen(_live())
            screen.snapshot = snapshot = _snapshot()
            await app.push_screen(screen)
            await pilot.pause()
            screen._scroll.scroll_to(y=5, animate=False)
            await pilot.pause()
            old_scroll = screen._scroll.scroll_y
            before = str(screen._title.content)
            evidence = os.environ.get("INFO_READOUT_EVIDENCE")
            if evidence:
                from scripts.visual_capture import save_capture
                save_capture(app, str(Path(evidence) / f"refresh-{route}-before.svg"))
            if route == "command":
                app._cmd_language("fr", app._system_notice)
            else:
                _write_elsewhere(config_dir, "fr")
                process_watcher(config_dir).poll_now()
            await pilot.pause()
            expected = text.tr("tui.small")(tui_small.info_panel_install_info_heading())
            assert expected != "Install info "
            assert str(screen._title.content).startswith(expected), (
                route,
                before,
                str(screen._title.content),
                expected,
            )
            assert str(screen._title.content) != before
            assert screen.snapshot is snapshot
            assert screen._scroll.scroll_y == old_scroll
            if evidence:
                save_capture(app, str(Path(evidence) / f"refresh-{route}-after.svg"))
    finally:
        text.set_active_locale("en")
        _reset_for_tests()
