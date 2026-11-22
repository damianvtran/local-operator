"""A mounted /info must repaint through both real language-setting routes."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.i18n import text
from local_operator.i18n.keys import tui_small
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.info_panel import InfoScreen
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_info_panel import _live, _snapshot
from tests.unit.tui.test_language_live import _boot, _write_elsewhere


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["command", "watcher", "external"])
@pytest.mark.parametrize("at_end", [False, True])
async def test_mounted_info_repaints_without_refetching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: str, at_end: bool
) -> None:
    from scripts.visual_capture import save_capture

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
            if at_end:
                screen._scroll.scroll_end(animate=False)
            else:
                screen._scroll.scroll_to(y=5, animate=False)
            await pilot.pause()
            old_scroll = screen._scroll.scroll_y
            before = str(screen._title.content)
            evidence = os.environ.get("INFO_READOUT_EVIDENCE")
            if evidence:
                save_capture(app, str(Path(evidence) / f"refresh-{route}-before.svg"))
            repainted = asyncio.Event()
            first_hints: list[str] = []
            original_repaint = screen._repaint

            def observe_first_paint() -> None:
                original_repaint()
                if text.active_locale() == "fr" and not first_hints:
                    # No event-loop yield: this is the first painted Static,
                    # before call_after_refresh can repair a stale-language hint.
                    first_hints.append(str(screen._hint.content))
                    if evidence:
                        save_capture(
                            app, str(Path(evidence) / f"refresh-{route}-{at_end}-immediate.svg")
                        )
                    repainted.set()

            monkeypatch.setattr(screen, "_repaint", observe_first_paint)
            if route == "command":
                app._cmd_language("fr", app._system_notice)
            elif route == "watcher":
                _write_elsewhere(config_dir, "fr")
                process_watcher(config_dir).poll_now()
            else:
                # A real other process, with only synthetic config and the test
                # environment; the app's automatic watcher must deliver the move.
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import sys; "
                        "from tests.unit.tui.test_language_live import _write_elsewhere; "
                        "_write_elsewhere(Path(sys.argv[1]), 'fr')",
                        str(config_dir),
                    ],
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert result.returncode == 0, result.stderr
                assert result.stderr == ""
            await asyncio.wait_for(repainted.wait(), timeout=15)
            assert "retour" in first_hints[0] and "copier" in first_hints[0], first_hints
            assert " back" not in first_hints[0] and " copy" not in first_hints[0], first_hints
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
            if at_end:
                assert screen._scroll.scroll_y == screen._scroll.max_scroll_y
            else:
                assert screen._scroll.scroll_y == old_scroll
            assert "retour" in str(screen._hint.content)
            assert "défiler" in str(screen._hint.content)
            if evidence:
                save_capture(app, str(Path(evidence) / f"refresh-{route}-after.svg"))
    finally:
        text.set_active_locale("en")
        _reset_for_tests()
