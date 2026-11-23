"""A mounted /info must repaint through both real language-setting routes."""

import ast
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from textual.pilot import Pilot

from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.i18n import text
from local_operator.i18n.keys import tui_small
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.info_panel import InfoScreen
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_info_panel import _live, _snapshot
from tests.unit.tui.test_language_live import _boot, _write_elsewhere

# CI installs only local_operator, not tests. The other-process writer must use
# installed production APIs even when the parent can import test helpers.
_EXTERNAL_LANGUAGE_WRITE = """\
from pathlib import Path
import sys
from local_operator import settings_io
from local_operator.config import ConfigManager

setting = settings_io.resolve_key("language")
assert setting is not None
settings_io._store(ConfigManager(Path(sys.argv[1])), setting.path, "fr")
"""


def test_external_language_writer_imports_only_installed_modules() -> None:
    # Parent PYTHONPATH may expose tests locally and mask a broken child import.
    # Pin the standalone payload's imports independently of that environment.
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(ast.parse(_EXTERNAL_LANGUAGE_WRITE))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports == {"pathlib", "sys", "local_operator", "local_operator.config"}


async def _await_scrollable(pilot: Pilot[None], screen: InfoScreen, *, minimum: float) -> None:
    """Wait until the seeded readout is laid out tall enough to scroll.

    A scroll request issued before the readout has been measured is clamped (or
    dropped) against ``max_scroll_y == 0``, which is how a request can look like
    it never happened. Establish the geometry first, then ask.
    """
    scroll = screen._scroll
    for _ in range(200):
        if scroll.max_scroll_y >= minimum:
            return
        await pilot.pause()
    # The last permitted pause can itself satisfy the condition; re-read once so a
    # slow-but-correct screen is not reported as a broken fixture.
    if scroll.max_scroll_y >= minimum:
        return
    raise AssertionError(
        "the seeded readout is too short to scroll, so the scroll premise has no "
        "meaning; this is a fixture fault, not a repaint bug",
        (scroll.scroll_y, scroll.max_scroll_y, minimum),
    )


async def _scroll_request_landed(
    pilot: Pilot[None], screen: InfoScreen, *, target: float | None
) -> None:
    """Wait until a scroll request has actually moved ``scroll_y``.

    ``scroll_to``/``scroll_end`` record a target, and the requested position is
    applied by a later refresh pass; on a loaded worker that pass can land after
    the single ``pause()`` this test used to allow. Capturing the baseline before
    it lands compares a settled post-repaint position against a pre-state that
    was never the intended one — the shape of the 2026-11-23 main failure, where
    the captured baseline was 0.0 although the test had asked for y=5 and the
    value read after the repaint was 5.

    The property is untouched: the caller still asserts ``scroll_y`` is unchanged
    across the repaint, so a repaint that really moves the reader still fails.
    """
    scroll = screen._scroll
    # Bounded, not timed: 200 turns is far beyond what a laid-out screen needs,
    # and an unbounded wait would burn the shard timeout instead of failing here.
    for _ in range(200):
        if target is None:
            if scroll.scroll_y >= scroll.max_scroll_y:
                return
        elif scroll.scroll_y >= target:
            return
        await pilot.pause()
    # The last permitted pause can itself satisfy the condition; re-read once so a
    # slow-but-correct screen is not reported as a request that never landed.
    landed = scroll.scroll_y >= scroll.max_scroll_y if target is None else scroll.scroll_y >= target
    if landed:
        return
    raise AssertionError(
        "the scroll request never landed before the baseline was read",
        (scroll.scroll_y, scroll.max_scroll_y, target),
    )


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
                await _await_scrollable(pilot, screen, minimum=1.0)
                screen._scroll.scroll_end(animate=False)
                await _scroll_request_landed(pilot, screen, target=None)
            else:
                await _await_scrollable(pilot, screen, minimum=5.0)
                screen._scroll.scroll_to(y=5, animate=False)
                await _scroll_request_landed(pilot, screen, target=5.0)
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
                        _EXTERNAL_LANGUAGE_WRITE,
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
