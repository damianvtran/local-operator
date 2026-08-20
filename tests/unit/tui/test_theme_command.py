"""``/theme``: the switch, the persistence, and the live preview's lifecycle.

The preview is the part with real failure modes, so the tests here are about
its LIFETIME: a highlight applies the candidate theme to the whole screen, a
dismissal (Esc, deleting the word, an empty match set) restores the theme the
user actually has, and a commit clears the stash BEFORE applying so the
restore branch cannot undo what was just chosen. Every test restores the
module-level theme singleton, because ``theme_mod`` is process state shared
with every other test in the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture()
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the config at a temp dir — a committed switch WRITES one."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _restore_theme():
    """The theme module is a process singleton; put dark back afterwards."""
    original = theme_mod.current_theme()
    yield
    theme_mod.set_theme(original)


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _type(pilot, app: OperatorApp, text: str) -> None:
    app.query_one(Editor).text = text
    await pilot.pause()
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [getattr(block, "_text", "") for block in app.query(NoticeBlock)]


@pytest.mark.asyncio
async def test_bare_theme_names_the_active_theme(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/theme")
        await pilot.pause()
        texts = _notices(app)
        assert any("Operator Dark" in text and "dark" in text for text in texts), texts


@pytest.mark.asyncio
async def test_switch_applies_persists_and_receipts(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/theme monokai")
        await pilot.pause()
        assert theme_mod.current_theme() == "monokai"
        # The receipt names both ends of the change and says it was saved.
        assert any("dark → monokai" in text for text in _notices(app))
        saved = yaml.safe_load((config_dir / "config.yml").read_text(encoding="utf-8"))
        assert saved["values"]["tui"]["theme"] == "monokai"


@pytest.mark.asyncio
async def test_unknown_theme_is_refused_and_nothing_moves(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/theme neon-nonexistent")
        await pilot.pause()
        assert theme_mod.current_theme() == "dark"
        assert not (config_dir / "config.yml").exists()
        assert any("unknown theme" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_arrowing_previews_and_esc_restores(config_dir: Path) -> None:
    """The whole screen is the swatch, and Esc hands the real theme back."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, app, "/theme ")
        picker = editor.picker
        assert picker.is_open()
        await pilot.press("down")
        await pilot.pause()
        await pilot.pause()
        previewed = theme_mod.current_theme()
        assert previewed != "dark", "arrowing to another row must apply its theme"
        assert app._theme_before_preview == "dark"
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert theme_mod.current_theme() == "dark"
        assert app._theme_before_preview is None
        # Browsing must never write: preview is try-on, not commitment.
        assert not (config_dir / "config.yml").exists()


@pytest.mark.asyncio
async def test_deleting_the_word_restores_too(config_dir: Path) -> None:
    """The buffer is the authority: no list, no preview."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, app, "/theme ")
        await pilot.press("down")
        await pilot.pause()
        assert theme_mod.current_theme() != "dark"
        await _type(pilot, app, "")
        await pilot.pause()
        assert theme_mod.current_theme() == "dark"
        assert app._theme_before_preview is None


@pytest.mark.asyncio
async def test_committing_a_previewed_theme_survives_the_list_closing(config_dir: Path) -> None:
    """Enter on a previewed row keeps it: the restore must not undo the choice.

    The ordering under test: `_cmd_theme` clears the stash BEFORE applying, so
    the close-of-list restore that follows submission finds no stash and the
    committed theme stands.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, app, "/theme ")
        picker = editor.picker
        names = [name for name, _ in picker.suggestions()]
        target = names.index("monokai")
        for _ in range(target):
            await pilot.press("down")
            await pilot.pause()
        assert picker.highlighted_name() == "monokai"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert theme_mod.current_theme() == "monokai"
        assert app._theme_before_preview is None
        saved = yaml.safe_load((config_dir / "config.yml").read_text(encoding="utf-8"))
        assert saved["values"]["tui"]["theme"] == "monokai"


@pytest.mark.asyncio
async def test_switch_reinks_settled_blocks(config_dir: Path) -> None:
    """A finalized notice must wear the NEW ramp after a switch (retheme seam)."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        block = NoticeBlock("a settled receipt", "warning")
        app._append_block(block)
        await pilot.pause()
        before = theme_mod.semantic_color("warning")
        app._run_slash_command("/theme monokai")
        await pilot.pause()
        await pilot.pause()
        after = theme_mod.semantic_color("warning")
        assert before != after, "monokai's warning must differ for this test to see the re-ink"
        rendered = block.renderable
        spans = getattr(rendered, "spans", [])
        colors = {
            str(span.style.color.name)
            for span in spans
            if getattr(span.style, "color", None) is not None
        }
        assert any(
            after in color for color in colors
        ), f"finalized notice still wears the old ramp: {colors} (wanted {after})"


@pytest.mark.asyncio
async def test_opening_the_list_does_not_flash_the_theme(config_dir: Path) -> None:
    """`/theme ` opens ON the current theme's row, previewing nothing (F2).

    The highlight previews live, so a list that opened on row 0 flashed every
    non-default user to whatever theme sorts first before they touched a key.
    The list must open with the highlight seeded on the active theme and the
    screen unmoved.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), theme_name="monokai")
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, app, "/theme ")
        await pilot.pause()
        assert theme_mod.current_theme() == "monokai", "opening the list must not switch themes"
        assert app._theme_before_preview is None, "no preview is standing before any movement"
        picker = editor.picker
        assert picker.highlighted_name() == "monokai", "the browse starts where the user is"


@pytest.mark.asyncio
async def test_preview_defers_offscreen_reink_to_the_settle(config_dir: Path) -> None:
    """A preview re-inks visible rows only; the settle sweep covers the rest (F1).

    The preview pays its cost per arrow key, so it is bounded to the viewport;
    whatever it skipped is corrected when the browse ends, however it ends.
    This pins the settle-on-restore path: blocks scrolled far offscreen wear
    the old ramp mid-preview and the current ramp once the list closes.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 20)) as pilot:
        await _boot(pilot, app)
        # Enough notices that the first is far above the fold.
        blocks = [NoticeBlock(f"receipt {index}", "warning") for index in range(40)]
        for block in blocks:
            app._append_block(block)
        await pilot.pause()
        offscreen = blocks[0]
        assert not app._is_on_screen(offscreen), "the probe block must be scrolled out"
        dark_warning = theme_mod.semantic_color("warning")

        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, app, "/theme ")
        await pilot.press("down")
        await pilot.pause()
        await pilot.pause()
        previewed = theme_mod.current_theme()
        assert previewed != "dark"
        previewed_warning = theme_mod.semantic_color("warning")
        assert previewed_warning != dark_warning, "pick a probe token the ramps disagree on"

        def _colors(block: NoticeBlock) -> set[str]:
            spans = getattr(block.renderable, "spans", [])
            return {
                str(span.style.color.name)
                for span in spans
                if getattr(span.style, "color", None) is not None
            }

        # Mid-preview: the offscreen block still wears dark's ink (skipped),
        # while an onscreen one wears the preview's.
        assert any(
            dark_warning in color for color in _colors(offscreen)
        ), "offscreen block should be skipped by the preview sweep"
        assert any(
            previewed_warning in color for color in _colors(blocks[-1])
        ), "onscreen block should carry the previewed ramp"

        # Settle (Esc restores dark): the skipped block is swept back too.
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert theme_mod.current_theme() == "dark"
        assert any(
            dark_warning in color for color in _colors(offscreen)
        ), "the settle sweep must cover what the preview skipped"


def test_boot_with_unknown_saved_theme_falls_back_to_dark() -> None:
    """A stale config.yml name must not keep the app from booting."""
    app = OperatorApp(lambda: _factory(FakeSession()), theme_name="theme-that-was-removed")
    assert theme_mod.current_theme() == "dark"
    assert app is not None
