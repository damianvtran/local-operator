"""``/language``: the command surface (slice C).

The command is the discoverable half of the ``language`` setting; its tests
are about the four shapes a user can meet — bare (what am I on), a write
(persists, receipts), a no-op (already there), and a refusal (unknown, or a
supported locale no wave has shipped yet). The LIVE half — a switch or a
foreign write moving what the running TUI resolves — is
``test_language_live.py``; both files share the isolated config dir, so a
committed switch never touches the operator's real config.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.cells import cell_len

from local_operator.config import ConfigManager
from local_operator.i18n import text as i18n_text
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _restore_locale() -> Iterator[None]:
    """The active locale is PROCESS state a committed switch moves; put it back.

    ``test_a_shipped_wave_locale_switches_and_persists`` leaves the process in
    ``fr``, and every later test in the same worker then resolves French copy:
    CI shard ``test (3.12, 1)`` failed twice on
    ``test_marker_chip.py::test_the_receipt_row_is_inked_as_chrome_not_as_the_user_s_prose``
    reading ``1 image jointe`` (PR #2377, reproduced locally on main ``8492cebb5``
    by running this file before that test). The sibling ``test_language_live.py``
    restores the same way; this file never patches the catalogue root, so a plain
    move back to ``en`` is enough.
    """
    yield
    i18n_text.set_active_locale("en")


@pytest.fixture()
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the config at a temp dir — a committed switch WRITES one."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _type(pilot, app: OperatorApp, text: str) -> None:
    # Set the line AND park the caret at the end, as typing it would: the slash
    # pickers are caret-anchored (inline detection), so a text-set that leaves
    # the caret at the origin sits before the slash where no command is live.
    editor = app.query_one(Editor)
    editor.text = text
    editor.move_cursor(editor._end_of_buffer())
    editor._sync_picker()
    await pilot.pause()
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [block.text() or "" for block in app.query(NoticeBlock)]


def _language_notices(app: OperatorApp) -> list[str]:
    """The command's own receipts — every one must fit the 70-cell budget."""
    return [
        n for n in _notices(app) if n.startswith("language:") or n.startswith("unknown language")
    ]


def _choice_map(picker) -> dict[str, ArgumentChoice]:
    """The open argument list as name -> row, narrowed for the type checker."""
    rows: dict[str, ArgumentChoice] = {}
    for name, choice in picker.suggestions():
        assert isinstance(choice, ArgumentChoice), choice
        rows[name] = choice
    return rows


def _stored(config_dir: Path) -> str | None:
    value = ConfigManager(config_dir).get_config().values.get("language")
    return value if isinstance(value, str) else None


@pytest.mark.asyncio
async def test_bare_language_names_the_active_language(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/language")
        await pilot.pause()
        texts = _notices(app)
        # Resolved language, its English name, the value space, and the space
        # as the browse — the accepted command shape.
        assert any(
            text.startswith("language: en (English)")
            and "/language <value>; space browses" in text
            and "; space" in text
            for text in texts
        ), texts
        for text in _language_notices(app):
            assert cell_len(text) <= 70, text
        # The bare receipt names the space browse instead of enumerating the
        # value space (which grows with every shipped locale), so it stays well
        # inside the 70-cell notice row for every accepted value.
        for value in ("en", "fr", "es", "ru", "vi", "ur", "hi", "auto"):
            app._run_slash_command(f"/language {value}")
            await pilot.pause()
            app._run_slash_command("/language")
            await pilot.pause()
            assert any(
                text.startswith(f"language: {'en' if value == 'auto' else value} (")
                and "/language <value>; space browses" in text
                for text in _notices(app)
            ), (value, _notices(app))
        for text in _language_notices(app):
            assert cell_len(text) <= 70, text


@pytest.mark.asyncio
async def test_a_switch_persists_the_same_key_the_settings_row_writes(
    config_dir: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/language en")
        await pilot.pause()
        # ONE truth: the registry key the /settings row owns, written through
        # the same facade (`settings_io.write_setting`).
        assert _stored(config_dir) == "en"
        assert any(
            text.startswith("language: en saved — active en (English)") for text in _notices(app)
        ), _notices(app)
        for text in _language_notices(app):
            assert cell_len(text) <= 70, text


@pytest.mark.asyncio
async def test_switching_to_auto_when_already_auto_writes_nothing(
    config_dir: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/language auto")
        await pilot.pause()
        # No config value existed; `auto` IS the absent value, so the no-op
        # receipt must not flip it into an explicitly stored one.
        assert _stored(config_dir) is None
        assert any("language: already auto" in text for text in _notices(app)), _notices(app)
        for text in _language_notices(app):
            assert cell_len(text) <= 70, text


@pytest.mark.asyncio
async def test_the_lang_alias_resolves_to_the_same_command(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/lang en")
        await pilot.pause()
        # The one-resolver rule: `/lang` reaches the same handler, the same
        # write and the same receipt.
        assert _stored(config_dir) == "en"
        assert any(
            text.startswith("language: en saved — active en (English)") for text in _notices(app)
        ), _notices(app)
        for text in _language_notices(app):
            assert cell_len(text) <= 70, text


@pytest.mark.asyncio
async def test_the_space_opens_the_choices_on_the_current_value(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, app, "/language ")
        picker = editor.picker
        assert picker.is_open(), "the space must open the choices list"
        rows = _choice_map(picker)
        assert list(rows) == ["auto", "en", "fr", "es", "ru", "vi", "ur", "hi"], list(rows)
        assert picker.highlighted_name() == "auto", "the browse starts where the user is"
        assert rows["auto"].detail == "← current"
        # After a committed switch the marker moves with the value.
        app._run_slash_command("/language en")
        await pilot.pause()
        editor.focus()
        await _type(pilot, app, "/language ")
        picker = editor.picker
        rows = _choice_map(picker)
        assert picker.highlighted_name() == "en"
        assert rows["en"].detail == "← current"
        assert rows["auto"].detail == ""


@pytest.mark.asyncio
async def test_unknown_and_unshipped_values_are_refused_with_the_affordance(
    config_dir: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        # Hindi and Urdu now ship; Chinese stays held after the incomplete
        # slash audit. A supported-but-unshipped locale may not write config.
        for value in ("klingon", "zh-CN"):
            app._run_slash_command(f"/language {value}")
            await pilot.pause()
            receipt = f"unknown language '{value}' — press space to list values"
            assert receipt in _notices(app), _notices(app)
            assert _stored(config_dir) is None
        for text in _language_notices(app):
            assert cell_len(text) <= 70, text
        # The echoed token is bounded by CELLS, not characters: a wide token
        # would otherwise push the refusal past the row.
        for token in ("語" * 30, "\\" * 20):
            app._run_slash_command("/language " + token)
            await pilot.pause()
            long_notice = _language_notices(app)[-1]
            assert "…" in long_notice, long_notice
            assert cell_len(long_notice) <= 70, long_notice


@pytest.mark.asyncio
async def test_a_shipped_wave_locale_switches_and_persists(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        # Wave 1 shipped es+fr: the call that was refused pre-wave now keys its
        # catalogue and stores the value through the shared facade.
        app._run_slash_command("/language fr")
        await pilot.pause()
        assert _stored(config_dir) == "fr"
        assert any(
            text.startswith("language: fr saved — active fr (French)") for text in _notices(app)
        ), _notices(app)
        for text in _language_notices(app):
            assert cell_len(text) <= 70, text
