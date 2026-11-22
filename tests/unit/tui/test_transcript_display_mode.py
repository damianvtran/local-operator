"""The transcript layout mode: the registry row, its accessor, and the live apply.

Slice 1 of the multi-level fold (`display.transcript_display`, request
2026-11-21). The fold layer that actually DRAWS the two layouts lands in a later
slice, so what is asserted here is the plumbing: the `/settings` row, the
unknown-is-default parse, the view's stored mode, the `/transcript` command and
``_on_config_change`` honouring a change. A test that painted a fold bar would
be asserting about a slice that does not exist yet.

Every app test points ``LOCAL_OPERATOR_CONFIG_DIR`` at a tmp_path: the command
and the page both WRITE config.yml, and a test on the real config dir would edit
the developer's own settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from local_operator import settings_io
from local_operator.settings_io import Kind
from local_operator.tui.app import OperatorApp
from local_operator.tui.settings import (
    DEFAULT_TRANSCRIPT_DISPLAY,
    TRANSCRIPT_DISPLAY_MODES,
    settings_reload,
    transcript_display_mode,
)
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _set_editor_line
from tests.unit.tui.waiting import settle


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # The display cache is per PROCESS, so a value left by an earlier test would
    # leak into this one's first read.
    settings_reload()
    yield tmp_path
    settings_reload()


def _app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


def _values(tmp_path: Path) -> dict[str, Any]:
    config = tmp_path / "config.yml"
    if not config.exists():
        return {}
    return yaml.safe_load(config.read_text()).get("values", {})


def _stored(tmp_path: Path, raw: Any) -> None:
    """Write the key AS A HAND EDIT WOULD: straight to the file, no registry."""
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"values": {"display.transcript_display": raw}}))


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def test_the_registry_row_is_a_flat_dotted_enum_in_live_appearance() -> None:
    """The `display.*` trap: a NESTED path writes a key nothing reads.

    The row must also be LIVE, because the whole point of the command and of the
    config-watch arm below is that the mode changes without a relaunch.
    """
    setting = settings_io.BY_KEY["display.transcript_display"]
    assert setting.kind is Kind.ENUM
    assert setting.path == ("display.transcript_display",)
    assert setting.section == "appearance"
    scope = {section.name: section.scope for section in settings_io.SECTIONS}
    assert scope[setting.section] is settings_io.Scope.LIVE
    assert setting.default == DEFAULT_TRANSCRIPT_DISPLAY
    assert [choice.value for choice in setting.choices] == list(TRANSCRIPT_DISPLAY_MODES)
    assert [choice.label for choice in setting.choices] == ["By turn", "By response"]
    # The consumer's own vocabulary is the same tuple, in the same order.
    assert TRANSCRIPT_DISPLAY_MODES == ("by-turn", "by-response")
    # The view's class default is a COPY (the widget reads no setting), so it is
    # pinned here rather than trusted to stay in step.
    assert TranscriptView._display_mode == DEFAULT_TRANSCRIPT_DISPLAY


@pytest.mark.parametrize(
    "raw",
    ["sideways", "", "../../etc/passwd", 42, None, ["by-response"]],
    ids=["unknown-token", "empty", "path", "int", "null", "list"],
)
def test_an_unreadable_stored_value_reads_as_the_default(tmp_path: Path, raw: Any) -> None:
    """A token this build does not know must paint the shipped layout.

    The registry validates every WRITE, but config.yml is a file people edit, and
    an older or newer build may have stored a token this one has never heard of.
    A parse that raised would take down a frame; one that returned the raw value
    would paint a layout that does not exist.
    """
    _stored(tmp_path, raw)
    settings_reload()
    assert transcript_display_mode() == DEFAULT_TRANSCRIPT_DISPLAY


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("by-response", "by-response"),
        ("  BY-RESPONSE  ", "by-response"),
        ("by-turn", "by-turn"),
    ],
    ids=["exact", "hand-edited-case-and-space", "default"],
)
def test_a_known_value_reads_back(tmp_path: Path, raw: str, expected: str) -> None:
    _stored(tmp_path, raw)
    settings_reload()
    assert transcript_display_mode() == expected


def test_an_absent_key_reads_as_the_default(tmp_path: Path) -> None:
    assert not (tmp_path / "config.yml").exists()
    settings_reload()
    assert transcript_display_mode() == DEFAULT_TRANSCRIPT_DISPLAY == "by-turn"


@pytest.mark.asyncio
async def test_the_view_stores_the_mode_and_hands_it_back() -> None:
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = app._transcript_view()
        assert view.display_mode == "by-turn"
        view.set_display_mode("by-response")
        assert view.display_mode == "by-response"
        # Switching back is the same one-token store, not a one-way latch.
        view.set_display_mode("by-turn")
        assert view.display_mode == "by-turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", ["by-response", "by-turn"])
async def test_the_stored_mode_applies_at_boot(tmp_path: Path, stored: str) -> None:
    """Boot reads the setting; it is not a value only a CHANGE can deliver.

    The stored token used to survive on disk and be ignored at mount: the page
    and the command both said "By response" while the transcript drew By turn
    until the next `/transcript` press. The stored value is read where the view
    is created, for every launch and every replay of the same view.
    """
    _stored(tmp_path, stored)
    # The settings cache is per PROCESS: this test writes the file after the
    # fixture's reload, so the boot has to see it the way a fresh process would.
    settings_reload()
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        assert app._transcript_view().display_mode == stored


@pytest.mark.asyncio
async def test_the_space_offers_both_modes_with_the_current_one_marked() -> None:
    """The command's argument list, through the real editor→app message path.

    ``_transcript_choices`` alone would not prove the list reaches the picker:
    the rows are pushed by ``on_argument_query_opened`` after the editor bubbles
    ``ArgumentQueryOpened``, so the assertion is made on the picker's own rows
    once the app has pushed them.
    """
    from local_operator.tui.widgets.editor import Editor

    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        _set_editor_line(editor, "/transcript ")
        await settle(
            pilot,
            lambda: len(editor.picker._choices) == 2,
            what="/transcript rows pushed",
        )
        rows = [(choice.name, choice.detail) for choice in editor.picker._choices]
        assert rows == [("by-turn", "← current"), ("by-response", "")], rows


@pytest.mark.asyncio
async def test_the_command_sets_the_setting_and_names_the_mode(tmp_path: Path) -> None:
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._run_slash_command("/transcript by-response")
        await pilot.pause()
        values = _values(tmp_path)
        assert values.get("display.transcript_display") == "by-response"
        # Flat-dotted: a nested mapping would be the key nothing reads.
        assert "display" not in values
        assert app._transcript_view().display_mode == "by-response"
        assert "Transcript: By response" in _notices(app)


@pytest.mark.asyncio
async def test_the_bare_command_reports_the_mode_and_both_choices() -> None:
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._run_slash_command("/transcript")
        await pilot.pause()
        assert "Transcript: By turn · choose by-turn or by-response" in _notices(app)


@pytest.mark.asyncio
async def test_an_unknown_token_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    app = _app()
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._run_slash_command("/transcript sideways")
        await pilot.pause()
        assert not (tmp_path / "config.yml").exists()
        notices = _notices(app)
        assert any("Unknown mode" in text and "by-turn" in text for text in notices), notices
        assert app._transcript_view().display_mode == "by-turn"
