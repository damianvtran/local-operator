"""The agent-messages rows in §5.6 F, and the accessible/narrow rendering rules.

F means the flow cannot resume — a plan lapse, an admin/ops kill, or a
scope-excluded organization (byte-identical by design). The page must say WHY
(the contract's own sentence), present the controls view-only instead of
letting a local toggle imply the flow resumed, and keep the consent/default-off
explanation legible at the supported narrow sizes and in both themes
(review round 1 Q3 / design round 1 D1–D4).

Driven through the REAL :class:`OperatorApp` (the only host that loads the
stylesheet); the posture arrives through the process-local availability record,
exactly as the status discipline writes it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.team_bridge import availability
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.settings_view import SettingsView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The F sentence v1 emits for EVERY cause (review round 2, R9): a lapsed
#: plan, a confirmed kill and an unreadable confirmation all read it.
NEUTRAL = "Agent messages are turned off"


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from local_operator.tui.settings import settings_reload

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    settings_reload()
    availability.clear()
    yield tmp_path
    availability.clear()


def _bridge_row_index(view: SettingsView, key: str) -> int:
    view._repaint()  # rows rebuild on paint; the record lands first
    for index, row in enumerate(view._rows):
        if row.setting is not None and row.setting.key == key:
            return index
    raise AssertionError(f"no {key} row — is the section visible?")


async def _open(size: tuple[int, int] = (100, 30)):
    app = OperatorApp(lambda: _factory(FakeSession()))
    pilot = app.run_test(size=size)
    return app, pilot


def _select(view: SettingsView, index: int) -> None:
    view._selected = index
    view._repaint()


@pytest.mark.asyncio
async def test_f_shows_the_state_sentence_and_refuses_every_editing_key() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        availability.record(
            view._manager.config_dir,
            visible=True,
            state="ready",
            checked_at=1.0,
            read_only=True,
            note=NEUTRAL,
        )
        index = _bridge_row_index(view, "team_bridge.enabled")
        _select(view, index)

        # The row explains the STATE — the contract's own sentence — instead of
        # the generic prerequisites help.
        assert view._detail_text.plain == NEUTRAL
        # No editing key is advertised for it.
        assert view._current_is_readonly() is True

        # `enter` opens nothing and reports the state; `space` writes nothing;
        # `r` rewrites nothing.
        await pilot.press("enter")
        assert view._editing is None and view._expanded is None
        assert view._error == NEUTRAL

        setting = settings_io.BY_KEY["team_bridge.enabled"]
        before = settings_io.read_setting(view._manager, setting)
        await pilot.press("space")
        after = settings_io.read_setting(view._manager, setting)
        assert before is False and after is False
        assert view._error == NEUTRAL

        await pilot.press("r")
        assert settings_io.read_setting(view._manager, setting) is False
        assert view._error == NEUTRAL


@pytest.mark.asyncio
async def test_the_neutral_sentence_renders_for_a_confirmed_kill() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        availability.record(
            view._manager.config_dir,
            visible=True,
            state="ready",
            checked_at=1.0,
            read_only=True,
            note=NEUTRAL,
        )
        _select(view, _bridge_row_index(view, "team_bridge.enabled"))
        assert view._detail_text.plain == NEUTRAL


@pytest.mark.asyncio
async def test_the_configured_cos_row_keeps_the_wake_explanation() -> None:
    """D1: off-default used to shed the whole sentence for `default: —`."""

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        availability.record(view._manager.config_dir, visible=True, state="ready", checked_at=1.0)

        # A configured (off-default) CoS value is exactly the state that used
        # to render `default: —   team_bridge.session` with no explanation.
        settings_io.write_setting(
            view._manager, settings_io.BY_KEY["team_bridge.session"], "cos-1234567890"
        )
        _select(view, _bridge_row_index(view, "team_bridge.session"))

        assert "badge only" in view._detail_text.plain
        assert "eligible organization" not in view._detail_text.plain


@pytest.mark.asyncio
async def test_the_default_off_helper_wraps_instead_of_losing_its_tail() -> None:
    """D4 at 60x20: the 74-cell helper must survive whole (two band rows)."""

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        availability.record(view._manager.config_dir, visible=True, state="ready", checked_at=1.0)
        index = _bridge_row_index(view, "team_bridge.enabled")
        _select(view, index)

        setting = settings_io.BY_KEY["team_bridge.enabled"]
        plain = view._detail_text.plain
        # The full sentence is on the band — wrapped, not truncated.
        assert " ".join(plain.split("\n")) == setting.help
        assert "eligible organization." in plain
        assert view._detail_two_line is True
        assert view._detail.styles.padding.top == 0

        # Leaving to a row without a wrap rung restores the band's blank top
        # row (no layout drift).
        other = next(
            index
            for index, row in enumerate(view._rows)
            if row.setting is not None and row.setting.section != "team_bridge"
        )
        _select(view, other)
        assert view._detail_two_line is False
        assert view._detail.styles.padding.top == 1


@pytest.mark.asyncio
async def test_the_bridge_explanation_uses_the_accessible_ink() -> None:
    """D1's ink: the H1 sentences resolve the `muted` rung, not `faint`/`dim`."""

    from local_operator.tui import theme as theme_mod

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        view = app.query_one(SettingsView)
        availability.record(view._manager.config_dir, visible=True, state="ready", checked_at=1.0)
        index = _bridge_row_index(view, "team_bridge.enabled")
        _select(view, index)

        from rich.color import Color as RichColor

        expected_muted = RichColor.parse(str(theme_mod.semantic_color("muted")))
        # getattr: a span's style is typed Style | str, and only a resolved
        # Style carries a Color; a raw string compares None and fails the rung.
        detail_spans = view._detail_text.spans
        assert any(getattr(s.style, "color", None) == expected_muted for s in detail_spans)
        # The value cell takes the same rung on these rows.
        row_text = view._row_text(view._rows[index], index, 100)
        row_spans = row_text.spans
        assert any(getattr(s.style, "color", None) == expected_muted for s in row_spans)
