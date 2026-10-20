"""The `/tag` and `/tags` command paths over the real app and store.

The store's own toggle/primary contract is pinned in
``tests/unit/session/test_session_tags.py`` and the sidebar's rows in
``test_sidebar_tag_groups.py``. This file is the seam between them: the real
``OperatorApp`` fills the argument list from the store, and one activation
writes one assignment.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from local_operator.paths import config_dir
from local_operator.session.session_tags import (
    read_categories,
    read_entity_tags,
    set_entity_tags,
    write_categories,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.sidebar_tags import read_tag_groups, write_tag_groups
from tests.unit.tui.test_app_pilot import FakeProviderController, FakeSession


class _TagSession(FakeSession):
    """A real 12-hex session id, which the tag store validates."""

    @property
    def session_id(self) -> str:
        return "abcdef123456"


@pytest.fixture(autouse=True)
def isolated_tags(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)


def _notices() -> tuple[list[tuple[str, str]], Callable[..., None]]:
    seen: list[tuple[str, str]] = []

    def notice(text: str, level: str = "info") -> None:
        seen.append((text, level))

    return seen, notice


def _app() -> OperatorApp:
    session = _TagSession()

    async def factory() -> FakeSession:
        return session

    return OperatorApp(factory, provider_controller=FakeProviderController())


async def _settle(app: OperatorApp, pilot) -> None:
    # The boot worker binds ``_session`` after the first paint. Poll it rather
    # than assuming a pause count: a cold shard under fleet load takes longer,
    # and every assertion here is about the bound conversation.
    for _ in range(60):
        await pilot.pause()
        if getattr(app, "_session", None) is not None:
            return
    raise AssertionError("the app never bound its session")


@pytest.mark.asyncio
async def test_bare_tag_opens_the_current_conversations_list() -> None:
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot)
        tag = write_categories(config_dir(), [(None, "Review")])[0]
        set_entity_tags(config_dir(), "session", "abcdef123456", [tag["id"]], tag["id"])
        notices, notice = _notices()
        app._cmd_tag("", notice)
        await pilot.pause()

        editor = app._editor()
        assert editor.text == "/tag "
        rows = [
            (name, choice)
            for name, choice in editor.picker.suggestions()
            if isinstance(choice, ArgumentChoice)
        ]
        assert [(name, choice.display, choice.detail) for name, choice in rows] == [
            ("Review", "[x] Review", "Primary")
        ]
        assert "1/8 applied" in editor.picker._notice
        assert notices == []


@pytest.mark.asyncio
async def test_selecting_an_existing_row_toggles_once() -> None:
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot)
        tag = write_categories(config_dir(), [(None, "Review")])[0]
        notices, notice = _notices()

        app._cmd_tag("Review", notice)
        state = read_entity_tags(config_dir(), "session", "abcdef123456")
        assert state["tags"] == [tag["id"]]
        assert notices[-1][0] == "Added “Review”."

        app._cmd_tag("Review", notice)
        state = read_entity_tags(config_dir(), "session", "abcdef123456")
        assert state["tags"] == []
        assert notices[-1][0] == "Removed “Review”."


@pytest.mark.asyncio
async def test_exact_new_name_creates_and_applies_in_one_activation() -> None:
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot)
        notices, notice = _notices()
        app._cmd_tag("Deep Work", notice)

        registry = read_categories(config_dir())
        assert [item["name"] for item in registry] == ["Deep Work"]
        assert read_entity_tags(config_dir(), "session", "abcdef123456") == {
            "tags": [registry[0]["id"]],
            "primary": registry[0]["id"],
            "revision": 1,
        }
        assert notices[-1][0] == "Created and applied “Deep Work”."

        # Reopening offers the create action for a prefix of an existing name,
        # but an EXACT existing name never grows a duplicate create row.
        rows, _line = app._tag_choices_with_create("Deep")
        assert [row.name for row in rows].count("Deep Work") == 1
        assert all(row.name != "Deep Work" or row.display.startswith("[x]") for row in rows)
        rows, _line = app._tag_choices_with_create("deep work")
        assert [row.name for row in rows] == ["Deep Work"]


@pytest.mark.asyncio
async def test_tags_on_off_persists_and_returns_to_flat() -> None:
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot)
        notices, notice = _notices()

        app._cmd_tags("on", notice)
        assert read_tag_groups(config_dir()) is True
        assert app._tag_groups is True
        assert notices[-1][0] == "Sidebar tag groups on — sections now split by primary tag."

        app._cmd_tags("off", notice)
        assert read_tag_groups(config_dir()) is False
        assert app._tag_groups is False
        assert notices[-1][0] == "Sidebar tag groups off — sections are flat again."

    # The preference survives a restart: a fresh app reads the SAME store.
    write_tag_groups(config_dir(), True)
    fresh = _app()
    async with fresh.run_test(size=(100, 30)) as pilot:
        await _settle(fresh, pilot)
        assert fresh._tag_groups is True


@pytest.mark.asyncio
async def test_enter_on_a_typed_name_toggles_and_creates_once() -> None:
    """The real composer path, not the handler: one Enter is one commit.

    The exact typed name makes the highlighted row unambiguous, so this is the
    frame in which Enter runs rather than merely completing. It must work for
    an existing tag (toggle) and for a novel exact name (create + apply), with
    no second confirmation and no second assignment.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot)
        tag = write_categories(config_dir(), [(None, "Review")])[0]
        editor = app._editor()

        editor.load_text("/tag Review")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert read_entity_tags(config_dir(), "session", "abcdef123456")["tags"] == [tag["id"]]
        assert editor.text == ""

        editor.load_text("/tag Review")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert read_entity_tags(config_dir(), "session", "abcdef123456")["tags"] == []

        editor.load_text("/tag Brand New")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        state = read_entity_tags(config_dir(), "session", "abcdef123456")
        registry = read_categories(config_dir())
        assert [item["name"] for item in registry] == ["Review", "Brand New"]
        brand_new = next(item["id"] for item in registry if item["name"] == "Brand New")
        assert state["tags"] == [brand_new]
