"""The `/tag` and `/tags` command paths over the real app and store.

The store's own toggle/primary contract is pinned in
``tests/unit/session/test_session_tags.py`` and the sidebar's rows in
``test_sidebar_tag_groups.py``. This file is the seam between them: the real
``OperatorApp`` fills the argument list from the store, and one activation
writes one assignment.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

import pytest

from local_operator.paths import config_dir
from local_operator.resume import SessionRow
from local_operator.session.protocol import RuntimeLocality
from local_operator.session.session_tags import (
    read_categories,
    read_entity_tags,
    set_entity_tags,
    write_categories,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.session_catalog import CatalogEntry
from local_operator.tui.sidebar_tags import read_tag_groups, write_tag_groups
from local_operator.tui.widgets.session_sidebar import SessionSidebar
from tests.unit.tui.test_app_pilot import FakeProviderController, FakeSession


class _TagSession(FakeSession):
    """A real 12-hex session id, which the tag store validates."""

    @property
    def session_id(self) -> str:
        return "abcdef123456"


class _RemoteTagSession(_TagSession):
    """A mesh viewer: a real id, but the runtime lives on another device."""

    runtime_locality: RuntimeLocality = "another-machine"


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


def _app(session: FakeSession | None = None) -> OperatorApp:
    bound = session if session is not None else _TagSession()

    async def factory() -> FakeSession:
        return bound

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


@pytest.mark.asyncio
async def test_tag_keyboard_arrows_enter_escape_and_inline_draft() -> None:
    """The key contract the design preflight names, on the real composer.

    Arrows move the highlight without writing; Escape closes with the draft
    intact and zero writes; Enter toggles once; an inline `/tag` inside a draft
    opens the list without consuming the prose around it.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot)
        review = write_categories(config_dir(), [(None, "Review")])[0]
        ops = write_categories(
            config_dir(),
            [(review["id"], "Review"), (None, "Ops")],
        )[1]
        set_entity_tags(
            config_dir(),
            "session",
            "abcdef123456",
            [review["id"], ops["id"]],
            review["id"],
        )
        editor = app._editor()
        before = read_entity_tags(config_dir(), "session", "abcdef123456")

        editor._set_text_and_caret("/tag ", len("/tag "))
        await pilot.pause()
        await pilot.pause()
        picker = editor.picker
        assert picker.is_open()
        first = picker.highlighted_name()
        await pilot.press("down")
        await pilot.pause()
        assert picker.highlighted_name() != first
        assert read_entity_tags(config_dir(), "session", "abcdef123456") == before

        await pilot.press("escape")
        await pilot.pause()
        assert not picker.is_open()
        assert editor.text == "/tag "
        assert read_entity_tags(config_dir(), "session", "abcdef123456") == before

        editor._set_text_and_caret("/tag Review", len("/tag Review"))
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        # Enter toggled the APPLIED primary OFF once; the oldest survivor is
        # promoted, which is the same rule the store unit tests pin.
        assert read_entity_tags(config_dir(), "session", "abcdef123456") == {
            "tags": [ops["id"]],
            "primary": ops["id"],
            "revision": 2,
        }

        editor._set_text_and_caret("keep this draft /tag", len("keep this draft /tag"))
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert "keep this draft" in editor.text
        assert editor.picker.is_open()
        await pilot.press("escape")
        await pilot.pause()
        assert editor.text.startswith("keep this draft")


@pytest.mark.asyncio
async def test_remote_viewer_refuses_the_write_and_grouping_agrees() -> None:
    """The two surfaces must answer the same question the same way.

    The store is local to THIS config root and ``_tag_group_snapshot`` skips
    every remote row's assignment, so accepting a write on a mesh viewer would
    receipt "applied" while the sidebar painted the same conversation under
    Untagged. The refusal is the surface that has no bucket to show.
    """
    app = _app(_RemoteTagSession())
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot)
        notices, notice = _notices()
        unavailable = "Tags aren't available for this conversation."

        app._cmd_tag("Deep Work", notice)
        assert notices == [(unavailable, "warning")]
        # Nothing was minted: no registry entry, no assignment file.
        assert read_categories(config_dir()) == []
        assert not (config_dir() / "tags" / "session" / "abcdef123456.json").exists()
        # The picker says the same thing, so no "[x] applied" row can exist.
        assert app._tag_argument_choices() == ([], unavailable)

        # A local row still groups, so the refusal is about locality rather
        # than grouping being off in this fixture.
        review = write_categories(config_dir(), [(None, "Review")])[0]
        set_entity_tags(config_dir(), "session", "111111111111", [review["id"]], review["id"])
        entries = [
            CatalogEntry(
                SessionRow("abcdef123456", 20, "Remote work", live_state="busy", locality="remote")
            ),
            CatalogEntry(SessionRow("111111111111", 10, "Local work", live_state="busy")),
        ]
        categories, primaries = app._tag_group_snapshot(config_dir(), entries)
        assert primaries == {"111111111111": review["id"]}

        sidebar = SessionSidebar()
        sidebar.set_entries(entries)
        sidebar.set_tag_groups(True, categories, primaries)
        painted = sidebar._display_rows()
        assert [kind for kind, _entry in painted] == [
            "header:active",
            "blank",
            f"header:tag:{review['id']}",
            "entry",
            "blank",
            "header:tag:untagged",
            "entry",
        ]
        assert [entry.id for _kind, entry in painted if entry is not None] == [
            "111111111111",
            "abcdef123456",
        ]


@pytest.mark.asyncio
async def test_partial_create_reports_created_not_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """The receipt for the half that landed: created, not applied.

    A flat failure would invite a retry that duplicates the name; the exception
    carries which half landed and the handler must say so. The category really
    survives, which is what makes retrying the APPLY the right next move.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot)
        from local_operator.session import session_tags

        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        original = session_tags._set_entity_tags
        monkeypatch.setattr(session_tags, "_set_entity_tags", fail)
        notices, notice = _notices()
        app._cmd_tag("Deep Work", notice)

        assert notices == [("Tag created, but couldn't apply it. Retry applying.", "warning")]
        assert [item["name"] for item in read_categories(config_dir())] == ["Deep Work"]
        assert read_entity_tags(config_dir(), "session", "abcdef123456")["tags"] == []

        # Retrying after the write path recovers APPLIES the surviving name
        # instead of minting a second category. Only the write seam is
        # restored: `monkeypatch.undo()` would also revert the isolated
        # HOME/config this test's autouse fixture installed.
        monkeypatch.setattr(session_tags, "_set_entity_tags", original)
        app._cmd_tag("Deep Work", notice)
        registry = read_categories(config_dir())
        assert [item["name"] for item in registry] == ["Deep Work"]
        assert read_entity_tags(config_dir(), "session", "abcdef123456")["tags"] == [
            registry[0]["id"]
        ]


@pytest.mark.asyncio
async def test_tags_on_during_an_in_flight_poll_never_paints_all_untagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mode must travel WITH the bucket map it was read under.

    A flat poll's snapshot has no registry; adopting it as "on" would file
    every row under Untagged for a frame until the deferred refresh landed.
    The flip is injected after ``collect`` returns (the real window), so this
    fails against any adoption that re-reads ``self._tag_groups``.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(app, pilot)
        app._session_sidebar.set_open(True)
        review = write_categories(config_dir(), [(None, "Review")])[0]
        set_entity_tags(config_dir(), "session", "abcdef123456", [review["id"]], review["id"])
        for _ in range(120):
            await pilot.pause()
            if not app._sidebar_refresh_pending:
                break

        adopted: list[tuple[bool, tuple[tuple[str, str], ...]]] = []
        original = SessionSidebar.set_tag_groups

        def spy(self, enabled, categories=(), primaries=None):
            adopted.append((bool(enabled), tuple(categories)))
            return original(self, enabled, categories, primaries)

        monkeypatch.setattr(SessionSidebar, "set_tag_groups", spy)
        real_to_thread = asyncio.to_thread
        state = {"flipped": False}

        async def to_thread_then_flip(func, /, *args, **kwargs):
            result = await real_to_thread(func, *args, **kwargs)
            if not state["flipped"] and getattr(func, "__name__", "") == "collect":
                # The operator's `/tags on` lands while this poll is in flight.
                state["flipped"] = True
                app._tag_groups = True
                app._sidebar_refresh_again = True
            return result

        monkeypatch.setattr(asyncio, "to_thread", to_thread_then_flip)
        app._refresh_sidebar()
        for _ in range(200):
            await pilot.pause()
            if (
                state["flipped"]
                and not app._sidebar_refresh_pending
                and app._session_sidebar._tag_grouping
            ):
                break

        assert state["flipped"], "the race never ran"
        ungrouped = [entry for entry in adopted if entry[0] and not entry[1]]
        assert ungrouped == [], "an on-adoption carried no registry: every row would paint Untagged"
        assert app._session_sidebar._tag_grouping is True
        assert app._session_sidebar._tag_categories == ((review["id"], "Review"),)
        # The deferred pass is the one that lands the registry; the point is
        # that no earlier adoption paired "on" with an empty bucket map.
        assert (True, ((review["id"], "Review"),)) in adopted
