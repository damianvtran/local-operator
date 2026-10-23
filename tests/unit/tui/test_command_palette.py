"""The app's ``ctrl+p`` command palette: catalogue, ranking, dispatch, states.

Two layers, tested for different reasons:

* ``build_palette_entries`` / ``rank_entries`` are pure, so the catalogue's
  shape — derived from the registries, duplicate labels kept distinct, an empty
  catalogue, the recent-sessions discovery cap — is asserted without a Screen.
* The widget tests drive the REAL ``OperatorApp`` (the lightweight hosts in
  other files declare no ``CSS_PATH``, so they cannot prove focus or a screen
  push). They cover the contract a provider cannot state on its own: ``ctrl+p``
  opens the app palette, a query filters it, Escape returns focus to the
  composer, and a selection runs through the typed dispatch path.
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.command import CommandList
from textual.fuzzy import Matcher

from local_operator import settings_io
from local_operator.slash_commands import SLASH_COMMANDS
from local_operator.tui import command_palette as palette_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.command_palette import (
    KIND_AGENT,
    KIND_COMMAND,
    KIND_SESSION,
    KIND_SETTING,
    KIND_TEAM,
    OperatorCommandPalette,
    PaletteEntry,
    build_palette_entries,
    palette_is_open,
    rank_entries,
)
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.conftest import await_session_boot
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class _Row:
    """The ``resume.SessionRow`` fields the catalogue reads."""

    def __init__(self, name: str, session_id: str) -> None:
        self.name = name
        self.id = session_id


def _sections() -> list[Any]:
    return list(settings_io.SECTIONS)


def _catalogue(**overrides: Any) -> list[PaletteEntry]:
    kwargs: dict[str, Any] = {
        "slash_commands": SLASH_COMMANDS,
        "sections": _sections(),
        "session_rows": (),
        "agent_rows": (),
        "team_names": (),
    }
    kwargs.update(overrides)
    return build_palette_entries(**kwargs)


class TestCatalogue:
    def test_commands_come_from_the_slash_registry_alone(self) -> None:
        """No second command table: one row per registry entry, same names."""
        entries = [e for e in _catalogue() if e.kind == KIND_COMMAND]
        assert [e.slash for e in entries] == [f"/{c.name}" for c in SLASH_COMMANDS]
        assert [e.label for e in entries] == [f"/{c.name}" for c in SLASH_COMMANDS]

    def test_settings_rows_carry_the_section_identity_not_a_slash_line(self) -> None:
        entries = [e for e in _catalogue() if e.kind == KIND_SETTING]
        assert [e.section for e in entries] == [s.name for s in settings_io.SECTIONS]
        assert all(not e.slash for e in entries), "a settings row must not route as a slash line"

    def test_agent_rows_are_addressed_by_registry_key_not_display_label(self) -> None:
        entries = [
            e
            for e in _catalogue(agent_rows=[("coder", "Coder (custom label)", "", "writes code")])
            if e.kind == KIND_AGENT
        ]
        assert entries[0].label == "Coder (custom label)"
        assert entries[0].slash == "/agent coder"

    def test_team_rows_route_through_the_team_command(self) -> None:
        entries = [e for e in _catalogue(team_names=["lopdev"]) if e.kind == KIND_TEAM]
        assert entries[0].slash == "/team lopdev"

    def test_duplicate_labels_remain_distinct_rows(self) -> None:
        """Two conversations can open with the same words; both must be reachable.

        The ids are carried in the description and the route, so one selection
        cannot resume the other.
        """
        rows = [_Row("same opening", "aaaa1111"), _Row("same opening", "bbbb2222")]
        entries = [e for e in _catalogue(session_rows=rows) if e.kind == KIND_SESSION]
        assert len(entries) == 2
        assert {e.slash for e in entries} == {"/resume aaaa1111", "/resume bbbb2222"}
        labels = [e.label for e in entries]
        assert labels == ["same opening", "same opening"]

    def test_empty_catalogue_is_empty_not_an_error(self) -> None:
        assert (
            build_palette_entries(slash_commands=[], sections=[], session_rows=(), agent_rows=())
            == []
        )

    def test_session_discovery_is_capped_but_search_reaches_every_row(self) -> None:
        rows = [_Row(f"conversation {i}", f"{i:08d}") for i in range(30)]
        entries = [e for e in _catalogue(session_rows=rows) if e.kind == KIND_SESSION]
        assert len(entries) == 30, "the search catalogue must be uncapped"
        assert sum(1 for e in entries if e.discovery) == 12
        hits = rank_entries(entries, Matcher("conversation 29"))
        assert hits and hits[0][0].slash == "/resume 00000029"


class TestRanking:
    def test_query_returns_best_score_first(self) -> None:
        hits = rank_entries(_catalogue(), Matcher("review"))
        assert hits
        assert hits[0][0].slash == "/review"
        assert all(score > 0 for _entry, score in hits)

    def test_alias_matches_without_being_displayed(self) -> None:
        """``quit`` finds ``/exit`` — the alias is identity, not decoration."""
        hits = rank_entries(_catalogue(), Matcher("quit"))
        assert [e.slash for e, _score in hits][:1] == ["/exit"]

    def test_equal_scores_keep_the_catalogue_order(self) -> None:
        """Stable sort, so equal-scoring rows stay in the order they were built.

        Two conversations with the same opening line score identically; the
        newer row must stay first rather than swapping under the cursor.
        """
        rows = [_Row("same opening", "aaaa1111"), _Row("same opening", "bbbb2222")]
        entries = [e for e in _catalogue(session_rows=rows) if e.kind == KIND_SESSION]
        hits = [e.slash for e, _score in rank_entries(entries, Matcher("same"))]
        assert hits == ["/resume aaaa1111", "/resume bbbb2222"]

    def test_no_match_is_an_empty_list(self) -> None:
        assert rank_entries(_catalogue(), Matcher("zzzzzzzz-no-such-thing")) == []


class TestBuildSeam:
    def test_palette_entries_reads_the_same_registries_the_surfaces_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = OperatorApp(lambda: _factory(FakeSession()))
        monkeypatch.setattr(app, "_agent_profile_rows", lambda: [("coder", "Coder", "", "d")])
        monkeypatch.setattr(app, "_known_team_names", lambda: ["lopdev"])
        entries = app._palette_entries()
        assert any(e.slash == "/agent coder" for e in entries)
        assert any(e.slash == "/team lopdev" for e in entries)
        assert any(e.slash == "/help" for e in entries)
        assert any(e.section == "model" for e in entries)


class TestWidget:
    @pytest.fixture
    def app(self) -> OperatorApp:
        return OperatorApp(lambda: _factory(FakeSession()))

    @pytest.mark.asyncio
    async def test_ctrl_p_opens_the_app_palette_and_escape_restores_focus(
        self, app: OperatorApp
    ) -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await await_session_boot(app)
            # A draft in the composer must survive the round trip: the palette
            # is a transient surface over the screen, not a mode that owns the
            # text box the user was typing in.
            await pilot.press(*"draft text")
            await pilot.pause()
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert palette_is_open(app)
            await pilot.press("escape")
            await pilot.pause()
            assert not palette_is_open(app)
            assert isinstance(
                app.focused, Editor
            ), "Escape must put the operator back in the composer, not nowhere"
            assert app.focused.text == "draft text"

    @pytest.mark.asyncio
    async def test_typing_filters_to_a_settings_section_and_selecting_opens_it(
        self, app: OperatorApp
    ) -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await await_session_boot(app)
            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press(*"wake trig")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert not palette_is_open(app)
            view = app._settings_view
            assert view is not None, "selecting a settings hit must open the settings page"
            current = view._rows[view._selected]
            assert (
                current.setting is not None
            ), "the landing must be inside the section, not on its header"
            assert current.setting.section == "wakes"

    @pytest.mark.asyncio
    async def test_a_command_hit_runs_through_the_typed_dispatch_path(
        self, app: OperatorApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(app, "_run_slash_command", lambda text: calls.append(text))
        async with app.run_test(size=(100, 30)) as pilot:
            await await_session_boot(app)
            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press(*"/help")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert calls == ["/help"]

    @pytest.mark.asyncio
    async def test_discovery_keeps_the_framework_actions_then_the_app_catalogue(
        self, app: OperatorApp
    ) -> None:
        """System actions are delegated to, not dropped, and stay first.

        Their position is the one thing a single provider buys: with several
        providers all discovery hits score 0 and ``App.COMMANDS`` is a set, so
        these rows would move between launches.
        """
        async with app.run_test(size=(100, 30)) as pilot:
            await await_session_boot(app)
            await pilot.press("ctrl+p")
            await pilot.pause()
            labels = [
                str(option.prompt).split("\n", 1)[0]
                for option in app.screen.query_one(CommandList).options
            ]
            assert "Quit" in labels and "Theme" in labels, labels
            assert "/help" in labels
            assert labels.index("Quit") < labels.index("/help")

    @pytest.mark.asyncio
    async def test_no_match_query_shows_the_localized_empty_state(self, app: OperatorApp) -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await await_session_boot(app)
            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press(*"zzzzzzzz")
            # The no-match row appears after the framework's half-second
            # countdown; a pause shorter than that would read the still-busy
            # state as "the palette said nothing".
            await pilot.pause(0.7)
            options = app.screen.query_one(CommandList).options
            texts = [str(option.prompt) for option in options]
            assert any("No matches found" in text for text in texts), texts

    @pytest.mark.asyncio
    async def test_an_empty_catalogue_opens_and_stays_quiet(
        self, app: OperatorApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No slash commands, sections, sessions or names: still a usable screen."""
        monkeypatch.setattr(app, "_palette_entries", lambda: [])
        async with app.run_test(size=(100, 30)) as pilot:
            await await_session_boot(app)
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert palette_is_open(app)
            await pilot.press(*"zz")
            await pilot.pause(0.7)
            await pilot.press("escape")
            await pilot.pause()
            assert not palette_is_open(app)


class TestFrameworkSeams:
    """The version-pinned Textual internals `OperatorCommandPalette` depends on."""

    def test_no_match_internals_still_exist(self) -> None:
        from textual.command import CommandPalette

        assert hasattr(CommandPalette, "_NO_MATCHES_COUNTDOWN")
        assert hasattr(CommandPalette, "_NO_MATCHES")
        assert issubclass(OperatorCommandPalette, CommandPalette)

    def test_placeholder_is_localized(self) -> None:
        assert palette_mod._t(palette_mod._p.placeholder()) in (
            "Search commands, settings, sessions, agents…",
        )
