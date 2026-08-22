"""`/team chart` routing dispatch, picker collision, and the widget's mode.

The routing dispatch table is pinned here the way ``test_slash_echo`` pins the
echo policy: each `/team …` form maps to exactly one outcome (a chart opened
for a named team, the talk-to path untouched, the `=` escape, the bare form
falling back to the attached team). The crux the design calls out — that adding
`chart` as a first-argument subcommand must NOT change `/team foo do x` and must
not hide a team literally named `chart` — is asserted directly.

The widget's mode is exercised through the real ``OperatorApp`` (not a bare
host) so the screen invariants hold: the SCREEN must not scroll (the chart
scrolls inside its own body; the dock is pinned), Esc restores the transcript,
and a wide org makes the BODY scroll (which is expected here, unlike the main
screen where a horizontal scrollbar is a bug).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tui.app import ORG_CHART_LAYOUT_CLASS, OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _registry(*teams: TeamEditFields) -> TeamRegistry:
    reg = TeamRegistry(Path(tempfile.mkdtemp()))
    for fields in teams:
        reg.create_team(fields)
    return reg


def _nested_registry() -> TeamRegistry:
    return _registry(
        TeamEditFields(
            name="pod-a",
            manager="mgr-a",
            members=[TeamMember(role="coder"), TeamMember(role="reviewer", count=2)],
        ),
        TeamEditFields(name="pod-b", manager="mgr-b", members=[TeamMember(role="scout")]),
        TeamEditFields(
            name="org",
            manager="director",
            members=[
                TeamMember(role="pod-a", kind="team"),
                TeamMember(role="pod-b", kind="team", count=2),
            ],
        ),
    )


def _user_rows(app: OperatorApp) -> list[str]:
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock)
    ]


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


# -- routing dispatch table -------------------------------------------------


@pytest.mark.asyncio
async def test_chart_name_opens_the_chart() -> None:
    session = FakeSession()
    session.team_registry = _nested_registry()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/team chart org")
        await pilot.pause()
        assert app._org_chart_view is not None
        assert app._org_chart_view.team_name == "org"
        # The chart is a MODE, so no user row and no prompt to the model.
        assert _user_rows(app) == []
        assert session.prompts == []


@pytest.mark.asyncio
async def test_talk_path_is_unchanged_by_the_subcommand() -> None:
    """`/team foo do x` must be byte-for-byte the pre-chart behaviour."""
    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="foo", manager="manager", members=[TeamMember(role="coder")])
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/team foo do the thing")
        await pilot.pause()
        assert app._org_chart_view is None
        assert [t.name for t in session.attached_teams] == ["foo"]
        assert session.prompts == ["do the thing"]


@pytest.mark.asyncio
async def test_chart_chart_charts_a_team_named_chart() -> None:
    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="chart", manager="m", members=[TeamMember(role="coder")])
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/team chart chart")
        await pilot.pause()
        assert app._org_chart_view is not None
        assert app._org_chart_view.team_name == "chart"


@pytest.mark.asyncio
async def test_equals_escape_talks_to_a_team_named_chart() -> None:
    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="chart", manager="m", members=[TeamMember(role="coder")])
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/team =chart do work")
        await pilot.pause()
        # No chart: the leading `=` routes to the talk-to path for team `chart`.
        assert app._org_chart_view is None
        assert [t.name for t in session.attached_teams] == ["chart"]
        assert session.prompts == ["do work"]


@pytest.mark.asyncio
async def test_bare_chart_uses_the_attached_team() -> None:
    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="org", manager="director", members=[TeamMember(role="coder")])
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        # Attach a team first, then bare `/team chart`.
        app._run_slash_command("/team org do work")
        await pilot.pause()
        app._run_slash_command("/team chart")
        await pilot.pause()
        assert app._org_chart_view is not None
        assert app._org_chart_view.team_name == "org"


@pytest.mark.asyncio
async def test_bare_chart_with_no_attached_team_is_an_error_not_a_crash() -> None:
    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="org", manager="director", members=[TeamMember(role="coder")])
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/team chart")
        await pilot.pause()
        assert app._org_chart_view is None  # nothing to chart, no mode entered


@pytest.mark.asyncio
async def test_chart_unknown_team_is_an_error() -> None:
    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="org", manager="d", members=[TeamMember(role="coder")])
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/team chart nope")
        await pilot.pause()
        assert app._org_chart_view is None


# -- picker collision -------------------------------------------------------


@pytest.mark.asyncio
async def test_picker_first_slot_offers_chart_subcommand_then_teams() -> None:
    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="org", manager="director", members=[TeamMember(role="coder")]),
        TeamEditFields(name="chart", manager="m", members=[TeamMember(role="coder")]),
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.text = "/team "
        editor.cursor_location = (0, len(editor.text))
        await pilot.pause()
        await pilot.pause()
        rows = [(c.name, c.detail) for c in editor.picker._choices]
    # The subcommand row is FIRST and tagged.
    assert rows[0] == ("chart", "subcommand")
    # A team literally named `chart` still appears as its own row (not hidden).
    assert ("chart", "2 roles · led by m") in rows
    assert any(name == "org" for name, _ in rows)


@pytest.mark.asyncio
async def test_picker_second_slot_reoffers_teams_feeding_the_chart() -> None:
    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="org", manager="director", members=[TeamMember(role="coder")]),
        TeamEditFields(name="chart", manager="m", members=[TeamMember(role="coder")]),
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.text = "/team chart "
        editor.cursor_location = (0, len(editor.text))
        await pilot.pause()
        await pilot.pause()
        names = [c.name for c in editor.picker._choices]
    # Second slot: `chart <name>` compound rows that complete to the chart.
    assert "chart org" in names
    assert "chart chart" in names


# -- widget mode invariants -------------------------------------------------


@pytest.mark.asyncio
async def test_mode_does_not_scroll_the_screen_and_esc_restores() -> None:
    session = FakeSession()
    session.team_registry = _nested_registry()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._open_org_chart_view("org")
        await pilot.pause()
        await pilot.pause()
        view = app._org_chart_view
        assert view is not None
        assert app.screen.has_class(ORG_CHART_LAYOUT_CLASS)
        # The SCREEN must not scroll — the chart scrolls inside its body, the
        # dock is pinned. This is the AGENTS.md invariant, and it holds in this
        # mode exactly as on the main screen.
        assert app.screen.virtual_size.height <= app.screen.size.height
        assert app.screen.virtual_size.width <= app.screen.size.width
        # Transcript hidden while the mode is up.
        assert app._transcript_view().display is False
        # Esc leaves and restores.
        await pilot.press("escape")
        await pilot.pause()
        assert app._org_chart_view is None
        assert app._transcript_view().display is True
        assert not app.screen.has_class(ORG_CHART_LAYOUT_CLASS)


@pytest.mark.asyncio
async def test_wide_org_scrolls_the_body_not_the_screen() -> None:
    session = FakeSession()
    session.team_registry = _nested_registry()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 20)) as pilot:
        await _boot(pilot, app)
        app._open_org_chart_view("org")
        await pilot.pause()
        await pilot.pause()
        view = app._org_chart_view
        assert view is not None
        # Zoom to detailed so the canvas is wider than the small viewport.
        await pilot.press("plus")
        await pilot.press("plus")
        await pilot.pause()
        canvas_w, _canvas_h = view.canvas_size
        # The canvas exceeds the body viewport width → the body's virtual size
        # is wider than its own size (horizontal scroll, EXPECTED here).
        assert view._body.virtual_size.width >= view._body.size.width
        assert canvas_w > view._body.size.width
        # But the SCREEN still does not scroll.
        assert app.screen.virtual_size.width <= app.screen.size.width


@pytest.mark.asyncio
async def test_zoom_keys_change_the_tier() -> None:
    session = FakeSession()
    session.team_registry = _nested_registry()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._open_org_chart_view("org")
        await pilot.pause()
        view = app._org_chart_view
        assert view is not None
        assert view._tier == 1  # standard default
        await pilot.press("plus")
        await pilot.pause()
        assert view._tier == 2
        await pilot.press("minus")
        await pilot.press("minus")
        await pilot.pause()
        assert view._tier == 0  # clamps at outline
