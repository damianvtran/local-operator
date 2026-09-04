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


async def _type(pilot, text: str) -> None:
    """Type ``text`` into the focused editor via real key presses.

    Post-#250 the picker's argument detection is caret-anchored: it syncs during
    ``load_text`` while the caret is still at the origin, so setting ``editor.text``
    directly and then moving the caret no longer opens the argument list. Driving
    the buffer through real key presses is the path the app actually takes and the
    only one that opens the picker — the same approach ``test_slash_echo`` uses.
    """
    for char in text:
        await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
    await pilot.pause()
    await pilot.pause()


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
async def test_chart_subcommand_is_case_insensitive() -> None:
    """minor-1: `/team Chart <name>` routes to the chart, not the talk path.

    The picker and the editor's hint suppression fold case, so a capitalised
    `Chart` visually implies charting; the router must agree or it would attach
    a team literally named `Chart` while the UI said otherwise.
    """

    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="org", manager="director", members=[TeamMember(role="coder")])
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/team Chart org")
        await pilot.pause()
        assert app._org_chart_view is not None
        assert app._org_chart_view.team_name == "org"
        # Routed to the chart, not the talk path.
        assert session.prompts == []
        assert session.attached_teams == []


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
async def test_picker_first_slot_offers_teams_first_then_chart_subcommand() -> None:
    """The `/team ` first slot lists TEAM NAMES first, then the `chart` row.

    Team-first is the ordering #250's D5 regression pins: a bare `/team ` + Tab
    must complete the sole/first TEAM (the common action is talking to a team),
    never the subcommand — so the `chart` row is offered but is NOT the default
    completion. It stays discoverable (present in the list, and the normal
    matcher ranks it up the moment the user types `c`/`ch`/`chart`).
    """

    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="org", manager="director", members=[TeamMember(role="coder")]),
        TeamEditFields(name="chart", manager="m", members=[TeamMember(role="coder")]),
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        await _type(pilot, "/team ")
        editor = app.query_one(Editor)
        rows = [(c.name, c.detail) for c in editor.picker._choices]
    # Team rows come FIRST; the subcommand row is present but LAST (never the
    # default Tab completion — see test_completing_a_team_name_keeps_the_parked_hint).
    assert rows, "the team list must be populated"
    assert rows[0][0] != "chart", f"a team must lead, not the subcommand: {rows}"
    # The `chart` subcommand row is still offered — discoverable, ranked by query.
    assert ("chart", "subcommand") in rows
    assert rows[-1] == ("chart", "subcommand"), f"subcommand must rank last: {rows}"
    # U3: a team literally named `chart` still appears, completing to the `=chart`
    # ESCAPE (not the bare name, which would route to the subcommand); its detail
    # names the talk path, so the collision is resolvable from the picker.
    # Detail is the D1/D2 shape: roster size only (`member_count()`, manager
    # excluded), no "led by" — the manager's name is what made this column wide
    # enough to silence descriptions at ordinary widths.
    assert ("=chart", "talk to team · 1 role") in rows
    # A non-colliding team keeps its plain name.
    assert any(name == "org" for name, _ in rows)
    # The bare `chart` name is NOT offered as a team row (it would mis-route).
    assert ("chart", "1 role") not in rows


@pytest.mark.asyncio
async def test_bare_tab_completes_a_team_not_the_chart_subcommand() -> None:
    """A bare `/team ` + Tab fills the sole TEAM, not the `chart` subcommand.

    The collision-safety guarantee behind team-first: Tab never silently opens a
    chart when the user meant to message their team. Mirrors #250's
    test_completing_a_team_name_keeps_the_parked_hint invariant for our ordering.
    """

    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(name="security", manager="manager", members=[TeamMember(role="coder")]),
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        await _type(pilot, "/team ")
        editor = app.query_one(Editor)
        assert editor.picker.is_open(), "the team list must open"
        await pilot.press("tab")
        await pilot.pause()
        text = editor.text
    assert text == "/team security ", text


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
        app.query_one(Editor).focus()
        await _type(pilot, "/team chart ")
        editor = app.query_one(Editor)
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


@pytest.mark.parametrize("size", [(100, 30), (140, 40)])
@pytest.mark.asyncio
async def test_the_chart_takes_the_whole_view_when_opened_from_the_splash(
    size: tuple[int, int],
) -> None:
    """The chart gets the same geometry from the splash as over a conversation.

    ``Screen.boot`` is a whole second layout (centred width-clamped input card,
    plus rows reserved below it in the dock's padding), and this mode replaces
    the region it lays out. Applied together the chart got the leftovers around
    a card that still held its clamp — 26 of 38 rows at 140x40, and at 100x30 an
    input shell clamped to 73 cells at column 12 rather than 96 at column 1.

    BOTH dimensions are asserted because the collision is not the same shape at
    every size: at 100x30 the composition reserves no rows at all, so a
    rows-only assertion passes while the card is still visibly clamped over the
    page (review round 1, F1/F2).
    """
    session = FakeSession()
    session.team_registry = _nested_registry()

    async def measure(seed_conversation: bool) -> tuple[int, int, int, int]:
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await _boot(pilot, app)
            if seed_conversation:
                app._append_block(UserBlock("hello"))
                await pilot.pause()
                assert not app.screen.has_class("boot"), size
            else:
                assert app.screen.has_class("boot"), size
            app._open_org_chart_view("org")
            await pilot.pause()
            await pilot.pause()
            view = app._org_chart_view
            assert view is not None
            dock = app.query_one("#input-dock")
            shell = app.query_one("#input-shell")
            assert not app.screen.has_class("boot-card"), (
                f"{size}: the boot card's clamp survived into the chart "
                f"(#input-shell is {shell.size.width} cells at x={shell.region.x})"
            )
            assert dock.outer_size.height == dock.size.height, size
            assert app.screen.virtual_size.height <= app.screen.size.height, size
            return (view.size.height, view.size.width, shell.size.width, shell.region.x)

    assert await measure(False) == await measure(True), size


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
async def test_end_jumps_to_the_right_edge_on_a_wide_chart() -> None:
    """U10 — `End` reaches the far-right column even with no vertical travel.

    On a wide flat team the chart overflows only the X axis (max_scroll_y == 0).
    Textual's `scroll_end` passes x=0 there (its "end" is bottom-LEFT), so it was
    a no-op — the fix pins max_scroll_x/max_scroll_y explicitly. `Home` returns
    to the origin, `End` reaches the right edge, both in one press.
    """

    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(
            name="wide",
            manager="boss",
            members=[TeamMember(role=f"m{i}") for i in range(12)],
        )
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._open_org_chart_view("wide")
        await pilot.pause()
        await pilot.pause()
        view = app._org_chart_view
        assert view is not None
        # Wide-only overflow: the X axis has travel, the Y axis does not.
        assert view._body.max_scroll_x > 0
        assert view._body.max_scroll_y == 0
        # End jumps to the far-right column (the axis that overflows).
        await pilot.press("end")
        await pilot.pause()
        assert view._body.scroll_offset.x == view._body.max_scroll_x
        # Home returns to the origin.
        await pilot.press("home")
        await pilot.pause()
        assert view._body.scroll_offset.x == 0


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


@pytest.mark.asyncio
async def test_footer_advertises_scroll() -> None:
    """U1 — the scroll affordance is named in the footer, not left undiscoverable."""
    session = FakeSession()
    session.team_registry = _nested_registry()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._open_org_chart_view("org")
        await pilot.pause()
        await pilot.pause()
        view = app._org_chart_view
        assert view is not None
        hint_row = "".join(h.rendered() for h in view._hints.query(type(view._exit_hint)))
    assert "scroll" in hint_row
    assert "↔↕" in hint_row


@pytest.mark.asyncio
async def test_fit_never_collapses_a_flat_team_to_a_box() -> None:
    """U2 — `f` on a wide flat team keeps members visible, never a bare box.

    A 12-member team overflows 80×24. Pressing `f` must NOT land on outline
    (which folds members to a `·N` badge); it stays on a member-showing tier and
    lets horizontal scroll carry the overflow.
    """

    session = FakeSession()
    session.team_registry = _registry(
        TeamEditFields(
            name="wide",
            manager="boss",
            members=[TeamMember(role=f"m{i}") for i in range(12)],
        )
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        app._open_org_chart_view("wide")
        await pilot.pause()
        await pilot.pause()
        view = app._org_chart_view
        assert view is not None
        await pilot.press("f")
        await pilot.pause()
        # Member tier, not outline: the roster is still drawn as boxes.
        assert view._tier >= 1
        # Every member box present (12 members + manager + team header).
        assert view.last_result is not None
        labels = " ".join(b.label for b in view.last_result.boxes)
        assert "m11" in labels  # the last member is a real box, not a badge


@pytest.mark.asyncio
async def test_legend_toggles_with_question_mark() -> None:
    """U5/U6 — `?` reveals the glyph legend + (declared) gloss in-mode."""
    session = FakeSession()
    session.team_registry = _nested_registry()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._open_org_chart_view("org")
        await pilot.pause()
        await pilot.pause()
        view = app._org_chart_view
        assert view is not None
        assert view._legend.display is False
        await pilot.press("question_mark")
        await pilot.pause()
        assert view._legend.display is True
        legend_text = view._legend.render()
        plain = legend_text.plain if hasattr(legend_text, "plain") else str(legend_text)
    assert "manager" in plain
    assert "declared" in plain


@pytest.mark.asyncio
async def test_hints_shed_without_clipping_a_word_at_narrow_width() -> None:
    """D3 — the hint row sheds whole hints rather than clipping mid-word."""
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
        visible = [h for h in view._hints.query(type(view._exit_hint)) if h.display]
        row = "".join(h.rendered() for h in visible)
    # esc always survives; the row never ends mid-word like "conversatio".
    assert "esc" in row
    assert "conversatio" not in row or "conversation" in row
