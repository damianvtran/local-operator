"""The start-session picker (P5b): its rows, its card, and the flow over the page.

Three layers, split the way the rest of the projects surfaces split them:

* the PURE half — :func:`start_targets` and :func:`filter_start_targets` — is
  pinned without a terminal, because the rows' order and their addresses are
  the feature and neither needs a screen to be wrong;
* the card's geometry and grammar are driven over the real page (the app is the
  only host that loads the shipped stylesheet);
* the boot flow is driven through the app's own handler with the creation core
  and the runtime engagement REPLACED, because the real ones spawn a detached
  runtime and write a session directory — what this file can honestly assert is
  the ORDER and the refusal handling, which is what the slice is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.projects_start import (
    AGENT_SECTION,
    NO_AGENT_NOTE,
    NO_MATCH_NOTE,
    NO_TEAM_NOTE,
    PLAIN_LABEL,
    START_CARD_CHROME_ROWS,
    START_CARD_ROW_CAP,
    TEAM_SECTION,
    StartPickerCard,
    StartTarget,
    filter_start_targets,
    start_targets,
)
from local_operator.tui.widgets.projects_view import ProjectsViewStartRequested
from tests.unit.tui.test_projects_view import (  # noqa: E402
    _boot,
    _factory,
    _open,
    _ProjectSession,
    _registry,
)

#: A real-looking mint: the store validates ids as 12 hex characters, so a
#: readable placeholder would be refused by the LINK rather than by the test.
STARTED_ID = "ab12cd34ef56"


def _team(name: str, *, manager: str = "manager", counts: tuple[int, ...] = (1, 1)) -> dict:
    return {
        "name": name,
        "label": name.title(),
        "manager": manager,
        "members": [{"role": f"r{i}", "count": c, "kind": "agent"} for i, c in enumerate(counts)],
    }


def _agent(name: str, description: str = "") -> dict:
    return {"name": name, "label": name.title(), "description": description}


# -- the pure half ----------------------------------------------------------


def test_the_rows_lead_with_teams_then_agents_then_the_plain_session() -> None:
    """The spec's §7.6.1 order, and the plain row LAST (the honest tail)."""
    rows = start_targets(
        teams=[_team("lopdev", counts=(1, 2)), _team("core")],
        agents=[_agent("coder", "implements"), _agent("reviewer", "reviews")],
    )
    assert [row.kind for row in rows] == ["team", "team", "agent", "agent", "plain"]
    assert [row.label for row in rows] == ["Lopdev", "Core", "Coder", "Reviewer", PLAIN_LABEL]
    # The catalogue's order is preserved — this is NOT a second sort.
    assert [row.name for row in rows][:2] == ["lopdev", "core"]
    # A team row names its manager and how many member copies it runs; the
    # manager is excluded from the count (`Team.member_count`).
    assert rows[0].detail == "manager · 3 roles"
    assert rows[1].detail == "manager · 2 roles"
    # An agent row carries the registry's one-line description.
    assert rows[2].detail == "implements"
    # The plain row asks the create core for no attachment at all.
    assert rows[4].name == "" and rows[4].detail == ""


def test_an_empty_registry_still_offers_a_plain_session() -> None:
    """D7: the feature has to work on a fresh install with nothing registered."""
    rows = start_targets(teams=[], agents=[])
    assert rows == [StartTarget(kind="plain", name="", label=PLAIN_LABEL)]
    card = StartPickerCard(rows)
    # Both sections still say what is missing rather than vanishing.
    painted = [line.text for line in card.painted_lines()]
    assert NO_TEAM_NOTE in painted and NO_AGENT_NOTE in painted
    assert PLAIN_LABEL in painted


def test_rows_without_a_name_are_dropped() -> None:
    """A target is addressed BY NAME, so a nameless row could only be refused."""
    rows = start_targets(teams=[{"name": "  "}, _team("core")], agents=[{}])
    assert [row.name for row in rows] == ["core", ""]


def test_the_filter_is_a_subsequence_over_the_name_and_the_description() -> None:
    rows = start_targets(
        teams=[_team("lopdev")],
        agents=[_agent("reviewer", "adversarial reading of diffs")],
    )
    # Subsequence, not prefix: "rvw" reaches the reviewer.
    assert [row.name for row in filter_start_targets(rows, "rvw")] == ["reviewer"]
    # The DESCRIPTION is searched too — a description the filter cannot see is
    # one nobody can search by.
    assert [row.name for row in filter_start_targets(rows, "adversarial")] == ["reviewer"]
    assert filter_start_targets(rows, "") == rows


def test_the_filter_narrows_the_plain_row_too() -> None:
    """A filter the reader typed must not be answered by a row kept for them."""
    rows = start_targets(teams=[_team("lopdev")], agents=[])
    assert [row.kind for row in filter_start_targets(rows, "zzz")] == []
    assert [row.kind for row in filter_start_targets(rows, "plain")] == ["plain"]


# -- the card ---------------------------------------------------------------


def test_the_card_paints_a_header_per_section_and_never_selects_one() -> None:
    card = StartPickerCard(
        start_targets(teams=[_team("core")], agents=[_agent("coder", "implements")])
    )
    lines = card.painted_lines()
    assert [line.text for line in lines] == [
        TEAM_SECTION,
        "Core · manager · 2 roles",
        AGENT_SECTION,
        "Coder · implements",
        PLAIN_LABEL,
    ]
    # Headers and the plain row's own line: only real rows carry a row index.
    assert [line.header for line in lines] == [True, False, True, False, False]
    assert [line.row for line in lines] == [-1, 0, -1, 1, 2]
    # The block is what the card's height is made of: chrome + painted lines.
    assert card._visible_lines == len(lines)


def test_the_geometry_the_card_reserves_is_the_block_it_paints() -> None:
    """Chrome + painted lines is the card's whole height (P5a's D1 property)."""
    card = StartPickerCard(start_targets(teams=[_team(f"t{i}") for i in range(6)], agents=[]))
    card.set_available(START_CARD_CHROME_ROWS + 3)
    # Three rows plus the teams header is what fits in three lines' budget.
    assert card._visible_lines <= 3
    assert card._visible <= START_CARD_ROW_CAP
    # A ground too short for even one line leaves nothing selectable, and
    # `enter` then refuses rather than answering with an unpainted row.
    card.set_available(2)
    assert card.painted_range() == range(0, 0)


def test_the_window_slides_to_keep_the_selection_painted() -> None:
    rows = start_targets(teams=[_team(f"t{i:02d}") for i in range(8)], agents=[])
    card = StartPickerCard(rows)
    card.set_available(START_CARD_CHROME_ROWS + 3)
    for _ in range(6):
        card.action_move(1)
    painted = set(card.painted_range())
    assert card.index in painted
    assert card._top > 0


def test_a_create_in_flight_refuses_a_second_pick() -> None:
    card = StartPickerCard(start_targets(teams=[_team("core")], agents=[]))
    posted: list[object] = []
    card.post_message = posted.append  # type: ignore[method-assign]
    card.action_choose()
    assert [m for m in posted if isinstance(m, StartPickerCard.Chosen)]
    posted.clear()
    card.set_pending()
    assert card.pending
    card.action_choose()
    assert not [m for m in posted if isinstance(m, StartPickerCard.Chosen)]
    # A refusal LEAVES the pending state: every refusal this card can meet is
    # answered by picking another row or closing.
    card.show_refusal("could not start a session: no runtime")
    assert not card.pending
    card.action_choose()
    assert [m for m in posted if isinstance(m, StartPickerCard.Chosen)]


def test_a_filter_that_matches_nothing_says_so() -> None:
    """A mistyped filter is a keystroke to take back, not an empty card."""
    card = StartPickerCard(start_targets(teams=[_team("core")], agents=[]))
    card._rows = filter_start_targets(card._all, "zzz")
    assert card._rows == []
    # Nothing SELECTABLE is painted — a mistyped filter is a keystroke to take
    # back, and the note row (painted by `_repaint`) is what says so; that
    # sentence is a DIFFERENT one from the empty-registry notes above (N2's
    # distinction, one card over).
    assert all(line.row < 0 for line in card.painted_lines())
    assert NO_MATCH_NOTE not in (NO_TEAM_NOTE, NO_AGENT_NOTE)


# -- the flow, over the real page -------------------------------------------


def _start_rows() -> list[StartTarget]:
    return start_targets(teams=[_team("core"), _team("lopdev")], agents=[_agent("coder", "impl")])


@pytest.mark.asyncio
async def test_s_opens_the_start_card_and_esc_puts_it_away(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        assert view._mode == "canvas"
        await pilot.press("s")
        await pilot.pause()
        assert view._mode == "start"
        card = view._start_card
        assert card is not None
        assert [row.kind for row in card.rows] == ["team", "team", "agent", "plain"]
        # The card carries the grammar (spec §3.3's picker ladder).
        legend = view._start_card.query_one("#projects-start-legend").render()
        assert "esc close" in str(legend)
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "canvas"
        assert view._start_card is None


@pytest.mark.asyncio
async def test_the_page_hint_row_is_blank_while_the_card_holds_the_keys(
    tmp_path: Path,
) -> None:
    """R2-3/D10 one mode over: page keys the card consumes must not be advertised."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        painted = [
            hint.rendered()
            for hint in view._hints.children
            if getattr(hint, "display", False) and hint.rendered().strip()
        ]
        assert painted == [], painted


@pytest.mark.asyncio
async def test_the_card_floats_over_the_canvas_without_moving_the_page(
    tmp_path: Path,
) -> None:
    """The P5a geometry contract, re-asserted for the family's second card."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        before = view.canvas_size
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        card = view._start_card
        assert card is not None
        assert card._available >= START_CARD_CHROME_ROWS
        assert card._visible_lines <= max(0, card._available - START_CARD_CHROME_ROWS)
        # The card lives inside the page's own content box and the SCREEN never
        # grows a scrollbar (the mode's shipped invariant).
        assert view.canvas_size == before
        content = view.content_region
        assert card.region.y >= content.y
        assert card.region.y + card.region.height <= content.y + content.height
        assert app.screen.virtual_size == app.screen.size


@pytest.mark.asyncio
async def test_choosing_a_row_names_the_project_and_the_target(tmp_path: Path) -> None:
    session = _ProjectSession()
    registry = _registry(tmp_path, "alpha")
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        posted: list[object] = []
        real_post = view.post_message

        def record(message: object) -> object:
            posted.append(message)
            return real_post(message)  # type: ignore[arg-type]

        view.post_message = record  # type: ignore[method-assign]
        card = view._start_card
        assert card is not None
        card.action_choose()
        await pilot.pause()
        requests = [m for m in posted if isinstance(m, ProjectsViewStartRequested)]
        assert len(requests) == 1
        assert requests[0].target.kind == "team"
        assert requests[0].target.name == "core"
        expected = registry.get_project_by_name("alpha")
        assert expected is not None
        assert requests[0].project_id == expected.id


@pytest.mark.asyncio
async def test_s_does_nothing_with_nothing_selected(tmp_path: Path) -> None:
    """No project, nothing to link and no snapshot to quote: no card."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        # An empty PAGE — the state a store that lost its last project leaves.
        view.load(views=[], updated_at=0.0, own_session=None)
        await pilot.pause()
        assert view.current_project_id() is None
        await pilot.press("s")
        await pilot.pause()
        assert view._start_card is None


# -- the boot flow (the app's own handler) ----------------------------------


async def _settle(pilot: Any, times: int = 8) -> None:
    for _ in range(times):
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_app_creates_links_kicks_off_and_hands_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONE order, and the session id it mints is the one every step sees."""
    session = _ProjectSession()
    registry = _registry(tmp_path, "alpha")
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    calls: list[tuple[str, Any]] = []

    async def fake_create(cwd: str, target: StartTarget) -> str:
        calls.append(("create", (cwd, target.kind, target.name)))
        return STARTED_ID

    async def fake_kickoff(session_id: str, cwd: str, project: Any) -> None:
        calls.append(("kickoff", (session_id, project.name)))

    def fake_resume(session_id: str, notice: Any) -> None:
        calls.append(("resume", session_id))

    monkeypatch.setattr(app, "_create_project_session", fake_create)
    monkeypatch.setattr(app, "_kick_off_project_session", fake_kickoff)
    monkeypatch.setattr(app, "_resume_session", fake_resume)

    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        view._start_card.action_choose()  # type: ignore[union-attr]
        await _settle(pilot)

    assert [name for name, _ in calls] == ["create", "kickoff", "resume"]
    # The page closed before the hand-off (the mode must be gone before the
    # conversation it hid comes back) — `_resume_session` itself is stubbed
    # above, so this is the page's own exit and not the reboot's.
    assert app._projects_view is None
    project = registry.get_project_by_name("alpha")
    assert project is not None
    # The auto-link is a WORKING link (the CoS exemption marks a FILING).
    assert project.sessions == [STARTED_ID]
    assert project.coordination_sessions == []


@pytest.mark.asyncio
async def test_a_refused_create_lands_in_the_card_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _ProjectSession()
    registry = _registry(tmp_path, "alpha")
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    handoffs: list[str] = []
    monkeypatch.setattr(
        app,
        "_create_project_session",
        lambda cwd, target: _raise(ValueError("there is no team named 'ghost'")),
    )
    monkeypatch.setattr(app, "_resume_session", lambda sid, notice: handoffs.append(sid))

    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(_start_rows())
        await pilot.press("s")
        await pilot.pause()
        view._start_card.action_choose()  # type: ignore[union-attr]
        await _settle(pilot)

        card = view._start_card
        assert card is not None
        assert card.pending is False
        assert "no team named 'ghost'" in card._note
        # The page never switched and the store never changed.
        assert view._mode == "start"
        project = registry.get_project_by_name("alpha")
        assert project is not None and project.sessions == []

    assert handoffs == []
    assert not list((tmp_path / "sessions").glob("*")) if (tmp_path / "sessions").exists() else True


async def _raise(error: Exception) -> None:
    raise error


@pytest.mark.asyncio
async def test_an_unknown_target_gets_the_rows_own_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create core refuses an unknown name with a bare ``KeyError``."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))

    async def unknown(cwd: str, target: StartTarget) -> str:
        raise KeyError(target.name)

    monkeypatch.setattr(app, "_create_project_session", unknown)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        view.set_start_rows(start_targets(teams=[_team("ghost")], agents=[]))
        await pilot.press("s")
        await pilot.pause()
        view._start_card.action_choose()  # type: ignore[union-attr]
        await _settle(pilot)
        card = view._start_card
        assert card is not None
        assert card._note == "could not start a session: no team named 'ghost'"


@pytest.mark.asyncio
async def test_the_desktop_create_core_materialises_the_team_binding(tmp_path: Path) -> None:
    """THE REAL CALLABLE, not a stand-in: what `_create_project_session` runs.

    ``DesktopSessions.create`` is what the pane's new-chat reaches through
    ``POST /v1/desktop/sessions``; this executes it against a synthetic root
    and asserts the two facts the whole design rests on — a 12-character hex id
    is minted, and the team the row named is on disk as the session's
    attachment BEFORE any runtime exists for it. No runtime is started (create
    writes records, it does not engage), so this needs no provider and no
    network.
    """
    from local_operator.resume import read_session_attachment
    from local_operator.server.utils.desktop_sessions import DesktopSessions
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    root = tmp_path / "root"
    root.mkdir()
    TeamRegistry(root).create_team(
        TeamEditFields(name="lopdev", members=[TeamMember(role="coder")])
    )
    pool = DesktopSessions(root)
    session_id = await pool.create(str(root), target={"kind": "team", "name": "lopdev"})

    assert len(session_id) == 12 and all(c in "0123456789abcdef" for c in session_id)
    directory = root / "sessions" / session_id
    attachment = read_session_attachment(directory)
    assert attachment is not None
    assert attachment.team == "lopdev"
    # And the refusal the app turns into a sentence is the core's own.
    with pytest.raises(KeyError):
        await pool.create(str(root), target={"kind": "team", "name": "ghost"})
