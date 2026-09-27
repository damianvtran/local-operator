"""The full-page ``/project`` view, driven through the REAL ``OperatorApp``.

The real app is the only host that loads ``local_operator.tcss``, so the layout
and colour assertions here are made against the shipped stylesheet. The
geometry assertions are the numbers behind the captured frames: the canvas is
sized in Python to what the renderers return, the body's virtual size equals
that canvas, and the SCREEN never grows a scrollbar — the page scrolls inside
its own body, exactly as ``/settings`` and ``/team chart`` do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.projects import ProjectEdit, ProjectRegistry
from local_operator.tui.app import PROJECTS_LAYOUT_CLASS, OperatorApp
from local_operator.tui.widgets.projects_view import ProjectsView
from local_operator.tui.widgets.transcript import UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

SESSION_ID = "ab12cd34ef56"


class _ProjectSession(FakeSession):
    @property
    def session_id(self) -> str:
        return SESSION_ID


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _registry(tmp_path: Path, *names: str) -> ProjectRegistry:
    registry = ProjectRegistry(tmp_path)
    for name in names:
        registry.create_project(ProjectEdit(name=name))
    return registry


async def _boot(pilot: Any, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _open(pilot: Any, app: OperatorApp, name: str = "alpha") -> ProjectsView:
    app._run_slash_command(f"/project show {name}")
    await pilot.pause()
    await pilot.pause()
    view = app._projects_view
    assert isinstance(view, ProjectsView)
    return view


@pytest.mark.asyncio
async def test_show_opens_a_full_page_mode_and_esc_restores(tmp_path: Path) -> None:
    """A MODE, not a modal: transcript hidden, dock greyed, esc puts it back."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._append_block(UserBlock("a turn worth keeping"))
        editor = app._editor()
        editor.focus()
        editor.load_text("half-typed prompt")
        await pilot.pause()

        view = await _open(pilot, app, "beta")
        assert app.screen.has_class(PROJECTS_LAYOUT_CLASS)
        assert not app._transcript_view().display
        assert view.has_focus
        # The cursor landed on the named project, not the first row.
        assert view.cursor == 1

        await pilot.press("escape")
        await pilot.pause()
        assert app._projects_view is None
        assert not app.screen.has_class(PROJECTS_LAYOUT_CLASS)
        assert app._transcript_view().display
        assert app._editor().text == "half-typed prompt"


@pytest.mark.asyncio
async def test_view_keys_switch_and_cycle(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta", "gamma")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)

        await pilot.press("2")
        await pilot.pause()
        assert view.view_type == "board"
        await pilot.press("3")
        await pilot.pause()
        assert view.view_type == "timeline"
        assert view.tier in ("week", "month", "quarter")
        await pilot.press("1")
        await pilot.pause()
        assert view.view_type == "list"
        # `v` cycles list → board → timeline → list.
        await pilot.press("v")
        await pilot.pause()
        assert view.view_type == "board"
        await pilot.press("v")
        await pilot.pause()
        assert view.view_type == "timeline"
        await pilot.press("v")
        await pilot.pause()
        assert view.view_type == "list"


@pytest.mark.asyncio
async def test_zoom_hint_is_advertised_only_on_the_timeline(tmp_path: Path) -> None:
    """`+/-` is TIME zoom: a hinted key that changes nothing is worse than none."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(150, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await pilot.pause()
        assert not view._zoom_hint.display  # list
        await pilot.press("2")
        await pilot.pause()
        assert not view._zoom_hint.display  # board
        await pilot.press("3")
        await pilot.pause()
        assert view._zoom_hint.display  # timeline


@pytest.mark.asyncio
async def test_timeline_zoom_keys_change_the_tier(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await pilot.press("3")
        await pilot.pause()
        # `+` moves toward finer time on the timeline; `-` back coarser.
        await pilot.press("plus")
        await pilot.pause()
        fine = view.tier
        await pilot.press("minus")
        await pilot.pause()
        assert view.tier != fine or view.tier == "quarter"
        await pilot.press("minus")
        await pilot.pause()
        assert view.tier == "quarter"


@pytest.mark.asyncio
async def test_list_cursor_clamps_and_reveals(tmp_path: Path) -> None:
    """The full-page exception AGENTS.md records: arrows CLAMP, never wrap."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, *[f"p{i:02d}" for i in range(30)])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "p00")
        assert view.cursor == 0

        # At the top, `up` is a dead key (clamped) — no teleport to the end.
        await pilot.press("up")
        await pilot.pause()
        assert view.cursor == 0

        for _ in range(40):
            await pilot.press("down")
        await pilot.pause()
        assert view.cursor == 29  # clamped at the last row
        await pilot.press("down")
        await pilot.pause()
        assert view.cursor == 29
        # The cursor row was revealed inside the body (it is row 29 of 30).
        assert view._body.scroll_offset.y > 0
        assert view._body.scroll_offset.y + view._body.size.height >= 29

        # Home/End are cursor ends on the list view.
        await pilot.press("home")
        await pilot.pause()
        assert view.cursor == 0
        await pilot.press("end")
        await pilot.pause()
        assert view.cursor == 29


@pytest.mark.asyncio
async def test_refresh_recomposes_and_keeps_the_reader_where_they_were(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "beta")
        assert view.cursor == 1
        before = view.canvas_size

        # A peer (another window, the agent's tool) creates a row while the
        # page is open; `r` is the operator's way to see it.
        session.project_registry.create_project(ProjectEdit(name="gamma"))
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        assert view.tracked == 3
        assert "gamma" in view._last.text.plain
        assert view.canvas_size[1] > before[1]
        # A refresh must not move the reader off the row they were reading.
        assert view.cursor == 1


@pytest.mark.asyncio
async def test_canvas_geometry_matches_the_pinned_static(tmp_path: Path) -> None:
    """The numbers behind the frames: virtual size == canvas, screen no scroll."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, *[f"p{i:02d}" for i in range(40)])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 20)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "p00")

        canvas_w, canvas_h = view.canvas_size
        assert (canvas_w, canvas_h) == (view._last.width, view._last.height)
        # The Static is pinned to the canvas, so the container's virtual size
        # equals it (scrollbars appear exactly when over).
        assert view._canvas.styles.width.value == canvas_w
        assert view._canvas.styles.height.value == canvas_h
        # The LIST canvas overflows vertically on a small terminal ...
        assert canvas_h > view._body.size.height
        assert view._body.virtual_size.height == canvas_h
        # ... and the SCREEN still does not scroll (the AGENTS.md invariant).
        assert app.screen.virtual_size.height <= app.screen.size.height
        assert app.screen.virtual_size.width <= app.screen.size.width

        # The board paints three fixed columns side by side; on a narrow
        # terminal that is wider than the viewport, so the BODY scrolls
        # horizontally while the screen still does not.
        await pilot.press("2")
        await pilot.pause()
        board_w, _board_h = view.canvas_size
        assert board_w >= 72  # three ~32-cell columns, right-trimmed
        assert view._body.size.width < board_w
        assert view._body.virtual_size.width == board_w
        assert app.screen.virtual_size.width <= app.screen.size.width


@pytest.mark.asyncio
async def test_opened_from_the_splash_the_page_takes_the_whole_region(
    tmp_path: Path,
) -> None:
    """`Screen.boot` is a whole second layout; the mode must shed it.

    Both dimensions are asserted because the collision is not the same shape at
    every size: a rows-only assertion passed while the input card stayed
    width-clamped over the page (the org chart's review round 1, F1/F2).
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")

    async def measure(seed_conversation: bool) -> tuple[int, int, int]:
        # A FRESH app per run: `run_test` is not re-entrant on one instance,
        # and the two runs must not share screen state.
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            if seed_conversation:
                app._append_block(UserBlock("hello"))
                await pilot.pause()
            else:
                assert app.screen.has_class("boot")
            view = await _open(pilot, app)
            assert not app.screen.has_class("boot")
            shell = app.query_one("#input-shell")
            return (view.size.height, view.size.width, shell.size.width)

    over_splash = await measure(False)
    over_talk = await measure(True)
    assert over_splash == over_talk


@pytest.mark.asyncio
async def test_reopening_retargets_without_duplicates(tmp_path: Path) -> None:
    """``remove()`` only POSTS a prune; a reopen inside that window must not
    mount a second page (the class-identified lesson org_chart_view records)."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        for _ in range(3):
            app._open_projects_view(highlight=None)
            app._close_projects_view()
        app._open_projects_view(highlight=None)
        await pilot.pause()
        assert len(app.query(ProjectsView)) == 1
        # A second open retargets the cursor instead of remounting.
        app._run_slash_command("/project show beta")
        await pilot.pause()
        assert len(app.query(ProjectsView)) == 1
        assert app._projects_view.cursor == 1
