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
from local_operator.tui.widgets.projects_view import (
    ProjectsView,
    ProjectsViewJumpRequested,
)
from local_operator.tui.widgets.subagent_view import HintButton
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
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


def _notices(app: OperatorApp) -> list[str]:
    """The transcript's notice texts, oldest first."""
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


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
        assert view._last is not None
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
        assert view._last is not None
        assert (canvas_w, canvas_h) == (view._last.width, view._last.height)
        # The Static is pinned to the canvas, so the container's virtual size
        # equals it (scrollbars appear exactly when over).
        width_scalar = view._canvas.styles.width
        height_scalar = view._canvas.styles.height
        assert width_scalar is not None and height_scalar is not None
        assert width_scalar.value == canvas_w
        assert height_scalar.value == canvas_h
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
        assert app._projects_view is not None
        assert app._projects_view.cursor == 1


# -- remediation round 1: the open reveal, the bar's row, the caps, the keys --


@pytest.mark.asyncio
async def test_show_reveals_the_named_row_on_a_store_longer_than_the_viewport(
    tmp_path: Path,
) -> None:
    """Q2: `/project show <name>` must land the reader ON the named row.

    The builder used to set the cursor and repaint, never scrolling — on a
    store longer than the viewport the page described a row that was 25 rows
    below the window.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, *[f"p{i:02d}" for i in range(40)])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "p39")
        assert view.cursor == 39
        offset = view._body.scroll_offset.y
        assert offset > 0
        assert offset <= 39 < offset + view._usable_height()


@pytest.mark.asyncio
async def test_downward_reveal_clears_the_horizontal_scrollbar_row(tmp_path: Path) -> None:
    """D1: with the h-bar up, the reveal must not park the cursor under it.

    The bar paints over the content region's LAST row, so the reveal's height
    has to be the same number ``max_scroll_y`` is computed from.
    """
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    for index in range(1, 31):
        registry.create_project(
            ProjectEdit(
                name=f"workstream-{index:02d}",
                description="a description wide enough to overflow the body",
            )
        )
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "workstream-01")
        assert view._body.scrollbar_size_horizontal == 1  # the bar is genuinely up
        await pilot.press("end")
        await pilot.pause()
        assert view.cursor == 29
        offset = view._body.scroll_offset.y
        assert 0 <= 29 - offset < view._usable_height()
        # And the physical check the widget's arithmetic exists to satisfy:
        # the row's screen line is above the bar's first line.
        bar = getattr(view._body, "_horizontal_scrollbar", None)
        if bar is not None:
            content_top = view._body.region.y + 1  # the body's top padding
            assert content_top + (29 - offset) < bar.region.y


@pytest.mark.asyncio
async def test_cursor_clamps_to_the_painted_rows_past_the_cap(tmp_path: Path) -> None:
    """D2: past ``PROJECTS_MAX`` the cursor may not rest on an unpainted row."""
    from local_operator.tui.projects_render import PROJECTS_MAX

    session = _ProjectSession()
    session.project_registry = _registry(
        tmp_path, *[f"queued-{index:03d}" for index in range(1, PROJECTS_MAX + 6)]
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "queued-001")
        await pilot.press("end")
        await pilot.pause()
        assert view.cursor == PROJECTS_MAX - 1  # the last PAINTED row, not row 204
        assert view._last is not None
        assert "▸" in view._last.text.plain
        assert "more not shown" in view._last.text.plain
        # The footer names the row the canvas paints, not the unpainted tail.
        assert "queued-200" in view.rendered_rows()[-1]


@pytest.mark.asyncio
async def test_zoom_survives_refresh_while_the_span_is_unchanged(tmp_path: Path) -> None:
    """F2: ``r`` must not discard a manual timeline zoom (the comment's claim)."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await pilot.press("3")
        await pilot.pause()
        await pilot.press("minus")
        await pilot.pause()
        zoomed = view.tier
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        assert view.tier == zoomed  # the span did not change; the zoom stands
        # A span CHANGE re-derives the auto tier (the stated rule).
        session.project_registry.create_project(ProjectEdit(name="far", target_date="2031-01-01"))
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        assert view.tier != zoomed


@pytest.mark.asyncio
async def test_hint_ladder_sheds_scroll_before_the_view_keys(tmp_path: Path) -> None:
    """U3: at 60 columns the newest view types stay advertised."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta", "gamma")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 20)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await pilot.pause()
        assert view._timeline_hint.display is True
        assert view._next_hint.display is True
        assert view._scroll_hint.display is False
        # Widening brings the scroll hint back into the ladder.
        await pilot.resize_terminal(140, 40)
        await pilot.pause()
        assert view._scroll_hint.display is True


@pytest.mark.asyncio
async def test_ctrl_c_closes_the_projects_page_like_its_siblings(tmp_path: Path) -> None:
    """Q1: the first ctrl+C dismisses the page (its warning then VISIBLE)."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        assert view.tracked == 1
        assert not app._transcript_view().display
        await pilot.press("ctrl+c")
        await pilot.pause()
        await pilot.pause()
        assert app._projects_view is None
        assert app._transcript_view().display
        assert app.is_running
        # The warning lands in a transcript that is visible again — the whole
        # point of closing the page in the same rung as its three siblings.
        assert app._exit_hint is not None
        assert "ctrl+c again to exit" in str(getattr(app._exit_hint, "_text", ""))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (150, 40)])
async def test_footer_shedding_keeps_the_page_geometry(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """Design round-2 scope: the U1/U2 footer changes move no geometry."""
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    registry.create_project(
        ProjectEdit(
            name="long-horizon-annotator-overhaul-with-many-many-words",
            description="a description that forces the footer to shed clauses",
        )
    )
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "long-horizon-annotator-overhaul-with-many-many-words")
        await pilot.pause()
        assert view._detail.region.height == 2  # the reserved box, unchanged
        width_scalar = view._canvas.styles.width
        height_scalar = view._canvas.styles.height
        assert width_scalar is not None and height_scalar is not None
        assert view.canvas_size == (width_scalar.value, height_scalar.value)
        assert app.screen.virtual_size == app.screen.size  # the screen never scrolls


@pytest.mark.asyncio
async def test_hint_row_has_no_leading_seam_when_scroll_sheds(tmp_path: Path) -> None:
    """UX round 2, U6: with `↔↕ scroll` shed the row opened with a dangling `·`."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta", "gamma")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 20)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        await pilot.pause()
        assert view._scroll_hint.display is False
        painted = [
            hint.rendered()
            for hint in view._hints.children
            if isinstance(hint, HintButton) and hint.display
        ]
        assert painted, "the hint row painted nothing"
        assert painted[0].startswith("1")  # the seam belongs to the row, not the hint
        assert "·" not in painted[0][:2]
        # Widening brings the scroll hint back as the row's FIRST hint — still
        # with no seam in front of it.
        await pilot.resize_terminal(140, 40)
        await pilot.pause()
        painted = [
            hint.rendered()
            for hint in view._hints.children
            if isinstance(hint, HintButton) and hint.display
        ]
        assert painted[0].startswith("↔↕")


@pytest.mark.asyncio
async def test_the_msg_hint_never_vanishes_while_create_is_advertised(
    tmp_path: Path,
) -> None:
    """F3: `m message` rides EVERY rung that still carries `c create`.

    The reviewer measured `m message` disappearing at canvas widths 109-131
    while `c create`/`r refresh` painted: a rung without `msg_hint` sat in
    front of an equal-width rung that had it, so the later rung was
    unreachable and widening the terminal made the hint vanish. This sweep is
    the invariant the fix restores, checked at every width.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(96, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        saw_create = False
        for width in range(96, 141):
            await pilot.resize_terminal(width, 24)
            await pilot.pause()
            if view._create_hint.display:
                saw_create = True
                assert (
                    view._msg_hint.display
                ), f"width {width}: `c create` is advertised without `m message`"
        assert saw_create, "the sweep never saw a rung carrying `c create`"


# -- S3b: the selection's jump, in the view that owns the cursor --------------


@pytest.mark.asyncio
async def test_enter_asks_the_host_to_open_the_selected_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`↵` posts ONE message naming the selection and its linked sessions (S3b)."""
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="alpha"), sessions=[SESSION_ID])
    registry.create_project(ProjectEdit(name="beta"))
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    seen: list[Any] = []
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        await _open(pilot, app, "alpha")
        # `post_message` is the WRONG seam: Textual delivers the key TO the
        # focused widget through it (and posts its shutdown handshake through
        # it), so an instance patch of it swallows the very press under test —
        # measured here as three captured posts (Key, Callback, Callback) and a
        # hung `run_test` exit. Recording the message at construction leaves
        # the machinery intact, and the real handler still receives it.
        original_init = ProjectsViewJumpRequested.__init__

        def _record(recorded_self: Any, **kwargs: Any) -> None:
            original_init(recorded_self, **kwargs)
            seen.append(recorded_self)

        monkeypatch.setattr(ProjectsViewJumpRequested, "__init__", _record)
        await pilot.press("enter")
        await pilot.pause()
    assert len(seen) == 1
    message = seen[0]
    assert message.project_name == "alpha"
    # The link's directory does not exist in this fixture and the receipt says
    # `missing` for exactly that state — the message carries the RECEIPT's word
    # (QA round 1, Q2), not the composed row's default `stopped`.
    assert message.sessions == ((SESSION_ID, "missing"),)


@pytest.mark.asyncio
async def test_jump_with_no_live_session_names_what_exists(tmp_path: Path) -> None:
    """`↵` on a stopped link: the page KEEPS the reader; its own footer
    names what exists (UX round 1, U4 — the transcript the mode hides is not
    asked to carry the sentence)."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "alpha")
        app.on_projects_view_jump_requested(
            ProjectsViewJumpRequested(
                project_id="alpha-id",
                project_name="alpha",
                sessions=((SESSION_ID, "stopped"),),
            )
        )
        await pilot.pause()
        assert app._projects_view is view  # the mode was not left
        notice = view.rendered_rows()[-1]
        assert "no live session to open for 'alpha'" in notice
        assert f"{SESSION_ID} [stopped]" in notice
        # ONE linked session: the notice spells the concrete command (UX r1, U4).
        assert f"/resume {SESSION_ID} starts it." in notice


@pytest.mark.asyncio
async def test_jump_with_a_live_session_switches_through_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live link goes through `_resume_session` — the one switch machinery."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    switched: list[str] = []
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        await _open(pilot, app, "alpha")
        monkeypatch.setattr(
            app,
            "_resume_session",
            lambda resume_id, notice, **kw: switched.append(resume_id),
        )
        app.on_projects_view_jump_requested(
            ProjectsViewJumpRequested(
                project_id="alpha-id",
                project_name="alpha",
                sessions=(("live-session-1", "live"),),
            )
        )
        await pilot.pause()
    assert switched == ["live-session-1"]
    assert app._projects_view is None


@pytest.mark.asyncio
async def test_jump_on_the_current_session_says_so(tmp_path: Path) -> None:
    """`↵` on the project THIS terminal is already in is answered, not rebooted."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        await _open(pilot, app, "alpha")
        app.on_projects_view_jump_requested(
            ProjectsViewJumpRequested(
                project_id="alpha-id",
                project_name="alpha",
                sessions=((SESSION_ID, "live"),),
            )
        )
        await pilot.pause()
        assert "already in 'alpha'" in _notices(app)[-1]
        assert app._projects_view is None


# -- round 1 remediation: seeding, reveal-on-switch, clamp repaint, hint -----


@pytest.mark.asyncio
async def test_nameless_show_seeds_the_cursor_onto_the_set(tmp_path: Path) -> None:
    """UX r1 U1: with links on p01/p02 the cursor must NOT stay on row 0."""
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="p00"))
    registry.create_project(ProjectEdit(name="p01"), sessions=[SESSION_ID])
    registry.create_project(ProjectEdit(name="p02"), sessions=[SESSION_ID])
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project show")
        await pilot.pause()
        await pilot.pause()
        view = app._projects_view
        assert view is not None and view._view == "board"
        found = registry.get_project_by_name("p01")
        assert found is not None
        assert view.current_project_id() == str(found.id)
        canvas = view._last.text.plain
        assert "▸◆p01" in canvas and "◆ p02" in canvas and "  p00" in canvas


@pytest.mark.asyncio
async def test_switching_to_board_reveals_the_selection(tmp_path: Path) -> None:
    """QA r1 Q1: the reveal re-runs once the new canvas's layout has landed."""
    from local_operator.tui.projects_render import board_position

    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    for index in range(20):
        registry.create_project(ProjectEdit(name=f"p{index:02d}"))
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _open(pilot, app, "p14")
        await pilot.press("2")
        await pilot.pause()
        await pilot.pause()
        view = app._projects_view
        assert view is not None
        position = board_position(view._views, view._cursor)
        assert position is not None
        offset = view._body.scroll_offset.y
        assert offset <= position[1] <= offset + max(view._usable_height() - 1, 0), (
            offset,
            position,
        )


@pytest.mark.asyncio
async def test_a_clamp_repaints_the_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent review r1 MINOR 3: clamping the cursor to a painted cell repaints."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta", "gamma")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "gamma")
        # Force the clamp: the position helper answers None for the cursor and a
        # real cell for index 1 (the shape a truncated canvas produces).
        from local_operator.tui.projects_render import board_position

        real = board_position

        def fake_position(index):  # noqa: ANN001 - test seam
            if index == view._cursor:
                return None
            return real(view._views, index)

        monkeypatch.setattr(view, "_position_for", fake_position)
        view._view = "board"
        view._cursor = 2
        view._scroll_cursor_into_view()
        assert view._cursor == 1  # clamped to the last painted project
        canvas = view._last.text.plain if view._last else ""
        assert "▸" in canvas, "the clamp must repaint the moved selection"


# -- round 2 remediation: refresh keeps the reader, the live pick, the title --


@pytest.mark.asyncio
async def test_a_refresh_never_re_seeds_the_reader(tmp_path: Path) -> None:
    """R2-3: the U1 seed rides an ENTRY; `r` must not move the reader back."""
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="p00"))
    registry.create_project(ProjectEdit(name="p01"), sessions=[SESSION_ID])
    registry.create_project(ProjectEdit(name="p02"), sessions=[SESSION_ID])
    session.project_registry = registry
    p00 = registry.get_project_by_name("p00")
    p01 = registry.get_project_by_name("p01")
    assert p00 is not None and p01 is not None
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project show")
        await pilot.pause()
        await pilot.pause()
        view = app._projects_view
        assert view is not None and view.current_project_id() == str(p01.id)
        await pilot.press("up")  # onto p00, outside the caller's set
        await pilot.pause()
        assert view.current_project_id() == str(p00.id)
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        # The refresh keeps the reader where they put themselves — re-seeding
        # here would contradict `load`'s own "a refresh must not move the
        # reader" contract (and the set itself is kept).
        assert view.current_project_id() == str(p00.id)
        assert "this session (2)" in view.rendered_rows()[0]


@pytest.mark.asyncio
async def test_the_terminals_own_session_wins_the_live_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NIT 6: with a sibling live FIRST in link order, `↵` on this terminal's
    own live project answers `already in` instead of switching away."""
    sibling = "cd" * 6
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))

    async def _noop_resume(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        return None

    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app)
        await _open(pilot, app, "alpha")
        monkeypatch.setattr(app, "_resume_session", _noop_resume, raising=False)
        # The SIBLING is first: `live[0]` would switch to it without the rule.
        app.on_projects_view_jump_requested(
            ProjectsViewJumpRequested(
                project_id="alpha-id",
                project_name="alpha",
                sessions=((sibling, "live"), (SESSION_ID, "live")),
            )
        )
        await pilot.pause()
        assert "already in 'alpha'" in _notices(app)[-1]
        assert SESSION_ID in _notices(app)[-1]
        # …and a sibling-only live set still switches (the fallback survives).
        app.on_projects_view_jump_requested(
            ProjectsViewJumpRequested(
                project_id="alpha-id",
                project_name="alpha",
                sessions=((sibling, "live"),),
            )
        )
        await pilot.pause()
        assert app._projects_view is None


@pytest.mark.asyncio
async def test_the_title_sheds_before_clipping(tmp_path: Path) -> None:
    """D2/D6: at 60 columns the set clause (board) and `zoom:` (timeline) shed
    instead of the row clipping mid-clause."""
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="p00"), sessions=[SESSION_ID])
    registry.create_project(ProjectEdit(name="p01"), sessions=[SESSION_ID])
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 20)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/project show")
        await pilot.pause()
        await pilot.pause()
        view = app._projects_view
        assert view is not None
        board_title = view.rendered_rows()[0]
        assert "this session" not in board_title, board_title
        assert "tracked" in board_title and "updated" in board_title
        app._run_slash_command("/project timeline")
        await pilot.pause()
        await pilot.pause()
        timeline_view = app._projects_view
        assert timeline_view is not None
        timeline_title = timeline_view.rendered_rows()[0]
        assert "zoom:" not in timeline_title, timeline_title
        assert "tracked" in timeline_title and "updated" in timeline_title


# -- S6d: sections, the ruler and the canvas clicks ---------------------------


def _grouped_registry(tmp_path: Path) -> ProjectRegistry:
    """Four projects, two teams, one bucket of one — name-sorted b < m < p < r.

    ``core`` = board-entry(0) + parity-spec(2); ``personal`` = mobile-sheet(1);
    ``no team`` = references(3). The canvas rows are therefore 0 core header,
    1..2 core rows, 3 personal header, 4 its row, 5 no-team header, 6 its row.
    """
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="board-entry", title="Board entry", team="core"))
    registry.create_project(
        ProjectEdit(name="mobile-sheet", title="Mobile sheet", team="personal", status="done")
    )
    registry.create_project(ProjectEdit(name="parity-spec", title="TUI parity spec", team="core"))
    registry.create_project(ProjectEdit(name="references", status="paused"))
    return registry


@pytest.mark.asyncio
async def test_shift_arrows_jump_sections_and_clamp(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _grouped_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "board-entry")
        assert [row["project"]["name"] for row in view._views] == [
            "board-entry",
            "mobile-sheet",
            "parity-spec",
            "references",
        ]
        assert view.cursor == 0
        await pilot.press("shift+down")
        await pilot.pause()
        assert view.cursor == 1  # personal's first row
        await pilot.press("shift+down")
        await pilot.pause()
        assert view.cursor == 3  # the `no team` bucket, last
        await pilot.press("shift+down")
        await pilot.pause()
        assert view.cursor == 3  # clamped at the end
        await pilot.press("shift+up")
        await pilot.pause()
        assert view.cursor == 1
        await pilot.press("shift+up")
        await pilot.pause()
        assert view.cursor == 0
        await pilot.press("shift+up")
        await pilot.pause()
        assert view.cursor == 0  # clamped at the top


@pytest.mark.asyncio
async def test_shift_arrows_do_nothing_without_teams(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha", "beta", "gamma")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "beta")
        assert view.cursor == 1
        await pilot.press("shift+down")
        await pilot.pause()
        assert view.cursor == 1
        await pilot.press("shift+up")
        await pilot.pause()
        assert view.cursor == 1


@pytest.mark.asyncio
async def test_the_rule_is_a_section_ruler_that_tracks_the_viewport(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _grouped_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 20)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "board-entry")
        rows = view.rendered_rows()
        assert rows[1].startswith("── core · 2 ")
        # The same width as the shipped plain rule (view 56 at 60 columns,
        # minus the two-cell inset): the ruler changes the CONTENT, not the row.
        assert len(rows[1]) == 54
        # The body shows 4 rows of a 7-row canvas: scrolling to the personal
        # header moves the ruler — and the cursor (still in core) earns the
        # dim `sel core` clause.
        view._body.scroll_to(y=3, animate=False)
        await pilot.pause()
        await pilot.pause()
        assert view.rendered_rows()[1].startswith("── personal · 1 · sel core ")


@pytest.mark.asyncio
async def test_clicks_select_rows_and_headers_jump(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _grouped_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "board-entry")
        canvas = view._canvas
        # A row click selects it and moves the keyboard cursor there (canvas
        # row 4 = mobile-sheet, in personal).
        await pilot.click(canvas, offset=(5, 4))
        await pilot.pause()
        assert view.cursor == 1
        # A header click jumps to that section's first row (row 5 = `no team`).
        await pilot.click(canvas, offset=(5, 5))
        await pilot.pause()
        assert view.cursor == 3
        # A header whose section already holds the cursor is a no-op.
        await pilot.click(canvas, offset=(5, 3))
        await pilot.pause()
        assert view.cursor == 1


@pytest.mark.asyncio
async def test_board_card_clicks_select_the_card(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _grouped_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "board-entry")
        await pilot.press("2")
        await pilot.pause()
        assert view.view_type == "board"
        from local_operator.tui.projects_render import board_position

        card = board_position(view._views, 2)  # parity-spec, core's second card
        assert card is not None
        x, y = card
        assert x == 0 and y > 0
        await pilot.click(view._canvas, offset=(x + 2, y))
        await pilot.pause()
        assert view.cursor == 2


@pytest.mark.asyncio
async def test_the_sentinel_bucket_answers_its_own_header_click(tmp_path: Path) -> None:
    """R1-2/U1: two `no team` sections, and each header reaches its own."""
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="alpha", team="core"))
    registry.create_project(ProjectEdit(name="binary", team="no team"))
    registry.create_project(ProjectEdit(name="charlie", team="no team"))
    registry.create_project(ProjectEdit(name="delta"))
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "alpha")
        assert [row["project"]["name"] for row in view._views] == [
            "alpha",
            "binary",
            "charlie",
            "delta",
        ]
        # rows: 0 core header, 1 alpha, 2 `no team` header, 3-4 its rows,
        # 5 `no team (unset)` header, 6 delta
        await pilot.click(view._canvas, offset=(5, 5))
        await pilot.pause()
        assert view.cursor == 3  # the teamless row, not the upper team
        await pilot.click(view._canvas, offset=(5, 2))
        await pilot.pause()
        assert view.cursor == 1  # the real team's first row


@pytest.mark.asyncio
async def test_timeline_chart_header_click_lands_on_the_chart(tmp_path: Path) -> None:
    """U3: the chart header targets its first DATED row, not the tail line."""
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="aaa-notes", team="core"))
    registry.create_project(ProjectEdit(name="zzz-ship", team="core", target_date="2026-10-04"))
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "aaa-notes")
        await pilot.press("3")
        await pilot.pause()
        assert view.view_type == "timeline"
        # rows: 0 axis, 1 the section header, 2 zzz-ship's chart row, 3 blank,
        # 4 the `no dates` tail line
        await pilot.click(view._canvas, offset=(5, 1))
        await pilot.pause()
        assert view.cursor == 1  # zzz-ship's dated row, not the tail line


@pytest.mark.asyncio
async def test_a_grouped_reveal_accounts_for_the_header_rows(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _grouped_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 20)) as pilot:
        await _boot(pilot, app)
        # references sits on canvas row 6 (two headers above it); a reveal
        # that used its INDEX would scroll 3 rows short.
        view = await _open(pilot, app, "references")
        assert view.cursor == 3
        # The reveal scrolls clear of the horizontal scrollbar's row (the
        # shipped downward-reveal rule): at 60 columns the rows are wider than
        # the viewport, so the usable height is 3 and the offset lands on 4.
        assert int(view._body.scroll_offset.y) == 4
        # The ruler tracks the viewport, and the cursor's own section (the
        # `no team` bucket) earns the sel clause.
        assert view.rendered_rows()[1].startswith("── personal · 1 · sel no team ")
        # A click after the scroll resolves through the same offset: widget
        # offsets ARE canvas cells, whatever the viewport shows.
        await pilot.click(view._canvas, offset=(5, 4))
        await pilot.pause()
        assert view.cursor == 1


@pytest.mark.asyncio
async def test_double_click_repeats_enter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _ProjectSession()
    session.project_registry = _grouped_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(140, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "board-entry")
        fired: list[bool] = []
        monkeypatch.setattr(view, "action_jump", lambda: fired.append(True))
        await pilot.click(view._canvas, offset=(5, 2), times=2)  # parity-spec's row
        await pilot.pause()
        assert fired, "the double click must repeat `↵`"
        assert view.cursor == 2  # the acting gesture moves the caret
