"""The viewed owner, not the root process, owns docked plans and children."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from rich.cells import cell_len

from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    JobState,
    SnapshotJobs,
    SnapshotSubagentComms,
    TodoPhaseState,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.subagent_panel import SubagentPanel
from local_operator.tui.widgets.subagent_view import entry_block, fold_trajectory
from local_operator.tui.widgets.todo_panel import TodoPanel
from local_operator.tui.widgets.tool_card import ToolCard
from tests.unit.tui.test_subagent_view import (
    FakeSession,
    _async_factory,
    _call,
    _open,
    _result,
)


def plan(text: str, status: str = "pending") -> list[dict[str, Any]]:
    return [{"name": "Work", "items": [{"text": text, "status": status, "reason": ""}]}]


def scoped_state() -> FrontendSessionState:
    return FrontendSessionState(
        session_id="sess",
        epoch="owner",
        selected_model=FrontendModelSpec(
            provider="test", model_id="model", display_name="test/model"
        ),
        todos=[TodoPhaseState.model_validate(row) for row in plan("Root plan")],
        jobs=[
            JobState(
                id="manager",
                type="task",
                session_id="manager-session",
                label="Coordinate review",
                todos=plan("Manager plan"),
                trajectory=[_call("wait", "jobs", op="peek")],
            ),
            JobState(
                id="sibling",
                type="task",
                session_id="sibling-session",
                label="Independent work",
                todos=plan("Sibling plan"),
            ),
            JobState(
                id="leaf",
                type="task",
                session_id="leaf-session",
                parent_job_id="manager",
                label="Inspect documentation",
                todos=plan("Leaf plan"),
                trajectory=[
                    _call("read", "read", path="README.md"),
                    _result("read", "read", "Synthetic documentation"),
                ],
            ),
        ],
    )


def install(session: Any, state: FrontendSessionState) -> None:
    session.owns_runtime = False
    session.frontend_state = state
    session.jobs = SnapshotJobs(state.jobs)
    session._subagent_comms = SnapshotSubagentComms(state.jobs)


def nested_state() -> FrontendSessionState:
    """``scoped_state`` one level deeper: the leaf child has a child of its own.

    Kept OUT of :func:`scoped_state` deliberately — that fixture is shared by
    tests that open ``leaf`` and assert the dock leaves with it.
    """
    state = scoped_state()
    return state.model_copy(
        update={
            "jobs": [
                *state.jobs,
                JobState(
                    id="grandchild",
                    type="task",
                    session_id="grandchild-session",
                    parent_job_id="leaf",
                    label="Check the changelog",
                    todos=plan("Grandchild plan"),
                ),
            ]
        }
    )


class CountingComms(SnapshotSubagentComms):
    """The follower facade, noting every graph walk it was asked for.

    The spy exists so the count on a row can be shown to come from a walk of
    the GRAPH rather than from anything read off the job row: a mark derived
    from the row itself would need no call here at all. It counts WALKS, not
    per-row lookups, because the mark is read from ONE grouping pass over
    ``nodes()`` (review round 1, F2): the old per-row ``children(job_id)``
    rebuilds the whole node list on every call, so a mark cost O(rows x nodes)
    and this counter is what makes that shape observable.
    """

    def __init__(self, jobs: Any) -> None:
        super().__init__(jobs)
        self.walks = 0
        self.children_asked: list[str] = []

    def nodes(self) -> list[Any]:
        self.walks += 1
        return super().nodes()

    def children(self, job_id: str | None) -> list[Any]:
        self.children_asked.append(str(job_id))
        return super().children(job_id)


def row_text(panel: SubagentPanel, job_id: str) -> str:
    """The string one mounted row is painting right now."""
    return str(panel._rows[job_id].content)


@pytest.mark.parametrize("size", [(100, 30), (80, 24)])
@pytest.mark.asyncio
async def test_manager_children_and_plans_follow_navigation_and_live_updates(size) -> None:
    state = scoped_state()
    session = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=size) as pilot:
        view = await _open(pilot, app, session.jobs.get("manager"))
        panel = app.query_one(SubagentPanel)
        todos = app.query_one(TodoPanel)
        assert set(panel._rows) == {"leaf"}
        assert "Manager plan" in str(todos._body.content)
        assert "Root plan" not in str(todos._body.content)
        await pilot.click(panel._rows["leaf"])
        await pilot.pause()
        assert view.job_id == "leaf"
        assert not panel.display
        assert "Leaf plan" in str(todos._body.content)
        assert "Manager plan" not in str(todos._body.content)
        card = view.query_one(ToolCard)
        await pilot.click(card)
        await pilot.pause()
        assert card._expanded
        await pilot.press("escape")
        await pilot.pause()
        assert view.job_id == "manager"
        assert set(panel._rows) == {"leaf"}
        owner = FrontendStateStore(state)
        follower = FrontendStateStore(state)
        updated = [
            (
                job.model_copy(update={"todos": plan("Manager updated"), "status": "running"})
                if job.id == "manager"
                else job
            )
            for job in state.jobs
        ]
        delta = owner.mutate(jobs=updated)
        assert delta is not None
        follower.apply_update(FrontendUpdate.model_validate_json(delta.model_dump_json()))
        install(session, follower.state)
        app._apply_frontend_state(follower.state)
        await pilot.pause()
        assert "Manager updated" in str(todos._body.content)
        assert "Manager plan" not in str(todos._body.content)
        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert view.job_id == "sibling"
        assert "Sibling plan" in str(todos._body.content)
        assert not panel.display
        await pilot.press("escape")
        await pilot.pause()
        assert app._subagent_view is None
        assert set(panel._rows) == {"manager", "sibling"}
        assert "Root plan" in str(todos._body.content)
        assert app.screen.size == app.screen.virtual_size
        assert not app.screen.show_vertical_scrollbar


@pytest.mark.asyncio
async def test_the_dock_says_whose_children_it_lists_and_which_rows_go_deeper() -> None:
    """Both halves of "where am I", on a roster that has a level under it.

    The rows are the DIRECT children of the page you have open, so ``Subagents``
    alone names a different list on every level, and a child with children
    paints the same row as a leaf until you drill in and compare two lists.
    """
    state = nested_state()
    session: Any = FakeSession()
    install(session, state)
    session._subagent_comms = comms = CountingComms(state.jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(SubagentPanel)
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()

        # Root scope, no page open: the bare header, and the mark on the one
        # row that has a level under it.
        assert panel.summary_text() == "Subagents   ctrl+g"
        assert set(panel._rows) == {"manager", "sibling"}
        assert "⊞1" in row_text(panel, "manager")
        assert "⊞" not in row_text(panel, "sibling")

        # The count came from a WALK of the comms graph — nothing about a mark
        # is read off the job row it is painted on — and `children` is asked
        # once by the resolver for the SCOPE it is listing, never once per row:
        # nothing here asks about `manager` or `sibling`, which is exactly what
        # a per-row count of their marks would have done (review round 1, F2).
        assert panel._children_counts == {"manager": 1, "sibling": 0}
        assert set(comms.children_asked) == {"None"}, comms.children_asked
        assert comms.walks, "the graph must be the source of the mark"

        await _open(pilot, app, session.jobs.get("manager"))
        await pilot.pause()
        # Scoped: the rows are now the manager's children, and the header says
        # so — this is the level the reader just arrived on.
        assert panel.summary_text() == "Subagents of Coordinate review   ctrl+g"
        assert set(panel._rows) == {"leaf"}
        assert panel._children_counts == {"leaf": 1}
        assert "⊞1" in row_text(panel, "leaf")
        # The newly-scoped page's own level was reached through the resolver,
        # and the mark on ITS row (leaf) never asked for leaf's children.
        assert comms.children_asked[-1] == "manager", comms.children_asked
        assert "leaf" not in comms.children_asked, comms.children_asked

        await pilot.press("escape")
        await pilot.pause()
        # Leaving the page takes the name of the scope with it: the header
        # names the level the rows are on, never the level just left.
        assert panel.summary_text() == "Subagents   ctrl+g"


@pytest.mark.asyncio
async def test_a_long_scope_label_cannot_push_the_header_past_its_row() -> None:
    """The scope is model-authored text, so the header caps it (`SCOPE_CEILING`).

    `#band` is content-sized, so an unbounded name in the header widens the
    dock rather than truncating inside it. The bound is the ceiling plus the
    header's own chrome and hint — read off the painted caption, not off the
    constant, so a future header that adds a segment fails here.
    """
    long_label = "Audit every merged MR for the ingest path and the scheduler"
    state = scoped_state()
    state = state.model_copy(
        update={
            "jobs": [
                job.model_copy(update={"label": long_label}) if job.id == "manager" else job
                for job in state.jobs
            ]
        }
    )
    session: Any = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(SubagentPanel)
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()
        await _open(pilot, app, session.jobs.get("manager"))
        await pilot.pause()
        caption = panel.summary_text()
        # 13 cells of `Subagents of ` + SCOPE_CEILING + 3 of HINT_GAP + `ctrl+g`.
        assert len(caption) <= 46, caption
        assert caption.startswith("Subagents of Audit every merged MR"), caption
        assert caption.endswith("   ctrl+g"), caption
        assert long_label not in caption, caption
        # And the header still occupies its one row of the dock.
        assert panel._header.size.height == 1


@pytest.mark.asyncio
async def test_a_host_with_no_comms_graph_docks_the_roster_without_marks() -> None:
    """The marks are a decoration: a missing graph costs them and nothing else.

    ``_refresh_band`` runs from the 1 Hz poll and from every ``Subagent*``
    handler, so a host with no lineage to walk — a local session, an embedder —
    must dock its ledger with no marks rather than raise or blank the dock.
    """
    state = scoped_state()
    session: Any = FakeSession()
    session.frontend_state = state
    session.jobs = SnapshotJobs(state.jobs)
    assert session._subagent_comms is None
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(SubagentPanel)
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()
        assert panel._children_counts == {}
        assert panel.display
        assert all("⊞" not in row_text(panel, job_id) for job_id in panel._rows)


@pytest.mark.parametrize("size", [(100, 30), (80, 24)])
@pytest.mark.asyncio
async def test_scoped_overflow_preserves_body_status_and_disclosure(size) -> None:
    state = scoped_state()
    long_plan = [
        {
            "name": "Implementation",
            "items": [
                {"text": f"Task {index}", "status": "pending", "reason": ""} for index in range(30)
            ],
        }
    ]
    state = state.model_copy(
        update={
            "jobs": [
                job.model_copy(update={"todos": long_plan}) if job.id == "manager" else job
                for job in state.jobs
            ]
        }
    )
    session = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=size) as pilot:
        view = await _open(pilot, app, session.jobs.get("manager"))
        todos = app.query_one(TodoPanel)
        for key in (None, "ctrl+t", "ctrl+down", "ctrl+t"):
            if key:
                await pilot.press(key)
            await pilot.pause()
            assert view._body.size.height >= 5
            assert todos._affordance.display
            assert "ctrl+t" in str(todos._affordance.content)
            status = app.query_one("#status-band")
            shell = app.query_one("#input-shell")
            assert shell.size.height >= 1
            assert status.region.bottom <= app.screen.region.bottom
            assert shell.region.contains_region(status.region)
            assert not app.screen.show_vertical_scrollbar
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one("#input-row").display


@pytest.mark.asyncio
async def test_open_manager_tracks_resumed_attempt_without_reentry() -> None:
    state = scoped_state()
    session: Any = FakeSession()
    install(session, state)
    loaded, unloaded = [], []

    async def load(job_id):  # noqa: ANN001, ANN202
        loaded.append(job_id)
        return True

    async def unload(job_id):  # noqa: ANN001, ANN202
        unloaded.append(job_id)

    session.load_job_trajectory = load
    session.unload_job_trajectory = unload
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, session.jobs.get("manager"))
        resumed = state.model_copy(
            update={
                "jobs": [
                    (
                        row.model_copy(
                            update={
                                "id": "manager-new",
                                "attempt_aliases": ["manager"],
                                "todos": plan("Resumed manager plan"),
                            }
                        )
                        if row.id == "manager"
                        else (
                            row.model_copy(update={"parent_job_id": "manager-new"})
                            if row.id == "leaf"
                            else row
                        )
                    )
                    for row in state.jobs
                ]
            }
        )
        install(session, resumed)
        app._apply_frontend_state(resumed)
        await pilot.pause()
        await app.workers.wait_for_complete(
            [worker for worker in app.workers if worker.group == "subagent-trajectory"]
        )
        await pilot.pause()
        assert view.job_id == "manager-new"
        assert "manager" in unloaded and "manager-new" in loaded
        assert "no longer on" not in " ".join(view.rendered_rows())
        assert "Resumed manager plan" in str(app.query_one(TodoPanel)._body.content)
        assert set(app.query_one(SubagentPanel)._rows) == {"leaf"}
        await pilot.click(app.query_one(SubagentPanel)._rows["leaf"])
        await pilot.pause()
        assert "back to parent" in " ".join(view.rendered_rows())
        await pilot.press("escape")
        await pilot.pause()
        assert view.job_id == "manager-new"


@pytest.mark.asyncio
async def test_child_todo_loading_unavailable_and_authoritative_clear_are_distinct() -> None:
    state = scoped_state()
    session = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _open(pilot, app, session.jobs.get("manager"))
        todos = app.query_one(TodoPanel)
        for phases, loading, expected in [
            (None, True, "Loading todos"),
            (None, False, "Todos unavailable"),
            ([], False, "No todos"),
        ]:
            state = state.model_copy(
                update={
                    "jobs": [
                        row.model_copy(update={"todos": phases}) if row.id == "manager" else row
                        for row in state.jobs
                    ]
                }
            )
            install(session, state)
            app._trajectory_state["manager"] = "loading" if loading else ""
            app._refresh_subagent_view()
            await pilot.pause()
            assert expected in str(todos._body.content)
            assert "Root plan" not in str(todos._body.content)
            assert "Manager plan" not in str(todos._body.content)


def test_canonical_edit_result_keeps_unified_diff() -> None:
    end = _result("edit", "edit", "Edited synthetic file")
    end["result"]["details"] = {
        "diff": ["--- before", "+++ after", "-old", "+new"],
        "added": 1,
        "removed": 1,
    }
    owner = FrontendStateStore(FrontendSessionState(session_id="root", epoch="owner"))
    delta = owner.mutate(
        jobs=[
            JobState(
                id="leaf",
                type="task",
                trajectory=[
                    _call("edit", "edit", path="synthetic.txt", old_text="old", new_text="new"),
                    end,
                ],
            )
        ]
    )
    assert delta is not None
    follower = FrontendStateStore(FrontendSessionState(session_id="root", epoch="owner"))
    follower.apply_update(FrontendUpdate.model_validate_json(delta.model_dump_json()))
    card = entry_block(fold_trajectory(follower.state.jobs[0].trajectory)[0])
    assert isinstance(card, ToolCard)
    assert card._diff == ["--- before", "+++ after", "-old", "+new"]
    assert card.can_expand()


@pytest.mark.asyncio
async def test_late_durable_todo_read_cannot_retarget_selected_owner(monkeypatch) -> None:
    from local_operator.tui.widgets import todo_panel

    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_read(*args):  # noqa: ANN002, ANN202
        entered.set()
        await release.wait()
        return plan("Old child")

    monkeypatch.setattr(todo_panel.asyncio, "to_thread", delayed_read)
    session = FakeSession()
    panel = TodoPanel()
    panel.sync(session, session_id="child-a", transcript_directory="synthetic-child-a")
    await entered.wait()
    pending = list(panel._todo_loads.values())
    panel.sync(session, session_id="child-b")
    release.set()
    await asyncio.gather(*pending)
    assert panel._selection == ("child-b", None)
    assert "Old child" not in str(panel._body.content)


def _dock_state(app: OperatorApp) -> tuple[bool, bool, bool]:
    band = app.query_one("#band")
    panel = app.query_one(SubagentPanel)
    todos = app.query_one(TodoPanel)
    return (panel.display, todos.display, band.has_class("has-slot"))


@pytest.mark.parametrize("leaf_has_plan", [True, False])
@pytest.mark.asyncio
async def test_opening_a_leaf_settles_the_dock_in_the_same_handler(leaf_has_plan: bool) -> None:
    """Re-scoping the dock to a leaf's (empty) roster must not straddle frames.

    Opening a leaf hides the child panel and re-decides the band's
    ``has-slot`` inset from whatever is STILL docked (the leaf's todo panel:
    a plan, or the authoritative "No todos" of an empty one). Both are decided
    inside ``_open_subagent_view`` itself, through ``_refresh_band``. A bare panel
    ``sync`` deferred to the next refresh left the inset to the 1 Hz poll, so
    the dock reflowed twice (panel hides; poll drops the inset a frame or a
    second later) and posted a four-widget ``messages.Layout`` cascade on
    whichever later frame ran it. That is what made a spinner tick "post
    layout" under xdist contention and the dock visibly jump after the page
    had painted.

    The invariant is stated as "what ``open`` returns with is what every later
    refresh reaches", read on the frame ``open`` returns on: that is the only
    place where "settled in the handler" is distinguishable from "settled
    eventually". Parametrised on the leaf's plan so both todo-panel branches
    (a rendered list, and the one-row empty state) are shown to settle with
    the roster in the same handler.
    """
    state = scoped_state()
    if not leaf_has_plan:
        state = state.model_copy(
            update={
                "jobs": [
                    job.model_copy(update={"todos": []}) if job.id == "leaf" else job
                    for job in state.jobs
                ]
            }
        )
    session = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        panel_shown, _, inset = _dock_state(app)
        assert panel_shown and inset, "fixture must start with a docked roster"

        app._open_subagent_view("leaf")
        # No pause: the dock has to be right on the frame that opened the page.
        settled = _dock_state(app)
        panel_shown, todos_shown, inset = settled
        assert panel_shown is False, "a leaf has no children to dock"
        assert inset is (panel_shown or todos_shown), "the inset follows what is still docked"

        # Nothing left over for a later frame to flip: the handler's answer is
        # the answer the deferred refresh and the poll both arrive at.
        for _ in range(3):
            await pilot.pause()
            assert _dock_state(app) == settled
        app._refresh_band()
        assert _dock_state(app) == settled


class WindowedJobs(SnapshotJobs):
    """``SnapshotJobs`` as the LIVE manager always is: with a retention window.

    The facade carries no ``retention_ms``, and ``_within_roster_window`` fails
    OPEN without one — identity, every row kept — so a fixture built on the
    bare facade cannot see a settled child age out at all. That is why the
    round-1 finding had to be reproduced with a probe that set the attribute by
    hand, and why the mark's agreement with the page had no test (review round
    1, F1): the disagreement only exists where a window does.
    """

    def __init__(self, values: Any, retention_ms: float) -> None:
        super().__init__(values)
        self.retention_ms = retention_ms


@pytest.mark.asyncio
async def test_a_mark_never_promises_a_level_that_would_open_empty() -> None:
    """The mark counts what the ROSTER would render, not what the graph holds.

    A child that settled longer ago than the manager's window is still in the
    comms graph, and the count used to read it straight out of the graph — so
    the dock painted ``Coordinate review ⊞1`` over a page that opened with no
    rows at all, and nothing on either surface said the level had aged out
    (review round 1, F1). Both halves are asserted, because the defect was
    their DISAGREEMENT: the row's promise and the page it opens.
    """
    state = scoped_state()
    settled = time.time() - 3600.0
    state = state.model_copy(
        update={
            "jobs": [
                (
                    job.model_copy(update={"status": "completed", "settled_at": settled})
                    if job.id == "leaf"
                    else job
                )
                for job in state.jobs
            ]
        }
    )
    session: Any = FakeSession()
    install(session, state)
    # The production default, so the fixture is the window a real dock runs.
    session.jobs = WindowedJobs(state.jobs, 5 * 60_000)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(SubagentPanel)
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()
        assert panel._children_counts == {"manager": 0, "sibling": 0}
        assert "⊞" not in row_text(panel, "manager"), row_text(panel, "manager")

        # The page that mark used to promise, resolved through the roster the
        # reader actually gets: one level down, and empty.
        scoped, _ = app._subagent_roster()
        assert [job.id for job in scoped] == ["manager", "sibling"], scoped
        await _open(pilot, app, session.jobs.get("manager"))
        await pilot.pause()
        page, _ = app._subagent_roster()
        assert page == [], page
        assert not panel._rows


@pytest.mark.asyncio
async def test_a_mark_costs_one_walk_of_the_graph_however_many_rows_list_it() -> None:
    """F2's shape: the mark was O(rows x nodes), one ``children()`` per row.

    ``SubagentComms.children`` rebuilds every node per call, so the per-row form
    cost 275 ms for ONE refresh at 500 rows — and it ran from the 1 Hz poll and
    from every ``Subagent*`` handler. The unit is the walks the facade sees, and
    the assertion is that the count does not move with the size of the roster it
    is answering for.
    """
    state = scoped_state()
    wide = [
        JobState(id=f"wide-{n}", type="task", parent_job_id="manager", label=f"wide {n}")
        for n in range(40)
    ]
    state = state.model_copy(update={"jobs": [*state.jobs, *wide]})
    session: Any = FakeSession()
    install(session, state)
    session._subagent_comms = comms = CountingComms(state.jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()
        one_row = [session.jobs.get("manager")]
        many_rows = session.jobs.list()

        comms.walks = 0
        assert app._subagent_child_counts(one_row) == {"manager": 41}
        one = comms.walks
        comms.walks = 0
        counts = app._subagent_child_counts(many_rows)
        many = comms.walks

        assert counts["manager"] == 41, counts
        assert one == many, (one, many)
        assert one < 5, one  # a per-row walk would be 43 here


@pytest.mark.parametrize("size", [(75, 30), (60, 30)])
@pytest.mark.asyncio
async def test_the_mark_survives_the_widths_where_it_used_to_disappear(
    size: tuple[int, int],
) -> None:
    """D2's cliff, at the two widths the finding measured it at.

    ``Coordinate review`` is 17 cells. With the mark appended BEFORE the
    truncation the row kept it only from 64 columns up, so at 75x30 and 60x30 a
    parent and a childless sibling painted the same row — the one thing the mark
    exists to prevent. The name yields those cells now, and a row still never
    overruns its width.
    """
    state = nested_state()
    session: Any = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=size) as pilot:
        panel = app.query_one(SubagentPanel)
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()
        assert set(panel._rows) == {"manager", "sibling"}
        assert "⊞1" in row_text(panel, "manager"), row_text(panel, "manager")
        assert "⊞" not in row_text(panel, "sibling"), row_text(panel, "sibling")
        assert app.screen.size == app.screen.virtual_size


@pytest.mark.parametrize("width", [40, 36])
@pytest.mark.asyncio
async def test_the_header_keeps_the_hint_inside_the_terminal_at_forty_columns(width: int) -> None:
    """D1: the constant ceiling alone painted the caption past the panel edge.

    At 40x30 the header was 43 cells in a 45-cell region, so `ctrl+g` sat in
    columns 38-43 — off screen — and the dock stopped following the terminal
    (panel width 45 at 40, 44 and 36 columns, against 38 at the same size with
    no scope). The scope is bounded by the LIVE row width now, and the hint's
    cells are charged before the scope's, so the way back out survives at any
    width where a caption exists at all.
    """
    state = scoped_state()
    session: Any = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(width, 30)) as pilot:
        panel = app.query_one(SubagentPanel)
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()
        await _open(pilot, app, session.jobs.get("manager"))
        await pilot.pause()
        caption = panel.summary_text()
        assert caption.endswith("ctrl+g"), caption
        assert cell_len(caption) <= panel._row_width(), (caption, panel._row_width())
        assert panel.size.width <= width, (panel.size.width, width)
        assert app.screen.size == app.screen.virtual_size


@pytest.mark.asyncio
async def test_the_compact_caption_names_the_level_it_counts() -> None:
    """D1: the summary had no scope segment at all, at the one size it is alone.

    At 40x20 the caption is the panel's ONLY representation, and it read a bare
    ``Subagents · 1 running ⣯ · ctrl+g`` while the counts under it were the open
    page's children — the unqualified word the scope exists to qualify. Counts
    yield to the level name now (``_SUMMARY_SHED_ORDER``), so the name survives
    the width that sheds them.
    """
    state = nested_state()
    session: Any = FakeSession()
    install(session, state)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(40, 20)) as pilot:
        panel = app.query_one(SubagentPanel)
        await pilot.pause()
        app._refresh_band()
        await pilot.pause()
        await _open(pilot, app, session.jobs.get("manager"))
        await pilot.pause()
        assert panel._header_compact, "the assertion is about the compact caption"
        caption = panel.summary_text()
        assert caption.startswith("Subagents of Coordinate"), caption
        assert "running" not in caption, caption  # a count yields to the name
        assert caption.endswith("ctrl+g"), caption
        assert cell_len(caption) <= panel._row_width(), (caption, panel._row_width())
