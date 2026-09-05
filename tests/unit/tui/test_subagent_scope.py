"""The viewed owner, not the root process, owns docked plans and children."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

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
    session.is_remote = True
    session.frontend_state = state
    session.jobs = SnapshotJobs(state.jobs)
    session._subagent_comms = SnapshotSubagentComms(state.jobs)


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
