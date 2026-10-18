"""The projects page's EDIT and MILESTONE flows (LOH-19 P2), through the REAL app.

Driven through ``OperatorApp`` with an isolated HOME/config, the same harness
the P1-P6 suites use, because every rule here is a CROSSING: the detail page's
state machine, the form's changed-only diff, the app's write through
``ProjectRegistry.update_project``/``set_milestone``, and the recomposition
that makes the page agree with the store afterwards.

What is deliberately NOT re-tested here: the create form's own gestures (its
suite owns them), the detail page's row grammar, and the store's validators
(their suites own them). The new surface is: `e` edits, `a` adds, `x` removes
behind a y/n question, and `esc` still pops exactly one level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.projects import ProjectEdit, ProjectMilestone, ProjectRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.projects_form import ProjectsFormPage
from local_operator.tui.widgets.projects_milestone import MilestonePrompt
from local_operator.tui.widgets.subagent_view import HintButton
from tests.unit.tui.test_projects_view import (
    _boot,
    _factory,
    _open,
    _ProjectSession,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _store(tmp_path: Path) -> ProjectRegistry:
    return ProjectRegistry(tmp_path)


def _registry(tmp_path: Path) -> ProjectRegistry:
    """One fully-populated project (one incomplete milestone) plus a sibling."""
    registry = ProjectRegistry(tmp_path)
    registry.create_project(
        ProjectEdit(
            name="parity-spec",
            title="TUI parity spec",
            team="core",
            owner="damian",
            tags=["tui", "spec"],
            estimate=5,
            start_date="2026-09-20",
            target_date="2026-10-04",
            description="TUI parity spec.\n\n## Goal\nOne reviewed spec.",
            milestones=[
                ProjectMilestone(name="groundwork", target_date="2026-09-20"),
                ProjectMilestone(name="spec shipped", target_date="2026-10-01"),
            ],
        )
    )
    registry.create_project(ProjectEdit(name="board-entry", title="Board entry", team="personal"))
    return registry


def _plain_registry(tmp_path: Path) -> ProjectRegistry:
    """One project with NO milestones: the session row is the only selectable."""
    registry = ProjectRegistry(tmp_path)
    registry.create_project(ProjectEdit(name="parity-spec", title="TUI parity spec"))
    return registry


async def _detail(pilot: Any, app: OperatorApp, view: Any) -> None:
    await pilot.press("d")
    await pilot.pause()
    await pilot.pause()
    assert view._mode == "detail"


def _form(view: Any) -> ProjectsFormPage:
    page = view._form_page
    assert isinstance(page, ProjectsFormPage)
    return page


def _prompt(view: Any) -> MilestonePrompt:
    prompt = view._milestone_prompt
    assert isinstance(prompt, MilestonePrompt)
    return prompt


def _hint_text(view: Any) -> str:
    """The painted hint row, whitespace-normalised (key and label sit 2 cells apart)."""
    painted = " ".join(
        hint.rendered()
        for hint in view._hints.children
        if isinstance(hint, HintButton) and hint.display
    )
    return " ".join(painted.split())


def _dump(tmp_path: Path, name: str = "parity-spec") -> dict[str, Any]:
    project = _store(tmp_path).get_project_by_name(name)
    assert project is not None
    return project.model_dump(mode="json")


def _without_clock(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "updated_at"}


# ---------------------------------------------------------------------------
# Opening the edit form
# ---------------------------------------------------------------------------


async def test_e_opens_the_edit_form_prefilled_with_every_field(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("e")
        await pilot.pause()
        await pilot.pause()

        assert view._mode == "form"
        page = _form(view)
        assert page.edit_mode
        assert page.display and not view._detail_page.display and not view._body.display
        assert page._title_input.value == "TUI parity spec"
        assert page._key_input.value == "parity-spec"
        assert page._description.text.startswith("TUI parity spec.\n\n## Goal")
        assert page._status_row.value == "active"
        assert page._tags_input.value == "tui, spec"
        assert page._team_input.value == "core"
        assert page._owner_input.value == "damian"
        assert page._start_input.value == "2026-09-20"
        assert page._target_input.value == "2026-10-04"
        assert page._estimate_input.value == "5"
        assert page._unit_row.value == "points"
        rows = view.rendered_rows()
        assert any("projects · edit project" in row for row in rows), rows
        # The key does not follow the title in an edit, and says so.
        assert any("changing it renames the key" in row for row in rows), rows


async def test_changing_one_field_writes_only_that_field(tmp_path: Path) -> None:
    """Per-field last-writer-wins: the other fields are not in the payload."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)

        await pilot.press("e")
        await pilot.pause()
        page = _form(view)
        page._title_input.focus()
        # Append to the prefilled value: `focus()` alone leaves the caret at
        # the start, so a plain typist's append is modelled explicitly.
        page._title_input.cursor_position = len(page._title_input.value)
        await pilot.press(*" v2")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

        after = _dump(tmp_path)
        assert after["title"] == "TUI parity spec v2"
        assert _without_clock(after) | {"title": ""} == _without_clock(before) | {"title": ""}
        assert view._mode == "detail", "a landed edit returns to the page"
        assert view._notice == "saved"
        assert any("TUI parity spec v2" in row for row in view.rendered_rows())


async def test_clearing_owner_team_title_and_dates_writes_the_clears(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("e")
        await pilot.pause()
        page = _form(view)
        page._title_input.value = ""
        page._team_input.value = ""
        page._owner_input.value = ""
        page._start_input.value = ""
        page._target_input.value = ""
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

        after = _dump(tmp_path)
        assert after["title"] is None
        assert after["team"] is None
        assert after["owner"] is None
        assert after["start_date"] is None
        assert after["target_date"] is None
        assert after["description"].startswith("TUI parity spec")  # untouched
        assert after["estimate"] == 5.0
        assert view._mode == "detail"


async def test_a_no_change_save_is_a_receipt_not_a_write(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)

        await pilot.press("e")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert view._mode == "form", "the receipt is painted in the form"
        assert view._notice == "no changes to save"
        assert _dump(tmp_path) == before, "nothing may be stamped for a no-op"


async def test_clearing_the_estimate_is_refused_inline_and_keeps_the_draft(
    tmp_path: Path,
) -> None:
    """The store has no clear state for an estimate; the form says so, no write."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)

        await pilot.press("e")
        await pilot.pause()
        page = _form(view)
        page._estimate_input.value = ""
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert view._mode == "form"
        assert app.focused is page._estimate_input
        assert any("cannot be cleared" in row for row in view.rendered_rows())
        assert _dump(tmp_path) == before


async def test_the_done_gate_refuses_a_status_change_in_the_form(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("e")
        await pilot.pause()
        page = _form(view)
        page._status_row.set_value("done")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert view._mode == "form"
        assert page._status_row.value == "done", "the draft keeps the reader's choice"
        assert any("milestone" in row and "error:" in row for row in view.rendered_rows())
        assert _dump(tmp_path)["status"] == "active"


async def test_a_store_refusal_keeps_the_edit_form_open_with_its_draft(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("e")
        await pilot.pause()
        page = _form(view)
        page._key_input.value = "board-entry"
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert view._mode == "form"
        assert page._key_input.value == "board-entry", "the draft survives a refusal"
        assert any("already exists" in row for row in view.rendered_rows())
        assert _dump(tmp_path)["name"] == "parity-spec"


# ---------------------------------------------------------------------------
# Adding a milestone
# ---------------------------------------------------------------------------


async def test_the_prompt_advertises_enter_and_tab_not_the_forms_save_key(
    tmp_path: Path,
) -> None:
    """The prompt owns its keys: `ctrl+s` here is the app's resume hotkey."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("a")
        await pilot.pause()
        hints = " ".join(
            hint.rendered()
            for hint in view._hints.children
            if isinstance(hint, HintButton) and hint.display
        )
        assert "tab" in hints and "next field" in hints
        assert "↵" in hints and " add" in hints
        assert "cancel" in hints
        assert "ctrl+s" not in hints, "the form's save key is not the prompt's add key"


async def test_a_adds_a_named_milestone_with_a_target_date(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("a")
        await pilot.pause()
        assert view._mode == "milestone"
        prompt = _prompt(view)
        assert prompt.display and view._detail_page.display, "the plan stays visible"
        assert prompt._name_input.has_focus

        prompt._name_input.value = "design round"
        prompt._date_input.value = "2026-10-08"
        await pilot.pause()
        await pilot.press("enter")  # from the name field: moves to the date
        await pilot.pause()
        assert prompt._date_input.has_focus
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None
        added = [m for m in project.milestones if m.name == "design round"]
        assert len(added) == 1 and added[0].target_date == "2026-10-08"
        assert view._mode == "detail" and not prompt.display
        assert view._notice == "milestone 'design round' added"
        assert any("design round" in row for row in view.rendered_rows())


async def test_a_adds_a_milestone_without_a_target_date(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("a")
        await pilot.pause()
        prompt = _prompt(view)
        prompt._name_input.value = "paperwork"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None
        added = [m for m in project.milestones if m.name == "paperwork"]
        assert len(added) == 1 and added[0].target_date is None


async def test_the_prompt_works_at_the_sixty_column_floor(tmp_path: Path) -> None:
    """The narrowest supported terminal: both fields visible, submit lands."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("a")
        await pilot.pause()
        prompt = _prompt(view)
        rows = prompt.readback()
        assert any("milestone: " in row for row in rows), rows
        assert any("target date: " in row for row in rows), rows

        prompt._name_input.value = "narrow"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None
        assert any(m.name == "narrow" for m in project.milestones)


async def test_an_invalid_date_is_refused_in_the_prompt_and_writes_nothing(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)

        await pilot.press("a")
        await pilot.pause()
        prompt = _prompt(view)
        prompt._name_input.value = "bad date"
        prompt._date_input.value = "not-a-date"
        await pilot.pause()
        await pilot.press("enter")  # name field -> date field
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert view._mode == "milestone"
        assert prompt._date_input.has_focus
        assert any("ISO YYYY-MM-DD" in row for row in prompt.readback())
        assert _dump(tmp_path) == before


async def test_an_empty_name_is_refused_in_the_prompt(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)

        await pilot.press("a")
        await pilot.pause()
        prompt = _prompt(view)
        prompt._name_input.value = "   "
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert view._mode == "milestone"
        assert any("needs a name" in row for row in prompt.readback())
        assert _dump(tmp_path) == before


async def test_a_blank_target_date_leaves_an_existing_milestones_date_alone(
    tmp_path: Path,
) -> None:
    """A blank optional date is UNTOUCHED, not a clear.

    ``MilestoneEdit``'s target_date is a tri-state (absent leaves it, ""/null
    clears it), so the prompt must send the field only when the reader typed a
    date — otherwise naming an existing milestone would erase a date they never
    meant to change. The echoed name with nothing else moved is the store's
    truthful no-change.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)

        await pilot.press("a")
        await pilot.pause()
        prompt = _prompt(view)
        prompt._name_input.value = "spec shipped"  # seeded with 2026-10-01
        assert prompt._date_input.value == ""  # the date is deliberately blank
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None
        unchanged = [m for m in project.milestones if m.name == "spec shipped"]
        assert len(unchanged) == 1 and unchanged[0].target_date == "2026-10-01"
        assert _dump(tmp_path) == before, "an echoed name with a blank date is a no-op"
        assert view._notice == "milestone 'spec shipped' was already exactly that"


async def test_a_naming_an_existing_milestone_reports_updated_not_added(
    tmp_path: Path,
) -> None:
    """`set_milestone` is add-or-update BY NAME; the receipt says which it did."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 34)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("a")
        await pilot.pause()
        prompt = _prompt(view)
        prompt._name_input.value = "spec shipped"
        prompt._date_input.value = "2026-10-09"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None
        changed = [m for m in project.milestones if m.name == "spec shipped"]
        assert len(changed) == 1 and changed[0].target_date == "2026-10-09"
        assert view._notice == "milestone 'spec shipped' updated"


# ---------------------------------------------------------------------------
# Removing a milestone: the inline y/n confirm
# ---------------------------------------------------------------------------


async def test_x_asks_y_n_and_n_keeps_the_milestone(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)
        page = view._detail_page

        await pilot.press("x")
        await pilot.pause()
        assert page.confirming and page.confirm_name == "groundwork"
        assert view._confirm_row.display
        assert any(
            "remove milestone 'groundwork'" in row for row in view.rendered_rows()
        )

        await pilot.press("n")
        await pilot.pause()
        assert not page.confirming and not view._confirm_row.display
        assert view._mode == "detail"
        assert _dump(tmp_path) == before


async def test_x_then_y_removes_the_milestone_and_recomposes(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("x")
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        await pilot.pause()

        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None
        assert [m.name for m in project.milestones] == ["spec shipped"]
        assert view._mode == "detail"
        assert view._notice == "milestone 'groundwork' removed"
        rows = view.rendered_rows()
        assert "milestones (0/1)" in " ".join(rows)
        assert not any("toggle groundwork" in row for row in rows)


async def test_x_off_a_milestone_row_says_what_it_needs(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _plain_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        page = view._detail_page
        assert page.selected_milestone() is None

        await pilot.press("x")
        await pilot.pause()
        assert not page.confirming
        assert view._notice == "put the cursor on a milestone row to remove"


async def test_a_create_opened_after_an_edit_is_still_a_fresh_create(
    tmp_path: Path,
) -> None:
    """`c` after an edit must not inherit its mode, values, or hint."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("e")
        await pilot.pause()
        assert _form(view).edit_mode
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "detail"

        await pilot.press("c")
        await pilot.pause()
        page = _form(view)
        assert not page.edit_mode
        assert page._title_input.value == "" and page._key_input.value == ""
        assert page._status_row.value == "planning"
        assert page._description.text.startswith("## Summary")
        assert any("follows the title" in row for row in view.rendered_rows())
        assert any("projects · new project" in row for row in view.rendered_rows())


# ---------------------------------------------------------------------------
# Discoverability: the page's own keys at the standard terminal sizes
# ---------------------------------------------------------------------------


async def test_the_detail_footer_advertises_edit_add_and_remove_at_100_columns(
    tmp_path: Path,
) -> None:
    """LOH-19 P2's keys must be DISCOVERABLE, not merely work.

    At 100x30 the page measures 96 cells, so the footer has 94. The row
    measured for a milestone selection is
    `↑↓ move · pgup/pgdn page · ↵ toggle groundwork · e edit · a add ·
    x remove · esc back` = 91 cells; `r refresh` would make it 104 and is the
    first of the canvas-era extras to shed.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        row = _hint_text(view)
        assert "e edit" in row, row
        assert "a add" in row, row
        assert "x remove" in row, row
        assert "toggle groundwork" in row, row
        assert "r refresh" not in row, row


async def test_the_detail_footer_keeps_edit_and_add_at_the_60_column_floor(
    tmp_path: Path,
) -> None:
    """At 60x24 the named context sheds first, never the keys.

    The page measures 56 cells, so the footer has 54. With the named context
    the row needs 61; the bare verb brings it to
    `↑↓ move · ↵ toggle · e edit · a add · esc back` = 50. `x remove` needs 12
    more cells and genuinely does not fit there — it is unadvertised and
    still works.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        row = _hint_text(view)
        assert "e edit" in row, row
        assert "a add" in row, row
        assert "↵ toggle" in row, row
        assert "toggle groundwork" not in row, row
        assert "x remove" not in row, row


# ---------------------------------------------------------------------------
# Esc pops one level, and Enter still toggles
# ---------------------------------------------------------------------------


async def test_esc_pops_one_level_from_each_new_surface(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        # form -> detail (a clean edit closes immediately, spec §7.7)
        await pilot.press("e")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "detail"

        # prompt -> detail
        await pilot.press("a")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "detail" and not view._milestone_prompt.display

        # confirm -> detail, with the milestone kept
        await pilot.press("x")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "detail" and not view._detail_page.confirming

        # detail -> canvas (the page itself, not transcript yet)
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "canvas"
        assert app._projects_view is view


async def test_enter_still_toggles_a_milestone_after_the_new_keys(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None
        assert project.milestones[0].completed_at is not None
