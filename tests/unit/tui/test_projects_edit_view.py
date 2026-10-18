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
    """The painted hint row, whitespace-normalised for containment checks."""
    painted = " ".join(
        hint.rendered()
        for hint in view._hints.children
        if isinstance(hint, HintButton) and hint.display
    )
    return " ".join(painted.split())


def _hint_width(view: Any) -> int:
    """The row's measured cell width — the number the ladder fits against."""
    from rich.cells import cell_len

    return sum(
        cell_len(hint.rendered())
        for hint in view._hints.children
        if isinstance(hint, HintButton) and hint.display
    )


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
    x remove · esc back` = 85 cells; `r refresh` would make it 97 and is the
    first of the canvas-era extras to shed. Every hint's key/label gap is ONE
    space (`HintButton._build` supplies it), not the two a leading-space label
    used to make — near the budget that rendered as `eedit`.
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
        # The exact painted strings: one space between key and label, seam
        # included. Two spaces here would mean a label carrying its own lead.
        assert view._move_hint.rendered() == "↑↓ move"
        assert view._edit_hint.rendered() == " · e edit"
        assert view._add_hint.rendered() == " · a add"
        assert view._remove_hint.rendered() == " · x remove"
        assert _hint_width(view) <= view.size.width - 2, _hint_width(view)


async def test_the_detail_footer_keeps_every_key_at_the_60_column_floor(
    tmp_path: Path,
) -> None:
    """At 60x24 the context sheds, never a write key (design D2 / UX U3 r1).

    The page measures 56 cells, so the footer has 54. The named context cannot
    coexist with the writes there; the bare verb plus a label-less `esc` buys
    the cells for the page's newest action:
    `↑↓ move · ↵ toggle · e edit · a add · x remove · esc` = 52. Removal is a
    P2 capability, not an optional extra — hiding it at a supported size made
    the feature undiscoverable.
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
        assert "x remove" in row, row
        assert "toggle groundwork" not in row, row
        assert "back" not in row, row
        assert view._move_hint.rendered() == "↑↓ move"
        assert view._edit_hint.rendered() == " · e edit"
        assert view._add_hint.rendered() == " · a add"
        assert view._remove_hint.rendered() == " · x remove"
        assert view._open_hint.rendered() == " · ↵ toggle"
        assert view._exit_hint.rendered() == " · esc"
        assert _hint_width(view) == 52, _hint_width(view)
        assert _hint_width(view) <= view.size.width - 2, _hint_width(view)


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


# ---------------------------------------------------------------------------
# Provenance and teardown (agent review round 1, R1)
# ---------------------------------------------------------------------------


def _without_project(view: Any, name: str) -> list[dict[str, Any]]:
    """The composed rows with one project removed — a vanished/recomposed store."""
    return [row for row in view._views if row["project"]["name"] != name]


async def test_an_orphaned_milestone_prompt_never_writes_another_project(
    tmp_path: Path,
) -> None:
    """R1's measured bug: a draft typed for A persisted on B after a recompose.

    The prompt is anchored on the project it OPENED on, and every way out of
    the detail disarms it. A submission that outlives its page is discarded —
    never re-aimed at whatever project the detail shows now.
    """
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("a")
        await pilot.pause()
        _prompt(view)._name_input.value = "belongs to parity"
        assert view._mode == "milestone" and view._milestone_project_id

        # The project the prompt was opened on leaves the composition.
        view.load(views=_without_project(view, "parity-spec"), updated_at=0.0)
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "canvas", "an orphaned prompt must not survive a recompose"
        assert view._milestone_prompt.display is False

        # Even a forced submission is discarded rather than sent anywhere.
        view._milestone_submitted("belongs to parity", "")
        await pilot.pause()
        await pilot.pause()
        for name in ("parity-spec", "board-entry"):
            project = _store(tmp_path).get_project_by_name(name)
            assert project is not None
            assert "belongs to parity" not in [m.name for m in project.milestones]


async def test_an_explicit_canvas_load_closes_the_prompt_and_the_question(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("a")
        await pilot.pause()
        view.load(views=view._views, view="board", updated_at=0.0)
        await pilot.pause()
        assert view._mode == "canvas" and not view._milestone_prompt.display

        await _detail(pilot, app, view)
        await pilot.press("x")
        await pilot.pause()
        assert view._detail_page.confirming
        view.load(views=view._views, view="board", updated_at=0.0)
        await pilot.pause()
        assert view._mode == "canvas"
        assert not view._detail_page.confirming
        assert view._confirm_row.display is False, "a stale question must not paint on the canvas"

        # Reopening the detail must NOT resurrect the question armed.
        await _detail(pilot, app, view)
        assert not view._detail_page.confirming
        assert view._confirm_row.display is False


async def test_a_vanished_project_never_leaves_a_question_behind(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("x")
        await pilot.pause()
        assert view._detail_page.confirming

        view.load(views=_without_project(view, "parity-spec"), updated_at=0.0)
        await pilot.pause()
        await pilot.pause()

        assert view._mode == "canvas"
        assert not view._detail_page.confirming
        assert view._confirm_row.display is False
        assert view._notice and "no longer in the store" in view._notice


# ---------------------------------------------------------------------------
# Read-only sessions (QA round 1, Q1)
# ---------------------------------------------------------------------------


async def _read_only_page(tmp_path: Path) -> tuple[OperatorApp, Any, Any]:
    """Boot a FOLLOWER session on a seeded store and open a project's detail.

    The store is this machine's (the test host has no other), which is exactly
    the mismatch the guard exists for: a page that can READ the owner's
    projects must not write them from a non-owning session.
    """
    session = _ProjectSession()
    session.owns_runtime = False
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    return app, session, None


async def test_a_follower_session_cannot_write_any_project_or_milestone(
    tmp_path: Path,
) -> None:
    """All five write paths refuse, say why, and leave the store byte-unchanged."""
    session = _ProjectSession()
    session.owns_runtime = False
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.pause()
        before = _dump(tmp_path)
        assert view._read_only is True
        assert view._notice and "Read-only" in view._notice, (
            "the state must be visible on entry"
        )

        # `c` must not open the create form.
        await pilot.press("c")
        await pilot.pause()
        assert view._mode == "canvas" and not view._form_page.display
        assert any("Read-only" in row for row in view.rendered_rows())

        await _detail(pilot, app, view)
        assert "e edit" not in _hint_text(view)
        assert "a add" not in _hint_text(view)
        assert "x remove" not in _hint_text(view)

        # `e` must not open the edit form.
        await pilot.press("e")
        await pilot.pause()
        assert view._mode == "detail"
        assert any("Read-only" in row for row in view.rendered_rows())

        # `a` must not open the prompt.
        await pilot.press("a")
        await pilot.pause()
        assert view._mode == "detail" and not view._milestone_prompt.display

        # `x` must not arm the question.
        await pilot.press("x")
        await pilot.pause()
        assert not view._detail_page.confirming

        # `↵` on the milestone must not toggle completion.
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"

        assert _dump(tmp_path) == before, "a follower's page must not write the store"


async def test_a_queued_message_from_a_follower_cannot_reach_the_store(
    tmp_path: Path,
) -> None:
    """The APP boundary refuses too, not only the page's keys (QA round 1, Q1).

    A message already in flight (or posted by a host that skipped the page)
    must not write anything for a follower session. Each of the five write
    messages is delivered directly and the store is compared in full.
    """
    from local_operator.tui.widgets.projects_view import (
        ProjectsViewEditSubmitted,
        ProjectsViewFormSubmitted,
        ProjectsViewMilestoneAddedRequested,
        ProjectsViewMilestoneRemoved,
        ProjectsViewMilestoneToggled,
    )

    session = _ProjectSession()
    session.owns_runtime = False
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)
        project_id = before["id"]

        for message in (
            ProjectsViewFormSubmitted(edit=ProjectEdit(name="follower-created")),
            ProjectsViewEditSubmitted(
                project_id=project_id,
                edit=ProjectEdit(title="FOLLOWER WRITE"),
                project_name="parity-spec",
            ),
            ProjectsViewMilestoneToggled(
                project_id, "groundwork", True, project_name="parity-spec"
            ),
            ProjectsViewMilestoneAddedRequested(
                project_id=project_id, name="follower-added", project_name="parity-spec"
            ),
            ProjectsViewMilestoneRemoved(
                project_id=project_id, name="groundwork", project_name="parity-spec"
            ),
        ):
            app.post_message(message)
            await pilot.pause()
            await pilot.pause()

        assert _dump(tmp_path) == before, "no queued write may land for a follower"
        assert any("Read-only" in row for row in view.rendered_rows())


async def test_a_normal_session_still_writes_after_the_read_only_guard(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        assert view._read_only is False
        assert "e edit" in _hint_text(view)

        await pilot.press("e")
        await pilot.pause()
        page = _form(view)
        page._title_input.focus()
        page._title_input.cursor_position = len(page._title_input.value)
        await pilot.press(*" ok")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()

        assert _dump(tmp_path)["title"] == "TUI parity spec ok"


async def test_a_bare_terminal_with_no_session_stays_writable(tmp_path: Path) -> None:
    """No session is not a follower: the local store is this terminal's own.

    The boot factory is HELD unresolved rather than returning ``None``: a
    factory that resolves to ``None`` is a pre-existing failed boot, while a
    terminal that has not (yet) adopted a session is the real state this rule
    is about. The page must be writable there.
    """
    import asyncio

    _store(tmp_path).create_project(ProjectEdit(name="parity-spec", title="TUI parity spec"))
    release = asyncio.Event()

    async def held_factory() -> Any:
        await release.wait()
        return None

    app = OperatorApp(held_factory)  # type: ignore[arg-type]
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(10):
            await pilot.pause()
            if app._projects_view is not None:
                break
        assert app._session is None, "no session was adopted"
        assert app._projects_read_only() is False
        app._open_projects_view()
        await pilot.pause()
        await pilot.pause()
        view = app._projects_view
        assert view is not None
        assert view._read_only is False
        view.action_create()
        await pilot.pause()
        assert view._mode == "form", "a bare terminal keeps the create page"
        release.set()


# ---------------------------------------------------------------------------
# The remove question owns the keyboard (UX round 1, U1)
# ---------------------------------------------------------------------------


async def test_tab_does_not_escape_the_remove_question(tmp_path: Path) -> None:
    """`x → Tab → Enter` used to TOGGLE the milestone under the open question."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        before = _dump(tmp_path)

        await pilot.press("x")
        await pilot.pause()
        assert view._detail_page.confirming
        await pilot.press("tab")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert not view._detail_page.confirming, "enter answers KEEP, it does not toggle"
        assert view._mode == "detail"
        assert _dump(tmp_path) == before, "nothing behind the question may be mutated"

        # The advertised yes still works after the tab trap.
        await pilot.press("x")
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        await pilot.pause()
        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None
        assert "groundwork" not in [m.name for m in project.milestones]


# ---------------------------------------------------------------------------
# A stale receipt and the done-gate copy (UX round 1, U2/U4)
# ---------------------------------------------------------------------------


async def test_typing_clears_the_no_change_receipt(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("e")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert view._mode == "form"
        assert view._notice == "no changes to save"

        page = _form(view)
        page._title_input.focus()
        page._title_input.cursor_position = len(page._title_input.value)
        await pilot.press(*"!")
        await pilot.pause()

        assert view._notice is None, "a dirty draft must not keep the no-change receipt"


async def test_the_done_gate_copy_names_no_force_done(tmp_path: Path) -> None:
    """The form has no force_done control; its refusal must not send readers there."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)

        await pilot.press("e")
        await pilot.pause()
        _form(view)._status_row.set_value("done")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        rows = view.rendered_rows()
        text = " ".join(rows)
        assert "cannot mark done while these milestones are open" in text, rows
        assert "groundwork" in text, rows
        assert "force_done" not in text, rows
        assert _dump(tmp_path)["status"] == "active"


# ---------------------------------------------------------------------------
# A long milestone name keeps both answers (design round 1, D1)
# ---------------------------------------------------------------------------


async def test_a_long_milestone_name_keeps_both_confirm_answers_visible(
    tmp_path: Path,
) -> None:
    """A legal 67-cell name at 60x24: the question WRAPS, it is not truncated.

    The old single-cell painter cut the whole sentence, so the destructive
    question showed neither `y` nor `n`. The row may take two or three lines;
    both answers and the name must be on screen.
    """
    long_name = "Release production migration and validate all customer integrations"
    from rich.cells import cell_len

    assert cell_len(long_name) == 67
    registry = ProjectRegistry(tmp_path)
    registry.create_project(
        ProjectEdit(
            name="parity-spec",
            title="TUI parity spec",
            milestones=[ProjectMilestone(name=long_name)],
        )
    )
    session = _ProjectSession()
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await _detail(pilot, app, view)
        await pilot.press("x")
        await pilot.pause()
        await pilot.pause()

        assert view._detail_page.confirming
        assert view._confirm_row.display
        rendered = "\n".join(view.rendered_rows())
        assert long_name in rendered, rendered
        assert "y remove" in rendered, rendered
        assert "n keep" in rendered, rendered
        assert view._confirm_row.size.height >= 2, view._confirm_row.size
        assert view._confirm_row.size.width <= view.size.width
        # Both answer keys are still live.
        await pilot.press("n")
        await pilot.pause()
        assert not view._detail_page.confirming
        project = _store(tmp_path).get_project_by_name("parity-spec")
        assert project is not None and project.milestones[0].name == long_name
