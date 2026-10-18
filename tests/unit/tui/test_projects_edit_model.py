"""The projects EDIT model (LOH-19 P2): the changed-only builder, as pure rules.

The edit form's whole contract is here, away from Textual: it must send ONLY
the fields the reader changed (per-field last-writer-wins), spell a clear the
way the store reads it, and refuse the one clear the store cannot express (an
estimate) BEFORE a write is attempted. Driving it through widgets would test
the message pump; these are the rules the widgets must not reinterpret.

The state is TEXT, not store types: the form's controls are text, and a
comparison of typed values ("5" against "5.0") would send fields nobody
changed. Normalisation lives in :meth:`ProjectEditState.from_project` and in
the builder, so the estimate's spelling and a description's trailing newline
cannot invent a write.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError

from local_operator.projects import ProjectDoneGateError, ProjectEdit
from local_operator.tui.projects_edit import (
    ProjectEditState,
    build_edit_plan,
    refusal_field,
)


def _project(**overrides: Any) -> dict[str, Any]:
    """One composed project payload, shaped like ``build_project_view``'s."""
    base: dict[str, Any] = {
        "id": "p1",
        "name": "alpha",
        "title": "Alpha",
        "description": "the body",
        "status": "active",
        "tags": ["tui", "spec"],
        "team": "core",
        "owner": "damian",
        "start_date": "2026-09-20",
        "target_date": "2026-10-04",
        "estimate": 5.0,
        "estimate_unit": "points",
    }
    base.update(overrides)
    return base


def _state(**overrides: Any) -> ProjectEditState:
    return replace(ProjectEditState.from_project(_project()), **overrides)


# ---------------------------------------------------------------------------
# Reading the state
# ---------------------------------------------------------------------------


def test_state_reads_every_editable_field_from_the_composed_project() -> None:
    state = ProjectEditState.from_project(_project())
    assert state == ProjectEditState(
        title="Alpha",
        name="alpha",
        description="the body",
        status="active",
        tags=("tui", "spec"),
        team="core",
        owner="damian",
        start_date="2026-09-20",
        target_date="2026-10-04",
        estimate="5",
        estimate_unit="points",
    )


def test_state_treats_a_null_field_as_the_empty_text_the_form_shows() -> None:
    """A store null and a cleared field are one value to the form — otherwise a
    project with no owner would look "changed" the moment the form opened."""
    state = ProjectEditState.from_project(
        _project(
            title=None,
            description=None,
            version=None,
            team=None,
            owner=None,
            start_date=None,
            target_date=None,
            estimate=None,
            tags=None,
            estimate_unit=None,
        )
    )
    assert state.title == ""
    assert state.description == ""
    assert state.team == ""
    assert state.owner == ""
    assert state.start_date == ""
    assert state.target_date == ""
    assert state.estimate == ""
    # The unit is NOT nullable in the store; a missing one reads as its default.
    assert state.estimate_unit == "points"
    assert state.tags == ()


@pytest.mark.parametrize(
    ("stored", "text"),
    [(5, "5"), (5.0, "5"), (5.5, "5.5"), (0.25, "0.25"), (1000.0, "1000")],
)
def test_state_spells_the_estimate_the_way_the_form_does(stored: float, text: str) -> None:
    assert ProjectEditState.from_project(_project(estimate=stored)).estimate == text


def test_state_keeps_a_status_word_this_build_does_not_know() -> None:
    """A row from a newer build loads verbatim (the store's own rule); the form
    must not normalise it into this build's vocabulary before the reader acts."""
    assert ProjectEditState.from_project(_project(status="blocked")).status == "blocked"


# ---------------------------------------------------------------------------
# Changed only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "changed", "expected"),
    [
        ("title", "Renamed", "Renamed"),
        ("name", "renamed-key", "renamed-key"),
        ("description", "new body", "new body"),
        ("status", "qa", "qa"),
        ("team", "personal", "personal"),
        ("owner", "someone", "someone"),
        ("start_date", "2026-09-21", "2026-09-21"),
        ("target_date", "2026-10-05", "2026-10-05"),
        ("estimate", "8.5", 8.5),
        ("estimate_unit", "days", "days"),
    ],
)
def test_one_changed_field_sends_exactly_that_field(
    field: str, changed: Any, expected: Any
) -> None:
    plan = build_edit_plan(_state(), _state(**{field: changed}))
    assert plan.edit is not None
    assert plan.edit.model_fields_set == {field}
    assert getattr(plan.edit, field) == expected
    assert plan.no_change is False and plan.refusal == ""


def test_changed_tags_send_the_whole_list() -> None:
    plan = build_edit_plan(_state(), _state(tags=("tui",)))
    assert plan.edit is not None
    assert plan.edit.model_fields_set == {"tags"}
    assert plan.edit.tags == ["tui"]


def test_several_changed_fields_ride_one_edit() -> None:
    plan = build_edit_plan(_state(), _state(title="Renamed", team="personal"))
    assert plan.edit is not None
    assert plan.edit.model_fields_set == {"title", "team"}


def test_an_unchanged_state_is_a_no_op_with_no_edit() -> None:
    plan = build_edit_plan(_state(), _state())
    assert plan.no_change is True
    assert plan.edit is None
    assert plan.refusal == "" and plan.field == ""


def test_text_differences_the_store_strips_are_not_changes() -> None:
    """The store strips a description and trims the one-line fields on write;
    a trailing newline the reader never meant to add is not an edit."""
    initial = _state()
    plan = build_edit_plan(initial, replace(initial, description="the body\n\n", title=" Alpha "))
    assert plan.no_change is True


def test_the_same_estimate_in_another_spelling_is_not_a_change() -> None:
    plan = build_edit_plan(_state(), _state(estimate="5.0"))
    assert plan.no_change is True


def test_a_reordered_tag_list_is_a_change_because_the_store_keeps_order() -> None:
    plan = build_edit_plan(_state(), _state(tags=("spec", "tui")))
    assert plan.edit is not None and plan.edit.model_fields_set == {"tags"}


# ---------------------------------------------------------------------------
# Clears
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("title", None),
        ("owner", None),
        ("team", None),
        ("description", ""),
        ("start_date", ""),
        ("target_date", ""),
        ("tags", []),
    ],
)
def test_clearing_a_field_sends_the_stores_clear_spelling(field: str, expected: Any) -> None:
    plan = build_edit_plan(_state(), _state(**{field: "" if field != "tags" else ()}))
    assert plan.edit is not None
    assert plan.edit.model_fields_set == {field}
    assert getattr(plan.edit, field) == expected


def test_an_empty_key_is_refused_with_the_forms_actionable_sentence() -> None:
    """The store's own message is about a "project name" and its grammar; the
    form calls the field a key, so it answers with the sentence that names it."""
    plan = build_edit_plan(_state(), _state(name=""))
    assert plan.edit is None and plan.no_change is False
    assert plan.field == "name"
    assert "key is required" in plan.refusal


def test_an_estimate_the_store_cannot_clear_is_refused_before_any_write() -> None:
    plan = build_edit_plan(_state(), _state(estimate=""))
    assert plan.edit is None
    assert plan.field == "estimate"
    assert "cannot be cleared" in plan.refusal


def test_an_unset_estimate_left_unset_is_a_no_op_not_a_refusal() -> None:
    initial = ProjectEditState.from_project(_project(estimate=None))
    plan = build_edit_plan(initial, replace(initial, title="Renamed"))
    assert plan.edit is not None and plan.edit.model_fields_set == {"title"}


def test_an_invalid_tag_is_refused_under_the_tags_field() -> None:
    """The store's tag grammar is the validator; the refusal names its field."""
    plan = build_edit_plan(_state(), _state(tags=("TUI",)))
    assert plan.edit is None and plan.no_change is False
    assert plan.field == "tags"
    assert "lowercase" in plan.refusal


def test_a_non_numeric_estimate_is_refused_with_the_stores_words() -> None:
    plan = build_edit_plan(_state(), _state(estimate="abc"))
    assert plan.edit is None and plan.field == "estimate"
    assert plan.refusal == "estimate must be a number"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def test_an_inverted_range_is_refused_on_the_target_date() -> None:
    plan = build_edit_plan(_state(), _state(start_date="2026-11-01"))
    assert plan.edit is None and plan.field == "target_date"
    assert "before the start date" in plan.refusal


def test_clearing_the_target_date_leaves_a_late_start_legal() -> None:
    plan = build_edit_plan(_state(), _state(start_date="2026-11-01", target_date=""))
    assert plan.edit is not None
    assert plan.edit.model_fields_set == {"start_date", "target_date"}


def test_one_date_alone_is_legal() -> None:
    initial = ProjectEditState.from_project(_project(target_date=None))
    plan = build_edit_plan(initial, replace(initial, start_date="2026-11-01"))
    assert plan.edit is not None


# ---------------------------------------------------------------------------
# Refusal routing (which field an app-side store refusal belongs under)
# ---------------------------------------------------------------------------


def test_the_done_gate_refusal_belongs_under_the_status_row() -> None:
    assert refusal_field(ProjectDoneGateError("milestones are incomplete: groundwork")) == "status"


def test_a_store_validation_error_belongs_under_its_own_field() -> None:
    try:
        ProjectEdit(start_date="not-a-date")
    except ValidationError as exc:
        assert refusal_field(exc) == "start_date"
    else:  # pragma: no cover - the validator must refuse
        raise AssertionError("ProjectEdit accepted a non-date")


def test_an_unattributed_refusal_falls_back_to_the_key_field() -> None:
    assert refusal_field(ValueError("project 'x' already exists")) == "key"
    assert refusal_field(KeyError("Project with id p1 not found")) == "key"
